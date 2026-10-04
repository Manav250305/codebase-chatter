from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from chatter.answer import ABSTENTION, AnswerResult, AnswerStatus
from chatter.answer_eval import (
    TimedGenerator,
    abstention_scores,
    classify,
    format_answer_report,
    format_review,
    run_answer_eval,
    save_answer_results,
)
from chatter.cli import make_app
from chatter.evaluate import EvalError
from conftest import FakeGenerator, HashEmbedder
from test_evaluate import QUESTIONS, write_eval


def result(text: str, status: AnswerStatus) -> AnswerResult:
    return AnswerResult(text, status, (), (), (), (), False, 0)


@pytest.mark.parametrize(
    ("text", "status", "expected"),
    [
        (ABSTENTION, AnswerStatus.ABSTAINED, "abstained"),
        (f"Partly [C1]. {ABSTENTION}", AnswerStatus.MIXED, "mixed"),
        ("The provided chunks do not mention OAuth.", AnswerStatus.ANSWERED, "near_miss_abstention"),
        ("It evicts the oldest entry [C1].", AnswerStatus.ANSWERED, "answered"),
    ],
)
def test_classify(text: str, status: AnswerStatus, expected: str) -> None:
    assert classify(result(text, status)) == expected


def test_abstention_scores() -> None:
    records = [
        {"expect_abstain": True, "classification": "abstained"},
        {"expect_abstain": True, "classification": "near_miss_abstention"},
        {"expect_abstain": False, "classification": "answered"},
        {"expect_abstain": False, "classification": "abstained"},
        {"expect_abstain": False, "classification": "mixed"},
    ]
    strict = abstention_scores(records, frozenset({"abstained"}))
    assert (strict["true_abstentions"], strict["false_abstentions"], strict["missed_abstentions"]) == (1, 1, 1)
    assert strict["precision"] == 0.5 and strict["recall"] == 0.5
    assert strict["false_abstention_rate"] == pytest.approx(1 / 3)
    lenient = abstention_scores(records, frozenset({"abstained", "mixed", "near_miss_abstention"}))
    assert lenient["recall"] == 1.0 and lenient["precision"] == 0.5
    assert abstention_scores([], frozenset({"abstained"}))["precision"] is None


def scripted_reply(system: str, user: str):  # type: ignore[no-untyped-def]
    """Abstain on the OAuth question, cite C1 (and a bogus C9) otherwise."""
    question = user.rsplit("Question:", 1)[-1]  # the context may mention anything
    if "oauth" in question.lower():
        text = ABSTENTION
    elif "evict" in question:
        text = "It drops the oldest entries until the cache fits [C1]."
    else:
        text = "It splits head and body [C1] [C9]."
    return iter([w + " " for w in text.split(" ")])


def run(tmp_path: Path, reply: object = scripted_reply, **kwargs: object):  # type: ignore[no-untyped-def]
    repo, questions, corpora = write_eval(tmp_path)
    gen = FakeGenerator(reply)
    results = run_answer_eval(
        questions, corpora, lambda n: HashEmbedder(name=n), lambda: gen,
        model_name="m", repo_root=repo, **kwargs,
    )
    return results, gen, questions


def test_run_answer_eval_records_each_question(tmp_path: Path) -> None:
    results, gen, _ = run(tmp_path)
    by_id = {r["id"]: r for r in results["questions"]}
    assert [r["id"] for r in results["questions"]] == ["e1", "e2", "e3"]
    assert len(gen.prompts) == 3  # one generation per question

    e1 = by_id["e1"]
    assert e1["classification"] == "answered"
    assert e1["relevant_in_context"] == {"strict": True, "contain": True}
    assert e1["citations"][0]["tag"] == "C1" and e1["invalid_tags"] == []
    assert e1["cited_relevant"]["strict"] is (e1["citations"][0]["chunk_id"] in e1["relevant"])
    assert e1["ttft_s"] is not None and e1["ttft_s"] >= 0 and e1["generated_tokens"] > 0
    assert e1["prompt_tokens"] > 0 and e1["context"] and e1["context"][0]["location"]

    assert by_id["e2"]["invalid_tags"] == ["C9"]
    e3 = by_id["e3"]
    assert e3["classification"] == "abstained" and e3["expect_abstain"]
    assert e3["cited_relevant"] == {"strict": None, "contain": None}  # negatives have no relevant ids

    summary = results["summary"]
    assert summary["classifications"]["abstained"] == 1 and summary["classifications"]["answered"] == 2
    assert summary["abstention"]["strict"]["precision"] == 1.0
    assert summary["abstention"]["strict"]["recall"] == 1.0
    assert summary["invalid_tags"] == 1 and summary["answers_with_invalid_tags"] == 1
    assert results["settings"]["retrieval"] == "default" and results["answer_model"] == "fake-llm"


def test_run_answer_eval_flags_near_miss_and_missed_abstention(tmp_path: Path) -> None:
    reply = lambda s, u: iter(["The provided chunks do not cover that topic."])  # noqa: E731
    results, _, _ = run(tmp_path, reply=reply)
    assert {r["classification"] for r in results["questions"]} == {"near_miss_abstention"}
    abst = results["summary"]["abstention"]
    assert abst["strict"]["recall"] == 0.0 and abst["strict"]["precision"] is None
    assert abst["lenient"]["recall"] == 1.0 and abst["lenient"]["false_abstention_rate"] == 1.0


def test_run_answer_eval_split_and_generator_failure(tmp_path: Path) -> None:
    repo, questions, corpora = write_eval(tmp_path)

    def broken():  # type: ignore[no-untyped-def]
        raise OSError("gated repo")

    with pytest.raises(EvalError, match="could not load the answer model: gated repo"):
        run_answer_eval(questions, corpora, lambda n: HashEmbedder(name=n), broken, model_name="m", repo_root=repo)


def test_timed_generator_measures_first_token_and_closes() -> None:
    gen = FakeGenerator(lambda s, u: iter(["", "  ", "Hello ", "world"]))
    timed = TimedGenerator(gen)
    stream = timed.generate("s", "u", max_new_tokens=10)
    assert next(stream) == ""  # blank pieces do not count as the first token
    assert timed.timing.first_token is None
    assert list(stream) == ["  ", "Hello ", "world"]
    assert timed.timing.first_token is not None and timed.timing.ended is not None
    assert "".join(timed.timing.text) == "  Hello world" and gen.closed


def test_reports_and_saved_files(tmp_path: Path) -> None:
    results, _, questions = run(tmp_path)
    report = format_answer_report(results)
    assert "abstention" in report and "strict" in report and "lenient" in report
    assert "e2   flow" in report and "C9" in report
    review = format_review(results)
    assert review.count("**Verdict:**") == 3 and "## e3 (negative" in review
    assert "> It splits head and body [C1] [C9]." in review and "invalid tags: C9" in review

    json_path, md_path = save_answer_results(results, tmp_path / "results")
    assert json_path.name.endswith("-answers.json") and md_path == json_path.with_suffix(".md")
    assert json.loads(json_path.read_text())["kind"] == "answers"
    assert md_path.read_text() == review


def test_cli_answers(tmp_path: Path) -> None:
    _, questions, _ = write_eval(tmp_path)
    configs: list[object] = []

    def factory(config):  # type: ignore[no-untyped-def]
        configs.append(config)
        return FakeGenerator(scripted_reply)

    app = make_app(embedder_factory=lambda name: HashEmbedder(name=name), generator_factory=factory)
    runner = CliRunner()
    result = runner.invoke(
        app, ["eval", str(questions), "--answers", "--answer-model", "tiny/model", "--split", "tune"]
    )
    assert result.exit_code == 0, result.output
    assert "classifications:" in result.output and "Review " in result.output
    assert configs[0].model_name == "tiny/model"  # type: ignore[attr-defined]
    saved = sorted((questions.parent / "results").glob("*-answers-tune.*"))
    assert [p.suffix for p in saved] == [".json", ".md"]

    clash = runner.invoke(app, ["eval", str(questions), "--answers", "--sweep-fusion"])
    assert clash.exit_code == 1 and "separate run" in clash.output
