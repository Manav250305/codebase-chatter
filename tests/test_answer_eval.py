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
    assert results["settings"]["retrieval"] == "dense+graph(hops=2, weight=2)"
    assert results["answer_model"] == "fake-llm"


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
    assert "- retrieval: dense+graph(hops=2, weight=2), top" in review
    legacy = {**results, "settings": {**results["settings"], "retrieval": "default"}}
    assert "- retrieval: default fused, top" in format_review(legacy)  # runs before the switch
    assert "> It splits head and body [C1] [C9]." in review and "invalid tags: C9" in review

    json_path, md_path = save_answer_results(results, tmp_path / "results")
    assert json_path.name.endswith("-answers-transformers.json") and md_path == json_path.with_suffix(".md")
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
    saved = sorted((questions.parent / "results").glob("*-answers-transformers-tune.*"))
    assert [p.suffix for p in saved] == [".json", ".md"]

    clash = runner.invoke(app, ["eval", str(questions), "--answers", "--sweep-fusion"])
    assert clash.exit_code == 1 and "separate run" in clash.output


# ---------------------------------------------------------------------------
# Context-relative abstention, memory, backend
# ---------------------------------------------------------------------------

from chatter.answer_eval import MemorySampler, context_relative_abstention  # noqa: E402


def _record(qid: str, corpus: str, classification: str, contain: bool | None, negative: bool = False) -> dict:
    return {
        "id": qid, "corpus": corpus, "classification": classification, "expect_abstain": negative,
        "relevant_in_context": {"strict": contain, "contain": contain},
    }


def test_context_relative_abstention_per_corpus() -> None:
    records = [
        _record("a", "lib", "answered", False),
        _record("b", "lib", "abstained", False),
        _record("c", "lib", "answered", True),  # evidence present: excluded
        _record("d", "app", "near_miss_abstention", False),
        _record("e", "app", "abstained", None, negative=True),
    ]
    cra = context_relative_abstention(records)
    answerable = cra["answerable"]
    assert answerable["overall"] == {
        "n": 3, "answered": 1, "abstained": 2, "answered_rate": pytest.approx(1 / 3),
        "abstained_rate": pytest.approx(2 / 3), "ids": ["a", "b", "d"],
    }
    assert answerable["by_corpus"]["lib"]["answered_rate"] == 0.5
    assert answerable["by_corpus"]["app"]["abstained"] == 1
    assert cra["with_negatives"]["overall"]["ids"] == ["a", "b", "d", "e"]
    assert cra["with_negatives"]["by_corpus"]["app"]["n"] == 2


def test_memory_sampler_records_rss_and_swap() -> None:
    with MemorySampler(interval=0.01) as sampler:
        blob = bytearray(20_000_000)  # make RSS move a little
        blob[::4096] = b"x" * len(blob[::4096])
    stats = sampler.stats()
    assert stats["peak_rss_mb"] > 20
    assert stats["swap_used_peak_mb"] >= stats["swap_used_start_mb"] >= 0
    assert set(stats) == {"peak_rss_mb", "swap_used_start_mb", "swap_used_peak_mb", "swap_used_end_mb"}


def test_answer_eval_records_memory_backend_and_context_abstention(tmp_path: Path) -> None:
    results, _, _ = run(tmp_path, backend="mlx")
    assert results["backend"] == "mlx" and results["schema"] == 3
    assert results["model_load"]["seconds"] >= 0 and results["model_load"]["peak_rss_mb"] > 0
    for record in results["questions"]:
        assert record["memory"]["peak_rss_mb"] > 0 and "swap_used_peak_mb" in record["memory"]
    summary = results["summary"]
    assert summary["memory"]["max_peak_rss_mb"] > 0
    assert "context_relative_abstention" in summary
    report = format_answer_report(results)
    assert "(mlx)" in report and "context-relative abstention" in report and "RSS GB" in report
    json_path, _ = save_answer_results(results, tmp_path / "r")
    assert json_path.name.endswith("-answers-mlx.json")


# ---------------------------------------------------------------------------
# Evidence prompt
# ---------------------------------------------------------------------------


def chunk_tag(user: str, chunk_id: str) -> str:
    marker = f'id="{chunk_id}"'
    head = user[: user.index(marker)]
    return head[head.rindex('tag="') + 5 :].split('"', 1)[0]


def evidence_reply(system: str, user: str):  # type: ignore[no-untyped-def]
    """Quote the relevant line for evict, a made-up line for parse, abstain on OAuth."""
    question = user.rsplit("Question:", 1)[-1].lower()
    if "oauth" in question:
        text = ABSTENTION
    elif "evict" in question:
        tag = chunk_tag(user, "lib/cache.py::evict_least_recently_used")
        text = f"Evidence:\n[{tag}] while len(cache) > capacity:\n\nAnswer:\nIt pops entries while over capacity [{tag}]."
    else:
        text = "Evidence:\n[C1] head, body = raw.split(CRLF)\n\nAnswer:\nIt splits on CRLF [C1]."
    return iter([w + " " for w in text.split(" ")])


def test_evidence_prompt_records_quotes_and_withholds_unverified(tmp_path: Path) -> None:
    results, gen, _ = run(tmp_path, evidence_reply, prompt="evidence")
    from chatter.answer import EVIDENCE_SYSTEM_PROMPT

    assert gen.prompts[0][0] == EVIDENCE_SYSTEM_PROMPT
    assert results["settings"]["prompt"] == "evidence" and len(results["settings"]["prompt_sha256"]) == 64
    by_id = {r["id"]: r for r in results["questions"]}
    e1, e2, e3 = by_id["e1"], by_id["e2"], by_id["e3"]

    assert e1["classification"] == "answered" and e1["raw_classification"] == "answered"
    assert e1["evidence"]["verification"] == "verified" and e1["evidence"]["in_relevant"] is True
    (quote,) = e1["evidence"]["quotes"]
    assert quote["result"] == "verified" and quote["path"] == "lib/cache.py" and quote["line"] == 3
    assert e1["evidence"]["format"] == {
        "evidence_section": True, "entries": 1, "answer_section": True,
        "parseable": True, "abstention_only": False, "compliant": True,
    }

    assert e2["classification"] == "abstained" and e2["raw_classification"] == "answered"
    assert e2["answer"] == ABSTENTION and "raw.split(CRLF)" in e2["raw_answer"]
    assert e2["evidence"]["forced_abstention"] and e2["evidence"]["in_relevant"] is False
    assert e2["citations"] == []

    assert e3["evidence"]["verification"] == "abstained" and e3["evidence"]["in_relevant"] is None
    assert e3["evidence"]["format"]["abstention_only"] and e3["evidence"]["format"]["compliant"]

    e = results["summary"]["evidence"]
    assert e["format"] == {"parseable": 2, "abstention_only": 1, "neither": 0, "neither_ids": []}
    assert e["verification"] == {"verified": 1, "partial": 0, "unverified": 1, "abstained": 1}
    assert e["forced_abstentions"] == ["e2"]
    assert e["in_relevant"]["count"] == 1 and e["in_relevant"]["answers"] == 1
    assert e["in_relevant"]["answerable"] == 2 and e["in_relevant"]["rate_of_answers"] == 1.0

    report = format_answer_report(results)
    assert report.index("EVIDENCE IN A RELEVANT CHUNK: 1/1 answers") < report.index("classifications:")
    assert "format compliance: 2/3 parseable Evidence section, 1 abstention only, 0 neither" in report
    review = format_review(results)
    assert "**Evidence check:** unverified — the model's answer was withheld" in review
    assert "**Model's reply (withheld):**" in review and "✓ [" in review and "lib/cache.py:3" in review
    assert "prompt: evidence" in review

    json_path, _ = save_answer_results(results, tmp_path / "results")
    assert json_path.name.endswith("-answers-transformers-evidence.json")


def test_plain_prompt_records_have_no_evidence(tmp_path: Path) -> None:
    results, _, _ = run(tmp_path)
    assert all(r["evidence"] is None and r["prompt"] == "plain" for r in results["questions"])
    assert results["summary"]["evidence"] is None and results["settings"]["prompt"] == "plain"
    assert "EVIDENCE" not in format_answer_report(results)


def test_unknown_prompt_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="unknown prompt"):
        run(tmp_path, prompt="cot")
