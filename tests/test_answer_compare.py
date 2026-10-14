from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from chatter.answer_compare import (
    FAIL,
    PASS,
    PENDING,
    compare_answers,
    context_differences,
    extract_grades,
    format_comparison,
    grades_path,
    load_criteria,
    load_grades,
    parse_threshold,
    run_metrics,
    write_grades,
)
from chatter.answer_eval import format_review, save_answer_results
from chatter.cli import make_app
from chatter.evaluate import EvalError
from conftest import FakeGenerator, HashEmbedder
from test_answer_eval import evidence_reply, run, scripted_reply

REPO = Path(__file__).resolve().parents[1]
CRITERIA = {
    "name": "t",
    "criteria": {
        "unsupported_answers": {"pass_if": "<= 1"},
        "false_abstentions": {"pass_if": "<= 1"},
        "relevant_citation_rate": {"pass_if": ">= 1/2"},
        "graded_regressions": {"pass_if": "== 0"},
    },
}


def record(qid: str, classification: str, *, in_ctx: bool = True, cites: bool = True,
           negative: bool = False, raw: str | None = None, context: str = "a.py:1-2") -> dict[str, Any]:
    return {
        "id": qid,
        "expect_abstain": negative,
        "classification": classification,
        "raw_classification": raw or classification,
        "relevant_in_context": {"strict": None if negative else in_ctx, "contain": None if negative else in_ctx},
        "cited_relevant": {"strict": None if negative else cites, "contain": None if negative else cites},
        "context": [{"chunk_id": "a.py::f", "location": context}],
    }


def results(*records: dict[str, Any]) -> dict[str, Any]:
    return {"questions": list(records), "summary": {}}


def criteria(tmp_path: Path, spec: dict[str, Any] = CRITERIA) -> dict[str, Any]:
    path = tmp_path / "criteria.yaml"
    path.write_text(yaml.safe_dump(spec))
    return load_criteria(path)


BASE = results(
    record("q1", "answered"),
    record("q2", "answered", in_ctx=False, cites=False),  # unsupported
    record("q3", "answered", in_ctx=False, cites=False),  # unsupported
    record("q4", "abstained", cites=False),  # false abstention
    record("n1", "abstained", negative=True),
)


def test_thresholds() -> None:
    assert parse_threshold("<= 2").passes(Fraction(2)) and not parse_threshold("<= 2").passes(Fraction(3))
    assert parse_threshold(">= 26/36").passes(Fraction(26, 36))
    assert not parse_threshold(">= 26/36").passes(Fraction(25, 36))
    assert parse_threshold("== 0").passes(Fraction(0))
    with pytest.raises(EvalError, match="bad pass_if"):
        parse_threshold("at most two")


def test_run_metrics() -> None:
    m = run_metrics(BASE)
    assert m["unsupported_answers"] == {"value": 2, "ids": ["q2", "q3"]}
    assert m["false_abstentions"] == {"value": 1, "ids": ["q4"]}
    assert m["relevant_citation_rate"]["value"] == Fraction(1, 4)


def test_compare_passes_fails_and_raw_values(tmp_path: Path) -> None:
    new = results(
        record("q1", "answered"),
        record("q2", "abstained", in_ctx=False, cites=False, raw="answered"),  # withheld
        record("q3", "answered", in_ctx=False, cites=False),
        record("q4", "answered"),
        record("n1", "abstained", negative=True),
    )
    outcome = compare_answers(BASE, new, criteria(tmp_path))
    rows = {r["criterion"]: r for r in outcome["criteria"]}
    assert rows["unsupported_answers"]["new"] == 1 and rows["unsupported_answers"]["raw"] == 2
    assert rows["unsupported_answers"]["status"] == PASS
    assert rows["false_abstentions"]["new"] == 1 and rows["false_abstentions"]["status"] == PASS
    assert rows["relevant_citation_rate"]["new"] == Fraction(2, 4) and rows["relevant_citation_rate"]["status"] == PASS
    assert rows["graded_regressions"]["status"] == PENDING
    assert outcome["decision"] == PENDING
    assert outcome["classification_changes"] == [("q2", "answered", "abstained"), ("q4", "abstained", "answered")]

    worse = results(*[record(r["id"], "abstained", negative=r["expect_abstain"], cites=False) for r in BASE["questions"]])
    assert compare_answers(BASE, worse, criteria(tmp_path))["decision"] == FAIL


def test_graded_regressions(tmp_path: Path) -> None:
    new = results(
        record("q1", "abstained", cites=False),  # was correct, now abstains: a regression
        record("q2", "answered", in_ctx=False, cites=False),
        record("q3", "answered", in_ctx=False, cites=False),
        record("q4", "answered"),
        record("n1", "abstained", negative=True),
    )
    base_grades = {"q1": "correct", "q2": "wrong", "q3": "wrong", "q4": "wrong", "n1": "correctly abstained"}
    rows = {r["criterion"]: r for r in compare_answers(BASE, new, criteria(tmp_path), base_grades=base_grades)["criteria"]}
    assert rows["graded_regressions"]["graded"]["ids"] == ["q1"] and rows["graded_regressions"]["status"] == FAIL

    kept = results(record("q1", "answered"), *new["questions"][1:])
    pending = compare_answers(BASE, kept, criteria(tmp_path), base_grades=base_grades)
    graded = {r["criterion"]: r for r in pending["criteria"]}["graded_regressions"]
    assert graded["status"] == PENDING and graded["graded"]["ungraded"] == ["q1"]
    for grade, status in (("partial", FAIL), ("correct", PASS)):
        outcome = compare_answers(BASE, kept, criteria(tmp_path), base_grades=base_grades, new_grades={"q1": grade})
        assert {r["criterion"]: r for r in outcome["criteria"]}["graded_regressions"]["status"] == status


def test_different_context_voids_the_comparison(tmp_path: Path) -> None:
    moved = results(*[{**r, "context": [{"chunk_id": "a.py::f", "location": "a.py:1-9"}]} if r["id"] == "q3" else r
                      for r in BASE["questions"]])
    assert context_differences(BASE, moved) == ["q3"]
    outcome = compare_answers(BASE, moved, criteria(tmp_path))
    assert outcome["decision"] == "VOID"
    assert "VOID: the runs answered from different context for q3" in format_comparison(outcome, "a", "b")


def test_load_criteria_rejects_unknown_names(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="unknown criteria"):
        criteria(tmp_path, {"criteria": {"vibes": {"pass_if": ">= 1"}}})


# ---------------------------------------------------------------------------
# Grades
# ---------------------------------------------------------------------------

REVIEW = """# Answer review

## q1 (behavior, stdlib, answerable)

**Verdict:** correct

## q2 (behavior, stdlib, answerable)

**Verdict:** **Partial** - misses the retry path

## q3 (location, chatter, answerable)

**Verdict:** _(correct / partial / wrong / correctly abstained / should have abstained)_

## n1 (negative, stdlib, should abstain)

**Verdict:** Correctly abstained.
"""


def test_extract_grades() -> None:
    grades, ungraded, problems = extract_grades(REVIEW)
    assert grades == {"q1": "correct", "q2": "partial", "n1": "correctly abstained"}
    assert ungraded == ["q3"] and problems == []
    _, _, bad = extract_grades(REVIEW.replace("correct\n", "kinda right\n", 1))
    assert bad and "q1: unknown verdict 'kinda right'" in bad[0]


def test_grades_file_round_trip(tmp_path: Path) -> None:
    results_path = tmp_path / "eval" / "results" / "run.json"
    path = grades_path(results_path)
    assert path == tmp_path / "eval" / "grades" / "run.yaml"
    write_grades(path, results_path, {"q1": "correct"})
    assert load_grades(path) == {"q1": "correct"}
    path.write_text(yaml.safe_dump({"grades": {"q1": "great"}}))
    with pytest.raises(EvalError, match="unknown grades"):
        load_grades(path)


# ---------------------------------------------------------------------------
# End to end: two real runs, grading, and the CLI
# ---------------------------------------------------------------------------


def two_runs(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
    base, _, _ = run(tmp_path / "a", scripted_reply)
    new, _, _ = run(tmp_path / "b", evidence_reply, prompt="evidence")
    return base, new


def test_plain_vs_evidence_runs_compare(tmp_path: Path) -> None:
    base, new = two_runs(tmp_path)
    assert context_differences(base, new) == []  # same retrieval and context for both prompts
    outcome = compare_answers(base, new, criteria(tmp_path))
    text = format_comparison(outcome, "base", "new")
    assert "EVIDENCE IN A RELEVANT CHUNK (new): 1/1 answers" in text
    assert "format compliance (new): 2/3 parseable" in text and "answers withheld as unverified: e2" in text
    assert "classification changes: e2 answered -> abstained" in text


def test_cli_grade_and_compare(tmp_path: Path) -> None:
    base, new = two_runs(tmp_path)
    results_dir = tmp_path / "eval" / "results"
    base_json, base_md = save_answer_results(base, results_dir)
    new_json, _ = save_answer_results({**new, "created": "2026-10-15T00:00:00+00:00"}, results_dir)
    filled = format_review(base).replace(
        "**Verdict:** _(correct / partial / wrong / correctly abstained / should have abstained)_",
        "**Verdict:** correct",
    )
    base_md.write_text(filled)
    criteria_path = tmp_path / "criteria.yaml"
    criteria_path.write_text(yaml.safe_dump(CRITERIA))

    app = make_app(embedder_factory=lambda n: HashEmbedder(name=n))
    graded = CliRunner().invoke(app, ["grade", str(base_md)])
    assert graded.exit_code == 0, graded.output
    grades_file = tmp_path / "eval" / "grades" / f"{base_json.stem}.yaml"
    assert grades_file.is_file() and "3 graded (correct 3)" in graded.output
    assert yaml.safe_load(grades_file.read_text())["results"].endswith(f"{base_json.stem}.json")

    compared = CliRunner().invoke(
        app, ["compare-answers", str(base_json), str(new_json), "--criteria", str(criteria_path)]
    )
    assert compared.exit_code == 0, compared.output
    # e2 was graded correct and the evidence run withheld it: a regression, found without new grades.
    assert "regressions: e2" in compared.output and "decision: FAIL" in compared.output

    missing = CliRunner().invoke(app, ["grade", str(tmp_path / "nope.md")])
    assert missing.exit_code == 1


def test_cli_eval_answers_with_evidence_prompt(tmp_path: Path) -> None:
    from test_evaluate import write_eval

    _, qpath, _ = write_eval(tmp_path)
    app = make_app(
        embedder_factory=lambda n: HashEmbedder(name=n),
        generator_factory=lambda config: FakeGenerator(evidence_reply),
    )
    result = CliRunner().invoke(app, ["eval", str(qpath), "--answers", "--prompt", "evidence"])
    assert result.exit_code == 0, result.output
    assert "EVIDENCE IN A RELEVANT CHUNK" in result.output and "prompt: evidence" in result.output
    saved = next((qpath.parent / "results").glob("*-evidence.json"))
    assert json.loads(saved.read_text())["settings"]["prompt"] == "evidence"
    wrong = CliRunner().invoke(app, ["eval", str(qpath), "--prompt", "evidence"])
    assert wrong.exit_code == 1 and "--prompt applies to --answers runs only" in wrong.output


def test_repository_criteria_match_the_recorded_baseline() -> None:
    spec = load_criteria(REPO / "eval" / "answer-criteria-evidence-first.yaml")
    assert set(spec["thresholds"]) == {
        "unsupported_answers", "false_abstentions", "relevant_citation_rate", "graded_regressions",
    }
    baseline = json.loads((REPO / spec["baseline"]["results"]).read_text())
    metrics = run_metrics(baseline)
    recorded = spec["criteria"]
    assert metrics["unsupported_answers"]["ids"] == recorded["unsupported_answers"]["baseline"]["questions"]
    assert metrics["false_abstentions"]["ids"] == recorded["false_abstentions"]["baseline"]["questions"]
    assert metrics["relevant_citation_rate"]["count"] == recorded["relevant_citation_rate"]["baseline"]["count"]
    assert spec["thresholds"]["relevant_citation_rate"].passes(metrics["relevant_citation_rate"]["value"])
