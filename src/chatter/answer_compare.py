"""Compare two answer-eval runs against a recorded success criterion.

Grades are your verdicts on a run's review file (``format_review``):
``extract_grades`` reads the ``**Verdict:**`` lines into
``eval/grades/<results stem>.yaml``::

    results: eval/results/<stem>.json
    grades:
      q01: correct
      q15: correctly abstained

The criterion file (e.g. eval/answer-criteria-evidence-first.yaml) names the
criteria and their thresholds (``pass_if``: ``<= 2``, ``>= 26/36``, ``== 0``).
The definitions live here, keyed by criterion name, and are scored on each
run's final output:

    unsupported_answers     answerable, no relevant or containing chunk in
                            context, final classification answered
    false_abstentions       answerable, final classification abstained
    relevant_citation_rate  answerable questions whose output cites a relevant
                            chunk (strict), over all answerable questions
    graded_regressions      graded correct in the base run, and in the new run
                            abstained or graded anything but correct

The comparison is void unless both runs answered the same questions from the
same context (chunk ids and shown lines).
"""

from __future__ import annotations

import operator
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml

from chatter.evaluate import EvalError

GRADES = ("correct", "partial", "wrong", "correctly abstained", "should have abstained")
CRITERIA = ("unsupported_answers", "false_abstentions", "relevant_citation_rate", "graded_regressions")
PASS, FAIL, PENDING = "PASS", "FAIL", "PENDING"

_SECTION = re.compile(r"^## (\S+) \(", re.MULTILINE)
_VERDICT = re.compile(r"^\*\*Verdict:\*\*(.*)$", re.MULTILINE)
_PLACEHOLDER = "correct / partial / wrong"
_THRESHOLD = re.compile(r"^\s*(<=|>=|==|<|>)\s*(\d+)(?:\s*/\s*(\d+))?\s*$")
_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "<=": operator.le, ">=": operator.ge, "==": operator.eq, "<": operator.lt, ">": operator.gt,
}


# ---------------------------------------------------------------------------
# Grades
# ---------------------------------------------------------------------------


def extract_grades(review: str) -> tuple[dict[str, str], list[str], list[str]]:
    """(grades, ungraded question ids, problems) from a filled-in review file."""
    grades: dict[str, str] = {}
    ungraded: list[str] = []
    problems: list[str] = []
    sections = list(_SECTION.finditer(review))
    for n, section in enumerate(sections):
        qid = section.group(1)
        end = sections[n + 1].start() if n + 1 < len(sections) else len(review)
        verdict = _VERDICT.search(review, section.end(), end)
        if verdict is None:
            problems.append(f"{qid}: no **Verdict:** line")
            continue
        value = _normalize_grade(verdict.group(1))
        grade = _grade(value)
        if not value or _PLACEHOLDER in value:
            ungraded.append(qid)
        elif grade is not None:
            grades[qid] = grade
        else:
            problems.append(f"{qid}: unknown verdict {verdict.group(1).strip()!r} (expected one of {', '.join(GRADES)})")
    return grades, ungraded, problems


def _grade(value: str) -> str | None:
    """A grade, allowing a trailing note ("partial - misses the retry path")."""
    for grade in sorted(GRADES, key=len, reverse=True):
        if value == grade or (value.startswith(grade) and value[len(grade)] in " -—–:,;("):
            return grade
    return None


def _normalize_grade(text: str) -> str:
    text = re.sub(r"[*_`]", "", text).strip().lower()
    return " ".join(text.split()).strip(" .")


def grades_path(results_path: Path) -> Path:
    """eval/results/<stem>.json -> eval/grades/<stem>.yaml"""
    return results_path.parent.parent / "grades" / f"{results_path.stem}.yaml"


def write_grades(path: Path, results_path: Path, grades: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"results": results_path.as_posix(), "grades": dict(grades)}
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def load_grades(path: Path) -> dict[str, str]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    grades = data.get("grades") if isinstance(data, dict) else None
    if not isinstance(grades, dict):
        raise EvalError(f"{path}: expected a mapping with 'grades'")
    bad = {q: g for q, g in grades.items() if g not in GRADES}
    if bad:
        raise EvalError(f"{path}: unknown grades {bad} (expected one of {', '.join(GRADES)})")
    return {str(q): str(g) for q, g in grades.items()}


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Threshold:
    op: str
    value: Fraction
    text: str

    def passes(self, value: Fraction) -> bool:
        return _OPS[self.op](value, self.value)


def parse_threshold(text: str) -> Threshold:
    match = _THRESHOLD.match(str(text))
    if match is None:
        raise EvalError(f"bad pass_if {text!r}: expected e.g. '<= 2', '>= 26/36', '== 0'")
    op, num, den = match.groups()
    return Threshold(op, Fraction(int(num), int(den) if den else 1), str(text).strip())


def load_criteria(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    criteria = data.get("criteria") if isinstance(data, dict) else None
    if not isinstance(criteria, dict) or not criteria:
        raise EvalError(f"{path}: expected a 'criteria' mapping")
    unknown = sorted(set(criteria) - set(CRITERIA))
    if unknown:
        raise EvalError(f"{path}: unknown criteria {unknown} (known: {', '.join(CRITERIA)})")
    thresholds = {name: parse_threshold(spec.get("pass_if", "")) for name, spec in criteria.items()}
    return {**data, "thresholds": thresholds}


def _answerable(results: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [r for r in results["questions"] if not r["expect_abstain"]]


def run_metrics(results: Mapping[str, Any], *, raw: bool = False) -> dict[str, Any]:
    """The classification-based criteria for one run (``raw``: before withholding)."""
    key = "raw_classification" if raw else "classification"
    answerable = _answerable(results)

    def cls(r: Mapping[str, Any]) -> str:
        return str(r.get(key) or r["classification"])

    unsupported = [r["id"] for r in answerable if r["relevant_in_context"]["contain"] is False and cls(r) == "answered"]
    abstained = [r["id"] for r in answerable if cls(r) == "abstained"]
    cited = [r["id"] for r in answerable if r["cited_relevant"]["strict"]]
    return {
        "unsupported_answers": {"value": Fraction(len(unsupported)), "ids": unsupported},
        "false_abstentions": {"value": Fraction(len(abstained)), "ids": abstained},
        "relevant_citation_rate": {
            "value": Fraction(len(cited), len(answerable)) if answerable else Fraction(0),
            "count": len(cited),
            "of": len(answerable),
        },
    }


def context_differences(base: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """Questions whose context (chunk ids and shown lines) differs, or that only one run has."""

    def contexts(results: Mapping[str, Any]) -> dict[str, list[tuple[str, str]]]:
        return {r["id"]: [(c["chunk_id"], c["location"]) for c in r["context"]] for r in results["questions"]}

    a, b = contexts(base), contexts(new)
    return sorted(q for q in set(a) | set(b) if a.get(q) != b.get(q))


def graded_regressions(
    base: Mapping[str, Any],
    new: Mapping[str, Any],
    base_grades: Mapping[str, str] | None,
    new_grades: Mapping[str, str] | None,
) -> dict[str, Any]:
    if base_grades is None:
        return {"status": PENDING, "reason": "no grades for the base run", "ids": [], "ungraded": []}
    new_class = {r["id"]: r["classification"] for r in new["questions"]}
    correct = sorted(q for q, g in base_grades.items() if g == "correct")
    regressions, ungraded = [], []
    for qid in correct:
        if new_class.get(qid) == "abstained":
            regressions.append(qid)
        elif new_grades is None or qid not in new_grades:
            ungraded.append(qid)
        elif new_grades[qid] != "correct":
            regressions.append(qid)
    return {"base_correct": correct, "ids": regressions, "ungraded": ungraded}


def compare_answers(
    base: Mapping[str, Any],
    new: Mapping[str, Any],
    criteria: Mapping[str, Any],
    *,
    base_grades: Mapping[str, str] | None = None,
    new_grades: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Score each criterion; the decision is PASS only if all pass and the comparison is valid."""
    differing = context_differences(base, new)
    base_m, new_m, raw_m = run_metrics(base), run_metrics(new), run_metrics(new, raw=True)
    rows: list[dict[str, Any]] = []
    for name, threshold in criteria["thresholds"].items():
        row: dict[str, Any] = {"criterion": name, "pass_if": threshold.text}
        if name == "graded_regressions":
            graded = graded_regressions(base, new, base_grades, new_grades)
            pending = "reason" in graded or bool(graded["ungraded"])
            row |= {"graded": graded}
            if "reason" in graded:
                row |= {"status": PENDING}
            else:
                value = Fraction(len(graded["ids"]))
                failed = not threshold.passes(value)  # regressions found already fail it
                row |= {"new": value, "status": FAIL if failed else (PENDING if pending else PASS)}
        else:
            row |= {
                "base": base_m[name]["value"],
                "new": new_m[name]["value"],
                "raw": raw_m[name]["value"] if name != "relevant_citation_rate" else None,
                "base_detail": base_m[name],
                "new_detail": new_m[name],
                "status": PASS if threshold.passes(new_m[name]["value"]) else FAIL,
            }
        rows.append(row)
    statuses = {row["status"] for row in rows}
    decision = "VOID" if differing else FAIL if FAIL in statuses else PENDING if PENDING in statuses else PASS
    changed = [
        (r["id"], b["classification"], r["classification"])
        for r, b in zip(new["questions"], base["questions"])
        if r["id"] == b["id"] and r["classification"] != b["classification"]
    ]
    return {
        "name": criteria.get("name"),
        "decision": decision,
        "context_differences": differing,
        "criteria": rows,
        "evidence": new.get("summary", {}).get("evidence"),
        "classification_changes": changed,
    }


def _fmt(value: Fraction | None, name: str, detail: Mapping[str, Any] | None = None) -> str:
    if value is None:
        return "-"
    if name == "relevant_citation_rate" and detail is not None:
        return f"{detail['count']}/{detail['of']} ({float(value):.3f})"
    return str(value) if value.denominator == 1 else f"{float(value):.3f}"


def format_comparison(outcome: Mapping[str, Any], base_label: str, new_label: str) -> str:
    lines = [f"answer comparison: {outcome['name'] or 'criteria'}", f"  base: {base_label}", f"  new:  {new_label}", ""]
    if outcome["context_differences"]:
        lines += [
            "VOID: the runs answered from different context for "
            + ", ".join(outcome["context_differences"]),
            "",
        ]
    e = outcome.get("evidence")
    if e:
        rel, fmt = e["in_relevant"], e["format"]
        rate = f"{rel['rate_of_answers']:.2f}" if rel["rate_of_answers"] is not None else "n/a"
        lines += [
            f"EVIDENCE IN A RELEVANT CHUNK (new): {rel['count']}/{rel['answers']} answers ({rate}); "
            f"{rel['count']}/{rel['answerable']} answerable questions",
            f"format compliance (new): {fmt['parseable']}/{e['n']} parseable Evidence section, "
            f"{fmt['abstention_only']} abstention only, {fmt['neither']} neither"
            + (f" ({', '.join(fmt['neither_ids'])})" if fmt["neither_ids"] else ""),
            "answers withheld as unverified: " + (", ".join(e["forced_abstentions"]) or "none"),
            "",
        ]
    lines.append(f"{'criterion':<24} {'base':>14} {'new':>14} {'new (raw)':>10} {'pass if':>9}  status")
    for row in outcome["criteria"]:
        name = row["criterion"]
        if name == "graded_regressions":
            g = row["graded"]
            new = _fmt(row.get("new"), name)
            note = g.get("reason") or (
                f"regressions: {', '.join(g['ids']) or 'none'}"
                + (f"; ungraded in new run: {', '.join(g['ungraded'])}" if g["ungraded"] else "")
            )
            lines.append(f"{name:<24} {'-':>14} {new:>14} {'-':>10} {row['pass_if']:>9}  {row['status']}  ({note})")
            continue
        lines.append(
            f"{name:<24} {_fmt(row['base'], name, row['base_detail']):>14} "
            f"{_fmt(row['new'], name, row['new_detail']):>14} {_fmt(row['raw'], name):>10} "
            f"{row['pass_if']:>9}  {row['status']}"
        )
        ids = row["new_detail"].get("ids")
        if ids is not None:
            lines.append(f"{'':<24} base: {', '.join(row['base_detail']['ids']) or 'none'}; new: {', '.join(ids) or 'none'}")
    if outcome["classification_changes"]:
        lines += ["", "classification changes: " + ", ".join(f"{q} {a} -> {b}" for q, a, b in outcome["classification_changes"])]
    lines += ["", f"decision: {outcome['decision']}"]
    return "\n".join(lines)


def describe_grades(grades: Mapping[str, str], ungraded: Sequence[str]) -> str:
    counts = {g: sum(v == g for v in grades.values()) for g in GRADES}
    text = ", ".join(f"{g} {n}" for g, n in counts.items() if n)
    return f"{len(grades)} graded ({text or 'none'})" + (f"; ungraded: {', '.join(ungraded)}" if ungraded else "")
