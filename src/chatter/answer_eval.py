"""Answer-quality eval: run `ask` on every question and record what happened.

Retrieval uses the default config (the same as ``chatter ask``). Per question:
    classification   answered | abstained | mixed | near_miss_abstention
                     (near-miss: not the exact abstention sentence, but the
                     answer says the context lacks it; see looks_like_abstention)
    relevant_in_context / cited_relevant
                     whether any relevant chunk (strict) or a chunk containing
                     one (containment) was given to the model / cited by it
    invalid_tags     cited tags that match no context block
    timing           time to first token (from the start of generation),
                     generated tokens per second, prompt tokens

Abstention precision/recall treats negative questions (expect_abstain) as the
positive class, scored two ways: strict (only the exact sentence counts as an
abstention) and lenient (exact, mixed, or near-miss). Correctness is judged by
hand: ``format_review`` renders a markdown file with a verdict line per answer.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chatter.answer import (
    ABSTENTION,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    AnswerResult,
    AnswerStatus,
    Generator,
    answer_question,
    looks_like_abstention,
)
from chatter.embed import Embedder
from chatter.evaluate import (
    Contains,
    EvalError,
    Question,
    exact_match,
    load_corpora,
    open_corpora,
    run_metadata,
    save_results,
    select_questions,
)

ANSWER_RESULTS_SCHEMA = 1
DEFAULT_ANSWER_TOP_K = 8  # same as `chatter ask`
CLASSIFICATIONS = ("answered", "abstained", "mixed", "near_miss_abstention")
LENIENT_ABSTAIN = frozenset({"abstained", "mixed", "near_miss_abstention"})


def classify(result: AnswerResult) -> str:
    if result.status is AnswerStatus.ABSTAINED:
        return "abstained"
    if result.status is AnswerStatus.MIXED:
        return "mixed"
    return "near_miss_abstention" if looks_like_abstention(result.text) else "answered"


@dataclass(slots=True)
class _Timing:
    started: float | None = None
    first_token: float | None = None
    ended: float | None = None
    text: list[str] = field(default_factory=list)


class TimedGenerator:
    """Wraps a Generator to time generation and keep the raw streamed text."""

    def __init__(self, inner: Generator) -> None:
        self._inner = inner
        self.timing = _Timing()

    @property
    def name(self) -> str:
        return self._inner.name

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        return self._inner.count_tokens(texts)

    def prompt_tokens(self, system: str, user: str) -> int:
        return self._inner.prompt_tokens(system, user)

    def generate(self, system: str, user: str, *, max_new_tokens: int) -> Iterator[str]:
        self.timing = timing = _Timing(started=time.perf_counter())
        stream = self._inner.generate(system, user, max_new_tokens=max_new_tokens)
        try:
            for piece in stream:
                if timing.first_token is None and piece.strip():
                    timing.first_token = time.perf_counter()
                timing.text.append(piece)
                yield piece
        finally:
            timing.ended = time.perf_counter()
            close = getattr(stream, "close", None)
            if close is not None:
                close()


def evaluate_answer(
    question: Question,
    result: AnswerResult,
    timing: _Timing,
    generator: Generator,
    contains: Contains,
) -> dict[str, Any]:
    """One question's record (see module docstring)."""
    relevant = list(question.relevant)
    context_ids = [block.chunk_id for block in result.blocks]
    cited_ids = [citation.block.chunk_id for citation in result.citations]

    def any_match(ids: Sequence[str], match: Contains) -> bool | None:
        if not relevant:
            return None
        return any(match(cid, rel) for cid in ids for rel in relevant)

    raw = "".join(timing.text)
    generated = generator.count_tokens([raw])[0] if raw else 0
    ttft = (
        timing.first_token - timing.started
        if timing.first_token is not None and timing.started is not None
        else None
    )
    decode_time = (
        timing.ended - timing.first_token
        if timing.ended is not None and timing.first_token is not None
        else None
    )
    return {
        "id": question.id,
        "corpus": question.corpus,
        "type": question.type,
        "split": question.split,
        "question": question.question,
        "expect_abstain": question.expect_abstain,
        "relevant": relevant,
        "classification": classify(result),
        "answer": result.text,
        "stopped_for_repetition": result.stopped_for_repetition,
        "context": [
            {"tag": b.tag, "chunk_id": b.chunk_id, "location": b.location, "truncated": b.truncated}
            for b in result.blocks
        ],
        "dropped_chunks": [hit.chunk_id for hit in result.dropped],
        "relevant_in_context": {
            "strict": any_match(context_ids, exact_match),
            "contain": any_match(context_ids, contains),
        },
        "citations": [
            {"tag": c.tag, "chunk_id": c.block.chunk_id, "location": c.block.location}
            for c in result.citations
        ],
        "invalid_tags": list(result.unknown_tags),
        "cited_relevant": {
            "strict": any_match(cited_ids, exact_match),
            "contain": any_match(cited_ids, contains),
        },
        "prompt_tokens": result.prompt_tokens,
        "generated_tokens": generated,
        "ttft_s": ttft,
        "tokens_per_s": generated / decode_time if decode_time and decode_time > 0 else None,
    }


def abstention_scores(records: Sequence[Mapping[str, Any]], abstain: frozenset[str]) -> dict[str, Any]:
    """Precision/recall with negatives (expect_abstain) as the positive class."""
    negatives = [r for r in records if r["expect_abstain"]]
    answerable = [r for r in records if not r["expect_abstain"]]
    tp = sum(r["classification"] in abstain for r in negatives)
    fp = sum(r["classification"] in abstain for r in answerable)
    fn = len(negatives) - tp
    predicted = tp + fp
    return {
        "negatives": len(negatives),
        "answerable": len(answerable),
        "true_abstentions": tp,
        "false_abstentions": fp,
        "missed_abstentions": fn,
        "precision": tp / predicted if predicted else None,
        "recall": tp / len(negatives) if negatives else None,
        "false_abstention_rate": fp / len(answerable) if answerable else None,
    }


def summarize_answers(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    answerable = [r for r in records if not r["expect_abstain"]]

    def rate(key: str, kind: str) -> float | None:
        values = [r[key][kind] for r in answerable if r[key][kind] is not None]
        return sum(values) / len(values) if values else None

    def median(key: str) -> float | None:
        values = [r[key] for r in records if r[key] is not None]
        return statistics.median(values) if values else None

    return {
        "n": len(records),
        "classifications": {c: sum(r["classification"] == c for r in records) for c in CLASSIFICATIONS},
        "abstention": {
            "strict": abstention_scores(records, frozenset({"abstained"})),
            "lenient": abstention_scores(records, LENIENT_ABSTAIN),
        },
        "answerable": {
            "n": len(answerable),
            "relevant_in_context": {k: rate("relevant_in_context", k) for k in ("strict", "contain")},
            "cited_relevant": {k: rate("cited_relevant", k) for k in ("strict", "contain")},
        },
        "invalid_tags": sum(len(r["invalid_tags"]) for r in records),
        "answers_with_invalid_tags": sum(bool(r["invalid_tags"]) for r in records),
        "stopped_for_repetition": sum(r["stopped_for_repetition"] for r in records),
        "median_ttft_s": median("ttft_s"),
        "median_tokens_per_s": median("tokens_per_s"),
        "median_prompt_tokens": median("prompt_tokens"),
    }


def run_answer_eval(
    questions_path: Path,
    corpora_path: Path,
    embedder_factory: Callable[[str], Embedder],
    generator_factory: Callable[[], Generator],
    *,
    model_name: str,
    repo_root: Path,
    stdlib_dir: Path | None = None,
    index_root: Path | None = None,
    split: str | None = None,
    top_k: int = DEFAULT_ANSWER_TOP_K,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    log: Callable[[str], None] = lambda _: None,
) -> dict[str, Any]:
    """Answer every selected question with the default retrieval config."""
    questions = select_questions(questions_path, split)
    embedder = embedder_factory(model_name)
    setup = open_corpora(
        questions,
        load_corpora(corpora_path),
        embedder,
        repo_root=repo_root,
        stdlib_dir=stdlib_dir,
        index_root=index_root,
        log=log,
    )
    try:
        generator = TimedGenerator(generator_factory())
    except (OSError, ValueError, RuntimeError, ImportError, MemoryError) as exc:
        raise EvalError(f"could not load the answer model: {exc}") from exc

    records = []
    for n, question in enumerate(questions, 1):
        hits = setup.retrievers[question.corpus].search(question.question, k=top_k)
        result = answer_question(
            question.question,
            hits,
            generator,
            max_context_tokens=max_context_tokens,
            max_new_tokens=max_new_tokens,
        )
        record = evaluate_answer(
            question, result, generator.timing, generator, setup.matchers[question.corpus]
        )
        records.append(record)
        log(f"[{n}/{len(questions)}] {question.id}: {record['classification']}")

    return {
        "schema": ANSWER_RESULTS_SCHEMA,
        "kind": "answers",
        **run_metadata(questions_path, repo_root, embedder, setup.info),
        "answer_model": generator.name,
        "settings": {
            "split": split or "all",
            "retrieval": "default",
            "top_k": top_k,
            "max_context_tokens": max_context_tokens,
            "max_new_tokens": max_new_tokens,
            "abstention_sentence": ABSTENTION,
        },
        "summary": summarize_answers(records),
        "questions": records,
    }


def save_answer_results(results: Mapping[str, Any], results_dir: Path) -> tuple[Path, Path]:
    """Write the JSON results and the markdown review file next to it."""
    json_path = save_results(results, results_dir, label="answers")
    review_path = json_path.with_suffix(".md")
    review_path.write_text(format_review(results), encoding="utf-8")
    return json_path, review_path


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def format_answer_report(results: Mapping[str, Any]) -> str:
    s = results["summary"]
    lines = [
        f"answer model: {results['answer_model']}   embedding model: {results['embedding_model']}   "
        f"commit: {(results['git'].get('commit') or 'none')[:10]}{' (dirty)' if results['git'].get('dirty') else ''}",
        f"split: {results['settings']['split']}   questions: {s['n']}   retrieval: default fused, "
        f"top {results['settings']['top_k']}",
        "",
        "classifications: " + ", ".join(f"{k} {v}" for k, v in s["classifications"].items()),
        "",
        f"{'abstention':<11} {'neg':>4} {'ans':>4} {'TP':>3} {'FP':>3} {'FN':>3} {'precision':>10} "
        f"{'recall':>7} {'false-abst':>11}",
    ]
    for kind in ("strict", "lenient"):
        a = s["abstention"][kind]
        lines.append(
            f"{kind:<11} {a['negatives']:>4} {a['answerable']:>4} {a['true_abstentions']:>3} "
            f"{a['false_abstentions']:>3} {a['missed_abstentions']:>3} {_pct(a['precision']):>10} "
            f"{_pct(a['recall']):>7} {_pct(a['false_abstention_rate']):>11}"
        )
    ans = s["answerable"]
    lines += [
        "",
        f"answerable questions: {ans['n']}   (strict/containment)",
        f"  relevant chunk in context: {_pct(ans['relevant_in_context']['strict'])}/"
        f"{_pct(ans['relevant_in_context']['contain'])}",
        f"  answer cites a relevant chunk: {_pct(ans['cited_relevant']['strict'])}/"
        f"{_pct(ans['cited_relevant']['contain'])}",
        f"invalid citation tags: {s['invalid_tags']} in {s['answers_with_invalid_tags']} answers   "
        f"repetition stops: {s['stopped_for_repetition']}",
        f"median TTFT {_seconds(s['median_ttft_s'])}, median {_rate(s['median_tokens_per_s'])}, "
        f"median prompt {s['median_prompt_tokens']} tokens",
        "",
        f"{'id':<4} {'type':<10} {'corpus':<7} {'classification':<21} {'in ctx':>7} {'cited':>7} "
        f"{'bad tags':>8} {'TTFT':>6} {'tok/s':>6}",
    ]
    for r in results["questions"]:
        def pair(key: str, r: Mapping[str, Any] = r) -> str:
            v = r[key]
            if v["strict"] is None:
                return "n/a"
            return f"{'y' if v['strict'] else 'n'}/{'y' if v['contain'] else 'n'}"

        lines.append(
            f"{r['id']:<4} {r['type']:<10} {r['corpus']:<7} {r['classification']:<21} "
            f"{pair('relevant_in_context'):>7} {pair('cited_relevant'):>7} "
            f"{','.join(r['invalid_tags']) or '-':>8} {_seconds(r['ttft_s']):>6} "
            f"{_rate(r['tokens_per_s'], unit=False):>6}"
        )
    return "\n".join(lines)


def format_review(results: Mapping[str, Any]) -> str:
    """Markdown for manual correctness review, one section per question."""
    s = results["summary"]
    lines = [
        f"# Answer review: {results['settings']['split']} split",
        "",
        f"- answer model: `{results['answer_model']}`; embedding model: `{results['embedding_model']}`",
        f"- commit: `{results['git'].get('commit')}`{' (dirty)' if results['git'].get('dirty') else ''}; "
        f"created {results['created']}",
        f"- retrieval: default fused, top {results['settings']['top_k']}, "
        f"max context {results['settings']['max_context_tokens']} tokens",
        "- classifications: " + ", ".join(f"{k} {v}" for k, v in s["classifications"].items()),
        "",
        "Mark one verdict per question: **correct**, **partial**, **wrong**, or for negatives "
        "**correctly abstained** / **should have abstained**. Near-miss abstentions are flagged by a "
        "heuristic; check them.",
    ]
    for r in results["questions"]:
        expected = "should abstain" if r["expect_abstain"] else "answerable"
        lines += [
            "",
            "---",
            "",
            f"## {r['id']} ({r['type']}, {r['corpus']}, {expected})",
            "",
            f"**Question:** {r['question']}",
            "",
            f"**Classification:** {r['classification']}"
            + (" (generation stopped: repetition)" if r["stopped_for_repetition"] else ""),
            "",
            "**Answer:**",
            "",
            *(f"> {line}" if line else ">" for line in (r["answer"] or "(empty)").split("\n")),
            "",
        ]
        if r["citations"]:
            lines.append("**Citations:**")
            lines += [f"- [{c['tag']}] `{c['chunk_id']}` {c['location']}" for c in r["citations"]]
        else:
            lines.append("**Citations:** none")
        if r["invalid_tags"]:
            lines.append(f"- invalid tags: {', '.join(r['invalid_tags'])}")
        lines += ["", "**Context given to the model:**"]
        lines += [
            f"- [{c['tag']}] `{c['chunk_id']}` {c['location']}{' (truncated)' if c['truncated'] else ''}"
            for c in r["context"]
        ]
        if r["relevant"]:
            lines += ["", "**Relevant (eval set):** " + ", ".join(f"`{x}`" for x in r["relevant"])]
            ctx, cited = r["relevant_in_context"], r["cited_relevant"]
            lines.append(
                f"- relevant in context: strict {ctx['strict']}, containment {ctx['contain']}; "
                f"cited: strict {cited['strict']}, containment {cited['contain']}"
            )
        lines += [
            "",
            f"_TTFT {_seconds(r['ttft_s'])}, {_rate(r['tokens_per_s'])}, "
            f"prompt {r['prompt_tokens']} tokens, {r['generated_tokens']} generated_",
            "",
            "**Verdict:** _(correct / partial / wrong / correctly abstained / should have abstained)_",
            "",
            "**Notes:**",
        ]
    return "\n".join(lines) + "\n"


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}s"


def _rate(value: float | None, *, unit: bool = True) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f} tok/s" if unit else f"{value:.1f}"
