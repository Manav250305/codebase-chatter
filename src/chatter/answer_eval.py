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

    memory           peak process RSS and system swap used while answering
                     (sampled every 0.25 s), plus the backend's own accelerator
                     memory figure where it reports one

Abstention precision/recall treats negative questions (expect_abstain) as the
positive class, scored two ways: strict (only the exact sentence counts as an
abstention) and lenient (exact, mixed, or near-miss). Context-relative
abstention looks only at questions whose context held no relevant or
containing chunk, and reports per corpus how often the model still answered.
Correctness is judged by hand: ``format_review`` renders a markdown file with
a verdict line per answer.
"""

from __future__ import annotations

import statistics
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from chatter.answer import (
    ABSTENTION,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    AnswerResult,
    AnswerStatus,
    Generator,
    answer_question,
    generator_memory,
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

ANSWER_RESULTS_SCHEMA = 2  # 2: memory, backend, context-relative abstention
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


class MemorySampler:
    """Samples process RSS and system swap in a background thread.

    Use as a context manager around one question; ``stats()`` then gives the
    peak RSS and the swap used at the start, at the peak, and at the end.
    """

    def __init__(self, interval: float = 0.25) -> None:
        self._interval = interval
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss = 0
        self._swap_start = self._swap_peak = self._swap_end = 0

    def __enter__(self) -> MemorySampler:
        self._swap_start = self._swap_peak = psutil.swap_memory().used
        self._sample()
        self._thread = threading.Thread(target=self._run, name="chatter-memory", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        self._swap_end = psutil.swap_memory().used

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample()

    def _sample(self) -> None:
        self._peak_rss = max(self._peak_rss, self._process.memory_info().rss)
        self._swap_peak = max(self._swap_peak, psutil.swap_memory().used)

    def stats(self) -> dict[str, float]:
        mb = 1e6
        return {
            "peak_rss_mb": self._peak_rss / mb,
            "swap_used_start_mb": self._swap_start / mb,
            "swap_used_peak_mb": self._swap_peak / mb,
            "swap_used_end_mb": self._swap_end / mb,
        }


class TimedGenerator:
    """Wraps a Generator to time generation and keep the raw streamed text."""

    def __init__(self, inner: Generator) -> None:
        self._inner = inner
        self.timing = _Timing()

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def backend(self) -> str | None:
        return getattr(self._inner, "backend", None)

    def memory_stats(self) -> dict[str, Any]:
        return generator_memory(self._inner)

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
    memory: Mapping[str, Any] | None = None,
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
        "memory": dict(memory or {}),
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


def context_relative_abstention(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How often the model answered when its context held no evidence.

    ``answerable``: answerable questions whose context had no relevant or
    containing chunk. ``with_negatives`` adds the negative questions, whose
    context can never hold the answer. Abstaining counts lenient
    classifications (abstained, mixed, near-miss). Reported overall and per
    corpus.
    """

    def tally(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        abstained = sum(r["classification"] in LENIENT_ABSTAIN for r in rows)
        n = len(rows)
        return {
            "n": n,
            "answered": n - abstained,
            "abstained": abstained,
            "answered_rate": (n - abstained) / n if n else None,
            "abstained_rate": abstained / n if n else None,
            "ids": [r["id"] for r in rows],
        }

    def group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        corpora = sorted({r["corpus"] for r in records})
        return {
            "overall": tally(rows),
            "by_corpus": {c: tally([r for r in rows if r["corpus"] == c]) for c in corpora},
        }

    no_evidence = [
        r for r in records if not r["expect_abstain"] and r["relevant_in_context"]["contain"] is False
    ]
    negatives = [r for r in records if r["expect_abstain"]]
    return {"answerable": group(no_evidence), "with_negatives": group(no_evidence + negatives)}


def _memory_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def values(key: str) -> list[float]:
        return [r["memory"][key] for r in records if r.get("memory", {}).get(key) is not None]

    rss, swap_peak, accel = values("peak_rss_mb"), values("swap_used_peak_mb"), values("accelerator_mb")
    starts, ends = values("swap_used_start_mb"), values("swap_used_end_mb")
    return {
        "max_peak_rss_mb": max(rss) if rss else None,
        "median_peak_rss_mb": statistics.median(rss) if rss else None,
        "max_swap_used_mb": max(swap_peak) if swap_peak else None,
        "swap_growth_mb": ends[-1] - starts[0] if starts and ends else None,
        "max_accelerator_mb": max(accel) if accel else None,
        "accelerator_metric": next(
            (r["memory"]["accelerator_metric"] for r in records if r.get("memory", {}).get("accelerator_metric")),
            None,
        ),
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
        "context_relative_abstention": context_relative_abstention(records),
        "memory": _memory_summary(records),
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
    backend: str = "transformers",
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
    load_started = time.perf_counter()
    try:
        with MemorySampler() as load_memory:
            generator = TimedGenerator(generator_factory())
    except (OSError, ValueError, RuntimeError, ImportError, MemoryError) as exc:
        raise EvalError(f"could not load the answer model: {exc}") from exc
    load = {"seconds": time.perf_counter() - load_started, **load_memory.stats()}
    backend = generator.backend or backend  # what actually ran, if the generator says
    log(f"loaded {generator.name} ({backend}) in {load['seconds']:.1f}s")

    records = []
    for n, question in enumerate(questions, 1):
        with MemorySampler() as memory:
            hits = setup.retrievers[question.corpus].search(question.question, k=top_k)
            result = answer_question(
                question.question,
                hits,
                generator,
                max_context_tokens=max_context_tokens,
                max_new_tokens=max_new_tokens,
            )
        record = evaluate_answer(
            question,
            result,
            generator.timing,
            generator,
            setup.matchers[question.corpus],
            memory={**memory.stats(), **generator.memory_stats()},
        )
        records.append(record)
        log(f"[{n}/{len(questions)}] {question.id}: {record['classification']}")

    return {
        "schema": ANSWER_RESULTS_SCHEMA,
        "kind": "answers",
        **run_metadata(questions_path, repo_root, embedder, setup.info),
        "answer_model": generator.name,
        "backend": backend,
        "model_load": load,
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
    json_path = save_results(results, results_dir, label=f"answers-{results.get('backend', 'transformers')}")
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
        f"answer model: {results['answer_model']} ({results.get('backend', 'transformers')})   "
        f"embedding model: {results['embedding_model']}   "
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
        *_memory_lines(results),
        "",
        *_context_relative_lines(s["context_relative_abstention"]),
        "",
        f"{'id':<4} {'type':<10} {'corpus':<7} {'classification':<21} {'in ctx':>7} {'cited':>7} "
        f"{'bad tags':>8} {'TTFT':>6} {'tok/s':>6} {'RSS GB':>7} {'swap GB':>8} {'accel GB':>9}",
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
            f"{_rate(r['tokens_per_s'], unit=False):>6} {_gb(r.get('memory', {}).get('peak_rss_mb')):>7} "
            f"{_gb(r.get('memory', {}).get('swap_used_peak_mb')):>8} "
            f"{_gb(r.get('memory', {}).get('accelerator_mb')):>9}"
        )
    return "\n".join(lines)


def _memory_lines(results: Mapping[str, Any]) -> list[str]:
    m = results["summary"].get("memory") or {}
    load = results.get("model_load") or {}
    lines = []
    if load:
        lines.append(
            f"model load {load['seconds']:.1f}s, RSS after load {_gb(load.get('peak_rss_mb'))} GB"
        )
    if m:
        lines.append(
            f"peak RSS {_gb(m['max_peak_rss_mb'])} GB (median {_gb(m['median_peak_rss_mb'])}), "
            f"max swap used {_gb(m['max_swap_used_mb'])} GB (growth {_gb(m['swap_growth_mb'])} GB), "
            f"accelerator {_gb(m['max_accelerator_mb'])} GB [{m['accelerator_metric'] or 'n/a'}]"
        )
    return lines


def _context_relative_lines(cra: Mapping[str, Any]) -> list[str]:
    lines = [
        "context-relative abstention (no relevant/containing chunk in context; abstained = lenient)",
        f"{'group':<17} {'corpus':<8} {'n':>3} {'answered':>9} {'abstained':>10}  questions",
    ]
    for group, label in (("answerable", "answerable"), ("with_negatives", "incl. negatives")):
        rows = [("all", cra[group]["overall"]), *cra[group]["by_corpus"].items()]
        for corpus, t in rows:
            if not t["n"]:
                lines.append(f"{label:<17} {corpus:<8} {0:>3} {'-':>9} {'-':>10}")
                continue
            lines.append(
                f"{label:<17} {corpus:<8} {t['n']:>3} {t['answered']:>3} ({t['answered_rate']:.2f}) "
                f"{t['abstained']:>3} ({t['abstained_rate']:.2f})  {', '.join(t['ids'])}"
            )
    return lines


def _gb(mb: float | None) -> str:
    return "n/a" if mb is None else f"{mb / 1000:.2f}"


def format_review(results: Mapping[str, Any]) -> str:
    """Markdown for manual correctness review, one section per question."""
    s = results["summary"]
    lines = [
        f"# Answer review: {results['settings']['split']} split",
        "",
        f"- answer model: `{results['answer_model']}` ({results.get('backend', 'transformers')}); "
        f"embedding model: `{results['embedding_model']}`",
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
            f"prompt {r['prompt_tokens']} tokens, {r['generated_tokens']} generated, "
            f"peak RSS {_gb(r.get('memory', {}).get('peak_rss_mb'))} GB, "
            f"swap {_gb(r.get('memory', {}).get('swap_used_peak_mb'))} GB_",
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
