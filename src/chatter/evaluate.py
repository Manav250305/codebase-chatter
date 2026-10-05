"""Retrieval evaluation: hit@1, hit@5, MRR@50 and recall@10 per retrieval mode.

Questions come from a YAML list (see eval/questions.yaml). Each names a
corpus; corpora are materialised from a manifest (eval/corpora.yaml) as either
a copy of stdlib packages from the running Python or a ``git archive`` of a
tag, then indexed in place with the requested embedding model.

Metrics (per question, per mode; ``first`` = best rank of any relevant chunk):
    hit@1      first == 1
    hit@5      first <= 5
    MRR@50     1 / first if first <= 50, else 0
    recall@10  |relevant within top 10| / |relevant|
    test@5     top-5 slots taken by test chunks (tests/, test_*.py, conftest.py)

Each metric is computed twice. Strict: a retrieved chunk matches a relevant
chunk only if the ids are equal. Containment: it also matches if it is in the
same file and its lines include all of the relevant chunk's lines (a class
containing a relevant method, a function containing a relevant nested one).

Negative questions (``type: negative``) are excluded from every aggregate;
their top results are still recorded. Every ``relevant`` id must exist in its
corpus's index, otherwise evaluation fails before scoring anything.

Splits: each question has ``split: tune`` (the default when absent) or
``split: heldout``. Aggregates are reported per split, and also combined in
the JSON for comparison with earlier runs. Tune on ``tune`` only; look at
``heldout`` to confirm, not to choose.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import io
import json
import shutil
import subprocess
import sys
import sysconfig
import tarfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from chatter.embed import Embedder
from chatter.extract import Chunk
from chatter.graph import is_test_path, read_graph_stats
from chatter.index import (
    SCHEMA_VERSION,
    build_index,
    chunk_ids_in_index,
    default_index_dir,
    index_data,
    read_manifest,
)
from chatter.retrieve import (
    BM25,
    DENSE,
    FUSED,
    MODES,
    RRF_K,
    GraphExpansion,
    Retriever,
    reciprocal_rank_fusion,
)

RESULTS_SCHEMA = 4  # 2: splits; 3: strict/containment; 4: "configs" replaces "fusion_sweep"
DEPTH = 50  # MRR cutoff and ranking depth recorded per question
CANDIDATES = max(4 * DEPTH, 50)  # per-retriever depth, as Retriever.search uses for k=DEPTH
SWEEP_RRF_K = (5, 10, 20, 60)
SWEEP_DENSE_WEIGHT = (1.0, 1.5, 2.0, 3.0)
QUESTION_TYPES = ("exact_name", "behavior", "location", "flow", "negative")
NEGATIVE = "negative"
SPLITS = ("tune", "heldout")
DEFAULT_SPLIT = "tune"
CORPUS_MARKER = ".corpus.json"


class EvalError(RuntimeError):
    """The eval inputs are inconsistent; nothing was scored."""


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    corpus: str
    type: str
    question: str
    relevant: tuple[str, ...]
    expect_abstain: bool
    split: str = DEFAULT_SPLIT

    @property
    def scored(self) -> bool:
        return self.type != NEGATIVE


@dataclass(frozen=True, slots=True)
class CorpusSpec:
    name: str
    path: Path
    source: Mapping[str, Any]


Contains = Callable[[str, str], bool]  # (retrieved id, relevant id) -> match


@dataclass(frozen=True, slots=True)
class Metrics:
    first_rank: int | None
    hit1: bool
    hit5: bool
    rr: float
    recall10: float


@dataclass(frozen=True, slots=True)
class Scores:
    strict: Metrics
    contain: Metrics
    test_slots5: int
    top: tuple[str, ...]  # ranked chunk ids, up to DEPTH


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def load_questions(path: Path) -> list[Question]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise EvalError(f"{path}: expected a YAML list of questions")
    problems: list[str] = []
    questions: list[Question] = []
    seen: set[str] = set()
    for n, item in enumerate(data, 1):
        where = f"{path} item {n}"
        if not isinstance(item, dict):
            problems.append(f"{where}: not a mapping")
            continue
        missing = [k for k in ("id", "corpus", "type", "question") if not item.get(k)]
        if missing:
            problems.append(f"{where}: missing {', '.join(missing)}")
            continue
        qid = str(item["id"])
        if qid in seen:
            problems.append(f"{qid}: duplicate id")
        seen.add(qid)
        qtype = str(item["type"])
        relevant = tuple(str(r) for r in (item.get("relevant") or []))
        if qtype not in QUESTION_TYPES:
            problems.append(f"{qid}: unknown type {qtype!r}")
        elif qtype == NEGATIVE and relevant:
            problems.append(f"{qid}: negative questions must have no relevant ids")
        elif qtype != NEGATIVE and not relevant:
            problems.append(f"{qid}: no relevant ids")
        split = str(item.get("split", DEFAULT_SPLIT))
        if split not in SPLITS:
            problems.append(f"{qid}: unknown split {split!r} (expected one of {', '.join(SPLITS)})")
        questions.append(
            Question(
                id=qid,
                corpus=str(item["corpus"]),
                type=qtype,
                question=str(item["question"]),
                relevant=relevant,
                expect_abstain=bool(item.get("expect_abstain", False)),
                split=split,
            )
        )
    if problems:
        raise EvalError("invalid questions:\n  " + "\n  ".join(problems))
    return questions


def load_corpora(path: Path) -> dict[str, CorpusSpec]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    base = Path(path).resolve().parent
    specs: dict[str, CorpusSpec] = {}
    for name, entry in data.items():
        if not isinstance(entry, dict) or "path" not in entry or "source" not in entry:
            raise EvalError(f"{path}: corpus {name!r} needs 'path' and 'source'")
        specs[str(name)] = CorpusSpec(str(name), base / entry["path"], dict(entry["source"]))
    return specs


def validate_relevant(questions: Sequence[Question], corpus_ids: Mapping[str, set[str]]) -> None:
    """Fail loudly if any relevant id is absent from its corpus index."""
    problems = [
        f"{q.id}: {rid} not in corpus {q.corpus!r}"
        for q in questions
        for rid in q.relevant
        if rid not in corpus_ids.get(q.corpus, set())
    ]
    if problems:
        raise EvalError(
            "relevant ids missing from the index (stale ids or wrong corpus):\n  "
            + "\n  ".join(problems)
        )


# ---------------------------------------------------------------------------
# Corpus materialisation
# ---------------------------------------------------------------------------


def prepare_corpus(
    spec: CorpusSpec, *, repo_root: Path, stdlib_dir: Path | None = None
) -> dict[str, Any]:
    """Materialise ``spec.path`` if missing or stale; return provenance info."""
    kind = spec.source.get("kind")
    if kind == "python_stdlib":
        info = _stdlib_provenance(spec, stdlib_dir)
        if _marker(spec.path) != info:
            _reset(spec.path)
            for package in spec.source["packages"]:
                shutil.copytree(
                    Path(info["stdlib_dir"]) / package,
                    spec.path / package,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
            _write_marker(spec.path, info)
        return info
    if kind == "git_archive":
        ref = str(spec.source["ref"])
        commit = _git(repo_root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
        if not commit:
            raise EvalError(f"corpus {spec.name!r}: git ref {ref!r} not found in {repo_root}")
        info = {"kind": kind, "ref": ref, "commit": commit}
        if _marker(spec.path) != info:
            _reset(spec.path)
            archive = subprocess.run(
                ["git", "archive", "--format=tar", commit],
                cwd=repo_root,
                check=True,
                capture_output=True,
            ).stdout
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(spec.path, filter="data")
            _write_marker(spec.path, info)
        return info
    raise EvalError(f"corpus {spec.name!r}: unknown source kind {kind!r}")


def _stdlib_provenance(spec: CorpusSpec, stdlib_dir: Path | None) -> dict[str, Any]:
    wanted = str(spec.source.get("python_version", ""))
    running = f"{sys.version_info.major}.{sys.version_info.minor}"
    if stdlib_dir is None and wanted and wanted != running:
        raise EvalError(
            f"corpus {spec.name!r} needs Python {wanted} stdlib; running Python {running}"
        )
    directory = stdlib_dir or Path(sysconfig.get_paths()["stdlib"])
    return {
        "kind": "python_stdlib",
        "python_version": wanted or running,
        "python_full_version": sys.version.split()[0],
        "packages": list(spec.source["packages"]),
        "stdlib_dir": str(directory),
    }


def _marker(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads((path / CORPUS_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_marker(path: Path, info: Mapping[str, Any]) -> None:
    (path / CORPUS_MARKER).write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")


def _reset(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def is_test_chunk(chunk_id: str) -> bool:
    return is_test_path(chunk_id.split("::", 1)[0])


def exact_match(retrieved: str, relevant: str) -> bool:
    return retrieved == relevant


def containment(lookup: Callable[[str], Chunk | None]) -> Contains:
    """Match when ``retrieved`` covers every line of ``relevant`` in the same file."""

    @functools.cache
    def lines(chunk_id: str) -> tuple[str, frozenset[int]] | None:
        chunk = lookup(chunk_id)
        return None if chunk is None else (chunk.path, frozenset(chunk.line_numbers()))

    def contains(retrieved: str, relevant: str) -> bool:
        if retrieved == relevant:
            return True
        outer, inner = lines(retrieved), lines(relevant)
        return (
            outer is not None
            and inner is not None
            and outer[0] == inner[0]
            and inner[1] <= outer[1]
        )

    return contains


def score_ranking(
    ranked: Sequence[str], relevant: Sequence[str], contains: Contains | None = None
) -> Scores:
    top = tuple(ranked[:DEPTH])
    return Scores(
        strict=_metrics(top, relevant, exact_match),
        contain=_metrics(top, relevant, contains or exact_match),
        test_slots5=sum(is_test_chunk(cid) for cid in top[:5]),
        top=top,
    )


def _metrics(top: Sequence[str], relevant: Sequence[str], match: Contains) -> Metrics:
    wanted = list(dict.fromkeys(relevant))
    first = next(
        (i for i, cid in enumerate(top, 1) if any(match(cid, rel) for rel in wanted)), None
    )
    found = sum(1 for rel in wanted if any(match(cid, rel) for cid in top[:10]))
    return Metrics(
        first_rank=first,
        hit1=first == 1,
        hit5=first is not None and first <= 5,
        rr=1.0 / first if first is not None else 0.0,
        recall10=found / len(wanted) if wanted else 0.0,
    )


def aggregate(rows: Sequence[tuple[Question, Scores]]) -> dict[str, Any]:
    """Mean strict and containment metrics over scored (non-negative) questions."""
    scored = [s for q, s in rows if q.scored]
    n = len(scored)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "strict": _mean_metrics([s.strict for s in scored]),
        "contain": _mean_metrics([s.contain for s in scored]),
        "test_slots@5": sum(s.test_slots5 for s in scored),
        "slots@5": sum(min(5, len(s.top)) for s in scored),
    }


def _mean_metrics(metrics: Sequence[Metrics]) -> dict[str, float]:
    n = len(metrics)
    return {
        "hit@1": sum(m.hit1 for m in metrics) / n,
        "hit@5": sum(m.hit5 for m in metrics) / n,
        "mrr@50": sum(m.rr for m in metrics) / n,
        "recall@10": sum(m.recall10 for m in metrics) / n,
    }


def summarize(rows: Sequence[tuple[Question, Scores]]) -> dict[str, Any]:
    """Overall and per-type aggregates, combined and for each split."""
    return {
        **_summarize_group(rows),
        "by_split": {
            split: _summarize_group([(q, s) for q, s in rows if q.split == split])
            for split in SPLITS
        },
    }


def _summarize_group(rows: Sequence[tuple[Question, Scores]]) -> dict[str, Any]:
    by_type = {
        qtype: aggregate([(q, s) for q, s in rows if q.type == qtype])
        for qtype in QUESTION_TYPES
        if qtype != NEGATIVE and any(q.type == qtype for q, _ in rows)
    }
    return {"overall": aggregate(rows), "by_type": by_type}


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CorpusSetup:
    retrievers: dict[str, Retriever]
    matchers: dict[str, Contains]
    info: dict[str, Any]


def select_questions(questions_path: Path, split: str | None) -> list[Question]:
    """Load questions, keeping one split (None: all)."""
    if split is not None and split not in SPLITS:
        raise EvalError(f"unknown split {split!r} (expected one of {', '.join(SPLITS)})")
    questions = [q for q in load_questions(questions_path) if split is None or q.split == split]
    if not questions:
        raise EvalError(f"no questions in split {split!r}")
    return questions


def open_corpora(
    questions: Sequence[Question],
    specs: Mapping[str, CorpusSpec],
    embedder: Embedder,
    *,
    repo_root: Path,
    stdlib_dir: Path | None = None,
    index_root: Path | None = None,
    log: Callable[[str], None] = lambda _: None,
) -> CorpusSetup:
    """Materialise and index every corpus the questions use; validate relevant ids.

    Indexes live in ``<corpus>/.chatter`` unless ``index_root`` is given, in
    which case corpus ``name`` is indexed into ``index_root / name``. An index
    built with another model or schema is rebuilt.
    """
    unknown = sorted({q.corpus for q in questions} - specs.keys())
    if unknown:
        raise EvalError(f"questions reference corpora not in the manifest: {', '.join(unknown)}")
    retrievers: dict[str, Retriever] = {}
    corpus_ids: dict[str, set[str]] = {}
    info: dict[str, Any] = {}
    for name in sorted({q.corpus for q in questions}):
        spec = specs[name]
        provenance = prepare_corpus(spec, repo_root=repo_root, stdlib_dir=stdlib_dir)
        index_dir = Path(index_root) / name if index_root else default_index_dir(spec.path)
        manifest = read_manifest(index_dir)
        rebuild = manifest is not None and (
            manifest.get("model") != embedder.name or manifest.get("schema") != SCHEMA_VERSION
        )
        try:
            stats = build_index(spec.path, embedder, index_dir=index_dir, rebuild=rebuild)
        except OSError as exc:
            raise EvalError(
                f"cannot write the index for corpus {name!r} at {index_dir}: {exc}. "
                "Use --index-root to keep eval indexes elsewhere."
            ) from exc
        log(f"corpus {name}: {stats.files} files, {stats.chunks} chunks, {stats.embedded_parts} embedded")
        retrievers[name] = Retriever.open(index_dir, embedder)
        corpus_ids[name] = chunk_ids_in_index(index_dir)
        info[name] = {
            **provenance,
            "files": stats.files,
            "chunks": stats.chunks,
            "graph": graph_coverage(index_dir),
        }
    validate_relevant(questions, corpus_ids)
    matchers = {name: containment(retriever.chunk) for name, retriever in retrievers.items()}
    return CorpusSetup(retrievers, matchers, info)


def graph_coverage(index_dir: Path) -> dict[str, Any] | None:
    """Call-graph resolution stats of the committed build (None if it has no graph)."""
    data = index_data(index_dir)
    stats = read_graph_stats(data.directory) if data is not None else None
    if stats is None:
        return None
    calls = stats.get("references", {}).get("call", 0)
    dropped_calls = sum(v for k, v in stats.get("dropped_by_reason", {}).items() if k.startswith("call:"))
    return {
        "call_sites": calls,
        "calls_resolved": calls - dropped_calls,
        "resolved_by_rule": stats.get("resolved_by_rule", {}),
        "dropped_by_reason": stats.get("dropped_by_reason", {}),
        "dropped_examples": stats.get("dropped_examples", {}),
        "edges_by_type": stats.get("edges_by_type", {}),
    }


def format_graph_coverage(corpora: Mapping[str, Any]) -> str:
    lines = ["call graph coverage (call sites resolved to repo code; dropped by reason)"]
    for name, info in corpora.items():
        graph = info.get("graph")
        if not graph:
            lines.append(f"  {name}: no graph")
            continue
        calls, resolved = graph["call_sites"], graph["calls_resolved"]
        share = resolved / calls if calls else 0.0
        edges = ", ".join(f"{k} {v}" for k, v in graph["edges_by_type"].items())
        lines.append(f"  {name}: {resolved}/{calls} call sites resolved ({share:.0%}); edges: {edges}")
        dropped = sorted(graph["dropped_by_reason"].items(), key=lambda kv: (-kv[1], kv[0]))
        lines.append("    dropped: " + ", ".join(f"{reason} {count}" for reason, count in dropped))
    return "\n".join(lines)


def run_metadata(
    questions_path: Path, repo_root: Path, embedder: Embedder, corpora_info: Mapping[str, Any]
) -> dict[str, Any]:
    """Provenance shared by every results file."""
    return {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": _git_info(repo_root),
        "questions_file": _relative(questions_path, repo_root),
        "questions_sha256": hashlib.sha256(Path(questions_path).read_bytes()).hexdigest(),
        "embedding_model": embedder.name,
        "embedding_fingerprint": embedder.fingerprint,
        "corpora": dict(corpora_info),
    }


def run_eval(
    questions_path: Path,
    corpora_path: Path,
    embedder_factory: Callable[[str], Embedder],
    *,
    model_name: str,
    repo_root: Path,
    stdlib_dir: Path | None = None,
    index_root: Path | None = None,
    split: str | None = None,
    configs: Sequence[RetrievalConfig] | None = None,
    log: Callable[[str], None] = lambda _: None,
) -> dict[str, Any]:
    """Materialise and index corpora, score every question in every mode.

    ``split`` restricts the run to one split (None: all questions); only the
    selected questions are validated, indexed for, and scored. ``configs``
    adds a table of retrieval configs; see ``evaluate_configs``. Corpus
    handling is described in ``open_corpora``.
    """
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
    retrievers, matchers, corpora_info = setup.retrievers, setup.matchers, setup.info
    per_mode: dict[str, list[tuple[Question, Scores]]] = {mode: [] for mode in MODES}
    for question in questions:
        for mode in MODES:
            hits = retrievers[question.corpus].search(question.question, k=DEPTH, mode=mode)
            ranked = [h.chunk_id for h in hits]
            per_mode[mode].append(
                (question, score_ranking(ranked, question.relevant, matchers[question.corpus]))
            )

    results: dict[str, Any] = {
        "schema": RESULTS_SCHEMA,
        **run_metadata(questions_path, repo_root, embedder, corpora_info),
        "settings": {"depth": DEPTH, "modes": list(MODES), "split": split or "all"},
        "summary": {mode: summarize(rows) for mode, rows in per_mode.items()},
        "questions": [
            {
                "id": q.id,
                "corpus": q.corpus,
                "type": q.type,
                "split": q.split,
                "scored": q.scored,
                "relevant": list(q.relevant),
                "modes": {
                    mode: {
                        "strict": dataclasses.asdict(scores.strict),
                        "contain": dataclasses.asdict(scores.contain),
                        "test_slots5": scores.test_slots5,
                        "top10": list(scores.top[:10]),
                    }
                    for mode, scores in ((m, per_mode[m][i][1]) for m in MODES)
                },
            }
            for i, q in enumerate(questions)
        ],
    }
    if configs:
        results["configs"] = evaluate_configs(questions, retrievers, matchers, configs)
    return results


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """One retrieval setup to score: a single retriever, or weighted RRF.

    ``graph_hops`` > 0 adds graph expansion (fused mode only) with RRF weight
    ``graph_weight``; ``counterpart`` names the config it is compared to.
    """

    name: str
    mode: str  # "bm25" | "dense" | "fused"
    rrf_k: int = RRF_K
    dense_weight: float = 1.0  # BM25 weight is always 1
    graph_hops: int = 0
    graph_weight: float = 1.0
    counterpart: str | None = None

    @property
    def expansion(self) -> GraphExpansion | None:
        if self.graph_hops <= 0:
            return None
        return GraphExpansion(hops=self.graph_hops, weight=self.graph_weight)


BM25_ONLY = RetrievalConfig("bm25-only", BM25)
DENSE_ONLY = RetrievalConfig("dense-only", DENSE)
DEFAULT_FUSED = RetrievalConfig("default", FUSED)


def sweep_configs(rrf_ks: Sequence[int], dense_weights: Sequence[float]) -> list[RetrievalConfig]:
    """Reference rows (bm25-only, dense-only, current default) plus the grid."""
    grid = [
        RetrievalConfig(f"k{k}-w{w:g}", FUSED, k, w)
        for k in rrf_ks
        for w in dense_weights
        if (k, w) != (DEFAULT_FUSED.rrf_k, DEFAULT_FUSED.dense_weight)
    ]
    return [BM25_ONLY, DENSE_ONLY, DEFAULT_FUSED, *grid]


def sweep_graph_configs(
    graph_weights: Sequence[float], graph_hops: Sequence[int]
) -> list[RetrievalConfig]:
    """Reference rows plus graph variants of the current default and k10-w3."""
    tuned = RetrievalConfig("k10-w3", FUSED, 10, 3.0)
    variants = [
        RetrievalConfig(
            f"{base.name}+g{hops}-w{weight:g}",
            FUSED,
            base.rrf_k,
            base.dense_weight,
            graph_hops=hops,
            graph_weight=weight,
            counterpart=base.name,
        )
        for base in (DEFAULT_FUSED, tuned)
        for hops in graph_hops
        for weight in graph_weights
    ]
    return [DENSE_ONLY, DEFAULT_FUSED, tuned, *variants]


def load_configs(path: Path) -> list[RetrievalConfig]:
    """Read frozen candidate configs.

    The file is a YAML list of configs, or a mapping with ``configs`` (that
    list) and an optional ``decision_rule`` section that the report shows.
    """
    return load_config_file(path)[0]


def load_config_file(path: Path) -> tuple[list[RetrievalConfig], dict[str, Any] | None]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    rule = None
    if isinstance(data, dict):
        rule = data.get("decision_rule")
        data = data.get("configs")
    if not isinstance(data, list) or not data:
        raise EvalError(f"{path}: expected a non-empty list of configs")
    problems: list[str] = []
    configs: list[RetrievalConfig] = []
    for n, item in enumerate(data, 1):
        if not isinstance(item, dict) or not item.get("name") or item.get("mode") not in MODES:
            problems.append(f"item {n}: needs a name and a mode in {', '.join(MODES)}")
            continue
        name = str(item["name"])
        rrf_k = item.get("rrf_k", RRF_K)
        weight = item.get("dense_weight", 1.0)
        hops = item.get("graph_hops", 0)
        graph_weight = item.get("graph_weight", 1.0)
        if not isinstance(rrf_k, int) or rrf_k < 1:
            problems.append(f"{name}: rrf_k must be a positive integer")
        if not isinstance(weight, int | float) or weight <= 0:
            problems.append(f"{name}: dense_weight must be positive")
        if not isinstance(hops, int) or hops < 0:
            problems.append(f"{name}: graph_hops must be a non-negative integer")
        elif hops > 0 and item["mode"] != FUSED:
            problems.append(f"{name}: graph expansion needs mode fused")
        if not isinstance(graph_weight, int | float) or graph_weight <= 0:
            problems.append(f"{name}: graph_weight must be positive")
        configs.append(
            RetrievalConfig(
                name, str(item["mode"]), int(rrf_k), float(weight),
                graph_hops=int(hops), graph_weight=float(graph_weight),
                counterpart=item.get("counterpart"),
            )
        )
    names = [c.name for c in configs]
    problems += [f"duplicate config name {n!r}" for n in sorted({n for n in names if names.count(n) > 1})]
    problems += [
        f"{c.name}: counterpart {c.counterpart!r} is not a config in this file"
        for c in configs
        if c.counterpart is not None and c.counterpart not in names
    ]
    if problems:
        raise EvalError(f"invalid configs in {path}:\n  " + "\n  ".join(problems))
    return configs, rule


def evaluate_configs(
    questions: Sequence[Question],
    retrievers: Mapping[str, Retriever],
    matchers: Mapping[str, Contains],
    configs: Sequence[RetrievalConfig],
) -> dict[str, Any]:
    """Score each config, with paired per-question comparisons against dense-only.

    Each question's BM25 and dense rankings are computed once at the depth
    the fused mode uses, then re-fused per config, so the default config
    equals the regular fused mode. Pairing uses the strict first relevant
    rank; a question with no relevant chunk in the top 50 loses to any rank,
    and two misses tie. Negative questions are excluded.
    """
    rankings = [(question, _base_rankings(retrievers[question.corpus], question)) for question in questions]
    corpus_of = {q.id: q.corpus for q in questions}
    reference = {
        q.id: score_ranking(_config_ranking(DENSE_ONLY, lists), q.relevant).strict.first_rank
        for q, lists in rankings
        if q.scored
    }
    corpora = sorted(set(corpus_of.values()))
    cells = []
    for config in configs:
        rows = [
            (q, score_ranking(_config_ranking_for(config, q, lists, retrievers), q.relevant, matchers[q.corpus]))
            for q, lists in rankings
        ]
        first = {q.id: s.strict.first_rank for q, s in rows if q.scored}
        first_contain = {q.id: s.contain.first_rank for q, s in rows if q.scored}
        cells.append(
            {
                **dataclasses.asdict(config),
                "overall": aggregate(rows),
                "by_corpus": {c: aggregate([r for r in rows if r[0].corpus == c]) for c in corpora},
                "first_ranks": first,
                "first_ranks_contain": first_contain,
                "vs_dense": paired_comparison(first, reference, corpus_of),
            }
        )
    by_name = {cell["name"]: cell for cell in cells}
    for cell in cells:
        other = by_name.get(cell.get("counterpart") or "")
        if other is not None:
            cell["vs_counterpart"] = paired_comparison(cell["first_ranks"], other["first_ranks"], corpus_of)
            cell["top3_losses"] = top3_losses(cell["first_ranks"], other["first_ranks"])
            cell["top3_losses_contain"] = top3_losses(
                cell["first_ranks_contain"], other["first_ranks_contain"]
            )
    return {"reference": DENSE_ONLY.name, "cells": cells}


def top3_losses(ranks: Mapping[str, int | None], counterpart: Mapping[str, int | None]) -> list[str]:
    """Questions where the counterpart had a relevant chunk in its top 3 and this ranks it lower."""
    lost = []
    for qid, theirs in sorted(counterpart.items()):
        mine = ranks.get(qid)
        if theirs is not None and theirs <= 3 and (mine is None or mine > theirs):
            lost.append(qid)
    return lost


def _config_ranking_for(
    config: RetrievalConfig,
    question: Question,
    lists: Mapping[str, Sequence[str]],
    retrievers: Mapping[str, Retriever],
) -> list[str]:
    """Graph configs search live (expansion needs the graph); others re-fuse ``lists``."""
    if config.expansion is None:
        return _config_ranking(config, lists)
    hits = retrievers[question.corpus].search(
        question.question,
        k=DEPTH,
        candidates=CANDIDATES,
        mode=FUSED,
        rrf_k=config.rrf_k,
        weights={BM25: 1.0, DENSE: config.dense_weight},
        expansion=config.expansion,
    )
    return [h.chunk_id for h in hits]


def paired_comparison(
    ranks: Mapping[str, int | None],
    reference: Mapping[str, int | None],
    corpus_of: Mapping[str, str],
) -> dict[str, Any]:
    """Wins/losses/ties of ``ranks`` against ``reference`` (lower rank wins)."""

    def key(rank: int | None) -> float:
        return float("inf") if rank is None else rank

    def tally(ids: Sequence[str]) -> dict[str, int]:
        wins = sum(key(ranks[i]) < key(reference[i]) for i in ids)
        losses = sum(key(ranks[i]) > key(reference[i]) for i in ids)
        return {"wins": wins, "losses": losses, "ties": len(ids) - wins - losses}

    ids = sorted(set(ranks) & set(reference))
    corpora = sorted({corpus_of[i] for i in ids})
    return {
        "overall": tally(ids),
        "by_corpus": {c: tally([i for i in ids if corpus_of[i] == c]) for c in corpora},
    }


def _base_rankings(retriever: Retriever, question: Question) -> dict[str, list[str]]:
    return {
        mode: [
            h.chunk_id
            for h in retriever.search(question.question, k=CANDIDATES, candidates=CANDIDATES, mode=mode)
        ]
        for mode in (BM25, DENSE)
    }


def _config_ranking(config: RetrievalConfig, lists: Mapping[str, Sequence[str]]) -> list[str]:
    if config.mode in (BM25, DENSE):
        return list(lists[config.mode][:DEPTH])
    fused = reciprocal_rank_fusion(
        lists, k=config.rrf_k, weights={BM25: 1.0, DENSE: config.dense_weight}
    )
    return [chunk_id for chunk_id, _, _ in fused[:DEPTH]]


def format_configs(table: Mapping[str, Any], decision_rule: Mapping[str, Any] | None = None) -> str:
    """Strict MRR@50 per corpus and overall, hit/recall, W/L/T vs dense-only and,
    for graph variants, vs their counterpart with any lost top-3 ranks."""
    cells = [c for c in table["cells"] if c["overall"].get("n")]
    if not cells:
        return "(no scored questions)"
    corpora = sorted(cells[0]["by_corpus"])
    wlt = lambda t: f"{t['wins']}/{t['losses']}/{t['ties']}"  # noqa: E731
    width = max(13, *(len(c["name"]) for c in cells))
    header = (
        f"{'config':<{width}} {'k':>3} {'w':>4} {'hop':>3} {'gw':>4}  "
        + " ".join(f"{'MRR ' + c:>12}" for c in corpora)
        + f" {'MRR all':>8} {'hit@1':>6} {'hit@5':>6} {'R@10':>6} {'cMRR':>6}  {'vs dense':>9}"
        + f"  {'vs counterpart':>14} {'top3 lost':>9}"
    )
    lines = [
        "configs (strict metrics; cMRR = containment MRR@50; W/L/T = paired first-rank "
        "wins/losses/ties; top3 lost = questions whose counterpart top-3 rank got worse)",
        header,
    ]
    for cell in cells:
        strict = cell["overall"]["strict"]
        fused = cell["mode"] == FUSED
        graph = cell.get("graph_hops", 0) > 0
        counterpart = (
            f"{wlt(cell['vs_counterpart']['overall'])} {cell['counterpart'][:8]}"
            if "vs_counterpart" in cell
            else "-"
        )
        lost = ",".join(cell["top3_losses"]) or "none" if "top3_losses" in cell else "-"
        lines.append(
            f"{cell['name']:<{width}} {cell['rrf_k'] if fused else '-':>3} "
            f"{format(cell['dense_weight'], 'g') if fused else '-':>4} "
            f"{cell['graph_hops'] if graph else '-':>3} {format(cell['graph_weight'], 'g') if graph else '-':>4}  "
            + " ".join(f"{cell['by_corpus'][c]['strict']['mrr@50']:>12.3f}" for c in corpora)
            + f" {strict['mrr@50']:>8.3f} {strict['hit@1']:>6.3f} {strict['hit@5']:>6.3f} "
            f"{strict['recall@10']:>6.3f} {cell['overall']['contain']['mrr@50']:>6.3f}  "
            f"{wlt(cell['vs_dense']['overall']):>9}  {counterpart:>14} {lost:>9}"
        )
    if decision_rule:
        lines += ["", "decision rule: " + " ".join(str(decision_rule.get("text", "")).split())]
        for cell in cells:
            if "vs_counterpart" in cell:
                vs = cell["vs_counterpart"]["overall"]
                passed = vs["wins"] > vs["losses"] and not cell["top3_losses"]
                lines.append(
                    f"  {cell['name']}: {'PASSES' if passed else 'fails'} on these questions "
                    f"(W/L {vs['wins']}/{vs['losses']} vs {cell['counterpart']}; "
                    f"top-3 losses: {', '.join(cell['top3_losses']) or 'none'}; "
                    f"containment top-3 losses: {', '.join(cell['top3_losses_contain']) or 'none'})"
                )
    return "\n".join(lines)


def save_results(results: Mapping[str, Any], results_dir: Path, *, label: str = "") -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = results["created"].replace(":", "").replace("-", "").split("+")[0]
    commit = (results["git"].get("commit") or "nogit")[:7]
    dirty = "-dirty" if results["git"].get("dirty") else ""
    split = results.get("settings", {}).get("split", "all")
    suffix = "" if split == "all" else f"-{split}"
    tag = f"-{label}" if label else ""
    path = results_dir / f"{stamp}-{commit}{dirty}{tag}{suffix}.json"
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return path


def _git_info(repo_root: Path) -> dict[str, Any]:
    commit = _git(repo_root, "rev-parse", "HEAD")
    status = _git(repo_root, "status", "--porcelain", "--untracked-files=no")
    return {"commit": commit or None, "dirty": bool(status)}


def _relative(path: Path, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def format_report(results: Mapping[str, Any]) -> str:
    lines = [
        f"embedding model: {results['embedding_model']}   commit: "
        f"{(results['git'].get('commit') or 'none')[:10]}{' (dirty)' if results['git'].get('dirty') else ''}",
        "corpora: "
        + ", ".join(
            f"{name} ({info['chunks']} chunks, {info.get('ref') or 'py' + info.get('python_full_version', '?')})"
            for name, info in results["corpora"].items()
        ),
    ]
    selected = results["settings"].get("split", "all")
    for split in SPLITS:
        if selected not in ("all", split):
            continue
        counts = [q for q in results["questions"] if q.get("split", DEFAULT_SPLIT) == split]
        scored = sum(q["scored"] for q in counts)
        lines += ["", f"== split: {split} ({scored} scored, {len(counts) - scored} negative)"]
        if not scored:
            lines.append("   (no scored questions)")
            continue
        lines.append(
            f"{'mode':<6} {'type':<11} {'n':>3} {'hit@1':>11} {'hit@5':>11} {'MRR@50':>11} "
            f"{'R@10':>11} {'test@5':>8}   (cells: strict/containment)"
        )
        for mode in results["settings"]["modes"]:
            summary = results["summary"][mode]["by_split"][split]
            groups = [("overall", summary["overall"]), *summary["by_type"].items()]
            for label, agg in groups:
                if agg.get("n"):
                    pair = lambda key: f"{agg['strict'][key]:.3f}/{agg['contain'][key]:.3f}"  # noqa: E731
                    lines.append(
                        f"{mode:<6} {label:<11} {agg['n']:>3} {pair('hit@1'):>11} "
                        f"{pair('hit@5'):>11} {pair('mrr@50'):>11} {pair('recall@10'):>11} "
                        f"{agg['test_slots@5']:>3}/{agg['slots@5']:<4}"
                    )
            lines.append("")

    lines.append(
        f"{'id':<4} {'split':<7} {'type':<10} {'corpus':<7} {'bm25':>7} {'dense':>7} {'fused':>7} "
        f"{'R@10':>9} {'test@5':>6}   (first relevant rank strict/containment; '-' = not in top 50)"
    )
    for q in results["questions"]:
        def rank(mode: str, q: Mapping[str, Any] = q) -> str:
            if not q["scored"]:
                return "n/a"
            strict, contain = (q["modes"][mode][kind]["first_rank"] for kind in ("strict", "contain"))
            return f"{strict or '-'}/{contain or '-'}"

        fused = q["modes"]["fused"]
        recall = (
            f"{fused['strict']['recall10']:.2f}/{fused['contain']['recall10']:.2f}"
            if q["scored"]
            else "n/a"
        )
        lines.append(
            f"{q['id']:<4} {q.get('split', DEFAULT_SPLIT):<7} {q['type']:<10} {q['corpus']:<7} "
            f"{rank('bm25'):>7} {rank('dense'):>7} {rank('fused'):>7} {recall:>9} "
            f"{fused['test_slots5']:>4}/5"
        )
    return "\n".join(lines)
