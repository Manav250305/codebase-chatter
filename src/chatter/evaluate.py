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
from chatter.index import (
    CHUNKS_FILE,
    IndexStorageError,
    build_index,
    default_index_dir,
    read_manifest,
)
from chatter.retrieve import MODES, Retriever

RESULTS_SCHEMA = 2  # 2: per-question "split" and summary[mode]["by_split"]
DEPTH = 50  # MRR cutoff and ranking depth recorded per question
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


@dataclass(frozen=True, slots=True)
class Scores:
    first_rank: int | None
    hit1: bool
    hit5: bool
    rr: float
    recall10: float
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
    path = PurePosixPath(chunk_id.split("::", 1)[0])
    return (
        any(part in ("tests", "test") for part in path.parts[:-1])
        or path.name.startswith("test_")
        or path.name.endswith("_test.py")
        or path.name == "conftest.py"
    )


def score_ranking(ranked: Sequence[str], relevant: Sequence[str]) -> Scores:
    top = tuple(ranked[:DEPTH])
    wanted = set(relevant)
    ranks = [i for i, cid in enumerate(top, 1) if cid in wanted]
    first = ranks[0] if ranks else None
    return Scores(
        first_rank=first,
        hit1=first == 1,
        hit5=first is not None and first <= 5,
        rr=1.0 / first if first is not None else 0.0,
        recall10=(len(wanted.intersection(top[:10])) / len(wanted)) if wanted else 0.0,
        test_slots5=sum(is_test_chunk(cid) for cid in top[:5]),
        top=top,
    )


def aggregate(rows: Sequence[tuple[Question, Scores]]) -> dict[str, Any]:
    """Mean metrics over scored (non-negative) questions."""
    scored = [(q, s) for q, s in rows if q.scored]
    n = len(scored)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "hit@1": sum(s.hit1 for _, s in scored) / n,
        "hit@5": sum(s.hit5 for _, s in scored) / n,
        "mrr@50": sum(s.rr for _, s in scored) / n,
        "recall@10": sum(s.recall10 for _, s in scored) / n,
        "test_slots@5": sum(s.test_slots5 for _, s in scored),
        "slots@5": sum(min(5, len(s.top)) for _, s in scored),
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


def run_eval(
    questions_path: Path,
    corpora_path: Path,
    embedder_factory: Callable[[str], Embedder],
    *,
    model_name: str,
    repo_root: Path,
    stdlib_dir: Path | None = None,
    index_root: Path | None = None,
    log: Callable[[str], None] = lambda _: None,
) -> dict[str, Any]:
    """Materialise and index corpora, score every question in every mode.

    Indexes live in ``<corpus>/.chatter`` unless ``index_root`` is given, in
    which case corpus ``name`` is indexed into ``index_root / name``.
    """
    questions = load_questions(questions_path)
    specs = load_corpora(corpora_path)
    unknown = sorted({q.corpus for q in questions} - specs.keys())
    if unknown:
        raise EvalError(f"questions reference corpora not in {corpora_path}: {', '.join(unknown)}")

    embedder = embedder_factory(model_name)
    corpora_info: dict[str, Any] = {}
    retrievers: dict[str, Retriever] = {}
    corpus_ids: dict[str, set[str]] = {}
    for name in sorted({q.corpus for q in questions}):
        spec = specs[name]
        info = prepare_corpus(spec, repo_root=repo_root, stdlib_dir=stdlib_dir)
        index_dir = Path(index_root) / name if index_root else default_index_dir(spec.path)
        manifest = read_manifest(index_dir)
        rebuild = manifest is not None and manifest.get("model") != embedder.name
        try:
            stats = build_index(spec.path, embedder, index_dir=index_dir, rebuild=rebuild)
        except IndexStorageError as exc:
            raise EvalError(f"{exc} Use --index-root to keep eval indexes elsewhere.") from exc
        log(f"corpus {name}: {stats.files} files, {stats.chunks} chunks, {stats.embedded_parts} embedded")
        retrievers[name] = Retriever.open(index_dir, embedder)
        corpus_ids[name] = _chunk_ids(index_dir)
        corpora_info[name] = {**info, "files": stats.files, "chunks": stats.chunks}

    validate_relevant(questions, corpus_ids)

    per_mode: dict[str, list[tuple[Question, Scores]]] = {mode: [] for mode in MODES}
    for question in questions:
        for mode in MODES:
            hits = retrievers[question.corpus].search(question.question, k=DEPTH, mode=mode)
            per_mode[mode].append(
                (question, score_ranking([h.chunk_id for h in hits], question.relevant))
            )

    return {
        "schema": RESULTS_SCHEMA,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": _git_info(repo_root),
        "questions_file": _relative(questions_path, repo_root),
        "questions_sha256": hashlib.sha256(Path(questions_path).read_bytes()).hexdigest(),
        "embedding_model": embedder.name,
        "embedding_fingerprint": embedder.fingerprint,
        "corpora": corpora_info,
        "settings": {"depth": DEPTH, "modes": list(MODES)},
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
                        **{k: v for k, v in dataclasses.asdict(scores).items() if k != "top"},
                        "top10": list(scores.top[:10]),
                    }
                    for mode, scores in ((m, per_mode[m][i][1]) for m in MODES)
                },
            }
            for i, q in enumerate(questions)
        ],
    }


def save_results(results: Mapping[str, Any], results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = results["created"].replace(":", "").replace("-", "").split("+")[0]
    commit = (results["git"].get("commit") or "nogit")[:7]
    dirty = "-dirty" if results["git"].get("dirty") else ""
    path = results_dir / f"{stamp}-{commit}{dirty}.json"
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return path


def _chunk_ids(index_dir: Path) -> set[str]:
    with (index_dir / CHUNKS_FILE).open(encoding="utf-8") as records:
        return {json.loads(line)["id"] for line in records}


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
    for split in SPLITS:
        counts = [q for q in results["questions"] if q.get("split", DEFAULT_SPLIT) == split]
        scored = sum(q["scored"] for q in counts)
        lines += ["", f"== split: {split} ({scored} scored, {len(counts) - scored} negative)"]
        if not scored:
            lines.append("   (no scored questions)")
            continue
        lines.append(
            f"{'mode':<6} {'type':<11} {'n':>3} {'hit@1':>6} {'hit@5':>6} {'MRR@50':>7} "
            f"{'R@10':>6} {'test@5':>8}"
        )
        for mode in results["settings"]["modes"]:
            summary = results["summary"][mode]["by_split"][split]
            groups = [("overall", summary["overall"]), *summary["by_type"].items()]
            for label, agg in groups:
                if agg.get("n"):
                    lines.append(
                        f"{mode:<6} {label:<11} {agg['n']:>3} {agg['hit@1']:>6.3f} "
                        f"{agg['hit@5']:>6.3f} {agg['mrr@50']:>7.3f} {agg['recall@10']:>6.3f} "
                        f"{agg['test_slots@5']:>3}/{agg['slots@5']:<4}"
                    )
            lines.append("")

    lines.append(
        f"{'id':<4} {'split':<7} {'type':<10} {'corpus':<7} {'bm25':>5} {'dense':>5} {'fused':>5} "
        f"{'R@10':>5} {'test@5':>6}   (first relevant rank; '-' = not in top 50)"
    )
    for q in results["questions"]:
        ranks = [q["modes"][m]["first_rank"] for m in ("bm25", "dense", "fused")]
        fused = q["modes"]["fused"]
        recall = f"{fused['recall10']:.2f}" if q["scored"] else "n/a"
        rank_text = [str(r) if r else ("-" if q["scored"] else "n/a") for r in ranks]
        lines.append(
            f"{q['id']:<4} {q.get('split', DEFAULT_SPLIT):<7} {q['type']:<10} {q['corpus']:<7} "
            f"{rank_text[0]:>5} {rank_text[1]:>5} {rank_text[2]:>5} {recall:>5} "
            f"{fused['test_slots5']:>4}/5"
        )
    return "\n".join(lines)
