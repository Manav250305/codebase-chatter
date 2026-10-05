"""Build a persistent hybrid (BM25 + dense) index over a repository's Python code.

Layout of an index directory (default ``<repo>/.chatter``)::

    manifest.json        schema, repo, embedding model, and the committed build
    gen-<id>/
        chunks.jsonl     one record per chunk: id, chunk fields, BM25 tokens,
                         content hash, vector rows [start, count], part lines
        vectors.npy      float32 L2-normalized part vectors, one row per part

Chunks whose embedding text exceeds the model's token budget are split by lines
into parts; each part repeats the chunk header and gets its own vector row.
Each chunk records a hash of its embedding text, so re-indexing copies the
vectors of unchanged chunks and only embeds new or changed ones. A build is
committed by atomically replacing manifest.json.
"""

from __future__ import annotations

import dataclasses
import filecmp
import functools
import hashlib
import json
import logging
import os
import re
import shutil
import uuid
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pathspec
import snowballstemmer

from chatter.embed import Embedder, SentenceTransformerEmbedder
from chatter.extract import DEFAULT_MAX_BYTES, Chunk, extract_source, read_source
from chatter.graph import GRAPH_FILES, GRAPH_VERSION, build_graph, collect_file_refs, write_graph

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 4  # 2: Snowball stems; 3: exact numpy vectors, no Chroma; 4: call graph
INDEX_DIRNAME = ".chatter"
MANIFEST_FILE = "manifest.json"
CHUNKS_FILE = "chunks.jsonl"
VECTORS_FILE = "vectors.npy"
GENERATION_PREFIX = "gen-"
_LEGACY_ENTRIES = frozenset({"chroma", CHUNKS_FILE})  # schema 1-2 layout

SKIP_DIRS = frozenset(
    {".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__", INDEX_DIRNAME}
)
SOURCE_SUFFIXES = frozenset({".py"})
HEADER_DOCSTRING_CHARS = 400

_IDENTIFIER_RE = re.compile(r"[^\W\d]\w*|\d+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


class IndexMismatchError(RuntimeError):
    """The on-disk index was built with a different model, schema, or repo."""


@dataclass(frozen=True, slots=True)
class IndexConfig:
    # Texts per embed call. sentence-transformers length-sorts within a call, so
    # larger calls pad less: 256 was ~2x faster than 64 for bge-small (MPS and CPU).
    batch_size: int = 256
    max_file_bytes: int = DEFAULT_MAX_BYTES


@dataclass(frozen=True, slots=True)
class IndexStats:
    files: int
    chunks: int
    embedded_parts: int
    reused_chunks: int
    deleted_chunks: int


@dataclass(frozen=True, slots=True)
class EmbeddingPart:
    """A slice of a chunk sent to the embedder. ``first``/``last`` index source lines."""

    index: int
    first: int
    last: int
    text: str


# ---------------------------------------------------------------------------
# Tokenization (BM25)
# ---------------------------------------------------------------------------


def tokenize_code(text: str, *, stem: bool = False) -> list[str]:
    """Lowercased identifiers plus their snake_case / camelCase parts.

    ``parseHTTPResponse`` -> ``parsehttpresponse parse http response``;
    ``max_retry_count`` -> ``max_retry_count max retry count``.
    With ``stem``, Snowball stems of the parts are added after them when they
    differ (``repeating`` -> ``repeating repeat``); unstemmed tokens are kept.
    """
    return [
        token
        for word in identifier_words(text)
        for token in tokenize_identifier(word, stem=stem)
    ]


def identifier_words(text: str) -> list[str]:
    """Identifier-like words (and digit runs) in ``text``, case preserved."""
    return _IDENTIFIER_RE.findall(text)


def tokenize_identifier(word: str, *, stem: bool = False) -> list[str]:
    """``word`` lowercased, then its snake_case/camelCase parts if it has several.

    With ``stem``, each part's Snowball stem follows when it differs; a word
    with a single part is its own part.
    """
    lower = word.lower()
    parts = [
        part.lower()
        for piece in word.split("_")
        if piece
        for part in _CAMEL_BOUNDARY_RE.split(piece)
        if part
    ]
    split = len(parts) > 1 or (parts and parts[0] != lower)
    tokens = [lower, *parts] if split else [lower]
    if stem:
        tokens += [s for part in (parts if split else [lower]) if (s := stem_word(part)) != part]
    return tokens


@functools.cache
def _english_stemmer() -> Any:
    return snowballstemmer.stemmer("english")


@functools.lru_cache(maxsize=65_536)
def stem_word(word: str) -> str:
    """Snowball (Porter2) English stem; digits and short words pass through."""
    return _english_stemmer().stemWord(word)


def bm25_tokens(chunk: Chunk) -> list[str]:
    return tokenize_code(f"{chunk.path} {chunk.qualname} {chunk.source}", stem=True)


# ---------------------------------------------------------------------------
# Chunk identity and embedding text
# ---------------------------------------------------------------------------


def assign_chunk_ids(chunks: Sequence[Chunk]) -> list[str]:
    """``path::qualname``; repeats within a file get ``~2``, ``~3`` in source order."""
    seen: Counter[str] = Counter()
    ids: list[str] = []
    for chunk in chunks:
        base = f"{chunk.path}::{chunk.qualname}"
        seen[base] += 1
        ids.append(base if seen[base] == 1 else f"{base}~{seen[base]}")
    return ids


def embedding_header(chunk: Chunk) -> str:
    lines = [f"# file: {chunk.path}", f"# symbol: {chunk.qualname} ({chunk.kind})"]
    if chunk.docstring:
        doc = " ".join(chunk.docstring.split())
        if len(doc) > HEADER_DOCSTRING_CHARS:
            doc = doc[:HEADER_DOCSTRING_CHARS].rstrip() + "…"
        lines.append(f"# doc: {doc}")
    return "\n".join(lines) + "\n"


def embedding_text(chunk: Chunk) -> str:
    return embedding_header(chunk) + chunk.source


def plan_parts(chunks: Sequence[Chunk], embedder: Embedder) -> list[list[EmbeddingPart]]:
    """Split each chunk into parts that fit ``embedder.max_tokens``.

    Whole texts are counted in one batch; per-line counts are only computed
    for oversized chunks. A single line longer than the budget becomes its
    own part and is truncated by the model.
    """
    texts = [embedding_text(chunk) for chunk in chunks]
    totals = embedder.count_tokens(texts)
    plans: list[list[EmbeddingPart]] = []
    for chunk, text, total in zip(chunks, texts, totals):
        last_line = chunk.source.count("\n")
        if total <= embedder.max_tokens:
            plans.append([EmbeddingPart(0, 0, last_line, text)])
        else:
            plans.append(_split_chunk(chunk, embedder))
    return plans


def _split_chunk(chunk: Chunk, embedder: Embedder) -> list[EmbeddingPart]:
    header = embedding_header(chunk)
    lines = chunk.source.split("\n")
    header_tokens, *line_tokens = embedder.count_tokens([header, *lines])
    budget = max(embedder.max_tokens - header_tokens, embedder.max_tokens // 4, 1)

    ranges: list[tuple[int, int]] = []
    first, used = 0, 0
    for i, count in enumerate(line_tokens):
        if i > first and used + count > budget:
            ranges.append((first, i - 1))
            first, used = i, 0
        used += count
    ranges.append((first, len(lines) - 1))
    return [
        EmbeddingPart(n, start, end, header + "\n".join(lines[start : end + 1]))
        for n, (start, end) in enumerate(ranges)
    ]


def content_hash(fingerprint: str, text: str) -> str:
    return hashlib.sha256(f"{fingerprint}\0{text}".encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Repository walking
# ---------------------------------------------------------------------------


def iter_source_files(
    repo: Path, suffixes: frozenset[str] = SOURCE_SUFFIXES
) -> Iterator[Path]:
    """Yield source files under ``repo`` in sorted order.

    Prunes ``SKIP_DIRS``, macOS AppleDouble ``._*`` files (created on exFAT/FAT
    volumes), and paths matched by any ``.gitignore`` (nested files apply
    relative to their directory; deeper files win). Symlinks are not followed,
    matching how git stores them and avoiding cycles and duplicates.
    """
    root = Path(repo).resolve()
    active_by_dir: dict[Path, list[tuple[Path, pathspec.GitIgnoreSpec]]] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        active = list(active_by_dir.pop(current, []))
        spec = _load_gitignore(current / ".gitignore")
        if spec is not None:
            active.append((current, spec))

        kept: list[str] = []
        for name in sorted(dirnames):
            child = current / name
            if name in SKIP_DIRS or child.is_symlink() or _ignored(active, child, is_dir=True):
                continue
            kept.append(name)
            active_by_dir[child] = active
        dirnames[:] = kept

        for name in sorted(filenames):
            path = current / name
            if path.suffix not in suffixes or name.startswith("._") or path.is_symlink():
                continue
            if not _ignored(active, path, is_dir=False):
                yield path


def _load_gitignore(path: Path) -> pathspec.GitIgnoreSpec | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    return pathspec.GitIgnoreSpec.from_lines(lines)


def _ignored(
    active: Sequence[tuple[Path, pathspec.GitIgnoreSpec]], path: Path, *, is_dir: bool
) -> bool:
    ignored = False
    for base, spec in active:
        rel = path.relative_to(base).as_posix() + ("/" if is_dir else "")
        verdict = spec.check_file(rel).include
        if verdict is not None:
            ignored = verdict
    return ignored


# ---------------------------------------------------------------------------
# Storage (shared with retrieve.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IndexData:
    """Files of one committed build (the generation the manifest points to)."""

    directory: Path
    chunks_path: Path
    vectors_path: Path


@dataclass(frozen=True, slots=True)
class _Previous:
    """What a re-index needs from the last build to reuse a chunk's vectors."""

    hash: str
    start: int
    count: int
    parts: list[list[int]]


def default_index_dir(repo: Path) -> Path:
    return Path(repo).resolve() / INDEX_DIRNAME


def read_manifest(index_dir: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((Path(index_dir) / MANIFEST_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def check_manifest(manifest: dict[str, Any], embedder: Embedder) -> None:
    if manifest.get("schema") != SCHEMA_VERSION:
        raise IndexMismatchError(
            f"index schema {manifest.get('schema')} != {SCHEMA_VERSION}; rebuild the index"
        )
    if manifest.get("model") != embedder.name:
        raise IndexMismatchError(
            f"index was built with model {manifest.get('model')!r}, not {embedder.name!r}; "
            "rebuild the index or use the same model"
        )


def index_data(index_dir: Path, manifest: dict[str, Any] | None = None) -> IndexData | None:
    """The committed build's files, or None if there is no (current-schema) build."""
    manifest = manifest if manifest is not None else read_manifest(index_dir)
    generation = (manifest or {}).get("data")
    if not generation:
        return None
    directory = Path(index_dir) / generation
    return IndexData(directory, directory / CHUNKS_FILE, directory / VECTORS_FILE)


def iter_records(chunks_path: Path) -> Iterator[dict[str, Any]]:
    with Path(chunks_path).open(encoding="utf-8") as records:
        for line in records:
            yield json.loads(line)


def load_vectors(path: Path) -> np.ndarray:
    """Memory-map the float32 part vectors (an empty array cannot be mapped)."""
    try:
        return np.load(path, mmap_mode="r")
    except ValueError:
        return np.load(path)


def chunk_ids_in_index(index_dir: Path) -> set[str]:
    data = index_data(index_dir)
    return set() if data is None else {record["id"] for record in iter_records(data.chunks_path)}


def chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return dataclasses.asdict(chunk)


def chunk_from_dict(data: dict[str, Any]) -> Chunk:
    return Chunk(
        **{
            **data,
            "imports": tuple(data["imports"]),
            "decorators": tuple(data["decorators"]),
            "spans": tuple((start, end) for start, end in data["spans"]),
        }
    )


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_index(
    repo: Path | str,
    embedder: Embedder | None = None,
    *,
    index_dir: Path | str | None = None,
    config: IndexConfig = IndexConfig(),
    rebuild: bool = False,
) -> IndexStats:
    """Index every Python file under ``repo``; re-running only embeds changes.

    Each build writes a new generation directory and then commits it by
    replacing ``manifest.json``; older generations are removed afterwards, so
    an interrupted build never leaves records and vectors out of step. A build
    whose output is byte-identical to the committed one is discarded, so an
    unchanged re-index leaves the index untouched.
    Unchanged chunks (same content hash) copy their vectors from the previous
    build. Raises ``IndexMismatchError`` if an existing index used a different
    model, schema, or repo, unless ``rebuild`` is set (nothing is reused).
    """
    repo_path = Path(repo).resolve()
    if not repo_path.is_dir():
        raise NotADirectoryError(repo_path)
    embedder = embedder if embedder is not None else SentenceTransformerEmbedder()
    out = Path(index_dir) if index_dir is not None else default_index_dir(repo_path)
    out.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(out)
    if manifest is not None and not rebuild:
        check_manifest(manifest, embedder)
        _check_same_repo(manifest, repo_path, out)
    previous: dict[str, _Previous] = {}
    old_vectors: np.ndarray | None = None
    data = None if rebuild else index_data(out, manifest)
    if data is not None:
        previous = {
            r["id"]: _Previous(r["hash"], r["rows"][0], r["rows"][1], r["parts"])
            for r in iter_records(data.chunks_path)
        }
        old_vectors = load_vectors(data.vectors_path)

    generation = f"{GENERATION_PREFIX}{uuid.uuid4().hex[:12]}"
    gen_dir = out / generation
    gen_dir.mkdir()
    seen: set[str] = set()
    files = reused = 0
    unchanged = False
    skeletons: list[Chunk] = []  # chunk metadata without source, for graph resolution
    record_ids: list[str] = []
    file_refs = []
    try:
        writer = _VectorWriter(gen_dir / "vectors.f32", embedder, config.batch_size)
        with (gen_dir / CHUNKS_FILE).open("w", encoding="utf-8") as records:
            for path in iter_source_files(repo_path):
                files += 1
                rel = path.relative_to(repo_path).as_posix()
                text = read_source(path, label=rel, max_bytes=config.max_file_bytes)
                chunks = extract_source(text, rel) if text is not None else []
                ids = assign_chunk_ids(chunks)
                if text is not None and chunks:
                    indices = list(range(len(record_ids), len(record_ids) + len(chunks)))
                    file_refs.append(collect_file_refs(text, rel, chunks, indices))
                record_ids += ids
                skeletons += [dataclasses.replace(c, source="", docstring=None, imports=()) for c in chunks]
                digests = [content_hash(embedder.fingerprint, embedding_text(c)) for c in chunks]
                stale = [
                    i
                    for i, (cid, digest) in enumerate(zip(ids, digests))
                    if old_vectors is None or cid not in previous or previous[cid].hash != digest
                ]
                plans = dict(zip(stale, plan_parts([chunks[i] for i in stale], embedder)))
                for i, (chunk_id, chunk, digest) in enumerate(zip(ids, chunks, digests)):
                    seen.add(chunk_id)
                    if i in plans:
                        parts = [[part.first, part.last] for part in plans[i]]
                        start = writer.add([part.text for part in plans[i]])
                    else:
                        old = previous[chunk_id]
                        assert old_vectors is not None
                        parts = old.parts
                        start = writer.copy(old_vectors[old.start : old.start + old.count])
                        reused += 1
                    record = {
                        "id": chunk_id,
                        "chunk": chunk_to_dict(chunk),
                        "tokens": bm25_tokens(chunk),
                        "hash": digest,
                        "rows": [start, len(parts)],
                        "parts": parts,
                    }
                    records.write(json.dumps(record, ensure_ascii=False) + "\n")
                if files % 500 == 0:
                    logger.info("indexed %d files (%d chunks)", files, len(seen))
        rows, dim = writer.finish(gen_dir / VECTORS_FILE)
        graph = build_graph(skeletons, record_ids, file_refs)
        write_graph(gen_dir, graph, len(record_ids))
        new_manifest = {
            "schema": SCHEMA_VERSION,
            "repo": str(repo_path),
            "model": embedder.name,
            "fingerprint": embedder.fingerprint,
            "max_tokens": embedder.max_tokens,
            "data": generation,
            "rows": rows,
            "dim": dim,
            "graph_version": GRAPH_VERSION,
        }
        unchanged = data is not None and _same_build(manifest, new_manifest, data, gen_dir)
        if not unchanged:
            _write_atomic(
                out / MANIFEST_FILE, json.dumps(new_manifest, indent=2, sort_keys=True) + "\n"
            )
    except BaseException:
        shutil.rmtree(gen_dir, ignore_errors=True)
        raise
    old_vectors = None  # release the memory map before deleting its file
    if unchanged:
        assert manifest is not None
        shutil.rmtree(gen_dir)  # identical to the committed build: keep that one
        generation = str(manifest["data"])
    _remove_stale_files(out, keep=generation, legacy=manifest is not None)
    return IndexStats(
        files=files,
        chunks=len(seen),
        embedded_parts=writer.embedded,
        reused_chunks=reused,
        deleted_chunks=len(previous.keys() - seen),
    )


def _check_same_repo(manifest: dict[str, Any], repo_path: Path, index_dir: Path) -> None:
    """Refuse to reuse an external index dir for a different repo.

    Re-indexing drops chunks that are no longer present, so pointing two
    repos at one directory would silently replace one index with the other.
    An index inside its own repo may move with it, so that case is allowed.
    """
    recorded = manifest.get("repo")
    if not recorded or Path(recorded) == repo_path:
        return
    if index_dir.resolve().is_relative_to(repo_path):
        return
    raise IndexMismatchError(
        f"the index at {index_dir} belongs to {recorded}, not {repo_path}; "
        "use a different index directory or rebuild it for this repo"
    )


def _same_build(
    old_manifest: dict[str, Any] | None,
    new_manifest: dict[str, Any],
    old: IndexData,
    gen_dir: Path,
) -> bool:
    """True if the new generation's manifest and files equal the committed ones."""
    if old_manifest is None:
        return False
    if {k: v for k, v in old_manifest.items() if k != "data"} != {
        k: v for k, v in new_manifest.items() if k != "data"
    }:
        return False
    names = (CHUNKS_FILE, VECTORS_FILE, *GRAPH_FILES)
    return all(
        (old.directory / name).exists()
        and filecmp.cmp(old.directory / name, gen_dir / name, shallow=False)
        for name in names
    )


def _remove_stale_files(index_dir: Path, *, keep: str, legacy: bool) -> None:
    """Delete older generations, and schema 1-2 (Chroma) files if ``legacy``.

    ``legacy`` is only set when the directory already held an index, so an
    unrelated ``chroma/`` or ``chunks.jsonl`` elsewhere is never touched.
    """
    for entry in index_dir.iterdir():
        stale_generation = entry.name.startswith(GENERATION_PREFIX) and entry.name != keep
        if not (stale_generation or (legacy and entry.name in _LEGACY_ENTRIES)):
            continue
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)


class _VectorWriter:
    """Writes L2-normalized float32 part vectors by row, embedding new texts in batches.

    Rows are assigned in record order; reused vectors are written at once and
    new ones when their batch is embedded, into a raw scratch file that
    ``finish`` turns into a ``.npy``.
    """

    def __init__(self, raw_path: Path, embedder: Embedder, batch_size: int) -> None:
        self._raw_path = raw_path
        self._file = raw_path.open("w+b")
        self._embedder = embedder
        self._batch_size = max(1, batch_size)
        self._pending: list[tuple[int, str]] = []  # (row, text)
        self._rows = 0
        self._dim: int | None = None
        self.embedded = 0

    def copy(self, vectors: np.ndarray) -> int:
        start = self._rows
        self._rows += len(vectors)
        self._write(start, np.asarray(vectors, dtype=np.float32))
        return start

    def add(self, texts: Sequence[str]) -> int:
        start = self._rows
        self._pending += [(start + i, text) for i, text in enumerate(texts)]
        self._rows += len(texts)
        if len(self._pending) >= self._batch_size:
            self.flush()
        return start

    def flush(self) -> None:
        while self._pending:
            batch, self._pending = self._pending[: self._batch_size], self._pending[self._batch_size :]
            vectors = _normalized(self._embedder.embed_documents([text for _, text in batch]))
            self.embedded += len(batch)
            # Rows within a batch are consecutive except where reused rows interleave.
            for (row, _), vector in zip(batch, vectors):
                self._write(row, vector[None, :])

    def finish(self, npy_path: Path) -> tuple[int, int]:
        self.flush()
        dim = self._dim or 0
        self._file.close()
        if self._rows == 0:
            np.save(npy_path, np.zeros((0, dim), dtype=np.float32))
        else:
            raw = np.memmap(self._raw_path, dtype=np.float32, mode="r", shape=(self._rows, dim))
            final = np.lib.format.open_memmap(
                npy_path, mode="w+", dtype=np.float32, shape=(self._rows, dim)
            )
            for start in range(0, self._rows, 65_536):
                final[start : start + 65_536] = raw[start : start + 65_536]
            final.flush()
            del final, raw
        self._raw_path.unlink()
        return self._rows, dim

    def _write(self, row: int, block: np.ndarray) -> None:
        if len(block) == 0:
            return
        if self._dim is None:
            self._dim = int(block.shape[1])
        elif block.shape[1] != self._dim:
            raise ValueError(f"embedding width changed from {self._dim} to {block.shape[1]}")
        self._file.seek(row * self._dim * 4)
        self._file.write(np.ascontiguousarray(block, dtype=np.float32).tobytes())


def _normalized(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return np.divide(array, norms, out=np.zeros_like(array), where=norms > 0)
