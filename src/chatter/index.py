"""Build a persistent hybrid (BM25 + dense) index over a repository's Python code.

Layout of an index directory (default ``<repo>/.chatter``)::

    manifest.json   schema version, embedding model, fingerprint
    chunks.jsonl    one record per chunk: id, chunk fields, BM25 tokens
    chroma/         Chroma collection of embedded chunk parts

Chunks whose embedding text exceeds the model's token budget are split by lines
into parts (``<chunk id>#<n>``); each part repeats the chunk header. Every part
stores a hash of its chunk's embedding text, so re-indexing only embeds chunks
that are new or changed and deletes chunks that disappeared.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
import pathspec
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

from chatter.embed import Embedder, SentenceTransformerEmbedder
from chatter.extract import DEFAULT_MAX_BYTES, Chunk, extract_file

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
INDEX_DIRNAME = ".chatter"
MANIFEST_FILE = "manifest.json"
CHUNKS_FILE = "chunks.jsonl"
CHROMA_DIR = "chroma"
COLLECTION_NAME = "chunks"

SKIP_DIRS = frozenset(
    {".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__", INDEX_DIRNAME}
)
SOURCE_SUFFIXES = frozenset({".py"})
HEADER_DOCSTRING_CHARS = 400
_CHROMA_PAGE = 5_000

_IDENTIFIER_RE = re.compile(r"[^\W\d]\w*|\d+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


class IndexMismatchError(RuntimeError):
    """The on-disk index was built with a different model or schema."""


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


def tokenize_code(text: str) -> list[str]:
    """Lowercased identifiers plus their snake_case / camelCase parts.

    ``parseHTTPResponse`` -> ``parsehttpresponse parse http response``;
    ``max_retry_count`` -> ``max_retry_count max retry count``.
    """
    tokens: list[str] = []
    for match in _IDENTIFIER_RE.finditer(text):
        word = match.group()
        lower = word.lower()
        tokens.append(lower)
        parts = [
            part.lower()
            for piece in word.split("_")
            if piece
            for part in _CAMEL_BOUNDARY_RE.split(piece)
            if part
        ]
        if len(parts) > 1 or (parts and parts[0] != lower):
            tokens.extend(parts)
    return tokens


def bm25_tokens(chunk: Chunk) -> list[str]:
    return tokenize_code(f"{chunk.path} {chunk.qualname} {chunk.source}")


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
# Storage helpers (shared with retrieve.py)
# ---------------------------------------------------------------------------


def default_index_dir(repo: Path) -> Path:
    return Path(repo).resolve() / INDEX_DIRNAME


def _client(index_dir: Path) -> ClientAPI:
    return chromadb.PersistentClient(
        path=str(Path(index_dir) / CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def open_collection(index_dir: Path, *, create: bool) -> Collection:
    client = _client(index_dir)
    if create:
        return client.get_or_create_collection(
            COLLECTION_NAME, embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
        )
    return client.get_collection(COLLECTION_NAME, embedding_function=None)


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


@dataclass(slots=True)
class _Stored:
    hash: str
    part_ids: set[str] = field(default_factory=set)


def build_index(
    repo: Path | str,
    embedder: Embedder | None = None,
    *,
    index_dir: Path | str | None = None,
    config: IndexConfig = IndexConfig(),
    rebuild: bool = False,
) -> IndexStats:
    """Index every Python file under ``repo``; re-running only embeds changes.

    Raises ``IndexMismatchError`` if an existing index used a different model
    or schema, unless ``rebuild`` is set (which drops all stored vectors).
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
    if rebuild:
        _drop_collection(out)
    collection = open_collection(out, create=True)
    stored = _stored_parts(collection)

    writer = _PartWriter(collection, embedder, config.batch_size)
    seen: set[str] = set()
    files = reused = 0
    chunks_tmp = out / (CHUNKS_FILE + ".tmp")
    with chunks_tmp.open("w", encoding="utf-8") as records:
        for path in iter_source_files(repo_path):
            files += 1
            rel = path.relative_to(repo_path).as_posix()
            chunks = extract_file(path, display_path=rel, max_bytes=config.max_file_bytes)
            for chunk_id, chunk in zip(assign_chunk_ids(chunks), chunks):
                seen.add(chunk_id)
                digest = content_hash(embedder.fingerprint, embedding_text(chunk))
                previous = stored.get(chunk_id)
                if previous is not None and previous.hash == digest:
                    reused += 1
                else:
                    stale = previous.part_ids if previous is not None else set()
                    writer.add(chunk_id, chunk, digest, stale)
                record = {"id": chunk_id, "chunk": chunk_to_dict(chunk), "tokens": bm25_tokens(chunk)}
                records.write(json.dumps(record, ensure_ascii=False) + "\n")
            if files % 500 == 0:
                logger.info("indexed %d files (%d chunks)", files, len(seen))
    writer.flush()

    removed = [cid for cid in stored if cid not in seen]
    _delete_ids(collection, [pid for cid in removed for pid in stored[cid].part_ids])

    os.replace(chunks_tmp, out / CHUNKS_FILE)
    _write_atomic(
        out / MANIFEST_FILE,
        json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "model": embedder.name,
                "fingerprint": embedder.fingerprint,
                "max_tokens": embedder.max_tokens,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return IndexStats(
        files=files,
        chunks=len(seen),
        embedded_parts=writer.embedded,
        reused_chunks=reused,
        deleted_chunks=len(removed),
    )


class _PartWriter:
    """Buffers changed chunks; splits, embeds, and upserts them in batches."""

    def __init__(self, collection: Collection, embedder: Embedder, batch_size: int) -> None:
        self._collection = collection
        self._embedder = embedder
        self._batch_size = max(1, batch_size)
        self._pending: list[tuple[str, Chunk, str, set[str]]] = []
        self.embedded = 0

    def add(self, chunk_id: str, chunk: Chunk, digest: str, stale: set[str]) -> None:
        self._pending.append((chunk_id, chunk, digest, stale))
        if len(self._pending) >= self._batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        pending, self._pending = self._pending, []
        plans = plan_parts([chunk for _, chunk, _, _ in pending], self._embedder)

        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        obsolete: list[str] = []
        for (chunk_id, _, digest, stale), parts in zip(pending, plans):
            new_ids = [f"{chunk_id}#{part.index}" for part in parts]
            obsolete.extend(sorted(stale - set(new_ids)))
            for part_id, part in zip(new_ids, parts):
                ids.append(part_id)
                texts.append(part.text)
                metadatas.append(
                    {
                        "chunk_id": chunk_id,
                        "hash": digest,
                        "part": part.index,
                        "parts": len(parts),
                        "first": part.first,
                        "last": part.last,
                    }
                )

        _delete_ids(self._collection, obsolete)
        for start in range(0, len(ids), self._batch_size):
            end = start + self._batch_size
            vectors = self._embedder.embed_documents(texts[start:end])
            self._collection.upsert(
                ids=ids[start:end], embeddings=vectors, metadatas=metadatas[start:end]
            )
            self.embedded += len(vectors)


def _stored_parts(collection: Collection) -> dict[str, _Stored]:
    """chunk id -> stored hash and part ids. Mixed hashes force a re-embed."""
    stored: dict[str, _Stored] = {}
    offset = 0
    while True:
        page = collection.get(include=["metadatas"], limit=_CHROMA_PAGE, offset=offset)
        ids = page["ids"]
        if not ids:
            return stored
        for part_id, meta in zip(ids, page["metadatas"] or []):
            chunk_id = str(meta["chunk_id"])
            entry = stored.setdefault(chunk_id, _Stored(hash=str(meta["hash"])))
            if entry.hash != meta["hash"]:
                entry.hash = ""  # interrupted earlier update; never matches
            entry.part_ids.add(part_id)
        offset += len(ids)


def _delete_ids(collection: Collection, ids: Sequence[str]) -> None:
    for start in range(0, len(ids), _CHROMA_PAGE):
        collection.delete(ids=list(ids[start : start + _CHROMA_PAGE]))


def _drop_collection(index_dir: Path) -> None:
    client = _client(index_dir)
    if any(c.name == COLLECTION_NAME for c in client.list_collections()):
        client.delete_collection(COLLECTION_NAME)
