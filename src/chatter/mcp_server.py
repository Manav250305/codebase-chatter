"""Read-only MCP server over a chatter index (stdio transport).

Tools:
    search(query, k, mode)        ranked chunks with file:line locations
    get_chunk(chunk_id)           one chunk's source and metadata
    list_symbols(path_prefix)     functions/classes/methods under a path
    ask(question, k)              optional: answer with the local generator

The server never builds or modifies an index: it opens the committed build
lazily on the first call and re-opens it when ``manifest.json`` changes, so a
``chatter index`` run elsewhere is picked up without a restart.

Configuration (CLI flags take precedence over the environment):
    CHATTER_REPO            repository root (default: current directory)
    CHATTER_INDEX_DIR       index directory (default: <repo>/.chatter)
    CHATTER_MCP_ENABLE_ASK  "1" to expose the ask tool
    CHATTER_BACKEND         answer backend for ask (mlx | transformers)
    CHATTER_ANSWER_MODEL    answer model for ask
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from chatter.answer import (
    Generator,
    GeneratorConfig,
    answer_question,
    classify_answer,
    format_line_ranges,
)
from chatter.embed import Embedder
from chatter.extract import Chunk
from chatter.index import MANIFEST_FILE, IndexMismatchError, default_index_dir, read_manifest
from chatter.retrieve import Retriever, hit_location, hit_to_dict

SERVER_NAME = "chatter"
MAX_SEARCH_K = 50
MAX_SYMBOLS = 2000
DEFAULT_ASK_K = 8
_TRUE = frozenset({"1", "true", "yes", "on"})

INSTRUCTIONS = """\
Search and read the indexed Python code of one repository. Every result carries a
chunk_id and a path:start-end location to cite.

- search(query, k, mode): hybrid BM25 + dense retrieval over functions, classes,
  methods and module-level code. Use natural language or identifiers.
- get_chunk(chunk_id): full source of a result, with its docstring and imports.
- list_symbols(path_prefix): browse the symbols defined under a path.
- ask(question), if enabled: a short answer from a small local model; prefer
  search + get_chunk and answer yourself when you can.

The index is read-only here and may lag behind the working tree until
`chatter index` is re-run."""

READ_ONLY = ToolAnnotations(
    read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
)


@dataclass(frozen=True, slots=True)
class ServerConfig:
    repo: Path
    index_dir: Path
    enable_ask: bool = False
    backend: str | None = None  # None: default_backend()
    answer_model: str | None = None


def config_from_env(
    *,
    repo: Path | None = None,
    index_dir: Path | None = None,
    enable_ask: bool | None = None,
    backend: str | None = None,
    answer_model: str | None = None,
    environ: Mapping[str, str] = os.environ,
) -> ServerConfig:
    """Explicit arguments win; otherwise the CHATTER_* environment variables."""
    repo_path = Path(repo or environ.get("CHATTER_REPO") or ".").resolve()
    index = index_dir or (Path(environ["CHATTER_INDEX_DIR"]) if environ.get("CHATTER_INDEX_DIR") else None)
    return ServerConfig(
        repo=repo_path,
        index_dir=Path(index).resolve() if index else default_index_dir(repo_path),
        enable_ask=(
            enable_ask
            if enable_ask is not None
            else environ.get("CHATTER_MCP_ENABLE_ASK", "").strip().lower() in _TRUE
        ),
        backend=backend or environ.get("CHATTER_BACKEND") or None,
        answer_model=answer_model or environ.get("CHATTER_ANSWER_MODEL") or None,
    )


def check_index(config: ServerConfig) -> str | None:
    """Problem with the configured index, or None if it looks usable."""
    manifest = read_manifest(config.index_dir)
    if manifest is None:
        return (
            f"No index found at {config.index_dir}. Build it first: "
            f"chatter index {config.repo}"
            + (f" --index-dir {config.index_dir}" if config.index_dir != default_index_dir(config.repo) else "")
        )
    return None


# ---------------------------------------------------------------------------
# Results (pydantic models give clients an output schema)
# ---------------------------------------------------------------------------


class SearchHit(BaseModel):
    rank: int
    chunk_id: str
    path: str
    qualname: str
    kind: str
    lines: list[int] = Field(description="[start, end] of the matching lines (1-based, inclusive)")
    location: str = Field(description="path:start-end to cite")
    signature: str | None = Field(description="First line of the definition, if any")
    score: float
    sources: list[str] = Field(description="Retrievers that found it: bm25, dense")


class SearchResult(BaseModel):
    query: str
    mode: str
    hits: list[SearchHit]


class ChunkResult(BaseModel):
    chunk_id: str
    path: str
    qualname: str
    kind: str
    parent: str | None
    start_line: int
    end_line: int
    location: str
    docstring: str | None
    decorators: list[str]
    imports: list[str]
    is_async: bool
    source: str


class Symbol(BaseModel):
    chunk_id: str
    path: str
    qualname: str
    kind: str
    location: str


class SymbolList(BaseModel):
    path_prefix: str
    total: int = Field(description="Matching symbols before the limit")
    truncated: bool
    symbols: list[Symbol]


class Citation(BaseModel):
    tag: str
    chunk_id: str
    location: str


class AskResult(BaseModel):
    answer: str
    status: str = Field(description="answered | abstained | mixed")
    citations: list[Citation]
    invalid_tags: list[str]
    context_chunk_ids: list[str]
    model: str
    backend: str | None


# ---------------------------------------------------------------------------
# Service: plain methods over a lazily opened, auto-reloading index
# ---------------------------------------------------------------------------


class ChatterService:
    """Read-only access to one index; safe to call from several worker threads."""

    def __init__(
        self,
        config: ServerConfig,
        embedder_factory: Callable[[str], Embedder],
        generator_factory: Callable[[GeneratorConfig], Generator],
    ) -> None:
        self.config = config
        self._embedder_factory = embedder_factory
        self._generator_factory = generator_factory
        self._lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self._retriever: Retriever | None = None
        self._embedder: Embedder | None = None
        self._stamp: tuple[int, int] | None = None
        self._generator: Generator | None = None

    def retriever(self) -> Retriever:
        """The current build, re-opened if manifest.json changed since last use."""
        manifest_path = self.config.index_dir / MANIFEST_FILE
        with self._lock:
            try:
                stat = manifest_path.stat()
            except OSError:
                raise ToolError(check_index(self.config) or f"cannot read {manifest_path}") from None
            stamp = (stat.st_mtime_ns, stat.st_size)
            if self._retriever is None or stamp != self._stamp:
                manifest = read_manifest(self.config.index_dir) or {}
                model = str(manifest.get("model", ""))
                if self._embedder is None or self._embedder.name != model:
                    try:
                        self._embedder = self._embedder_factory(model)
                    except (OSError, ValueError, RuntimeError, ImportError) as exc:
                        raise ToolError(f"could not load embedding model {model!r}: {exc}") from exc
                try:
                    self._retriever = Retriever.open(self.config.index_dir, self._embedder)
                except (IndexMismatchError, FileNotFoundError) as exc:
                    raise ToolError(f"{exc}. Re-run: chatter index {self.config.repo} --rebuild") from exc
                self._stamp = stamp
            return self._retriever

    def search(self, query: str, k: int = 10, mode: str = "fused") -> SearchResult:
        if not query.strip():
            raise ToolError("query is empty")
        hits = self.retriever().search(query, k=k, mode=mode)  # type: ignore[arg-type]
        results = []
        for rank, hit in enumerate(hits, 1):
            data = hit_to_dict(rank, hit)
            results.append(
                SearchHit(
                    rank=rank,
                    chunk_id=hit.chunk_id,
                    path=data["path"],
                    qualname=data["qualname"],
                    kind=data["kind"],
                    lines=data["lines"],
                    location=hit_location(hit),
                    signature=_signature(hit.chunk),
                    score=hit.score,
                    sources=data["sources"],
                )
            )
        return SearchResult(query=query, mode=mode, hits=results)

    def get_chunk(self, chunk_id: str) -> ChunkResult:
        chunk = self.retriever().chunk(chunk_id)
        if chunk is None:
            raise ToolError(f"unknown chunk_id {chunk_id!r}; use search or list_symbols to find ids")
        return ChunkResult(
            chunk_id=chunk_id,
            path=chunk.path,
            qualname=chunk.qualname,
            kind=chunk.kind,
            parent=chunk.parent,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            location=_location(chunk),
            docstring=chunk.docstring,
            decorators=list(chunk.decorators),
            imports=list(chunk.imports),
            is_async=chunk.is_async,
            source=chunk.source,
        )

    def list_symbols(
        self, path_prefix: str = "", kind: str | None = None, limit: int = 200
    ) -> SymbolList:
        prefix = path_prefix.strip().removeprefix("./")
        matches = sorted(
            (
                (chunk.path, chunk.start_line, chunk_id, chunk)
                for chunk_id, chunk in self.retriever().chunks()
                if chunk.kind != "module"
                and chunk.path.startswith(prefix)
                and (kind is None or chunk.kind == kind)
            ),
            key=lambda item: item[:3],
        )
        return SymbolList(
            path_prefix=prefix,
            total=len(matches),
            truncated=len(matches) > limit,
            symbols=[
                Symbol(
                    chunk_id=chunk_id,
                    path=chunk.path,
                    qualname=chunk.qualname,
                    kind=chunk.kind,
                    location=_location(chunk),
                )
                for _, _, chunk_id, chunk in matches[:limit]
            ],
        )

    def ask(self, question: str, k: int = DEFAULT_ASK_K) -> AskResult:
        if not self.config.enable_ask:
            raise ToolError("ask is disabled; start the server with --enable-ask")
        if not question.strip():
            raise ToolError("question is empty")
        hits = self.retriever().search(question, k=k)
        if not hits:
            return AskResult(
                answer="No indexed code matched the question.",
                status="abstained",
                citations=[],
                invalid_tags=[],
                context_chunk_ids=[],
                model="",
                backend=None,
            )
        with self._generate_lock:  # one generation at a time; the model is not shared safely
            if self._generator is None:
                config = GeneratorConfig(model_name=self.config.answer_model, backend=self.config.backend)
                try:
                    self._generator = self._generator_factory(config)
                except (OSError, ValueError, RuntimeError, ImportError, MemoryError) as exc:
                    raise ToolError(f"could not load the answer model: {exc}") from exc
            generator = self._generator
            result = answer_question(question, hits, generator)
        return AskResult(
            answer=result.text,
            status=classify_answer(result.text).value,
            citations=[
                Citation(tag=c.tag, chunk_id=c.block.chunk_id, location=c.block.location)
                for c in result.citations
            ],
            invalid_tags=list(result.unknown_tags),
            context_chunk_ids=[block.chunk_id for block in result.blocks],
            model=generator.name,
            backend=getattr(generator, "backend", None),
        )


def _location(chunk: Chunk) -> str:
    return f"{chunk.path}:{format_line_ranges(chunk.line_numbers())}"


def _signature(chunk: Chunk) -> str | None:
    if chunk.kind == "module":
        return None
    for line in chunk.source.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith(("@", "#")):
            return stripped
    return None


# ---------------------------------------------------------------------------
# MCP wiring
# ---------------------------------------------------------------------------

Mode = Literal["fused", "bm25", "dense"]
SymbolKind = Literal["function", "method", "class", "lambda"]


def build_server(service: ChatterService) -> MCPServer:
    server = MCPServer(name=SERVER_NAME, instructions=INSTRUCTIONS)

    @server.tool(annotations=READ_ONLY)
    def search(
        query: Annotated[str, Field(description="Natural-language question or identifiers")],
        k: Annotated[int, Field(ge=1, le=MAX_SEARCH_K, description="Number of results")] = 10,
        mode: Annotated[Mode, Field(description="fused (default), bm25 only, or dense only")] = "fused",
    ) -> SearchResult:
        """Find the most relevant code chunks for a query, with file:line locations."""
        return service.search(query, k, mode)

    @server.tool(annotations=READ_ONLY)
    def get_chunk(
        chunk_id: Annotated[str, Field(description="A chunk_id from search or list_symbols")],
    ) -> ChunkResult:
        """Full source and metadata of one chunk."""
        return service.get_chunk(chunk_id)

    @server.tool(annotations=READ_ONLY)
    def list_symbols(
        path_prefix: Annotated[
            str, Field(description="Repo-relative path prefix, e.g. 'src/chatter/' ('' for all)")
        ] = "",
        kind: Annotated[SymbolKind | None, Field(description="Only this kind of symbol")] = None,
        limit: Annotated[int, Field(ge=1, le=MAX_SYMBOLS, description="Maximum symbols returned")] = 200,
    ) -> SymbolList:
        """List functions, classes and methods defined under a path, in file order."""
        return service.list_symbols(path_prefix, kind, limit)

    if service.config.enable_ask:

        @server.tool(annotations=READ_ONLY)
        def ask(
            question: Annotated[str, Field(description="Question about the code")],
            k: Annotated[int, Field(ge=1, le=20, description="Chunks given to the model")] = DEFAULT_ASK_K,
        ) -> AskResult:
            """Answer from retrieved code with a small local model (slow; cites chunk ids)."""
            return service.ask(question, k)

    return server


def run_stdio(service: ChatterService) -> None:
    build_server(service).run("stdio")

