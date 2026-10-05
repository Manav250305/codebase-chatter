from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client
from typer.testing import CliRunner

from chatter.answer import ABSTENTION, GeneratorConfig
from chatter.cli import make_app
from chatter.index import build_index
from chatter.mcp_server import ChatterService, ServerConfig, build_server, check_index, config_from_env
from chatter.retrieve import Retriever
from conftest import FakeGenerator, HashEmbedder, WriteRepo

REPO = {
    "net/http.py": '''
        """HTTP helpers."""
        import json

        def parse_http_response(raw):
            """Split status line, headers, and body."""
            head, _, body = raw.partition("\\r\\n\\r\\n")
            return head, body

        class RetryPolicy:
            @staticmethod
            def backoff_delay(attempt):
                return 2 ** attempt
    ''',
    "store/cache.py": '''
        def evict_least_recently_used(cache, capacity):
            while len(cache) > capacity:
                cache.popitem(last=False)
    ''',
    "tests/test_cache.py": "def test_evict():\n    assert True\n",
}


def make_service(
    root: Path,
    *,
    enable_ask: bool = False,
    reply: object = "It splits head and body [C1].",
    index_dir: Path | None = None,
) -> tuple[ChatterService, dict[str, Any]]:
    log: dict[str, Any] = {"embedders": [], "generators": []}

    def embedder_factory(name: str) -> HashEmbedder:
        log["embedders"].append(name)
        return HashEmbedder(name=name)

    def generator_factory(config: GeneratorConfig) -> FakeGenerator:
        log["generators"].append(config)
        return FakeGenerator(reply)

    config = ServerConfig(
        repo=root, index_dir=index_dir or root / ".chatter", enable_ask=enable_ask, backend="mlx"
    )
    return ChatterService(config, embedder_factory, generator_factory), log


@pytest.fixture
def indexed(write_repo: WriteRepo) -> Path:
    root = write_repo(REPO)
    build_index(root, HashEmbedder())
    return root


def call(service: ChatterService, tool: str, arguments: dict[str, Any] | None = None) -> Any:
    """Call a tool through a real MCP client session (in-memory transport)."""

    async def main() -> Any:
        async with Client(build_server(service)) as client:
            return await client.call_tool(tool, arguments or {})

    return anyio.run(main)


def tools(service: ChatterService) -> dict[str, Any]:
    async def main() -> Any:
        async with Client(build_server(service)) as client:
            return {t.name: t for t in (await client.list_tools()).tools}

    return anyio.run(main)


def error_text(result: Any) -> str:
    assert result.is_error, result
    return " ".join(getattr(block, "text", "") for block in result.content)


# ---------------------------------------------------------------------------
# Tool listing
# ---------------------------------------------------------------------------


def test_tools_are_read_only_with_schemas(indexed: Path) -> None:
    service, _ = make_service(indexed)
    listed = tools(service)
    assert set(listed) == {"search", "get_chunk", "list_symbols"}  # ask is opt-in
    for tool in listed.values():
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.destructive_hint is False
        assert tool.output_schema is not None
    search_props = listed["search"].input_schema["properties"]
    assert set(search_props) == {"query", "k", "mode"}
    assert search_props["mode"]["enum"] == ["fused", "bm25", "dense"]
    assert search_props["mode"]["default"] == "fused"
    assert search_props["k"]["minimum"] == 1 and search_props["k"]["maximum"] == 50
    kind = listed["list_symbols"].input_schema["properties"]["kind"]
    assert kind["anyOf"][0]["enum"] == ["function", "method", "class", "lambda"]


def test_ask_tool_listed_only_when_enabled(indexed: Path) -> None:
    service, _ = make_service(indexed, enable_ask=True)
    assert "ask" in tools(service)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_matches_retriever(indexed: Path) -> None:
    service, log = make_service(indexed)
    result = call(service, "search", {"query": "parse_http_response", "k": 3})
    assert not result.is_error
    data = result.structured_content
    assert data["query"] == "parse_http_response" and data["mode"] == "fused"
    expected = Retriever.open(indexed / ".chatter", HashEmbedder()).search("parse_http_response", k=3)
    assert [h["chunk_id"] for h in data["hits"]] == [h.chunk_id for h in expected]
    top = data["hits"][0]
    assert top["rank"] == 1 and top["chunk_id"] == "net/http.py::parse_http_response"
    assert top["location"] == "net/http.py:4-7" and top["lines"] == [4, 7]
    assert top["signature"] == "def parse_http_response(raw):"
    assert top["sources"] == ["bm25", "dense"] and top["score"] > 0
    assert log["embedders"] == ["fake-hash"]  # model taken from the index manifest


def test_search_modes_and_decorated_signature(indexed: Path) -> None:
    service, _ = make_service(indexed)
    bm25 = call(service, "search", {"query": "backoff_delay", "mode": "bm25"}).structured_content
    assert all(h["sources"] == ["bm25"] for h in bm25["hits"])
    hit = next(h for h in bm25["hits"] if h["chunk_id"] == "net/http.py::RetryPolicy.backoff_delay")
    assert hit["signature"] == "def backoff_delay(attempt):"  # decorator line skipped
    dense = call(service, "search", {"query": "backoff_delay", "mode": "dense"}).structured_content
    assert all(h["sources"] == ["dense"] for h in dense["hits"])


def test_search_validates_arguments(indexed: Path) -> None:
    service, _ = make_service(indexed)
    assert call(service, "search", {"query": "x", "k": 0}).is_error
    assert call(service, "search", {"query": "x", "k": 51}).is_error
    assert call(service, "search", {"query": "x", "mode": "hybrid"}).is_error
    assert "query is empty" in error_text(call(service, "search", {"query": "   "}))


# ---------------------------------------------------------------------------
# get_chunk / list_symbols
# ---------------------------------------------------------------------------


def test_get_chunk_returns_source_and_metadata(indexed: Path) -> None:
    service, _ = make_service(indexed)
    data = call(service, "get_chunk", {"chunk_id": "net/http.py::RetryPolicy.backoff_delay"}).structured_content
    assert data["kind"] == "method" and data["parent"] == "RetryPolicy"
    assert data["decorators"] == ["@staticmethod"] and data["imports"] == ["import json"]
    assert data["source"].startswith("    @staticmethod\n    def backoff_delay")
    assert data["location"] == "net/http.py:10-12"

    module = call(service, "get_chunk", {"chunk_id": "net/http.py::<module>"}).structured_content
    assert module["kind"] == "module" and module["docstring"] == "HTTP helpers."
    assert module["location"] == "net/http.py:1-2"


def test_get_chunk_unknown_id_is_a_readable_error(indexed: Path) -> None:
    service, _ = make_service(indexed)
    message = error_text(call(service, "get_chunk", {"chunk_id": "nope.py::missing"}))
    assert "unknown chunk_id 'nope.py::missing'" in message and "search or list_symbols" in message


def test_list_symbols_prefix_kind_and_limit(indexed: Path) -> None:
    service, _ = make_service(indexed)
    everything = call(service, "list_symbols").structured_content
    assert [s["chunk_id"] for s in everything["symbols"]] == [
        "net/http.py::parse_http_response",
        "net/http.py::RetryPolicy",
        "net/http.py::RetryPolicy.backoff_delay",
        "store/cache.py::evict_least_recently_used",
        "tests/test_cache.py::test_evict",
    ]  # file order; module chunks are not symbols
    assert everything["total"] == 5 and not everything["truncated"]

    net = call(service, "list_symbols", {"path_prefix": "./net/"}).structured_content
    assert net["path_prefix"] == "net/" and net["total"] == 3
    methods = call(service, "list_symbols", {"kind": "method"}).structured_content
    assert [s["qualname"] for s in methods["symbols"]] == ["RetryPolicy.backoff_delay"]
    limited = call(service, "list_symbols", {"limit": 2}).structured_content
    assert limited["total"] == 5 and limited["truncated"] and len(limited["symbols"]) == 2
    assert call(service, "list_symbols", {"path_prefix": "nowhere/"}).structured_content["total"] == 0


# ---------------------------------------------------------------------------
# ask (opt-in)
# ---------------------------------------------------------------------------


def test_ask_answers_with_citations(indexed: Path) -> None:
    service, log = make_service(indexed, enable_ask=True)
    data = call(service, "ask", {"question": "how is an http response parsed"}).structured_content
    assert data["status"] == "answered" and data["answer"] == "It splits head and body [C1]."
    assert data["citations"][0]["tag"] == "C1" and data["citations"][0]["location"]
    assert data["context_chunk_ids"] and data["model"] == "fake-llm"
    assert log["generators"] == [GeneratorConfig(model_name=None, backend="mlx")]
    call(service, "ask", {"question": "again"})
    assert len(log["generators"]) == 1  # the model is loaded once


def test_ask_reports_abstention(indexed: Path) -> None:
    service, _ = make_service(indexed, enable_ask=True, reply=ABSTENTION)
    assert call(service, "ask", {"question": "oauth?"}).structured_content["status"] == "abstained"


def test_ask_disabled_service_refuses() -> None:
    service, _ = make_service(Path("/nonexistent"))
    with pytest.raises(Exception, match="ask is disabled"):
        service.ask("anything")


def test_ask_model_load_failure_is_a_tool_error(indexed: Path) -> None:
    service, _ = make_service(indexed, enable_ask=True)

    def broken(config: GeneratorConfig) -> FakeGenerator:
        raise OSError("gated repo")

    service._generator_factory = broken  # type: ignore[assignment]
    assert "could not load the answer model: gated repo" in error_text(call(service, "ask", {"question": "x"}))


# ---------------------------------------------------------------------------
# Read-only behaviour, reloading, configuration
# ---------------------------------------------------------------------------


def _snapshot(directory: Path) -> dict[str, tuple[int, int]]:
    return {
        str(p.relative_to(directory)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in directory.rglob("*")
        if p.is_file()
    }


def test_tools_never_modify_the_index(indexed: Path) -> None:
    service, _ = make_service(indexed, enable_ask=True)
    before = _snapshot(indexed / ".chatter")
    call(service, "search", {"query": "cache"})
    call(service, "get_chunk", {"chunk_id": "store/cache.py::evict_least_recently_used"})
    call(service, "list_symbols")
    call(service, "ask", {"question": "what evicts?"})
    assert _snapshot(indexed / ".chatter") == before


def test_reindex_is_picked_up_without_restart(indexed: Path) -> None:
    service, _ = make_service(indexed)
    assert call(service, "list_symbols", {"path_prefix": "extra/"}).structured_content["total"] == 0
    (indexed / "extra").mkdir()
    (indexed / "extra" / "new.py").write_text("def brand_new():\n    pass\n")
    build_index(indexed, HashEmbedder())
    symbols = call(service, "list_symbols", {"path_prefix": "extra/"}).structured_content["symbols"]
    assert [s["chunk_id"] for s in symbols] == ["extra/new.py::brand_new"]


def test_missing_index_is_reported(tmp_path: Path) -> None:
    config = ServerConfig(repo=tmp_path, index_dir=tmp_path / ".chatter")
    assert "No index found" in (check_index(config) or "")
    service, _ = make_service(tmp_path)
    assert "No index found" in error_text(call(service, "search", {"query": "x"}))


def test_config_from_env(tmp_path: Path) -> None:
    env = {
        "CHATTER_REPO": str(tmp_path),
        "CHATTER_INDEX_DIR": str(tmp_path / "idx"),
        "CHATTER_MCP_ENABLE_ASK": "yes",
        "CHATTER_BACKEND": "transformers",
        "CHATTER_ANSWER_MODEL": "x/y",
    }
    config = config_from_env(environ=env)
    assert config == ServerConfig(tmp_path.resolve(), (tmp_path / "idx").resolve(), True, "transformers", "x/y")
    explicit = config_from_env(repo=tmp_path / "r", enable_ask=False, environ=env)
    assert explicit.repo == (tmp_path / "r").resolve() and explicit.enable_ask is False
    default = config_from_env(repo=tmp_path, environ={})
    assert default.index_dir == tmp_path.resolve() / ".chatter" and not default.enable_ask


def test_cli_mcp_fails_fast_without_index(tmp_path: Path) -> None:
    app = make_app(embedder_factory=lambda name: HashEmbedder(name=name))
    result = CliRunner().invoke(app, ["mcp", "--path", str(tmp_path)])
    assert result.exit_code == 1 and "No index found" in result.output


@pytest.mark.slow
def test_stdio_server_end_to_end(indexed: Path) -> None:
    """Spawn `chatter mcp` as a real stdio subprocess (real embedding model)."""
    from mcp import StdioServerParameters

    root = indexed
    build_index(root, None, rebuild=True)  # real model index
    chatter = Path(sys.executable).parent / "chatter"
    params = StdioServerParameters(command=str(chatter), args=["mcp", "--path", str(root)])

    async def main() -> Any:
        async with Client(params) as client:
            listed = {t.name for t in (await client.list_tools()).tools}
            result = await client.call_tool("search", {"query": "parse an HTTP response", "k": 3})
            return listed, result

    listed, result = anyio.run(main)
    assert listed == {"search", "get_chunk", "list_symbols"}
    assert result.structured_content["hits"][0]["chunk_id"] == "net/http.py::parse_http_response"
