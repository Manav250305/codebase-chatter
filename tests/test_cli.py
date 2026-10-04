from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from chatter.answer import ABSTENTION, DEFAULT_ANSWER_MODEL, GeneratorConfig
from chatter.cli import make_app
from chatter.embed import DEFAULT_MODEL
from conftest import FakeGenerator, HashEmbedder, WriteRepo

runner = CliRunner()

REPO = {
    "net/http.py": '''
        """HTTP helpers."""
        import json

        def parse_http_response(raw):
            """Split status line, headers, and body."""
            head, _, body = raw.partition("\\r\\n\\r\\n")
            return head, body

        class RetryPolicy:
            def backoff_delay(self, attempt):
                return 2 ** attempt
    ''',
    "cfg.py": "import os\n\ndef f():\n    pass\n\nA = 1\nB = 2\n",
}


class Harness:
    """App with offline factories; records which models were requested."""

    def __init__(self, reply: object = "It splits the raw response [C1].") -> None:
        self.embedder_models: list[str] = []
        self.generator_configs: list[GeneratorConfig] = []
        self.generator = FakeGenerator(reply)
        self.generator_error: Exception | None = None
        self.app = make_app(
            embedder_factory=self._embedder, generator_factory=self._generator
        )

    def _embedder(self, name: str) -> HashEmbedder:
        self.embedder_models.append(name)
        return HashEmbedder(name=name)

    def _generator(self, config: GeneratorConfig) -> FakeGenerator:
        self.generator_configs.append(config)
        if self.generator_error is not None:
            raise self.generator_error
        return self.generator

    def run(self, *args: str):  # type: ignore[no-untyped-def]
        return runner.invoke(self.app, list(args))


@pytest.fixture
def repo(write_repo: WriteRepo) -> Path:
    return write_repo(REPO)


@pytest.fixture
def indexed(repo: Path) -> Callable[..., Harness]:
    def make(reply: object = "It splits the raw response [C1].") -> Harness:
        harness = Harness(reply)
        result = harness.run("index", str(repo))
        assert result.exit_code == 0, result.output
        return harness

    return make


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


def test_index_prints_counts_and_timing(repo: Path) -> None:
    h = Harness()
    result = h.run("index", str(repo))
    assert result.exit_code == 0, result.output
    assert "Indexed 2 files: 6 chunks (6 embedded, 0 unchanged, 0 deleted) in " in result.output
    assert f"model: {DEFAULT_MODEL}" in result.output
    assert h.embedder_models == [DEFAULT_MODEL]

    again = h.run("index", str(repo), "--batch-size", "2")
    assert "(0 embedded, 6 unchanged, 0 deleted)" in again.output


def test_index_reports_deletions(repo: Path) -> None:
    h = Harness()
    h.run("index", str(repo))
    (repo / "cfg.py").unlink()
    result = h.run("index", str(repo))
    assert "Indexed 1 files: 4 chunks (0 embedded, 4 unchanged, 2 deleted)" in result.output


def test_index_reuses_model_recorded_in_index(repo: Path) -> None:
    h = Harness()
    h.run("index", str(repo), "--model", "custom/embedder")
    h.run("index", str(repo))
    assert h.embedder_models == ["custom/embedder", "custom/embedder"]


def test_index_model_mismatch_requires_rebuild(repo: Path) -> None:
    h = Harness()
    h.run("index", str(repo))
    result = h.run("index", str(repo), "--model", "other/model")
    assert result.exit_code == 1
    assert "built with 'BAAI/bge-small-en-v1.5'" in result.output and "--rebuild" in result.output
    assert h.embedder_models == [DEFAULT_MODEL]  # never loaded the new model

    rebuilt = h.run("index", str(repo), "--model", "other/model", "--rebuild")
    assert rebuilt.exit_code == 0 and "(6 embedded" in rebuilt.output
    assert "model: other/model" in rebuilt.output


def test_index_missing_directory(tmp_path: Path) -> None:
    result = Harness().run("index", str(tmp_path / "nope"))
    assert result.exit_code == 1 and "Not a directory" in result.output


def test_index_embedding_model_load_failure(repo: Path) -> None:
    def broken(name: str) -> HashEmbedder:
        raise OSError(f"{name} is not a valid model identifier")

    app = make_app(embedder_factory=broken)
    result = runner.invoke(app, ["index", str(repo), "--model", "nope/nope"])
    assert result.exit_code == 1
    assert "Could not load embedding model 'nope/nope'" in result.output


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", [["search", "x"], ["ask", "x"]])
def test_no_index_yet(command: list[str], tmp_path: Path) -> None:
    result = Harness().run(*command, "--path", str(tmp_path))
    assert result.exit_code == 1
    assert "No index found" in result.output and f"chatter index {tmp_path}" in result.output


def test_search_table(indexed: Callable[..., Harness], repo: Path) -> None:
    result = indexed().run("search", "parse_http_response", "-k", "3", "--path", str(repo))
    assert result.exit_code == 0, result.output
    rows = result.output.strip().splitlines()
    assert len(rows) == 3
    rank, score, sources, location, chunk_id = rows[0].split()
    assert (rank, sources) == ("1", "bm25+dense")
    assert location == "net/http.py:4-7" and chunk_id == "net/http.py::parse_http_response"
    assert float(score) > 0


def test_search_module_chunk_shows_all_spans(indexed: Callable[..., Harness], repo: Path) -> None:
    result = indexed().run("search", "A B os", "--path", str(repo))
    row = next(line for line in result.output.splitlines() if "cfg.py::<module>" in line)
    assert "cfg.py:1-1,6-7" in row


def test_search_json(indexed: Callable[..., Harness], repo: Path) -> None:
    result = indexed().run("search", "backoff delay", "--json", "-k", "2", "--path", str(repo))
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert [d["rank"] for d in data] == [1, 2]
    top = data[0]
    assert top["chunk_id"] == "net/http.py::RetryPolicy.backoff_delay"
    assert top["kind"] == "method" and top["path"] == "net/http.py"
    assert top["lines"] == [10, 11] and top["spans"] == [[10, 11]]
    assert set(top["sources"]) <= {"bm25", "dense"} and top["ranks"] and top["raw_scores"]


def test_search_empty_results(write_repo: WriteRepo) -> None:
    root = write_repo({"README.md": "no code"})
    h = Harness()
    h.run("index", str(root))
    assert h.run("search", "anything", "--path", str(root)).output.strip() == "No results."
    assert json.loads(h.run("search", "anything", "--json", "--path", str(root)).output) == []


def test_search_uses_index_model_and_reports_mismatch(repo: Path) -> None:
    h = Harness()
    h.run("index", str(repo), "--model", "custom/embedder")
    h.run("search", "x", "--path", str(repo))
    assert h.embedder_models[-1] == "custom/embedder"

    app = make_app(embedder_factory=lambda name: HashEmbedder(name="something-else"))
    result = runner.invoke(app, ["search", "x", "--path", str(repo)])
    assert result.exit_code == 1
    assert "custom/embedder" in result.output and "--rebuild" in result.output


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


def test_ask_prints_answer_and_citations(indexed: Callable[..., Harness], repo: Path) -> None:
    h = indexed("It splits on the blank line [C1] and returns head and body [C1].")
    result = h.run("ask", "how is an http response parsed", "--path", str(repo))
    assert result.exit_code == 0, result.output
    assert "It splits on the blank line [C1]" in result.output
    citations = result.output.split("Citations:")[1]
    assert "[C1] net/http.py::parse_http_response  net/http.py:4-7" in citations
    assert h.generator_configs == [GeneratorConfig(model_name=None, backend="transformers")]
    assert h.generator_configs[0].resolved_model == DEFAULT_ANSWER_MODEL
    system, user = h.generator.prompts[0]
    assert "<chunk tag=\"C1\" id=\"net/http.py::parse_http_response\"" in user


def test_ask_model_and_budget_flags(indexed: Callable[..., Harness], repo: Path) -> None:
    h = indexed("Answer [C1].")
    result = h.run(
        "ask", "parse http response", "--path", str(repo), "-k", "2",
        "--model", "Qwen/Qwen2.5-Coder-1.5B-Instruct", "--max-context-tokens", "256",
    )
    assert result.exit_code == 0, result.output
    assert h.generator_configs[0].model_name == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    system, user = h.generator.prompts[0]
    assert h.generator.prompt_tokens(system, user) <= 256


def test_ask_abstention_is_reported_distinctly(indexed: Callable[..., Harness], repo: Path) -> None:
    result = indexed(ABSTENTION).run("ask", "how is auth handled", "--path", str(repo))
    assert result.exit_code == 0, result.output
    assert "NOT FOUND: the retrieved code does not contain the answer." in result.output
    assert "Citations:" not in result.output
    searched = result.output.split("Searched:")[1]
    assert "[C1]" in searched and ("net/http.py" in searched or "cfg.py" in searched)


def test_ask_mixed_abstention_note(indexed: Callable[..., Harness], repo: Path) -> None:
    result = indexed(f"Partly: it parses [C1]. {ABSTENTION}").run(
        "ask", "parse and auth", "--path", str(repo)
    )
    assert "does not fully answer" in result.output and "Citations:" in result.output


def test_ask_unknown_tags_and_uncited_answers(indexed: Callable[..., Harness], repo: Path) -> None:
    result = indexed("See [C1] and [C42].").run("ask", "parse", "--path", str(repo), "-k", "2")
    assert "unknown tag(s): C42" in result.output
    uncited = indexed("Some answer without tags.").run("ask", "parse", "--path", str(repo))
    assert "(none - the answer cites no chunks)" in uncited.output


def test_ask_repetition_notice(indexed: Callable[..., Harness], repo: Path) -> None:
    def looping(system: str, user: str):  # type: ignore[no-untyped-def]
        yield "It parses responses [C1].\n"
        while True:
            yield "The function splits the raw text.\n"

    h = indexed(looping)
    result = h.run("ask", "parse", "--path", str(repo), "--max-new-tokens", "5000")
    assert result.exit_code == 0, result.output
    assert "repeating itself" in result.output and h.generator.closed
    assert "[C1] net/http.py::parse_http_response" in result.output


def test_ask_empty_retrieval_never_loads_model(write_repo: WriteRepo) -> None:
    root = write_repo({"notes.txt": "nothing"})
    h = Harness()
    h.run("index", str(root))
    result = h.run("ask", "anything", "--path", str(root))
    assert result.exit_code == 0
    assert "nothing to answer from" in result.output and h.generator_configs == []


def test_ask_model_load_failure_suggests_search(indexed: Callable[..., Harness], repo: Path) -> None:
    h = indexed()
    h.generator_error = OSError("You are trying to access a gated repo.")
    result = h.run("ask", "parse", "--path", str(repo))
    assert result.exit_code == 1
    assert f"Could not load answer model '{DEFAULT_ANSWER_MODEL}'" in result.output
    assert "gated repo" in result.output and "`chatter search` still works" in result.output
    assert h.run("search", "parse", "--path", str(repo)).exit_code == 0


def test_ask_context_too_small(indexed: Callable[..., Harness], repo: Path) -> None:
    question = "parse " + "very " * 400 + "long question"
    result = indexed().run("ask", question, "--path", str(repo), "--max-context-tokens", "300")
    assert result.exit_code == 1 and "--max-context-tokens" in result.output


def test_help_lists_commands() -> None:
    result = Harness().run("--help")
    assert result.exit_code == 0
    for command in ("index", "search", "ask"):
        assert command in result.output


# ---------------------------------------------------------------------------
# --index-dir / CHATTER_INDEX_DIR
# ---------------------------------------------------------------------------


def test_index_dir_flag_keeps_index_outside_repo(repo: Path, tmp_path: Path) -> None:
    h = Harness()
    external = tmp_path / "idx"
    result = h.run("index", str(repo), "--index-dir", str(external))
    assert result.exit_code == 0, result.output
    assert f"Index: {external}" in result.output
    assert (external / "manifest.json").exists() and not (repo / ".chatter").exists()

    found = h.run("search", "parse_http_response", "--path", str(repo), "--index-dir", str(external))
    assert found.exit_code == 0 and "net/http.py::parse_http_response" in found.output
    asked = h.run("ask", "parse", "--path", str(repo), "--index-dir", str(external))
    assert asked.exit_code == 0 and "Citations:" in asked.output

    missing = h.run("search", "x", "--path", str(repo))  # default location has no index
    assert missing.exit_code == 1 and "No index found" in missing.output


def test_index_dir_env_var_and_flag_precedence(repo: Path, tmp_path: Path) -> None:
    h = Harness()
    env_dir, flag_dir = tmp_path / "from_env", tmp_path / "from_flag"
    env = {"CHATTER_INDEX_DIR": str(env_dir)}
    assert runner.invoke(h.app, ["index", str(repo)], env=env).exit_code == 0
    assert (env_dir / "manifest.json").exists()
    searched = runner.invoke(h.app, ["search", "backoff", "--path", str(repo)], env=env)
    assert searched.exit_code == 0 and "RetryPolicy.backoff_delay" in searched.output

    runner.invoke(h.app, ["index", str(repo), "--index-dir", str(flag_dir)], env=env)
    assert (flag_dir / "manifest.json").exists()  # the flag wins over the env var


def test_missing_external_index_message(repo: Path, tmp_path: Path) -> None:
    result = Harness().run("search", "x", "--path", str(repo), "--index-dir", str(tmp_path / "none"))
    assert result.exit_code == 1
    assert f"--index-dir {tmp_path / 'none'}" in result.output


def test_external_index_dir_cannot_be_shared_between_repos(
    write_repo: WriteRepo, repo: Path, tmp_path: Path
) -> None:
    other = tmp_path / "other_repo"
    other.mkdir()
    (other / "x.py").write_text("def x(): pass\n")
    shared = tmp_path / "shared"
    h = Harness()
    assert h.run("index", str(repo), "--index-dir", str(shared)).exit_code == 0
    clash = h.run("index", str(other), "--index-dir", str(shared))
    assert clash.exit_code == 1
    assert f"belongs to {repo.resolve()}" in clash.output and "--rebuild" in clash.output
    found = h.run("search", "parse_http_response", "--path", str(repo), "--index-dir", str(shared))
    assert "net/http.py::parse_http_response" in found.output  # first repo's index intact

    rebuilt = h.run("index", str(other), "--index-dir", str(shared), "--rebuild")
    assert rebuilt.exit_code == 0 and "Indexed 1 files" in rebuilt.output


def test_default_index_moves_with_its_repo(repo: Path, tmp_path: Path) -> None:
    import shutil

    h = Harness()
    h.run("index", str(repo))
    moved = tmp_path / "moved_repo"
    shutil.copytree(repo, moved)
    result = h.run("index", str(moved))
    assert result.exit_code == 0, result.output
    assert "(0 embedded, 6 unchanged, 0 deleted)" in result.output


def test_ask_backend_option(indexed: Callable[..., Harness], repo: Path) -> None:
    h = indexed("Answer [C1].")
    result = h.run("ask", "parse", "--path", str(repo), "--backend", "mlx")
    assert result.exit_code == 0, result.output
    config = h.generator_configs[0]
    assert (config.backend, config.model_name) == ("mlx", None)
    assert config.resolved_model == "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    bad = h.run("ask", "parse", "--path", str(repo), "--backend", "onnx")
    assert bad.exit_code == 2
