from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chatter.embed import (
    BGE_QUERY_PREFIX,
    DEFAULT_MODEL,
    EmbedderConfig,
    SentenceTransformerEmbedder,
    default_query_prefix,
)


class FakeTokenizer:
    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2

    def __call__(self, texts: Sequence[str], **kwargs: Any) -> dict[str, list[list[int]]]:
        assert kwargs["add_special_tokens"] is False
        return {"input_ids": [list(range(len(t.split()))) for t in texts]}


class FakeModel:
    max_seq_length = 512

    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.encoded: list[list[str]] = []

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        assert kwargs["normalize_embeddings"] is True
        self.encoded.append(list(texts))
        return np.ones((len(texts), 4), dtype=np.float32) / 2


def make(config: EmbedderConfig = EmbedderConfig()) -> tuple[SentenceTransformerEmbedder, FakeModel]:
    model = FakeModel()
    return SentenceTransformerEmbedder(config, model=model), model


def test_bge_query_prefix_applies_to_queries_only() -> None:
    embedder, model = make()
    assert embedder.name == DEFAULT_MODEL == "BAAI/bge-small-en-v1.5"
    embedder.embed_documents(["def f(): pass", "class A: ..."])
    embedder.embed_query("where is f defined")
    assert model.encoded == [
        ["def f(): pass", "class A: ..."],
        [BGE_QUERY_PREFIX + "where is f defined"],
    ]


def test_token_counting_ignores_query_prefix() -> None:
    embedder, _ = make()
    assert embedder.count_tokens(["a b c", ""]) == [3, 0]
    assert embedder.max_tokens == 510  # 512 minus [CLS]/[SEP]


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("BAAI/bge-small-en-v1.5", BGE_QUERY_PREFIX),
        ("BAAI/bge-base-en", BGE_QUERY_PREFIX),
        ("BAAI/bge-m3", ""),
        ("sentence-transformers/all-MiniLM-L6-v2", ""),
        ("jinaai/jina-embeddings-v2-base-code", ""),
    ],
)
def test_default_query_prefix(model_name: str, expected: str) -> None:
    assert default_query_prefix(model_name) == expected


def test_explicit_prefixes_override_defaults() -> None:
    embedder, model = make(
        EmbedderConfig(query_prefix="query: ", document_prefix="passage: ")
    )
    embedder.embed_documents(["x"])
    embedder.embed_query("y")
    assert model.encoded == [["passage: x"], ["query: y"]]
    assert embedder.count_tokens(["x"]) == [2]  # document prefix counts toward budget
    assert '"passage: "' in embedder.fingerprint


def test_fingerprint_ignores_query_prefix() -> None:
    a, _ = make(EmbedderConfig(query_prefix="q1: "))
    b, _ = make(EmbedderConfig(query_prefix="q2: "))
    assert a.fingerprint == b.fingerprint


def test_embed_empty_batch_skips_model() -> None:
    embedder, model = make()
    assert embedder.embed_documents([]) == [] and embedder.count_tokens([]) == []
    assert model.encoded == []


def test_trust_remote_code_off_by_default() -> None:
    assert EmbedderConfig().trust_remote_code is False


@pytest.mark.slow
def test_real_model_smoke(tmp_path: Path) -> None:
    """End to end with the real default model (downloads it on first run)."""
    from chatter.index import build_index
    from chatter.retrieve import Retriever

    fixtures = Path(__file__).parent / "fixtures"
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text((fixtures / "sample.py").read_text())

    embedder = SentenceTransformerEmbedder()
    assert embedder.max_tokens == 510
    stats = build_index(repo, embedder)
    assert stats.chunks > 10

    vectors = embedder.embed_documents(["def add(a, b): return a + b"])
    assert len(vectors[0]) == 384
    assert abs(sum(x * x for x in vectors[0]) - 1.0) < 1e-3

    hits = Retriever.open(repo / ".chatter", embedder).search("function that adds two numbers", k=3)
    assert "sample.py::plain" in [h.chunk_id for h in hits]
