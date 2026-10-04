"""Embedding models behind a small protocol so indexing and tests stay decoupled.

Token counts here exclude special tokens ([CLS]/[SEP]); ``max_tokens`` is the
model's sequence limit minus those, i.e. the budget for real text.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def max_tokens(self) -> int: ...

    @property
    def fingerprint(self) -> str:
        """Identifies everything that affects document vectors (not queries)."""
        ...

    def count_tokens(self, texts: Sequence[str]) -> list[int]: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class EmbedderConfig:
    model_name: str = DEFAULT_MODEL
    query_prefix: str | None = None  # None: pick a default for the model family
    document_prefix: str = ""
    device: str | None = None  # None: let sentence-transformers choose
    batch_size: int = 64
    trust_remote_code: bool = False


def default_query_prefix(model_name: str) -> str:
    """BGE v1/v1.5 English models expect an instruction prefix on queries only."""
    name = model_name.lower()
    if "bge-" in name and "-en" in name and "m3" not in name:
        return BGE_QUERY_PREFIX
    return ""


class SentenceTransformerEmbedder:
    """``Embedder`` backed by sentence-transformers. ``model`` is injectable for tests."""

    def __init__(self, config: EmbedderConfig = EmbedderConfig(), *, model: Any = None) -> None:
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                config.model_name,
                device=config.device,
                trust_remote_code=config.trust_remote_code,
            )
        self._config = config
        self._model = model
        self._query_prefix = (
            config.query_prefix
            if config.query_prefix is not None
            else default_query_prefix(config.model_name)
        )
        special = model.tokenizer.num_special_tokens_to_add(pair=False)
        self._max_tokens = int(model.max_seq_length) - special

    @property
    def name(self) -> str:
        return self._config.model_name

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def fingerprint(self) -> str:
        doc_prefix = json.dumps(self._config.document_prefix)
        return f"{self.name}|max_tokens={self._max_tokens}|document_prefix={doc_prefix}"

    @property
    def query_prefix(self) -> str:
        return self._query_prefix

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        if not texts:
            return []
        prefix = self._config.document_prefix
        encoded = self._model.tokenizer(
            [prefix + text for text in texts],
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )
        return [len(ids) for ids in encoded["input_ids"]]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        prefix = self._config.document_prefix
        return self._encode([prefix + text for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self._query_prefix + text])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self._config.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()
