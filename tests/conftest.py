from __future__ import annotations

import math
import textwrap
import zlib
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from chatter.index import tokenize_code


class HashEmbedder:
    """Deterministic offline embedder: hashed bag of identifier tokens.

    Tokens are whitespace-separated words, so token budgets are easy to reason
    about in tests. Records every call for idempotency assertions.
    """

    def __init__(self, *, max_tokens: int = 512, dim: int = 64, name: str = "fake-hash") -> None:
        self._max_tokens = max_tokens
        self._dim = dim
        self._name = name
        self.document_batches: list[list[str]] = []
        self.queries: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def fingerprint(self) -> str:
        return f"{self._name}|{self._max_tokens}|{self._dim}"

    @property
    def embedded_texts(self) -> list[str]:
        return [text for batch in self.document_batches for text in batch]

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        return [len(text.split()) for text in texts]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_batches.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in tokenize_code(text):
            vec[zlib.crc32(token.encode()) % self._dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            vec[0], norm = 1.0, 1.0
        return [x / norm for x in vec]


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder()


WriteRepo = Callable[[dict[str, str]], Path]


@pytest.fixture
def write_repo(tmp_path: Path) -> WriteRepo:
    """Create files (dedented) under ``tmp_path / "repo"`` and return the repo root."""

    def write(files: dict[str, str]) -> Path:
        root = tmp_path / "repo"
        root.mkdir(exist_ok=True)
        for rel, content in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")
        return root

    return write


class FakeGenerator:
    """Offline ``Generator``: whitespace tokens, scripted streamed replies.

    ``reply`` is a string, or a callable (system, user) -> iterable of pieces,
    which allows infinite (looping) streams. Records prompts and whether the
    stream was closed.
    """

    def __init__(self, reply: object = "An answer [C1].", *, name: str = "fake-llm") -> None:
        self._reply = reply
        self._name = name
        self.prompts: list[tuple[str, str]] = []
        self.closed = False
        self.pieces_sent = 0

    @property
    def name(self) -> str:
        return self._name

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        return [len(text.split()) for text in texts]

    def prompt_tokens(self, system: str, user: str) -> int:
        return len(system.split()) + len(user.split()) + 10  # + chat template

    def generate(self, system: str, user: str, *, max_new_tokens: int):  # type: ignore[no-untyped-def]
        self.prompts.append((system, user))
        pieces = (
            self._reply(system, user)
            if callable(self._reply)
            else [w + " " for w in str(self._reply).split(" ")]
        )
        try:
            for i, piece in enumerate(pieces):
                if i >= max_new_tokens:
                    return
                self.pieces_sent += 1
                yield piece
        finally:
            self.closed = True
