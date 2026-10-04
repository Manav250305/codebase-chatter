"""Hybrid retrieval: BM25 over identifier tokens + dense vectors, fused with RRF.

Known gap: retrieval cannot abstain. The dense retriever always returns its
nearest neighbours, so a query with no real answer still yields hits; only the
``sources``/``raw_scores`` fields hint that BM25 found nothing.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from chromadb.api.models.Collection import Collection
from rank_bm25 import BM25Okapi

from chatter.embed import Embedder, EmbedderConfig, SentenceTransformerEmbedder
from chatter.extract import Chunk
from chatter.index import (
    CHUNKS_FILE,
    check_manifest,
    chunk_from_dict,
    identifier_words,
    open_collection,
    read_manifest,
    tokenize_code,
    tokenize_identifier,
)

RRF_K = 60
BM25 = "bm25"
DENSE = "dense"
FUSED = "fused"
Mode = Literal["fused", "bm25", "dense"]
MODES: tuple[Mode, ...] = ("bm25", "dense", "fused")


@dataclass(frozen=True, slots=True)
class Hit:
    chunk_id: str
    chunk: Chunk
    score: float  # fused RRF score
    sources: tuple[str, ...]  # retrievers that surfaced this chunk, e.g. ("bm25", "dense")
    ranks: dict[str, int]  # 1-based rank per retriever
    raw_scores: dict[str, float]  # BM25 score / cosine similarity per retriever
    lines: tuple[int, int]  # best citation range (the matching part for split chunks)


# NLTK's English stopword list (forms with apostrophes omitted: the tokenizer
# never produces them; "don't" becomes "don" + "t", both listed).
ENGLISH_STOPWORDS = frozenset(
    """
    i me my myself we our ours ourselves you your yours yourself yourselves he him
    his himself she her hers herself it its itself they them their theirs
    themselves what which who whom this that these those am is are was were be
    been being have has had having do does did doing a an the and but if or
    because as until while of at by for with about against between into through
    during before after above below to from up down in out on off over under
    again further then once here there when where why how all any both each few
    more most other some such no nor not only own same so than too very s t can
    will just don should now d ll m o re ve y ain aren couldn didn doesn hadn
    hasn haven isn ma mightn mustn needn shan shouldn wasn weren won wouldn
    """.split()
)
_BACKTICK_RE = re.compile(r"`([^`]*)`")


def bm25_query_tokens(query: str, symbol_names: frozenset[str] = frozenset()) -> list[str]:
    """BM25 tokens for a natural-language query, with English stopwords removed.

    A word is never dropped if it is inside backticks, looks like an
    identifier (contains ``_`` or a capital after its first letter, as in
    camelCase/PascalCase), or equals an indexed symbol name (case-insensitive).
    Documents are tokenized without stopword removal.
    """
    tokens: list[str] = []
    position = 0
    for match in _BACKTICK_RE.finditer(query):
        tokens += _filtered_tokens(query[position : match.start()], symbol_names)
        tokens += tokenize_code(match.group(1))
        position = match.end()
    tokens += _filtered_tokens(query[position:], symbol_names)
    return tokens


def _filtered_tokens(text: str, symbol_names: frozenset[str]) -> list[str]:
    tokens: list[str] = []
    for word in identifier_words(text):
        lower = word.lower()
        if lower in ENGLISH_STOPWORDS and not (
            _looks_like_identifier(word) or lower in symbol_names
        ):
            continue
        tokens += tokenize_identifier(word)
    return tokens


def _looks_like_identifier(word: str) -> bool:
    inner_capital = any(c.isupper() for c in word[1:])
    return "_" in word or (inner_capital and any(c.islower() for c in word))


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


class PositiveIdfBM25(BM25Okapi):
    """BM25Okapi with Lucene's idf, ``log(1 + (N - n + 0.5) / (n + 0.5))``.

    Okapi's idf is <= 0 for terms in at least half the documents (always the
    case in a one-file repo), so real matches could score zero or negative.
    This idf is strictly positive: matches score > 0, non-matches exactly 0.
    """

    def _calc_idf(self, nd: Mapping[str, int]) -> None:
        for word, freq in nd.items():
            self.idf[word] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))


class BM25Index:
    def __init__(self, ids: Sequence[str], token_lists: Sequence[Sequence[str]]) -> None:
        self._ids = list(ids)
        has_tokens = any(token_lists)
        self._bm25 = PositiveIdfBM25([list(t) for t in token_lists]) if has_tokens else None

    def search(self, query_tokens: Sequence[str], n: int) -> list[tuple[str, float]]:
        """Top ``n`` (id, score) with score > 0, best first."""
        if self._bm25 is None or not query_tokens or n <= 0:
            return []
        scores = self._bm25.get_scores(list(query_tokens))
        matched = np.flatnonzero(scores > 0)
        order = matched[np.argsort(-scores[matched], kind="stable")][:n]
        return [(self._ids[i], float(scores[i])) for i in order]


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]], k: int = RRF_K
) -> list[tuple[str, float, dict[str, int]]]:
    """Fuse ranked id lists: score = sum over retrievers of 1 / (k + rank).

    Ties are broken by best single rank, then id, so output is deterministic.
    """
    scores: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for source, ids in rankings.items():
        for rank, item in enumerate(ids, start=1):
            if source in ranks[item]:
                continue  # duplicate within one ranking: keep the best rank
            scores[item] += 1.0 / (k + rank)
            ranks[item][source] = rank
    return sorted(
        ((item, score, ranks[item]) for item, score in scores.items()),
        key=lambda entry: (-entry[1], min(entry[2].values()), entry[0]),
    )


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class Retriever:
    def __init__(
        self,
        chunks: Mapping[str, Chunk],
        bm25: BM25Index,
        collection: Collection,
        embedder: Embedder,
    ) -> None:
        self._chunks = dict(chunks)
        self._symbol_names = frozenset(
            c.name.lower() for c in self._chunks.values() if c.kind != "module"
        )
        self._bm25 = bm25
        self._collection = collection
        self._embedder = embedder

    @classmethod
    def open(cls, index_dir: Path | str, embedder: Embedder | None = None) -> Retriever:
        index_path = Path(index_dir)
        manifest = read_manifest(index_path)
        if manifest is None:
            raise FileNotFoundError(f"no index found at {index_path}; build it first")
        if embedder is None:
            embedder = SentenceTransformerEmbedder(EmbedderConfig(model_name=manifest["model"]))
        check_manifest(manifest, embedder)

        chunks: dict[str, Chunk] = {}
        ids: list[str] = []
        token_lists: list[list[str]] = []
        with (index_path / CHUNKS_FILE).open(encoding="utf-8") as records:
            for line in records:
                record = json.loads(line)
                chunks[record["id"]] = chunk_from_dict(record["chunk"])
                ids.append(record["id"])
                token_lists.append(record["tokens"])
        return cls(
            chunks,
            BM25Index(ids, token_lists),
            open_collection(index_path, create=False),
            embedder,
        )

    def __len__(self) -> int:
        return len(self._chunks)

    def chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def search(
        self, query: str, k: int = 10, *, candidates: int | None = None, mode: Mode = "fused"
    ) -> list[Hit]:
        """Top ``k`` chunks for ``query`` with fused scores and provenance.

        ``mode`` restricts retrieval to one retriever ("bm25" or "dense"); its
        ranking is then passed through RRF unchanged (scores are 1/(k + rank)).
        """
        if mode not in MODES:
            raise ValueError(f"unknown retrieval mode {mode!r}; expected one of {MODES}")
        if k <= 0 or not query.strip() or not self._chunks:
            return []
        n = candidates if candidates is not None else max(4 * k, 50)

        bm25_hits = (
            self._bm25.search(bm25_query_tokens(query, self._symbol_names), n)
            if mode != DENSE
            else []
        )
        dense_hits = self._dense_search(query, n) if mode != BM25 else []
        fused = reciprocal_rank_fusion(
            {BM25: [cid for cid, _ in bm25_hits], DENSE: [cid for cid, _, _ in dense_hits]}
        )

        bm25_scores = dict(bm25_hits)
        dense_info = {cid: (similarity, lines) for cid, similarity, lines in dense_hits}
        hits: list[Hit] = []
        for chunk_id, score, ranks in fused[:k]:
            chunk = self._chunks[chunk_id]
            raw: dict[str, float] = {}
            if chunk_id in bm25_scores:
                raw[BM25] = bm25_scores[chunk_id]
            lines = (chunk.start_line, chunk.end_line)
            if chunk_id in dense_info:
                raw[DENSE], lines = dense_info[chunk_id]
            hits.append(
                Hit(
                    chunk_id=chunk_id,
                    chunk=chunk,
                    score=score,
                    sources=tuple(s for s in (BM25, DENSE) if s in ranks),
                    ranks=ranks,
                    raw_scores=raw,
                    lines=lines,
                )
            )
        return hits

    def _dense_search(self, query: str, n: int) -> list[tuple[str, float, tuple[int, int]]]:
        """Best part per chunk: (chunk id, cosine similarity, part line range)."""
        count = self._collection.count()
        if count == 0:
            return []
        result = self._collection.query(
            query_embeddings=[self._embedder.embed_query(query)],
            n_results=min(count, 2 * n),  # split chunks may occupy several slots
            include=["metadatas", "distances"],
        )
        metadatas = (result["metadatas"] or [[]])[0]
        distances = (result["distances"] or [[]])[0]
        hits: list[tuple[str, float, tuple[int, int]]] = []
        seen: set[str] = set()
        for meta, distance in zip(metadatas, distances):
            chunk_id = str(meta["chunk_id"])
            chunk = self._chunks.get(chunk_id)
            if chunk is None or chunk_id in seen:
                continue
            seen.add(chunk_id)
            hits.append((chunk_id, 1.0 - float(distance), _part_lines(chunk, meta)))
            if len(hits) == n:
                break
        return hits


def _part_lines(chunk: Chunk, meta: Mapping[str, object]) -> tuple[int, int]:
    if meta.get("parts", 1) == 1:
        return (chunk.start_line, chunk.end_line)
    numbers = chunk.line_numbers()
    first = min(int(meta["first"]), len(numbers) - 1)  # type: ignore[arg-type]
    last = min(int(meta["last"]), len(numbers) - 1)  # type: ignore[arg-type]
    return (numbers[first], numbers[last])
