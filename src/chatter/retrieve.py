"""Hybrid retrieval: BM25 over identifier tokens + dense vectors, fused with RRF.

Known gap: retrieval cannot abstain. The dense retriever always returns its
nearest neighbours, so a query with no real answer still yields hits; only the
``sources``/``raw_scores`` fields hint that BM25 found nothing.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi

from chatter.embed import Embedder, EmbedderConfig, SentenceTransformerEmbedder
from chatter.extract import Chunk
from chatter.index import (
    check_manifest,
    chunk_from_dict,
    identifier_words,
    index_data,
    iter_records,
    load_vectors,
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


def bm25_query_tokens(
    query: str, symbol_names: frozenset[str] = frozenset(), *, stem: bool = True
) -> list[str]:
    """BM25 tokens for a natural-language query, with English stopwords removed.

    A word is never dropped if it is inside backticks, looks like an
    identifier (contains ``_`` or a capital after its first letter, as in
    camelCase/PascalCase), or equals an indexed symbol name (case-insensitive).
    Documents are tokenized without stopword removal. ``stem`` adds Snowball
    stems as for document tokens (see ``tokenize_identifier``).
    """
    tokens: list[str] = []
    position = 0
    for match in _BACKTICK_RE.finditer(query):
        tokens += _filtered_tokens(query[position : match.start()], symbol_names, stem)
        tokens += tokenize_code(match.group(1), stem=stem)
        position = match.end()
    tokens += _filtered_tokens(query[position:], symbol_names, stem)
    return tokens


def _filtered_tokens(text: str, symbol_names: frozenset[str], stem: bool) -> list[str]:
    tokens: list[str] = []
    for word in identifier_words(text):
        lower = word.lower()
        if lower in ENGLISH_STOPWORDS and not (
            _looks_like_identifier(word) or lower in symbol_names
        ):
            continue
        tokens += tokenize_identifier(word, stem=stem)
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


def id_ranks(ids: Sequence[str]) -> np.ndarray:
    """Position of each id in sorted order: the tie-break key (lower wins)."""
    ranks = np.empty(len(ids), dtype=np.int64)
    ranks[np.argsort(np.asarray(ids, dtype=object), kind="stable")] = np.arange(len(ids))
    return ranks


def top_k(
    scores: np.ndarray, k: int, tiebreak: np.ndarray, *, positive_only: bool = False
) -> np.ndarray:
    """Indices of the ``k`` highest scores, best first; equal scores by ``tiebreak``.

    Exact and deterministic: argpartition finds the k-th best score, every
    candidate tied with it is kept, and a stable lexicographic sort on
    (-score, tiebreak) picks the final order.
    """
    candidates = np.flatnonzero(scores > 0) if positive_only else np.arange(len(scores))
    if k <= 0 or len(candidates) == 0:
        return np.empty(0, dtype=np.int64)
    values = scores[candidates]
    if k < len(values):
        kth = values[np.argpartition(-values, k - 1)[k - 1]]
        keep = values >= kth
        candidates, values = candidates[keep], values[keep]
    order = np.lexsort((tiebreak[candidates], -values))
    return candidates[order][:k]


class BM25Index:
    def __init__(self, ids: Sequence[str], token_lists: Sequence[Sequence[str]]) -> None:
        self._ids = list(ids)
        self._ranks = id_ranks(self._ids)
        has_tokens = any(token_lists)
        self._bm25 = PositiveIdfBM25([list(t) for t in token_lists]) if has_tokens else None

    def search(self, query_tokens: Sequence[str], n: int) -> list[tuple[str, float]]:
        """Top ``n`` (id, score) with score > 0, best first; ties by chunk id."""
        if self._bm25 is None or not query_tokens or n <= 0:
            return []
        scores = self._bm25.get_scores(list(query_tokens))
        order = top_k(scores, n, self._ranks, positive_only=True)
        return [(self._ids[i], float(scores[i])) for i in order]


class DenseIndex:
    """Exact cosine search over L2-normalized part vectors, ranked per chunk.

    A chunk's score is its best part's cosine similarity. Rows of one chunk
    are contiguous, starting at ``row_starts[i]``.
    """

    def __init__(
        self,
        ids: Sequence[str],
        vectors: np.ndarray,
        row_starts: Sequence[int],
        row_counts: Sequence[int],
    ) -> None:
        self._ids = list(ids)
        self._ranks = id_ranks(self._ids)
        self._vectors = vectors
        self._starts = np.asarray(row_starts, dtype=np.int64)
        self._counts = np.asarray(row_counts, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._ids)

    def search(self, query: Sequence[float], n: int) -> list[tuple[str, float, int]]:
        """Top ``n`` (chunk id, cosine similarity, best part index), best first."""
        if n <= 0 or not self._ids or self._vectors.shape[0] == 0:
            return []
        q = np.asarray(query, dtype=np.float32)
        norm = float(np.linalg.norm(q))
        if norm > 0:
            q = q / norm
        part_scores = np.asarray(self._vectors @ q, dtype=np.float32)
        chunk_scores = np.maximum.reduceat(part_scores, self._starts)
        hits = []
        for i in top_k(chunk_scores, n, self._ranks):
            start, count = self._starts[i], self._counts[i]
            best_part = int(np.argmax(part_scores[start : start + count]))
            hits.append((self._ids[i], float(chunk_scores[i]), best_part))
        return hits


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    k: int = RRF_K,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[str, float, dict[str, int]]]:
    """Fuse ranked id lists: score = sum over retrievers of weight / (k + rank).

    ``weights`` defaults to 1 for every retriever. Ties are broken by best
    single rank, then id, so output is deterministic.
    """
    scores: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for source, ids in rankings.items():
        weight = 1.0 if weights is None else weights.get(source, 1.0)
        for rank, item in enumerate(ids, start=1):
            if source in ranks[item]:
                continue  # duplicate within one ranking: keep the best rank
            scores[item] += weight / (k + rank)
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
        parts: Mapping[str, Sequence[Sequence[int]]],
        bm25: BM25Index,
        dense: DenseIndex,
        embedder: Embedder,
    ) -> None:
        self._chunks = dict(chunks)
        self._parts = dict(parts)
        self._symbol_names = frozenset(
            c.name.lower() for c in self._chunks.values() if c.kind != "module"
        )
        self._bm25 = bm25
        self._dense = dense
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
        data = index_data(index_path, manifest)
        if data is None:
            raise FileNotFoundError(f"index at {index_path} has no committed build")

        chunks: dict[str, Chunk] = {}
        parts: dict[str, list[list[int]]] = {}
        ids: list[str] = []
        token_lists: list[list[str]] = []
        starts: list[int] = []
        counts: list[int] = []
        for record in iter_records(data.chunks_path):
            chunk_id = record["id"]
            chunks[chunk_id] = chunk_from_dict(record["chunk"])
            parts[chunk_id] = record["parts"]
            ids.append(chunk_id)
            token_lists.append(record["tokens"])
            starts.append(record["rows"][0])
            counts.append(record["rows"][1])
        vectors = load_vectors(data.vectors_path)
        if vectors.shape[0] != sum(counts) or vectors.shape[0] != manifest.get("rows"):
            raise FileNotFoundError(
                f"index at {index_path} is inconsistent ({vectors.shape[0]} vectors for "
                f"{sum(counts)} parts); rebuild it"
            )
        return cls(
            chunks,
            parts,
            BM25Index(ids, token_lists),
            DenseIndex(ids, vectors, starts, counts),
            embedder,
        )

    def __len__(self) -> int:
        return len(self._chunks)

    def chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def search(
        self,
        query: str,
        k: int = 10,
        *,
        candidates: int | None = None,
        mode: Mode = "fused",
        rrf_k: int = RRF_K,
        weights: Mapping[str, float] | None = None,
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
            {BM25: [cid for cid, _ in bm25_hits], DENSE: [cid for cid, _, _ in dense_hits]},
            k=rrf_k,
            weights=weights,
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
        if len(self._dense) == 0 or n <= 0:
            return []
        return [
            (chunk_id, similarity, _part_lines(self._chunks[chunk_id], self._parts[chunk_id], part))
            for chunk_id, similarity, part in self._dense.search(self._embedder.embed_query(query), n)
        ]


def _part_lines(chunk: Chunk, parts: Sequence[Sequence[int]], part: int) -> tuple[int, int]:
    """File line range of one embedding part (the whole chunk if unsplit)."""
    if len(parts) <= 1:
        return (chunk.start_line, chunk.end_line)
    numbers = chunk.line_numbers()
    first, last = parts[part]
    return (numbers[min(first, len(numbers) - 1)], numbers[min(last, len(numbers) - 1)])
