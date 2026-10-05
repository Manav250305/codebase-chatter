"""Hybrid retrieval: BM25 over identifier tokens + dense vectors, fused with RRF.

Known gap: retrieval cannot abstain. The dense retriever always returns its
nearest neighbours, so a query with no real answer still yields hits; only the
``sources``/``raw_scores`` fields hint that BM25 found nothing.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from rank_bm25 import BM25Okapi

from chatter.embed import Embedder, EmbedderConfig, SentenceTransformerEmbedder
from chatter.extract import Chunk, format_line_ranges
from chatter.graph import GraphIndex
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
GRAPH = "graph"

# Graph expansion constants (fixed; only hops and weight are tuned).
GRAPH_SEEDS = 5  # top fused chunks whose neighbours are considered
GRAPH_PER_SEED = 8  # neighbours kept per seed, by weight x relevance
GRAPH_BUDGET = 3  # chunks from outside the base top window allowed into it
GRAPH_PROTECTED = 3  # base ranks 1..3 are never changed by expansion
GRAPH_WINDOW = 10  # the "top k" the budget protects
GRAPH_HOP_DECAY = 0.5  # contribution carried to the next hop
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
        return self.search_with_scores(query, n)[0]

    def search_with_scores(
        self, query: Sequence[float], n: int
    ) -> tuple[list[tuple[str, float, int]], np.ndarray]:
        """``search`` plus every chunk's cosine similarity (index order)."""
        if not self._ids or self._vectors.shape[0] == 0:
            return [], np.zeros(len(self._ids), dtype=np.float32)
        q = np.asarray(query, dtype=np.float32)
        norm = float(np.linalg.norm(q))
        if norm > 0:
            q = q / norm
        part_scores = np.asarray(self._vectors @ q, dtype=np.float32)
        chunk_scores = np.maximum.reduceat(part_scores, self._starts)
        if n <= 0:
            return [], chunk_scores
        hits = []
        for i in top_k(chunk_scores, n, self._ranks):
            start, count = self._starts[i], self._counts[i]
            best_part = int(np.argmax(part_scores[start : start + count]))
            hits.append((self._ids[i], float(chunk_scores[i]), best_part))
        return hits, chunk_scores


@dataclass(frozen=True, slots=True)
class GraphExpansion:
    """Graph-expanded retrieval: neighbours of the top fused chunks, re-fused.

    The graph list scores a neighbour by the sum over seeds that reach it of
    ``edge weight / log(2 + in-degree) x relevance x 1/(rrf_k + seed rank)``,
    where relevance is the neighbour's dense cosine over the query's best.
    It is fused with BM25 and dense using ``weight``. Ranks 1-3 stay those of
    the base fusion, and at most ``GRAPH_BUDGET`` chunks from outside the base
    top ``GRAPH_WINDOW`` may enter it.
    """

    hops: int = 1
    weight: float = 1.0


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


def hit_location(hit: Hit) -> str:
    """``path:a-b`` for a hit (all line ranges for an unsplit module chunk)."""
    chunk = hit.chunk
    if chunk.spans and hit.lines == (chunk.start_line, chunk.end_line):
        return f"{chunk.path}:{format_line_ranges(chunk.line_numbers())}"
    return f"{chunk.path}:{hit.lines[0]}-{hit.lines[1]}"


def hit_to_dict(rank: int, hit: Hit) -> dict[str, Any]:
    """JSON-ready hit, shared by ``chatter search --json`` and the MCP server."""
    chunk = hit.chunk
    return {
        "rank": rank,
        "chunk_id": hit.chunk_id,
        "path": chunk.path,
        "qualname": chunk.qualname,
        "kind": chunk.kind,
        "lines": list(hit.lines),
        "spans": [list(span) for span in (chunk.spans or ((chunk.start_line, chunk.end_line),))],
        "score": hit.score,
        "sources": list(hit.sources),
        "ranks": hit.ranks,
        "raw_scores": hit.raw_scores,
    }


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
        graph: GraphIndex | None = None,
    ) -> None:
        self._chunks = dict(chunks)
        self._parts = dict(parts)
        self._ids = list(self._chunks)  # index order, as in the dense and graph arrays
        self._index_of = {chunk_id: i for i, chunk_id in enumerate(self._ids)}
        self._id_ranks = id_ranks(self._ids)
        self._graph = graph
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
            GraphIndex.load(data.directory),
        )

    @property
    def has_graph(self) -> bool:
        return self._graph is not None

    def __len__(self) -> int:
        return len(self._chunks)

    def chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def chunks(self) -> Iterator[tuple[str, Chunk]]:
        """(chunk id, chunk) for every indexed chunk, in index order."""
        return iter(self._chunks.items())

    def search(
        self,
        query: str,
        k: int = 10,
        *,
        candidates: int | None = None,
        mode: Mode = "fused",
        rrf_k: int = RRF_K,
        weights: Mapping[str, float] | None = None,
        expansion: GraphExpansion | None = None,
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
        dense_hits, chunk_scores = self._dense_search(query, n) if mode != BM25 else ([], None)
        rankings = {BM25: [cid for cid, _ in bm25_hits], DENSE: [cid for cid, _, _ in dense_hits]}
        fused = reciprocal_rank_fusion(rankings, k=rrf_k, weights=weights)
        if expansion is not None and mode == FUSED and self._graph is not None and fused:
            if expansion.hops < 1 or expansion.weight < 0:
                raise ValueError(f"invalid graph expansion {expansion}")
            assert chunk_scores is not None
            base = [cid for cid, _, _ in fused]
            rankings[GRAPH] = self._graph_ranking(base, chunk_scores, expansion, rrf_k)
            expanded = reciprocal_rank_fusion(
                rankings, k=rrf_k, weights={**(weights or {}), GRAPH: expansion.weight}
            )
            by_id = {cid: (score, ranks) for cid, score, ranks in expanded}
            fused = [(cid, *by_id[cid]) for cid in _apply_budget(base, [cid for cid, _, _ in expanded])]

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
                    sources=tuple(s for s in (BM25, DENSE, GRAPH) if s in ranks),
                    ranks=ranks,
                    raw_scores=raw,
                    lines=lines,
                )
            )
        return hits

    def _dense_search(
        self, query: str, n: int
    ) -> tuple[list[tuple[str, float, tuple[int, int]]], np.ndarray]:
        """Best part per chunk (chunk id, cosine, part lines), and every chunk's cosine."""
        if len(self._dense) == 0:
            return [], np.zeros(len(self._ids), dtype=np.float32)
        hits, scores = self._dense.search_with_scores(self._embedder.embed_query(query), max(n, 0))
        return [
            (chunk_id, similarity, _part_lines(self._chunks[chunk_id], self._parts[chunk_id], part))
            for chunk_id, similarity, part in hits
        ], scores

    def _graph_ranking(
        self, base: Sequence[str], cosine: np.ndarray, expansion: GraphExpansion, rrf_k: int
    ) -> list[str]:
        """Neighbours of the top base chunks, best first (see ``GraphExpansion``)."""
        assert self._graph is not None
        best = float(cosine.max()) if len(cosine) else 0.0
        relevance = np.clip(cosine / best, 0.0, 1.0) if best > 0 else np.zeros_like(cosine)
        seeds = [self._index_of[cid] for cid in base[:GRAPH_SEEDS]]
        order = graph_ranking(
            self._graph, seeds, relevance, self._id_ranks, hops=expansion.hops, rrf_k=rrf_k
        )
        return [self._ids[i] for i in order]


def graph_ranking(
    graph: GraphIndex,
    seeds: Sequence[int],
    relevance: np.ndarray,
    tiebreak: np.ndarray,
    *,
    hops: int,
    rrf_k: int,
) -> list[int]:
    """Chunk indices reached from ``seeds`` (best seed first), best first.

    A seed at rank r contributes 1/(rrf_k + r). Each neighbour of a frontier
    node scores ``edge weight / log(2 + in-degree) x relevance``; each node
    keeps its GRAPH_PER_SEED best neighbours, which add ``contribution x score``
    and carry ``GRAPH_HOP_DECAY`` of it to the next hop. Ties: ``tiebreak``.
    """
    in_degree = graph.in_degree
    frontier = [(node, 1.0 / (rrf_k + rank)) for rank, node in enumerate(seeds, start=1)]
    scores: dict[int, float] = defaultdict(float)
    for _ in range(hops):
        reached: dict[int, float] = {}
        for node, contribution in frontier:
            candidates = [
                (neighbor, weight / math.log(2 + int(in_degree[neighbor])) * float(relevance[neighbor]))
                for neighbor, weight in graph.neighbors(node).items()
            ]
            candidates = sorted(
                (c for c in candidates if c[1] > 0), key=lambda c: (-c[1], int(tiebreak[c[0]]))
            )[:GRAPH_PER_SEED]
            for neighbor, strength in candidates:
                value = contribution * strength
                scores[neighbor] += value
                reached[neighbor] = max(reached.get(neighbor, 0.0), value * GRAPH_HOP_DECAY)
        frontier = sorted(reached.items())
    return sorted(scores, key=lambda i: (-scores[i], int(tiebreak[i])))


def _apply_budget(base: Sequence[str], expanded: Sequence[str]) -> list[str]:
    """Expanded order, except: base ranks 1..GRAPH_PROTECTED are kept as they are,
    and at most GRAPH_BUDGET chunks from outside the base top GRAPH_WINDOW enter it."""
    result = list(base[:GRAPH_PROTECTED])
    taken = set(result)
    window = set(base[:GRAPH_WINDOW])
    promoted = 0
    for chunk_id in expanded:
        if len(result) >= GRAPH_WINDOW:
            break
        if chunk_id in taken:
            continue
        if chunk_id not in window:
            if promoted >= GRAPH_BUDGET:
                continue  # over budget: it may still appear below the window
            promoted += 1
        result.append(chunk_id)
        taken.add(chunk_id)
    result += [chunk_id for chunk_id in expanded if chunk_id not in taken]
    return result


def _part_lines(chunk: Chunk, parts: Sequence[Sequence[int]], part: int) -> tuple[int, int]:
    """File line range of one embedding part (the whole chunk if unsplit)."""
    if len(parts) <= 1:
        return (chunk.start_line, chunk.end_line)
    numbers = chunk.line_numbers()
    first, last = parts[part]
    return (numbers[min(first, len(numbers) - 1)], numbers[min(last, len(numbers) - 1)])
