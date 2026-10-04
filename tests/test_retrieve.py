from __future__ import annotations

from pathlib import Path

import pytest
from rank_bm25 import BM25Okapi

from chatter.index import build_index
from chatter.retrieve import BM25Index, PositiveIdfBM25, Retriever, reciprocal_rank_fusion
from conftest import HashEmbedder, WriteRepo


def open_retriever(root: Path, embedder: HashEmbedder) -> Retriever:
    build_index(root, embedder)
    return Retriever.open(root / ".chatter", embedder)


# ---------------------------------------------------------------------------
# BM25 on tiny corpora
# ---------------------------------------------------------------------------


TINY_CORPORA = [
    [["parse", "json"]],  # one document: every term is in 100% of docs
    [["parse", "json"], ["parse", "xml"]],  # term in 100%
    [["parse", "json"], ["load", "xml"]],  # term in exactly 50%
    [["parse", "a"], ["parse", "b"], ["c"], ["d"]],  # term in 50%
]


@pytest.mark.parametrize("corpus", TINY_CORPORA)
def test_okapi_scores_real_matches_at_or_below_zero_on_tiny_corpora(
    corpus: list[list[str]],
) -> None:
    # Documents the upstream behaviour this module works around.
    scores = BM25Okapi(corpus).get_scores(["parse"])
    matching = [s for s, doc in zip(scores, corpus) if "parse" in doc]
    assert all(s <= 0 for s in matching)


@pytest.mark.parametrize("corpus", TINY_CORPORA)
def test_positive_idf_bm25_scores_matches_above_zero_and_others_zero(
    corpus: list[list[str]],
) -> None:
    scores = PositiveIdfBM25(corpus).get_scores(["parse"])
    for score, doc in zip(scores, corpus):
        assert (score > 0) if "parse" in doc else (score == 0)


def test_bm25_index_drops_non_matches_and_orders_by_score() -> None:
    index = BM25Index(
        ["a", "b", "c"],
        [["parse", "parse", "json"], ["parse", "xml", "xml", "xml"], ["other"]],
    )
    assert [cid for cid, _ in index.search(["parse"], 10)] == ["a", "b"]
    assert index.search(["missing"], 10) == []
    assert index.search([], 10) == []
    assert [cid for cid, _ in index.search(["parse"], 1)] == ["a"]


def test_bm25_index_empty_corpora() -> None:
    assert BM25Index([], []).search(["x"], 5) == []
    assert BM25Index(["a"], [[]]).search(["x"], 5) == []


# ---------------------------------------------------------------------------
# Reciprocal rank fusion
# ---------------------------------------------------------------------------


def test_rrf_scores_and_provenance() -> None:
    fused = reciprocal_rank_fusion({"bm25": ["a", "b"], "dense": ["b", "c"]}, k=60)
    by_id = {item: (score, ranks) for item, score, ranks in fused}
    assert by_id["b"] == (pytest.approx(1 / 62 + 1 / 61), {"bm25": 2, "dense": 1})
    assert by_id["a"] == (pytest.approx(1 / 61), {"bm25": 1})
    assert by_id["c"] == (pytest.approx(1 / 62), {"dense": 2})
    assert [item for item, _, _ in fused] == ["b", "a", "c"]


def test_rrf_ties_are_deterministic_and_duplicates_ignored() -> None:
    fused = reciprocal_rank_fusion({"x": ["b", "b"], "y": ["a"]})
    assert [(item, ranks) for item, _, ranks in fused] == [("a", {"y": 1}), ("b", {"x": 1})]
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"x": []}) == []


# ---------------------------------------------------------------------------
# Retriever end to end (offline embedder)
# ---------------------------------------------------------------------------


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
    "store/cache.py": '''
        def evict_least_recently_used(cache, capacity):
            while len(cache) > capacity:
                cache.popitem(last=False)
    ''',
}


def test_search_returns_scored_hits_with_provenance(
    write_repo: WriteRepo, embedder: HashEmbedder
) -> None:
    retriever = open_retriever(write_repo(REPO), embedder)
    hits = retriever.search("parseHttpResponse", k=3)
    assert len(hits) == 3
    top = hits[0]
    assert top.chunk_id == "net/http.py::parse_http_response"
    assert top.sources == ("bm25", "dense") and set(top.ranks) == {"bm25", "dense"}
    assert top.raw_scores["bm25"] > 0
    assert top.lines == (top.chunk.start_line, top.chunk.end_line) == (4, 7)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_dense_only_hits_are_labelled(write_repo: WriteRepo, embedder: HashEmbedder) -> None:
    retriever = open_retriever(write_repo(REPO), embedder)
    hits = retriever.search("evict", k=10)
    labels = {h.chunk_id: h.sources for h in hits}
    assert labels["store/cache.py::evict_least_recently_used"] == ("bm25", "dense")
    assert all(s == ("dense",) for cid, s in labels.items() if "evict" not in cid)


def test_query_with_no_matches(write_repo: WriteRepo, embedder: HashEmbedder) -> None:
    retriever = open_retriever(write_repo(REPO), embedder)
    hits = retriever.search("zebra quokka narwhal", k=5)
    # BM25 finds nothing; dense still returns neighbours (no abstention yet).
    assert hits and all(h.sources == ("dense",) and "bm25" not in h.raw_scores for h in hits)
    assert retriever.search("   ", k=5) == []
    assert retriever.search("parse", k=0) == []


def test_empty_index_search(write_repo: WriteRepo, embedder: HashEmbedder) -> None:
    retriever = open_retriever(write_repo({"notes.txt": "hi"}), embedder)
    assert len(retriever) == 0
    assert retriever.search("anything") == []


def test_duplicate_names_both_retrievable(write_repo: WriteRepo, embedder: HashEmbedder) -> None:
    root = write_repo(
        {
            "a/util.py": "def helper():\n    return 'a'\n",
            "b/util.py": "def helper():\n    return 'b'\n",
        }
    )
    hits = open_retriever(root, embedder).search("helper", k=5)
    assert {h.chunk_id for h in hits} == {"a/util.py::helper", "b/util.py::helper"}
    assert all(h.sources == ("bm25", "dense") for h in hits)


def test_split_chunk_hit_cites_matching_part(write_repo: WriteRepo) -> None:
    lines = [f"    step_{i} = run(stage_{i})\n" for i in range(60)]
    lines[45] = "    checksum = verify_signature(payload)\n"
    root = write_repo({"big.py": "def pipeline():\n" + "".join(lines)})
    emb = HashEmbedder(max_tokens=40, dim=1024)  # wide enough to avoid bucket collisions
    (hit,) = open_retriever(root, emb).search("verify_signature payload", k=1)
    assert hit.chunk_id == "big.py::pipeline"
    start, end = hit.lines
    assert start <= 47 <= end and (start, end) != (1, 61)


def test_split_module_chunk_cites_real_file_lines(write_repo: WriteRepo) -> None:
    consts = "".join(f"CONST_{i} = {i}\n" for i in range(40))
    source = f"import os\n\ndef f():\n    pass\n\n{consts}TARGET_SETTING = 'special'\n"
    root = write_repo({"cfg.py": source})
    (hit,) = open_retriever(root, HashEmbedder(max_tokens=20, dim=1024)).search("TARGET_SETTING", k=1)
    assert hit.chunk_id == "cfg.py::<module>"
    target_line = source.splitlines().index("TARGET_SETTING = 'special'") + 1
    assert hit.lines[0] <= target_line == hit.lines[1]


def test_open_requires_existing_index(tmp_path: Path, embedder: HashEmbedder) -> None:
    with pytest.raises(FileNotFoundError):
        Retriever.open(tmp_path, embedder)


def test_open_rejects_other_model(write_repo: WriteRepo, embedder: HashEmbedder) -> None:
    root = write_repo({"a.py": "def f(): pass\n"})
    build_index(root, embedder)
    from chatter.index import IndexMismatchError

    with pytest.raises(IndexMismatchError):
        Retriever.open(root / ".chatter", HashEmbedder(name="different"))


def test_search_reflects_reindex(write_repo: WriteRepo, embedder: HashEmbedder) -> None:
    root = write_repo({"a.py": "def old_name():\n    pass\n"})
    open_retriever(root, embedder)
    (root / "a.py").write_text("def new_name():\n    pass\n")
    hits = open_retriever(root, embedder).search("old_name new_name", k=5)
    assert [h.chunk_id for h in hits] == ["a.py::new_name"]
