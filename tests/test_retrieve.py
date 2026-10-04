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


def test_single_retriever_modes(write_repo: WriteRepo, embedder: HashEmbedder) -> None:
    retriever = open_retriever(write_repo(REPO), embedder)
    queries_before = len(embedder.queries)
    bm25 = retriever.search("evict cache", k=10, mode="bm25")
    assert bm25 and all(h.sources == ("bm25",) for h in bm25)
    assert len(embedder.queries) == queries_before  # bm25 mode never embeds the query
    dense = retriever.search("evict cache", k=10, mode="dense")
    assert dense and all(h.sources == ("dense",) for h in dense)
    fused = retriever.search("evict cache", k=10)
    assert [h.chunk_id for h in fused] == [h.chunk_id for h in retriever.search("evict cache", k=10, mode="fused")]
    with pytest.raises(ValueError, match="unknown retrieval mode"):
        retriever.search("x", mode="hybrid")  # type: ignore[arg-type]


def test_single_mode_preserves_retriever_order(write_repo: WriteRepo, embedder: HashEmbedder) -> None:
    retriever = open_retriever(write_repo(REPO), embedder)
    raw = retriever._bm25.search(["cache", "evict"], 50)
    assert [h.chunk_id for h in retriever.search("cache evict", k=50, mode="bm25")] == [c for c, _ in raw]


# ---------------------------------------------------------------------------
# BM25 query stopwords
# ---------------------------------------------------------------------------


from chatter.retrieve import ENGLISH_STOPWORDS, bm25_query_tokens  # noqa: E402


def test_stopwords_removed_from_queries() -> None:
    assert bm25_query_tokens("What stops the local model from repeating the same sentence?", stem=False) == [
        "stops", "local", "model", "repeating", "sentence",
    ]
    assert bm25_query_tokens("When an asyncio program's main coroutine finishes", stem=False) == [
        "asyncio", "program", "main", "coroutine", "finishes",
    ]
    assert bm25_query_tokens("how is it done and why", stem=False) == ["done"]
    assert bm25_query_tokens("what is this", stem=False) == []


def test_backticked_words_are_never_dropped() -> None:
    assert bm25_query_tokens("what does `from` do in `for x in y`", stem=False) == [
        "from", "for", "x", "in", "y",
    ]
    assert bm25_query_tokens("unclosed `the backtick", stem=False) == ["unclosed", "backtick"]


def test_identifier_like_words_are_never_dropped() -> None:
    assert bm25_query_tokens("where is_set and doesNot and IsDone used", stem=False) == [
        "is_set", "is", "set", "doesnot", "does", "not", "isdone", "is", "done", "used",
    ]
    # A capital only in first position is ordinary sentence case, not an identifier.
    assert bm25_query_tokens("The What", stem=False) == []


def test_symbol_names_are_never_dropped() -> None:
    names = frozenset({"once", "close"})
    assert bm25_query_tokens("run it once then close", names, stem=False) == ["run", "once", "close"]
    assert bm25_query_tokens("run it once then close", stem=False) == ["run", "close"]


def test_stopword_list_is_lowercase_without_apostrophes() -> None:
    assert len(ENGLISH_STOPWORDS) > 120
    assert all(w == w.lower() and "'" not in w for w in ENGLISH_STOPWORDS)


def test_retriever_applies_query_stopwords_to_bm25_only(
    write_repo: WriteRepo, embedder: HashEmbedder
) -> None:
    root = write_repo(
        {
            "prose.py": '"""The of and to in is it that the of and to."""\n\ndef helper():\n    pass\n',
            "cache.py": "def evict(cache):\n    cache.clear()\n",
        }
    )
    retriever = open_retriever(root, embedder)
    assert retriever.search("what is the", k=5, mode="bm25") == []
    hits = retriever.search("how is the cache evicted", k=5, mode="bm25")
    assert [h.chunk_id for h in hits] == ["cache.py::evict"]
    assert retriever.search("what is the", k=5, mode="dense")  # dense sees the full query
    assert retriever.search("`helper`", k=5, mode="bm25")[0].chunk_id == "prose.py::helper"


# ---------------------------------------------------------------------------
# Stemming
# ---------------------------------------------------------------------------


from chatter.index import stem_word  # noqa: E402


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("repeating", ["repeating", "repeat"]),
        ("RepetitionGuard", ["repetitionguard", "repetition", "guard", "repetit"]),
        ("max_retry_count", ["max_retry_count", "max", "retry", "count", "retri"]),
        ("vectors tasks", ["vectors", "vector", "tasks", "task"]),
        ("run utf8 b 42", ["run", "utf8", "b", "42"]),  # stems equal to the word are not repeated
    ],
)
def test_tokenize_code_with_stems(text: str, expected: list[str]) -> None:
    from chatter.index import tokenize_code

    assert tokenize_code(text, stem=True) == expected


def test_tokenize_code_is_unstemmed_by_default() -> None:
    from chatter.index import tokenize_code

    assert tokenize_code("repeating RepetitionGuard") == [
        "repeating", "repetitionguard", "repetition", "guard",
    ]


def test_query_tokens_are_stemmed_like_documents() -> None:
    assert bm25_query_tokens("What stops the local model from repeating?") == [
        "stops", "stop", "local", "model", "repeating", "repeat",
    ]
    assert stem_word("cancelled") == "cancel" and stem_word("headers") == "header"


def test_stemming_lets_inflected_queries_match(write_repo: WriteRepo, embedder: HashEmbedder) -> None:
    root = write_repo({"q.py": "def cancel_task(task):\n    task.cancel()\n", "o.py": "def other():\n    pass\n"})
    retriever = open_retriever(root, embedder)
    hits = retriever.search("which tasks get cancelled", k=5, mode="bm25")
    assert [h.chunk_id for h in hits] == ["q.py::cancel_task"]


def test_index_built_with_old_schema_requires_rebuild(write_repo: WriteRepo, embedder: HashEmbedder) -> None:
    import json as _json

    from chatter.index import IndexMismatchError

    root = write_repo({"a.py": "def f(): pass\n"})
    build_index(root, embedder)
    manifest_path = root / ".chatter" / "manifest.json"
    manifest = _json.loads(manifest_path.read_text())
    manifest_path.write_text(_json.dumps({**manifest, "schema": 1}))
    with pytest.raises(IndexMismatchError, match="schema 1"):
        Retriever.open(root / ".chatter", embedder)
    with pytest.raises(IndexMismatchError, match="schema 1"):
        build_index(root, embedder)
    build_index(root, embedder, rebuild=True)
    assert Retriever.open(root / ".chatter", embedder).search("f", k=1)
