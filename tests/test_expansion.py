from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from chatter.graph import CALLS, CONTAINS, TESTED_BY, GraphIndex
from chatter.index import build_index, index_data
from chatter.retrieve import (
    GRAPH_BUDGET,
    GRAPH_PER_SEED,
    GraphExpansion,
    Retriever,
    _apply_budget,
    graph_ranking,
)
from conftest import HashEmbedder, WriteRepo


def csr(n: int, edges: list[tuple[int, int, int]]) -> GraphIndex:
    edges = sorted(edges)
    sources = np.array([s for s, _, _ in edges], dtype=np.int64)
    indptr = np.concatenate([[0], np.cumsum(np.bincount(sources, minlength=n))]) if edges else np.zeros(n + 1)
    return GraphIndex(
        indptr.astype(np.int64),
        np.array([t for _, t, _ in edges], dtype=np.int32),
        np.array([k for _, _, k in edges], dtype=np.uint8),
    )


# ---------------------------------------------------------------------------
# Budget and protected ranks
# ---------------------------------------------------------------------------


def test_apply_budget_keeps_top3_and_limits_promotions() -> None:
    base = [f"b{i}" for i in range(20)]  # b0..b9 form the base window
    expanded = ["x1", "x2", "b5", "x3", "x4", "x5", "b0", "b1", "b2", "b3", "b4", "b6", "b7", "b8", "b9", *base[10:]]
    result = _apply_budget(base, expanded)
    assert result[:3] == ["b0", "b1", "b2"]  # protected, whatever the expansion says
    top10 = result[:10]
    assert [c for c in top10 if not c.startswith("b")] == ["x1", "x2", "x3"]  # budget of 3
    assert top10[3:] == ["x1", "x2", "b5", "x3", "b3", "b4", "b6"]
    assert set(result) == set(expanded) and len(result) == len(expanded)  # nothing lost
    assert result.index("x4") > 9 and result.index("x5") > 9  # over budget: below the window


def test_apply_budget_with_short_lists() -> None:
    assert _apply_budget(["a", "b"], ["c", "a", "b"]) == ["a", "b", "c"]
    assert _apply_budget([], []) == []


# ---------------------------------------------------------------------------
# Graph ranking
# ---------------------------------------------------------------------------


def test_graph_ranking_per_seed_cap_relevance_and_hub_damping() -> None:
    n = 14
    star = [(0, i, CALLS) for i in range(1, 13)]  # seed 0 calls 1..12
    hub_callers = [(13, 12, CALLS)] + [(i, 12, CALLS) for i in range(1, 6)]  # 12 is widely called
    graph = csr(n, star + hub_callers)
    relevance = np.ones(n)
    relevance[3] = 0.0  # irrelevant to the query: never reached
    ranks = np.arange(n)
    order = graph_ranking(graph, [0], relevance, ranks, hops=1, rrf_k=60)
    assert len(order) == GRAPH_PER_SEED  # 11 relevant neighbours, capped at 8
    assert 3 not in order and 12 not in order  # irrelevant; hub damped below the cap
    assert order == sorted(order)  # equal scores: tie-break order


def test_graph_ranking_two_hops_and_reverse_edges() -> None:
    graph = csr(5, [(0, 1, CALLS), (1, 2, CALLS), (3, 0, CALLS), (4, 0, TESTED_BY), (1, 4, CONTAINS)])
    relevance = np.ones(5)
    one = graph_ranking(graph, [0], relevance, np.arange(5), hops=1, rrf_k=60)
    assert set(one) == {1, 3}  # callee and caller; tested_by never expanded
    two = graph_ranking(graph, [0], relevance, np.arange(5), hops=2, rrf_k=60)
    assert set(two) == {0, 1, 2, 3, 4}  # 2 via 1; 4 via contains from 1; back to 0
    assert two.index(1) < two.index(2)  # hop decay


# ---------------------------------------------------------------------------
# Retriever integration
# ---------------------------------------------------------------------------


REPO = {
    "orders.py": """
        def process_orders(orders):
            validated = check_inventory(orders)
            return schedule_delivery(validated)

        def check_inventory(orders):
            return [o for o in orders if o.stock > 0]

        def schedule_delivery(items):
            return sorted(items, key=lambda i: i.eta)
    """,
    "noise.py": "".join(
        f"def unrelated_{i}(value):\n    return value * {i}\n\n" for i in range(30)
    ),
    "tests/test_orders.py": """
        from orders import process_orders

        def test_process_orders():
            assert process_orders([]) == []
    """,
}


def indexed_repo(write_repo: WriteRepo) -> Path:
    root = write_repo(REPO)
    build_index(root, HashEmbedder(dim=256))
    return root


def retriever(write_repo: WriteRepo) -> Retriever:
    return Retriever.open(indexed_repo(write_repo) / ".chatter", HashEmbedder(dim=256))


def test_expansion_is_off_by_default(write_repo: WriteRepo) -> None:
    r = retriever(write_repo)
    assert r.has_graph
    plain = r.search("process orders", k=20)
    assert all("graph" not in h.sources for h in plain)


def test_expansion_reaches_callees_and_keeps_top3(write_repo: WriteRepo) -> None:
    r = retriever(write_repo)
    base = r.search("process orders", k=20)
    expanded = r.search("process orders", k=20, expansion=GraphExpansion(hops=1, weight=2.0))
    assert [h.chunk_id for h in expanded[:3]] == [h.chunk_id for h in base[:3]]
    reached = {h.chunk_id for h in expanded if "graph" in h.sources}
    assert "orders.py::schedule_delivery" in reached
    assert "tests/test_orders.py::test_process_orders" not in reached  # tested_by excluded
    base_ids = [h.chunk_id for h in base]
    new_ids = [h.chunk_id for h in expanded]
    assert new_ids.index("orders.py::schedule_delivery") <= base_ids.index("orders.py::schedule_delivery")
    promoted = [c for c in new_ids[:10] if c not in base_ids[:10]]
    assert len(promoted) <= GRAPH_BUDGET


def test_expansion_needs_fused_mode_and_a_graph(write_repo: WriteRepo) -> None:
    index_dir = indexed_repo(write_repo) / ".chatter"
    r = Retriever.open(index_dir, HashEmbedder(dim=256))
    dense = r.search("process orders", k=10, mode="dense", expansion=GraphExpansion())
    assert all(h.sources == ("dense",) for h in dense)
    data = index_data(index_dir)
    assert data is not None
    for name in ("graph_indptr.npy", "graph_targets.npy", "graph_types.npy", "graph.json"):
        (data.directory / name).unlink()
    without = Retriever.open(index_dir, HashEmbedder(dim=256))
    assert not without.has_graph
    assert [h.chunk_id for h in without.search("process orders", k=10, expansion=GraphExpansion())] == [
        h.chunk_id for h in without.search("process orders", k=10)
    ]


_SCRIPT = """
import json, sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
from conftest import HashEmbedder
from chatter.retrieve import GraphExpansion, Retriever
r = Retriever.open(sys.argv[3], HashEmbedder(dim=256))
out = {}
for hops in (1, 2):
    hits = r.search("process orders delivery", k=40, candidates=200, expansion=GraphExpansion(hops=hops, weight=1.5))
    out[str(hops)] = [[h.chunk_id, round(h.score, 12), list(h.sources)] for h in hits]
print(json.dumps(out))
"""


def test_expansion_is_identical_across_processes(write_repo: WriteRepo) -> None:
    root = indexed_repo(write_repo)
    runs = []
    for seed in ("3", "424242"):
        result = subprocess.run(
            [sys.executable, "-c", _SCRIPT, str(Path(__file__).parent),
             str(Path(__file__).parents[1] / "src"), str(root / ".chatter")],
            capture_output=True, text=True, check=True, env={**os.environ, "PYTHONHASHSEED": seed},
        )
        runs.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert runs[0] == runs[1] and runs[0]["1"]
