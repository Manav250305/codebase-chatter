from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from chatter.extract import extract_source
from chatter.graph import (
    EDGE_TYPES,
    EDGE_WEIGHTS,
    TESTED_BY,
    GraphIndex,
    build_graph,
    collect_file_refs,
    is_test_path,
    module_name,
    write_graph,
)
from chatter.index import assign_chunk_ids


def graph_for(files: dict[str, str]):  # type: ignore[no-untyped-def]
    """Extract and resolve in memory: (edge set of (src id, dst id, type), stats, graph, ids)."""
    chunks, ids, file_refs = [], [], []
    for path in sorted(files):
        text = textwrap.dedent(files[path]).lstrip("\n")
        file_chunks = extract_source(text, path)
        indices = list(range(len(chunks), len(chunks) + len(file_chunks)))
        chunks += file_chunks
        ids += assign_chunk_ids(file_chunks)
        file_refs.append(collect_file_refs(text, path, file_chunks, indices))
    graph = build_graph(chunks, ids, file_refs)
    edges = {
        (ids[s], ids[d], EDGE_TYPES[int(k)])
        for s, d, k in zip(graph.sources.tolist(), graph.targets.tolist(), graph.types.tolist())
    }
    return edges, graph.stats, graph, ids


def has(edges: set, src: str, dst: str, kind: str = "calls") -> bool:  # type: ignore[type-arg]
    return (src, dst, kind) in edges


# ---------------------------------------------------------------------------
# Module names and test paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "module"),
    [
        ("asyncio/runners.py", "asyncio.runners"),
        ("src/chatter/index.py", "chatter.index"),
        ("pkg/__init__.py", "pkg"),
        ("setup.py", "setup"),
    ],
)
def test_module_name(path: str, module: str) -> None:
    assert module_name(path) == module


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("tests/test_a.py", True),
        ("pkg/tests/helpers.py", True),
        ("test_api.py", True),
        ("api_test.py", True),
        ("tests/conftest.py", True),
        ("src/chatter/answer.py", False),
        ("email/test.py", False),
    ],
)
def test_is_test_path(path: str, expected: bool) -> None:
    assert is_test_path(path) is expected


# ---------------------------------------------------------------------------
# Bare calls, scopes, shadowing
# ---------------------------------------------------------------------------


def test_bare_calls_nested_defs_and_shadowing() -> None:
    edges, stats, _, _ = graph_for(
        {
            "a.py": """
                def helper():
                    return 1

                def main():
                    return helper()

                def outer():
                    def inner():
                        return helper()
                    return inner()

                def shadowed(helper):
                    return helper()

                def assigned():
                    helper = lambda: 2
                    return helper()

                def builtin_and_unknown():
                    len([])
                    missing_function()
            """
        }
    )
    assert has(edges, "a.py::main", "a.py::helper")
    assert has(edges, "a.py::outer", "a.py::outer.inner")  # nested def in the enclosing scope
    assert has(edges, "a.py::outer.inner", "a.py::helper")  # through the scope chain
    assert not any(src == "a.py::shadowed" and kind == "calls" for src, _, kind in edges)
    assert not any(src == "a.py::assigned" and kind == "calls" for src, _, kind in edges)
    dropped = stats["dropped_by_reason"]
    assert dropped["call:local_variable"] == 2
    assert dropped["call:builtin"] == 1 and dropped["call:unknown_name"] == 1


def test_last_definition_wins_and_class_body_does_not_shadow() -> None:
    edges, _, _, _ = graph_for(
        {
            "a.py": """
                def name():
                    return 1

                def name():
                    return 2

                class K:
                    name = "attribute"

                    def m(self):
                        return name()
            """
        }
    )
    assert has(edges, "a.py::K.m", "a.py::name~2")  # module function, not the class attribute
    assert not has(edges, "a.py::K.m", "a.py::name")


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


IMPORTS = {
    "pkg/__init__.py": "from .util import tool\n",
    "pkg/util.py": """
        def tool():
            return 1

        def other():
            return 2
    """,
    "pkg/sub/__init__.py": "",
    "pkg/sub/deep.py": """
        from .. import util
        from ..util import other as o
        from ...outside import nothing

        def go():
            util.tool()
            o()
    """,
    "app.py": """
        import pkg.util as u
        import pkg.util
        from pkg.util import tool as t
        from pkg import tool
        from pkg.util import *
        import os.path

        def a():
            u.tool()

        def b():
            pkg.util.other()

        def c():
            t()

        def d():
            tool()

        def e():
            other()

        def f():
            from pkg.util import other as local_other
            local_other()

        def g():
            os.path.join("x", "y")
    """,
}


def test_imports_aliases_reexports_and_wildcards() -> None:
    edges, stats, _, _ = graph_for(IMPORTS)
    assert has(edges, "app.py::a", "pkg/util.py::tool")  # import a.b as z
    assert has(edges, "app.py::b", "pkg/util.py::other")  # import a.b; a.b.f()
    assert has(edges, "app.py::c", "pkg/util.py::tool")  # from a import b as z
    assert has(edges, "app.py::d", "pkg/util.py::tool")  # package re-export in __init__
    assert has(edges, "app.py::e", "pkg/util.py::other")  # from a import *
    assert has(edges, "app.py::f", "pkg/util.py::other")  # function-local import
    assert stats["dropped_by_reason"]["call:external"] >= 1  # os.path.join


def test_relative_imports() -> None:
    edges, _, _, _ = graph_for(IMPORTS)
    assert has(edges, "pkg/sub/deep.py::go", "pkg/util.py::tool")  # from .. import util
    assert has(edges, "pkg/sub/deep.py::go", "pkg/util.py::other")  # from ..util import other as o


# ---------------------------------------------------------------------------
# self / cls / super / MRO, instantiation, contains, inheritance
# ---------------------------------------------------------------------------


CLASSES = {
    "shapes.py": """
        class Base:
            def __init__(self):
                self.ready = True

            def run(self):
                return self.step()

            def step(self):
                return 1

        class Mixin:
            def describe(self):
                return "mixin"

        class Child(Mixin, Base):
            def step(self):
                return super().step() + 1

            def go(self):
                self.describe()
                return self.run()

            @classmethod
            def make(cls):
                return cls.build()

            @classmethod
            def build(cls):
                return cls()

            def bad(self):
                return self.ready.upper()

        def factory():
            return Child()
    """
}


def test_self_cls_super_and_mro() -> None:
    edges, stats, _, _ = graph_for(CLASSES)
    assert has(edges, "shapes.py::Base.run", "shapes.py::Base.step")  # self.m in the defining class
    assert has(edges, "shapes.py::Child.go", "shapes.py::Base.run")  # inherited through the MRO
    assert has(edges, "shapes.py::Child.go", "shapes.py::Mixin.describe")  # MRO order: Mixin first
    assert has(edges, "shapes.py::Child.step", "shapes.py::Base.step")  # super() skips Child
    assert has(edges, "shapes.py::Child.make", "shapes.py::Child.build")  # cls.m
    assert stats["dropped_by_reason"]["call:instance_attribute"] == 1  # self.ready.upper()


def test_instantiation_inherits_init_and_contains() -> None:
    edges, _, _, _ = graph_for(CLASSES)
    assert has(edges, "shapes.py::factory", "shapes.py::Child", "instantiates")
    assert has(edges, "shapes.py::factory", "shapes.py::Base.__init__")  # __init__ via MRO
    assert has(edges, "shapes.py::Child", "shapes.py::Base", "inherits")
    assert has(edges, "shapes.py::Child", "shapes.py::Mixin", "inherits")
    assert has(edges, "shapes.py::Child", "shapes.py::Child.go", "contains")
    assert has(edges, "shapes.py::Base", "shapes.py::Base.__init__", "contains")


# ---------------------------------------------------------------------------
# with / async with, callback references, dynamic calls
# ---------------------------------------------------------------------------


def test_context_managers_and_callbacks() -> None:
    edges, stats, _, _ = graph_for(
        {
            "res.py": """
                class Res:
                    def __enter__(self):
                        return self

                    def __exit__(self, *exc):
                        self.close()

                    def close(self):
                        pass

                class ARes:
                    async def __aenter__(self):
                        return self

                    async def __aexit__(self, *exc):
                        pass

                def callback(result):
                    pass

                def use(lock, loop):
                    with Res() as r, lock:
                        loop.call_soon(callback)

                async def use_async():
                    async with ARes():
                        pass

                class Group:
                    def create(self, task):
                        task.add_done_callback(self._on_done)
                        task.add_done_callback(cb=self.missing)

                    def _on_done(self, task):
                        pass
            """
        }
    )
    assert has(edges, "res.py::use", "res.py::Res.__enter__", "context_manager")
    assert has(edges, "res.py::use", "res.py::Res.__exit__", "context_manager")
    assert has(edges, "res.py::use_async", "res.py::ARes.__aenter__", "context_manager")
    assert has(edges, "res.py::use_async", "res.py::ARes.__aexit__", "context_manager")
    assert has(edges, "res.py::Res.__exit__", "res.py::Res.close")
    assert has(edges, "res.py::use", "res.py::callback", "references")
    assert has(edges, "res.py::Group.create", "res.py::Group._on_done", "references")
    assert not any(dst == "res.py::Group.missing" for _, dst, _ in edges)
    dropped = stats["dropped_by_reason"]
    assert dropped["call:untyped_receiver"] >= 2  # loop.call_soon, task.add_done_callback


def test_dynamic_calls_are_dropped_not_guessed() -> None:
    edges, stats, _, _ = graph_for(
        {
            "dyn.py": """
                def a():
                    pass

                HANDLERS = {"a": a}

                def dispatch(name):
                    HANDLERS[name]()
                    getattr(dispatch, name)()
                    make()()
            """
        }
    )
    assert not any(src == "dyn.py::dispatch" and kind == "calls" for src, dst, kind in edges if dst == "dyn.py::a")
    dropped = stats["dropped_by_reason"]
    assert dropped["call:dynamic"] >= 2  # subscript call, call-result call
    assert stats["references"]["call"] >= 4


def test_syntax_errors_do_not_break_resolution() -> None:
    edges, _, _, _ = graph_for(
        {
            "broken.py": """
                def ok():
                    return helper()

                def bad(:
                    pass

                def helper():
                    return 1
            """
        }
    )
    assert has(edges, "broken.py::ok", "broken.py::helper")


# ---------------------------------------------------------------------------
# tested_by
# ---------------------------------------------------------------------------


def test_edges_from_test_files_are_tested_by_and_never_expanded() -> None:
    files = {
        "lib.py": """
            def helper():
                return 1

            def api():
                return helper()
        """,
        "tests/test_lib.py": """
            from lib import api, helper

            class TestApi:
                def test_api(self):
                    assert api() == helper()
        """,
        "lib_test.py": "from lib import api\n\ndef test_it():\n    api()\n",
    }
    edges, _, graph, ids = graph_for(files)
    assert has(edges, "lib.py::api", "lib.py::helper", "calls")  # normal code unaffected
    assert has(edges, "tests/test_lib.py::TestApi.test_api", "lib.py::api", "tested_by")
    assert has(edges, "tests/test_lib.py::TestApi.test_api", "lib.py::helper", "tested_by")
    assert has(edges, "tests/test_lib.py::TestApi", "tests/test_lib.py::TestApi.test_api", "tested_by")
    assert has(edges, "lib_test.py::test_it", "lib.py::api", "tested_by")
    assert not any(src.startswith(("tests/", "lib_test")) and kind != "tested_by" for src, _, kind in edges)

    index = GraphIndex(*_csr(graph, len(ids)))
    api = ids.index("lib.py::api")
    assert set(index.neighbors(api)) == {ids.index("lib.py::helper")}  # test callers excluded
    assert TESTED_BY not in EDGE_WEIGHTS


# ---------------------------------------------------------------------------
# Storage and adjacency
# ---------------------------------------------------------------------------


def _csr(graph, n):  # type: ignore[no-untyped-def]
    counts = np.bincount(graph.sources, minlength=n)
    return np.concatenate([[0], np.cumsum(counts)]), graph.targets, graph.types


def test_write_and_load_graph(tmp_path: Path) -> None:
    edges, stats, graph, ids = graph_for(CLASSES)
    write_graph(tmp_path, graph, len(ids))
    index = GraphIndex.load(tmp_path)
    assert index is not None and index.n == len(ids)
    loaded = {
        (ids[s], ids[t], kind) for s in range(len(ids)) for t, kind in index.edges_from(s)
    }
    assert loaded == edges
    child = ids.index("shapes.py::Child")
    neighbors = index.neighbors(child)
    assert neighbors[ids.index("shapes.py::Base")] == EDGE_WEIGHTS[6][0]  # inherits, forward
    assert neighbors[ids.index("shapes.py::factory")] == 1.0  # instantiated by: reverse edge
    assert neighbors[ids.index("shapes.py::Child.go")] == EDGE_WEIGHTS[5][0]  # contains
    go = ids.index("shapes.py::Child.go")
    assert index.neighbors(go)[child] == EDGE_WEIGHTS[5][1]  # contained by: reverse weight
    assert index.in_degree[ids.index("shapes.py::Base.step")] >= 2


def test_graph_is_deterministic() -> None:
    first = graph_for(IMPORTS)[2]
    second = graph_for(IMPORTS)[2]
    assert np.array_equal(first.sources, second.sources)
    assert np.array_equal(first.targets, second.targets)
    assert np.array_equal(first.types, second.types)
