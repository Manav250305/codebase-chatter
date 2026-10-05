from __future__ import annotations

import json
from pathlib import Path

import pytest

from chatter.extract import Chunk, extract_source
from chatter.index import (
    MANIFEST_FILE,
    SCHEMA_VERSION,
    IndexConfig,
    IndexMismatchError,
    assign_chunk_ids,
    build_index,
    chunk_from_dict,
    chunk_to_dict,
    embedding_header,
    embedding_text,
    index_data,
    iter_records,
    iter_source_files,
    load_vectors,
    plan_parts,
    tokenize_code,
)
from conftest import HashEmbedder, WriteRepo


def chunk(code: str, path: str = "pkg/mod.py") -> list[Chunk]:
    import textwrap

    return extract_source(textwrap.dedent(code), path)


def chunks_file(index_dir: Path) -> Path:
    data = index_data(index_dir)
    assert data is not None, f"no committed build in {index_dir}"
    return data.chunks_path


def stored_ids(index_dir: Path) -> list[str]:
    """One id per stored vector row: ``<chunk id>#<part>``."""
    return sorted(
        f"{record['id']}#{part}"
        for record in iter_records(chunks_file(index_dir))
        for part in range(record["rows"][1])
    )


def record_ids(index_dir: Path) -> list[str]:
    return [record["id"] for record in iter_records(chunks_file(index_dir))]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("max_retry_count", ["max_retry_count", "max", "retry", "count"]),
        ("parseHTTPResponse", ["parsehttpresponse", "parse", "http", "response"]),
        ("HTTPServer", ["httpserver", "http", "server"]),
        ("getX", ["getx", "get", "x"]),
        ("_private __dunder__", ["_private", "private", "__dunder__", "dunder"]),
        ("plain", ["plain"]),
        ("utf8 base64Encode", ["utf8", "base64encode", "base64", "encode"]),
        ("x = 42 + y", ["x", "42", "y"]),
        ("größe_wert", ["größe_wert", "größe", "wert"]),
        ("", []),
    ],
)
def test_tokenize_code(text: str, expected: list[str]) -> None:
    assert tokenize_code(text) == expected


# ---------------------------------------------------------------------------
# Chunk ids and embedding text
# ---------------------------------------------------------------------------


def test_chunk_ids_suffix_duplicates_within_a_file() -> None:
    chunks = chunk('''
        """Doc."""
        class C:
            @property
            def x(self): return 1
            @x.setter
            def x(self, v): pass
            @x.deleter
            def x(self): pass
        if A:
            def f(): pass
        else:
            def f(): pass
    ''')
    assert assign_chunk_ids(chunks) == [
        "pkg/mod.py::<module>",
        "pkg/mod.py::C",
        "pkg/mod.py::C.x",
        "pkg/mod.py::C.x~2",
        "pkg/mod.py::C.x~3",
        "pkg/mod.py::f",
        "pkg/mod.py::f~2",
    ]


def test_embedding_text_has_header_then_source() -> None:
    (c,) = chunk('''
        def add(a, b):
            """Add numbers."""
            return a + b
    ''')
    assert embedding_header(c) == (
        "# file: pkg/mod.py\n# symbol: add (function)\n# doc: Add numbers.\n"
    )
    assert embedding_text(c) == embedding_header(c) + c.source


def test_embedding_header_truncates_long_docstrings() -> None:
    (c,) = chunk(f'def f():\n    """{"word " * 500}"""\n')
    doc_line = embedding_header(c).splitlines()[-1]
    assert doc_line.endswith("…") and len(doc_line) < 420


def test_chunk_round_trips_through_json() -> None:
    for c in chunk('"""D."""\nimport os\nX = 1\n\nY = 2\n@dec\nasync def f(): pass\n'):
        assert chunk_from_dict(json.loads(json.dumps(chunk_to_dict(c)))) == c


# ---------------------------------------------------------------------------
# Splitting oversized chunks
# ---------------------------------------------------------------------------


def test_small_chunk_is_one_part() -> None:
    (c,) = chunk("def f():\n    return 1\n")
    (parts,) = plan_parts([c], HashEmbedder(max_tokens=100))
    assert len(parts) == 1
    assert (parts[0].first, parts[0].last, parts[0].text) == (0, 1, embedding_text(c))


def test_oversized_chunk_split_by_lines_within_budget() -> None:
    body = "".join(f"    value_{i} = compute(alpha, beta, gamma)\n" for i in range(60))
    (c,) = chunk(f'def big():\n    """Big one."""\n{body}')
    emb = HashEmbedder(max_tokens=50)
    (parts,) = plan_parts([c], emb)
    assert len(parts) > 1
    header = embedding_header(c)
    lines = c.source.split("\n")
    # Parts tile the source exactly, in order, each with the header.
    assert parts[0].first == 0 and parts[-1].last == len(lines) - 1
    for prev, cur in zip(parts, parts[1:]):
        assert cur.first == prev.last + 1
    for i, part in enumerate(parts):
        assert part.index == i
        assert part.text == header + "\n".join(lines[part.first : part.last + 1])
        assert emb.count_tokens([part.text])[0] <= emb.max_tokens


def test_single_line_longer_than_budget_becomes_its_own_part() -> None:
    (c,) = chunk("def f():\n    x = 1\n    y = [" + ", ".join(["a"] * 200) + "]\n    z = 2\n")
    (parts,) = plan_parts([c], HashEmbedder(max_tokens=30))
    assert [(p.first, p.last) for p in parts] == [(0, 1), (2, 2), (3, 3)]


# ---------------------------------------------------------------------------
# Repository walking
# ---------------------------------------------------------------------------


def test_iter_source_files_skips_dirs_and_respects_gitignore(write_repo: WriteRepo) -> None:
    root = write_repo(
        {
            ".gitignore": "build/\n*_generated.py\n!keep_generated.py\n",
            "a.py": "",
            "README.md": "",
            "x_generated.py": "",
            "keep_generated.py": "",
            "build/out.py": "",
            ".git/hooks/h.py": "",
            ".venv/lib/site.py": "",
            "node_modules/pkg/x.py": "",
            "sub/__pycache__/c.py": "",
            "sub/venv/v.py": "",
            "sub/.gitignore": "local_*.py\n/only_here.py\n",
            "sub/b.py": "",
            "sub/local_x.py": "",
            "sub/only_here.py": "",
            "sub/deeper/only_here.py": "",
            "other/local_x.py": "",
            "._a.py": "",  # macOS AppleDouble metadata on exFAT volumes
        }
    )
    got = [p.relative_to(root.resolve()).as_posix() for p in iter_source_files(root)]
    assert got == [
        "a.py",
        "keep_generated.py",
        "other/local_x.py",
        "sub/b.py",
        "sub/deeper/only_here.py",
    ]


def test_iter_source_files_does_not_follow_symlinks(write_repo: WriteRepo, tmp_path: Path) -> None:
    root = write_repo({"real/a.py": "", "outside_target/o.py": ""})
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "e.py").write_text("")
    try:
        (root / "link_dir").symlink_to(outside, target_is_directory=True)
        (root / "link.py").symlink_to(root / "real" / "a.py")
        (root / "loop").symlink_to(root, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unsupported on this filesystem")
    got = [p.relative_to(root.resolve()).as_posix() for p in iter_source_files(root)]
    assert got == ["outside_target/o.py", "real/a.py"]


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------


def test_empty_repo(write_repo: WriteRepo, embedder: HashEmbedder) -> None:
    root = write_repo({"README.md": "# nothing"})
    stats = build_index(root, embedder)
    assert (stats.files, stats.chunks, stats.embedded_parts) == (0, 0, 0)
    index_dir = root / ".chatter"
    assert chunks_file(index_dir).read_text() == ""
    assert json.loads((index_dir / MANIFEST_FILE).read_text())["model"] == "fake-hash"
    assert stored_ids(index_dir) == []


def test_single_file(write_repo: WriteRepo, embedder: HashEmbedder) -> None:
    root = write_repo(
        {
            "calc.py": '''
                """Calculator."""
                import math

                def add(a, b):
                    return a + b

                class Calc:
                    def mul(self, a, b):
                        return a * b
            '''
        }
    )
    stats = build_index(root, embedder)
    assert (stats.files, stats.chunks, stats.embedded_parts, stats.reused_chunks) == (1, 4, 4, 0)
    index_dir = root / ".chatter"
    expected = ["calc.py::<module>", "calc.py::Calc", "calc.py::Calc.mul", "calc.py::add"]
    assert sorted(record_ids(index_dir)) == expected
    assert stored_ids(index_dir) == sorted(f"{cid}#0" for cid in expected)
    record = json.loads(chunks_file(index_dir).read_text().splitlines()[0])
    assert {"id", "chunk", "tokens"} <= record.keys()


def test_duplicate_names_across_files(write_repo: WriteRepo, embedder: HashEmbedder) -> None:
    root = write_repo(
        {
            "a/util.py": "def helper():\n    return 'a'\n",
            "b/util.py": "def helper():\n    return 'b'\n",
        }
    )
    stats = build_index(root, embedder)
    assert stats.chunks == 2
    assert record_ids(root / ".chatter") == ["a/util.py::helper", "b/util.py::helper"]


def test_oversized_chunk_stored_as_parts(write_repo: WriteRepo) -> None:
    body = "".join(f"    step_{i} = run(stage_{i}, options)\n" for i in range(80))
    root = write_repo({"big.py": f"def pipeline():\n{body}"})
    emb = HashEmbedder(max_tokens=40)
    stats = build_index(root, emb)
    ids = stored_ids(root / ".chatter")
    assert stats.chunks == 1 and stats.embedded_parts == len(ids) > 1
    assert all(pid.startswith("big.py::pipeline#") for pid in ids)
    assert max(emb.count_tokens(emb.embedded_texts)) <= 40


def test_reindex_is_idempotent(write_repo: WriteRepo, embedder: HashEmbedder) -> None:
    body = "".join(f"    s{i} = f(x{i})\n" for i in range(50))
    root = write_repo(
        {
            "a.py": "X = 1\n\ndef f():\n    pass\n\nclass K:\n    def m(self): pass\n",
            "b.py": f"def long():\n{body}",
        }
    )
    emb = HashEmbedder(max_tokens=40)
    index_dir = root / ".chatter"
    first = build_index(root, emb)
    snapshot = (
        chunks_file(index_dir).read_bytes(),
        (index_dir / MANIFEST_FILE).read_bytes(),
        stored_ids(index_dir),
    )
    calls = len(emb.document_batches)

    second = build_index(root, emb)
    assert second.embedded_parts == 0 and len(emb.document_batches) == calls
    assert (second.chunks, second.reused_chunks, second.deleted_chunks) == (first.chunks, first.chunks, 0)
    assert snapshot == (
        chunks_file(index_dir).read_bytes(),
        (index_dir / MANIFEST_FILE).read_bytes(),
        stored_ids(index_dir),
    )


def test_reindex_embeds_only_changes_and_removes_deleted(
    write_repo: WriteRepo, embedder: HashEmbedder
) -> None:
    root = write_repo(
        {
            "a.py": "def keep():\n    return 1\n\ndef change():\n    return 1\n",
            "gone.py": "def bye():\n    pass\n",
        }
    )
    build_index(root, embedder)
    (root / "a.py").write_text(
        "\n\ndef keep():\n    return 1\n\ndef change():\n    return 2\n\ndef new():\n    pass\n"
    )
    (root / "gone.py").unlink()
    before = len(embedder.embedded_texts)

    stats = build_index(root, embedder)
    newly_embedded = embedder.embedded_texts[before:]
    assert (stats.embedded_parts, stats.reused_chunks, stats.deleted_chunks) == (2, 1, 1)
    assert sorted(t.splitlines()[1] for t in newly_embedded) == [
        "# symbol: change (function)",
        "# symbol: new (function)",
    ]
    index_dir = root / ".chatter"
    assert stored_ids(index_dir) == ["a.py::change#0", "a.py::keep#0", "a.py::new#0"]
    keep = next(
        json.loads(line)["chunk"]
        for line in chunks_file(index_dir).read_text().splitlines()
        if json.loads(line)["id"] == "a.py::keep"
    )
    assert keep["start_line"] == 3  # moved lines are refreshed without re-embedding


def test_shrinking_chunk_drops_stale_parts(write_repo: WriteRepo) -> None:
    body = "".join(f"    s{i} = f(x{i})\n" for i in range(50))
    root = write_repo({"m.py": f"def g():\n{body}"})
    emb = HashEmbedder(max_tokens=40)
    build_index(root, emb)
    assert len(stored_ids(root / ".chatter")) > 1
    (root / "m.py").write_text("def g():\n    return 1\n")
    build_index(root, emb)
    assert stored_ids(root / ".chatter") == ["m.py::g#0"]


def test_model_mismatch_requires_rebuild(write_repo: WriteRepo, embedder: HashEmbedder) -> None:
    root = write_repo({"a.py": "def f(): pass\n"})
    build_index(root, embedder)
    other = HashEmbedder(name="other-model", dim=32)
    with pytest.raises(IndexMismatchError, match="other-model"):
        build_index(root, other)
    stats = build_index(root, other, rebuild=True)
    assert stats.embedded_parts == 1
    assert json.loads((root / ".chatter" / MANIFEST_FILE).read_text())["model"] == "other-model"


def test_custom_index_dir_and_batching(write_repo: WriteRepo, tmp_path: Path) -> None:
    root = write_repo({f"m{i}.py": f"def f{i}():\n    pass\n" for i in range(7)})
    emb = HashEmbedder()
    out = tmp_path / "idx"
    stats = build_index(root, emb, index_dir=out, config=IndexConfig(batch_size=3))
    assert stats.chunks == 7 and (out / MANIFEST_FILE).exists()
    assert not (root / ".chatter").exists()
    assert [len(b) for b in emb.document_batches] == [3, 3, 1]


def test_missing_repo_raises(tmp_path: Path, embedder: HashEmbedder) -> None:
    with pytest.raises(NotADirectoryError):
        build_index(tmp_path / "nope", embedder)


def test_unwritable_index_dir_raises_os_error(
    write_repo: WriteRepo, embedder: HashEmbedder, tmp_path: Path
) -> None:
    import os

    root = write_repo({"a.py": "def f(): pass\n"})
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        if os.access(locked, os.W_OK):
            pytest.skip("running with privileges that ignore directory permissions")
        with pytest.raises(PermissionError):
            build_index(root, embedder, index_dir=locked)
        assert embedder.document_batches == []  # failed before any embedding work
    finally:
        locked.chmod(0o700)


# ---------------------------------------------------------------------------
# Storage layout: generations, vectors, no-op rebuilds, legacy cleanup
# ---------------------------------------------------------------------------


def test_vectors_are_normalized_float32_and_memory_mapped(
    write_repo: WriteRepo, embedder: HashEmbedder
) -> None:
    import numpy as np

    root = write_repo({"a.py": "def f():\n    return 1\n\nclass K:\n    def m(self): pass\n"})
    build_index(root, embedder)
    data = index_data(root / ".chatter")
    assert data is not None and data.directory.name.startswith("gen-")
    vectors = load_vectors(data.vectors_path)
    assert isinstance(vectors, np.memmap) and vectors.dtype == np.float32
    assert vectors.shape == (3, 64)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    manifest = json.loads((root / ".chatter" / MANIFEST_FILE).read_text())
    assert (manifest["rows"], manifest["dim"], manifest["schema"]) == (3, 64, SCHEMA_VERSION)
    rows = [r["rows"] for r in iter_records(data.chunks_path)]
    assert [start for start, _ in rows] == [0, 1, 2]  # contiguous, in record order


def test_unchanged_reindex_keeps_the_committed_generation(
    write_repo: WriteRepo, embedder: HashEmbedder
) -> None:
    root = write_repo({"a.py": "def f(): pass\n"})
    index_dir = root / ".chatter"
    build_index(root, embedder)
    first = index_data(index_dir)
    build_index(root, embedder)
    assert index_data(index_dir) == first
    assert sorted(p.name for p in index_dir.iterdir()) == sorted([first.directory.name, MANIFEST_FILE])


def test_changed_reindex_replaces_generation_and_removes_old(
    write_repo: WriteRepo, embedder: HashEmbedder
) -> None:
    root = write_repo({"a.py": "def f(): pass\n"})
    index_dir = root / ".chatter"
    build_index(root, embedder)
    first = index_data(index_dir)
    (root / "a.py").write_text("def g(): pass\n")
    build_index(root, embedder)
    second = index_data(index_dir)
    assert second != first and not first.directory.exists()
    assert record_ids(index_dir) == ["a.py::g"]


def test_failed_build_leaves_committed_index_intact(
    write_repo: WriteRepo, embedder: HashEmbedder
) -> None:
    root = write_repo({"a.py": "def f(): pass\n"})
    index_dir = root / ".chatter"
    build_index(root, embedder)
    committed = index_data(index_dir)
    (root / "b.py").write_text("def g(): pass\n")

    class Exploding(HashEmbedder):
        def embed_documents(self, texts):  # type: ignore[no-untyped-def]
            raise RuntimeError("embedder crashed")

    with pytest.raises(RuntimeError, match="embedder crashed"):
        build_index(root, Exploding())
    assert index_data(index_dir) == committed  # manifest still points at the old build
    assert record_ids(index_dir) == ["a.py::f"]
    assert [p.name for p in index_dir.iterdir() if p.name.startswith("gen-")] == [committed.directory.name]


def test_stray_generations_are_cleaned_up(write_repo: WriteRepo, embedder: HashEmbedder) -> None:
    root = write_repo({"a.py": "def f(): pass\n"})
    index_dir = root / ".chatter"
    build_index(root, embedder)
    (index_dir / "gen-crashed").mkdir()  # e.g. left by a killed process
    (root / "a.py").write_text("def g(): pass\n")
    build_index(root, embedder)
    assert [p.name for p in index_dir.iterdir() if p.name.startswith("gen-")] == [
        index_data(index_dir).directory.name
    ]


def test_legacy_chroma_index_needs_rebuild_and_is_cleaned(
    write_repo: WriteRepo, embedder: HashEmbedder
) -> None:
    root = write_repo({"a.py": "def f(): pass\n"})
    index_dir = root / ".chatter"
    index_dir.mkdir()
    (index_dir / "chroma").mkdir()
    (index_dir / "chroma" / "chroma.sqlite3").write_bytes(b"old")
    (index_dir / "chunks.jsonl").write_text("{}\n")
    (index_dir / MANIFEST_FILE).write_text(json.dumps({"schema": 2, "model": embedder.name}))
    with pytest.raises(IndexMismatchError, match="schema 2"):
        build_index(root, embedder)
    build_index(root, embedder, rebuild=True)
    assert not (index_dir / "chroma").exists() and not (index_dir / "chunks.jsonl").exists()
    assert record_ids(index_dir) == ["a.py::f"]


def test_unrelated_files_in_a_fresh_index_dir_are_kept(
    write_repo: WriteRepo, embedder: HashEmbedder, tmp_path: Path
) -> None:
    root = write_repo({"a.py": "def f(): pass\n"})
    shared = tmp_path / "shared"
    (shared / "chroma").mkdir(parents=True)  # not ours: no manifest yet
    (shared / "chunks.jsonl").write_text("user data\n")
    build_index(root, embedder, index_dir=shared)
    assert (shared / "chroma").is_dir() and (shared / "chunks.jsonl").read_text() == "user data\n"


# ---------------------------------------------------------------------------
# Call graph in the index
# ---------------------------------------------------------------------------


def test_index_writes_call_graph_across_files(write_repo: WriteRepo, embedder: HashEmbedder) -> None:
    from chatter.graph import GRAPH_FILES, GraphIndex, read_graph_stats

    root = write_repo(
        {
            "pkg/__init__.py": "",
            "pkg/util.py": "def tool():\n    return 1\n",
            "app.py": "from pkg.util import tool\n\ndef main():\n    return tool()\n",
        }
    )
    build_index(root, embedder)
    data = index_data(root / ".chatter")
    assert all((data.directory / name).exists() for name in GRAPH_FILES)
    ids = record_ids(root / ".chatter")
    graph = GraphIndex.load(data.directory)
    main = ids.index("app.py::main")
    assert (ids.index("pkg/util.py::tool"), "calls") in graph.edges_from(main)
    stats = read_graph_stats(data.directory)
    assert stats["resolved_by_rule"]["import"] == 1 and stats["references"]["call"] == 1


def test_graph_counts_in_noop_rebuild(write_repo: WriteRepo, embedder: HashEmbedder) -> None:
    root = write_repo({"a.py": "def f():\n    return g()\n\ndef g():\n    return 1\n"})
    build_index(root, embedder)
    first = index_data(root / ".chatter")
    build_index(root, embedder)
    assert index_data(root / ".chatter") == first  # graph identical too: no new generation
