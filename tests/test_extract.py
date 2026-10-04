from __future__ import annotations

import os
import random
import textwrap
from pathlib import Path

import pytest

from chatter.extract import Chunk, extract_file, extract_source

FIXTURES = Path(__file__).parent / "fixtures"


def extract(code: str, path: str = "mod.py") -> list[Chunk]:
    return extract_source(textwrap.dedent(code), path)


def by_qualname(chunks: list[Chunk]) -> dict[str, Chunk]:
    return {c.qualname: c for c in chunks}


def assert_source_consistent(chunks: list[Chunk], text: str) -> None:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for c in chunks:
        assert 1 <= c.start_line <= c.end_line <= len(lines), c
        assert c.source == "\n".join(lines[c.start_line - 1 : c.end_line]), c


# ---------------------------------------------------------------------------
# Fixture: well-formed sample covering most constructs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample() -> dict[str, Chunk]:
    return by_qualname(extract_file(FIXTURES / "sample.py"))


def test_sample_symbols_in_source_order() -> None:
    chunks = extract_file(FIXTURES / "sample.py")
    assert [c.qualname for c in chunks] == [
        "plain",
        "decorated",
        "fetch",
        "fetch.inner",
        "Base",
        "Base.class_attr",
        "Base.__init__",
        "Base.value",
        "Base.make",
        "Base.outer",
        "Base.outer.helper",
        "Base.outer.helper.deepest",
        "Base.outer.Local",
        "Base.outer.Local.local_method",
        "Base.Nested",
        "Base.Nested.nested_method",
        "maybe",
        "square",
        "one_liner",
    ]
    assert_source_consistent(chunks, (FIXTURES / "sample.py").read_text())


def test_plain_function(sample: dict[str, Chunk]) -> None:
    c = sample["plain"]
    assert (c.kind, c.parent, c.start_line, c.end_line) == ("function", None, 14, 19)
    assert c.docstring == "Add two numbers.\n\nReturns the sum."
    assert c.source.startswith("def plain(a: int, b: int = 2) -> int:")
    assert c.source.endswith("return a + b")
    assert c.path.endswith("sample.py")
    assert c.language == "python"
    assert not c.is_async


def test_decorators_included_in_span(sample: dict[str, Chunk]) -> None:
    c = sample["decorated"]
    assert c.start_line == 22  # first decorator line, not the `def` line
    assert c.source.startswith("@decorator\n@other.decorator(")
    assert c.decorators == ("@decorator", '@other.decorator( "arg", flag=True, )')


def test_async_function_and_nested_async(sample: dict[str, Chunk]) -> None:
    assert sample["fetch"].is_async
    assert sample["fetch"].docstring == "Fetch a URL."
    inner = sample["fetch.inner"]
    assert inner.is_async and inner.kind == "function" and inner.parent == "fetch"


def test_methods_and_parent(sample: dict[str, Chunk]) -> None:
    assert sample["Base"].kind == "class"
    assert sample["Base"].docstring == "A base class."
    for name in ("__init__", "value", "make", "outer"):
        c = sample[f"Base.{name}"]
        assert c.kind == "method" and c.parent == "Base"
    make = sample["Base.make"]
    assert make.is_async and make.decorators == ("@staticmethod",)
    assert make.start_line == 53 and make.source.lstrip().startswith("@staticmethod")


def test_nested_functions_and_classes(sample: dict[str, Chunk]) -> None:
    helper = sample["Base.outer.helper"]
    assert helper.kind == "function"  # a def inside a method is not a method
    assert helper.parent == "Base.outer"
    assert sample["Base.outer.helper.deepest"].parent == "Base.outer.helper"
    local = sample["Base.outer.Local"]
    assert local.kind == "class" and local.parent == "Base.outer"
    assert sample["Base.outer.Local.local_method"].kind == "method"
    nested = sample["Base.Nested.nested_method"]
    assert nested.kind == "method" and nested.docstring == "Nested."


def test_nested_chunk_ranges_inside_parent(sample: dict[str, Chunk]) -> None:
    for c in sample.values():
        if c.parent:
            p = sample[c.parent]
            assert p.start_line <= c.start_line <= c.end_line <= p.end_line


def test_definitions_inside_compound_statements(sample: dict[str, Chunk]) -> None:
    assert sample["maybe"].parent is None
    assert sample["maybe"].kind == "function"


def test_named_lambdas(sample: dict[str, Chunk]) -> None:
    assert sample["square"].kind == "lambda"
    assert sample["square"].source == "square = lambda n: n * n"
    assert sample["Base.class_attr"].parent == "Base"


def test_one_liner(sample: dict[str, Chunk]) -> None:
    c = sample["one_liner"]
    assert c.start_line == c.end_line == 84


def test_module_imports_normalized_and_shared(sample: dict[str, Chunk]) -> None:
    expected = (
        "from __future__ import annotations",
        "import os",
        "from typing import Any, TYPE_CHECKING",
        "from collections.abc import Iterator",
    )
    assert sample["plain"].imports == expected
    assert sample["Base.Nested.nested_method"].imports == expected


def test_local_imports_only_on_enclosing_chunks(sample: dict[str, Chunk]) -> None:
    assert "import json" in sample["fetch"].imports
    assert "import json" not in sample["fetch.inner"].imports
    assert "import json" not in sample["plain"].imports


# ---------------------------------------------------------------------------
# Docstrings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('"""Triple."""', "Triple."),
        ("'single'", "single"),
        ('r"""raw \\n kept"""', "raw \\n kept"),
        ('"a" "b"', "ab"),
        ('"""\n    Indented\n      more\n    """', "Indented\n  more"),
        ('f"not {a} docstring"', None),
        ('b"bytes"', None),
        ("x = 1", None),
        ("# comment\n    \"\"\"After comment.\"\"\"", "After comment."),
        ('"""first"""; x = 1', "first"),
    ],
)
def test_docstring_variants(body: str, expected: str | None) -> None:
    (c,) = extract_source(f"def f():\n    {body}\n", "m.py")
    assert c.docstring == expected


def test_docstring_not_taken_from_second_statement() -> None:
    (c,) = extract('''
        def f():
            x = 1
            """Not a docstring."""
    ''')
    assert c.docstring is None


# ---------------------------------------------------------------------------
# Empty / trivial inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", ["", "\n\n", "   \n\t\n", "# only a comment\n", "x = 1\nprint(x)\n"])
def test_no_symbols(code: str) -> None:
    assert extract_source(code, "m.py") == []


def test_lambda_inside_function_is_not_a_chunk() -> None:
    chunks = extract('''
        def f():
            g = lambda: 1
            return sorted([], key=lambda x: x)
    ''')
    assert [c.qualname for c in chunks] == ["f"]


def test_lambda_without_name_is_not_a_chunk() -> None:
    assert extract("handlers = [lambda: 1]\nobj.attr = lambda: 2\n") == []


def test_overloads_and_property_setters_all_kept() -> None:
    chunks = extract('''
        class C:
            @property
            def x(self): return 1
            @x.setter
            def x(self, v): pass
    ''')
    xs = [c for c in chunks if c.name == "x"]
    assert [c.decorators for c in xs] == [("@property",), ("@x.setter",)]


def test_decorated_class() -> None:
    (c,) = extract('''
        @dataclass(frozen=True)
        class Point:
            x: int
    ''')
    assert c.kind == "class" and c.decorators == ("@dataclass(frozen=True)",)
    assert c.start_line == 2 and c.end_line == 4


def test_comment_between_decorator_and_def() -> None:
    (c,) = extract('''
        @cache
        # explain
        def f():
            pass
    ''')
    assert c.start_line == 2 and c.decorators == ("@cache",)


def test_unicode_identifiers() -> None:
    (c,) = extract('''
        def größe(ä):
            """Ünïcödé."""
            return ä
    ''')
    assert c.name == "größe" and c.docstring == "Ünïcödé."


def test_match_statement_definitions() -> None:
    chunks = extract('''
        match cmd:
            case "x":
                def handler():
                    pass
    ''')
    assert [c.name for c in chunks] == ["handler"]


def test_deeply_nested_expression_does_not_recurse() -> None:
    code = "def f():\n    return " + " + ".join(["1"] * 20_000) + "\n"
    (c,) = extract_source(code, "m.py")
    assert c.name == "f"


def test_many_lines_and_wide_lines() -> None:
    # Rows/columns > 256 crashed py-tree-sitter 0.26.0 via Point.row; keep as a regression.
    code = "".join(f"def f{i}():\n    return {'1 + ' * 100}1\n\n" for i in range(400))
    for _ in range(3):
        chunks = extract_source(code, "m.py")
    assert len(chunks) == 400
    assert (chunks[-1].start_line, chunks[-1].end_line) == (1198, 1199)


def test_deeply_nested_definitions() -> None:
    depth = 60
    code = "".join(f"{'    ' * i}def f{i}():\n" for i in range(depth)) + "    " * depth + "pass\n"
    chunks = extract_source(code, "m.py")
    assert len(chunks) == depth
    assert chunks[-1].qualname == ".".join(f"f{i}" for i in range(depth))


# ---------------------------------------------------------------------------
# Syntax errors: extract what we can, never crash
# ---------------------------------------------------------------------------


def test_broken_method_does_not_swallow_rest_of_file() -> None:
    chunks = extract_file(FIXTURES / "broken.py")
    got = {c.qualname: c for c in chunks}
    assert list(got) == [
        "good_before",
        "Broken",
        "Broken.ok",
        "Broken.bad",
        "Broken.also_ok",
        "good_after",
        "good_async",
    ]
    assert got["Broken.ok"].kind == "method"
    assert (got["Broken.also_ok"].start_line, got["Broken.also_ok"].end_line) == (15, 17)
    assert got["Broken.also_ok"].docstring == "Still here."
    assert (got["good_after"].start_line, got["good_after"].end_line) == (20, 22)
    assert got["good_after"].docstring == "After the break."
    assert got["good_async"].is_async and got["good_async"].decorators == ("@decorator",)
    assert got["good_async"].start_line == 25
    assert got["Broken"].start_line == 8 and got["Broken"].end_line == 17
    assert all(c.imports == ("import os",) for c in chunks)
    assert_source_consistent(chunks, (FIXTURES / "broken.py").read_text())


def test_unclosed_bracket_at_module_level() -> None:
    chunks = extract('''
        CONFIG = {
            "a": 1,

        def after():
            return 1

        class K:
            def m(self):
                pass
    ''')
    assert {"after", "K", "K.m"} <= {c.qualname for c in chunks}


def test_unterminated_string_inside_function() -> None:
    chunks = extract('''
        def ok():
            return 1

        def bad():
            return "unterminated

        def later():
            return 2
    ''')
    names = [c.name for c in chunks]
    assert names[0] == "ok" and "bad" in names and names[-1] == "later"


def test_broken_header_still_yields_named_chunk() -> None:
    chunks = extract('''
        async def broken(a, b
            return a

        def fine():
            pass
    ''')
    got = by_qualname(chunks)
    assert got["broken"].is_async and got["broken"].kind == "function"
    assert got["fine"].start_line == 5


def test_broken_class_recovers_nested_class_methods() -> None:
    chunks = extract('''
        class Outer:
            x = (

            class Inner:
                def m(self):
                    pass
    ''')
    got = by_qualname(chunks)
    assert got["Outer.Inner.m"].kind == "method"


def test_dangling_decorator() -> None:
    chunks = extract('''
        @decorator

        def f():
            pass

        @lonely
    ''')
    assert "f" in {c.name for c in chunks}


@pytest.mark.parametrize(
    "code",
    [
        "def",
        "def (",
        "class :",
        "async",
        "@",
        "@\n@\n",
        "def f(:\n",
        ")))\n]]]\n",
        "def f():\n\treturn 1\n        return 2\n",  # inconsistent indentation
        "\x0c\ndef f():\n    pass\n",
        "class A:\n    def",
        "lambda: (",
    ],
)
def test_garbage_does_not_crash(code: str) -> None:
    chunks = extract_source(code, "m.py")
    assert_source_consistent(chunks, code)


def test_every_prefix_of_sample_is_safe() -> None:
    text = (FIXTURES / "sample.py").read_text()
    for cut in range(0, len(text), 7):
        prefix = text[:cut]
        assert_source_consistent(extract_source(prefix, "m.py"), prefix)


def test_random_line_deletions_are_safe() -> None:
    rng = random.Random(1234)
    lines = (FIXTURES / "sample.py").read_text().split("\n")
    for _ in range(150):
        kept = [line for line in lines if rng.random() > 0.15]
        text = "\n".join(kept)
        chunks = extract_source(text, "m.py")
        assert_source_consistent(chunks, text)


def test_random_bytes_are_safe() -> None:
    rng = random.Random(99)
    for _ in range(50):
        raw = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 400)))
        extract_source(raw, "m.py")


# ---------------------------------------------------------------------------
# Encodings and line endings
# ---------------------------------------------------------------------------


def test_latin1_coding_cookie() -> None:
    (c,) = extract_file(FIXTURES / "latin1.py")
    assert c.name == "café" and c.docstring == "Café au lait."


def test_invalid_utf8_without_cookie_is_replaced(tmp_path: Path) -> None:
    f = tmp_path / "bad.py"
    f.write_bytes(b'def f():\n    """caf\xe9"""\n    return 1\n')
    (c,) = extract_file(f)
    assert c.name == "f" and c.docstring == "caf\ufffd"


def test_utf8_bom(tmp_path: Path) -> None:
    f = tmp_path / "bom.py"
    f.write_bytes(b"\xef\xbb\xbfdef f():\n    pass\n")
    (c,) = extract_file(f)
    assert c.start_line == 1 and c.source.startswith("def f")


def test_bogus_coding_cookie_falls_back(tmp_path: Path) -> None:
    f = tmp_path / "cookie.py"
    f.write_bytes(b"# coding: no-such-codec\ndef f():\n    pass\n")
    assert [c.name for c in extract_file(f)] == ["f"]


@pytest.mark.parametrize("newline", ["\r\n", "\r"])
def test_non_unix_newlines(newline: str) -> None:
    code = newline.join(["def a():", "    pass", "", "def b():", "    pass", ""])
    chunks = extract_source(code, "m.py")
    assert [(c.name, c.start_line, c.end_line) for c in chunks] == [("a", 1, 2), ("b", 4, 5)]
    assert "\r" not in chunks[0].source


def test_unicode_line_separators_do_not_shift_lines() -> None:
    code = 'X = "a\u2028b"\ndef f():\n    pass\n'
    (c,) = extract_source(code, "m.py")
    assert (c.start_line, c.source) == (2, "def f():\n    pass")


# ---------------------------------------------------------------------------
# File-level handling
# ---------------------------------------------------------------------------


def test_huge_file_skipped(tmp_path: Path) -> None:
    f = tmp_path / "big.py"
    f.write_text("def f():\n    pass\n" + "# pad\n" * 1000)
    assert extract_file(f, max_bytes=100) == []
    assert len(extract_file(f)) == 1


def test_binary_file_skipped(tmp_path: Path) -> None:
    f = tmp_path / "bin.py"
    f.write_bytes(b"def f():\n    pass\n\x00\x01")
    assert extract_file(f) == []


def test_missing_file_and_directory(tmp_path: Path) -> None:
    assert extract_file(tmp_path / "nope.py") == []
    assert extract_file(tmp_path) == []


def test_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "empty.py"
    f.write_bytes(b"")
    assert extract_file(f) == []


def test_display_path(tmp_path: Path) -> None:
    f = tmp_path / "m.py"
    f.write_text("def f(): pass\n")
    (c,) = extract_file(f, display_path="pkg/m.py")
    assert c.path == "pkg/m.py"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unsupported")
def test_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "real.py"
    target.write_text("def f(): pass\n")
    link = tmp_path / "link.py"
    dangling = tmp_path / "dangling.py"
    try:
        link.symlink_to(target)
        dangling.symlink_to(tmp_path / "missing.py")
    except OSError:
        pytest.skip("cannot create symlinks on this filesystem")
    (c,) = extract_file(link)
    assert c.name == "f" and c.path == link.as_posix()
    assert extract_file(dangling) == []


def test_chunk_is_immutable() -> None:
    (c,) = extract("def f(): pass\n")
    with pytest.raises(AttributeError):
        c.name = "g"  # type: ignore[misc]
