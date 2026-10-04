"""Symbol-level chunk extraction for Python source using tree-sitter.

Each function, method, class, and named lambda becomes one ``Chunk`` carrying
its location, enclosing scope, docstring, relevant imports, and exact source.
Each file also gets one ``module`` chunk (named ``<module>``) holding the module
docstring, top-level imports, assignments, and import guards such as
``if TYPE_CHECKING:``. Its ``source`` contains only those statements, in order.

Files with syntax errors never raise. When the parse tree contains errors, the
file is split into segments at column-0 ``def``/``class``/decorator lines and
each segment is parsed on its own, so one broken definition cannot swallow the
rest of the file. Segments that still fail produce a best-effort chunk from the
header line, and their bodies are recursed into to recover nested symbols.
"""

from __future__ import annotations

import ast
import bisect
import inspect
import io
import logging
import re
import tokenize
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser, Tree

logger = logging.getLogger(__name__)

Kind = Literal["module", "function", "method", "class", "lambda"]
ScopeKind = Literal["module", "class", "function"]

DEFAULT_MAX_BYTES = 2_000_000
MODULE_NAME = "<module>"  # cannot collide with a Python identifier

_FUNCTION = "function_definition"
_CLASS = "class_definition"
_DECORATED = "decorated_definition"
_IMPORT_TYPES = frozenset({"import_statement", "import_from_statement", "future_import_statement"})
_STRING_TYPES = frozenset({"string", "concatenated_string"})
_GUARD_TYPES = frozenset({"if_statement", "try_statement"})

_HEADER_RE = re.compile(r"(async[ \t]+)?(def|class)[ \t]+([^\W\d]\w*)")
_LEADING_WS_RE = re.compile(r"[ \t\f]*")
_IMPORT_NOISE_RE = re.compile(r"#[^\n]*|[()\\]")


@dataclass(frozen=True, slots=True)
class Chunk:
    """One extracted symbol. Lines are 1-based and inclusive.

    ``spans`` lists the line ranges ``source`` was taken from when they are not
    contiguous (module chunks); otherwise it is empty and the span is
    ``(start_line, end_line)``.
    """

    path: str
    name: str
    kind: Kind
    start_line: int
    end_line: int
    parent: str | None
    docstring: str | None
    imports: tuple[str, ...]
    source: str
    language: str = "python"
    decorators: tuple[str, ...] = ()
    is_async: bool = False
    spans: tuple[tuple[int, int], ...] = ()

    @property
    def qualname(self) -> str:
        return f"{self.parent}.{self.name}" if self.parent else self.name

    def line_numbers(self) -> list[int]:
        """File line number of each line in ``source``."""
        spans = self.spans or ((self.start_line, self.end_line),)
        return [line for start, end in spans for line in range(start, end + 1)]


@dataclass(frozen=True, slots=True)
class _Scope:
    qualname: str | None
    kind: ScopeKind


@dataclass(frozen=True, slots=True)
class _FileContext:
    path: str
    lines: tuple[str, ...]
    module_imports: tuple[str, ...]
    local_import_rows: tuple[int, ...]  # 0-based rows, sorted
    local_import_texts: tuple[str, ...]


_MODULE_SCOPE = _Scope(None, "module")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_file(
    path: Path | str,
    *,
    display_path: str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> list[Chunk]:
    """Extract chunks from a Python file on disk.

    Unreadable, binary (NUL-containing), and oversized files yield no chunks.
    ``display_path`` overrides the path recorded on each chunk (e.g. a
    repo-relative path).
    """
    file_path = Path(path)
    label = display_path if display_path is not None else file_path.as_posix()
    try:
        size = file_path.stat().st_size
        if size > max_bytes:
            logger.info("skipping %s: %d bytes exceeds limit of %d", label, size, max_bytes)
            return []
        raw = file_path.read_bytes()
    except OSError as exc:
        logger.warning("skipping %s: %s", label, exc)
        return []
    if b"\x00" in raw:
        logger.info("skipping %s: looks binary", label)
        return []
    return extract_source(raw, label)


def extract_source(source: str | bytes, path: str) -> list[Chunk]:
    """Extract chunks from Python source text. Never raises on bad syntax."""
    text = decode_source(source) if isinstance(source, bytes) else source
    text = normalize_newlines(text)
    if not text.strip():
        return []

    lines = tuple(text.split("\n"))
    parser = _new_parser()
    tree = parser.parse(text.encode("utf-8"))
    ctx = _file_context(path, lines, tree.root_node)

    if not tree.root_node.has_error:
        chunks = _walk(tree.root_node, ctx, row_offset=0, scope=_MODULE_SCOPE, skip_errors=False)
        roots = [(tree.root_node, 0)]
    else:
        chunks = _extract_segmented(parser, ctx, 0, len(lines), indent=0, scope=_MODULE_SCOPE)
        roots = _segment_roots(parser, lines)
    module = _module_chunk(ctx, roots)
    if module is not None:
        chunks.append(module)
    return sorted(chunks, key=lambda c: (c.start_line, -c.end_line))


def decode_source(raw: bytes) -> str:
    """Decode Python source bytes honoring BOM / PEP 263 coding cookies.

    Falls back to UTF-8 with replacement characters so line structure and
    ASCII identifiers survive even in mis-encoded files.
    """
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
    except SyntaxError:
        encoding = "utf-8"
    try:
        text = raw.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        text = raw.decode("utf-8", errors="replace")
    return text.removeprefix("﻿")


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


@cache
def _language() -> Language:
    return Language(tspython.language())


def _new_parser() -> Parser:
    return Parser(_language())


def _parse(parser: Parser, text: str) -> Tree:
    return parser.parse(text.encode("utf-8"))


def _node_text(node: Node) -> str:
    return (node.text or b"").decode("utf-8", errors="replace")


def _collapse_ws(text: str) -> str:
    return " ".join(text.split())


# Points are read by tuple index, never ``.row``/``.column``: in py-tree-sitter
# 0.26.0 those attribute getters drop a reference on every access, which frees
# ints > 256 while still in use and corrupts the heap on any file > 256 lines.


def _start_row(node: Node) -> int:
    return node.start_point[0]


def _end_row(node: Node) -> int:
    """Last 0-based row actually occupied by ``node``."""
    end_row, end_column = node.end_point[0], node.end_point[1]
    if end_column == 0 and end_row > _start_row(node):
        return end_row - 1
    return end_row


def _children_reversed(node: Node) -> list[Node]:
    return list(reversed(node.children))


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


def _file_context(path: str, lines: tuple[str, ...], root: Node) -> _FileContext:
    module_imports: list[str] = []
    local: list[tuple[int, str]] = []
    stack: list[tuple[Node, bool]] = [(root, False)]
    while stack:
        node, in_def = stack.pop()
        if node.type in _IMPORT_TYPES:
            if node.has_error:
                continue
            text = _normalize_import(_node_text(node))
            if in_def:
                local.append((_start_row(node), text))
            else:
                module_imports.append(text)
            continue
        child_in_def = in_def or node.type in (_FUNCTION, _CLASS)
        stack.extend((child, child_in_def) for child in _children_reversed(node))
    local.sort(key=lambda item: item[0])
    return _FileContext(
        path=path,
        lines=lines,
        module_imports=tuple(dict.fromkeys(module_imports)),
        local_import_rows=tuple(row for row, _ in local),
        local_import_texts=tuple(text for _, text in local),
    )


def _normalize_import(text: str) -> str:
    """``from a import (\\n  b,  # c\\n)`` -> ``from a import b``.

    Import statements cannot contain string literals, so stripping comments,
    parentheses, and backslash continuations textually is safe.
    """
    text = _IMPORT_NOISE_RE.sub(" ", text)
    return _collapse_ws(text).rstrip(",").replace(" ,", ",")


def _imports_for(ctx: _FileContext, start_row: int, end_row: int) -> tuple[str, ...]:
    lo = bisect.bisect_left(ctx.local_import_rows, start_row)
    hi = bisect.bisect_right(ctx.local_import_rows, end_row)
    if lo == hi:
        return ctx.module_imports
    return tuple(dict.fromkeys((*ctx.module_imports, *ctx.local_import_texts[lo:hi])))


# ---------------------------------------------------------------------------
# Tree walk (clean or partially-clean trees)
# ---------------------------------------------------------------------------


def _walk(
    root: Node,
    ctx: _FileContext,
    *,
    row_offset: int,
    scope: _Scope,
    skip_errors: bool,
) -> list[Chunk]:
    """Collect chunks from ``root`` iteratively (deep expression trees are common)."""
    chunks: list[Chunk] = []
    stack: list[tuple[Node, _Scope]] = [(root, scope)]
    while stack:
        node, current = stack.pop()

        if node.type in (_FUNCTION, _CLASS, _DECORATED):
            definition, span = _unwrap_decorated(node)
            if definition is None or (skip_errors and span.has_error):
                continue
            chunk = _definition_chunk(definition, span, ctx, row_offset, current)
            if chunk is None:
                continue
            chunks.append(chunk)
            body = definition.child_by_field_name("body")
            if body is not None:
                child_kind: ScopeKind = "class" if chunk.kind == "class" else "function"
                stack.append((body, _Scope(chunk.qualname, child_kind)))
            continue

        if current.kind != "function" and node.type == "expression_statement":
            chunk = _lambda_chunk(node, ctx, row_offset, current)
            if chunk is not None:
                chunks.append(chunk)
                continue

        stack.extend((child, current) for child in _children_reversed(node))
    return chunks


def _unwrap_decorated(node: Node) -> tuple[Node | None, Node]:
    """Return (definition node, node whose span includes decorators)."""
    if node.type != _DECORATED:
        return node, node
    definition = node.child_by_field_name("definition")
    if definition is None or definition.type not in (_FUNCTION, _CLASS):
        return None, node
    return definition, node


def _definition_chunk(
    definition: Node,
    span: Node,
    ctx: _FileContext,
    row_offset: int,
    scope: _Scope,
) -> Chunk | None:
    name_node = definition.child_by_field_name("name")
    if name_node is None or name_node.is_missing:
        return None
    name = _node_text(name_node)
    if not name:
        return None

    kind: Kind
    if definition.type == _CLASS:
        kind = "class"
    else:
        kind = "method" if scope.kind == "class" else "function"

    start_row = _start_row(span) + row_offset
    end_row = _end_row(span) + row_offset
    decorators = tuple(
        _collapse_ws(_node_text(child)) for child in span.children if child.type == "decorator"
    )
    return Chunk(
        path=ctx.path,
        name=name,
        kind=kind,
        start_line=start_row + 1,
        end_line=end_row + 1,
        parent=scope.qualname,
        docstring=_docstring(definition),
        imports=_imports_for(ctx, start_row, end_row),
        source=_source(ctx, start_row, end_row),
        decorators=decorators,
        is_async=any(child.type == "async" for child in definition.children),
    )


def _lambda_chunk(
    statement: Node, ctx: _FileContext, row_offset: int, scope: _Scope
) -> Chunk | None:
    """Chunk ``name = lambda ...`` at module or class scope."""
    left = _named_lambda_target(statement)
    if left is None:
        return None
    start_row = _start_row(statement) + row_offset
    end_row = _end_row(statement) + row_offset
    return Chunk(
        path=ctx.path,
        name=_node_text(left),
        kind="lambda",
        start_line=start_row + 1,
        end_line=end_row + 1,
        parent=scope.qualname,
        docstring=None,
        imports=_imports_for(ctx, start_row, end_row),
        source=_source(ctx, start_row, end_row),
    )


def _named_lambda_target(statement: Node) -> Node | None:
    """The identifier in ``name = lambda ...``, else None."""
    if statement.type != "expression_statement" or statement.named_child_count != 1:
        return None
    assignment = statement.named_children[0]
    if assignment.type != "assignment":
        return None
    left = assignment.child_by_field_name("left")
    right = assignment.child_by_field_name("right")
    if left is None or right is None or left.type != "identifier" or right.type != "lambda":
        return None
    return left


def _docstring(definition: Node) -> str | None:
    body = definition.child_by_field_name("body")
    if body is None:
        return None
    first = next((c for c in body.named_children if c.type != "comment"), None)
    return None if first is None else _string_statement_value(first)


def _string_statement_value(statement: Node) -> str | None:
    """Value of a bare string-literal statement (a docstring candidate)."""
    if statement.type != "expression_statement" or statement.named_child_count != 1:
        return None
    literal = statement.named_children[0]
    if literal.type not in _STRING_TYPES:
        return None
    return _string_value(_node_text(literal))


def _string_value(literal: str) -> str | None:
    try:
        value = ast.literal_eval(literal)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None  # f-strings and malformed literals are not docstrings
    if not isinstance(value, str):
        return None
    return inspect.cleandoc(value)


def _source(ctx: _FileContext, start_row: int, end_row: int) -> str:
    return "\n".join(ctx.lines[start_row : end_row + 1])


# ---------------------------------------------------------------------------
# Module chunk
# ---------------------------------------------------------------------------


def _module_chunk(ctx: _FileContext, roots: list[tuple[Node, int]]) -> Chunk | None:
    """Collect module-level docstring, imports, assignments, and import guards.

    ``roots`` are (module node, row offset) pairs: the whole file when it parses
    cleanly, otherwise one per recovery segment. Statements with errors are
    skipped. Comment lines directly above an included statement come with it.
    """
    ranges: list[tuple[int, int]] = []
    docstring: str | None = None
    seen_statement = False
    for root, offset in roots:
        comments: list[Node] = []
        previous: Node | None = None
        for node in root.named_children:
            if node.type == "comment":
                trailing = previous is not None and _start_row(node) == _end_row(previous)
                if not trailing:
                    comments.append(node)
                previous = node
                continue
            previous = node
            if node.has_error:
                comments, seen_statement = [], True
                continue
            value = None if seen_statement else _string_statement_value(node)
            seen_statement = True
            if value is not None:
                docstring = value
            elif not _is_module_statement(node):
                comments = []
                continue
            start = _start_row(node)
            for comment in reversed(comments):
                if _end_row(comment) != start - 1:
                    break
                start = _start_row(comment)
            ranges.append((start + offset, _end_row(node) + offset))
            comments = []

    if not ranges:
        return None
    merged = _merge_ranges(ranges)
    return Chunk(
        path=ctx.path,
        name=MODULE_NAME,
        kind="module",
        start_line=merged[0][0] + 1,
        end_line=merged[-1][1] + 1,
        parent=None,
        docstring=docstring,
        imports=ctx.module_imports,
        source="\n".join(_source(ctx, start, end) for start, end in merged),
        spans=tuple((start + 1, end + 1) for start, end in merged) if len(merged) > 1 else (),
    )


def _is_module_statement(node: Node) -> bool:
    if node.type in _IMPORT_TYPES or node.type == "type_alias_statement":
        return True
    if node.type == "expression_statement":
        return _is_plain_assignment(node)
    if node.type in _GUARD_TYPES:
        return _is_import_guard(node)
    return False


def _is_plain_assignment(statement: Node) -> bool:
    if statement.named_child_count != 1 or _named_lambda_target(statement) is not None:
        return False
    return statement.named_children[0].type in ("assignment", "augmented_assignment")


def _is_import_guard(node: Node) -> bool:
    """An ``if``/``try`` whose every branch holds only imports and assignments."""
    blocks = [child for child in node.children if child.type == "block"]
    for clause in node.children:
        if clause.type.endswith("_clause"):
            blocks.extend(child for child in clause.children if child.type == "block")
    if not blocks:
        return False
    for block in blocks:
        for statement in block.named_children:
            if statement.type in ("comment", "pass_statement") or statement.type in _IMPORT_TYPES:
                continue
            if statement.type == "expression_statement" and _is_plain_assignment(statement):
                continue
            if statement.type in _GUARD_TYPES and _is_import_guard(statement):
                continue
            return False
    return True


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent inclusive row ranges."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _segment_roots(parser: Parser, lines: tuple[str, ...]) -> list[tuple[Node, int]]:
    """Parse each top-level recovery segment on its own (for broken files)."""
    return [
        (_parse(parser, "\n".join(lines[segment.start : segment.end])).root_node, segment.start)
        for segment in _split_segments(list(lines), 0)
    ]


# ---------------------------------------------------------------------------
# Error recovery: segment by column-0 definitions and parse each separately
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Segment:
    start: int  # absolute 0-based row, inclusive
    end: int  # absolute 0-based row, exclusive
    header: int | None  # absolute row of the def/class line, if any


def _extract_segmented(
    parser: Parser,
    ctx: _FileContext,
    start: int,
    end: int,
    *,
    indent: int,
    scope: _Scope,
) -> list[Chunk]:
    """Recover chunks from rows ``[start, end)`` whose code sits at ``indent``."""
    dedented = [_strip_indent(line, indent) for line in ctx.lines[start:end]]
    chunks: list[Chunk] = []
    for segment in _split_segments(dedented, start):
        text = "\n".join(dedented[segment.start - start : segment.end - start])
        tree = _parse(parser, text)
        if not tree.root_node.has_error:
            chunks.extend(
                _walk(tree.root_node, ctx, row_offset=segment.start, scope=scope, skip_errors=False)
            )
        elif segment.header is None:
            chunks.extend(
                _walk(tree.root_node, ctx, row_offset=segment.start, scope=scope, skip_errors=True)
            )
        else:
            chunks.extend(_recover_broken(parser, ctx, segment, dedented, start, indent, scope))
    return chunks


def _split_segments(dedented: list[str], base: int) -> list[_Segment]:
    """Split at column-0 headers; a run of decorators joins the header after it."""
    starts: list[tuple[int, int | None]] = []  # (row, header row)
    in_decorators = False
    for i, line in enumerate(dedented):
        row = base + i
        if line.startswith("@"):
            if not in_decorators:
                starts.append((row, None))
                in_decorators = True
        elif _HEADER_RE.match(line):
            if in_decorators:
                starts[-1] = (starts[-1][0], row)
                in_decorators = False
            else:
                starts.append((row, row))
        elif line and not line[0].isspace() and not line.startswith(("#", ")", "]", "}")):
            in_decorators = False

    end = base + len(dedented)
    if not starts or starts[0][0] != base:
        starts.insert(0, (base, None))
    return [
        _Segment(start=row, end=starts[i + 1][0] if i + 1 < len(starts) else end, header=header)
        for i, (row, header) in enumerate(starts)
    ]


def _recover_broken(
    parser: Parser,
    ctx: _FileContext,
    segment: _Segment,
    dedented: list[str],
    base: int,
    indent: int,
    scope: _Scope,
) -> list[Chunk]:
    """Best-effort chunk for a definition that does not parse, plus its children."""
    assert segment.header is not None
    match = _HEADER_RE.match(dedented[segment.header - base])
    assert match is not None
    is_async, keyword, name = bool(match.group(1)), match.group(2), match.group(3)

    kind: Kind
    if keyword == "class":
        kind = "class"
    else:
        kind = "method" if scope.kind == "class" else "function"

    last = _last_content_row(ctx.lines, segment.start, segment.end)
    decorators = tuple(
        _collapse_ws(dedented[row - base])
        for row in range(segment.start, segment.header)
        if dedented[row - base].startswith("@")
    )
    chunk = Chunk(
        path=ctx.path,
        name=name,
        kind=kind,
        start_line=segment.start + 1,
        end_line=last + 1,
        parent=scope.qualname,
        docstring=None,
        imports=_imports_for(ctx, segment.start, last),
        source=_source(ctx, segment.start, last),
        decorators=decorators,
        is_async=is_async,
    )

    body_start = segment.header + 1
    body_indent = _body_indent(ctx.lines, body_start, segment.end, indent)
    if body_indent is None:
        return [chunk]
    child_scope = _Scope(chunk.qualname, "class" if kind == "class" else "function")
    children = _extract_segmented(
        parser, ctx, body_start, segment.end, indent=body_indent, scope=child_scope
    )
    return [chunk, *children]


def _body_indent(lines: tuple[str, ...], start: int, end: int, outer: int) -> int | None:
    """Indent width of the first non-blank line in rows [start, end) if deeper than ``outer``."""
    for row in range(start, end):
        line = lines[row]
        if line.strip() and not line.lstrip().startswith("#"):
            width = _indent_width(line)
            return width if width > outer else None
    return None


def _last_content_row(lines: tuple[str, ...], start: int, end: int) -> int:
    for row in range(end - 1, start - 1, -1):
        if lines[row].strip():
            return row
    return start


def _indent_width(line: str) -> int:
    match = _LEADING_WS_RE.match(line)
    return match.end() if match else 0


def _strip_indent(line: str, width: int) -> str:
    """Remove up to ``width`` leading whitespace characters."""
    return line[min(width, _indent_width(line)) :]
