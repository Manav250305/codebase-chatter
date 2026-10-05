"""Call graph between chunks: extraction, resolution, storage (Phase A).

Pass 1 (``collect_file_refs``) walks one file's tree-sitter parse and records
raw references, each attributed to the innermost chunk whose lines contain
it: call sites (the callee expression), callables passed as arguments
(callback references), ``with`` items, class bases, imports, and the names a
function binds locally (parameters, assignment targets), which shadow outer
names.

Pass 2 (``build_graph``) resolves references against a repo-wide symbol
table and produces typed edges between chunk indices.

Resolution rules (Phase A):
    bare names       innermost function scope outwards (nested defs, local
                     imports; parameters/assignments stop resolution), then
                     module definitions and imports; class scope is skipped
                     as in Python
    imports          ``import a.b [as z]``, ``from [.]a import b [as z]``,
                     relative imports, package re-exports, ``import *``
    self / cls       the enclosing method's class, then its MRO (C3 over
                     repo classes; external bases end the walk)
    super()          the MRO after the enclosing class
    attribute chains through modules and classes (``mod.f``, ``Cls.m``)
    instantiation    ``Cls(...)``: instantiates Cls, calls Cls.__init__ via MRO
    with             ``with Cls(...)``: Cls.__enter__/__exit__ via MRO
                     (``async with``: __aenter__/__aexit__)
    callbacks        a function/method/class passed as an argument value
    contains         parent chunk -> nested chunk
    inherits         class -> resolved base class

Anything else (local variables used as receivers, instance attributes,
subscripts, call results, builtins, external libraries) is dropped and
counted by reason. Edges whose source chunk is in a test file get type
``tested_by``: stored, never used for expansion.
"""

from __future__ import annotations

import builtins
import json
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from chatter.extract import Chunk, normalize_newlines

GRAPH_VERSION = 2  # resolver version; 2: finer drop reasons (literal/subscript/external base)

# Edge type codes (stable: stored in graph_types.npy).
CALLS = 1
INSTANTIATES = 2
REFERENCES = 3
CONTEXT_MANAGER = 4
CONTAINS = 5
INHERITS = 6
TESTED_BY = 7
EDGE_TYPES = {
    CALLS: "calls",
    INSTANTIATES: "instantiates",
    REFERENCES: "references",
    CONTEXT_MANAGER: "context_manager",
    CONTAINS: "contains",
    INHERITS: "inherits",
    TESTED_BY: "tested_by",
}
# Expansion weights by edge type, (forward, reverse). TESTED_BY is never expanded.
EDGE_WEIGHTS: dict[int, tuple[float, float]] = {
    CALLS: (1.0, 1.0),
    INSTANTIATES: (1.0, 1.0),
    REFERENCES: (1.0, 1.0),
    CONTEXT_MANAGER: (1.0, 1.0),
    CONTAINS: (0.7, 0.5),
    INHERITS: (0.5, 0.5),
}

INDPTR_FILE = "graph_indptr.npy"
TARGETS_FILE = "graph_targets.npy"
TYPES_FILE = "graph_types.npy"
META_FILE = "graph.json"
GRAPH_FILES = (INDPTR_FILE, TARGETS_FILE, TYPES_FILE, META_FILE)

_BUILTINS = frozenset(dir(builtins))
_CLASS_SCOPE = -1  # names bound in a class body (not visible to its methods)
_EXAMPLES_PER_REASON = 8
_FUNCTION_KINDS = frozenset({"function", "method", "lambda"})


def is_test_path(path: str) -> bool:
    """Test code: under tests/ or test/, test_*.py, *_test.py, or conftest.py."""
    p = PurePosixPath(path)
    return (
        any(part in ("tests", "test") for part in p.parts[:-1])
        or p.name.startswith("test_")
        or p.name.endswith("_test.py")
        or p.name == "conftest.py"
    )


def module_name(path: str) -> str:
    """Dotted module for a repo-relative path; ``src/`` is a source root."""
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if len(parts) > 1 and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


# ---------------------------------------------------------------------------
# Pass 1: raw references per file
# ---------------------------------------------------------------------------

Expr = tuple[str, ...]  # ("name", "attr", ...); first item may be "super()"


@dataclass(frozen=True, slots=True)
class Ref:
    kind: str  # call | reference | with | async_with | base
    source: int  # chunk index
    expr: Expr | None  # None: not a resolvable shape (subscript, call result, ...)
    line: int
    text: str  # for examples in stats
    shape: str = ""  # why expr is None: subscript | call_result | literal_receiver | dynamic


@dataclass(frozen=True, slots=True)
class Import:
    scope: int | None  # chunk index of the binding function; None = module scope
    name: str  # local name bound
    target: str  # absolute dotted name, or "" if unresolvable (relative past root)
    from_import: bool  # from X import name (target = X.name)
    wildcard: bool = False  # from X import * (target = X)


@dataclass(slots=True)
class FileRefs:
    path: str
    module: str
    is_package: bool
    chunk_indices: list[int]
    refs: list[Ref] = field(default_factory=list)
    imports: list[Import] = field(default_factory=list)
    local_names: dict[int | None, set[str]] = field(default_factory=lambda: defaultdict(set))


def _parser() -> Parser:
    return Parser(Language(tspython.language()))


def collect_file_refs(
    text: str,
    path: str,
    chunks: Sequence[Chunk],
    chunk_indices: Sequence[int],
) -> FileRefs:
    """Raw references in one file, attributed to the innermost containing chunk."""
    lines_to_chunk = _innermost_chunk_by_line(chunks, chunk_indices)
    module_chunk = next(
        (idx for chunk, idx in zip(chunks, chunk_indices) if chunk.kind == "module"), None
    )
    scope_kinds = {idx: chunk.kind for chunk, idx in zip(chunks, chunk_indices)}

    def owner(line: int) -> int | None:
        return lines_to_chunk.get(line, module_chunk)

    def scope(line: int) -> int | None:
        """Function chunk owning names bound at ``line``; None = module, _CLASS_SCOPE = ignore."""
        idx = lines_to_chunk.get(line)
        if idx is None:
            return None
        return idx if scope_kinds.get(idx) in _FUNCTION_KINDS else _CLASS_SCOPE

    def bind(line: int, names: Iterator[str]) -> None:
        owner_scope = scope(line)
        if owner_scope != _CLASS_SCOPE:  # class bodies don't leak into method scopes
            result.local_names[owner_scope].update(names)

    module = module_name(path)
    result = FileRefs(
        path=path,
        module=module,
        is_package=PurePosixPath(path).name == "__init__.py",
        chunk_indices=list(chunk_indices),
    )
    tree = _parser().parse(normalize_newlines(text).encode("utf-8"))
    stack: list[Node] = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "ERROR" or node.is_missing:
            continue
        line = node.start_point[0] + 1
        kind = node.type
        if kind == "call":
            source = owner(line)
            function = node.child_by_field_name("function")
            if source is not None and function is not None:
                expr = _expr(function)
                shape = "" if expr is not None else _shape(function)
                result.refs.append(Ref("call", source, expr, line, _text(function), shape))
                arguments = node.child_by_field_name("arguments")
                if arguments is not None:
                    for arg in arguments.named_children:
                        value = arg.child_by_field_name("value") if arg.type == "keyword_argument" else arg
                        if value is not None and value.type in ("identifier", "attribute"):
                            expr = _expr(value)
                            if expr is not None:
                                result.refs.append(Ref("reference", source, expr, value.start_point[0] + 1, _text(value)))
        elif kind == "with_statement":
            is_async = any(child.type == "async" for child in node.children)
            source = owner(line)
            for item in _with_items(node):
                value = item.child_by_field_name("value")
                if value is not None and value.type == "as_pattern":
                    value = value.named_children[0] if value.named_children else None
                    target = item.child_by_field_name("value")
                    alias = target.child_by_field_name("alias") if target is not None else None
                    if alias is not None:
                        bind(line, _identifiers(alias))
                if value is not None and value.type == "call" and source is not None:
                    function = value.child_by_field_name("function")
                    if function is not None:
                        result.refs.append(
                            Ref("async_with" if is_async else "with", source, _expr(function), line, _text(function))
                        )
        elif kind == "class_definition":
            name = node.child_by_field_name("name")
            source = lines_to_chunk.get(name.start_point[0] + 1) if name is not None else None
            bases = node.child_by_field_name("superclasses")
            if source is not None and bases is not None:
                for base in bases.named_children:
                    if base.type in ("identifier", "attribute"):
                        result.refs.append(Ref("base", source, _expr(base), base.start_point[0] + 1, _text(base)))
        elif kind == "import_statement" and scope(line) != _CLASS_SCOPE:
            for name_node in node.children_by_field_name("name"):
                _record_import(result, scope(line), name_node, None)
        elif kind == "import_from_statement" and scope(line) != _CLASS_SCOPE:
            module_node = node.child_by_field_name("module_name")
            base, level = _import_base(module_node)
            if any(child.type == "wildcard_import" for child in node.children):
                target = _absolute(module, result.is_package, base, level)
                result.imports.append(Import(scope(line), "*", target, True, wildcard=True))
            for name_node in node.children_by_field_name("name"):
                _record_import(result, scope(line), name_node, (base, level))
        elif kind in ("parameters", "lambda_parameters"):
            bind(line, _parameter_names(node))
        elif kind in ("assignment", "augmented_assignment", "for_statement", "for_in_clause"):
            left = node.child_by_field_name("left")
            if left is not None:
                bind(line, _identifiers(left))
        stack.extend(reversed(node.children))
    return result


def _record_import(
    result: FileRefs, scope: int | None, node: Node, from_base: tuple[str, int] | None
) -> None:
    if node.type == "aliased_import":
        dotted = node.child_by_field_name("name")
        alias = node.child_by_field_name("alias")
        name = _text(dotted) if dotted is not None else ""
        bound = _text(alias) if alias is not None else name
        explicit_alias = True
    else:
        name = _text(node)
        bound = name.split(".")[0]
        explicit_alias = False
    if not name:
        return
    if from_base is None:  # import a.b [as z]
        target = name if explicit_alias else name.split(".")[0]
        result.imports.append(Import(scope, bound, target, False))
    else:  # from base import name [as z]
        base, level = from_base
        module = _absolute(result.module, result.is_package, base, level)
        target = f"{module}.{name}" if module else ""
        result.imports.append(Import(scope, bound, target, True))


def _import_base(node: Node | None) -> tuple[str, int]:
    if node is None:
        return "", 0
    if node.type == "relative_import":
        prefix = next((c for c in node.children if c.type == "import_prefix"), None)
        dotted = next((c for c in node.children if c.type == "dotted_name"), None)
        return (_text(dotted) if dotted is not None else ""), len(_text(prefix)) if prefix else 0
    return _text(node), 0


def _absolute(module: str, is_package: bool, base: str, level: int) -> str:
    """Resolve a (possibly relative) import base to an absolute dotted module."""
    if level == 0:
        return base
    parts = module.split(".") if module else []
    package = parts if is_package else parts[:-1]
    if level - 1 > len(package):
        return ""
    package = package[: len(package) - (level - 1)]
    return ".".join([*package, *([base] if base else [])])


def _expr(node: Node) -> Expr | None:
    """Flatten ``a.b.c`` / ``super().m`` into a tuple; None for other shapes."""
    names: list[str] = []
    while node.type == "attribute":
        attribute = node.child_by_field_name("attribute")
        obj = node.child_by_field_name("object")
        if attribute is None or obj is None:
            return None
        names.append(_text(attribute))
        node = obj
    if node.type == "identifier":
        names.append(_text(node))
    elif node.type == "call":
        function = node.child_by_field_name("function")
        if function is None or function.type != "identifier" or _text(function) != "super":
            return None
        names.append("super()")
    else:
        return None
    return tuple(reversed(names))


_LITERALS = frozenset(
    {"string", "concatenated_string", "integer", "float", "list", "dictionary", "tuple", "set",
     "list_comprehension", "dictionary_comprehension", "set_comprehension", "generator_expression",
     "true", "false", "none"}
)


def _shape(node: Node) -> str:
    """Why a callee is not a name chain: what its innermost receiver is."""
    while node.type == "attribute":
        obj = node.child_by_field_name("object")
        if obj is None:
            break
        node = obj
    if node.type == "subscript":
        return "subscript"
    if node.type == "call":
        return "call_result"
    if node.type in _LITERALS:
        return "literal_receiver"
    return "dynamic"


def _with_items(node: Node) -> Iterator[Node]:
    for child in node.children:
        if child.type == "with_clause":
            yield from (c for c in child.named_children if c.type == "with_item")


def _identifiers(node: Node) -> Iterator[str]:
    """Names bound by a target (identifier, tuple/list patterns); not attributes."""
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type == "identifier":
            yield _text(current)
        elif current.type in ("pattern_list", "tuple_pattern", "list_pattern", "as_pattern_target", "tuple", "list", "list_splat_pattern"):
            stack.extend(current.named_children)


def _parameter_names(node: Node) -> Iterator[str]:
    for param in node.named_children:
        if param.type == "identifier":
            yield _text(param)
        else:
            name = param.child_by_field_name("name")
            if name is not None and name.type == "identifier":
                yield _text(name)
            else:
                yield from (_text(c) for c in param.named_children if c.type == "identifier")


def _text(node: Node | None) -> str:
    return (node.text or b"").decode("utf-8", errors="replace") if node is not None else ""


def _innermost_chunk_by_line(chunks: Sequence[Chunk], indices: Sequence[int]) -> dict[int, int]:
    """Line -> index of the smallest non-module chunk covering it."""
    best: dict[int, tuple[int, int]] = {}
    for chunk, idx in zip(chunks, indices):
        if chunk.kind == "module":
            continue
        size = chunk.end_line - chunk.start_line
        for line in range(chunk.start_line, chunk.end_line + 1):
            current = best.get(line)
            if current is None or size < current[0]:
                best[line] = (size, idx)
    return {line: idx for line, (_, idx) in best.items()}


# ---------------------------------------------------------------------------
# Pass 2: resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Value:
    kind: str  # module | chunk | external | stop
    name: str = ""  # module name, external dotted name, or stop reason
    index: int = -1  # chunk index for kind == "chunk"


@dataclass(slots=True)
class Graph:
    sources: np.ndarray
    targets: np.ndarray
    types: np.ndarray
    stats: dict[str, Any]


class _Resolver:
    def __init__(self, chunks: Sequence[Chunk], ids: Sequence[str], files: Sequence[FileRefs]) -> None:
        self.chunks = chunks
        self.ids = ids
        self.index_of = {cid: i for i, cid in enumerate(ids)}
        self.files = {f.module: f for f in files}
        self.file_of_chunk: dict[int, FileRefs] = {i: f for f in files for i in f.chunk_indices}
        # Symbols: module -> name -> chunk (last definition wins, as at runtime).
        self.top: dict[str, dict[str, int]] = defaultdict(dict)
        self.members: dict[int, dict[str, int]] = defaultdict(dict)  # class/function -> child name -> chunk
        self.by_qualname: dict[tuple[str, str], int] = {}
        for i, chunk in enumerate(chunks):
            if chunk.kind == "module":
                continue
            module = module_name(chunk.path)
            self.by_qualname[(chunk.path, chunk.qualname)] = i
            if chunk.parent is None:
                self.top[module][chunk.name] = i
        for i, chunk in enumerate(chunks):
            if chunk.kind != "module" and chunk.parent is not None:
                parent = self.by_qualname.get((chunk.path, chunk.parent))
                if parent is not None:
                    self.members[parent][chunk.name] = i
        self.module_imports: dict[str, dict[str, Import]] = defaultdict(dict)
        self.local_imports: dict[int, dict[str, Import]] = defaultdict(dict)
        self.wildcards: dict[str, list[str]] = defaultdict(list)
        self.local_names: dict[tuple[str, int | None], set[str]] = {}
        for f in files:
            for imp in f.imports:
                if imp.wildcard:
                    self.wildcards[f.module].append(imp.target)
                elif imp.scope is None:
                    self.module_imports[f.module][imp.name] = imp
                else:
                    self.local_imports[imp.scope][imp.name] = imp
            for scope, names in f.local_names.items():
                self.local_names[(f.module, scope)] = names
        self.bases: dict[int, list[int]] = defaultdict(list)
        self.external_bases: set[int] = set()  # classes with a base outside the repo
        self._mro_cache: dict[int, list[int]] = {}

    # --- values ---------------------------------------------------------

    def module_member(self, module: str, name: str, depth: int = 0) -> Value:
        sub = f"{module}.{name}" if module else name
        if sub in self.files:
            return Value("module", sub)
        if name in self.top.get(module, {}):
            return Value("chunk", index=self.top[module][name])
        imp = self.module_imports.get(module, {}).get(name)
        if imp is not None and depth < 8:  # re-export
            return self.import_value(imp, depth + 1)
        for star in self.wildcards.get(module, []):
            if depth < 8 and star in self.files:
                value = self.module_member(star, name, depth + 1)
                if value.kind != "stop":
                    return value
        if module in self.files:
            return Value("stop", "module_attr_unknown")
        return Value("external", sub)

    def import_value(self, imp: Import, depth: int = 0) -> Value:
        if not imp.target:
            return Value("stop", "relative_import_outside_repo")
        if not imp.from_import:
            return Value("module", imp.target) if imp.target in self.files else Value("external", imp.target)
        module, _, name = imp.target.rpartition(".")
        if imp.target in self.files:
            return Value("module", imp.target)
        if module in self.files:
            return self.module_member(module, name, depth)
        return Value("external", imp.target)

    def scope_chain(self, index: int) -> list[int]:
        """Function chunks from ``index`` outwards (classes skipped, as in Python)."""
        chain = []
        current: int | None = index
        while current is not None:
            chunk = self.chunks[current]
            if chunk.kind in _FUNCTION_KINDS:
                chain.append(current)
            current = self.by_qualname.get((chunk.path, chunk.parent)) if chunk.parent else None
        return chain

    def lookup(self, name: str, source: int) -> tuple[Value, str]:
        chunk = self.chunks[source]
        module = module_name(chunk.path)
        for scope in self.scope_chain(source):
            if name in self.members.get(scope, {}):
                return Value("chunk", index=self.members[scope][name]), "local_def"
            imp = self.local_imports.get(scope, {}).get(name)
            if imp is not None:
                return self.import_value(imp), "local_import"
            if name in self.local_names.get((module, scope), set()):
                return Value("stop", "local_variable"), ""
        if name in self.top.get(module, {}):
            return Value("chunk", index=self.top[module][name]), "module_def"
        imp = self.module_imports.get(module, {}).get(name)
        if imp is not None:
            return self.import_value(imp), "import"
        if name in self.local_names.get((module, None), set()):
            return Value("stop", "module_variable"), ""
        for star in self.wildcards.get(module, []):
            if star in self.files:
                value = self.module_member(star, name)
                if value.kind == "chunk":
                    return value, "wildcard_import"
        if name in _BUILTINS:
            return Value("stop", "builtin"), ""
        return Value("stop", "unknown_name"), ""

    def enclosing_class(self, source: int) -> int | None:
        """Class of the nearest enclosing method (``self``/``cls``/``super()`` owner)."""
        current: int | None = source
        while current is not None:
            chunk = self.chunks[current]
            if chunk.kind == "method":
                return self.by_qualname.get((chunk.path, chunk.parent)) if chunk.parent else None
            current = self.by_qualname.get((chunk.path, chunk.parent)) if chunk.parent else None
        return None

    def mro(self, cls: int) -> list[int]:
        """C3 linearization over repo classes (falls back to DFS on conflicts)."""
        if cls in self._mro_cache:
            return self._mro_cache[cls]
        self._mro_cache[cls] = [cls]  # cycle guard
        sequences = [list(self.mro(base)) for base in self.bases.get(cls, [])]
        sequences.append(list(self.bases.get(cls, [])))
        result = [cls]
        while any(sequences):
            for seq in sequences:
                if not seq:
                    continue
                head = seq[0]
                if not any(head in other[1:] for other in sequences):
                    break
            else:  # inconsistent hierarchy: depth-first order instead
                rest = [c for seq in sequences for c in seq if c not in result]
                result += list(dict.fromkeys(rest))
                break
            result.append(head)
            sequences = [[c for c in seq if c != head] for seq in sequences]
        self._mro_cache[cls] = result
        return result

    def class_attr(self, cls: int, name: str, skip_self: bool = False) -> int | None:
        for klass in self.mro(cls)[1 if skip_self else 0 :]:
            member = self.members.get(klass, {}).get(name)
            if member is not None:
                return member
        return None

    def resolve(self, expr: Expr | None, source: int) -> tuple[Value, str]:
        """Resolve an expression to a value and the rule that found it."""
        if expr is None:
            return Value("stop", "dynamic"), ""
        head, rest = expr[0], expr[1:]
        if head in ("self", "cls") or head == "super()":
            cls = self.enclosing_class(source)
            if cls is None:
                return Value("stop", "self_outside_method"), ""
            if not rest:
                return Value("chunk", index=cls), "self_mro"
            member = self.class_attr(cls, rest[0], skip_self=head == "super()")
            if member is None:
                if any(klass in self.external_bases for klass in self.mro(cls)):
                    return Value("stop", "inherited_from_external_base"), ""
                return Value("stop", "instance_attribute"), ""
            value, rule = Value("chunk", index=member), "super_mro" if head == "super()" else "self_mro"
            rest = rest[1:]
        else:
            value, rule = self.lookup(head, source)
        for name in rest:
            if value.kind == "module":
                value, rule = self.module_member(value.name, name), "module_attr"
            elif value.kind == "chunk" and self.chunks[value.index].kind == "class":
                member = self.class_attr(value.index, name)
                value = Value("chunk", index=member) if member is not None else Value("stop", "class_attr_unknown")
                rule = "class_attr"
            elif value.kind == "chunk":
                return Value("stop", "function_attribute"), ""
            elif value.kind == "external":
                return value, ""
            else:
                stop = value.name
                if stop in ("local_variable", "module_variable"):
                    stop = "untyped_receiver" if stop == "local_variable" else "module_variable_receiver"
                return Value("stop", stop), ""
        return value, rule


def build_graph(chunks: Sequence[Chunk], ids: Sequence[str], files: Sequence[FileRefs]) -> Graph:
    """Resolve all references into deduplicated, sorted typed edges plus stats."""
    resolver = _Resolver(chunks, ids, files)
    stats: dict[str, Any] = {
        "version": GRAPH_VERSION,
        "references": Counter(),
        "resolved_by_rule": Counter(),
        "dropped_by_reason": Counter(),
        "dropped_examples": defaultdict(list),
    }
    edges: set[tuple[int, int, int]] = set()

    def add(src: int, dst: int, kind: int) -> None:
        if src != dst:
            edges.add((src, dst, kind))

    # Bases first: the MRO needs them.
    all_refs = [ref for f in files for ref in f.refs]
    for ref in all_refs:
        if ref.kind == "base":
            value, _ = resolver.resolve(ref.expr, ref.source)
            if value.kind == "chunk" and chunks[value.index].kind == "class":
                resolver.bases[ref.source].append(value.index)
            else:
                resolver.external_bases.add(ref.source)
    resolver._mro_cache.clear()

    for ref in all_refs:
        counted = ref.kind != "reference"  # unresolved argument values are usually data
        if counted:
            stats["references"][ref.kind] += 1
        value, rule = resolver.resolve(ref.expr, ref.source)
        target = value.index if value.kind == "chunk" else None
        target_kind = chunks[target].kind if target is not None else None
        if ref.kind == "base":
            if target_kind == "class":
                add(ref.source, target, INHERITS)
                stats["resolved_by_rule"]["base:" + rule] += 1
            else:
                _drop(stats, ref, value, chunks)
        elif ref.kind == "call":
            if target_kind == "class":
                add(ref.source, target, INSTANTIATES)
                init = resolver.class_attr(target, "__init__")
                if init is not None:
                    add(ref.source, init, CALLS)
                stats["resolved_by_rule"]["instantiation"] += 1
            elif target_kind in _FUNCTION_KINDS:
                add(ref.source, target, CALLS)
                stats["resolved_by_rule"][rule or "call"] += 1
            else:
                _drop(stats, ref, value, chunks)
        elif ref.kind in ("with", "async_with"):
            if target_kind == "class":
                enter, exit_ = ("__aenter__", "__aexit__") if ref.kind == "async_with" else ("__enter__", "__exit__")
                found = False
                for dunder in (enter, exit_):
                    method = resolver.class_attr(target, dunder)
                    if method is not None:
                        add(ref.source, method, CONTEXT_MANAGER)
                        found = True
                if found:
                    stats["resolved_by_rule"]["context_manager"] += 1
                else:
                    _drop(stats, ref, Value("stop", "context_manager_not_in_repo"), chunks)
            else:
                _drop(stats, ref, value, chunks)
        elif ref.kind == "reference" and target_kind in (*_FUNCTION_KINDS, "class"):
            add(ref.source, target, REFERENCES)
            stats["references"]["reference"] += 1
            stats["resolved_by_rule"]["callback_reference"] += 1

    for i, chunk in enumerate(chunks):
        if chunk.kind != "module" and chunk.parent is not None:
            parent = resolver.by_qualname.get((chunk.path, chunk.parent))
            if parent is not None:
                add(parent, i, CONTAINS)

    tests = np.array([is_test_path(chunk.path) for chunk in chunks], dtype=bool)
    typed = sorted({(s, d, TESTED_BY if tests[s] else k) for s, d, k in edges})
    array = np.array(typed, dtype=np.int64).reshape(-1, 3)
    stats["edges_by_type"] = Counter(EDGE_TYPES[int(k)] for k in array[:, 2])
    return Graph(
        sources=array[:, 0].astype(np.int32),
        targets=array[:, 1].astype(np.int32),
        types=array[:, 2].astype(np.uint8),
        stats=_plain(stats),
    )


def _drop(stats: dict[str, Any], ref: Ref, value: Value, chunks: Sequence[Chunk]) -> None:
    if ref.expr is None and ref.shape:
        reason = ref.shape
    elif value.kind == "external":
        reason = "external"
    elif value.kind == "chunk":
        reason = f"not_callable_{chunks[value.index].kind}"
    else:
        reason = value.name or "unresolved"
    reason = f"{ref.kind}:{reason}"
    stats["dropped_by_reason"][reason] += 1
    examples = stats["dropped_examples"][reason]
    if len(examples) < _EXAMPLES_PER_REASON:
        examples.append(f"{chunks[ref.source].path}:{ref.line} {ref.text}")


def _plain(stats: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in stats.items():
        if isinstance(value, Counter):
            out[key] = dict(sorted(value.items(), key=lambda kv: (-kv[1], kv[0])))
        elif isinstance(value, defaultdict):
            out[key] = dict(sorted(value.items()))
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def write_graph(directory: Path, graph: Graph, n_chunks: int) -> None:
    """CSR adjacency over chunk indices (record order) plus stats."""
    counts = np.bincount(graph.sources, minlength=n_chunks) if len(graph.sources) else np.zeros(n_chunks, dtype=np.int64)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    np.save(directory / INDPTR_FILE, indptr)
    np.save(directory / TARGETS_FILE, graph.targets.astype(np.int32))
    np.save(directory / TYPES_FILE, graph.types.astype(np.uint8))
    meta = {"types": {str(k): v for k, v in EDGE_TYPES.items()}, "stats": graph.stats}
    (directory / META_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_graph_stats(directory: Path) -> dict[str, Any] | None:
    try:
        return json.loads((Path(directory) / META_FILE).read_text(encoding="utf-8"))["stats"]
    except (OSError, ValueError, KeyError):
        return None


class GraphIndex:
    """Read-only typed adjacency in both directions, for expansion."""

    def __init__(self, indptr: np.ndarray, targets: np.ndarray, types: np.ndarray) -> None:
        n = len(indptr) - 1
        self.n = n
        self._indptr = np.asarray(indptr)
        self._targets = np.asarray(targets)
        self._types = np.asarray(types)
        sources = np.repeat(np.arange(n, dtype=np.int64), np.diff(self._indptr))
        order = np.lexsort((sources, self._targets))  # by target, then source: deterministic
        self._rev_sources = sources[order]
        self._rev_types = self._types[order]
        rev_counts = np.bincount(self._targets, minlength=n) if len(self._targets) else np.zeros(n, dtype=np.int64)
        self._rev_indptr = np.concatenate([[0], np.cumsum(rev_counts)]).astype(np.int64)
        expandable = np.isin(self._types, list(EDGE_WEIGHTS))
        self.in_degree = (
            np.bincount(self._targets[expandable], minlength=n) if expandable.any() else np.zeros(n, dtype=np.int64)
        )

    @classmethod
    def load(cls, directory: Path) -> GraphIndex | None:
        directory = Path(directory)
        if not all((directory / name).exists() for name in GRAPH_FILES):
            return None
        return cls(
            np.load(directory / INDPTR_FILE),
            np.load(directory / TARGETS_FILE),
            np.load(directory / TYPES_FILE),
        )

    def neighbors(self, node: int) -> dict[int, float]:
        """Expandable neighbours in both directions with their best edge weight."""
        weights: dict[int, float] = {}
        start, end = self._indptr[node], self._indptr[node + 1]
        for target, kind in zip(self._targets[start:end].tolist(), self._types[start:end].tolist()):
            if kind in EDGE_WEIGHTS:
                weights[target] = max(weights.get(target, 0.0), EDGE_WEIGHTS[kind][0])
        start, end = self._rev_indptr[node], self._rev_indptr[node + 1]
        for source, kind in zip(self._rev_sources[start:end].tolist(), self._rev_types[start:end].tolist()):
            if kind in EDGE_WEIGHTS:
                weights[source] = max(weights.get(source, 0.0), EDGE_WEIGHTS[kind][1])
        weights.pop(node, None)
        return weights

    def edges_from(self, node: int) -> list[tuple[int, str]]:
        start, end = self._indptr[node], self._indptr[node + 1]
        return [
            (int(t), EDGE_TYPES[int(k)])
            for t, k in zip(self._targets[start:end].tolist(), self._types[start:end].tolist())
        ]
