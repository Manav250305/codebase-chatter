# Codebase Chatter

Local tool that indexes a code repository with code-aware chunking and answers
questions about functions/modules, with `file:line-range` citations.

## Stack
- Python 3.11+, tree-sitter (py-tree-sitter + tree-sitter-python)
- Embeddings: sentence-transformers (local); store: Chroma; keyword: rank-bm25
- CLI: typer; tests: pytest

## Conventions
- Type hints everywhere; small pure functions; no global state
- Chunk by symbol (function/method/class), never fixed windows
- Every chunk carries: file path, start/end line, language, parent class,
  docstring, imports
- Handle corner cases: empty files, syntax errors, non-UTF8 files, huge files,
  nested functions/classes, decorators, async defs, lambdas, symlinks
- Must be efficient on large repos: stream files, batch embeddings, skip
  .git/node_modules/.venv, respect .gitignore
- Write tests alongside each module before moving on

## Roadmap (do in order, one at a time)
1. Python symbol extractor (tree-sitter) + tests
2. Indexer: embeddings + BM25 hybrid retrieval
3. CLI: `chatter index <path>`, `chatter ask "<question>"`
4. Eval set in eval/ with baseline retrieval hit-rate (before adding features)
5. Call graph + graph-expanded retrieval, re-measure
6. Summary-enriched chunks, re-measure
7. Agentic tool loop, git history, incremental indexing, MCP server
8. Abstention: retrieval currently cannot say "not found"
