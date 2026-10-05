# Codebase Chatter

Local tool that indexes a code repository with code-aware chunking and answers
questions about functions/modules, with `file:line-range` citations.

## Stack
- Python 3.11+, tree-sitter (py-tree-sitter + tree-sitter-python)
- Embeddings: sentence-transformers (local); store: exact cosine search over
  memory-mapped float32 .npy vectors; keyword: rank-bm25
- Answers: local Qwen3-4B-Instruct-2507; mlx-lm (4-bit) by default on Apple
  Silicon, transformers elsewhere (`--backend` selects either)
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
7. Agentic tool loop, git history, incremental indexing, MCP server (MCP done:
   `chatter mcp`, read-only)
8. Abstention: retrieval currently cannot say "not found"
- Optional: MLX generator for Apple Silicon (done; default there)

## Known issues (fix after the eval baseline)
- Orphan comments are not in any chunk: a comment block above a def/class
  that is separated from it by a blank line belongs to neither the module
  chunk nor the definition, so it can never be retrieved. Example: the
  tree-sitter Point workaround comment in src/chatter/extract.py, which is
  the answer to eval question q17.
