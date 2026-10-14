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
5. Call graph + graph-expanded retrieval, re-measure (done: dense + call graph
   is the default; see Eval status)
6. Summary-enriched chunks, re-measure
7. Agentic tool loop, git history, incremental indexing, MCP server (MCP done:
   `chatter mcp`, read-only)
8. Abstention: retrieval currently cannot say "not found"
- Optional: MLX generator for Apple Silicon (done; default there)

## Eval status
- 2026-10-14: held-out v1 (h01-h18 in eval/questions.yaml) is spent. Its one
  run (eval/results/20261014T170530-c37049f.json) applied the frozen rules in
  eval/candidates.yaml and eval/candidates-graph.yaml. No fusion config beat
  dense-only. dense-only+g2-w2 (dense retrieval plus 2-hop call-graph
  expansion at weight 2) did, 8/7/21 with no top-3 losses, and is now the
  default for search, ask and MCP (DEFAULT_SEARCH in src/chatter/retrieve.py).
  The margin is one question, so treat it as a small gain. Choosing or
  confirming anything new needs a fresh held-out set (v2); h01-h18 are for
  regression tracking only.

## Known issues (fix after the eval baseline)
- Orphan comments are not in any chunk: a comment block above a def/class
  that is separated from it by a blank line belongs to neither the module
  chunk nor the definition, so it can never be retrieved. Example: the
  tree-sitter Point workaround comment in src/chatter/extract.py, which is
  the answer to eval question q17.
