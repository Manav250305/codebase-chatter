# Resume bullets

Numbers come from the files in `eval/results/` and `eval/grades/`. The
retrieval figures are strict MRR@50 and recall@10 over 36 scored questions
(20 tune, 16 held-out) unless a split is named.

- Built Codebase Chatter, a local code question-answering tool with
  `file:line` citations. It uses tree-sitter symbol chunking, bge-small dense
  retrieval with static call-graph expansion, and a 4-bit Qwen3-4B answer
  model on MLX, and ships a read-only MCP server and 447 tests.
- Designed a 40-question retrieval eval with tune and held-out splits and
  decision rules frozen before the held-out run. It rejected a fusion config
  that led on tune (MRR@50 0.574 vs 0.524 for dense-only) but trailed on
  held-out (0.296 vs 0.310).
- Added call-graph expansion (calls, inheritance, protocol implementations;
  2,191 of 7,304 stdlib call sites resolved). It beat dense-only 8 wins,
  7 losses, 21 ties with no lost top-3 hits and raised recall@10 from 0.557
  to 0.608, and became the default.
- Tested an evidence-first answer prompt, with quotes checked in code,
  against a success criterion recorded before the run. A subagent graded
  both runs (the baseline spot-checked by a human, the new run blind). The
  prompt failed all four criteria (false abstentions rose from 2 to 17 of 36)
  and was kept opt-in.
