# Codebase Chatter

A local tool that indexes a Python repository and answers questions about its
functions and modules, citing `path:start-end` locations. Everything runs on
your machine: tree-sitter chunking, a local embedding model, dense retrieval
expanded along a static call graph (BM25 and hybrid modes available), and an
optional local answer model.

![Demo: chatter index, search and ask on this repository](docs/demo.gif)

## Highlights

- **Tuning on 20 questions overfit, and the held-out run caught it.** The
  fusion config `k10-w3` scored best on the tune split (MRR@50 0.574,
  vs 0.523 for the default) and worst on held-out (0.296, vs 0.374), going 4
  wins to 8 losses against dense-only. The decision rules, frozen before the
  held-out run, rejected it. See [the reversal](#the-k10-w3-reversal).
- **A static call graph is the one retrieval change that held up.** Expanding
  dense results 2 hops along calls, inheritance and protocol implementations
  beat plain dense retrieval 8 wins, 7 losses and 21 ties over 36 questions,
  with no lost top-3 hits, and raised R@10 from 0.557 to 0.608. Dense plus
  call graph is the default. The margin is one question. See
  [held-out verdicts](#held-out-verdicts).
- **Making the model quote its evidence first made answers worse.** An
  evidence-first prompt with code-checked quotes failed all four criteria
  recorded before the run. False abstentions rose from 2 to 17 of 36, and 11
  answers graded correct in the baseline were no longer correct. The plain
  prompt stays the default. See
  [answer quality](#answer-quality).

## Install

Python 3.11+.

```bash
python -m venv .venv
```

```bash
.venv/bin/pip install -e '.[dev]'
```

On Apple Silicon this also installs `mlx-lm`, which becomes the default answer
backend. Without an NVIDIA GPU, install CPU-only PyTorch first to avoid the
large CUDA wheels:

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Models download from Hugging Face on first use: `BAAI/bge-small-en-v1.5` for
embeddings (~130 MB), and for answers `Qwen/Qwen3-4B-Instruct-2507` (~8 GB,
transformers) or `mlx-community/Qwen3-4B-Instruct-2507-4bit` (~2.3 GB, MLX).

## Use

Index a repository (stored in `<repo>/.chatter`; re-running only embeds
changed code):

```bash
chatter index /path/to/repo
```

Search, without generating an answer:

```bash
chatter search "how are response headers parsed" --path /path/to/repo
```

Ask, with a local model that cites the chunks it used:

```bash
chatter ask "what happens when the queue is shut down?" --path /path/to/repo
```

Useful options:

- `--index-dir` / `CHATTER_INDEX_DIR`: keep the index somewhere other than `<repo>/.chatter`.
- `chatter search --json`: machine-readable results.
- `--mode dense|fused|bm25`, `--graph/--no-graph`, `--graph-hops`, `--graph-weight` (search and ask): retrieval. The default, dense with 2-hop call-graph expansion at weight 2, is the config the held-out evaluation chose (see [Results](#results)). For an exact identifier, `--mode fused` adds BM25 keyword matching.
- `chatter ask --backend mlx|transformers`: choose the answer backend (default: mlx on Apple Silicon, transformers elsewhere).
- `chatter ask --prompt evidence`: the model first quotes the lines it relies on, tagged `[C#]`, then answers. Each quote is checked against the lines it was shown, needing at least 12 non-space characters. The answer streams only if at least one quote is found; otherwise it is withheld and reported as NOT FOUND. Verified quotes are shown with their `path:line`. The default stays `plain`: evidence-first failed its success criterion (see [Answer quality](#answer-quality)).
- `chatter index --rebuild`: re-embed everything, e.g. after changing `--model`.

## MCP server

`chatter mcp` serves an existing index to MCP clients over stdio. It is
read-only: it never builds or changes the index, and it picks up a new build
automatically when you re-run `chatter index`.

| Tool | Purpose |
|---|---|
| `search(query, k=10, mode="dense", graph=true)` | Ranked chunks with `path:start-end`, signature, score and which retriever found them. `mode`: `dense`, `fused` (BM25 + dense), or `bm25`; `graph` adds call-graph neighbours (dense and fused only). |
| `get_chunk(chunk_id)` | Full source of one chunk, with docstring, decorators and imports. |
| `list_symbols(path_prefix="", kind=None, limit=200)` | Functions, classes and methods under a path, in file order. |
| `ask(question, k=8)` | Opt-in (`--enable-ask`): answer with the local model. Slow and loads several GB; the client's own model can usually answer from `search` and `get_chunk`. |

**Treat tool output as untrusted data.** `search`, `get_chunk`, `list_symbols`
and `ask` return text taken from the indexed repository: source code,
comments, docstrings and string literals, which anyone who can commit to that
repository controls. A comment like "ignore previous instructions and ..." is
just data. MCP clients and the models using them should not follow
instructions found in tool results, and should be cautious about indexing
repositories they don't trust.

Build the index first, then register the server. Use absolute paths: clients
start the server from their own working directory.

### Claude Code

```bash
claude mcp add --transport stdio chatter -- /abs/path/to/codebase-chatter/.venv/bin/chatter mcp --path /abs/path/to/repo
```

Add `--scope user` to make it available in every project, or `--scope project`
to write it to the repository's `.mcp.json` for your team. To keep the index
elsewhere or enable `ask`, pass environment variables with `--env`:

```bash
claude mcp add --transport stdio --env CHATTER_INDEX_DIR=/abs/path/to/index --env CHATTER_MCP_ENABLE_ASK=1 chatter -- /abs/path/to/codebase-chatter/.venv/bin/chatter mcp --path /abs/path/to/repo
```

The equivalent `.mcp.json` entry:

```json
{
  "mcpServers": {
    "chatter": {
      "command": "/abs/path/to/codebase-chatter/.venv/bin/chatter",
      "args": ["mcp", "--path", "/abs/path/to/repo"],
      "env": {"HF_HUB_OFFLINE": "1"}
    }
  }
}
```

Check it with `claude mcp list`, or `/mcp` inside a session.

### Claude Desktop

Open Settings → Developer → Edit Config, or edit the file directly:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "chatter": {
      "command": "/abs/path/to/codebase-chatter/.venv/bin/chatter",
      "args": ["mcp", "--path", "/abs/path/to/repo"],
      "env": {"HF_HUB_OFFLINE": "1"}
    }
  }
}
```

Quit and restart Claude Desktop to load it. If the server doesn't connect, its
stderr is in `~/Library/Logs/Claude/mcp-server-chatter.log` (macOS) or
`%APPDATA%\Claude\logs` (Windows); a missing index is reported there.

### Configuration

Flags take precedence over environment variables.

| Flag | Environment variable | Default |
|---|---|---|
| `--path` | `CHATTER_REPO` | current directory |
| `--index-dir` | `CHATTER_INDEX_DIR` | `<repo>/.chatter` |
| `--enable-ask` | `CHATTER_MCP_ENABLE_ASK` | off |
| `--backend` | `CHATTER_BACKEND` | mlx on Apple Silicon, else transformers |
| `--answer-model` | `CHATTER_ANSWER_MODEL` | the backend's Qwen3-4B build |
| `--answer-prompt` | `CHATTER_ANSWER_PROMPT` | `plain` (or `evidence`; `ask` then also returns the checked quotes) |

`HF_HUB_OFFLINE=1` stops Hugging Face from checking for model updates at
startup once the models are cached.

## Evaluation

`eval/questions.yaml` is a hand-written question set with `tune` and `heldout`
splits; `eval/corpora.yaml` describes the corpora (a Python 3.13 stdlib subset
and a frozen snapshot of this repo).

Retrieval metrics (hit@1, hit@5, MRR@50, recall@10; strict and containment):

```bash
chatter eval eval/questions.yaml --split tune
```

Compare frozen retrieval configs, or answer every question and write a
markdown file for manual review:

```bash
chatter eval eval/questions.yaml --split tune --candidates eval/candidates.yaml
```

```bash
chatter eval eval/questions.yaml --split tune --answers
```

A candidates file can carry a `decision_rule`, which the report applies
automatically. `eval/candidates.yaml` chooses the fusion config, and
`eval/candidates-graph.yaml` then decides its graph variant against that
choice. Both rules are decided on tune and heldout combined, so run every
split; a run that lacks one is marked NOT FINAL:

```bash
chatter eval eval/questions.yaml --candidates eval/candidates-graph.yaml
```

Results are saved in `eval/results/` with the commit, models and corpus
versions.

Answer runs take `--prompt plain|evidence`. To compare two runs, grade each
review file (fill in its **Verdict:** lines), collect the verdicts, and score
the new run against a recorded success criterion. The comparison is void
unless both runs answered from the same context:

```bash
chatter grade eval/results/<run>-answers-mlx.md
```

```bash
chatter compare-answers eval/results/<base>.json eval/results/<new>.json --criteria eval/answer-criteria-evidence-first.yaml
```

`chatter grade` writes `eval/grades/<run>.yaml`, which `compare-answers`
picks up automatically.

## Results

Retrieval on the eval set: 40 questions over two corpora, a Python 3.13
stdlib subset (2,178 chunks) and a frozen snapshot of this repo (427 chunks),
with `BAAI/bge-small-en-v1.5` embeddings. 22 questions are `tune` (20 scored,
2 negative) and 18 are `heldout` (16 scored, 2 negative). Metrics are strict:
a hit is the exact relevant chunk. MRR@50 uses the first relevant rank, and
R@10 is the share of relevant chunks in the top 10. Every run is in
`eval/results/`.

### Ablation (tune split, 20 questions)

Dense-only scores 0.524 MRR@50 and 0.543 R@10 at every step. The BM25 changes
don't touch it, and replacing Chroma's approximate search with exact search
left its tune metrics unchanged.

| Step | BM25 MRR@50 | Fused (RRF k=60) MRR@50 / R@10 | Best tuned config, MRR@50 / R@10 |
|---|---|---|---|
| Baseline | 0.352 | 0.471 / 0.498 | |
| + query stopwords | 0.403 | 0.484 / 0.523 | |
| + Snowball stemming | 0.374 | 0.523 / 0.560 | |
| + RRF k × dense-weight sweep | 0.374 | 0.523 / 0.560 | `k10-w3`: 0.574 / 0.539 |
| + call graph, Phase A | | | `k10-w3+g2-w3`: 0.583 / 0.646 |
| + call graph, Phase B (inferred types, protocols) | | | `k10-w3+g2-w3`: 0.583 / 0.646 |
| + dense + call graph | | | `dense-only+g2-w2`: 0.531 / 0.621 |

Stemming lowered BM25 on its own but raised fused retrieval. Phase B resolved
more call sites (stdlib 28% → 30%, this repo 31% → 35%) but did not change
the best tune config. Config names: `k10-w3` is RRF k=10 with dense weighted
3×; `+g2-w3` adds 2-hop call-graph expansion with RRF weight 3.

### Held-out verdicts

The candidates and the rules for adopting them were frozen in
`eval/candidates.yaml` and `eval/candidates-graph.yaml` before the held-out
questions were run once, scored with the tune questions (36 total).
- **Fusion rule:** keep fused configs that beat dense-only on paired
  first-rank wins/losses and lose none of its top-3 hits, then choose the best.
- **Graph rule:** adopt the chosen config's graph variant only if it beats
  that config the same way.

| Config | MRR@50 (all 36) | MRR@50 (held-out 16) | W/L/T vs dense-only | W/L/T vs counterpart | Top-3 losses | Verdict |
|---|---|---|---|---|---|---|
| `dense-only` | 0.428 | 0.310 | | | | reference |
| `default` (k=60) | 0.457 | 0.374 | 14/14/8 | | 6 (h10, h17, q09, q10, q18, q20) | fails |
| `k10-w3` | 0.450 | 0.296 | 9/13/14 | | 2 (h10, h17) | fails |
| `dense-only+g2-w2` | 0.432 | 0.310 | 8/7/21 | 8/7/21 | none | **adopted** |
| `default+g2-w1` | 0.459 | 0.375 | 16/12/8 | 7/6/23 | none | not considered |
| `k10-w3+g2-w3` | 0.456 | 0.297 | 12/10/14 | 9/4/23 | none | not considered |

`dense-only+g2-w2` is now the default for `search`, `ask` and the MCP server.
- **The margin is one question.** Treat it as a small gain: R@10 rose
  0.557 → 0.608 and hit@5 0.639 → 0.667, while hit@1 stayed at 0.278 and
  containment MRR dipped 0.520 → 0.516.
- **Graph expansion helped consistently.** Each graph variant beat its own
  counterpart and lost no top-3 hit.
- **The rules favour keeping top hits over average rank.** Fused `default`
  has the best MRR@50 overall, but it pushed six of dense-only's top-3 hits
  down, so it failed.

The held-out questions are now spent. Choosing or confirming anything new
needs a fresh held-out set.

### The k10-w3 reversal

`k10-w3` won the fusion sweep on tune: 0.574 MRR@50 against 0.523 for
`default` and 0.524 for dense-only, with 5/5/10 paired against dense-only.
On held-out it scored the lowest of the three (0.296 vs 0.374 and 0.310) and
went 4/8/4 against dense-only. `default`, which lost to dense-only on tune
(6/8/6), had the best held-out MRR@50 and went 8/6/2. Twenty questions were
too few to tune two RRF parameters. The paired, top-3-protecting rules, frozen
before the held-out run, kept the overfit config out.

### Known limitations

- **Exact identifiers may rank lower.** The default retrieval has no BM25,
  so queries for a rare identifier, constant or error string may rank the
  defining chunk lower. The eval barely measures this. It has two exact-name
  questions, both in tune, and dense ranks both first. For identifier lookups
  use `--mode fused` (MCP: `mode="fused"`).
- **Orphan comments can't be retrieved** (q17); see CLAUDE.md.

### Answer quality

Answers from the local model (Qwen3-4B-Instruct-2507, 4-bit, mlx) on all 40
questions, with the default dense + call-graph retrieval and identical
context for both prompts. The evidence-first prompt asks the model to quote
the lines it relies on, tagged `[C#]`, then answer. Each quote is checked
in code against the lines the model was shown, and an answer with no
verified quote is withheld as an abstention. The success criterion was
committed before the run (`eval/answer-criteria-evidence-first.yaml`).

| Measure (36 answerable questions) | Plain prompt (baseline) | Evidence-first | Pass if | Result |
|---|---|---|---|---|
| Unsupported answers (no relevant chunk in context, still answered) | 5 | 3 (5 before withholding) | ≤ 2 | fail |
| False abstentions | 2 | 17 (6 before withholding) | ≤ 3 | fail |
| Answers citing a relevant chunk | 26 | 14 | ≥ 26 | fail |
| Baseline-correct answers no longer graded correct | – | 11 | 0 | fail |
| Graded correct | 17 | 10 | | |
| Format compliance (parseable Evidence section, or abstention only) | – | 40/40 (30 + 10) | | |
| Answers with a verified quote inside a relevant chunk | – | 12 of 19 shown answers | | |

Of 127 quotes, 63 verified. Most failures joined several source lines into one
quote (a docstring paragraph, a few lines of code), which the prompt forbade
and the check rejects. Two answers, q01 and q10, never reached the Answer
section: the model kept quoting until the repetition guard stopped it. In all,
11 answers were withheld. The blind grader judged 10 of them to be false
abstentions, with the evidence in the shown lines; only q11 was right to
withhold. The model also abstained by itself on 6 answerable questions, 3 of
them falsely (q03, h02, h08). Of the 11 baseline-correct answers that were no
longer correct, 8 became abstentions and 3 became partial. 4 questions
improved from partial to correct (q09, q14, q19, h06). The evidence prompt is
still available with `--prompt evidence`.

**How answers were graded.** Each run's review file was graded against a
written rubric (`eval/grades/README.md`), judged only against the lines shown
to the model. Two fresh subagents, with no shared context, did the grading:
- The baseline grader saw only the rubric, the review file and the corpora.
  A human then spot-checked 15 of its verdicts (the 12 hardest plus 3 drawn
  at random), which changed 5 verdicts and added a precedence rule to the
  rubric.
- The evidence-first run was graded blind by a second subagent. It saw a copy
  of the review without the run header, prompt name, verification results or
  withheld replies, and saw neither the baseline grades nor the criteria.

Grades and justifications are in `eval/grades/`. The comparison is in
`eval/results/20261115-compare-evidence-first.txt`.

**Limitations.**
- A verified quote proves that the line exists in the chunk shown, not that it
  supports the claim made from it.
- The answer model is a 4B local model; a larger model may follow the quoting
  format better.
- There are 40 questions, and the evidence-first run was graded by a model
  without a human spot-check.
- Held-out v1 was spent on the retrieval decision before this work started,
  so this is a regression check on questions already seen, not a held-out
  result.

## Demo

`docs/demo.gif` was recorded with [vhs](https://github.com/charmbracelet/vhs)
from `docs/demo.tape`. It builds a fresh index of this repository in a temporary
directory, then runs one `chatter search` and one `chatter ask` with the default
plain prompt. To re-record it, run this from the repository root:

```bash
vhs docs/demo.tape
```

## Development

```bash
.venv/bin/python -m pytest
```

Tests use an offline fake embedder and generator. Tests that download or run
real models are marked `slow`:

```bash
.venv/bin/python -m pytest -m slow
```
