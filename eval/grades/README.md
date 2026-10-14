# Grades

Manual verdicts on answer runs, one file per run: `<results stem>.yaml`,
next to `eval/results/<results stem>.json`.

```yaml
results: eval/results/20261014T172908-152daa3-answers-mlx.json
grades:
  q01: correct
  q03: partial
  q15: correctly abstained
```

Verdicts: `correct`, `partial`, `wrong`, and, for negative questions,
`correctly abstained` or `should have abstained`. Write them into the run's
review file (`**Verdict:** correct`; a note may follow, e.g. `partial - misses
the retry path`), then collect them:

```bash
chatter grade eval/results/20261014T172908-152daa3-answers-mlx.md
```

`chatter compare-answers` reads these files to score `graded_regressions`.
