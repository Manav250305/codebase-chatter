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

## Rubric

Grounding is judged only against the chunks and lines the model was shown
(listed under "Context given to the model" in the review file), not against
the full repository.

- **correct**: right AND supported by the chunks shown to the model.
- **partial**: some claims supported and right, others unsupported or wrong.
- **wrong**: incorrect, OR a false abstention when the evidence was in context
  (note "false abstention").
- **correctly abstained**: abstained, and the shown context didn't contain the
  answer.
- **should have abstained**: answered without support in the shown context,
  EVEN IF factually right (note "right from memory" when it is).

These five verdicts apply to every question, answerable or negative.

Precedence: if the shown context did not contain the answer, **should have
abstained** takes precedence over correct, partial and wrong. **wrong** is
reserved for answers where the evidence was shown and the model got it wrong
(or abstained falsely). Support is judged from the shown lines, module
docstrings included, not from the eval set's relevant ids.

## Format

Verdicts: `correct`, `partial`, `wrong`, `correctly abstained` or
`should have abstained`. Write them into the run's
review file (`**Verdict:** correct`; a note may follow, e.g. `partial - misses
the retry path`), then collect them:

```bash
chatter grade eval/results/20261014T172908-152daa3-answers-mlx.md
```

`chatter compare-answers` reads these files to score `graded_regressions`.
