# Results directory notes

## `lab_alpha0.5.json`

This is the first completed `alpha=0.5` run (300/300 steps, lab server). Kept
as-is intentionally: it's the only complete run with `zero_shot_accuracy`
recorded in every eval checkpoint, since that field was later removed from
`evaluate()` in `lightweight_train.py` (rollout-0-alone scoring, used to
compare against majority-vote accuracy). All runs after that code change
only have `test_accuracy`, `agreement_gap`, and `mean_entropy` at each
checkpoint -- no `zero_shot_accuracy`.

Use this file specifically if you need the zero-shot-vs-majority-vote
comparison; use the other (later) result files for the main
`alpha=0.0` / `alpha=0.5` / `oracle` comparison.
