# Results directory structure

## `final_300steps_100test/`
The core comparison. Each file completed the full 300 training steps AND
ran `--final_eval_all` (a full 100-question held-out test set evaluation) at
the end, so the reported `test_accuracy` / `agreement_gap` / `mean_entropy`
numbers here are the trustworthy ones to cite.

- `alpha0.0.json` — plain SRT (self-consistency reward only, no entropy bonus)
- `alpha0.3.json` — entropy bonus, alpha=0.3
- `alpha0.5.json` — entropy bonus, alpha=0.5 ("my method")
- `oracle.json` — ground-truth reward ceiling baseline

Note: `alpha0.0.json` and `alpha0.3.json` are currently being extended to 900
steps on the lab server, to test whether reward-hacking collapse eventually
appears with more training budget. This file reflects the 300-step
checkpoint; once the 900-step run finishes its own `--final_eval_all`, this
file should be replaced/updated with the new numbers, and the old 300-step
version noted as superseded.

## `double_check_300steps_no_100test/`
Completed the full 300 training steps, but was **not** verified against the
full 100-question test set (no `--final_eval_all` run, predates that
feature). Keep as a secondary sanity check / cross-reference only —
**do not cite these numbers as the primary result**, since they weren't
confirmed against the same held-out test set as the `final_300steps_100test/`
files.

- `alpha0.5.json` — first completed `alpha=0.5` run. Also the reference run
  for zero-shot data (every eval checkpoint recorded `zero_shot_accuracy`,
  a field removed from later runs — see `zero_shot/` below).

## `zero_shot/`
- `zero_shot_reference.json` — copy of `final_300steps_100test/alpha0.0.json`.
  Contains a `pretrain_eval` field: the model's accuracy on the held-out test
  set *before any training*, i.e. the true zero-shot baseline
  (`test_accuracy: 0.30`). Use this number as the "before training" point of
  comparison against the trained results in `final_300steps_100test/`.

## `day1_exploration/`
Early, pre-training-script exploration, run via API calls to hosted models
rather than the local Qwen training pipeline. Kept for reference /
methodology narrative, not part of the main results comparison.

- `pilot_*` — Day 1 pilot: 7 hosted models, up to 20 questions each,
  zero-shot vs majority-vote accuracy and reward-hack gap. Only 4 models
  (`llama-3.1-8b-instant`, `qwen2.5-32b`,
  `meta-llama/llama-4-scout-17b-16e-instruct`, `o4-mini`) completed all 20
  questions; the rest stalled early (API rate limits).
- `alpha_sweep_*` — follow-up sweep of the 3 strongest models from the pilot
  (`llama-4-scout`, `o4-mini`, `qwen2.5-32b`) across alpha = 0.01 / 0.1 / 0.5,
  testing the entropy-bonus reward formula before moving to local training.

## `archive_incomplete/`
Partial/stale runs, superseded by the complete versions above. Kept only as
raw backups — not to be used in any reported number.

- `oracle_121steps.json` — an early oracle attempt that crashed at step 121
  (superseded by `final_300steps_100test/oracle.json`, which completed all
  300 steps).
- `alpha0.0_75steps.json` — an early RunPod alpha=0.0 attempt that stopped at
  step 75 (superseded by `final_300steps_100test/alpha0.0.json`).
