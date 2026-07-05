"""
curate_and_export.py -- Build a small, leak-proof training pool for
lightweight_train.py from the existing dapo_unlabeled/train.parquet.

Verified on disk: 17398 rows, columns
  ['prompt', 'source', 'id', 'data_source', 'ability', 'reward_model', 'extra_info']
reward_model = {'style': 'rule', 'ground_truth': 'LABEL_BY_SELF_CONSISTENCY',
                'solution_hidden_during_training': '<real answer>'}
extra_info = {'split': 'train', 'index': N}  -- matches `id` exactly.

So the file already carries an explicit original-dataset index -- just under
the column name `id`, not one of the names lightweight_train.py's
fallback-detection checks for ('original_dapo_idx', 'dapo_idx', 'source_idx').
The real answer is deliberately hidden behind a different key
(solution_hidden_during_training) so it can't accidentally leak into the
reward -- exactly the paper's self-training setup, left untouched here.

This script:
  1. Excludes every row whose `id` is in data/eval_indices.json's main_100
     (Day-1/2's 100-question eval set), matched by VALUE, not row position --
     closing the leak-proof gap lightweight_train.py's own warning flags.
  2. Deterministically samples --n_train rows (recommended 200-500) from
     what's left.
  3. Writes them out with `id` renamed to `original_dapo_idx`, so
     lightweight_train.py's existing index-column auto-detection picks it
     up with NO code changes needed there.

Run with: python -m mini_entropy_srt.curate_and_export --n_train 300
"""
import argparse
import json
from pathlib import Path

import pandas as pd

# Inlined rather than imported from baselines.py: that module also pulls in
# matplotlib (for its plotting), which isn't installed in every training
# environment (e.g. the GPU cluster's yuval_rl env) and isn't needed here.
SEED = 42
INDICES_FILE = Path(__file__).resolve().parent / "data" / "eval_indices.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                         default=str(Path.home() / "data/dapo_unlabeled/train.parquet"),
                         help="The existing full DAPO-unlabeled train file to sample from.")
    parser.add_argument("--n_train", type=int, default=300,
                         help="Number of training questions to curate (recommended 200-500).")
    parser.add_argument("--seed", type=int, default=SEED,
                         help="Sampling seed (defaults to Day-1's seed, 42).")
    parser.add_argument("--output", type=str,
                         default=str(Path.home() / "data/dapo_unlabeled/train_curated.parquet"))
    args = parser.parse_args()

    if not INDICES_FILE.exists():
        raise SystemExit(f"{INDICES_FILE} not found -- run baselines.py first to build it.")
    eval_data = json.loads(INDICES_FILE.read_text())
    eval_indices = set(eval_data["main_100"])
    print(f"Loaded {len(eval_indices)} Day-1/2 eval indices (main_100) from {INDICES_FILE}.")

    df = pd.read_parquet(args.input)
    if "id" not in df.columns:
        raise SystemExit(f"Expected an 'id' column on {args.input} to use as the original "
                          f"dataset index -- schema has changed from what this script was "
                          f"written against, re-check the file by hand.")

    before = len(df)
    pool = df[~df["id"].isin(eval_indices)].reset_index(drop=True)
    print(f"Excluded {before - len(pool)} eval rows from the {before}-row pool "
          f"(matched by 'id' VALUE, not row position).")

    if args.n_train > len(pool):
        raise SystemExit(f"Requested n_train={args.n_train} but only {len(pool)} non-eval rows remain.")

    sampled = pool.sample(n=args.n_train, random_state=args.seed).reset_index(drop=True)
    sampled = sampled.rename(columns={"id": "original_dapo_idx"})

    overlap = eval_indices & set(sampled["original_dapo_idx"])
    assert not overlap, f"BUG: {len(overlap)} sampled rows overlap the eval set: {overlap}"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(out_path)

    print(f"Wrote {len(sampled)} training rows to {out_path}")
    print(f"Verified: 0 overlap with the {len(eval_indices)} Day-1/2 eval indices (main_100).")
    print(f"\nPoint lightweight_train.py at this file with:\n  --train_parquet {out_path}")


if __name__ == "__main__":
    main()
