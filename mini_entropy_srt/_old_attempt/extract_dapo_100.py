"""
Step 1: Extract a reliable 100-sample test set from the DAPO parquet.

Usage:
    uv run --no-project --with pandas --with pyarrow python scripts/extract_dapo_100.py

Output:
    ~/data/dapo_unlabeled/train_100.parquet   (100 random rows, seed=42)
"""
import argparse
import pathlib
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="~/data/dapo_unlabeled/train.parquet")
    parser.add_argument("--output", default="~/data/dapo_unlabeled/train_100.parquet")
    parser.add_argument("--n",      type=int, default=100)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    src = pathlib.Path(args.input).expanduser()
    dst = pathlib.Path(args.output).expanduser()

    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found.\n"
            "Run examples/data_preprocess/dapo.py first to download the dataset."
        )

    df = pd.read_parquet(src)
    print(f"Loaded {len(df)} rows from {src}")

    sample = df.sample(n=min(args.n, len(df)), random_state=args.seed)
    sample = sample.reset_index(drop=True)

    dst.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(dst, index=False)
    print(f"Saved {len(sample)} samples → {dst}")

    # Quick sanity check
    loaded = pd.read_parquet(dst)
    assert len(loaded) == len(sample), "Row count mismatch after save"
    print("Sanity check passed.")


if __name__ == "__main__":
    main()
