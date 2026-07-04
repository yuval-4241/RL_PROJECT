"""
Day 2: sweeps alpha over the SAME cached rollouts (no re-inference, no
training). For each model, computes per-question majority-vote accuracy,
agreement (self-consistency), answer-distribution entropy, and mean total
reward, at each alpha in ALPHAS.

Only mean total reward actually moves with alpha here -- accuracy, agreement,
and entropy are properties of the model's ANSWER DISTRIBUTION alone, fixed
once the rollouts are generated. alpha only reweights the reward assigned to
answers that already exist; it can't change which answer is the majority.
That's expected, not a bug: it's the same reason Day 3's honesty rule says
never compare total reward between conditions (of course it's higher with a
bonus added) -- only held-out accuracy over TRAINING steps can show whether
the bonus actually helps, which needs real training, not this offline sweep.

Run with: python -m mini_entropy_srt.alpha_sweep
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mini_entropy_srt import entropy_reward, repo_utils

ALPHAS = [0.01, 0.10, 0.50]
MIN_PARSE_RATE = 0.90

THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "results"


def _all_cached_contents(model: str, prompt_idx: int, n_rollouts: int) -> list:
    """Zero-shot (rollout_idx 0) plus all N rollouts, for parse-rate purposes.

    A missing file (generation never completed -- e.g. a run that hit a rate
    limit mid-question) counts as empty content, i.e. honestly "no answer",
    rather than crashing the whole sweep over one incomplete question."""
    directory = entropy_reward.cache_dir_for_model(model)
    contents = []
    for r in range(0, n_rollouts + 1):
        path = directory / f"{prompt_idx:04d}_{r:02d}.json"
        if not path.exists():
            contents.append("")
            continue
        contents.append(json.loads(path.read_text())["content"])
    return contents


def infer_n_rollouts(model: str, prompt_indices: list) -> int:
    """Max cached-generations-minus-zero-shot across ALL pilot questions, so an
    interrupted run (some questions incomplete) doesn't undercount the rollout
    count the model was actually run with."""
    directory = entropy_reward.cache_dir_for_model(model)
    best = 0
    for prompt_idx in prompt_indices:
        n_files = len(list(directory.glob(f"{prompt_idx:04d}_*.json")))
        best = max(best, n_files - 1)
    return max(best, 0)


def compute_parse_rate(model: str, prompt_indices: list, n_rollouts: int) -> float:
    """Fraction of (zero-shot + rollout) generations with a non-empty \\boxed{} answer."""
    total = parsed = 0
    for p_idx in prompt_indices:
        for content in _all_cached_contents(model, p_idx, n_rollouts):
            total += 1
            if repo_utils.extract_boxed_answer(content) is not None:
                parsed += 1
    return parsed / total if total else 0.0


def sweep_alpha(model: str, prompt_indices: list, ground_truths: list, n_rollouts: int) -> pd.DataFrame:
    rows = []
    for alpha in ALPHAS:
        accs, agreements, entropies, total_rewards = [], [], [], []
        for p_idx, gt in zip(prompt_indices, ground_truths):
            group = entropy_reward.compute_rewards_for_question(model, p_idx, gt, alpha, n_rollouts)
            valid = [r for r in group["per_rollout"] if r["answer"] is not None]
            agreement = sum(r["majority_reward"] for r in valid) / len(valid) if valid else 0.0

            accs.append(group["majority_correct"])
            agreements.append(agreement)
            entropies.append(group["entropy"])
            total_rewards.append(sum(r["reward"] for r in group["per_rollout"]) / len(group["per_rollout"]))

        rows.append(
            {
                "model": model,
                "alpha": alpha,
                "accuracy": sum(accs) / len(accs),
                "agreement": sum(agreements) / len(agreements),
                "mean_entropy": sum(entropies) / len(entropies),
                "mean_total_reward": sum(total_rewards) / len(total_rewards),
            }
        )
    return pd.DataFrame(rows)


def plot_sweep(df: pd.DataFrame, model: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    metrics = ["accuracy", "agreement", "mean_entropy", "mean_total_reward"]
    for ax, metric in zip(axes, metrics):
        ax.plot(df["alpha"], df[metric], marker="o")
        ax.set_xlabel("alpha")
        ax.set_xscale("log")
        ax.set_title(metric)
    fig.suptitle(f"Day 2 alpha sweep -- {model} (cached rollouts, no training)")
    plt.tight_layout()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "__")
    plt.savefig(RESULTS_DIR / f"alpha_sweep_{safe_model}.png")
    plt.close(fig)


def _write_sweep_summary(df: pd.DataFrame) -> None:
    """Merges with whatever's already on disk (keyed by model), same pattern
    as baselines._write_results -- separate per-model sweeps accumulate
    instead of overwriting each other."""
    summary_path = RESULTS_DIR / "alpha_sweep_summary.csv"
    new_models = set(df["model"])

    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        existing = existing[~existing["model"].isin(new_models)]
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    combined.sort_values(["model", "alpha"]).to_csv(summary_path, index=False)


def diversity_audit(model: str, prompt_indices: list, ground_truths: list, n_rollouts: int) -> pd.DataFrame:
    """For each question: entropy, the actual distinct answers (so genuine
    variety vs. spam is directly inspectable), and whether the vote was
    correct -- then cross-checks whether high-entropy questions tend to be
    the ones the vote gets wrong (the signal this whole project targets)."""
    rows = []
    for p_idx, gt in zip(prompt_indices, ground_truths):
        group = entropy_reward.compute_rewards_for_question(model, p_idx, gt, alpha=0.0, n_rollouts=n_rollouts)
        distinct_answers = sorted({r["answer"] for r in group["per_rollout"] if r["answer"] is not None})
        rows.append(
            {
                "model": model,
                "prompt_idx": p_idx,
                "entropy": group["entropy"],
                "n_distinct_answers": len(distinct_answers),
                "distinct_answers": distinct_answers,
                "n_unparsed": group["n_unparsed"],
                "majority_correct": bool(group["majority_correct"]),
            }
        )
    df = pd.DataFrame(rows)

    median_entropy = df["entropy"].median()
    high = df[df["entropy"] > median_entropy]
    low = df[df["entropy"] <= median_entropy]
    print(f"[diversity_audit] {model}: vote accuracy on HIGH-entropy questions (> median {median_entropy:.2f}): "
          f"{high['majority_correct'].mean():.2f} (n={len(high)})")
    print(f"[diversity_audit] {model}: vote accuracy on LOW-entropy questions: "
          f"{low['majority_correct'].mean():.2f} (n={len(low)})")

    for _, row in df.iterrows():
        if row["entropy"] > median_entropy and row["n_unparsed"] >= max(n_rollouts // 2, 1):
            print(
                f"[diversity_audit] {model} prompt_idx={row['prompt_idx']}: high entropy but "
                f"{row['n_unparsed']}/{n_rollouts} unparsed -- inspect: {row['distinct_answers']}"
            )

    return df


def run_alpha_sweep(models: list, prompt_indices: list, ground_truths: list) -> dict:
    results = {}
    for model in models:
        n_rollouts = infer_n_rollouts(model, prompt_indices)
        if n_rollouts <= 0:
            print(f"[alpha_sweep] {model}: no cached rollouts found, skipping.")
            continue

        rate = compute_parse_rate(model, prompt_indices, n_rollouts)
        print(f"[alpha_sweep] {model}: parse rate {rate:.1%} (n_rollouts={n_rollouts})")
        if rate < MIN_PARSE_RATE:
            print(f"[alpha_sweep] {model}: SKIPPED -- parse rate below {MIN_PARSE_RATE:.0%} threshold.")
            continue

        df = sweep_alpha(model, prompt_indices, ground_truths, n_rollouts)
        plot_sweep(df, model)
        _write_sweep_summary(df)
        diversity_audit(model, prompt_indices, ground_truths, n_rollouts)
        results[model] = df

    return results


if __name__ == "__main__":
    from mini_entropy_srt import baselines

    prompt_indices, _, ground_truths = baselines.load_pilot_questions()
    sweeps = run_alpha_sweep(baselines.DEFAULT_MODELS, prompt_indices, ground_truths)
    for model, df in sweeps.items():
        print(f"\n=== {model} ===")
        print(df.to_string(index=False))
