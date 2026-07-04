"""
Pre-Day-3 sanity check (no GPU needed): confirms the per-rollout surprisal
bonus in entropy_reward.py actually changes the RLOO training signal, rather
than canceling out under the leave-one-out baseline.

The RLOO formula is transcribed verbatim from the repo's real
compute_rloo_outcome_advantage (verl/trainer/ppo/core_algos.py:248-287),
quoted rather than imported: that function takes torch tensors shaped for a
full training batch (response-length token dimension, GPU dtype), and
importing verl.trainer would pull in torch/ray as a dependency of THIS
Groq/lab-API-only, inference-only project just to run a scalar arithmetic
check. The formula itself is three lines and is copied exactly, not
reimplemented or guessed:

    scores[i] = scores[i] * n/(n-1) - mean(scores) * n/(n-1)      (n > 1)

which is algebraically the leave-one-out baseline:
    advantage_i = scores_i - mean_excluding_i
"""
from mini_entropy_srt import entropy_reward


def compute_rloo_advantages(scores: list) -> list:
    """Verbatim port of verl/trainer/ppo/core_algos.py:compute_rloo_outcome_advantage,
    specialized to one scalar total-reward per rollout (no token dimension,
    no batching across prompts -- this checks a single question's group)."""
    n = len(scores)
    if n <= 1:
        return [0.0] * n
    mean = sum(scores) / n
    return [s * n / (n - 1) - mean * n / (n - 1) for s in scores]


def invariance_test(model: str, prompt_idx: int, n_rollouts: int, alpha: float = 0.1) -> dict:
    """On one question's rollout group, compares RLOO advantages computed from
    the plain majority reward vs. the alpha-bonused reward. They MUST differ --
    if identical, the bonus is acting as a per-question constant and cancels
    under the leave-one-out baseline (the exact bug this design avoids)."""
    contents = entropy_reward.load_cached_rollouts(model, prompt_idx, n_rollouts)
    group = entropy_reward.compute_group_rewards(contents, alpha)

    plain_scores = [r["majority_reward"] for r in group["per_rollout"]]
    bonus_scores = [r["reward"] for r in group["per_rollout"]]

    plain_advantages = compute_rloo_advantages(plain_scores)
    bonus_advantages = compute_rloo_advantages(bonus_scores)

    identical = all(abs(p - b) < 1e-9 for p, b in zip(plain_advantages, bonus_advantages))

    return {
        "model": model,
        "prompt_idx": prompt_idx,
        "alpha": alpha,
        "entropy": group["entropy"],
        "answers": [r["answer"] for r in group["per_rollout"]],
        "plain_advantages": plain_advantages,
        "bonus_advantages": bonus_advantages,
        "identical": identical,
    }


def plumbing_test(model: str, prompt_idx: int, n_rollouts: int) -> dict:
    """Same batch, alpha=0 vs. alpha=0.1: the raw reward values AND the
    resulting advantages must differ once alpha is nonzero."""
    contents = entropy_reward.load_cached_rollouts(model, prompt_idx, n_rollouts)
    zero = entropy_reward.compute_group_rewards(contents, alpha=0.0)
    nonzero = entropy_reward.compute_group_rewards(contents, alpha=0.1)

    rewards_differ = any(
        abs(a["reward"] - b["reward"]) > 1e-9 for a, b in zip(zero["per_rollout"], nonzero["per_rollout"])
    )
    adv_zero = compute_rloo_advantages([r["reward"] for r in zero["per_rollout"]])
    adv_nonzero = compute_rloo_advantages([r["reward"] for r in nonzero["per_rollout"]])
    advantages_differ = any(abs(a - b) > 1e-9 for a, b in zip(adv_zero, adv_nonzero))

    return {
        "model": model,
        "prompt_idx": prompt_idx,
        "rewards_differ": rewards_differ,
        "advantages_differ": advantages_differ,
    }


def _report(result: dict) -> bool:
    """Prints a test result and returns whether it passed."""
    passed = result["identical"] is False if "identical" in result else (
        result["rewards_differ"] and result["advantages_differ"]
    )
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("  PASS" if passed else "  FAIL")
    return passed


if __name__ == "__main__":
    import sys

    from mini_entropy_srt import baselines, alpha_sweep

    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5-32b"
    prompt_indices, _, _ = baselines.load_pilot_questions()
    prompt_idx = prompt_indices[0]
    n_rollouts = alpha_sweep.infer_n_rollouts(model, [prompt_idx])

    print(f"=== invariance test: {model}, prompt_idx={prompt_idx}, n_rollouts={n_rollouts} ===")
    inv = invariance_test(model, prompt_idx, n_rollouts)
    inv_passed = _report(inv)

    print(f"\n=== plumbing test: {model}, prompt_idx={prompt_idx} ===")
    plumb = plumbing_test(model, prompt_idx, n_rollouts)
    plumb_passed = _report(plumb)

    if inv_passed and plumb_passed:
        print("\nBoth checks passed: the per-rollout bonus produces a real, "
              "non-canceling gradient signal under RLOO.")
    else:
        print("\nFAILED: the bonus is not affecting the RLOO advantages as designed -- stop before Day 3.")
