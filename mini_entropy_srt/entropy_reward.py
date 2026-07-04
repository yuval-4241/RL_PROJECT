"""
Day 2: offline entropy-augmented reward, computed on cached text rollouts
(no re-inference, no training -- same frozen-model measurement as Day 1).

    reward_i = majority_reward_i + alpha * surprisal_i

  majority_reward_i : 1 if rollout i's extracted answer equals the group's
                       majority answer, else 0 (the plain SRT / baseline reward).
  surprisal_i        : -log(p(a_i)), where p(a_i) is the empirical frequency of
                       rollout i's OWN extracted answer within its group of N
                       rollouts for that question. Rare answers get a bigger
                       bonus than common ones.

CRITICAL: surprisal_i varies per rollout (rare answer -> bigger number). A
per-question constant (e.g. a uniform +alpha*H added to every rollout in the
group) cancels under RLOO's leave-one-out baseline and produces zero training
signal -- see gradient_checks.py, which verifies this empirically against the
repo's real compute_rloo_outcome_advantage.

The GROUP MEAN of surprisal_i equals the Shannon entropy H of the answer
distribution (nats, natural log) -- kept for logging/plots even though the
per-rollout values differ:
    mean_i(surprisal_i) = sum_a p(a) * -log(p(a)) = H

Empty/unparsed answers (repo_utils.extract_boxed_answer returned None) are
EXCLUDED from the answer distribution -- they aren't a real "answer" and
shouldn't inflate entropy or receive a rarity bonus (that would reward
spamming unparseable text instead of genuine answer variety).
"""
import json
import math
from collections import Counter
from pathlib import Path

from mini_entropy_srt import repo_utils

CACHE_DIR = Path(__file__).resolve().parent / "cache"


def cache_dir_for_model(model: str) -> Path:
    return CACHE_DIR / model.replace("/", "__")


def load_cached_rollouts(model: str, prompt_idx: int, n_rollouts: int) -> list:
    """The N cached rollout texts (rollout_idx 1..n_rollouts) for one question.

    A missing file (a generation that never completed, e.g. a run that hit a
    rate limit mid-question) is treated as empty content -- honestly "no
    answer", same as a generation that came back blank, rather than crashing."""
    directory = cache_dir_for_model(model)
    contents = []
    for r in range(1, n_rollouts + 1):
        path = directory / f"{prompt_idx:04d}_{r:02d}.json"
        if not path.exists():
            contents.append("")
            continue
        contents.append(json.loads(path.read_text())["content"])
    return contents


def compute_group_rewards(contents: list, alpha: float) -> dict:
    """Per-rollout reward for one question's group of rollouts.

    contents: raw rollout text, all for the SAME question.
    Returns {"majority_answer", "entropy", "per_rollout": [...]}, where
    per_rollout has one entry per input rollout, same order, each with
    {"answer", "majority_reward", "p_answer", "surprisal", "reward"}.
    """
    answers = [repo_utils.extract_boxed_answer(c) for c in contents]
    valid_answers = [a for a in answers if a is not None]
    n_valid = len(valid_answers)

    if n_valid == 0:
        empty_entry = {"answer": None, "majority_reward": 0.0, "p_answer": 0.0, "surprisal": 0.0, "reward": 0.0}
        return {"majority_answer": None, "entropy": 0.0, "per_rollout": [dict(empty_entry) for _ in contents]}

    counts = Counter(valid_answers)
    majority_answer, _ = counts.most_common(1)[0]
    freqs = {a: c / n_valid for a, c in counts.items()}
    entropy = -sum(p * math.log(p) for p in freqs.values())  # nats; equals mean surprisal below

    per_rollout = []
    for answer in answers:
        if answer is None:
            # Unparsed rollout: no majority reward, no rarity bonus -- rewarding
            # rarity of a non-answer would incentivize spamming unparseable text.
            per_rollout.append({"answer": None, "majority_reward": 0.0, "p_answer": 0.0, "surprisal": 0.0, "reward": 0.0})
            continue
        majority_reward = 1.0 if answer == majority_answer else 0.0
        p_answer = freqs[answer]
        surprisal = -math.log(p_answer)
        per_rollout.append(
            {
                "answer": answer,
                "majority_reward": majority_reward,
                "p_answer": p_answer,
                "surprisal": surprisal,
                "reward": majority_reward + alpha * surprisal,
            }
        )

    return {"majority_answer": majority_answer, "entropy": entropy, "per_rollout": per_rollout}


def compute_rewards_for_question(
    model: str, prompt_idx: int, ground_truth: str, alpha: float, n_rollouts: int
) -> dict:
    """compute_group_rewards() plus ground-truth accuracy fields, reading rollouts from cache."""
    contents = load_cached_rollouts(model, prompt_idx, n_rollouts)
    group = compute_group_rewards(contents, alpha)

    majority_answer = group["majority_answer"]
    majority_correct = (
        repo_utils.score_against_ground_truth("\\boxed{" + majority_answer + "}", ground_truth)
        if majority_answer is not None
        else 0.0
    )
    group.update(
        {
            "model": model,
            "prompt_idx": prompt_idx,
            "ground_truth": ground_truth,
            "n_rollouts": n_rollouts,
            "majority_correct": majority_correct,
            "n_unparsed": sum(1 for r in group["per_rollout"] if r["answer"] is None),
        }
    )
    return group
