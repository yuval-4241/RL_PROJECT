"""
Shannon entropy utilities for batch-level diversity reward.

Formula:
    R_total = R_consistency + alpha * H_batch

Where H_batch is the Shannon entropy of the extracted-answer distribution
across all model responses for a single prompt.

Example (16 rollouts per prompt):
  All 16 say "42"            -> H_batch = 0.0  (fully collapsed, no bonus)
  Split 8/8 between two      -> H_batch = 1.0  (max for 2 answers)
  All 16 different answers   -> H_batch = 1.0  (normalized max)
"""
import math
from collections import Counter
from typing import List, Optional


def batch_shannon_entropy(answers: List[Optional[str]], normalize: bool = True) -> float:
    """
    Shannon entropy over the distribution of extracted answers.

    Args:
        answers: list of extracted answer strings from the batch
                 (one entry per model rollout for the same prompt).
                 None and empty-string entries are filtered out.
        normalize: if True, divide by log2(n_unique) so result is in [0, 1].

    Returns:
        H_batch  (float, >= 0)
    """
    answers = [a for a in answers if a is not None and a != ""]
    n = len(answers)
    if n == 0:
        return 0.0

    counts = Counter(answers)
    n_unique = len(counts)
    if n_unique == 1:
        return 0.0

    entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())

    if normalize:
        entropy /= math.log2(n_unique)

    return entropy
