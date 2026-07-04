"""
Tests for the batch Shannon entropy reward.

Formula:  R_total = R_consistency + alpha * H_batch

H_batch = normalized Shannon entropy over the distribution of extracted answers
across all model rollouts for the same prompt.

No GPU required — standard Python only (math module).
"""
import importlib.util
import math
import pathlib
import pytest

# Load entropy_reward directly from file — skips verl's heavy __init__.py
_REWARD_FILE = (
    pathlib.Path(__file__).parent.parent.parent
    / "verl" / "utils" / "reward_score" / "entropy_reward.py"
)
_spec = importlib.util.spec_from_file_location("entropy_reward", _REWARD_FILE)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

batch_shannon_entropy = _mod.batch_shannon_entropy


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_empty_list_returns_zero():
    assert batch_shannon_entropy([]) == 0.0

def test_none_and_empty_strings_ignored():
    # None and "" are not parsable answers — should be filtered out
    assert batch_shannon_entropy([None, "", None]) == 0.0

def test_all_same_answer_entropy_is_zero():
    # All 32 rollouts say "42" → model collapsed → H = 0
    answers = ["42"] * 32
    assert batch_shannon_entropy(answers) == 0.0

def test_two_equal_answers_entropy_is_one():
    # 16 say "1", 16 say "2" → maximum entropy for 2 options → normalized = 1.0
    answers = ["1"] * 16 + ["2"] * 16
    assert math.isclose(batch_shannon_entropy(answers), 1.0, abs_tol=1e-9)

def test_all_different_answers_entropy_is_one():
    # All 32 rollouts give a different answer → max entropy → normalized = 1.0
    answers = [str(i) for i in range(32)]
    assert math.isclose(batch_shannon_entropy(answers), 1.0, abs_tol=1e-9)

def test_entropy_is_between_zero_and_one():
    answers = ["1"] * 20 + ["2"] * 8 + ["3"] * 4
    h = batch_shannon_entropy(answers)
    assert 0.0 <= h <= 1.0

def test_more_diverse_answers_give_higher_entropy():
    collapsed  = ["42"] * 30 + ["7"] * 2          # mostly one answer
    mixed      = ["42"] * 16 + ["7"] * 8 + ["3"] * 8  # three answers
    assert batch_shannon_entropy(mixed) > batch_shannon_entropy(collapsed)

def test_single_response_entropy_is_zero():
    # Only one response → no diversity possible → H = 0
    assert batch_shannon_entropy(["42"]) == 0.0


# ---------------------------------------------------------------------------
# Normalization flag
# ---------------------------------------------------------------------------

def test_unnormalized_entropy_two_equal():
    # Without normalization: H = -2 * (0.5 * log2(0.5)) = 1.0 bit
    answers = ["1"] * 8 + ["2"] * 8
    h_norm   = batch_shannon_entropy(answers, normalize=True)
    h_unnorm = batch_shannon_entropy(answers, normalize=False)
    assert math.isclose(h_norm, 1.0, abs_tol=1e-9)
    assert math.isclose(h_unnorm, 1.0, abs_tol=1e-9)  # coincides for 2 answers

def test_unnormalized_entropy_four_equal():
    # 4 equally likely answers: H = log2(4) = 2 bits (unnormalized), 1.0 (normalized)
    answers = ["1"] * 8 + ["2"] * 8 + ["3"] * 8 + ["4"] * 8
    h_norm   = batch_shannon_entropy(answers, normalize=True)
    h_unnorm = batch_shannon_entropy(answers, normalize=False)
    assert math.isclose(h_norm,   1.0, abs_tol=1e-9)
    assert math.isclose(h_unnorm, 2.0, abs_tol=1e-9)

def test_normalized_always_le_one():
    for answers in [
        ["1"] * 32,
        ["1"] * 16 + ["2"] * 16,
        [str(i) for i in range(32)],
        ["1"] * 20 + ["2"] * 7 + ["3"] * 5,
    ]:
        assert batch_shannon_entropy(answers, normalize=True) <= 1.0


# ---------------------------------------------------------------------------
# Connection to R_total = R_consistency + alpha * H_batch
# ---------------------------------------------------------------------------

def test_reward_hacking_scenario():
    """
    When the model collapses (all same answer), H_batch = 0.
    The entropy bonus disappears → only R_consistency contributes.
    """
    R_consistency = 1.0
    alpha = 0.1
    answers = ["1"] * 32  # reward hacking

    H_batch = batch_shannon_entropy(answers)
    R_total = R_consistency + alpha * H_batch

    assert H_batch == 0.0
    assert math.isclose(R_total, R_consistency)

def test_diverse_response_gets_bonus():
    """
    When responses are diverse, H_batch > 0 → R_total > R_consistency.
    """
    R_consistency = 0.5
    alpha = 0.1
    answers = ["1"] * 16 + ["2"] * 8 + ["3"] * 8

    H_batch = batch_shannon_entropy(answers)
    R_total = R_consistency + alpha * H_batch

    assert H_batch > 0.0
    assert R_total > R_consistency

def test_alpha_zero_ignores_entropy():
    """alpha = 0 makes the formula identical to the baseline."""
    R_consistency = 0.8
    answers = [str(i) for i in range(32)]
    H_batch = batch_shannon_entropy(answers)

    R_total = R_consistency + 0.0 * H_batch
    assert math.isclose(R_total, R_consistency)

@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2, 0.5])
def test_higher_alpha_gives_higher_bonus(alpha):
    R_consistency = 0.5
    answers = ["1"] * 16 + ["2"] * 16  # H = 1.0
    H_batch = batch_shannon_entropy(answers)
    R_total = R_consistency + alpha * H_batch
    assert math.isclose(R_total, R_consistency + alpha, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Parametrized DAPO-style scenarios
# ---------------------------------------------------------------------------

DAPO_SCENARIOS = [
    ("all_correct_collapsed",  ["42"] * 32,                                   0.0),
    ("majority_correct",       ["42"] * 26 + ["7"] * 6,                       None),
    ("split_two_answers",      ["42"] * 16 + ["7"] * 16,                      1.0),
    ("three_way_split",        ["1"] * 11 + ["2"] * 11 + ["3"] * 10,         None),
    ("fully_diverse",          [str(i) for i in range(32)],                   1.0),
]

@pytest.mark.parametrize("name,answers,expected_h", DAPO_SCENARIOS)
def test_dapo_scenario_entropy(name, answers, expected_h):
    h = batch_shannon_entropy(answers)
    assert 0.0 <= h <= 1.0, f"{name}: H={h} not in [0,1]"
    if expected_h is not None:
        assert math.isclose(h, expected_h, abs_tol=1e-9), f"{name}: expected H={expected_h}, got {h}"
