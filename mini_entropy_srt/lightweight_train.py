"""
lightweight_train.py — Minimal RLOO self-rewarded training loop.

Avoids verl/vLLM/FSDP entirely. Only depends on torch + transformers, both already
confirmed working on this GPU. Implements the same algorithm as the paper's SRT:
majority-vote self-consistency reward + RLOO leave-one-out advantage, with an optional
per-rollout entropy (surprisal) bonus.

Usage:
    python lightweight_train.py --alpha 0.0 --n_steps 20 --n_rollouts 4   # smoke test
    python lightweight_train.py --alpha 0.3 --n_steps 500 --n_rollouts 8  # real run
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_ground_truth(row):
    """
    DAPO's parquet does NOT have a plain 'answer' or 'ground_truth' column.
    The true answer lives inside the 'reward_model' struct, typically as
    reward_model['ground_truth']. Handle numpy/dict wrapping defensively.
    """
    rm = row.get("reward_model", None)
    if rm is None:
        return None
    if hasattr(rm, "item"):
        try:
            rm = rm.item()
        except Exception:
            pass
    if isinstance(rm, dict):
        return rm.get("ground_truth", None)
    return None


def extract_prompt_text(prompt_field, tokenizer=None):
    """
    DAPO's parquet stores 'prompt' as a numpy array containing one dict:
        array([{'role': 'user', 'content': '...'}], dtype=object)
    Extract the content string, and apply the tokenizer's chat template if available
    so the model sees properly formatted instruction-following input.
    """
    # Unwrap numpy array / list wrapper
    if hasattr(prompt_field, "tolist"):
        prompt_field = prompt_field.tolist()
    if isinstance(prompt_field, (list, tuple)) and len(prompt_field) > 0:
        messages = prompt_field
    elif isinstance(prompt_field, dict):
        messages = [prompt_field]
    elif isinstance(prompt_field, str):
        return prompt_field
    else:
        return str(prompt_field)

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass  # fall through to plain content extraction below

    # Fallback: just grab the content string(s)
    parts = [m.get("content", "") for m in messages if isinstance(m, dict)]
    return "\n".join(parts) if parts else str(prompt_field)


# ---------------------------------------------------------------------------
# Answer extraction (matches the repo's \boxed{} convention)
# ---------------------------------------------------------------------------
def extract_boxed_answer(text: str):
    """Extract the content of the LAST \\boxed{...} in text. Returns None if absent."""
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start : i - 1].strip()


def answers_match(a: str, b: str) -> bool:
    """Loose string-equality check on normalized answers."""
    if a is None or b is None:
        return False
    norm = lambda s: s.strip().replace(" ", "").replace("$", "")
    return norm(a) == norm(b)


# ---------------------------------------------------------------------------
# Reward: majority-vote + PER-ROLLOUT surprisal bonus (the validated design)
# ---------------------------------------------------------------------------
def compute_rewards(answers: list, ground_truth: str, alpha: float):
    """
    answers: list of extracted answer strings (one per rollout), may contain None
    ground_truth: the true answer for this question (used only for logging/eval,
                  NOT used in the SRT training reward itself)
    alpha: entropy-bonus strength

    Returns: (rewards, majority_answer, agreement, true_accuracy_this_batch)
    """
    n = len(answers)
    valid = [a for a in answers if a is not None]
    if not valid:
        # nothing parsed: zero reward everywhere, zero bonus
        return [0.0] * n, None, 0.0, 0.0

    counts = Counter(valid)
    majority_answer, majority_count = counts.most_common(1)[0]
    agreement = majority_count / len(valid)

    # per-rollout surprisal: -log( p(this rollout's answer) ), using EXCLUDED empties
    total_valid = len(valid)
    probs = {ans: c / total_valid for ans, c in counts.items()}

    rewards = []
    true_correct = 0
    for a in answers:
        if a is None:
            rewards.append(0.0)  # empties get zero reward (and were excluded from p(a))
            continue
        majority_reward = 1.0 if a == majority_answer else 0.0
        surprisal = -math.log(max(probs[a], 1e-9))
        rewards.append(majority_reward + alpha * surprisal)
        if ground_truth is not None and answers_match(a, ground_truth):
            true_correct += 1

    true_accuracy = true_correct / n
    return rewards, majority_answer, agreement, true_accuracy


# ---------------------------------------------------------------------------
# RLOO leave-one-out advantage (the exact math, ~5 lines)
# ---------------------------------------------------------------------------
def rloo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    n = rewards.shape[0]
    if n <= 1:
        return torch.zeros_like(rewards)
    baselines = (rewards.sum() - rewards) / (n - 1)
    return rewards - baselines


# ---------------------------------------------------------------------------
# Gradient invariance sanity check — run BEFORE any real training
# ---------------------------------------------------------------------------
def gradient_invariance_check(alpha: float):
    """
    Confirms the per-rollout bonus actually changes the RLOO advantages
    (i.e. the zero-gradient bug is NOT present). Uses a synthetic example
    mirroring idx 28 from the Day-2 analysis (majority wrong, one correct rare answer).

    IMPORTANT: if the user's real training alpha is 0.0 (a deliberate baseline/
    control run), there is nothing to check -- the bonus is intentionally off --
    so we skip rather than raise a false failure.
    """
    if alpha == 0.0:
        print("[gradient check] alpha=0.0 requested (baseline run, bonus intentionally "
              "off) -- skipping invariance check, nothing to verify.\n")
        return

    answers = ["6", "6", "6", "6", "6", "3", "6", "9/2"]  # majority "6" is wrong; "3" is correct
    ground_truth = "3"

    rewards_no_bonus, _, _, _ = compute_rewards(answers, ground_truth, alpha=0.0)
    rewards_with_bonus, _, _, _ = compute_rewards(answers, ground_truth, alpha=alpha)

    adv_no_bonus = rloo_advantages(torch.tensor(rewards_no_bonus))
    adv_with_bonus = rloo_advantages(torch.tensor(rewards_with_bonus))

    # index of the lone correct rollout ("3")
    idx_correct = answers.index("3")
    a0 = adv_no_bonus[idx_correct].item()
    a1 = adv_with_bonus[idx_correct].item()

    print(f"[gradient check] advantage for rare-correct rollout: "
          f"no-bonus={a0:.4f}  with-bonus(alpha={alpha})={a1:.4f}")
    if math.isclose(a0, a1, abs_tol=1e-6):
        raise RuntimeError(
            "GRADIENT INVARIANCE CHECK FAILED: advantages identical with/without bonus. "
            "The entropy bonus is being cancelled (zero-gradient bug). STOPPING before training."
        )
    print("[gradient check] PASSED — bonus changes the advantage. Safe to proceed.\n")


# ---------------------------------------------------------------------------
# Rollout generation (plain transformers .generate(), no vLLM)
# ---------------------------------------------------------------------------
def generate_rollouts(model, tokenizer, prompt: str, n_rollouts: int, max_new_tokens: int):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            num_return_sequences=n_rollouts,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    texts = [
        tokenizer.decode(seq[prompt_len:], skip_special_tokens=True) for seq in outputs
    ]
    return texts, outputs  # outputs includes prompt+completion token ids


def compute_sequence_logprobs(model, tokenizer, sequences: torch.Tensor, prompt_len: int):
    """Forward pass WITH gradients on the already-generated sequences; sum log-probs
    of the generated (post-prompt) tokens only.

    IMPORTANT: sequences come from model.generate() under torch.no_grad(), so they are
    plain integer token ids with no grad history attached (which is correct -- you can't
    backprop through discrete sampling). Gradients flow through the MODEL'S PARAMETERS
    via this fresh forward pass, not through the sequence tensor itself. Make sure the
    model is in train() mode (not eval()) so dropout/gradient-checkpointing behave correctly.
    """
    model.train()
    sequences = sequences.detach()  # ensure no stale graph from generation is attached
    attention_mask = (sequences != tokenizer.pad_token_id).long()
    outputs = model(input_ids=sequences, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, :-1, :]  # predict token t+1 from position t
    targets = sequences[:, 1:]
    log_probs_all = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs_all.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    # only sum over the GENERATED portion (after the prompt)
    gen_mask = torch.zeros_like(targets, dtype=torch.bool)
    gen_mask[:, prompt_len - 1 :] = True
    seq_log_probs = (token_log_probs * gen_mask).sum(dim=1)
    return seq_log_probs


# ---------------------------------------------------------------------------
# One training step: generate -> reward -> advantage -> policy-gradient update
# ---------------------------------------------------------------------------
def training_step(model, tokenizer, optimizer, prompt, ground_truth, alpha, n_rollouts, max_new_tokens, debug=False):
    texts, gen_outputs = generate_rollouts(model, tokenizer, prompt, n_rollouts, max_new_tokens)
    answers = [extract_boxed_answer(t) for t in texts]

    if debug:
        print(f"  [debug] ground_truth={ground_truth!r}")
        for i, (t, a) in enumerate(zip(texts, answers)):
            print(f"  [debug] rollout {i}: extracted={a!r} | raw_tail={t[-150:]!r}")

    rewards, majority_answer, agreement, true_accuracy = compute_rewards(answers, ground_truth, alpha)

    advantages = rloo_advantages(torch.tensor(rewards, dtype=torch.float32, device=model.device))

    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    log_probs = compute_sequence_logprobs(model, tokenizer, gen_outputs, prompt_len)

    loss = -(advantages.detach() * log_probs).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    mean_entropy = 0.0
    valid_answers = [a for a in answers if a is not None]
    if valid_answers:
        counts = Counter(valid_answers)
        total = len(valid_answers)
        mean_entropy = -sum((c / total) * math.log(c / total) for c in counts.values())

    return {
        "loss": loss.item(),
        "agreement": agreement,
        "true_accuracy": true_accuracy,
        "entropy": mean_entropy,
    }


# ---------------------------------------------------------------------------
# Held-out evaluation
# ---------------------------------------------------------------------------
def evaluate(model, tokenizer, test_df, n_eval_questions, n_rollouts, max_new_tokens):
    model.eval()
    accs, gaps, ents = [], [], []
    sample = test_df.sample(min(n_eval_questions, len(test_df)), random_state=42)
    for _, row in sample.iterrows():
        prompt = extract_prompt_text(row["prompt"], tokenizer=tokenizer)
        texts, _ = generate_rollouts(model, tokenizer, prompt, n_rollouts, max_new_tokens)
        answers = [extract_boxed_answer(t) for t in texts]
        _, majority_answer, agreement, true_accuracy = compute_rewards(
            answers, extract_ground_truth(row), alpha=0.0
        )
        valid_answers = [a for a in answers if a is not None]
        entropy = 0.0
        if valid_answers:
            counts = Counter(valid_answers)
            total = len(valid_answers)
            entropy = -sum((c / total) * math.log(c / total) for c in counts.values())
        accs.append(true_accuracy)
        gaps.append(agreement - true_accuracy)
        ents.append(entropy)
    model.train()
    return {
        "test_accuracy": sum(accs) / len(accs) if accs else 0.0,
        "agreement_gap": sum(gaps) / len(gaps) if gaps else 0.0,
        "mean_entropy": sum(ents) / len(ents) if ents else 0.0,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_parquet", type=str,
                         default=str(Path.home() / "data/dapo_unlabeled/train.parquet"))
    parser.add_argument("--test_parquet", type=str,
                         default=str(Path.home() / "data/srt_test_dataset/test.parquet"))
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--n_rollouts", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--n_eval_questions", type=int, default=10)
    parser.add_argument("--debug_steps", type=int, default=2,
                         help="Print raw generated text + extracted answers for the "
                              "first N training steps, to diagnose empty-parse issues.")
    parser.add_argument("--output", type=str, default="results/lightweight_run.json")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for training-question sampling (reproducibility).")
    parser.add_argument("--eval_indices_path", type=str,
                         default=str(Path.home() / "RL_Project/mini_entropy_srt/data/eval_indices.json"),
                         help="Path to Day-1's eval_indices.json; those rows are EXCLUDED from "
                              "training to avoid train/eval leakage.")
    args = parser.parse_args()

    import random as _random
    _random.seed(args.seed)
    import numpy as _np
    _np.random.seed(args.seed)

    print(f"=== Lightweight RLOO training: alpha={args.alpha}, n_steps={args.n_steps} ===\n")

    print("Running gradient invariance check first...")
    gradient_invariance_check(args.alpha)

    print(f"Loading model {args.model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16
    ).to("cuda")
    model.gradient_checkpointing_enable()  # trade compute for memory
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # SGD needs ~1x extra memory for its state vs AdamW's ~2x -- important on an 11GB GPU
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    print("Loading data...")
    train_df = pd.read_parquet(args.train_parquet)
    test_df = pd.read_parquet(args.test_parquet)

    # Exclude Day-1 eval-set rows from training to avoid train/eval leakage
    eval_path = Path(args.eval_indices_path)
    if eval_path.exists():
        with open(eval_path) as f:
            eval_data = json.load(f)
        # eval_indices.json may store indices under various keys depending on how it
        # was written on Day 1 -- check a few common shapes defensively.
        eval_indices = (
            eval_data.get("main_100")
            or eval_data.get("eval_indices")
            or eval_data.get("indices")
            or []
        )
        before = len(train_df)
        train_df = train_df.reset_index(drop=True)
        train_df = train_df.drop(index=[i for i in eval_indices if i < len(train_df)], errors="ignore")
        print(f"Excluded {before - len(train_df)} Day-1 eval rows from training pool "
              f"(loaded from {eval_path}).")
    else:
        print(f"WARNING: eval_indices_path not found at {eval_path} -- "
              f"could not exclude Day-1 eval rows. Proceeding with full train set "
              f"(risk of train/eval overlap if eval questions come from this same file).")

    prompt_col = "prompt" if "prompt" in train_df.columns else train_df.columns[0]
    print(f"Train rows (after exclusion): {len(train_df)}, Test rows: {len(test_df)}")

    history = []
    for step in range(args.n_steps):
        row = train_df.sample(1, random_state=args.seed + step).iloc[0]
        prompt = extract_prompt_text(row[prompt_col], tokenizer=tokenizer)
        ground_truth = extract_ground_truth(row)

        step_stats = training_step(
            model, tokenizer, optimizer, prompt, ground_truth,
            args.alpha, args.n_rollouts, args.max_new_tokens,
            debug=(step < args.debug_steps),
        )
        print(f"step {step:4d} | loss={step_stats['loss']:.4f} "
              f"agreement={step_stats['agreement']:.3f} "
              f"train_acc={step_stats['true_accuracy']:.3f} "
              f"entropy={step_stats['entropy']:.3f}")

        record = {"step": step, **step_stats}

        if step % args.eval_every == 0:
            eval_stats = evaluate(
                model, tokenizer, test_df, args.n_eval_questions,
                args.n_rollouts, args.max_new_tokens,
            )
            record.update(eval_stats)
            print(f"  [eval] test_acc={eval_stats['test_accuracy']:.3f} "
                  f"gap={eval_stats['agreement_gap']:.3f} "
                  f"entropy={eval_stats['mean_entropy']:.3f}")

        history.append(record)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "history": history}, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
