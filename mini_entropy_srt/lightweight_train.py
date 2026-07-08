"""
lightweight_train.py — Minimal RLOO self-rewarded training loop.

Avoids verl/vLLM/FSDP entirely. Only depends on torch + transformers, both already
confirmed working on this GPU. Implements the same algorithm as the paper's SRT:
majority-vote self-consistency reward + RLOO leave-one-out advantage, with an optional
per-rollout entropy (surprisal) bonus.

\\boxed{} extraction and ground-truth matching reuse repo_utils (the actual SRT repo's
math_verify-based equivalence checker) -- same as baselines.py/entropy_reward.py -- so
answers like \\boxed{0.5} and \\boxed{1/2} are correctly treated as equal, matching every
other Day 1/2 script.

Usage (both work -- see the sys.path bootstrap below for why):
    python -m mini_entropy_srt.lightweight_train --alpha 0.0 --n_steps 20 --n_rollouts 4   # from RL_Project/
    python lightweight_train.py --alpha 0.0 --n_steps 20 --n_rollouts 4                    # from mini_entropy_srt/
"""

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

# Must be set BEFORE `import torch` -- the CUDA allocator reads this env var once,
# at torch's own import/first-CUDA-init time. Reduces OOMs caused by allocator
# fragmentation (a contiguous block not found even though enough total free memory
# exists) rather than genuinely insufficient memory -- previously only set when
# launching via run_smoke_test.sh, so it silently didn't apply when running this
# script directly. setdefault() so an explicitly-exported value from the shell
# still wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
import torch
import torch.utils.checkpoint as torch_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

# Makes `from mini_entropy_srt import repo_utils` resolve even when this file is run
# as a bare script from inside mini_entropy_srt/ itself (python lightweight_train.py),
# not just via `python -m mini_entropy_srt.lightweight_train` from the parent dir --
# in the bare-script case, Python only puts this file's OWN directory on sys.path,
# so the package containing it isn't otherwise importable.
_PARENT_DIR = Path(__file__).resolve().parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from mini_entropy_srt import repo_utils


def is_placeholder_label(gt):
    """DAPO's unlabeled train set uses a placeholder like 'LABEL_BY_SELF_CONSISTENCY'
    instead of a real answer -- this is INTENTIONAL for self-training (no answer key
    during training). Detect it so we don't compute a meaningless 'accuracy' against it."""
    if gt is None:
        return True
    if isinstance(gt, str) and "SELF_CONSISTENCY" in gt.upper():
        return True
    return False


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


def extract_oracle_ground_truth(row):
    """
    For --oracle mode ONLY: reads the REAL answer, hidden at
    reward_model['solution_hidden_during_training'] -- kept deliberately separate from
    reward_model['ground_truth'] (which stays the 'LABEL_BY_SELF_CONSISTENCY' placeholder)
    so the real answer can't accidentally leak into the normal SRT reward. Same
    numpy/dict defensive unwrapping as extract_ground_truth.
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
        return rm.get("solution_hidden_during_training", None)
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
# Reward: majority-vote + PER-ROLLOUT surprisal bonus (the validated design)
# ---------------------------------------------------------------------------
def compute_rewards(answers: list, ground_truth: str, alpha: float):
    """
    answers: list of extracted answer strings (one per rollout), may contain None
    ground_truth: the true answer for this question (used only for logging/eval,
                  NOT used in the SRT training reward itself)
    alpha: entropy-bonus strength

    Ground-truth comparisons use repo_utils.score_against_ground_truth (the real
    SRT repo's math_verify-based checker), same as baselines.py/entropy_reward.py --
    so e.g. \\boxed{0.5} and \\boxed{1/2} are correctly scored as equal, instead of a
    naive string comparison undercounting correctness on differently-formatted
    but equivalent answers.

    Returns: (rewards, majority_answer, agreement, avg_rollout_accuracy, majority_correct)
      avg_rollout_accuracy : fraction of ALL rollouts individually correct (paper's avg@k) --
                             informational only, NOT the same thing as majority_correct.
      majority_correct     : 1.0/0.0, whether the MAJORITY-VOTE answer itself matches
                             ground truth -- this is what Day 1/2's reward_hack_gap
                             (agreement - majority_vote_accuracy) is defined against;
                             use THIS, not avg_rollout_accuracy, for that comparison.
    """
    n = len(answers)
    valid = [a for a in answers if a is not None]
    if not valid:
        # nothing parsed: zero reward everywhere, zero bonus
        return [0.0] * n, None, 0.0, 0.0, 0.0

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
        if ground_truth is not None and repo_utils.score_against_ground_truth("\\boxed{" + a + "}", ground_truth):
            true_correct += 1

    avg_rollout_accuracy = true_correct / n
    majority_correct = (
        repo_utils.score_against_ground_truth("\\boxed{" + majority_answer + "}", ground_truth)
        if ground_truth is not None
        else 0.0
    )
    return rewards, majority_answer, agreement, avg_rollout_accuracy, majority_correct


def compute_oracle_rewards(answers: list, ground_truth: str):
    """
    Ground-truth oracle reward (classic verifiable-reward RL): reward_i = 1.0 if
    rollout i's answer matches the REAL ground truth, 0.0 otherwise -- no majority
    vote, no entropy bonus, no self-consistency involved at all. This is the ceiling
    baseline: what training looks like if the reward signal were perfect, instead of
    the majority-vote proxy every other run in this project uses.

    Still computes agreement/majority_answer purely as DIAGNOSTIC stats (never used
    in the reward itself) so this run's output has the same shape as compute_rewards()
    and can be logged/plotted the same way as every other baseline.

    Returns: (rewards, majority_answer, agreement, avg_rollout_accuracy, majority_correct)
    """
    n = len(answers)
    valid = [a for a in answers if a is not None]

    rewards = []
    true_correct = 0
    for a in answers:
        if a is None or ground_truth is None:
            rewards.append(0.0)
            continue
        correct = repo_utils.score_against_ground_truth("\\boxed{" + a + "}", ground_truth)
        rewards.append(1.0 if correct else 0.0)
        if correct:
            true_correct += 1
    avg_rollout_accuracy = true_correct / n if n else 0.0

    majority_answer, agreement, majority_correct = None, 0.0, 0.0
    if valid:
        counts = Counter(valid)
        majority_answer, majority_count = counts.most_common(1)[0]
        agreement = majority_count / len(valid)
        majority_correct = (
            repo_utils.score_against_ground_truth("\\boxed{" + majority_answer + "}", ground_truth)
            if ground_truth is not None
            else 0.0
        )

    return rewards, majority_answer, agreement, avg_rollout_accuracy, majority_correct


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
PROBE_ALPHA = 0.1  # fixed, comfortably-sized -- deliberately independent of the
# user's real --alpha. This check validates the REWARD/ADVANTAGE CODE is wired
# correctly, not what the user intends to train with -- if it only ran when
# --alpha happened to be nonzero, a baseline run (--alpha 0.0, the "plain SRT"
# control condition) would never verify the bonus mechanism before you also
# launch the paired --alpha>0 run, possibly days apart.


def gradient_invariance_check(alpha: float):
    """
    Confirms the per-rollout bonus actually changes the RLOO advantages
    (i.e. the zero-gradient bug is NOT present). Uses a synthetic example
    mirroring idx 28 from the Day-2 analysis (majority wrong, one correct rare answer).

    Always runs using PROBE_ALPHA, regardless of the user's real --alpha (including
    0.0) -- see PROBE_ALPHA's comment for why.
    """
    print(f"[gradient check] real training alpha={alpha} "
          f"(probe alpha={PROBE_ALPHA} used for this check, independent of the above)")

    answers = ["6", "6", "6", "6", "6", "3", "6", "9/2"]  # majority "6" is wrong; "3" is correct
    ground_truth = "3"

    rewards_no_bonus, _, _, _, _ = compute_rewards(answers, ground_truth, alpha=0.0)
    rewards_with_bonus, _, _, _, _ = compute_rewards(answers, ground_truth, alpha=PROBE_ALPHA)

    adv_no_bonus = rloo_advantages(torch.tensor(rewards_no_bonus))
    adv_with_bonus = rloo_advantages(torch.tensor(rewards_with_bonus))

    # index of the lone correct rollout ("3")
    idx_correct = answers.index("3")
    a0 = adv_no_bonus[idx_correct].item()
    a1 = adv_with_bonus[idx_correct].item()

    print(f"[gradient check] advantage for rare-correct rollout: "
          f"no-bonus={a0:.4f}  with-bonus(probe alpha={PROBE_ALPHA})={a1:.4f}")
    if math.isclose(a0, a1, abs_tol=1e-6):
        raise RuntimeError(
            "GRADIENT INVARIANCE CHECK FAILED: advantages identical with/without bonus. "
            "The entropy bonus is being cancelled (zero-gradient bug). STOPPING before training."
        )
    print("[gradient check] PASSED — bonus changes the advantage. Safe to proceed.\n")


# ---------------------------------------------------------------------------
# Rollout generation (plain transformers .generate(), no vLLM)
# ---------------------------------------------------------------------------
def generate_rollouts(model, tokenizer, prompt: str, n_rollouts: int,
                       base_max_tokens: int = 2048, escalated_max_tokens: int = 4096,
                       batch_generation: bool = False):
    """
    Generate n_rollouts completions.

    Default (batch_generation=False): ONE AT A TIME (num_return_sequences=1 per call)
    rather than batching all n_rollouts into a single .generate() call. HF's .generate()
    runs a normal forward pass over the full prompt for its first ("prefill") step, and
    that forward pass ALSO unconditionally upcasts its logits to fp32 (same underlying
    behavior as the training forward pass, just inside a different code path) --
    batching n_rollouts copies of the prompt together multiplies that tensor by
    n_rollouts, which can OOM for long-enough prompts even though generation itself
    runs under torch.no_grad() (no backward buffers needed, but the forward-pass peak
    is still real). Processing one rollout at a time bounds that peak to a single
    prompt's prefill, independent of n_rollouts -- needed on memory-constrained GPUs
    (the original 11GB SLURM/lab-server cards this project was built against).

    batch_generation=True: all n_rollouts in ONE .generate() call (num_return_sequences=
    n_rollouts). Plain (non-batched) autoregressive generation never fully saturates a
    modern GPU's compute -- each token depends on the previous one, so there's limited
    parallelism within a single sequence. Batching multiple sequences together lets the
    GPU process them in parallel instead, meaningfully faster wall-clock time -- but
    reintroduces the prefill-memory-multiplies-by-n_rollouts risk above, so only safe
    on GPUs with comfortable VRAM headroom (e.g. dedicated rented GPUs, not the original
    11GB constrained cards this project fought OOMs on all session).

    Uses base_max_tokens by default; any rollout that gets cut off (no EOS token
    reached, i.e. truncated mid-generation) is automatically regenerated ONCE at
    escalated_max_tokens. This matches the Day-1/2 pilot's escalation policy: default
    2048, escalate to 4096 only when truncated.

    Gradient checkpointing (needed for the TRAINING forward pass) forces
    use_cache=False inside .generate(), which disables KV-caching and makes
    autoregressive generation reprocess the whole sequence for every new token --
    dramatically slower, especially on older GPUs with weak fp16 throughput
    (e.g. Pascal/1080 Ti, no Tensor Cores). Generation does no backprop, so
    checkpointing buys nothing here -- disabled for generation, restored after.
    """
    was_checkpointing = model.is_gradient_checkpointing
    if was_checkpointing:
        model.gradient_checkpointing_disable()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    eos_id = tokenizer.eos_token_id

    if batch_generation:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=base_max_tokens,
                do_sample=True,
                temperature=1.0,
                num_return_sequences=n_rollouts,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        truncated_mask = [eos_id not in seq[prompt_len:].tolist() for seq in outputs]
        n_truncated = sum(truncated_mask)

        if n_truncated > 0:
            with torch.no_grad():
                escalated_outputs = model.generate(
                    **inputs,
                    max_new_tokens=escalated_max_tokens,
                    do_sample=True,
                    temperature=1.0,
                    num_return_sequences=n_truncated,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            esc_iter = iter(escalated_outputs)
            max_len = max(outputs.shape[1], escalated_outputs.shape[1])
            padded = torch.full((outputs.shape[0], max_len), tokenizer.pad_token_id,
                                 dtype=outputs.dtype, device=outputs.device)
            padded[:, : outputs.shape[1]] = outputs
            for i, was_truncated in enumerate(truncated_mask):
                if was_truncated:
                    esc_seq = next(esc_iter)
                    padded[i, : esc_seq.shape[0]] = esc_seq
            outputs = padded
        all_outputs = list(outputs)
    else:
        all_outputs = []
        for _ in range(n_rollouts):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=base_max_tokens,
                    do_sample=True,
                    temperature=1.0,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            all_outputs.append(out[0])
            torch.cuda.empty_cache()

        # Detect truncation: a rollout is truncated if it never produced the EOS token
        # within the generated span (i.e. it used the full budget without stopping naturally).
        truncated_mask = [eos_id not in seq[prompt_len:].tolist() for seq in all_outputs]
        n_truncated = sum(truncated_mask)

        if n_truncated > 0:
            # Re-generate ONLY the truncated rollouts, one at a time, at the escalated budget
            for i, was_truncated in enumerate(truncated_mask):
                if not was_truncated:
                    continue
                with torch.no_grad():
                    esc_out = model.generate(
                        **inputs,
                        max_new_tokens=escalated_max_tokens,
                        do_sample=True,
                        temperature=1.0,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )
                all_outputs[i] = esc_out[0]
                torch.cuda.empty_cache()

    if was_checkpointing:
        model.gradient_checkpointing_enable()  # restore for the training forward pass

    max_len = max(o.shape[0] for o in all_outputs)
    outputs = torch.full((n_rollouts, max_len), tokenizer.pad_token_id,
                          dtype=all_outputs[0].dtype, device=all_outputs[0].device)
    for i, o in enumerate(all_outputs):
        outputs[i, : o.shape[0]] = o

    texts = [
        tokenizer.decode(seq[prompt_len:], skip_special_tokens=True) for seq in outputs
    ]
    return texts, outputs, n_truncated


# Tokens per chunk when computing the vocab-projection + log-softmax inside
# compute_sequence_logprobs. HF's built-in ForCausalLM.forward() computes this over
# the FULL sequence length in one shot and unconditionally upcasts to fp32 (baked into
# Qwen2's own code, not something we control) -- at escalated sequence lengths
# (~4096 tokens) x ~150k vocab x 4 bytes, that single tensor is ~2.5GB, enough to OOM
# an 11GB GPU once the model + SGD momentum state are also resident. Chunking bounds
# that tensor's size to LOGIT_CHUNK_SIZE regardless of total sequence length.
LOGIT_CHUNK_SIZE = 512


def _chunk_logprob_fn(lm_head, hidden_chunk, target_chunk):
    logits_chunk = lm_head(hidden_chunk).float()
    log_probs_chunk = torch.log_softmax(logits_chunk, dim=-1)
    return log_probs_chunk.gather(2, target_chunk.unsqueeze(-1)).squeeze(-1)


def compute_sequence_logprobs(model, tokenizer, sequences: torch.Tensor, prompt_len: int):
    """Forward pass WITH gradients on the already-generated sequences; sum log-probs
    of the generated (post-prompt) tokens only.

    Runs the transformer body once (its own gradient checkpointing applies exactly as
    before), then applies the vocab projection (lm_head) + log-softmax in
    LOGIT_CHUNK_SIZE-token chunks via torch.utils.checkpoint, instead of materializing
    the full (seq_len x vocab_size) logits tensor at once. Each chunk's logits are
    discarded immediately after computing its log-probs and only recomputed (cheaply --
    just lm_head + softmax, not the whole transformer body) if/when the eventual single
    .backward() call reaches that chunk. Mathematically identical to the non-chunked
    version; only peak memory differs.

    IMPORTANT: sequences come from model.generate() under torch.no_grad(), so they are
    plain integer token ids with no grad history attached (which is correct -- you can't
    backprop through discrete sampling). Gradients flow through the MODEL'S PARAMETERS
    via this fresh forward pass, not through the sequence tensor itself. Make sure the
    model is in train() mode (not eval()) so dropout/gradient-checkpointing behave correctly.
    """
    model.train()
    sequences = sequences.detach()  # ensure no stale graph from generation is attached
    attention_mask = (sequences != tokenizer.pad_token_id).long()

    hidden_states = model.model(
        input_ids=sequences, attention_mask=attention_mask, use_cache=False
    ).last_hidden_state
    hidden_states = hidden_states[:, :-1, :]  # predict token t+1 from position t
    targets = sequences[:, 1:]

    seq_len = hidden_states.shape[1]
    gen_mask = torch.zeros(seq_len, dtype=torch.bool, device=sequences.device)
    gen_mask[prompt_len - 1 :] = True  # only the GENERATED portion (after the prompt)

    chunk_sums = []
    for start in range(0, seq_len, LOGIT_CHUNK_SIZE):
        end = min(start + LOGIT_CHUNK_SIZE, seq_len)
        if not gen_mask[start:end].any():
            continue  # pure-prompt chunk contributes nothing -- skip its lm_head pass entirely
        chunk_log_probs = torch_checkpoint.checkpoint(
            _chunk_logprob_fn,
            model.lm_head,
            hidden_states[:, start:end, :],
            targets[:, start:end],
            use_reentrant=False,
        )
        chunk_mask = gen_mask[start:end].unsqueeze(0)
        chunk_sums.append((chunk_log_probs * chunk_mask).sum(dim=1))

    if not chunk_sums:
        return torch.zeros(sequences.shape[0], device=sequences.device)
    seq_log_probs = torch.stack(chunk_sums, dim=0).sum(dim=0)
    return seq_log_probs


# ---------------------------------------------------------------------------
# One training step: generate -> reward -> advantage -> policy-gradient update
# ---------------------------------------------------------------------------
def training_step(model, tokenizer, optimizer, prompt, ground_truth, alpha, n_rollouts,
                   base_max_tokens, escalated_max_tokens, debug=False, oracle=False,
                   batch_generation=False):
    texts, gen_outputs, n_truncated = generate_rollouts(
        model, tokenizer, prompt, n_rollouts, base_max_tokens, escalated_max_tokens,
        batch_generation=batch_generation,
    )
    answers = [repo_utils.extract_boxed_answer(t) for t in texts]

    if debug:
        print(f"  [debug] ground_truth={ground_truth!r}  truncated_and_escalated={n_truncated}/{n_rollouts}")
        for i, (t, a) in enumerate(zip(texts, answers)):
            print(f"  [debug] rollout {i}: extracted={a!r} | raw_tail={t[-150:]!r}")

    if oracle:
        rewards, majority_answer, agreement, avg_rollout_accuracy, _ = compute_oracle_rewards(answers, ground_truth)
    else:
        rewards, majority_answer, agreement, avg_rollout_accuracy, _ = compute_rewards(answers, ground_truth, alpha)
    train_acc_is_meaningful = not is_placeholder_label(ground_truth)

    advantages = rloo_advantages(torch.tensor(rewards, dtype=torch.float32, device=model.device))

    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]

    # Forward+backward ONE ROLLOUT AT A TIME rather than batched together. Qwen2's own
    # forward() upcasts the full per-position logits to fp32 (logits = logits.float(),
    # baked into the model, not something we control) -- that tensor scales with
    # batch_size x seq_len x vocab_size, and this vocab is ~150k. Batching all rollouts
    # together OOM'd on longer (escalated) sequences. Gradients are additive, so
    # accumulating per-rollout .backward() calls before one optimizer.step() gives the
    # mathematically identical result to the batched version -- just n_rollouts separate
    # smaller forward passes instead of one big one.
    optimizer.zero_grad()
    total_loss = 0.0
    n = len(gen_outputs)
    for i in range(n):
        log_prob_i = compute_sequence_logprobs(model, tokenizer, gen_outputs[i : i + 1], prompt_len)
        loss_i = -(advantages[i].detach() * log_prob_i).mean() / n
        loss_i.backward()
        total_loss += loss_i.item()
        del log_prob_i, loss_i
        torch.cuda.empty_cache()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Release cached-but-unused CUDA memory. On a tight 11GB GPU with a long
    # multi-day run ahead, fragmentation compounds step over step even when
    # each individual step would otherwise fit -- this is what caused the OOM
    # inside generate()'s prefill forward pass on a later step.
    torch.cuda.empty_cache()

    mean_entropy = 0.0
    valid_answers = [a for a in answers if a is not None]
    if valid_answers:
        counts = Counter(valid_answers)
        total = len(valid_answers)
        mean_entropy = -sum((c / total) * math.log(c / total) for c in counts.values())

    return {
        "loss": total_loss,
        "agreement": agreement,
        "true_accuracy": avg_rollout_accuracy if train_acc_is_meaningful else None,
        "entropy": mean_entropy,
    }


# ---------------------------------------------------------------------------
# Held-out evaluation
# ---------------------------------------------------------------------------
def evaluate(model, tokenizer, test_df, n_eval_questions, n_rollouts, base_max_tokens, escalated_max_tokens,
             batch_generation=False):
    """test_accuracy/agreement_gap use MAJORITY-VOTE correctness (majority_correct),
    matching Day 1/2's reward_hack_gap = agreement - majority_vote_accuracy exactly --
    NOT avg_rollout_accuracy (avg@k), which is a different, always-more-pessimistic
    number the paper itself tracks as a separate plot (Fig 3 vs Fig 4).

    zero_shot_accuracy is no longer computed here -- already-saved runs (e.g.
    alpha=0.5's completed 300-step result) have it for reference; not needed going
    forward."""
    model.eval()
    accs, gaps, ents = [], [], []
    sample = test_df.sample(min(n_eval_questions, len(test_df)), random_state=42)
    for _, row in sample.iterrows():
        prompt = extract_prompt_text(row["prompt"], tokenizer=tokenizer)
        texts, _, _ = generate_rollouts(model, tokenizer, prompt, n_rollouts, base_max_tokens, escalated_max_tokens,
                                         batch_generation=batch_generation)
        answers = [repo_utils.extract_boxed_answer(t) for t in texts]
        ground_truth = extract_ground_truth(row)
        _, majority_answer, agreement, _, majority_correct = compute_rewards(
            answers, ground_truth, alpha=0.0
        )
        valid_answers = [a for a in answers if a is not None]
        entropy = 0.0
        if valid_answers:
            counts = Counter(valid_answers)
            total = len(valid_answers)
            entropy = -sum((c / total) * math.log(c / total) for c in counts.values())
        accs.append(majority_correct)
        gaps.append(agreement - majority_correct)
        ents.append(entropy)
        torch.cuda.empty_cache()  # same fragmentation mitigation as training_step
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
    parser.add_argument("--oracle", action="store_true",
                         help="Ground-truth oracle baseline: reward = 1.0 if a rollout's "
                              "answer matches the REAL answer (hidden at "
                              "reward_model.solution_hidden_during_training), 0.0 "
                              "otherwise -- no majority vote, no entropy bonus. --alpha "
                              "is ignored when this is set.")
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--n_rollouts", type=int, default=4)
    parser.add_argument("--base_max_tokens", type=int, default=2048,
                         help="Default generation token budget (Day-1/2 convention).")
    parser.add_argument("--escalated_max_tokens", type=int, default=4096,
                         help="Token budget used to auto-retry ONLY the rollouts that "
                              "got truncated at base_max_tokens (Day-1/2 convention).")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--n_eval_questions", type=int, default=10)
    parser.add_argument("--debug_steps", type=int, default=2,
                         help="Print raw generated text + extracted answers for the "
                              "first N training steps, to diagnose empty-parse issues.")
    parser.add_argument("--output", type=str, default="results/lightweight_run.json")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                         help="Where to save/load model+optimizer state for crash resume. "
                              "Defaults to --output with a .pt extension. If this file AND "
                              "--output both already exist when the run starts, training "
                              "resumes from the checkpointed step instead of starting over.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for training-question sampling (reproducibility).")
    parser.add_argument("--eval_indices_path", type=str,
                         default=str(Path.home() / "RL_Project/mini_entropy_srt/data/eval_indices.json"),
                         help="Path to Day-1's eval_indices.json; those rows are EXCLUDED from "
                              "training to avoid train/eval leakage.")
    parser.add_argument("--final_eval_all", action="store_true",
                         help="After training finishes, run ONE extra evaluation pass over "
                              "ALL test rows (not just --n_eval_questions) on the final trained "
                              "model. The during-training checks repeatedly sample the SAME "
                              "fixed small subset (by design, for a fair before/after trend) -- "
                              "this gives a separate, less sample-biased number for the final "
                              "report headline, at the cost of extra time (proportional to the "
                              "full test set instead of --n_eval_questions).")
    parser.add_argument("--batch_generation", action="store_true",
                         help="Generate all n_rollouts in one batched .generate() call "
                              "instead of one at a time. Meaningfully faster (plain "
                              "autoregressive generation rarely saturates a modern GPU's "
                              "compute when run one sequence at a time), but reintroduces "
                              "a prefill-memory-multiplies-by-n_rollouts risk -- only use "
                              "on GPUs with comfortable VRAM headroom (e.g. a dedicated "
                              "rented GPU), not the original 11GB constrained cards this "
                              "project was built against.")
    args = parser.parse_args()

    import random as _random
    _random.seed(args.seed)
    import numpy as _np
    _np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"Seed set to {args.seed} (matches Day-1 pilot convention: seed=42, "
          f"reproducible sampling).")

    mode_str = "ORACLE (ground-truth reward, alpha ignored)" if args.oracle else f"alpha={args.alpha}"
    print(f"=== Lightweight RLOO training: {mode_str}, n_steps={args.n_steps} ===\n")

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

    # Crash-resume: if both the checkpoint and a prior results file exist, load the
    # trained weights/optimizer state and continue from the next step instead of
    # restarting from the frozen pretrained model. Without this, every OOM/crash lost
    # all trained progress even though the LOGGED metrics were already safe via
    # save_progress() -- this repeatedly cost real GPU-hours on the shared lab server.
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else Path(args.output).with_suffix(".pt")
    out_path_for_resume_check = Path(args.output)
    resuming = checkpoint_path.exists() and out_path_for_resume_check.exists()
    start_step = 0
    if resuming:
        print(f"Found existing checkpoint at {checkpoint_path} -- resuming instead of starting over.")
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"] + 1
        with open(out_path_for_resume_check) as f:
            resumed_results = json.load(f)
        pretrain_eval = resumed_results["pretrain_eval"]
        history = resumed_results["history"]
        print(f"Resuming at step {start_step} (already had {len(history)} completed steps logged).")

    print("Loading data...")
    train_df = pd.read_parquet(args.train_parquet)
    test_df = pd.read_parquet(args.test_parquet)

    # Exclude Day-1 eval-set rows from training to avoid train/eval leakage.
    eval_path = Path(args.eval_indices_path)
    if eval_path.exists():
        with open(eval_path) as f:
            eval_data = json.load(f)
        # Verified against the real eval_indices.json on disk: the correct key is
        # "main_100" (the full 100-question eval pool Day 1 carved out; pilot_20 is
        # nested inside it, so excluding main_100 covers both).
        eval_indices = eval_data.get("main_100", [])

        # eval_indices.json's numbers are row positions in the ORIGINAL
        # ftajwar/deduplicated_dapo_dataset, not necessarily in train_parquet --
        # dapo_unlabeled/train.parquet doesn't exist yet (as of this audit), so
        # whether positional exclusion is even valid depends entirely on how
        # curate_and_export.py (not yet written) constructs it. Prefer an explicit
        # index column if the export preserved one; else fall back to positional
        # exclusion with a loud warning so this assumption isn't silently trusted.
        index_col = next((c for c in ("original_dapo_idx", "dapo_idx", "source_idx") if c in train_df.columns), None)
        before = len(train_df)
        if index_col is not None:
            train_df = train_df[~train_df[index_col].isin(eval_indices)].reset_index(drop=True)
            print(f"Excluded {before - len(train_df)} Day-1 eval rows from training pool "
                  f"via explicit '{index_col}' column (loaded from {eval_path}).")
        else:
            train_df = train_df.reset_index(drop=True)
            train_df = train_df.drop(index=[i for i in eval_indices if i < len(train_df)], errors="ignore")
            print(f"Excluded {before - len(train_df)} Day-1 eval rows from training pool "
                  f"(loaded from {eval_path}).\n"
                  f"WARNING: no explicit source-index column found on train_parquet -- this "
                  f"exclusion assumed row POSITION in train_parquet matches the original "
                  f"dataset index in eval_indices.json. If curate_and_export.py filters, "
                  f"reorders, or subsets rows, this assumption is WRONG and this exclusion "
                  f"is not actually leak-proof. Verify, or add one of "
                  f"('original_dapo_idx', 'dapo_idx', 'source_idx') to the export.")
    else:
        print(f"WARNING: eval_indices_path not found at {eval_path} -- "
              f"could not exclude Day-1 eval rows. Proceeding with full train set "
              f"(risk of train/eval overlap if eval questions come from this same file).")

    prompt_col = "prompt" if "prompt" in train_df.columns else train_df.columns[0]
    print(f"Train rows (after exclusion): {len(train_df)}, Test rows: {len(test_df)}")

    if not resuming:
        # Pre-training baseline eval removed (was: evaluate() on the frozen model before
        # any optimizer.step()). No longer computed -- the periodic during-training evals
        # (step 0 onward) are kept for the trend curve, just no separate "before" snapshot.
        pretrain_eval = None
        history = []

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    final_eval_all_result = None

    def save_progress():
        # Called after EVERY step, not just at the end -- a long run that crashes
        # (OOM, node preemption, etc.) partway through previously lost ALL completed
        # steps, since the only json.dump() call was after the full loop. Overwrites
        # the same file each time, so this is always the latest complete state.
        with open(out_path, "w") as f:
            json.dump({
                "args": vars(args),
                "pretrain_eval": pretrain_eval,
                "history": history,
                "final_eval_all": final_eval_all_result,
            }, f, indent=2)

    def save_checkpoint(step: int):
        # Companion to save_progress(): that saves the LOGGED metrics, this saves the
        # actual trained weights/optimizer state, so a crash can resume training itself
        # instead of only preserving the numbers from steps already completed.
        torch.save(
            {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "step": step},
            checkpoint_path,
        )

    for step in range(start_step, args.n_steps):
        row = train_df.sample(1, random_state=args.seed + step).iloc[0]
        prompt = extract_prompt_text(row[prompt_col], tokenizer=tokenizer)
        ground_truth = extract_oracle_ground_truth(row) if args.oracle else extract_ground_truth(row)

        step_stats = training_step(
            model, tokenizer, optimizer, prompt, ground_truth,
            args.alpha, args.n_rollouts, args.base_max_tokens, args.escalated_max_tokens,
            debug=(step < args.debug_steps), oracle=args.oracle,
            batch_generation=args.batch_generation,
        )
        acc_str = f"{step_stats['true_accuracy']:.3f}" if step_stats['true_accuracy'] is not None else "N/A (unlabeled)"
        print(f"step {step:4d} | loss={step_stats['loss']:.4f} "
              f"agreement={step_stats['agreement']:.3f} "
              f"train_acc={acc_str} "
              f"entropy={step_stats['entropy']:.3f}")

        record = {"step": step, **step_stats}

        if step % args.eval_every == 0:
            eval_stats = evaluate(
                model, tokenizer, test_df, args.n_eval_questions,
                args.n_rollouts, args.base_max_tokens, args.escalated_max_tokens,
                batch_generation=args.batch_generation,
            )
            record.update(eval_stats)
            print(f"  [eval] majority_acc={eval_stats['test_accuracy']:.3f} "
                  f"gap={eval_stats['agreement_gap']:.3f} "
                  f"entropy={eval_stats['mean_entropy']:.3f}")

        history.append(record)
        save_progress()
        save_checkpoint(step)

    if args.final_eval_all:
        print(f"\nRunning final evaluation on all {len(test_df)} test rows "
              f"(final trained model, separate from the fixed {args.n_eval_questions}-question "
              f"trend checks during training)...")
        final_eval_all_result = evaluate(
            model, tokenizer, test_df, len(test_df),
            args.n_rollouts, args.base_max_tokens, args.escalated_max_tokens,
            batch_generation=args.batch_generation,
        )
        print(f"  [final, all {len(test_df)} questions] "
              f"majority_acc={final_eval_all_result['test_accuracy']:.3f} "
              f"gap={final_eval_all_result['agreement_gap']:.3f} "
              f"entropy={final_eval_all_result['mean_entropy']:.3f}")
        save_progress()

    print(f"\nSaved results to {out_path}")
    print(f"Training complete -- {checkpoint_path} can be deleted, or left in place "
          f"(a future run with the same --output would otherwise treat it as a resume point).")


if __name__ == "__main__":
    main()
