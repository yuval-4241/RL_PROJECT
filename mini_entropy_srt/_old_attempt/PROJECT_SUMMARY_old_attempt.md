# Entropy-Augmented Reward for SRT — Project Summary

**Goal:** Extend Self-Rewarding Training (SRT) with a batch-entropy bonus to prevent reward hacking / reward collapse, evaluated on the DAPO math dataset.

**The problem being solved:** In standard SRT, the self-consistency pseudo-reward is gamed when the model collapses to always emitting the same answer. Every generation matches, so pseudo-reward → 1.0, but true accuracy → 0.

**The fix (core contribution):**

$$R_{total} = R_{consistency} + \alpha \cdot \mathcal{H}_{batch}$$

where $\mathcal{H}_{batch}$ is the normalized Shannon entropy of the answer distribution across a prompt's rollout batch. Collapsing to one answer forces $\mathcal{H}_{batch} = 0$, removing the bonus and breaking the hack.

**Deliverable:** A Jupyter notebook `entropy_reward_experiment.ipynb` in the `srtorigin` repo, structured in 5 steps matching the project plan.

## What each step does

1. **Baseline verification (100-sample test):** Loads DAPO (`ftajwar/deduplicated_dapo_dataset`), extracts a fixed 100-sample test set, loads the model via `transformers` (`AutoModelForCausalLM`), generates 16 rollouts per prompt, and scores them. Visualizes the gap between pseudo-reward and true accuracy (the reward-hack signal).
2. **Custom reward function:** Implements `EntropyAugmentedRewardManager`, extending the repo's `SelfLearningRewardManager`. Parses `\boxed{}` answers, computes P(a), calculates normalized Shannon entropy, and adds the α·H bonus. Includes the exact 2-line patch for `verl/workers/reward_manager/self_learning.py`.
3. **α tuning:** Sweeps α ∈ {0.01, 0.10, 0.50} on the same 100 rollouts (no re-inference), plots accuracy / pseudo-reward / entropy / total reward, plus a diversity audit to ensure high entropy is genuine math, not random-token spamming.
4. **Metric analysis:** Summary table + bar chart comparing baseline vs α runs; auto-selects best α by highest accuracy + lowest KL proxy.
5. **Scale-up:** Curates the "easy one-third" DAPO subset by pass rate, and exports `run_entropy_srt.sh` for full RLOO training with the tuned α.

## Key technical decisions

- **No API keys** — all 3 models are open-weight, loaded locally from HuggingFace Hub.
- **Uses the repo's real scoring functions** (`_default_compute_score`, `_extract_verifiable_part_of_solution` from `verl.utils.reward_score`) so results are directly comparable to the original SRT code.
- Config values (16 generations, temp=1.0, top_p=1.0, max_prompt_length=1024) mirror `experiment_scripts/srt.sh`.

## Models chosen (open, math-strong)

| Model | HF ID |
|---|---|
| Qwen2.5-Math-7B | `Qwen/Qwen2.5-Math-7B` (repo baseline) |
| DeepSeek-R1-Distill-Qwen-7B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |
| NuminaMath-7B-CoT | `AI-MO/NuminaMath-7B-CoT` |

## To reproduce on GPU cluster

1. Apply the two-line patch from Step 2b to `verl/workers/reward_manager/self_learning.py`.
2. Set `TRAIN_DATASET_PATH` and `MODEL_PATH` in `run_entropy_srt.sh`.
3. Run `bash run_entropy_srt.sh` and monitor WandB for accuracy, pseudo-reward, and KL.
