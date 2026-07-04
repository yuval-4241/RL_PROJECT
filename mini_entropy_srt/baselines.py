"""
Day 1: frozen-model baselines on Groq.

For each model, on a fixed 20-question pilot (nested in the 100-question main
eval set, same seed):
  - zero-shot accuracy (1 sample vs ground truth) -- the floor.
  - majority-vote accuracy (16-rollout consensus vs ground truth).
  - agreement / pseudo-reward (how often rollouts match the majority) -- no
    ground truth needed, this is what SRT trains on.
  - reward-hack gap = agreement - majority_vote_accuracy.

Run with: python -m mini_entropy_srt.baselines
"""
import json
import random
from collections import Counter
from pathlib import Path

import datasets
import matplotlib.pyplot as plt
import pandas as pd

from mini_entropy_srt import deepseek_client, groq_client, lab_client, openai_client, qwen_math_client, repo_utils

# Model -> client module. Anything not listed here is assumed to be a Groq
# model. qwen2.5-32b lives on the lab's own GPU server; DeepSeek and
# Qwen2.5-Math-7B are served via the HF Inference Providers router
# (Featherless AI) since neither is hosted on Groq or their own official API
# reliably; o4-mini is OpenAI's own API directly.
CLIENT_FOR_MODEL = {
    "qwen2.5-32b": lab_client,
    deepseek_client.MODEL: deepseek_client,
    openai_client.MODEL: openai_client,
    qwen_math_client.MODEL: qwen_math_client,
}

# Confirmed clean, complete (20/20 questions) Day 2 model set:
#   - qwen2.5-32b: lab server, 97.2% parse rate.
#   - meta-llama/llama-4-scout-17b-16e-instruct: Groq, 100% parse rate.
#   - o4-mini: OpenAI direct, 95% majority-vote accuracy, negative reward-hack gap.
# Excluded, not deleted from the project:
#   - llama-3.1-8b-instant: dropped after Day 1 (44% empty parses, 3% acc).
#   - llama-3.3-70b-versatile: real but only 100k/day Groq cap, exhausts
#     after ~7 questions; partial data exists, not a clean full run.
#   - openai/gpt-oss-20b: hit a hard 413 at every budget tried (2048/2200:
#     0.6% parse; 8192: over its own 8000 TPM ceiling) -- broken on Groq.
#     Also broken on the lab server (empty content on ~1/3 of real
#     questions even at 16k tokens -- genuine pathological reasoning loops,
#     not a budget problem).
#   - qwen/qwen3-32b: 100% empty content across all cached generations --
#     its real Groq TPM ceiling (6000) is too small for it to finish
#     reasoning on real DAPO problems, same failure class as gpt-oss-20b.
#   - DeepSeek (Featherless via HF router): credits added, but no validated
#     clean smoke test completed yet -- add once tested.
#   - Qwen2.5-Math-7B (Featherless via HF router, base model matching
#     srt.sh's default): client just built, 5/5 clean in ad-hoc testing,
#     pending the same small-scale smoke test as every other model before
#     promotion here.
DEFAULT_MODELS = ["qwen2.5-32b", "meta-llama/llama-4-scout-17b-16e-instruct", openai_client.MODEL]

DATASET_PATH = "ftajwar/deduplicated_dapo_dataset"
SEED = 42
N_MAIN = 100
N_PILOT = 20
N_ROLLOUTS = 16
ZERO_SHOT_TEMPERATURE = 1.0
ROLLOUT_TEMPERATURE = 1.0

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"
RESULTS_DIR = THIS_DIR / "results"
INDICES_FILE = DATA_DIR / "eval_indices.json"

INSTRUCTION_FOLLOWING = "Let's think step by step and output the final answer within \\boxed{}."


def build_eval_indices(dataset_len: int) -> dict:
    """Seeded 100-question main eval set; the 20-pilot is the first 20 of it (nested)."""
    if INDICES_FILE.exists():
        return json.loads(INDICES_FILE.read_text())

    rng = random.Random(SEED)
    all_idx = list(range(dataset_len))
    rng.shuffle(all_idx)
    main_100 = sorted(all_idx[:N_MAIN])
    pilot_20 = main_100[:N_PILOT]

    payload = {"seed": SEED, "main_100": main_100, "pilot_20": pilot_20}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDICES_FILE.write_text(json.dumps(payload, indent=2))
    return payload


def load_pilot_questions():
    dataset = datasets.load_dataset(DATASET_PATH, split="train", trust_remote_code=True)
    indices = build_eval_indices(len(dataset))

    pilot_indices = indices["pilot_20"]
    questions, ground_truths = [], []
    for idx in pilot_indices:
        row = dataset[idx]
        questions.append(row["prompt"] + " " + INSTRUCTION_FOLLOWING)
        ground_truths.append(str(row["answer"]))

    return pilot_indices, questions, ground_truths


def _write_results(rows: list) -> tuple:
    """Writes whatever rows were collected so far -- called after every model
    and again at the end, so a later model crashing never loses earlier ones.

    Merges with whatever's already on disk (keyed by model), so running each
    model as a separate invocation accumulates into one combined results file
    instead of each run overwriting the last one's models."""
    raw_path = RESULTS_DIR / "pilot_raw.csv"
    new_models = {r["model"] for r in rows}

    if raw_path.exists():
        existing = pd.read_csv(raw_path)
        existing = existing[~existing["model"].isin(new_models)]
        df = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) if rows else existing
    elif rows:
        df = pd.DataFrame(rows)
    else:
        return None, None

    if df.empty:
        return None, None

    summary = df.groupby("model").agg(
        zero_shot_accuracy=("zero_shot_correct", "mean"),
        majority_vote_accuracy=("majority_correct", "mean"),
        agreement=("agreement", "mean"),
    )
    summary["reward_hack_gap"] = summary["agreement"] - summary["majority_vote_accuracy"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "pilot_raw.csv", index=False)
    summary.to_csv(RESULTS_DIR / "pilot_summary.csv")

    ax = summary[["zero_shot_accuracy", "majority_vote_accuracy", "agreement"]].plot(
        kind="bar", figsize=(8, 5)
    )
    ax.set_ylabel("Rate")
    ax.set_title("Day 1 pilot (20 questions): zero-shot vs majority-vote vs agreement")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pilot_plot.png")
    plt.close(ax.get_figure())

    return df, summary


def _write_detailed(detail_rows: list) -> None:
    """One row per generation (question text, ground truth, extracted answer,
    correctness, whether it matched the majority) -- the full log behind the
    aggregated numbers in pilot_raw.csv. Same merge-by-model behavior as
    _write_results so separate per-model runs accumulate into one file."""
    detail_path = RESULTS_DIR / "pilot_detailed.csv"
    new_models = {r["model"] for r in detail_rows}

    if detail_path.exists():
        existing = pd.read_csv(detail_path)
        existing = existing[~existing["model"].isin(new_models)]
        df = pd.concat([existing, pd.DataFrame(detail_rows)], ignore_index=True) if detail_rows else existing
    elif detail_rows:
        df = pd.DataFrame(detail_rows)
    else:
        return

    if df.empty:
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.sort_values(["model", "prompt_idx", "rollout_idx"]).to_csv(detail_path, index=False)


def run_pilot(models=None, n_rollouts=N_ROLLOUTS) -> tuple:
    models = models or DEFAULT_MODELS

    for backend in {CLIENT_FOR_MODEL.get(m, groq_client) for m in models}:
        backend.verify_models_exist([m for m in models if CLIENT_FOR_MODEL.get(m, groq_client) is backend])

    prompt_indices, questions, ground_truths = load_pilot_questions()

    rows = []
    detail_rows = []
    failed_models = {}
    for model in models:
        client = CLIENT_FOR_MODEL.get(model, groq_client)
        try:
            for p_idx, question, gt in zip(prompt_indices, questions, ground_truths):
                zero_shot = client.generate(
                    model=model,
                    prompt=question,
                    prompt_idx=p_idx,
                    rollout_idx=0,
                    temperature=ZERO_SHOT_TEMPERATURE,
                )
                zero_shot_answer = repo_utils.extract_boxed_answer(zero_shot["content"])
                zero_shot_correct = repo_utils.score_against_ground_truth(zero_shot["content"], gt)

                rollout_jobs = [
                    dict(
                        model=model,
                        prompt=question,
                        prompt_idx=p_idx,
                        rollout_idx=r + 1,
                        temperature=ROLLOUT_TEMPERATURE,
                    )
                    for r in range(n_rollouts)
                ]
                rollouts = client.generate_batch(rollout_jobs)
                answers = [repo_utils.extract_boxed_answer(r["content"]) for r in rollouts]
                valid_answers = [a for a in answers if a]

                majority_answer = None
                if valid_answers:
                    majority_answer, majority_count = Counter(valid_answers).most_common(1)[0]
                    agreement = majority_count / len(valid_answers)
                    majority_correct = repo_utils.score_against_ground_truth(
                        "\\boxed{" + majority_answer + "}", gt
                    )
                else:
                    agreement, majority_correct = 0.0, 0.0

                rows.append(
                    {
                        "model": model,
                        "prompt_idx": p_idx,
                        "zero_shot_correct": zero_shot_correct,
                        "majority_correct": majority_correct,
                        "agreement": agreement,
                    }
                )

                shared_detail_fields = {
                    "model": model,
                    "prompt_idx": p_idx,
                    "question": question,
                    "ground_truth": gt,
                    "majority_answer": majority_answer,
                    "agreement": agreement,
                    "majority_correct": majority_correct,
                }
                detail_rows.append(
                    {
                        **shared_detail_fields,
                        "rollout_idx": 0,
                        "extracted_answer": zero_shot_answer,
                        "correct": bool(zero_shot_correct),
                        "is_majority_answer": majority_answer is not None and zero_shot_answer == majority_answer,
                    }
                )
                for r_idx, (answer, rollout) in enumerate(zip(answers, rollouts), start=1):
                    rollout_correct = repo_utils.score_against_ground_truth(rollout["content"], gt)
                    detail_rows.append(
                        {
                            **shared_detail_fields,
                            "rollout_idx": r_idx,
                            "extracted_answer": answer,
                            "correct": bool(rollout_correct),
                            "is_majority_answer": majority_answer is not None and answer == majority_answer,
                        }
                    )
        except Exception as e:
            failed_models[model] = str(e)
            print(f"[run_pilot] {model} failed, moving on to the next model: {e}")
            continue
        finally:
            # Write after every model so a later crash never loses earlier results.
            _write_results(rows)
            _write_detailed(detail_rows)

    if failed_models:
        print(f"[run_pilot] Finished with {len(failed_models)} failed model(s): {list(failed_models)}")
        print("[run_pilot] Results below only cover the model(s) that succeeded.")

    _write_detailed(detail_rows)
    return _write_results(rows)


if __name__ == "__main__":
    _, pilot_summary = run_pilot()
    print(pilot_summary)
