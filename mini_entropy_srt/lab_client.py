"""
Lab GPU server client (Tailscale-only, shared with other lab members).

Same on-disk cache format and truncation-escalation retry as groq_client.py,
so it's a drop-in for run_pilot()'s per-model dispatch. Differs because the
backend differs:
  - No TPM throttle: the server queues excess requests itself rather than
    rate-limiting (see the lab user guide), so there's nothing to throttle
    against -- just a polite concurrency cap.
  - Uses `max_tokens`, not Groq/OpenAI's `max_completion_tokens`.
  - Longer request timeout: a cold model can take 10-60s to load onto the
    GPU, and busy-queue waits can run a few minutes.
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError

LAB_BASE_URL = "http://100.110.96.81:8000/v1"

MODELS = ["qwen2.5-32b"]

DEFAULT_MAX_TOKENS = 1536
ESCALATED_MAX_TOKENS = 2200

CACHE_DIR = Path(__file__).resolve().parent / "cache"

# Polite concurrency cap on a shared GPU -- the server handles fairness via a
# queue, but there's no reason to hammer it with a high worker count.
MAX_CONCURRENCY = 3
MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 30.0
REQUEST_TIMEOUT_SECONDS = 180.0

_client_singleton: Optional[OpenAI] = None
_client_lock = threading.Lock()


def _build_client() -> OpenAI:
    api_key = os.environ.get("LAB_LLM_TOKEN")
    if not api_key:
        raise RuntimeError(
            "LAB_LLM_TOKEN is not set. Put it in RL_Project/.env or export it before running."
        )
    return OpenAI(api_key=api_key, base_url=LAB_BASE_URL, timeout=REQUEST_TIMEOUT_SECONDS)


def _get_client() -> OpenAI:
    global _client_singleton
    with _client_lock:
        if _client_singleton is None:
            _client_singleton = _build_client()
        return _client_singleton


def verify_models_exist(models=MODELS) -> None:
    """Fails loudly if any requested model id isn't currently served by the lab server."""
    client = _get_client()
    available = {m.id for m in client.models.list().data}
    missing = [m for m in models if m not in available]
    if missing:
        raise RuntimeError(
            f"Lab server model(s) not available: {missing}.\nAvailable models: {sorted(available)}"
        )
    print(f"Verified {len(models)} lab-server model(s) exist: {models}")


def _cache_path(model: str, prompt_idx: int, rollout_idx: int) -> Path:
    safe_model = model.replace("/", "__")
    directory = CACHE_DIR / safe_model
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{prompt_idx:04d}_{rollout_idx:02d}.json"


def generate(
    model: str,
    prompt: str,
    prompt_idx: int,
    rollout_idx: int,
    temperature: float,
    max_completion_tokens: Optional[int] = None,
) -> dict:
    """Single cached generation. Same signature as groq_client.generate for drop-in use."""
    cache_file = _cache_path(model, prompt_idx, rollout_idx)
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    if max_completion_tokens is None:
        max_completion_tokens = DEFAULT_MAX_TOKENS

    client = _get_client()
    current_max_tokens = max_completion_tokens
    escalated = False
    last_err = None

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=current_max_tokens,
            )
            finish_reason = resp.choices[0].finish_reason
            if (
                finish_reason == "length"
                and not escalated
                and current_max_tokens < ESCALATED_MAX_TOKENS
            ):
                # Cut off mid-derivation -- retry once with more room instead
                # of caching a truncated non-answer. Fires at most once.
                current_max_tokens = ESCALATED_MAX_TOKENS
                escalated = True
                continue

            result = {
                "model": model,
                "prompt_idx": prompt_idx,
                "rollout_idx": rollout_idx,
                "content": resp.choices[0].message.content,
                "raw": resp.model_dump(),
                "max_tokens_used": current_max_tokens,
                "escalated": escalated,
                "truncated": finish_reason == "length",
            }
            cache_file.write_text(json.dumps(result, indent=2))
            return result
        except (RateLimitError, APIConnectionError, APITimeoutError, APIError) as e:
            last_err = e
            sleep_s = min(BASE_BACKOFF_SECONDS * (2**attempt), MAX_BACKOFF_SECONDS)
            time.sleep(sleep_s)

    raise RuntimeError(f"Lab server generation failed after {MAX_RETRIES} retries: {last_err}")


def generate_batch(jobs: list) -> list:
    """jobs: list of kwarg-dicts for generate(). Runs with bounded concurrency."""
    results = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        futures = {pool.submit(generate, **job): i for i, job in enumerate(jobs)}
        for future, i in futures.items():
            results[i] = future.result()
    return results
