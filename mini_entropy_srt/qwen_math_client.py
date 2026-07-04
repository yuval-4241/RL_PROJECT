"""
Qwen2.5-Math-7B client, served via Hugging Face's Inference Providers router
(provider: Featherless AI). This is the exact base model (not -Instruct) that
the original SRT repo's srt.sh points at by default (MODEL_PATH=Qwen/Qwen2.5-Math-7B)
and that the paper itself uses as its primary real-math model.

Verified before building: the -Instruct variant garbles into repetitive
nonsense tokens (`SEEK`, `Leone`, `ebx`, ...) on this deployment 20-40% of the
time depending on path (HF router vs. direct Featherless); the base model
showed zero garbling across 5 test rollouts via this exact HF-router path.
Direct Featherless access (separate FEATHERLESS_API_KEY) was tried too but
was less reliable (frequent capacity 503s, and the base model ignored the
actual prompt once) -- the HF router path is what's used here.

Same on-disk cache format and truncation-escalation retry as
deepseek_client.py. No TPM throttle: this backend exposes no rate-limit
headers (checked empirically), so pacing is retry/backoff on actual errors
rather than pre-emptive, same as deepseek_client.py and lab_client.py.
"""
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from openai import APIConnectionError, APIError, OpenAI, RateLimitError

HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"

# Pinned explicitly (not ":fastest") so a provider swap can't silently change
# behavior/pricing underneath us -- Featherless is the only provider anyway.
MODEL = "Qwen/Qwen2.5-Math-7B:featherless-ai"
MODELS = [MODEL]

# Not a reasoning model, but DAPO problems are hard enough that it sometimes
# needs more than a short budget: 2/5 test rollouts hit 1024 tokens without
# finishing. 2048 base + 4096 escalation mirrors the margin used elsewhere.
DEFAULT_MAX_TOKENS = 2048
ESCALATED_MAX_TOKENS = 4096

CACHE_DIR = Path(__file__).resolve().parent / "cache"

MAX_CONCURRENCY = 3
MAX_RETRIES = 8
BASE_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 30.0
REQUEST_TIMEOUT_SECONDS = 180.0

_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)

_client_singleton: Optional[OpenAI] = None
_client_lock = threading.Lock()


def _build_client() -> OpenAI:
    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        raise RuntimeError(
            "HF_TOKEN is not set. Put it in RL_Project/.env or export it before running. "
            "Needs a fine-grained token with 'Make calls to Inference Providers' permission."
        )
    return OpenAI(api_key=api_key, base_url=HF_ROUTER_BASE_URL, timeout=REQUEST_TIMEOUT_SECONDS)


def _get_client() -> OpenAI:
    global _client_singleton
    with _client_lock:
        if _client_singleton is None:
            _client_singleton = _build_client()
        return _client_singleton


def verify_models_exist(models=MODELS) -> None:
    """A real (cheap) completion call, not client.models.list() -- the HF
    router's model list is the full multi-provider catalog, not filtered to
    what this token/provider combination can actually serve."""
    client = _get_client()
    for model in models:
        client.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}], max_tokens=1)
    print(f"Verified {len(models)} model(s) reachable via HF router: {models}")


def _cache_path(model: str, prompt_idx: int, rollout_idx: int) -> Path:
    safe_model = model.replace("/", "__").replace(":", "__")
    directory = CACHE_DIR / safe_model
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{prompt_idx:04d}_{rollout_idx:02d}.json"


def _parse_retry_after(error: Exception) -> Optional[float]:
    response = getattr(error, "response", None)
    header_val = response.headers.get("retry-after") if response is not None else None
    if header_val:
        try:
            return float(header_val)
        except ValueError:
            pass
    match = _RETRY_AFTER_RE.search(str(error))
    return float(match.group(1)) if match else None


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
        except (RateLimitError, APIConnectionError, APIError) as e:
            last_err = e
            retry_after = _parse_retry_after(e)
            sleep_s = retry_after if retry_after is not None else BASE_BACKOFF_SECONDS * (2**attempt)
            time.sleep(min(sleep_s, MAX_BACKOFF_SECONDS))

    raise RuntimeError(f"Qwen-Math (HF router) generation failed after {MAX_RETRIES} retries: {last_err}")


def generate_batch(jobs: list) -> list:
    """jobs: list of kwarg-dicts for generate(). Runs with bounded concurrency."""
    results = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        futures = {pool.submit(generate, **job): i for i, job in enumerate(jobs)}
        for future, i in futures.items():
            results[i] = future.result()
    return results
