"""
DeepSeek-R1-Distill-Qwen-32B client, served via Hugging Face's Inference
Providers router (provider: Featherless AI) -- DeepSeek's own official API
doesn't serve this model (only deepseek-v4-flash/-pro) and Groq doesn't host
any DeepSeek model currently, so this is the actual available path to it.

Same on-disk cache format and truncation-escalation retry as groq_client.py /
lab_client.py. No TPM throttle: this backend exposes no rate-limit headers at
all (checked empirically), so there's nothing to calibrate a throttle target
against -- same situation as the lab server, handled the same way (modest
concurrency, retry/backoff on actual errors instead of pre-emptive pacing).

CRITICAL (this is why gpt-oss-20b's Groq deployment failed): R1-Distill-Qwen
produces a long <think> chain before the final answer. Too small a token
budget truncates mid-reasoning and returns empty/no answer. 8192 is the
default here for that reason. This provider's `content` field, unlike some
reasoning-model deployments, contains the ENTIRE reasoning trace inline
followed by the boxed answer -- NOT split into a separate `reasoning_content`
field (verified empirically: hasattr(message, 'reasoning_content') is False
on this deployment). That's fine either way: extract_boxed_answer() takes the
LAST \\boxed{} in the text regardless of what precedes it. Still coded
defensively below -- if a future provider swap DOES populate
reasoning_content, it's captured for reference but never used for parsing.
"""
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError

HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"

# deepseek-ai/DeepSeek-R1-Distill-Qwen-32B is only live on Featherless AI
# among HF's inference providers (checked via the model's
# inferenceProviderMapping) -- pinned explicitly rather than ":fastest" so a
# provider swap can't silently change behavior/pricing underneath us.
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B:featherless-ai"
MODELS = [MODEL]

DEFAULT_MAX_TOKENS = 8192
ESCALATED_MAX_TOKENS = 12288

CACHE_DIR = Path(__file__).resolve().parent / "cache"

MAX_CONCURRENCY = 3  # no visible rate-limit headers to calibrate against; stay modest
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
    """Fails loudly if the model isn't live for its pinned provider.

    Doesn't use client.models.list() -- the HF router's model list is the
    full multi-provider catalog, not filtered to what THIS token/provider
    combination can actually serve. A real (cheap) completion call is the
    honest check.
    """
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
            msg = resp.choices[0].message
            finish_reason = resp.choices[0].finish_reason

            if (
                finish_reason == "length"
                and not escalated
                and current_max_tokens < ESCALATED_MAX_TOKENS
            ):
                # Cut off mid-<think>, not out of things to say -- retry once
                # with more room instead of caching a truncated non-answer.
                current_max_tokens = ESCALATED_MAX_TOKENS
                escalated = True
                continue

            result = {
                "model": model,
                "prompt_idx": prompt_idx,
                "rollout_idx": rollout_idx,
                "content": msg.content,  # answer parsing MUST use only this field
                "reasoning_content": getattr(msg, "reasoning_content", None),  # reference only, never parsed
                "raw": resp.model_dump(),
                "max_tokens_used": current_max_tokens,
                "escalated": escalated,
                "truncated": finish_reason == "length",
            }
            cache_file.write_text(json.dumps(result, indent=2))
            return result
        except (RateLimitError, APIConnectionError, APITimeoutError, APIError) as e:
            last_err = e
            retry_after = _parse_retry_after(e)
            sleep_s = retry_after if retry_after is not None else BASE_BACKOFF_SECONDS * (2**attempt)
            time.sleep(min(sleep_s, MAX_BACKOFF_SECONDS))

    raise RuntimeError(f"DeepSeek (HF router) generation failed after {MAX_RETRIES} retries: {last_err}")


def generate_batch(jobs: list) -> list:
    """jobs: list of kwarg-dicts for generate(). Runs with bounded concurrency."""
    results = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        futures = {pool.submit(generate, **job): i for i, job in enumerate(jobs)}
        for future, i in futures.items():
            results[i] = future.result()
    return results
