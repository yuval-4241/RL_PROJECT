"""
OpenAI o4-mini client (api.openai.com directly, not via Groq or any router).
Added as a third strong model: a cost-efficient reasoning model with no
Groq-style daily token cap, on a paid account.

Same on-disk cache format and truncation-escalation retry as groq_client.py.
This account's real limit (200,000 TPM, checked via response headers) is far
more generous than any Groq model used so far, so the throttle mostly stays
out of the way -- included anyway for consistency and because it's cheap
insurance against ever exceeding it under concurrency.

o4-mini is a reasoning model, but unlike Groq's gpt-oss-20b/qwen3-32b (which
need reasoning_format=hidden to keep <think> out of `content`) or the
DeepSeek/Featherless deployment (which inlines the whole reasoning trace into
`content`), OpenAI's API keeps `content` clean by default -- no special
params needed. Reasoning token spend is only visible via
usage.completion_tokens_details.reasoning_tokens, not as text. Answer parsing
uses `content` only, same as every other client.
"""
import json
import os
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from openai import APIConnectionError, APIError, OpenAI, RateLimitError

MODEL = "o4-mini"
MODELS = [MODEL]

DEFAULT_MAX_COMPLETION_TOKENS = 8192
ESCALATED_MAX_COMPLETION_TOKENS = 12288
ESCALATION_MULTIPLIER = 1.5

CACHE_DIR = Path(__file__).resolve().parent / "cache"

MAX_CONCURRENCY = 5
MAX_RETRIES = 8
BASE_BACKOFF_SECONDS = 1.5
MAX_BACKOFF_SECONDS = 30.0

# Real TPM, read from this account's actual x-ratelimit-limit-tokens header.
MODEL_TPM_LIMITS = {MODEL: 200000}
TPM_SAFETY_MARGIN = 0.85
TPM_WINDOW_SECONDS = 60.0
CHARS_PER_TOKEN_ESTIMATE = 4
REQUEST_SIZE_SAFETY_MARGIN = 0.97  # hard per-request cap, independent of the softer pacing target

_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)

_client_singleton: Optional[OpenAI] = None
_client_lock = threading.Lock()

_token_ledger: dict = {}
_token_ledger_lock = threading.Lock()


def _target_tpm(model: str) -> int:
    return int(MODEL_TPM_LIMITS[model] * TPM_SAFETY_MARGIN)


def _hard_completion_cap(model: str, prompt: str) -> int:
    prompt_tokens_estimate = len(prompt) // CHARS_PER_TOKEN_ESTIMATE
    return max(int(MODEL_TPM_LIMITS[model] * REQUEST_SIZE_SAFETY_MARGIN) - prompt_tokens_estimate, 1)


def _prune_ledger(model: str, now: float) -> deque:
    dq = _token_ledger.setdefault(model, deque())
    while dq and now - dq[0][0] > TPM_WINDOW_SECONDS:
        dq.popleft()
    return dq


def _throttle_for_tpm(model: str, estimated_tokens: int) -> None:
    """Same deadlock-safe design as groq_client._throttle_for_tpm: once the
    ledger is empty, let the request through regardless of its own size."""
    target = _target_tpm(model)
    while True:
        with _token_ledger_lock:
            now = time.time()
            dq = _prune_ledger(model, now)
            used = sum(tokens for _, tokens in dq)
            if used + estimated_tokens <= target or not dq:
                return
            wait_s = TPM_WINDOW_SECONDS - (now - dq[0][0])
        time.sleep(max(wait_s, 0.5))


def _record_tokens(model: str, tokens: int) -> None:
    with _token_ledger_lock:
        _prune_ledger(model, time.time()).append((time.time(), tokens))


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


def _build_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in RL_Project/.env or export it before running.")
    return OpenAI(api_key=api_key)


def _get_client() -> OpenAI:
    global _client_singleton
    with _client_lock:
        if _client_singleton is None:
            _client_singleton = _build_client()
        return _client_singleton


def verify_models_exist(models=MODELS) -> None:
    client = _get_client()
    available = {m.id for m in client.models.list().data}
    missing = [m for m in models if m not in available]
    if missing:
        raise RuntimeError(f"OpenAI model(s) not available: {missing}.")
    print(f"Verified {len(models)} OpenAI model(s) exist: {models}")


def _cache_path(model: str, prompt_idx: int, rollout_idx: int) -> Path:
    directory = CACHE_DIR / model.replace("/", "__")
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
        max_completion_tokens = DEFAULT_MAX_COMPLETION_TOKENS

    hard_cap = _hard_completion_cap(model, prompt)
    max_completion_tokens = min(max_completion_tokens, hard_cap)

    client = _get_client()
    current_max_tokens = max_completion_tokens
    escalation_ceiling = min(
        max(ESCALATED_MAX_COMPLETION_TOKENS, int(max_completion_tokens * ESCALATION_MULTIPLIER)),
        hard_cap,
    )
    escalated = False
    last_err = None

    for attempt in range(MAX_RETRIES):
        estimated_tokens = len(prompt) // CHARS_PER_TOKEN_ESTIMATE + current_max_tokens
        _throttle_for_tpm(model, estimated_tokens)
        try:
            # o-series reasoning models only accept temperature=1 (verified:
            # 0.7 gets a 400 "Only the default (1) value is supported").
            # Passed through rather than silently dropped -- callers get a
            # clear error if they ever pass something else, instead of a
            # silently-ignored parameter.
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=current_max_tokens,
                temperature=temperature,
            )
            usage_tokens = resp.usage.total_tokens if resp.usage else estimated_tokens
            _record_tokens(model, usage_tokens)

            finish_reason = resp.choices[0].finish_reason
            if finish_reason == "length" and not escalated and current_max_tokens < escalation_ceiling:
                current_max_tokens = escalation_ceiling
                escalated = True
                continue

            msg = resp.choices[0].message
            result = {
                "model": model,
                "prompt_idx": prompt_idx,
                "rollout_idx": rollout_idx,
                "content": msg.content,
                "raw": resp.model_dump(),
                "max_completion_tokens_used": current_max_tokens,
                "escalated": escalated,
                "truncated": finish_reason == "length",
            }
            cache_file.write_text(json.dumps(result, indent=2))
            return result
        except (RateLimitError, APIConnectionError, APIError) as e:
            last_err = e
            if isinstance(e, RateLimitError):
                _record_tokens(model, estimated_tokens)
            retry_after = _parse_retry_after(e)
            sleep_s = retry_after if retry_after is not None else BASE_BACKOFF_SECONDS * (2**attempt)
            time.sleep(min(sleep_s, MAX_BACKOFF_SECONDS))

    raise RuntimeError(f"OpenAI generation failed after {MAX_RETRIES} retries: {last_err}")


def generate_batch(jobs: list) -> list:
    results = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        futures = {pool.submit(generate, **job): i for i, job in enumerate(jobs)}
        for future, i in futures.items():
            results[i] = future.result()
    return results
