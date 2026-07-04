"""
Groq client: OpenAI-compatible, bounded concurrency, a real per-model TPM
throttle, retry-after-aware backoff, runtime model-existence check, disk
cache keyed by (model, prompt_idx, rollout_idx).

Nothing is ever regenerated once cached -- reruns only fill in what's missing.
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

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# llama-3.1-8b-instant dropped after Day 1: ~44% empty \boxed{} extractions
# and 3% accuracy -- too weak for DAPO, and its blanks would poison the
# entropy/surprisal computation in Day 2.
MODELS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]

# Both reasoning models: set reasoning_format so the visible `content` still
# ends with the final \boxed{} instead of raw <think> traces.
REASONING_MODELS = {"openai/gpt-oss-20b", "qwen/qwen3-32b"}

# DAPO problems often need multi-step algebra before the model reaches
# \boxed{}; 768 was measured to truncate ~1/3 of llama-3.1-8b-instant
# rollouts mid-derivation (finish_reason="length", no boxed answer -> None).
# 1536 is the base budget most questions finish comfortably within.
# Reasoning models burn hidden <think> tokens even though the visible content
# is short, so they get more headroom.
DEFAULT_MAX_COMPLETION_TOKENS = 1536
REASONING_MAX_COMPLETION_TOKENS = 2048

# Per-model overrides for anything that needs a different budget than the
# reasoning/non-reasoning default above. qwen/qwen3-32b measured truncating
# its <think> trace at 1024 tokens before ever reaching an answer; 4096 clears
# that comfortably (verified: real calls finished in ~1800 tokens).
# openai/gpt-oss-20b at 2048 (the old reasoning default) burned its entire
# budget on hidden <think> reasoning and returned EMPTY visible content on
# real DAPO problems (0.6% parse rate) -- 8192 gives it room to both finish
# reasoning and write the boxed answer; under test to see if that's enough.
MODEL_MAX_COMPLETION_TOKENS = {
    "qwen/qwen3-32b": 4096,
    "openai/gpt-oss-20b": 8192,
}

# If a rollout hits finish_reason="length" at the base budget, it's cut off
# mid-derivation, not actually out of things to say -- retry that one rollout
# once at a higher ceiling instead of accepting a truncated non-answer. Only
# the questions that need it pay the extra TPM cost. Computed relative to the
# base budget (see _resolve_max_completion_tokens) so it always exceeds it,
# even for models with a large override like qwen/qwen3-32b.
ESCALATED_MAX_COMPLETION_TOKENS = 2200
ESCALATION_MULTIPLIER = 1.5

CACHE_DIR = Path(__file__).resolve().parent / "cache"

MAX_CONCURRENCY = 3  # bounded hard by the TPM throttle below, not just worker count
MAX_RETRIES = 10
BASE_BACKOFF_SECONDS = 1.5
MAX_BACKOFF_SECONDS = 60.0

# Real per-model TPM throttle. Limits read from this account's actual
# x-ratelimit-limit-tokens response headers (they differ per model), with a
# safety margin so bursts of concurrent requests don't tip over the edge.
MODEL_TPM_LIMITS = {
    "openai/gpt-oss-20b": 8000,
    "qwen/qwen3-32b": 6000,
    "llama-3.3-70b-versatile": 12000,
    "meta-llama/llama-4-scout-17b-16e-instruct": 30000,
}
TPM_SAFETY_MARGIN = 0.85
DEFAULT_TARGET_TPM = 5000  # fallback for any model not in the table above


def _target_tpm(model: str) -> int:
    limit = MODEL_TPM_LIMITS.get(model)
    return int(limit * TPM_SAFETY_MARGIN) if limit else DEFAULT_TARGET_TPM


# A single request whose (prompt + completion) tokens exceed the model's TPM
# limit gets a hard 413 "Request too large" -- immediate and non-retryable,
# unlike a 429 which is transient. Both the base budget AND the escalated
# retry must stay under this, independent of the softer _target_tpm pacing
# above (qwen/qwen3-32b hit this for real: base 4096 was fine alone, but
# escalating to 6144 pushed a single request past its 6000 TPM hard cap).
REQUEST_SIZE_SAFETY_MARGIN = 0.97


def _hard_completion_cap(model: str, prompt: str) -> Optional[int]:
    """Max completion tokens keeping ONE request under the model's real
    per-request TPM ceiling. None if the model has no known hard limit."""
    limit = MODEL_TPM_LIMITS.get(model)
    if limit is None:
        return None
    prompt_tokens_estimate = len(prompt) // CHARS_PER_TOKEN_ESTIMATE
    return max(int(limit * REQUEST_SIZE_SAFETY_MARGIN) - prompt_tokens_estimate, 1)


TPM_WINDOW_SECONDS = 60.0
CHARS_PER_TOKEN_ESTIMATE = 4  # rough pre-flight estimate before we know actual usage

_client_singleton: Optional[OpenAI] = None
_client_lock = threading.Lock()

_token_ledger: dict = {}  # model -> deque[(timestamp, tokens)]
_token_ledger_lock = threading.Lock()

_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)


def _prune_ledger(model: str, now: float) -> deque:
    dq = _token_ledger.setdefault(model, deque())
    while dq and now - dq[0][0] > TPM_WINDOW_SECONDS:
        dq.popleft()
    return dq


def _throttle_for_tpm(model: str, estimated_tokens: int) -> None:
    """Blocks until sending `estimated_tokens` for `model` should stay under its target TPM.

    If a single request's own cost exceeds the entire target (e.g. a large
    token budget on a tight TPM cap), waiting for OTHER usage to free up can
    never satisfy `used + estimated_tokens <= target` even at used=0 -- that
    used to spin forever. Once the ledger is empty (nothing else in flight),
    let it through regardless; it's the best that can be done, and the
    request's own usage will simply decay out of the window naturally."""
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
    header_val = None
    if response is not None:
        header_val = response.headers.get("retry-after")
    if header_val:
        try:
            return float(header_val)
        except ValueError:
            pass
    match = _RETRY_AFTER_RE.search(str(error))
    if match:
        return float(match.group(1))
    return None


def _build_client() -> OpenAI:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Put it in RL_Project/.env or export it before running."
        )
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def _get_client() -> OpenAI:
    global _client_singleton
    with _client_lock:
        if _client_singleton is None:
            _client_singleton = _build_client()
        return _client_singleton


def verify_models_exist(models=MODELS) -> None:
    """Fails loudly if any requested model id isn't currently served by Groq.

    Never silently substitutes a different model.
    """
    client = _get_client()
    available = {m.id for m in client.models.list().data}
    missing = [m for m in models if m not in available]
    if missing:
        raise RuntimeError(
            f"Groq model(s) not available: {missing}.\nAvailable models: {sorted(available)}"
        )
    print(f"Verified {len(models)} Groq models exist: {models}")


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
    """Single cached generation. Returns {"content": str, "raw": dict, ...}."""
    cache_file = _cache_path(model, prompt_idx, rollout_idx)
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    if max_completion_tokens is None:
        if model in MODEL_MAX_COMPLETION_TOKENS:
            max_completion_tokens = MODEL_MAX_COMPLETION_TOKENS[model]
        elif model in REASONING_MODELS:
            max_completion_tokens = REASONING_MAX_COMPLETION_TOKENS
        else:
            max_completion_tokens = DEFAULT_MAX_COMPLETION_TOKENS

    hard_cap = _hard_completion_cap(model, prompt)
    if hard_cap is not None:
        max_completion_tokens = min(max_completion_tokens, hard_cap)

    client = _get_client()

    def _build_kwargs(token_budget: int) -> dict:
        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=token_budget,
        )
        if model in REASONING_MODELS:
            # Not a typed kwarg on this openai-python SDK version's
            # create() signature -- Groq-specific params go through extra_body.
            kwargs["extra_body"] = {"reasoning_format": "hidden"}
        return kwargs

    current_max_tokens = max_completion_tokens
    # Always exceeds the base budget, even for a model with a large override
    # (e.g. qwen/qwen3-32b's 4096 base -> 6144 escalated, not the flat 2200)
    # -- but never past the hard per-request cap computed above.
    escalation_ceiling = max(ESCALATED_MAX_COMPLETION_TOKENS, int(max_completion_tokens * ESCALATION_MULTIPLIER))
    if hard_cap is not None:
        escalation_ceiling = min(escalation_ceiling, hard_cap)
    escalated = False
    last_err = None
    for attempt in range(MAX_RETRIES):
        estimated_tokens = len(prompt) // CHARS_PER_TOKEN_ESTIMATE + current_max_tokens
        _throttle_for_tpm(model, estimated_tokens)
        try:
            resp = client.chat.completions.create(**_build_kwargs(current_max_tokens))
            usage_tokens = resp.usage.total_tokens if resp.usage else estimated_tokens
            _record_tokens(model, usage_tokens)

            finish_reason = resp.choices[0].finish_reason
            if (
                finish_reason == "length"
                and not escalated
                and current_max_tokens < escalation_ceiling
            ):
                # Cut off mid-derivation, not out of things to say -- retry
                # this rollout once with more room instead of caching a
                # truncated non-answer. `escalated` guards against looping;
                # this can fire at most once per generation.
                current_max_tokens = escalation_ceiling
                escalated = True
                continue

            result = {
                "model": model,
                "prompt_idx": prompt_idx,
                "rollout_idx": rollout_idx,
                "content": resp.choices[0].message.content,
                "raw": resp.model_dump(),
                "max_completion_tokens_used": current_max_tokens,
                "escalated": escalated,
                "truncated": finish_reason == "length",
            }
            cache_file.write_text(json.dumps(result, indent=2))
            return result
        except (RateLimitError, APIConnectionError, APIError) as e:
            last_err = e
            # A 429 means our estimate undershot -- charge the ledger for what
            # we attempted so the throttle backs off on the next loop too.
            if isinstance(e, RateLimitError):
                _record_tokens(model, estimated_tokens)
            retry_after = _parse_retry_after(e)
            sleep_s = retry_after if retry_after is not None else BASE_BACKOFF_SECONDS * (2**attempt)
            time.sleep(min(sleep_s, MAX_BACKOFF_SECONDS))

    raise RuntimeError(f"Groq generation failed after {MAX_RETRIES} retries: {last_err}")


def generate_batch(jobs: list) -> list:
    """jobs: list of kwarg-dicts for generate(). Runs with bounded concurrency."""
    results = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        futures = {pool.submit(generate, **job): i for i, job in enumerate(jobs)}
        for future, i in futures.items():
            results[i] = future.result()
    return results
