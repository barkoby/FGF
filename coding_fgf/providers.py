"""Shared bounded provider calls. Never changes the requested provider/model."""
from __future__ import annotations
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

class RetryExhausted(RuntimeError):
    """A provider request has already used its complete retry budget."""

def retry_config() -> tuple[float, float, float, int]:
    initial = max(0.1, float(os.getenv("CODING_FGF_API_BACKOFF_INITIAL_SECONDS", "2")))
    maximum = max(initial, float(os.getenv("CODING_FGF_API_BACKOFF_MAX_SECONDS", "120")))
    jitter = max(0.0, float(os.getenv("CODING_FGF_API_BACKOFF_JITTER_SECONDS", "0.5")))
    attempts = int(os.getenv("CODING_FGF_API_MAX_ATTEMPTS", "3"))
    if attempts <= 0:
        raise ValueError("CODING_FGF_API_MAX_ATTEMPTS must be positive")
    return initial, maximum, jitter, attempts

def status_code(error: BaseException) -> int | None:
    value = getattr(error, "status_code", None)
    if value is None:
        value = getattr(getattr(error, "response", None), "status_code", None)
    return value if isinstance(value, int) else None

def is_transient(error: BaseException) -> bool:
    if isinstance(error, RetryExhausted):
        return False
    code = status_code(error)
    if code is not None:
        return code in {408, 409, 429} or 500 <= code < 600
    names = {cls.__name__ for cls in type(error).__mro__}
    if names & {"AuthenticationError", "PermissionDeniedError", "NotFoundError",
                "BadRequestError", "GoogleVertexPermanentError"}:
        return False
    if names & {"Timeout", "TimeoutError", "APITimeoutError", "ConnectTimeout",
                "ReadTimeout", "ConnectionError", "APIConnectionError",
                "RateLimitError", "InternalServerError", "TransportError"}:
        return True
    message = str(error).lower()
    return any(marker in message for marker in
               ("rate limit", "ratelimit", "timed out", "timeout",
                "temporarily unavailable", "service unavailable"))

def request_with_retry(call: Callable[[], Any], *, event_logger=None, label="api") -> Any:
    initial, maximum, jitter, attempts = retry_config()
    delay = initial
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            if not is_transient(exc):
                raise
            if attempt == attempts:
                raise RetryExhausted(f"{label} failed after {attempts} attempts: {type(exc).__name__}") from exc
            if event_logger:
                event_logger(f"{label}:retry:attempt={attempt}:error={type(exc).__name__}")
            time.sleep(delay + random.uniform(0.0, jitter))
            delay = min(maximum, delay * 2)

@dataclass(frozen=True)
class StructuredResult:
    data: Any
    model_used: str
    usage: dict[str, int]
    provider: str

def structured_generate(prompt: str, schema_name: str, model: str, *,
                        provider="openai", google_project="", google_location="",
                        google_credentials="", temperature=None, event_logger=None) -> StructuredResult:
    retry_config()
    output_attempts = int(os.getenv("CODING_FGF_MODEL_OUTPUT_MAX_ATTEMPTS", "6"))
    if output_attempts <= 0:
        raise ValueError("CODING_FGF_MODEL_OUTPUT_MAX_ATTEMPTS must be positive")
    if provider == "google":
        from .google_vertex import generate_json_with_metadata, google_config
        result = generate_json_with_metadata(
            prompt, schema_name, model,
            google_config(google_project or None, google_location or None, google_credentials or None),
            event_logger=event_logger, temperature=0.0 if temperature is None else temperature)
        return StructuredResult(result["data"], model, result["usage"], provider)
    if provider != "openai":
        raise ValueError(f"Unsupported LLM provider: {provider}")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    from openai import OpenAI
    client = OpenAI(max_retries=0, timeout=float(os.getenv("CODING_FGF_OPENAI_TIMEOUT_SECONDS", "90")))
    kwargs = {"model": model, "input": prompt, "text": {"format": {"type": "json_object"}}}
    if temperature is not None:
        kwargs["temperature"] = temperature
    total_usage = {}
    for attempt in range(output_attempts):
        response = request_with_retry(lambda: client.responses.create(**kwargs),
                                      event_logger=event_logger, label=f"openai:{schema_name}")
        raw_usage = getattr(response, "usage", None) or {}
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = raw_usage.get(key) if isinstance(raw_usage, dict) else getattr(raw_usage, key, None)
            if isinstance(value, int):
                total_usage[key] = total_usage.get(key, 0) + value
        text = getattr(response, "output_text", "")
        try:
            if not text:
                text = response.output[0].content[0].text
            data = json.loads(text)
        except (ValueError, IndexError, AttributeError, TypeError) as exc:
            if attempt + 1 == output_attempts:
                raise RuntimeError(f"Invalid JSON after {output_attempts} model-output attempts") from exc
            continue
        return StructuredResult(data, model, total_usage, provider)
    raise RuntimeError("No structured response")
