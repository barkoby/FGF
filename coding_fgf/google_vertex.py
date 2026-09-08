from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

import requests

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


class GoogleVertexError(RuntimeError):
    pass


class GoogleVertexPermanentError(GoogleVertexError):
    pass


@dataclass(frozen=True)
class GoogleVertexConfig:
    project: str
    location: str = "global"
    credentials: str = ""
    timeout_seconds: float = 180.0


def google_config(
    project: str | None = None,
    location: str | None = None,
    credentials: str | None = None,
) -> GoogleVertexConfig:
    project = project or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_PROJECT")
    if not project or project.startswith(chr(36) + "{"):
        raise ValueError("A Google project is required; set GOOGLE_CLOUD_PROJECT or --google-project")
    credentials = credentials or os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    if credentials and not os.path.isfile(credentials):
        raise ValueError("Google credentials file not found; use its path inside the container")
    return GoogleVertexConfig(
        project=project,
        location=location or os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("GOOGLE_LOCATION") or "global",
        credentials=credentials or os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""),
        timeout_seconds=float(os.getenv("CODING_FGF_GOOGLE_TIMEOUT_SECONDS", "180")),
    )


from .providers import retry_config as _retry_config, request_with_retry


def _model_output_max_attempts() -> int:
    value = int(os.getenv("CODING_FGF_MODEL_OUTPUT_MAX_ATTEMPTS", "6"))
    if value <= 0:
        raise ValueError("CODING_FGF_MODEL_OUTPUT_MAX_ATTEMPTS must be positive")
    return value


def _sleep(delay: float, jitter: float) -> None:
    time.sleep(delay + (random.uniform(0.0, jitter) if jitter else 0.0))


def _access_token(config: GoogleVertexConfig) -> str:
    try:
        import google.auth
        from google.auth.transport.requests import Request
    except Exception as exc:  # pragma: no cover - exercised by environment
        raise GoogleVertexPermanentError("google-auth is required for Google Vertex AI mode") from exc

    from google.auth.exceptions import DefaultCredentialsError, RefreshError
    scopes = [CLOUD_PLATFORM_SCOPE]
    try:
        if config.credentials:
            credentials, _ = google.auth.load_credentials_from_file(config.credentials, scopes=scopes)
        else:
            credentials, _ = google.auth.default(scopes=scopes)
        credentials.refresh(Request())
    except (DefaultCredentialsError, RefreshError) as exc:
        error = GoogleVertexPermanentError(
            "Google ADC credentials are missing or invalid; configure GOOGLE_APPLICATION_CREDENTIALS "
            "to a readable container path or provide working application-default credentials")
        if getattr(exc, "retryable", False):
            error.status_code = 503
        raise error from exc
    token = getattr(credentials, "token", None)
    if not token:
        raise GoogleVertexPermanentError("Google ADC did not produce an access token")
    return str(token)


def _vertex_base_url(config: GoogleVertexConfig) -> str:
    host = "aiplatform.googleapis.com" if config.location == "global" else f"{config.location}-aiplatform.googleapis.com"
    return f"https://{host}/v1/projects/{config.project}/locations/{config.location}/publishers/google/models"


def _post_vertex(
    url: str,
    payload: dict[str, Any],
    config: GoogleVertexConfig,
    event_logger: Callable[[str], None] | None,
    event_prefix: str,
) -> dict[str, Any]:
    def call():
        token = _access_token(config)
        response = requests.post(
            url, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload, timeout=config.timeout_seconds)
        if response.status_code in {400, 401, 403, 404}:
            error = GoogleVertexPermanentError(f"Vertex call failed with HTTP {response.status_code}")
            error.status_code = response.status_code
            raise error
        response.raise_for_status()
        return response.json()
    return request_with_retry(call, event_logger=event_logger, label=event_prefix)


def extract_generate_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for candidate in response.get("candidates", []) or []:
        content = candidate.get("content", {}) or {}
        for part in content.get("parts", []) or []:
            text = part.get("text")
            if text:
                parts.append(str(text))
    if parts:
        return "\n".join(parts).strip()
    text = response.get("text")
    if text:
        return str(text).strip()
    raise GoogleVertexError(f"No text found in Vertex generateContent response: {json.dumps(response)[:500]}")


def parse_json_text(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start_obj = cleaned.find("{")
        start_arr = cleaned.find("[")
        starts = [idx for idx in (start_obj, start_arr) if idx >= 0]
        if not starts:
            raise
        start = min(starts)
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def generate_json_with_metadata(
    prompt: str,
    schema_name: str,
    model: str,
    config: GoogleVertexConfig,
    event_logger: Callable[[str], None] | None = None,
    temperature: float = 0.0,
    max_output_tokens: int | None = None,
) -> Any:
    url = f"{_vertex_base_url(config)}/{model}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
            "maxOutputTokens": max_output_tokens or int(os.getenv("CODING_FGF_GOOGLE_MAX_OUTPUT_TOKENS", "4096")),
        },
    }
    initial_delay, max_delay, jitter, max_attempts = _retry_config()
    parse_max_attempts = _model_output_max_attempts()
    delay = initial_delay
    attempt = 0
    usage = {}
    while True:
        attempt += 1
        response = _post_vertex(url, payload, config, event_logger, f"google:{schema_name}")
        raw_usage = response.get("usageMetadata", {}) or {}
        for key, out in (("promptTokenCount", "input_tokens"), ("candidatesTokenCount", "output_tokens"),
                         ("totalTokenCount", "total_tokens")):
            if isinstance(raw_usage.get(key), int):
                usage[out] = usage.get(out, 0) + raw_usage[key]
        try:
            data = parse_json_text(extract_generate_text(response))
            return {"data": data, "usage": usage}
        except Exception as exc:
            if attempt >= parse_max_attempts:
                raise GoogleVertexError(
                    f"Vertex JSON parse failed for {schema_name} after {attempt} model-output attempts: {exc}"
                ) from exc
            if event_logger:
                event_logger(f"google:{schema_name}:json_retry:attempt={attempt}:error={type(exc).__name__}:sleep={delay:.1f}")
            _sleep(delay, jitter)
            delay = min(delay * 2.0, max_delay)


def generate_json(prompt, schema_name, model, config, event_logger=None,
                  temperature=0.0, max_output_tokens=None):
    return generate_json_with_metadata(prompt, schema_name, model, config,
        event_logger, temperature, max_output_tokens)["data"]


def _extract_embedding(prediction: dict[str, Any]) -> list[float]:
    if isinstance(prediction.get("embeddings"), dict):
        values = prediction["embeddings"].get("values")
        if values is not None:
            return values
    if isinstance(prediction.get("embedding"), dict):
        values = prediction["embedding"].get("values")
        if values is not None:
            return values
    values = prediction.get("values")
    if values is not None:
        return values
    raise GoogleVertexError(f"No embedding values found in prediction: {json.dumps(prediction)[:500]}")


def embed_texts(
    texts: list[str],
    model: str,
    config: GoogleVertexConfig,
    event_logger: Callable[[str], None] | None = None,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    if not texts:
        return []
    url = f"{_vertex_base_url(config)}/{model}:predict"
    batch_size = max(1, int(os.getenv("CODING_FGF_GOOGLE_EMBED_BATCH_SIZE", "32")))
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        payload = {
            "instances": [
                {
                    "content": text,
                    "task_type": task_type,
                }
                for text in batch
            ]
        }
        response = _post_vertex(url, payload, config, event_logger, "google:embedding")
        predictions = response.get("predictions", []) or []
        if len(predictions) != len(batch):
            raise GoogleVertexError(f"Expected {len(batch)} embeddings, received {len(predictions)}")
        vectors.extend(_extract_embedding(prediction) for prediction in predictions)
    from .embeddings import validate_vectors
    return validate_vectors(vectors, len(texts))
