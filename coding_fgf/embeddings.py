from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

from .constants import DEFAULT_EMBEDDING_MODEL
from .io import ensure_dir


def deterministic_embedding(text: str, dim: int = 256) -> list[float]:
    vec = [0.0 for _ in range(dim)]
    tokens = text.lower().split()
    if not tokens:
        tokens = [text]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(value * value for value in vec))
    if norm > 0:
        vec = [value / norm for value in vec]
    return vec


class EmbeddingClient:
    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        offline: bool = False,
        provider: str = "openai",
        google_project: str = "",
        google_location: str = "",
        google_credentials: str = "",
    ) -> None:
        self.model = model
        self.offline = offline
        self.provider = provider
        self.google_project = google_project
        self.google_location = google_location
        self.google_credentials = google_credentials

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "google":
            if self.offline:
                return [deterministic_embedding(text) for text in texts]
            from .google_vertex import embed_texts, google_config

            return embed_texts(
                texts,
                self.model,
                google_config(self.google_project or None, self.google_location or None, self.google_credentials or None),
            )
        if self.provider != "openai":
            raise ValueError(f"Unsupported embedding provider: {self.provider}")
        if self.offline or not os.getenv("OPENAI_API_KEY"):
            return [deterministic_embedding(text) for text in texts]
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI()
            response = client.embeddings.create(model=self.model, input=texts)
            return [item.embedding for item in response.data]
        except Exception:
            return [deterministic_embedding(text) for text in texts]


def embed_records(
    records: Iterable[dict[str, object]],
    cache_path: Path,
    model: str = DEFAULT_EMBEDDING_MODEL,
    offline: bool = False,
    provider: str = "openai",
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
) -> list[dict[str, object]]:
    ensure_dir(cache_path.parent)
    cached: dict[str, dict[str, object]] = {}
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    cached[str(row["hash"])] = row
    out: list[dict[str, object]] = []
    missing: list[tuple[str, dict[str, object]]] = []
    for record in records:
        text = str(record.get("text", ""))
        key_material = (model + "\n" + text) if provider == "openai" else (provider + "\n" + model + "\n" + text)
        key = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
        if key in cached:
            out.append({**record, "embedding": cached[key]["embedding"]})
        else:
            missing.append((key, record))
    if missing:
        client = EmbeddingClient(
            model=model,
            offline=offline,
            provider=provider,
            google_project=google_project,
            google_location=google_location,
            google_credentials=google_credentials,
        )
        embeddings = client.embed([str(record.get("text", "")) for _, record in missing])
        with cache_path.open("a", encoding="utf-8") as fh:
            for (key, record), embedding in zip(missing, embeddings):
                row = {"hash": key, "provider": provider, "model": model, "id": record.get("id"), "embedding": embedding}
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                out.append({**record, "embedding": embedding})
    return out
