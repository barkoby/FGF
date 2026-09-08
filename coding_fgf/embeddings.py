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
        if self.offline:
            return [deterministic_embedding(text) for text in texts]
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for live embeddings; use --offline for tests")
        from openai import OpenAI
        from .providers import request_with_retry
        client = OpenAI(max_retries=0, timeout=float(os.getenv("CODING_FGF_OPENAI_TIMEOUT_SECONDS", "90")))
        response = request_with_retry(lambda: client.embeddings.create(model=self.model, input=texts),
                                      label="openai:embedding")
        items = list(response.data)
        if items and all(hasattr(item, "index") for item in items):
            if sorted(item.index for item in items) != list(range(len(texts))):
                raise ValueError("Invalid embedding response indices")
            items.sort(key=lambda item: item.index)
        return validate_vectors([item.embedding for item in items], len(texts))



CACHE_VERSION = 2
VERBALIZATION_VERSION = "coding_fgf_ontology_v1"

def validate_vectors(vectors, expected_count: int, dimension: int | None = None):
    if len(vectors) != expected_count:
        raise ValueError(f"Expected {expected_count} embeddings, received {len(vectors)}")
    clean = []
    for vector in vectors:
        if not isinstance(vector, (list, tuple)) or not vector:
            raise ValueError("Embedding vectors must be nonempty numeric sequences")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector):
            raise ValueError("Embedding vectors must contain finite numbers")
        dimension = dimension or len(vector)
        if len(vector) != dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {dimension}, received {len(vector)}")
        clean.append([float(v) for v in vector])
    return clean

def embedding_key(model: str, text: str, *, provider="openai", offline=False,
                  verbalization_version=VERBALIZATION_VERSION) -> str:
    payload = {"version": CACHE_VERSION, "provider": provider, "model": model,
               "mode": "offline" if offline else "live", "text": text,
               "verbalization_version": verbalization_version}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def cached_embeddings(texts, cache_path: Path, embed, *, model, provider="openai",
                      offline=False, verbalization_version=VERBALIZATION_VERSION):
    """Load only provenance-complete v2 rows; legacy cache rows remain untouched."""
    mode = "offline" if offline else "live"
    metadata = {"version": CACHE_VERSION, "provider": provider, "model": model,
                "mode": mode, "verbalization_version": verbalization_version}
    cached = {}
    if cache_path.exists():
        for line_number, line in enumerate(cache_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"Malformed embedding cache at line {line_number}") from exc
            if all(row.get(k) == v for k, v in metadata.items()):
                if not isinstance(row.get("dimension"), int):
                    raise ValueError("Embedding cache entry lacks dimension")
                cached[row["hash"]] = validate_vectors([row["embedding"]], 1, row["dimension"])[0]
    keys = [embedding_key(model, text, provider=provider, offline=offline,
                          verbalization_version=verbalization_version) for text in texts]
    missing = dict((key, text) for key, text in zip(keys, texts) if key not in cached)
    new_rows = []
    if missing:
        vectors = validate_vectors(embed(list(missing.values())), len(missing))
        for key, vector in zip(missing, vectors):
            cached[key] = vector
            new_rows.append({**metadata, "hash": key, "dimension": len(vector), "embedding": vector})
    result = validate_vectors([cached[key] for key in keys], len(keys))
    if new_rows:
        ensure_dir(cache_path.parent)
        with cache_path.open("a", encoding="utf-8") as fh:
            for row in new_rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
    return result

def embed_records(records: Iterable[dict[str, object]], cache_path: Path,
                  model: str = DEFAULT_EMBEDDING_MODEL, offline: bool = False,
                  provider: str = "openai", google_project: str = "",
                  google_location: str = "", google_credentials: str = "") -> list[dict[str, object]]:
    records = list(records)
    client = EmbeddingClient(model, offline, provider, google_project, google_location, google_credentials)
    vectors = cached_embeddings([str(record.get("text", "")) for record in records],
                                cache_path, client.embed, model=model, provider=provider, offline=offline)
    return [{**record, "embedding": vector} for record, vector in zip(records, vectors)]
