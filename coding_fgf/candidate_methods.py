from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .constants import DEFAULT_EMBEDDING_MODEL
from .io import ensure_dir


METHOD_OPENAI_SMALL = "openai_small"
METHOD_LEVENSHTEIN = "levenshtein"
METHOD_BM25 = "bm25"
METHOD_HYBRID_LEVENSHTEIN_DENSE = "hybrid_levenshtein_dense"
METHOD_HYBRID_BM25_DENSE = "hybrid_bm25_dense"
DEFAULT_METHODS = [
    METHOD_OPENAI_SMALL,
    METHOD_LEVENSHTEIN,
    METHOD_BM25,
    METHOD_HYBRID_LEVENSHTEIN_DENSE,
    METHOD_HYBRID_BM25_DENSE,
]
VERBALIZATION_VERSION = "coding_fgf_ontology_v1"

EmbeddingFunction = Callable[[str, Sequence[str]], list[list[float]]]
ProgressLogger = Callable[[str], None]


@dataclass(frozen=True)
class RankedCandidate:
    rank: int
    uri: str
    record: dict[str, Any]
    score: float
    method_scores: dict[str, float] = field(default_factory=dict)

    def to_artifact(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "uri": self.uri,
            "id": self.record.get("id", ""),
            "kind": self.record.get("kind", ""),
            "local_name": self.record.get("local_name", ""),
            "label": self.record.get("label", ""),
            "score": self.score,
            "method_scores": self.method_scores,
        }


def validate_methods(methods: Sequence[str]) -> list[str]:
    unknown = [method for method in methods if method not in DEFAULT_METHODS]
    if unknown:
        raise ValueError(f"Unknown candidate generation method(s): {', '.join(unknown)}")
    return list(methods)


def token_list(value: object) -> list[str]:
    text = re.sub(r"[_\-]+", " ", str(value or ""))
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return [part.lower() for part in re.split(r"\W+", text) if part]


def compact_entity_name(record: dict[str, Any]) -> str:
    if record.get("source_table") or record.get("source_column"):
        return " ".join(str(record.get(key, "")) for key in ("source_table", "source_column", "local_name", "label"))
    return " ".join(str(record.get(key, "")) for key in ("local_name", "label", "uri"))


def record_text(record: dict[str, Any]) -> str:
    return str(record.get("text") or compact_entity_name(record))


def normalized_levenshtein_similarity(left: object, right: object) -> float:
    left_text = " ".join(token_list(left))
    right_text = " ".join(token_list(right))
    return normalized_levenshtein_similarity_text(left_text, right_text)


def normalized_levenshtein_similarity_text(left_text: str, right_text: str) -> float:
    if not left_text and not right_text:
        return 1.0
    if not left_text or not right_text:
        return 0.0
    distance = levenshtein_distance(left_text, right_text)
    return 1.0 - (distance / max(len(left_text), len(right_text)))


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            current.append(min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


class Bm25Index:
    def __init__(self, records: Sequence[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.records = list(records)
        self.k1 = k1
        self.b = b
        self.docs = [token_list(record_text(record)) for record in self.records]
        self.avgdl = sum(len(doc) for doc in self.docs) / len(self.docs) if self.docs else 0.0
        self.doc_freq: dict[str, int] = {}
        for doc in self.docs:
            for token in set(doc):
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

    def score(self, query: str, doc_index: int) -> float:
        if not self.docs:
            return 0.0
        query_tokens = token_list(query)
        doc = self.docs[doc_index]
        if not query_tokens or not doc:
            return 0.0
        freqs: dict[str, int] = {}
        for token in doc:
            freqs[token] = freqs.get(token, 0) + 1
        n_docs = len(self.docs)
        score = 0.0
        for token in query_tokens:
            df = self.doc_freq.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            tf = freqs.get(token, 0)
            denom = tf + self.k1 * (1.0 - self.b + self.b * len(doc) / (self.avgdl or 1.0))
            score += idf * (tf * (self.k1 + 1.0)) / (denom or 1.0)
        return score

    def rank(self, query: str) -> list[tuple[dict[str, Any], float]]:
        rows = [(record, self.score(query, idx)) for idx, record in enumerate(self.records)]
        return sorted(rows, key=lambda item: (-item[1], str(item[0].get("uri", ""))))


def embedding_cache_key(model: str, text: str, verbalization_version: str = VERBALIZATION_VERSION) -> str:
    payload = {"model": model, "text": text, "verbalization_version": verbalization_version}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


class OpenAIEmbeddingCache:
    def __init__(
        self,
        cache_path: Path,
        model: str = DEFAULT_EMBEDDING_MODEL,
        embedder: EmbeddingFunction | None = None,
        batch_size: int = 64,
        logger: ProgressLogger | None = None,
    ) -> None:
        self.cache_path = cache_path
        self.model = model
        self.embedder = embedder
        self.batch_size = batch_size
        self.logger = logger
        ensure_dir(cache_path.parent)
        self.cache: dict[str, list[float]] = {}
        if cache_path.exists():
            for raw in cache_path.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if row.get("model") == model and isinstance(row.get("embedding"), list):
                    self.cache[str(row["text_hash"])] = [float(v) for v in row["embedding"]]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        output: list[list[float] | None] = [None for _ in texts]
        missing: list[tuple[int, str, str]] = []
        for idx, text in enumerate(texts):
            key = embedding_cache_key(self.model, text)
            cached = self.cache.get(key)
            if cached is None:
                missing.append((idx, key, text))
            else:
                output[idx] = cached
        if missing:
            self._log(
                "embedding_cache:miss",
                texts=len(texts),
                missing=len(missing),
                cache_hits=len(texts) - len(missing),
                model=self.model,
            )
            vectors = self._embed_missing(missing)
            with self.cache_path.open("a", encoding="utf-8") as fh:
                for (idx, key, text), vector in zip(missing, vectors):
                    clean_vector = [float(v) for v in vector]
                    self.cache[key] = clean_vector
                    output[idx] = clean_vector
                    fh.write(
                        json.dumps(
                            {
                                "model": self.model,
                                "text_hash": key,
                                "text_preview": text[:120],
                                "verbalization_version": VERBALIZATION_VERSION,
                                "embedding": clean_vector,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
            self._log("embedding_cache:write", rows=len(missing), cache_path=str(self.cache_path))
        return [vector for vector in output if vector is not None]

    def _embed_missing(self, missing: Sequence[tuple[int, str, str]]) -> list[list[float]]:
        texts = [item[2] for item in missing]
        if self.embedder:
            return self.embedder(self.model, texts)
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is required for uncached dense candidate evaluation; "
                "the evaluator does not silently fall back to deterministic embeddings."
            )
        from openai import OpenAI  # type: ignore

        client = OpenAI()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            self._log(
                "embedding_api:batch:start",
                batch_start=start + 1,
                batch_end=start + len(batch),
                batch_total=len(texts),
                model=self.model,
            )
            response = client.embeddings.create(model=self.model, input=batch)
            vectors.extend([[float(v) for v in item.embedding] for item in response.data])
            self._log(
                "embedding_api:batch:complete",
                batch_start=start + 1,
                batch_end=start + len(batch),
                batch_total=len(texts),
                model=self.model,
            )
        return vectors

    def _log(self, event: str, **fields: Any) -> None:
        if not self.logger:
            return
        details = " ".join(f"{key}={value}" for key, value in fields.items())
        self.logger(f"{event}" + (f" {details}" if details else ""))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class DenseIndex:
    def __init__(self, records_by_kind: dict[str, list[dict[str, Any]]], cache: OpenAIEmbeddingCache) -> None:
        self.records_by_kind = records_by_kind
        self.cache = cache
        self.vectors: dict[str, list[float]] = {}
        all_records = [record for records in records_by_kind.values() for record in records]
        embeddings = cache.embed_texts([record_text(record) for record in all_records])
        for record, vector in zip(all_records, embeddings):
            self.vectors[str(record.get("uri", ""))] = vector

    def rank(self, source: dict[str, Any], kind: str) -> list[tuple[dict[str, Any], float]]:
        records = self.records_by_kind.get(kind, [])
        if not records:
            return []
        source_vector = self.cache.embed_texts([record_text(source)])[0]
        rows = [
            (record, cosine_similarity(source_vector, self.vectors.get(str(record.get("uri", "")), [])))
            for record in records
        ]
        return sorted(rows, key=lambda item: (-item[1], str(item[0].get("uri", ""))))


@dataclass
class CandidateMethodContext:
    methods: list[str]
    target_records: list[dict[str, Any]]
    cache_dir: Path
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    rrf_k: int = 60
    embedder: EmbeddingFunction | None = None
    logger: ProgressLogger | None = None
    records_by_kind: dict[str, list[dict[str, Any]]] = field(init=False)
    levenshtein_text_by_uri: dict[str, str] = field(init=False)
    bm25_by_kind: dict[str, Bm25Index] = field(init=False)
    dense_index: DenseIndex | None = field(init=False, default=None)
    ranking_cache: dict[tuple[str, str], list[RankedCandidate]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.methods = validate_methods(self.methods)
        self.records_by_kind = {}
        for record in self.target_records:
            self.records_by_kind.setdefault(str(record.get("kind", "")), []).append(record)
        self.levenshtein_text_by_uri = {
            str(record.get("uri", "")): " ".join(token_list(compact_entity_name(record))) for record in self.target_records
        }
        self.bm25_by_kind = {kind: Bm25Index(records) for kind, records in self.records_by_kind.items()}
        needs_dense = any("dense" in method or method == METHOD_OPENAI_SMALL for method in self.methods)
        if needs_dense:
            cache = OpenAIEmbeddingCache(
                self.cache_dir / f"{self.embedding_model}.jsonl",
                self.embedding_model,
                embedder=self.embedder,
                logger=self.logger,
            )
            self.dense_index = DenseIndex(self.records_by_kind, cache)

    def method_configs(self, k_values: Sequence[int], max_candidates: int) -> dict[str, Any]:
        return {
            "methods": {
                METHOD_OPENAI_SMALL: {
                    "model": self.embedding_model,
                    "similarity": "cosine_similarity",
                    "candidate_filter": "same_source_kind",
                },
                METHOD_LEVENSHTEIN: {
                    "similarity": "1 - levenshtein_distance / max_length",
                    "candidate_filter": "same_source_kind",
                },
                METHOD_BM25: {
                    "similarity": "okapi_bm25",
                    "k1": 1.5,
                    "b": 0.75,
                    "candidate_filter": "same_source_kind",
                },
                METHOD_HYBRID_LEVENSHTEIN_DENSE: {
                    "fusion": "reciprocal_rank_fusion",
                    "rrf_k": self.rrf_k,
                    "components": [METHOD_LEVENSHTEIN, METHOD_OPENAI_SMALL],
                },
                METHOD_HYBRID_BM25_DENSE: {
                    "fusion": "reciprocal_rank_fusion",
                    "rrf_k": self.rrf_k,
                    "components": [METHOD_BM25, METHOD_OPENAI_SMALL],
                },
            },
            "embedding_model": self.embedding_model,
            "k_values": list(k_values),
            "max_candidates": max_candidates,
            "cache_dir": str(self.cache_dir),
            "verbalization_version": VERBALIZATION_VERSION,
            "llm_matching": "disabled",
        }


def rank_method(
    context: CandidateMethodContext,
    method: str,
    source: dict[str, Any],
    max_candidates: int = 20,
) -> tuple[list[RankedCandidate], dict[str, int]]:
    full = _rank_full(context, method, source)
    rank_by_uri = {candidate.uri: candidate.rank for candidate in full}
    return full[:max_candidates], rank_by_uri


def _rank_full(context: CandidateMethodContext, method: str, source: dict[str, Any]) -> list[RankedCandidate]:
    source_key = str(source.get("id") or record_text(source))
    cache_key = (method, source_key)
    cached = context.ranking_cache.get(cache_key)
    if cached is not None:
        return cached
    ranked = _rank_full_uncached(context, method, source)
    context.ranking_cache[cache_key] = ranked
    return ranked


def _rank_full_uncached(context: CandidateMethodContext, method: str, source: dict[str, Any]) -> list[RankedCandidate]:
    kind = str(source.get("kind", ""))
    if method == METHOD_OPENAI_SMALL:
        if context.dense_index is None:
            raise RuntimeError("Dense index was not initialized")
        return _rows_to_ranked(context.dense_index.rank(source, kind), "dense_score")
    if method == METHOD_LEVENSHTEIN:
        query = " ".join(token_list(compact_entity_name(source)))
        rows = [
            (
                record,
                normalized_levenshtein_similarity_text(
                    query,
                    context.levenshtein_text_by_uri.get(str(record.get("uri", "")), ""),
                ),
            )
            for record in context.records_by_kind.get(kind, [])
        ]
        return _rows_to_ranked(sorted(rows, key=lambda item: (-item[1], str(item[0].get("uri", "")))), "levenshtein_score")
    if method == METHOD_BM25:
        query = record_text(source)
        return _rows_to_ranked(context.bm25_by_kind.get(kind, Bm25Index([])).rank(query), "bm25_score")
    if method == METHOD_HYBRID_LEVENSHTEIN_DENSE:
        return reciprocal_rank_fusion(
            [_rank_full(context, METHOD_LEVENSHTEIN, source), _rank_full(context, METHOD_OPENAI_SMALL, source)],
            rrf_k=context.rrf_k,
            score_name="hybrid_levenshtein_dense_score",
        )
    if method == METHOD_HYBRID_BM25_DENSE:
        return reciprocal_rank_fusion(
            [_rank_full(context, METHOD_BM25, source), _rank_full(context, METHOD_OPENAI_SMALL, source)],
            rrf_k=context.rrf_k,
            score_name="hybrid_bm25_dense_score",
        )
    raise ValueError(f"Unknown method {method}")


def _rows_to_ranked(rows: Sequence[tuple[dict[str, Any], float]], score_name: str) -> list[RankedCandidate]:
    out: list[RankedCandidate] = []
    for rank, (record, score) in enumerate(rows, start=1):
        uri = str(record.get("uri", ""))
        out.append(RankedCandidate(rank=rank, uri=uri, record=record, score=float(score), method_scores={score_name: float(score)}))
    return out


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RankedCandidate]],
    rrf_k: int = 60,
    score_name: str = "rrf_score",
) -> list[RankedCandidate]:
    by_uri: dict[str, RankedCandidate] = {}
    scores: dict[str, float] = {}
    method_scores: dict[str, dict[str, float]] = {}
    for ranking in rankings:
        for candidate in ranking:
            by_uri.setdefault(candidate.uri, candidate)
            scores[candidate.uri] = scores.get(candidate.uri, 0.0) + 1.0 / (rrf_k + candidate.rank)
            method_scores.setdefault(candidate.uri, {}).update(candidate.method_scores)
    ordered = sorted(scores, key=lambda uri: (-scores[uri], uri))
    out: list[RankedCandidate] = []
    for rank, uri in enumerate(ordered, start=1):
        base = by_uri[uri]
        all_scores = dict(method_scores.get(uri, {}))
        all_scores[score_name] = scores[uri]
        out.append(RankedCandidate(rank=rank, uri=uri, record=base.record, score=scores[uri], method_scores=all_scores))
    return out


def normalized_candidate_artifact(
    scenario: str,
    method: str,
    source: dict[str, Any],
    candidates: Sequence[RankedCandidate],
) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "method": method,
        "source_id": source.get("id", ""),
        "source_kind": source.get("kind", ""),
        "source_table": source.get("source_table", ""),
        "source_column": source.get("source_column", ""),
        "source_table_role": source.get("source_table_role") or source.get("table_role", ""),
        "source_column_role": source.get("source_column_role", ""),
        "candidate_count": len(candidates),
        "candidates": [candidate.to_artifact() for candidate in candidates],
    }
