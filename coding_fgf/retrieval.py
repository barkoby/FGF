from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .io import ensure_dir, read_jsonl, write_jsonl
from .lexical import words


def build_index(embedded_records: Iterable[dict[str, object]], work_dir: Path) -> Path:
    rows = list(embedded_records)
    ensure_dir(work_dir)
    if not rows:
        raise ValueError("No embedded records to index")
    vectors = [row["embedding"] for row in rows]
    write_jsonl(work_dir / "index_meta.jsonl", [{k: v for k, v in row.items() if k != "embedding"} for row in rows])
    (work_dir / "index_vectors.json").write_text(json.dumps(vectors), encoding="utf-8")
    try:
        import numpy as np  # type: ignore
        import faiss  # type: ignore

        np_vectors = np.array(vectors, dtype="float32")
        index = faiss.IndexFlatL2(np_vectors.shape[1])
        index.add(np_vectors)
        faiss.write_index(index, str(work_dir / "index.faiss"))
    except Exception as exc:
        (work_dir / "faiss_error.txt").write_text(str(exc), encoding="utf-8")
    return work_dir


def _load_vectors(work_dir: Path) -> list[list[float]]:
    return json.loads((work_dir / "index_vectors.json").read_text(encoding="utf-8"))


def _l2(left: list[float], right: list[float]) -> float:
    return sum((a - b) * (a - b) for a, b in zip(left, right))


def _lexical_bonus(source: dict[str, object], candidate: dict[str, object]) -> float:
    source_name = source.get("source_column") or source.get("source_table") or source.get("local_name") or source.get("label")
    candidate_name = " ".join(str(candidate.get(key, "")) for key in ("local_name", "label", "uri"))
    source_words = words(source_name)
    candidate_words = words(candidate_name)
    if not source_words or not candidate_words:
        return 0.0
    overlap = len(source_words & candidate_words)
    if source_words <= candidate_words or candidate_words <= source_words:
        overlap += 2
    return min(1.0, overlap * 0.35)


def retrieve_candidates(
    source_records: Iterable[dict[str, object]],
    work_dir: Path,
    k: int = 20,
    same_kind: bool = True,
) -> list[dict[str, object]]:
    meta = read_jsonl(work_dir / "index_meta.jsonl")
    vectors = _load_vectors(work_dir)
    rows: list[dict[str, object]] = []
    for source in source_records:
        source_clean = {key: value for key, value in source.items() if key != "embedding"}
        if source.get("kind") == "class" and source.get("table_role") == "join_table":
            rows.append({"source": source_clean, "candidates": []})
            continue
        query = list(source["embedding"])  # type: ignore[arg-type]
        distances = [_l2(vector, query) for vector in vectors]
        order = sorted(range(len(distances)), key=lambda idx: distances[idx] - _lexical_bonus(source, meta[idx]))
        candidates = []
        for idx in order:
            candidate = meta[int(idx)]
            if same_kind and candidate.get("kind") != source.get("kind"):
                continue
            candidate_row = dict(candidate)
            candidate_row["distance"] = float(distances[int(idx)])
            candidates.append(candidate_row)
            if len(candidates) >= k:
                break
        rows.append({"source": source_clean, "candidates": candidates})
    return rows


def write_candidates(path: Path, rows: Iterable[dict[str, object]]) -> None:
    write_jsonl(path, rows)


def read_candidates(path: Path) -> list[dict[str, object]]:
    return read_jsonl(path)
