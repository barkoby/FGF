from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

from ..candidate_gold import normalize_uri


NULL_TOKENS = {None, "", "NO_MATCH", "null"}


def candidate_rank_map(candidate_row: Mapping[str, Any]) -> dict[str, int]:
    return {normalize_uri(candidate.get("uri")): int(candidate.get("rank", idx + 1)) for idx, candidate in enumerate(candidate_row.get("candidates", []) or [])}


def validate_match_response(
    data: Mapping[str, Any] | None,
    candidate_row: Mapping[str, Any],
    method: str,
    allowed_uris: Iterable[str] | None = None,
) -> dict[str, Any]:
    source = candidate_row.get("source", {}) or {}
    source_id = str(source.get("id", ""))
    candidates = list(candidate_row.get("candidates", []) or [])
    by_uri = {normalize_uri(candidate.get("uri")): candidate for candidate in candidates}
    allowed = {normalize_uri(uri) for uri in (allowed_uris if allowed_uris is not None else by_uri)}
    rank_by_uri = candidate_rank_map(candidate_row)
    rows = list((data or {}).get("matches", []) or [])
    if len(rows) != 1:
        return invalid_result(candidate_row, method, "missing_source_id", f"Expected exactly one decision, got {len(rows)}")
    row = rows[0]
    if str(row.get("source_id", "")) != source_id:
        return invalid_result(candidate_row, method, "missing_source_id", "Decision source_id did not match the requested source")
    target_uri = _normalize_target(row.get("target_uri"))
    if target_uri is not None and target_uri not in allowed:
        return invalid_result(candidate_row, method, "invalid_target_uri", "Selected target_uri was not supplied as an allowed candidate")
    selected = by_uri.get(target_uri or "", {})
    decision = "no_match" if target_uri is None else "selected"
    target_id = row.get("target_id") or selected.get("id")
    return {
        **base_result(candidate_row, method),
        "predicted_target_uri": target_uri,
        "predicted_target_id": None if not target_id else str(target_id),
        "predicted_target_kind": selected.get("kind", ""),
        "candidate_rank": rank_by_uri.get(target_uri or ""),
        "confidence": _float(row.get("confidence")),
        "decision": decision,
        "null_category": "" if target_uri else str(row.get("null_category", "")),
        "reason": str(row.get("reason", "")),
        "evidence_summary": row.get("evidence_summary", []),
        "api_error": False,
        "invalid_selection": False,
        "error_type": "",
    }


def base_result(candidate_row: Mapping[str, Any], method: str) -> dict[str, Any]:
    source = candidate_row.get("source", {}) or {}
    return {
        "scenario": candidate_row.get("scenario", ""),
        "method": method,
        "source_id": source.get("id", ""),
        "source_kind": source.get("kind", ""),
        "source_table": source.get("source_table", ""),
        "source_column": source.get("source_column", ""),
        "source_table_role": source.get("source_table_role") or source.get("table_role", ""),
        "source_column_role": source.get("source_column_role", ""),
        "gold_target_uris": [],
        "predicted_target_uri": None,
        "predicted_target_id": None,
        "predicted_target_kind": "",
        "candidate_rank": None,
        "confidence": 0.0,
        "decision": "no_match",
        "null_category": "",
        "correct": False,
        "error_type": "",
        "reason": "",
        "api_error": False,
        "invalid_selection": False,
        "runtime_seconds": 0.0,
    }


def invalid_result(candidate_row: Mapping[str, Any], method: str, error_type: str, reason: str) -> dict[str, Any]:
    return {
        **base_result(candidate_row, method),
        "decision": "invalid",
        "error_type": error_type,
        "reason": reason,
        "invalid_selection": True,
    }


def api_error_result(candidate_row: Mapping[str, Any], method: str, reason: str) -> dict[str, Any]:
    return {
        **base_result(candidate_row, method),
        "decision": "api_error",
        "error_type": "api_error",
        "reason": reason,
        "api_error": True,
    }


def attach_gold_and_score(row: dict[str, Any], gold_by_source: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    out = dict(row)
    gold = [normalize_uri(uri) for uri in gold_by_source.get(str(row.get("source_id", "")), [])]
    predicted = normalize_uri(row.get("predicted_target_uri")) if row.get("predicted_target_uri") else None
    out["gold_target_uris"] = gold
    out["correct"] = bool(predicted and predicted in set(gold))
    return out


def aggregate_rows(rows: Sequence[Mapping[str, Any]], group_fields: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in group_fields)].append(row)
    out: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        metric = {field: value for field, value in zip(group_fields, key)}
        metric.update(_metric_values(group))
        out.append(metric)
    return out


def confusion_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("scenario", "")), str(row.get("method", "")))].append(row)
    out = []
    for (scenario, method), group in sorted(grouped.items()):
        selected_correct = selected_wrong = selected_when_gold_null = 0
        null_when_gold_match = null_when_gold_null = invalid = api_error = 0
        for row in group:
            gold = set(row.get("gold_target_uris", []) or [])
            predicted = row.get("predicted_target_uri")
            if row.get("api_error"):
                api_error += 1
                continue
            if row.get("invalid_selection"):
                invalid += 1
                continue
            if gold:
                if predicted and predicted in gold:
                    selected_correct += 1
                elif predicted:
                    selected_wrong += 1
                else:
                    null_when_gold_match += 1
            elif predicted:
                selected_when_gold_null += 1
            else:
                null_when_gold_null += 1
        out.append(
            {
                "scenario": scenario,
                "method": method,
                "selected_correct": selected_correct,
                "selected_wrong": selected_wrong,
                "selected_when_gold_null": selected_when_gold_null,
                "null_when_gold_match": null_when_gold_match,
                "null_when_gold_null": null_when_gold_null,
                "invalid_selection": invalid,
                "api_error": api_error,
            }
        )
    return out


def aggregate_self_consistency(
    samples: Sequence[Mapping[str, Any]],
    candidate_row: Mapping[str, Any],
    confidence_threshold: float = 0.5,
) -> dict[str, Any]:
    rank_by_uri = candidate_rank_map(candidate_row)
    valid_samples = [sample for sample in samples if not sample.get("invalid_selection") and not sample.get("api_error")]
    if not valid_samples:
        if samples and all(sample.get("invalid_selection") for sample in samples):
            return invalid_result(candidate_row, "self_consistency", "no_valid_self_consistency_samples", "No valid self-consistency samples were available")
        return api_error_result(candidate_row, "self_consistency", "No valid self-consistency samples were available")
    key_for = lambda sample: str(sample.get("predicted_target_uri") or "null")
    counts = Counter(key_for(sample) for sample in valid_samples)
    confidence_by_key: dict[str, list[float]] = defaultdict(list)
    for sample in valid_samples:
        confidence_by_key[key_for(sample)].append(_float(sample.get("confidence")))
    max_votes = max(counts.values())
    tied = [key for key, value in counts.items() if value == max_votes]

    def sort_key(key: str) -> tuple[float, int, int]:
        avg_conf = sum(confidence_by_key[key]) / len(confidence_by_key[key])
        rank = rank_by_uri.get(key, 10**9) if key != "null" else 10**9
        non_null = 1 if key != "null" else 0
        return (-avg_conf, rank, -non_null)

    winner = sorted(tied, key=sort_key)[0]
    avg_conf = sum(confidence_by_key[winner]) / len(confidence_by_key[winner])
    if len(tied) > 1 and winner != "null" and avg_conf < confidence_threshold:
        winner = "null"
        avg_conf = sum(confidence_by_key.get("null", [0.0])) / len(confidence_by_key.get("null", [0.0]))
    base = validate_match_response(
        {"matches": [{"source_id": candidate_row["source"]["id"], "target_uri": None if winner == "null" else winner, "confidence": avg_conf}]},
        candidate_row,
        "self_consistency",
    )
    distribution = dict(counts)
    base.update(
        {
            "vote_count": counts[winner],
            "num_samples": len(samples),
            "vote_distribution": distribution,
            "disagreement_score": 1.0 - (counts[winner] / max(1, len(valid_samples))),
            "all_sampled_decisions": [dict(sample) for sample in samples],
            "tie": len(tied) > 1,
            "reason": f"Majority self-consistency decision over {len(valid_samples)} valid samples",
        }
    )
    return base


def _metric_values(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gold_rows = [row for row in group if row.get("gold_target_uris")]
    tp = sum(1 for row in gold_rows if row.get("correct"))
    fp = sum(1 for row in gold_rows if row.get("predicted_target_uri") and not row.get("correct") and not row.get("invalid_selection") and not row.get("api_error"))
    fn = len(gold_rows) - tp
    predicted_matches = sum(1 for row in group if row.get("predicted_target_uri") and not row.get("invalid_selection") and not row.get("api_error"))
    predicted_nulls = sum(1 for row in group if not row.get("predicted_target_uri") and not row.get("invalid_selection") and not row.get("api_error"))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / len(gold_rows) if gold_rows else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    ranks = [_int(row.get("candidate_rank")) for row in group if _int(row.get("candidate_rank")) is not None]
    confidences = [_float(row.get("confidence")) for row in group if row.get("confidence") is not None]
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "evaluated_sources": len(group),
        "gold_mapped_sources": len(gold_rows),
        "predicted_matches": predicted_matches,
        "predicted_nulls": predicted_nulls,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "invalid_selections": sum(1 for row in group if row.get("invalid_selection")),
        "api_errors": sum(1 for row in group if row.get("api_error")),
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "avg_candidate_rank": sum(ranks) / len(ranks) if ranks else "",
        "runtime_seconds": sum(_float(row.get("runtime_seconds")) for row in group),
        "all_source_predicted_matches": predicted_matches,
        "all_source_predicted_nulls": predicted_nulls,
        "unlabeled_sources": sum(1 for row in group if not row.get("gold_target_uris")),
        "error_rate": (fp + fn + sum(1 for row in group if row.get("invalid_selection") or row.get("api_error"))) / len(group) if group else 0.0,
    }


def _normalize_target(value: Any) -> str | None:
    if value in NULL_TOKENS:
        return None
    normalized = normalize_uri(value)
    return normalized or None


def _float(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        number = float(value)
        if math.isnan(number):
            return None
        return int(number)
    except (TypeError, ValueError):
        return None
