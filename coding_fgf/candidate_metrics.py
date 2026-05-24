from __future__ import annotations

from typing import Any, Iterable, Sequence

from .candidate_gold import GoldIssue, GoldMapping
from .candidate_methods import (
    METHOD_BM25,
    METHOD_HYBRID_BM25_DENSE,
    METHOD_HYBRID_LEVENSHTEIN_DENSE,
    METHOD_LEVENSHTEIN,
    METHOD_OPENAI_SMALL,
    RankedCandidate,
)


FAILURE_CATEGORIES = [
    "zero_candidates",
    "correct_target_not_retrieved",
    "correct_target_rank_too_low",
    "wrong_kind_filtering",
    "source_not_in_gold",
    "target_uri_normalization_error",
    "weak_verbalization",
    "lexical_rename_failure",
    "dense_semantic_failure",
    "bm25_tokenization_failure",
    "hybrid_fusion_failure",
    "ambiguous_candidates",
    "other",
]


def found_field(k: int) -> str:
    return f"found_at_{k}"


def first_correct_rank(gold_target_uris: Iterable[str], rank_by_uri: dict[str, int], max_rank: int = 20) -> int | None:
    ranks = [rank_by_uri[uri] for uri in gold_target_uris if uri in rank_by_uri and rank_by_uri[uri] <= max_rank]
    return min(ranks) if ranks else None


def first_correct_rank_anywhere(gold_target_uris: Iterable[str], rank_by_uri: dict[str, int]) -> int | None:
    ranks = [rank_by_uri[uri] for uri in gold_target_uris if uri in rank_by_uri]
    return min(ranks) if ranks else None


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def evaluate_mapping(
    mapping: GoldMapping,
    method: str,
    candidates: Sequence[RankedCandidate],
    rank_by_uri: dict[str, int],
    k_values: Sequence[int],
    target_kind_by_uri: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    max_k = max(k_values) if k_values else 20
    rank = first_correct_rank(mapping.gold_target_uris, rank_by_uri, max_rank=max_k)
    any_rank = first_correct_rank_anywhere(mapping.gold_target_uris, rank_by_uri)
    row = {
        "scenario": mapping.scenario,
        "method": method,
        "source_id": mapping.source_id,
        "source_kind": mapping.source_kind,
        "source_table": mapping.source_table,
        "source_column": mapping.source_column,
        "source_table_role": mapping.source_table_role,
        "source_column_role": mapping.source_column_role,
        "gold_target_uris": tuple(mapping.gold_target_uris),
        "rank_of_first_gold": rank,
        "rank_of_first_gold_anywhere": any_rank,
        "reciprocal_rank": reciprocal_rank(rank),
        "candidate_count": len(candidates),
        "zero_candidates": 1 if not candidates else 0,
    }
    for k in k_values:
        row[found_field(k)] = 1 if rank and rank <= k else 0
    failure_category = ""
    if rank is None:
        failure_category = assign_failure_category(
            method,
            len(candidates),
            any_rank,
            candidates,
            mapping.gold_target_uris,
            target_kind_by_uri or {},
            mapping.source_kind,
        )
    row["failure_category"] = failure_category

    per_gold_rows = []
    for gold_uri in mapping.gold_target_uris:
        out = dict(row)
        out["gold_target_uri"] = gold_uri
        out.pop("gold_target_uris", None)
        per_gold_rows.append(out)
    return row, per_gold_rows


def assign_failure_category(
    method: str,
    candidate_count: int,
    rank_anywhere: int | None,
    candidates: Sequence[RankedCandidate],
    gold_target_uris: Sequence[str],
    target_kind_by_uri: dict[str, str],
    source_kind: str,
) -> str:
    if candidate_count == 0:
        return "zero_candidates"
    if any(target_kind_by_uri.get(uri) and target_kind_by_uri.get(uri) != source_kind for uri in gold_target_uris):
        return "wrong_kind_filtering"
    if rank_anywhere and rank_anywhere > 20:
        return "correct_target_rank_too_low"
    if len(candidates) >= 2 and abs(float(candidates[0].score) - float(candidates[1].score)) <= 1e-12:
        return "ambiguous_candidates"
    if method == METHOD_LEVENSHTEIN:
        return "lexical_rename_failure"
    if method == METHOD_OPENAI_SMALL:
        return "dense_semantic_failure"
    if method == METHOD_BM25:
        return "bm25_tokenization_failure"
    if method in {METHOD_HYBRID_LEVENSHTEIN_DENSE, METHOD_HYBRID_BM25_DENSE}:
        return "hybrid_fusion_failure"
    return "correct_target_not_retrieved"


def failure_row_from_eval(unit_row: dict[str, Any], candidates: Sequence[RankedCandidate]) -> dict[str, Any] | None:
    if unit_row.get("rank_of_first_gold"):
        return None
    return {
        "scenario": unit_row.get("scenario", ""),
        "method": unit_row.get("method", ""),
        "source_id": unit_row.get("source_id", ""),
        "source_kind": unit_row.get("source_kind", ""),
        "source_table": unit_row.get("source_table", ""),
        "source_column": unit_row.get("source_column", ""),
        "gold_target_uri": ";".join(unit_row.get("gold_target_uris", ())),
        "top_candidate_uris": ";".join(candidate.uri for candidate in candidates[:20]),
        "failure_category": unit_row.get("failure_category", ""),
        "candidate_count": unit_row.get("candidate_count", 0),
        "rank_of_first_gold": unit_row.get("rank_of_first_gold") or "",
        "reciprocal_rank": unit_row.get("reciprocal_rank", 0.0),
    }


def failure_row_from_issue(issue: GoldIssue, method: str) -> dict[str, Any]:
    return {
        "scenario": issue.scenario,
        "method": method,
        "source_id": "",
        "source_kind": issue.target_kind,
        "source_table": "",
        "source_column": "",
        "gold_target_uri": issue.target_uri,
        "top_candidate_uris": "",
        "failure_category": issue.failure_category,
        "candidate_count": 0,
        "rank_of_first_gold": "",
        "reciprocal_rank": 0.0,
    }


def aggregate_by_scenario(unit_rows: Sequence[dict[str, Any]], k_values: Sequence[int]) -> list[dict[str, Any]]:
    groups = _group_rows(unit_rows, ["scenario", "method", "source_kind"], include_all_source_kind=True)
    return _aggregate_groups(groups, k_values, ["scenario", "method", "source_kind"])


def aggregate_overall(unit_rows: Sequence[dict[str, Any]], k_values: Sequence[int]) -> list[dict[str, Any]]:
    groups = _group_rows(unit_rows, ["method", "source_kind"], include_all_source_kind=True)
    return _aggregate_groups(groups, k_values, ["method", "source_kind"])


def aggregate_by_role(unit_rows: Sequence[dict[str, Any]], k_values: Sequence[int]) -> list[dict[str, Any]]:
    groups = _group_rows(
        unit_rows,
        ["scenario", "method", "source_kind", "source_table_role", "source_column_role"],
        include_all_source_kind=True,
    )
    return _aggregate_groups(groups, k_values, ["scenario", "method", "source_kind", "source_table_role", "source_column_role"])


def aggregate_overall_by_role(unit_rows: Sequence[dict[str, Any]], k_values: Sequence[int]) -> list[dict[str, Any]]:
    groups = _group_rows(
        unit_rows,
        ["method", "source_kind", "source_table_role", "source_column_role"],
        include_all_source_kind=True,
    )
    return _aggregate_groups(groups, k_values, ["method", "source_kind", "source_table_role", "source_column_role"])


def _group_rows(
    rows: Sequence[dict[str, Any]],
    fields: Sequence[str],
    include_all_source_kind: bool = False,
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        keys = [tuple(row.get(field, "") for field in fields)]
        if include_all_source_kind and "source_kind" in fields:
            all_key = []
            for field in fields:
                all_key.append("all" if field == "source_kind" else row.get(field, ""))
            keys.append(tuple(all_key))
        for key in keys:
            groups.setdefault(key, []).append(row)
    return groups


def _aggregate_groups(
    groups: dict[tuple[Any, ...], list[dict[str, Any]]],
    k_values: Sequence[int],
    fields: Sequence[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        base = {field: key[idx] for idx, field in enumerate(fields)}
        n = len(rows)
        found_ranks = [int(row["rank_of_first_gold"]) for row in rows if row.get("rank_of_first_gold")]
        for k in k_values:
            metric = dict(base)
            metric["k"] = k
            metric["recall_at_k"] = sum(int(row.get(found_field(k), 0)) for row in rows) / n if n else 0.0
            metric["mrr"] = sum(float(row.get("reciprocal_rank", 0.0)) for row in rows) / n if n else 0.0
            metric["mean_rank_first_gold"] = (sum(found_ranks) / len(found_ranks)) if found_ranks else None
            metric["candidate_coverage"] = sum(1 for row in rows if int(row.get("candidate_count", 0)) > 0) / n if n else 0.0
            metric["zero_candidate_rate"] = sum(int(row.get("zero_candidates", 0)) for row in rows) / n if n else 0.0
            metric["number_of_gold_mapped_sources"] = n
            metric["number_of_sources_evaluated"] = n
            out.append(metric)
    return out
