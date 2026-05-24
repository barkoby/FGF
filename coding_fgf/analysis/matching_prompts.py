from __future__ import annotations

import json
from typing import Any, Sequence

from ..llm import compact_match_request


METHOD_CURRENT_VALIDATED = "current_validated"
METHOD_COT_PROMPT = "cot_prompt"
METHOD_SELF_CONSISTENCY = "self_consistency"
METHOD_CHAIN_OF_VERIFICATION = "chain_of_verification"

MATCHING_METHODS = [
    METHOD_CURRENT_VALIDATED,
    METHOD_COT_PROMPT,
    METHOD_SELF_CONSISTENCY,
    METHOD_CHAIN_OF_VERIFICATION,
]

PROMPT_VERSIONS = {
    METHOD_CURRENT_VALIDATED: "match_current_validated_v1",
    METHOD_COT_PROMPT: "match_cot_style_v1",
    METHOD_SELF_CONSISTENCY: "match_self_consistency_v1",
    "chain_of_verification_stage1": "match_cov_stage1_v1",
    "chain_of_verification_stage2": "match_cov_stage2_v1",
    "current_validated_review": "match_current_no_match_review_v1",
    "current_validated_referee": "match_current_referee_v1",
}

NULL_CATEGORIES = {
    "",
    "raw_identifier",
    "fk_needs_object_property",
    "no_semantic_candidate",
    "ambiguous_candidate",
    "insufficient_context",
}


def validate_methods(methods: Sequence[str]) -> list[str]:
    unknown = [method for method in methods if method not in MATCHING_METHODS]
    if unknown:
        raise ValueError(f"Unknown matching method(s): {', '.join(unknown)}")
    return list(methods)


def match_current_validated_v1(candidate_row: dict[str, Any], top_k: int = 16) -> str:
    return (
        "Choose the best ontology match for this one source entity.\n"
        "Return JSON: {\"matches\":[{\"source_id\":...,\"target_uri\":null|...,"
        "\"target_id\":null|...,\"confidence\":0..1,\"reason\":\"...\"}]}.\n"
        "Only choose one of the candidate uri/id values. Use null when no candidate fits. "
        "Never map generic primary-key or foreign-key id columns to semantic data properties "
        "unless the column name/value is an actual domain identifier such as paper_id, doi, url, or email.\n\n"
        + json.dumps(compact_match_request(candidate_row, max_candidates=top_k), ensure_ascii=False)
    )


def match_current_validated_table_v1(candidate_rows: Sequence[dict[str, Any]], top_k: int = 16) -> str:
    payload = {
        "prompt_version": "match_current_validated_v1",
        "task": "Choose ontology matches for source schema entities from one source table using shared table context.",
        "required_output": {
            "matches": [
                {
                    "source_id": "source id from request",
                    "target_uri": "candidate uri or null",
                    "target_id": "candidate id or null",
                    "confidence": "0..1",
                    "decision": "selected or no_match",
                    "null_category": "one null category when target_uri is null",
                    "reason": "short evidence-based explanation",
                }
            ]
        },
        "constraints": [
            "Return JSON only.",
            "Return one decision for every source_id in requests.",
            "Choose at most one candidate per source_id and only from the supplied candidate uri/id values.",
            "Use null only when no candidate expresses the same class/property semantics.",
            "Use consistency across the table: class, data columns, object/FK records, PK/FK graph, sample values, and candidate domain/range should agree.",
            "Raw primary-key or foreign-key id columns should normally be null for datatype properties unless the ontology property is explicitly an identifier.",
            "Foreign-key records and association-table records should be considered for object properties.",
            "When target_uri is null, set null_category to exactly one of: raw_identifier, fk_needs_object_property, no_semantic_candidate, ambiguous_candidate, insufficient_context.",
            "Do not invent benchmark-specific URI rewrites, mappings, or repair shortcuts.",
        ],
        "request": _table_match_payload(candidate_rows, top_k),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def match_current_no_match_review_v1(candidate_rows: Sequence[dict[str, Any]], top_k: int = 16) -> str:
    payload = {
        "prompt_version": "match_current_no_match_review_v1",
        "task": "Review previous null match decisions. Select a supplied candidate only if semantic evidence is clear.",
        "constraints": [
            "Return JSON only.",
            "Return one decision for every source_id in requests.",
            "Use null when candidates are only lexical coincidences or raw FK/PK datatype mismatches.",
            "Do not invent new candidates or benchmark-specific repairs.",
        ],
        "request": _table_match_payload(candidate_rows, top_k),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def match_current_referee_v1(
    candidate_rows: Sequence[dict[str, Any]],
    prior_decisions: Sequence[dict[str, Any]],
    sibling_selected_matches: Sequence[dict[str, Any]],
    top_k: int = 16,
) -> str:
    payload = {
        "prompt_version": "match_current_referee_v1",
        "task": "Referee uncertain match decisions using supplied candidates, table context, sibling matches, and generic schema/ontology evidence.",
        "required_output": {
            "matches": [
                {
                    "source_id": "source id from request",
                    "target_uri": "candidate uri or null",
                    "target_id": "candidate id or null",
                    "confidence": "0..1",
                    "decision": "selected or keep_no_match",
                    "null_category": "required if target_uri is null",
                    "reason": "short evidence-based explanation",
                }
            ]
        },
        "constraints": [
            "Return JSON only.",
            "Return one decision for every source_id in requests.",
            "Choose at most one candidate per source_id and only from supplied candidate uri/id values.",
            "Do not change already selected sibling matches; use them only as consistency evidence.",
            "Recover a match only when candidate kind, name/comment, domain/range, FK/PK role, sample values, and sibling table context support it.",
            "For null target_uri, null_category must be exactly one of: raw_identifier, fk_needs_object_property, no_semantic_candidate, ambiguous_candidate, insufficient_context.",
            "Do not invent benchmark-specific URI rewrites, mappings, or repair shortcuts.",
        ],
        "request": _table_match_payload(candidate_rows, top_k),
        "prior_decisions": list(prior_decisions),
        "sibling_selected_matches": list(sibling_selected_matches),
        "null_categories": sorted(NULL_CATEGORIES - {""}),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def match_cot_style_v1(candidate_row: dict[str, Any], top_k: int = 16) -> str:
    payload = _rich_match_payload(candidate_row, top_k)
    return (
        "You are matching one relational source element to one ontology target candidate.\n"
        "Reason internally, but return JSON only. Do not reveal a chain of thought.\n"
        "Consider source kind, source table and column, table role, column role, primary keys, foreign keys, "
        "sample values, target kind, target label/local name/comment, target domain/range, and sibling consistency "
        "within the same source table.\n"
        "Select exactly one candidate or null. target_uri must be null or one of the supplied candidate URIs. "
        "Use null when no candidate expresses the same class/property semantics.\n"
        "Return exactly this shape: {\"matches\":[{\"source_id\":\"...\",\"target_uri\":\"candidate uri or null\","
        "\"target_id\":\"candidate id or null\",\"confidence\":0.0,\"decision\":\"selected|no_match\","
        "\"null_category\":\"raw_identifier|fk_needs_object_property|no_semantic_candidate|ambiguous_candidate|"
        "insufficient_context|\",\"evidence_summary\":[\"short evidence item 1\",\"short evidence item 2\","
        "\"short evidence item 3\"],\"reason\":\"one-sentence final explanation\"}]}.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def match_self_consistency_v1(candidate_row: dict[str, Any], top_k: int = 16) -> str:
    payload = _rich_match_payload(candidate_row, top_k)
    return (
        "You are independently sampling a matching decision for one source element. "
        "Reason internally, return JSON only, and do not provide a long chain of thought.\n"
        "Select exactly one candidate or null. target_uri must be null or one of the supplied candidate URIs. "
        "Use null when no candidate expresses the same class/property semantics.\n"
        "Return exactly this shape: {\"matches\":[{\"source_id\":\"...\",\"target_uri\":\"candidate uri or null\","
        "\"target_id\":\"candidate id or null\",\"confidence\":0.0,\"decision\":\"selected|no_match\","
        "\"null_category\":\"raw_identifier|fk_needs_object_property|no_semantic_candidate|ambiguous_candidate|"
        "insufficient_context|\",\"evidence_summary\":[\"short evidence item 1\",\"short evidence item 2\","
        "\"short evidence item 3\"],\"reason\":\"one-sentence final explanation\"}]}.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def match_cov_stage1_v1(candidate_row: dict[str, Any], top_k: int = 16) -> str:
    payload = _rich_match_payload(candidate_row, top_k)
    return (
        "Review the supplied ontology candidates for one source element. Return JSON only.\n"
        "Select the top 3 candidates or fewer. For each shortlisted candidate, provide exactly 3 concise, "
        "evidence-based reasons. You may include a null option when no candidate seems valid. "
        "Do not invent candidates; every target_uri must be one of the supplied candidate URIs.\n"
        "Return exactly this shape: {\"source_id\":\"...\",\"shortlist\":[{\"target_uri\":\"candidate uri\","
        "\"target_id\":\"candidate id\",\"candidate_rank\":1,\"confidence\":0.0,"
        "\"reasons\":[\"reason 1\",\"reason 2\",\"reason 3\"]}],"
        "\"null_option\":{\"allowed\":true,\"reasons\":[\"reason 1\",\"reason 2\",\"reason 3\"]}}.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def match_cov_stage2_v1(
    candidate_row: dict[str, Any],
    stage1: dict[str, Any],
    top_k: int = 16,
) -> str:
    payload = {
        "source": _source_payload(candidate_row.get("source", {})),
        "shortlist": stage1.get("shortlist", []),
        "null_option": stage1.get("null_option", {"allowed": False, "reasons": []}),
        "candidate_uris_allowed": [
            item.get("target_uri")
            for item in stage1.get("shortlist", [])
            if item.get("target_uri")
        ],
        "original_candidate_budget": top_k,
    }
    return (
        "Verify the shortlisted match choices for one source element. Return JSON only.\n"
        "Select the best final match or null. The final target_uri must be one of the shortlisted candidate URIs "
        "or null. Do not invent candidates.\n"
        "Return exactly this shape: {\"matches\":[{\"source_id\":\"...\",\"target_uri\":\"shortlisted candidate uri or null\","
        "\"target_id\":\"shortlisted candidate id or null\",\"confidence\":0.0,\"decision\":\"selected|no_match\","
        "\"null_category\":\"raw_identifier|fk_needs_object_property|no_semantic_candidate|ambiguous_candidate|"
        "insufficient_context|\",\"selected_from_stage1_rank\":1,"
        "\"reason\":\"concise final verification explanation\"}]}.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _rich_match_payload(candidate_row: dict[str, Any], top_k: int) -> dict[str, Any]:
    return {
        "source": _source_payload(candidate_row.get("source", {})),
        "candidates": [_candidate_payload(candidate) for candidate in candidate_row.get("candidates", [])[:top_k]],
    }


def _table_match_payload(candidate_rows: Sequence[dict[str, Any]], top_k: int) -> dict[str, Any]:
    rows = list(candidate_rows)
    first_source = rows[0].get("source", {}) if rows else {}
    return {
        "table_context": {
            "source_table": first_source.get("source_table") or first_source.get("id", ""),
            "table_role": first_source.get("source_table_role") or first_source.get("table_role", ""),
            "primary_key": first_source.get("primary_key", []),
            "foreign_keys": first_source.get("foreign_keys", []),
            "source_entities": [_source_payload(row.get("source", {})) for row in rows],
        },
        "requests": [_rich_match_payload(row, top_k) for row in rows],
    }


def _source_payload(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": source.get("id"),
        "kind": source.get("kind"),
        "uri": source.get("uri"),
        "source_table": source.get("source_table", ""),
        "source_column": source.get("source_column", ""),
        "source_table_role": source.get("source_table_role") or source.get("table_role", ""),
        "source_column_role": source.get("source_column_role", ""),
        "primary_key": source.get("primary_key", []),
        "foreign_keys": source.get("foreign_keys", []),
        "sample_values": source.get("sample_values", []),
        "text": source.get("text", ""),
    }


def _candidate_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": candidate.get("rank"),
        "id": candidate.get("id"),
        "uri": candidate.get("uri"),
        "kind": candidate.get("kind"),
        "local_name": candidate.get("local_name", ""),
        "label": candidate.get("label", ""),
        "comment": candidate.get("comment", ""),
        "domain": candidate.get("domain", []),
        "range": candidate.get("range", []),
        "score": candidate.get("score"),
        "method_scores": candidate.get("method_scores", {}),
        "text": candidate.get("text", ""),
    }
