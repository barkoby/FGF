"""Request-scoped matching contracts; never infer missing semantic mappings."""
from __future__ import annotations

import math


def match_output_schema(candidate_row):
    candidates = candidate_row.get("candidates", [])
    uris = list(dict.fromkeys(str(c["uri"]) for c in candidates))
    ids = list(dict.fromkeys(str(c["id"]) for c in candidates if c.get("id")))
    properties = {
        "source_id": {"type": "string", "enum": [str(candidate_row["source"]["id"])]},
        "target_uri": {"type": ["string", "null"], "enum": uris + [None]},
        "target_id": {"type": ["string", "null"], "enum": ids + [None]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    }
    return {"type": "object", "properties": {"matches": {
        "type": "array", "minItems": 1, "maxItems": 1,
        "items": {"type": "object", "properties": properties,
                  "required": list(properties), "additionalProperties": False}}},
        "required": ["matches"], "additionalProperties": False}


def match_response_issues(data, candidate_row):
    rows = data.get("matches") if isinstance(data, dict) else None
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        return ["expected_exactly_one_match_object"]
    row = rows[0]
    issues = []
    if row.get("source_id") != candidate_row["source"]["id"]:
        issues.append("source_id_mismatch: copy the request source.id exactly")
    allowed = {str(c["uri"]): c for c in candidate_row.get("candidates", [])}
    uri, target_id = row.get("target_uri"), row.get("target_id")
    if uri is None:
        if target_id is not None:
            issues.append("null_target_requires_null_target_id")
    elif not isinstance(uri, str) or uri not in allowed:
        issues.append("target_uri_not_in_supplied_candidates: use a supplied URI or null")
    elif target_id is not None and target_id != allowed[uri].get("id"):
        issues.append("target_id_uri_mismatch: use the ID belonging to the selected URI")
    confidence = row.get("confidence", 0.0)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        issues.append("confidence_must_be_finite_number_between_zero_and_one")
    return issues


def validation_feedback(issues):
    return ("\nThe previous response failed validation: " + "; ".join(issues)
            + ". Return one corrected decision. Keep the exact request source.id. "
            "An internal ontology ID is not a URI. Use null for both target fields when no candidate fits.\n")
