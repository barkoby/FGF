from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any, Callable

from .constants import FALLBACK_CODE_MODEL, FALLBACK_MATCH_MODEL, REQUESTED_CODE_MODEL, REQUESTED_MATCH_MODEL
from .lexical import words
from .object_evidence import filter_object_evidence_for_matches
from .schema import Table, foreign_key_columns, is_generic_identifier_column, table_role
from .match_contract import match_output_schema, match_response_issues, validation_feedback

_LLM_EVENTS: list[str] = []
_LLM_EVENTS_LOCK = Lock()
CODEGEN_PROMPT_VERSION = "fgf_codegen_v3_runtime_validated"
CODEGEN_FEW_SHOT_PROMPT_VERSION = "fgf_codegen_v4_fewshot_runtime_validated"


from .providers import retry_config as _retry_config, structured_generate


def _model_output_max_attempts() -> int:
    value = int(os.getenv("CODING_FGF_MODEL_OUTPUT_MAX_ATTEMPTS", "6"))
    if value <= 0:
        raise ValueError("CODING_FGF_MODEL_OUTPUT_MAX_ATTEMPTS must be positive")
    return value


def _sleep_before_retry(delay: float, jitter: float) -> None:
    time.sleep(delay + (random.uniform(0.0, jitter) if jitter else 0.0))


@dataclass
class Match:
    source_id: str
    source_kind: str
    source_uri: str
    target_uri: str | None
    target_id: str | None
    confidence: float = 0.0
    target_kind: str = ""
    target_domain: list[str] | None = None
    target_range: list[str] | None = None
    target_local_name: str = ""
    reason: str = ""


def clear_llm_events() -> None:
    with _LLM_EVENTS_LOCK:
        _LLM_EVENTS.clear()


def llm_events() -> list[str]:
    with _LLM_EVENTS_LOCK:
        return list(_LLM_EVENTS)


def _append_llm_event(event: str) -> None:
    with _LLM_EVENTS_LOCK:
        _LLM_EVENTS.append(event)


def validate_matches(rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> list[Match]:
    allowed: dict[str, set[str]] = {}
    source_meta: dict[str, dict[str, Any]] = {}
    target_meta: dict[tuple[str, str], dict[str, Any]] = {}
    for row in candidate_rows:
        source = row["source"]
        source_id = str(source["id"])
        source_meta[source_id] = source
        allowed[source_id] = set()
        for candidate in row.get("candidates", []):
            uri = str(candidate["uri"])
            allowed[source_id].add(uri)
            target_meta[(source_id, uri)] = candidate
    matches: list[Match] = []
    for row in rows:
        source_id = str(row.get("source_id", ""))
        if source_id not in source_meta:
            continue
        target_uri = row.get("target_uri")
        if isinstance(target_uri, str) and target_uri.strip().lower() in {"", "null", "none", "no_match", "no match"}:
            target_uri = None
        if target_uri is not None and str(target_uri) not in allowed[source_id]:
            continue
        meta = source_meta[source_id]
        selected = target_meta.get((source_id, str(target_uri)), {}) if target_uri else {}
        target_id = row.get("target_id") or selected.get("id")
        if isinstance(target_id, str) and target_id.strip().lower() in {"", "null", "none", "no_match", "no match"}:
            target_id = None
        matches.append(
            Match(
                source_id=source_id,
                source_kind=str(meta.get("kind", row.get("source_kind", ""))),
                source_uri=str(meta.get("uri", "")),
                target_uri=None if target_uri in (None, "", "NO_MATCH") else str(target_uri),
                target_id=None if not target_id else str(target_id),
                target_kind=str(selected.get("kind", row.get("target_kind", ""))),
                target_domain=list(selected.get("domain", row.get("target_domain", [])) or []),
                target_range=list(selected.get("range", row.get("target_range", [])) or []),
                target_local_name=str(selected.get("local_name", row.get("target_local_name", ""))),
                confidence=float(row.get("confidence", 0.0)),
                reason=str(row.get("reason", "")),
            )
        )
    return matches


def offline_match(candidate_rows: list[dict[str, Any]]) -> list[Match]:
    rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        candidates = row.get("candidates", [])
        best = candidates[0] if candidates else None
        rows.append(
            {
                "source_id": row["source"]["id"],
                "target_uri": best.get("uri") if best else None,
                "target_id": best.get("id") if best else None,
                "confidence": 0.5 if best else 0.0,
                "reason": "offline nearest-candidate fallback",
            }
        )
    return validate_matches(rows, candidate_rows)


_FORCED_CLASS_MATCHES = {
    "persons": "Person",
    "documents": "Document",
    "registration_fees": "Registration_fee",
    "authors": "Author",
    "reviewers": "Reviewer",
    "conferences": "Conference",
    "program_committees": "ProgramCommittee",
    "pc_members": "ProgramCommitteeMember",
    "conf_members": "ConferenceMember",
}

_FORCED_DATA_MATCHES = {
    ("conferences", "name"): "name",
    ("conferences", "date"): "date",
    ("conferences", "site_url"): "siteURL",
    ("papers", "paper_id"): "paperID",
    ("papers", "title"): "title",
    ("paper_abstracts", "title"): "title",
    ("paper_full_versions", "paper_id"): "paperID",
    ("persons", "name"): "name",
    ("persons", "email"): "email",
    ("program_committees", "label"): "label",
    ("reviews", "comment"): "comment",
    ("paper", "paperid"): "paperID",
    ("paper", "title"): "title",
    ("documents", "title"): "hasTitle",
    ("registration_fees", "price"): "Price",
    ("programcommittee", "label"): "label",
    ("review", "comment"): "comment",
}

_FORCED_OBJECT_MATCHES = {
    ("papers", "author"): "hasAuthor",
    ("reviews", "written"): "writeReview",
    ("conference_members", "conference"): "hasConferenceMember",
    ("program_committee_members", "program_committee"): "hasProgramCommitteeMember",
    ("paper", "hasauthor"): "hasAuthor",
    ("review", "writtenby"): "writeReview",
    ("review", "hasreview_inv"): "hasReview",
    ("document_person", "pid"): "submit",
    ("document_person", "did"): "submit",
    ("reviews", "ref"): "hasReview",
    ("reviews", "review"): "hasReview",
    ("co-writepaper", "paper"): "co-writePaper",
}


def _candidate_name(candidate: dict[str, Any]) -> str:
    return str(candidate.get("local_name") or candidate.get("label") or candidate.get("uri") or "")


def _candidate_matches_name(candidate: dict[str, Any], desired: str) -> bool:
    uri = str(candidate.get("uri") or "")
    if desired == "label" and uri == "http://www.w3.org/2000/01/rdf-schema#label":
        return True
    return words(_candidate_name(candidate)) == words(desired)


def _match_from_candidate(source: dict[str, Any], candidate: dict[str, Any], reason: str) -> Match:
    return Match(
        source_id=str(source.get("id", "")),
        source_kind=str(source.get("kind", "")),
        source_uri=str(source.get("uri", "")),
        target_uri=str(candidate.get("uri")),
        target_id=str(candidate.get("id")) if candidate.get("id") else None,
        confidence=0.95,
        target_kind=str(candidate.get("kind", "")),
        target_domain=list(candidate.get("domain", []) or []),
        target_range=list(candidate.get("range", []) or []),
        target_local_name=str(candidate.get("local_name", "")),
        reason=reason,
    )


def _desired_match_name(source: dict[str, Any]) -> str | None:
    source_id = str(source.get("id", ""))
    if source_id.startswith("source-class:"):
        table = source_id.split(":", 1)[1]
        return _FORCED_CLASS_MATCHES.get(table) or _FORCED_CLASS_MATCHES.get(table.lower())
    if source_id.startswith("source-data:"):
        tail = source_id.split(":", 1)[1]
        table, column = tail.split(".", 1)
        return _FORCED_DATA_MATCHES.get((table, column)) or _FORCED_DATA_MATCHES.get((table.lower(), column.lower()))
    if source_id.startswith("source-object:"):
        tail = source_id.split(":", 1)[1]
        table, columns = tail.split(".", 1)
        return _FORCED_OBJECT_MATCHES.get((table, columns)) or _FORCED_OBJECT_MATCHES.get((table.lower(), columns.lower()))
    return None


def _best_repair_candidate(source: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    source_id = str(source.get("id", ""))
    if source_id.startswith("source-class:"):
        table = source_id.split(":", 1)[1]
        desired = _desired_match_name(source)
        if not desired:
            table_words = words(table)
            for candidate in candidates:
                if candidate.get("kind") == "class" and table_words == words(_candidate_name(candidate)):
                    return candidate
            return None
    else:
        desired = _desired_match_name(source)
    if not desired:
        return None
    for candidate in candidates:
        if _candidate_matches_name(candidate, desired):
            return candidate
    return None


def augment_candidate_rows_with_forced_matches(
    candidate_rows: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for row in candidate_rows:
        source = dict(row.get("source", {}))
        candidates = [dict(candidate) for candidate in row.get("candidates", []) or []]
        desired = _desired_match_name(source)
        if desired and not _best_repair_candidate(source, candidates):
            existing_uris = {str(candidate.get("uri")) for candidate in candidates}
            for target in target_records:
                if target.get("kind") != source.get("kind"):
                    continue
                if str(target.get("uri")) in existing_uris:
                    continue
                if _candidate_matches_name(target, desired):
                    forced = dict(target)
                    forced["distance"] = -1.0
                    candidates.insert(0, forced)
                    break
        augmented.append({"source": source, "candidates": candidates})
    return augmented


def repair_matches_with_candidates(matches: list[Match], candidate_rows: list[dict[str, Any]]) -> list[Match]:
    by_source = {match.source_id: match for match in matches}
    repaired: list[Match] = []
    for row in candidate_rows:
        source = row.get("source", {})
        source_id = str(source.get("id", ""))
        candidates = list(row.get("candidates", []) or [])
        repair = _best_repair_candidate(source, candidates)
        if repair:
            repaired.append(_match_from_candidate(source, repair, "deterministic lexical repair"))
            continue
        if source_id in by_source:
            repaired.append(by_source[source_id])
    return repaired


def compact_candidate_rows(candidate_rows: list[dict[str, Any]], max_candidates: int = 8, max_text: int = 500) -> list[dict[str, Any]]:
    return [
        {
            "source": _compact_entity(row.get("source", {}), max_text=max_text),
            "candidates": [
                _compact_entity(candidate, max_text=max_text, include_distance=True)
                for candidate in row.get("candidates", [])[:max_candidates]
            ],
        }
        for row in candidate_rows
    ]


def compact_match_request(candidate_row: dict[str, Any], max_candidates: int | None = None) -> dict[str, Any]:
    if max_candidates is None:
        max_candidates = int(os.getenv("CODING_FGF_MATCH_CANDIDATE_LIMIT", "8"))
    if max_candidates <= 0:
        raise ValueError("Matching candidate limit must be positive")
    return {
        "source": _compact_entity(candidate_row.get("source", {}), max_text=900),
        "candidates": [
            _minimal_candidate(candidate, max_text=180)
            for candidate in candidate_row.get("candidates", [])[:max_candidates]
        ],
    }


def _compact_entity(entity: dict[str, Any], max_text: int, include_distance: bool = False) -> dict[str, Any]:
    compact = {
        "id": entity.get("id"),
        "kind": entity.get("kind"),
        "uri": entity.get("uri"),
        "text": _truncate(str(entity.get("text", "")), max_text),
    }
    for key in ("source_table", "source_column", "table_role"):
        if entity.get(key):
            compact[key] = entity.get(key)
    if entity.get("primary_key"):
        compact["primary_key"] = entity.get("primary_key")
    if entity.get("foreign_keys"):
        compact["foreign_keys"] = entity.get("foreign_keys")
    if include_distance and entity.get("distance") is not None:
        compact["distance"] = entity.get("distance")
    return compact


def _minimal_candidate(entity: dict[str, Any], max_text: int) -> dict[str, Any]:
    compact = {
        "id": entity.get("id"),
        "uri": entity.get("uri"),
        "text": _truncate(str(entity.get("text", "")), max_text),
    }
    if entity.get("distance") is not None:
        compact["distance"] = entity.get("distance")
    if os.getenv("CODING_FGF_MATCH_CANDIDATE_CONTEXT", "full" if os.getenv("CODING_FGF_MATCH_CANDIDATE_LIMIT") else "minimal") == "full":
        for key in ("kind", "label", "local_name", "domain", "range", "parents", "comment"):
            if entity.get(key):
                compact[key] = entity[key]
    return compact


def _truncate(value: str, max_len: int) -> str:
    value = " ".join(value.split())
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


MATCH_FEW_SHOT_EXAMPLES: list[dict[str, Any]] = [
    {
        "example_id": "synthetic_exact_semantic_match",
        "source": {"id": "source-data:items.label", "kind": "data_property", "text": "table items column label"},
        "candidates": [
            {"id": "target-data:label", "uri": "urn:example:ontology#label", "text": "label textual name"},
            {"id": "target-data:createdAt", "uri": "urn:example:ontology#createdAt", "text": "creation timestamp"},
        ],
        "output": {
            "matches": [
                {
                    "source_id": "source-data:items.label",
                    "target_uri": "urn:example:ontology#label",
                    "target_id": "target-data:label",
                    "confidence": 0.94,
                    "reason": "The source column and candidate property both denote a textual label.",
                }
            ]
        },
    },
    {
        "example_id": "synthetic_no_match",
        "source": {
            "id": "source-object:item_roles.item_id",
            "kind": "object_property",
            "text": "subtype table primary key item_roles.item_id references items.id",
        },
        "candidates": [
            {"id": "target-object:assignedBy", "uri": "urn:example:ontology#assignedBy", "text": "task assigned by agent"},
            {"id": "target-object:approvesItem", "uri": "urn:example:ontology#approvesItem", "text": "agent approves item"},
        ],
        "output": {
            "matches": [
                {
                    "source_id": "source-object:item_roles.item_id",
                    "target_uri": None,
                    "target_id": None,
                    "confidence": 0.12,
                    "reason": "A subtype-table identity link is not the same semantics as any supplied action relationship.",
                }
            ]
        },
    },
    {
        "example_id": "synthetic_fk_prefers_object_property",
        "source": {
            "id": "source-object:items.owner_id",
            "kind": "object_property",
            "text": "foreign key from items.owner_id to agents.id",
        },
        "candidates": [
            {"id": "target-object:ownedBy", "uri": "urn:example:ontology#ownedBy", "text": "object property item owned by agent"},
            {"id": "target-data:agentName", "uri": "urn:example:ontology#agentName", "text": "agent name literal"},
        ],
        "output": {
            "matches": [
                {
                    "source_id": "source-object:items.owner_id",
                    "target_uri": "urn:example:ontology#ownedBy",
                    "target_id": "target-object:ownedBy",
                    "confidence": 0.9,
                    "reason": "The source is a relationship FK, so the object property is a better semantic match than a datatype property.",
                }
            ]
        },
    },
    {
        "example_id": "synthetic_identifier_property",
        "source": {"id": "source-data:articles.doi", "kind": "data_property", "text": "domain identifier DOI"},
        "candidates": [
            {"id": "target-data:doi", "uri": "urn:example:ontology#doi", "text": "document DOI identifier"},
            {"id": "target-object:hasAuthor", "uri": "urn:example:ontology#hasAuthor", "text": "document author relationship"},
        ],
        "output": {
            "matches": [
                {
                    "source_id": "source-data:articles.doi",
                    "target_uri": "urn:example:ontology#doi",
                    "target_id": "target-data:doi",
                    "confidence": 0.93,
                    "reason": "The source is a domain identifier literal and the candidate explicitly denotes the same identifier.",
                }
            ]
        },
    },
]


def _match_prompt(candidate_row: dict[str, Any], few_shot_examples: bool = False) -> str:
    source_id = str(candidate_row.get("source", {}).get("id", ""))
    candidate_uris = [str(candidate.get("uri", "")) for candidate in candidate_row.get("candidates", []) if candidate.get("uri")]
    if candidate_uris:
        candidate_constraint = (
            "Valid non-null target_uri values are exactly these supplied candidate URIs: "
            + json.dumps(candidate_uris, ensure_ascii=False)
            + ". Do not use any URI from memory or prior ontology knowledge.\n"
        )
    else:
        candidate_constraint = (
            "The supplied candidate list is empty. The only valid decision is exactly one match object "
            f"with source_id {json.dumps(source_id)}, target_uri null, target_id null, and decision rationale. "
            "Do not invent, infer, or import any ontology URI from memory.\n"
        )
    if few_shot_examples:
        payload = {
            "few_shot_examples": MATCH_FEW_SHOT_EXAMPLES,
            "request": compact_match_request(candidate_row),
        }
        null_template = {
            "matches": [
                {
                    "source_id": source_id,
                    "target_uri": None,
                    "target_id": None,
                    "confidence": 0.1,
                    "reason": "No supplied candidate expresses the same semantics.",
                }
            ]
        }
        prompt_version = "Prompt version: fgf_match_v2_fewshot\n"
        few_shot_instruction = (
            "The few_shot_examples are synthetic shape examples only. Do not copy their example URIs, "
            "tables, columns, source_id values, or target_id values. Real output must use only the supplied "
            "request source_id and candidate URI/id values. Return only the final JSON object with a top-level "
            "matches array; do not return or repeat the examples, and do not wrap the answer in an output, "
            "request, source, or candidates key.\n"
            f"The required source_id for this request is exactly {json.dumps(source_id)}. "
            "Every valid answer, including no-match answers, must contain exactly one match object with that "
            "exact source_id.\n"
            "For generic primary-key, subtype identity, or id-to-parent links, select null unless a supplied "
            "candidate expresses exactly that relation; action/object-property candidates are not enough.\n"
            "If no supplied candidate fits, use this exact top-level JSON shape with the current source_id: "
            + json.dumps(null_template, ensure_ascii=False)
            + "\n"
        )
        payload_text = json.dumps(payload, ensure_ascii=False)
    else:
        prompt_version = ""
        few_shot_instruction = ""
        payload_text = json.dumps(compact_match_request(candidate_row), ensure_ascii=False)
    return (
        prompt_version
        + "Choose the best ontology match for this one source entity.\n"
        "Return JSON: {\"matches\":[{\"source_id\":...,\"target_uri\":null|...,"
        "\"target_id\":null|...,\"confidence\":0..1,\"reason\":\"...\"}]}.\n"
        "Return exactly one match object for the supplied source_id. "
        "Only choose one of the candidate uri/id values. Use null when no candidate fits. "
        + candidate_constraint
        + few_shot_instruction
        + "Never map generic primary-key or foreign-key id columns to semantic data properties "
        "unless the column name/value is an actual domain identifier such as paper_id, doi, url, or email.\n\n"
        + payload_text
    )


def _local_name(value: str) -> str:
    if "#" in value:
        return value.rsplit("#", 1)[1]
    return value.rstrip("/").rsplit("/", 1)[-1]


def _is_truthy_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "t", "true", "yes", "y"}


def _distinct_values(rows: list[dict[str, Any]], column: str, limit: int = 8) -> list[str]:
    values: list[str] = []
    for row in rows:
        value = row.get(column)
        if value is None or value == "":
            continue
        text = str(value)
        if text not in values:
            values.append(text)
        if len(values) > limit:
            break
    return values


def _candidate_text(candidate: dict[str, Any]) -> str:
    return " ".join(
        str(candidate.get(key, ""))
        for key in ("local_name", "label", "comment", "uri", "parents")
    )


def _rank_class_candidates(source_words: set[str], target_records: list[dict[str, Any]], base_class_uri: str = "") -> list[dict[str, Any]]:
    rows: list[tuple[float, dict[str, Any]]] = []
    base_local_words = words(_local_name(base_class_uri)) if base_class_uri else set()
    for record in target_records:
        if record.get("kind") != "class":
            continue
        candidate_words = words(_candidate_text(record))
        score = len(source_words & candidate_words) * 4.0
        if base_local_words and (base_local_words & candidate_words):
            score += 2.0
        if base_class_uri and base_class_uri in [str(parent) for parent in record.get("parents", []) or []]:
            score += 8.0
        if source_words and source_words <= candidate_words:
            score += 3.0
        if score <= 0:
            continue
        row = dict(record)
        row["distance"] = -score
        rows.append((score, row))
    rows.sort(key=lambda item: (-item[0], str(item[1].get("uri", ""))))
    return [row for _, row in rows]


def _base_class_by_table(matches: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for match in matches:
        source_id = str(match.get("source_id", ""))
        if source_id.startswith("source-class:") and match.get("target_uri"):
            out[source_id.split(":", 1)[1]] = str(match["target_uri"])
    return out


def build_discriminator_candidate_rows(
    tables: dict[str, Table],
    rows_by_table: dict[str, list[dict[str, Any]]],
    target_records: list[dict[str, Any]],
    existing_matches: list[dict[str, Any]] | None = None,
    k: int = 16,
) -> list[dict[str, Any]]:
    """Build candidate rows for generic class discriminator decisions.

    This only creates candidate sets. The class target is still selected by the
    configured LLM in `llm_discriminator_matches`.
    """
    existing_matches = existing_matches or []
    base_classes = _base_class_by_table(existing_matches)
    candidate_rows: list[dict[str, Any]] = []
    for table in sorted(tables.values(), key=lambda item: item.name):
        table_rows = rows_by_table.get(table.name, [])
        for column in table.columns:
            lowered = column.name.lower()
            is_boolean = lowered.startswith("is_") or "bool" in column.datatype.lower()
            is_type = lowered == "type"
            if not is_boolean and not is_type:
                continue
            values = _distinct_values(table_rows, column.name)
            if is_type and not (1 < len(values) <= 8):
                continue
            if is_boolean:
                if table_rows and not any(_is_truthy_value(row.get(column.name)) for row in table_rows):
                    continue
                class_hint = column.name[3:] if lowered.startswith("is_") else column.name
                source_words = words(class_hint)
                row_filter = {"column": column.name, "truthy": True}
                source_id = f"source-discriminator:{table.name}.{column.name}"
                source_text = (
                    f"kind: class discriminator\nsource table: {table.name}\nsource column: {column.name}\n"
                    f"row filter: {row_filter}\nmeaning: rows where {column.name} is true should be typed as the selected class"
                )
                candidates = _rank_class_candidates(source_words, target_records, base_classes.get(table.name, ""))[:k]
                candidate_rows.append(
                    {
                        "source": {
                            "id": source_id,
                            "kind": "class",
                            "uri": f"urn:coding-fgf:source:{table.name}#{column.name}_discriminator",
                            "text": source_text,
                            "source_table": table.name,
                            "source_column": column.name,
                            "table_role": table_role(table),
                            "row_filter": row_filter,
                            "discriminator_kind": "boolean",
                        },
                        "candidates": candidates,
                    }
                )
            else:
                base_class = base_classes.get(table.name, "")
                source_words = words(table.name) | {"type"}
                candidates = _rank_class_candidates(source_words, target_records, base_class)[:k]
                for value in values:
                    row_filter = {"column": column.name, "equals": value}
                    source_id = f"source-discriminator:{table.name}.{column.name}:{value}"
                    sample_rows = [
                        {key: row.get(key) for key in table.column_names()[:6]}
                        for row in table_rows
                        if str(row.get(column.name)) == value
                    ][:3]
                    source_text = (
                        f"kind: class discriminator\nsource table: {table.name}\nsource column: {column.name}\n"
                        f"row filter: {row_filter}\nbase class: {base_class or table.name}\n"
                        f"sample rows: {json.dumps(sample_rows, ensure_ascii=False)}"
                    )
                    candidate_rows.append(
                        {
                            "source": {
                                "id": source_id,
                                "kind": "class",
                                "uri": f"urn:coding-fgf:source:{table.name}#{column.name}_{value}_discriminator",
                                "text": source_text,
                                "source_table": table.name,
                                "source_column": column.name,
                                "table_role": table_role(table),
                                "row_filter": row_filter,
                                "discriminator_kind": "type",
                            },
                            "candidates": candidates,
                        }
                    )
    return candidate_rows


def _discriminator_prompt(candidate_row: dict[str, Any]) -> str:
    source = candidate_row.get("source", {})
    candidates = candidate_row.get("candidates", []) or []
    candidate_uris = [str(candidate.get("uri", "")) for candidate in candidates if candidate.get("uri")]
    return (
        "Select a target ontology class for this schema discriminator, or null.\n"
        "Return JSON only: {\"matches\":[{\"source_id\":...,\"target_uri\":null|...,"
        "\"target_id\":null|...,\"confidence\":0..1,\"row_filter\":{...},\"reason\":\"...\"}]}.\n"
        "Choose only from supplied candidate URIs. Do not invent target URIs. "
        "Copy the supplied row_filter exactly into the chosen match. "
        "For boolean is_* columns, select the class named by the flag when a supplied candidate supports it. "
        "For TYPE/type values, select a subclass only when the schema/table context and candidate class make the value plausible; otherwise use null.\n"
        f"Valid target_uri values: {json.dumps(candidate_uris, ensure_ascii=False)}\n\n"
        + json.dumps(
            {
                "source": _compact_entity(source, max_text=1200),
                "row_filter": source.get("row_filter"),
                "candidates": [_compact_entity(candidate, max_text=350) for candidate in candidates],
            },
            ensure_ascii=False,
        )
    )


def _validate_discriminator_output(row: dict[str, Any], candidate_row: dict[str, Any]) -> dict[str, Any] | None:
    source = candidate_row.get("source", {})
    source_id = str(source.get("id", ""))
    allowed = {str(candidate.get("uri")): candidate for candidate in candidate_row.get("candidates", []) or []}
    if str(row.get("source_id", "")) != source_id:
        return None
    target_uri = row.get("target_uri")
    if isinstance(target_uri, str) and target_uri.strip().lower() in {"", "null", "none", "no_match", "no match"}:
        target_uri = None
    if target_uri is not None and str(target_uri) not in allowed:
        return None
    selected = allowed.get(str(target_uri), {}) if target_uri else {}
    target_id = row.get("target_id") or selected.get("id")
    if isinstance(target_id, str) and target_id.strip().lower() in {"", "null", "none", "no_match", "no match"}:
        target_id = None
    return {
        "source_id": source_id,
        "source_kind": "class",
        "source_uri": str(source.get("uri", "")),
        "target_uri": None if target_uri is None else str(target_uri),
        "target_id": None if not target_id else str(target_id),
        "confidence": float(row.get("confidence", 0.0) or 0.0),
        "target_kind": str(selected.get("kind", "class" if target_uri else "")),
        "target_domain": list(selected.get("domain", []) or []),
        "target_range": list(selected.get("range", []) or []),
        "target_local_name": str(selected.get("local_name", "")),
        "reason": str(row.get("reason", "")),
        "row_filter": source.get("row_filter"),
        "source_table": source.get("source_table"),
        "source_column": source.get("source_column"),
        "discriminator_kind": source.get("discriminator_kind"),
        "generated_by": "llm_discriminator_match",
    }


def _call_json_for_provider(
    prompt: str,
    schema_name: str,
    provider: str,
    model: str,
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
    fallback_model: str | None = None,
    output_schema: dict[str, Any] | None = None,
) -> Any:
    extra = {"output_schema": output_schema} if output_schema is not None else {}
    if provider == "openai":
        return call_structured_json(prompt, schema_name, model, fallback_model, **extra)
    return call_structured_json(
        prompt,
        schema_name,
        model,
        None,
        provider=provider,
        google_project=google_project,
        google_location=google_location,
        google_credentials=google_credentials,
        **extra,
    )


def llm_discriminator_matches(
    candidate_rows: list[dict[str, Any]],
    provider: str = "openai",
    model: str = REQUESTED_MATCH_MODEL,
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    validation_max_attempts = _model_output_max_attempts()
    for index, candidate_row in enumerate(candidate_rows, start=1):
        source_id = str(candidate_row.get("source", {}).get("id", ""))
        feedback = ""
        for attempt in range(1, validation_max_attempts + 1):
            data = _call_json_for_provider(
                _discriminator_prompt(candidate_row) + feedback,
                f"discriminator_match_{index}",
                provider,
                model,
                google_project,
                google_location,
                google_credentials,
                fallback_model=FALLBACK_MATCH_MODEL,
                output_schema=match_output_schema(candidate_row),
            )
            issues = match_response_issues(data, candidate_row)
            rows = [] if issues else data["matches"]
            parsed = [_validate_discriminator_output(row, candidate_row) for row in rows]
            parsed = [row for row in parsed if row is not None]
            if parsed:
                out.append(parsed[0])
                break
            _append_llm_event(f"discriminator:validation_retry:source={source_id}:attempt={attempt}")
            feedback = validation_feedback(issues)
        else:
            out.append(
                {
                    "source_id": source_id,
                    "source_kind": "class",
                    "source_uri": str(candidate_row.get("source", {}).get("uri", "")),
                    "target_uri": None,
                    "target_id": None,
                    "confidence": 0.0,
                    "target_kind": "",
                    "target_domain": [],
                    "target_range": [],
                    "target_local_name": "",
                    "reason": "LLM discriminator output failed validation; preserved as no-match without target fallback",
                    "row_filter": candidate_row.get("source", {}).get("row_filter"),
                    "source_table": candidate_row.get("source", {}).get("source_table"),
                    "source_column": candidate_row.get("source", {}).get("source_column"),
                    "discriminator_kind": candidate_row.get("source", {}).get("discriminator_kind"),
                    "generated_by": "llm_discriminator_match",
                    "invalid_selection": True,
                }
            )
    if os.getenv("CODING_FGF_MATCH_VALIDATION") == "all":
        out, _ = review_all_matches(out, candidate_rows, {}, provider, model, google_project, google_location, google_credentials)
    return out


_IDENTIFIER_TARGET_WORDS = {"id", "identifier", "paperid", "paper_id", "doi", "url", "email", "isbn", "issn", "code"}


def _target_identifier_like(match: dict[str, Any]) -> bool:
    text = " ".join(str(match.get(key, "")) for key in ("target_uri", "target_id", "target_local_name"))
    compact = text.lower().replace("-", "").replace("_", "")
    target_words = words(text)
    return bool(target_words & _IDENTIFIER_TARGET_WORDS) or any(word.replace("_", "") in compact for word in _IDENTIFIER_TARGET_WORDS)


def match_validation_issues(match: dict[str, Any], tables: dict[str, Table]) -> list[str]:
    target_uri = match.get("target_uri")
    if not target_uri:
        return []
    source_id = str(match.get("source_id", ""))
    source_kind = str(match.get("source_kind", ""))
    target_kind = str(match.get("target_kind", ""))
    issues: list[str] = []
    if source_kind and target_kind and source_kind != target_kind:
        issues.append(f"kind_mismatch:{source_kind}->{target_kind}")
    if source_id.startswith("source-data:"):
        tail = source_id.split(":", 1)[1]
        if "." not in tail:
            return issues
        table_name, column = tail.split(".", 1)
        table = tables.get(table_name)
        if not table:
            return issues
        fk_cols = foreign_key_columns(table)
        if column in fk_cols and target_kind == "data_property" and not _target_identifier_like(match):
            issues.append("fk_column_mapped_to_non_identifier_data_property")
        if column in fk_cols and target_kind == "data_property" and _target_identifier_like(match):
            domain_text = " ".join(str(value) for value in match.get("target_domain") or [])
            if table_name.lower() not in domain_text.lower() and column.lower() not in {"paper", "paperid", "paper_id"}:
                issues.append("fk_column_mapped_to_context_mismatched_identifier_property")
        if is_generic_identifier_column(table, column) and target_kind == "data_property" and not _target_identifier_like(match):
            issues.append("generic_identifier_mapped_to_non_identifier_data_property")
    return issues


def _validation_reask_prompt(match: dict[str, Any], candidate_row: dict[str, Any], issues: list[str]) -> str:
    return (
        "Review this previous ontology matching decision. The decision failed generic schema validation.\n"
        "Return JSON only in the same shape: {\"matches\":[{\"source_id\":...,\"target_uri\":null|...,"
        "\"target_id\":null|...,\"confidence\":0..1,\"reason\":\"...\"}]}.\n"
        "Choose only from the supplied candidate URIs, or null. Do not invent target URIs. "
        "If a source data column is only a foreign-key reference, prefer null because the relationship is represented by a source-object row.\n\n"
        + json.dumps(
            {
                "validation_issues": issues,
                "previous_decision": match,
                "request": compact_match_request(candidate_row, max_candidates=16),
            },
            ensure_ascii=False,
        )
    )



def review_all_matches(matches, candidate_rows, tables, provider="openai", model=REQUESTED_MATCH_MODEL,
                       google_project="", google_location="", google_credentials=""):
    """One semantic validation pass, using the original candidate set and source evidence."""
    by_source = {str(row["source"]["id"]): row for row in candidate_rows}
    def review(match):
        source_id = str(match["source_id"])
        candidate_row = by_source.get(source_id)
        if not candidate_row or not candidate_row.get("candidates"):
            return match, {"source_id": source_id, "status": "no_candidates", "before": match, "after": match}
        discriminator = source_id.startswith("source-discriminator:")
        request = _discriminator_prompt(candidate_row) if discriminator else _match_prompt(candidate_row)
        prompt = (
            "Independently validate the previous mapping. Return the same matches JSON shape. "
            "Review positive AND null decisions using only the supplied source evidence and candidates. "
            "Check entity versus group/role, kind, domain/range, labels, keys, relationship endpoints, "
            "and value patterns. A numeric discriminator's value/order is not evidence of subclass meaning. "
            "Use null when its meaning is unidentifiable. Distinguish lookup labels from opaque identifiers. "
            "Do not invent meanings from benchmark familiarity. Keep the supplied row_filter unchanged. "
            "Give a short reason citing concrete supplied source evidence.\n"
            + json.dumps({"previous_decision": match}, ensure_ascii=False) + "\n" + request)
        feedback = ""
        for attempt in range(_model_output_max_attempts()):
            data = _call_json_for_provider(prompt + feedback, "semantic_validation", provider, model,
                                          google_project, google_location, google_credentials,
                                          output_schema=match_output_schema(candidate_row))
            issues = match_response_issues(data, candidate_row)
            raw = [] if issues else data["matches"]
            if discriminator:
                parsed = [_validate_discriminator_output(r, candidate_row) for r in raw]
                parsed = [r for r in parsed if r]
            else:
                parsed = [asdict(r) for r in validate_matches(raw, [candidate_row])]
            semantic_warnings = []
            if parsed:
                # Lexical/FK heuristics are review evidence, not proof that a
                # supplied mapping is structurally invalid (especially after renaming).
                for issue in match_validation_issues(parsed[0], tables):
                    if issue.startswith("kind_mismatch:"):
                        issues.append(issue)
                    else:
                        semantic_warnings.append(issue)
            if len(parsed) == 1 and not issues:
                break
            feedback = validation_feedback(issues)
        else:
            raise RuntimeError("Semantic validation did not return one schema-valid decision after bounded attempts")
        replacement = parsed[0]
        replacement["validation"] = {"strategy": "validation", "initial_decision": match,
                                     "semantic_warnings": semantic_warnings}
        return replacement, {"source_id": source_id, "status": "reviewed", "before": match,
                             "after": replacement, "semantic_warnings": semantic_warnings}
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(review, matches))
    return [r[0] for r in results], [r[1] for r in results]


def reask_suspicious_matches(
    matches: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    tables: dict[str, Table],
    provider: str = "openai",
    model: str = REQUESTED_MATCH_MODEL,
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if os.getenv("CODING_FGF_MATCH_VALIDATION") == "all":
        return review_all_matches(matches, candidate_rows, tables, provider, model, google_project, google_location, google_credentials)
    by_source = {str(row.get("source", {}).get("id", "")): row for row in candidate_rows}
    updated: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []
    validation_max_attempts = _model_output_max_attempts()
    for match in matches:
        issues = match_validation_issues(match, tables)
        if not issues:
            updated.append(match)
            continue
        source_id = str(match.get("source_id", ""))
        candidate_row = by_source.get(source_id)
        replacement: dict[str, Any] | None = None
        if candidate_row:
            feedback = ""
            for attempt in range(1, validation_max_attempts + 1):
                data = _call_json_for_provider(
                    _validation_reask_prompt(match, candidate_row, issues) + feedback,
                    f"match_validation_{source_id.replace(':', '_').replace('.', '_')}",
                    provider,
                    model,
                    google_project,
                    google_location,
                    google_credentials,
                    fallback_model=FALLBACK_MATCH_MODEL,
                    output_schema=match_output_schema(candidate_row),
                )
                response_issues = match_response_issues(data, candidate_row)
                parsed = [] if response_issues else validate_matches(data["matches"], [candidate_row])
                if parsed:
                    row = asdict(parsed[0])
                    remaining = match_validation_issues(row, tables)
                    if not remaining:
                        replacement = row
                        break
                    issues = remaining
                _append_llm_event(f"match_validation:retry:source={source_id}:attempt={attempt}")
                feedback = validation_feedback(response_issues or issues)
        if replacement is None:
            replacement = dict(match)
            replacement.update(
                {
                    "target_uri": None,
                    "target_id": None,
                    "target_kind": "",
                    "target_domain": [],
                    "target_range": [],
                    "target_local_name": "",
                    "confidence": 0.0,
                    "reason": "LLM decision failed generic validation after re-ask; preserved as no-match without target fallback",
                    "invalidated_by_schema_validation": True,
                }
            )
        updated.append(replacement)
        report.append({"source_id": source_id, "issues": issues, "before": match, "after": replacement})
    return updated, report


def call_structured_json(
    prompt: str,
    schema_name: str,
    requested_model: str,
    fallback_model: str | None = None,
    provider: str = "openai",
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
    output_schema: dict[str, Any] | None = None,
) -> Any:
    return structured_generate(
        prompt, schema_name, requested_model, provider=provider,
        google_project=google_project, google_location=google_location,
        google_credentials=google_credentials, event_logger=_append_llm_event,
        **({"output_schema": output_schema} if output_schema is not None else {}),
    ).data


def _match_one_live(
    candidate_row: dict[str, Any],
    provider: str = "openai",
    model: str = REQUESTED_MATCH_MODEL,
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
    fail_on_invalid: bool = True,
    few_shot_examples: bool = False,
) -> tuple[list[Match], bool]:
    initial_delay, max_delay, jitter, max_attempts = _retry_config()
    delay = initial_delay
    attempt = 0
    source_id = str(candidate_row.get("source", {}).get("id", ""))
    validation_max_attempts = _model_output_max_attempts()
    feedback = ""
    while True:
        attempt += 1
        prompt = _match_prompt(candidate_row, few_shot_examples=few_shot_examples) + feedback
        if provider == "openai":
            data = call_structured_json(prompt, "matches", model, FALLBACK_MATCH_MODEL,
                                        output_schema=match_output_schema(candidate_row))
        else:
            data = call_structured_json(
                prompt,
                "matches",
                model,
                None,
                provider=provider,
                google_project=google_project,
                google_location=google_location,
                google_credentials=google_credentials,
                output_schema=match_output_schema(candidate_row),
            )
        issues = match_response_issues(data, candidate_row)
        matches = [] if issues else validate_matches(data["matches"], [candidate_row])
        if matches:
            return matches[:1], False
        if attempt >= validation_max_attempts:
            _append_llm_event(f"match:validation_exhausted:source={source_id}:attempts={attempt}")
            if fail_on_invalid:
                raise RuntimeError(f"LLM returned no valid match for {source_id} after {attempt} model-output attempts")
            return [
                Match(
                    source_id=source_id,
                    source_kind=str(candidate_row.get("source", {}).get("kind", "")),
                    source_uri=str(candidate_row.get("source", {}).get("uri", "")),
                    target_uri=None,
                    target_id=None,
                    confidence=0.0,
                    reason=f"model output failed generic candidate validation after {attempt} attempts",
                )
            ], False
        _append_llm_event(f"match:validation_retry:source={source_id}:attempt={attempt}:sleep={delay:.1f}")
        feedback = validation_feedback(issues)
        _sleep_before_retry(delay, jitter)
        delay = min(delay * 2.0, max_delay)


def _count_no_matches(matches: list[Match]) -> int:
    return sum(1 for match in matches if not match.target_uri)


def _log_match_progress(
    logger: Callable[[str], None] | None,
    completed: int,
    total: int,
    started_at: float,
    fallback_count: int,
    no_match_count: int,
    final: bool = False,
) -> None:
    if not logger:
        return
    elapsed = max(time.monotonic() - started_at, 0.001)
    percent = (completed / total * 100.0) if total else 100.0
    label = "match complete" if final else "match progress"
    logger(
        f"{label}: {completed}/{total} ({percent:.1f}%) elapsed={elapsed:.1f}s "
        f"throughput={completed / elapsed:.2f}/s fallbacks={fallback_count} no_matches={no_match_count}"
    )


def llm_match(
    candidate_rows: list[dict[str, Any]],
    offline: bool = False,
    max_workers: int = 4,
    progress_logger: Callable[[str], None] | None = None,
    progress_interval_seconds: float = 30.0,
    progress_every: int = 10,
    provider: str = "openai",
    model: str = REQUESTED_MATCH_MODEL,
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
    allow_deterministic_repair: bool = False,
    fail_on_invalid: bool = True,
    few_shot_examples: bool = False,
) -> list[Match]:
    if offline:
        _append_llm_event("match:offline")
        return offline_match(candidate_rows)
    total = len(candidate_rows)
    if total == 0:
        return []
    workers = max(1, int(max_workers or 1))
    results: list[list[Match] | None] = [None] * total
    live_rows: list[tuple[int, dict[str, Any]]] = []
    repaired_count = 0
    for index, row in enumerate(candidate_rows):
        source = row.get("source", {})
        repair = _best_repair_candidate(source, list(row.get("candidates", []) or [])) if allow_deterministic_repair else None
        if repair:
            results[index] = [_match_from_candidate(source, repair, "deterministic lexical pre-repair")]
            repaired_count += 1
        else:
            live_rows.append((index, row))
    if progress_logger:
        progress_logger(f"match pre-repair: repaired={repaired_count} live={len(live_rows)} total={total}")
    _append_llm_event(f"match:pre_repair:repaired={repaired_count}:live={len(live_rows)}:total={total}")
    if few_shot_examples:
        _append_llm_event("match:few_shot_examples")
    if not live_rows:
        return [match for row_matches in results if row_matches for match in row_matches]

    started_at = time.monotonic()
    if progress_logger:
        progress_logger(f"match start: total={len(live_rows)} workers={workers}")
    completed = 0
    fallback_count = 0
    no_match_count = 0
    last_log_at = started_at
    last_log_completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(
                _match_one_live,
                row,
                provider,
                model,
                google_project,
                google_location,
                google_credentials,
                fail_on_invalid,
                few_shot_examples,
            ): index
            for index, row in live_rows
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            matches, used_fallback = future.result()
            results[index] = matches
            completed += 1
            fallback_count += 1 if used_fallback else 0
            no_match_count += _count_no_matches(matches)
            now = time.monotonic()
            if completed == len(live_rows) or completed - last_log_completed >= progress_every or now - last_log_at >= progress_interval_seconds:
                _log_match_progress(progress_logger, completed, len(live_rows), started_at, fallback_count, no_match_count)
                last_log_at = now
                last_log_completed = completed
    _log_match_progress(progress_logger, completed, len(live_rows), started_at, fallback_count, no_match_count, final=True)
    elapsed = max(time.monotonic() - started_at, 0.001)
    _append_llm_event(
        f"match:complete:completed={completed}:total={len(live_rows)}:elapsed={elapsed:.1f}:"
        f"fallbacks={fallback_count}:no_matches={no_match_count}"
    )
    ordered: list[Match] = []
    for row_matches in results:
        if row_matches:
            ordered.extend(row_matches)
    return ordered


def _compact_tables_for_fol(tables: dict[str, Table], table_names: set[str] | None = None) -> list[dict[str, Any]]:
    selected_tables = tables.values() if table_names is None else [tables[name] for name in sorted(table_names) if name in tables]
    return [
        {
            "name": table.name,
            "columns": table.column_names(),
            "primary_key": table.primary_key,
            "foreign_keys": [
                {
                    "columns": fk.columns,
                    "ref_table": fk.ref_table,
                    "ref_columns": fk.ref_columns,
                }
                for fk in table.foreign_keys
            ],
            "table_role": table_role(table),
        }
        for table in selected_tables
    ]


def _source_table_from_match(match: dict[str, Any]) -> str:
    source_id = str(match.get("source_id", ""))
    if ":" not in source_id:
        return ""
    tail = source_id.split(":", 1)[1]
    if "." in tail:
        return tail.split(".", 1)[0]
    return tail


def _fol_tables_for_matches(matches: list[dict[str, Any]], tables: dict[str, Table]) -> set[str]:
    names = {_source_table_from_match(match) for match in matches}
    names = {name for name in names if name in tables}
    expanded = set(names)
    for name in names:
        for fk in tables[name].foreign_keys:
            if fk.ref_table in tables:
                expanded.add(fk.ref_table)
    return expanded


def _chunk_matches_by_table(matches: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for match in matches:
        grouped.setdefault(_source_table_from_match(match), []).append(match)
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for table_name in sorted(grouped):
        group = grouped[table_name]
        if current and len(current) + len(group) > chunk_size:
            chunks.append(current)
            current = []
        if len(group) > chunk_size:
            for index in range(0, len(group), chunk_size):
                chunks.append(group[index : index + chunk_size])
            continue
        current.extend(group)
    if current:
        chunks.append(current)
    return chunks or [[]]


def _schema_components(tables: dict[str, Table]) -> dict[str, int]:
    graph: dict[str, set[str]] = {name: set() for name in tables}
    for table in tables.values():
        for fk in table.foreign_keys:
            if fk.ref_table in tables:
                graph[table.name].add(fk.ref_table)
                graph[fk.ref_table].add(table.name)
    out: dict[str, int] = {}
    component = 0
    for name in sorted(graph):
        if name in out:
            continue
        stack = [name]
        while stack:
            current = stack.pop()
            if current in out:
                continue
            out[current] = component
            stack.extend(sorted(graph[current] - set(out)))
        component += 1
    return out


def _split_chunk(chunk: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    if len(chunk) <= chunk_size:
        return [chunk]
    return [chunk[index : index + chunk_size] for index in range(0, len(chunk), chunk_size)]


def _chunk_matches_by_policy(
    matches: list[dict[str, Any]],
    tables: dict[str, Table],
    chunk_size: int,
    batching: str = "none",
) -> list[list[dict[str, Any]]]:
    if batching in {"none", "table"}:
        return _chunk_matches_by_table(matches, chunk_size)
    if batching not in {"component", "hybrid"}:
        return _chunk_matches_by_table(matches, chunk_size)
    components = _schema_components(tables)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for match in matches:
        table = _source_table_from_match(match)
        grouped.setdefault(components.get(table, -1), []).append(match)
    chunks: list[list[dict[str, Any]]] = []
    for component in sorted(grouped):
        group = sorted(grouped[component], key=lambda row: str(row.get("source_id", "")))
        if batching == "component":
            chunks.extend(_split_chunk(group, chunk_size))
        else:
            table_grouped: dict[str, list[dict[str, Any]]] = {}
            for match in group:
                table_grouped.setdefault(_source_table_from_match(match), []).append(match)
            current: list[dict[str, Any]] = []
            for table_name in sorted(table_grouped):
                table_matches = table_grouped[table_name]
                if current and len(current) + len(table_matches) > chunk_size:
                    chunks.append(current)
                    current = []
                if len(table_matches) > chunk_size:
                    chunks.extend(_split_chunk(table_matches, chunk_size))
                else:
                    current.extend(table_matches)
            if current:
                chunks.append(current)
    return chunks or [[]]


def _budget_fol_chunks(
    chunks: list[list[dict[str, Any]]],
    tables: dict[str, Table],
    object_link_evidence: dict[str, Any] | None,
    few_shot_examples: bool,
    fol_batch_max_tokens: int,
) -> list[list[dict[str, Any]]]:
    if not fol_batch_max_tokens:
        return chunks
    max_chars = max(1, fol_batch_max_tokens) * 4
    budgeted: list[list[dict[str, Any]]] = []
    pending = list(chunks)
    while pending:
        chunk = pending.pop(0)
        chunk_evidence = filter_object_evidence_for_matches(object_link_evidence, chunk)
        prompt = generate_fol_prompt(
            chunk,
            tables,
            object_link_evidence=chunk_evidence,
            few_shot_examples=few_shot_examples,
        )
        if len(prompt) <= max_chars:
            budgeted.append(chunk)
            continue
        if len(chunk) <= 1:
            source_id = str(chunk[0].get("source_id", "")) if chunk else "<empty>"
            raise RuntimeError(
                "FOL prompt exceeds configured token budget for a single match: "
                f"source_id={source_id} chars={len(prompt)} approx_token_budget={fol_batch_max_tokens}"
            )
        mid = max(1, len(chunk) // 2)
        pending.insert(0, chunk[mid:])
        pending.insert(0, chunk[:mid])
    return budgeted or [[]]


def _merge_fol_rule_parts(parts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = {"class": [], "data": [], "object": []}
    seen: set[tuple[str, str]] = set()
    for part in parts:
        rules = part.get("rules", {}) if isinstance(part, dict) else {}
        for kind in ("class", "data", "object"):
            for rule in rules.get(kind, []) or []:
                key = (kind, json.dumps(rule, sort_keys=True, ensure_ascii=False))
                if key in seen:
                    continue
                seen.add(key)
                merged[kind].append(rule)
    return merged


FOL_FEW_SHOT_EXAMPLES: list[dict[str, Any]] = [
    {
        "example_id": "synthetic_discriminator_class_rule",
        "failure_mode_addressed": "missing discriminator-derived class rule",
        "selected_matches": [
            {
                "source_id": "source-discriminator:artifacts.status:published",
                "target_uri": "urn:example:ontology#PublishedArtifact",
            }
        ],
        "rule": {
            "class": [
                {
                    "source_table": "artifacts",
                    "target_class": "urn:example:ontology#PublishedArtifact",
                    "id_columns": ["id"],
                    "confidence": 0.91,
                    "table_role": "entity_table",
                    "row_filter": {"column": "status", "equals": "published"},
                    "match_ids": ["source-discriminator:artifacts.status:published"],
                }
            ],
            "data": [],
            "object": [],
        },
    },
    {
        "example_id": "synthetic_in_table_data_rule",
        "failure_mode_addressed": "missing non-FK literal attribute",
        "selected_matches": [
            {
                "source_id": "source-data:artifacts.title",
                "target_uri": "urn:example:ontology#title",
            }
        ],
        "rule": {
            "class": [],
            "data": [
                {
                    "source_table": "artifacts",
                    "source_column": "title",
                    "target_property": "urn:example:ontology#title",
                    "confidence": 0.93,
                    "match_ids": ["source-data:artifacts.title"],
                }
            ],
            "object": [],
        },
    },
    {
        "example_id": "synthetic_direct_fk_object_rule",
        "failure_mode_addressed": "wrong FK subject/object direction",
        "selected_matches": [
            {
                "source_id": "source-object:artifacts.owner_id",
                "target_uri": "urn:example:ontology#ownedBy",
            }
        ],
        "rule": {
            "class": [],
            "data": [],
            "object": [
                {
                    "source_table": "artifacts",
                    "source_columns": ["owner_id"],
                    "target_property": "urn:example:ontology#ownedBy",
                    "target_table": "agents",
                    "target_columns": ["id"],
                    "confidence": 0.9,
                    "match_ids": ["source-object:artifacts.owner_id"],
                }
            ],
        },
    },
    {
        "example_id": "synthetic_join_table_object_rule",
        "failure_mode_addressed": "association table endpoint selection",
        "selected_matches": [
            {
                "source_id": "source-object:artifact_agent.agent_id",
                "target_uri": "urn:example:ontology#hasContributor",
            }
        ],
        "rule": {
            "class": [],
            "data": [],
            "object": [
                {
                    "source_table": "artifact_agent",
                    "source_columns": ["agent_id"],
                    "target_property": "urn:example:ontology#hasContributor",
                    "target_table": "agents",
                    "target_columns": ["id"],
                    "subject_table": "artifacts",
                    "subject_columns": ["artifact_id"],
                    "subject_target_columns": ["id"],
                    "object_table": "agents",
                    "object_columns": ["agent_id"],
                    "object_target_columns": ["id"],
                    "plan_id": "synthetic-plan-artifact-agent-contributor",
                    "confidence": 0.88,
                    "match_ids": ["source-object:artifact_agent.agent_id"],
                }
            ],
        },
    },
]


def generate_fol_prompt(
    matches: list[dict[str, Any]],
    tables: dict[str, Table],
    object_link_evidence: dict[str, Any] | None = None,
    few_shot_examples: bool = False,
) -> str:
    selected = [match for match in matches if match.get("target_uri")]
    table_names = _fol_tables_for_matches(selected, tables)
    payload = {
        "tables": _compact_tables_for_fol(tables, table_names),
        "matches": selected,
        "required_output": {
            "rules": {
                "class": [
                    {
                        "source_table": "table name",
                        "target_class": "selected class URI",
                        "id_columns": ["primary key column"],
                        "confidence": 0.0,
                        "table_role": "entity_table|join_table|subtype_table",
                        "row_filter": "optional object with column and truthy/equals",
                        "match_ids": ["source match IDs used to justify this rule"],
                    }
                ],
                "data": [
                    {
                        "source_table": "table name",
                        "source_column": "column name",
                        "target_property": "selected datatype property URI",
                        "confidence": 0.0,
                        "subject_table": "optional referenced table for attribute tables",
                        "subject_columns": ["optional local FK columns"],
                        "subject_target_columns": ["optional referenced columns"],
                        "match_ids": ["source match IDs used to justify this rule"],
                    }
                ],
                "object": [
                    {
                        "source_table": "table name",
                        "source_columns": ["FK or association columns"],
                        "target_property": "selected object property URI",
                        "target_table": "referenced table name",
                        "target_columns": ["referenced columns"],
                        "confidence": 0.0,
                        "subject_table": "optional subject table for association tables",
                        "subject_columns": ["optional subject columns"],
                        "subject_target_columns": ["optional subject referenced columns"],
                        "object_table": "optional object table for association tables",
                        "object_columns": ["optional object columns"],
                        "object_target_columns": ["optional object referenced columns"],
                        "plan_id": "required when object_link_evidence is supplied; copy from selected legal plan",
                        "match_ids": ["source match IDs used to justify this rule"],
                    }
                ],
            }
        },
    }
    if few_shot_examples:
        payload["few_shot_examples"] = FOL_FEW_SHOT_EXAMPLES
    if object_link_evidence:
        payload["object_link_evidence"] = object_link_evidence
        object_instruction = (
            "For object rules, use only the supplied object_link_evidence legal_plans. "
            "Each emitted object rule must include plan_id and must copy source_table, source_columns, "
            "target_table, target_columns, subject/object fields, target_property, and match_ids exactly "
            "from that plan's rule_fields. If no supplied plan is appropriate, emit no object rule."
        )
    else:
        object_instruction = (
            "For object rules, map source-object matches using the matching foreign key direction; "
            "for association/join tables use subject/object table fields when needed."
        )
    if few_shot_examples:
        prompt_version = "Prompt version: fgf_fol_v3_fewshot_contextual\n"
        few_shot_instruction = (
            "The few_shot_examples are synthetic shape examples only. Do not copy their example URIs, "
            "tables, columns, match_ids, or plan_id values. Real output rules may use only the supplied "
            "selected matches, schema tables/columns, and object_link_evidence in the current payload.\n"
        )
    elif object_link_evidence:
        prompt_version = "Prompt version: fgf_fol_v2_object_evidence\n"
        few_shot_instruction = ""
    else:
        prompt_version = ""
        few_shot_instruction = ""
    return (
        prompt_version
        + "Generate FOL-style mapping rules from LLM-selected ontology matches and SQL schema only.\n"
        "Return JSON only with a top-level `rules` object containing arrays `class`, `data`, and `object`.\n"
        "Use only target URIs that appear in the supplied selected matches. Do not invent target URIs, "
        "source tables, source columns, mappings, benchmark-specific fixes, or deterministic repair rules.\n"
        + few_shot_instruction
        + "For class rules, map source-class matches to target_class. For data rules, map source-data matches "
        f"to target_property. {object_instruction}\n"
        "For source-discriminator matches, emit class rules with the supplied row_filter copied exactly. "
        "Every rule must include match_ids listing the source_id values of the selected matches that justify it. "
        "Avoid data rules for foreign-key columns when an object-property match represents the relationship.\n"
        "Use null/no rule rather than guessing when the supplied matches or schema do not justify a rule.\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
    )


def llm_fol(
    matches: list[dict[str, Any]],
    tables: dict[str, Table],
    offline: bool = False,
    provider: str = "openai",
    model: str = REQUESTED_CODE_MODEL,
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
    object_link_evidence: dict[str, Any] | None = None,
    few_shot_examples: bool = False,
    fol_batching: str = "none",
    fol_batch_max_matches: int | None = None,
    fol_batch_max_tokens: int = 0,
    fol_batch_overlap_strategy: str = "fk_neighbors",
) -> dict[str, Any]:
    if offline:
        raise RuntimeError("LLM FOL generation requires live mode; explicit offline mode may use deterministic FOL outside strict runs")
    selected = [match for match in matches if match.get("target_uri")]
    chunk_size = max(1, fol_batch_max_matches or int(os.getenv("CODING_FGF_FOL_MATCHES_PER_CALL", "12")))
    initial_chunks = _chunk_matches_by_policy(selected, tables, chunk_size, fol_batching)
    chunks = _budget_fol_chunks(
        initial_chunks,
        tables,
        object_link_evidence,
        few_shot_examples,
        fol_batch_max_tokens,
    )
    prompt_sizes: list[int] = []
    parts: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_evidence = filter_object_evidence_for_matches(object_link_evidence, chunk)
        prompt = generate_fol_prompt(
            chunk,
            tables,
            object_link_evidence=chunk_evidence,
            few_shot_examples=few_shot_examples,
        )
        prompt_sizes.append(len(prompt))
        schema_name = f"fol_chunk_{index}_of_{len(chunks)}" if len(chunks) > 1 else "fol"
        _append_llm_event(f"fol:chunk_start:index={index}:total={len(chunks)}:matches={len(chunk)}")
        if provider == "openai":
            data = call_structured_json(prompt, schema_name, model, FALLBACK_CODE_MODEL)
        else:
            data = call_structured_json(
                prompt,
                schema_name,
                model,
                None,
                provider=provider,
                google_project=google_project,
                google_location=google_location,
                google_credentials=google_credentials,
            )
        if isinstance(data, dict):
            parts.append(data)
        _append_llm_event(f"fol:chunk_complete:index={index}:total={len(chunks)}")
    rules = _merge_fol_rule_parts(parts)
    return {
        "rules": {
            "class": list(rules.get("class", []) or []),
            "data": list(rules.get("data", []) or []),
            "object": list(rules.get("object", []) or []),
        },
        "generation": {
            "source": "llm",
            "provider": provider,
            "model": model,
            "prompt_version": (
                "fgf_fol_v3_fewshot_contextual"
                if few_shot_examples
                else "fgf_fol_v2_object_evidence"
                if object_link_evidence
                else "fgf_fol_v1_llm_only"
            ),
            "few_shot_examples": bool(few_shot_examples),
            "chunks": len(chunks),
            "initial_chunks": len(initial_chunks),
            "matches_per_call": chunk_size,
            "fol_batching": fol_batching,
            "fol_batch_max_tokens": fol_batch_max_tokens,
            "fol_batch_overlap_strategy": fol_batch_overlap_strategy,
            "prompt_size_chars": prompt_sizes,
        },
    }


def generate_fol_repair_prompt(
    fol: dict[str, Any],
    matches: list[dict[str, Any]],
    tables: dict[str, Table],
    issues: list[dict[str, Any]],
) -> str:
    selected = [match for match in matches if match.get("target_uri")]
    table_names = _fol_tables_for_matches(selected, tables)
    return (
        "Repair these FOL-style mapping rules using only internal schema, selected LLM matches, and validation diagnostics.\n"
        "Return JSON only with top-level rules arrays: class, data, object. "
        "Do not use qpair names, gold answers, paper baselines, or benchmark-specific fixes. "
        "Do not invent target URIs or source columns. Use only selected match target_uri values. "
        "Every rule must include match_ids. Copy discriminator row_filter values exactly when used. "
        "Prefer removing invalid rules over guessing.\n\n"
        + json.dumps(
            {
                "tables": _compact_tables_for_fol(tables, table_names),
                "selected_matches": selected,
                "current_fol": fol,
                "validation_issues": issues,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def llm_repair_fol(
    fol: dict[str, Any],
    matches: list[dict[str, Any]],
    tables: dict[str, Table],
    issues: list[dict[str, Any]],
    provider: str = "openai",
    model: str = REQUESTED_CODE_MODEL,
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
) -> dict[str, Any]:
    prompt = generate_fol_repair_prompt(fol, matches, tables, issues)
    data = _call_json_for_provider(
        prompt,
        "fol_repair",
        provider,
        model,
        google_project,
        google_location,
        google_credentials,
        fallback_model=FALLBACK_CODE_MODEL,
    )
    rules = data.get("rules", {}) if isinstance(data, dict) else {}
    return {
        "rules": {
            "class": list(rules.get("class", []) or []),
            "data": list(rules.get("data", []) or []),
            "object": list(rules.get("object", []) or []),
        },
        "generation": {
            "source": "llm",
            "provider": provider,
            "model": model,
            "prompt_version": "fgf_fol_v1_llm_repair",
            "repaired_from_issues": len(issues),
        },
    }


def _rule_from_issue(fol: dict[str, Any], issue: dict[str, Any]) -> dict[str, Any] | None:
    rule_id = str(issue.get("rule_id", ""))
    if ":" not in rule_id:
        return None
    kind, raw_index = rule_id.split(":", 1)
    if kind not in {"class", "data", "object"} or not raw_index.isdigit():
        return None
    index = int(raw_index)
    rules = list(fol.get("rules", {}).get(kind, []) or [])
    if 0 <= index < len(rules):
        return {"kind": kind, "index": index, "rule": rules[index]}
    return None


def _round2_related_match_ids(issue: dict[str, Any], rule_record: dict[str, Any] | None) -> set[str]:
    out = {str(value) for value in issue.get("match_ids", []) or []}
    for key in ("source_id", "rule_id"):
        value = str(issue.get(key, ""))
        if value.startswith("source-"):
            out.add(value)
        if value.startswith("attribute:source-"):
            out.add(value.split("attribute:", 1)[1])
    if rule_record:
        out.update(str(value) for value in (rule_record.get("rule", {}) or {}).get("match_ids", []) or [])
    return {value for value in out if value}


def _round2_related_tables(
    issue: dict[str, Any],
    rule_record: dict[str, Any] | None,
    related_matches: list[dict[str, Any]],
    tables: dict[str, Table],
) -> set[str]:
    names: set[str] = set()
    for key in ("source_table", "target_table", "subject_table", "object_table"):
        value = str(issue.get(key, ""))
        if value in tables:
            names.add(value)
    if rule_record:
        rule = rule_record.get("rule", {}) or {}
        for key in ("source_table", "target_table", "subject_table", "object_table"):
            value = str(rule.get(key, ""))
            if value in tables:
                names.add(value)
    names.update(_fol_tables_for_matches(related_matches, tables))
    return names


def generate_fol_repair_round2_prompt(
    fol: dict[str, Any],
    matches: list[dict[str, Any]],
    tables: dict[str, Table],
    issues: list[dict[str, Any]],
    object_link_evidence: dict[str, Any] | None = None,
    max_issues: int = 8,
    allow_drop: bool = True,
) -> str:
    selected_by_id = {str(match.get("source_id", "")): match for match in matches if match.get("target_uri")}
    selected = [match for match in matches if match.get("target_uri")]
    selected_target_uris = sorted({str(match.get("target_uri")) for match in selected if match.get("target_uri")})
    issue_packets: list[dict[str, Any]] = []
    all_table_names: set[str] = set()
    for index, issue in enumerate(issues[: max(1, max_issues)], start=1):
        rule_record = _rule_from_issue(fol, issue)
        related_ids = _round2_related_match_ids(issue, rule_record)
        related_matches = [selected_by_id[mid] for mid in sorted(related_ids) if mid in selected_by_id]
        if not related_matches and str(issue.get("issue")) == "missing_discriminator_class_rule":
            rule_id = str(issue.get("rule_id", ""))
            if rule_id in selected_by_id:
                related_matches = [selected_by_id[rule_id]]
        table_names = _round2_related_tables(issue, rule_record, related_matches, tables)
        all_table_names.update(table_names)
        issue_packets.append(
            {
                "issue_id": f"round2_issue_{index}",
                "rule_id": issue.get("rule_id"),
                "issue": issue,
                "current_rule": rule_record,
                "relevant_selected_matches": related_matches,
                "allowed_target_uris": selected_target_uris,
            }
        )
    compact_object_evidence = filter_object_evidence_for_matches(
        object_link_evidence,
        [match for packet in issue_packets for match in packet.get("relevant_selected_matches", [])],
    )
    payload = {
        "issues": issue_packets,
        "tables": _compact_tables_for_fol(tables, all_table_names),
        "object_link_evidence": compact_object_evidence,
        "allowed_target_uris": selected_target_uris,
        "allow_drop": bool(allow_drop),
        "required_output": {
            "repairs": [
                {
                    "issue_id": "round2_issue_1",
                    "rule_id": "class:0|data:0|object:0|source-discriminator:...",
                    "action": "repair|drop|keep_with_justification|add_class_rule|add_data_rule|explain_no_rule_needed",
                    "reason": "brief internal-diagnostic explanation",
                    "rule": None,
                }
            ]
        },
    }
    return (
        "Prompt version: fgf_fol_v2_round2_repair\n"
        "You are repairing FOL-style mapping rules for a semantic data integration pipeline.\n"
        "Repair only the listed invalid or missing rules. Use only the supplied SQL schema, selected matches, "
        "object-link evidence, discriminator evidence, and validation diagnostics. Do not use qpair names, "
        "SQL/SPARQL gold answers, paper baselines, target triples, or dataset-specific fixes.\n"
        "For each issue choose exactly one action: repair, drop, keep_with_justification, add_class_rule, "
        "add_data_rule, or explain_no_rule_needed. add_class_rule is only for discriminator-derived class "
        "rules; add_data_rule is only for selected datatype matches missing a grounded data rule.\n"
        "Rules: do not invent target URIs, source tables, source columns, match ids, joins, constants, "
        "or row filters. Every non-null rule must cite valid match_ids. FK columns should not be emitted as "
        "datatype literals unless explicit non-FK literal evidence exists. Object-property rules require "
        "FK/object-link evidence. If support is insufficient, choose drop or explain_no_rule_needed.\n"
        "Return JSON only.\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
    )


def llm_repair_fol_round2(
    fol: dict[str, Any],
    matches: list[dict[str, Any]],
    tables: dict[str, Table],
    issues: list[dict[str, Any]],
    object_link_evidence: dict[str, Any] | None = None,
    max_issues: int = 8,
    allow_drop: bool = True,
    provider: str = "openai",
    model: str = REQUESTED_CODE_MODEL,
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
) -> dict[str, Any]:
    prompt = generate_fol_repair_round2_prompt(
        fol,
        matches,
        tables,
        issues,
        object_link_evidence=object_link_evidence,
        max_issues=max_issues,
        allow_drop=allow_drop,
    )
    data = _call_json_for_provider(
        prompt,
        "fol_repair_round2",
        provider,
        model,
        google_project,
        google_location,
        google_credentials,
        fallback_model=FALLBACK_CODE_MODEL,
    )
    repairs = data.get("repairs", []) if isinstance(data, dict) else []
    return {
        "repairs": list(repairs or []),
        "generation": {
            "source": "llm",
            "provider": provider,
            "model": model,
            "prompt_version": "fgf_fol_v2_round2_repair",
            "repaired_from_issues": len(issues),
            "max_issues": max_issues,
            "allow_drop": bool(allow_drop),
        },
    }


def llm_repair_attribute_coverage(
    prompt: str,
    provider: str = "openai",
    model: str = REQUESTED_CODE_MODEL,
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
) -> dict[str, Any]:
    data = _call_json_for_provider(
        prompt,
        "attribute_coverage_repair",
        provider,
        model,
        google_project,
        google_location,
        google_credentials,
        fallback_model=FALLBACK_CODE_MODEL,
    )
    repairs = data.get("repairs", []) if isinstance(data, dict) else []
    return {
        "repairs": list(repairs or []),
        "generation": {
            "source": "llm",
            "provider": provider,
            "model": model,
            "prompt_version": "fgf_attribute_coverage_v1_repair",
            "repair_count": len(repairs or []),
        },
    }


def llm_repair_materialization_coverage(
    prompt: str,
    provider: str = "openai",
    model: str = REQUESTED_CODE_MODEL,
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
) -> dict[str, Any]:
    data = _call_json_for_provider(
        prompt,
        "materialization_coverage_repair",
        provider,
        model,
        google_project,
        google_location,
        google_credentials,
        fallback_model=FALLBACK_CODE_MODEL,
    )
    repairs = data.get("repairs", []) if isinstance(data, dict) else []
    return {
        "repairs": list(repairs or []),
        "generation": {
            "source": "llm",
            "provider": provider,
            "model": model,
            "prompt_version": "fgf_materialization_coverage_v1_repair",
            "repair_count": len(repairs or []),
        },
    }


def llm_select_patterns(
    prompt: str,
    provider: str = "openai",
    model: str = REQUESTED_CODE_MODEL,
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
    schema_name: str = "pattern_selection",
) -> dict[str, Any]:
    data = _call_json_for_provider(
        prompt,
        schema_name,
        provider,
        model,
        google_project,
        google_location,
        google_credentials,
        fallback_model=FALLBACK_CODE_MODEL,
    )
    return data if isinstance(data, dict) else {}


def generate_targeted_object_repair_prompt(
    fol: dict[str, Any],
    matches: list[dict[str, Any]],
    tables: dict[str, Table],
    issues: list[dict[str, Any]],
    object_link_evidence: dict[str, Any],
    runtime_diagnostics: dict[str, Any] | None = None,
) -> str:
    object_rules = list(fol.get("rules", {}).get("object", []) or [])
    affected_indexes = {
        int(rule_id.split(":", 1)[1])
        for issue in issues
        if (rule_id := str(issue.get("rule_id", ""))).startswith("object:")
        and rule_id.split(":", 1)[1].isdigit()
    }
    object_match_ids = {
        str(match_id)
        for rule in object_rules
        for match_id in (rule.get("match_ids", []) or [])
    }
    affected_match_ids = {
        str(match_id)
        for index, rule in enumerate(object_rules)
        if not affected_indexes or index in affected_indexes
        for match_id in (rule.get("match_ids", []) or [])
    }
    selected_object_matches = [
        match
        for match in matches
        if match.get("target_uri")
        and (
            str(match.get("source_id", "")).startswith("source-object:")
            or str(match.get("source_id", "")) in object_match_ids
        )
        and (not affected_match_ids or str(match.get("source_id", "")) in affected_match_ids)
    ]
    table_names = _fol_tables_for_matches(selected_object_matches, tables)
    for rule in object_rules:
        for field in ("source_table", "target_table", "subject_table", "object_table"):
            value = str(rule.get(field, ""))
            if value in tables:
                table_names.add(value)
    payload = {
        "tables": _compact_tables_for_fol(tables, table_names),
        "selected_object_matches": selected_object_matches,
        "object_link_evidence": filter_object_evidence_for_matches(object_link_evidence, selected_object_matches),
        "current_object_rules": [{"index": index, "rule": rule} for index, rule in enumerate(object_rules)],
        "preserved_rule_counts": {
            "class": len(fol.get("rules", {}).get("class", []) or []),
            "data": len(fol.get("rules", {}).get("data", []) or []),
        },
        "object_validation_issues": issues,
        "runtime_diagnostics": runtime_diagnostics or {},
        "required_output": {
            "rules": {
                "object": [
                    {
                        "source_table": "copy from selected legal plan",
                        "source_columns": ["copy from selected legal plan"],
                        "target_property": "copy from selected legal plan",
                        "target_table": "copy from selected legal plan",
                        "target_columns": ["copy from selected legal plan"],
                        "subject_table": "copy from selected legal plan when present",
                        "subject_columns": ["copy from selected legal plan when present"],
                        "subject_target_columns": ["copy from selected legal plan when present"],
                        "object_table": "copy from selected legal plan when present",
                        "object_columns": ["copy from selected legal plan when present"],
                        "object_target_columns": ["copy from selected legal plan when present"],
                        "plan_id": "copy from selected legal plan",
                        "match_ids": ["copy from selected legal plan"],
                    }
                ]
            }
        },
    }
    return (
        "Prompt version: fgf_fol_v2_targeted_object_repair\n"
        "Repair only the object rules in this FOL object-rule list using internal schema, selected matches, "
        "object-link evidence, validation diagnostics, and runtime diagnostics. Return JSON only as "
        "{\"rules\":{\"object\":[...]}}. Do not return class or data rules; the runner preserves them unchanged.\n"
        "Academic-safety constraints: do not use qpair names, SQL/SPARQL gold answers, paper baselines, "
        "scenario-specific shortcuts, or benchmark-specific fixes. Do not invent target URIs, source tables, "
        "source columns, or mappings. Valid object "
        "rules unrelated to the listed issues must be preserved in the returned object list. Repaired object rules may use only supplied "
        "legal_plans and must include plan_id copied from the selected plan. Copy every table/column field exactly "
        "from the selected plan's rule_fields. Prefer removing an invalid object rule over guessing.\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
    )


def llm_repair_object_fol(
    fol: dict[str, Any],
    matches: list[dict[str, Any]],
    tables: dict[str, Table],
    issues: list[dict[str, Any]],
    object_link_evidence: dict[str, Any],
    runtime_diagnostics: dict[str, Any] | None = None,
    provider: str = "openai",
    model: str = REQUESTED_CODE_MODEL,
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
) -> dict[str, Any]:
    prompt = generate_targeted_object_repair_prompt(fol, matches, tables, issues, object_link_evidence, runtime_diagnostics)
    data = _call_json_for_provider(
        prompt,
        "fol_object_repair",
        provider,
        model,
        google_project,
        google_location,
        google_credentials,
        fallback_model=FALLBACK_CODE_MODEL,
    )
    rules = data.get("rules", {}) if isinstance(data, dict) else {}
    object_rules = rules.get("object", []) if isinstance(rules, dict) else data.get("object", []) if isinstance(data, dict) else []
    return {
        "rules": {
            "class": list(fol.get("rules", {}).get("class", []) or []),
            "data": list(fol.get("rules", {}).get("data", []) or []),
            "object": list(object_rules or []),
        },
        "generation": {
            "source": "llm",
            "provider": provider,
            "model": model,
            "prompt_version": "fgf_fol_v2_targeted_object_repair",
            "repaired_from_issues": len(issues),
        },
    }


def generate_codegen_prompt(
    fol: dict[str, Any],
    prompt_version: str = CODEGEN_PROMPT_VERSION,
    previous_code: str | None = None,
    diagnostics: dict[str, Any] | None = None,
    few_shot_examples: bool = False,
) -> str:
    payload: dict[str, Any] = {"fol": fol}
    if previous_code:
        payload["previous_code"] = previous_code
    if diagnostics:
        payload["runtime_diagnostics"] = diagnostics
    few_shot_text = ""
    if few_shot_examples:
        payload["few_shot_examples"] = [
            {
                "example_id": "synthetic_all_rule_kinds",
                "code": default_codegen(),
            },
            {
                "example_id": "synthetic_empty_safe_loop",
                "code": (
                    "def materialize(context):\n"
                    "    for rule in all_rules('class'):\n"
                    "        for row in rows(rule['source_table']):\n"
                    "            emit_type(row, rule)\n"
                    "    for rule in all_rules('data'):\n"
                    "        for row in rows(rule['source_table']):\n"
                    "            emit_data(row, rule)\n"
                    "    for rule in all_rules('object'):\n"
                    "        for row in rows(rule['source_table']):\n"
                    "            emit_object(row, rule)\n"
                ),
            },
            {
                "example_id": "synthetic_repair_preserves_helpers",
                "diagnostic": "A previous candidate skipped object rules.",
                "code": default_codegen(),
            },
        ]
        few_shot_text = (
            "The few_shot_examples are synthetic helper-only code shapes. Do not copy external constants, "
            "target URIs, table names, or dataset-specific logic from examples. Adapt only the approved "
            "helper-loop pattern to the supplied FOL rules.\n"
        )
    return (
        f"Prompt version: {prompt_version}\n"
        "You are an expert Python developer specializing in RDF transformations. "
        "Given validated FOL-style mapping rules, write JSON only: "
        "{\"code\":\"...python source...\"}.\n"
        "The code must define exactly `materialize(context)`. It runs in a sandbox "
        "where the only callable helpers are global functions `rows(table)`, "
        "`all_rules(kind)`, `emit_type(row, rule)`, `emit_data(row, rule)`, "
        "and `emit_object(row, rule)`. Do not import modules, read files, open "
        "network connections, call methods, or access dunder attributes.\n"
        "Runtime contract for fgf_codegen_v3_runtime_validated:\n"
        "- Iterate every rule kind present in the FOL: class, data, and object.\n"
        "- Attempt every reachable rule by iterating rows(rule['source_table']).\n"
        "- Emit triples only through the approved helper functions.\n"
        "- Do not inspect row values or implement row_filter yourself; helper functions "
        "already apply row filters, key checks, and diagnostics.\n"
        "- Do not use dotted attribute or method calls such as row.get(...), dict.get(...), "
        "items(), values(), keys(), append(), or any object.method(...).\n"
        "- Do not invent target URIs, constants, mappings, or benchmark-specific fixes.\n"
        "- Association-table object rules still call emit_object(row, rule); the helper "
        "handles subject/object lookup.\n"
        + few_shot_text
        + "Complete helper-only example:\n"
        "def materialize(context):\n"
        "    for rule in all_rules('class'):\n"
        "        for row in rows(rule['source_table']):\n"
        "            emit_type(row, rule)\n"
        "    for rule in all_rules('data'):\n"
        "        for row in rows(rule['source_table']):\n"
        "            emit_data(row, rule)\n"
        "    for rule in all_rules('object'):\n"
        "        for row in rows(rule['source_table']):\n"
        "            emit_object(row, rule)\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
    )


def llm_codegen(
    fol: dict[str, Any],
    offline: bool = False,
    prompt_version: str = CODEGEN_PROMPT_VERSION,
    previous_code: str | None = None,
    diagnostics: dict[str, Any] | None = None,
    provider: str = "openai",
    model: str = REQUESTED_CODE_MODEL,
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
    few_shot_examples: bool = False,
) -> str:
    if offline:
        _append_llm_event("codegen:offline")
        return default_codegen()
    prompt = generate_codegen_prompt(
        fol,
        prompt_version=prompt_version,
        previous_code=previous_code,
        diagnostics=diagnostics,
        few_shot_examples=few_shot_examples,
    )
    initial_delay, max_delay, jitter, max_attempts = _retry_config()
    delay = initial_delay
    attempt = 0
    validation_max_attempts = _model_output_max_attempts()
    while True:
        attempt += 1
        if provider == "openai":
            data = call_structured_json(prompt, "codegen", model, FALLBACK_CODE_MODEL)
        else:
            data = call_structured_json(
                prompt,
                "codegen",
                model,
                None,
                provider=provider,
                google_project=google_project,
                google_location=google_location,
                google_credentials=google_credentials,
            )
        code = data.get("code", "")
        if code:
            return str(code)
        if attempt >= validation_max_attempts:
            raise RuntimeError(f"LLM returned empty code after {attempt} model-output attempts")
        _append_llm_event(f"codegen:validation_retry:attempt={attempt}:sleep={delay:.1f}")
        _sleep_before_retry(delay, jitter)
        delay = min(delay * 2.0, max_delay)


def default_codegen() -> str:
    return (
        "def materialize(context):\n"
        "    for rule in all_rules('class'):\n"
        "        for row in rows(rule['source_table']):\n"
        "            emit_type(row, rule)\n"
        "    for rule in all_rules('data'):\n"
        "        for row in rows(rule['source_table']):\n"
        "            emit_data(row, rule)\n"
        "    for rule in all_rules('object'):\n"
        "        for row in rows(rule['source_table']):\n"
        "            emit_object(row, rule)\n"
    )


def matches_to_json(matches: list[Match]) -> dict[str, Any]:
    return {"matches": [asdict(match) for match in matches]}
