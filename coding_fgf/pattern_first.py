from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from .lexical import words
from .materialization_coverage import internal_materialization_score, materialization_coverage_summary
from .schema import SqlData, Table, foreign_key_columns, is_generic_identifier_column, table_key, table_role


TEXT_PROFILE_WORDS = {
    "title": {"title", "name", "label"},
    "abstract": {"abstract", "summary"},
    "review": {"review", "comment", "text", "description"},
    "isbn": {"isbn"},
    "date": {"date", "time", "year"},
    "email": {"email", "mail"},
    "url": {"url", "uri", "link", "homepage", "web"},
}

PATTERN_TYPES = {
    "SCHEMA_ENTITY",
    "FK_OBJECT",
    "JOIN_TABLE_OBJECT",
    "ASSOCIATION_ENTITY",
    "SUBCLASS_PKFK",
    "WEAK_ENTITY",
    "LITERAL_ATTRIBUTE",
    "IDENTIFIER_AS_URI",
}

FORBIDDEN_PROMPT_TERMS = {
    "qpair",
    "gold",
    "llm4vkg",
    "false positive",
    "false negative",
    "paper score",
    "expected triple",
    "sparql answer",
    "sql answer",
}


def _local(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[1]
    return uri.rstrip("/").rsplit("/", 1)[-1]


def _is_uri(value: Any) -> bool:
    text = str(value or "")
    return text.startswith(("http://", "https://", "urn:"))


def _target_words(record: dict[str, Any]) -> set[str]:
    return words(" ".join(str(record.get(key, "")) for key in ("target_uri", "target_id", "target_local_name", "local_name", "id")))


def _column_profile_flags(table: str, column: str, values: list[str]) -> dict[str, Any]:
    column_words = words(f"{table} {column}")
    joined = " ".join(values[:50]).lower()
    flags: dict[str, Any] = {}
    for name, indicators in TEXT_PROFILE_WORDS.items():
        flags[f"{name}_like"] = bool(column_words & indicators)
    flags["contains_at_sign"] = any("@" in value for value in values[:20])
    flags["contains_url_prefix"] = any(value.lower().startswith(("http://", "https://")) for value in values[:20])
    flags["contains_isbn_shape"] = bool(re.search(r"\b(?:97[89][-\s]?)?\d[-\s]?\d{2,5}[-\s]?\d{2,7}[-\s]?[0-9x]\b", joined))
    flags["text_like"] = any(flags.get(f"{name}_like") for name in ("title", "abstract", "review", "isbn", "email", "url"))
    return flags


def _identifier_likeness(table: Table, column: str, distinct: int, non_null: int) -> float:
    score = 0.0
    lowered = column.lower()
    if column in table.primary_key:
        score += 0.6
    if column in foreign_key_columns(table):
        score += 0.5
    if lowered == "id" or lowered.endswith("_id") or lowered in {"pid", "rid", "aid", "oid", "cid", "uid"}:
        score += 0.4
    if non_null and distinct / max(1, non_null) > 0.95:
        score += 0.2
    return min(1.0, score)


def value_profile(table: Table, data: SqlData, column: str) -> dict[str, Any]:
    rows = data.rows.get(table.name, [])
    values = [row.get(column) for row in rows]
    non_empty = [str(value) for value in values if value not in (None, "")]
    lengths = [len(value) for value in non_empty]
    distinct = len(set(non_empty))
    row_count = len(rows)
    non_null = len(non_empty)
    flags = _column_profile_flags(table.name, column, non_empty)
    return {
        "table": table.name,
        "column": column,
        "row_count": row_count,
        "non_null_count": non_null,
        "null_rate": 1.0 - (non_null / row_count) if row_count else 1.0,
        "distinct_count": distinct,
        "distinct_ratio": distinct / max(1, non_null),
        "examples": non_empty[:5],
        "min_text_length": min(lengths) if lengths else 0,
        "max_text_length": max(lengths) if lengths else 0,
        "mean_text_length": sum(lengths) / len(lengths) if lengths else 0.0,
        "identifier_likeness": _identifier_likeness(table, column, distinct, non_null),
        **flags,
    }


def build_schema_graph(tables: dict[str, Table], data: SqlData) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    value_profiles: dict[str, dict[str, Any]] = {}
    for table_name, table in sorted(tables.items()):
        nodes.append(
            {
                "id": f"table:{table_name}",
                "kind": "table",
                "table": table_name,
                "role": table_role(table),
                "primary_key": list(table.primary_key),
                "row_count": len(data.rows.get(table_name, [])),
            }
        )
        for column in table.columns:
            profile = value_profile(table, data, column.name)
            value_profiles[f"{table_name}.{column.name}"] = profile
            nodes.append(
                {
                    "id": f"column:{table_name}.{column.name}",
                    "kind": "column",
                    "table": table_name,
                    "column": column.name,
                    "datatype": column.datatype,
                    "nullable": column.nullable,
                    "is_primary_key": column.name in table.primary_key,
                    "is_foreign_key": column.name in foreign_key_columns(table),
                    "profile": profile,
                }
            )
            edges.append({"source": f"table:{table_name}", "target": f"column:{table_name}.{column.name}", "kind": "contains_column"})
            if column.name in table.primary_key:
                edges.append({"source": f"table:{table_name}", "target": f"column:{table_name}.{column.name}", "kind": "primary_key"})
        for fk in table.foreign_keys:
            edges.append(
                {
                    "source": f"table:{table_name}",
                    "target": f"table:{fk.ref_table}",
                    "kind": "foreign_key",
                    "columns": list(fk.columns),
                    "ref_columns": list(fk.ref_columns),
                }
            )
            edges.append(
                {
                    "source": f"table:{fk.ref_table}",
                    "target": f"table:{table_name}",
                    "kind": "reverse_foreign_key",
                    "columns": list(fk.ref_columns),
                    "ref_columns": list(fk.columns),
                }
            )
            if set(fk.columns) == set(table.primary_key):
                edges.append({"source": f"table:{table_name}", "target": f"table:{fk.ref_table}", "kind": "primary_key_as_foreign_key"})
    return {
        "version": "pattern_schema_graph_v1",
        "nodes": nodes,
        "edges": edges,
        "value_profiles": value_profiles,
    }


def _source_id_parts(source_id: str) -> tuple[str, str, list[str]]:
    if ":" not in source_id:
        return "", "", []
    kind, tail = source_id.split(":", 1)
    if "." not in tail:
        return kind, tail, []
    table, cols = tail.split(".", 1)
    return kind, table, [part for part in cols.split(",") if part]


def _candidate_rows_by_source(candidate_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in candidate_rows:
        source_id = str((row.get("source") or {}).get("id", ""))
        if source_id:
            out[source_id] = row
    return out


def _selected_by_source(matches: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(match.get("source_id")): match for match in matches if match.get("source_id") and match.get("target_uri")}


def _target_options(source_id: str, matches: list[dict[str, Any]], candidate_row: dict[str, Any] | None) -> list[dict[str, Any]]:
    selected = _selected_by_source(matches).get(source_id)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    if selected and selected.get("target_uri"):
        uri = str(selected["target_uri"])
        seen.add(uri)
        out.append(
            {
                "target_uri": uri,
                "target_id": selected.get("target_id"),
                "target_kind": selected.get("target_kind") or selected.get("kind"),
                "target_local_name": selected.get("target_local_name"),
                "target_domain": selected.get("target_domain") or [],
                "target_range": selected.get("target_range") or [],
                "selected_match": True,
                "source_match": selected,
                "rank": 0,
            }
        )
    for rank, candidate in enumerate((candidate_row or {}).get("candidates", []) or [], start=1):
        uri = str(candidate.get("uri") or candidate.get("target_uri") or "")
        if not uri or uri in seen:
            continue
        seen.add(uri)
        out.append(
            {
                "target_uri": uri,
                "target_id": candidate.get("id") or candidate.get("target_id"),
                "target_kind": candidate.get("kind") or candidate.get("target_kind"),
                "target_local_name": candidate.get("local_name") or candidate.get("target_local_name"),
                "target_domain": candidate.get("domain") or candidate.get("target_domain") or [],
                "target_range": candidate.get("range") or candidate.get("target_range") or [],
                "selected_match": False,
                "source_match": selected,
                "rank": rank,
                "score": candidate.get("score"),
            }
        )
    return out


def _kind_allows(option: dict[str, Any], target_kind: str) -> bool:
    kind = str(option.get("target_kind") or "").lower()
    if target_kind == "class":
        return "class" in kind or kind in {"", "ontology_class"}
    if target_kind == "data":
        return "data" in kind or "datatype" in kind or kind in {"property", "rdf_property", ""}
    if target_kind == "object":
        return "object" in kind or kind in {"property", "rdf_property", ""}
    return False


def _domain_range_score(option: dict[str, Any], domain_classes: list[str], range_classes: list[str]) -> int:
    def side_score(values: list[str], classes: list[str]) -> int:
        if not values:
            return 1
        scores: list[int] = []
        for value in values:
            vwords = words(_local(value))
            for cls in classes:
                cwords = words(_local(cls))
                scores.append(2 if vwords == cwords else 1 if vwords & cwords else 0)
        return max(scores, default=0)

    return side_score([str(v) for v in option.get("target_domain") or []], domain_classes) + side_score(
        [str(v) for v in option.get("target_range") or []], range_classes
    )


def _class_map_from_matches(matches: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for match in matches:
        source_id = str(match.get("source_id", ""))
        if source_id.startswith("source-class:") and match.get("target_uri"):
            table = source_id.split(":", 1)[1]
            uri = str(match["target_uri"])
            if uri not in out[table]:
                out[table].append(uri)
    return dict(out)


def _profile_compatible_with_target(profile: dict[str, Any], option: dict[str, Any]) -> bool:
    target = _target_words(option)
    if profile.get("identifier_likeness", 0) >= 0.7:
        identifier_targets = {"id", "identifier", "doi", "url", "email", "isbn", "issn", "code"}
        return bool(target & identifier_targets)
    text_targets = {"title", "name", "label", "abstract", "review", "comment", "text", "isbn", "date", "email", "url"}
    if target & text_targets:
        return bool(profile.get("text_like") or profile.get("mean_text_length", 0) >= 3)
    return True


def _pattern_record(
    *,
    pattern_id: str,
    pattern_type: str,
    source_id: str,
    source_tables: list[str],
    source_columns: list[str],
    target_option: dict[str, Any],
    rule_fields: dict[str, Any],
    evidence: dict[str, Any],
    score: float,
) -> dict[str, Any]:
    return {
        "pattern_id": pattern_id,
        "pattern_type": pattern_type,
        "source_id": source_id,
        "source_tables": source_tables,
        "source_columns": source_columns,
        "target_uri": target_option.get("target_uri"),
        "target_id": target_option.get("target_id"),
        "target_kind": target_option.get("target_kind"),
        "target_local_name": target_option.get("target_local_name"),
        "target_domain": target_option.get("target_domain") or [],
        "target_range": target_option.get("target_range") or [],
        "from_selected_match": bool(target_option.get("selected_match")),
        "candidate_rank": target_option.get("rank"),
        "score": score,
        "rule_fields": rule_fields,
        "evidence": evidence,
    }


def extract_pattern_candidates(
    tables: dict[str, Table],
    data: SqlData,
    matches: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    schema_graph: dict[str, Any],
) -> list[dict[str, Any]]:
    by_source = _candidate_rows_by_source(candidate_rows)
    class_by_table = _class_map_from_matches(matches)
    value_profiles = schema_graph.get("value_profiles", {})
    patterns: list[dict[str, Any]] = []

    for table_name, table in sorted(tables.items()):
        source_id = f"source-class:{table_name}"
        if table.primary_key and any(column.name not in foreign_key_columns(table) for column in table.columns):
            for option in _target_options(source_id, matches, by_source.get(source_id)):
                if not _kind_allows(option, "class") or not _is_uri(option.get("target_uri")):
                    continue
                pattern_type = "SUBCLASS_PKFK" if any(fk.columns == table.primary_key for fk in table.foreign_keys) else "SCHEMA_ENTITY"
                lexical = 1.0 if words(table_name) & _target_words(option) else 0.25
                patterns.append(
                    _pattern_record(
                        pattern_id=f"{pattern_type}:{source_id}:{len(patterns)}",
                        pattern_type=pattern_type,
                        source_id=source_id,
                        source_tables=[table_name],
                        source_columns=list(table.primary_key),
                        target_option=option,
                        rule_fields={
                            "source_table": table_name,
                            "target_class": option.get("target_uri"),
                            "id_columns": list(table_key(table)),
                            "table_role": table_role(table),
                            "match_ids": [source_id],
                        },
                        evidence={"table_role": table_role(table), "row_count": len(data.rows.get(table_name, [])), "lexical_score": lexical},
                        score=3.0 + lexical + (1.0 if option.get("selected_match") else 0.0),
                    )
                )
        for column in table.column_names():
            source_id = f"source-data:{table_name}.{column}"
            profile = value_profiles.get(f"{table_name}.{column}", {})
            for option in _target_options(source_id, matches, by_source.get(source_id)):
                if not _kind_allows(option, "data") or not _is_uri(option.get("target_uri")):
                    continue
                fk_like = column in foreign_key_columns(table) or is_generic_identifier_column(table, column)
                target_id_like = bool(_target_words(option) & {"id", "identifier", "doi", "url", "email", "isbn", "issn", "code"})
                if fk_like and not target_id_like:
                    continue
                if not _profile_compatible_with_target(profile, option):
                    continue
                rule_fields = {
                    "source_table": table_name,
                    "source_column": column,
                    "target_property": option.get("target_uri"),
                    "match_ids": [source_id],
                }
                subject_fk = None
                if table_name not in class_by_table:
                    for fk in table.foreign_keys:
                        if class_by_table.get(fk.ref_table):
                            subject_fk = fk
                            break
                if subject_fk:
                    rule_fields.update(
                        {
                            "subject_table": subject_fk.ref_table,
                            "subject_columns": list(subject_fk.columns),
                            "subject_target_columns": list(subject_fk.ref_columns),
                        }
                    )
                patterns.append(
                    _pattern_record(
                        pattern_id=f"LITERAL_ATTRIBUTE:{source_id}:{len(patterns)}",
                        pattern_type="LITERAL_ATTRIBUTE",
                        source_id=source_id,
                        source_tables=[table_name],
                        source_columns=[column],
                        target_option=option,
                        rule_fields=rule_fields,
                        evidence={"profile": profile, "fk_like": fk_like, "target_id_like": target_id_like},
                        score=2.0 + (1.0 if option.get("selected_match") else 0.0) + min(1.0, float(profile.get("distinct_ratio", 0.0) or 0.0)),
                    )
                )
            if is_generic_identifier_column(table, column):
                patterns.append(
                    {
                        "pattern_id": f"IDENTIFIER_AS_URI:{table_name}.{column}",
                        "pattern_type": "IDENTIFIER_AS_URI",
                        "source_id": source_id,
                        "source_tables": [table_name],
                        "source_columns": [column],
                        "target_uri": None,
                        "target_kind": "uri_key",
                        "rule_fields": {"source_table": table_name, "id_columns": [column]},
                        "evidence": {"profile": profile},
                        "score": 1.0 + float(profile.get("identifier_likeness", 0.0) or 0.0),
                    }
                )

    for table_name, table in sorted(tables.items()):
        for fk in table.foreign_keys:
            source_id = f"source-object:{table_name}.{','.join(fk.columns)}"
            options = [option for option in _target_options(source_id, matches, by_source.get(source_id)) if _kind_allows(option, "object")]
            for option in options:
                if not _is_uri(option.get("target_uri")):
                    continue
                source_classes = class_by_table.get(table_name, [])
                target_classes = class_by_table.get(fk.ref_table, [])
                if _domain_range_score(option, source_classes, target_classes) <= 0:
                    continue
                patterns.append(
                    _pattern_record(
                        pattern_id=f"FK_OBJECT:{source_id}:{len(patterns)}",
                        pattern_type="FK_OBJECT",
                        source_id=source_id,
                        source_tables=[table_name, fk.ref_table],
                        source_columns=list(fk.columns),
                        target_option=option,
                        rule_fields={
                            "source_table": table_name,
                            "source_columns": list(fk.columns),
                            "target_property": option.get("target_uri"),
                            "target_table": fk.ref_table,
                            "target_columns": list(fk.ref_columns),
                            "match_ids": [source_id],
                        },
                        evidence={"join_evidence": "direct_fk", "domain_range_score": _domain_range_score(option, source_classes, target_classes)},
                        score=3.0 + _domain_range_score(option, source_classes, target_classes) + (1.0 if option.get("selected_match") else 0.0),
                    )
                )

        if len(table.foreign_keys) >= 2:
            payload_columns = [
                column.name
                for column in table.columns
                if column.name not in foreign_key_columns(table) and column.name not in table.primary_key
            ]
            pattern_type = "ASSOCIATION_ENTITY" if payload_columns else "JOIN_TABLE_OBJECT"
            for left in table.foreign_keys:
                for right in table.foreign_keys:
                    if left is right:
                        continue
                    source_id = f"source-object:{table_name}.{','.join(left.columns)}"
                    for option in _target_options(source_id, matches, by_source.get(source_id)):
                        if not _kind_allows(option, "object") or not _is_uri(option.get("target_uri")):
                            continue
                        domain_classes = class_by_table.get(left.ref_table, [])
                        range_classes = class_by_table.get(right.ref_table, [])
                        dr_score = _domain_range_score(option, domain_classes, range_classes)
                        if dr_score <= 0:
                            continue
                        patterns.append(
                            _pattern_record(
                                pattern_id=f"{pattern_type}:{source_id}:{right.ref_table}:{len(patterns)}",
                                pattern_type=pattern_type,
                                source_id=source_id,
                                source_tables=[table_name, left.ref_table, right.ref_table],
                                source_columns=list(left.columns) + list(right.columns),
                                target_option=option,
                                rule_fields={
                                    "source_table": table_name,
                                    "source_columns": list(left.columns) + list(right.columns),
                                    "target_property": option.get("target_uri"),
                                    "target_table": right.ref_table,
                                    "target_columns": list(right.ref_columns),
                                    "subject_table": left.ref_table,
                                    "subject_columns": list(left.columns),
                                    "subject_target_columns": list(left.ref_columns),
                                    "object_table": right.ref_table,
                                    "object_columns": list(right.columns),
                                    "object_target_columns": list(right.ref_columns),
                                    "match_ids": [source_id],
                                },
                                evidence={"join_evidence": "join_table", "payload_columns": payload_columns, "domain_range_score": dr_score},
                                score=3.0 + dr_score + (0.5 if not payload_columns else 0.25) + (1.0 if option.get("selected_match") else 0.0),
                            )
                        )
    patterns.sort(key=lambda item: (float(item.get("score", 0.0)), bool(item.get("from_selected_match"))), reverse=True)
    return patterns


def summarize_pattern_candidates(patterns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    for pattern in patterns:
        counts[str(pattern.get("pattern_type"))] += 1
    for pattern_type in sorted(counts):
        rows.append({"pattern_type": pattern_type, "count": counts[pattern_type]})
    return rows


def build_pattern_selection_prompts(patterns: list[dict[str, Any]], budget_chars: int = 24000, variant: int = 1) -> list[dict[str, Any]]:
    ordered = list(patterns)
    if ordered and variant > 1:
        offset = (variant - 1) % len(ordered)
        ordered = ordered[offset:] + ordered[:offset]
    header = (
        "Select schema-grounded FGF mapping patterns from supplied candidates only. "
        "Return JSON only: {\"selected_pattern_ids\":[...],\"rejected\":[{\"pattern_id\":\"...\",\"reason\":\"...\"}]}. "
        "Use only candidate pattern_id values. Prefer structurally supported class, literal, FK, join, and URI-key patterns. "
        "Do not invent identifiers, source fields, joins, or target resources."
    )
    prompts: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pattern in ordered:
        groups[str(pattern.get("pattern_type"))].append(pattern)
    for pattern_type, group in groups.items():
        batch: list[dict[str, Any]] = []
        for pattern in group:
            compact = {
                "pattern_id": pattern.get("pattern_id"),
                "pattern_type": pattern.get("pattern_type"),
                "source_id": pattern.get("source_id"),
                "source_tables": pattern.get("source_tables"),
                "source_columns": pattern.get("source_columns"),
                "target_uri": pattern.get("target_uri"),
                "target_kind": pattern.get("target_kind"),
                "target_local_name": pattern.get("target_local_name"),
                "from_selected_match": pattern.get("from_selected_match"),
                "score": pattern.get("score"),
                "evidence": pattern.get("evidence"),
            }
            trial_batch = batch + [compact]
            payload = {"pattern_type": pattern_type, "candidates": trial_batch}
            prompt = f"{header}\n\n{json.dumps(payload, ensure_ascii=False)}"
            if len(prompt) > budget_chars and batch:
                prompts.append({"pattern_type": pattern_type, "prompt": f"{header}\n\n{json.dumps({'pattern_type': pattern_type, 'candidates': batch}, ensure_ascii=False)}", "char_count": len(f"{header}\n\n{json.dumps({'pattern_type': pattern_type, 'candidates': batch}, ensure_ascii=False)}")})
                batch = [compact]
            else:
                batch = trial_batch
        if batch:
            prompt = f"{header}\n\n{json.dumps({'pattern_type': pattern_type, 'candidates': batch}, ensure_ascii=False)}"
            prompts.append({"pattern_type": pattern_type, "prompt": prompt, "char_count": len(prompt)})
    return prompts


def validate_prompt_no_leakage(prompt: str) -> bool:
    lowered = prompt.lower()
    return not any(term in lowered for term in FORBIDDEN_PROMPT_TERMS)


def select_patterns_from_llm(patterns: list[dict[str, Any]], responses: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {str(pattern.get("pattern_id")): pattern for pattern in patterns}
    selected_ids: list[str] = []
    rejected: list[dict[str, Any]] = []
    invalid_ids: list[str] = []
    for response in responses:
        ids = response.get("selected_pattern_ids", []) if isinstance(response, dict) else []
        for pattern_id in ids:
            pid = str(pattern_id)
            if pid in by_id and pid not in selected_ids:
                selected_ids.append(pid)
            elif pid not in by_id:
                invalid_ids.append(pid)
        rejected.extend(response.get("rejected", []) if isinstance(response, dict) else [])
    selected = [by_id[pattern_id] for pattern_id in selected_ids]
    return selected, {"selected_pattern_ids": selected_ids, "invalid_pattern_ids": invalid_ids, "rejected": rejected}


def infer_uri_keys(selected_patterns: list[dict[str, Any]], tables: dict[str, Table], data: SqlData) -> dict[str, Any]:
    table_patterns: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pattern in selected_patterns:
        for table in pattern.get("source_tables", []) or []:
            if table in tables:
                table_patterns[table].append(pattern)
    templates: dict[str, dict[str, Any]] = {}
    consistency_issues: list[dict[str, Any]] = []
    for table_name, table in sorted(tables.items()):
        if table_name not in table_patterns:
            continue
        candidates: list[tuple[float, list[str], str]] = []
        if table.primary_key:
            candidates.append((3.0, list(table.primary_key), "primary_key"))
        for column in table.column_names():
            profile = value_profile(table, data, column)
            if profile["null_rate"] == 0 and profile["distinct_ratio"] >= 0.98:
                candidates.append((2.0 + float(profile["identifier_likeness"]), [column], "unique_non_null_profile"))
        if not candidates:
            candidates.append((0.5, table_key(table), "fallback_table_key"))
        _, id_columns, reason = max(candidates, key=lambda item: item[0])
        parent_fk = next((fk for fk in table.foreign_keys if fk.columns == table.primary_key and table.primary_key), None)
        if parent_fk:
            reason = "subclass_pkfk_parent_identity"
        templates[table_name] = {
            "table": table_name,
            "id_columns": id_columns,
            "reason": reason,
            "uri_template": f"urn:coding-fgf:source:{{scenario}}/{table_name}/" + "_".join([f"{{{col}}}" for col in id_columns]),
            "pattern_ids": [str(pattern.get("pattern_id")) for pattern in table_patterns[table_name]],
        }
        for pattern in table_patterns[table_name]:
            fields = pattern.get("rule_fields", {}) or {}
            if fields.get("id_columns") and list(fields.get("id_columns") or []) != id_columns:
                consistency_issues.append(
                    {
                        "pattern_id": pattern.get("pattern_id"),
                        "table": table_name,
                        "existing_id_columns": fields.get("id_columns"),
                        "canonical_id_columns": id_columns,
                    }
                )
    return {"version": "uri_key_inference_v1", "templates": templates, "consistency_issues": consistency_issues}


def _rule_key(kind: str, rule: dict[str, Any]) -> str:
    return f"{kind}:{json.dumps(rule, sort_keys=True, default=str)}"


def fol_target_coverage(fol: dict[str, Any]) -> dict[str, Any]:
    rules = fol.get("rules", {}) or {}
    by_kind: dict[str, set[str]] = {"class": set(), "data": set(), "object": set()}
    for rule in rules.get("class", []) or []:
        if rule.get("target_class"):
            by_kind["class"].add(str(rule["target_class"]))
    for rule in rules.get("data", []) or []:
        if rule.get("target_property"):
            by_kind["data"].add(str(rule["target_property"]))
    for rule in rules.get("object", []) or []:
        if rule.get("target_property"):
            by_kind["object"].add(str(rule["target_property"]))
    all_targets = set().union(*by_kind.values())
    return {
        "class": sorted(by_kind["class"]),
        "data": sorted(by_kind["data"]),
        "object": sorted(by_kind["object"]),
        "all": sorted(all_targets),
        "counts": {kind: len(values) for kind, values in by_kind.items()} | {"all": len(all_targets)},
    }


def fol_rule_diff(stage_c_fol: dict[str, Any], stage_f_fol: dict[str, Any]) -> dict[str, Any]:
    diff: dict[str, Any] = {"dropped": {}, "added": {}, "counts": {}}
    for kind in ("class", "data", "object"):
        stage_c = {_rule_key(kind, rule): rule for rule in (stage_c_fol.get("rules", {}) or {}).get(kind, []) or []}
        stage_f = {_rule_key(kind, rule): rule for rule in (stage_f_fol.get("rules", {}) or {}).get(kind, []) or []}
        dropped = [stage_c[key] for key in sorted(set(stage_c) - set(stage_f))]
        added = [stage_f[key] for key in sorted(set(stage_f) - set(stage_c))]
        diff["dropped"][kind] = dropped
        diff["added"][kind] = added
        diff["counts"][kind] = {
            "stage_c": len(stage_c),
            "stage_f": len(stage_f),
            "dropped": len(dropped),
            "added": len(added),
        }
    return diff


def _source_refs(fol: dict[str, Any]) -> dict[str, list[str]]:
    refs: dict[str, set[str]] = {"tables": set(), "columns": set()}
    for kind, rules in (fol.get("rules", {}) or {}).items():
        for rule in rules or []:
            for key in ("source_table", "target_table", "subject_table", "object_table"):
                if rule.get(key):
                    refs["tables"].add(str(rule[key]))
            source_table = str(rule.get("source_table") or "")
            for key in ("source_column",):
                if source_table and rule.get(key):
                    refs["columns"].add(f"{source_table}.{rule[key]}")
            for key in ("source_columns", "target_columns", "id_columns", "subject_columns", "object_columns"):
                for column in rule.get(key) or []:
                    if source_table:
                        refs["columns"].add(f"{source_table}.{column}")
    return {key: sorted(value) for key, value in refs.items()}


def _uri_template_map(fol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for rule in (fol.get("rules", {}) or {}).get("class", []) or []:
        table = str(rule.get("source_table") or "")
        target = str(rule.get("target_class") or "")
        if not table or not target:
            continue
        mapping[f"{table}->{target}"] = {
            "source_table": table,
            "target_class": target,
            "id_columns": list(rule.get("id_columns") or []),
            "uri_template": rule.get("uri_template"),
        }
    return mapping


def stagec_vs_stagef_audit(
    stage_c_fol: dict[str, Any],
    stage_f_fol: dict[str, Any],
    *,
    compiler_report: dict[str, Any] | None = None,
    uri_key_report: dict[str, Any] | None = None,
    mode: str = "conservative",
) -> dict[str, Any]:
    stage_c_coverage = fol_target_coverage(stage_c_fol)
    stage_f_coverage = fol_target_coverage(stage_f_fol)
    target_diff: dict[str, Any] = {}
    for kind in ("class", "data", "object", "all"):
        c_targets = set(stage_c_coverage.get(kind, []) or [])
        f_targets = set(stage_f_coverage.get(kind, []) or [])
        missing = sorted(c_targets - f_targets)
        added = sorted(f_targets - c_targets)
        target_diff[kind] = {
            "stage_c_count": len(c_targets),
            "stage_f_count": len(f_targets),
            "missing_from_stage_f": missing,
            "added_by_stage_f": added,
            "coverage_drop_ratio": len(missing) / max(1, len(c_targets)),
        }
    rule_diff = fol_rule_diff(stage_c_fol, stage_f_fol)
    c_templates = _uri_template_map(stage_c_fol)
    f_templates = _uri_template_map(stage_f_fol)
    template_changes: list[dict[str, Any]] = []
    for key, c_value in sorted(c_templates.items()):
        f_value = f_templates.get(key)
        if not f_value:
            template_changes.append({"mapping": key, "reason": "missing_in_stage_f", "stage_c": c_value, "stage_f": None})
        elif c_value.get("id_columns") and f_value.get("id_columns") and c_value.get("id_columns") != f_value.get("id_columns"):
            template_changes.append({"mapping": key, "reason": "id_columns_changed", "stage_c": c_value, "stage_f": f_value})
    source_c = _source_refs(stage_c_fol)
    source_f = _source_refs(stage_f_fol)
    source_diff = {
        "tables_missing_from_stage_f": sorted(set(source_c["tables"]) - set(source_f["tables"])),
        "columns_missing_from_stage_f": sorted(set(source_c["columns"]) - set(source_f["columns"])),
        "tables_added_by_stage_f": sorted(set(source_f["tables"]) - set(source_c["tables"])),
        "columns_added_by_stage_f": sorted(set(source_f["columns"]) - set(source_c["columns"])),
    }
    compiler = compiler_report or {}
    rejected = compiler.get("rejected", []) or []
    return {
        "version": "pattern_preservation_audit_v1",
        "mode": mode,
        "replacement_kind": "additive_overlay" if mode in {"overlay", "conservative"} else "replacement",
        "target_coverage_diff": target_diff,
        "fol_rule_diff": rule_diff,
        "uri_template_diff": {"changes": template_changes, "stage_c": c_templates, "stage_f": f_templates},
        "source_usage_diff": source_diff,
        "compiler_rejections": rejected,
        "compiler_rejection_counts": dict(Counter(str(item.get("reason", "")) for item in rejected)),
        "uri_key_consistency_issues": (uri_key_report or {}).get("consistency_issues", []),
    }


def _safe_pattern_rule(kind: str, rule: dict[str, Any], tables: dict[str, Table]) -> tuple[bool, str]:
    source_table = str(rule.get("source_table") or "")
    if source_table not in tables:
        return False, "unknown_source_table"
    table = tables[source_table]
    if kind == "class":
        for column in rule.get("id_columns") or table_key(table):
            if column not in table.column_names():
                return False, "unknown_class_id_column"
    elif kind == "data":
        column = str(rule.get("source_column") or "")
        if column not in table.column_names():
            return False, "unknown_data_source_column"
        if column in foreign_key_columns(table) and not any(token in _local(str(rule.get("target_property", ""))).lower() for token in ("id", "identifier", "doi", "url", "email", "isbn", "issn", "code")):
            return False, "fk_like_literal_without_identifier_target"
    elif kind == "object":
        target_table = str(rule.get("target_table") or "")
        if target_table not in tables:
            return False, "unknown_object_target_table"
        for column in rule.get("source_columns") or []:
            if column not in table.column_names():
                return False, "unknown_object_source_column"
        for column in rule.get("target_columns") or table_key(tables[target_table]):
            if column not in tables[target_table].column_names():
                return False, "unknown_object_target_column"
    return True, "ok"


def apply_pattern_preservation(
    stage_c_fol: dict[str, Any],
    pattern_fol: dict[str, Any],
    tables: dict[str, Table],
    *,
    mode: str = "conservative",
    compiler_report: dict[str, Any] | None = None,
    uri_key_report: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if mode not in {"replace", "overlay", "conservative"}:
        raise ValueError(f"Unsupported pattern preservation mode: {mode}")
    audit = stagec_vs_stagef_audit(
        stage_c_fol,
        pattern_fol,
        compiler_report=compiler_report,
        uri_key_report=uri_key_report,
        mode=mode,
    )
    coverage = audit["target_coverage_diff"]
    rejection_reasons: list[str] = []
    if mode == "replace":
        for kind, limit in (("all", 0.02), ("class", 0.02), ("data", 0.02), ("object", 0.02)):
            if float(coverage[kind]["coverage_drop_ratio"]) > limit:
                rejection_reasons.append(f"{kind}_target_coverage_drop_gt_{limit}")
        if audit["uri_template_diff"]["changes"]:
            rejection_reasons.append("uri_template_changed_or_missing")
        decision = {
            "mode": mode,
            "fallback_used": bool(rejection_reasons),
            "rejection_reasons": rejection_reasons,
            "selected_fol": "stage_c" if rejection_reasons else "pattern_replacement",
        }
        return (stage_c_fol if rejection_reasons else pattern_fol), {"audit": audit, "decision": decision}

    merged = {
        "rules": {
            "class": [dict(rule) for rule in (stage_c_fol.get("rules", {}) or {}).get("class", []) or []],
            "data": [dict(rule) for rule in (stage_c_fol.get("rules", {}) or {}).get("data", []) or []],
            "object": [dict(rule) for rule in (stage_c_fol.get("rules", {}) or {}).get("object", []) or []],
        },
        "generation": {
            **(stage_c_fol.get("generation", {}) or {}),
            "pattern_preservation_mode": mode,
            "pattern_first_overlay": True,
        },
    }
    seen = {_rule_key(kind, rule) for kind in ("class", "data", "object") for rule in merged["rules"][kind]}
    added: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for kind in ("class", "data", "object"):
        for rule in (pattern_fol.get("rules", {}) or {}).get(kind, []) or []:
            key = _rule_key(kind, rule)
            if key in seen:
                continue
            ok, reason = _safe_pattern_rule(kind, rule, tables)
            if not ok:
                rejected.append({"kind": kind, "reason": reason, "rule": rule})
                continue
            merged["rules"][kind].append(dict(rule))
            seen.add(key)
            added.append({"kind": kind, "target": rule.get("target_class") or rule.get("target_property"), "pattern_id": rule.get("pattern_id")})
    decision = {
        "mode": mode,
        "fallback_used": False,
        "selected_fol": "stage_c_plus_safe_pattern_overlay",
        "rules_added": len(added),
        "rules_rejected": len(rejected),
        "added": added,
        "rejected": rejected,
        "rejection_reasons": [],
    }
    return merged, {"audit": audit, "decision": decision}


def compile_patterns_to_fol(
    selected_patterns: list[dict[str, Any]],
    tables: dict[str, Table],
    uri_keys: dict[str, Any],
    *,
    safe_domain_range_type_completion: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rules: dict[str, list[dict[str, Any]]] = {"class": [], "data": [], "object": []}
    report = {
        "compiled": [],
        "rejected": [],
        "type_completion": [],
        "rule_counts": {},
    }
    allowed_targets = {str(pattern.get("target_uri")) for pattern in selected_patterns if pattern.get("target_uri")}
    class_targets_by_table: dict[str, list[str]] = defaultdict(list)
    seen_rules: set[str] = set()
    templates = uri_keys.get("templates", {}) if isinstance(uri_keys, dict) else {}

    def add_rule(kind: str, rule: dict[str, Any], pattern: dict[str, Any]) -> None:
        key = _rule_key(kind, rule)
        if key in seen_rules:
            return
        seen_rules.add(key)
        rules[kind].append(rule)
        report["compiled"].append({"pattern_id": pattern.get("pattern_id"), "pattern_type": pattern.get("pattern_type"), "kind": kind})

    for pattern in selected_patterns:
        pattern_type = str(pattern.get("pattern_type"))
        fields = dict(pattern.get("rule_fields") or {})
        target_uri = pattern.get("target_uri")
        if pattern_type == "IDENTIFIER_AS_URI":
            continue
        if not target_uri or str(target_uri) not in allowed_targets:
            report["rejected"].append({"pattern_id": pattern.get("pattern_id"), "reason": "target_not_selected"})
            continue
        source_table = str(fields.get("source_table") or "")
        if source_table not in tables:
            report["rejected"].append({"pattern_id": pattern.get("pattern_id"), "reason": "unknown_source_table", "source_table": source_table})
            continue
        if pattern_type in {"SCHEMA_ENTITY", "SUBCLASS_PKFK", "WEAK_ENTITY"}:
            id_columns = list((templates.get(source_table) or {}).get("id_columns") or fields.get("id_columns") or table_key(tables[source_table]))
            if any(column not in tables[source_table].column_names() for column in id_columns):
                report["rejected"].append({"pattern_id": pattern.get("pattern_id"), "reason": "unknown_id_column", "id_columns": id_columns})
                continue
            rule = {
                "source_table": source_table,
                "target_class": str(target_uri),
                "id_columns": id_columns,
                "confidence": float(pattern.get("score", 0.0) or 0.0),
                "table_role": table_role(tables[source_table]),
                "match_ids": list(fields.get("match_ids") or [pattern.get("source_id")]),
                "pattern_id": pattern.get("pattern_id"),
                "uri_template": (templates.get(source_table) or {}).get("uri_template"),
            }
            add_rule("class", rule, pattern)
            class_targets_by_table[source_table].append(str(target_uri))
        elif pattern_type == "LITERAL_ATTRIBUTE":
            column = str(fields.get("source_column") or "")
            if column not in tables[source_table].column_names():
                report["rejected"].append({"pattern_id": pattern.get("pattern_id"), "reason": "unknown_source_column", "source_column": column})
                continue
            if column in foreign_key_columns(tables[source_table]) and not (pattern.get("evidence", {}) or {}).get("target_id_like"):
                report["rejected"].append({"pattern_id": pattern.get("pattern_id"), "reason": "fk_like_literal_rejected", "source_column": column})
                continue
            rule = {
                "source_table": source_table,
                "source_column": column,
                "target_property": str(target_uri),
                "confidence": float(pattern.get("score", 0.0) or 0.0),
                "match_ids": list(fields.get("match_ids") or [pattern.get("source_id")]),
                "pattern_id": pattern.get("pattern_id"),
            }
            for optional in ("subject_table", "subject_columns", "subject_target_columns"):
                if fields.get(optional):
                    rule[optional] = fields[optional]
            add_rule("data", rule, pattern)
        elif pattern_type in {"FK_OBJECT", "JOIN_TABLE_OBJECT", "ASSOCIATION_ENTITY"}:
            target_table = str(fields.get("target_table") or "")
            if target_table not in tables:
                report["rejected"].append({"pattern_id": pattern.get("pattern_id"), "reason": "unknown_target_table", "target_table": target_table})
                continue
            source_columns = list(fields.get("source_columns") or [])
            if any(column not in tables[source_table].column_names() for column in source_columns):
                report["rejected"].append({"pattern_id": pattern.get("pattern_id"), "reason": "unknown_source_columns", "source_columns": source_columns})
                continue
            rule = {
                "source_table": source_table,
                "source_columns": source_columns,
                "target_property": str(target_uri),
                "target_table": target_table,
                "target_columns": list(fields.get("target_columns") or table_key(tables[target_table])),
                "confidence": float(pattern.get("score", 0.0) or 0.0),
                "match_ids": list(fields.get("match_ids") or [pattern.get("source_id")]),
                "pattern_id": pattern.get("pattern_id"),
            }
            for optional in ("subject_table", "subject_columns", "subject_target_columns", "object_table", "object_columns", "object_target_columns"):
                if fields.get(optional):
                    rule[optional] = fields[optional]
            add_rule("object", rule, pattern)
            if safe_domain_range_type_completion:
                for role_table in [rule.get("subject_table") or source_table, rule.get("object_table") or target_table]:
                    if role_table in templates and class_targets_by_table.get(str(role_table)):
                        for class_uri in class_targets_by_table[str(role_table)]:
                            type_rule = {
                                "source_table": str(role_table),
                                "target_class": class_uri,
                                "id_columns": list((templates.get(str(role_table)) or {}).get("id_columns") or table_key(tables[str(role_table)])),
                                "confidence": 0.5,
                                "table_role": table_role(tables[str(role_table)]),
                                "match_ids": [f"source-class:{role_table}"],
                                "inferred_from": "safe_domain_range_type_completion",
                            }
                            add_rule("class", type_rule, pattern)
                            report["type_completion"].append({"table": role_table, "target_class": class_uri, "pattern_id": pattern.get("pattern_id")})
        else:
            report["rejected"].append({"pattern_id": pattern.get("pattern_id"), "reason": "unsupported_pattern_type", "pattern_type": pattern_type})

    report["rule_counts"] = {kind: len(values) for kind, values in rules.items()}
    fol = {
        "rules": rules,
        "generation": {
            "source": "pattern_first_deterministic_fol_compiler",
            "prompt_version": "fgf_pattern_first_v1",
            "pattern_count": len(selected_patterns),
            "safe_domain_range_type_completion": safe_domain_range_type_completion,
        },
    }
    return fol, report


def pattern_selection_internal_score(report: dict[str, Any], materialization_report: dict[str, Any] | None = None) -> tuple[int, int, int, int, int, int]:
    compiler = report.get("compiler_report", {}) or {}
    rejected = len(compiler.get("rejected", []) or [])
    rule_counts = compiler.get("rule_counts", {}) or {}
    class_rules = int(rule_counts.get("class", 0) or 0)
    data_rules = int(rule_counts.get("data", 0) or 0)
    object_rules = int(rule_counts.get("object", 0) or 0)
    uri_issues = len(((report.get("uri_key_inference") or {}).get("consistency_issues") or []))
    if materialization_report:
        mat_score = internal_materialization_score(materialization_report)
        mat_summary = materialization_coverage_summary(materialization_report)
        invalid = int(mat_summary.get("invalid_rdf_count", 0) or 0)
        zero = int(mat_summary.get("zero_emission_selected_targets", 0) or 0)
    else:
        mat_score = (0,)
        invalid = 0
        zero = 0
    return (
        -rejected,
        -uri_issues,
        -invalid,
        -zero,
        class_rules + data_rules + object_rules,
        sum(mat_score) if all(isinstance(value, (int, float)) for value in mat_score) else 0,
    )
