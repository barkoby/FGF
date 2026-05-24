from __future__ import annotations

import json
from typing import Any

from .lexical import words
from .schema import ForeignKey, SqlData, Table, table_key, table_role


OBJECT_EVIDENCE_VERSION = "object_link_evidence_v1"


def _source_object_parts(source_id: str) -> tuple[str, list[str]]:
    if not source_id.startswith("source-object:") or "." not in source_id:
        return "", []
    tail = source_id.split(":", 1)[1]
    table, columns = tail.split(".", 1)
    return table, [part for part in columns.split(",") if part]


def _local(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[1]
    return uri.rstrip("/").rsplit("/", 1)[-1]


def _class_words(class_uri: str) -> set[str]:
    return words(_local(class_uri))


def _property_side_score(property_uris: list[str], class_uris: list[str]) -> int:
    if not property_uris or not class_uris:
        return 1
    scores: list[int] = []
    for prop_uri in property_uris:
        prop_words = _class_words(prop_uri)
        for class_uri in class_uris:
            class_words = _class_words(class_uri)
            scores.append(2 if prop_words == class_words else 1 if prop_words & class_words else 0)
    return max(scores, default=0)


def _table_classes(matches: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for match in matches:
        source_id = str(match.get("source_id", ""))
        target_uri = match.get("target_uri")
        if not source_id.startswith("source-class:") or not target_uri:
            continue
        table = source_id.split(":", 1)[1]
        out.setdefault(table, [])
        uri = str(target_uri)
        if uri not in out[table]:
            out[table].append(uri)
    return out


def _row_index(rows: list[dict[str, Any]], columns: list[str]) -> set[tuple[Any, ...]]:
    return {tuple(row.get(column) for column in columns) for row in rows}


def _nonempty_key(row: dict[str, Any], columns: list[str]) -> tuple[Any, ...] | None:
    key = tuple(row.get(column) for column in columns)
    if not columns or any(value is None or value == "" for value in key):
        return None
    return key


def _direct_reachability(plan: dict[str, Any], data: SqlData | None) -> dict[str, int]:
    if data is None:
        return {}
    source_rows = data.rows.get(str(plan["source_table"]), [])
    target_rows = data.rows.get(str(plan["target_table"]), [])
    target_columns = list(plan.get("target_columns") or [])
    target_keys = _row_index(target_rows, target_columns)
    nonempty = 0
    hits = 0
    for row in source_rows:
        key = _nonempty_key(row, list(plan.get("source_columns") or []))
        if key is None:
            continue
        nonempty += 1
        if key in target_keys:
            hits += 1
    return {
        "source_rows": len(source_rows),
        "nonempty_source_keys": nonempty,
        "target_key_hits": hits,
        "estimated_emissions": hits,
    }


def _association_reachability(plan: dict[str, Any], data: SqlData | None) -> dict[str, int]:
    if data is None:
        return {}
    source_rows = data.rows.get(str(plan["source_table"]), [])
    subject_rows = data.rows.get(str(plan["subject_table"]), [])
    object_rows = data.rows.get(str(plan["object_table"]), [])
    subject_target_columns = list(plan.get("subject_target_columns") or [])
    object_target_columns = list(plan.get("object_target_columns") or [])
    subject_keys = _row_index(subject_rows, subject_target_columns)
    object_keys = _row_index(object_rows, object_target_columns)
    nonempty_subject = 0
    nonempty_object = 0
    subject_hits = 0
    object_hits = 0
    both_hits = 0
    for row in source_rows:
        subject_key = _nonempty_key(row, list(plan.get("subject_columns") or []))
        object_key = _nonempty_key(row, list(plan.get("object_columns") or []))
        if subject_key is not None:
            nonempty_subject += 1
        if object_key is not None:
            nonempty_object += 1
        subject_ok = subject_key in subject_keys if subject_key is not None else False
        object_ok = object_key in object_keys if object_key is not None else False
        subject_hits += 1 if subject_ok else 0
        object_hits += 1 if object_ok else 0
        both_hits += 1 if subject_ok and object_ok else 0
    return {
        "source_rows": len(source_rows),
        "nonempty_subject_keys": nonempty_subject,
        "nonempty_object_keys": nonempty_object,
        "subject_key_hits": subject_hits,
        "object_key_hits": object_hits,
        "estimated_emissions": both_hits,
    }


def _domain_range_compatible(match: dict[str, Any], subject_classes: list[str], object_classes: list[str]) -> bool:
    domain = [str(value) for value in match.get("target_domain") or []]
    range_ = [str(value) for value in match.get("target_range") or []]
    return _property_side_score(domain, subject_classes) > 0 and _property_side_score(range_, object_classes) > 0


def _plan_id(source_id: str, orientation: str, fields: dict[str, Any]) -> str:
    stable = {
        key: fields.get(key)
        for key in (
            "source_table",
            "source_columns",
            "target_table",
            "target_columns",
            "subject_table",
            "subject_columns",
            "object_table",
            "object_columns",
        )
        if fields.get(key)
    }
    return f"{source_id}|{orientation}|{json.dumps(stable, sort_keys=True, separators=(',', ':'))}"


def _direct_plan(
    source_id: str,
    match: dict[str, Any],
    tables: dict[str, Table],
    data: SqlData | None,
    class_by_table: dict[str, list[str]],
    source_table: str,
    fk: ForeignKey,
) -> dict[str, Any] | None:
    subject_classes = class_by_table.get(source_table, [])
    object_classes = class_by_table.get(fk.ref_table, [])
    if not _domain_range_compatible(match, subject_classes, object_classes):
        return None
    fields = {
        "source_table": source_table,
        "source_columns": list(fk.columns),
        "target_property": str(match.get("target_uri", "")),
        "target_table": fk.ref_table,
        "target_columns": list(fk.ref_columns),
        "match_ids": [source_id],
    }
    return {
        "plan_id": _plan_id(source_id, "direct", fields),
        "orientation": "direct_fk",
        "confidence_label": "direct_fk",
        "rule_fields": fields,
        "subject_table": source_table,
        "object_table": fk.ref_table,
        "subject_classes": subject_classes,
        "object_classes": object_classes,
        "reachability": _direct_reachability(fields, data),
    }


def _inverse_plan(
    source_id: str,
    match: dict[str, Any],
    tables: dict[str, Table],
    data: SqlData | None,
    class_by_table: dict[str, list[str]],
    source_table: str,
    fk: ForeignKey,
) -> dict[str, Any] | None:
    subject_classes = class_by_table.get(fk.ref_table, [])
    object_classes = class_by_table.get(source_table, [])
    if not _domain_range_compatible(match, subject_classes, object_classes):
        return None
    object_key = table_key(tables[source_table])
    fields = {
        "source_table": source_table,
        "source_columns": list(fk.columns),
        "target_property": str(match.get("target_uri", "")),
        "target_table": source_table,
        "target_columns": list(object_key),
        "subject_table": fk.ref_table,
        "subject_columns": list(fk.columns),
        "subject_target_columns": list(fk.ref_columns),
        "object_table": source_table,
        "object_columns": list(object_key),
        "object_target_columns": list(object_key),
        "match_ids": [source_id],
    }
    return {
        "plan_id": _plan_id(source_id, "inverse", fields),
        "orientation": "inverse_fk",
        "confidence_label": "reverse_fk",
        "rule_fields": fields,
        "subject_table": fk.ref_table,
        "object_table": source_table,
        "subject_classes": subject_classes,
        "object_classes": object_classes,
        "reachability": _association_reachability(fields, data),
    }


def _join_plan(
    source_id: str,
    match: dict[str, Any],
    data: SqlData | None,
    class_by_table: dict[str, list[str]],
    source_table: str,
    subject_fk: ForeignKey,
    object_fk: ForeignKey,
    orientation: str,
) -> dict[str, Any] | None:
    subject_classes = class_by_table.get(subject_fk.ref_table, [])
    object_classes = class_by_table.get(object_fk.ref_table, [])
    if not _domain_range_compatible(match, subject_classes, object_classes):
        return None
    fields = {
        "source_table": source_table,
        "source_columns": list(subject_fk.columns) + list(object_fk.columns),
        "target_property": str(match.get("target_uri", "")),
        "target_table": object_fk.ref_table,
        "target_columns": list(object_fk.ref_columns),
        "subject_table": subject_fk.ref_table,
        "subject_columns": list(subject_fk.columns),
        "subject_target_columns": list(subject_fk.ref_columns),
        "object_table": object_fk.ref_table,
        "object_columns": list(object_fk.columns),
        "object_target_columns": list(object_fk.ref_columns),
        "match_ids": [source_id],
    }
    return {
        "plan_id": _plan_id(source_id, orientation, fields),
        "orientation": orientation,
        "confidence_label": "join_table",
        "rule_fields": fields,
        "subject_table": subject_fk.ref_table,
        "object_table": object_fk.ref_table,
        "subject_classes": subject_classes,
        "object_classes": object_classes,
        "reachability": _association_reachability(fields, data),
    }


def build_object_link_evidence(
    matches: list[dict[str, Any]],
    tables: dict[str, Table],
    data: SqlData | None = None,
    allow_weak_object_links: bool = False,
) -> dict[str, Any]:
    class_by_table = _table_classes(matches)
    entries: list[dict[str, Any]] = []
    for match in matches:
        source_id = str(match.get("source_id", ""))
        if not match.get("target_uri") or not source_id.startswith("source-object:"):
            continue
        source_table, source_columns = _source_object_parts(source_id)
        table = tables.get(source_table)
        if not table:
            continue
        matched_fks = [fk for fk in table.foreign_keys if list(fk.columns) == source_columns]
        plans: list[dict[str, Any]] = []
        for fk in matched_fks:
            direct = _direct_plan(source_id, match, tables, data, class_by_table, source_table, fk)
            inverse = _inverse_plan(source_id, match, tables, data, class_by_table, source_table, fk)
            if direct:
                plans.append(direct)
            if inverse:
                plans.append(inverse)
            if table_role(table) == "join_table" or len(table.foreign_keys) >= 2:
                for other_fk in table.foreign_keys:
                    if list(other_fk.columns) == list(fk.columns):
                        continue
                    subject_first = _join_plan(source_id, match, data, class_by_table, source_table, fk, other_fk, "join_subject_matched_fk")
                    object_first = _join_plan(source_id, match, data, class_by_table, source_table, other_fk, fk, "join_object_matched_fk")
                    if subject_first:
                        plans.append(subject_first)
                    if object_first:
                        plans.append(object_first)
        entries.append(
            {
                "source_id": source_id,
                "source_table": source_table,
                "source_columns": source_columns,
                "target_uri": str(match.get("target_uri")),
                "target_domain": list(match.get("target_domain") or []),
                "target_range": list(match.get("target_range") or []),
                "target_local_name": str(match.get("target_local_name", "")),
                "source_table_classes": class_by_table.get(source_table, []),
                "legal_plans": plans,
                "weak_name_only_allowed": bool(allow_weak_object_links),
            }
        )
    return {"version": OBJECT_EVIDENCE_VERSION, "entries": entries, "allow_weak_object_links": bool(allow_weak_object_links)}


def legal_plan_index(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for entry in evidence.get("entries", []) or []:
        for plan in entry.get("legal_plans", []) or []:
            plan_id = str(plan.get("plan_id", ""))
            if plan_id:
                out[plan_id] = plan
    return out


def filter_object_evidence_for_matches(evidence: dict[str, Any] | None, matches: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not evidence:
        return None
    wanted = {str(match.get("source_id", "")) for match in matches}
    return {
        "version": evidence.get("version", OBJECT_EVIDENCE_VERSION),
        "entries": [entry for entry in evidence.get("entries", []) or [] if str(entry.get("source_id", "")) in wanted],
    }


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def validate_object_rules_against_evidence(fol: dict[str, Any], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    plans = legal_plan_index(evidence)
    issues: list[dict[str, Any]] = []
    for index, rule in enumerate(fol.get("rules", {}).get("object", []) or []):
        rule_id = f"object:{index}"
        plan_id = str(rule.get("plan_id", ""))
        if not plan_id:
            issues.append({"rule_id": rule_id, "issue": "missing_object_plan_id"})
            continue
        plan = plans.get(plan_id)
        if not plan:
            issues.append({"rule_id": rule_id, "issue": "unknown_object_plan_id", "plan_id": plan_id})
            continue
        expected = plan.get("rule_fields", {}) or {}
        for field, expected_value in expected.items():
            actual = rule.get(field)
            if isinstance(expected_value, list):
                actual_value = _as_list(actual)
            else:
                actual_value = actual
            if actual_value != expected_value:
                issues.append(
                    {
                        "rule_id": rule_id,
                        "issue": "object_rule_does_not_match_plan",
                        "plan_id": plan_id,
                        "field": field,
                        "expected": expected_value,
                        "actual": actual_value,
                    }
                )
        reachability = plan.get("reachability") or {}
        if int(reachability.get("source_rows", 0) or 0) > 0 and int(reachability.get("estimated_emissions", 0) or 0) == 0:
            issues.append(
                {
                    "rule_id": rule_id,
                    "issue": "object_plan_zero_reachability",
                    "plan_id": plan_id,
                    "reachability": reachability,
                }
            )
    return issues


def object_evidence_summary(evidence: dict[str, Any]) -> dict[str, Any]:
    entries = evidence.get("entries", []) or []
    plan_count = sum(len(entry.get("legal_plans", []) or []) for entry in entries)
    by_label: dict[str, int] = {}
    for entry in entries:
        for plan in entry.get("legal_plans", []) or []:
            label = str(plan.get("confidence_label") or plan.get("orientation") or "unknown")
            by_label[label] = by_label.get(label, 0) + 1
    reachable = sum(
        1
        for entry in entries
        for plan in entry.get("legal_plans", []) or []
        if int((plan.get("reachability") or {}).get("estimated_emissions", 0) or 0) > 0
    )
    return {
        "version": evidence.get("version", OBJECT_EVIDENCE_VERSION),
        "object_matches": len(entries),
        "legal_plans": plan_count,
        "reachable_plans": reachable,
        "plans_by_confidence_label": by_label,
        "allow_weak_object_links": bool(evidence.get("allow_weak_object_links", False)),
    }
