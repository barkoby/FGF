from __future__ import annotations

import copy
import json
from collections import Counter
from typing import Any

from .schema import SqlData, Table, foreign_key_columns


def _source_data_parts(source_id: str) -> tuple[str, str] | None:
    if not source_id.startswith("source-data:") or "." not in source_id:
        return None
    tail = source_id.split(":", 1)[1]
    table, column = tail.split(".", 1)
    return table, column


def _data_rule_supports_match(rule: dict[str, Any], match: dict[str, Any]) -> bool:
    source_id = str(match.get("source_id", ""))
    if source_id in {str(value) for value in rule.get("match_ids", []) or []}:
        return True
    parts = _source_data_parts(source_id)
    if not parts:
        return False
    table, column = parts
    return (
        str(rule.get("source_table", "")) == table
        and str(rule.get("source_column", "")) == column
        and str(rule.get("target_property", "")) == str(match.get("target_uri", ""))
    )


def _selected_datatype_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        match
        for match in matches
        if match.get("target_uri") and str(match.get("source_id", "")).startswith("source-data:")
    ]


def _allowed_target_uris(matches: list[dict[str, Any]]) -> set[str]:
    return {str(match.get("target_uri")) for match in _selected_datatype_matches(matches) if match.get("target_uri")}


def _valid_match_ids(rule: dict[str, Any], matches: list[dict[str, Any]]) -> bool:
    selected = {str(match.get("source_id")) for match in matches if match.get("target_uri")}
    match_ids = [str(value) for value in rule.get("match_ids", []) or []]
    return bool(match_ids) and any(match_id in selected for match_id in match_ids)


def _rule_has_selected_data_provenance(rule: dict[str, Any], matches: list[dict[str, Any]]) -> bool:
    return any(_data_rule_supports_match(rule, match) for match in _selected_datatype_matches(matches))


def attribute_coverage_diagnostics(
    matches: list[dict[str, Any]],
    fol: dict[str, Any],
    tables: dict[str, Table],
    data: SqlData | None = None,
) -> dict[str, Any]:
    """Check selected datatype matches against FOL data rules without using evaluation answers."""
    selected_data = _selected_datatype_matches(matches)
    data_rules = list(fol.get("rules", {}).get("data", []) or [])
    issues: list[dict[str, Any]] = []
    covered = 0
    fk_like_data_rules = 0
    allowed_targets = _allowed_target_uris(matches)

    for rule_index, rule in enumerate(data_rules):
        table = tables.get(str(rule.get("source_table", "")))
        column = str(rule.get("source_column", ""))
        rule_id = f"data:{rule_index}"
        if not table:
            issues.append(
                {
                    "rule_id": rule_id,
                    "issue": "data_rule_unknown_source_table",
                    "source_table": rule.get("source_table"),
                    "source_column": column,
                    "target_property": rule.get("target_property"),
                }
            )
            continue
        if column not in table.column_names():
            issues.append(
                {
                    "rule_id": rule_id,
                    "issue": "data_rule_unknown_source_column",
                    "source_table": table.name,
                    "source_column": column,
                    "target_property": rule.get("target_property"),
                }
            )
        target_property = str(rule.get("target_property", ""))
        if target_property and target_property not in allowed_targets:
            issues.append(
                {
                    "rule_id": rule_id,
                    "issue": "data_rule_target_not_selected",
                    "source_table": table.name,
                    "source_column": column,
                    "target_property": target_property,
                }
            )
        if not _valid_match_ids(rule, matches):
            issues.append(
                {
                    "rule_id": rule_id,
                    "issue": "literal_provenance_missing",
                    "source_table": table.name,
                    "source_column": column,
                    "target_property": target_property,
                }
            )
        elif not _rule_has_selected_data_provenance(rule, matches):
            issues.append(
                {
                    "rule_id": rule_id,
                    "issue": "literal_provenance_too_broad",
                    "source_table": table.name,
                    "source_column": column,
                    "target_property": target_property,
                }
            )
        if table and column in foreign_key_columns(table) and not rule.get("subject_table"):
            fk_like_data_rules += 1
            issues.append(
                {
                    "rule_id": rule_id,
                    "issue": "fk_like_data_rule",
                    "source_table": table.name,
                    "source_column": column,
                    "target_property": rule.get("target_property"),
                }
            )

    for match in selected_data:
        source_id = str(match.get("source_id", ""))
        parts = _source_data_parts(source_id)
        if not parts:
            continue
        table_name, column = parts
        table = tables.get(table_name)
        supported = any(_data_rule_supports_match(rule, match) for rule in data_rules)
        if supported:
            covered += 1
        else:
            issues.append(
                {
                    "rule_id": f"attribute:{source_id}",
                    "issue": "missing_data_rule",
                    "source_id": source_id,
                    "source_table": table_name,
                    "source_column": column,
                    "target_uri": match.get("target_uri"),
                    "target_id": match.get("target_id"),
                    "match_ids": [source_id],
                }
            )
        if table and column in foreign_key_columns(table):
            issues.append(
                {
                    "rule_id": f"attribute:{source_id}",
                    "issue": "fk_like_selected_datatype_match",
                    "source_id": source_id,
                    "source_table": table_name,
                    "source_column": column,
                    "target_uri": match.get("target_uri"),
                }
            )

    return {
        "selected_datatype_matches": len(selected_data),
        "datatype_matches_with_rules": covered,
        "datatype_matches_missing_rules": max(0, len(selected_data) - covered),
        "fk_like_data_rules": fk_like_data_rules,
        "issues_by_type": dict(Counter(str(issue.get("issue", "unknown")) for issue in issues)),
        "issues": issues,
    }


def _table_excerpt(table: Table) -> dict[str, Any]:
    return {
        "name": table.name,
        "columns": [{"name": column.name, "datatype": column.datatype, "nullable": column.nullable} for column in table.columns],
        "primary_key": table.primary_key,
        "foreign_keys": [
            {"columns": fk.columns, "ref_table": fk.ref_table, "ref_columns": fk.ref_columns}
            for fk in table.foreign_keys
        ],
    }


def _source_value_summary(data: SqlData | None, table_name: str, column: str, limit: int = 5) -> dict[str, Any]:
    rows = list((data.rows.get(table_name, []) if data else []) or [])
    samples: list[str] = []
    non_empty = 0
    distinct: set[str] = set()
    for row in rows:
        value = row.get(column)
        if value in (None, ""):
            continue
        non_empty += 1
        text = str(value)
        distinct.add(text)
        if len(samples) < limit and text not in samples:
            samples.append(text)
    return {
        "table": table_name,
        "column": column,
        "row_count": len(rows),
        "non_empty_count": non_empty,
        "distinct_sample_count": min(len(distinct), limit),
        "sample_values": samples,
    }


def build_attribute_coverage_repair_prompt(
    report: dict[str, Any],
    fol: dict[str, Any],
    matches: list[dict[str, Any]],
    tables: dict[str, Table],
    data: SqlData | None = None,
    max_issues: int = 12,
) -> str:
    """Build a compact, evaluation-blind repair prompt for datatype-rule coverage."""
    issues = list(report.get("issues", []) or [])[:max_issues]
    referenced_tables = {
        str(issue.get("source_table", ""))
        for issue in issues
        if str(issue.get("source_table", "")) in tables
    }
    referenced_columns = {
        (str(issue.get("source_table", "")), str(issue.get("source_column", "")))
        for issue in issues
        if str(issue.get("source_table", "")) in tables
    }
    payload = {
        "task": "Repair datatype-property FOL rules using only supplied schema, selected matches, current data rules, and source summaries.",
        "allowed_actions": ["add_data_rule", "repair", "drop", "keep_with_justification"],
        "constraints": [
            "Return JSON only with a top-level repairs array.",
            "Use only target URIs that appear in selected_datatype_matches.",
            "Use only source tables and columns from schema_excerpt.",
            "Every added or repaired rule must cite valid match_ids from selected_datatype_matches.",
            "Do not materialize FK-like columns as literals unless the rule is explicitly justified as an identifier literal.",
            "If support is insufficient, choose drop or keep_with_justification.",
            "Do not use evaluation answers, benchmark queries, evaluation-derived result sets, or dataset-specific corrections.",
        ],
        "output_shape": {
            "repairs": [
                {
                    "issue_id": "data:0 or attribute:source-data:Table.column",
                    "action": "add_data_rule|repair|drop|keep_with_justification",
                    "reason": "short schema-grounded explanation",
                    "rule": None,
                }
            ]
        },
        "issues": issues,
        "selected_datatype_matches": [
            {
                "source_id": match.get("source_id"),
                "target_uri": match.get("target_uri"),
                "target_id": match.get("target_id"),
                "target_local_name": match.get("target_local_name"),
            }
            for match in _selected_datatype_matches(matches)
        ],
        "current_data_rules": list(fol.get("rules", {}).get("data", []) or []),
        "schema_excerpt": [_table_excerpt(tables[name]) for name in sorted(referenced_tables)],
        "source_value_summaries": [
            _source_value_summary(data, table_name, column)
            for table_name, column in sorted(referenced_columns)
            if column in tables[table_name].column_names()
        ],
    }
    return (
        "You are repairing datatype-property FOL mapping rules for a semantic data integration pipeline.\n"
        "Repair only the listed attribute coverage diagnostics. Ground every change in the supplied selected "
        "matches, schema excerpt, current rules, and source value summaries.\n"
        "Do not invent source tables, source columns, target URIs, match ids, constants, or dataset-specific fixes.\n"
        "Return JSON only.\n\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def _rule_is_grounded_data_rule(rule: dict[str, Any], matches: list[dict[str, Any]], tables: dict[str, Table]) -> bool:
    table_name = str(rule.get("source_table", ""))
    column = str(rule.get("source_column", ""))
    table = tables.get(table_name)
    if not table or column not in table.column_names():
        return False
    target = str(rule.get("target_property", ""))
    if target not in _allowed_target_uris(matches):
        return False
    match_ids = {str(value) for value in rule.get("match_ids", []) or []}
    if not match_ids:
        return False
    for match in _selected_datatype_matches(matches):
        source_id = str(match.get("source_id", ""))
        parts = _source_data_parts(source_id)
        if not parts:
            continue
        if (
            source_id in match_ids
            and parts == (table_name, column)
            and str(match.get("target_uri", "")) == target
        ):
            return True
    return False


def _parse_data_rule_id(rule_id: str) -> int | None:
    if not rule_id.startswith("data:"):
        return None
    raw = rule_id.split(":", 1)[1]
    return int(raw) if raw.isdigit() else None


def apply_attribute_coverage_repairs(
    fol: dict[str, Any],
    repairs: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    tables: dict[str, Table],
    allow_drop: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply attribute repairs only when rules are grounded in selected matches and schema."""
    rules = copy.deepcopy(fol.get("rules", {}) or {})
    for kind in ("class", "data", "object"):
        rules.setdefault(kind, [])
    data_rules = list(rules.get("data", []) or [])
    action_counts: Counter[str] = Counter()
    rejected: list[dict[str, Any]] = []

    for repair in repairs:
        action = str(repair.get("action", ""))
        action_counts[action] += 1
        issue_id = str(repair.get("issue_id") or repair.get("rule_id") or "")
        raw_rule = repair.get("rule")
        rule = dict(raw_rule) if isinstance(raw_rule, dict) else None
        data_index = _parse_data_rule_id(issue_id)

        if action == "drop" and allow_drop and data_index is not None and 0 <= data_index < len(data_rules):
            data_rules[data_index] = {"_dropped": True}
            continue
        if action == "repair" and rule and data_index is not None and 0 <= data_index < len(data_rules):
            if _rule_is_grounded_data_rule(rule, matches, tables):
                data_rules[data_index] = rule
            else:
                rejected.append({"issue_id": issue_id, "action": action, "reason": "ungrounded_repaired_rule"})
            continue
        if action == "add_data_rule" and rule:
            if _rule_is_grounded_data_rule(rule, matches, tables):
                data_rules.append(rule)
            else:
                rejected.append({"issue_id": issue_id, "action": action, "reason": "ungrounded_added_rule"})
            continue
        if action == "keep_with_justification":
            continue
        rejected.append({"issue_id": issue_id, "action": action, "reason": "unsupported_action_or_missing_rule"})

    rules["data"] = [rule for rule in data_rules if not rule.get("_dropped")]
    summary = {
        "rules_added": int(action_counts.get("add_data_rule", 0)),
        "rules_repaired": int(action_counts.get("repair", 0)),
        "rules_dropped": int(action_counts.get("drop", 0)),
        "rules_kept_with_justification": int(action_counts.get("keep_with_justification", 0)),
        "repairs_rejected": len(rejected),
        "rejected_repairs": rejected,
        "action_counts": dict(action_counts),
    }
    return {"rules": {kind: list(rules.get(kind, []) or []) for kind in ("class", "data", "object")}}, summary
