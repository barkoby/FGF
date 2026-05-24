from __future__ import annotations

import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from rdflib import Graph, Literal, RDF, URIRef

from .schema import SqlData, Table, foreign_key_columns


_TEXT_HINT_WORDS = {
    "abstract",
    "comment",
    "description",
    "isbn",
    "issn",
    "keyword",
    "name",
    "review",
    "summary",
    "text",
    "title",
}


def _source_data_parts(source_id: str) -> tuple[str, str] | None:
    if not source_id.startswith("source-data:") or "." not in source_id:
        return None
    tail = source_id.split(":", 1)[1]
    table, column = tail.split(".", 1)
    return table, column


def _match_type(match: dict[str, Any]) -> str:
    source_id = str(match.get("source_id", ""))
    target_kind = str(match.get("target_kind", "") or match.get("kind", "")).lower()
    if source_id.startswith(("source-class:", "source-discriminator:")) or target_kind == "class":
        return "class"
    if source_id.startswith("source-data:") or target_kind in {"datatype", "datatype_property", "data_property"}:
        return "datatype_property"
    return "object_property"


def _selected_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [match for match in matches if match.get("target_uri")]


def _allowed_target_uris(matches: list[dict[str, Any]]) -> set[str]:
    return {str(match.get("target_uri")) for match in _selected_matches(matches)}


def _valid_match_ids(rule: dict[str, Any], matches: list[dict[str, Any]]) -> bool:
    selected = {str(match.get("source_id")) for match in _selected_matches(matches)}
    rule_ids = {str(value) for value in rule.get("match_ids", []) or []}
    return bool(rule_ids) and bool(rule_ids & selected)


def _rule_target(kind: str, rule: dict[str, Any]) -> str:
    if kind == "class":
        return str(rule.get("target_class", ""))
    return str(rule.get("target_property", ""))


def _rule_id(kind: str, index: int, rule: dict[str, Any]) -> str:
    return str(rule.get("rule_id") or rule.get("id") or f"{kind}:{index}")


def _rules_for_match(fol: dict[str, Any], match: dict[str, Any]) -> list[dict[str, Any]]:
    target = str(match.get("target_uri", ""))
    source_id = str(match.get("source_id", ""))
    out: list[dict[str, Any]] = []
    for kind, rules in (fol.get("rules", {}) or {}).items():
        for index, rule in enumerate(rules or []):
            if not isinstance(rule, dict):
                continue
            target_matches = _rule_target(kind, rule) == target
            id_matches = source_id in {str(value) for value in rule.get("match_ids", []) or []}
            if target_matches or id_matches:
                out.append({"kind": kind, "index": index, "rule_id": _rule_id(kind, index, rule), "rule": rule})
    return out


def _table_columns(tables: dict[str, Table], table: str) -> set[str]:
    return set(tables[table].column_names()) if table in tables else set()


def _row_summary(data: SqlData, table: str, column: str | None = None, limit: int = 5) -> dict[str, Any]:
    rows = list(data.rows.get(table, []) or [])
    if column is None:
        return {"table": table, "row_count": len(rows)}
    non_empty = 0
    null_empty = 0
    values: set[str] = set()
    samples: list[str] = []
    identifier_like = 0
    for row in rows:
        value = row.get(column)
        if value is None or value == "":
            null_empty += 1
            continue
        text = str(value)
        non_empty += 1
        values.add(text)
        if len(samples) < limit and text not in samples:
            samples.append(text)
        if _identifier_like_literal(text):
            identifier_like += 1
    return {
        "table": table,
        "column": column,
        "row_count": len(rows),
        "non_empty_count": non_empty,
        "null_empty_count": null_empty,
        "null_empty_rate": null_empty / len(rows) if rows else 0.0,
        "distinct_count": len(values),
        "identifier_like_rate": identifier_like / non_empty if non_empty else 0.0,
        "sample_values": samples,
    }


def _is_text_like_column(table: str, column: str) -> bool:
    lowered = f"{table}_{column}".lower()
    return any(word in lowered for word in _TEXT_HINT_WORDS)


def _identifier_like_literal(value: str) -> bool:
    text = str(value).strip()
    if not text:
        return False
    if len(text) <= 3 and re.fullmatch(r"[A-Za-z0-9_-]+", text):
        return True
    if re.fullmatch(r"[A-Za-z]?\d+[A-Za-z]?", text):
        return True
    if re.fullmatch(r"[A-Za-z0-9_-]{1,12}", text) and not re.search(r"\s", text):
        return True
    return False


def _graph_counts(graph_path: Path) -> dict[str, dict[str, Any]]:
    graph = Graph()
    if graph_path.exists():
        graph.parse(str(graph_path))
    counts: dict[str, dict[str, Any]] = {}
    for subject, predicate, obj in graph:
        if predicate == RDF.type and isinstance(obj, URIRef):
            key = str(obj)
            record = counts.setdefault(key, {"class_triples": 0, "predicate_triples": 0, "subjects": set(), "objects": set(), "literal_samples": []})
            record["class_triples"] += 1
            record["subjects"].add(str(subject))
            record["objects"].add(str(obj))
        key = str(predicate)
        record = counts.setdefault(key, {"class_triples": 0, "predicate_triples": 0, "subjects": set(), "objects": set(), "literal_samples": []})
        record["predicate_triples"] += 1
        record["subjects"].add(str(subject))
        record["objects"].add(str(obj))
        if isinstance(obj, Literal) and len(record["literal_samples"]) < 8:
            record["literal_samples"].append(str(obj))
    for record in counts.values():
        record["subject_count"] = len(record.pop("subjects"))
        record["object_distinct_count"] = len(record.pop("objects"))
    return counts


def _target_count(match_type: str, target_uri: str, counts: dict[str, dict[str, Any]]) -> int:
    record = counts.get(target_uri, {})
    if match_type == "class":
        return int(record.get("class_triples", 0) or 0)
    return int(record.get("predicate_triples", 0) or 0)


def _runtime_rule_stats(runtime_log: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = runtime_log.get("rule_stats", {}) if isinstance(runtime_log, dict) else {}
    return {str(key): dict(value or {}) for key, value in (raw or {}).items()}


def _add_issue(issues: list[dict[str, Any]], issue: dict[str, Any]) -> None:
    issues.append({key: value for key, value in issue.items() if value not in (None, "", [], {})})


def materialization_coverage_diagnostics(
    matches: list[dict[str, Any]],
    fol: dict[str, Any],
    tables: dict[str, Table],
    data: SqlData,
    graph_path: Path,
    runtime_log: dict[str, Any],
    code_text: str = "",
) -> dict[str, Any]:
    """Gold-blind coverage checks over selected matches, FOL, generated RDF, and runtime diagnostics."""
    counts = _graph_counts(graph_path)
    stats_by_rule = _runtime_rule_stats(runtime_log)
    selected = _selected_matches(matches)
    issues: list[dict[str, Any]] = []
    target_records: list[dict[str, Any]] = []

    for match in selected:
        target_uri = str(match.get("target_uri", ""))
        match_type = _match_type(match)
        count = _target_count(match_type, target_uri, counts)
        graph_record = counts.get(target_uri, {})
        rules = _rules_for_match(fol, match)
        source_table = ""
        source_column = ""
        source_parts = _source_data_parts(str(match.get("source_id", "")))
        if source_parts:
            source_table, source_column = source_parts
        target_record = {
            "source_id": match.get("source_id"),
            "target_uri": target_uri,
            "match_type": match_type,
            "source_table": source_table,
            "source_column": source_column,
            "appears_in_generated_rdf": count > 0,
            "emitted_triple_count": count,
            "subject_count": graph_record.get("subject_count", 0),
            "object_literal_distinct_count": graph_record.get("object_distinct_count", 0),
            "literal_samples": graph_record.get("literal_samples", []),
            "rules": [{"rule_id": rule["rule_id"], "kind": rule["kind"]} for rule in rules],
        }
        if source_table and source_column and source_table in tables and source_column in tables[source_table].column_names():
            target_record["source_value_summary"] = _row_summary(data, source_table, source_column)
        target_records.append(target_record)

        if count == 0:
            _add_issue(
                issues,
                {
                    "issue_id": f"target_zero:{len(issues)+1}",
                    "issue": "selected_target_zero_emission",
                    "source_id": match.get("source_id"),
                    "target_uri": target_uri,
                    "match_type": match_type,
                    "source_table": source_table,
                    "source_column": source_column,
                    "rule_ids": [rule["rule_id"] for rule in rules],
                },
            )
        if match_type == "datatype_property" and source_table and source_column and _is_text_like_column(source_table, source_column):
            samples = [str(value) for value in graph_record.get("literal_samples", [])]
            if count == 0:
                _add_issue(
                    issues,
                    {
                        "issue_id": f"text_missing:{len(issues)+1}",
                        "issue": "text_like_selected_evidence_missing_literals",
                        "source_id": match.get("source_id"),
                        "target_uri": target_uri,
                        "source_table": source_table,
                        "source_column": source_column,
                    },
                )
            elif samples and sum(1 for sample in samples if _identifier_like_literal(sample)) / len(samples) >= 0.75:
                _add_issue(
                    issues,
                    {
                        "issue_id": f"text_identifier:{len(issues)+1}",
                        "issue": "text_like_literal_identifier_like",
                        "source_id": match.get("source_id"),
                        "target_uri": target_uri,
                        "source_table": source_table,
                        "source_column": source_column,
                        "literal_samples": samples[:5],
                    },
                )
        if source_table and source_column and source_table in tables and source_column in tables[source_table].column_names():
            summary = _row_summary(data, source_table, source_column)
            row_guard = max(20, int(summary.get("non_empty_count", 0) or summary.get("row_count", 0) or 0) * 3)
            if count > row_guard:
                _add_issue(
                    issues,
                    {
                        "issue_id": f"overbroad:{len(issues)+1}",
                        "issue": "overbroad_emission",
                        "source_id": match.get("source_id"),
                        "target_uri": target_uri,
                        "source_table": source_table,
                        "source_column": source_column,
                        "emitted_triple_count": count,
                        "source_non_empty_count": summary.get("non_empty_count"),
                        "guardrail": row_guard,
                    },
                )

    for kind, rules in (fol.get("rules", {}) or {}).items():
        for index, rule in enumerate(rules or []):
            if not isinstance(rule, dict):
                continue
            rid = _rule_id(kind, index, rule)
            stat = stats_by_rule.get(rid, {})
            source_table = str(rule.get("source_table", ""))
            if source_table and source_table not in tables:
                _add_issue(issues, {"issue_id": f"unknown_table:{rid}", "issue": "unknown_source_table", "rule_id": rid, "source_table": source_table})
            source_columns: list[str] = []
            if kind == "data" and rule.get("source_column"):
                source_columns.append(str(rule.get("source_column")))
            for key in ("source_columns", "subject_columns", "object_columns"):
                source_columns.extend(str(value) for value in rule.get(key, []) or [])
            for column in source_columns:
                if source_table in tables and column not in _table_columns(tables, source_table):
                    _add_issue(
                        issues,
                        {
                            "issue_id": f"unknown_column:{rid}:{column}",
                            "issue": "unknown_source_column",
                            "rule_id": rid,
                            "source_table": source_table,
                            "source_column": column,
                        },
                    )
            if kind == "data" and source_table in tables and str(rule.get("source_column", "")) in foreign_key_columns(tables[source_table]) and not rule.get("subject_table"):
                _add_issue(
                    issues,
                    {
                        "issue_id": f"fk_literal:{rid}",
                        "issue": "fk_like_datatype_literal_without_justification",
                        "rule_id": rid,
                        "source_table": source_table,
                        "source_column": rule.get("source_column"),
                        "target_uri": rule.get("target_property"),
                    },
                )
            reachable = int(stat.get("reachable_rows", 0) or 0)
            helper_calls = int(stat.get("helper_calls", 0) or 0)
            emitted = int(stat.get("emitted_triples", 0) or 0)
            if reachable > 0 and emitted == 0:
                _add_issue(
                    issues,
                    {
                        "issue_id": f"zero_rule:{rid}",
                        "issue": "reachable_rule_zero_emission",
                        "rule_id": rid,
                        "kind": kind,
                        "source_table": source_table,
                        "helper_calls": helper_calls,
                        "reachable_rows": reachable,
                        "failure_counts": stat.get("failure_counts", {}),
                        "failure_samples": stat.get("failure_samples", []),
                    },
                )
            if reachable > 0 and helper_calls == 0:
                _add_issue(
                    issues,
                    {
                        "issue_id": f"zero_branch:{rid}",
                        "issue": "generated_code_path_zero_helper_calls",
                        "rule_id": rid,
                        "kind": kind,
                        "source_table": source_table,
                        "reachable_rows": reachable,
                    },
                )
            failures = stat.get("failure_counts", {}) or {}
            if int(failures.get("missing_target_row", 0) or 0) + int(failures.get("missing_object_row", 0) or 0) > 0 and emitted == 0:
                _add_issue(
                    issues,
                    {
                        "issue_id": f"zero_join:{rid}",
                        "issue": "zero_row_join",
                        "rule_id": rid,
                        "kind": kind,
                        "source_table": source_table,
                        "failure_counts": failures,
                        "failure_samples": stat.get("failure_samples", []),
                    },
                )

    issue_counts = dict(Counter(str(issue.get("issue", "unknown")) for issue in issues))
    zero_targets = sum(1 for record in target_records if not record.get("appears_in_generated_rdf"))
    return {
        "selected_targets": target_records,
        "selected_target_count": len(target_records),
        "selected_targets_with_emission": len(target_records) - zero_targets,
        "zero_emission_selected_targets": zero_targets,
        "generated_triples": int(runtime_log.get("generated_triples", 0) or 0),
        "invalid_triple_count": int(runtime_log.get("invalid_triple_count", 0) or 0),
        "issues": issues,
        "issues_by_type": issue_counts,
        "code_summary": {
            "contains_materialize": "def materialize" in code_text,
            "helper_mentions": {
                "emit_type": code_text.count("emit_type"),
                "emit_data": code_text.count("emit_data"),
                "emit_object": code_text.count("emit_object"),
            },
        },
    }


def materialization_coverage_summary(report: dict[str, Any]) -> dict[str, Any]:
    issues = list(report.get("issues", []) or [])
    counts = report.get("issues_by_type", {}) or {}
    return {
        "selected_target_count": report.get("selected_target_count", 0),
        "selected_targets_with_emission": report.get("selected_targets_with_emission", 0),
        "zero_emission_selected_targets": report.get("zero_emission_selected_targets", 0),
        "issue_count": len(issues),
        "unknown_reference_issues": int(counts.get("unknown_source_table", 0)) + int(counts.get("unknown_source_column", 0)),
        "fk_like_literal_issues": int(counts.get("fk_like_datatype_literal_without_justification", 0)),
        "overbroad_issues": int(counts.get("overbroad_emission", 0)),
        "zero_rule_issues": int(counts.get("reachable_rule_zero_emission", 0)),
        "zero_branch_issues": int(counts.get("generated_code_path_zero_helper_calls", 0)),
        "invalid_triple_count": report.get("invalid_triple_count", 0),
        "generated_triples": report.get("generated_triples", 0),
    }


def _table_excerpt(table: Table) -> dict[str, Any]:
    return {
        "name": table.name,
        "columns": [{"name": column.name, "datatype": column.datatype, "nullable": column.nullable} for column in table.columns],
        "primary_key": table.primary_key,
        "foreign_keys": [{"columns": fk.columns, "ref_table": fk.ref_table, "ref_columns": fk.ref_columns} for fk in table.foreign_keys],
    }


def _issue_context(
    issue: dict[str, Any],
    report: dict[str, Any],
    fol: dict[str, Any],
    matches: list[dict[str, Any]],
    tables: dict[str, Table],
    data: SqlData,
) -> dict[str, Any]:
    source_ids = {str(issue.get("source_id", ""))}
    rule_ids = {str(issue.get("rule_id", ""))}
    rule_ids.update(str(value) for value in issue.get("rule_ids", []) or [])
    related_matches = [
        {"source_id": match.get("source_id"), "target_uri": match.get("target_uri"), "target_id": match.get("target_id"), "target_local_name": match.get("target_local_name")}
        for match in _selected_matches(matches)
        if str(match.get("source_id", "")) in source_ids or str(match.get("target_uri", "")) == str(issue.get("target_uri", ""))
    ]
    related_rules: list[dict[str, Any]] = []
    table_names: set[str] = set()
    source_summaries: list[dict[str, Any]] = []
    for kind, rules in (fol.get("rules", {}) or {}).items():
        for index, rule in enumerate(rules or []):
            rid = _rule_id(kind, index, rule)
            if rid in rule_ids or _rule_target(kind, rule) == str(issue.get("target_uri", "")):
                related_rules.append({"rule_id": rid, "kind": kind, "rule": rule})
                for key in ("source_table", "target_table", "subject_table", "object_table"):
                    value = str(rule.get(key, ""))
                    if value in tables:
                        table_names.add(value)
                source_table = str(rule.get("source_table", ""))
                if source_table in tables:
                    if rule.get("source_column"):
                        source_summaries.append(_row_summary(data, source_table, str(rule.get("source_column"))))
                    else:
                        source_summaries.append(_row_summary(data, source_table))
    source_table = str(issue.get("source_table", ""))
    source_column = str(issue.get("source_column", ""))
    if source_table in tables:
        table_names.add(source_table)
        if source_column and source_column in tables[source_table].column_names():
            source_summaries.append(_row_summary(data, source_table, source_column))
    return {
        "issue": issue,
        "related_selected_matches": related_matches,
        "current_rules": related_rules,
        "schema_excerpt": [_table_excerpt(tables[name]) for name in sorted(table_names)],
        "source_summaries": source_summaries[:6],
        "allowed_target_uris": sorted(_allowed_target_uris(matches)),
    }


def _prompt_from_contexts(contexts: list[dict[str, Any]]) -> str:
    payload = {
        "task": "Repair materialization coverage using only supplied internal evidence.",
        "allowed_actions": [
            "repair",
            "drop",
            "keep_with_justification",
            "add_class_rule",
            "add_data_rule",
            "add_object_rule",
            "request_code_repair",
        ],
        "constraints": [
            "Return JSON only with a top-level repairs array.",
            "Use only target URIs listed in allowed_target_uris.",
            "Use only source tables and source columns from schema_excerpt.",
            "Every non-null rule must cite valid match_ids from related_selected_matches.",
            "Do not add row constants except values shown in source_summaries.",
            "If support is insufficient, choose drop or keep_with_justification.",
            "Use only this internal evidence; external benchmark feedback is forbidden.",
        ],
        "output_shape": {
            "repairs": [
                {
                    "issue_id": "issue id from payload",
                    "rule_id": "class:0|data:0|object:0 or empty",
                    "action": "repair|drop|keep_with_justification|add_class_rule|add_data_rule|add_object_rule|request_code_repair",
                    "reason": "short internal-evidence explanation",
                    "rule": None,
                }
            ]
        },
        "issue_contexts": contexts,
    }
    return (
        "Prompt version: fgf_materialization_coverage_v1_repair\n"
        "Repair only listed internal materialization coverage issues. Ground every change in the supplied "
        "matches, schema, source summaries, current rules, and runtime diagnostics.\n"
        "Do not invent target URIs, source tables, source columns, match ids, joins, constants, or scenario fixes.\n"
        "Return JSON only.\n\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def build_materialization_repair_prompts(
    report: dict[str, Any],
    fol: dict[str, Any],
    matches: list[dict[str, Any]],
    tables: dict[str, Table],
    data: SqlData,
    budget_chars: int = 24000,
) -> list[dict[str, Any]]:
    issues = list(report.get("issues", []) or [])
    if not issues:
        return []
    budget = max(1200, int(budget_chars or 24000))
    prompts: list[dict[str, Any]] = []
    current_contexts: list[dict[str, Any]] = []
    current_issue_ids: list[str] = []
    for issue in issues:
        context = _issue_context(issue, report, fol, matches, tables, data)
        candidate_contexts = current_contexts + [context]
        prompt = _prompt_from_contexts(candidate_contexts)
        if current_contexts and len(prompt) > budget:
            final_prompt = _prompt_from_contexts(current_contexts)
            prompts.append({"prompt": final_prompt, "issue_ids": current_issue_ids, "char_count": len(final_prompt), "split": True})
            current_contexts = [context]
            current_issue_ids = [str(issue.get("issue_id"))]
            continue
        current_contexts = candidate_contexts
        current_issue_ids.append(str(issue.get("issue_id")))
    if current_contexts:
        final_prompt = _prompt_from_contexts(current_contexts)
        if len(final_prompt) > budget and len(current_contexts) == 1:
            context = copy.deepcopy(current_contexts[0])
            context["source_summaries"] = [
                {key: value for key, value in summary.items() if key != "sample_values"}
                for summary in context.get("source_summaries", [])
            ]
            context["issue"] = {key: value for key, value in context.get("issue", {}).items() if key not in {"failure_samples", "literal_samples"}}
            final_prompt = _prompt_from_contexts([context])
        prompts.append({"prompt": final_prompt, "issue_ids": current_issue_ids, "char_count": len(final_prompt), "split": len(prompts) > 0})
    return prompts


def _parse_rule_id(rule_id: str) -> tuple[str, int] | None:
    if ":" not in rule_id:
        return None
    kind, raw = rule_id.split(":", 1)
    if kind in {"class", "data", "object"} and raw.isdigit():
        return kind, int(raw)
    return None


def _rule_is_grounded(rule: dict[str, Any], kind: str, matches: list[dict[str, Any]], tables: dict[str, Table]) -> bool:
    allowed_targets = _allowed_target_uris(matches)
    target = _rule_target(kind, rule)
    if target not in allowed_targets:
        return False
    if not _valid_match_ids(rule, matches):
        return False
    source_table = str(rule.get("source_table", ""))
    if source_table not in tables:
        return False
    source_columns = set()
    if kind == "data":
        source_columns.add(str(rule.get("source_column", "")))
    for key in ("source_columns", "subject_columns", "object_columns"):
        source_columns.update(str(value) for value in rule.get(key, []) or [])
    if any(column and column not in _table_columns(tables, source_table) for column in source_columns):
        return False
    for table_key, column_key in (("target_table", "target_columns"), ("subject_table", "subject_target_columns"), ("object_table", "object_target_columns")):
        table_name = str(rule.get(table_key, ""))
        if table_name:
            if table_name not in tables:
                return False
            if any(str(column) not in _table_columns(tables, table_name) for column in rule.get(column_key, []) or []):
                return False
    return True


def apply_materialization_coverage_repairs(
    fol: dict[str, Any],
    repairs: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    tables: dict[str, Table],
    allow_drop: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rules = copy.deepcopy(fol.get("rules", {}) or {})
    for kind in ("class", "data", "object"):
        rules.setdefault(kind, [])
    action_counts: Counter[str] = Counter()
    rejected: list[dict[str, Any]] = []
    code_repair_requests = 0

    for repair in repairs:
        action = str(repair.get("action", ""))
        action_counts[action] += 1
        rule_id = str(repair.get("rule_id") or "")
        parsed = _parse_rule_id(rule_id)
        raw_rule = repair.get("rule")
        rule = dict(raw_rule) if isinstance(raw_rule, dict) else None
        if action == "request_code_repair":
            code_repair_requests += 1
            continue
        if action == "drop" and allow_drop and parsed:
            kind, index = parsed
            if 0 <= index < len(rules.get(kind, [])):
                rules[kind][index] = {"_dropped": True}
            continue
        if action in {"repair", "add_class_rule", "add_data_rule", "add_object_rule"} and rule:
            kind = parsed[0] if action == "repair" and parsed else (
                "class" if action == "add_class_rule" else "data" if action == "add_data_rule" else "object"
            )
            if not _rule_is_grounded(rule, kind, matches, tables):
                rejected.append({"rule_id": rule_id, "action": action, "reason": "ungrounded_rule"})
                continue
            if action == "repair" and parsed:
                _, index = parsed
                if 0 <= index < len(rules.get(kind, [])):
                    rules[kind][index] = rule
                else:
                    rejected.append({"rule_id": rule_id, "action": action, "reason": "rule_index_out_of_range"})
            else:
                rules[kind].append(rule)
            continue
        if action == "keep_with_justification":
            continue
        rejected.append({"rule_id": rule_id, "action": action, "reason": "unsupported_action_or_missing_rule"})

    cleaned_rules = {
        kind: [rule for rule in rules.get(kind, []) if isinstance(rule, dict) and not rule.get("_dropped")]
        for kind in ("class", "data", "object")
    }
    return {
        "rules": cleaned_rules,
        "generation": {**(fol.get("generation", {}) or {}), "materialization_coverage_repair": True},
    }, {
        "action_counts": dict(action_counts),
        "rules_added": int(action_counts.get("add_class_rule", 0)) + int(action_counts.get("add_data_rule", 0)) + int(action_counts.get("add_object_rule", 0)),
        "rules_repaired": int(action_counts.get("repair", 0)),
        "rules_dropped": int(action_counts.get("drop", 0)),
        "rules_kept_with_justification": int(action_counts.get("keep_with_justification", 0)),
        "code_repair_requests": code_repair_requests,
        "repairs_rejected": len(rejected),
        "rejected_repairs": rejected,
    }


def internal_materialization_score(report: dict[str, Any]) -> tuple[int, int, int, int, int, int, int]:
    summary = materialization_coverage_summary(report)
    return (
        -int(summary.get("invalid_triple_count", 0) or 0),
        -int(summary.get("zero_emission_selected_targets", 0) or 0),
        -int(summary.get("unknown_reference_issues", 0) or 0),
        -int(summary.get("fk_like_literal_issues", 0) or 0),
        -int(summary.get("overbroad_issues", 0) or 0),
        int(summary.get("selected_targets_with_emission", 0) or 0),
        int(summary.get("generated_triples", 0) or 0),
    )


def materialization_repair_accepted(before: dict[str, Any], after: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    before_summary = materialization_coverage_summary(before)
    after_summary = materialization_coverage_summary(after)
    before_triples = int(before_summary.get("generated_triples", 0) or 0)
    after_triples = int(after_summary.get("generated_triples", 0) or 0)
    if int(after_summary.get("invalid_triple_count", 0) or 0) > int(before_summary.get("invalid_triple_count", 0) or 0):
        reasons.append("invalid_triple_count_worse")
    if before_triples and after_triples < before_triples * 0.5:
        reasons.append("generated_triples_collapsed")
    if int(after_summary.get("overbroad_issues", 0) or 0) > int(before_summary.get("overbroad_issues", 0) or 0):
        reasons.append("overbroad_issue_count_worse")
    if internal_materialization_score(after) < internal_materialization_score(before):
        reasons.append("internal_materialization_score_worse")
    return not reasons, reasons
