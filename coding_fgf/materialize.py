from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from rdflib import Graph, Literal, RDF, URIRef

from .constants import SOURCE_BASE
from .sandbox import run_generated_materializer
from .schema import ForeignKey, SqlData, Table, key_tuple, table_key


def safe_fragment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_") or "id"


def instance_uri(scenario: str, table: str, row: dict[str, Optional[str]], id_columns: list[str]) -> str:
    values = [row.get(col) for col in id_columns if row.get(col) is not None]
    if not values:
        values = [row.get(col) for col in row if row.get(col) is not None][:1]
    key = "_".join(safe_fragment(str(v)) for v in values) or "row"
    return f"{SOURCE_BASE}{scenario}/{table}/{key}"


def normalize_literal_value(value: object) -> str:
    text = str(value)
    return text.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t").replace("\\\\", "\\")


def _is_uri_like(value: object) -> bool:
    text = str(value or "")
    return text.startswith(("http://", "https://", "urn:"))


def _invalid_triples(triples: list[tuple[str, str, str]]) -> list[dict[str, Any]]:
    invalid: list[dict[str, Any]] = []
    for index, triple in enumerate(triples):
        if not isinstance(triple, tuple) or len(triple) != 3:
            invalid.append({"index": index, "reason": "invalid_triple_shape", "triple": repr(triple)})
            continue
        subject, predicate, obj = triple
        if not _is_uri_like(subject):
            invalid.append({"index": index, "reason": "invalid_subject_uri", "triple": repr(triple)})
        if not _is_uri_like(predicate):
            invalid.append({"index": index, "reason": "invalid_predicate_uri", "triple": repr(triple)})
        if not (isinstance(obj, str) and obj.startswith("literal:")) and not _is_uri_like(obj):
            invalid.append({"index": index, "reason": "invalid_object", "triple": repr(triple)})
    return invalid


def _sample_row(row: dict[str, Any], limit: int = 6) -> dict[str, Any]:
    return {str(key): row.get(key) for key in list(row.keys())[:limit]}


def _primary_key_parent_fk(table: Table) -> ForeignKey | None:
    if not table.primary_key:
        return None
    for fk in table.foreign_keys:
        if fk.columns == table.primary_key:
            return fk
    return None


def canonical_identity(
    tables: dict[str, Table],
    table: str,
    row: dict[str, Optional[str]],
    id_columns: list[str],
) -> tuple[str, dict[str, Optional[str]], list[str]]:
    current_table = table
    current_row = dict(row)
    current_id_columns = list(id_columns or table_key(tables[current_table]))
    seen: set[str] = set()
    while current_table in tables and current_table not in seen:
        seen.add(current_table)
        parent_fk = _primary_key_parent_fk(tables[current_table])
        if not parent_fk:
            break
        parent_row = {
            ref_col: current_row.get(source_col)
            for source_col, ref_col in zip(parent_fk.columns, parent_fk.ref_columns)
        }
        if any(value is None or value == "" for value in parent_row.values()):
            break
        current_table = parent_fk.ref_table
        current_row = parent_row
        current_id_columns = list(parent_fk.ref_columns or table_key(tables[current_table]))
    return current_table, current_row, current_id_columns


def canonical_instance_uri(
    scenario: str,
    tables: dict[str, Table],
    table: str,
    row: dict[str, Optional[str]],
    id_columns: list[str],
) -> str:
    canonical_table, canonical_row, canonical_id_columns = canonical_identity(tables, table, row, id_columns)
    return instance_uri(scenario, canonical_table, canonical_row, canonical_id_columns)


def build_context(scenario: str, tables: dict[str, Table], data: SqlData, fol: dict[str, Any]) -> dict[str, Any]:
    class_ids = {
        rule["source_table"]: list(rule.get("id_columns") or table_key(tables[rule["source_table"]]))
        for rule in fol.get("rules", {}).get("class", [])
        if rule.get("source_table") in tables
    }
    rule_stats: dict[str, dict[str, Any]] = {}
    rule_ids: dict[int, str] = {}
    for kind, rules in (fol.get("rules", {}) or {}).items():
        for index, rule in enumerate(rules or []):
            if not isinstance(rule, dict):
                continue
            rule_id = str(rule.get("rule_id") or rule.get("id") or f"{kind}:{index}")
            source_table = str(rule.get("source_table") or "")
            reachable_rows = len(data.rows.get(source_table, []))
            rule_ids[id(rule)] = rule_id
            rule_stats[rule_id] = {
                "rule_id": rule_id,
                "kind": kind,
                "source_table": source_table,
                "reachable_rows": reachable_rows,
                "helper_calls": 0,
                "emitted_triples": 0,
                "failure_counts": {},
                "failure_samples": [],
            }

    def rule_id_for(rule: dict[str, Any]) -> str:
        return rule_ids.get(id(rule), str(rule.get("rule_id") or rule.get("id") or "unknown"))

    def note_helper_call(rule: dict[str, Any]) -> None:
        stats = rule_stats.get(rule_id_for(rule))
        if stats is not None:
            stats["helper_calls"] = int(stats.get("helper_calls", 0)) + 1

    def note_emit(rule: dict[str, Any]) -> None:
        stats = rule_stats.get(rule_id_for(rule))
        if stats is not None:
            stats["emitted_triples"] = int(stats.get("emitted_triples", 0)) + 1

    def note_failure(rule: dict[str, Any], reason: str, row: dict[str, Any] | None = None, details: dict[str, Any] | None = None) -> None:
        stats = rule_stats.get(rule_id_for(rule))
        if stats is None:
            return
        counts = stats.setdefault("failure_counts", {})
        counts[reason] = int(counts.get(reason, 0)) + 1
        samples = stats.setdefault("failure_samples", [])
        if len(samples) < 3:
            sample: dict[str, Any] = {"reason": reason}
            if row is not None:
                sample["row"] = _sample_row(row)
            if details:
                sample["details"] = details
            samples.append(sample)
    row_indexes = {
        table_name: {key_tuple(row, table_key(tables[table_name])) for row in rows}
        for table_name, rows in data.rows.items()
        if table_name in tables
    }

    def row_exists(table: str, row: dict[str, Optional[str]], key_columns: list[str]) -> bool:
        if table not in row_indexes:
            return False
        key_columns = key_columns or table_key(tables[table])
        if any(row.get(column) is None or row.get(column) == "" for column in key_columns):
            return False
        return key_tuple(row, key_columns) in row_indexes[table]

    def row_matches_filter(row: dict[str, Any], rule: dict[str, Any]) -> bool:
        row_filter = rule.get("row_filter")
        if not row_filter:
            return True
        column = row_filter.get("column")
        value = row.get(column)
        if row_filter.get("truthy"):
            return str(value).strip().lower() in {"1", "t", "true", "yes", "y"}
        if "equals" in row_filter:
            return str(value).strip().lower() == str(row_filter["equals"]).strip().lower()
        return True

    def emit_type(row: dict[str, Any], rule: dict[str, Any]) -> tuple[str, str, str] | None:
        note_helper_call(rule)
        if not row_matches_filter(row, rule):
            note_failure(rule, "row_filter_rejected", row)
            return None
        table = rule["source_table"]
        triple = (
            canonical_instance_uri(scenario, tables, table, row, list(rule.get("id_columns") or class_ids.get(table, []))),
            str(RDF.type),
            rule["target_class"],
        )
        note_emit(rule)
        return triple

    def emit_data(row: dict[str, Any], rule: dict[str, Any]) -> tuple[str, str, str] | None:
        note_helper_call(rule)
        value = row.get(rule["source_column"])
        if value is None or value == "":
            note_failure(rule, "empty_source_value", row, {"source_column": rule.get("source_column")})
            return None
        table = rule["source_table"]
        if rule.get("subject_table"):
            subject_table = rule["subject_table"]
            subject_cols = list(rule.get("subject_columns") or [])
            subject_target_cols = list(rule.get("subject_target_columns") or table_key(tables[subject_table]))
            subject_row = {target_col: row.get(source_col) for source_col, target_col in zip(subject_cols, subject_target_cols)}
            if any(value is None or value == "" for value in subject_row.values()):
                note_failure(rule, "missing_subject_key", row, {"subject_table": subject_table})
                return None
            if not row_exists(subject_table, subject_row, subject_target_cols):
                note_failure(rule, "missing_subject_row", row, {"subject_table": subject_table, "subject_row": subject_row})
                return None
            triple = (
                canonical_instance_uri(scenario, tables, subject_table, subject_row, subject_target_cols),
                rule["target_property"],
                "literal:" + normalize_literal_value(value),
            )
            note_emit(rule)
            return triple
        triple = (
            canonical_instance_uri(scenario, tables, table, row, class_ids.get(table, table_key(tables[table]))),
            rule["target_property"],
            "literal:" + normalize_literal_value(value),
        )
        note_emit(rule)
        return triple

    def emit_object(row: dict[str, Any], rule: dict[str, Any]) -> tuple[str, str, str] | None:
        note_helper_call(rule)
        table = rule["source_table"]
        if rule.get("subject_table") and rule.get("object_table"):
            subject_table = rule["subject_table"]
            object_table = rule["object_table"]
            subject_cols = list(rule.get("subject_columns") or [])
            object_cols = list(rule.get("object_columns") or [])
            subject_target_cols = list(rule.get("subject_target_columns") or table_key(tables[subject_table]))
            object_target_cols = list(rule.get("object_target_columns") or table_key(tables[object_table]))
            subject_row = {target_col: row.get(source_col) for source_col, target_col in zip(subject_cols, subject_target_cols)}
            object_row = {target_col: row.get(source_col) for source_col, target_col in zip(object_cols, object_target_cols)}
            if any(value is None or value == "" for value in subject_row.values()):
                note_failure(rule, "missing_subject_key", row, {"subject_table": subject_table})
                return None
            if any(value is None or value == "" for value in object_row.values()):
                note_failure(rule, "missing_object_key", row, {"object_table": object_table})
                return None
            if not row_exists(subject_table, subject_row, subject_target_cols):
                note_failure(rule, "missing_subject_row", row, {"subject_table": subject_table, "subject_row": subject_row})
                return None
            if not row_exists(object_table, object_row, object_target_cols):
                note_failure(rule, "missing_object_row", row, {"object_table": object_table, "object_row": object_row})
                return None
            triple = (
                canonical_instance_uri(scenario, tables, subject_table, subject_row, subject_target_cols),
                rule["target_property"],
                canonical_instance_uri(scenario, tables, object_table, object_row, object_target_cols),
            )
            note_emit(rule)
            return triple

        target_table = rule["target_table"]
        source_cols = list(rule.get("source_columns") or [])
        target_cols = list(rule.get("target_columns") or table_key(tables[target_table]))
        if not source_cols:
            note_failure(rule, "missing_source_columns", row, {"target_table": target_table})
            return None
        target_row = {target_col: row.get(source_col) for source_col, target_col in zip(source_cols, target_cols)}
        if any(value is None or value == "" for value in target_row.values()):
            note_failure(rule, "missing_target_key", row, {"target_table": target_table, "target_row": target_row})
            return None
        if not row_exists(target_table, target_row, target_cols):
            note_failure(rule, "missing_target_row", row, {"target_table": target_table, "target_row": target_row})
            return None
        triple = (
            canonical_instance_uri(scenario, tables, table, row, class_ids.get(table, table_key(tables[table]))),
            rule["target_property"],
            canonical_instance_uri(scenario, tables, target_table, target_row, target_cols),
        )
        note_emit(rule)
        return triple

    return {
        "rows": data.rows,
        "rules": fol.get("rules", {}),
        "emit_type": emit_type,
        "emit_data": emit_data,
        "emit_object": emit_object,
        "runtime_log": {
            "scenario": scenario,
            "rule_stats": rule_stats,
            "invalid_triples": [],
            "invalid_triple_count": 0,
            "generated_triples": 0,
            "status": "pending",
        },
    }


def materialize_graph(
    scenario: str,
    tables: dict[str, Table],
    data: SqlData,
    fol: dict[str, Any],
    code: str | None = None,
) -> Graph:
    graph, _ = materialize_graph_with_log(scenario, tables, data, fol, code=code)
    return graph


def materialize_graph_with_log(
    scenario: str,
    tables: dict[str, Table],
    data: SqlData,
    fol: dict[str, Any],
    code: str | None = None,
    raise_on_invalid: bool = True,
) -> tuple[Graph, dict[str, Any]]:
    context = build_context(scenario, tables, data, fol)
    if code:
        triples = run_generated_materializer(code, context)
    else:
        triples = []
        for rule in context["rules"].get("class", []):
            for row in context["rows"].get(rule["source_table"], []):
                triple = context["emit_type"](row, rule)
                if triple:
                    triples.append(triple)
        for rule in context["rules"].get("data", []):
            for row in context["rows"].get(rule["source_table"], []):
                triple = context["emit_data"](row, rule)
                if triple:
                    triples.append(triple)
        for rule in context["rules"].get("object", []):
            for row in context["rows"].get(rule["source_table"], []):
                triple = context["emit_object"](row, rule)
                if triple:
                    triples.append(triple)

    invalid = _invalid_triples(triples)
    runtime_log = context["runtime_log"]
    runtime_log["generated_triples"] = len(triples)
    runtime_log["invalid_triples"] = invalid
    runtime_log["invalid_triple_count"] = len(invalid)
    runtime_log["status"] = "invalid_triples" if invalid else "success"
    if invalid and raise_on_invalid:
        raise ValueError(f"Generated materializer emitted {len(invalid)} invalid RDF triple(s)")

    graph = Graph()
    for subject, predicate, obj in triples:
        rdf_obj = Literal(obj[len("literal:") :]) if isinstance(obj, str) and obj.startswith("literal:") else URIRef(obj)
        graph.add((URIRef(subject), URIRef(predicate), rdf_obj))
    return graph, runtime_log


def materialize_to_file(
    scenario: str,
    tables: dict[str, Table],
    data: SqlData,
    fol_path: Path,
    output_path: Path,
    code_path: Path | None = None,
    require_code: bool = True,
) -> Path:
    fol = json.loads(fol_path.read_text(encoding="utf-8"))
    if require_code and (code_path is None or not code_path.exists()):
        raise FileNotFoundError("Generated code is required for RDF materialization; no deterministic materialization fallback is allowed")
    code = code_path.read_text(encoding="utf-8") if code_path and code_path.exists() else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_path.with_suffix(".materialization_log.json")
    try:
        graph, runtime_log = materialize_graph_with_log(scenario, tables, data, fol, code=code, raise_on_invalid=False)
        log_path.write_text(json.dumps(runtime_log, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        if int(runtime_log.get("invalid_triple_count", 0) or 0):
            raise ValueError(f"Generated materializer emitted {runtime_log['invalid_triple_count']} invalid RDF triple(s)")
        graph.serialize(destination=str(output_path), format="turtle")
    except Exception as exc:
        if not log_path.exists():
            log_path.write_text(
                json.dumps({"scenario": scenario, "status": "error", "error": f"{type(exc).__name__}: {exc}"}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        raise
    return output_path
