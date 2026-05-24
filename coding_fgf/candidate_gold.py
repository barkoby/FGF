from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .constants import PAPER_SCENARIOS
from .evaluate import parse_qpair
from .ontology import parse_ontology_records, source_schema_records
from .schema import (
    Table,
    find_ontology_file,
    foreign_key_columns,
    is_generic_identifier_column,
    parse_copy_data,
    parse_sql_dump,
    table_role,
)


STANDARD_PREFIXES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}

SOURCE_KIND_BY_TARGET_KIND = {
    "class": "class",
    "data_property": "data_property",
    "object_property": "object_property",
}

SQL_KEYWORDS = {
    "and",
    "as",
    "asc",
    "avg",
    "by",
    "cast",
    "character",
    "coalesce",
    "concat",
    "count",
    "desc",
    "distinct",
    "from",
    "group",
    "having",
    "in",
    "inner",
    "is",
    "join",
    "left",
    "like",
    "limit",
    "not",
    "null",
    "on",
    "or",
    "order",
    "outer",
    "right",
    "select",
    "sum",
    "then",
    "union",
    "varying",
    "where",
}


@dataclass(frozen=True)
class GoldMapping:
    scenario: str
    source_id: str
    source_kind: str
    source_table: str
    source_column: str
    source_table_role: str
    source_column_role: str
    gold_target_uris: tuple[str, ...]
    query_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GoldIssue:
    scenario: str
    query_id: str
    target_uri: str
    target_kind: str
    failure_category: str
    reason: str


@dataclass(frozen=True)
class ScenarioCandidateData:
    scenario: str
    scenario_dir: Path
    tables: dict[str, Table]
    source_records: list[dict[str, Any]]
    target_records: list[dict[str, Any]]
    gold_mappings: list[GoldMapping]
    gold_issues: list[GoldIssue]


def resolve_scenarios(rodi_root: Path, all_scenarios: bool, scenarios: Iterable[str] | None = None) -> list[str]:
    selected = [s.strip() for s in (scenarios or []) if s and s.strip()]
    if all_scenarios or not selected:
        selected = list(PAPER_SCENARIOS)
    for scenario in selected:
        scenario_dir(rodi_root, scenario)
    return selected


def scenario_dir(rodi_root: Path, scenario: str) -> Path:
    for candidate in (rodi_root / "data" / scenario, rodi_root / scenario):
        if candidate.exists():
            _validate_scenario_dir(candidate, scenario)
            return candidate
    raise FileNotFoundError(f"Could not find scenario {scenario} below {rodi_root}")


def _validate_scenario_dir(path: Path, scenario: str) -> None:
    missing = []
    if not (path / "dump.sql").exists():
        missing.append("dump.sql")
    try:
        find_ontology_file(path)
    except FileNotFoundError:
        missing.append("ontology.ttl|ontology.owl|ontology.rdf")
    if not (path / "queries").is_dir():
        missing.append("queries/")
    if missing:
        raise FileNotFoundError(f"Scenario {scenario} at {path} is missing {', '.join(missing)}")


def normalize_uri(value: object, prefixes: dict[str, str] | None = None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1].strip()
    if text.startswith("URIRef(") and text.endswith(")"):
        text = text[len("URIRef(") : -1].strip("'\" ")
    if prefixes and ":" in text and not re.match(r"^[a-z][a-z0-9+.-]*://", text, flags=re.I):
        prefix, local = text.split(":", 1)
        if prefix in prefixes:
            text = prefixes[prefix] + local
    return text


def sparql_prefixes(sparql: str) -> dict[str, str]:
    prefixes = dict(STANDARD_PREFIXES)
    for prefix, uri in re.findall(r"\bPREFIX\s+([^:\s]*):\s*<([^>]+)>", sparql, flags=re.I):
        prefixes[prefix] = uri
    return prefixes


def sparql_target_uris(sparql: str, target_by_uri: dict[str, dict[str, Any]]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    prefixes = sparql_prefixes(sparql)
    seen: set[str] = set()
    targets: list[tuple[str, str]] = []
    unknown: list[tuple[str, str]] = []

    def add(raw: str) -> None:
        uri = normalize_uri(raw, prefixes)
        if not uri or uri in seen:
            return
        seen.add(uri)
        record = target_by_uri.get(uri)
        if record:
            targets.append((uri, str(record.get("kind", ""))))
        else:
            unknown.append((uri, "unknown"))

    for uri in re.findall(r"<([^>]+)>", sparql):
        add(uri)

    token_re = re.compile(r"(?<![A-Za-z0-9_?])([A-Za-z_][\w-]*|):([A-Za-z_][\w.-]*)")
    for prefix, local in token_re.findall(sparql):
        if prefix in STANDARD_PREFIXES:
            continue
        if prefix in prefixes:
            add(f"{prefix}:{local}")
    return targets, unknown


def load_scenario_data(rodi_root: Path, scenario: str) -> ScenarioCandidateData:
    src = scenario_dir(rodi_root, scenario)
    tables = parse_sql_dump(src / "dump.sql")
    data = parse_copy_data(src / "dump.sql")
    target_records = [record.to_dict() for record in parse_ontology_records(find_ontology_file(src))]
    source_records = [record.to_dict() for record in source_schema_records(tables, data.rows)]
    for record in source_records:
        record["source_table_role"] = record.get("table_role", "")
        record["source_column_role"] = source_column_role(record, tables)
    gold_mappings, gold_issues = load_gold_mappings(src, scenario, tables, source_records, target_records)
    return ScenarioCandidateData(
        scenario=scenario,
        scenario_dir=src,
        tables=tables,
        source_records=source_records,
        target_records=target_records,
        gold_mappings=gold_mappings,
        gold_issues=gold_issues,
    )


def load_gold_mappings(
    scenario_path: Path,
    scenario: str,
    tables: dict[str, Table],
    source_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
) -> tuple[list[GoldMapping], list[GoldIssue]]:
    source_by_id = {str(record.get("id")): record for record in source_records}
    target_by_uri = {normalize_uri(record.get("uri")): record for record in target_records}
    gold_by_source: dict[str, set[str]] = {}
    query_ids_by_source: dict[str, set[str]] = {}
    issues: list[GoldIssue] = []

    for qpair_path in sorted((scenario_path / "queries").glob("*.qpair")):
        qpair = parse_qpair(qpair_path)
        if qpair.get("disabled"):
            continue
        targets, unknown_targets = sparql_target_uris(str(qpair.get("sparql", "")), target_by_uri)
        for uri, kind in unknown_targets:
            issues.append(
                GoldIssue(
                    scenario=scenario,
                    query_id=str(qpair.get("id", qpair_path.stem)),
                    target_uri=uri,
                    target_kind=kind,
                    failure_category="target_uri_normalization_error",
                    reason="SPARQL target URI was not found in the parsed target ontology",
                )
            )
        usage = sql_usage(str(qpair.get("sql", "")), tables)
        for target_uri, target_kind in targets:
            source_ids = candidate_source_ids_for_target_kind(target_kind, usage, tables, source_by_id)
            if not source_ids:
                issues.append(
                    GoldIssue(
                        scenario=scenario,
                        query_id=str(qpair.get("id", qpair_path.stem)),
                        target_uri=target_uri,
                        target_kind=target_kind,
                        failure_category="source_not_in_gold",
                        reason="No source record could be aligned from the qpair SQL usage",
                    )
                )
                continue
            for source_id in source_ids:
                gold_by_source.setdefault(source_id, set()).add(target_uri)
                query_ids_by_source.setdefault(source_id, set()).add(str(qpair.get("id", qpair_path.stem)))

    mappings: list[GoldMapping] = []
    for source_id, target_uris in sorted(gold_by_source.items()):
        source = source_by_id[source_id]
        mappings.append(
            GoldMapping(
                scenario=scenario,
                source_id=source_id,
                source_kind=str(source.get("kind", "")),
                source_table=str(source.get("source_table", "")),
                source_column=str(source.get("source_column", "")),
                source_table_role=str(source.get("source_table_role") or source.get("table_role") or ""),
                source_column_role=str(source.get("source_column_role", "")),
                gold_target_uris=tuple(sorted(target_uris)),
                query_ids=tuple(sorted(query_ids_by_source.get(source_id, set()))),
            )
        )
    return mappings, issues


@dataclass(frozen=True)
class SqlUsage:
    tables: tuple[str, ...]
    columns: tuple[tuple[str, str], ...]


def sql_usage(sql: str, tables: dict[str, Table]) -> SqlUsage:
    used_tables, aliases = _sql_tables_and_aliases(sql, tables)
    columns = _sql_columns(sql, used_tables, aliases, tables)
    return SqlUsage(tables=tuple(sorted(used_tables)), columns=tuple(sorted(columns)))


def _clean_identifier(value: str) -> str:
    value = value.strip().strip('"')
    if "." in value:
        value = value.split(".")[-1].strip('"')
    return value


def _sql_tables_and_aliases(sql: str, tables: dict[str, Table]) -> tuple[set[str], dict[str, str]]:
    known = {name.lower(): name for name in tables}
    used: set[str] = set()
    aliases: dict[str, str] = {}
    table_re = re.compile(
        r"\b(?:FROM|JOIN)\s+(?:(?:\"[^\"]+\"|[A-Za-z_][\w$]*)\.)?(\"[^\"]+\"|[A-Za-z_][\w$]*)(?:\s+(?:AS\s+)?(\"[^\"]+\"|[A-Za-z_][\w$]*))?",
        flags=re.I,
    )
    for match in table_re.finditer(sql):
        raw_table = _clean_identifier(match.group(1))
        table_name = known.get(raw_table.lower())
        if not table_name:
            continue
        used.add(table_name)
        aliases[table_name.lower()] = table_name
        alias = _clean_identifier(match.group(2) or "")
        if alias and alias.lower() not in SQL_KEYWORDS:
            aliases[alias.lower()] = table_name
    return used, aliases


def _sql_columns(
    sql: str,
    used_tables: set[str],
    aliases: dict[str, str],
    tables: dict[str, Table],
) -> set[tuple[str, str]]:
    columns: set[tuple[str, str]] = set()
    qualified_re = re.compile(r'(?:"([^"]+)"|([A-Za-z_][\w$]*))\s*\.\s*(?:"([^"]+)"|([A-Za-z_][\w$]*))')
    for match in qualified_re.finditer(sql):
        qualifier = _clean_identifier(match.group(1) or match.group(2) or "")
        column = _clean_identifier(match.group(3) or match.group(4) or "")
        table_name = aliases.get(qualifier.lower())
        if table_name and column in {col.name for col in tables[table_name].columns}:
            columns.add((table_name, column))

    if len(used_tables) == 1:
        table_name = next(iter(used_tables))
        table_columns = {col.name for col in tables[table_name].columns}
        for token in re.findall(r'(?:"([^"]+)"|(?<![.])\b([A-Za-z_][\w$]*)\b)', sql):
            word = _clean_identifier(token[0] or token[1] or "")
            if word in table_columns:
                columns.add((table_name, word))
    else:
        column_to_tables: dict[str, list[str]] = {}
        for table_name in used_tables:
            for column in tables[table_name].columns:
                column_to_tables.setdefault(column.name.lower(), []).append(table_name)
        for word in re.findall(r"(?<![.])\b([A-Za-z_][\w$]*)\b", sql):
            if word.lower() in SQL_KEYWORDS:
                continue
            owners = column_to_tables.get(word.lower(), [])
            if len(owners) == 1:
                columns.add((owners[0], next(col.name for col in tables[owners[0]].columns if col.name.lower() == word.lower())))
    return columns


def candidate_source_ids_for_target_kind(
    target_kind: str,
    usage: SqlUsage,
    tables: dict[str, Table],
    source_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    source_kind = SOURCE_KIND_BY_TARGET_KIND.get(target_kind)
    if source_kind == "class":
        return [source_id for table in usage.tables if (source_id := f"source-class:{table}") in source_by_id]
    if source_kind == "data_property":
        out = []
        for table, column in usage.columns:
            source_id = f"source-data:{table}.{column}"
            if source_id in source_by_id:
                out.append(source_id)
        return sorted(set(out))
    if source_kind == "object_property":
        out: list[str] = []
        used_tables = set(usage.tables)
        for table_name in usage.tables:
            table = tables.get(table_name)
            if not table:
                continue
            for fk in table.foreign_keys:
                if used_tables and fk.ref_table not in used_tables and len(used_tables) > 1:
                    continue
                source_id = f"source-object:{table_name}.{','.join(fk.columns)}"
                if source_id in source_by_id:
                    out.append(source_id)
        return sorted(set(out))
    return []


def source_column_role(source: dict[str, Any], tables: dict[str, Table]) -> str:
    kind = str(source.get("kind", ""))
    table_name = str(source.get("source_table", ""))
    column = str(source.get("source_column", ""))
    table = tables.get(table_name)
    if not table or kind == "class" or not column:
        return ""
    if kind == "object_property":
        return "foreign_key"
    if "," in column:
        return "foreign_key"
    if column in table.primary_key:
        return "primary_key"
    if column in foreign_key_columns(table):
        return "foreign_key"
    if is_generic_identifier_column(table, column):
        return "generic_identifier"
    return "attribute"


def table_roles(tables: dict[str, Table]) -> dict[str, str]:
    return {name: table_role(table) for name, table in tables.items()}
