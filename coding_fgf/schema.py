from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Column:
    name: str
    datatype: str = "text"
    nullable: bool = True


@dataclass(frozen=True)
class ForeignKey:
    columns: list[str]
    ref_table: str
    ref_columns: list[str]


@dataclass
class Table:
    name: str
    columns: list[Column] = field(default_factory=list)
    primary_key: list[str] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


@dataclass
class SqlData:
    rows: dict[str, list[dict[str, Optional[str]]]] = field(default_factory=dict)


def foreign_key_columns(table: Table) -> set[str]:
    return {column for fk in table.foreign_keys for column in fk.columns}


def table_role(table: Table) -> str:
    column_names = set(table.column_names())
    fk_columns = foreign_key_columns(table)
    non_fk_columns = column_names - fk_columns
    pk_columns = set(table.primary_key)
    if len(table.primary_key) >= 2 and pk_columns == column_names:
        return "join_table"
    if table.primary_key and set(table.primary_key).issubset(fk_columns):
        return "subtype_table"
    if len(table.foreign_keys) >= 2 and non_fk_columns <= pk_columns and len(column_names) <= len(fk_columns) + 1:
        return "join_table"
    return "entity_table"


def is_generic_identifier_column(table: Table, column: str) -> bool:
    lowered = column.lower()
    if lowered == "id":
        return True
    if column in table.primary_key and lowered.endswith("_id"):
        return True
    if column in foreign_key_columns(table):
        return True
    return False


def _clean_identifier(value: str) -> str:
    value = value.strip().rstrip(",")
    if "." in value:
        value = value.split(".")[-1]
    return value.strip().strip('"')


def _split_csv_identifiers(value: str) -> list[str]:
    return [_clean_identifier(part) for part in value.split(",") if part.strip()]


def parse_sql_dump(path: Path) -> dict[str, Table]:
    text = path.read_text(encoding="utf-8", errors="replace")
    tables: dict[str, Table] = {}

    ident_re = r'(?:"([^"]+)"|([\w]+))'
    qualified_ident_re = rf"(?:(?:\"[^\"]+\"|[\w]+)\.)?{ident_re}"
    create_re = re.compile(rf"CREATE\s+TABLE\s+(?:ONLY\s+)?{qualified_ident_re}\s*\((.*?)\);", re.I | re.S)
    for match in create_re.finditer(text):
        table = Table(name=_clean_identifier(match.group(1) or match.group(2) or ""))
        for raw_line in match.group(3).splitlines():
            line = raw_line.strip().rstrip(",")
            if not line or line.upper().startswith(("CONSTRAINT", "PRIMARY KEY", "FOREIGN KEY", "UNIQUE", "CHECK")):
                pk_match = re.search(r"PRIMARY\s+KEY\s*\(([^)]+)\)", line, re.I)
                if pk_match:
                    table.primary_key = _split_csv_identifiers(pk_match.group(1))
                continue
            col_match = re.match(r'(?:"([^"]+)"|([\w]+))\s+(.+)$', line)
            if not col_match:
                continue
            name = _clean_identifier(col_match.group(1) or col_match.group(2) or "")
            rest = col_match.group(3).strip()
            datatype = re.split(r"\s+(?:NOT\s+NULL|NULL|DEFAULT|COLLATE|CONSTRAINT|PRIMARY\s+KEY)\b", rest, maxsplit=1, flags=re.I)[0]
            nullable = "NOT NULL" not in rest.upper()
            table.columns.append(Column(name=name, datatype=datatype.strip(), nullable=nullable))
            if re.search(r"\bPRIMARY\s+KEY\b", rest, re.I):
                table.primary_key = [name]
        tables[table.name] = table

    pk_re = re.compile(rf"ALTER\s+TABLE\s+(?:ONLY\s+)?{qualified_ident_re}[^;]*PRIMARY\s+KEY\s*\(([^)]+)\)", re.I)
    for match in pk_re.finditer(text):
        table = tables.get(_clean_identifier(match.group(1) or match.group(2) or ""))
        if table:
            table.primary_key = _split_csv_identifiers(match.group(3))

    fk_re = re.compile(
        rf"ALTER\s+TABLE\s+(?:ONLY\s+)?{qualified_ident_re}[^;]*"
        rf"FOREIGN\s+KEY\s*\(([^)]+)\)\s+REFERENCES\s+(?:(?:\"[^\"]+\"|[\w]+)\.)?{ident_re}\s*\(([^)]+)\)",
        re.I,
    )
    for match in fk_re.finditer(text):
        table = tables.get(_clean_identifier(match.group(1) or match.group(2) or ""))
        if table:
            table.foreign_keys.append(
                ForeignKey(
                    columns=_split_csv_identifiers(match.group(3)),
                    ref_table=_clean_identifier(match.group(4) or match.group(5) or ""),
                    ref_columns=_split_csv_identifiers(match.group(6)),
                )
            )

    return tables


def parse_copy_data(path: Path) -> SqlData:
    rows: dict[str, list[dict[str, Optional[str]]]] = {}
    copy_re = re.compile(r'COPY\s+(?:(?:"[^"]+"|[\w]+)\.)?(?:"([^"]+)"|([\w]+))\s*\(([^)]+)\)\s+FROM\s+stdin;', re.I)
    current_table: Optional[str] = None
    current_columns: list[str] = []

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n\r")
            if current_table is None:
                match = copy_re.match(line)
                if match:
                    current_table = _clean_identifier(match.group(1) or match.group(2) or "")
                    current_columns = _split_csv_identifiers(match.group(3))
                    rows.setdefault(current_table, [])
                continue
            if line == r"\.":
                current_table = None
                current_columns = []
                continue
            values = line.split("\t")
            row: dict[str, Optional[str]] = {}
            for col, value in zip(current_columns, values):
                row[col] = None if value == r"\N" else value
            rows[current_table].append(row)

    return SqlData(rows=rows)


def key_tuple(row: dict[str, Optional[str]], columns: list[str]) -> tuple[Optional[str], ...]:
    return tuple(row.get(col) for col in columns)


def table_key(table: Table) -> list[str]:
    if table.primary_key:
        return table.primary_key
    if table.columns:
        return [table.columns[0].name]
    return []


def find_ontology_file(scenario_dir: Path) -> Path:
    for name in ("ontology.ttl", "ontology.owl", "ontology.rdf"):
        candidate = scenario_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No ontology file found in {scenario_dir}")
