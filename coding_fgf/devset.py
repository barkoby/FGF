from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from typing import Optional

from .constants import PAPER_SCENARIOS
from .io import ensure_dir, write_csv
from .schema import SqlData, Table, key_tuple, parse_copy_data, parse_sql_dump, table_key


def stable_keep(table: str, key: tuple[Optional[str], ...], fraction: float, seed: str) -> bool:
    raw = f"{seed}|{table}|{'|'.join('' if v is None else str(v) for v in key)}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF) < fraction


def is_join_like(table: Table) -> bool:
    fk_cols = {col for fk in table.foreign_keys for col in fk.columns}
    return bool(fk_cols) and set(table.column_names()).issubset(fk_cols)


def sampling_key(table: Table) -> list[str]:
    return table_key(table) if table.primary_key else table.column_names()


def select_dev_rows(
    tables: dict[str, Table],
    data: SqlData,
    fraction: float = 0.1,
    seed: str = "coding-fgf-dev10-v1",
) -> dict[str, list[dict[str, Optional[str]]]]:
    if fraction == 1.0:
        return {name: [dict(row) for row in data.rows.get(name, [])] for name in tables}
    selected: dict[str, set[tuple[Optional[str], ...]]] = {name: set() for name in tables}
    rows_by_key: dict[str, dict[tuple[Optional[str], ...], dict[str, Optional[str]]]] = {}
    rows_by_columns: dict[tuple[str, tuple[str, ...]], dict[tuple[Optional[str], ...], dict[str, Optional[str]]]] = {}

    def row_lookup(table_name: str, columns: list[str]) -> dict[tuple[Optional[str], ...], dict[str, Optional[str]]]:
        cache_key = (table_name, tuple(columns))
        if cache_key not in rows_by_columns:
            rows_by_columns[cache_key] = {key_tuple(row, columns): row for row in data.rows.get(table_name, [])}
        return rows_by_columns[cache_key]

    def selected_key(table_name: str, row: dict[str, Optional[str]]) -> tuple[Optional[str], ...]:
        return key_tuple(row, sampling_key(tables[table_name]))

    def referenced_selected_key(fk_table: str, ref_columns: list[str], ref_key: tuple[Optional[str], ...]) -> tuple[Optional[str], ...] | None:
        row = row_lookup(fk_table, ref_columns).get(ref_key)
        return selected_key(fk_table, row) if row else None

    for name, table in tables.items():
        pk = sampling_key(table)
        rows_by_key[name] = {key_tuple(row, pk): row for row in data.rows.get(name, [])}
        if is_join_like(table):
            continue
        for row in data.rows.get(name, []):
            key = key_tuple(row, pk)
            if stable_keep(name, key, fraction, seed):
                selected[name].add(key)

    changed = True
    while changed:
        changed = False
        for name, table in tables.items():
            pk = table_key(table)
            for key in list(selected[name]):
                row = rows_by_key.get(name, {}).get(key)
                if not row:
                    continue
                for fk in table.foreign_keys:
                    ref_key = key_tuple(row, fk.columns)
                    if any(v is None for v in ref_key):
                        continue
                    parent_key = referenced_selected_key(fk.ref_table, fk.ref_columns, ref_key)
                    if parent_key is not None and parent_key not in selected.get(fk.ref_table, set()):
                        selected[fk.ref_table].add(parent_key)
                        changed = True

    for name, table in tables.items():
        if not is_join_like(table):
            continue
        pk = sampling_key(table)
        for row in data.rows.get(name, []):
            endpoints_kept = True
            for fk in table.foreign_keys:
                ref_key = key_tuple(row, fk.columns)
                parent_key = referenced_selected_key(fk.ref_table, fk.ref_columns, ref_key)
                if parent_key is None or parent_key not in selected.get(fk.ref_table, set()):
                    endpoints_kept = False
                    break
            if endpoints_kept and stable_keep(name, key_tuple(row, pk), fraction, seed):
                selected[name].add(key_tuple(row, pk))

    return {
        name: [row for row in data.rows.get(name, []) if key_tuple(row, sampling_key(table)) in selected[name]]
        for name, table in tables.items()
    }


def _copy_value(value: Optional[str]) -> str:
    return r"\N" if value is None else str(value)


def write_sampled_dump(original_dump: Path, output_dump: Path, sampled_rows: dict[str, list[dict[str, Optional[str]]]]) -> None:
    copy_re = re.compile(r'COPY\s+(?:(?:"[^"]+"|[\w]+)\.)?(?:"([^"]+)"|([\w]+))\s*\(([^)]+)\)\s+FROM\s+stdin;', re.I)
    lines = original_dump.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        match = copy_re.match(line)
        if not match:
            out.append(line)
            i += 1
            continue
        table = (match.group(1) or match.group(2) or "").strip('"')
        columns = [c.strip().strip('"') for c in match.group(3).split(",")]
        out.append(line)
        for row in sampled_rows.get(table, []):
            if not all(col in row for col in columns):
                continue
            out.append("\t".join(_copy_value(row.get(col)) for col in columns))
        out.append(r"\.")
        i += 1
        while i < len(lines) and lines[i] != r"\.":
            i += 1
        if i < len(lines):
            i += 1
    output_dump.write_text("\n".join(out) + "\n", encoding="utf-8")


def create_devset(
    rodi_root: Path,
    out_dir: Path,
    scenarios: list[str] | None = None,
    fraction: float = 0.1,
    seed: str = "coding-fgf-dev10-v1",
) -> list[dict[str, object]]:
    data_dir = rodi_root / "data"
    scenarios = scenarios or PAPER_SCENARIOS
    ensure_dir(out_dir)
    summary: list[dict[str, object]] = []
    for scenario in scenarios:
        src = data_dir / scenario
        if not src.exists():
            raise FileNotFoundError(f"Missing scenario {scenario} at {src}")
        dst = out_dir / scenario
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("dump.sql", ".DS_Store"))
        tables = parse_sql_dump(src / "dump.sql")
        data = parse_copy_data(src / "dump.sql")
        sampled = select_dev_rows(tables, data, fraction=fraction, seed=seed)
        write_sampled_dump(src / "dump.sql", dst / "dump.sql", sampled)
        for table_name, table in tables.items():
            full = len(data.rows.get(table_name, []))
            dev = len(sampled.get(table_name, []))
            summary.append(
                {
                    "scenario": scenario,
                    "table": table_name,
                    "full_rows": full,
                    "dev_rows": dev,
                    "fraction": fraction,
                    "seed": seed,
                }
            )
    write_csv(out_dir / "devset_summary.csv", summary, ["scenario", "table", "full_rows", "dev_rows", "fraction", "seed"])
    return summary
