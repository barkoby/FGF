from __future__ import annotations

from pathlib import Path

from .constants import SOURCE_BASE
from .io import ensure_dir
from .schema import Table, table_key


def turtle_literal(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def generate_source_r2rml(tables: dict[str, Table], scenario: str, output_path: Path, db_schema: str = "") -> Path:
    ensure_dir(output_path.parent)
    lines = [
        "@prefix rr: <http://www.w3.org/ns/r2rml#> .",
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        f"@prefix src: <{SOURCE_BASE}{scenario}#> .",
        "",
    ]
    for table in sorted(tables.values(), key=lambda t: t.name):
        pk = table_key(table)
        template_key = pk[0] if pk else (table.columns[0].name if table.columns else "id")
        tm = f"<#TriplesMap_{table.name}>"
        lines.extend(
            [
                f"{tm} a rr:TriplesMap ;",
                (
                    "  rr:logicalTable [ "
                    f"rr:sqlQuery {turtle_literal('SELECT * FROM ' + _sql_identifier(db_schema) + '.' + _sql_identifier(table.name))} "
                    "] ;"
                    if db_schema
                    else f"  rr:logicalTable [ rr:tableName {turtle_literal(table.name)} ] ;"
                ),
                "  rr:subjectMap [",
                f"    rr:template {turtle_literal(SOURCE_BASE + scenario + '/' + table.name + '/{' + template_key + '}')} ;",
                f"    rr:class src:{table.name}",
                "  ] ;",
            ]
        )
        predicate_maps: list[str] = []
        for col in table.columns:
            predicate_maps.append(
                "  rr:predicateObjectMap [ "
                f"rr:predicate src:{table.name}_{col.name} ; "
                f"rr:objectMap [ rr:column {turtle_literal(col.name)} ] "
                "]"
            )
        for fk in table.foreign_keys:
            if not fk.columns or not fk.ref_columns:
                continue
            predicate_maps.append(
                "  rr:predicateObjectMap [ "
                f"rr:predicate src:{table.name}_{'_'.join(fk.columns)}_to_{fk.ref_table} ; "
                f"rr:objectMap [ rr:parentTriplesMap <#TriplesMap_{fk.ref_table}> ; "
                f"rr:joinCondition [ rr:child {turtle_literal(fk.columns[0])} ; rr:parent {turtle_literal(fk.ref_columns[0])} ] ] "
                "]"
            )
        lines.append(" ;\n".join(predicate_maps) + " .")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_morph_config(
    mapping_path: Path,
    output_path: Path,
    db_url: str,
    db_user: str,
    db_password: str,
    config_path: Path,
) -> Path:
    ensure_dir(config_path.parent)
    config = "\n".join(
        [
            "[DataSource1]",
            f"mappings={mapping_path.as_posix()}",
            f"db_url={db_url}",
            f"db_user={db_user}",
            f"db_password={db_password}",
            "db_type=postgresql",
            "",
            "[CONFIGURATION]",
            f"output_file={output_path.as_posix()}",
            "output_format=N-TRIPLES",
            "",
        ]
    )
    config_path.write_text(config, encoding="utf-8")
    return config_path


def materialize_with_morphkgc(config_path: Path) -> bool:
    try:
        import morph_kgc  # type: ignore
    except ImportError:
        return False
    graph = morph_kgc.materialize(str(config_path))
    output = None
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("output_file="):
            output = Path(line.split("=", 1)[1])
            break
    if output is not None:
        ensure_dir(output.parent)
        graph.serialize(destination=str(output), format="turtle")
    return True
