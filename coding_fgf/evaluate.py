from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

from rdflib import Graph, URIRef

from .io import write_csv

QPAIR_FIELDS = ["name", "orderNum", "sql", "sparql", "entityIdCols", "entityIdVars", "categories", "disabled"]


def clean_qpair_value(value: str) -> str:
    value = value.replace("\\n\\", " ").replace("\\\n", " ")
    value = value.replace("\\n", " ").replace("\\", " ")
    return re.sub(r"\s+", " ", value).strip()


def qpair_section(text: str, name: str, next_names: list[str]) -> str:
    stop = "|".join(re.escape(n) + r"\s*=" for n in next_names)
    match = re.search(rf"(?ms)^\s*{name}\s*=(.*?)(?=^\s*(?:{stop})|\Z)", text)
    return clean_qpair_value(match.group(1)) if match else ""


def parse_qpair(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    name = re.search(r"(?m)^\s*name\s*=(.+)$", text)
    categories = re.search(r"(?m)^\s*categories\s*=(.+)$", text)
    disabled = re.search(r"(?m)^\s*disabled\s*=\s*(.+)$", text)
    return {
        "id": path.stem,
        "name": name.group(1).strip() if name else path.stem,
        "sql": qpair_section(text, "sql", [field for field in QPAIR_FIELDS if field != "sql"]),
        "sparql": qpair_section(text, "sparql", [field for field in QPAIR_FIELDS if field != "sparql"]),
        "categories": [c.strip() for c in (categories.group(1) if categories else "").split(",") if c.strip()],
        "disabled": bool(disabled and disabled.group(1).strip().lower() in {"1", "true", "yes"}),
    }


def normalize_sparql_value(value: Any) -> str:
    if isinstance(value, URIRef):
        return "##iri##"
    return str(value)


def sparql_results(graph: Graph, query: str) -> list[Any]:
    rows = []
    for row in graph.query(query):
        values = [normalize_sparql_value(value) for value in row]
        rows.append(values[0] if len(values) == 1 else values)
    return rows


def calculate_precision_recall_f1(res: list[Any], ref: list[Any]) -> tuple[float, float, float]:
    if ref and isinstance(ref[0], list):
        idx_not_iri: list[int] = []
        if res and res[0]:
            first = res[0]
            if isinstance(first, list):
                idx_not_iri = [idx for idx, value in enumerate(first) if value != "##iri##"]
        if idx_not_iri:
            res = [[item[i] for i in idx_not_iri] for item in res if isinstance(item, list)]
            ref = [[item[i] for i in idx_not_iri] for item in ref if isinstance(item, list)]
        res = [str(item) for item in res if item]
        ref = [str(item) for item in ref if item]
    if not res and not ref:
        return 1.0, 1.0, 1.0
    matched_res = sum(1 for item in res if item in ref)
    matched_ref = sum(1 for item in ref if item in res)
    precision = matched_res / len(res) if res else 0.0
    recall = matched_ref / len(ref) if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def execute_sql_psycopg2(sql: str, dbname: str, host: str, port: int, user: str, password: str) -> list[Any]:
    import psycopg2  # type: ignore

    conn = psycopg2.connect(dbname=dbname, host=host, port=port, user=user, password=password)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
    finally:
        conn.close()
    out = []
    for row in rows:
        values = [str(value) for value in row]
        out.append(values[0] if len(values) == 1 else values)
    return out


def evaluate_graph(
    graph_path: Path,
    qpair_dir: Path,
    sql_executor: Callable[[str], list[Any]],
    output_dir: Path,
) -> dict[str, Any]:
    graph = Graph()
    graph.parse(str(graph_path))
    details: list[dict[str, Any]] = []
    for qpair_path in sorted(qpair_dir.glob("*.qpair")):
        qpair = parse_qpair(qpair_path)
        if qpair.get("disabled"):
            continue
        ref = sql_executor(qpair["sql"])
        res = sparql_results(graph, qpair["sparql"])
        precision, recall, f1 = calculate_precision_recall_f1(res, ref)
        details.append(
            {
                "id": qpair["id"],
                "name": qpair["name"],
                "categories": ",".join(qpair["categories"]),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "sql_results": ref,
                "sparql_results": res,
            }
        )
    avg_p = sum(row["precision"] for row in details) / len(details) if details else 0.0
    avg_r = sum(row["recall"] for row in details) / len(details) if details else 0.0
    avg_f1 = sum(row["f1"] for row in details) / len(details) if details else 0.0
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics_details.json").write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "summary.csv", [{"precision": avg_p, "recall": avg_r, "f1": avg_f1}], ["precision", "recall", "f1"])
    return {"precision": avg_p, "recall": avg_r, "f1": avg_f1, "count": len(details)}
