"""Summarize diagnostic heuristics from existing evaluation output."""
from __future__ import annotations
import argparse
import csv
import json
import math
import re
from pathlib import Path

def classify(row):
    sql, sparql = row["sql_count"], row["sparql_count"]
    recall, precision = row["recall"], row["precision"]
    category = row["categories"]
    if (sql == 0 and sparql == 0) or (precision == 1.0 and recall == 1.0):
        return None
    if sparql == 0 and sql > 0:
        return "zero_sparql_answers"
    if sql == 0 and sparql > 0:
        return "overgenerated_sparql_answers"
    if recall < 1.0 and ("object" in category.lower() or "ref" in re.split(r"[,;\s]+", category.lower())):
        return "missing_or_wrong_link_path"
    if recall < 1.0:
        return "missing_attribute_or_class"
    if precision < 1.0 and sparql > sql:
        return "overgenerated_sparql_answers"
    return "other"

def normalized_row(raw, scenario, is_json):
    required = ("precision", "recall", "sql_results", "sparql_results") if is_json else (
        "precision", "recall", "sql_count", "sparql_count")
    missing = [name for name in required if name not in raw or raw[name] is None or raw[name] == ""]
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))
    if "categories" not in raw and "category" not in raw:
        raise ValueError("Missing required field: categories (or category)")
    row = {"scenario": raw.get("scenario") or scenario}
    if not row["scenario"]:
        raise ValueError("Cannot determine scenario; pass --scenario or use runs/<scenario>/eval/metrics_details.json")
    for key in ("precision", "recall"):
        value = float(raw[key])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be a finite number between 0 and 1")
        row[key] = value
    for count, results in (("sql_count", "sql_results"), ("sparql_count", "sparql_results")):
        if is_json:
            if not isinstance(raw[results], list):
                raise ValueError(f"{results} must be an array")
            row[count] = len(raw[results])
        else:
            value = float(raw[count])
            if not math.isfinite(value) or value < 0 or not value.is_integer():
                raise ValueError(f"{count} must be a nonnegative integer")
            row[count] = int(value)
    category = raw.get("categories", raw.get("category"))
    row["categories"] = ",".join(map(str, category)) if isinstance(category, list) else str(category)
    return row

def summarize(metrics: Path, output: Path, scenario=None):
    if not metrics.is_file():
        raise ValueError(f"Metrics file not found: {metrics}")
    inferred = metrics.parent.parent.name if metrics.parent.name == "eval" and metrics.parent.parent.parent.name == "runs" else None
    is_json = metrics.suffix.lower() == ".json"
    if is_json:
        if scenario and inferred and scenario != inferred:
            raise ValueError("--scenario conflicts with the scenario output path")
        raw_rows = json.loads(metrics.read_text(encoding="utf-8"))
        if not isinstance(raw_rows, list):
            raise ValueError("Evaluation JSON must contain an array of query results")
        if not scenario and not inferred and (not raw_rows or not all(isinstance(row, dict) and row.get("scenario") for row in raw_rows)):
            raise ValueError("Cannot determine scenario; pass --scenario")
    elif metrics.suffix.lower() == ".csv":
        with metrics.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fields = set(reader.fieldnames or [])
            if not {"precision", "recall", "sql_count", "sparql_count"} <= fields or not fields & {"categories", "category"}:
                raise ValueError("CSV lacks required evaluation fields")
            raw_rows = list(reader)
    else:
        raise ValueError("--metrics must be an evaluation .json or .csv file")
    if not raw_rows and not scenario and not inferred:
        raise ValueError("Cannot determine scenario; pass --scenario")
    counts = {}
    for index, raw in enumerate(raw_rows, 1):
        if not isinstance(raw, dict):
            raise ValueError(f"Metrics row {index} must be an object")
        try:
            row = normalized_row(raw, scenario or inferred, is_json)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Metrics row {index}: {exc}") from exc
        if scenario and row["scenario"] != scenario:
            continue
        category = classify(row)
        if category:
            key = row["scenario"], category
            counts[key] = counts.get(key, 0) + 1
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scenario", "failure_category", "count"])
        writer.writeheader()
        writer.writerows({"scenario": s, "failure_category": c, "count": n}
                         for (s, c), n in sorted(counts.items()))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", default="/outputs/failure_modes_summary.csv")
    parser.add_argument("--scenario")
    args = parser.parse_args()
    try:
        summarize(Path(args.metrics), Path(args.output), args.scenario)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output}")

if __name__ == "__main__":
    main()
