from __future__ import annotations

import argparse
import csv
from pathlib import Path


def classify(row: dict[str, str]) -> str:
    sql = int(float(row.get("sql_count") or 0))
    sparql = int(float(row.get("sparql_count") or 0))
    recall = float(row.get("recall") or 0.0)
    precision = float(row.get("precision") or 0.0)
    category = row.get("category", "")
    if sparql == 0 and sql > 0:
        return "zero_sparql_answers"
    if recall < 1.0 and "object" in category:
        return "missing_or_wrong_link_path"
    if recall < 1.0:
        return "missing_attribute_or_class"
    if precision < 1.0 and sparql > sql:
        return "overgenerated_sparql_answers"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize missed SQL-vs-SPARQL answers from per-qpair metrics.")
    parser.add_argument("--metrics", default="/outputs/per_qpair_metrics.csv")
    parser.add_argument("--output", default="/outputs/failure_modes_summary.csv")
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()
    in_path = Path(args.metrics)
    if not in_path.exists():
        raise SystemExit(f"Metrics file not found: {in_path}")
    counts: dict[tuple[str, str], int] = {}
    with in_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if args.scenario and row.get("scenario") != args.scenario:
                continue
            key = (row.get("scenario", ""), classify(row))
            counts[key] = counts.get(key, 0) + 1
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scenario", "failure_category", "count"])
        writer.writeheader()
        for (scenario, category), count in sorted(counts.items()):
            writer.writerow({"scenario": scenario, "failure_category": category, "count": count})
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
