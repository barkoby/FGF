"""Post-selection evaluation of frozen portfolio artifacts."""
from pathlib import Path
from .evaluate import evaluate_graph
from .io import write_csv

FIELDS = ["scenario", "arm", "selected", "status", "precision", "recall", "f1", "query_count"]

def write_fol_ablation_report(scenario, records, selected_arm, qpair_dir, sql_executor, output_dir):
    output_dir = Path(output_dir)
    rows = []
    for record in records:
        row = dict.fromkeys(FIELDS, "")
        row.update(scenario=scenario, arm=record["arm"], selected=record["arm"] == selected_arm)
        path = Path(record["work"]) / "import.ttl"
        if record.get("hard_rejections") or not path.exists():
            row["status"] = "failed"
        elif sql_executor is None:
            row["status"] = "evaluation_skipped"
        else:
            try:
                result = evaluate_graph(path, qpair_dir, sql_executor, output_dir / record["arm"] / "eval")
                row.update(status="success", precision=result["precision"], recall=result["recall"],
                           f1=result["f1"], query_count=result["count"])
            except Exception as exc:
                row["status"] = "evaluation_error:" + type(exc).__name__
        rows.append(row)
    selected = next((row for row in rows if row["selected"]), None)
    if selected:
        rows.append({**selected, "arm": "portfolio_selected"})
    write_csv(output_dir / "fol_selection_comparison.csv", rows, FIELDS)
    return rows
