from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .compare import emitted_triple_count, no_match_rate, read_summary_csv
from .constants import PAPER_SCENARIOS, PAPER_TARGET_F1
from .io import ensure_dir, read_json, read_jsonl, write_csv

LEDGER_FIELDS = [
    "run_id",
    "timestamp",
    "scenario",
    "phase",
    "iteration",
    "target_f1",
    "precision",
    "recall",
    "f1",
    "delta_vs_target",
    "beats_target_by_0_01",
    "elapsed_seconds",
    "mode",
    "no_match_rate",
    "emitted_triples",
    "selected_matches",
    "anonymous_target_count",
    "invalid_target_uri_count",
    "join_table_class_matches",
    "generic_id_data_matches",
    "class_rules",
    "data_rules",
    "object_rules",
    "pre_repaired",
    "live_rows",
    "live_fallbacks",
    "live_no_matches",
    "live_elapsed_seconds",
    "estimated_api_cost_usd",
    "estimated_api_cost_source",
    "improvement_notes",
    "artifact_path",
]

SUMMARY_FIELDS = [
    "scenario",
    "target_f1",
    "first_sweep_f1",
    "final_precision",
    "final_recall",
    "final_f1",
    "delta_vs_target",
    "status",
    "passed",
    "runs",
    "improvement_iterations",
    "total_elapsed_seconds",
    "total_live_match_seconds",
    "total_estimated_api_cost_usd",
    "final_no_match_rate",
    "final_selected_matches",
    "final_class_rules",
    "final_data_rules",
    "final_object_rules",
    "anonymous_target_count",
    "invalid_target_uri_count",
    "join_table_class_matches",
    "generic_id_data_matches",
    "failed_qpairs",
    "top_failure_cause",
    "improvement_notes",
    "final_artifact_path",
]


def parse_match_event_stats(events: list[str]) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "pre_repaired": None,
        "live_rows": None,
        "live_fallbacks": None,
        "live_no_matches": None,
        "live_elapsed_seconds": None,
    }
    for event in events:
        pre = re.match(r"match:pre_repair:repaired=(\d+):live=(\d+):total=(\d+)$", event)
        if pre:
            stats["pre_repaired"] = int(pre.group(1))
            stats["live_rows"] = int(pre.group(2))
            continue
        complete = re.match(r"match:complete:completed=(\d+):total=(\d+):elapsed=([0-9.]+):fallbacks=(\d+):no_matches=(\d+)$", event)
        if complete:
            stats["live_rows"] = int(complete.group(2))
            stats["live_elapsed_seconds"] = float(complete.group(3))
            stats["live_fallbacks"] = int(complete.group(4))
            stats["live_no_matches"] = int(complete.group(5))
    return stats


def scenario_metadata(run_root: Path, scenario: str) -> dict[str, Any]:
    metadata_path = run_root / "run_metadata.json"
    if not metadata_path.exists():
        return {}
    metadata = read_json(metadata_path)
    for row in metadata.get("scenario_results", []):
        if row.get("scenario") == scenario:
            return row
    return {}


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _cost_setting(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return float(default)


def _estimate_api_cost(scenario_work: Path, metadata: Mapping[str, Any], event_stats: Mapping[str, Any]) -> tuple[float, str]:
    mode = str(metadata.get("mode", ""))
    if mode == "offline":
        return 0.0, "offline"
    match_cost = _cost_setting("CODING_FGF_EST_MATCH_COST_PER_CALL_USD", "0.001")
    codegen_cost = _cost_setting("CODING_FGF_EST_CODEGEN_COST_PER_CALL_USD", "0.005")
    embedding_cost = _cost_setting("CODING_FGF_EST_EMBEDDING_COST_PER_RECORD_USD", "0.000002")
    live_rows = _optional_int(event_stats.get("live_rows")) or 0
    embedding_records = len(read_jsonl(scenario_work / "target_records.jsonl")) + len(read_jsonl(scenario_work / "source_records.jsonl"))
    cost = live_rows * match_cost + embedding_records * embedding_cost + codegen_cost
    source = (
        "estimated from request counts; "
        f"match_calls={live_rows}@{match_cost}, embedding_records={embedding_records}@{embedding_cost}, codegen_calls=1@{codegen_cost}"
    )
    return round(cost, 6), source


def _next_iteration(benchmark_root: Path, scenario: str) -> int:
    ledger = benchmark_root / "benchmark_ledger.csv"
    if not ledger.exists():
        return 1
    with ledger.open("r", encoding="utf-8", newline="") as fh:
        return 1 + sum(1 for row in csv.DictReader(fh) if row.get("scenario") == scenario)


def build_ledger_record(
    benchmark_root: Path,
    run_root: Path,
    scenario: str,
    elapsed_seconds: float | None = None,
    run_id: str | None = None,
    phase: str = "first_sweep",
    improvement_notes: str = "",
) -> dict[str, Any]:
    scenario_work = run_root / "runs" / scenario
    summary = read_summary_csv(scenario_work / "eval" / "summary.csv") or {}
    diagnostics = json.loads((scenario_work / "mapping_diagnostics.json").read_text(encoding="utf-8")) if (scenario_work / "mapping_diagnostics.json").exists() else {}
    metadata = scenario_metadata(run_root, scenario)
    event_stats = parse_match_event_stats(list(metadata.get("llm_events", []) or []))
    estimated_cost, estimated_cost_source = _estimate_api_cost(scenario_work, metadata, event_stats)
    target = PAPER_TARGET_F1.get(scenario)
    f1 = summary.get("f1")
    delta = None if f1 is None or target is None else f1 - target
    timestamp = datetime.now().replace(microsecond=0).isoformat()
    return {
        "run_id": run_id or timestamp.replace(":", "").replace("-", ""),
        "timestamp": timestamp,
        "scenario": scenario,
        "phase": phase,
        "iteration": _next_iteration(benchmark_root, scenario),
        "target_f1": target,
        "precision": summary.get("precision"),
        "recall": summary.get("recall"),
        "f1": f1,
        "delta_vs_target": delta,
        "beats_target_by_0_01": bool(f1 is not None and target is not None and f1 >= target + 0.01),
        "elapsed_seconds": elapsed_seconds,
        "mode": metadata.get("mode", ""),
        "no_match_rate": no_match_rate(scenario_work),
        "emitted_triples": emitted_triple_count(scenario_work / "import.ttl"),
        "selected_matches": diagnostics.get("selected_matches"),
        "anonymous_target_count": diagnostics.get("anonymous_target_count"),
        "invalid_target_uri_count": diagnostics.get("invalid_target_uri_count"),
        "join_table_class_matches": diagnostics.get("join_table_class_matches"),
        "generic_id_data_matches": diagnostics.get("generic_id_data_matches"),
        "class_rules": diagnostics.get("class_rules"),
        "data_rules": diagnostics.get("data_rules"),
        "object_rules": diagnostics.get("object_rules"),
        "pre_repaired": event_stats.get("pre_repaired"),
        "live_rows": event_stats.get("live_rows"),
        "live_fallbacks": event_stats.get("live_fallbacks"),
        "live_no_matches": event_stats.get("live_no_matches"),
        "live_elapsed_seconds": event_stats.get("live_elapsed_seconds"),
        "estimated_api_cost_usd": estimated_cost,
        "estimated_api_cost_source": estimated_cost_source,
        "improvement_notes": improvement_notes,
        "artifact_path": str(scenario_work),
    }


def _rewrite_ledger_schema(ledger_path: Path) -> None:
    if not ledger_path.exists():
        return
    with ledger_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames == LEDGER_FIELDS:
            return
        rows = list(reader)
    with ledger_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in LEDGER_FIELDS})


def append_ledger_row(ledger_path: Path, record: Mapping[str, Any]) -> None:
    ensure_dir(ledger_path.parent)
    exists = ledger_path.exists()
    _rewrite_ledger_schema(ledger_path)
    with ledger_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: record.get(field, "") for field in LEDGER_FIELDS})


def failed_qpairs(scenario_work: Path) -> list[dict[str, Any]]:
    details_path = scenario_work / "eval" / "metrics_details.json"
    if not details_path.exists():
        return []
    rows = json.loads(details_path.read_text(encoding="utf-8"))
    return [row for row in rows if float(row.get("f1", 0.0)) < 1.0]


def likely_failure_cause(row: Mapping[str, Any]) -> str:
    sql_count = len(row.get("sql_results", []) or [])
    sparql_count = len(row.get("sparql_results", []) or [])
    if sql_count and not sparql_count:
        return "missing RDF class/property/path"
    if sparql_count and not sql_count:
        return "false-positive RDF output"
    if sql_count == sparql_count:
        return "value or literal-normalization mismatch"
    return "partial coverage or extra rows"


def write_scenario_report(report_path: Path, record: Mapping[str, Any]) -> None:
    scenario_work = Path(str(record.get("artifact_path", "")))
    failures = failed_qpairs(scenario_work)
    matches = read_json(scenario_work / "matches.json").get("matches", []) if (scenario_work / "matches.json").exists() else []
    fol = read_json(scenario_work / "fol.json").get("rules", {}) if (scenario_work / "fol.json").exists() else {}
    lines = [
        f"# Benchmark Report: {record.get('scenario')}",
        "",
        f"- F1: {record.get('f1')} (target {record.get('target_f1')}, delta {record.get('delta_vs_target')})",
        f"- Beats target +0.01: {record.get('beats_target_by_0_01')}",
        f"- Mode: {record.get('mode')}",
        f"- No-match rate: {record.get('no_match_rate')}",
        f"- Rules: class={len(fol.get('class', []))}, data={len(fol.get('data', []))}, object={len(fol.get('object', []))}",
        f"- Selected matches: {sum(1 for match in matches if match.get('target_uri'))}",
        "",
        "## Failed QPairs",
        "",
    ]
    if failures:
        lines.extend(
            [
                "| ID | Name | F1 | SQL Rows | SPARQL Rows | Likely Cause |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in failures:
            lines.append(
                f"| {row.get('id')} | {row.get('name')} | {row.get('f1')} | "
                f"{len(row.get('sql_results', []) or [])} | {len(row.get('sparql_results', []) or [])} | {likely_failure_cause(row)} |"
            )
    else:
        lines.append("No failed qpairs.")
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            f"- Anonymous targets: {record.get('anonymous_target_count')}",
            f"- Invalid target URIs: {record.get('invalid_target_uri_count')}",
            f"- Join-table class matches: {record.get('join_table_class_matches')}",
            f"- Generic id data matches: {record.get('generic_id_data_matches')}",
            f"- Pre-repaired/live rows: {record.get('pre_repaired')}/{record.get('live_rows')}",
            f"- Live fallbacks/no-matches: {record.get('live_fallbacks')}/{record.get('live_no_matches')}",
        ]
    )
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def record_benchmark_run(
    benchmark_root: Path,
    run_root: Path,
    scenario: str,
    elapsed_seconds: float | None = None,
    run_id: str | None = None,
    phase: str = "first_sweep",
    improvement_notes: str = "",
) -> dict[str, Any]:
    record = build_ledger_record(
        benchmark_root,
        run_root,
        scenario,
        elapsed_seconds=elapsed_seconds,
        run_id=run_id,
        phase=phase,
        improvement_notes=improvement_notes,
    )
    append_ledger_row(benchmark_root / "benchmark_ledger.csv", record)
    write_scenario_report(benchmark_root / "reports" / f"{scenario}.md", record)
    return record


def read_ledger_rows(benchmark_root: Path) -> list[dict[str, str]]:
    ledger_path = benchmark_root / "benchmark_ledger.csv"
    if not ledger_path.exists():
        return []
    with ledger_path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _scenario_summary_row(scenario: str, rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    target = PAPER_TARGET_F1.get(scenario)
    if not rows:
        row = {field: "" for field in SUMMARY_FIELDS}
        row.update(
            {
                "scenario": scenario,
                "target_f1": target,
                "status": "pending",
                "passed": False,
                "runs": 0,
                "improvement_iterations": 0,
            }
        )
        return row
    first = rows[0]
    final = rows[-1]
    final_f1 = _optional_float(final.get("f1"))
    passed = bool(final_f1 is not None and target is not None and final_f1 >= target + 0.01)
    status = "passed" if passed else ("pending" if final_f1 is None else "blocked")
    scenario_work = Path(str(final.get("artifact_path", "")))
    failures = failed_qpairs(scenario_work) if scenario_work.exists() else []
    cause_counts = Counter(likely_failure_cause(row) for row in failures)
    notes = [str(row.get("improvement_notes", "")).strip() for row in rows if str(row.get("improvement_notes", "")).strip()]
    return {
        "scenario": scenario,
        "target_f1": target,
        "first_sweep_f1": first.get("f1", ""),
        "final_precision": final.get("precision", ""),
        "final_recall": final.get("recall", ""),
        "final_f1": final.get("f1", ""),
        "delta_vs_target": None if final_f1 is None or target is None else final_f1 - target,
        "status": status,
        "passed": passed,
        "runs": len(rows),
        "improvement_iterations": max(0, len(rows) - 1),
        "total_elapsed_seconds": sum(_optional_float(row.get("elapsed_seconds")) or 0.0 for row in rows),
        "total_live_match_seconds": sum(_optional_float(row.get("live_elapsed_seconds")) or 0.0 for row in rows),
        "total_estimated_api_cost_usd": round(sum(_optional_float(row.get("estimated_api_cost_usd")) or 0.0 for row in rows), 6),
        "final_no_match_rate": final.get("no_match_rate", ""),
        "final_selected_matches": final.get("selected_matches", ""),
        "final_class_rules": final.get("class_rules", ""),
        "final_data_rules": final.get("data_rules", ""),
        "final_object_rules": final.get("object_rules", ""),
        "anonymous_target_count": final.get("anonymous_target_count", ""),
        "invalid_target_uri_count": final.get("invalid_target_uri_count", ""),
        "join_table_class_matches": final.get("join_table_class_matches", ""),
        "generic_id_data_matches": final.get("generic_id_data_matches", ""),
        "failed_qpairs": len(failures),
        "top_failure_cause": cause_counts.most_common(1)[0][0] if cause_counts else "",
        "improvement_notes": " | ".join(notes),
        "final_artifact_path": final.get("artifact_path", ""),
    }


def summarize_benchmark(benchmark_root: Path) -> list[dict[str, Any]]:
    rows = read_ledger_rows(benchmark_root)
    by_scenario: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(str(row.get("scenario", "")), []).append(row)
    scenario_order = list(PAPER_SCENARIOS)
    scenario_order.extend(sorted(scenario for scenario in by_scenario if scenario and scenario not in PAPER_SCENARIOS))
    summary_rows = [_scenario_summary_row(scenario, by_scenario.get(scenario, [])) for scenario in scenario_order]
    write_csv(benchmark_root / "benchmark_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_benchmark_summary_markdown(benchmark_root / "benchmark_summary.md", summary_rows)
    return summary_rows


def write_benchmark_summary_markdown(path: Path, rows: list[Mapping[str, Any]]) -> None:
    total_elapsed = sum(_optional_float(row.get("total_elapsed_seconds")) or 0.0 for row in rows)
    total_live = sum(_optional_float(row.get("total_live_match_seconds")) or 0.0 for row in rows)
    total_cost = sum(_optional_float(row.get("total_estimated_api_cost_usd")) or 0.0 for row in rows)
    beaten = [str(row["scenario"]) for row in rows if row.get("status") == "passed"]
    blocked = [str(row["scenario"]) for row in rows if row.get("status") == "blocked"]
    pending = [str(row["scenario"]) for row in rows if row.get("status") == "pending"]
    lines = [
        "# coding_fgf RODI Benchmark Summary",
        "",
        f"- Scenarios beaten: {len(beaten)}",
        f"- Scenarios blocked: {len(blocked)}",
        f"- Scenarios pending: {len(pending)}",
        f"- Total elapsed wall-clock seconds: {total_elapsed:.2f}",
        f"- Total live match seconds: {total_live:.2f}",
        f"- Estimated API cost USD: {total_cost:.6f}",
        "",
        "## Scenario Results",
        "",
        "| Scenario | Target | First F1 | Final F1 | Delta | Status | Iterations | Runtime s | Live s | Est. Cost USD | Failed QPairs | Top Failure Cause |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        values = {field: "" if row.get(field) is None else row.get(field, "") for field in SUMMARY_FIELDS}
        lines.append(
            "| {scenario} | {target_f1} | {first_sweep_f1} | {final_f1} | {delta_vs_target} | {status} | "
            "{improvement_iterations} | {total_elapsed_seconds} | {total_live_match_seconds} | "
            "{total_estimated_api_cost_usd} | {failed_qpairs} | {top_failure_cause} |".format(**values)
        )
    lines.extend(
        [
            "",
            "## Final Lists",
            "",
            f"- Beaten: {', '.join(beaten) if beaten else 'none'}",
            f"- Blocked: {', '.join(blocked) if blocked else 'none'}",
            f"- Pending: {', '.join(pending) if pending else 'none'}",
            "",
            "## Improvement Notes",
            "",
        ]
    )
    for row in rows:
        notes = str(row.get("improvement_notes", "")).strip()
        if notes:
            lines.append(f"- {row.get('scenario')}: {notes}")
    if not any(str(row.get("improvement_notes", "")).strip() for row in rows):
        lines.append("No improvement notes recorded yet.")
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
