from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from .candidate_gold import load_scenario_data, normalize_uri, resolve_scenarios
from .candidate_methods import (
    DEFAULT_METHODS,
    CandidateMethodContext,
    normalized_candidate_artifact,
    rank_method,
    validate_methods,
)
from .candidate_metrics import (
    aggregate_by_role,
    aggregate_by_scenario,
    aggregate_overall,
    aggregate_overall_by_role,
    evaluate_mapping,
    failure_row_from_eval,
    failure_row_from_issue,
    found_field,
)
from .candidate_report import write_plots, write_report
from .constants import DEFAULT_EMBEDDING_MODEL
from .io import ensure_dir, write_csv, write_json, write_jsonl


DEFAULT_K_VALUES = [1, 3, 5, 8, 10, 16, 20]
_PROGRESS_LOG_PATH: Path | None = None


PER_SOURCE_FIELDS = [
    "scenario",
    "method",
    "source_id",
    "source_kind",
    "source_table",
    "source_column",
    "source_table_role",
    "source_column_role",
    "gold_target_uri",
    "rank_of_first_gold",
    "reciprocal_rank",
    "found_at_1",
    "found_at_3",
    "found_at_5",
    "found_at_8",
    "found_at_10",
    "found_at_16",
    "found_at_20",
    "candidate_count",
    "zero_candidates",
    "failure_category",
]

METRIC_FIELDS = [
    "scenario",
    "method",
    "source_kind",
    "k",
    "recall_at_k",
    "mrr",
    "mean_rank_first_gold",
    "candidate_coverage",
    "zero_candidate_rate",
    "number_of_gold_mapped_sources",
    "number_of_sources_evaluated",
]

OVERALL_FIELDS = [field for field in METRIC_FIELDS if field != "scenario"]

ROLE_FIELDS = [
    "scenario",
    "method",
    "source_kind",
    "source_table_role",
    "source_column_role",
    "k",
    "recall_at_k",
    "mrr",
    "mean_rank_first_gold",
    "candidate_coverage",
    "zero_candidate_rate",
    "number_of_gold_mapped_sources",
    "number_of_sources_evaluated",
]

OVERALL_ROLE_FIELDS = [field for field in ROLE_FIELDS if field != "scenario"]

FAILURE_FIELDS = [
    "scenario",
    "method",
    "source_id",
    "source_kind",
    "source_table",
    "source_column",
    "gold_target_uri",
    "top_candidate_uris",
    "failure_category",
    "candidate_count",
    "rank_of_first_gold",
    "reciprocal_rank",
]


def run_candidate_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    methods = validate_methods(args.methods, args.embedding_provider)
    k_values = sorted(set(int(k) for k in args.k_values))
    if not k_values or max(k_values) > 20:
        raise SystemExit("--k-values must include values between 1 and 20")
    output_dir = ensure_dir(Path(args.output_dir))
    cache_dir = ensure_dir(Path(args.cache_dir))
    scenarios = resolve_scenarios(Path(args.rodi_root), bool(args.all_scenarios), args.scenarios)
    configure_progress_log(output_dir / "candidate_eval.log")
    total_method_runs = len(scenarios) * len(methods)
    completed_method_runs = 0
    log_progress(
        "run:start",
        scenarios=len(scenarios),
        methods=",".join(methods),
        k_values=",".join(str(k) for k in k_values),
        output_dir=str(output_dir),
        cache_dir=str(cache_dir),
    )

    per_gold_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    candidate_artifact_rows: list[dict[str, Any]] = []
    method_config: dict[str, Any] | None = None

    for scenario_index, scenario in enumerate(scenarios, start=1):
        scenario_started_at = time.perf_counter()
        log_progress("scenario:start", scenario=scenario, scenario_index=scenario_index, scenario_total=len(scenarios))
        log_progress("scenario:load:start", scenario=scenario)
        data = load_scenario_data(Path(args.rodi_root), scenario)
        log_progress(
            "scenario:load:complete",
            scenario=scenario,
            sources=len(data.source_records),
            targets=len(data.target_records),
            gold_mapped_sources=len(data.gold_mappings),
            gold_issues=len(data.gold_issues),
            elapsed_seconds=elapsed(scenario_started_at),
        )
        target_kind_by_uri = {normalize_uri(record.get("uri")): str(record.get("kind", "")) for record in data.target_records}
        gold_by_source = {mapping.source_id: mapping for mapping in data.gold_mappings}
        log_progress("scenario:context:start", scenario=scenario)

        def context_logger(message: str, scenario_name: str = scenario) -> None:
            event, _, details = message.partition(" ")
            log_progress(event, scenario=scenario_name, details=details)

        context = CandidateMethodContext(
            methods=methods,
            target_records=data.target_records,
            cache_dir=cache_dir,
            embedding_model=args.embedding_model,
            embedding_provider=args.embedding_provider, google_project=args.google_project,
            google_location=args.google_location, google_credentials=args.google_credentials,
            logger=context_logger,
        )
        method_config = context.method_configs(k_values, max(k_values))
        log_progress("scenario:context:complete", scenario=scenario, elapsed_seconds=elapsed(scenario_started_at))

        for method in methods:
            method_started_at = time.perf_counter()
            method_unit_before = len(unit_rows)
            method_failures_before = len(failure_rows)
            method_candidates_before = len(candidate_artifact_rows)
            log_progress(
                "method:start",
                scenario=scenario,
                method=method,
                method_run_index=completed_method_runs + 1,
                method_run_total=total_method_runs,
                sources=len(data.source_records),
            )
            progress_interval = max(1, len(data.source_records) // 10)
            for source_index, source in enumerate(data.source_records, start=1):
                source_id = str(source.get("id", ""))
                mapping = gold_by_source.get(source_id)
                candidates, rank_by_uri = rank_method(context, method, source, max_candidates=max(k_values))
                candidate_artifact_rows.append({**normalized_candidate_artifact(scenario, method, source, candidates),
                    "embedding_provider": args.embedding_provider, "embedding_model": args.embedding_model,
                    "embedding_mode": "live"})
                if mapping:
                    unit_row, gold_rows = evaluate_mapping(
                        mapping,
                        method,
                        candidates,
                        rank_by_uri,
                        k_values,
                        target_kind_by_uri=target_kind_by_uri,
                    )
                    unit_rows.append(unit_row)
                    per_gold_rows.extend(gold_rows)
                    failure = failure_row_from_eval(unit_row, candidates)
                    if failure:
                        failure_rows.append(failure)
                if source_index == len(data.source_records) or source_index % progress_interval == 0:
                    log_progress(
                        "method:progress",
                        scenario=scenario,
                        method=method,
                        sources_done=source_index,
                        sources_total=len(data.source_records),
                        candidate_rows=len(candidate_artifact_rows) - method_candidates_before,
                        evaluated_sources=len(unit_rows) - method_unit_before,
                    )
            for issue in data.gold_issues:
                failure_rows.append(failure_row_from_issue(issue, method))
            completed_method_runs += 1
            log_progress(
                "method:complete",
                scenario=scenario,
                method=method,
                sources=len(data.source_records),
                evaluated_sources=len(unit_rows) - method_unit_before,
                failures=len(failure_rows) - method_failures_before,
                candidate_rows=len(candidate_artifact_rows) - method_candidates_before,
                elapsed_seconds=elapsed(method_started_at),
            )

        log_progress(
            "scenario:complete",
            scenario=scenario,
            scenario_index=scenario_index,
            scenario_total=len(scenarios),
            elapsed_seconds=elapsed(scenario_started_at),
        )

    log_progress("metrics:aggregate:start", evaluated_sources=len(unit_rows), candidate_rows=len(candidate_artifact_rows))
    metrics_by_scenario = aggregate_by_scenario(unit_rows, k_values)
    metrics_overall = aggregate_overall(unit_rows, k_values)
    metrics_by_role = aggregate_by_role(unit_rows, k_values)
    metrics_overall_by_role = aggregate_overall_by_role(unit_rows, k_values)
    log_progress(
        "metrics:aggregate:complete",
        metrics_by_scenario=len(metrics_by_scenario),
        metrics_overall=len(metrics_overall),
        metrics_by_role=len(metrics_by_role),
        metrics_overall_by_role=len(metrics_overall_by_role),
    )

    log_progress("outputs:write:start", output_dir=str(output_dir))
    output_specs = [
        ("per_source_candidate_ranks.csv", lambda: write_csv(output_dir / "per_source_candidate_ranks.csv", _prepare_per_source_rows(per_gold_rows, k_values), PER_SOURCE_FIELDS), len(per_gold_rows)),
        ("metrics_by_scenario.csv", lambda: write_csv(output_dir / "metrics_by_scenario.csv", metrics_by_scenario, METRIC_FIELDS), len(metrics_by_scenario)),
        ("metrics_overall.csv", lambda: write_csv(output_dir / "metrics_overall.csv", metrics_overall, OVERALL_FIELDS), len(metrics_overall)),
        ("metrics_by_role.csv", lambda: write_csv(output_dir / "metrics_by_role.csv", metrics_by_role, ROLE_FIELDS), len(metrics_by_role)),
        ("metrics_overall_by_role.csv", lambda: write_csv(output_dir / "metrics_overall_by_role.csv", metrics_overall_by_role, OVERALL_ROLE_FIELDS), len(metrics_overall_by_role)),
        ("failure_analysis.csv", lambda: write_csv(output_dir / "failure_analysis.csv", failure_rows, FAILURE_FIELDS), len(failure_rows)),
        ("candidates_by_method.jsonl", lambda: write_jsonl(output_dir / "candidates_by_method.jsonl", candidate_artifact_rows), len(candidate_artifact_rows)),
    ]
    for filename, writer, rows in output_specs:
        file_started_at = time.perf_counter()
        log_progress("outputs:file:start", file=filename, rows=rows)
        writer()
        log_progress("outputs:file:complete", file=filename, rows=rows, elapsed_seconds=elapsed(file_started_at))

    log_progress("outputs:file:start", file="method_configs.json", rows=1)
    write_json(
        output_dir / "method_configs.json",
        {
            **(method_config or {}),
            "scenarios": scenarios,
            "scenario_selection": "PAPER_SCENARIOS" if args.all_scenarios or not args.scenarios else "explicit",
            "no_llm_matching_default": True,
        },
    )
    log_progress("outputs:file:complete", file="method_configs.json", rows=1)
    log_progress("report:start", file="report.md")
    write_report(output_dir, metrics_by_scenario, metrics_overall, methods, k_values)
    log_progress("report:complete", file="report.md")
    log_progress("plots:start", dir=str(output_dir / "plots"))
    write_plots(output_dir, metrics_by_scenario, metrics_overall, methods, k_values)
    log_progress("plots:complete", dir=str(output_dir / "plots"))
    log_progress("outputs:write:complete", output_dir=str(output_dir))
    log_progress(
        "run:complete",
        scenarios=len(scenarios),
        methods=len(methods),
        gold_mapped_sources=len({(row["scenario"], row["source_id"]) for row in unit_rows}),
        candidate_rows=len(candidate_artifact_rows),
        elapsed_seconds=elapsed(started_at),
    )
    return {
        "scenarios": scenarios,
        "methods": methods,
        "gold_mapped_sources": len({(row["scenario"], row["source_id"]) for row in unit_rows}),
        "candidate_rows": len(candidate_artifact_rows),
        "output_dir": str(output_dir),
    }


def log_progress(event: str, **fields: Any) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    line = f"{timestamp} candidate_eval:{event}" + (f" {details}" if details else "")
    print(line, flush=True)
    if _PROGRESS_LOG_PATH is not None:
        with _PROGRESS_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def configure_progress_log(path: Path) -> None:
    global _PROGRESS_LOG_PATH
    ensure_dir(path.parent)
    path.write_text("", encoding="utf-8")
    _PROGRESS_LOG_PATH = path


def elapsed(started_at: float) -> str:
    return f"{time.perf_counter() - started_at:.2f}"


def _prepare_per_source_rows(rows: Sequence[dict[str, Any]], k_values: Sequence[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        clean = dict(row)
        for k in DEFAULT_K_VALUES:
            clean.setdefault(found_field(k), "")
        for k in k_values:
            clean[found_field(k)] = row.get(found_field(k), 0)
        if clean.get("rank_of_first_gold") is None:
            clean["rank_of_first_gold"] = ""
        out.append(clean)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m coding_fgf.eval_candidates")
    parser.add_argument("--rodi-root", default="..")
    parser.add_argument("--all-scenarios", action="store_true", help="Evaluate the configured paper RODI scenarios")
    parser.add_argument("--scenarios", nargs="*", default=None, help="Explicit scenario names")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--k-values", nargs="+", type=int, default=list(DEFAULT_K_VALUES))
    parser.add_argument("--output-dir", default="outputs/candidate_eval")
    parser.add_argument("--cache-dir", default=".cache/candidate_embeddings")
    parser.add_argument("--embedding-provider", choices=["openai", "google"], default="openai")
    parser.add_argument("--google-project", default="")
    parser.add_argument("--google-location", default="")
    parser.add_argument("--google-credentials", default="")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--no-llm-matching",
        action="store_true",
        default=True,
        help="Kept for CLI clarity; candidate evaluation never invokes LLM matching.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.no_llm_matching:
        raise SystemExit("LLM matching is not supported by this retrieval-only evaluator")
    summary = run_candidate_evaluation(args)
    print(
        "Wrote candidate evaluation to {output_dir} for {scenario_count} scenarios and {method_count} methods".format(
            output_dir=summary["output_dir"],
            scenario_count=len(summary["scenarios"]),
            method_count=len(summary["methods"]),
        )
    )


if __name__ == "__main__":
    main()
