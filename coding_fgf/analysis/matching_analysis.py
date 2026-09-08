from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..candidate_gold import load_scenario_data, normalize_uri, resolve_scenarios, scenario_dir
from ..constants import PAPER_SCENARIOS, REQUESTED_MATCH_MODEL, DEFAULT_EMBEDDING_MODEL
from ..providers import structured_generate, is_transient, status_code, RetryExhausted
from ..candidate_methods import normalize_method
from ..io import ensure_dir, read_jsonl, write_csv, write_json, write_jsonl
from .matching_metrics import (
    aggregate_rows,
    aggregate_self_consistency,
    api_error_result,
    attach_gold_and_score,
    confusion_rows,
    invalid_result,
    validate_match_response,
)
from .matching_prompts import (
    MATCHING_METHODS,
    METHOD_CHAIN_OF_VERIFICATION,
    METHOD_COT_PROMPT,
    METHOD_CURRENT_VALIDATED,
    METHOD_SELF_CONSISTENCY,
    PROMPT_VERSIONS,
    match_cot_style_v1,
    match_cov_stage1_v1,
    match_cov_stage2_v1,
    match_current_no_match_review_v1,
    match_current_referee_v1,
    match_current_validated_table_v1,
    match_self_consistency_v1,
    validate_methods,
)
from .plot_matching_heatmaps import write_heatmaps


DEFAULT_CANDIDATE_METHOD = "dense"
DEFAULT_TOP_K = 16
DEFAULT_OUTPUT_DIR = "outputs/matching_analysis"
DEFAULT_CANDIDATE_ARTIFACT = "outputs/candidate_eval/candidates_by_method.jsonl"
DEV10_PREFIX = "fgf_dev10_"

RESULT_FIELDS = [
    "llm_provider", "model_used", "models_used",
    "scenario",
    "method",
    "source_id",
    "source_kind",
    "source_table",
    "source_column",
    "source_table_role",
    "source_column_role",
    "gold_target_uris",
    "predicted_target_uri",
    "predicted_target_id",
    "predicted_target_kind",
    "candidate_rank",
    "confidence",
    "decision",
    "null_category",
    "correct",
    "error_type",
    "reason",
    "api_error",
    "invalid_selection",
    "runtime_seconds",
    "vote_distribution",
    "disagreement_score",
    "all_sampled_decisions",
    "stage1_shortlist",
    "stage1_reasons",
    "selected_from_stage1_rank",
    "initial_selected_target",
    "validation_result",
    "validation_changed",
]

METRIC_FIELDS = [
    "scenario",
    "method",
    "precision",
    "recall",
    "f1",
    "evaluated_sources",
    "gold_mapped_sources",
    "predicted_matches",
    "predicted_nulls",
    "true_positives",
    "false_positives",
    "false_negatives",
    "invalid_selections",
    "api_errors",
    "avg_confidence",
    "avg_candidate_rank",
    "runtime_seconds",
    "all_source_predicted_matches",
    "all_source_predicted_nulls",
    "unlabeled_sources",
    "error_rate",
]

KIND_METRIC_FIELDS = ["scenario", "source_kind", "method"] + [field for field in METRIC_FIELDS if field not in {"scenario", "method"}]
OVERALL_METRIC_FIELDS = ["method"] + [field for field in METRIC_FIELDS if field not in {"scenario", "method"}]

CONFUSION_FIELDS = [
    "scenario",
    "method",
    "selected_correct",
    "selected_wrong",
    "selected_when_gold_null",
    "null_when_gold_match",
    "null_when_gold_null",
    "invalid_selection",
    "api_error",
]

_PROGRESS_LOG_PATH: Path | None = None


@dataclass(frozen=True)
class JsonCallResult:
    data: Mapping[str, Any]
    model_used: str
    usage: Mapping[str, Any]


class ProviderJsonCaller:
    def __init__(self, model, fallback_model="", temperature=0.0, logger=None, *,
                 provider="openai", google_project="", google_location="", google_credentials=""):
        self.model, self.fallback_model = model, fallback_model
        self.temperature, self.logger, self.provider = temperature, logger, provider
        self.google = dict(google_project=google_project, google_location=google_location,
                           google_credentials=google_credentials)

    def call(self, prompt, schema_name):
        models = [self.model] + ([self.fallback_model] if self.fallback_model and self.fallback_model != self.model else [])
        for index, model in enumerate(models):
            try:
                result = structured_generate(prompt, schema_name, model, provider=self.provider,
                    temperature=self.temperature, **self.google,
                    event_logger=(lambda message: self.logger(message)) if self.logger else None)
                return JsonCallResult(result.data, result.model_used, result.usage)
            except Exception as exc:
                # An explicitly requested model fallback is only useful for an unavailable model.
                if index + 1 == len(models) or not (status_code(exc) == 404 or type(exc).__name__ == "NotFoundError"):
                    raise
                if self.logger:
                    self.logger("api:fallback_model", provider=self.provider, model=models[index + 1])
        raise RuntimeError("No configured generation model")

class OpenAIJsonCaller(ProviderJsonCaller):
    """Compatibility adapter for existing OpenAI callers."""

def make_caller(args, temperature):
    provider = getattr(args, "llm_provider", "openai")
    if provider == "openai":
        return OpenAIJsonCaller(args.model, args.fallback_model, temperature, logger=log_progress)
    return ProviderJsonCaller(args.model, args.fallback_model, temperature, logger=log_progress,
        provider=provider, google_project=args.google_project, google_location=args.google_location,
        google_credentials=args.google_credentials)

def stamp_models(row, *results):
    models = list(dict.fromkeys(result.model_used for result in results))
    row["models_used"] = models
    row["model_used"] = ",".join(models)
    return row


def run_candidate_matching(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    methods = validate_methods(args.methods)
    output_dir = ensure_dir(Path(args.output_dir))
    configure_progress_log(output_dir / "matching_analysis.log")
    scenarios = resolve_matching_scenarios(Path(args.rodi_root), bool(args.all_scenarios), args.scenarios, bool(args.dev10_scenarios))
    log_progress(
        "run:start",
        scenarios=len(scenarios),
        methods=",".join(methods),
        candidate_method=args.candidate_method,
        top_k=args.top_k,
        self_consistency_top_k=args.self_consistency_top_k,
        current_group_size=args.current_group_size,
        max_workers=args.max_workers,
        no_api_error_rows=_no_api_error_rows(),
    )
    scenario_rows = load_candidate_artifact_rows(
        Path(args.candidate_artifact),
        Path(args.rodi_root),
        scenarios,
        args.candidate_method,
        args.top_k,
        args.embedding_provider, args.embedding_model,
    )
    results: list[dict[str, Any]] = []
    usage_totals: dict[str, int] = {}
    for scenario in scenarios:
        scenario_started = time.perf_counter()
        rows, gold_by_source = scenario_rows[scenario]
        log_progress("scenario:start", scenario=scenario, sources=len(rows), gold_mapped_sources=len(gold_by_source))
        for method in methods:
            method_started = time.perf_counter()
            method_top_k = args.self_consistency_top_k if method == METHOD_SELF_CONSISTENCY else args.top_k
            log_progress("method:start", scenario=scenario, method=method, sources=len(rows), top_k=method_top_k)
            method_rows = run_method(
                rows,
                gold_by_source,
                method,
                args,
                usage_totals,
            )
            results.extend(method_rows)
            log_progress(
                "method:complete",
                scenario=scenario,
                method=method,
                rows=len(method_rows),
                api_errors=sum(1 for row in method_rows if row.get("api_error")),
                invalid=sum(1 for row in method_rows if row.get("invalid_selection")),
                elapsed_seconds=elapsed(method_started),
            )
        log_progress("scenario:complete", scenario=scenario, elapsed_seconds=elapsed(scenario_started))

    log_progress("metrics:start", rows=len(results))
    metrics_by_dataset = aggregate_rows(results, ["scenario", "method"])
    metrics_by_kind = aggregate_rows(results, ["scenario", "source_kind", "method"])
    metrics_overall = aggregate_rows(results, ["method"])
    confusion = confusion_rows(results)
    log_progress("metrics:complete", by_dataset=len(metrics_by_dataset), by_kind=len(metrics_by_kind), overall=len(metrics_overall))

    write_outputs(output_dir, results, metrics_by_dataset, metrics_by_kind, metrics_overall, confusion, args, scenarios, usage_totals)
    log_progress("run:complete", rows=len(results), elapsed_seconds=elapsed(started))
    return {"rows": len(results), "output_dir": str(output_dir), "scenarios": scenarios, "methods": methods}


def run_method(
    rows: list[dict[str, Any]],
    gold_by_source: Mapping[str, Sequence[str]],
    method: str,
    args: argparse.Namespace,
    usage_totals: dict[str, int],
) -> list[dict[str, Any]]:
    if method == METHOD_CURRENT_VALIDATED:
        raw = run_current_validated(rows, args, usage_totals)
    elif method == METHOD_COT_PROMPT:
        raw = run_single_prompt_method(rows, method, args, usage_totals)
    elif method == METHOD_SELF_CONSISTENCY:
        raw = run_self_consistency(rows, args, usage_totals)
    elif method == METHOD_CHAIN_OF_VERIFICATION:
        raw = run_chain_of_verification(rows, args, usage_totals)
    else:
        raise ValueError(f"Unsupported method {method}")
    return [attach_gold_and_score({**row, "llm_provider": getattr(args, "llm_provider", "openai")}, gold_by_source) for row in raw]


def run_single_prompt_method(
    rows: list[dict[str, Any]],
    method: str,
    args: argparse.Namespace,
    usage_totals: dict[str, int],
) -> list[dict[str, Any]]:
    temperature = temperature_for_method(method, args)
    caller = make_caller(args, temperature)

    def worker(row: dict[str, Any], worker_count: int) -> dict[str, Any]:
        started = time.perf_counter()
        prompt = match_cot_style_v1(row, args.top_k)
        result = caller.call(prompt, "match_cot")
        add_usage(usage_totals, result.usage)
        out = stamp_models(validate_match_response(result.data, row, method), result)
        out["runtime_seconds"] = time.perf_counter() - started
        return out

    return adaptive_map(
        rows,
        worker,
        lambda row, error, attempts, workers: _final_api_row(row, method, error, attempts, workers),
        max_workers=args.max_workers,
        api_retries=args.api_retries,
        logger=log_progress,
        context=f"{rows[0].get('scenario', '')}:{method}" if rows else method,
    )


def run_self_consistency(rows: list[dict[str, Any]], args: argparse.Namespace, usage_totals: dict[str, int]) -> list[dict[str, Any]]:
    caller = make_caller(args, float(args.self_consistency_temperature))

    def worker(row: dict[str, Any], worker_count: int) -> dict[str, Any]:
        started = time.perf_counter()
        samples = []
        prompt_top_k = int(args.self_consistency_top_k)
        allowed_uris = [candidate.get("uri") for candidate in list(row.get("candidates", []) or [])[:prompt_top_k]]
        for sample_index in range(args.self_consistency_samples):
            prompt = match_self_consistency_v1(row, prompt_top_k)
            result = caller.call(prompt, f"match_self_consistency_{sample_index + 1}")
            add_usage(usage_totals, result.usage)
            sample = validate_match_response(result.data, row, METHOD_SELF_CONSISTENCY, allowed_uris=allowed_uris)
            stamp_models(sample, result)
            sample["sample_index"] = sample_index + 1
            samples.append(sample)
        out = aggregate_self_consistency(samples, row)
        out["models_used"] = list(dict.fromkeys(m for sample in samples for m in sample["models_used"]))
        out["model_used"] = ",".join(out["models_used"])
        out["runtime_seconds"] = time.perf_counter() - started
        return out

    return adaptive_map(
        rows,
        worker,
        lambda row, error, attempts, workers: _final_api_row(row, METHOD_SELF_CONSISTENCY, error, attempts, workers),
        max_workers=args.max_workers,
        api_retries=args.api_retries,
        logger=log_progress,
        context=f"{rows[0].get('scenario', '')}:self_consistency" if rows else "self_consistency",
    )


def run_chain_of_verification(rows: list[dict[str, Any]], args: argparse.Namespace, usage_totals: dict[str, int]) -> list[dict[str, Any]]:
    caller = make_caller(args, temperature_for_method(METHOD_CHAIN_OF_VERIFICATION, args))

    def worker(row: dict[str, Any], worker_count: int) -> dict[str, Any]:
        started = time.perf_counter()
        stage1 = caller.call(match_cov_stage1_v1(row, args.top_k), "match_cov_stage1")
        add_usage(usage_totals, stage1.usage)
        stage1_valid, stage1_clean = validate_stage1(stage1.data, row)
        if not stage1_valid:
            out = invalid_result(row, METHOD_CHAIN_OF_VERIFICATION, "invalid_stage1", "Stage 1 shortlist was invalid")
            out["runtime_seconds"] = time.perf_counter() - started
            stamp_models(out, stage1)
            out["stage1_shortlist"] = stage1_clean.get("shortlist", [])
            return out
        allowed = [item["target_uri"] for item in stage1_clean.get("shortlist", [])]
        stage2 = caller.call(match_cov_stage2_v1(row, stage1_clean, args.top_k), "match_cov_stage2")
        add_usage(usage_totals, stage2.usage)
        out = stamp_models(validate_match_response(stage2.data, row, METHOD_CHAIN_OF_VERIFICATION, allowed_uris=allowed), stage1, stage2)
        out["runtime_seconds"] = time.perf_counter() - started
        out["stage1_shortlist"] = stage1_clean.get("shortlist", [])
        out["stage1_reasons"] = {item["target_uri"]: item.get("reasons", []) for item in stage1_clean.get("shortlist", [])}
        if out.get("predicted_target_uri"):
            out["selected_from_stage1_rank"] = next(
                (item.get("candidate_rank") for item in stage1_clean.get("shortlist", []) if item.get("target_uri") == out.get("predicted_target_uri")),
                "",
            )
        return out

    return adaptive_map(
        rows,
        worker,
        lambda row, error, attempts, workers: _final_api_row(row, METHOD_CHAIN_OF_VERIFICATION, error, attempts, workers),
        max_workers=args.max_workers,
        api_retries=args.api_retries,
        logger=log_progress,
        context=f"{rows[0].get('scenario', '')}:chain_of_verification" if rows else "chain_of_verification",
    )


def run_current_validated(rows: list[dict[str, Any]], args: argparse.Namespace, usage_totals: dict[str, int]) -> list[dict[str, Any]]:
    caller = make_caller(args, temperature_for_method(METHOD_CURRENT_VALIDATED, args))
    indexed = list(enumerate(rows))
    results: list[dict[str, Any] | None] = [None] * len(rows)
    current_group_size = max(1, int(args.current_group_size))
    groups = group_candidate_rows(indexed, max_group_size=current_group_size)

    def group_worker(group: list[tuple[int, dict[str, Any]]], worker_count: int) -> dict[int, dict[str, Any]]:
        group_started = time.perf_counter()
        group_rows = [row for _, row in group]
        result = caller.call(match_current_validated_table_v1(group_rows, args.top_k), "match_current_validated")
        add_usage(usage_totals, result.usage)
        by_source = _validated_by_source(result.data, group_rows, METHOD_CURRENT_VALIDATED)
        for item in by_source.values():
            stamp_models(item, result)
        out: dict[int, dict[str, Any]] = {}
        for index, row in group:
            source_id = str(row["source"]["id"])
            item = by_source.get(source_id)
            if item is None:
                item = stamp_models(invalid_result(row, METHOD_CURRENT_VALIDATED, "missing_source_id", "LLM omitted this source or selected an invalid candidate"), result)
            item["runtime_seconds"] = (time.perf_counter() - group_started) / max(1, len(group))
            item["initial_selected_target"] = item.get("predicted_target_uri")
            item["validation_result"] = "initial_valid" if not item.get("invalid_selection") else "initial_invalid"
            item["validation_changed"] = False
            out[index] = item
        return out

    mapped = adaptive_map(
        groups,
        group_worker,
        lambda group, error, attempts, workers: {index: _final_api_row(row, METHOD_CURRENT_VALIDATED, error, attempts, workers) for index, row in group},
        max_workers=args.max_workers,
        api_retries=args.api_retries,
        logger=log_progress,
        context=f"{rows[0].get('scenario', '')}:current_validated" if rows else "current_validated",
    )
    for group_result in mapped:
        for index, result in group_result.items():
            results[index] = result

    review_candidates = [(index, rows[index]) for index, result in enumerate(results) if result and not result.get("predicted_target_uri") and not result.get("api_error")]
    if review_candidates:
        log_progress("method:review:start", method=METHOD_CURRENT_VALIDATED, groups=len(group_candidate_rows(review_candidates, max_group_size=current_group_size)))
        review_groups = group_candidate_rows(review_candidates, max_group_size=current_group_size)

        def review_worker(group: list[tuple[int, dict[str, Any]]], worker_count: int) -> dict[int, dict[str, Any]]:
            group_rows = [row for _, row in group]
            result = caller.call(match_current_no_match_review_v1(group_rows, args.top_k), "match_current_review")
            add_usage(usage_totals, result.usage)
            by_source = _validated_by_source(result.data, group_rows, METHOD_CURRENT_VALIDATED)
            for item in by_source.values():
                stamp_models(item, result)
            out = {}
            for index, row in group:
                candidate = by_source.get(str(row["source"]["id"]))
                if candidate and candidate.get("predicted_target_uri"):
                    candidate["validation_result"] = "review_selected"
                    candidate["validation_changed"] = candidate.get("predicted_target_uri") != results[index].get("predicted_target_uri")  # type: ignore[union-attr]
                    out[index] = candidate
            return out

        for group_result in adaptive_map(review_groups, review_worker, lambda group, error, attempts, workers: {index: _final_api_row(row, METHOD_CURRENT_VALIDATED, f"Review stage: {error}", attempts, workers) for index, row in group}, args.max_workers, args.api_retries, log_progress, f"{rows[0].get('scenario', '')}:current_review"):
            for index, result in group_result.items():
                results[index] = result

    referee_candidates = [(index, rows[index]) for index, result in enumerate(results) if result and (not result.get("predicted_target_uri") or result.get("invalid_selection")) and not result.get("api_error")]
    if referee_candidates:
        log_progress("method:referee:start", method=METHOD_CURRENT_VALIDATED, groups=len(group_candidate_rows(referee_candidates, max_group_size=current_group_size)))
        referee_groups = group_candidate_rows(referee_candidates, max_group_size=current_group_size)

        def referee_worker(group: list[tuple[int, dict[str, Any]]], worker_count: int) -> dict[int, dict[str, Any]]:
            group_rows = [row for _, row in group]
            prior = [_compact_prior(results[index]) for index, _ in group]
            siblings = [_compact_prior(result) for result in results if result and result.get("predicted_target_uri")][:40]
            result = caller.call(match_current_referee_v1(group_rows, prior, siblings, args.top_k), "match_current_referee")
            add_usage(usage_totals, result.usage)
            by_source = _validated_by_source(result.data, group_rows, METHOD_CURRENT_VALIDATED)
            for item in by_source.values():
                stamp_models(item, result)
            out = {}
            for index, row in group:
                candidate = by_source.get(str(row["source"]["id"]))
                if candidate:
                    candidate["validation_result"] = "referee_selected" if candidate.get("predicted_target_uri") else "referee_no_match"
                    candidate["validation_changed"] = candidate.get("predicted_target_uri") != results[index].get("predicted_target_uri")  # type: ignore[union-attr]
                    out[index] = candidate
            return out

        for group_result in adaptive_map(referee_groups, referee_worker, lambda group, error, attempts, workers: {index: _final_api_row(row, METHOD_CURRENT_VALIDATED, f"Referee stage: {error}", attempts, workers) for index, row in group}, args.max_workers, args.api_retries, log_progress, f"{rows[0].get('scenario', '')}:current_referee"):
            for index, result in group_result.items():
                results[index] = result

    return [result if result is not None else api_error_result(row, METHOD_CURRENT_VALIDATED, "No result produced") for result, row in zip(results, rows)]


def adaptive_map(
    tasks: Sequence[Any],
    worker: Callable[[Any, int], Any],
    api_error_builder: Callable[[Any, str, int, int], Any],
    max_workers: int = 4,
    api_retries: int = 2,
    logger: Callable[[str, Any], None] | None = None,
    context: str = "",
) -> list[Any]:
    if api_retries < 0:
        raise ValueError("api_retries must be nonnegative")
    pending = list(enumerate(tasks))
    output = [None] * len(tasks)
    attempts = {index: 0 for index, _ in pending}
    workers = max(1, int(max_workers))
    while pending:
        window, remaining = pending[:workers], pending[workers:]
        retry = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for index, task in window:
                attempts[index] += 1
                futures[executor.submit(worker, task, workers)] = (index, task)
            for future in as_completed(futures):
                index, task = futures[future]
                try:
                    output[index] = future.result()
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    if is_transient(exc) and attempts[index] <= api_retries:
                        retry.append((index, task))
                    elif _no_api_error_rows():
                        raise RuntimeError(f"API task failed in {context} after {attempts[index]} attempts: {message}") from exc
                    else:
                        output[index] = api_error_builder(task, message, attempts[index], workers)
                        if logger:
                            logger("api:error_row", context=context, attempts=attempts[index], error=message)
        if retry:
            next_workers = _next_worker_count(workers)
            if next_workers < workers and logger:
                logger("api:workers_reduce", context=context, from_workers=workers, to_workers=next_workers)
            workers = next_workers
            time.sleep(_api_backoff_seconds(max(attempts[index] for index, _ in retry)))
        pending = sorted(retry) + remaining
    return output


def load_candidate_artifact_rows(
    artifact_path: Path,
    rodi_root: Path,
    scenarios: Sequence[str],
    candidate_method: str,
    top_k: int,
    embedding_provider: str = "openai",
    embedding_model: str | None = None,
) -> dict[str, tuple[list[dict[str, Any]], dict[str, tuple[str, ...]]]]:
    if not artifact_path.exists():
        raise FileNotFoundError(f"Candidate artifact not found: {artifact_path}")
    raw_rows = read_jsonl(artifact_path)
    legacy_config_path = artifact_path.with_name("method_configs.json")
    legacy_config = {}
    if any(row.get("method") == "openai_small" for row in raw_rows) and legacy_config_path.exists():
        legacy_config = json.loads(legacy_config_path.read_text(encoding="utf-8"))
        if not isinstance(legacy_config, dict):
            raise ValueError("Legacy candidate model provenance must be an object in method_configs.json")
    candidate_method = normalize_method(candidate_method, embedding_provider)
    by_key = {}
    for row in raw_rows:
        method = str(row.get("method"))
        row_provider = row.get("embedding_provider")
        legacy = method == "openai_small" and row_provider is None
        if legacy:
            row_provider = legacy_config.get("embedding_provider", "openai")
        normalized = normalize_method(method, row_provider or embedding_provider)
        if normalized != candidate_method:
            continue
        if row_provider != embedding_provider:
            raise ValueError("Candidate artifact embedding provider provenance is missing or incompatible")
        row_model = row.get("embedding_model") or (legacy_config.get("embedding_model") if legacy else None)
        if not row_model:
            raise ValueError("Candidate artifact lacks embedding model provenance; keep method_configs.json beside legacy artifacts")
        if embedding_model and row_model != embedding_model:
            raise ValueError("Candidate artifact embedding model is incompatible")
        if row.get("embedding_mode", "live" if legacy else None) != "live":
            raise ValueError("Candidate artifact is not a provenance-complete live artifact")
        by_key[(str(row.get("scenario")), normalized, str(row.get("source_id")))] = row
    out: dict[str, tuple[list[dict[str, Any]], dict[str, tuple[str, ...]]]] = {}
    for scenario in scenarios:
        artifact_scenario = base_scenario_for_artifact(scenario)
        data = load_scenario_data(rodi_root, scenario)
        source_by_id = {str(record.get("id")): record for record in data.source_records}
        target_by_uri = {normalize_uri(record.get("uri")): record for record in data.target_records}
        gold_by_source = {mapping.source_id: mapping.gold_target_uris for mapping in data.gold_mappings}
        rows = []
        for source_id in sorted(source_by_id):
            artifact = by_key.get((artifact_scenario, candidate_method, source_id))
            if artifact is None:
                raise FileNotFoundError(
                    f"Missing candidate artifact row for scenario={scenario} artifact_scenario={artifact_scenario} "
                    f"method={candidate_method} source={source_id}"
                )
            source = dict(source_by_id[source_id])
            source["source_table_role"] = source.get("source_table_role") or source.get("table_role", "")
            candidates = []
            for candidate in list(artifact.get("candidates", []) or [])[:top_k]:
                merged = dict(target_by_uri.get(normalize_uri(candidate.get("uri")), {}))
                merged.update(candidate)
                candidates.append(merged)
            rows.append({"scenario": scenario, "source": source, "candidates": candidates})
        out[scenario] = (rows, gold_by_source)
    return out


def resolve_matching_scenarios(rodi_root: Path, all_scenarios: bool, scenarios: Iterable[str] | None, dev10_scenarios: bool) -> list[str]:
    selected = [str(s).strip() for s in (scenarios or []) if str(s).strip()]
    if dev10_scenarios:
        if selected:
            for scenario in selected:
                scenario_dir(rodi_root, scenario)
            return selected
        return discover_dev10_scenarios(rodi_root)
    return resolve_scenarios(rodi_root, all_scenarios, selected or None)


def discover_dev10_scenarios(rodi_root: Path) -> list[str]:
    candidates: list[str] = []
    search_roots = [rodi_root / "data", rodi_root]
    for root in search_roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob(f"{DEV10_PREFIX}*")):
            if not path.is_dir():
                continue
            scenario = path.name
            scenario_dir(rodi_root, scenario)
            candidates.append(scenario)
        if candidates:
            break
    if not candidates:
        raise FileNotFoundError(f"Could not find dev10 scenario folders matching {DEV10_PREFIX}* below {rodi_root}")
    return candidates


def base_scenario_for_artifact(scenario: str) -> str:
    return scenario[len(DEV10_PREFIX) :] if scenario.startswith(DEV10_PREFIX) else scenario


def validate_stage1(data: Mapping[str, Any], candidate_row: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    source_id = str(candidate_row.get("source", {}).get("id", ""))
    allowed = {normalize_uri(candidate.get("uri")): candidate for candidate in candidate_row.get("candidates", []) or []}
    if str(data.get("source_id", "")) != source_id:
        return False, {"shortlist": []}
    shortlist = []
    for item in list(data.get("shortlist", []) or [])[:3]:
        uri = normalize_uri(item.get("target_uri"))
        if uri not in allowed:
            return False, {"shortlist": shortlist}
        reasons = list(item.get("reasons", []) or [])
        if len(reasons) != 3:
            return False, {"shortlist": shortlist}
        candidate = allowed[uri]
        shortlist.append(
            {
                "target_uri": uri,
                "target_id": item.get("target_id") or candidate.get("id"),
                "candidate_rank": int(item.get("candidate_rank") or candidate.get("rank") or 0),
                "confidence": float(item.get("confidence") or 0.0),
                "reasons": reasons,
            }
        )
    null_option = data.get("null_option", {}) or {}
    return True, {"source_id": source_id, "shortlist": shortlist, "null_option": null_option}


def _validated_by_source(data: Mapping[str, Any], rows: Sequence[dict[str, Any]], method: str) -> dict[str, dict[str, Any]]:
    out = {}
    for row in rows:
        source_id = str(row["source"]["id"])
        matching_rows = [candidate for candidate in list((data or {}).get("matches", []) or []) if str(candidate.get("source_id", "")) == source_id]
        if matching_rows:
            out[source_id] = validate_match_response({"matches": matching_rows[:1]}, row, method)
    return out


def group_candidate_rows(indexed_rows: Sequence[tuple[int, dict[str, Any]]], max_group_size: int = 12) -> list[list[tuple[int, dict[str, Any]]]]:
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for item in indexed_rows:
        _, row = item
        source = row.get("source", {}) or {}
        key = str(source.get("source_table") or source.get("id") or "unknown")
        grouped.setdefault(key, []).append(item)
    chunks = []
    for key in sorted(grouped):
        rows = grouped[key]
        for start in range(0, len(rows), max_group_size):
            chunks.append(rows[start : start + max_group_size])
    return chunks


def write_outputs(
    output_dir: Path,
    results: list[dict[str, Any]],
    metrics_by_dataset: list[dict[str, Any]],
    metrics_by_kind: list[dict[str, Any]],
    metrics_overall: list[dict[str, Any]],
    confusion: list[dict[str, Any]],
    args: argparse.Namespace,
    scenarios: Sequence[str],
    usage_totals: Mapping[str, int],
) -> None:
    log_progress("outputs:write:start", output_dir=str(output_dir))
    write_jsonl(output_dir / "matching_results.jsonl", [_json_ready(row) for row in results])
    log_progress("outputs:write:file", path=str(output_dir / "matching_results.jsonl"), rows=len(results))
    write_csv(output_dir / "metrics_by_dataset.csv", metrics_by_dataset, METRIC_FIELDS)
    log_progress("outputs:write:file", path=str(output_dir / "metrics_by_dataset.csv"), rows=len(metrics_by_dataset))
    write_csv(output_dir / "metrics_by_kind.csv", metrics_by_kind, KIND_METRIC_FIELDS)
    log_progress("outputs:write:file", path=str(output_dir / "metrics_by_kind.csv"), rows=len(metrics_by_kind))
    write_csv(output_dir / "metrics_overall.csv", metrics_overall, OVERALL_METRIC_FIELDS)
    log_progress("outputs:write:file", path=str(output_dir / "metrics_overall.csv"), rows=len(metrics_overall))
    write_csv(output_dir / "confusion_by_dataset.csv", confusion, CONFUSION_FIELDS)
    log_progress("outputs:write:file", path=str(output_dir / "confusion_by_dataset.csv"), rows=len(confusion))
    write_json(
        output_dir / "method_configs.json",
        {
            "llm_provider": args.llm_provider,
            "model": args.model,
            "fallback_model": args.fallback_model,
            "candidate_method": args.candidate_method,
            "top_k": args.top_k,
            "self_consistency_top_k": args.self_consistency_top_k,
            "current_group_size": args.current_group_size,
            "prompt_versions": PROMPT_VERSIONS,
            "temperature": args.temperature,
            "self_consistency_temperature": args.self_consistency_temperature,
            "self_consistency_samples": args.self_consistency_samples,
            "max_workers": args.max_workers,
            "api_retries": args.api_retries,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "scenarios": list(scenarios),
            "usage_totals": dict(usage_totals),
            "candidate_artifact": str(args.candidate_artifact),
            "cache_dir": str(args.cache_dir),
            "use_warmed_embeddings": bool(args.use_warmed_embeddings),
            "dev10_scenarios": bool(args.dev10_scenarios),
            "no_api_error_rows": _no_api_error_rows(),
        },
    )
    log_progress("outputs:write:file", path=str(output_dir / "method_configs.json"), rows=1)
    write_report(output_dir / "report.md", metrics_by_dataset, metrics_overall, args)
    log_progress("outputs:write:file", path=str(output_dir / "report.md"), rows=1)
    log_progress("heatmaps:start", output_dir=str(output_dir / "heatmaps"))
    write_heatmaps(output_dir, metrics_by_dataset, args.methods)
    log_progress("heatmaps:complete", output_dir=str(output_dir / "heatmaps"))
    log_progress("outputs:write:complete", output_dir=str(output_dir))


def write_report(path: Path, metrics_by_dataset: Sequence[Mapping[str, Any]], metrics_overall: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> None:
    methods = list(args.methods)
    scenarios = sorted({str(row.get("scenario", "")) for row in metrics_by_dataset})
    by_dataset = {(str(row.get("scenario", "")), str(row.get("method", ""))): row for row in metrics_by_dataset}
    by_method = {str(row.get("method", "")): row for row in metrics_overall}
    baseline = by_method.get(METHOD_CURRENT_VALIDATED, {})
    best_by_dataset = []
    for scenario in scenarios:
        scenario_rows = [row for row in metrics_by_dataset if str(row.get("scenario", "")) == scenario]
        best_by_dataset.append(max(scenario_rows, key=lambda row: (float(row.get("f1", 0.0)), float(row.get("recall", 0.0))), default={}))
    current_best = [
        scenario
        for scenario, best_row in zip(scenarios, best_by_dataset)
        if best_row.get("method") == METHOD_CURRENT_VALIDATED
    ]
    lines = [
        "# Matching Analysis",
        "",
        f"Candidate method: `{args.candidate_method}`; top-k: `{args.top_k}`.",
        f"Model: `{args.model}`; fallback: `{args.fallback_model}`.",
        "",
        "## F1 By Dataset",
        "",
        _metric_table(metrics_by_dataset, scenarios, methods, "f1"),
        "",
        "## Precision By Dataset",
        "",
        _metric_table(metrics_by_dataset, scenarios, methods, "precision"),
        "",
        "## Recall By Dataset",
        "",
        _metric_table(metrics_by_dataset, scenarios, methods, "recall"),
        "",
        "## Overall",
        "",
        "| Method | Precision | Recall | F1 | API errors | Invalid selections |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(metrics_overall, key=lambda item: str(item.get("method", ""))):
        lines.append(
            "| `{method}` | {precision:.3f} | {recall:.3f} | {f1:.3f} | {api_errors} | {invalid_selections} |".format(
                method=row.get("method"),
                precision=float(row.get("precision", 0.0)),
                recall=float(row.get("recall", 0.0)),
                f1=float(row.get("f1", 0.0)),
                api_errors=row.get("api_errors", 0),
                invalid_selections=row.get("invalid_selections", 0),
            )
        )
    best = max(metrics_overall, key=lambda row: float(row.get("f1", 0.0)), default={})
    lines.extend(
        [
            "",
            "## Best Method By Dataset",
            "",
            "| Dataset | Best method | F1 | Precision | Recall |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for scenario, row in zip(scenarios, best_by_dataset):
        lines.append(
            "| `{scenario}` | `{method}` | {f1:.3f} | {precision:.3f} | {recall:.3f} |".format(
                scenario=scenario,
                method=row.get("method", ""),
                f1=float(row.get("f1", 0.0)),
                precision=float(row.get("precision", 0.0)),
                recall=float(row.get("recall", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Prompt Comparisons",
            "",
            "| Comparison | Overall F1 delta vs current_validated | Note |",
            "| --- | ---: | --- |",
        ]
    )
    baseline_f1 = float(baseline.get("f1", 0.0))
    for method in methods:
        if method == METHOD_CURRENT_VALIDATED:
            continue
        row = by_method.get(method, {})
        delta = float(row.get("f1", 0.0)) - baseline_f1
        note = "improves" if delta > 0 else "ties" if abs(delta) < 1e-12 else "does not improve"
        lines.append(f"| `{method}` | {delta:.3f} | {note} |")
    lines.extend(
        [
            "",
            "## Failure Signals",
            "",
            "| Method | False positives | False negatives | Invalid selections | API errors |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(metrics_overall, key=lambda item: str(item.get("method", ""))):
        lines.append(
            "| `{method}` | {fp} | {fn} | {invalid} | {api} |".format(
                method=row.get("method"),
                fp=row.get("false_positives", 0),
                fn=row.get("false_negatives", 0),
                invalid=row.get("invalid_selections", 0),
                api=row.get("api_errors", 0),
            )
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Best overall method: `{best.get('method', '')}` with F1 {float(best.get('f1', 0.0)):.3f}.",
            f"- Datasets where `current_validated` is still best: {', '.join(f'`{name}`' for name in current_best) if current_best else 'none'}.",
            f"- `cot_prompt` improvement over baseline: {float(by_method.get(METHOD_COT_PROMPT, {}).get('f1', 0.0)) - baseline_f1:.3f} F1.",
            f"- `self_consistency` improvement over baseline: {float(by_method.get(METHOD_SELF_CONSISTENCY, {}).get('f1', 0.0)) - baseline_f1:.3f} F1.",
            f"- `chain_of_verification` improvement over baseline: {float(by_method.get(METHOD_CHAIN_OF_VERIFICATION, {}).get('f1', 0.0)) - baseline_f1:.3f} F1.",
            "- Primary metrics use gold-mapped sources only; unlabeled/null behavior is reported separately in the CSV metrics.",
            "- API errors are retained as rows and included in metrics unless no-api-error-row mode is enabled for a live resumable run.",
            "- If candidate Recall@16 remains high while F1 is materially lower, the remaining bottleneck is match selection rather than retrieval.",
            f"- Recommended final method by this run: `{best.get('method', '')}`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric_table(rows: Sequence[Mapping[str, Any]], scenarios: Sequence[str], methods: Sequence[str], metric: str) -> str:
    by_key = {(str(row.get("scenario", "")), str(row.get("method", ""))): row for row in rows}
    lines = ["| Dataset | " + " | ".join(f"`{method}`" for method in methods) + " |"]
    lines.append("| --- | " + " | ".join("---:" for _ in methods) + " |")
    for scenario in scenarios:
        values = [float(by_key.get((scenario, method), {}).get(metric, 0.0)) for method in methods]
        lines.append("| `{}` | {} |".format(scenario, " | ".join(f"{value:.3f}" for value in values)))
    return "\n".join(lines)


def temperature_for_method(method: str, args: argparse.Namespace) -> float:
    if method == METHOD_SELF_CONSISTENCY:
        return float(args.self_consistency_temperature)
    return float(args.temperature)


_USAGE_LOCK = threading.Lock()

def add_usage(usage_totals: dict[str, int], usage: Mapping[str, Any]) -> None:
    with _USAGE_LOCK:
        for key, value in usage.items():
            try:
                usage_totals[key] = usage_totals.get(key, 0) + int(value)
            except (TypeError, ValueError):
                continue


def _usage_dict(usage: Any) -> dict[str, int]:
    if isinstance(usage, dict):
        return {str(key): int(value) for key, value in usage.items() if isinstance(value, int)}
    out = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if isinstance(value, int):
            out[key] = value
    return out


def _final_api_row(row: dict[str, Any], method: str, error: str, attempts: int, workers: int) -> dict[str, Any]:
    out = api_error_result(row, method, f"{error}; attempts={attempts}; workers={workers}")
    out["runtime_seconds"] = 0.0
    return out


def _no_api_error_rows() -> bool:
    return os.getenv("CODING_FGF_NO_API_ERROR_ROWS", "").strip().lower() in {"1", "true", "yes"}


def _is_blocking_api_error(message: str) -> bool:
    lowered = message.lower()
    blocking_markers = (
        "openai_api_key is not set",
        "authenticationerror",
        "permissiondeniederror",
        "notfounderror",
        "badrequesterror",
        "error code: 400",
        "unsupported parameter",
        "invalid api key",
        "model_not_found",
    )
    retriable_markers = ("ratelimiterror", "timeout", "apiconnectionerror", "internalservererror", "service unavailable")
    return any(marker in lowered for marker in blocking_markers) and not any(marker in lowered for marker in retriable_markers)


def _api_backoff_seconds(attempts: int) -> float:
    configured = float(os.getenv("CODING_FGF_OPENAI_RATE_LIMIT_BACKOFF_SECONDS", "60"))
    return min(300.0, max(0.0, configured * min(max(1, attempts), 3)))


def _next_worker_count(current_workers: int) -> int:
    if current_workers > 8:
        return 8
    if current_workers > 4:
        return 4
    if current_workers > 2:
        return 2
    return 1


def _compact_prior(row: Mapping[str, Any] | None) -> dict[str, Any]:
    row = row or {}
    return {
        "source_id": row.get("source_id", ""),
        "target_uri": row.get("predicted_target_uri"),
        "candidate_rank": row.get("candidate_rank"),
        "confidence": row.get("confidence", 0.0),
        "decision": row.get("decision", ""),
        "null_category": row.get("null_category", ""),
        "reason": row.get("reason", ""),
        "validation_result": row.get("validation_result", ""),
    }


def _json_ready(row: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for key in RESULT_FIELDS:
        value = row.get(key, "")
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            value = ""
        out[key] = value
    return out


def configure_progress_log(path: Path) -> None:
    global _PROGRESS_LOG_PATH
    ensure_dir(path.parent)
    path.write_text("", encoding="utf-8")
    _PROGRESS_LOG_PATH = path


def log_progress(event: str, **fields: Any) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    line = f"{timestamp} matching_analysis:{event}" + (f" {details}" if details else "")
    print(line, flush=True)
    if _PROGRESS_LOG_PATH is not None:
        with _PROGRESS_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def elapsed(started_at: float) -> str:
    return f"{time.perf_counter() - started_at:.2f}"


def git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True, check=False)
        return result.stdout.strip()
    except Exception:
        return ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m coding_fgf.analysis.matching_analysis")
    parser.add_argument("--rodi-root", default="..")
    parser.add_argument("--all-scenarios", action="store_true", default=False)
    parser.add_argument("--dev10-scenarios", action="store_true", default=False)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--scenario", dest="scenario_one", action="append")
    parser.add_argument("--methods", nargs="+", default=list(MATCHING_METHODS))
    parser.add_argument("--method", dest="method_one", action="append")
    parser.add_argument("--candidate-method", default=DEFAULT_CANDIDATE_METHOD)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--candidate-artifact", default=DEFAULT_CANDIDATE_ARTIFACT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", default=".cache/candidate_embeddings")
    parser.add_argument("--use-warmed-embeddings", action="store_true", default=True)
    parser.add_argument("--regenerate-candidates", action="store_true", help="Reserved; matching analysis does not regenerate candidates by default")
    parser.add_argument("--allow-embedding-api", action="store_true", help="Reserved; embeddings are not regenerated unless candidate regeneration is implemented")
    parser.add_argument("--llm-provider", choices=["openai", "google"], default="openai")
    parser.add_argument("--embedding-provider", choices=["openai", "google"], default="openai")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--google-project", default="")
    parser.add_argument("--google-location", default="")
    parser.add_argument("--google-credentials", default="")
    parser.add_argument("--model", default=REQUESTED_MATCH_MODEL)
    parser.add_argument("--fallback-model", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--self-consistency-temperature", type=float, default=0.4)
    parser.add_argument("--self-consistency-top-k", type=int, default=2)
    parser.add_argument("--current-group-size", type=int, default=12)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--api-retries", type=int, default=2)
    parser.add_argument("--self-consistency-samples", type=int, default=5)
    parser.add_argument("--list-methods", action="store_true")
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.scenario_one:
        args.scenarios = list(args.scenario_one)
    if args.method_one:
        args.methods = list(args.method_one)
    if args.list_methods:
        print("\n".join(MATCHING_METHODS))
        return
    if args.list_scenarios:
        if args.dev10_scenarios:
            print("\n".join(discover_dev10_scenarios(Path(args.rodi_root))))
        else:
            print("\n".join(PAPER_SCENARIOS))
        return
    args.methods = validate_methods(args.methods)
    if args.dry_run:
        scenarios = resolve_matching_scenarios(Path(args.rodi_root), bool(args.all_scenarios), args.scenarios, bool(args.dev10_scenarios))
        print(
            json.dumps(
                {
                    "scenarios": scenarios,
                    "methods": args.methods,
                    "candidate_method": args.candidate_method,
                    "top_k": args.top_k,
                    "self_consistency_top_k": args.self_consistency_top_k,
                    "current_group_size": args.current_group_size,
                    "dev10_scenarios": args.dev10_scenarios,
                    "no_api_error_rows": _no_api_error_rows(),
                },
                indent=2,
            )
        )
        return
    if args.regenerate_candidates or args.allow_embedding_api:
        raise SystemExit("Candidate/embedding regeneration is intentionally not implemented in matching analysis")
    summary = run_candidate_matching(args)
    print(f"Wrote matching analysis to {summary['output_dir']} for {len(summary['scenarios'])} scenarios and {len(summary['methods'])} methods")


if __name__ == "__main__":
    main()
