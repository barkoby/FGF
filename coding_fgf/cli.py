from __future__ import annotations

import argparse
import copy
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .baseline import bootstrap_llm4vkg, check_llm4vkg_resources, default_llm4vkg_root, load_postgres_dumps
from .benchmark import record_benchmark_run, summarize_benchmark
from .attribute_coverage import (
    apply_attribute_coverage_repairs,
    attribute_coverage_diagnostics,
    build_attribute_coverage_repair_prompt,
)
from .compare import compare_runs, compare_to_paper
from .constants import DEFAULT_EMBEDDING_MODEL, PAPER_SCENARIOS, REQUESTED_CODE_MODEL, REQUESTED_MATCH_MODEL
from .devset import create_devset
from .fol_ablation import write_fol_ablation_report
from .embeddings import embed_records
from .evaluate import evaluate_graph, execute_sql_psycopg2
from .fol import fol_validation_issues, mapping_diagnostics, matches_to_fol, validate_fol
from .io import ensure_dir, read_json, read_jsonl, write_csv, write_json, write_jsonl
from .fol_repair_round2 import apply_round2_repairs, fol_repair_round2_summary, issue_type_counts
from .llm import (
    CODEGEN_FEW_SHOT_PROMPT_VERSION,
    CODEGEN_PROMPT_VERSION,
    augment_candidate_rows_with_forced_matches,
    build_discriminator_candidate_rows,
    clear_llm_events,
    llm_discriminator_matches,
    llm_codegen,
    llm_fol,
    llm_events,
    llm_repair_attribute_coverage,
    llm_repair_fol,
    llm_repair_fol_round2,
    llm_repair_materialization_coverage,
    llm_repair_object_fol,
    llm_select_patterns,
    llm_match,
    matches_to_json,
    reask_suspicious_matches,
    repair_matches_with_candidates,
    generate_fol_repair_round2_prompt,
)
from .logging_utils import log_info
from .materialize import materialize_graph_with_log, materialize_to_file
from .materialization_coverage import (
    apply_materialization_coverage_repairs,
    build_materialization_repair_prompts,
    internal_materialization_score,
    materialization_coverage_diagnostics,
    materialization_coverage_summary,
    materialization_repair_accepted,
)
from .morphkgc import generate_source_r2rml, materialize_with_morphkgc, write_morph_config
from .object_evidence import build_object_link_evidence, object_evidence_summary, validate_object_rules_against_evidence
from .ontology import enrich_source_records_from_morphkgc, parse_ontology_records, source_schema_records, write_records
from .pattern_first import (
    apply_pattern_preservation,
    build_pattern_selection_prompts,
    build_schema_graph,
    compile_patterns_to_fol,
    extract_pattern_candidates,
    fol_target_coverage,
    infer_uri_keys,
    pattern_selection_internal_score,
    select_patterns_from_llm,
    stagec_vs_stagef_audit,
    summarize_pattern_candidates,
    validate_prompt_no_leakage,
)
from .retrieval import build_index, retrieve_candidates, write_candidates
from .sandbox import SandboxViolation, validate_generated_code
from .schema import find_ontology_file, parse_copy_data, parse_sql_dump


def add_provider_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llm-provider", choices=["openai", "google"], default=os.getenv("CODING_FGF_LLM_PROVIDER", "openai"))
    parser.add_argument("--llm-model", default=os.getenv("CODING_FGF_LLM_MODEL", ""))
    parser.add_argument("--embedding-provider", choices=["openai", "google"], default=os.getenv("CODING_FGF_EMBEDDING_PROVIDER", "openai"))
    parser.add_argument("--google-project", default=os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_PROJECT") or "")
    parser.add_argument("--google-location", default=os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("GOOGLE_LOCATION") or "global")
    parser.add_argument("--google-credentials", default=os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""))


def add_fewshot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fewshot", choices=["none", "generic", "failure_patterns"], default=os.getenv("CODING_FGF_FEWSHOT", "none"))
    parser.add_argument("--fewshot-max-examples", type=int, default=int(os.getenv("CODING_FGF_FEWSHOT_MAX_EXAMPLES", "4")))
    parser.add_argument("--fewshot-include-matching", action="store_true")
    parser.add_argument("--fewshot-include-fol", action="store_true")
    parser.add_argument("--fewshot-include-codegen", action="store_true")


def add_fol_ablation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fol-repair-mode",
        choices=["standard", "round2_only"],
        default=os.getenv("CODING_FGF_FOL_REPAIR_MODE", "standard"),
    )
    parser.add_argument(
        "--fol-repair-context",
        choices=["global", "batched"],
        default=os.getenv("CODING_FGF_FOL_REPAIR_CONTEXT", "global"),
    )
    parser.add_argument("--fol-repair-rounds", type=int, choices=[1, 2], default=int(os.getenv("CODING_FGF_FOL_REPAIR_ROUNDS", "1")))
    parser.add_argument("--fol-repair-round2", action="store_true")
    parser.add_argument("--fol-repair-max-issues-per-prompt", type=int, default=int(os.getenv("CODING_FGF_FOL_REPAIR_MAX_ISSUES", "8")))
    parser.add_argument("--fol-repair-allow-drop", dest="fol_repair_allow_drop", action="store_true", default=True)
    parser.add_argument("--fol-repair-no-drop", dest="fol_repair_allow_drop", action="store_false")
    parser.add_argument("--fol-repair-preservation-gate", action="store_true")
    parser.add_argument("--fol-batching", choices=["none", "table", "component", "hybrid"], default=os.getenv("CODING_FGF_FOL_BATCHING", "none"))
    parser.add_argument("--fol-batch-max-matches", type=int, default=int(os.getenv("CODING_FGF_FOL_BATCH_MAX_MATCHES", "0")) or None)
    parser.add_argument("--fol-batch-max-tokens", type=int, default=int(os.getenv("CODING_FGF_FOL_BATCH_MAX_TOKENS", "0")))
    parser.add_argument("--fol-batch-overlap-strategy", default=os.getenv("CODING_FGF_FOL_BATCH_OVERLAP", "fk_neighbors"))
    parser.add_argument("--attribute-coverage", action="store_true")
    parser.add_argument("--attribute-coverage-validation", action="store_true")
    parser.add_argument("--attribute-coverage-repair", action="store_true")
    parser.add_argument("--allow-weak-object-links", action="store_true")


def add_source_rdf_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--use-source-rdf", action="store_true")
    parser.add_argument("--source-rdf-backend", choices=["morphkgc"], default=os.getenv("CODING_FGF_SOURCE_RDF_BACKEND", "morphkgc"))


def add_materialization_coverage_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--materialization-coverage-validation", action="store_true")
    parser.add_argument("--materialization-coverage-repair", action="store_true")
    parser.add_argument("--materialization-repair-context", choices=["batched"], default=os.getenv("CODING_FGF_MATERIALIZATION_REPAIR_CONTEXT", "batched"))
    parser.add_argument("--materialization-repair-candidates", type=int, default=int(os.getenv("CODING_FGF_MATERIALIZATION_REPAIR_CANDIDATES", "1")))
    parser.add_argument("--materialization-repair-budget-chars", type=int, default=int(os.getenv("CODING_FGF_MATERIALIZATION_REPAIR_BUDGET_CHARS", "24000")))
    parser.add_argument("--internal-rerank-repair-candidates", action="store_true")


def add_pattern_first_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pattern-first-fgf", action="store_true")
    parser.add_argument("--pattern-schema-graph", action="store_true")
    parser.add_argument("--pattern-candidate-expansion", action="store_true")
    parser.add_argument("--uri-key-inference", action="store_true")
    parser.add_argument("--deterministic-pattern-compiler", action="store_true")
    parser.add_argument("--pattern-candidates", type=int, default=int(os.getenv("CODING_FGF_PATTERN_CANDIDATES", "1")))
    parser.add_argument("--internal-rerank-pattern-candidates", action="store_true")
    parser.add_argument("--pattern-repair-budget-chars", type=int, default=int(os.getenv("CODING_FGF_PATTERN_REPAIR_BUDGET_CHARS", "24000")))
    parser.add_argument("--pattern-selector-context", choices=["batched"], default=os.getenv("CODING_FGF_PATTERN_SELECTOR_CONTEXT", "batched"))
    parser.add_argument("--safe-domain-range-type-completion", action="store_true")
    parser.add_argument("--pattern-materialization-validation", action="store_true")
    parser.add_argument(
        "--pattern-preservation-mode",
        choices=["replace", "overlay", "conservative"],
        default=os.getenv("CODING_FGF_PATTERN_PRESERVATION_MODE", "conservative"),
    )


FOL_PORTFOLIO_ALLOWED_ARMS = {"full9_default", "stage2_hybrid", "stage2c_round2_only"}
FOL_PORTFOLIO_TIEBREAK_PRIORITY = {
    "stage2_hybrid": 0,
    "stage2c_round2_only": 1,
    "full9_default": 2,
}
FOL_PORTFOLIO_FORBIDDEN_SELECTION_INPUTS = (
    "summary.csv",
    "eval",
    "qpair",
    "gold",
    "llm4vkg",
    "paper_comparison",
)


def add_fol_portfolio_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fol-portfolio", action="store_true")
    parser.add_argument(
        "--fol-portfolio-arms",
        default=os.getenv("CODING_FGF_FOL_PORTFOLIO_ARMS", "full9_default,stage2_hybrid,stage2c_round2_only"),
    )
    parser.add_argument(
        "--fol-portfolio-selector",
        choices=["internal_materialization"],
        default=os.getenv("CODING_FGF_FOL_PORTFOLIO_SELECTOR", "internal_materialization"),
    )


def _materialization_coverage_validation_enabled(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "materialization_coverage_validation", False)
        or getattr(args, "materialization_coverage_repair", False)
    )


def _materialization_coverage_repair_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "materialization_coverage_repair", False))


def _pattern_first_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "pattern_first_fgf", False))


def _pattern_materialization_validation_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "pattern_materialization_validation", False) or getattr(args, "internal_rerank_pattern_candidates", False))


def _live_offline(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "offline", False)):
        return True
    if (
        getattr(args, "llm_provider", "openai") == "openai"
        or getattr(args, "embedding_provider", "openai") == "openai"
    ) and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for live OpenAI mode; pass --offline only for explicit deterministic test mode")
    return False


def _fol_rule_counts(fol: dict[str, Any]) -> dict[str, int]:
    rules = fol.get("rules", {}) or {}
    return {kind: len(rules.get(kind, []) or []) for kind in ("class", "data", "object")}


_TARGET_ROLE_PREFIXES = ("class:", "data_property:", "datatype_property:", "object_property:")
_CRITICAL_FOL_REPAIR_ISSUES = {
    "target_not_in_selected_matches",
    "missing_match_ids",
    "match_ids_do_not_reference_selected_matches",
}


def _canonicalize_role_prefixed_target(value: Any) -> tuple[Any, bool]:
    if not isinstance(value, str):
        return value, False
    for prefix in _TARGET_ROLE_PREFIXES:
        if value.startswith(prefix):
            candidate = value[len(prefix) :]
            if candidate.startswith(("http://", "https://", "urn:")):
                return candidate, True
    return value, False


def _canonicalize_fol_target_uris(fol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove accidental role prefixes from FOL target URI fields.

    The repair prompt sometimes returns values such as ``class:http://...`` in
    fields that should contain only the URI. This normalization is intentionally
    narrow and gold-blind: it only strips known syntactic role prefixes.
    """
    out = copy.deepcopy(fol)
    rules = out.setdefault("rules", {})
    changes: list[dict[str, Any]] = []
    for kind, field in (("class", "target_class"), ("data", "target_property"), ("object", "target_property")):
        for index, rule in enumerate(rules.get(kind, []) or []):
            old_value = rule.get(field)
            new_value, changed = _canonicalize_role_prefixed_target(old_value)
            if changed:
                rule[field] = new_value
                changes.append(
                    {
                        "rule_id": f"{kind}:{index}",
                        "field": field,
                        "old": old_value,
                        "new": new_value,
                    }
                )
    return out, {"canonicalized_target_uri_count": len(changes), "canonicalized_target_uris": changes}


def _fol_target_values_by_kind(fol: dict[str, Any]) -> dict[str, set[str]]:
    rules = fol.get("rules", {}) or {}
    return {
        "class": {str(rule.get("target_class")) for rule in rules.get("class", []) or [] if rule.get("target_class")},
        "data": {str(rule.get("target_property")) for rule in rules.get("data", []) or [] if rule.get("target_property")},
        "object": {str(rule.get("target_property")) for rule in rules.get("object", []) or [] if rule.get("target_property")},
    }


def _selected_target_values_by_kind(matches: list[dict[str, Any]]) -> dict[str, set[str]]:
    selected = {"class": set(), "data": set(), "object": set()}
    for match in matches:
        target = match.get("target_uri")
        if not target:
            continue
        source_id = str(match.get("source_id", ""))
        if source_id.startswith(("source-class:", "source-discriminator:")):
            selected["class"].add(str(target))
        elif source_id.startswith("source-data:"):
            selected["data"].add(str(target))
        elif source_id.startswith("source-object:"):
            selected["object"].add(str(target))
    return selected


def _class_filter_signatures(fol: dict[str, Any]) -> set[str]:
    signatures: set[str] = set()
    for rule in fol.get("rules", {}).get("class", []) or []:
        row_filter = rule.get("row_filter")
        if row_filter:
            signatures.add(
                json_dumps_stable(
                    {
                        "target_class": rule.get("target_class"),
                        "source_table": rule.get("source_table"),
                        "row_filter": row_filter,
                    }
                )
            )
    return signatures


def _object_grounding_signatures(fol: dict[str, Any]) -> set[str]:
    signatures: set[str] = set()
    for rule in fol.get("rules", {}).get("object", []) or []:
        signatures.add(
            json_dumps_stable(
                {
                    "target_property": rule.get("target_property"),
                    "source_table": rule.get("source_table"),
                    "source_columns": list(rule.get("source_columns") or []),
                    "target_table": rule.get("target_table"),
                    "target_columns": list(rule.get("target_columns") or []),
                    "subject_table": rule.get("subject_table"),
                    "subject_columns": list(rule.get("subject_columns") or []),
                    "object_table": rule.get("object_table"),
                    "object_columns": list(rule.get("object_columns") or []),
                    "match_ids": list(rule.get("match_ids") or []),
                }
            )
        )
    return signatures


def _fol_preservation_gate_decision(
    original_fol: dict[str, Any],
    repaired_fol: dict[str, Any],
    original_issues: list[dict[str, Any]],
    repaired_issues: list[dict[str, Any]],
    matches: list[dict[str, Any]],
) -> tuple[bool, list[str], dict[str, Any]]:
    accepted, rejection_reasons = _fol_repair_acceptance_decision(
        original_fol,
        repaired_fol,
        original_issues,
        repaired_issues,
        matches,
    )
    report: dict[str, Any] = {
        "base_acceptance": accepted,
        "base_rejection_reasons": list(rejection_reasons),
        "critical_issues_after": {},
        "dropped_selected_targets": {},
        "dropped_discriminator_class_rules": 0,
        "dropped_object_groundings": 0,
    }

    critical_after = issue_type_counts(
        [issue for issue in repaired_issues if str(issue.get("issue")) in _CRITICAL_FOL_REPAIR_ISSUES]
    )
    report["critical_issues_after"] = critical_after
    for issue_type, count in critical_after.items():
        if count:
            rejection_reasons.append(f"critical_fol_issue_after_repair:{issue_type}")

    selected = _selected_target_values_by_kind(matches)
    original_targets = _fol_target_values_by_kind(original_fol)
    repaired_targets = _fol_target_values_by_kind(repaired_fol)
    for kind in ("class", "data", "object"):
        protected_targets = original_targets[kind] & selected[kind]
        dropped = sorted(protected_targets - repaired_targets[kind])
        report["dropped_selected_targets"][kind] = dropped
        if dropped:
            rejection_reasons.append(f"preservation_gate_dropped_selected_{kind}_targets")

    original_filters = _class_filter_signatures(original_fol)
    repaired_filters = _class_filter_signatures(repaired_fol)
    dropped_filters = original_filters - repaired_filters
    report["dropped_discriminator_class_rules"] = len(dropped_filters)
    if dropped_filters:
        rejection_reasons.append("preservation_gate_dropped_filtered_class_rules")

    original_object_groundings = _object_grounding_signatures(original_fol)
    repaired_object_groundings = _object_grounding_signatures(repaired_fol)
    dropped_object_groundings = original_object_groundings - repaired_object_groundings
    report["dropped_object_groundings"] = len(dropped_object_groundings)
    if dropped_object_groundings and (original_targets["object"] & selected["object"]):
        rejection_reasons.append("preservation_gate_changed_object_grounding")

    deduped_reasons = list(dict.fromkeys(rejection_reasons))
    report["rejection_reasons"] = deduped_reasons
    report["accepted"] = not deduped_reasons
    return not deduped_reasons, deduped_reasons, report


def _attribute_coverage_validation_enabled(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "attribute_coverage", False)
        or getattr(args, "attribute_coverage_validation", False)
        or getattr(args, "attribute_coverage_repair", False)
    )


def _attribute_coverage_repair_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "attribute_coverage_repair", False))


def _fol_repair_acceptance_decision(
    original_fol: dict[str, Any],
    repaired_fol: dict[str, Any],
    original_issues: list[dict[str, Any]],
    repaired_issues: list[dict[str, Any]],
    matches: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    original_counts = _fol_rule_counts(original_fol)
    repaired_counts = _fol_rule_counts(repaired_fol)
    original_total = sum(original_counts.values())
    repaired_total = sum(repaired_counts.values())
    selected_match_count = sum(1 for match in matches if match.get("target_uri"))
    rejection_reasons: list[str] = []

    if len(repaired_issues) > len(original_issues):
        rejection_reasons.append("validation_issue_count_worse")
    if original_total > 0 and repaired_total == 0:
        rejection_reasons.append("repaired_fol_removed_all_rules")
    if selected_match_count > 0 and repaired_total == 0:
        rejection_reasons.append("selected_matches_but_no_repaired_rules")
    if original_total > 0 and repaired_total * 2 < original_total:
        rejection_reasons.append("repaired_rule_count_below_50_percent")
    for kind in ("class", "data", "object"):
        if original_counts[kind] > 0 and repaired_counts[kind] == 0:
            rejection_reasons.append(f"repaired_fol_removed_{kind}_rules")
    return not rejection_reasons, rejection_reasons


def _targeted_object_repair_acceptance_decision(
    original_fol: dict[str, Any],
    repaired_fol: dict[str, Any],
    original_issues: list[dict[str, Any]],
    repaired_issues: list[dict[str, Any]],
    matches: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    accepted, rejection_reasons = _fol_repair_acceptance_decision(
        original_fol,
        repaired_fol,
        original_issues,
        repaired_issues,
        matches,
    )
    original_rules = original_fol.get("rules", {}) or {}
    repaired_rules = repaired_fol.get("rules", {}) or {}
    for kind in ("class", "data"):
        original_json = json_dumps_stable(original_rules.get(kind, []) or [])
        repaired_json = json_dumps_stable(repaired_rules.get(kind, []) or [])
        if original_json != repaired_json:
            rejection_reasons.append(f"targeted_object_repair_modified_{kind}_rules")
    original_object_issues = [issue for issue in original_issues if str(issue.get("rule_id", "")).startswith("object:")]
    repaired_object_issues = [issue for issue in repaired_issues if str(issue.get("rule_id", "")).startswith("object:")]
    if len(repaired_object_issues) > len(original_object_issues):
        rejection_reasons.append("targeted_object_issue_count_worse")
    return accepted and not rejection_reasons, rejection_reasons


def _all_fol_issues(
    fol: dict[str, Any],
    matches: list[dict[str, Any]],
    tables: dict[str, Any],
    object_link_evidence: dict[str, Any] | None = None,
    include_attribute_coverage: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    issues = fol_validation_issues(fol, matches, tables)
    if object_link_evidence:
        issues = issues + validate_object_rules_against_evidence(fol, object_link_evidence)
    attribute_report: dict[str, Any] = {}
    if include_attribute_coverage:
        attribute_report = attribute_coverage_diagnostics(matches, fol, tables)
        issues = issues + list(attribute_report.get("issues", []) or [])
    return issues, attribute_report


def _run_attribute_coverage_stage(
    *,
    work: Path,
    fol: dict[str, Any],
    matches: list[dict[str, Any]],
    tables: dict[str, Any],
    data: Any,
    repair_enabled: bool,
    offline: bool,
    provider: str,
    model: str,
    google_project: str,
    google_location: str,
    google_credentials: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report_before = attribute_coverage_diagnostics(matches, fol, tables, data)
    write_json(work / "attribute_coverage_report.json", report_before)

    action_summary: dict[str, Any] = {
        "rules_added": 0,
        "rules_repaired": 0,
        "rules_dropped": 0,
        "rules_kept_with_justification": 0,
        "repairs_rejected": 0,
        "rejected_repairs": [],
        "action_counts": {},
    }
    report_after = report_before
    repair_response: dict[str, Any] = {}
    repair_attempted = bool(repair_enabled and report_before.get("issues") and not offline)
    repair_accepted = False
    rejection_reasons: list[str] = []
    candidate = fol

    if repair_attempted:
        prompt = build_attribute_coverage_repair_prompt(report_before, fol, matches, tables, data)
        write_json(
            work / "attribute_coverage_repair_prompt.json",
            {
                "prompt_version": "fgf_attribute_coverage_v1_repair",
                "prompt": prompt,
                "leakage_guard": "selected matches, schema excerpts, source summaries, current FOL data rules, and validator diagnostics only",
            },
        )
        try:
            repair_response = llm_repair_attribute_coverage(
                prompt,
                provider=provider,
                model=model,
                google_project=google_project,
                google_location=google_location,
                google_credentials=google_credentials,
            )
            candidate_raw, action_summary = apply_attribute_coverage_repairs(
                fol,
                list(repair_response.get("repairs", []) or []),
                matches,
                tables,
                allow_drop=True,
            )
            candidate = validate_fol(candidate_raw, tables)
            candidate["generation"] = {
                **fol.get("generation", {}),
                "attribute_coverage_repair": repair_response.get("generation", {}),
            }
            report_after = attribute_coverage_diagnostics(matches, candidate, tables, data)
            core_before = fol_validation_issues(fol, matches, tables)
            core_after = fol_validation_issues(candidate, matches, tables)
            accepted, coverage_rejections = _fol_repair_acceptance_decision(
                fol,
                candidate,
                core_before + list(report_before.get("issues", []) or []),
                core_after + list(report_after.get("issues", []) or []),
                matches,
            )
            if len(core_after) > len(core_before):
                accepted = False
                coverage_rejections.append("core_fol_validation_worse")
            if len(report_after.get("issues", []) or []) > len(report_before.get("issues", []) or []):
                accepted = False
                coverage_rejections.append("attribute_coverage_issue_count_worse")
            coverage_decreased = int(report_after.get("datatype_matches_with_rules", 0) or 0) < int(
                report_before.get("datatype_matches_with_rules", 0) or 0
            )
            fk_like_decreased = int(report_after.get("fk_like_data_rules", 0) or 0) < int(
                report_before.get("fk_like_data_rules", 0) or 0
            )
            if coverage_decreased and not fk_like_decreased:
                accepted = False
                coverage_rejections.append("datatype_rule_coverage_worse")
            if (
                fk_like_decreased
                and len(report_after.get("issues", []) or []) <= len(report_before.get("issues", []) or [])
                and len(core_after) <= len(core_before)
            ):
                coverage_rejections = [
                    reason
                    for reason in coverage_rejections
                    if reason != "repaired_fol_removed_data_rules"
                ]
                accepted = not coverage_rejections
            repair_accepted = accepted
            rejection_reasons = coverage_rejections
        except Exception as exc:
            repair_accepted = False
            rejection_reasons = [f"attribute_coverage_repair_error:{exc}"]
            report_after = report_before
            candidate = fol

    final_fol = candidate if repair_accepted else fol
    final_report = report_after if repair_accepted else report_before
    write_json(work / "fol_after_attribute_coverage.json", final_fol)
    summary = {
        "attribute_coverage_validation_enabled": True,
        "attribute_coverage_repair_enabled": bool(repair_enabled),
        "repair_attempted": repair_attempted,
        "repair_accepted": repair_accepted,
        "fallback_used": bool(repair_attempted and not repair_accepted),
        "rejection_reasons": rejection_reasons,
        "selected_datatype_matches": int(report_before.get("selected_datatype_matches", 0) or 0),
        "datatype_matches_with_rules_before": int(report_before.get("datatype_matches_with_rules", 0) or 0),
        "datatype_matches_with_rules_after": int(final_report.get("datatype_matches_with_rules", 0) or 0),
        "datatype_matches_missing_rules_before": int(report_before.get("datatype_matches_missing_rules", 0) or 0),
        "datatype_matches_missing_rules_after": int(final_report.get("datatype_matches_missing_rules", 0) or 0),
        "fk_like_data_rules_before": int(report_before.get("fk_like_data_rules", 0) or 0),
        "fk_like_data_rules_after": int(final_report.get("fk_like_data_rules", 0) or 0),
        "issues_before": len(report_before.get("issues", []) or []),
        "issues_after": len(final_report.get("issues", []) or []),
        "issues_by_type_before": report_before.get("issues_by_type", {}),
        "issues_by_type_after": final_report.get("issues_by_type", {}),
        "rules_added": int(action_summary.get("rules_added", 0) or 0),
        "rules_repaired": int(action_summary.get("rules_repaired", 0) or 0),
        "rules_dropped": int(action_summary.get("rules_dropped", 0) or 0),
        "rules_kept_with_justification": int(action_summary.get("rules_kept_with_justification", 0) or 0),
        "repairs_rejected": int(action_summary.get("repairs_rejected", 0) or 0),
    }
    write_csv(work / "attribute_coverage_summary.csv", [summary], list(summary.keys()))
    write_json(
        work / "attribute_coverage_report_after.json",
        {
            "report_before": report_before,
            "report_after": report_after,
            "final_report": final_report,
            "summary": summary,
            "repair_response": repair_response,
            "action_summary": action_summary,
        },
    )
    return final_fol, summary


def _run_fol_repair_round2(
    *,
    work: Path,
    fol: dict[str, Any],
    matches: list[dict[str, Any]],
    tables: dict[str, Any],
    issues_before: list[dict[str, Any]],
    issues_after_round1: list[dict[str, Any]],
    object_link_evidence: dict[str, Any] | None,
    provider: str,
    model: str,
    google_project: str,
    google_location: str,
    google_credentials: str,
    max_issues: int,
    allow_drop: bool,
    include_attribute_coverage: bool,
    preservation_gate: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    prompt = generate_fol_repair_round2_prompt(
        fol,
        matches,
        tables,
        issues_after_round1,
        object_link_evidence=object_link_evidence,
        max_issues=max_issues,
        allow_drop=allow_drop,
    )
    write_json(
        work / "fol_repair_round2_prompt.json",
        {
            "prompt_version": "fgf_fol_v2_round2_repair",
            "max_issues": max_issues,
            "allow_drop": allow_drop,
            "prompt": prompt,
            "leakage_guard": "schema, selected matches, validator diagnostics, object evidence, and source summaries only",
        },
    )
    response = llm_repair_fol_round2(
        fol,
        matches,
        tables,
        issues_after_round1,
        object_link_evidence=object_link_evidence,
        max_issues=max_issues,
        allow_drop=allow_drop,
        provider=provider,
        model=model,
        google_project=google_project,
        google_location=google_location,
        google_credentials=google_credentials,
    )
    candidate_raw, action_counts = apply_round2_repairs(fol, list(response.get("repairs", []) or []), allow_drop=allow_drop)
    hygiene_report: dict[str, Any] = {}
    if preservation_gate:
        candidate_raw, hygiene_report = _canonicalize_fol_target_uris(candidate_raw)
    candidate = validate_fol(candidate_raw, tables)
    candidate["generation"] = {
        **fol.get("generation", {}),
        "repair_round2": response.get("generation", {}),
    }
    round2_issues, attribute_report = _all_fol_issues(
        candidate,
        matches,
        tables,
        object_link_evidence=object_link_evidence,
        include_attribute_coverage=include_attribute_coverage,
    )
    if preservation_gate:
        accepted, rejection_reasons, preservation_report = _fol_preservation_gate_decision(
            fol,
            candidate,
            issues_after_round1,
            round2_issues,
            matches,
        )
        preservation_report["target_uri_hygiene"] = hygiene_report
    else:
        accepted, rejection_reasons = _fol_repair_acceptance_decision(
            fol,
            candidate,
            issues_after_round1,
            round2_issues,
            matches,
        )
        preservation_report = {}
    if len(round2_issues) > len(issues_after_round1):
        accepted = False
        rejection_reasons.append("round2_validation_issue_count_worse")
    summary = fol_repair_round2_summary(issues_before, issues_after_round1, round2_issues, action_counts)
    summary.update(
        {
            "repair_round2_attempted": True,
            "repair_round2_accepted": accepted,
            "repair_round2_rejection_reasons": rejection_reasons,
            "repair_round2_response": response,
            "attribute_coverage": attribute_report,
            "preservation_gate": preservation_report,
        }
    )
    write_json(work / "fol_validation_report_round2.json", {**summary, "issues_after_round2": round2_issues})
    write_json(work / "fol_rules_after_round2.json", candidate)
    write_csv(work / "fol_rounds_summary.csv", [summary], list(summary.keys()))
    return (candidate if accepted else fol), (round2_issues if accepted else issues_after_round1), summary


def _run_fol_single_round2_style_repair(
    *,
    work: Path,
    fol: dict[str, Any],
    matches: list[dict[str, Any]],
    tables: dict[str, Any],
    issues_before_repair: list[dict[str, Any]],
    object_link_evidence: dict[str, Any] | None,
    provider: str,
    model: str,
    google_project: str,
    google_location: str,
    google_credentials: str,
    max_issues: int,
    allow_drop: bool,
    include_attribute_coverage: bool,
    preservation_gate: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    write_json(
        work / "fol_validation_report_before_repair.json",
        {
            "issues_before_repair": issues_before_repair,
            "issues_by_type_before_repair": issue_type_counts(issues_before_repair),
            "rule_counts_before_repair": _fol_rule_counts(fol),
            "repair_mode": "round2_only",
        },
    )
    write_json(work / "fol_rules_before_repair.json", fol)

    action_counts: dict[str, int] = {}
    response: dict[str, Any] = {}
    issues_after_single = list(issues_before_repair)
    candidate = fol
    accepted = False
    fallback_used = bool(issues_before_repair)
    rejection_reasons: list[str] = []
    repair_error = ""
    attribute_report: dict[str, Any] = {}
    preservation_report: dict[str, Any] = {}

    if issues_before_repair:
        prompt = generate_fol_repair_round2_prompt(
            fol,
            matches,
            tables,
            issues_before_repair,
            object_link_evidence=object_link_evidence,
            max_issues=max_issues,
            allow_drop=allow_drop,
        )
        write_json(
            work / "fol_repair_single_round2_style_prompt.json",
            {
                "prompt_version": "fgf_fol_v2_round2_repair",
                "repair_mode": "round2_only",
                "max_issues": max_issues,
                "allow_drop": allow_drop,
                "prompt": prompt,
                "leakage_guard": "schema, selected matches, validator diagnostics, object evidence, and source summaries only",
            },
        )
        try:
            response = llm_repair_fol_round2(
                fol,
                matches,
                tables,
                issues_before_repair,
                object_link_evidence=object_link_evidence,
                max_issues=max_issues,
                allow_drop=allow_drop,
                provider=provider,
                model=model,
                google_project=google_project,
                google_location=google_location,
                google_credentials=google_credentials,
            )
            candidate_raw, action_counts = apply_round2_repairs(
                fol,
                list(response.get("repairs", []) or []),
                allow_drop=allow_drop,
            )
            hygiene_report: dict[str, Any] = {}
            if preservation_gate:
                candidate_raw, hygiene_report = _canonicalize_fol_target_uris(candidate_raw)
            candidate = validate_fol(candidate_raw, tables)
            candidate["generation"] = {
                **fol.get("generation", {}),
                "repair_single_round2_style": response.get("generation", {}),
            }
            issues_after_single, attribute_report = _all_fol_issues(
                candidate,
                matches,
                tables,
                object_link_evidence=object_link_evidence,
                include_attribute_coverage=include_attribute_coverage,
            )
            if preservation_gate:
                accepted, rejection_reasons, preservation_report = _fol_preservation_gate_decision(
                    fol,
                    candidate,
                    issues_before_repair,
                    issues_after_single,
                    matches,
                )
                preservation_report["target_uri_hygiene"] = hygiene_report
            else:
                accepted, rejection_reasons = _fol_repair_acceptance_decision(
                    fol,
                    candidate,
                    issues_before_repair,
                    issues_after_single,
                    matches,
                )
                preservation_report = {}
            if len(issues_after_single) > len(issues_before_repair):
                accepted = False
                rejection_reasons.append("single_round2_style_validation_issue_count_worse")
            fallback_used = not accepted
        except Exception as exc:
            repair_error = str(exc)
            candidate = fol
            issues_after_single = list(issues_before_repair)
            rejection_reasons = ["single_round2_style_repair_error"]
            accepted = False
            fallback_used = True
    else:
        accepted = False
        fallback_used = False

    summary: dict[str, Any] = {
        "repair_mode": "round2_only",
        "single_round2_style_attempted": bool(issues_before_repair),
        "single_round2_style_accepted": accepted,
        "fallback_used": fallback_used,
        "rejection_reasons": rejection_reasons,
        "repair_error": repair_error,
        "issues_before_repair": len(issues_before_repair),
        "issues_after_single_repair": len(issues_after_single),
        "issues_by_type_before_repair": issue_type_counts(issues_before_repair),
        "issues_by_type_after_single_repair": issue_type_counts(issues_after_single),
        "rules_added": int(action_counts.get("add_class_rule", 0)) + int(action_counts.get("add_data_rule", 0)),
        "rules_repaired": int(action_counts.get("repair", 0)),
        "rules_dropped": int(action_counts.get("drop", 0)),
        "rules_kept_with_justification": int(action_counts.get("keep_with_justification", 0)),
        "explain_no_rule_needed": int(action_counts.get("explain_no_rule_needed", 0)),
        "attribute_coverage": attribute_report,
        "repair_response": response,
        "preservation_gate": preservation_report if issues_before_repair else {},
    }
    write_json(
        work / "fol_validation_report_after_single_round2_style_repair.json",
        {
            **summary,
            "issues_before_repair_detail": issues_before_repair,
            "issues_after_single_repair_detail": issues_after_single,
        },
    )
    write_json(work / "fol_rules_after_single_round2_style_repair.json", candidate)
    csv_summary = {key: value for key, value in summary.items() if key not in {"repair_response", "attribute_coverage"}}
    write_csv(work / "fol_single_round2_style_summary.csv", [csv_summary], list(csv_summary.keys()))
    return (candidate if accepted else fol), (issues_after_single if accepted else issues_before_repair), summary


def _pattern_first_fol_stage(
    *,
    scenario: str,
    work: Path,
    tables: dict[str, Any],
    data: Any,
    matches: list[dict[str, Any]],
    offline: bool,
    provider: str,
    model: str,
    google_project: str,
    google_location: str,
    google_credentials: str,
    pattern_candidates: int,
    budget_chars: int,
    safe_domain_range_type_completion: bool,
) -> dict[str, Any]:
    if offline:
        raise RuntimeError("Stage F pattern-first FOL requires live bounded LLM pattern selection")
    write_json(work / "matches_before_pattern_first.json", {"matches": matches})
    candidate_rows = read_jsonl(work / "candidates.jsonl")
    schema_graph = build_schema_graph(tables, data)
    write_json(work / "schema_graph.json", schema_graph)
    write_json(work / "schema_value_profiles.json", schema_graph.get("value_profiles", {}))
    patterns = extract_pattern_candidates(tables, data, matches, candidate_rows, schema_graph)
    write_json(work / "pattern_candidates.json", {"patterns": patterns})
    write_csv(work / "pattern_candidate_summary.csv", summarize_pattern_candidates(patterns), ["pattern_type", "count"])
    if not patterns:
        raise RuntimeError("Stage F produced no bounded pattern candidates")

    compiled_candidates: list[dict[str, Any]] = []
    candidate_count = max(1, int(pattern_candidates or 1))
    for candidate_index in range(1, candidate_count + 1):
        candidate_dir = ensure_dir(work / "pattern_candidate_runs" / f"candidate_{candidate_index}")
        prompts = build_pattern_selection_prompts(patterns, budget_chars=budget_chars, variant=candidate_index)
        if not all(validate_prompt_no_leakage(record["prompt"]) for record in prompts):
            raise RuntimeError("Stage F pattern prompt failed leakage guard")
        write_json(candidate_dir / "pattern_selection_prompt.json", {"prompts": prompts, "budget_chars": budget_chars})
        responses: list[dict[str, Any]] = []
        for prompt_index, prompt_record in enumerate(prompts, start=1):
            responses.append(
                llm_select_patterns(
                    prompt_record["prompt"],
                    provider=provider,
                    model=model,
                    google_project=google_project,
                    google_location=google_location,
                    google_credentials=google_credentials,
                    schema_name=f"pattern_selection_{scenario}_{candidate_index}_{prompt_index}",
                )
            )
        selected, selection_report = select_patterns_from_llm(patterns, responses)
        if not selected:
            raise RuntimeError(f"Stage F candidate {candidate_index} selected no valid patterns")
        write_json(candidate_dir / "pattern_selection_decisions.json", {"responses": responses, **selection_report})
        write_json(candidate_dir / "pattern_selected_matches.json", {"patterns": selected})
        uri_keys = infer_uri_keys(selected, tables, data)
        write_json(candidate_dir / "uri_key_inference.json", uri_keys)
        write_json(candidate_dir / "uri_template_consistency_report.json", {"issues": uri_keys.get("consistency_issues", [])})
        fol, compiler_report = compile_patterns_to_fol(
            selected,
            tables,
            uri_keys,
            safe_domain_range_type_completion=safe_domain_range_type_completion,
        )
        if not any(fol.get("rules", {}).get(kind) for kind in ("class", "data", "object")):
            raise RuntimeError(f"Stage F candidate {candidate_index} compiled zero rules")
        write_json(candidate_dir / "fol_after_pattern_compilation.json", fol)
        write_json(candidate_dir / "fol.json", fol)
        write_json(candidate_dir / "pattern_compiler_report.json", compiler_report)
        candidate_record = {
            "candidate_index": candidate_index,
            "candidate_dir": str(candidate_dir),
            "selected_patterns": len(selected),
            "prompt_count": len(prompts),
            "max_prompt_chars": max([int(record.get("char_count", 0) or 0) for record in prompts] or [0]),
            "compiler_report": compiler_report,
            "uri_key_inference": uri_keys,
            "score": list(pattern_selection_internal_score({"compiler_report": compiler_report, "uri_key_inference": uri_keys})),
        }
        write_json(candidate_dir / "pattern_candidate_summary.json", candidate_record)
        compiled_candidates.append(candidate_record)

    selected_record = max(compiled_candidates, key=lambda record: tuple(record.get("score", [])))
    selected_dir = Path(str(selected_record["candidate_dir"]))
    fol = read_json(selected_dir / "fol.json")
    for name in (
        "pattern_selection_prompt.json",
        "pattern_selection_decisions.json",
        "pattern_selected_matches.json",
        "uri_key_inference.json",
        "uri_template_consistency_report.json",
        "fol_after_pattern_compilation.json",
        "pattern_compiler_report.json",
    ):
        _copy_if_exists(selected_dir / name, work / name)
    write_json(
        work / "selected_pattern_candidate.json",
        {
            "selected_candidate": selected_record["candidate_index"],
            "selection_metric": "compile_time_internal_pattern_score",
            "candidates": compiled_candidates,
        },
    )
    fol["generation"] = {
        **(fol.get("generation", {}) or {}),
        "provider": provider,
        "model": model,
        "stage": "Stage F pattern-first deterministic FOL compilation",
        "selected_candidate": selected_record["candidate_index"],
        "pattern_candidates": candidate_count,
    }
    return fol


def _run_fol_standard_repair(
    *,
    work: Path,
    fol: dict[str, Any],
    matches: list[dict[str, Any]],
    tables: dict[str, Any],
    issues: list[dict[str, Any]],
    object_link_evidence: dict[str, Any] | None,
    provider: str,
    model: str,
    google_project: str,
    google_location: str,
    google_credentials: str,
    include_attribute_coverage: bool,
    repair_context: str = "global",
    max_issues: int = 8,
    allow_drop: bool = True,
    preservation_gate: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    report: dict[str, Any] = {
        "repair_mode": "standard" if repair_context == "global" else "standard_contextual_round2",
        "repair_context": repair_context,
        "issues_before": issues,
        "repair_attempted": bool(issues),
        "issues_after": issues,
        "original_rule_counts": _fol_rule_counts(fol),
        "repair_accepted": False,
    }
    if not issues:
        return fol, issues, report
    if repair_context == "batched":
        repaired, final_issues, contextual_summary = _run_fol_single_round2_style_repair(
            work=work,
            fol=fol,
            matches=matches,
            tables=tables,
            issues_before_repair=issues,
            object_link_evidence=object_link_evidence,
            provider=provider,
            model=model,
            google_project=google_project,
            google_location=google_location,
            google_credentials=google_credentials,
            max_issues=max_issues,
            allow_drop=allow_drop,
            include_attribute_coverage=include_attribute_coverage,
            preservation_gate=preservation_gate,
        )
        report["contextual_round2"] = {
            key: value for key, value in contextual_summary.items() if key != "repair_response"
        }
        report["issues_after"] = final_issues
        report["repair_accepted"] = bool(contextual_summary.get("single_round2_style_accepted", False))
        report["fallback_used"] = bool(contextual_summary.get("fallback_used", False))
        return repaired, final_issues, report
    try:
        repaired_raw = llm_repair_fol(
            fol,
            matches,
            tables,
            issues,
            provider=provider,
            model=model,
            google_project=google_project,
            google_location=google_location,
            google_credentials=google_credentials,
        )
        hygiene_report: dict[str, Any] = {}
        if preservation_gate:
            repaired_raw, hygiene_report = _canonicalize_fol_target_uris(repaired_raw)
        repaired = validate_fol(repaired_raw, tables)
        repaired["generation"] = {
            **fol.get("generation", {}),
            "repair": repaired_raw.get("generation", {}),
        }
        repaired_issues, repaired_attribute_report = _all_fol_issues(
            repaired,
            matches,
            tables,
            object_link_evidence=object_link_evidence,
            include_attribute_coverage=include_attribute_coverage,
        )
        report["issues_after"] = repaired_issues
        report["repaired_rule_counts"] = _fol_rule_counts(repaired)
        if include_attribute_coverage:
            report["attribute_coverage_after_round1"] = repaired_attribute_report
        if preservation_gate:
            repair_accepted, rejection_reasons, preservation_report = _fol_preservation_gate_decision(
                fol,
                repaired,
                issues,
                repaired_issues,
                matches,
            )
            preservation_report["target_uri_hygiene"] = hygiene_report
            report["preservation_gate"] = preservation_report
        else:
            repair_accepted, rejection_reasons = _fol_repair_acceptance_decision(
                fol,
                repaired,
                issues,
                repaired_issues,
                matches,
            )
        report["repair_rejection_reasons"] = rejection_reasons
        if repair_accepted:
            report["repair_accepted"] = True
            return repaired, repaired_issues, report
        report["rejected_repair_issues"] = repaired_issues
        report["issues_after"] = issues
        return fol, issues, report
    except Exception as exc:
        report["repair_error"] = str(exc)
        report["issues_after"] = issues
        return fol, issues, report


def _apply_fol_repair_arm(
    *,
    arm: str,
    work: Path,
    frozen_fol: dict[str, Any],
    matches: list[dict[str, Any]],
    tables: dict[str, Any],
    issues: list[dict[str, Any]],
    object_link_evidence: dict[str, Any] | None,
    provider: str,
    model: str,
    google_project: str,
    google_location: str,
    google_credentials: str,
    max_issues: int,
    allow_drop: bool,
    include_attribute_coverage: bool,
    preservation_gate: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if arm == "pre_repair_preserved":
        final_fol = frozen_fol
        final_issues = list(issues)
        hygiene_report: dict[str, Any] = {}
        if preservation_gate:
            final_fol, hygiene_report = _canonicalize_fol_target_uris(frozen_fol)
            final_fol = validate_fol(final_fol, tables)
            final_issues, _attribute_report = _all_fol_issues(
                final_fol,
                matches,
                tables,
                object_link_evidence=object_link_evidence,
                include_attribute_coverage=include_attribute_coverage,
            )
        return final_fol, final_issues, {
            "repair_mode": "pre_repair_preserved",
            "repair_attempted": False,
            "issues_before": issues,
            "issues_after": final_issues,
            "target_uri_hygiene": hygiene_report,
            "preservation_gate": bool(preservation_gate),
        }
    if arm == "standard":
        return _run_fol_standard_repair(
            work=work,
            fol=frozen_fol,
            matches=matches,
            tables=tables,
            issues=issues,
            object_link_evidence=object_link_evidence,
            provider=provider,
            model=model,
            google_project=google_project,
            google_location=google_location,
            google_credentials=google_credentials,
            include_attribute_coverage=include_attribute_coverage,
            preservation_gate=preservation_gate,
        )
    if arm == "round2_only":
        fol, final_issues, summary = _run_fol_single_round2_style_repair(
            work=work,
            fol=frozen_fol,
            matches=matches,
            tables=tables,
            issues_before_repair=issues,
            object_link_evidence=object_link_evidence,
            provider=provider,
            model=model,
            google_project=google_project,
            google_location=google_location,
            google_credentials=google_credentials,
            max_issues=max_issues,
            allow_drop=allow_drop,
            include_attribute_coverage=include_attribute_coverage,
            preservation_gate=preservation_gate,
        )
        report = {key: value for key, value in summary.items() if key != "repair_response"}
        report["repair_mode"] = "round2_only"
        report["issues_before"] = issues
        report["issues_after"] = final_issues
        return fol, final_issues, report
    if arm == "standard_then_round2":
        round1_fol, round1_issues, report = _run_fol_standard_repair(
            work=work,
            fol=frozen_fol,
            matches=matches,
            tables=tables,
            issues=issues,
            object_link_evidence=object_link_evidence,
            provider=provider,
            model=model,
            google_project=google_project,
            google_location=google_location,
            google_credentials=google_credentials,
            include_attribute_coverage=include_attribute_coverage,
            preservation_gate=preservation_gate,
        )
        report["repair_mode"] = "standard_then_round2"
        report["issues_after_round1_kept"] = round1_issues
        if not round1_issues:
            report["round2"] = {"repair_round2_attempted": False, "reason": "no_remaining_issues"}
            return round1_fol, round1_issues, report
        try:
            final_fol, final_issues, round2_summary = _run_fol_repair_round2(
                work=work,
                fol=round1_fol,
                matches=matches,
                tables=tables,
                issues_before=issues,
                issues_after_round1=round1_issues,
                object_link_evidence=object_link_evidence,
                provider=provider,
                model=model,
                google_project=google_project,
                google_location=google_location,
                google_credentials=google_credentials,
                max_issues=max_issues,
                allow_drop=allow_drop,
                include_attribute_coverage=include_attribute_coverage,
                preservation_gate=preservation_gate,
            )
            report["round2"] = {key: value for key, value in round2_summary.items() if key != "repair_round2_response"}
            report["issues_after"] = final_issues
            return final_fol, final_issues, report
        except Exception as exc:
            report["round2"] = {
                "repair_round2_attempted": True,
                "repair_round2_accepted": False,
                "repair_round2_error": str(exc),
            }
            report["issues_after"] = round1_issues
            return round1_fol, round1_issues, report
    raise ValueError(f"Unknown FOL repair ablation arm: {arm}")


def _copy_frozen_upstream_artifacts(src: Path, dst: Path) -> None:
    ensure_dir(dst)
    for name in (
        "source_records.jsonl",
        "target_records.jsonl",
        "candidates.jsonl",
        "candidate_audit.json",
        "matches.json",
        "match_validation_report.json",
        "discriminator_candidates.jsonl",
        "discriminator_matches.json",
        "object_link_evidence.json",
        "fol_rules_before_repair.json",
        "fol_validation_report_before_repair.json",
    ):
        source = src / name
        if source.exists():
            shutil.copy2(source, dst / name)


def _fewshot_enabled(args: argparse.Namespace, component: str) -> bool:
    legacy_flag = bool(getattr(args, f"{component}_few_shot_examples", False))
    policy = str(getattr(args, "fewshot", "none") or "none")
    if policy == "none":
        return legacy_flag
    include_flags = {
        "match": bool(getattr(args, "fewshot_include_matching", False)),
        "fol": bool(getattr(args, "fewshot_include_fol", False)),
        "codegen": bool(getattr(args, "fewshot_include_codegen", False)),
    }
    if any(include_flags.values()):
        return include_flags.get(component, False) or legacy_flag
    return True


def _source_context_arg(args: argparse.Namespace) -> str:
    if bool(getattr(args, "use_source_rdf", False)):
        backend = str(getattr(args, "source_rdf_backend", "morphkgc"))
        if backend != "morphkgc":
            raise SystemExit(f"Unsupported source RDF backend: {backend}")
        return "morphkgc"
    return str(getattr(args, "source_context", "schema"))


def _write_leakage_risk_report(work: Path) -> None:
    report = """# Leakage Risk Report

This run uses only schema, ontology, selected matches, validator diagnostics, runtime diagnostics, source data summaries, and optional source-RDF summaries before output selection. SQL/SPARQL answer sets, gold mappings, paper baselines, qpair failures, and target triples derived from evaluation are reserved for post-hoc reporting only.

| Improvement | Allowed inputs | Leakage risk | Guard |
|---|---|---:|---|
| FOL repair round 2 | selected matches, schema, validator issues, object evidence | Low | Prompt and artifacts forbid qpair/gold/paper feedback and invented URIs/tables/columns. |
| Source RDF grounding | source schema/data, generated source R2RML, source RDF summaries | Low | Used as source-side evidence only; final RDF still comes from generated FGF code. |
| Context-safe FOL batching | selected matches, schema graph, FK edges, object evidence | Low | Changes prompt packaging only. |
| Few-shot examples | synthetic abstract patterns | Medium | Examples must avoid RODI scenario names, qpair IDs, gold answers, scores, real benchmark URIs, and copied rules. |
| Attribute coverage | selected datatype matches, schema, FOL rules | Low | Checks selected-match realization without querying benchmark answers. |
| Object-link evidence | schema/FK graph, selected matches, reachability summaries | Low-to-medium | Weak name-only links are disabled unless explicitly requested. |

Acceptance and final reporting are based on generated RDF after materialization; evaluation feedback is not used to choose matches, rules, code, or repair outputs.
"""
    (work / "leakage_risk_report.md").write_text(report, encoding="utf-8")


def json_dumps_stable(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def scenario_dir(root: Path, scenario: str) -> Path:
    if (root / "data" / scenario).exists():
        return root / "data" / scenario
    if (root / scenario).exists():
        return root / scenario
    raise FileNotFoundError(f"Could not find scenario {scenario} below {root}")


def cmd_devset(args: argparse.Namespace) -> None:
    summary = create_devset(
        rodi_root=Path(args.rodi_root),
        out_dir=Path(args.out),
        scenarios=args.scenarios.split(",") if args.scenarios else PAPER_SCENARIOS,
        fraction=args.fraction,
        seed=args.seed,
    )
    print(f"Wrote {len(summary)} sampled table summaries to {Path(args.out) / 'devset_summary.csv'}")


def stage_verbalize(
    scenario: str,
    root: Path,
    work: Path,
    source_context: str = "schema",
    db_host: str = "postgres",
    db_port: str = "5432",
    db_name: str = "",
    db_user: str = "postgres",
    db_password: str = "postgres",
) -> None:
    log_info(f"{scenario}: verbalize start source_context={source_context}")
    src = scenario_dir(root, scenario)
    tables = parse_sql_dump(src / "dump.sql")
    data = parse_copy_data(src / "dump.sql")
    target = parse_ontology_records(find_ontology_file(src))
    source = source_schema_records(tables, data.rows)
    if source_context == "morphkgc":
        ensure_dir(work)
        mapping_path = generate_source_r2rml(tables, scenario, work / "source_r2rml.ttl", db_schema=scenario)
        output_path = work / "source_graph.ttl"
        config_path = write_morph_config(
            mapping_path=mapping_path,
            output_path=output_path,
            db_url=f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{scenario}",
            db_user=db_user,
            db_password=db_password,
            config_path=work / "morphkgc.ini",
        )
        if not materialize_with_morphkgc(config_path) or not output_path.exists():
            raise RuntimeError("Morph-KGC source-context mode failed to produce source_graph.ttl")
        source = enrich_source_records_from_morphkgc(source, tables, scenario, output_path)
    elif source_context != "schema":
        raise ValueError(f"Unsupported source context: {source_context}")
    ensure_dir(work)
    write_records(work, target, source)
    log_info(f"{scenario}: verbalize complete targets={len(target)} sources={len(source)}")


def cmd_verbalize(args: argparse.Namespace) -> None:
    stage_verbalize(args.scenario, Path(args.rodi_root), Path(args.work))
    print(f"Wrote verbalized records to {args.work}")


def cmd_source_rdf(args: argparse.Namespace) -> None:
    src = scenario_dir(Path(args.rodi_root), args.scenario)
    work = ensure_dir(Path(args.work))
    tables = parse_sql_dump(src / "dump.sql")
    mapping = generate_source_r2rml(tables, args.scenario, work / "source_r2rml.ttl", db_schema=args.scenario)
    db_url = args.db_url or f"postgresql://{args.db_user}:{args.db_password}@{args.db_host}:{args.db_port}/{args.db_name}"
    config = write_morph_config(
        mapping_path=mapping,
        output_path=work / "source_graph.ttl",
        db_url=db_url,
        db_user=args.db_user,
        db_password=args.db_password,
        config_path=work / "morphkgc.ini",
    )
    ran = materialize_with_morphkgc(config) if args.run else False
    print(f"Wrote Morph-KGC config to {config}; materialized={ran}")


def stage_index(
    work: Path,
    embedding_model: str,
    offline: bool,
    embedding_provider: str = "openai",
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
) -> None:
    log_info(f"{work.name}: index start offline={offline} embedding_provider={embedding_provider} embedding_model={embedding_model}")
    records = read_jsonl(work / "target_records.jsonl")
    embedded = embed_records(
        records,
        work / "embedding_cache.jsonl",
        model=embedding_model,
        offline=offline,
        provider=embedding_provider,
        google_project=google_project,
        google_location=google_location,
        google_credentials=google_credentials,
    )
    build_index(embedded, work)
    log_info(f"{work.name}: index complete records={len(records)}")


def cmd_index(args: argparse.Namespace) -> None:
    stage_index(
        Path(args.work),
        args.embedding_model,
        args.offline,
        embedding_provider=args.embedding_provider,
        google_project=args.google_project,
        google_location=args.google_location,
        google_credentials=args.google_credentials,
    )
    print(f"Wrote index artifacts to {args.work}")


def stage_retrieve(
    work: Path,
    embedding_model: str,
    k: int,
    offline: bool,
    embedding_provider: str = "openai",
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
    allow_forced_candidates: bool = False,
) -> None:
    log_info(f"{work.name}: retrieve start k={k} offline={offline} embedding_provider={embedding_provider} embedding_model={embedding_model}")
    source = read_jsonl(work / "source_records.jsonl")
    target = read_jsonl(work / "target_records.jsonl")
    embedded_source = embed_records(
        source,
        work / "source_embedding_cache.jsonl",
        model=embedding_model,
        offline=offline,
        provider=embedding_provider,
        google_project=google_project,
        google_location=google_location,
        google_credentials=google_credentials,
    )
    candidates = retrieve_candidates(embedded_source, work, k=k)
    if allow_forced_candidates:
        candidates = augment_candidate_rows_with_forced_matches(candidates, target)
    write_candidates(work / "candidates.jsonl", candidates)
    log_info(f"{work.name}: retrieve complete sources={len(source)} candidate_rows={len(candidates)}")


def cmd_retrieve(args: argparse.Namespace) -> None:
    stage_retrieve(
        Path(args.work),
        args.embedding_model,
        args.k,
        args.offline,
        embedding_provider=args.embedding_provider,
        google_project=args.google_project,
        google_location=args.google_location,
        google_credentials=args.google_credentials,
        allow_forced_candidates=args.allow_forced_candidates,
    )
    print(f"Wrote candidates to {Path(args.work) / 'candidates.jsonl'}")


def stage_match(
    work: Path,
    offline: bool,
    match_workers: int = 4,
    llm_provider: str = "openai",
    llm_model: str = "",
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
    allow_deterministic_repair: bool = False,
    fail_on_invalid: bool = True,
    match_few_shot_examples: bool = False,
) -> None:
    model = llm_model or REQUESTED_MATCH_MODEL
    log_info(
        f"{work.name}: match start offline={offline} workers={match_workers} llm_provider={llm_provider} "
        f"llm_model={model} few_shot_examples={match_few_shot_examples}"
    )
    candidates = read_jsonl(work / "candidates.jsonl")
    matches = llm_match(
        candidates,
        offline=offline,
        max_workers=match_workers,
        progress_logger=log_info,
        provider=llm_provider,
        model=model,
        google_project=google_project,
        google_location=google_location,
        google_credentials=google_credentials,
        allow_deterministic_repair=allow_deterministic_repair,
        fail_on_invalid=fail_on_invalid,
        few_shot_examples=match_few_shot_examples,
    )
    if allow_deterministic_repair:
        matches = repair_matches_with_candidates(matches, candidates)
    write_json(work / "matches.json", matches_to_json(matches))
    log_info(f"{work.name}: match complete matches={len(matches)}")


def cmd_match(args: argparse.Namespace) -> None:
    stage_match(
        Path(args.work),
        args.offline,
        match_workers=args.match_workers,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        google_project=args.google_project,
        google_location=args.google_location,
        google_credentials=args.google_credentials,
        allow_deterministic_repair=args.allow_deterministic_match_repair,
        fail_on_invalid=not args.allow_invalid_match_fallback,
        match_few_shot_examples=_fewshot_enabled(args, "match"),
    )
    print(f"Wrote matches to {Path(args.work) / 'matches.json'}")


def stage_fol(
    scenario: str,
    root: Path,
    work: Path,
    offline: bool = False,
    use_llm: bool = True,
    llm_provider: str = "openai",
    llm_model: str = "",
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
    fol_object_evidence: bool = False,
    fol_targeted_object_repair: bool = False,
    fol_few_shot_examples: bool = False,
    fol_repair_rounds: int = 1,
    fol_repair_round2: bool = False,
    fol_repair_mode: str = "standard",
    fol_repair_context: str = "global",
    fol_repair_max_issues_per_prompt: int = 8,
    fol_repair_allow_drop: bool = True,
    fol_repair_preservation_gate: bool = False,
    fol_batching: str = "none",
    fol_batch_max_matches: int | None = None,
    fol_batch_max_tokens: int = 0,
    fol_batch_overlap_strategy: str = "fk_neighbors",
    attribute_coverage: bool = False,
    attribute_coverage_validation: bool = False,
    attribute_coverage_repair: bool = False,
    allow_weak_object_links: bool = False,
    pattern_first_fgf: bool = False,
    pattern_candidates: int = 1,
    pattern_repair_budget_chars: int = 24000,
    pattern_selector_context: str = "batched",
    safe_domain_range_type_completion: bool = False,
    pattern_preservation_mode: str = "conservative",
) -> None:
    model = llm_model or REQUESTED_CODE_MODEL
    legacy_attribute_coverage = bool(attribute_coverage)
    attribute_coverage_validation = bool(attribute_coverage_validation or attribute_coverage_repair or attribute_coverage)
    attribute_coverage_repair = bool(attribute_coverage_repair)
    log_info(
        f"{scenario}: fol start offline={offline} use_llm={use_llm} llm_provider={llm_provider} llm_model={model} "
        f"object_evidence={fol_object_evidence} targeted_object_repair={fol_targeted_object_repair} "
        f"few_shot_examples={fol_few_shot_examples} repair_rounds={fol_repair_rounds} "
        f"repair_mode={fol_repair_mode} repair_context={fol_repair_context} batching={fol_batching} "
        f"preservation_gate={fol_repair_preservation_gate} "
        f"attribute_coverage_validation={attribute_coverage_validation} attribute_coverage_repair={attribute_coverage_repair} "
        f"pattern_first_fgf={pattern_first_fgf} pattern_preservation_mode={pattern_preservation_mode}"
    )
    src = scenario_dir(root, scenario)
    tables = parse_sql_dump(src / "dump.sql")
    data = parse_copy_data(src / "dump.sql")
    matches = read_json(work / "matches.json")["matches"]
    if use_llm and not offline:
        candidates = read_jsonl(work / "candidates.jsonl")
        matches, validation_report = reask_suspicious_matches(
            matches,
            candidates,
            tables,
            provider=llm_provider,
            model=model,
            google_project=google_project,
            google_location=google_location,
            google_credentials=google_credentials,
        )
        write_json(work / "match_validation_report.json", {"rechecked": validation_report})
        if validation_report:
            write_json(work / "matches.json", {"matches": matches})
        target_records = read_jsonl(work / "target_records.jsonl")
        discriminator_rows = build_discriminator_candidate_rows(tables, data.rows, target_records, matches, k=16)
        existing_source_ids = {str(match.get("source_id", "")) for match in matches}
        discriminator_rows = [
            row for row in discriminator_rows if str(row.get("source", {}).get("id", "")) not in existing_source_ids
        ]
        if discriminator_rows:
            write_jsonl(work / "discriminator_candidates.jsonl", discriminator_rows)
            discriminator_matches = llm_discriminator_matches(
                discriminator_rows,
                provider=llm_provider,
                model=model,
                google_project=google_project,
                google_location=google_location,
                google_credentials=google_credentials,
            )
            write_json(work / "discriminator_matches.json", {"matches": discriminator_matches})
            matches = matches + discriminator_matches
            write_json(work / "matches.json", {"matches": matches})
            log_info(f"{scenario}: discriminator matching complete rows={len(discriminator_rows)}")
        object_link_evidence: dict[str, Any] | None = None
        if fol_object_evidence:
            object_link_evidence = build_object_link_evidence(
                matches,
                tables,
                data,
                allow_weak_object_links=allow_weak_object_links,
            )
            write_json(work / "object_link_evidence.json", object_link_evidence)
            log_info(f"{scenario}: object evidence built {object_evidence_summary(object_link_evidence)}")
        if pattern_first_fgf:
            if pattern_selector_context != "batched":
                raise ValueError("Stage F pattern selector only supports batched context")
            raw_fol = _pattern_first_fol_stage(
                scenario=scenario,
                work=work,
                tables=tables,
                data=data,
                matches=matches,
                offline=offline,
                provider=llm_provider,
                model=model,
                google_project=google_project,
                google_location=google_location,
                google_credentials=google_credentials,
                pattern_candidates=pattern_candidates,
                budget_chars=pattern_repair_budget_chars,
                safe_domain_range_type_completion=safe_domain_range_type_completion,
            )
        else:
            raw_fol = llm_fol(
                matches,
                tables,
                offline=offline,
                provider=llm_provider,
                model=model,
                google_project=google_project,
                google_location=google_location,
                google_credentials=google_credentials,
                object_link_evidence=object_link_evidence,
                few_shot_examples=fol_few_shot_examples,
                fol_batching=fol_batching,
                fol_batch_max_matches=fol_batch_max_matches,
                fol_batch_max_tokens=fol_batch_max_tokens,
                fol_batch_overlap_strategy=fol_batch_overlap_strategy,
            )
        fol = validate_fol(raw_fol, tables)
        fol["generation"] = raw_fol.get(
            "generation",
            {
                "source": "llm",
                "provider": llm_provider,
                "model": model,
                "prompt_version": "fgf_fol_v2_object_evidence" if object_link_evidence else "fgf_fol_v1_llm_only",
                "few_shot_examples": bool(fol_few_shot_examples),
                "fol_batching": fol_batching,
            },
        )
        initial_hygiene_report: dict[str, Any] = {}
        if fol_repair_preservation_gate:
            fol, initial_hygiene_report = _canonicalize_fol_target_uris(fol)
            fol = validate_fol(fol, tables)
            fol["generation"] = raw_fol.get("generation", fol.get("generation", {}))
        issues, attribute_report = _all_fol_issues(
            fol,
            matches,
            tables,
            object_link_evidence=object_link_evidence,
            include_attribute_coverage=legacy_attribute_coverage,
        )
        if legacy_attribute_coverage:
            write_json(work / "attribute_coverage_summary.json", attribute_report)
        repair_report: dict[str, Any] = {
            "issues_before": issues,
            "repair_attempted": False,
            "issues_after": [],
            "original_rule_counts": _fol_rule_counts(fol),
            "object_evidence_enabled": bool(object_link_evidence),
            "targeted_object_repair_enabled": bool(fol_targeted_object_repair),
            "few_shot_examples_enabled": bool(fol_few_shot_examples),
            "object_evidence_summary": object_evidence_summary(object_link_evidence) if object_link_evidence else {},
            "fol_repair_rounds": fol_repair_rounds,
            "fol_repair_mode": fol_repair_mode,
            "fol_repair_context": fol_repair_context,
            "fol_repair_preservation_gate": bool(fol_repair_preservation_gate),
            "initial_target_uri_hygiene": initial_hygiene_report,
            "fol_batching": fol_batching,
            "attribute_coverage_enabled": bool(legacy_attribute_coverage),
            "attribute_coverage_validation_enabled": bool(attribute_coverage_validation),
            "attribute_coverage_repair_enabled": bool(attribute_coverage_repair),
            "attribute_coverage": attribute_report,
            "allow_weak_object_links": bool(allow_weak_object_links),
            "pattern_first_fgf": bool(pattern_first_fgf),
            "pattern_candidates": int(pattern_candidates or 1),
        }
        if fol_repair_preservation_gate:
            write_json(work / "fol_rules_before_repair.json", fol)
            write_json(
                work / "fol_validation_report_before_repair.json",
                {
                    "issues_before_repair": issues,
                    "issues_by_type_before_repair": issue_type_counts(issues),
                    "rule_counts_before_repair": _fol_rule_counts(fol),
                    "target_uri_hygiene": initial_hygiene_report,
                    "repair_preservation_gate": True,
                },
            )
        if fol_repair_mode == "round2_only":
            try:
                fol, final_issues, single_summary = _run_fol_single_round2_style_repair(
                    work=work,
                    fol=fol,
                    matches=matches,
                    tables=tables,
                    issues_before_repair=issues,
                    object_link_evidence=object_link_evidence,
                    provider=llm_provider,
                    model=model,
                    google_project=google_project,
                    google_location=google_location,
                    google_credentials=google_credentials,
                    max_issues=fol_repair_max_issues_per_prompt,
                    allow_drop=fol_repair_allow_drop,
                    include_attribute_coverage=legacy_attribute_coverage,
                    preservation_gate=fol_repair_preservation_gate,
                )
                repair_report["repair_attempted"] = bool(issues)
                repair_report["repair_mode"] = "round2_only"
                repair_report["single_round2_style"] = {
                    key: value
                    for key, value in single_summary.items()
                    if key not in {"repair_response"}
                }
                repair_report["issues_after"] = final_issues
            except Exception as exc:
                repair_report["repair_attempted"] = bool(issues)
                repair_report["repair_mode"] = "round2_only"
                repair_report["repair_accepted"] = False
                repair_report["fallback_used"] = True
                repair_report["repair_error"] = str(exc)
                repair_report["issues_after"] = issues
                log_info(f"{scenario}: single round2-style fol repair failed; keeping post-batching FOL error={exc}")
            if attribute_coverage_validation:
                fol, attribute_stage = _run_attribute_coverage_stage(
                    work=work,
                    fol=fol,
                    matches=matches,
                    tables=tables,
                    data=data,
                    repair_enabled=attribute_coverage_repair,
                    offline=offline,
                    provider=llm_provider,
                    model=model,
                    google_project=google_project,
                    google_location=google_location,
                    google_credentials=google_credentials,
                )
                repair_report["attribute_coverage_stage"] = attribute_stage
            if fol_repair_preservation_gate:
                fol, final_hygiene_report = _canonicalize_fol_target_uris(fol)
                fol = validate_fol(fol, tables)
                final_issues, _final_attribute_report = _all_fol_issues(
                    fol,
                    matches,
                    tables,
                    object_link_evidence=object_link_evidence,
                    include_attribute_coverage=legacy_attribute_coverage,
                )
                repair_report["final_target_uri_hygiene"] = final_hygiene_report
                repair_report["issues_after"] = final_issues
            write_json(work / "fol_validation_report.json", repair_report)
            write_json(work / "fol.json", fol)
            write_json(work / "mapping_diagnostics.json", mapping_diagnostics(matches, fol, tables))
            log_info(f"{scenario}: fol complete diagnostics={work / 'mapping_diagnostics.json'}")
            return
        if issues:
            repair_report["repair_attempted"] = True
            try:
                object_issues = [issue for issue in issues if str(issue.get("rule_id", "")).startswith("object:")]
                if fol_repair_context == "batched" and fol_batching != "none":
                    if fol_targeted_object_repair and object_link_evidence and object_issues:
                        repair_report["targeted_object_repair_context_control"] = (
                            "delegated_to_contextual_round2"
                        )
                    fol, issues, contextual_report = _run_fol_standard_repair(
                        work=work,
                        fol=fol,
                        matches=matches,
                        tables=tables,
                        issues=issues,
                        object_link_evidence=object_link_evidence,
                        provider=llm_provider,
                        model=model,
                        google_project=google_project,
                        google_location=google_location,
                        google_credentials=google_credentials,
                        include_attribute_coverage=legacy_attribute_coverage,
                        repair_context="batched",
                        max_issues=fol_repair_max_issues_per_prompt,
                        allow_drop=fol_repair_allow_drop,
                        preservation_gate=fol_repair_preservation_gate,
                    )
                    repair_report.update(contextual_report)
                    repaired_raw = None
                elif fol_targeted_object_repair and object_link_evidence and object_issues:
                    repair_report["repair_mode"] = "targeted_object"
                    repaired_raw = llm_repair_object_fol(
                        fol,
                        matches,
                        tables,
                        object_issues,
                        object_link_evidence,
                        provider=llm_provider,
                        model=model,
                        google_project=google_project,
                        google_location=google_location,
                        google_credentials=google_credentials,
                    )
                elif fol_targeted_object_repair and object_link_evidence:
                    repair_report["repair_mode"] = "targeted_object_skipped_no_object_issues"
                    repair_report["repair_attempted"] = False
                    repair_report["issues_after"] = issues
                    repair_report["repair_accepted"] = False
                    repaired_raw = None
                else:
                    repair_report["repair_mode"] = "general"
                    repaired_raw = llm_repair_fol(
                        fol,
                        matches,
                        tables,
                        issues,
                        provider=llm_provider,
                        model=model,
                        google_project=google_project,
                        google_location=google_location,
                        google_credentials=google_credentials,
                    )
                if repaired_raw is not None:
                    hygiene_report: dict[str, Any] = {}
                    if fol_repair_preservation_gate:
                        repaired_raw, hygiene_report = _canonicalize_fol_target_uris(repaired_raw)
                    repaired = validate_fol(repaired_raw, tables)
                    repaired["generation"] = {
                        **fol.get("generation", {}),
                        "repair": repaired_raw.get("generation", {}),
                    }
                    repaired_issues, repaired_attribute_report = _all_fol_issues(
                        repaired,
                        matches,
                        tables,
                        object_link_evidence=object_link_evidence,
                        include_attribute_coverage=legacy_attribute_coverage,
                    )
                    repair_report["issues_after"] = repaired_issues
                    if legacy_attribute_coverage:
                        repair_report["attribute_coverage_after_round1"] = repaired_attribute_report
                    repair_report["repaired_rule_counts"] = _fol_rule_counts(repaired)
                    if repair_report.get("repair_mode") == "targeted_object":
                        repair_accepted, rejection_reasons = _targeted_object_repair_acceptance_decision(
                            fol,
                            repaired,
                            issues,
                            repaired_issues,
                            matches,
                        )
                    elif fol_repair_preservation_gate:
                        repair_accepted, rejection_reasons, preservation_report = _fol_preservation_gate_decision(
                            fol,
                            repaired,
                            issues,
                            repaired_issues,
                            matches,
                        )
                        preservation_report["target_uri_hygiene"] = hygiene_report
                        repair_report["preservation_gate"] = preservation_report
                    else:
                        repair_accepted, rejection_reasons = _fol_repair_acceptance_decision(
                            fol,
                            repaired,
                            issues,
                            repaired_issues,
                            matches,
                        )
                    repair_report["repair_rejection_reasons"] = rejection_reasons
                    if repair_accepted:
                        fol = repaired
                        repair_report["repair_accepted"] = True
                        issues = repaired_issues
                    else:
                        repair_report["repair_accepted"] = False
                        repair_report["rejected_repair_issues"] = repaired_issues
                        repair_report["issues_after"] = issues
            except Exception as exc:
                repair_report["issues_after"] = issues
                repair_report["repair_accepted"] = False
                repair_report["repair_error"] = str(exc)
                log_info(f"{scenario}: fol repair failed; keeping original LLM FOL error={exc}")
        else:
            repair_report["issues_after"] = issues
        current_round1_issues = list(issues)
        repair_report["issues_after_round1_kept"] = current_round1_issues
        if (fol_repair_round2 or fol_repair_rounds >= 2) and current_round1_issues:
            try:
                fol, final_issues, round2_summary = _run_fol_repair_round2(
                    work=work,
                    fol=fol,
                    matches=matches,
                    tables=tables,
                    issues_before=repair_report.get("issues_before", []),
                    issues_after_round1=current_round1_issues,
                    object_link_evidence=object_link_evidence,
                    provider=llm_provider,
                    model=model,
                    google_project=google_project,
                    google_location=google_location,
                    google_credentials=google_credentials,
                    max_issues=fol_repair_max_issues_per_prompt,
                    allow_drop=fol_repair_allow_drop,
                    include_attribute_coverage=legacy_attribute_coverage,
                    preservation_gate=fol_repair_preservation_gate,
                )
                repair_report["round2"] = {
                    key: value
                    for key, value in round2_summary.items()
                    if key not in {"repair_round2_response"}
                }
                repair_report["issues_after"] = final_issues
            except Exception as exc:
                repair_report["round2"] = {
                    "repair_round2_attempted": True,
                    "repair_round2_accepted": False,
                    "repair_round2_error": str(exc),
                }
                log_info(f"{scenario}: fol round2 repair failed; keeping round1 FOL error={exc}")
        if attribute_coverage_validation:
            fol, attribute_stage = _run_attribute_coverage_stage(
                work=work,
                fol=fol,
                matches=matches,
                tables=tables,
                data=data,
                repair_enabled=attribute_coverage_repair,
                offline=offline,
                provider=llm_provider,
                model=model,
                google_project=google_project,
                google_location=google_location,
                google_credentials=google_credentials,
            )
            repair_report["attribute_coverage_stage"] = attribute_stage
        if fol_repair_preservation_gate:
            fol, final_hygiene_report = _canonicalize_fol_target_uris(fol)
            fol = validate_fol(fol, tables)
            final_issues, _final_attribute_report = _all_fol_issues(
                fol,
                matches,
                tables,
                object_link_evidence=object_link_evidence,
                include_attribute_coverage=legacy_attribute_coverage,
            )
            repair_report["final_target_uri_hygiene"] = final_hygiene_report
            repair_report["issues_after"] = final_issues
        write_json(work / "fol_validation_report.json", repair_report)
    else:
        fol = validate_fol(matches_to_fol(matches, tables), tables)
        fol["generation"] = {"source": "deterministic", "reason": "explicit offline or deterministic FOL mode"}
        if attribute_coverage_validation:
            fol, _attribute_stage = _run_attribute_coverage_stage(
                work=work,
                fol=fol,
                matches=matches,
                tables=tables,
                data=data,
                repair_enabled=False,
                offline=True,
                provider=llm_provider,
                model=model,
                google_project=google_project,
                google_location=google_location,
                google_credentials=google_credentials,
            )
    write_json(work / "fol.json", fol)
    write_json(work / "mapping_diagnostics.json", mapping_diagnostics(matches, fol, tables))
    log_info(f"{scenario}: fol complete diagnostics={work / 'mapping_diagnostics.json'}")


def cmd_fol(args: argparse.Namespace) -> None:
    stage_fol(
        args.scenario,
        Path(args.rodi_root),
        Path(args.work),
        offline=args.offline,
        use_llm=not args.deterministic_fol,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        google_project=args.google_project,
        google_location=args.google_location,
        google_credentials=args.google_credentials,
        fol_object_evidence=bool(getattr(args, "fol_object_evidence", False)),
        fol_targeted_object_repair=bool(getattr(args, "fol_targeted_object_repair", False)),
        fol_few_shot_examples=_fewshot_enabled(args, "fol"),
        fol_repair_rounds=int(getattr(args, "fol_repair_rounds", 1)),
        fol_repair_round2=bool(getattr(args, "fol_repair_round2", False)),
        fol_repair_mode=str(getattr(args, "fol_repair_mode", "standard")),
        fol_repair_context=str(getattr(args, "fol_repair_context", "global")),
        fol_repair_max_issues_per_prompt=int(getattr(args, "fol_repair_max_issues_per_prompt", 8)),
        fol_repair_allow_drop=bool(getattr(args, "fol_repair_allow_drop", True)),
        fol_repair_preservation_gate=bool(getattr(args, "fol_repair_preservation_gate", False)),
        fol_batching=str(getattr(args, "fol_batching", "none")),
        fol_batch_max_matches=getattr(args, "fol_batch_max_matches", None),
        fol_batch_max_tokens=int(getattr(args, "fol_batch_max_tokens", 0)),
        fol_batch_overlap_strategy=str(getattr(args, "fol_batch_overlap_strategy", "fk_neighbors")),
        attribute_coverage=bool(getattr(args, "attribute_coverage", False)),
        attribute_coverage_validation=_attribute_coverage_validation_enabled(args),
        attribute_coverage_repair=_attribute_coverage_repair_enabled(args),
        allow_weak_object_links=bool(getattr(args, "allow_weak_object_links", False)),
        pattern_first_fgf=_pattern_first_enabled(args),
        pattern_candidates=int(getattr(args, "pattern_candidates", 1)),
        pattern_repair_budget_chars=int(getattr(args, "pattern_repair_budget_chars", 24000)),
        pattern_selector_context=str(getattr(args, "pattern_selector_context", "batched")),
        safe_domain_range_type_completion=bool(getattr(args, "safe_domain_range_type_completion", False)),
    )
    print(f"Wrote FOL rules to {Path(args.work) / 'fol.json'}")


def _expected_rule_kinds(fol: dict[str, Any]) -> list[str]:
    rules = fol.get("rules", {}) or {}
    return [kind for kind in ("class", "data", "object") if rules.get(kind)]


def _runtime_code_score(code: str, fol: dict[str, Any], log: dict[str, Any], execution_ok: bool) -> tuple[float, float, float, float, float, float]:
    expected_kinds = set(_expected_rule_kinds(fol))
    stats = list((log.get("rule_stats") or {}).values())
    covered_kinds = {
        str(stat.get("kind"))
        for stat in stats
        if int(stat.get("helper_calls", 0) or 0) > 0 and str(stat.get("kind")) in expected_kinds
    }
    helper_coverage = len(covered_kinds) / len(expected_kinds) if expected_kinds else 1.0
    reachable_zero = sum(
        1
        for stat in stats
        if int(stat.get("reachable_rows", 0) or 0) > 0
        and int(stat.get("helper_calls", 0) or 0) > 0
        and int(stat.get("emitted_triples", 0) or 0) == 0
    )
    skipped_reachable = sum(
        1
        for stat in stats
        if int(stat.get("reachable_rows", 0) or 0) > 0 and int(stat.get("helper_calls", 0) or 0) == 0
    )
    invalid = int(log.get("invalid_triple_count", 0) or 0)
    produced = int(log.get("generated_triples", 0) or 0)
    safety_ok = execution_ok and invalid == 0 and skipped_reachable == 0
    return (
        1.0 if safety_ok else 0.0,
        helper_coverage,
        float(produced),
        float(-reachable_zero),
        float(-invalid),
        float(-len(code.splitlines())),
    )


def _summarize_runtime_log(fol: dict[str, Any], log: dict[str, Any]) -> dict[str, Any]:
    stats = list((log.get("rule_stats") or {}).values())
    expected_kinds = set(_expected_rule_kinds(fol))
    covered_kinds = sorted(
        {
            str(stat.get("kind"))
            for stat in stats
            if int(stat.get("helper_calls", 0) or 0) > 0 and str(stat.get("kind")) in expected_kinds
        }
    )
    return {
        "expected_rule_kinds": sorted(expected_kinds),
        "covered_rule_kinds": covered_kinds,
        "generated_triples": int(log.get("generated_triples", 0) or 0),
        "invalid_triple_count": int(log.get("invalid_triple_count", 0) or 0),
        "reachable_zero_output_rules": sum(
            1
            for stat in stats
            if int(stat.get("reachable_rows", 0) or 0) > 0
            and int(stat.get("helper_calls", 0) or 0) > 0
            and int(stat.get("emitted_triples", 0) or 0) == 0
        ),
        "skipped_reachable_rules": sum(
            1
            for stat in stats
            if int(stat.get("reachable_rows", 0) or 0) > 0 and int(stat.get("helper_calls", 0) or 0) == 0
        ),
        "zero_output_samples": [
            {
                "rule_id": stat.get("rule_id"),
                "kind": stat.get("kind"),
                "source_table": stat.get("source_table"),
                "failure_counts": stat.get("failure_counts", {}),
                "failure_samples": stat.get("failure_samples", []),
            }
            for stat in stats
            if int(stat.get("reachable_rows", 0) or 0) > 0
            and int(stat.get("helper_calls", 0) or 0) > 0
            and int(stat.get("emitted_triples", 0) or 0) == 0
        ][:8],
    }


def _run_codegen_candidate(
    *,
    work: Path,
    index: int | str,
    code: str,
    fol: dict[str, Any],
    scenario: str,
    tables: object,
    data: object,
) -> dict[str, Any]:
    candidate_path = work / f"generated_fgf.candidate_{index}.py"
    candidate_path.write_text(code, encoding="utf-8")
    log_path = work / f"generated_fgf.candidate_{index}.materialization_log.json"
    static_ok = True
    execution_ok = False
    error = ""
    runtime_log: dict[str, Any] = {
        "status": "not_run",
        "rule_stats": {},
        "generated_triples": 0,
        "invalid_triple_count": 0,
        "invalid_triples": [],
    }
    try:
        validate_generated_code(code)
    except SandboxViolation as exc:
        static_ok = False
        error = f"{type(exc).__name__}: {exc}"
    if static_ok:
        try:
            _, runtime_log = materialize_graph_with_log(
                scenario,
                tables,  # type: ignore[arg-type]
                data,  # type: ignore[arg-type]
                fol,
                code=code,
                raise_on_invalid=False,
            )
            execution_ok = int(runtime_log.get("invalid_triple_count", 0) or 0) == 0
        except Exception as exc:
            runtime_log["status"] = getattr(exc, "status", "error")
            error = f"{type(exc).__name__}: {exc}"
    runtime_log["candidate_index"] = index
    runtime_log["static_ok"] = static_ok
    runtime_log["execution_ok"] = execution_ok
    if error:
        runtime_log["error"] = error
    write_json(log_path, runtime_log)
    score = _runtime_code_score(code, fol, runtime_log, execution_ok)
    summary = _summarize_runtime_log(fol, runtime_log)
    return {
        "index": index,
        "path": str(candidate_path),
        "materialization_log": str(log_path),
        "static_ok": static_ok,
        "execution_ok": execution_ok,
        "error": error,
        "score": list(score),
        **summary,
    }


def _needs_codegen_repair(record: dict[str, Any]) -> bool:
    return (
        not record.get("static_ok")
        or not record.get("execution_ok")
        or float(record.get("invalid_triple_count", 0) or 0) > 0
        or float(len(record.get("covered_rule_kinds", []))) < float(len(record.get("expected_rule_kinds", [])))
        or int(record.get("skipped_reachable_rules", 0) or 0) > 0
        or int(record.get("reachable_zero_output_rules", 0) or 0) > 0
    )


def _repair_accepted(before: dict[str, Any], after: dict[str, Any]) -> bool:
    if not after.get("static_ok") or not after.get("execution_ok"):
        return False
    before_triples = int(before.get("generated_triples", 0) or 0)
    after_triples = int(after.get("generated_triples", 0) or 0)
    if before_triples and after_triples < before_triples * 0.5:
        return False
    if int(after.get("invalid_triple_count", 0) or 0) > int(before.get("invalid_triple_count", 0) or 0):
        return False
    if int(after.get("skipped_reachable_rules", 0) or 0) > int(before.get("skipped_reachable_rules", 0) or 0):
        return False
    if int(after.get("reachable_zero_output_rules", 0) or 0) > int(before.get("reachable_zero_output_rules", 0) or 0):
        return False
    return tuple(after.get("score", [])) >= tuple(before.get("score", []))


def _codegen_self_consistent(
    work: Path,
    fol: dict[str, Any],
    offline: bool,
    scenario: str | None,
    tables: object | None,
    data: object | None,
    self_consistency: int,
    prompt_version: str,
    llm_provider: str = "openai",
    llm_model: str = "",
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
    codegen_few_shot_examples: bool = False,
) -> tuple[str, dict[str, Any]]:
    runtime_enabled = scenario is not None and tables is not None and data is not None
    rounds = max(1, int(self_consistency or 1))
    if runtime_enabled:
        rounds = max(3, rounds)
    candidates: list[dict[str, Any]] = []
    model = llm_model or REQUESTED_CODE_MODEL
    effective_prompt_version = CODEGEN_FEW_SHOT_PROMPT_VERSION if codegen_few_shot_examples else prompt_version
    for index in range(1, rounds + 1):
        code = llm_codegen(
            fol,
            offline=offline,
            prompt_version=effective_prompt_version,
            provider=llm_provider,
            model=model,
            google_project=google_project,
            google_location=google_location,
            google_credentials=google_credentials,
            few_shot_examples=codegen_few_shot_examples,
        )
        if runtime_enabled:
            record = _run_codegen_candidate(
                work=work,
                index=index,
                code=code,
                fol=fol,
                scenario=str(scenario),
                tables=tables,
                data=data,
            )
        else:
            path = work / f"generated_fgf.candidate_{index}.py"
            path.write_text(code, encoding="utf-8")
            record = {"index": index, "path": str(path), "static_ok": True, "execution_ok": None, "score": [0.0], "code": code}
        record["code"] = code
        candidates.append(record)

    selected = max(candidates, key=lambda row: tuple(row.get("score", [])))
    repair_record: dict[str, Any] | None = None
    if runtime_enabled and _needs_codegen_repair(selected):
        diagnostics = {
            "selected_candidate": {key: value for key, value in selected.items() if key != "code"},
            "repair_constraints": [
                "Use only supplied FOL rules and approved helpers.",
                "Do not invent target URIs, constants, mappings, qpair fixes, or dataset-specific logic.",
                "Repair iteration strategy, rule grouping, or subject/object helper use only.",
            ],
        }
        repair_code = llm_codegen(
            fol,
            offline=offline,
            prompt_version=effective_prompt_version,
            previous_code=str(selected.get("code", "")),
            diagnostics=diagnostics,
            provider=llm_provider,
            model=model,
            google_project=google_project,
            google_location=google_location,
            google_credentials=google_credentials,
            few_shot_examples=codegen_few_shot_examples,
        )
        repair_record = _run_codegen_candidate(
            work=work,
            index="repair_1",
            code=repair_code,
            fol=fol,
            scenario=str(scenario),
            tables=tables,
            data=data,
        )
        repair_record["code"] = repair_code
        candidates.append(repair_record)
        if _repair_accepted(selected, repair_record):
            selected = repair_record

    audit_candidates = [{key: value for key, value in candidate.items() if key != "code"} for candidate in candidates]
    audit = {
        "prompt_version": effective_prompt_version,
        "few_shot_examples": bool(codegen_few_shot_examples),
        "runtime_selection_enabled": runtime_enabled,
        "selected_index": selected.get("index"),
        "repair_attempted": repair_record is not None,
        "repair_accepted": repair_record is not None and selected is repair_record,
        "candidates": audit_candidates,
    }
    write_json(work / "codegen_candidate_scores.json", audit)
    return str(selected.get("code", "")), audit


def stage_codegen(
    work: Path,
    offline: bool,
    scenario: str | None = None,
    tables: object | None = None,
    data: object | None = None,
    codegen_self_consistency: int | None = None,
    prompt_version: str = CODEGEN_PROMPT_VERSION,
    llm_provider: str = "openai",
    llm_model: str = "",
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
    codegen_few_shot_examples: bool = False,
) -> None:
    rounds = codegen_self_consistency
    if rounds is None:
        rounds = int(os.getenv("CODING_FGF_CODEGEN_SELF_CONSISTENCY", "3"))
    model = llm_model or REQUESTED_CODE_MODEL
    effective_prompt_version = CODEGEN_FEW_SHOT_PROMPT_VERSION if codegen_few_shot_examples else prompt_version
    log_info(
        f"{work.name}: codegen start offline={offline} candidates={rounds} prompt={effective_prompt_version} "
        f"llm_provider={llm_provider} llm_model={model} few_shot_examples={codegen_few_shot_examples}"
    )
    fol = read_json(work / "fol.json")
    code, audit = _codegen_self_consistent(
        work,
        fol,
        offline,
        scenario=scenario,
        tables=tables,
        data=data,
        self_consistency=rounds,
        prompt_version=effective_prompt_version,
        llm_provider=llm_provider,
        llm_model=model,
        google_project=google_project,
        google_location=google_location,
        google_credentials=google_credentials,
        codegen_few_shot_examples=codegen_few_shot_examples,
    )
    (work / "generated_fgf.py").write_text(code, encoding="utf-8")
    log_info(f"{work.name}: codegen complete selected={audit.get('selected_index')} repair_accepted={audit.get('repair_accepted')}")


def _codegen_and_materialize_with_round2_only_fallback(
    *,
    scenario: str,
    scenario_work: Path,
    tables: object,
    data: object,
    args: argparse.Namespace,
    offline: bool,
    output: Path,
) -> dict[str, Any]:
    """Run codegen/materialization, restoring pre-repair FOL for guarded repair failures."""

    def _run_once() -> None:
        stage_codegen(
            scenario_work,
            offline=offline,
            scenario=scenario,
            tables=tables,
            data=data,
            codegen_self_consistency=getattr(args, "codegen_self_consistency", None),
            prompt_version=getattr(args, "codegen_prompt_version", CODEGEN_PROMPT_VERSION),
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
            codegen_few_shot_examples=_fewshot_enabled(args, "codegen"),
        )
        materialize_to_file(
            scenario,
            tables,
            data,
            scenario_work / "fol.json",
            output,
            code_path=scenario_work / "generated_fgf.py",
        )

    try:
        _run_once()
        return {"round2_only_runtime_fallback_used": False, "repair_preservation_runtime_fallback_used": False}
    except Exception as exc:
        preservation_gate = bool(getattr(args, "fol_repair_preservation_gate", False))
        if str(getattr(args, "fol_repair_mode", "standard")) != "round2_only" and not preservation_gate:
            raise
        before_path = scenario_work / "fol_rules_before_repair.json"
        if not before_path.exists():
            raise
        rejected_fol_path = scenario_work / "fol_rules_rejected_after_invalid_materialization.json"
        rejected_code_path = scenario_work / "generated_fgf.rejected_round2_only.py"
        rejected_report_path = scenario_work / "fol_round2_only_runtime_fallback.json"
        try:
            current_fol = read_json(scenario_work / "fol.json")
            write_json(rejected_fol_path, current_fol)
        except Exception:
            pass
        try:
            if (scenario_work / "generated_fgf.py").exists():
                rejected_code_path.write_text(
                    (scenario_work / "generated_fgf.py").read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
        except Exception:
            pass

        fallback_fol = read_json(before_path)
        write_json(scenario_work / "fol.json", fallback_fol)
        report = {
            "round2_only_runtime_fallback_used": True,
            "repair_preservation_runtime_fallback_used": preservation_gate,
            "reason": f"{type(exc).__name__}: {exc}",
            "restored_fol": str(before_path),
            "rejected_fol": str(rejected_fol_path),
            "rejected_code": str(rejected_code_path),
        }
        write_json(rejected_report_path, report)
        log_info(
            f"{scenario}: round2_only runtime fallback to unrepaired post-batching FOL "
            f"reason={type(exc).__name__}: {exc}"
        )
        _run_once()
        return report


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def _fol_portfolio_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "fol_portfolio", False))


def _fol_portfolio_arms(args: argparse.Namespace) -> list[str]:
    arms = [arm.strip() for arm in str(getattr(args, "fol_portfolio_arms", "")).split(",") if arm.strip()]
    invalid = [arm for arm in arms if arm not in FOL_PORTFOLIO_ALLOWED_ARMS]
    if invalid:
        raise SystemExit(f"Unknown FOL portfolio arm(s): {', '.join(invalid)}")
    if not arms:
        raise SystemExit("FOL portfolio requires at least one arm")
    return arms


def _fol_portfolio_arm_settings(arm: str) -> dict[str, Any]:
    if arm == "full9_default":
        return {
            "fol_batching": "none",
            "fol_repair_mode": "standard",
            "fol_repair_rounds": 1,
            "fol_repair_context": "global",
            "fol_few_shot_examples": False,
            "attribute_coverage_validation": False,
            "attribute_coverage_repair": False,
            "fol_repair_max_issues_per_prompt": 8,
        }
    if arm == "stage2_hybrid":
        return {
            "fol_batching": "hybrid",
            "fol_repair_mode": "standard",
            "fol_repair_rounds": 2,
            "fol_repair_context": "global",
            "fol_few_shot_examples": False,
            "attribute_coverage_validation": False,
            "attribute_coverage_repair": False,
            "fol_repair_max_issues_per_prompt": 8,
        }
    if arm == "stage2c_round2_only":
        return {
            "fol_batching": "hybrid",
            "fol_repair_mode": "round2_only",
            "fol_repair_rounds": 1,
            "fol_repair_context": "global",
            "fol_few_shot_examples": False,
            "attribute_coverage_validation": False,
            "attribute_coverage_repair": False,
            "fol_repair_max_issues_per_prompt": 8,
        }
    raise ValueError(f"Unknown FOL portfolio arm: {arm}")


def _fol_portfolio_prepare_upstream(
    *,
    scenario: str,
    dev_root: Path,
    scenario_work: Path,
    args: argparse.Namespace,
    offline: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Freeze match/discriminator context once for all FOL portfolio arms."""

    src = scenario_dir(dev_root, scenario)
    tables = parse_sql_dump(src / "dump.sql")
    data = parse_copy_data(src / "dump.sql")
    matches = read_json(scenario_work / "matches.json")["matches"]
    model = args.llm_model or REQUESTED_CODE_MODEL
    if not offline:
        candidates = read_jsonl(scenario_work / "candidates.jsonl")
        matches, validation_report = reask_suspicious_matches(
            matches,
            candidates,
            tables,
            provider=args.llm_provider,
            model=model,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
        )
        write_json(scenario_work / "match_validation_report.json", {"rechecked": validation_report})
        if validation_report:
            write_json(scenario_work / "matches.json", {"matches": matches})
        target_records = read_jsonl(scenario_work / "target_records.jsonl")
        discriminator_rows = build_discriminator_candidate_rows(tables, data.rows, target_records, matches, k=16)
        existing_source_ids = {str(match.get("source_id", "")) for match in matches}
        discriminator_rows = [
            row for row in discriminator_rows if str(row.get("source", {}).get("id", "")) not in existing_source_ids
        ]
        if discriminator_rows:
            write_jsonl(scenario_work / "discriminator_candidates.jsonl", discriminator_rows)
            discriminator_matches = llm_discriminator_matches(
                discriminator_rows,
                provider=args.llm_provider,
                model=model,
                google_project=args.google_project,
                google_location=args.google_location,
                google_credentials=args.google_credentials,
            )
            write_json(scenario_work / "discriminator_matches.json", {"matches": discriminator_matches})
            matches = matches + discriminator_matches
            write_json(scenario_work / "matches.json", {"matches": matches})
            log_info(f"{scenario}: fol portfolio discriminator matching complete rows={len(discriminator_rows)}")
    return matches, tables, data


def _generate_fol_portfolio_arm(
    *,
    scenario: str,
    dev_root: Path,
    scenario_work: Path,
    arm: str,
    matches: list[dict[str, Any]],
    tables: dict[str, Any],
    data: dict[str, Any],
    args: argparse.Namespace,
    offline: bool,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    settings = _fol_portfolio_arm_settings(arm)
    arm_work = ensure_dir(scenario_work / "fol_portfolio" / arm)
    _copy_frozen_upstream_artifacts(scenario_work, arm_work)
    write_json(arm_work / "matches.json", {"matches": matches})
    model = args.llm_model or REQUESTED_CODE_MODEL
    if offline:
        raw_fol = matches_to_fol(matches, tables)
        raw_fol["generation"] = {"source": "explicit_offline", "fol_portfolio_arm": arm}
    else:
        raw_fol = llm_fol(
            matches,
            tables,
            offline=offline,
            provider=args.llm_provider,
            model=model,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
            object_link_evidence=None,
            few_shot_examples=bool(settings["fol_few_shot_examples"]),
            fol_batching=str(settings["fol_batching"]),
            fol_batch_max_matches=getattr(args, "fol_batch_max_matches", None),
            fol_batch_max_tokens=int(getattr(args, "fol_batch_max_tokens", 0)),
            fol_batch_overlap_strategy=str(getattr(args, "fol_batch_overlap_strategy", "fk_neighbors")),
        )
    fol = validate_fol(raw_fol, tables)
    fol["generation"] = raw_fol.get(
        "generation",
        {
            "source": "llm",
            "provider": args.llm_provider,
            "model": model,
            "prompt_version": "fgf_fol_v1_llm_only",
            "fol_portfolio_arm": arm,
            "fol_batching": settings["fol_batching"],
        },
    )
    issues, _attribute_report = _all_fol_issues(
        fol,
        matches,
        tables,
        object_link_evidence=None,
        include_attribute_coverage=False,
    )
    write_json(arm_work / "fol_rules_before_repair.json", fol)
    write_json(
        arm_work / "fol_validation_report_before_repair.json",
        {
            "fol_portfolio_arm": arm,
            "issues_before_repair": issues,
            "issues_by_type_before_repair": issue_type_counts(issues),
            "rule_counts_before_repair": _fol_rule_counts(fol),
            "settings": settings,
        },
    )
    repair_report: dict[str, Any] = {
        "fol_portfolio_arm": arm,
        "settings": settings,
        "issues_before": issues,
        "issues_after": issues,
        "repair_attempted": bool(issues) and not offline,
    }
    if offline:
        repair_report["repair_skipped"] = "explicit_offline"
    elif str(settings["fol_repair_mode"]) == "round2_only":
        fol, issues, single_summary = _run_fol_single_round2_style_repair(
            work=arm_work,
            fol=fol,
            matches=matches,
            tables=tables,
            issues_before_repair=issues,
            object_link_evidence=None,
            provider=args.llm_provider,
            model=model,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
            max_issues=int(settings["fol_repair_max_issues_per_prompt"]),
            allow_drop=bool(getattr(args, "fol_repair_allow_drop", True)),
            include_attribute_coverage=False,
            preservation_gate=False,
        )
        repair_report["single_round2_style"] = {k: v for k, v in single_summary.items() if k != "repair_response"}
    else:
        if issues:
            fol, issues, standard_report = _run_fol_standard_repair(
                work=arm_work,
                fol=fol,
                matches=matches,
                tables=tables,
                issues=issues,
                object_link_evidence=None,
                provider=args.llm_provider,
                model=model,
                google_project=args.google_project,
                google_location=args.google_location,
                google_credentials=args.google_credentials,
                include_attribute_coverage=False,
                preservation_gate=False,
            )
            repair_report.update(standard_report)
        if int(settings["fol_repair_rounds"]) >= 2 and issues:
            try:
                fol, issues, round2_summary = _run_fol_repair_round2(
                    work=arm_work,
                    fol=fol,
                    matches=matches,
                    tables=tables,
                    issues_before=repair_report.get("issues_before", []),
                    issues_after_round1=issues,
                    object_link_evidence=None,
                    provider=args.llm_provider,
                    model=model,
                    google_project=args.google_project,
                    google_location=args.google_location,
                    google_credentials=args.google_credentials,
                    max_issues=int(settings["fol_repair_max_issues_per_prompt"]),
                    allow_drop=bool(getattr(args, "fol_repair_allow_drop", True)),
                    include_attribute_coverage=False,
                    preservation_gate=False,
                )
                repair_report["round2"] = {k: v for k, v in round2_summary.items() if k != "repair_round2_response"}
            except Exception as exc:
                repair_report["round2"] = {
                    "repair_round2_attempted": True,
                    "repair_round2_accepted": False,
                    "repair_round2_error": str(exc),
                }
    repair_report["issues_after"] = issues
    repair_report["issues_by_type_after"] = issue_type_counts(issues)
    write_json(arm_work / "fol_validation_report.json", repair_report)
    write_json(arm_work / "fol.json", fol)
    write_json(arm_work / "mapping_diagnostics.json", mapping_diagnostics(matches, fol, tables))
    return arm_work, fol, issues, repair_report


def _invalid_triple_count_from_log(log: dict[str, Any]) -> int:
    if "invalid_triple_count" in log:
        return int(log.get("invalid_triple_count", 0) or 0)
    invalid = log.get("invalid_triples", [])
    if isinstance(invalid, list):
        return len(invalid)
    if isinstance(invalid, dict):
        return len(invalid)
    return 0


def _role_prefixed_target_count(fol: dict[str, Any]) -> int:
    count = 0
    for kind, target_key in (("class", "class_uri"), ("data", "property_uri"), ("object", "property_uri")):
        for rule in (fol.get("rules", {}) or {}).get(kind, []) or []:
            value = str(rule.get(target_key, ""))
            if any(value.startswith(prefix) for prefix in _TARGET_ROLE_PREFIXES):
                count += 1
    return count


def _fol_portfolio_issue_counts(issues: list[dict[str, Any]]) -> dict[str, int]:
    return issue_type_counts(issues)


def _fol_portfolio_hard_rejections(record: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not record.get("import_exists"):
        reasons.append("missing_import_ttl")
    if int(record.get("generated_triples", 0) or 0) <= 0:
        reasons.append("zero_generated_triples")
    if int(record.get("invalid_triples", 0) or 0) > 0:
        reasons.append("invalid_triples")
    status = str(record.get("materialization_status", "") or "")
    if status and status not in {"success", "ok"}:
        reasons.append("materialization_status_not_success")
    if int(record.get("prefixed_target_uri_count", 0) or 0) > 0:
        reasons.append("role_prefixed_target_uri")
    issue_counts = record.get("issues_by_type", {}) or {}
    if int(issue_counts.get("target_not_in_selected_matches", 0) or 0) > 0:
        reasons.append("invented_target_uri")
    if record.get("codegen_error"):
        reasons.append("codegen_or_materialization_error")
    return reasons


def _fol_portfolio_record(
    *,
    scenario: str,
    arm: str,
    arm_work: Path,
    fol: dict[str, Any],
    issues: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    tables: dict[str, Any],
    data: dict[str, Any],
) -> dict[str, Any]:
    output = arm_work / "import.ttl"
    log_path = output.with_suffix(".materialization_log.json")
    log = read_json(log_path) if log_path.exists() else {}
    code_text = (arm_work / "generated_fgf.py").read_text(encoding="utf-8") if (arm_work / "generated_fgf.py").exists() else ""
    materialization_report: dict[str, Any] = {}
    materialization_summary: dict[str, Any] = {}
    if output.exists() and log:
        materialization_report = materialization_coverage_diagnostics(
            matches,
            fol,
            tables,
            data,
            output,
            log,
            code_text=code_text,
        )
        materialization_summary = materialization_coverage_summary(materialization_report)
        write_json(arm_work / "fol_portfolio_materialization_coverage_report.json", materialization_report)
        write_csv(
            arm_work / "fol_portfolio_materialization_coverage_summary.csv",
            [materialization_summary],
            list(materialization_summary.keys()),
        )
    issue_counts = _fol_portfolio_issue_counts(issues)
    record = {
        "scenario": scenario,
        "arm": arm,
        "work": str(arm_work),
        "import_exists": output.exists(),
        "generated_triples": int(log.get("generated_triples", 0) or 0),
        "invalid_triples": _invalid_triple_count_from_log(log),
        "materialization_status": log.get("status", ""),
        "issues_after": len(issues),
        "issues_by_type": issue_counts,
        "critical_issue_count": sum(int(issue_counts.get(name, 0) or 0) for name in _CRITICAL_FOL_REPAIR_ISSUES),
        "prefixed_target_uri_count": _role_prefixed_target_count(fol),
        "rule_counts": _fol_rule_counts(fol),
        "selected_targets_with_emission": int(materialization_summary.get("selected_targets_with_emission", 0) or 0),
        "zero_emission_selected_targets": int(materialization_summary.get("zero_emission_selected_targets", 0) or 0),
        "unknown_reference_issues": int(materialization_summary.get("unknown_reference_issues", 0) or 0),
        "overbroad_issues": int(materialization_summary.get("overbroad_issues", 0) or 0),
        "fk_like_literal_issues": int(materialization_summary.get("fk_like_literal_issues", 0) or 0),
        "zero_rule_issues": int(materialization_summary.get("zero_rule_issues", 0) or 0),
    }
    record["hard_rejections"] = _fol_portfolio_hard_rejections(record)
    write_json(arm_work / "fol_portfolio_candidate_report.json", record)
    return record


def _median_int(values: list[int]) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return int(round((ordered[mid - 1] + ordered[mid]) / 2))


def _score_fol_portfolio_candidate(record: dict[str, Any], median_generated: int) -> tuple[int, ...]:
    generated = int(record.get("generated_triples", 0) or 0)
    triple_sanity = 0
    if median_generated > 0:
        # Five-percent buckets keep the selector from overfitting tiny count differences while still
        # penalizing clear collapse/explosion relative to the portfolio's internally observed center.
        triple_sanity = -int(round((abs(generated - median_generated) / max(1, median_generated)) * 20))
    rule_counts = record.get("rule_counts", {}) or {}
    rule_kind_coverage = sum(1 for kind in ("class", "data", "object") if int(rule_counts.get(kind, 0) or 0) > 0)
    priority = FOL_PORTFOLIO_TIEBREAK_PRIORITY.get(str(record.get("arm", "")), 99)
    return (
        triple_sanity,
        -int(record.get("critical_issue_count", 0) or 0),
        int(record.get("selected_targets_with_emission", 0) or 0),
        -int(record.get("zero_emission_selected_targets", 0) or 0),
        -int(record.get("unknown_reference_issues", 0) or 0),
        -int(record.get("overbroad_issues", 0) or 0),
        -int(record.get("fk_like_literal_issues", 0) or 0),
        -int(record.get("zero_rule_issues", 0) or 0),
        rule_kind_coverage,
        -int(record.get("issues_after", 0) or 0),
        -priority,
    )


def _select_fol_portfolio_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if not record.get("hard_rejections")]
    if not valid:
        raise RuntimeError("FOL portfolio produced no internally valid candidates")
    median_generated = _median_int([int(record.get("generated_triples", 0) or 0) for record in valid])
    for record in records:
        record["median_generated_triples"] = median_generated
        record["selector_score"] = list(_score_fol_portfolio_candidate(record, median_generated))
    return max(valid, key=lambda record: tuple(record.get("selector_score", [])))


def _copy_selected_fol_portfolio_artifacts(selected_work: Path, scenario_work: Path) -> None:
    for name in (
        "fol.json",
        "fol_validation_report.json",
        "mapping_diagnostics.json",
        "generated_fgf.py",
        "codegen_candidate_scores.json",
        "import.ttl",
        "import.materialization_log.json",
    ):
        _copy_if_exists(selected_work / name, scenario_work / name)


def _run_fol_portfolio(*, scenario, dev_root, scenario_work, args, offline):
    arms = _fol_portfolio_arms(args)
    if str(getattr(args, "fol_portfolio_selector", "internal_materialization")) != "internal_materialization":
        raise SystemExit("Only internal_materialization FOL portfolio selector is supported")
    matches, tables, data = _fol_portfolio_prepare_upstream(
        scenario=scenario, dev_root=dev_root, scenario_work=scenario_work, args=args, offline=offline)
    records = []
    for arm in arms:
        arm_work = scenario_work / "fol_portfolio" / arm
        try:
            arm_work, fol, issues, repair_report = _generate_fol_portfolio_arm(
                scenario=scenario, dev_root=dev_root, scenario_work=scenario_work, arm=arm,
                matches=matches, tables=tables, data=data, args=args, offline=offline)
            arm_args = argparse.Namespace(**vars(args))
            arm_args.fol_repair_mode = _fol_portfolio_arm_settings(arm)["fol_repair_mode"]
            arm_args.fewshot = "none"
            arm_args.codegen_few_shot_examples = False
            runtime_fallback = _codegen_and_materialize_with_round2_only_fallback(
                scenario=scenario, scenario_work=arm_work, tables=tables, data=data,
                args=arm_args, offline=offline, output=arm_work / "import.ttl")
            final_fol = read_json(arm_work / "fol.json")
            record = _fol_portfolio_record(
                scenario=scenario, arm=arm, arm_work=arm_work, fol=final_fol,
                issues=fol_validation_issues(final_fol, matches, tables),
                matches=matches, tables=tables, data=data)
            record["repair_report"] = {k: v for k, v in repair_report.items()
                                       if k not in {"repair_response", "repair_round2_response"}}
            record["runtime_fallback"] = runtime_fallback
        except Exception as exc:
            record = {"scenario": scenario, "arm": arm, "work": str(arm_work),
                      "codegen_error": f"{type(exc).__name__}: {exc}",
                      "materialization_status": getattr(exc, "status", "error"),
                      "import_exists": False}
        record["hard_rejections"] = _fol_portfolio_hard_rejections(record)
        ensure_dir(arm_work)
        write_json(arm_work / "fol_portfolio_candidate_report.json", record)
        records.append(record)
    write_json(scenario_work / "fol_portfolio_candidate_report.json", {"candidates": records})
    try:
        selected = _select_fol_portfolio_candidate(records)
    except RuntimeError:
        if getattr(args, "fol_ablation_report", False):
            write_fol_ablation_report(scenario, records, None, None, None, scenario_work / "fol_ablation")
        raise
    selected_work = Path(selected["work"])
    _copy_selected_fol_portfolio_artifacts(selected_work, scenario_work)
    report = {"selector": "internal_materialization",
              "academic_safety": {"forbidden_inputs": list(FOL_PORTFOLIO_FORBIDDEN_SELECTION_INPUTS),
                                  "uses_qpair_or_gold_feedback": False},
              "selected_arm": selected["arm"], "selected_score": selected.get("selector_score", []),
              "candidates": records}
    write_json(scenario_work / "fol_portfolio_candidate_report.json", {"candidates": records})
    write_json(scenario_work / "fol_portfolio_selection_report.json", report)
    write_json(scenario_work / "selected_fol_arm.json",
               {"selected_arm": selected["arm"], "selected_work": str(selected_work),
                "selector_score": selected.get("selector_score", [])})
    # This boundary is deliberately after the immutable selection decision is persisted.
    if getattr(args, "fol_ablation_report", False):
        def sql_exec(sql):
            return execute_sql_psycopg2(sql, dbname=scenario, host=args.db_host,
                port=args.db_port, user=args.db_user, password=args.db_password)
        write_fol_ablation_report(scenario, records, selected["arm"],
            scenario_dir(dev_root, scenario) / "queries",
            None if args.dry_run_db else sql_exec, scenario_work / "fol_ablation")
    return report


def _run_materialization_coverage_stage(
    *,
    scenario: str,
    scenario_work: Path,
    tables: dict[str, Any],
    data: Any,
    args: argparse.Namespace,
    offline: bool,
    output: Path,
) -> dict[str, Any]:
    """Run gold-blind materialization coverage validation/repair before qpair evaluation."""
    if not _materialization_coverage_validation_enabled(args):
        return {"materialization_coverage_validation_enabled": False}

    pass0_dir = ensure_dir(scenario_work / "pass0")
    pass0_output = pass0_dir / "import.ttl"
    pass0_log = pass0_dir / "import.materialization_log.json"
    pass0_fol = pass0_dir / "fol.json"
    pass0_code = pass0_dir / "generated_fgf.py"
    output_log = output.with_suffix(".materialization_log.json")
    _copy_if_exists(output, pass0_output)
    _copy_if_exists(output_log, pass0_log)
    _copy_if_exists(scenario_work / "fol.json", pass0_fol)
    _copy_if_exists(scenario_work / "generated_fgf.py", pass0_code)

    matches = read_json(scenario_work / "matches.json")["matches"]
    fol = read_json(scenario_work / "fol.json")
    code_text = (scenario_work / "generated_fgf.py").read_text(encoding="utf-8") if (scenario_work / "generated_fgf.py").exists() else ""
    runtime_log = read_json(output_log) if output_log.exists() else {}
    report_before = materialization_coverage_diagnostics(
        matches,
        fol,
        tables,
        data,
        output,
        runtime_log,
        code_text=code_text,
    )
    summary_before = materialization_coverage_summary(report_before)
    write_json(scenario_work / "materialization_coverage_report.json", report_before)
    write_csv(scenario_work / "materialization_coverage_summary.csv", [summary_before], list(summary_before.keys()))

    decisions: dict[str, Any] = {
        "materialization_coverage_validation_enabled": True,
        "materialization_coverage_repair_enabled": _materialization_coverage_repair_enabled(args),
        "pass0": {
            "import_ttl": str(pass0_output),
            "materialization_log": str(pass0_log),
            "fol": str(pass0_fol),
            "code": str(pass0_code),
            "summary": summary_before,
            "score": list(internal_materialization_score(report_before)),
        },
        "repair_candidates": [],
        "selected_candidate": None,
        "repair_accepted": False,
        "fallback_used": False,
    }
    if not _materialization_coverage_repair_enabled(args):
        write_json(scenario_work / "materialization_repair_decisions.json", decisions)
        return decisions

    if str(getattr(args, "materialization_repair_context", "batched")) != "batched":
        raise ValueError("Stage E materialization repair only supports batched context")

    prompts = build_materialization_repair_prompts(
        report_before,
        fol,
        matches,
        tables,
        data,
        budget_chars=int(getattr(args, "materialization_repair_budget_chars", 24000)),
    )
    write_json(
        scenario_work / "materialization_repair_prompt.json",
        {
            "prompt_version": "fgf_materialization_coverage_v1_repair",
            "budget_chars": int(getattr(args, "materialization_repair_budget_chars", 24000)),
            "prompt_count": len(prompts),
            "prompts": prompts,
        },
    )
    if not prompts:
        decisions["repair_skipped_reason"] = "no_materialization_coverage_issues"
        write_json(scenario_work / "materialization_repair_decisions.json", decisions)
        return decisions

    candidate_count = max(1, int(getattr(args, "materialization_repair_candidates", 1) or 1))
    accepted_candidates: list[dict[str, Any]] = []
    for candidate_index in range(1, candidate_count + 1):
        candidate_repairs: list[dict[str, Any]] = []
        prompt_responses: list[dict[str, Any]] = []
        for prompt_index, prompt_record in enumerate(prompts, start=1):
            if offline:
                response = {"repairs": [], "generation": {"source": "offline", "prompt_index": prompt_index}}
            else:
                response = llm_repair_materialization_coverage(
                    str(prompt_record.get("prompt", "")),
                    provider=args.llm_provider,
                    model=args.llm_model,
                    google_project=args.google_project,
                    google_location=args.google_location,
                    google_credentials=args.google_credentials,
                )
            prompt_responses.append(
                {
                    "prompt_index": prompt_index,
                    "issue_ids": prompt_record.get("issue_ids", []),
                    "char_count": prompt_record.get("char_count", 0),
                    "repair_count": len(response.get("repairs", []) or []),
                    "generation": response.get("generation", {}),
                }
            )
            candidate_repairs.extend(response.get("repairs", []) or [])

        candidate_fol, repair_summary = apply_materialization_coverage_repairs(
            fol,
            candidate_repairs,
            matches,
            tables,
        )
        candidate_work = ensure_dir(scenario_work / f"materialization_repair_candidate_{candidate_index}")
        write_json(candidate_work / "fol.json", candidate_fol)
        write_json(candidate_work / "materialization_repair_responses.json", {"responses": prompt_responses, "repairs": candidate_repairs})
        candidate_output = candidate_work / "import.ttl"
        candidate_error = ""
        candidate_report: dict[str, Any] = {}
        accepted = False
        rejection_reasons: list[str] = []
        try:
            stage_codegen(
                candidate_work,
                offline=offline,
                scenario=scenario,
                tables=tables,
                data=data,
                codegen_self_consistency=getattr(args, "codegen_self_consistency", None),
                prompt_version=getattr(args, "codegen_prompt_version", CODEGEN_PROMPT_VERSION),
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
                google_project=args.google_project,
                google_location=args.google_location,
                google_credentials=args.google_credentials,
                codegen_few_shot_examples=_fewshot_enabled(args, "codegen"),
            )
            materialize_to_file(
                scenario,
                tables,
                data,
                candidate_work / "fol.json",
                candidate_output,
                code_path=candidate_work / "generated_fgf.py",
            )
            candidate_log = read_json(candidate_output.with_suffix(".materialization_log.json"))
            candidate_code = (candidate_work / "generated_fgf.py").read_text(encoding="utf-8")
            candidate_report = materialization_coverage_diagnostics(
                matches,
                candidate_fol,
                tables,
                data,
                candidate_output,
                candidate_log,
                code_text=candidate_code,
            )
            accepted, rejection_reasons = materialization_repair_accepted(report_before, candidate_report)
        except Exception as exc:
            candidate_error = f"{type(exc).__name__}: {exc}"
            accepted = False
            rejection_reasons = ["candidate_materialization_error"]

        candidate_record = {
            "candidate_index": candidate_index,
            "work": str(candidate_work),
            "repair_summary": repair_summary,
            "prompt_responses": prompt_responses,
            "accepted_by_internal_criteria": accepted,
            "rejection_reasons": rejection_reasons,
            "error": candidate_error,
            "summary": materialization_coverage_summary(candidate_report) if candidate_report else {},
            "score": list(internal_materialization_score(candidate_report)) if candidate_report else [],
        }
        write_json(candidate_work / "materialization_coverage_report.json", candidate_report)
        write_json(candidate_work / "materialization_repair_candidate_summary.json", candidate_record)
        decisions["repair_candidates"].append(candidate_record)
        if accepted:
            accepted_candidates.append(candidate_record)
            if not bool(getattr(args, "internal_rerank_repair_candidates", False)):
                break

    selected: dict[str, Any] | None = None
    if accepted_candidates:
        if bool(getattr(args, "internal_rerank_repair_candidates", False)):
            selected = max(accepted_candidates, key=lambda row: tuple(row.get("score", [])))
        else:
            selected = accepted_candidates[0]

    if selected:
        selected_work = Path(str(selected["work"]))
        _copy_if_exists(selected_work / "fol.json", scenario_work / "fol.json")
        _copy_if_exists(selected_work / "generated_fgf.py", scenario_work / "generated_fgf.py")
        _copy_if_exists(selected_work / "generated_fgf.py", scenario_work / "code_after_materialization_repair.py")
        _copy_if_exists(selected_work / "import.ttl", output)
        _copy_if_exists(selected_work / "import.materialization_log.json", output_log)
        _copy_if_exists(selected_work / "fol.json", scenario_work / "fol_after_materialization_repair.json")
        decisions["selected_candidate"] = selected.get("candidate_index")
        decisions["repair_accepted"] = True
        decisions["selected_by_internal_rerank"] = bool(getattr(args, "internal_rerank_repair_candidates", False))
    else:
        _copy_if_exists(pass0_fol, scenario_work / "fol.json")
        _copy_if_exists(pass0_code, scenario_work / "generated_fgf.py")
        _copy_if_exists(pass0_output, output)
        _copy_if_exists(pass0_log, output_log)
        decisions["fallback_used"] = True
        decisions["fallback_reason"] = "no_candidate_passed_internal_materialization_criteria"

    final_log = read_json(output_log) if output_log.exists() else {}
    final_code = (scenario_work / "generated_fgf.py").read_text(encoding="utf-8") if (scenario_work / "generated_fgf.py").exists() else ""
    final_fol = read_json(scenario_work / "fol.json") if (scenario_work / "fol.json").exists() else fol
    final_report = materialization_coverage_diagnostics(matches, final_fol, tables, data, output, final_log, code_text=final_code)
    decisions["final_summary"] = materialization_coverage_summary(final_report)
    write_json(scenario_work / "materialization_coverage_report_after.json", final_report)
    write_json(scenario_work / "materialization_repair_decisions.json", decisions)
    return decisions


def _run_pattern_materialization_stage(
    *,
    scenario: str,
    scenario_work: Path,
    tables: dict[str, Any],
    data: Any,
    args: argparse.Namespace,
    offline: bool,
    output: Path,
) -> dict[str, Any]:
    if not _pattern_first_enabled(args) or not _pattern_materialization_validation_enabled(args):
        return {"pattern_materialization_validation_enabled": False}
    matches = read_json(scenario_work / "matches.json")["matches"]
    output_log = output.with_suffix(".materialization_log.json")
    pass0_fol = read_json(scenario_work / "fol.json")
    pass0_code = (scenario_work / "generated_fgf.py").read_text(encoding="utf-8") if (scenario_work / "generated_fgf.py").exists() else ""
    pass0_log = read_json(output_log) if output_log.exists() else {}
    pass0_report = materialization_coverage_diagnostics(matches, pass0_fol, tables, data, output, pass0_log, code_text=pass0_code)
    pass0_summary = materialization_coverage_summary(pass0_report)
    write_json(scenario_work / "pattern_materialization_validation_report.json", pass0_report)
    write_csv(scenario_work / "pattern_materialization_validation_summary.csv", [pass0_summary], list(pass0_summary.keys()))
    compiler_report = read_json(scenario_work / "pattern_compiler_report.json") if (scenario_work / "pattern_compiler_report.json").exists() else {}
    uri_keys = read_json(scenario_work / "uri_key_inference.json") if (scenario_work / "uri_key_inference.json").exists() else {}
    selected_score = tuple(pattern_selection_internal_score({"compiler_report": compiler_report, "uri_key_inference": uri_keys}, pass0_report))
    decisions: dict[str, Any] = {
        "pattern_materialization_validation_enabled": True,
        "internal_rerank_pattern_candidates": bool(getattr(args, "internal_rerank_pattern_candidates", False)),
        "pass0": {"candidate_index": 1, "summary": pass0_summary, "score": list(selected_score)},
        "candidates": [],
        "selected_candidate": 1,
        "selection_metric": "gold_blind_pattern_internal_diagnostics",
    }
    if not getattr(args, "internal_rerank_pattern_candidates", False) or int(getattr(args, "pattern_candidates", 1) or 1) <= 1:
        write_json(scenario_work / "pattern_candidate_rerank_report.json", decisions)
        return decisions

    selected_record = decisions["pass0"]
    candidate_root = scenario_work / "pattern_candidate_runs"
    for candidate_dir in sorted(candidate_root.glob("candidate_*")):
        try:
            candidate_index = int(candidate_dir.name.rsplit("_", 1)[1])
        except Exception:
            candidate_index = 0
        if candidate_index <= 1 or not (candidate_dir / "fol.json").exists():
            continue
        stage_codegen(
            candidate_dir,
            offline=offline,
            scenario=scenario,
            tables=tables,
            data=data,
            codegen_self_consistency=getattr(args, "codegen_self_consistency", None),
            prompt_version=getattr(args, "codegen_prompt_version", CODEGEN_PROMPT_VERSION),
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
            codegen_few_shot_examples=_fewshot_enabled(args, "codegen"),
        )
        candidate_output = candidate_dir / "import.ttl"
        materialize_to_file(
            scenario,
            tables,
            data,
            candidate_dir / "fol.json",
            candidate_output,
            code_path=candidate_dir / "generated_fgf.py",
        )
        candidate_log = read_json(candidate_output.with_suffix(".materialization_log.json"))
        candidate_code = (candidate_dir / "generated_fgf.py").read_text(encoding="utf-8")
        candidate_fol = read_json(candidate_dir / "fol.json")
        candidate_report = materialization_coverage_diagnostics(matches, candidate_fol, tables, data, candidate_output, candidate_log, code_text=candidate_code)
        write_json(candidate_dir / "pattern_materialization_validation_report.json", candidate_report)
        compiler = read_json(candidate_dir / "pattern_compiler_report.json") if (candidate_dir / "pattern_compiler_report.json").exists() else {}
        uri_key_report = read_json(candidate_dir / "uri_key_inference.json") if (candidate_dir / "uri_key_inference.json").exists() else {}
        score = tuple(pattern_selection_internal_score({"compiler_report": compiler, "uri_key_inference": uri_key_report}, candidate_report))
        candidate_record = {
            "candidate_index": candidate_index,
            "candidate_dir": str(candidate_dir),
            "summary": materialization_coverage_summary(candidate_report),
            "score": list(score),
        }
        decisions["candidates"].append(candidate_record)
        if score > selected_score:
            selected_score = score
            selected_record = candidate_record
    decisions["selected_candidate"] = selected_record.get("candidate_index")
    if int(selected_record.get("candidate_index", 1) or 1) > 1:
        selected_dir = Path(str(selected_record["candidate_dir"]))
        _copy_if_exists(selected_dir / "fol.json", scenario_work / "fol.json")
        _copy_if_exists(selected_dir / "generated_fgf.py", scenario_work / "generated_fgf.py")
        _copy_if_exists(selected_dir / "import.ttl", output)
        _copy_if_exists(selected_dir / "import.materialization_log.json", output_log)
        for name in (
            "pattern_selection_prompt.json",
            "pattern_selection_decisions.json",
            "pattern_selected_matches.json",
            "uri_key_inference.json",
            "uri_template_consistency_report.json",
            "fol_after_pattern_compilation.json",
            "pattern_compiler_report.json",
        ):
            _copy_if_exists(selected_dir / name, scenario_work / name)
    write_json(scenario_work / "pattern_candidate_rerank_report.json", decisions)
    return decisions


def cmd_codegen(args: argparse.Namespace) -> None:
    stage_codegen(
        Path(args.work),
        args.offline,
        codegen_self_consistency=getattr(args, "codegen_self_consistency", None),
        prompt_version=getattr(args, "codegen_prompt_version", CODEGEN_PROMPT_VERSION),
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        google_project=args.google_project,
        google_location=args.google_location,
        google_credentials=args.google_credentials,
        codegen_few_shot_examples=_fewshot_enabled(args, "codegen"),
    )
    print(f"Wrote generated FGF to {Path(args.work) / 'generated_fgf.py'}")


def cmd_materialize(args: argparse.Namespace) -> None:
    log_info(f"{args.scenario}: materialize start")
    src = scenario_dir(Path(args.rodi_root), args.scenario)
    tables = parse_sql_dump(src / "dump.sql")
    data = parse_copy_data(src / "dump.sql")
    work = Path(args.work)
    output = materialize_to_file(
        args.scenario,
        tables,
        data,
        work / "fol.json",
        Path(args.output),
        code_path=work / "generated_fgf.py",
    )
    log_info(f"{args.scenario}: materialize complete output={output}")
    print(f"Wrote materialized RDF to {output}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    log_info(f"{args.scenario}: evaluate start")
    qdir = scenario_dir(Path(args.rodi_root), args.scenario) / "queries"

    def sql_exec(sql: str) -> list[object]:
        return execute_sql_psycopg2(
            sql,
            dbname=args.db_name or args.scenario,
            host=args.db_host,
            port=args.db_port,
            user=args.db_user,
            password=args.db_password,
        )

    result = evaluate_graph(Path(args.graph), qdir, sql_exec, Path(args.out))
    log_info(f"{args.scenario}: evaluate complete f1={result['f1']:.4f} count={result['count']}")
    print(f"Average F1={result['f1']:.4f} over {result['count']} qpairs")


def cmd_run_dev10(args: argparse.Namespace) -> None:
    work = ensure_dir(Path(args.work))
    _write_leakage_risk_report(work)
    dev_root = work / "devset"
    offline = _live_offline(args)
    log_info(f"devset start scenarios={args.scenarios}")
    create_devset(
        Path(args.rodi_root),
        dev_root,
        scenarios=args.scenarios.split(",") if args.scenarios else PAPER_SCENARIOS,
        fraction=getattr(args, "fraction", 0.1),
        seed=getattr(args, "seed", "coding-fgf-dev10-v1"),
    )
    log_info(f"devset complete root={dev_root}")
    for scenario in (args.scenarios.split(",") if args.scenarios else PAPER_SCENARIOS):
        scenario = scenario.strip()
        scenario_work = ensure_dir(work / scenario)
        stage_verbalize(scenario, dev_root, scenario_work, source_context=_source_context_arg(args))
        stage_index(
            scenario_work,
            args.embedding_model,
            offline=offline,
            embedding_provider=args.embedding_provider,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
        )
        stage_retrieve(
            scenario_work,
            args.embedding_model,
            k=args.k,
            offline=offline,
            embedding_provider=args.embedding_provider,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
            allow_forced_candidates=bool(getattr(args, "allow_forced_candidates", False) or offline),
        )
        stage_match(
            scenario_work,
            offline=offline,
            match_workers=args.match_workers,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
            allow_deterministic_repair=bool(getattr(args, "allow_deterministic_match_repair", False) or offline),
            fail_on_invalid=not bool(getattr(args, "allow_invalid_match_fallback", False)),
            match_few_shot_examples=_fewshot_enabled(args, "match"),
        )
        src = scenario_dir(dev_root, scenario)
        tables = parse_sql_dump(src / "dump.sql")
        data = parse_copy_data(src / "dump.sql")
        output = scenario_work / "import.ttl"
        if _fol_portfolio_enabled(args):
            log_info(f"{scenario}: fol portfolio start arms={getattr(args, 'fol_portfolio_arms', '')}")
            _run_fol_portfolio(
                scenario=scenario,
                dev_root=dev_root,
                scenario_work=scenario_work,
                args=args,
                offline=offline,
            )
        else:
            stage_fol(
                scenario,
                dev_root,
                scenario_work,
                offline=offline,
                use_llm=not bool(getattr(args, "deterministic_fol", False)) and not offline,
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
                google_project=args.google_project,
                google_location=args.google_location,
                google_credentials=args.google_credentials,
                fol_object_evidence=bool(getattr(args, "fol_object_evidence", False)),
                fol_targeted_object_repair=bool(getattr(args, "fol_targeted_object_repair", False)),
                fol_few_shot_examples=_fewshot_enabled(args, "fol"),
                fol_repair_rounds=int(getattr(args, "fol_repair_rounds", 1)),
                fol_repair_round2=bool(getattr(args, "fol_repair_round2", False)),
                fol_repair_mode=str(getattr(args, "fol_repair_mode", "standard")),
                fol_repair_context=str(getattr(args, "fol_repair_context", "global")),
                fol_repair_max_issues_per_prompt=int(getattr(args, "fol_repair_max_issues_per_prompt", 8)),
                fol_repair_allow_drop=bool(getattr(args, "fol_repair_allow_drop", True)),
                fol_repair_preservation_gate=bool(getattr(args, "fol_repair_preservation_gate", False)),
                fol_batching=str(getattr(args, "fol_batching", "none")),
                fol_batch_max_matches=getattr(args, "fol_batch_max_matches", None),
                fol_batch_max_tokens=int(getattr(args, "fol_batch_max_tokens", 0)),
                fol_batch_overlap_strategy=str(getattr(args, "fol_batch_overlap_strategy", "fk_neighbors")),
                attribute_coverage=bool(getattr(args, "attribute_coverage", False)),
                attribute_coverage_validation=_attribute_coverage_validation_enabled(args),
                attribute_coverage_repair=_attribute_coverage_repair_enabled(args),
                allow_weak_object_links=bool(getattr(args, "allow_weak_object_links", False)),
                pattern_first_fgf=_pattern_first_enabled(args),
                pattern_candidates=int(getattr(args, "pattern_candidates", 1)),
                pattern_repair_budget_chars=int(getattr(args, "pattern_repair_budget_chars", 24000)),
                pattern_selector_context=str(getattr(args, "pattern_selector_context", "batched")),
                safe_domain_range_type_completion=bool(getattr(args, "safe_domain_range_type_completion", False)),
            )
            log_info(f"{scenario}: materialize start")
            _codegen_and_materialize_with_round2_only_fallback(
                scenario=scenario,
                scenario_work=scenario_work,
                tables=tables,
                data=data,
                args=args,
                offline=offline,
                output=output,
            )
        log_info(f"{scenario}: materialize complete output={output}")
        pattern_materialization = _run_pattern_materialization_stage(
            scenario=scenario,
            scenario_work=scenario_work,
            tables=tables,
            data=data,
            args=args,
            offline=offline,
            output=output,
        )
        if pattern_materialization.get("pattern_materialization_validation_enabled"):
            log_info(
                f"{scenario}: pattern materialization validation complete "
                f"selected_candidate={pattern_materialization.get('selected_candidate')}"
            )
        materialization_coverage = _run_materialization_coverage_stage(
            scenario=scenario,
            scenario_work=scenario_work,
            tables=tables,
            data=data,
            args=args,
            offline=offline,
            output=output,
        )
        if materialization_coverage.get("materialization_coverage_validation_enabled"):
            log_info(
                f"{scenario}: materialization coverage complete "
                f"repair_accepted={materialization_coverage.get('repair_accepted')}"
            )
        print(f"{scenario}: wrote {output}")


def _run_unit_test_gate(test_command: str) -> None:
    log_info(f"benchmark tests start command={test_command}")
    result = subprocess.run(test_command, shell=True)
    if result.returncode != 0:
        raise SystemExit(f"Unit test gate failed with exit code {result.returncode}; live benchmark was not run")
    log_info("benchmark tests complete")


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=True)
    except Exception:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not raw or raw.lstrip().startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            clean_key = key.strip()
            if clean_key and not os.environ.get(clean_key):
                os.environ[clean_key] = value.strip().strip('"')


def cmd_run_one_benchmark(args: argparse.Namespace) -> None:
    scenarios = _split_scenarios(args.scenario)
    if len(scenarios) != 1:
        raise SystemExit("run-one-benchmark requires exactly one scenario")
    scenario = scenarios[0]
    _load_env_file(Path(args.env_file))
    _run_unit_test_gate(args.test_command)

    benchmark_root = ensure_dir(Path(args.benchmark_root))
    smoke_work = benchmark_root / "smoke" / scenario
    live_work = benchmark_root / "paper_compare" / scenario
    log_info(f"{scenario}: benchmark offline smoke start")
    cmd_run_dev10(
        argparse.Namespace(
            rodi_root=args.rodi_root,
            work=str(smoke_work),
            scenarios=scenario,
            embedding_model=args.embedding_model,
            k=args.k,
            offline=True,
            match_workers=args.match_workers,
            fraction=args.fraction,
            seed=args.seed,
            codegen_self_consistency=args.codegen_self_consistency,
            codegen_prompt_version=args.codegen_prompt_version,
            source_context=getattr(args, "source_context", "schema"),
            llm_provider=getattr(args, "llm_provider", "openai"),
            llm_model=getattr(args, "llm_model", ""),
            embedding_provider=getattr(args, "embedding_provider", "openai"),
            google_project=getattr(args, "google_project", ""),
            google_location=getattr(args, "google_location", "global"),
            google_credentials=getattr(args, "google_credentials", ""),
            allow_forced_candidates=getattr(args, "allow_forced_candidates", False),
            allow_deterministic_match_repair=getattr(args, "allow_deterministic_match_repair", False),
            allow_invalid_match_fallback=getattr(args, "allow_invalid_match_fallback", False),
            deterministic_fol=getattr(args, "deterministic_fol", False),
            fol_object_evidence=getattr(args, "fol_object_evidence", False),
            fol_targeted_object_repair=getattr(args, "fol_targeted_object_repair", False),
            fol_few_shot_examples=getattr(args, "fol_few_shot_examples", False),
            match_few_shot_examples=getattr(args, "match_few_shot_examples", False),
            codegen_few_shot_examples=getattr(args, "codegen_few_shot_examples", False),
            fol_repair_rounds=getattr(args, "fol_repair_rounds", 1),
            fol_repair_round2=getattr(args, "fol_repair_round2", False),
            fol_repair_mode=getattr(args, "fol_repair_mode", "standard"),
            fol_repair_context=getattr(args, "fol_repair_context", "global"),
            fol_repair_max_issues_per_prompt=getattr(args, "fol_repair_max_issues_per_prompt", 8),
            fol_repair_allow_drop=getattr(args, "fol_repair_allow_drop", True),
            fol_repair_preservation_gate=getattr(args, "fol_repair_preservation_gate", False),
            fol_batching=getattr(args, "fol_batching", "none"),
            fol_batch_max_matches=getattr(args, "fol_batch_max_matches", None),
            fol_batch_max_tokens=getattr(args, "fol_batch_max_tokens", 0),
            fol_batch_overlap_strategy=getattr(args, "fol_batch_overlap_strategy", "fk_neighbors"),
            attribute_coverage=getattr(args, "attribute_coverage", False),
            attribute_coverage_validation=getattr(args, "attribute_coverage_validation", False),
            attribute_coverage_repair=getattr(args, "attribute_coverage_repair", False),
            allow_weak_object_links=getattr(args, "allow_weak_object_links", False),
            fewshot=getattr(args, "fewshot", "none"),
            fewshot_max_examples=getattr(args, "fewshot_max_examples", 4),
            fewshot_include_matching=getattr(args, "fewshot_include_matching", False),
            fewshot_include_fol=getattr(args, "fewshot_include_fol", False),
            fewshot_include_codegen=getattr(args, "fewshot_include_codegen", False),
            use_source_rdf=getattr(args, "use_source_rdf", False),
            source_rdf_backend=getattr(args, "source_rdf_backend", "morphkgc"),
            materialization_coverage_validation=getattr(args, "materialization_coverage_validation", False),
            materialization_coverage_repair=getattr(args, "materialization_coverage_repair", False),
            materialization_repair_context=getattr(args, "materialization_repair_context", "batched"),
            materialization_repair_candidates=getattr(args, "materialization_repair_candidates", 1),
            materialization_repair_budget_chars=getattr(args, "materialization_repair_budget_chars", 24000),
            internal_rerank_repair_candidates=getattr(args, "internal_rerank_repair_candidates", False),
            pattern_first_fgf=getattr(args, "pattern_first_fgf", False),
            pattern_schema_graph=getattr(args, "pattern_schema_graph", False),
            pattern_candidate_expansion=getattr(args, "pattern_candidate_expansion", False),
            uri_key_inference=getattr(args, "uri_key_inference", False),
            deterministic_pattern_compiler=getattr(args, "deterministic_pattern_compiler", False),
            pattern_candidates=getattr(args, "pattern_candidates", 1),
            internal_rerank_pattern_candidates=getattr(args, "internal_rerank_pattern_candidates", False),
            pattern_repair_budget_chars=getattr(args, "pattern_repair_budget_chars", 24000),
            pattern_selector_context=getattr(args, "pattern_selector_context", "batched"),
            safe_domain_range_type_completion=getattr(args, "safe_domain_range_type_completion", False),
            pattern_materialization_validation=getattr(args, "pattern_materialization_validation", False),
        )
    )
    log_info(f"{scenario}: benchmark offline smoke complete")

    started = time.monotonic()
    cmd_run_paper_compare(
        argparse.Namespace(
            rodi_root=args.rodi_root,
            work=str(live_work),
            scenarios=scenario,
            fraction=args.fraction,
            seed=args.seed,
            embedding_model=args.embedding_model,
            k=args.k,
            match_workers=args.match_workers,
            offline=args.offline,
            dry_run_db=args.dry_run_db,
            db_loader=args.db_loader,
            compose_root=args.compose_root,
            db_host=args.db_host,
            db_port=args.db_port,
            db_name=args.db_name,
            db_user=args.db_user,
            db_password=args.db_password,
            codegen_self_consistency=args.codegen_self_consistency,
            codegen_prompt_version=args.codegen_prompt_version,
            source_context=getattr(args, "source_context", "schema"),
            llm_provider=getattr(args, "llm_provider", "openai"),
            llm_model=getattr(args, "llm_model", ""),
            embedding_provider=getattr(args, "embedding_provider", "openai"),
            google_project=getattr(args, "google_project", ""),
            google_location=getattr(args, "google_location", "global"),
            google_credentials=getattr(args, "google_credentials", ""),
            allow_forced_candidates=getattr(args, "allow_forced_candidates", False),
            allow_deterministic_match_repair=getattr(args, "allow_deterministic_match_repair", False),
            allow_invalid_match_fallback=getattr(args, "allow_invalid_match_fallback", False),
            deterministic_fol=getattr(args, "deterministic_fol", False),
            fol_object_evidence=getattr(args, "fol_object_evidence", False),
            fol_targeted_object_repair=getattr(args, "fol_targeted_object_repair", False),
            fol_few_shot_examples=getattr(args, "fol_few_shot_examples", False),
            match_few_shot_examples=getattr(args, "match_few_shot_examples", False),
            codegen_few_shot_examples=getattr(args, "codegen_few_shot_examples", False),
            fol_repair_rounds=getattr(args, "fol_repair_rounds", 1),
            fol_repair_round2=getattr(args, "fol_repair_round2", False),
            fol_repair_mode=getattr(args, "fol_repair_mode", "standard"),
            fol_repair_context=getattr(args, "fol_repair_context", "global"),
            fol_repair_max_issues_per_prompt=getattr(args, "fol_repair_max_issues_per_prompt", 8),
            fol_repair_allow_drop=getattr(args, "fol_repair_allow_drop", True),
            fol_repair_preservation_gate=getattr(args, "fol_repair_preservation_gate", False),
            fol_batching=getattr(args, "fol_batching", "none"),
            fol_batch_max_matches=getattr(args, "fol_batch_max_matches", None),
            fol_batch_max_tokens=getattr(args, "fol_batch_max_tokens", 0),
            fol_batch_overlap_strategy=getattr(args, "fol_batch_overlap_strategy", "fk_neighbors"),
            attribute_coverage=getattr(args, "attribute_coverage", False),
            attribute_coverage_validation=getattr(args, "attribute_coverage_validation", False),
            attribute_coverage_repair=getattr(args, "attribute_coverage_repair", False),
            allow_weak_object_links=getattr(args, "allow_weak_object_links", False),
            fewshot=getattr(args, "fewshot", "none"),
            fewshot_max_examples=getattr(args, "fewshot_max_examples", 4),
            fewshot_include_matching=getattr(args, "fewshot_include_matching", False),
            fewshot_include_fol=getattr(args, "fewshot_include_fol", False),
            fewshot_include_codegen=getattr(args, "fewshot_include_codegen", False),
            use_source_rdf=getattr(args, "use_source_rdf", False),
            source_rdf_backend=getattr(args, "source_rdf_backend", "morphkgc"),
            materialization_coverage_validation=getattr(args, "materialization_coverage_validation", False),
            materialization_coverage_repair=getattr(args, "materialization_coverage_repair", False),
            materialization_repair_context=getattr(args, "materialization_repair_context", "batched"),
            materialization_repair_candidates=getattr(args, "materialization_repair_candidates", 1),
            materialization_repair_budget_chars=getattr(args, "materialization_repair_budget_chars", 24000),
            internal_rerank_repair_candidates=getattr(args, "internal_rerank_repair_candidates", False),
            pattern_first_fgf=getattr(args, "pattern_first_fgf", False),
            pattern_schema_graph=getattr(args, "pattern_schema_graph", False),
            pattern_candidate_expansion=getattr(args, "pattern_candidate_expansion", False),
            uri_key_inference=getattr(args, "uri_key_inference", False),
            deterministic_pattern_compiler=getattr(args, "deterministic_pattern_compiler", False),
            pattern_candidates=getattr(args, "pattern_candidates", 1),
            internal_rerank_pattern_candidates=getattr(args, "internal_rerank_pattern_candidates", False),
            pattern_repair_budget_chars=getattr(args, "pattern_repair_budget_chars", 24000),
            pattern_selector_context=getattr(args, "pattern_selector_context", "batched"),
            safe_domain_range_type_completion=getattr(args, "safe_domain_range_type_completion", False),
            pattern_materialization_validation=getattr(args, "pattern_materialization_validation", False),
        )
    )
    elapsed = time.monotonic() - started
    record = record_benchmark_run(
        benchmark_root,
        live_work,
        scenario,
        elapsed_seconds=elapsed,
        phase=args.phase,
        improvement_notes=args.improvement_notes,
    )
    log_info(f"{scenario}: benchmark recorded f1={record.get('f1')} target={record.get('target_f1')}")
    print(f"{scenario}: benchmark ledger={benchmark_root / 'benchmark_ledger.csv'} report={benchmark_root / 'reports' / (scenario + '.md')}")


def cmd_summarize_benchmark(args: argparse.Namespace) -> None:
    benchmark_root = Path(args.benchmark_root)
    rows = summarize_benchmark(benchmark_root)
    passed = sum(1 for row in rows if row.get("status") == "passed")
    blocked = sum(1 for row in rows if row.get("status") == "blocked")
    pending = sum(1 for row in rows if row.get("status") == "pending")
    print(
        f"Wrote benchmark summary to {benchmark_root / 'benchmark_summary.md'} "
        f"(passed={passed}, blocked={blocked}, pending={pending})"
    )


def _split_scenarios(value: str) -> list[str]:
    return [scenario.strip() for scenario in value.split(",") if scenario.strip()]


def _db_load_failed(rows: list[dict[str, object]]) -> bool:
    return any(row.get("returncode") not in (None, 0) for row in rows if not row.get("dry_run"))


def _llm_mode(offline: bool, events: list[str]) -> str:
    if offline:
        return "offline"
    if any(
        event.startswith("match:source_fallback")
        or event.startswith("match:partial_fallback")
        or event.startswith("codegen:fallback")
        for event in events
    ):
        return "fallback"
    return "live"


def _prepare_frozen_fol_input(
    *,
    scenario: str,
    dev_root: Path,
    upstream_work: Path,
    args: argparse.Namespace,
    offline: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    src = scenario_dir(dev_root, scenario)
    tables = parse_sql_dump(src / "dump.sql")
    data = parse_copy_data(src / "dump.sql")
    matches = read_json(upstream_work / "matches.json")["matches"]
    model = args.llm_model or REQUESTED_CODE_MODEL
    object_link_evidence: dict[str, Any] | None = None
    attribute_report: dict[str, Any] = {}

    if not offline:
        candidates = read_jsonl(upstream_work / "candidates.jsonl")
        matches, validation_report = reask_suspicious_matches(
            matches,
            candidates,
            tables,
            provider=args.llm_provider,
            model=model,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
        )
        write_json(upstream_work / "match_validation_report.json", {"rechecked": validation_report})
        if validation_report:
            write_json(upstream_work / "matches.json", {"matches": matches})
        target_records = read_jsonl(upstream_work / "target_records.jsonl")
        discriminator_rows = build_discriminator_candidate_rows(tables, data.rows, target_records, matches, k=16)
        existing_source_ids = {str(match.get("source_id", "")) for match in matches}
        discriminator_rows = [
            row for row in discriminator_rows if str(row.get("source", {}).get("id", "")) not in existing_source_ids
        ]
        if discriminator_rows:
            write_jsonl(upstream_work / "discriminator_candidates.jsonl", discriminator_rows)
            discriminator_matches = llm_discriminator_matches(
                discriminator_rows,
                provider=args.llm_provider,
                model=model,
                google_project=args.google_project,
                google_location=args.google_location,
                google_credentials=args.google_credentials,
            )
            write_json(upstream_work / "discriminator_matches.json", {"matches": discriminator_matches})
            matches = matches + discriminator_matches
            write_json(upstream_work / "matches.json", {"matches": matches})
            log_info(f"{scenario}: controlled ablation discriminator matching complete rows={len(discriminator_rows)}")
        if bool(getattr(args, "fol_object_evidence", False)):
            object_link_evidence = build_object_link_evidence(
                matches,
                tables,
                data,
                allow_weak_object_links=bool(getattr(args, "allow_weak_object_links", False)),
            )
            write_json(upstream_work / "object_link_evidence.json", object_link_evidence)
        raw_fol = llm_fol(
            matches,
            tables,
            offline=offline,
            provider=args.llm_provider,
            model=model,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
            object_link_evidence=object_link_evidence,
            few_shot_examples=_fewshot_enabled(args, "fol"),
            fol_batching=str(getattr(args, "fol_batching", "none")),
            fol_batch_max_matches=getattr(args, "fol_batch_max_matches", None),
            fol_batch_max_tokens=int(getattr(args, "fol_batch_max_tokens", 0)),
            fol_batch_overlap_strategy=str(getattr(args, "fol_batch_overlap_strategy", "fk_neighbors")),
        )
        fol = validate_fol(raw_fol, tables)
        fol["generation"] = raw_fol.get(
            "generation",
            {
                "source": "llm",
                "provider": args.llm_provider,
                "model": model,
                "prompt_version": "fgf_fol_v1_llm_only",
                "few_shot_examples": _fewshot_enabled(args, "fol"),
                "fol_batching": str(getattr(args, "fol_batching", "none")),
            },
        )
    else:
        fol = validate_fol(matches_to_fol(matches, tables), tables)
        fol["generation"] = {"source": "deterministic", "reason": "explicit offline mode"}

    issues, attribute_report = _all_fol_issues(
        fol,
        matches,
        tables,
        object_link_evidence=object_link_evidence,
        include_attribute_coverage=bool(getattr(args, "attribute_coverage", False)),
    )
    write_json(upstream_work / "fol_rules_before_repair.json", fol)
    write_json(
        upstream_work / "fol_validation_report_before_repair.json",
        {
            "issues_before_repair": issues,
            "issues_by_type_before_repair": issue_type_counts(issues),
            "rule_counts": _fol_rule_counts(fol),
            "attribute_coverage": attribute_report,
            "fol_batching": str(getattr(args, "fol_batching", "none")),
            "object_evidence_enabled": bool(object_link_evidence),
        },
    )
    write_json(upstream_work / "mapping_diagnostics_before_repair.json", mapping_diagnostics(matches, fol, tables))
    return fol, matches, tables, data, object_link_evidence, issues, attribute_report


def cmd_run_fol_repair_ablation(args: argparse.Namespace) -> None:
    work = ensure_dir(Path(args.work))
    _write_leakage_risk_report(work)
    scenarios = _split_scenarios(args.scenarios)
    arms = [arm.strip() for arm in str(args.fol_repair_arms).split(",") if arm.strip()]
    allowed_arms = {"standard", "round2_only", "standard_then_round2", "pre_repair_preserved"}
    invalid_arms = [arm for arm in arms if arm not in allowed_arms]
    if invalid_arms:
        raise SystemExit(f"Unknown FOL repair arm(s): {', '.join(invalid_arms)}")
    dev_root = work / "devset"
    offline = _live_offline(args)
    log_info(f"controlled fol repair devset start scenarios={scenarios} fraction={args.fraction}")
    create_devset(
        rodi_root=Path(args.rodi_root),
        out_dir=dev_root,
        scenarios=scenarios,
        fraction=args.fraction,
        seed=args.seed,
    )
    log_info(f"controlled fol repair devset complete root={dev_root}")
    log_info("controlled fol repair db load start")
    db_results = load_postgres_dumps(
        dev_root,
        scenarios,
        db_host=args.db_host,
        db_port=args.db_port,
        db_user=args.db_user,
        db_password=args.db_password,
        db_name=args.db_name,
        dry_run=args.dry_run_db,
        loader=args.db_loader,
        compose_root=Path(args.compose_root) if args.compose_root else Path(args.rodi_root),
        reset_databases=True,
    )
    log_info("controlled fol repair db load complete")
    if _db_load_failed(db_results):
        write_json(work / "run_metadata.json", {"db_results": db_results})
        raise SystemExit(f"PostgreSQL load failed; see {work / 'run_metadata.json'}")

    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "scenarios": scenarios,
        "arms": arms,
        "fraction": args.fraction,
        "seed": args.seed,
        "fol_batching": str(getattr(args, "fol_batching", "none")),
        "fol_repair_preservation_gate": bool(getattr(args, "fol_repair_preservation_gate", False)),
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model or REQUESTED_MATCH_MODEL,
        "embedding_provider": args.embedding_provider,
        "embedding_model": args.embedding_model,
        "db_results": db_results,
        "results": [],
    }
    for scenario in scenarios:
        clear_llm_events()
        scenario_root = ensure_dir(work / scenario)
        upstream_work = ensure_dir(scenario_root / "upstream")
        log_info(f"{scenario}: controlled upstream start")
        stage_verbalize(
            scenario,
            dev_root,
            upstream_work,
            source_context=_source_context_arg(args),
            db_host=args.db_host,
            db_port=args.db_port,
            db_name=args.db_name,
            db_user=args.db_user,
            db_password=args.db_password,
        )
        stage_index(
            upstream_work,
            args.embedding_model,
            offline=offline,
            embedding_provider=args.embedding_provider,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
        )
        stage_retrieve(
            upstream_work,
            args.embedding_model,
            k=args.k,
            offline=offline,
            embedding_provider=args.embedding_provider,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
            allow_forced_candidates=bool(getattr(args, "allow_forced_candidates", False) or offline),
        )
        stage_match(
            upstream_work,
            offline=offline,
            match_workers=args.match_workers,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
            allow_deterministic_repair=bool(getattr(args, "allow_deterministic_match_repair", False) or offline),
            fail_on_invalid=not bool(getattr(args, "allow_invalid_match_fallback", False)),
            match_few_shot_examples=_fewshot_enabled(args, "match"),
        )
        frozen_fol, matches, tables, data, object_link_evidence, issues, _attribute_report = _prepare_frozen_fol_input(
            scenario=scenario,
            dev_root=dev_root,
            upstream_work=upstream_work,
            args=args,
            offline=offline,
        )
        log_info(f"{scenario}: controlled upstream complete issues={len(issues)}")

        for arm in arms:
            arm_work = ensure_dir(scenario_root / arm)
            _copy_frozen_upstream_artifacts(upstream_work, arm_work)
            write_json(arm_work / "fol_rules_before_repair.json", frozen_fol)
            write_json(arm_work / "fol.json", frozen_fol)
            log_info(f"{scenario}/{arm}: repair arm start")
            fol, final_issues, repair_report = _apply_fol_repair_arm(
                arm=arm,
                work=arm_work,
                frozen_fol=frozen_fol,
                matches=matches,
                tables=tables,
                issues=issues,
                object_link_evidence=object_link_evidence,
                provider=args.llm_provider,
                model=args.llm_model or REQUESTED_CODE_MODEL,
                google_project=args.google_project,
                google_location=args.google_location,
                google_credentials=args.google_credentials,
                max_issues=int(getattr(args, "fol_repair_max_issues_per_prompt", 8)),
                allow_drop=bool(getattr(args, "fol_repair_allow_drop", True)),
                include_attribute_coverage=bool(getattr(args, "attribute_coverage", False)),
                preservation_gate=bool(getattr(args, "fol_repair_preservation_gate", False)),
            )
            write_json(arm_work / "fol.json", fol)
            write_json(
                arm_work / "fol_validation_report.json",
                {
                    **repair_report,
                    "controlled_ablation": True,
                    "arm": arm,
                    "issues_before_repair": issues,
                    "issues_after_repair": final_issues,
                    "issues_by_type_before_repair": issue_type_counts(issues),
                    "issues_by_type_after_repair": issue_type_counts(final_issues),
                },
            )
            write_json(arm_work / "mapping_diagnostics.json", mapping_diagnostics(matches, fol, tables))
            output = arm_work / "import.ttl"
            arm_args = argparse.Namespace(**vars(args))
            arm_args.fol_repair_mode = "round2_only" if arm == "round2_only" else "standard"
            runtime_fallback = _codegen_and_materialize_with_round2_only_fallback(
                scenario=scenario,
                scenario_work=arm_work,
                tables=tables,
                data=data,
                args=arm_args,
                offline=offline,
                output=output,
            )
            if not args.dry_run_db:
                qdir = scenario_dir(dev_root, scenario) / "queries"

                def sql_exec(sql: str, scenario_name: str = scenario) -> list[object]:
                    return execute_sql_psycopg2(
                        sql,
                        dbname=scenario_name,
                        host=args.db_host,
                        port=args.db_port,
                        user=args.db_user,
                        password=args.db_password,
                    )

                evaluation = evaluate_graph(output, qdir, sql_exec, arm_work / "eval")
            else:
                evaluation = {"precision": None, "recall": None, "f1": None, "count": 0}
            materialization = read_json(arm_work / "import.materialization_log.json") if (arm_work / "import.materialization_log.json").exists() else {}
            codegen_audit = read_json(arm_work / "codegen_candidate_scores.json") if (arm_work / "codegen_candidate_scores.json").exists() else {}
            row = {
                "scenario": scenario,
                "arm": arm,
                "precision": evaluation.get("precision"),
                "recall": evaluation.get("recall"),
                "f1": evaluation.get("f1"),
                "issues_before": len(issues),
                "issues_after": len(final_issues),
                "issues_by_type_before": issue_type_counts(issues),
                "issues_by_type_after": issue_type_counts(final_issues),
                "generated_triples": materialization.get("generated_triples", 0),
                "invalid_triples": materialization.get("invalid_triple_count", 0),
                "codegen_selected": codegen_audit.get("selected_index"),
                "codegen_repair_accepted": codegen_audit.get("repair_accepted"),
                "runtime_fallback_used": runtime_fallback.get("round2_only_runtime_fallback_used", False),
                "work": str(arm_work),
            }
            rows.append(row)
            metadata["results"].append(row)
            write_json(arm_work / "controlled_ablation_result.json", row)
            log_info(f"{scenario}/{arm}: complete f1={evaluation.get('f1')} issues_after={len(final_issues)}")

    csv_rows = [
        {
            **row,
            "issues_by_type_before": str(row["issues_by_type_before"]),
            "issues_by_type_after": str(row["issues_by_type_after"]),
        }
        for row in rows
    ]
    write_csv(
        work / "controlled_fol_repair_summary.csv",
        csv_rows,
        [
            "scenario",
            "arm",
            "precision",
            "recall",
            "f1",
            "issues_before",
            "issues_after",
            "issues_by_type_before",
            "issues_by_type_after",
            "generated_triples",
            "invalid_triples",
            "codegen_selected",
            "codegen_repair_accepted",
            "runtime_fallback_used",
            "work",
        ],
    )
    write_json(work / "run_metadata.json", metadata)
    print(f"Wrote controlled FOL repair ablation to {work / 'controlled_fol_repair_summary.csv'}")


def cmd_run_paper_compare(args: argparse.Namespace) -> None:
    if getattr(args, "fol_ablation_report", False) and not _fol_portfolio_enabled(args):
        raise SystemExit("--fol-ablation-report requires --fol-portfolio")
    work = ensure_dir(Path(args.work))
    _write_leakage_risk_report(work)
    scenarios = _split_scenarios(args.scenarios)
    dev_root = work / "devset"
    run_root = ensure_dir(work / "runs")
    comparison_root = ensure_dir(work / "comparison")
    offline = _live_offline(args)

    log_info(f"paper compare devset start scenarios={scenarios} fraction={args.fraction}")
    create_devset(
        rodi_root=Path(args.rodi_root),
        out_dir=dev_root,
        scenarios=scenarios,
        fraction=args.fraction,
        seed=args.seed,
    )
    log_info(f"paper compare devset complete root={dev_root}")
    log_info("paper compare db load start")
    db_results = load_postgres_dumps(
        dev_root,
        scenarios,
        db_host=args.db_host,
        db_port=args.db_port,
        db_user=args.db_user,
        db_password=args.db_password,
        db_name=args.db_name,
        dry_run=args.dry_run_db,
        loader=args.db_loader,
        compose_root=Path(args.compose_root) if args.compose_root else Path(args.rodi_root),
        reset_databases=True,
    )
    log_info("paper compare db load complete")
    metadata: dict[str, object] = {
        "scenarios": scenarios,
        "fraction": args.fraction,
        "seed": args.seed,
        "offline": offline,
        "openai_api_key_present": bool(os.getenv("OPENAI_API_KEY")),
        "llm_provider": getattr(args, "llm_provider", "openai"),
        "llm_model": getattr(args, "llm_model", "") or REQUESTED_MATCH_MODEL,
        "embedding_provider": getattr(args, "embedding_provider", "openai"),
        "embedding_model": args.embedding_model,
        "source_context": _source_context_arg(args),
        "match_few_shot_examples": _fewshot_enabled(args, "match"),
        "fol_few_shot_examples": _fewshot_enabled(args, "fol"),
        "codegen_few_shot_examples": _fewshot_enabled(args, "codegen"),
        "fol_repair_rounds": int(getattr(args, "fol_repair_rounds", 1)),
        "fol_repair_mode": str(getattr(args, "fol_repair_mode", "standard")),
        "fol_repair_context": str(getattr(args, "fol_repair_context", "global")),
        "fol_batching": str(getattr(args, "fol_batching", "none")),
        "attribute_coverage_validation": _attribute_coverage_validation_enabled(args),
        "attribute_coverage_repair": _attribute_coverage_repair_enabled(args),
        "fewshot": str(getattr(args, "fewshot", "none")),
        "use_source_rdf": bool(getattr(args, "use_source_rdf", False)),
        "google_project": getattr(args, "google_project", ""),
        "google_location": getattr(args, "google_location", ""),
        "google_credentials_present": bool(getattr(args, "google_credentials", "")),
        "db_results": db_results,
        "scenario_results": [],
    }
    if _db_load_failed(db_results):
        write_json(work / "run_metadata.json", metadata)
        raise SystemExit(f"PostgreSQL load failed; see {work / 'run_metadata.json'}")

    for scenario in scenarios:
        clear_llm_events()
        scenario_work = ensure_dir(run_root / scenario)
        stage_verbalize(
            scenario,
            dev_root,
            scenario_work,
            source_context=_source_context_arg(args),
            db_host=args.db_host,
            db_port=args.db_port,
            db_name=args.db_name,
            db_user=args.db_user,
            db_password=args.db_password,
        )
        stage_index(
            scenario_work,
            args.embedding_model,
            offline=offline,
            embedding_provider=args.embedding_provider,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
        )
        stage_retrieve(
            scenario_work,
            args.embedding_model,
            k=args.k,
            offline=offline,
            embedding_provider=args.embedding_provider,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
            allow_forced_candidates=bool(getattr(args, "allow_forced_candidates", False) or offline),
        )
        stage_match(
            scenario_work,
            offline=offline,
            match_workers=args.match_workers,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
            allow_deterministic_repair=bool(getattr(args, "allow_deterministic_match_repair", False) or offline),
            fail_on_invalid=not bool(getattr(args, "allow_invalid_match_fallback", False)),
            match_few_shot_examples=_fewshot_enabled(args, "match"),
        )
        src = scenario_dir(dev_root, scenario)
        tables = parse_sql_dump(src / "dump.sql")
        data = parse_copy_data(src / "dump.sql")
        output = scenario_work / "import.ttl"
        if _fol_portfolio_enabled(args):
            log_info(f"{scenario}: fol portfolio start arms={getattr(args, 'fol_portfolio_arms', '')}")
            portfolio_selection = _run_fol_portfolio(
                scenario=scenario,
                dev_root=dev_root,
                scenario_work=scenario_work,
                args=args,
                offline=offline,
            )
            runtime_fallback = {
                "fol_portfolio_enabled": True,
                "fol_portfolio_selected_arm": portfolio_selection.get("selected_arm"),
                "fol_portfolio_selector": portfolio_selection.get("selector"),
            }
        else:
            stage_fol(
                scenario,
                dev_root,
                scenario_work,
                offline=offline,
                use_llm=not bool(getattr(args, "deterministic_fol", False)) and not offline,
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
                google_project=args.google_project,
                google_location=args.google_location,
                google_credentials=args.google_credentials,
                fol_object_evidence=bool(getattr(args, "fol_object_evidence", False)),
                fol_targeted_object_repair=bool(getattr(args, "fol_targeted_object_repair", False)),
                fol_few_shot_examples=_fewshot_enabled(args, "fol"),
                fol_repair_rounds=int(getattr(args, "fol_repair_rounds", 1)),
                fol_repair_round2=bool(getattr(args, "fol_repair_round2", False)),
                fol_repair_mode=str(getattr(args, "fol_repair_mode", "standard")),
                fol_repair_context=str(getattr(args, "fol_repair_context", "global")),
                fol_repair_max_issues_per_prompt=int(getattr(args, "fol_repair_max_issues_per_prompt", 8)),
                fol_repair_allow_drop=bool(getattr(args, "fol_repair_allow_drop", True)),
                fol_repair_preservation_gate=bool(getattr(args, "fol_repair_preservation_gate", False)),
                fol_batching=str(getattr(args, "fol_batching", "none")),
                fol_batch_max_matches=getattr(args, "fol_batch_max_matches", None),
                fol_batch_max_tokens=int(getattr(args, "fol_batch_max_tokens", 0)),
                fol_batch_overlap_strategy=str(getattr(args, "fol_batch_overlap_strategy", "fk_neighbors")),
                attribute_coverage=bool(getattr(args, "attribute_coverage", False)),
                attribute_coverage_validation=_attribute_coverage_validation_enabled(args),
                attribute_coverage_repair=_attribute_coverage_repair_enabled(args),
                allow_weak_object_links=bool(getattr(args, "allow_weak_object_links", False)),
                pattern_first_fgf=_pattern_first_enabled(args),
                pattern_candidates=int(getattr(args, "pattern_candidates", 1)),
                pattern_repair_budget_chars=int(getattr(args, "pattern_repair_budget_chars", 24000)),
                pattern_selector_context=str(getattr(args, "pattern_selector_context", "batched")),
                safe_domain_range_type_completion=bool(getattr(args, "safe_domain_range_type_completion", False)),
            )
            log_info(f"{scenario}: materialize start")
            runtime_fallback = _codegen_and_materialize_with_round2_only_fallback(
                scenario=scenario,
                scenario_work=scenario_work,
                tables=tables,
                data=data,
                args=args,
                offline=offline,
                output=output,
            )
        log_info(f"{scenario}: materialize complete output={output}")
        pattern_materialization = _run_pattern_materialization_stage(
            scenario=scenario,
            scenario_work=scenario_work,
            tables=tables,
            data=data,
            args=args,
            offline=offline,
            output=output,
        )
        if pattern_materialization.get("pattern_materialization_validation_enabled"):
            log_info(
                f"{scenario}: pattern materialization validation complete "
                f"selected_candidate={pattern_materialization.get('selected_candidate')}"
            )
        materialization_coverage = _run_materialization_coverage_stage(
            scenario=scenario,
            scenario_work=scenario_work,
            tables=tables,
            data=data,
            args=args,
            offline=offline,
            output=output,
        )
        if materialization_coverage.get("materialization_coverage_validation_enabled"):
            log_info(
                f"{scenario}: materialization coverage complete "
                f"repair_accepted={materialization_coverage.get('repair_accepted')}"
            )
        if not args.dry_run_db:
            qdir = scenario_dir(dev_root, scenario) / "queries"

            def sql_exec(sql: str, scenario_name: str = scenario) -> list[object]:
                return execute_sql_psycopg2(
                    sql,
                    dbname=scenario_name,
                    host=args.db_host,
                    port=args.db_port,
                    user=args.db_user,
                    password=args.db_password,
                )

            log_info(f"{scenario}: evaluate start")
            evaluation = evaluate_graph(output, qdir, sql_exec, scenario_work / "eval")
            log_info(f"{scenario}: evaluate complete f1={evaluation['f1']} count={evaluation['count']}")
        else:
            evaluation = {"precision": None, "recall": None, "f1": None, "count": 0}
        events = llm_events()
        scenario_meta = {
            "scenario": scenario,
            "work": str(scenario_work),
            "import_ttl": str(output),
            "evaluation": evaluation,
            "mapping_diagnostics": read_json(scenario_work / "mapping_diagnostics.json")
            if (scenario_work / "mapping_diagnostics.json").exists()
            else {},
            "llm_events": events,
            "mode": _llm_mode(offline, events),
            "llm_provider": args.llm_provider,
            "llm_model": args.llm_model or REQUESTED_MATCH_MODEL,
            "embedding_provider": args.embedding_provider,
            "embedding_model": args.embedding_model,
            "source_context": _source_context_arg(args),
            "match_few_shot_examples": _fewshot_enabled(args, "match"),
            "fol_few_shot_examples": _fewshot_enabled(args, "fol"),
            "codegen_few_shot_examples": _fewshot_enabled(args, "codegen"),
            "fol_repair_rounds": int(getattr(args, "fol_repair_rounds", 1)),
            "fol_repair_context": str(getattr(args, "fol_repair_context", "global")),
            "fol_batching": str(getattr(args, "fol_batching", "none")),
            "fol_round2_only_runtime_fallback": runtime_fallback,
            "pattern_first_fgf": _pattern_first_enabled(args),
            "pattern_materialization": pattern_materialization,
            "materialization_coverage": materialization_coverage,
            "materialization_coverage_validation": _materialization_coverage_validation_enabled(args),
            "materialization_coverage_repair": _materialization_coverage_repair_enabled(args),
            "fewshot": str(getattr(args, "fewshot", "none")),
            "use_source_rdf": bool(getattr(args, "use_source_rdf", False)),
            "google_project": args.google_project,
            "google_location": args.google_location,
        }
        metadata["scenario_results"].append(scenario_meta)  # type: ignore[union-attr]
        print(f"{scenario}: F1={evaluation['f1']} mode={scenario_meta['mode']} artifacts={scenario_work}")

    log_info("paper compare output start")
    paper_rows = compare_to_paper(run_root, comparison_root, scenarios)
    metadata["paper_comparison"] = paper_rows
    write_json(work / "run_metadata.json", metadata)
    log_info(f"paper compare output complete root={comparison_root}")
    print(f"Wrote paper comparison to {comparison_root}")


def cmd_check_llm4vkg(args: argparse.Namespace) -> None:
    root = Path(args.llm4vkg_root) if args.llm4vkg_root else default_llm4vkg_root(Path(args.rodi_root))
    status = check_llm4vkg_resources(root, require_logmap=args.require_logmap)
    for check in status.checks:
        marker = "OK" if check.ok else "MISSING"
        print(f"{marker}: {check.name} ({check.detail})")
    if not status.ok:
        raise SystemExit(2)


def cmd_run_llm4vkg(args: argparse.Namespace) -> None:
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    llm4vkg_root = Path(args.llm4vkg_root) if args.llm4vkg_root else default_llm4vkg_root(Path(args.rodi_root))
    result = bootstrap_llm4vkg(
        llm4vkg_root=llm4vkg_root,
        devset_root=Path(args.devset_root),
        output_root=Path(args.output),
        scenarios=scenarios,
        api_names=[name.strip() for name in args.api_names.split(",") if name.strip()],
        db_host=args.db_host,
        db_port=args.db_port,
        db_user=args.db_user,
        db_password=args.db_password,
        db_name=args.db_name,
        run_generation=not args.skip_generation,
        run_evaluation=not args.skip_evaluation,
        dry_run_db=args.dry_run_db,
        java_home=Path(args.java_home) if args.java_home else None,
        db_loader=args.db_loader,
        compose_root=Path(args.compose_root) if args.compose_root else Path(args.rodi_root),
        lightweight_retriever=args.lightweight_retriever,
    )
    print(f"Wrote LLM4VKG run metadata to {Path(args.output) / 'llm4vkg_run.json'}")
    print(f"Resource status ok={result['status']['ok']}; discovered metrics={len(result['metrics'])}")


def cmd_compare(args: argparse.Namespace) -> None:
    baseline_rows = []
    if args.baseline_jsonl:
        baseline_rows = read_jsonl(Path(args.baseline_jsonl))
    if args.baseline_csv:
        import csv

        with Path(args.baseline_csv).open("r", encoding="utf-8", newline="") as fh:
            baseline_rows = list(csv.DictReader(fh))
    rows = compare_runs(
        coding_root=Path(args.coding_root),
        baseline_rows=baseline_rows,
        output_dir=Path(args.output),
        scenarios=[s.strip() for s in args.scenarios.split(",") if s.strip()],
        similarity_margin=args.similarity_margin,
    )
    promoted = sum(1 for row in rows if row["promoted"])
    print(f"Wrote comparison to {args.output}; promoted={promoted}/{len(rows)}")


def add_db_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db-host", default=os.getenv("DB_HOST", "localhost"))
    parser.add_argument("--db-port", type=int, default=int(os.getenv("DB_PORT", "5432")))
    parser.add_argument("--db-name", default=os.getenv("POSTGRES_DB", "postgres"))
    parser.add_argument("--db-user", default=os.getenv("POSTGRES_USER", "postgres"))
    parser.add_argument("--db-password", default=os.getenv("POSTGRES_PASSWORD", "postgres"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coding-fgf")
    sub = parser.add_subparsers(required=True)

    devset = sub.add_parser("devset")
    devset.add_argument("--rodi-root", default="..")
    devset.add_argument("--out", default="work/dev10")
    devset.add_argument("--fraction", type=float, default=0.1)
    devset.add_argument("--seed", default="coding-fgf-dev10-v1")
    devset.add_argument("--scenarios", default=",".join(PAPER_SCENARIOS))
    devset.set_defaults(func=cmd_devset)

    verbalize = sub.add_parser("verbalize")
    verbalize.add_argument("--scenario", required=True)
    verbalize.add_argument("--rodi-root", default="..")
    verbalize.add_argument("--work", required=True)
    verbalize.set_defaults(func=cmd_verbalize)

    source_rdf = sub.add_parser("source-rdf")
    source_rdf.add_argument("--scenario", required=True)
    source_rdf.add_argument("--rodi-root", default="..")
    source_rdf.add_argument("--work", required=True)
    source_rdf.add_argument("--db-url", default="")
    source_rdf.add_argument("--run", action="store_true")
    add_db_args(source_rdf)
    source_rdf.set_defaults(func=cmd_source_rdf)

    index = sub.add_parser("index")
    index.add_argument("--work", required=True)
    index.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    index.add_argument("--offline", action="store_true")
    add_provider_args(index)
    index.set_defaults(func=cmd_index)

    retrieve = sub.add_parser("retrieve")
    retrieve.add_argument("--work", required=True)
    retrieve.add_argument("--k", type=int, default=20)
    retrieve.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    retrieve.add_argument("--offline", action="store_true")
    retrieve.add_argument("--allow-forced-candidates", action="store_true")
    add_provider_args(retrieve)
    retrieve.set_defaults(func=cmd_retrieve)

    match = sub.add_parser("match")
    match.add_argument("--work", required=True)
    match.add_argument("--offline", action="store_true")
    match.add_argument("--match-workers", type=int, default=int(os.getenv("CODING_FGF_MATCH_WORKERS", "4")))
    match.add_argument("--allow-deterministic-match-repair", action="store_true")
    match.add_argument("--allow-invalid-match-fallback", action="store_true")
    match.add_argument("--match-few-shot-examples", action="store_true")
    add_provider_args(match)
    add_fewshot_args(match)
    match.set_defaults(func=cmd_match)

    fol = sub.add_parser("fol")
    fol.add_argument("--scenario", required=True)
    fol.add_argument("--rodi-root", default="..")
    fol.add_argument("--work", required=True)
    fol.add_argument("--offline", action="store_true")
    fol.add_argument("--deterministic-fol", action="store_true")
    fol.add_argument("--fol-object-evidence", action="store_true")
    fol.add_argument("--fol-targeted-object-repair", action="store_true")
    fol.add_argument("--fol-few-shot-examples", action="store_true")
    add_provider_args(fol)
    add_fol_ablation_args(fol)
    add_fewshot_args(fol)
    add_pattern_first_args(fol)
    fol.set_defaults(func=cmd_fol)

    codegen = sub.add_parser("codegen")
    codegen.add_argument("--work", required=True)
    codegen.add_argument("--offline", action="store_true")
    codegen.add_argument("--codegen-self-consistency", type=int, default=int(os.getenv("CODING_FGF_CODEGEN_SELF_CONSISTENCY", "3")))
    codegen.add_argument("--codegen-prompt-version", default=os.getenv("CODING_FGF_CODEGEN_PROMPT_VERSION", CODEGEN_PROMPT_VERSION))
    codegen.add_argument("--codegen-few-shot-examples", action="store_true")
    add_provider_args(codegen)
    add_fewshot_args(codegen)
    codegen.set_defaults(func=cmd_codegen)

    materialize = sub.add_parser("materialize")
    materialize.add_argument("--scenario", required=True)
    materialize.add_argument("--rodi-root", default="..")
    materialize.add_argument("--work", required=True)
    materialize.add_argument("--output", required=True)
    materialize.set_defaults(func=cmd_materialize)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--scenario", required=True)
    evaluate.add_argument("--rodi-root", default="..")
    evaluate.add_argument("--graph", required=True)
    evaluate.add_argument("--out", required=True)
    add_db_args(evaluate)
    evaluate.set_defaults(func=cmd_evaluate)

    run = sub.add_parser("run-dev10")
    run.add_argument("--rodi-root", default="..")
    run.add_argument("--work", default="work/dev10_run")
    run.add_argument("--scenarios", default=",".join(PAPER_SCENARIOS))
    run.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    run.add_argument("--k", type=int, default=20)
    run.add_argument("--offline", action="store_true")
    run.add_argument("--source-context", choices=["schema", "morphkgc"], default=os.getenv("CODING_FGF_SOURCE_CONTEXT", "schema"))
    run.add_argument("--match-workers", type=int, default=int(os.getenv("CODING_FGF_MATCH_WORKERS", "4")))
    run.add_argument("--codegen-self-consistency", type=int, default=int(os.getenv("CODING_FGF_CODEGEN_SELF_CONSISTENCY", "3")))
    run.add_argument("--codegen-prompt-version", default=os.getenv("CODING_FGF_CODEGEN_PROMPT_VERSION", CODEGEN_PROMPT_VERSION))
    run.add_argument("--fraction", type=float, default=float(os.getenv("CODING_FGF_FRACTION", "0.1")))
    run.add_argument("--seed", default=os.getenv("CODING_FGF_SEED", "coding-fgf-dev10-v1"))
    run.add_argument("--allow-forced-candidates", action="store_true")
    run.add_argument("--allow-deterministic-match-repair", action="store_true")
    run.add_argument("--allow-invalid-match-fallback", action="store_true")
    run.add_argument("--match-few-shot-examples", action="store_true")
    run.add_argument("--deterministic-fol", action="store_true")
    run.add_argument("--fol-object-evidence", action="store_true")
    run.add_argument("--fol-targeted-object-repair", action="store_true")
    run.add_argument("--fol-few-shot-examples", action="store_true")
    run.add_argument("--codegen-few-shot-examples", action="store_true")
    add_provider_args(run)
    add_fol_ablation_args(run)
    add_fewshot_args(run)
    add_source_rdf_args(run)
    add_materialization_coverage_args(run)
    add_pattern_first_args(run)
    run.set_defaults(func=cmd_run_dev10)

    one = sub.add_parser("run-one-benchmark")
    one.add_argument("--scenario", required=True)
    one.add_argument("--rodi-root", default="..")
    one.add_argument("--benchmark-root", default=os.getenv("CODING_FGF_BENCHMARK_ROOT", "work/benchmark_runs"))
    one.add_argument("--fraction", type=float, default=float(os.getenv("CODING_FGF_FRACTION", "0.1")))
    one.add_argument("--seed", default=os.getenv("CODING_FGF_SEED", "coding-fgf-dev10-v1"))
    one.add_argument("--embedding-model", default=os.getenv("FGF_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    one.add_argument("--k", type=int, default=int(os.getenv("CODING_FGF_K", "20")))
    one.add_argument("--match-workers", type=int, default=int(os.getenv("CODING_FGF_MATCH_WORKERS", "4")))
    one.add_argument("--codegen-self-consistency", type=int, default=int(os.getenv("CODING_FGF_CODEGEN_SELF_CONSISTENCY", "3")))
    one.add_argument("--codegen-prompt-version", default=os.getenv("CODING_FGF_CODEGEN_PROMPT_VERSION", CODEGEN_PROMPT_VERSION))
    one.add_argument("--test-command", default="pytest -q -p no:cacheprovider tests/test_core.py tests/test_baseline_compare.py")
    one.add_argument("--env-file", default="../.env")
    one.add_argument("--phase", default=os.getenv("CODING_FGF_BENCHMARK_PHASE", "first_sweep"))
    one.add_argument("--improvement-notes", default=os.getenv("CODING_FGF_IMPROVEMENT_NOTES", ""))
    one.add_argument("--offline", action="store_true")
    one.add_argument("--source-context", choices=["schema", "morphkgc"], default=os.getenv("CODING_FGF_SOURCE_CONTEXT", "schema"))
    add_provider_args(one)
    one.add_argument("--allow-forced-candidates", action="store_true")
    one.add_argument("--allow-deterministic-match-repair", action="store_true")
    one.add_argument("--allow-invalid-match-fallback", action="store_true")
    one.add_argument("--match-few-shot-examples", action="store_true")
    one.add_argument("--deterministic-fol", action="store_true")
    one.add_argument("--fol-object-evidence", action="store_true")
    one.add_argument("--fol-targeted-object-repair", action="store_true")
    one.add_argument("--fol-few-shot-examples", action="store_true")
    one.add_argument("--codegen-few-shot-examples", action="store_true")
    add_fol_ablation_args(one)
    add_fewshot_args(one)
    add_source_rdf_args(one)
    add_materialization_coverage_args(one)
    add_pattern_first_args(one)
    one.add_argument("--dry-run-db", action="store_true")
    one.add_argument("--db-loader", choices=["psql", "docker-compose"], default=os.getenv("CODING_FGF_DB_LOADER", "psql"))
    one.add_argument("--compose-root", default="")
    add_db_args(one)
    one.set_defaults(func=cmd_run_one_benchmark)

    summary = sub.add_parser("summarize-benchmark")
    summary.add_argument("--benchmark-root", default=os.getenv("CODING_FGF_BENCHMARK_ROOT", "work/benchmark_runs"))
    summary.set_defaults(func=cmd_summarize_benchmark)

    paper = sub.add_parser("run-paper-compare")
    paper.add_argument("--rodi-root", default="..")
    paper.add_argument("--work", default=os.getenv("CODING_FGF_WORK", "work/paper_compare"))
    paper.add_argument("--scenarios", default=os.getenv("SCENARIOS", "cmt_renamed"))
    paper.add_argument("--fraction", type=float, default=float(os.getenv("CODING_FGF_FRACTION", "0.1")))
    paper.add_argument("--seed", default=os.getenv("CODING_FGF_SEED", "coding-fgf-dev10-v1"))
    paper.add_argument("--embedding-model", default=os.getenv("FGF_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    paper.add_argument("--source-context", choices=["schema", "morphkgc"], default=os.getenv("CODING_FGF_SOURCE_CONTEXT", "schema"))
    paper.add_argument("--k", type=int, default=int(os.getenv("CODING_FGF_K", "20")))
    paper.add_argument("--match-workers", type=int, default=int(os.getenv("CODING_FGF_MATCH_WORKERS", "4")))
    paper.add_argument("--codegen-self-consistency", type=int, default=int(os.getenv("CODING_FGF_CODEGEN_SELF_CONSISTENCY", "3")))
    paper.add_argument("--codegen-prompt-version", default=os.getenv("CODING_FGF_CODEGEN_PROMPT_VERSION", CODEGEN_PROMPT_VERSION))
    paper.add_argument("--offline", action="store_true")
    add_provider_args(paper)
    paper.add_argument("--allow-forced-candidates", action="store_true")
    paper.add_argument("--allow-deterministic-match-repair", action="store_true")
    paper.add_argument("--allow-invalid-match-fallback", action="store_true")
    paper.add_argument("--match-few-shot-examples", action="store_true")
    paper.add_argument("--deterministic-fol", action="store_true")
    paper.add_argument("--fol-object-evidence", action="store_true")
    paper.add_argument("--fol-targeted-object-repair", action="store_true")
    paper.add_argument("--fol-few-shot-examples", action="store_true")
    paper.add_argument("--codegen-few-shot-examples", action="store_true")
    add_fol_ablation_args(paper)
    add_fewshot_args(paper)
    add_source_rdf_args(paper)
    add_materialization_coverage_args(paper)
    add_pattern_first_args(paper)
    add_fol_portfolio_args(paper)
    paper.add_argument("--fol-ablation-report", action="store_true")
    paper.add_argument("--dry-run-db", action="store_true")
    paper.add_argument("--db-loader", choices=["psql", "docker-compose"], default=os.getenv("CODING_FGF_DB_LOADER", "psql"))
    paper.add_argument("--compose-root", default="")
    add_db_args(paper)
    paper.set_defaults(func=cmd_run_paper_compare)

    repair_ablation = sub.add_parser("run-fol-repair-ablation")
    repair_ablation.add_argument("--rodi-root", default="..")
    repair_ablation.add_argument("--work", default=os.getenv("CODING_FGF_WORK", "work/fol_repair_ablation"))
    repair_ablation.add_argument("--scenarios", default=os.getenv("SCENARIOS", "cmt_renamed"))
    repair_ablation.add_argument("--fraction", type=float, default=float(os.getenv("CODING_FGF_FRACTION", "0.1")))
    repair_ablation.add_argument("--seed", default=os.getenv("CODING_FGF_SEED", "coding-fgf-dev10-v1"))
    repair_ablation.add_argument("--embedding-model", default=os.getenv("FGF_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    repair_ablation.add_argument("--source-context", choices=["schema", "morphkgc"], default=os.getenv("CODING_FGF_SOURCE_CONTEXT", "schema"))
    repair_ablation.add_argument("--k", type=int, default=int(os.getenv("CODING_FGF_K", "20")))
    repair_ablation.add_argument("--match-workers", type=int, default=int(os.getenv("CODING_FGF_MATCH_WORKERS", "4")))
    repair_ablation.add_argument("--codegen-self-consistency", type=int, default=int(os.getenv("CODING_FGF_CODEGEN_SELF_CONSISTENCY", "3")))
    repair_ablation.add_argument("--codegen-prompt-version", default=os.getenv("CODING_FGF_CODEGEN_PROMPT_VERSION", CODEGEN_PROMPT_VERSION))
    repair_ablation.add_argument("--fol-repair-arms", default=os.getenv("CODING_FGF_FOL_REPAIR_ARMS", "standard,round2_only,standard_then_round2"))
    repair_ablation.add_argument("--offline", action="store_true")
    add_provider_args(repair_ablation)
    repair_ablation.add_argument("--allow-forced-candidates", action="store_true")
    repair_ablation.add_argument("--allow-deterministic-match-repair", action="store_true")
    repair_ablation.add_argument("--allow-invalid-match-fallback", action="store_true")
    repair_ablation.add_argument("--match-few-shot-examples", action="store_true")
    repair_ablation.add_argument("--deterministic-fol", action="store_true")
    repair_ablation.add_argument("--fol-object-evidence", action="store_true")
    repair_ablation.add_argument("--fol-targeted-object-repair", action="store_true")
    repair_ablation.add_argument("--fol-few-shot-examples", action="store_true")
    repair_ablation.add_argument("--codegen-few-shot-examples", action="store_true")
    add_fol_ablation_args(repair_ablation)
    add_fewshot_args(repair_ablation)
    add_source_rdf_args(repair_ablation)
    add_materialization_coverage_args(repair_ablation)
    repair_ablation.add_argument("--dry-run-db", action="store_true")
    repair_ablation.add_argument("--db-loader", choices=["psql", "docker-compose"], default=os.getenv("CODING_FGF_DB_LOADER", "psql"))
    repair_ablation.add_argument("--compose-root", default="")
    add_db_args(repair_ablation)
    repair_ablation.set_defaults(func=cmd_run_fol_repair_ablation)

    check = sub.add_parser("check-llm4vkg")
    check.add_argument("--rodi-root", default="..")
    check.add_argument("--llm4vkg-root", default="")
    check.add_argument("--require-logmap", action="store_true")
    check.set_defaults(func=cmd_check_llm4vkg)

    baseline = sub.add_parser("run-llm4vkg")
    baseline.add_argument("--rodi-root", default="..")
    baseline.add_argument("--llm4vkg-root", default="")
    baseline.add_argument("--devset-root", required=True)
    baseline.add_argument("--output", default="work/baselines/llm4vkg_dev10")
    baseline.add_argument("--scenarios", default="cmt_renamed")
    baseline.add_argument("--api-names", default=os.getenv("LLM4VKG_API_NAMES", "gpt_4o"))
    baseline.add_argument("--skip-generation", action="store_true")
    baseline.add_argument("--skip-evaluation", action="store_true")
    baseline.add_argument("--dry-run-db", action="store_true")
    baseline.add_argument("--java-home", default=os.getenv("JAVA_HOME_17", ""))
    baseline.add_argument("--db-loader", choices=["psql", "docker-compose"], default=os.getenv("CODING_FGF_DB_LOADER", "psql"))
    baseline.add_argument("--compose-root", default="")
    baseline.add_argument("--lightweight-retriever", action="store_true")
    add_db_args(baseline)
    baseline.set_defaults(func=cmd_run_llm4vkg)

    compare = sub.add_parser("compare")
    compare.add_argument("--coding-root", required=True)
    compare.add_argument("--baseline-csv", default="")
    compare.add_argument("--baseline-jsonl", default="")
    compare.add_argument("--output", required=True)
    compare.add_argument("--scenarios", default="cmt_renamed")
    compare.add_argument("--similarity-margin", type=float, default=0.05)
    compare.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)

