from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required for release config wrappers") from exc

BOOL_FLAGS = {
    "fol_portfolio": "--fol-portfolio",
    "fol_ablation_report": "--fol-ablation-report",
    "attribute_coverage_validation": "--attribute-coverage-validation",
    "attribute_coverage_repair": "--attribute-coverage-repair",
    "fewshot_include_matching": "--fewshot-include-matching",
    "fewshot_include_fol": "--fewshot-include-fol",
    "fewshot_include_codegen": "--fewshot-include-codegen",
}
VALUE_FLAGS = {
    "retrieval_metric": "--retrieval-metric",
    "match_candidate_limit": "--match-candidate-limit",
    "match_candidate_context": "--match-candidate-context",
    "match_validation": "--match-validation",
    "rodi_root": "--rodi-root",
    "work": "--work",
    "fraction": "--fraction",
    "seed": "--seed",
    "embedding_model": "--embedding-model",
    "embedding_provider": "--embedding-provider",
    "llm_provider": "--llm-provider",
    "llm_model": "--llm-model",
    "google_project": "--google-project",
    "google_location": "--google-location",
    "google_credentials": "--google-credentials",
    "k": "--k",
    "match_workers": "--match-workers",
    "codegen_self_consistency": "--codegen-self-consistency",
    "db_loader": "--db-loader",
    "fol_portfolio_selector": "--fol-portfolio-selector",
    "fol_batching": "--fol-batching",
    "fol_repair_context": "--fol-repair-context",
    "fol_batch_max_tokens": "--fol-batch-max-tokens",
    "fol_repair_max_issues_per_prompt": "--fol-repair-max-issues-per-prompt",
    "fewshot": "--fewshot",
}


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return expand_env(data)


def scenario_arg(config: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    scenarios = config.get("scenarios") or ["cmt_renamed"]
    if isinstance(scenarios, str):
        return scenarios
    return ",".join(str(s) for s in scenarios)


def pipeline_command(config: dict[str, Any], scenario: str | None = None, extra_args: list[str] | None = None) -> list[str]:
    cmd = [sys.executable, "-m", "coding_fgf", "run-paper-compare"]
    cfg = dict(config)
    cfg["scenarios"] = scenario_arg(config, scenario)
    cmd.extend(["--scenarios", str(cfg["scenarios"])])
    for key, flag in VALUE_FLAGS.items():
        if key in cfg and cfg[key] not in (None, ""):
            if key == "fol_portfolio_selector" and not cfg.get("fol_portfolio"):
                continue
            cmd.extend([flag, str(cfg[key])])
    if cfg.get("fol_portfolio_arms"):
        arms = cfg["fol_portfolio_arms"]
        if isinstance(arms, list):
            arms = ",".join(str(a) for a in arms)
        cmd.extend(["--fol-portfolio-arms", str(arms)])
    for key, flag in BOOL_FLAGS.items():
        if cfg.get(key):
            cmd.append(flag)
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/openai_best_fol_default.yaml")
    parser.add_argument("--scenario", default=None, help="Override scenario list from config, comma-separated")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing it")


def run_or_print(cmd: list[str], dry_run: bool) -> int:
    print(" ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd)


def ablation_command(config: dict[str, Any], kind: str, *, scenario=None,
                     output_dir=None, candidate_artifact=None, extra_args=None) -> list[str]:
    base = Path(str(config.get("work", "/outputs/run"))) / "ablations"
    scenarios = scenario_arg(config, scenario).split(",")
    if kind == "fol_selection":
        cfg = dict(config, work=str(output_dir or base / kind), fol_portfolio=True,
                   fol_ablation_report=True)
        cfg.setdefault("fol_portfolio_arms", ["full9_default", "stage2_hybrid", "stage2c_round2_only"])
        cfg.setdefault("fol_portfolio_selector", "internal_materialization")
        return pipeline_command(cfg, scenario, extra_args)
    module = "coding_fgf.eval_candidates" if kind == "candidate_generation" else "coding_fgf.analysis.matching_analysis"
    cmd = [sys.executable, "-m", module, "--rodi-root", str(config.get("rodi_root", "/data")),
           "--output-dir", str(output_dir or base / kind), "--scenarios", *scenarios,
           "--cache-dir", str(base / "embedding_cache")]
    fields = ["embedding_provider", "embedding_model", "google_project", "google_location", "google_credentials"]
    for key in fields:
        value = config.get(key)
        if value:
            cmd.extend(["--" + key.replace("_", "-"), str(value)])
    if kind == "candidate_generation":
        values = config.get("k_values") or [config.get("k", 16)]
        cmd.extend(["--k-values", *map(str, values)])
    else:
        cmd.extend(["--candidate-artifact", str(candidate_artifact or base / "candidate_generation" / "candidates_by_method.jsonl")])
        for key, flag in (("llm_provider", "--llm-provider"), ("llm_model", "--model"),
                          ("k", "--top-k"), ("match_workers", "--max-workers"),
                          ("fallback_model", "--fallback-model"), ("api_retries", "--api-retries")):
            if config.get(key) is not None:
                cmd.extend([flag, str(config[key])])
    return cmd + list(extra_args or [])
