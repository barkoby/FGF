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
    "attribute_coverage_validation": "--attribute-coverage-validation",
    "attribute_coverage_repair": "--attribute-coverage-repair",
    "fewshot_include_matching": "--fewshot-include-matching",
    "fewshot_include_fol": "--fewshot-include-fol",
    "fewshot_include_codegen": "--fewshot-include-codegen",
}
VALUE_FLAGS = {
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
