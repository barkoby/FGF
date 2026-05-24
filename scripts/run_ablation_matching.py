from __future__ import annotations

import argparse
import subprocess
import sys
from _config import load_config, scenario_arg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM matching ablations over candidate artifacts.")
    parser.add_argument("--config", default="configs/openai_best_fol_default.yaml")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--candidate-artifact", default="/outputs/candidate_eval/candidates_by_method.jsonl")
    parser.add_argument("--output-dir", default="/outputs/matching_analysis")
    parser.add_argument("--dry-run", action="store_true")
    args, unknown = parser.parse_known_args()
    config = load_config(args.config)
    scenarios = scenario_arg(config, args.scenario).split(",")
    cmd = [sys.executable, "-m", "coding_fgf.analysis.matching_analysis", "--rodi-root", str(config.get("rodi_root", "/data")), "--output-dir", args.output_dir, "--candidate-artifact", args.candidate_artifact, "--scenarios", *scenarios]
    if config.get("llm_model"):
        cmd.extend(["--model", str(config["llm_model"])])
    cmd.extend(unknown)
    print(" ".join(cmd))
    if args.dry_run:
        return
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
