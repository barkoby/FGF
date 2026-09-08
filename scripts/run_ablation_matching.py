from __future__ import annotations
import argparse
from _config import add_common_args, load_config, ablation_command, run_or_print

def main():
    parser = argparse.ArgumentParser(description="Run the matching ablation.")
    add_common_args(parser)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--candidate-artifact", default=None)
    args, extra = parser.parse_known_args()
    cmd = ablation_command(load_config(args.config), "matching", scenario=args.scenario,
        output_dir=args.output_dir, candidate_artifact=args.candidate_artifact, extra_args=extra)
    raise SystemExit(run_or_print(cmd, args.dry_run))

if __name__ == "__main__":
    main()
