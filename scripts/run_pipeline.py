from __future__ import annotations

import argparse
from _config import add_common_args, load_config, pipeline_command, run_or_print


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FGF pipeline from a release YAML config.")
    add_common_args(parser)
    args, unknown = parser.parse_known_args()
    config = load_config(args.config)
    cmd = pipeline_command(config, scenario=args.scenario, extra_args=unknown)
    raise SystemExit(run_or_print(cmd, args.dry_run))


if __name__ == "__main__":
    main()
