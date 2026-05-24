from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coding_fgf.cli import build_parser
from _config import load_config, pipeline_command

FORBIDDEN = ["sk-" + "proj-", "OPENAI_API_KEY=" + "sk", "AI" + "za", "application_default_credentials.json{"]


def main() -> None:
    root = ROOT
    config_path = root / "configs" / "openai_best_fol_default.yaml"
    config = load_config(config_path)
    assert config["fol_portfolio"] is True
    assert "stage2_hybrid" in config["fol_portfolio_arms"]
    cmd = pipeline_command(config, scenario="cmt_renamed")
    assert "--fol-portfolio" in cmd
    assert "--fol-portfolio-selector" in cmd
    parser = build_parser()
    parser.parse_args(["run-paper-compare", "--fol-portfolio", "--fol-portfolio-arms", "full9_default,stage2_hybrid,stage2c_round2_only"])
    for path in [root / ".env.example", root / "README.md", config_path]:
        text = path.read_text(encoding="utf-8")
        for marker in FORBIDDEN:
            assert marker not in text, f"forbidden marker {marker!r} in {path}"
    print(json.dumps({"status": "ok", "default_config": str(config_path), "portfolio_default": True}, indent=2))


if __name__ == "__main__":
    main()
