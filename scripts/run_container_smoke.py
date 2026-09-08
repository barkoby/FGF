"""Offline fixture exercising PostgreSQL, portfolio execution, RDF and reports."""
from pathlib import Path
import json
import os
import sys
import tempfile
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "scripts")]
from test_core import write_fixture
from coding_fgf.cli import main
from run_failure_analysis import summarize

def forbidden(*args, **kwargs):
    raise AssertionError("The container smoke must not call a live provider")

def run():
    with tempfile.TemporaryDirectory(prefix="fgf-smoke-") as scratch:
        root = Path(scratch)
        write_fixture(root / "data")
        work = root / "output"
        args = ["run-paper-compare", "--rodi-root", str(root), "--work", str(work),
                "--scenarios", "mini", "--fraction", "1", "--offline",
                "--fol-portfolio", "--fol-ablation-report",
                "--db-host", os.getenv("DB_HOST", "smoke-postgres"),
                "--db-user", "postgres", "--db-password", "postgres"]
        with patch("openai.OpenAI", forbidden), patch("coding_fgf.llm.call_structured_json", forbidden), patch("coding_fgf.google_vertex._post_vertex", forbidden):
            try:
                main(args)
            except Exception:
                for report in work.rglob("fol_portfolio_candidate_report.json"):
                    print(report, report.read_text(), flush=True)
                raise
        scenario = work / "runs" / "mini"
        assert (scenario / "import.ttl").stat().st_size > 0
        metrics = scenario / "eval" / "metrics_details.json"
        assert len(json.loads(metrics.read_text())) > 0
        assert (scenario / "fol_ablation" / "fol_selection_comparison.csv").exists()
        summarize(metrics, work / "failure_summary.csv")
        print("CONTAINER_SMOKE_OK: PostgreSQL, generated RDF, evaluation, FOL comparison, failure analysis")

if __name__ == "__main__":
    run()
