"""Public-release regressions; all provider calls are mocked."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

from coding_fgf.embeddings import EmbeddingClient, embed_records
from coding_fgf.constants import PAPER_SCENARIOS, PAPER_TARGET_F1

NINE = ["cmt_renamed", "conference_renamed", "sigkdd_renamed", "cmt_structured",
        "conference_structured", "sigkdd_structured", "sigkdd_mixed",
        "conference_nofks", "cmt_denormalized"]

def test_paper_suite_defaults_preserved_without_distributing_scores():
    assert PAPER_SCENARIOS == NINE
    assert PAPER_TARGET_F1 == {}

def test_openai_missing_key_is_not_offline(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        EmbeddingClient(offline=False).embed(["hello"])

def test_openai_failure_is_not_cached_as_live(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "mock")
    monkeypatch.setattr("openai.OpenAI", Mock(side_effect=RuntimeError("provider unavailable")))
    cache = tmp_path / "cache.jsonl"
    with pytest.raises(RuntimeError, match="provider unavailable"):
        embed_records([{"id": "a", "text": "hello"}], cache)
    assert not cache.exists() or not cache.read_text().strip()

def test_offline_cache_is_not_reused_live(tmp_path, monkeypatch):
    cache = tmp_path / "cache.jsonl"
    records = [{"id": "a", "text": "hello"}]
    embed_records(records, cache, offline=True)
    live = Mock(return_value=[[0.25, 0.75]])
    monkeypatch.setattr(EmbeddingClient, "embed", live)
    assert embed_records(records, cache)[0]["embedding"] == [0.25, 0.75]
    assert live.call_count == 1

def test_partial_cache_hits_preserve_order(tmp_path, monkeypatch):
    cache = tmp_path / "cache.jsonl"
    records = [{"id": str(i), "text": str(i)} for i in range(3)]
    monkeypatch.setattr(EmbeddingClient, "embed", lambda self, texts: [[float(t), 1.0] for t in texts])
    embed_records(records[1:2], cache)
    assert [r["id"] for r in embed_records(records, cache)] == ["0", "1", "2"]

@pytest.mark.parametrize("vectors", [[], [[float("nan")]], [[1.0], [1.0, 2.0]]])
def test_malformed_vectors_rejected_before_cache(tmp_path, monkeypatch, vectors):
    monkeypatch.setattr(EmbeddingClient, "embed", lambda self, texts: vectors)
    cache = tmp_path / "cache.jsonl"
    with pytest.raises(ValueError):
        embed_records([{"text": "a"}, {"text": "b"}], cache)
    assert not cache.exists() or not cache.read_text().strip()

@pytest.mark.parametrize("module", ["coding_fgf.llm", "coding_fgf.google_vertex"])
def test_retry_defaults_are_bounded(monkeypatch, module):
    import importlib
    monkeypatch.delenv("CODING_FGF_API_MAX_ATTEMPTS", raising=False)
    assert importlib.import_module(module)._retry_config()[-1] == 3
    monkeypatch.setenv("CODING_FGF_API_MAX_ATTEMPTS", "0")
    with pytest.raises(ValueError):
        importlib.import_module(module)._retry_config()

def test_google_requires_explicit_project(monkeypatch):
    from coding_fgf.google_vertex import google_config
    for name in ("GOOGLE_CLOUD_PROJECT", "GOOGLE_PROJECT"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="project"):
        google_config()

def test_failure_analysis_accepts_real_json_and_excludes_success(tmp_path):
    metrics = tmp_path / "runs" / "mini" / "eval" / "metrics_details.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text(json.dumps([
        {"id":"ok","categories":"class","precision":1,"recall":1,"f1":1,
         "sql_results":["a"],"sparql_results":["a"]},
        {"id":"bad","categories":"object","precision":0,"recall":0,"f1":0,
         "sql_results":["a"],"sparql_results":[]}
    ]))
    out = tmp_path / "failures.csv"
    result = subprocess.run([sys.executable, "scripts/run_failure_analysis.py",
        "--metrics", str(metrics), "--output", str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    import csv
    assert list(csv.DictReader(out.open())) == [
        {"scenario":"mini","failure_category":"zero_sparql_answers","count":"1"}]

def test_matching_has_no_implicit_fallback():
    from coding_fgf.analysis.matching_analysis import build_parser
    assert build_parser().parse_args([]).fallback_model == ""
