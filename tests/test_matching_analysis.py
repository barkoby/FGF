from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import coding_fgf.analysis.matching_analysis as matching_analysis
from coding_fgf.analysis.matching_analysis import adaptive_map, validate_stage1
from coding_fgf.analysis.matching_metrics import (
    aggregate_rows,
    aggregate_self_consistency,
    attach_gold_and_score,
    validate_match_response,
)
from coding_fgf.analysis.matching_prompts import (
    MATCHING_METHODS,
    match_cot_style_v1,
    match_cov_stage1_v1,
    match_cov_stage2_v1,
    match_current_validated_table_v1,
    match_self_consistency_v1,
)
from coding_fgf.analysis.plot_matching_heatmaps import write_heatmaps


def candidate_row() -> dict:
    return {
        "scenario": "mini",
        "method": "openai_small",
        "source": {
            "id": "source-data:people.email",
            "kind": "data_property",
            "source_table": "people",
            "source_column": "email",
            "source_table_role": "entity",
            "source_column_role": "attribute",
            "text": "people email address",
            "primary_key_columns": ["id"],
            "foreign_keys": [],
            "samples": ["ada@example.org"],
        },
        "candidates": [
            {
                "id": "target-data:email",
                "uri": "http://ex#email",
                "kind": "data_property",
                "rank": 1,
                "score": 0.95,
                "text": "email address",
                "label": "email",
                "comment": "Electronic mail address",
                "domain": ["http://ex#Person"],
                "range": ["http://www.w3.org/2001/XMLSchema#string"],
            },
            {
                "id": "target-data:title",
                "uri": "http://ex#title",
                "kind": "data_property",
                "rank": 2,
                "score": 0.45,
                "text": "paper title",
                "label": "title",
                "comment": "",
                "domain": ["http://ex#Paper"],
                "range": ["http://www.w3.org/2001/XMLSchema#string"],
            },
        ],
    }


def write_minimal_rodi_scenario(root: Path, scenario: str) -> None:
    scenario_dir = root / scenario
    (scenario_dir / "queries").mkdir(parents=True)
    (scenario_dir / "dump.sql").write_text(
        """
CREATE TABLE people (
  id integer PRIMARY KEY,
  email text
);
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (scenario_dir / "ontology.ttl").write_text(
        """
@prefix ex: <http://ex#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:Person a owl:Class ; rdfs:label "Person" .
ex:email a owl:DatatypeProperty ;
  rdfs:domain ex:Person ;
  rdfs:range xsd:string .
""".lstrip(),
        encoding="utf-8",
    )
    (scenario_dir / "queries" / "q1.qpair").write_text(
        """
name = q1
sql = SELECT id FROM people
sparql = PREFIX ex: <http://ex#> SELECT ?x WHERE { ?x a ex:Person . }
""".lstrip(),
        encoding="utf-8",
    )


def write_candidate_artifact(path: Path, scenario: str) -> None:
    rows = [
        {
            "scenario": scenario,
            "method": "openai_small",
            "source_id": "source-class:people",
            "candidates": [{"id": "class:http://ex#Person", "uri": "http://ex#Person", "kind": "class", "rank": 1}],
        },
        {
            "scenario": scenario,
            "method": "openai_small",
            "source_id": "source-data:people.email",
            "candidates": [{"id": "data_property:http://ex#email", "uri": "http://ex#email", "kind": "data_property", "rank": 1}],
        },
        {
            "scenario": scenario,
            "method": "openai_small",
            "source_id": "source-data:people.id",
            "candidates": [],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    path.with_name("method_configs.json").write_text(json.dumps({"embedding_model":"text-embedding-3-small"}), encoding="utf-8")


def test_prompt_construction_for_each_method() -> None:
    row = candidate_row()
    prompts = [
        match_current_validated_table_v1([row]),
        match_cot_style_v1(row),
        match_self_consistency_v1(row),
        match_cov_stage1_v1(row),
        match_cov_stage2_v1(row, {"source_id": row["source"]["id"], "shortlist": [row["candidates"][0]]}),
    ]
    for prompt in prompts:
        assert "JSON" in prompt
        assert row["source"]["id"] in prompt
        assert "http://ex#email" in prompt
    assert set(MATCHING_METHODS) == {
        "current_validated",
        "cot_prompt",
        "self_consistency",
        "chain_of_verification",
    }


def test_json_validation_accepts_valid_candidate_and_scores_multi_gold() -> None:
    row = candidate_row()
    parsed = validate_match_response(
        {
            "matches": [
                {
                    "source_id": row["source"]["id"],
                    "target_uri": "http://ex#email",
                    "target_id": "target-data:email",
                    "confidence": 0.88,
                    "decision": "selected",
                    "reason": "Email evidence",
                }
            ]
        },
        row,
        "cot_prompt",
    )
    scored = attach_gold_and_score(parsed, {row["source"]["id"]: ["http://ex#altEmail", "http://ex#email"]})
    assert scored["candidate_rank"] == 1
    assert scored["correct"] is True


def test_invalid_selected_uri_is_rejected() -> None:
    row = candidate_row()
    parsed = validate_match_response(
        {"matches": [{"source_id": row["source"]["id"], "target_uri": "http://ex#invented"}]},
        row,
        "cot_prompt",
    )
    assert parsed["invalid_selection"] is True
    assert parsed["error_type"] == "invalid_target_uri"


def test_missing_source_id_is_invalid() -> None:
    row = candidate_row()
    parsed = validate_match_response({"matches": []}, row, "cot_prompt")
    assert parsed["invalid_selection"] is True
    assert parsed["error_type"] == "missing_source_id"


def test_null_decision_is_preserved() -> None:
    row = candidate_row()
    parsed = validate_match_response(
        {
            "matches": [
                {
                    "source_id": row["source"]["id"],
                    "target_uri": None,
                    "confidence": 0.2,
                    "decision": "no_match",
                    "null_category": "ambiguous_candidate",
                }
            ]
        },
        row,
        "cot_prompt",
    )
    assert parsed["predicted_target_uri"] is None
    assert parsed["decision"] == "no_match"
    assert parsed["null_category"] == "ambiguous_candidate"


def test_precision_recall_f1_calculation() -> None:
    row = candidate_row()
    good = attach_gold_and_score(
        validate_match_response(
            {"matches": [{"source_id": row["source"]["id"], "target_uri": "http://ex#email"}]},
            row,
            "cot_prompt",
        ),
        {row["source"]["id"]: ["http://ex#email"]},
    )
    bad_source = {**row, "source": {**row["source"], "id": "source-data:people.name"}}
    bad = attach_gold_and_score(
        validate_match_response(
            {"matches": [{"source_id": "source-data:people.name", "target_uri": "http://ex#title"}]},
            bad_source,
            "cot_prompt",
        ),
        {"source-data:people.name": ["http://ex#email"]},
    )
    metrics = aggregate_rows([good, bad], ["method"])[0]
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)


def test_self_consistency_aggregation_and_tie_breaking() -> None:
    row = candidate_row()
    samples = [
        validate_match_response(
            {"matches": [{"source_id": row["source"]["id"], "target_uri": "http://ex#email", "confidence": 0.9}]},
            row,
            "self_consistency",
        ),
        validate_match_response(
            {"matches": [{"source_id": row["source"]["id"], "target_uri": "http://ex#title", "confidence": 0.7}]},
            row,
            "self_consistency",
        ),
    ]
    aggregated = aggregate_self_consistency(samples, row, confidence_threshold=0.5)
    assert aggregated["predicted_target_uri"] == "http://ex#email"
    assert aggregated["tie"] is True
    assert aggregated["vote_distribution"]["http://ex#email"] == 1


def test_self_consistency_uses_own_top_k_and_rejects_outside_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    row = candidate_row()
    row["candidates"].append(
        {
            "id": "target-data:abstract",
            "uri": "http://ex#abstract",
            "kind": "data_property",
            "rank": 3,
            "score": 0.25,
            "text": "abstract",
        }
    )
    seen_top_k: list[int] = []

    def fake_prompt(candidate: dict, top_k: int) -> str:
        seen_top_k.append(top_k)
        return "prompt"

    class FakeCaller:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def call(self, prompt: str, schema_name: str) -> matching_analysis.JsonCallResult:
            return matching_analysis.JsonCallResult(
                data={"matches": [{"source_id": row["source"]["id"], "target_uri": "http://ex#email", "confidence": 0.8}]},
                model_used="fake",
                usage={"input_tokens": 1, "output_tokens": 1},
            )

    monkeypatch.setattr(matching_analysis, "match_self_consistency_v1", fake_prompt)
    monkeypatch.setattr(matching_analysis, "OpenAIJsonCaller", FakeCaller)
    args = SimpleNamespace(
        model="fake",
        fallback_model="fake",
        self_consistency_temperature=0.4,
        self_consistency_samples=1,
        self_consistency_top_k=2,
        max_workers=1,
        api_retries=0,
    )
    results = matching_analysis.run_self_consistency([row], args, {})
    assert seen_top_k == [2]
    assert results[0]["predicted_target_uri"] == "http://ex#email"

    outside_prompt = validate_match_response(
        {"matches": [{"source_id": row["source"]["id"], "target_uri": "http://ex#abstract"}]},
        row,
        "self_consistency",
        allowed_uris=[candidate["uri"] for candidate in row["candidates"][:2]],
    )
    assert outside_prompt["invalid_selection"] is True
    assert outside_prompt["error_type"] == "invalid_target_uri"


def test_matching_analysis_parser_defaults_and_dry_run_include_self_consistency_top_k(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = matching_analysis.build_parser()
    args = parser.parse_args([])
    assert args.top_k == 16
    assert args.self_consistency_top_k == 2
    assert args.current_group_size == 12

    data_root = tmp_path / "data"
    write_minimal_rodi_scenario(data_root, "sigkdd_mixed")
    matching_analysis.main(["--rodi-root", str(tmp_path), "--dry-run", "--scenario", "sigkdd_mixed", "--method", "self_consistency"])
    output = json.loads(capsys.readouterr().out)
    assert output["top_k"] == 16
    assert output["self_consistency_top_k"] == 2
    assert output["current_group_size"] == 12
    assert output["methods"] == ["self_consistency"]


def test_dev10_discovery_and_artifact_alias_join(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    write_minimal_rodi_scenario(data_root, "fgf_dev10_mini")
    write_minimal_rodi_scenario(data_root, "other")
    scenarios = matching_analysis.discover_dev10_scenarios(tmp_path)
    assert scenarios == ["fgf_dev10_mini"]
    assert matching_analysis.base_scenario_for_artifact("fgf_dev10_mini") == "mini"

    artifact_path = tmp_path / "candidates.jsonl"
    write_candidate_artifact(artifact_path, "mini")
    loaded = matching_analysis.load_candidate_artifact_rows(
        artifact_path,
        tmp_path,
        ["fgf_dev10_mini"],
        "openai_small",
        16,
    )
    rows, gold = loaded["fgf_dev10_mini"]
    assert {row["scenario"] for row in rows} == {"fgf_dev10_mini"}
    assert rows[0]["candidates"][0]["uri"] == "http://ex#Person"
    assert gold["source-class:people"] == ("http://ex#Person",)


def test_current_group_size_can_force_single_source_groups() -> None:
    row1 = candidate_row()
    row2 = {**candidate_row(), "source": {**candidate_row()["source"], "id": "source-data:people.name", "source_column": "name"}}
    groups = matching_analysis.group_candidate_rows([(0, row1), (1, row2)], max_group_size=1)
    assert len(groups) == 2
    assert all(len(group) == 1 for group in groups)


def test_adaptive_runner_continues_after_successful_windows() -> None:
    seen: list[tuple[str, int]] = []

    def worker(task: str, workers: int) -> dict:
        seen.append((task, workers))
        return {"task": task}

    rows = adaptive_map(
        ["a", "b", "c", "d", "e"],
        worker,
        lambda task, error, attempts, workers: {"api_error": True},
        max_workers=2,
        api_retries=0,
    )
    assert [row["task"] for row in rows] == ["a", "b", "c", "d", "e"]
    assert [task for task, _ in seen] == ["a", "b", "c", "d", "e"]


def test_chain_of_verification_stage_parsing_and_stage2_validation() -> None:
    row = candidate_row()
    ok, stage1 = validate_stage1(
        {
            "source_id": row["source"]["id"],
            "shortlist": [
                {
                    "target_uri": "http://ex#email",
                    "target_id": "target-data:email",
                    "candidate_rank": 1,
                    "confidence": 0.9,
                    "reasons": ["name", "sample", "domain"],
                }
            ],
            "null_option": {"allowed": True, "reasons": ["a", "b", "c"]},
        },
        row,
    )
    assert ok is True
    assert stage1["shortlist"][0]["target_uri"] == "http://ex#email"
    parsed = validate_match_response(
        {"matches": [{"source_id": row["source"]["id"], "target_uri": "http://ex#email"}]},
        row,
        "chain_of_verification",
        allowed_uris=["http://ex#email"],
    )
    assert parsed["invalid_selection"] is False


def test_heatmap_generation_from_fake_metrics(tmp_path: Path) -> None:
    rows = []
    for scenario in ("s1", "s2"):
        for method in MATCHING_METHODS:
            rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "precision": 0.5,
                    "recall": 0.6,
                    "f1": 0.55,
                    "evaluated_sources": 10,
                    "invalid_selections": 1,
                    "api_errors": 0,
                    "avg_candidate_rank": 2.0,
                    "error_rate": 0.2,
                }
            )
    write_heatmaps(tmp_path, rows, MATCHING_METHODS)
    assert (tmp_path / "heatmaps" / "f1_by_dataset_method.png").exists()
    assert (tmp_path / "heatmaps" / "s1_method_metrics.png").exists()


def test_adaptive_runner_drops_workers_and_returns_api_error_rows(monkeypatch) -> None:
    monkeypatch.setattr(matching_analysis.time, "sleep", lambda _: None)
    worker_counts: list[int] = []
    events: list[tuple[str, dict]] = []

    def worker(task: str, workers: int) -> dict:
        worker_counts.append(workers)
        raise TimeoutError("simulated API failure")

    def builder(task: str, error: str, attempts: int, workers: int) -> dict:
        return {"task": task, "api_error": True, "attempts": attempts, "workers": workers, "error": error}

    rows = adaptive_map(
        ["x"],
        worker,
        builder,
        max_workers=4,
        api_retries=2,
        logger=lambda event, **fields: events.append((event, fields)),
        context="test",
    )
    assert worker_counts == [4, 2, 1]
    assert rows == [{"task": "x", "api_error": True, "attempts": 3, "workers": 1, "error": "TimeoutError: simulated API failure"}]
    assert [event for event, _ in events].count("api:workers_reduce") == 2
    assert events[-1][0] == "api:error_row"


def test_adaptive_runner_no_api_error_mode_reduces_and_recovers_within_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    worker_counts: list[int] = []
    events: list[tuple[str, dict]] = []
    calls = {"count": 0}

    class RateLimitError(Exception):
        pass

    def worker(task: str, workers: int) -> dict:
        worker_counts.append(workers)
        calls["count"] += 1
        if calls["count"] < 3:
            raise RateLimitError("simulated pressure")
        return {"task": task, "api_error": False, "workers": workers}

    monkeypatch.setenv("CODING_FGF_NO_API_ERROR_ROWS", "1")
    monkeypatch.setenv("CODING_FGF_OPENAI_RATE_LIMIT_BACKOFF_SECONDS", "0")
    monkeypatch.setattr(matching_analysis.time, "sleep", lambda _seconds: None)

    rows = adaptive_map(
        ["x"],
        worker,
        lambda task, error, attempts, workers: {"api_error": True},
        max_workers=4,
        api_retries=2,
        logger=lambda event, **fields: events.append((event, fields)),
        context="test_no_error_rows",
    )
    assert worker_counts == [4, 2, 1]
    assert rows == [{"task": "x", "api_error": False, "workers": 1}]
    assert [fields["to_workers"] for event, fields in events if event == "api:workers_reduce"] == [2, 1]
    assert "api:error_row" not in [event for event, _ in events]


def test_adaptive_runner_no_api_error_mode_uses_16_worker_ladder(monkeypatch: pytest.MonkeyPatch) -> None:
    worker_counts: list[int] = []
    calls = {"count": 0}

    class RateLimitError(Exception):
        pass

    def worker(task: str, workers: int) -> dict:
        worker_counts.append(workers)
        calls["count"] += 1
        if calls["count"] < 5:
            raise RateLimitError("simulated pressure")
        return {"task": task, "api_error": False, "workers": workers}

    monkeypatch.setenv("CODING_FGF_NO_API_ERROR_ROWS", "1")
    monkeypatch.setenv("CODING_FGF_OPENAI_RATE_LIMIT_BACKOFF_SECONDS", "0")
    monkeypatch.setattr(matching_analysis.time, "sleep", lambda _seconds: None)

    rows = adaptive_map(
        ["x"],
        worker,
        lambda task, error, attempts, workers: {"api_error": True},
        max_workers=16,
        api_retries=4,
        context="test_16_ladder",
    )
    assert worker_counts == [16, 8, 4, 2, 1]
    assert rows == [{"task": "x", "api_error": False, "workers": 1}]


def test_adaptive_runner_no_api_error_mode_raises_setup_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def worker(task: str, workers: int) -> dict:
        raise RuntimeError("BadRequestError: invalid model")

    monkeypatch.setenv("CODING_FGF_NO_API_ERROR_ROWS", "1")
    with pytest.raises(RuntimeError, match="API task failed"):
        adaptive_map(
            ["x"],
            worker,
            lambda task, error, attempts, workers: {"api_error": True},
            max_workers=16,
            api_retries=0,
            context="setup_failure",
        )


def test_analysis_module_does_not_import_generation_or_materialization() -> None:
    module_text = Path("coding_fgf/analysis/matching_analysis.py").read_text(encoding="utf-8")
    forbidden = ("from ..fol", "import coding_fgf.fol", "from ..materialize", "from ..sandbox", "from ..morphkgc")
    assert not any(token in module_text for token in forbidden)


def test_existing_llm_module_does_not_import_analysis() -> None:
    module_text = Path("coding_fgf/llm.py").read_text(encoding="utf-8")
    assert "coding_fgf.analysis" not in module_text


@pytest.mark.live_llm
def test_live_llm_one_source_smoke() -> None:
    if not os.getenv("RUN_LIVE_LLM") or not os.getenv("OPENAI_API_KEY"):
        pytest.skip("set RUN_LIVE_LLM=1 and OPENAI_API_KEY for live smoke")
    from coding_fgf.analysis.matching_analysis import OpenAIJsonCaller

    row = candidate_row()
    prompt = match_cot_style_v1(row)
    caller = OpenAIJsonCaller(model=os.getenv("OPENAI_MATCH_MODEL", "gpt-4.1-nano"), fallback_model="gpt-4.1-nano", temperature=0.0)
    result = caller.call(prompt, "live_smoke")
    parsed = validate_match_response(result.data, row, "cot_prompt")
    assert parsed["invalid_selection"] is False


def test_json_artifact_fields_are_serializable() -> None:
    row = candidate_row()
    parsed = validate_match_response(
        {"matches": [{"source_id": row["source"]["id"], "target_uri": "http://ex#email"}]},
        row,
        "cot_prompt",
    )
    json.dumps(parsed)
