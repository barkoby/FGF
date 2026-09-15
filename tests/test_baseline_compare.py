from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

import pytest

from coding_fgf.baseline import (
    check_llm4vkg_resources,
    db_load_commands,
    default_search_path_from_dump,
    sync_llm4vkg_devset,
    write_llm4vkg_config,
)
from coding_fgf.cli import _fol_repair_acceptance_decision, main
from coding_fgf.benchmark import record_benchmark_run, summarize_benchmark
from coding_fgf.compare import compare_runs, compare_to_paper, decide_promotion
from coding_fgf.llm import (
    augment_candidate_rows_with_forced_matches,
    build_discriminator_candidate_rows,
    clear_llm_events,
    compact_match_request,
    generate_fol_repair_prompt,
    llm_events,
    llm_match,
    _match_one_live,
    _match_prompt,
    generate_fol_prompt,
    match_validation_issues,
    repair_matches_with_candidates,
    validate_matches,
)
from coding_fgf.fol import fol_validation_issues
from coding_fgf.logging_utils import log_info
from coding_fgf.schema import Column, ForeignKey, Table
from test_core import work_path, write_fixture


def test_llm4vkg_resource_checks_and_config_generation() -> None:
    root = work_path("baseline_config")
    llm = root / "LLM4VKG"
    (llm / "resources" / "ontop").mkdir(parents=True)
    (llm / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

    status = check_llm4vkg_resources(llm)
    by_name = {check.name: check.ok for check in status.checks}
    assert by_name["root"]
    assert by_name["pyproject"]
    assert not by_name["ontop"]

    config = write_llm4vkg_config(llm, api_names=["gpt_4o"], db_host="db", db_port=15432, db_user="u", db_password="p")
    text = config.read_text(encoding="utf-8")
    assert "db_config" in text
    assert "api_names = ['gpt_4o']" in text
    assert "os.getenv('DB_HOST', 'db')" in text


def test_devset_sync_and_db_command_construction() -> None:
    root = work_path("baseline_sync")
    devset = root / "devset"
    write_fixture(devset)
    llm = root / "llm"

    target = sync_llm4vkg_devset(devset, llm, ["mini"])
    assert (target / "mini" / "dump.sql").exists()

    commands = db_load_commands(devset, ["mini"], db_host="localhost", db_port=5432, db_user="postgres", db_name="postgres")
    assert commands[0][0] == "psql"
    assert str(devset / "mini" / "dump.sql") in commands[0]


def test_default_search_path_from_dump_uses_non_public_schema() -> None:
    root = work_path("dump_search_path")
    dump = root / "dump.sql"
    dump.write_text("SET search_path = npd, pg_catalog;\nCREATE TABLE x (id integer);\n", encoding="utf-8")

    assert default_search_path_from_dump(dump) == '"npd", public'


def test_validate_matches_accepts_string_null_as_llm_null_decision() -> None:
    candidate_rows = [
        {
            "source": {
                "id": "source-class:join_table",
                "kind": "class",
                "uri": "urn:source#join_table",
            },
            "candidates": [],
        }
    ]
    rows = [
        {
            "source_id": "source-class:join_table",
            "target_uri": "null",
            "target_id": "NO_MATCH",
            "confidence": 0.0,
            "reason": "no supplied candidate fits",
        }
    ]

    matches = validate_matches(rows, candidate_rows)

    assert len(matches) == 1
    assert matches[0].target_uri is None
    assert matches[0].target_id is None


def test_match_prompt_for_empty_candidates_forbids_invented_uris() -> None:
    prompt = _match_prompt(
        {
            "source": {
                "id": "source-class:join_table",
                "kind": "class",
                "uri": "urn:source#join_table",
                "text": "kind: class local name: join table",
            },
            "candidates": [],
        }
    )

    assert "The supplied candidate list is empty" in prompt
    assert "target_uri null" in prompt
    assert "Do not invent" in prompt


def test_fol_prompt_is_scoped_to_relevant_tables() -> None:
    root = work_path("fol_relevant_tables")
    scenario = write_fixture(root)
    from coding_fgf.schema import parse_sql_dump

    tables = parse_sql_dump(scenario / "dump.sql")
    prompt = generate_fol_prompt(
        [
            {
                "source_id": "source-class:papers",
                "source_kind": "class",
                "target_uri": "http://ex#Paper",
                "target_id": "class:http://ex#Paper",
            }
        ],
        tables,
    )

    assert '"name": "papers"' in prompt
    assert '"name": "people"' in prompt


def test_discriminator_candidate_rows_detect_boolean_flags() -> None:
    tables = {
        "Person": Table(
            "Person",
            [Column("ID"), Column("is_Author", "boolean"), Column("name")],
            ["ID"],
        )
    }
    target_records = [
        {"id": "class:http://ex#Author", "kind": "class", "uri": "http://ex#Author", "local_name": "Author", "label": "author"},
        {"id": "class:http://ex#Paper", "kind": "class", "uri": "http://ex#Paper", "local_name": "Paper", "label": "paper"},
    ]

    rows = build_discriminator_candidate_rows(
        tables,
        {"Person": [{"ID": "1", "is_Author": "t", "name": "Ada"}]},
        target_records,
        [],
    )

    assert rows[0]["source"]["id"] == "source-discriminator:Person.is_Author"
    assert rows[0]["source"]["row_filter"] == {"column": "is_Author", "truthy": True}
    assert rows[0]["candidates"][0]["uri"] == "http://ex#Author"


def test_match_validation_flags_fk_data_property_mismatch() -> None:
    tables = {
        "Paper": Table(
            "Paper",
            [Column("ID"), Column("hasAuthor")],
            ["ID"],
            [ForeignKey(["hasAuthor"], "Person", ["ID"])],
        ),
        "Person": Table("Person", [Column("ID")], ["ID"]),
    }
    match = {
        "source_id": "source-data:Paper.hasAuthor",
        "source_kind": "data_property",
        "target_uri": "http://ex#title",
        "target_kind": "data_property",
        "target_local_name": "title",
    }

    issues = match_validation_issues(match, tables)

    assert "fk_column_mapped_to_non_identifier_data_property" in issues


def test_fol_validation_requires_discriminator_rule_and_match_ids() -> None:
    tables = {"Person": Table("Person", [Column("ID"), Column("is_Author")], ["ID"])}
    matches = [
        {
            "source_id": "source-discriminator:Person.is_Author",
            "target_uri": "http://ex#Author",
            "row_filter": {"column": "is_Author", "truthy": True},
        }
    ]
    fol = {"rules": {"class": [], "data": [], "object": []}}

    issues = fol_validation_issues(fol, matches, tables)

    assert any(issue["issue"] == "missing_discriminator_class_rule" for issue in issues)


def test_fol_repair_prompt_uses_internal_diagnostics_only() -> None:
    tables = {"Person": Table("Person", [Column("ID"), Column("is_Author")], ["ID"])}
    prompt = generate_fol_repair_prompt(
        {"rules": {"class": [], "data": [], "object": []}},
        [{"source_id": "source-discriminator:Person.is_Author", "target_uri": "http://ex#Author"}],
        tables,
        [{"issue": "missing_discriminator_class_rule"}],
    )

    assert "validation_issues" in prompt
    assert "Q08" not in prompt
    assert "LLM4VKG" not in prompt


def test_fol_repair_rejects_empty_repair_when_original_had_rules() -> None:
    original = {"rules": {"class": [{"target_class": "http://ex#Paper"}], "data": [], "object": []}}
    repaired = {"rules": {"class": [], "data": [], "object": []}}

    accepted, reasons = _fol_repair_acceptance_decision(
        original,
        repaired,
        [{"issue": "missing_match_ids"}],
        [],
        [{"source_id": "source-class:papers", "target_uri": "http://ex#Paper"}],
    )

    assert not accepted
    assert "repaired_fol_removed_all_rules" in reasons
    assert "selected_matches_but_no_repaired_rules" in reasons


def test_fol_repair_rejects_rule_count_collapse_below_half() -> None:
    original = {
        "rules": {
            "class": [{"target_class": "http://ex#Paper"}, {"target_class": "http://ex#Person"}],
            "data": [{"target_property": "http://ex#title"}, {"target_property": "http://ex#name"}],
            "object": [],
        }
    }
    repaired = {"rules": {"class": [{"target_class": "http://ex#Paper"}], "data": [{"target_property": "http://ex#title"}], "object": []}}

    accepted, reasons = _fol_repair_acceptance_decision(
        original,
        repaired,
        [{"issue": "missing_match_ids"}],
        [],
        [{"source_id": "source-class:papers", "target_uri": "http://ex#Paper"}],
    )

    assert accepted
    assert "repaired_rule_count_below_50_percent" not in reasons

    repaired["rules"]["data"] = []
    accepted, reasons = _fol_repair_acceptance_decision(
        original,
        repaired,
        [{"issue": "missing_match_ids"}],
        [],
        [{"source_id": "source-class:papers", "target_uri": "http://ex#Paper"}],
    )
    assert not accepted
    assert "repaired_rule_count_below_50_percent" in reasons
    assert "repaired_fol_removed_data_rules" in reasons


def test_fol_repair_accepts_coverage_preserving_issue_reduction() -> None:
    original = {
        "rules": {
            "class": [{"target_class": "http://ex#Paper"}],
            "data": [{"target_property": "http://ex#title"}],
            "object": [{"target_property": "http://ex#writtenBy"}],
        }
    }
    repaired = {
        "rules": {
            "class": [{"target_class": "http://ex#Paper"}],
            "data": [{"target_property": "http://ex#title"}],
            "object": [{"target_property": "http://ex#writtenBy"}],
        }
    }

    accepted, reasons = _fol_repair_acceptance_decision(
        original,
        repaired,
        [{"issue": "missing_match_ids"}],
        [],
        [{"source_id": "source-class:papers", "target_uri": "http://ex#Paper"}],
    )

    assert accepted
    assert reasons == []


def test_promotion_decision_uses_similarity_margin_and_invalid_rules() -> None:
    yes = decide_promotion("cmt_renamed", coding_f1=0.82, llm4vkg_f1=0.86, invalid_rules=0, similarity_margin=0.05)
    assert yes.promoted
    no = decide_promotion("cmt_renamed", coding_f1=0.80, llm4vkg_f1=0.86, invalid_rules=0, similarity_margin=0.05)
    assert not no.promoted
    invalid = decide_promotion("cmt_renamed", coding_f1=0.90, llm4vkg_f1=0.86, invalid_rules=1, similarity_margin=0.05)
    assert not invalid.promoted


def test_compare_runs_outputs_csv_and_markdown() -> None:
    root = work_path("compare")
    coding = root / "coding" / "cmt_renamed"
    (coding / "eval").mkdir(parents=True)
    with (coding / "eval" / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["precision", "recall", "f1"])
        writer.writeheader()
        writer.writerow({"precision": 0.8, "recall": 0.9, "f1": 0.84})
    (coding / "matches.json").write_text(json.dumps({"matches": [{"target_uri": "u1"}, {"target_uri": None}]}), encoding="utf-8")
    (coding / "fol.json").write_text(json.dumps({"rules": {"class": [{"x": 1}], "data": [], "object": []}}), encoding="utf-8")
    (coding / "import.ttl").write_text("<s> <p> <o> .\n", encoding="utf-8")

    rows = compare_runs(
        coding_root=root / "coding",
        baseline_rows=[{"scenario": "cmt_renamed", "f1": "0.86"}],
        output_dir=root / "out",
        scenarios=["cmt_renamed"],
        similarity_margin=0.05,
    )
    assert rows[0]["promoted"]
    assert rows[0]["no_match_rate"] == 0.5
    assert (root / "out" / "comparison.csv").exists()
    assert "cmt_renamed" in (root / "out" / "comparison.md").read_text(encoding="utf-8")


def test_compare_to_paper_outputs_csv_and_markdown() -> None:
    root = work_path("paper_compare")
    coding = root / "coding" / "cmt_renamed"
    (coding / "eval").mkdir(parents=True)
    with (coding / "eval" / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["precision", "recall", "f1"])
        writer.writeheader()
        writer.writerow({"precision": 0.9, "recall": 0.85, "f1": 0.87})
    (coding / "matches.json").write_text(json.dumps({"matches": [{"target_uri": "u1"}]}), encoding="utf-8")
    (coding / "fol.json").write_text(json.dumps({"rules": {"class": [{"x": 1}], "data": [], "object": []}}), encoding="utf-8")
    (coding / "import.ttl").write_text("<s> <p> <o> .\n", encoding="utf-8")

    rows = compare_to_paper(root / "coding", root / "out", ["cmt_renamed"])

    assert rows[0]["paper_f1"] is None
    assert rows[0]["delta_vs_paper"] is None
    assert not rows[0]["beats_paper_by_0_01"]
    assert (root / "out" / "paper_comparison.csv").exists()
    assert "Paper Comparison" in (root / "out" / "paper_comparison.md").read_text(encoding="utf-8")


def test_benchmark_ledger_and_failed_qpair_report() -> None:
    root = work_path("benchmark_report")
    run_root = root / "run"
    scenario_work = run_root / "runs" / "cmt_renamed"
    (scenario_work / "eval").mkdir(parents=True)
    with (scenario_work / "eval" / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["precision", "recall", "f1"])
        writer.writeheader()
        writer.writerow({"precision": 0.9, "recall": 0.8, "f1": 0.85})
    (scenario_work / "eval" / "metrics_details.json").write_text(
        json.dumps(
            [
                {
                    "id": "QX",
                    "name": "Broken Values",
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "sql_results": ["a\nb"],
                    "sparql_results": ["a\\nb"],
                }
            ]
        ),
        encoding="utf-8",
    )
    (scenario_work / "mapping_diagnostics.json").write_text(
        json.dumps(
            {
                "selected_matches": 3,
                "anonymous_target_count": 0,
                "invalid_target_uri_count": 0,
                "join_table_class_matches": 0,
                "generic_id_data_matches": 1,
                "class_rules": 1,
                "data_rules": 1,
                "object_rules": 1,
            }
        ),
        encoding="utf-8",
    )
    (scenario_work / "matches.json").write_text(json.dumps({"matches": [{"target_uri": "u"}, {"target_uri": None}]}), encoding="utf-8")
    (scenario_work / "fol.json").write_text(json.dumps({"rules": {"class": [{"x": 1}], "data": [], "object": []}}), encoding="utf-8")
    (scenario_work / "import.ttl").write_text("<s> <p> <o> .\n", encoding="utf-8")
    (run_root / "run_metadata.json").write_text(
        json.dumps(
            {
                "scenario_results": [
                    {
                        "scenario": "cmt_renamed",
                        "mode": "live",
                        "llm_events": [
                            "match:pre_repair:repaired=18:live=190:total=208",
                            "match:complete:completed=190:total=190:elapsed=845.1:fallbacks=7:no_matches=74",
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    record = record_benchmark_run(root / "benchmark_runs", run_root, "cmt_renamed", elapsed_seconds=900.0)

    assert record["live_rows"] == 190
    ledger = (root / "benchmark_runs" / "benchmark_ledger.csv").read_text(encoding="utf-8")
    assert "cmt_renamed" in ledger
    assert "0.85" in ledger
    report = (root / "benchmark_runs" / "reports" / "cmt_renamed.md").read_text(encoding="utf-8")
    assert "Broken Values" in report
    assert "value or literal-normalization mismatch" in report


def _write_benchmark_artifacts(
    run_root: Path,
    scenario: str,
    f1: float,
    live_rows: int,
    live_elapsed: float,
    failed: bool = False,
) -> None:
    scenario_work = run_root / "runs" / scenario
    (scenario_work / "eval").mkdir(parents=True)
    with (scenario_work / "eval" / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["precision", "recall", "f1"])
        writer.writeheader()
        writer.writerow({"precision": f1, "recall": f1, "f1": f1})
    details = []
    if failed:
        details.append(
            {
                "id": "Q1",
                "name": "Missing Path",
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "sql_results": ["expected"],
                "sparql_results": [],
            }
        )
    (scenario_work / "eval" / "metrics_details.json").write_text(json.dumps(details), encoding="utf-8")
    (scenario_work / "mapping_diagnostics.json").write_text(
        json.dumps(
            {
                "selected_matches": 4,
                "anonymous_target_count": 0,
                "invalid_target_uri_count": 0,
                "join_table_class_matches": 0,
                "generic_id_data_matches": 0,
                "class_rules": 2,
                "data_rules": 1,
                "object_rules": 1,
            }
        ),
        encoding="utf-8",
    )
    (scenario_work / "matches.json").write_text(json.dumps({"matches": [{"target_uri": "u"}, {"target_uri": None}]}), encoding="utf-8")
    (scenario_work / "fol.json").write_text(
        json.dumps({"rules": {"class": [{"x": 1}, {"x": 2}], "data": [{"x": 3}], "object": [{"x": 4}]}}),
        encoding="utf-8",
    )
    (scenario_work / "import.ttl").write_text("<s> <p> <o> .\n", encoding="utf-8")
    (scenario_work / "target_records.jsonl").write_text(json.dumps({"id": "t1"}) + "\n", encoding="utf-8")
    (scenario_work / "source_records.jsonl").write_text(json.dumps({"id": "s1"}) + "\n", encoding="utf-8")
    (run_root / "run_metadata.json").write_text(
        json.dumps(
            {
                "scenario_results": [
                    {
                        "scenario": scenario,
                        "mode": "live",
                        "llm_events": [
                            f"match:pre_repair:repaired=2:live={live_rows}:total={live_rows + 2}",
                            f"match:complete:completed={live_rows}:total={live_rows}:elapsed={live_elapsed}:fallbacks=0:no_matches=1",
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_benchmark_summary_tracks_first_final_iterations_runtime_cost_and_failures(monkeypatch) -> None:
    from coding_fgf.constants import PAPER_TARGET_F1
    # Invented fixture thresholds, not benchmark reference scores.
    monkeypatch.setitem(PAPER_TARGET_F1, "fixture_ok", 0.8)
    monkeypatch.setitem(PAPER_TARGET_F1, "fixture_blocked", 0.7)
    root = work_path("benchmark_summary")
    benchmark_root = root / "benchmark_runs"
    first_run = root / "first"
    final_run = root / "final"
    blocked_run = root / "blocked"

    _write_benchmark_artifacts(first_run, "fixture_ok", f1=0.80, live_rows=10, live_elapsed=30.0, failed=True)
    _write_benchmark_artifacts(final_run, "fixture_ok", f1=0.88, live_rows=6, live_elapsed=20.0, failed=False)
    _write_benchmark_artifacts(blocked_run, "fixture_blocked", f1=0.50, live_rows=5, live_elapsed=12.0, failed=True)

    record_benchmark_run(benchmark_root, first_run, "fixture_ok", elapsed_seconds=45.0, run_id="first", phase="first_sweep")
    record_benchmark_run(
        benchmark_root,
        final_run,
        "fixture_ok",
        elapsed_seconds=35.0,
        run_id="final",
        phase="improvement",
        improvement_notes="general lexical repair",
    )
    record_benchmark_run(benchmark_root, blocked_run, "fixture_blocked", elapsed_seconds=25.0, run_id="blocked")

    rows = summarize_benchmark(benchmark_root)
    by_scenario = {row["scenario"]: row for row in rows}

    cmt = by_scenario["fixture_ok"]
    assert cmt["first_sweep_f1"] == "0.8"
    assert cmt["final_f1"] == "0.88"
    assert cmt["status"] == "passed"
    assert cmt["improvement_iterations"] == 1
    assert cmt["total_elapsed_seconds"] == 80.0
    assert cmt["total_live_match_seconds"] == 50.0
    assert cmt["total_estimated_api_cost_usd"] > 0
    assert "general lexical repair" in cmt["improvement_notes"]

    conference = by_scenario["fixture_blocked"]
    assert conference["status"] == "blocked"
    assert conference["failed_qpairs"] == 1
    assert conference["top_failure_cause"] == "missing RDF class/property/path"
    assert by_scenario["sigkdd_renamed"]["status"] == "pending"

    summary_csv = (benchmark_root / "benchmark_summary.csv").read_text(encoding="utf-8")
    assert "total_estimated_api_cost_usd" in summary_csv
    summary_md = (benchmark_root / "benchmark_summary.md").read_text(encoding="utf-8")
    assert "Scenarios beaten: 1" in summary_md
    assert "Blocked: fixture_blocked" in summary_md


def test_run_one_benchmark_refuses_live_when_tests_fail(monkeypatch) -> None:
    root = work_path("run_one_gate")
    import coding_fgf.cli as cli

    class FailedTests:
        returncode = 1

    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: FailedTests())
    monkeypatch.setattr(cli, "cmd_run_dev10", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("offline smoke should not run")))
    monkeypatch.setattr(cli, "cmd_run_paper_compare", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("live run should not run")))

    try:
        main(["run-one-benchmark", "--scenario", "mini", "--benchmark-root", str(root / "bench"), "--rodi-root", str(root / "rodi")])
    except SystemExit as exc:
        assert "Unit test gate failed" in str(exc)
    else:
        raise AssertionError("run-one-benchmark accepted a failing test gate")


def test_env_file_loader_overrides_empty_compose_default(monkeypatch) -> None:
    root = work_path("env_loader")
    env_path = root / ".env"
    env_path.write_text("OPENAI_API_KEY=live-key\n", encoding="utf-8")

    import coding_fgf.cli as cli

    monkeypatch.setenv("OPENAI_API_KEY", "")

    cli._load_env_file(env_path)

    assert os.environ["OPENAI_API_KEY"] == "live-key"


def test_run_paper_compare_cli_smoke_without_live_openai(monkeypatch) -> None:
    root = work_path("paper_cli")
    data_root = root / "rodi" / "data"
    write_fixture(data_root)

    import coding_fgf.cli as cli

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        cli,
        "load_postgres_dumps",
        lambda *args, **kwargs: [{"scenario": "mini", "step": "load_dump", "returncode": 0}],
    )
    monkeypatch.setattr(cli, "execute_sql_psycopg2", lambda *args, **kwargs: ["ada@example.org"])

    main(
        [
            "run-paper-compare",
            "--rodi-root",
            str(root / "rodi"),
            "--work",
            str(root / "work"),
            "--scenarios",
            "mini",
            "--fraction",
            "1.0",
            "--offline",
        ]
    )

    metadata = json.loads((root / "work" / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["scenario_results"][0]["mode"] == "offline"
    assert (root / "work" / "runs" / "mini" / "eval" / "summary.csv").exists()
    assert (root / "work" / "comparison" / "paper_comparison.csv").exists()


def test_live_matching_uses_one_compact_request_per_source(monkeypatch) -> None:
    rows = [
        {
            "source": {
                "id": f"source-class:t{i}",
                "kind": "class",
                "uri": f"urn:s{i}",
                "text": "source context " + ("x " * 1000),
            },
            "candidates": [
                {
                    "id": f"class:http://ex#T{i}",
                    "kind": "class",
                    "uri": f"http://ex#T{i}",
                    "text": "candidate context " + ("y " * 1000),
                    "distance": 0.1,
                }
            ],
        }
        for i in range(2)
    ]
    prompts: list[str] = []

    import coding_fgf.llm as llm_module

    def fake_call(prompt: str, schema_name: str, requested_model: str, fallback_model: str | None, **kwargs) -> dict[str, object]:
        prompts.append(prompt)
        payload = json.loads(prompt.split("\n\n", 1)[1])
        source_id = payload["source"]["id"]
        candidate = payload["candidates"][0]
        return {
            "matches": [
                {
                    "source_id": source_id,
                    "target_uri": candidate["uri"],
                    "target_id": candidate["id"],
                    "confidence": 0.9,
                    "reason": "test",
                }
            ]
        }

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(llm_module, "call_structured_json", fake_call)

    matches = llm_match(rows, offline=False)

    assert len(matches) == 2
    assert len(prompts) == 2
    assert all(len(prompt) < 3000 for prompt in prompts)
    assert prompts[0].count("source-class:") == 1
    compact = compact_match_request(rows[0])
    assert len(compact["source"]["text"]) <= 900
    assert len(compact["candidates"][0]["text"]) <= 180


def test_live_matching_thread_pool_preserves_order_and_events(monkeypatch) -> None:
    rows = [
        {
            "source": {"id": f"source-class:t{i}", "kind": "class", "uri": f"urn:s{i}", "text": "source"},
            "candidates": [{"id": f"class:http://ex#T{i}", "kind": "class", "uri": f"http://ex#T{i}", "text": "target"}],
        }
        for i in range(4)
    ]
    import coding_fgf.llm as llm_module

    real_executor = llm_module.ThreadPoolExecutor
    worker_counts: list[int | None] = []

    class RecordingExecutor(real_executor):  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            worker_counts.append(kwargs.get("max_workers"))
            super().__init__(*args, **kwargs)

    def fake_call(prompt: str, schema_name: str, requested_model: str, fallback_model: str | None, **kwargs) -> dict[str, object]:
        payload = json.loads(prompt.split("\n\n", 1)[1])
        source_id = payload["source"]["id"]
        candidate = payload["candidates"][0]
        return {
            "matches": [
                {
                    "source_id": source_id,
                    "target_uri": candidate["uri"],
                    "target_id": candidate["id"],
                    "confidence": 0.9,
                    "reason": "test",
                }
            ]
        }

    clear_llm_events()
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(llm_module, "ThreadPoolExecutor", RecordingExecutor)
    monkeypatch.setattr(llm_module, "call_structured_json", fake_call)

    matches = llm_match(rows, offline=False, max_workers=4)

    assert worker_counts == [4]
    assert [match.source_id for match in matches] == [row["source"]["id"] for row in rows]
    assert not any(
        event.startswith("match:source_fallback")
        or event.startswith("match:partial_fallback")
        or event.startswith("codegen:fallback")
        for event in llm_events()
    )
    assert any(event.startswith("match:complete:") and "fallbacks=0" in event for event in llm_events())


def test_live_matching_no_api_fallback_on_call_error(monkeypatch) -> None:
    row = {
        "source": {"id": "source-class:t", "kind": "class", "uri": "urn:s", "text": "source"},
        "candidates": [{"id": "class:http://ex#T", "kind": "class", "uri": "http://ex#T", "text": "target"}],
    }
    import coding_fgf.llm as llm_module

    def fail_call(prompt: str, schema_name: str, requested_model: str, fallback_model: str | None, **kwargs) -> dict[str, object]:
        raise RuntimeError("api pressure")

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("CODING_FGF_API_MAX_ATTEMPTS", "1")
    monkeypatch.setattr(llm_module, "call_structured_json", fail_call)

    with pytest.raises(RuntimeError, match="api pressure"):
        llm_match([row], offline=False, max_workers=1)


def test_live_matching_bounds_invalid_model_output_retries(monkeypatch) -> None:
    row = {
        "source": {"id": "source-class:t", "kind": "class", "uri": "urn:s", "text": "source"},
        "candidates": [{"id": "class:http://ex#T", "kind": "class", "uri": "http://ex#T", "text": "target"}],
    }
    import coding_fgf.llm as llm_module

    calls = 0

    def invalid_call(prompt: str, schema_name: str, requested_model: str, fallback_model: str | None, **kwargs) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"matches": [{"source_id": "source-class:t", "target_uri": "http://ex#NotACandidate"}]}

    clear_llm_events()
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("CODING_FGF_MODEL_OUTPUT_MAX_ATTEMPTS", "2")
    monkeypatch.setattr(llm_module, "call_structured_json", invalid_call)

    matches, used_fallback = _match_one_live(row, fail_on_invalid=False)

    assert calls == 2
    assert not used_fallback
    assert matches[0].target_uri is None
    assert "failed generic candidate validation" in matches[0].reason
    assert any(event.startswith("match:validation_exhausted:source=source-class:t") for event in llm_events())


def test_live_matching_progress_logs(monkeypatch) -> None:
    rows = [
        {
            "source": {"id": f"source-class:t{i}", "kind": "class", "uri": f"urn:s{i}", "text": "source"},
            "candidates": [{"id": f"class:http://ex#T{i}", "kind": "class", "uri": f"http://ex#T{i}", "text": "target"}],
        }
        for i in range(3)
    ]
    import coding_fgf.llm as llm_module

    def fake_call(prompt: str, schema_name: str, requested_model: str, fallback_model: str | None, **kwargs) -> dict[str, object]:
        payload = json.loads(prompt.split("\n\n", 1)[1])
        candidate = payload["candidates"][0]
        return {"matches": [{"source_id": payload["source"]["id"], "target_uri": candidate["uri"], "target_id": candidate["id"]}]}

    logs: list[str] = []
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(llm_module, "call_structured_json", fake_call)

    llm_match(rows, offline=False, max_workers=2, progress_logger=logs.append, progress_interval_seconds=0, progress_every=2)

    assert any("match start: total=3 workers=2" in line for line in logs)
    assert any("%" in line and "throughput=" in line for line in logs)
    assert any(line.startswith("match complete: 3/3") for line in logs)


def test_timestamped_log_info(capsys) -> None:
    log_info("hello")
    output = capsys.readouterr().out.strip()
    assert re.match(r"^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\] hello$", output)


def test_deterministic_lexical_repair_prefers_obvious_match() -> None:
    row = {
        "source": {"id": "source-class:conferences", "kind": "class", "uri": "urn:conferences"},
        "candidates": [
            {"id": "class:http://cmt#Preference", "kind": "class", "uri": "http://cmt#Preference", "local_name": "Preference"},
            {"id": "class:http://cmt#Conference", "kind": "class", "uri": "http://cmt#Conference", "local_name": "Conference"},
        ],
    }
    matches = llm_match([row], offline=True)
    repaired = repair_matches_with_candidates(matches, [row])
    assert repaired[0].target_uri == "http://cmt#Conference"


def test_deterministic_pre_repair_skips_live_api(monkeypatch) -> None:
    row = {
        "source": {"id": "source-class:conferences", "kind": "class", "uri": "urn:conferences"},
        "candidates": [
            {"id": "class:http://cmt#Preference", "kind": "class", "uri": "http://cmt#Preference", "local_name": "Preference"},
            {"id": "class:http://cmt#Conference", "kind": "class", "uri": "http://cmt#Conference", "local_name": "Conference"},
        ],
    }
    import coding_fgf.llm as llm_module

    def fail_call(prompt: str, schema_name: str, requested_model: str, fallback_model: str | None, **kwargs) -> dict[str, object]:
        raise AssertionError("live API should be skipped for deterministic repair")

    logs: list[str] = []
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(llm_module, "call_structured_json", fail_call)

    matches = llm_match([row], offline=False, progress_logger=logs.append, allow_deterministic_repair=True)

    assert matches[0].target_uri == "http://cmt#Conference"
    assert matches[0].reason == "deterministic lexical pre-repair"
    assert logs == ["match pre-repair: repaired=1 live=0 total=1"]


def test_forced_candidate_augmentation_recovers_exact_repair_targets() -> None:
    rows = [
        {
            "source": {"id": "source-class:program_committees", "kind": "class", "uri": "urn:pc"},
            "candidates": [{"id": "class:http://cmt#ProgramCommitteeChair", "kind": "class", "uri": "http://cmt#ProgramCommitteeChair"}],
        },
        {
            "source": {"id": "source-object:reviews.written", "kind": "object_property", "uri": "urn:written"},
            "candidates": [{"id": "object_property:http://cmt#writtenBy", "kind": "object_property", "uri": "http://cmt#writtenBy"}],
        },
        {
            "source": {"id": "source-class:persons", "kind": "class", "uri": "urn:persons"},
            "candidates": [{"id": "class:http://cmt#ProgramCommittee", "kind": "class", "uri": "http://cmt#ProgramCommittee"}],
        },
    ]
    targets = [
        {"id": "class:http://cmt#ProgramCommittee", "kind": "class", "uri": "http://cmt#ProgramCommittee", "local_name": "ProgramCommittee"},
        {"id": "class:http://cmt#Person", "kind": "class", "uri": "http://cmt#Person", "local_name": "Person"},
        {
            "id": "object_property:http://cmt#writeReview",
            "kind": "object_property",
            "uri": "http://cmt#writeReview",
            "local_name": "writeReview",
        },
    ]

    augmented = augment_candidate_rows_with_forced_matches(rows, targets)

    assert augmented[0]["candidates"][0]["uri"] == "http://cmt#ProgramCommittee"
    assert augmented[1]["candidates"][0]["uri"] == "http://cmt#writeReview"
    assert augmented[2]["candidates"][0]["uri"] == "http://cmt#Person"


def test_domain_code_primary_key_can_materialize_as_semantic_data_property() -> None:
    from coding_fgf.fol import matches_to_fol
    from coding_fgf.schema import Column, Table

    tables = {
        "country": Table("country", [Column("code"), Column("name")], ["code"], []),
    }
    matches = [
        {
            "source_id": "source-class:country",
            "target_uri": "http://ex#Country",
            "target_local_name": "Country",
            "confidence": 0.95,
        },
        {
            "source_id": "source-data:country.code",
            "target_uri": "http://ex#carCode",
            "target_local_name": "carCode",
            "target_domain": ["http://ex#Country"],
            "confidence": 0.95,
        },
    ]

    fol = matches_to_fol(matches, tables)

    assert fol["rules"]["data"] == [
        {"source_table": "country", "source_column": "code", "target_property": "http://ex#carCode", "confidence": 0.95}
    ]
