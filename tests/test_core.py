from __future__ import annotations

import json
import argparse
import inspect
import shutil
from pathlib import Path

import pytest
from rdflib import Graph, Literal, RDF, URIRef

import coding_fgf.cli as cli_mod
from coding_fgf.attribute_coverage import (
    apply_attribute_coverage_repairs,
    attribute_coverage_diagnostics,
    build_attribute_coverage_repair_prompt,
)
from coding_fgf.devset import select_dev_rows, stable_keep, write_sampled_dump
from coding_fgf.embeddings import embed_records
from coding_fgf.evaluate import calculate_precision_recall_f1, evaluate_graph, parse_qpair
from coding_fgf.fol import _class_compatible, matches_to_fol, validate_fol
from coding_fgf.io import read_json, write_json, write_jsonl
from coding_fgf.lexical import words
from coding_fgf.llm import (
    CODEGEN_FEW_SHOT_PROMPT_VERSION,
    CODEGEN_PROMPT_VERSION,
    FOL_FEW_SHOT_EXAMPLES,
    MATCH_FEW_SHOT_EXAMPLES,
    _budget_fol_chunks,
    _match_prompt,
    default_codegen,
    generate_codegen_prompt,
    generate_fol_prompt,
    generate_fol_repair_round2_prompt,
    generate_targeted_object_repair_prompt,
    llm_fol,
)
from coding_fgf.cli import (
    _apply_fol_repair_arm,
    _canonicalize_fol_target_uris,
    _codegen_and_materialize_with_round2_only_fallback,
    _codegen_self_consistent,
    _copy_frozen_upstream_artifacts,
    _fewshot_enabled,
    _fol_portfolio_arm_settings,
    _fol_portfolio_hard_rejections,
    _fol_portfolio_arms,
    _fol_preservation_gate_decision,
    _score_fol_portfolio_candidate,
    _select_fol_portfolio_candidate,
    _run_fol_single_round2_style_repair,
    _run_fol_standard_repair,
    _source_context_arg,
    _targeted_object_repair_acceptance_decision,
    build_parser,
)
from coding_fgf.fol_repair_round2 import apply_round2_repairs
from coding_fgf.llm import offline_match
from coding_fgf.materialize import _invalid_triples, materialize_graph, materialize_graph_with_log, materialize_to_file, normalize_literal_value
from coding_fgf.materialization_coverage import (
    apply_materialization_coverage_repairs,
    build_materialization_repair_prompts,
    internal_materialization_score,
    materialization_coverage_diagnostics,
)
from coding_fgf.morphkgc import generate_source_r2rml, write_morph_config
from coding_fgf.object_evidence import build_object_link_evidence, object_evidence_summary, validate_object_rules_against_evidence
from coding_fgf.ontology import enrich_source_records_from_morphkgc, parse_ontology_records, source_schema_records, write_records
from coding_fgf.pattern_first import (
    build_pattern_selection_prompts,
    build_schema_graph,
    compile_patterns_to_fol,
    extract_pattern_candidates,
    infer_uri_keys,
    pattern_selection_internal_score,
    select_patterns_from_llm,
    validate_prompt_no_leakage,
)
from coding_fgf.retrieval import build_index, retrieve_candidates
from coding_fgf.sandbox import SandboxViolation, validate_generated_code
from coding_fgf.schema import Column, ForeignKey, SqlData, Table, parse_copy_data, parse_sql_dump, table_role


def work_path(name: str) -> Path:
    path = Path("tests") / "_work" / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def write_fixture(root: Path) -> Path:
    scenario = root / "mini"
    queries = scenario / "queries"
    queries.mkdir(parents=True)
    (scenario / "dump.sql").write_text(
        "\n".join(
            [
                "CREATE TABLE people (",
                "  id integer NOT NULL,",
                "  name text,",
                "  email text",
                ");",
                "CREATE TABLE papers (",
                "  id integer NOT NULL,",
                "  title text,",
                "  author_id integer",
                ");",
                "ALTER TABLE ONLY people ADD CONSTRAINT people_pkey PRIMARY KEY (id);",
                "ALTER TABLE ONLY papers ADD CONSTRAINT papers_pkey PRIMARY KEY (id);",
                "ALTER TABLE ONLY papers ADD CONSTRAINT papers_author FOREIGN KEY (author_id) REFERENCES people(id);",
                "COPY people (id, name, email) FROM stdin;",
                "1\tAda\tada@example.org",
                "2\tGrace\tgrace@example.org",
                r"\.",
                "COPY papers (id, title, author_id) FROM stdin;",
                "10\tEngines\t1",
                "11\tCompilers\t2",
                r"\.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (scenario / "ontology.ttl").write_text(
        "\n".join(
            [
                "@prefix : <http://ex#> .",
                "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
                "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
                "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
                ":Person rdf:type owl:Class ; rdfs:label \"person\" .",
                ":Paper rdf:type owl:Class ; rdfs:label \"paper\" .",
                ":email rdf:type owl:DatatypeProperty ; rdfs:domain :Person .",
                ":title rdf:type owl:DatatypeProperty ; rdfs:domain :Paper .",
                ":hasAuthor rdf:type owl:ObjectProperty ; rdfs:domain :Paper ; rdfs:range :Person .",
            ]
        ),
        encoding="utf-8",
    )
    (queries / "Q01.qpair").write_text(
        "name=Emails\n"
        "sql=SELECT email FROM people\n"
        "sparql=PREFIX : <http://ex#> SELECT ?email WHERE { ?p :email ?email }\n"
        "categories=attrib\n",
        encoding="utf-8",
    )
    return scenario


def test_schema_parse_and_dev_sampling_preserves_fk_parent() -> None:
    tmp_path = work_path("schema")
    scenario = write_fixture(tmp_path)
    tables = parse_sql_dump(scenario / "dump.sql")
    data = parse_copy_data(scenario / "dump.sql")
    assert tables["papers"].foreign_keys[0].ref_table == "people"
    sampled = select_dev_rows(tables, data, fraction=1.0, seed="test")
    assert len(sampled["people"]) == 2
    assert len(sampled["papers"]) == 2
    out = tmp_path / "sampled.sql"
    write_sampled_dump(scenario / "dump.sql", out, sampled)
    assert "Ada" in out.read_text(encoding="utf-8")


def test_morphkgc_source_context_enrichment_preserves_source_ids() -> None:
    tmp_path = work_path("morphkgc_enrich")
    people = Table("people", [Column("id"), Column("name")], ["id"], [])
    papers = Table(
        "papers",
        [Column("id"), Column("author_id")],
        ["id"],
        [ForeignKey(["author_id"], "people", ["id"])],
    )
    tables = {"people": people, "papers": papers}
    records = source_schema_records(
        tables,
        {
            "people": [{"id": "1", "name": "Ada"}],
            "papers": [{"id": "10", "author_id": "1"}],
        },
    )
    graph = Graph()
    graph.add((URIRef("urn:coding-fgf:source:mini/people/1"), RDF.type, URIRef("urn:coding-fgf:source:mini#people")))
    graph.add((URIRef("urn:coding-fgf:source:mini/people/1"), URIRef("urn:coding-fgf:source:mini#people_name"), Literal("Ada")))
    graph.add(
        (
            URIRef("urn:coding-fgf:source:mini/papers/10"),
            URIRef("urn:coding-fgf:source:mini#papers_author_id_to_people"),
            URIRef("urn:coding-fgf:source:mini/people/1"),
        )
    )
    source_graph = tmp_path / "source_graph.ttl"
    graph.serialize(destination=str(source_graph), format="turtle")

    enriched = enrich_source_records_from_morphkgc(records, tables, "mini", source_graph)

    assert [record.id for record in enriched] == [record.id for record in records]
    by_id = {record.id: record for record in enriched}
    assert "morphkgc instance count: 1" in by_id["source-class:people"].text
    assert "morphkgc object samples: Ada" in by_id["source-data:people.name"].text
    assert "morphkgc edge count: 1" in by_id["source-object:papers.author_id"].text


def test_run_paper_compare_source_context_cli_defaults_to_schema() -> None:
    parser = build_parser()
    args = parser.parse_args(["run-paper-compare"])
    assert args.source_context == "schema"
    args = parser.parse_args(["run-paper-compare", "--source-context", "morphkgc"])
    assert args.source_context == "morphkgc"


def test_fol_object_evidence_cli_flags_default_off() -> None:
    parser = build_parser()
    args = parser.parse_args(["run-paper-compare"])
    assert args.match_few_shot_examples is False
    assert args.fol_object_evidence is False
    assert args.fol_targeted_object_repair is False
    assert args.fol_few_shot_examples is False
    assert args.codegen_few_shot_examples is False
    args = parser.parse_args(
        [
            "run-paper-compare",
            "--match-few-shot-examples",
            "--fol-object-evidence",
            "--fol-targeted-object-repair",
            "--codegen-few-shot-examples",
        ]
    )
    assert args.match_few_shot_examples is True
    assert args.fol_object_evidence is True
    assert args.fol_targeted_object_repair is True
    assert args.codegen_few_shot_examples is True
    args = parser.parse_args(["run-paper-compare", "--fol-few-shot-examples"])
    assert args.fol_few_shot_examples is True


def test_ablation_cli_flags_default_to_baseline_behavior() -> None:
    parser = build_parser()
    args = parser.parse_args(["run-paper-compare"])
    assert args.fol_repair_mode == "standard"
    assert args.fol_repair_context == "global"
    assert args.fol_repair_rounds == 1
    assert args.fol_repair_round2 is False
    assert args.fol_repair_preservation_gate is False
    assert args.fol_batching == "none"
    assert args.attribute_coverage is False
    assert args.attribute_coverage_validation is False
    assert args.attribute_coverage_repair is False
    assert args.allow_weak_object_links is False
    assert args.fewshot == "none"
    assert args.use_source_rdf is False
    assert args.fol_portfolio is False
    assert args.fol_portfolio_selector == "internal_materialization"
    assert _fewshot_enabled(args, "match") is False
    assert _source_context_arg(args) == "schema"

    args = parser.parse_args(
        [
            "run-paper-compare",
            "--fol-repair-mode",
            "round2_only",
            "--fol-repair-context",
            "batched",
            "--fol-repair-rounds",
            "2",
            "--fol-repair-round2",
            "--fol-repair-preservation-gate",
            "--fol-batching",
            "hybrid",
            "--attribute-coverage-validation",
            "--attribute-coverage-repair",
            "--fewshot",
            "generic",
            "--fewshot-include-fol",
            "--use-source-rdf",
        ]
    )
    assert args.fol_repair_mode == "round2_only"
    assert args.fol_repair_context == "batched"
    assert args.fol_repair_rounds == 2
    assert args.fol_repair_round2 is True
    assert args.fol_repair_preservation_gate is True
    assert args.fol_batching == "hybrid"
    assert args.attribute_coverage is False
    assert args.attribute_coverage_validation is True
    assert args.attribute_coverage_repair is True
    assert _fewshot_enabled(args, "match") is False
    assert _fewshot_enabled(args, "fol") is True
    assert _fewshot_enabled(args, "codegen") is False
    assert _source_context_arg(args) == "morphkgc"
    args = parser.parse_args(
        [
            "run-paper-compare",
            "--fol-portfolio",
            "--fol-portfolio-arms",
            "full9_default,stage2_hybrid,stage2c_round2_only",
            "--fol-portfolio-selector",
            "internal_materialization",
        ]
    )
    assert args.fol_portfolio is True
    assert _fol_portfolio_arms(args) == ["full9_default", "stage2_hybrid", "stage2c_round2_only"]


def test_controlled_fol_repair_ablation_cli_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-fol-repair-ablation",
            "--fol-batching",
            "hybrid",
            "--fol-repair-arms",
            "standard,round2_only,standard_then_round2,pre_repair_preserved",
            "--fol-repair-preservation-gate",
        ]
    )
    assert args.func.__name__ == "cmd_run_fol_repair_ablation"
    assert args.fol_batching == "hybrid"
    assert args.fol_repair_arms == "standard,round2_only,standard_then_round2,pre_repair_preserved"
    assert args.fol_repair_mode == "standard"
    assert args.fol_repair_preservation_gate is True


def test_fol_portfolio_arm_presets_are_exact() -> None:
    full9 = _fol_portfolio_arm_settings("full9_default")
    assert full9["fol_batching"] == "none"
    assert full9["fol_repair_mode"] == "standard"
    assert full9["fol_repair_rounds"] == 1
    assert full9["fol_few_shot_examples"] is False
    assert full9["attribute_coverage_validation"] is False

    stage2 = _fol_portfolio_arm_settings("stage2_hybrid")
    assert stage2["fol_batching"] == "hybrid"
    assert stage2["fol_repair_mode"] == "standard"
    assert stage2["fol_repair_rounds"] == 2
    assert stage2["fol_repair_max_issues_per_prompt"] == 8

    stage2c = _fol_portfolio_arm_settings("stage2c_round2_only")
    assert stage2c["fol_batching"] == "hybrid"
    assert stage2c["fol_repair_mode"] == "round2_only"
    assert stage2c["fol_repair_rounds"] == 1


def _portfolio_record(
    arm: str,
    generated: int,
    issues: int = 0,
    selected: int = 0,
    zero: int = 0,
    invalid: int = 0,
    hard: list[str] | None = None,
) -> dict[str, object]:
    return {
        "arm": arm,
        "generated_triples": generated,
        "invalid_triples": invalid,
        "issues_after": issues,
        "issues_by_type": {},
        "critical_issue_count": 0,
        "prefixed_target_uri_count": 0,
        "selected_targets_with_emission": selected,
        "zero_emission_selected_targets": zero,
        "unknown_reference_issues": 0,
        "overbroad_issues": 0,
        "fk_like_literal_issues": 0,
        "zero_rule_issues": 0,
        "rule_counts": {"class": 1, "data": 1, "object": 1},
        "hard_rejections": hard or [],
    }


def test_fol_portfolio_hard_rejects_invalid_and_invented_targets() -> None:
    invalid = _portfolio_record("stage2_hybrid", 100, invalid=1)
    invalid["import_exists"] = True
    invalid["materialization_status"] = "success"
    assert "invalid_triples" in _fol_portfolio_hard_rejections(invalid)

    invented = _portfolio_record("stage2_hybrid", 100)
    invented["import_exists"] = True
    invented["materialization_status"] = "success"
    invented["issues_by_type"] = {"target_not_in_selected_matches": 1}
    assert "invented_target_uri" in _fol_portfolio_hard_rejections(invented)


def test_fol_portfolio_selector_uses_internal_triple_sanity_and_tiebreaks() -> None:
    records = [
        _portfolio_record("full9_default", 10597, issues=7, selected=48, zero=33),
        _portfolio_record("stage2_hybrid", 12397, issues=20, selected=49, zero=33),
        _portfolio_record("stage2c_round2_only", 13818, issues=4, selected=53, zero=31),
    ]
    selected = _select_fol_portfolio_candidate(records)
    assert selected["arm"] == "stage2_hybrid"
    assert "f1" not in selected
    assert _score_fol_portfolio_candidate(selected, 12397)[0] == 0

    records = [
        _portfolio_record("full9_default", 9439, issues=7, selected=43, zero=29),
        _portfolio_record("stage2_hybrid", 11486, issues=3, selected=44, zero=30),
        _portfolio_record("stage2c_round2_only", 10907, issues=8, selected=43, zero=31),
    ]
    assert _select_fol_portfolio_candidate(records)["arm"] == "stage2c_round2_only"

    records = [
        _portfolio_record("full9_default", 10938, issues=0, selected=52, zero=63),
        _portfolio_record("stage2_hybrid", 9783, issues=25, selected=46, zero=71, hard=["invalid_triples"]),
        _portfolio_record("stage2c_round2_only", 10100, issues=8, selected=46, zero=70),
    ]
    assert _select_fol_portfolio_candidate(records)["arm"] == "full9_default"


def test_fol_portfolio_selector_source_excludes_evaluation_feedback() -> None:
    source = inspect.getsource(_select_fol_portfolio_candidate) + inspect.getsource(_score_fol_portfolio_candidate)
    forbidden = ["summary.csv", "qpair", "gold", "llm4vkg", "paper_comparison", "precision", "recall", "f1"]
    lowered = source.lower()
    assert all(term not in lowered for term in forbidden)


def test_fol_portfolio_replay_selects_known_restoration_arms_without_eval_files() -> None:
    root = Path("work")
    required = [
        root / "full9_fraction1_20260517/openai_sigkdd_renamed/runs/sigkdd_renamed",
        root / "improve_fgf_20260524_openai_stage2_guarded/runs/sigkdd_renamed",
        root / "full9_fraction1_20260517/openai_sigkdd_structured/runs/sigkdd_structured",
        root / "improve_fgf_20260523_openai_first/stage2_hybrid/runs/sigkdd_structured",
        root / "improve_fgf_20260523_openai_first/stage2c_round2_only_v2/runs/sigkdd_structured",
        root / "full9_fraction1_20260517/openai_sigkdd_mixed/runs/sigkdd_mixed",
        root / "improve_fgf_20260523_openai_first/stage2_hybrid/runs/sigkdd_mixed",
        root / "improve_fgf_20260523_openai_first/stage2c_round2_only_v2/runs/sigkdd_mixed",
    ]
    if not all(path.exists() for path in required):
        pytest.skip("local run artifacts unavailable")

    def replay_record(arm: str, path: Path) -> dict[str, object]:
        materialization = json.loads((path / "import.materialization_log.json").read_text(encoding="utf-8"))
        report = json.loads((path / "fol_validation_report.json").read_text(encoding="utf-8"))
        issues = report.get("issues_after", []) or []
        issue_counts: dict[str, int] = {}
        for issue in issues:
            name = str(issue.get("issue", ""))
            issue_counts[name] = issue_counts.get(name, 0) + 1
        record = _portfolio_record(
            arm,
            int(materialization.get("generated_triples", 0) or 0),
            issues=len(issues),
            invalid=int(materialization.get("invalid_triple_count", 0) or 0),
        )
        record["import_exists"] = (path / "import.ttl").exists()
        record["materialization_status"] = materialization.get("status", "success")
        record["issues_by_type"] = issue_counts
        record["critical_issue_count"] = sum(
            issue_counts.get(name, 0)
            for name in {
                "target_not_in_selected_matches",
                "missing_match_ids",
                "match_ids_do_not_reference_selected_matches",
            }
        )
        record["hard_rejections"] = _fol_portfolio_hard_rejections(record)
        return record

    sigkdd_renamed = [
        replay_record("full9_default", required[0]),
        replay_record("stage2_hybrid", required[1]),
    ]
    assert _select_fol_portfolio_candidate(sigkdd_renamed)["arm"] == "full9_default"

    sigkdd_structured = [
        replay_record("full9_default", required[2]),
        replay_record("stage2_hybrid", required[3]),
        replay_record("stage2c_round2_only", required[4]),
    ]
    assert _select_fol_portfolio_candidate(sigkdd_structured)["arm"] == "stage2_hybrid"

    sigkdd_mixed = [
        replay_record("full9_default", required[5]),
        replay_record("stage2_hybrid", required[6]),
        replay_record("stage2c_round2_only", required[7]),
    ]
    assert _select_fol_portfolio_candidate(sigkdd_mixed)["arm"] == "stage2c_round2_only"


def _json_payload_from_prompt(prompt: str) -> dict[str, object]:
    return json.loads(prompt[prompt.index("{") :])


def test_match_few_shot_prompt_enabled_only_and_safe() -> None:
    row = {
        "source_id": "source-data:articles.title",
        "source": {
            "id": "source-data:articles.title",
            "text": "source data column articles.title",
            "kind": "data_property",
        },
        "candidates": [
            {
                "uri": "http://target.example/title",
                "id": "target-data:title",
                "kind": "datatype_property",
                "text": "target datatype property title",
                "score": 0.91,
                "rank": 1,
            }
        ],
    }

    disabled = _match_prompt(row)
    enabled = _match_prompt(row, few_shot_examples=True)

    assert "few_shot_examples" not in disabled
    assert "Prompt version: fgf_match_v2_fewshot" in enabled
    assert enabled.count('"example_id"') == 4
    assert len(MATCH_FEW_SHOT_EXAMPLES) == 4
    assert "Return JSON:" in enabled
    assert "Valid non-null target_uri values are exactly these supplied candidate URIs" in enabled
    for forbidden in ("cmt_", "conference_", "sigkdd", "Q38", "LLM4VKG", "BootOX", "gold answers"):
        assert forbidden not in enabled


def test_fol_few_shot_prompt_enabled_only_and_safe() -> None:
    tables = {"artifacts": Table("artifacts", [Column("id"), Column("title")], ["id"])}
    matches = [{"source_id": "source-data:artifacts.title", "target_uri": "http://target.example/title"}]

    disabled = generate_fol_prompt(matches, tables)
    enabled = generate_fol_prompt(matches, tables, few_shot_examples=True)

    assert "few_shot_examples" not in disabled
    assert "Prompt version: fgf_fol_v3_fewshot_contextual" in enabled
    assert enabled.count('"example_id"') == 4
    assert len(FOL_FEW_SHOT_EXAMPLES) == 4
    assert "Return JSON only" in enabled
    assert "Use only target URIs that appear in the supplied selected matches" in enabled
    for forbidden in ("cmt_", "conference_", "sigkdd", "Q38", "LLM4VKG", "BootOX", "gold answers"):
        assert forbidden not in enabled


def test_generic_fewshot_example_files_are_synthetic_and_safe() -> None:
    root = Path("prompts") / "examples"
    files = [
        root / "matching" / "generic.json",
        root / "fol_generation" / "generic.json",
        root / "codegen" / "generic.json",
    ]
    forbidden = ("cmt_", "conference_", "sigkdd", "mondial", "npd_", "Q38", "LLM4VKG", "BootOX", "gold", "benchmark score")
    for path in files:
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        parsed = json.loads(text)
        assert parsed["examples"]
        lowered = text.lower()
        for token in forbidden:
            assert token.lower() not in lowered


def test_fol_prompt_context_excludes_unrelated_tables_and_matches() -> None:
    tables = {
        "artifacts": Table("artifacts", [Column("id"), Column("title")], ["id"]),
        "unrelated": Table("unrelated", [Column("id"), Column("label")], ["id"]),
    }
    matches = [{"source_id": "source-data:artifacts.title", "target_uri": "http://target.example/title"}]

    payload = _json_payload_from_prompt(generate_fol_prompt(matches, tables, few_shot_examples=True))

    table_names = {table["name"] for table in payload["tables"]}
    assert table_names == {"artifacts"}
    assert payload["matches"] == matches


def test_llm_fol_filters_chunk_context_and_object_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    tables = {
        "agents": Table("agents", [Column("id")], ["id"]),
        "artifacts": Table(
            "artifacts",
            [Column("id"), Column("owner_id")],
            ["id"],
            [ForeignKey(["owner_id"], "agents", ["id"])],
        ),
        "tasks": Table(
            "tasks",
            [Column("id"), Column("assignee_id")],
            ["id"],
            [ForeignKey(["assignee_id"], "agents", ["id"])],
        ),
    }
    data = SqlData(rows={"agents": [{"id": "1"}], "artifacts": [{"id": "10", "owner_id": "1"}], "tasks": [{"id": "20", "assignee_id": "1"}]})
    matches = [
        {"source_id": "source-object:artifacts.owner_id", "target_uri": "http://target.example/ownedBy"},
        {"source_id": "source-object:tasks.assignee_id", "target_uri": "http://target.example/assignedTo"},
    ]
    evidence = build_object_link_evidence(matches, tables, data)
    prompts: list[str] = []

    def fake_call(prompt: str, *args: object, **kwargs: object) -> dict[str, object]:
        prompts.append(prompt)
        return {"rules": {"class": [], "data": [], "object": []}}

    monkeypatch.setenv("CODING_FGF_FOL_MATCHES_PER_CALL", "1")
    monkeypatch.setattr("coding_fgf.llm.call_structured_json", fake_call)

    result = llm_fol(matches, tables, object_link_evidence=evidence, few_shot_examples=True)

    assert result["generation"]["prompt_version"] == "fgf_fol_v3_fewshot_contextual"
    assert len(prompts) == 2
    seen_ids = set()
    for prompt in prompts:
        payload = _json_payload_from_prompt(prompt)
        assert len(payload["matches"]) == 1
        assert len(payload["object_link_evidence"]["entries"]) == 1
        match_id = payload["matches"][0]["source_id"]
        seen_ids.add(match_id)
        assert payload["object_link_evidence"]["entries"][0]["source_id"] == match_id
    assert seen_ids == {"source-object:artifacts.owner_id", "source-object:tasks.assignee_id"}


def test_budget_fol_chunks_splits_until_prompts_fit_budget() -> None:
    tables = {
        "papers": Table("papers", [Column("id"), Column("title"), Column("abstract")], ["id"]),
    }
    matches = [
        {
            "source_id": f"source-data:papers.col{index}",
            "target_uri": f"http://ex#prop{index}",
            "target_kind": "datatype_property",
            "reason": "semantic evidence " * 80,
        }
        for index in range(3)
    ]
    one_len = len(generate_fol_prompt([matches[0]], tables))
    all_len = len(generate_fol_prompt(matches, tables))
    assert all_len > one_len
    token_budget = max(1, (one_len + 400) // 4)

    chunks = _budget_fol_chunks([matches], tables, None, False, token_budget)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(generate_fol_prompt(chunk, tables)) <= token_budget * 4


def test_budget_fol_chunks_fails_when_single_match_exceeds_budget() -> None:
    tables = {"papers": Table("papers", [Column("id"), Column("title")], ["id"])}
    matches = [
        {
            "source_id": "source-data:papers.title",
            "target_uri": "http://ex#title",
            "target_kind": "datatype_property",
            "reason": "semantic evidence " * 40,
        }
    ]

    with pytest.raises(RuntimeError, match="single match"):
        _budget_fol_chunks([matches], tables, None, False, 1)


def test_object_link_evidence_generates_reachable_direct_fk_plan() -> None:
    tables = {
        "persons": Table("persons", [Column("id")], ["id"]),
        "papers": Table(
            "papers",
            [Column("id"), Column("author_id")],
            ["id"],
            [ForeignKey(["author_id"], "persons", ["id"])],
        ),
    }
    data = SqlData(rows={"persons": [{"id": "1"}], "papers": [{"id": "10", "author_id": "1"}]})
    matches = [
        {"source_id": "source-class:papers", "target_uri": "http://ex#Paper", "target_local_name": "Paper"},
        {"source_id": "source-class:persons", "target_uri": "http://ex#Person", "target_local_name": "Person"},
        {
            "source_id": "source-object:papers.author_id",
            "target_uri": "http://ex#hasAuthor",
            "target_local_name": "hasAuthor",
            "target_domain": ["http://ex#Paper"],
            "target_range": ["http://ex#Person"],
        },
    ]
    evidence = build_object_link_evidence(matches, tables, data)
    plans = evidence["entries"][0]["legal_plans"]
    direct = next(plan for plan in plans if plan["orientation"] == "direct_fk")
    assert direct["rule_fields"]["target_table"] == "persons"
    assert direct["reachability"]["estimated_emissions"] == 1


def test_object_link_evidence_generates_join_table_plans() -> None:
    tables = {
        "persons": Table("persons", [Column("id")], ["id"]),
        "papers": Table("papers", [Column("id")], ["id"]),
        "paper_author": Table(
            "paper_author",
            [Column("pid"), Column("aid")],
            ["pid", "aid"],
            [ForeignKey(["pid"], "papers", ["id"]), ForeignKey(["aid"], "persons", ["id"])],
        ),
    }
    data = SqlData(
        rows={
            "persons": [{"id": "1"}],
            "papers": [{"id": "10"}],
            "paper_author": [{"pid": "10", "aid": "1"}],
        }
    )
    matches = [
        {"source_id": "source-class:papers", "target_uri": "http://ex#Paper", "target_local_name": "Paper"},
        {"source_id": "source-class:persons", "target_uri": "http://ex#Person", "target_local_name": "Person"},
        {
            "source_id": "source-object:paper_author.pid",
            "target_uri": "http://ex#hasAuthor",
            "target_local_name": "hasAuthor",
            "target_domain": ["http://ex#Paper"],
            "target_range": ["http://ex#Person"],
        },
    ]
    evidence = build_object_link_evidence(matches, tables, data)
    plans = evidence["entries"][0]["legal_plans"]
    join = next(plan for plan in plans if plan["orientation"] == "join_subject_matched_fk")
    assert join["rule_fields"]["subject_table"] == "papers"
    assert join["rule_fields"]["object_table"] == "persons"
    assert join["reachability"]["estimated_emissions"] == 1


def test_object_rule_must_reference_legal_plan_id_and_copy_fields() -> None:
    tables = {
        "persons": Table("persons", [Column("id")], ["id"]),
        "papers": Table(
            "papers",
            [Column("id"), Column("author_id")],
            ["id"],
            [ForeignKey(["author_id"], "persons", ["id"])],
        ),
    }
    matches = [
        {"source_id": "source-class:papers", "target_uri": "http://ex#Paper", "target_local_name": "Paper"},
        {"source_id": "source-class:persons", "target_uri": "http://ex#Person", "target_local_name": "Person"},
        {
            "source_id": "source-object:papers.author_id",
            "target_uri": "http://ex#hasAuthor",
            "target_local_name": "hasAuthor",
            "target_domain": ["http://ex#Paper"],
            "target_range": ["http://ex#Person"],
        },
    ]
    evidence = build_object_link_evidence(matches, tables)
    plan = evidence["entries"][0]["legal_plans"][0]
    valid_rule = dict(plan["rule_fields"], plan_id=plan["plan_id"])
    assert validate_object_rules_against_evidence({"rules": {"object": [valid_rule]}}, evidence) == []

    invented = dict(valid_rule, target_table="invented")
    issues = validate_object_rules_against_evidence({"rules": {"object": [invented]}}, evidence)
    assert issues[0]["issue"] == "object_rule_does_not_match_plan"
    assert issues[0]["field"] == "target_table"

    missing_plan = dict(plan["rule_fields"])
    issues = validate_object_rules_against_evidence({"rules": {"object": [missing_plan]}}, evidence)
    assert issues[0]["issue"] == "missing_object_plan_id"


def test_object_rule_zero_reachability_is_flagged_from_internal_data_only() -> None:
    tables = {
        "persons": Table("persons", [Column("id")], ["id"]),
        "papers": Table(
            "papers",
            [Column("id"), Column("author_id")],
            ["id"],
            [ForeignKey(["author_id"], "persons", ["id"])],
        ),
    }
    data = SqlData(rows={"persons": [], "papers": [{"id": "10", "author_id": "1"}]})
    matches = [
        {"source_id": "source-class:papers", "target_uri": "http://ex#Paper", "target_local_name": "Paper"},
        {"source_id": "source-class:persons", "target_uri": "http://ex#Person", "target_local_name": "Person"},
        {
            "source_id": "source-object:papers.author_id",
            "target_uri": "http://ex#hasAuthor",
            "target_local_name": "hasAuthor",
            "target_domain": ["http://ex#Paper"],
            "target_range": ["http://ex#Person"],
        },
    ]
    evidence = build_object_link_evidence(matches, tables, data)
    plan = next(plan for plan in evidence["entries"][0]["legal_plans"] if plan["orientation"] == "direct_fk")
    rule = dict(plan["rule_fields"], plan_id=plan["plan_id"])
    issues = validate_object_rules_against_evidence({"rules": {"object": [rule]}}, evidence)
    assert any(issue["issue"] == "object_plan_zero_reachability" for issue in issues)


def test_object_evidence_fol_prompt_requires_legal_plan_id() -> None:
    tables = {
        "persons": Table("persons", [Column("id")], ["id"]),
        "papers": Table(
            "papers",
            [Column("id"), Column("author_id")],
            ["id"],
            [ForeignKey(["author_id"], "persons", ["id"])],
        ),
    }
    matches = [
        {"source_id": "source-class:papers", "target_uri": "http://ex#Paper", "target_local_name": "Paper"},
        {"source_id": "source-class:persons", "target_uri": "http://ex#Person", "target_local_name": "Person"},
        {
            "source_id": "source-object:papers.author_id",
            "target_uri": "http://ex#hasAuthor",
            "target_local_name": "hasAuthor",
            "target_domain": ["http://ex#Paper"],
            "target_range": ["http://ex#Person"],
        },
    ]
    evidence = build_object_link_evidence(matches, tables)
    prompt = generate_fol_prompt(matches, tables, object_link_evidence=evidence)
    assert "fgf_fol_v2_object_evidence" in prompt
    assert "legal_plans" in prompt
    assert "plan_id" in prompt


def test_targeted_object_repair_prompt_excludes_gold_and_preserves_non_object_rules() -> None:
    tables = {
        "persons": Table("persons", [Column("id")], ["id"]),
        "papers": Table(
            "papers",
            [Column("id"), Column("author_id")],
            ["id"],
            [ForeignKey(["author_id"], "persons", ["id"])],
        ),
    }
    matches = [
        {"source_id": "source-class:papers", "target_uri": "http://ex#Paper", "target_local_name": "Paper"},
        {"source_id": "source-class:persons", "target_uri": "http://ex#Person", "target_local_name": "Person"},
        {
            "source_id": "source-object:papers.author_id",
            "target_uri": "http://ex#hasAuthor",
            "target_local_name": "hasAuthor",
            "target_domain": ["http://ex#Paper"],
            "target_range": ["http://ex#Person"],
        },
    ]
    evidence = build_object_link_evidence(matches, tables)
    fol = {
        "rules": {
            "class": [{"source_table": "papers", "target_class": "http://ex#Paper", "id_columns": ["id"]}],
            "data": [],
            "object": [{"source_table": "papers", "target_property": "http://ex#hasAuthor", "target_table": "persons"}],
        }
    }
    prompt = generate_targeted_object_repair_prompt(
        fol,
        matches,
        tables,
        [{"rule_id": "object:0", "issue": "missing_object_plan_id"}],
        evidence,
    )
    lowered = prompt.lower()
    assert "runner preserves them unchanged" in lowered
    assert "qpair" in lowered and "gold answers" in lowered
    assert "baseline" in lowered
    assert "sql expected" not in lowered
    assert "sparql actual" not in lowered


def test_targeted_object_repair_acceptance_rejects_rule_collapse_and_non_object_changes() -> None:
    original = {
        "rules": {
            "class": [{"source_table": "papers", "target_class": "http://ex#Paper"}],
            "data": [{"source_table": "papers", "source_column": "title", "target_property": "http://ex#title"}],
            "object": [{"source_table": "papers", "target_property": "http://ex#hasAuthor", "target_table": "persons"}],
        }
    }
    matches = [{"source_id": "source-object:papers.author_id", "target_uri": "http://ex#hasAuthor"}]
    repaired_empty = {"rules": {"class": [], "data": [], "object": []}}
    accepted, reasons = _targeted_object_repair_acceptance_decision(original, repaired_empty, [], [], matches)
    assert accepted is False
    assert "repaired_fol_removed_all_rules" in reasons

    repaired_changed_data = {
        "rules": {
            "class": original["rules"]["class"],
            "data": [{"source_table": "papers", "source_column": "name", "target_property": "http://ex#title"}],
            "object": original["rules"]["object"],
        }
    }
    accepted, reasons = _targeted_object_repair_acceptance_decision(original, repaired_changed_data, [], [], matches)
    assert accepted is False
    assert "targeted_object_repair_modified_data_rules" in reasons


def test_fol_repair_round2_prompt_is_grounded_and_excludes_evaluation_feedback() -> None:
    tables = {
        "papers": Table("papers", [Column("id"), Column("author_id")], ["id"]),
        "persons": Table("persons", [Column("id")], ["id"]),
    }
    matches = [
        {"source_id": "source-object:papers.author_id", "target_uri": "http://ex#hasAuthor", "match_id": "m1"},
        {"source_id": "source-class:persons", "target_uri": "http://ex#Person", "match_id": "m2"},
    ]
    fol = {"rules": {"class": [], "data": [], "object": [{"target_property": "http://ex#hasAuthor", "source_table": "papers"}]}}
    issues = [{"rule_id": "object:0", "issue": "unknown_object_columns", "column": "invented"}]

    prompt = generate_fol_repair_round2_prompt(fol, matches, tables, issues, max_issues=1)
    payload = _json_payload_from_prompt(prompt)

    assert "repair|drop|keep_with_justification" in prompt
    assert "SQL/SPARQL gold answers" in prompt
    assert "paper baselines" in prompt
    assert payload["issues"][0]["issue"]["issue"] == "unknown_object_columns"
    assert set(payload["allowed_target_uris"]) == {"http://ex#hasAuthor", "http://ex#Person"}
    for forbidden in ("Q38", "LLM4VKG", "BootOX", "sql expected", "sparql actual"):
        assert forbidden.lower() not in prompt.lower()


def test_fol_repair_round2_apply_repairs_drop_repair_and_add() -> None:
    fol = {
        "rules": {
            "class": [],
            "data": [{"source_table": "papers", "source_column": "author_id", "target_property": "http://ex#badLiteral"}],
            "object": [{"source_table": "papers", "target_property": "http://ex#badObject"}],
        }
    }
    repairs = [
        {"issue_id": "data:0", "action": "drop", "reason": "FK-like literal"},
        {
            "issue_id": "object:0",
            "action": "repair",
            "reason": "grounded FK",
            "rule": {"source_table": "papers", "target_property": "http://ex#hasAuthor", "target_table": "persons"},
        },
        {
            "issue_id": "missing:0",
            "action": "add_class_rule",
            "reason": "discriminator match",
            "rule": {"source_table": "persons", "target_class": "http://ex#Person"},
        },
    ]

    repaired, counts = apply_round2_repairs(fol, repairs)

    assert counts["drop"] == 1
    assert counts["repair"] == 1
    assert counts["add_class_rule"] == 1
    assert repaired["rules"]["data"] == []
    assert repaired["rules"]["object"][0]["target_property"] == "http://ex#hasAuthor"
    assert repaired["rules"]["class"][0]["target_class"] == "http://ex#Person"


def test_single_round2_style_repair_uses_one_round2_call_and_skips_old_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = work_path("single_round2_style")
    tables = {"papers": Table("papers", [Column("id"), Column("title")], ["id"])}
    matches = [{"source_id": "source-data:papers.title", "target_uri": "http://ex#title"}]
    fol = {
        "rules": {
            "class": [],
            "data": [
                {
                    "source_table": "papers",
                    "source_column": "bad_title",
                    "target_property": "http://ex#title",
                    "match_ids": ["source-data:papers.title"],
                }
            ],
            "object": [],
        }
    }
    issues = [{"rule_id": "data:0", "issue": "unknown_source_column", "source_column": "bad_title"}]
    calls = {"round2": 0}

    def fake_round2(*args: object, **kwargs: object) -> dict[str, object]:
        calls["round2"] += 1
        assert args[3] == issues
        return {
            "generation": {"source": "mock_round2"},
            "repairs": [
                {
                    "issue_id": "data:0",
                    "action": "repair",
                    "reason": "Use the schema-grounded title column.",
                    "rule": {
                        "source_table": "papers",
                        "source_column": "title",
                        "target_property": "http://ex#title",
                        "match_ids": ["source-data:papers.title"],
                    },
                }
            ],
        }

    monkeypatch.setattr(cli_mod, "llm_repair_fol_round2", fake_round2)
    monkeypatch.setattr(cli_mod, "llm_repair_fol", lambda *args, **kwargs: pytest.fail("old round-1 repair called"))
    monkeypatch.setattr(cli_mod, "llm_repair_object_fol", lambda *args, **kwargs: pytest.fail("targeted object repair called"))

    repaired, final_issues, summary = _run_fol_single_round2_style_repair(
        work=tmp_path,
        fol=fol,
        matches=matches,
        tables=tables,
        issues_before_repair=issues,
        object_link_evidence=None,
        provider="openai",
        model="mock",
        google_project="",
        google_location="",
        google_credentials="",
        max_issues=8,
        allow_drop=True,
        include_attribute_coverage=False,
    )

    assert calls["round2"] == 1
    assert summary["single_round2_style_accepted"] is True
    assert summary["fallback_used"] is False
    assert final_issues == []
    assert repaired["rules"]["data"][0]["source_column"] == "title"
    assert (tmp_path / "fol_repair_single_round2_style_prompt.json").exists()
    prompt_payload = json.loads((tmp_path / "fol_repair_single_round2_style_prompt.json").read_text(encoding="utf-8"))
    prompt = prompt_payload["prompt"].lower()
    for forbidden in ("q38", "llm4vkg", "bootox", "sql expected", "sparql actual"):
        assert forbidden not in prompt


def test_single_round2_style_repair_falls_back_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = work_path("single_round2_fallback")
    tables = {"papers": Table("papers", [Column("id"), Column("title")], ["id"])}
    matches = [{"source_id": "source-data:papers.title", "target_uri": "http://ex#title"}]
    fol = {
        "rules": {
            "class": [],
            "data": [
                {
                    "source_table": "papers",
                    "source_column": "bad_title",
                    "target_property": "http://ex#title",
                    "match_ids": ["source-data:papers.title"],
                }
            ],
            "object": [],
        }
    }
    issues = [{"rule_id": "data:0", "issue": "unknown_source_column", "source_column": "bad_title"}]

    def fail_round2(*args: object, **kwargs: object) -> dict[str, object]:
        raise ValueError("invalid repair json")

    monkeypatch.setattr(cli_mod, "llm_repair_fol_round2", fail_round2)

    repaired, final_issues, summary = _run_fol_single_round2_style_repair(
        work=tmp_path,
        fol=fol,
        matches=matches,
        tables=tables,
        issues_before_repair=issues,
        object_link_evidence=None,
        provider="openai",
        model="mock",
        google_project="",
        google_location="",
        google_credentials="",
        max_issues=8,
        allow_drop=True,
        include_attribute_coverage=False,
    )

    assert repaired == fol
    assert final_issues == issues
    assert summary["fallback_used"] is True
    assert summary["single_round2_style_accepted"] is False
    assert "invalid repair json" in summary["repair_error"]


def test_single_round2_style_repair_falls_back_when_diagnostics_worsen(monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = work_path("single_round2_worse")
    tables = {"papers": Table("papers", [Column("id"), Column("title")], ["id"])}
    matches = [{"source_id": "source-data:papers.title", "target_uri": "http://ex#title"}]
    fol = {
        "rules": {
            "class": [],
            "data": [
                {
                    "source_table": "papers",
                    "source_column": "bad_title",
                    "target_property": "http://ex#title",
                    "match_ids": ["source-data:papers.title"],
                }
            ],
            "object": [],
        }
    }
    issues = [{"rule_id": "data:0", "issue": "unknown_source_column", "source_column": "bad_title"}]

    def worse_round2(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "generation": {"source": "mock_round2"},
            "repairs": [
                {
                    "issue_id": "data:0",
                    "action": "repair",
                    "reason": "Syntactically valid repair, but validator diagnostics worsen.",
                    "rule": {
                        "source_table": "papers",
                        "source_column": "title",
                        "target_property": "http://ex#title",
                        "match_ids": ["source-data:papers.title"],
                    },
                }
            ],
        }

    monkeypatch.setattr(cli_mod, "llm_repair_fol_round2", worse_round2)
    monkeypatch.setattr(
        cli_mod,
        "_all_fol_issues",
        lambda *args, **kwargs: (
            issues + [{"rule_id": "data:0", "issue": "extra_validator_issue"}],
            {},
        ),
    )

    repaired, final_issues, summary = _run_fol_single_round2_style_repair(
        work=tmp_path,
        fol=fol,
        matches=matches,
        tables=tables,
        issues_before_repair=issues,
        object_link_evidence=None,
        provider="openai",
        model="mock",
        google_project="",
        google_location="",
        google_credentials="",
        max_issues=8,
        allow_drop=True,
        include_attribute_coverage=False,
    )

    assert repaired == fol
    assert final_issues == issues
    assert summary["fallback_used"] is True
    assert summary["single_round2_style_accepted"] is False
    assert "single_round2_style_validation_issue_count_worse" in summary["rejection_reasons"]


def test_standard_repair_uses_global_context_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = work_path("standard_repair_global_context")
    tables = {"papers": Table("papers", [Column("id"), Column("title")], ["id"])}
    matches = [{"source_id": "source-data:papers.title", "target_uri": "http://ex#title"}]
    fol = {
        "rules": {
            "class": [],
            "data": [
                {
                    "source_table": "papers",
                    "source_column": "bad_title",
                    "target_property": "http://ex#title",
                    "match_ids": ["source-data:papers.title"],
                }
            ],
            "object": [],
        }
    }
    issues = [{"rule_id": "data:0", "issue": "unknown_source_column", "source_column": "bad_title"}]
    calls = {"global": 0, "round2": 0}

    def fake_global(*args: object, **kwargs: object) -> dict[str, object]:
        calls["global"] += 1
        return {
            "rules": {
                "class": [],
                "data": [
                    {
                        "source_table": "papers",
                        "source_column": "title",
                        "target_property": "http://ex#title",
                        "match_ids": ["source-data:papers.title"],
                    }
                ],
                "object": [],
            },
            "generation": {"source": "mock_global"},
        }

    def fail_round2(*args: object, **kwargs: object) -> dict[str, object]:
        calls["round2"] += 1
        raise AssertionError("round2 repair should not run in global standard mode")

    monkeypatch.setattr(cli_mod, "llm_repair_fol", fake_global)
    monkeypatch.setattr(cli_mod, "llm_repair_fol_round2", fail_round2)

    repaired, final_issues, report = _run_fol_standard_repair(
        work=tmp_path,
        fol=fol,
        matches=matches,
        tables=tables,
        issues=issues,
        object_link_evidence=None,
        provider="openai",
        model="mock",
        google_project="",
        google_location="",
        google_credentials="",
        include_attribute_coverage=False,
    )

    assert calls == {"global": 1, "round2": 0}
    assert report["repair_mode"] == "standard"
    assert report["repair_context"] == "global"
    assert report["repair_accepted"] is True
    assert final_issues == []
    assert repaired["rules"]["data"][0]["source_column"] == "title"


def test_batched_standard_repair_uses_scoped_round2_context(monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = work_path("standard_repair_batched_context")
    tables = {
        "authors": Table("authors", [Column("id"), Column("name")], ["id"]),
        "papers": Table("papers", [Column("id"), Column("title")], ["id"]),
    }
    matches = [
        {"source_id": "source-data:papers.title", "target_uri": "http://ex#title"},
        {"source_id": "source-data:authors.name", "target_uri": "http://ex#name"},
    ]
    fol = {
        "rules": {
            "class": [],
            "data": [
                {
                    "source_table": "papers",
                    "source_column": "bad_title",
                    "target_property": "http://ex#title",
                    "match_ids": ["source-data:papers.title"],
                },
                {
                    "source_table": "authors",
                    "source_column": "name",
                    "target_property": "http://ex#name",
                    "match_ids": ["source-data:authors.name"],
                },
            ],
            "object": [],
        }
    }
    issues = [{"rule_id": "data:0", "issue": "unknown_source_column", "source_column": "bad_title"}]
    calls = {"global": 0, "round2": 0}

    def fail_global(*args: object, **kwargs: object) -> dict[str, object]:
        calls["global"] += 1
        raise AssertionError("global repair should not run in batched standard mode")

    def fake_round2(*args: object, **kwargs: object) -> dict[str, object]:
        calls["round2"] += 1
        return {
            "generation": {"source": "mock_round2"},
            "repairs": [
                {
                    "issue_id": "data:0",
                    "action": "repair",
                    "reason": "Use the schema-grounded column for the scoped issue.",
                    "rule": {
                        "source_table": "papers",
                        "source_column": "title",
                        "target_property": "http://ex#title",
                        "match_ids": ["source-data:papers.title"],
                    },
                }
            ],
        }

    monkeypatch.setattr(cli_mod, "llm_repair_fol", fail_global)
    monkeypatch.setattr(cli_mod, "llm_repair_fol_round2", fake_round2)

    repaired, final_issues, report = _run_fol_standard_repair(
        work=tmp_path,
        fol=fol,
        matches=matches,
        tables=tables,
        issues=issues,
        object_link_evidence=None,
        provider="openai",
        model="mock",
        google_project="",
        google_location="",
        google_credentials="",
        include_attribute_coverage=False,
        repair_context="batched",
        max_issues=1,
        allow_drop=True,
    )

    assert calls == {"global": 0, "round2": 1}
    assert report["repair_mode"] == "standard_contextual_round2"
    assert report["repair_context"] == "batched"
    assert report["repair_accepted"] is True
    assert final_issues == []
    assert repaired["rules"]["data"][0]["source_column"] == "title"
    assert repaired["rules"]["data"][1]["source_table"] == "authors"

    prompt = json.loads((tmp_path / "fol_repair_single_round2_style_prompt.json").read_text(encoding="utf-8"))["prompt"]
    assert "source-data:papers.title" in prompt
    assert "source-data:authors.name" not in prompt
    assert '"name": "papers"' in prompt
    assert '"name": "authors"' not in prompt
    for forbidden in ("Q38", "LLM4VKG", "BootOX", "SQL expected", "SPARQL actual"):
        assert forbidden not in prompt


def test_stage_fol_targeted_object_repair_uses_batched_context_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = work_path("targeted_object_batched_context")
    root = tmp_path / "root"
    scenario_dir = root / "toy"
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "dump.sql").write_text("-- mocked\n", encoding="utf-8")

    tables = {
        "papers": Table(
            "papers",
            [Column("id"), Column("author_id")],
            ["id"],
            [ForeignKey(["author_id"], "persons", ["id"])],
        ),
        "persons": Table("persons", [Column("id")], ["id"]),
    }
    matches = [
        {
            "source_id": "source-object:papers.author_id",
            "target_uri": "http://ex#hasAuthor",
            "target_local_name": "hasAuthor",
            "target_kind": "object_property",
        }
    ]
    object_issue = [{"rule_id": "object:0", "issue": "missing_object_plan_id"}]
    write_json(tmp_path / "matches.json", {"matches": matches})
    write_jsonl(tmp_path / "candidates.jsonl", [])
    write_jsonl(tmp_path / "target_records.jsonl", [])

    fol = {
        "rules": {
            "class": [],
            "data": [],
            "object": [
                {
                    "source_table": "papers",
                    "target_property": "http://ex#hasAuthor",
                    "target_table": "persons",
                    "match_ids": ["source-object:papers.author_id"],
                }
            ],
        }
    }
    calls = {"round2": 0}

    monkeypatch.setattr(cli_mod, "parse_sql_dump", lambda path: tables)
    monkeypatch.setattr(cli_mod, "parse_copy_data", lambda path: SqlData({"papers": [], "persons": []}))
    monkeypatch.setattr(cli_mod, "reask_suspicious_matches", lambda *args, **kwargs: (matches, []))
    monkeypatch.setattr(cli_mod, "build_discriminator_candidate_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(cli_mod, "build_object_link_evidence", lambda *args, **kwargs: {"legal_plans": []})
    monkeypatch.setattr(cli_mod, "llm_fol", lambda *args, **kwargs: fol)
    monkeypatch.setattr(
        cli_mod,
        "llm_repair_object_fol",
        lambda *args, **kwargs: pytest.fail("global targeted object repair should not run"),
    )

    issue_calls = iter([(object_issue, {}), ([], {})])
    monkeypatch.setattr(cli_mod, "_all_fol_issues", lambda *args, **kwargs: next(issue_calls))

    def fake_round2(*args: object, **kwargs: object) -> dict[str, object]:
        calls["round2"] += 1
        assert args[3] == object_issue
        return {
            "generation": {"source": "mock_round2"},
            "repairs": [
                {
                    "issue_id": "round2_issue_1",
                    "rule_id": "object:0",
                    "action": "keep_with_justification",
                    "reason": "The test keeps the supplied object rule using scoped context.",
                    "rule": None,
                }
            ],
        }

    monkeypatch.setattr(cli_mod, "llm_repair_fol_round2", fake_round2)

    cli_mod.stage_fol(
        "toy",
        root,
        tmp_path,
        offline=False,
        use_llm=True,
        llm_provider="google",
        llm_model="gemini-test",
        fol_object_evidence=True,
        fol_targeted_object_repair=True,
        fol_repair_context="batched",
        fol_batching="hybrid",
        fol_repair_max_issues_per_prompt=1,
    )

    report = read_json(tmp_path / "fol_validation_report.json")
    assert calls["round2"] == 1
    assert report["repair_mode"] == "standard_contextual_round2"
    assert report["targeted_object_repair_context_control"] == "delegated_to_contextual_round2"
    prompt = json.loads((tmp_path / "fol_repair_single_round2_style_prompt.json").read_text(encoding="utf-8"))["prompt"]
    for forbidden in ("Q38", "LLM4VKG", "BootOX", "SQL expected", "SPARQL actual"):
        assert forbidden not in prompt


def test_round2_only_runtime_failure_restores_pre_repair_fol(monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = work_path("single_round2_runtime_fallback")
    original = {"class": [{"rule_id": "class:original"}], "data": [], "object": []}
    repaired = {"class": [{"rule_id": "class:repaired"}], "data": [], "object": []}
    write_json(tmp_path / "fol_rules_before_repair.json", original)
    write_json(tmp_path / "fol.json", repaired)
    args = argparse.Namespace(
        fol_repair_mode="round2_only",
        codegen_self_consistency=3,
        codegen_prompt_version=CODEGEN_PROMPT_VERSION,
        llm_provider="openai",
        llm_model="test-model",
        google_project="",
        google_location="",
        google_credentials="",
        fewshot="none",
        codegen_few_shot_examples=False,
    )
    calls = {"codegen": 0, "materialize": 0}

    def fake_codegen(*args: object, **kwargs: object) -> None:
        calls["codegen"] += 1
        (tmp_path / "generated_fgf.py").write_text("# generated\n", encoding="utf-8")

    def fake_materialize(*args: object, **kwargs: object) -> Path:
        calls["materialize"] += 1
        if calls["materialize"] == 1:
            raise ValueError("Generated materializer emitted invalid RDF triples")
        output = args[4] if len(args) > 4 else kwargs["output"]
        Path(output).write_text("@prefix ex: <http://example.org/> .\n", encoding="utf-8")
        return Path(output)

    monkeypatch.setattr(cli_mod, "stage_codegen", fake_codegen)
    monkeypatch.setattr(cli_mod, "materialize_to_file", fake_materialize)

    summary = _codegen_and_materialize_with_round2_only_fallback(
        scenario="toy",
        scenario_work=tmp_path,
        tables={},
        data={},
        args=args,
        offline=False,
        output=tmp_path / "import.ttl",
    )

    assert calls == {"codegen": 2, "materialize": 2}
    assert summary["round2_only_runtime_fallback_used"] is True
    assert read_json(tmp_path / "fol.json") == original
    assert (tmp_path / "fol_rules_rejected_after_invalid_materialization.json").exists()
    assert (tmp_path / "fol_round2_only_runtime_fallback.json").exists()


def test_apply_standard_then_round2_arm_invokes_round1_then_round2(monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = work_path("standard_then_round2_arm")
    calls: list[str] = []
    frozen = {"rules": {"class": [], "data": [], "object": []}}
    round1_fol = {"rules": {"class": [{"source_table": "papers"}], "data": [], "object": []}}
    round2_fol = {"rules": {"class": [{"source_table": "papers"}, {"source_table": "authors"}], "data": [], "object": []}}
    initial_issues = [{"rule_id": "class:0", "issue": "missing_match_ids"}]
    round1_issues = [{"rule_id": "class:1", "issue": "missing_match_ids"}]

    def fake_standard(**kwargs: object) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
        calls.append("standard")
        assert kwargs["issues"] == initial_issues
        return round1_fol, round1_issues, {"repair_mode": "standard", "issues_after": round1_issues}

    def fake_round2(**kwargs: object) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
        calls.append("round2")
        assert kwargs["fol"] == round1_fol
        assert kwargs["issues_before"] == initial_issues
        assert kwargs["issues_after_round1"] == round1_issues
        return round2_fol, [], {"repair_round2_attempted": True, "repair_round2_accepted": True}

    monkeypatch.setattr(cli_mod, "_run_fol_standard_repair", fake_standard)
    monkeypatch.setattr(cli_mod, "_run_fol_repair_round2", fake_round2)

    fol, issues, report = _apply_fol_repair_arm(
        arm="standard_then_round2",
        work=tmp_path,
        frozen_fol=frozen,
        matches=[],
        tables={},
        issues=initial_issues,
        object_link_evidence=None,
        provider="openai",
        model="mock",
        google_project="",
        google_location="",
        google_credentials="",
        max_issues=8,
        allow_drop=True,
        include_attribute_coverage=False,
    )

    assert calls == ["standard", "round2"]
    assert fol == round2_fol
    assert issues == []
    assert report["repair_mode"] == "standard_then_round2"


def test_target_uri_hygiene_canonicalizes_role_prefixed_targets() -> None:
    fol = {
        "rules": {
            "class": [{"target_class": "class:http://ex#Paper"}],
            "data": [{"target_property": "data_property:http://ex#title"}],
            "object": [{"target_property": "object_property:http://ex#writtenBy"}],
        }
    }

    cleaned, report = _canonicalize_fol_target_uris(fol)

    assert cleaned["rules"]["class"][0]["target_class"] == "http://ex#Paper"
    assert cleaned["rules"]["data"][0]["target_property"] == "http://ex#title"
    assert cleaned["rules"]["object"][0]["target_property"] == "http://ex#writtenBy"
    assert report["canonicalized_target_uri_count"] == 3


def test_preservation_gate_rejects_unresolved_critical_repair_issues() -> None:
    fol = {"rules": {"class": [{"target_class": "http://ex#Paper", "source_table": "papers"}], "data": [], "object": []}}
    matches = [{"source_id": "source-class:papers", "target_uri": "http://ex#Paper"}]

    accepted, reasons, report = _fol_preservation_gate_decision(
        fol,
        fol,
        [{"issue": "missing_match_ids", "rule_id": "class:0"}],
        [{"issue": "missing_match_ids", "rule_id": "class:0"}],
        matches,
    )

    assert not accepted
    assert "critical_fol_issue_after_repair:missing_match_ids" in reasons
    assert report["critical_issues_after"]["missing_match_ids"] == 1


def test_preservation_gate_rejects_dropped_selected_targets() -> None:
    original = {
        "rules": {
            "class": [{"target_class": "http://ex#Paper", "source_table": "papers"}],
            "data": [{"target_property": "http://ex#title", "source_table": "papers", "source_column": "title"}],
            "object": [],
        }
    }
    repaired = {"rules": {"class": [{"target_class": "http://ex#Paper", "source_table": "papers"}], "data": [], "object": []}}
    matches = [
        {"source_id": "source-class:papers", "target_uri": "http://ex#Paper"},
        {"source_id": "source-data:papers.title", "target_uri": "http://ex#title"},
    ]

    accepted, reasons, report = _fol_preservation_gate_decision(original, repaired, [{"issue": "x"}], [], matches)

    assert not accepted
    assert "preservation_gate_dropped_selected_data_targets" in reasons
    assert report["dropped_selected_targets"]["data"] == ["http://ex#title"]


def test_preservation_gate_rejects_object_direction_drift() -> None:
    original = {
        "rules": {
            "class": [],
            "data": [],
            "object": [
                {
                    "target_property": "http://ex#writtenBy",
                    "source_table": "papers",
                    "source_columns": ["author_id"],
                    "target_table": "authors",
                    "target_columns": ["id"],
                    "match_ids": ["source-object:papers.author_id"],
                }
            ],
        }
    }
    repaired = {
        "rules": {
            "class": [],
            "data": [],
            "object": [
                {
                    "target_property": "http://ex#writtenBy",
                    "source_table": "authors",
                    "source_columns": ["id"],
                    "target_table": "papers",
                    "target_columns": ["author_id"],
                    "match_ids": ["source-object:papers.author_id"],
                }
            ],
        }
    }
    matches = [{"source_id": "source-object:papers.author_id", "target_uri": "http://ex#writtenBy"}]

    accepted, reasons, report = _fol_preservation_gate_decision(original, repaired, [{"issue": "x"}], [], matches)

    assert not accepted
    assert "preservation_gate_changed_object_grounding" in reasons
    assert report["dropped_object_groundings"] == 1


def test_pre_repair_preserved_arm_skips_llm_and_uses_frozen_fol(monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = work_path("pre_repair_preserved_arm")
    frozen = {
        "rules": {
            "class": [
                {
                    "target_class": "class:http://ex#Paper",
                    "source_table": "papers",
                    "match_ids": ["source-class:papers"],
                }
            ],
            "data": [],
            "object": [],
        }
    }
    tables = {"papers": Table("papers", [Column("id")], ["id"])}
    matches = [{"source_id": "source-class:papers", "target_uri": "http://ex#Paper"}]

    def fail_repair(*args: object, **kwargs: object) -> None:
        raise AssertionError("pre_repair_preserved must not invoke LLM repair")

    monkeypatch.setattr(cli_mod, "llm_repair_fol", fail_repair)
    monkeypatch.setattr(cli_mod, "llm_repair_fol_round2", fail_repair)

    fol, issues, report = _apply_fol_repair_arm(
        arm="pre_repair_preserved",
        work=tmp_path,
        frozen_fol=frozen,
        matches=matches,
        tables=tables,
        issues=[{"rule_id": "class:0", "issue": "target_not_in_selected_matches"}],
        object_link_evidence=None,
        provider="openai",
        model="mock",
        google_project="",
        google_location="",
        google_credentials="",
        max_issues=8,
        allow_drop=True,
        include_attribute_coverage=False,
        preservation_gate=True,
    )

    assert report["repair_mode"] == "pre_repair_preserved"
    assert fol["rules"]["class"][0]["target_class"] == "http://ex#Paper"
    assert issues == []


def test_copy_frozen_upstream_artifacts_preserves_inputs() -> None:
    src = work_path("copy_frozen_src")
    dst = work_path("copy_frozen_dst")
    (src / "matches.json").write_text('{"matches": [{"source_id": "s"}]}', encoding="utf-8")
    (src / "fol_rules_before_repair.json").write_text('{"rules": {"class": []}}', encoding="utf-8")
    (src / "ignored.txt").write_text("ignore me", encoding="utf-8")

    _copy_frozen_upstream_artifacts(src, dst)

    assert (dst / "matches.json").read_text(encoding="utf-8") == '{"matches": [{"source_id": "s"}]}'
    assert (dst / "fol_rules_before_repair.json").read_text(encoding="utf-8") == '{"rules": {"class": []}}'
    assert not (dst / "ignored.txt").exists()


def test_attribute_coverage_flags_missing_data_rules_and_fk_like_literals() -> None:
    tables = {
        "people": Table("people", [Column("id"), Column("name")], ["id"]),
        "papers": Table(
            "papers",
            [Column("id"), Column("title"), Column("author_id")],
            ["id"],
            [ForeignKey(["author_id"], "people", ["id"])],
        ),
    }
    matches = [
        {"source_id": "source-data:papers.title", "target_uri": "http://ex#title", "target_kind": "datatype_property"},
        {"source_id": "source-data:papers.author_id", "target_uri": "http://ex#authorId", "target_kind": "datatype_property"},
    ]
    fol = {"rules": {"data": [{"source_table": "papers", "source_column": "author_id", "target_property": "http://ex#authorId"}]}}

    report = attribute_coverage_diagnostics(matches, fol, tables)

    assert report["selected_datatype_matches"] == 2
    assert report["datatype_matches_with_rules"] == 1
    assert report["datatype_matches_missing_rules"] == 1
    assert report["fk_like_data_rules"] == 1
    assert {issue["issue"] for issue in report["issues"]} >= {"missing_data_rule", "fk_like_data_rule"}


def test_attribute_coverage_flags_unknown_source_columns() -> None:
    tables = {"papers": Table("papers", [Column("id"), Column("title")], ["id"])}
    matches = [{"source_id": "source-data:papers.title", "target_uri": "http://ex#title", "target_kind": "datatype_property"}]
    fol = {
        "rules": {
            "data": [
                {
                    "source_table": "papers",
                    "source_column": "missing_title",
                    "target_property": "http://ex#title",
                    "match_ids": ["source-data:papers.title"],
                }
            ]
        }
    }

    report = attribute_coverage_diagnostics(matches, fol, tables)

    assert any(issue["issue"] == "data_rule_unknown_source_column" for issue in report["issues"])


def test_attribute_coverage_repair_prompt_is_grounded_and_evaluation_blind() -> None:
    tables = {"papers": Table("papers", [Column("id"), Column("title")], ["id"])}
    data = SqlData({"papers": [{"id": "1", "title": "Synthetic title"}]})
    matches = [{"source_id": "source-data:papers.title", "target_uri": "http://ex#title", "target_id": "target-data:title"}]
    fol = {"rules": {"class": [], "data": [], "object": []}}
    report = attribute_coverage_diagnostics(matches, fol, tables, data)

    prompt = build_attribute_coverage_repair_prompt(report, fol, matches, tables, data)
    lowered = prompt.lower()

    assert "http://ex#title" in prompt
    assert "papers" in prompt
    assert "title" in prompt
    for forbidden in ("sparql", "gold", "paper baseline", "qpair", "expected output", "cmt_renamed"):
        assert forbidden not in lowered


def test_attribute_coverage_repairs_reject_invented_targets_and_columns() -> None:
    tables = {"papers": Table("papers", [Column("id"), Column("title")], ["id"])}
    matches = [{"source_id": "source-data:papers.title", "target_uri": "http://ex#title", "target_kind": "datatype_property"}]
    fol = {"rules": {"class": [], "data": [], "object": []}}
    repairs = [
        {
            "issue_id": "attribute:source-data:papers.title",
            "action": "add_data_rule",
            "rule": {
                "source_table": "papers",
                "source_column": "invented",
                "target_property": "http://ex#title",
                "match_ids": ["source-data:papers.title"],
            },
        },
        {
            "issue_id": "attribute:source-data:papers.title",
            "action": "add_data_rule",
            "rule": {
                "source_table": "papers",
                "source_column": "title",
                "target_property": "http://ex#invented",
                "match_ids": ["source-data:papers.title"],
            },
        },
    ]

    repaired, summary = apply_attribute_coverage_repairs(fol, repairs, matches, tables)

    assert repaired["rules"]["data"] == []
    assert summary["repairs_rejected"] == 2


def test_attribute_coverage_repairs_accept_grounded_data_rule() -> None:
    tables = {"papers": Table("papers", [Column("id"), Column("title")], ["id"])}
    matches = [{"source_id": "source-data:papers.title", "target_uri": "http://ex#title", "target_kind": "datatype_property"}]
    fol = {"rules": {"class": [], "data": [], "object": []}}
    repairs = [
        {
            "issue_id": "attribute:source-data:papers.title",
            "action": "add_data_rule",
            "rule": {
                "source_table": "papers",
                "source_column": "title",
                "target_property": "http://ex#title",
                "match_ids": ["source-data:papers.title"],
            },
        }
    ]

    repaired, summary = apply_attribute_coverage_repairs(fol, repairs, matches, tables)

    assert repaired["rules"]["data"][0]["target_property"] == "http://ex#title"
    assert summary["repairs_rejected"] == 0


def test_attribute_coverage_stage_accepts_fk_like_literal_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    work = work_path("attribute_coverage_stage_fk_drop")
    tables = {
        "people": Table("people", [Column("id")], ["id"]),
        "papers": Table(
            "papers",
            [Column("id"), Column("author_id")],
            ["id"],
            [ForeignKey(["author_id"], "people", ["id"])],
        ),
    }
    data = SqlData({"papers": [{"id": "1", "author_id": "7"}]})
    matches = [
        {
            "source_id": "source-data:papers.author_id",
            "target_uri": "http://ex#authorIdentifier",
            "target_kind": "datatype_property",
        }
    ]
    fol = {
        "rules": {
            "class": [{"source_table": "papers", "target_class": "http://ex#Paper", "id_columns": ["id"]}],
            "data": [
                {
                    "source_table": "papers",
                    "source_column": "author_id",
                    "target_property": "http://ex#authorIdentifier",
                    "match_ids": ["source-data:papers.author_id"],
                }
            ],
            "object": [],
        }
    }

    def fake_repair(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "repairs": [
                {
                    "issue_id": "data:0",
                    "action": "drop",
                    "reason": "FK-like literal rule is unsupported as a datatype fact.",
                    "rule": None,
                }
            ],
            "generation": {"source": "test"},
        }

    monkeypatch.setattr(cli_mod, "llm_repair_attribute_coverage", fake_repair)

    repaired, summary = cli_mod._run_attribute_coverage_stage(
        work=work,
        fol=fol,
        matches=matches,
        tables=tables,
        data=data,
        repair_enabled=True,
        offline=False,
        provider="google",
        model="gemini-3.1-flash-lite",
        google_project="",
        google_location="",
        google_credentials="",
    )

    assert summary["repair_accepted"] is True
    assert summary["fallback_used"] is False
    assert repaired["rules"]["data"] == []
    assert (work / "attribute_coverage_summary.csv").exists()


def _materialization_coverage_fixture(tmp_path: Path) -> tuple[list[dict[str, object]], dict[str, object], dict[str, Table], SqlData, Path, dict[str, object]]:
    tables = {
        "papers": Table(
            name="papers",
            columns=[Column("id", "integer"), Column("title", "text"), Column("author_id", "integer")],
            primary_key=["id"],
            foreign_keys=[ForeignKey(columns=["author_id"], ref_table="people", ref_columns=["id"])],
        ),
        "people": Table(name="people", columns=[Column("id", "integer"), Column("name", "text")], primary_key=["id"]),
    }
    data = SqlData(
        rows={
            "papers": [
                {"id": "1", "title": "A Semantic Paper", "author_id": "10"},
                {"id": "2", "title": "Another Semantic Paper", "author_id": "11"},
            ],
            "people": [{"id": "10", "name": "Ada"}, {"id": "11", "name": "Ben"}],
        }
    )
    matches = [
        {"source_id": "source-data:papers.title", "target_uri": "http://ex/title", "target_id": "title", "target_local_name": "title"},
        {"source_id": "source-data:papers.author_id", "target_uri": "http://ex/authorId", "target_id": "authorId", "target_local_name": "author id"},
        {"source_id": "source-class:papers", "target_uri": "http://ex/Paper", "target_id": "Paper", "target_local_name": "Paper", "target_kind": "class"},
    ]
    fol = {
        "rules": {
            "class": [{"source_table": "papers", "target_class": "http://ex/Paper", "id_columns": ["id"], "match_ids": ["source-class:papers"]}],
            "data": [
                {"source_table": "papers", "source_column": "missing_title", "target_property": "http://ex/title", "match_ids": ["source-data:papers.title"]},
                {"source_table": "papers", "source_column": "author_id", "target_property": "http://ex/authorId", "match_ids": ["source-data:papers.author_id"]},
            ],
            "object": [],
        }
    }
    graph = Graph()
    for index in range(25):
        graph.add((URIRef(f"http://ex/paper/{index}"), URIRef("http://ex/title"), Literal(str(index))))
    graph.add((URIRef("http://ex/paper/1"), RDF.type, URIRef("http://ex/Paper")))
    graph_path = tmp_path / "import.ttl"
    graph.serialize(destination=str(graph_path), format="turtle")
    runtime_log = {
        "generated_triples": 26,
        "invalid_triple_count": 0,
        "rule_stats": {
            "class:0": {"reachable_rows": 2, "helper_calls": 2, "emitted_triples": 1, "failure_counts": {}},
            "data:0": {"reachable_rows": 2, "helper_calls": 2, "emitted_triples": 0, "failure_counts": {"empty_source_value": 2}},
            "data:1": {"reachable_rows": 2, "helper_calls": 0, "emitted_triples": 0, "failure_counts": {}},
        },
    }
    return matches, fol, tables, data, graph_path, runtime_log


def test_materialization_coverage_detects_zero_unknown_fk_and_overbroad(tmp_path: Path) -> None:
    matches, fol, tables, data, graph_path, runtime_log = _materialization_coverage_fixture(tmp_path)
    report = materialization_coverage_diagnostics(matches, fol, tables, data, graph_path, runtime_log, code_text="def materialize(context): pass")
    issues = report["issues_by_type"]
    assert issues["reachable_rule_zero_emission"] >= 1
    assert issues["generated_code_path_zero_helper_calls"] >= 1
    assert issues["unknown_source_column"] >= 1
    assert issues["fk_like_datatype_literal_without_justification"] >= 1
    assert issues["overbroad_emission"] >= 1


def test_materialization_coverage_detects_selected_target_zero_emission(tmp_path: Path) -> None:
    matches, fol, tables, data, graph_path, runtime_log = _materialization_coverage_fixture(tmp_path)
    matches.append({"source_id": "source-data:papers.title", "target_uri": "http://ex/missingTitle", "target_local_name": "missing title"})
    report = materialization_coverage_diagnostics(matches, fol, tables, data, graph_path, runtime_log)
    assert report["issues_by_type"]["selected_target_zero_emission"] >= 1
    assert report["issues_by_type"]["text_like_selected_evidence_missing_literals"] >= 1


def test_materialization_repair_prompt_is_budgeted_and_evaluation_blind(tmp_path: Path) -> None:
    matches, fol, tables, data, graph_path, runtime_log = _materialization_coverage_fixture(tmp_path)
    report = materialization_coverage_diagnostics(matches, fol, tables, data, graph_path, runtime_log)
    prompts = build_materialization_repair_prompts(report, fol, matches, tables, data, budget_chars=3000)
    assert prompts
    assert all(record["char_count"] <= 3000 for record in prompts)
    text = "\n".join(record["prompt"] for record in prompts)
    forbidden = ["LLM4VKG", "SPARQL", "SQL", "qpair", "gold mapping", "paper score", "expected triples", "false positives", "false negatives"]
    assert not any(token in text for token in forbidden)


def test_materialization_repairs_reject_invented_targets_and_columns() -> None:
    tables = {"papers": Table(name="papers", columns=[Column("id"), Column("title")], primary_key=["id"])}
    matches = [{"source_id": "source-data:papers.title", "target_uri": "http://ex/title"}]
    fol = {"rules": {"class": [], "data": [], "object": []}}
    repairs = [
        {
            "action": "add_data_rule",
            "rule": {"source_table": "papers", "source_column": "title", "target_property": "http://ex/invented", "match_ids": ["source-data:papers.title"]},
        },
        {
            "action": "add_data_rule",
            "rule": {"source_table": "papers", "source_column": "invented_column", "target_property": "http://ex/title", "match_ids": ["source-data:papers.title"]},
        },
    ]
    repaired, summary = apply_materialization_coverage_repairs(fol, repairs, matches, tables)
    assert repaired["rules"]["data"] == []
    assert summary["repairs_rejected"] == 2


def test_materialization_coverage_cli_defaults_and_stage_e_flags() -> None:
    parser = build_parser()
    defaults = parser.parse_args(["run-paper-compare"])
    assert defaults.materialization_coverage_validation is False
    assert defaults.materialization_coverage_repair is False
    assert defaults.materialization_repair_context == "batched"
    args = parser.parse_args(
        [
            "run-paper-compare",
            "--llm-provider",
            "google",
            "--embedding-provider",
            "google",
            "--materialization-coverage-validation",
            "--materialization-coverage-repair",
            "--materialization-repair-context",
            "batched",
            "--materialization-repair-candidates",
            "3",
            "--materialization-repair-budget-chars",
            "4000",
            "--internal-rerank-repair-candidates",
        ]
    )
    assert args.materialization_coverage_validation is True
    assert args.materialization_coverage_repair is True
    assert args.materialization_repair_candidates == 3
    assert args.materialization_repair_budget_chars == 4000
    assert args.internal_rerank_repair_candidates is True


def test_materialization_internal_rerank_uses_gold_blind_diagnostics() -> None:
    weak = {
        "selected_target_count": 3,
        "selected_targets_with_emission": 1,
        "zero_emission_selected_targets": 2,
        "generated_triples": 100,
        "invalid_triple_count": 0,
        "issues": [{"issue": "selected_target_zero_emission"}, {"issue": "overbroad_emission"}],
        "issues_by_type": {"selected_target_zero_emission": 1, "overbroad_emission": 1},
    }
    strong = {
        "selected_target_count": 3,
        "selected_targets_with_emission": 3,
        "zero_emission_selected_targets": 0,
        "generated_triples": 90,
        "invalid_triple_count": 0,
        "issues": [],
        "issues_by_type": {},
    }
    assert internal_materialization_score(strong) > internal_materialization_score(weak)


def test_object_evidence_summary_records_confidence_labels_and_weak_default() -> None:
    tables = {
        "persons": Table("persons", [Column("id")], ["id"]),
        "papers": Table(
            "papers",
            [Column("id"), Column("author_id")],
            ["id"],
            [ForeignKey(["author_id"], "persons", ["id"])],
        ),
    }
    matches = [{"source_id": "source-object:papers.author_id", "target_uri": "http://ex#hasAuthor"}]
    evidence = build_object_link_evidence(matches, tables)
    summary = object_evidence_summary(evidence)

    assert evidence["allow_weak_object_links"] is False
    assert summary["allow_weak_object_links"] is False
    assert summary["plans_by_confidence_label"]["direct_fk"] >= 1


def test_dev_sampling_preserves_fk_parent_with_reordered_reference_columns() -> None:
    province = Table("province", [Column("name"), Column("country")], ["name", "country"], [])
    city = Table(
        "city",
        [Column("name"), Column("country"), Column("province")],
        ["name", "country", "province"],
        [ForeignKey(["country", "province"], "province", ["country", "name"])],
    )
    tables = {"province": province, "city": city}
    data = SqlData(
        {
            "province": [{"name": "Albania", "country": "AL"}],
            "city": [{"name": "Tirana", "country": "AL", "province": "Albania"}],
        }
    )
    fraction = 0.5
    seed = next(
        str(i)
        for i in range(1000)
        if stable_keep("city", ("Tirana", "AL", "Albania"), fraction, str(i))
        and not stable_keep("province", ("Albania", "AL"), fraction, str(i))
    )
    sampled = select_dev_rows(tables, data, fraction=fraction, seed=seed)
    assert sampled["city"] == data.rows["city"]
    assert sampled["province"] == data.rows["province"]


def test_dev_sampling_no_pk_uses_full_row_key() -> None:
    sea = Table("sea", [Column("name")], ["name"], [])
    islandin = Table(
        "islandin",
        [Column("island"), Column("sea"), Column("lake"), Column("river")],
        [],
        [ForeignKey(["sea"], "sea", ["name"])],
    )
    tables = {"sea": sea, "islandin": islandin}
    data = SqlData(
        {
            "sea": [{"name": "Banda Sea"}],
            "islandin": [
                {"island": "Sulawesi", "sea": "Banda Sea", "lake": None, "river": None},
                {"island": "Sulawesi", "sea": "Java Sea", "lake": None, "river": None},
            ],
        }
    )
    fraction = 0.5
    seed = next(
        str(i)
        for i in range(1000)
        if stable_keep("islandin", ("Sulawesi", "Banda Sea", None, None), fraction, str(i))
        and not stable_keep("islandin", ("Sulawesi", "Java Sea", None, None), fraction, str(i))
        and not stable_keep("sea", ("Banda Sea",), fraction, str(i))
    )
    sampled = select_dev_rows(tables, data, fraction=fraction, seed=seed)
    assert sampled["islandin"] == [{"island": "Sulawesi", "sea": "Banda Sea", "lake": None, "river": None}]
    assert sampled["sea"] == [{"name": "Banda Sea"}]


def test_quoted_hyphen_identifiers_are_sampled_in_copy_blocks() -> None:
    tmp_path = work_path("quoted_identifiers")
    dump = tmp_path / "dump.sql"
    dump.write_text(
        "\n".join(
            [
                'CREATE TABLE "Paper" ("ID" integer NOT NULL);',
                'CREATE TABLE "Person" ("ID" integer NOT NULL);',
                'CREATE TABLE "co-writePaper" (',
                '  "Co-author" integer NOT NULL,',
                '  "Paper" integer NOT NULL',
                ");",
                'ALTER TABLE ONLY "Paper" ADD CONSTRAINT "PaperPK" PRIMARY KEY ("ID");',
                'ALTER TABLE ONLY "Person" ADD CONSTRAINT "PersonPK" PRIMARY KEY ("ID");',
                'ALTER TABLE ONLY "co-writePaper" ADD CONSTRAINT "co-writePaperPK" PRIMARY KEY ("Co-author", "Paper");',
                'ALTER TABLE ONLY "co-writePaper" ADD CONSTRAINT "FKco-writePaperPaper" FOREIGN KEY ("Paper") REFERENCES "Paper"("ID");',
                'ALTER TABLE ONLY "co-writePaper" ADD CONSTRAINT "co-writePaper_Co-author_fkey" FOREIGN KEY ("Co-author") REFERENCES "Person"("ID");',
                'COPY "Paper" ("ID") FROM stdin;',
                "947",
                "999",
                r"\.",
                'COPY "Person" ("ID") FROM stdin;',
                "1",
                r"\.",
                'COPY "co-writePaper" ("Co-author", "Paper") FROM stdin;',
                "1\t947",
                "1\t999",
                r"\.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    tables = parse_sql_dump(dump)
    data = parse_copy_data(dump)
    assert "co-writePaper" in tables
    assert tables["co-writePaper"].column_names() == ["Co-author", "Paper"]
    assert tables["co-writePaper"].foreign_keys[0].ref_table == "Paper"
    assert data.rows["co-writePaper"][0] == {"Co-author": "1", "Paper": "947"}

    out = tmp_path / "sampled.sql"
    write_sampled_dump(
        dump,
        out,
        {
            "Paper": [{"ID": "947"}],
            "Person": [{"ID": "1"}],
            "co-writePaper": [{"Co-author": "1", "Paper": "947"}],
        },
    )
    sampled_text = out.read_text(encoding="utf-8")
    assert 'COPY "co-writePaper" ("Co-author", "Paper") FROM stdin;' in sampled_text
    assert "1\t947" in sampled_text
    assert "1\t999" not in sampled_text


def test_sampled_dump_keeps_duplicate_table_copy_columns_separate() -> None:
    tmp_path = work_path("duplicate_table_copy")
    dump = tmp_path / "dump.sql"
    dump.write_text(
        "\n".join(
            [
                "SET search_path = a, pg_catalog;",
                'CREATE TABLE borders ("Entity" text NOT NULL, "LargeArea" text NOT NULL);',
                "SET search_path = b, pg_catalog;",
                "CREATE TABLE borders (country1 text NOT NULL, country2 text NOT NULL);",
                'COPY borders ("Entity", "LargeArea") FROM stdin;',
                "Country-A\tArea-A",
                r"\.",
                "COPY borders (country1, country2) FROM stdin;",
                "AA\tBB",
                r"\.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "sampled.sql"
    write_sampled_dump(
        dump,
        out,
        {
            "borders": [
                {"Entity": "Country-A", "LargeArea": "Area-A"},
                {"country1": "AA", "country2": "BB"},
            ]
        },
    )
    sampled_text = out.read_text(encoding="utf-8")
    assert 'COPY borders ("Entity", "LargeArea") FROM stdin;\nCountry-A\tArea-A\n\\.' in sampled_text
    assert "COPY borders (country1, country2) FROM stdin;\nAA\tBB\n\\." in sampled_text
    assert r"\N" not in sampled_text


def test_table_role_distinguishes_subtypes_from_join_tables() -> None:
    authors = Table(
        "authors",
        [Column("id")],
        ["id"],
        [ForeignKey(["id"], "conf_members", ["id"]), ForeignKey(["id"], "users", ["id"])],
    )
    reviewers = Table(
        "reviewers",
        [Column("id")],
        ["id"],
        [ForeignKey(["id"], "conf_members", ["id"]), ForeignKey(["id"], "users", ["id"])],
    )
    paper_author = Table(
        "paper_author",
        [Column("aid"), Column("pid")],
        ["aid", "pid"],
        [ForeignKey(["aid"], "authors", ["id"]), ForeignKey(["pid"], "papers", ["id"])],
    )
    conference_members = Table(
        "conference_members",
        [Column("conference"), Column("conference_member")],
        ["conference", "conference_member"],
        [ForeignKey(["conference"], "conferences", ["id"])],
    )

    assert table_role(authors) == "subtype_table"
    assert table_role(reviewers) == "subtype_table"
    assert table_role(paper_author) == "join_table"
    assert table_role(conference_members) == "join_table"


def test_safe_token_expansion_for_conference_and_program_committee() -> None:
    conference_words = words("conferences")
    assert "conference" in conference_words
    assert "conferenceerence" not in conference_words
    assert _class_compatible(
        "conferences",
        {
            "target_uri": "http://cmt#Conference",
            "target_local_name": "Conference",
            "target_id": "class:http://cmt#Conference",
        },
    )
    assert _class_compatible("program_committees", {"target_uri": "http://cmt#ProgramCommittee", "target_local_name": "ProgramCommittee"})
    assert _class_compatible("pc_members", {"target_uri": "http://cmt#ProgramCommitteeMember", "target_local_name": "ProgramCommitteeMember"})


def test_ontology_verbalization_and_retrieval() -> None:
    tmp_path = work_path("retrieval")
    scenario = write_fixture(tmp_path)
    tables = parse_sql_dump(scenario / "dump.sql")
    data = parse_copy_data(scenario / "dump.sql")
    target = parse_ontology_records(scenario / "ontology.ttl")
    source = source_schema_records(tables, data.rows)
    work = tmp_path / "work"
    write_records(work, target, source)
    embedded_target = embed_records([r.to_dict() for r in target], work / "embedding_cache.jsonl", offline=True)
    build_index(embedded_target, work)
    embedded_source = embed_records([r.to_dict() for r in source], work / "source_embedding_cache.jsonl", offline=True)
    rows = retrieve_candidates(embedded_source, work, k=2)
    assert rows
    assert all(len(row["candidates"]) <= 2 for row in rows)
    people_row = next(row for row in rows if row["source"]["id"] == "source-class:people")
    assert people_row["candidates"][0]["uri"] == "http://ex#Person"


def test_ontology_parser_skips_blank_node_targets() -> None:
    tmp_path = work_path("blank_nodes")
    ontology = tmp_path / "ontology.ttl"
    ontology.write_text(
        "\n".join(
            [
                "@prefix : <http://ex#> .",
                "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
                "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
                ":Person rdf:type owl:Class .",
                "[] rdf:type owl:Class .",
            ]
        ),
        encoding="utf-8",
    )
    records = parse_ontology_records(ontology)
    assert [record.uri for record in records if record.kind == "class"] == ["http://ex#Person"]
    data_uris = {record.uri for record in records if record.kind == "data_property"}
    assert "http://www.w3.org/2000/01/rdf-schema#label" in data_uris
    assert "http://www.w3.org/2000/01/rdf-schema#comment" in data_uris


def test_morph_config_generation() -> None:
    tmp_path = work_path("morph")
    scenario = write_fixture(tmp_path)
    tables = parse_sql_dump(scenario / "dump.sql")
    mapping = generate_source_r2rml(tables, "mini", tmp_path / "source_r2rml.ttl")
    config = write_morph_config(mapping, tmp_path / "source.ttl", "postgresql://localhost/postgres", "u", "p", tmp_path / "morph.ini")
    assert "rr:TriplesMap" in mapping.read_text(encoding="utf-8")
    assert "output_file=" in config.read_text(encoding="utf-8")


def test_matching_fol_sandbox_and_materialization() -> None:
    tmp_path = work_path("materialize")
    scenario = write_fixture(tmp_path)
    tables = parse_sql_dump(scenario / "dump.sql")
    data = parse_copy_data(scenario / "dump.sql")
    target = parse_ontology_records(scenario / "ontology.ttl")
    source = source_schema_records(tables, data.rows)
    target_by_uri = {r.uri: r for r in target}
    target_for_source = {
        "source-class:people": target_by_uri["http://ex#Person"],
        "source-class:papers": target_by_uri["http://ex#Paper"],
        "source-data:people.email": target_by_uri["http://ex#email"],
        "source-data:papers.title": target_by_uri["http://ex#title"],
        "source-object:papers.author_id": target_by_uri["http://ex#hasAuthor"],
    }
    candidates = [
        {"source": r.to_dict(), "candidates": [target_for_source[r.id].to_dict()]}
        for r in source
        if r.id in target_for_source
    ]
    matches = offline_match(candidates)
    fol = validate_fol(matches_to_fol([m.__dict__ for m in matches], tables), tables)
    code = (
        "def materialize(context):\n"
        "    for rule in all_rules('class'):\n"
        "        for row in rows(rule['source_table']):\n"
        "            emit_type(row, rule)\n"
    )
    validate_generated_code(code)
    try:
        validate_generated_code("import os\ndef materialize(context):\n    pass\n")
    except SandboxViolation:
        pass
    else:
        raise AssertionError("unsafe code accepted")
    try:
        validate_generated_code("def materialize(context):\n    while True:\n        pass\n")
    except SandboxViolation:
        pass
    else:
        raise AssertionError("unbounded loop accepted")
    graph = materialize_graph("mini", tables, data, fol)
    assert (None, RDF.type, URIRef("http://ex#Person")) in graph
    assert (URIRef("urn:coding-fgf:source:mini/papers/10"), URIRef("http://ex#hasAuthor"), URIRef("urn:coding-fgf:source:mini/people/1")) in graph


def test_runtime_codegen_prompt_version_and_contract() -> None:
    prompt = generate_codegen_prompt({"rules": {"class": [], "data": [], "object": []}})
    assert CODEGEN_PROMPT_VERSION == "fgf_codegen_v3_runtime_validated"
    assert "Prompt version: fgf_codegen_v3_runtime_validated" in prompt
    assert "Iterate every rule kind" in prompt
    assert "Do not use dotted attribute or method calls" in prompt
    assert "helper functions already apply row filters" in prompt
    assert "Do not invent target URIs" in prompt


def test_codegen_few_shot_prompt_enabled_only_and_safe() -> None:
    fol = {"rules": {"class": [], "data": [], "object": []}}

    disabled = generate_codegen_prompt(fol)
    enabled = generate_codegen_prompt(
        fol,
        prompt_version=CODEGEN_FEW_SHOT_PROMPT_VERSION,
        few_shot_examples=True,
    )

    assert "few_shot_examples" not in disabled
    assert CODEGEN_FEW_SHOT_PROMPT_VERSION == "fgf_codegen_v4_fewshot_runtime_validated"
    assert "Prompt version: fgf_codegen_v4_fewshot_runtime_validated" in enabled
    assert enabled.count('"example_id"') == 3
    assert "Emit triples only through the approved helper functions" in enabled
    assert "Do not use dotted attribute or method calls" in enabled
    assert "Do not invent target URIs" in enabled
    for forbidden in ("cmt_", "conference_", "sigkdd", "Q38", "LLM4VKG", "BootOX", "gold answers"):
        assert forbidden not in enabled


def test_fol_prompt_requires_llm_only_supplied_matches() -> None:
    tables = {"people": Table("people", [Column("id"), Column("email")], ["id"])}
    matches = [
        {
            "source_id": "source-class:people",
            "target_uri": "http://ex#Person",
            "confidence": 0.9,
        }
    ]
    prompt = generate_fol_prompt(matches, tables)
    assert "Generate FOL-style mapping rules" in prompt
    assert "Use only target URIs that appear in the supplied selected matches" in prompt
    assert "Do not invent target URIs" in prompt
    assert "http://ex#Person" in prompt


def test_runtime_materialization_diagnostics_record_zero_outputs() -> None:
    tables = {"people": Table("people", [Column("id"), Column("email")], ["id"])}
    data = SqlData(rows={"people": [{"id": "1", "email": ""}]})
    fol = {
        "rules": {
            "class": [{"source_table": "people", "target_class": "http://ex#Person", "id_columns": ["id"]}],
            "data": [{"source_table": "people", "source_column": "email", "target_property": "http://ex#email"}],
            "object": [],
        }
    }
    _, log = materialize_graph_with_log("mini", tables, data, fol, code=default_codegen())
    data_stats = [stat for stat in log["rule_stats"].values() if stat["kind"] == "data"][0]
    assert data_stats["helper_calls"] == 1
    assert data_stats["emitted_triples"] == 0
    assert data_stats["failure_counts"]["empty_source_value"] == 1
    assert data_stats["failure_samples"][0]["row"]["id"] == "1"


def test_materialize_to_file_requires_generated_code_by_default() -> None:
    tmp_path = work_path("materialize_requires_code")
    fol_path = tmp_path / "fol.json"
    fol_path.write_text(json.dumps({"rules": {"class": [], "data": [], "object": []}}), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Generated code is required"):
        materialize_to_file(
            "mini",
            {},
            SqlData(rows={}),
            fol_path,
            tmp_path / "out.ttl",
            code_path=tmp_path / "missing_generated_fgf.py",
        )


def test_runtime_invalid_triple_detection() -> None:
    invalid = _invalid_triples(
        [
            ("not-a-uri", "http://ex#p", "literal:value"),
            ("urn:s", "not-a-uri", "urn:o"),
            ("urn:s", "http://ex#p", "not-a-uri"),
        ]
    )
    assert {row["reason"] for row in invalid} == {"invalid_subject_uri", "invalid_predicate_uri", "invalid_object"}


def test_codegen_runtime_selection_prefers_complete_executing_candidate(monkeypatch) -> None:
    tmp_path = work_path("codegen_runtime_selection")
    scenario = write_fixture(tmp_path)
    tables = parse_sql_dump(scenario / "dump.sql")
    data = parse_copy_data(scenario / "dump.sql")
    fol = {
        "rules": {
            "class": [
                {"source_table": "people", "target_class": "http://ex#Person", "id_columns": ["id"]},
                {"source_table": "papers", "target_class": "http://ex#Paper", "id_columns": ["id"]},
            ],
            "data": [
                {"source_table": "people", "source_column": "email", "target_property": "http://ex#email"},
                {"source_table": "papers", "source_column": "title", "target_property": "http://ex#title"},
            ],
            "object": [
                {
                    "source_table": "papers",
                    "source_columns": ["author_id"],
                    "target_table": "people",
                    "target_columns": ["id"],
                    "target_property": "http://ex#hasAuthor",
                }
            ],
        }
    }
    class_only = (
        "def materialize(context):\n"
        "    for rule in all_rules('class'):\n"
        "        for row in rows(rule['source_table']):\n"
        "            emit_type(row, rule)\n"
    )
    data_only = (
        "def materialize(context):\n"
        "    for rule in all_rules('data'):\n"
        "        for row in rows(rule['source_table']):\n"
        "            emit_data(row, rule)\n"
    )
    generated = [class_only, data_only, default_codegen()]

    def fake_codegen(*args, **kwargs):
        return generated.pop(0) if generated else default_codegen()

    monkeypatch.setattr("coding_fgf.cli.llm_codegen", fake_codegen)
    code, audit = _codegen_self_consistent(
        tmp_path,
        fol,
        offline=False,
        scenario="mini",
        tables=tables,
        data=data,
        self_consistency=1,
        prompt_version=CODEGEN_PROMPT_VERSION,
    )
    assert code == default_codegen()
    assert audit["selected_index"] == 3
    assert (tmp_path / "generated_fgf.candidate_1.py").exists()
    assert (tmp_path / "generated_fgf.candidate_2.py").exists()
    assert (tmp_path / "generated_fgf.candidate_3.py").exists()
    assert (tmp_path / "codegen_candidate_scores.json").exists()


def test_materialization_normalizes_copy_escaped_newline_literals() -> None:
    tables = {"reviews": Table("reviews", [Column("id"), Column("comment")], ["id"])}
    data = SqlData(rows={"reviews": [{"id": "1", "comment": "line one\\nline two"}]})
    fol = {
        "rules": {
            "class": [{"source_table": "reviews", "target_class": "http://cmt#Review", "id_columns": ["id"]}],
            "data": [{"source_table": "reviews", "source_column": "comment", "target_property": "http://www.w3.org/2000/01/rdf-schema#comment"}],
            "object": [],
        }
    }
    graph = materialize_graph("cmt", tables, data, fol)
    assert normalize_literal_value("line one\\nline two") == "line one\nline two"
    assert (
        URIRef("urn:coding-fgf:source:cmt/reviews/1"),
        URIRef("http://www.w3.org/2000/01/rdf-schema#comment"),
        Literal("line one\nline two"),
    ) in graph


def test_subtype_identity_and_paper_author_materialization() -> None:
    tables = {
        "persons": Table("persons", [Column("id")], ["id"]),
        "authors": Table("authors", [Column("id")], ["id"], [ForeignKey(["id"], "persons", ["id"])]),
        "papers": Table("papers", [Column("id"), Column("author")], ["id"], [ForeignKey(["author"], "authors", ["id"])]),
        "paper_full_versions": Table("paper_full_versions", [Column("id")], ["id"], [ForeignKey(["id"], "papers", ["id"])]),
    }
    data = SqlData(
        rows={
            "persons": [{"id": "1"}],
            "authors": [{"id": "1"}],
            "papers": [{"id": "10", "author": "1"}],
            "paper_full_versions": [{"id": "10"}],
        }
    )
    fol = {
        "rules": {
            "class": [
                {"source_table": "persons", "target_class": "http://cmt#Person", "id_columns": ["id"]},
                {"source_table": "authors", "target_class": "http://cmt#Author", "id_columns": ["id"]},
                {"source_table": "papers", "target_class": "http://cmt#Paper", "id_columns": ["id"]},
                {"source_table": "paper_full_versions", "target_class": "http://cmt#PaperFullVersion", "id_columns": ["id"]},
            ],
            "data": [],
            "object": [
                {
                    "source_table": "papers",
                    "source_columns": ["author"],
                    "target_property": "http://cmt#hasAuthor",
                    "target_table": "authors",
                    "target_columns": ["id"],
                }
            ],
        }
    }
    graph = materialize_graph("cmt", tables, data, fol)
    person_uri = URIRef("urn:coding-fgf:source:cmt/persons/1")
    paper_uri = URIRef("urn:coding-fgf:source:cmt/papers/10")
    assert (person_uri, RDF.type, URIRef("http://cmt#Person")) in graph
    assert (person_uri, RDF.type, URIRef("http://cmt#Author")) in graph
    assert (paper_uri, RDF.type, URIRef("http://cmt#Paper")) in graph
    assert (paper_uri, RDF.type, URIRef("http://cmt#PaperFullVersion")) in graph
    assert (paper_uri, URIRef("http://cmt#hasAuthor"), person_uri) in graph


def test_join_table_class_blocked_and_missing_endpoint_inferred() -> None:
    tables = {
        "program_committees": Table("program_committees", [Column("id")], ["id"]),
        "pc_members": Table("pc_members", [Column("id")], ["id"]),
        "program_committee_members": Table(
            "program_committee_members",
            [Column("program_committee"), Column("program_committee_member")],
            ["program_committee", "program_committee_member"],
            [ForeignKey(["program_committee"], "program_committees", ["id"])],
        ),
    }
    matches = [
        {
            "source_id": "source-class:program_committee_members",
            "target_uri": "http://cmt#ProgramCommitteeMember",
            "target_local_name": "ProgramCommitteeMember",
        },
        {
            "source_id": "source-class:program_committees",
            "target_uri": "http://cmt#ProgramCommittee",
            "target_local_name": "ProgramCommittee",
        },
        {
            "source_id": "source-class:pc_members",
            "target_uri": "http://cmt#ProgramCommitteeMember",
            "target_local_name": "ProgramCommitteeMember",
        },
        {
            "source_id": "source-object:program_committee_members.program_committee",
            "target_uri": "http://cmt#hasProgramCommitteeMember",
            "target_local_name": "hasProgramCommitteeMember",
            "target_domain": ["http://cmt#ProgramCommittee"],
            "target_range": ["http://cmt#ProgramCommitteeMember"],
        },
    ]
    fol = validate_fol(matches_to_fol(matches, tables), tables)
    assert all(rule["source_table"] != "program_committee_members" for rule in fol["rules"]["class"])
    assert fol["rules"]["object"][0]["subject_table"] == "program_committees"
    assert fol["rules"]["object"][0]["object_table"] == "pc_members"

    data = SqlData(
        rows={
            "program_committees": [{"id": "1"}],
            "pc_members": [],
            "program_committee_members": [{"program_committee": "1", "program_committee_member": "99"}],
        }
    )
    graph = materialize_graph("cmt", tables, data, fol)
    assert (None, URIRef("http://cmt#hasProgramCommitteeMember"), None) not in graph

    data.rows["pc_members"] = [{"id": "99"}]
    graph = materialize_graph("cmt", tables, data, fol)
    assert (
        URIRef("urn:coding-fgf:source:cmt/program_committees/1"),
        URIRef("http://cmt#hasProgramCommitteeMember"),
        URIRef("urn:coding-fgf:source:cmt/pc_members/99"),
    ) in graph


def test_conference_class_data_and_program_committee_label_rules() -> None:
    tables = {
        "conferences": Table("conferences", [Column("id"), Column("name"), Column("date"), Column("site_url")], ["id"]),
        "program_committees": Table("program_committees", [Column("id"), Column("label")], ["id"]),
        "reviews": Table("reviews", [Column("id"), Column("comment")], ["id"]),
    }
    matches = [
        {"source_id": "source-class:conferences", "target_uri": "http://cmt#Conference", "target_local_name": "Conference"},
        {"source_id": "source-data:conferences.name", "target_uri": "http://cmt#name", "target_local_name": "name", "target_domain": ["http://cmt#Conference"]},
        {"source_id": "source-data:conferences.date", "target_uri": "http://cmt#date", "target_local_name": "date", "target_domain": ["http://cmt#Conference"]},
        {"source_id": "source-data:conferences.site_url", "target_uri": "http://cmt#siteURL", "target_local_name": "siteURL", "target_domain": ["http://cmt#Conference"]},
        {
            "source_id": "source-class:program_committees",
            "target_uri": "http://cmt#ProgramCommittee",
            "target_local_name": "ProgramCommittee",
        },
        {
            "source_id": "source-data:program_committees.label",
            "target_uri": "http://www.w3.org/2000/01/rdf-schema#label",
            "target_local_name": "label",
        },
        {"source_id": "source-class:reviews", "target_uri": "http://cmt#Review", "target_local_name": "Review"},
        {
            "source_id": "source-data:reviews.comment",
            "target_uri": "http://www.w3.org/2000/01/rdf-schema#comment",
            "target_local_name": "comment",
        },
    ]
    fol = validate_fol(matches_to_fol(matches, tables), tables)
    assert {rule["source_table"] for rule in fol["rules"]["class"]} == {"conferences", "program_committees", "reviews"}
    properties = {(rule["source_table"], rule["source_column"]): rule["target_property"] for rule in fol["rules"]["data"]}
    assert properties[("conferences", "name")] == "http://cmt#name"
    assert properties[("conferences", "date")] == "http://cmt#date"
    assert properties[("conferences", "site_url")] == "http://cmt#siteURL"
    assert properties[("program_committees", "label")] == "http://www.w3.org/2000/01/rdf-schema#label"
    assert properties[("reviews", "comment")] == "http://www.w3.org/2000/01/rdf-schema#comment"


def test_reviewer_write_review_inverse_materialization() -> None:
    tables = {
        "persons": Table("persons", [Column("id")], ["id"]),
        "conf_members": Table("conf_members", [Column("id")], ["id"], [ForeignKey(["id"], "persons", ["id"])]),
        "reviewers": Table("reviewers", [Column("id")], ["id"], [ForeignKey(["id"], "conf_members", ["id"])]),
        "reviews": Table("reviews", [Column("id"), Column("written")], ["id"], [ForeignKey(["written"], "reviewers", ["id"])]),
    }
    matches = [
        {"source_id": "source-class:persons", "target_uri": "http://cmt#Person", "target_local_name": "Person"},
        {"source_id": "source-class:conf_members", "target_uri": "http://cmt#ConferenceMember", "target_local_name": "ConferenceMember"},
        {"source_id": "source-class:reviewers", "target_uri": "http://cmt#Reviewer", "target_local_name": "Reviewer"},
        {"source_id": "source-class:reviews", "target_uri": "http://cmt#Review", "target_local_name": "Review"},
        {
            "source_id": "source-object:reviews.written",
            "target_uri": "http://cmt#writeReview",
            "target_local_name": "writeReview",
            "target_domain": ["http://cmt#Reviewer"],
            "target_range": ["http://cmt#Review"],
        },
    ]
    fol = validate_fol(matches_to_fol(matches, tables), tables)
    assert fol["rules"]["object"][0]["subject_table"] == "reviewers"
    data = SqlData(
        rows={
            "persons": [{"id": "1"}],
            "conf_members": [{"id": "1"}],
            "reviewers": [{"id": "1"}],
            "reviews": [{"id": "7", "written": "1"}],
        }
    )
    graph = materialize_graph("cmt", tables, data, fol)
    assert (
        URIRef("urn:coding-fgf:source:cmt/persons/1"),
        URIRef("http://cmt#writeReview"),
        URIRef("urn:coding-fgf:source:cmt/reviews/7"),
    ) in graph


def test_fk_owned_attribute_table_emits_data_on_referenced_subject() -> None:
    tables = {
        "persons": Table("persons", [Column("id")], ["id"]),
        "emails": Table(
            "emails",
            [Column("person"), Column("value")],
            ["person", "value"],
            [ForeignKey(["person"], "persons", ["id"])],
        ),
    }
    matches = [
        {"source_id": "source-class:persons", "target_uri": "http://conference#Person", "target_local_name": "Person"},
        {
            "source_id": "source-data:emails.value",
            "target_uri": "http://conference#has_an_email",
            "target_local_name": "has_an_email",
            "target_domain": ["http://conference#Person"],
        },
    ]
    fol = validate_fol(matches_to_fol(matches, tables), tables)
    data_rules = fol["rules"]["data"]
    assert data_rules[0]["subject_table"] == "persons"
    assert data_rules[0]["subject_columns"] == ["person"]

    data = SqlData(rows={"persons": [{"id": "1"}], "emails": [{"person": "1", "value": "ada@example.org"}]})
    graph = materialize_graph("conference", tables, data, fol)
    assert (
        URIRef("urn:coding-fgf:source:conference/persons/1"),
        URIRef("http://conference#has_an_email"),
        Literal("ada@example.org"),
    ) in graph


def test_structured_discriminator_columns_emit_filtered_types_and_links() -> None:
    tables = {
        "Document": Table("Document", [Column("ID")], ["ID"]),
        "Person": Table(
            "Person",
            [Column("ID"), Column("is_Author"), Column("is_Co-author"), Column("is_Reviewer")],
            ["ID"],
        ),
        "Paper": Table(
            "Paper",
            [Column("ID"), Column("paperID"), Column("title"), Column("hasAuthor"), Column("TYPE")],
            ["ID"],
            [ForeignKey(["ID"], "Document", ["ID"]), ForeignKey(["hasAuthor"], "Person", ["ID"])],
        ),
        "Review": Table(
            "Review",
            [Column("ID"), Column("writtenBy"), Column("comment")],
            ["ID"],
            [ForeignKey(["ID"], "Document", ["ID"]), ForeignKey(["writtenBy"], "Person", ["ID"])],
        ),
        "ProgramCommittee": Table("ProgramCommittee", [Column("ID"), Column("label")], ["ID"]),
    }
    matches = [
        {"source_id": "source-class:Person", "target_uri": "http://cmt#Person", "target_local_name": "Person"},
        {"source_id": "source-class:Paper", "target_uri": "http://cmt#Paper", "target_local_name": "Paper"},
        {"source_id": "source-class:Review", "target_uri": "http://cmt#Review", "target_local_name": "Review"},
        {
            "source_id": "source-class:ProgramCommittee",
            "target_uri": "http://cmt#ProgramCommittee",
            "target_local_name": "ProgramCommittee",
        },
        {
            "source_id": "source-data:Paper.paperID",
            "target_uri": "http://cmt#paperID",
            "target_local_name": "paperID",
            "target_domain": ["http://cmt#Paper"],
        },
        {
            "source_id": "source-data:Review.comment",
            "target_uri": "http://www.w3.org/2000/01/rdf-schema#comment",
            "target_local_name": "comment",
        },
        {
            "source_id": "source-object:Paper.hasAuthor",
            "target_uri": "http://cmt#hasAuthor",
            "target_local_name": "hasAuthor",
            "target_domain": ["http://cmt#Paper"],
            "target_range": ["http://cmt#Author"],
        },
        {
            "source_id": "source-object:Review.writtenBy",
            "target_uri": "http://cmt#writeReview",
            "target_local_name": "writeReview",
            "target_domain": ["http://cmt#Reviewer"],
            "target_range": ["http://cmt#Review"],
        },
    ]
    fol = validate_fol(matches_to_fol(matches, tables), tables)
    class_targets = {rule["target_class"] for rule in fol["rules"]["class"]}
    assert {"http://cmt#Author", "http://cmt#Co-author", "http://cmt#Reviewer"} <= class_targets
    assert {"http://cmt#PaperFullVersion", "http://cmt#PaperAbstract", "http://cmt#Program_committee"} <= class_targets

    data = SqlData(
        rows={
            "Document": [{"ID": "10"}, {"ID": "11"}, {"ID": "20"}],
            "Person": [{"ID": "1", "is_Author": "t", "is_Co-author": "f", "is_Reviewer": "t"}],
            "Paper": [
                {"ID": "10", "paperID": "P1", "title": "full", "hasAuthor": "1", "TYPE": "1"},
                {"ID": "11", "paperID": "A1", "title": "abstract", "hasAuthor": "1", "TYPE": "2"},
            ],
            "Review": [{"ID": "20", "writtenBy": "1", "comment": "solid"}],
            "ProgramCommittee": [{"ID": "7", "label": "PC"}],
        }
    )
    graph = materialize_graph("structured", tables, data, fol)
    person_uri = URIRef("urn:coding-fgf:source:structured/Person/1")
    paper_uri = URIRef("urn:coding-fgf:source:structured/Document/10")
    abstract_uri = URIRef("urn:coding-fgf:source:structured/Document/11")
    review_uri = URIRef("urn:coding-fgf:source:structured/Document/20")
    pc_uri = URIRef("urn:coding-fgf:source:structured/ProgramCommittee/7")
    assert (person_uri, RDF.type, URIRef("http://cmt#Author")) in graph
    assert (person_uri, RDF.type, URIRef("http://cmt#Reviewer")) in graph
    assert (paper_uri, RDF.type, URIRef("http://cmt#PaperFullVersion")) in graph
    assert (abstract_uri, RDF.type, URIRef("http://cmt#PaperAbstract")) in graph
    assert (pc_uri, RDF.type, URIRef("http://cmt#Program_committee")) in graph
    assert (paper_uri, URIRef("http://cmt#hasAuthor"), person_uri) in graph
    assert (person_uri, URIRef("http://cmt#writeReview"), review_uri) in graph
    assert (review_uri, URIRef("http://www.w3.org/2000/01/rdf-schema#comment"), Literal("solid")) in graph


def test_document_type_discriminators_and_review_inverse_links() -> None:
    tables = {
        "Document": Table("Document", [Column("ID"), Column("hasTitle"), Column("TYPE")], ["ID"]),
        "Person": Table(
            "Person",
            [Column("ID"), Column("Name"), Column("is_Author_of_paper_student"), Column("TYPE")],
            ["ID"],
        ),
        "Review": Table(
            "Review",
            [Column("ID"), Column("hasReview_Inv")],
            ["ID"],
            [ForeignKey(["ID"], "Document", ["ID"]), ForeignKey(["hasReview_Inv"], "Document", ["ID"])],
        ),
    }
    matches = [
        {"source_id": "source-class:Document", "target_uri": "http://sigkdd#Document", "target_local_name": "Document"},
        {"source_id": "source-class:Person", "target_uri": "http://sigkdd#Person", "target_local_name": "Person"},
        {"source_id": "source-class:Review", "target_uri": "http://sigkdd#Review", "target_local_name": "Review"},
        {
            "source_id": "source-data:Document.hasTitle",
            "target_uri": "http://sigkdd#hasTitle",
            "target_local_name": "hasTitle",
        },
        {
            "source_id": "source-data:Person.Name",
            "target_uri": "http://sigkdd#Name",
            "target_local_name": "Name",
            "target_domain": ["http://sigkdd#Person"],
        },
        {
            "source_id": "source-data:Person.TYPE",
            "target_uri": "http://sigkdd#Name",
            "target_local_name": "Name",
            "target_domain": ["http://sigkdd#Person"],
        },
        {
            "source_id": "source-data:Person.is_Author_of_paper_student",
            "target_uri": "http://sigkdd#Name",
            "target_local_name": "Name",
            "target_domain": ["http://sigkdd#Person"],
        },
        {
            "source_id": "source-object:Review.hasReview_Inv",
            "target_uri": "http://sigkdd#hasReview",
            "target_local_name": "hasReview",
            "target_domain": ["http://sigkdd#Paper"],
            "target_range": ["http://sigkdd#Review"],
        },
    ]
    fol = validate_fol(matches_to_fol(matches, tables), tables)
    class_targets = {rule["target_class"] for rule in fol["rules"]["class"]}
    assert {"http://sigkdd#Paper", "http://sigkdd#Abstract", "http://sigkdd#Author"} <= class_targets
    assert not any(rule["source_column"] in {"TYPE", "is_Author_of_paper_student"} for rule in fol["rules"]["data"])

    data = SqlData(
        rows={
            "Document": [
                {"ID": "10", "hasTitle": "paper title", "TYPE": "1"},
                {"ID": "11", "hasTitle": "abstract title", "TYPE": "2"},
                {"ID": "20", "hasTitle": "review title", "TYPE": "3"},
            ],
            "Person": [{"ID": "1", "Name": "Ada", "is_Author_of_paper_student": "t", "TYPE": "7"}],
            "Review": [{"ID": "20", "hasReview_Inv": "10"}],
        }
    )
    graph = materialize_graph("sigkdd", tables, data, fol)
    paper_uri = URIRef("urn:coding-fgf:source:sigkdd/Document/10")
    abstract_uri = URIRef("urn:coding-fgf:source:sigkdd/Document/11")
    review_uri = URIRef("urn:coding-fgf:source:sigkdd/Document/20")
    person_uri = URIRef("urn:coding-fgf:source:sigkdd/Person/1")
    assert (paper_uri, RDF.type, URIRef("http://sigkdd#Paper")) in graph
    assert (abstract_uri, RDF.type, URIRef("http://sigkdd#Abstract")) in graph
    assert (person_uri, RDF.type, URIRef("http://sigkdd#Author")) in graph
    assert (paper_uri, URIRef("http://sigkdd#hasReview"), review_uri) in graph
    assert (paper_uri, URIRef("http://sigkdd#hasTitle"), Literal("paper title")) in graph


def test_mixed_lowercase_discriminators_fee_aliases_and_submit_links() -> None:
    tables = {
        "documents": Table("documents", [Column("id"), Column("title"), Column("type", "integer")], ["id"]),
        "persons": Table(
            "persons",
            [
                Column("id"),
                Column("name"),
                Column("author_paper_student", "boolean"),
                Column("program_committee_member", "boolean"),
                Column("listener", "boolean"),
                Column("type", "integer"),
            ],
            ["id"],
        ),
        "document_person": Table(
            "document_person",
            [Column("pid"), Column("did")],
            ["pid", "did"],
            [ForeignKey(["pid"], "persons", ["id"]), ForeignKey(["did"], "documents", ["id"])],
        ),
        "registration_fees": Table(
            "registration_fees",
            [Column("id"), Column("price"), Column("type", "integer")],
            ["id"],
        ),
        "reviews": Table(
            "reviews",
            [Column("id"), Column("ref")],
            ["id"],
            [ForeignKey(["id"], "documents", ["id"]), ForeignKey(["ref"], "documents", ["id"])],
        ),
    }
    matches = [
        {"source_id": "source-class:documents", "target_uri": "http://sigkdd#Document", "target_local_name": "Document"},
        {"source_id": "source-class:persons", "target_uri": "http://sigkdd#Person", "target_local_name": "Person"},
        {"source_id": "source-class:reviews", "target_uri": "http://sigkdd#Review", "target_local_name": "Review"},
        {
            "source_id": "source-class:registration_fees",
            "target_uri": "http://sigkdd#Registration_Student",
            "target_local_name": "Registration_Student",
        },
        {
            "source_id": "source-data:documents.title",
            "target_uri": "http://sigkdd#hasTitle",
            "target_local_name": "hasTitle",
        },
        {
            "source_id": "source-data:persons.name",
            "target_uri": "http://sigkdd#Name",
            "target_local_name": "Name",
            "target_domain": ["http://sigkdd#Person"],
        },
        {
            "source_id": "source-data:persons.author_paper_student",
            "target_uri": "http://sigkdd#Name",
            "target_local_name": "Name",
            "target_domain": ["http://sigkdd#Person"],
        },
        {
            "source_id": "source-data:registration_fees.price",
            "target_uri": "http://sigkdd#Price",
            "target_local_name": "Price",
            "target_domain": ["http://sigkdd#Fee"],
        },
        {
            "source_id": "source-object:document_person.pid",
            "target_uri": "http://sigkdd#submit",
            "target_local_name": "submit",
            "target_domain": ["http://sigkdd#Author"],
            "target_range": ["http://sigkdd#Paper"],
        },
        {
            "source_id": "source-object:reviews.ref",
            "target_uri": "http://sigkdd#hasReview",
            "target_local_name": "hasReview",
            "target_domain": ["http://sigkdd#Paper"],
            "target_range": ["http://sigkdd#Review"],
        },
    ]
    fol = validate_fol(matches_to_fol(matches, tables), tables)
    class_targets = {rule["target_class"] for rule in fol["rules"]["class"]}
    assert {
        "http://sigkdd#Paper",
        "http://sigkdd#Abstract",
        "http://sigkdd#Author",
        "http://sigkdd#Author_of_paper_student",
        "http://sigkdd#Student",
        "http://sigkdd#Program_Committee_member",
        "http://sigkdd#Listener",
        "http://sigkdd#Registration_fee",
        "http://sigkdd#Fee",
    } <= class_targets
    assert not any(
        rule["source_column"] in {"author_paper_student", "program_committee_member", "listener", "type"}
        for rule in fol["rules"]["data"]
    )
    assert any(rule["source_table"] == "registration_fees" and rule["target_property"] == "http://sigkdd#Price" for rule in fol["rules"]["data"])

    data = SqlData(
        rows={
            "documents": [
                {"id": "10", "title": "paper title", "type": "1"},
                {"id": "11", "title": "abstract title", "type": "2"},
                {"id": "20", "title": "review title", "type": "3"},
            ],
            "persons": [
                {
                    "id": "1",
                    "name": "Ada",
                    "author_paper_student": "t",
                    "program_committee_member": "f",
                    "listener": "t",
                    "type": "0",
                }
            ],
            "document_person": [{"pid": "1", "did": "10"}],
            "registration_fees": [{"id": "30", "price": "757", "type": "0"}],
            "reviews": [{"id": "20", "ref": "10"}],
        }
    )
    graph = materialize_graph("sigkdd_mixed", tables, data, fol)
    paper_uri = URIRef("urn:coding-fgf:source:sigkdd_mixed/documents/10")
    review_uri = URIRef("urn:coding-fgf:source:sigkdd_mixed/documents/20")
    person_uri = URIRef("urn:coding-fgf:source:sigkdd_mixed/persons/1")
    fee_uri = URIRef("urn:coding-fgf:source:sigkdd_mixed/registration_fees/30")
    assert (paper_uri, RDF.type, URIRef("http://sigkdd#Paper")) in graph
    assert (person_uri, RDF.type, URIRef("http://sigkdd#Author")) in graph
    assert (person_uri, RDF.type, URIRef("http://sigkdd#Student")) in graph
    assert (person_uri, URIRef("http://sigkdd#submit"), paper_uri) in graph
    assert (paper_uri, URIRef("http://sigkdd#hasReview"), review_uri) in graph
    assert (fee_uri, RDF.type, URIRef("http://sigkdd#Registration_fee")) in graph
    assert (fee_uri, URIRef("http://sigkdd#Price"), Literal("757")) in graph


def test_evaluation_metric_and_graph_eval() -> None:
    tmp_path = work_path("evaluate")
    precision, recall, f1 = calculate_precision_recall_f1(["a", "b"], ["a", "c"])
    assert precision == 0.5
    assert recall == 0.5
    assert f1 == 0.5
    assert calculate_precision_recall_f1([], []) == (1.0, 1.0, 1.0)

    scenario = write_fixture(tmp_path)
    (scenario / "queries" / "Q99.qpair").write_text(
        "name=Disabled Missing Table\n"
        'sql=SELECT COUNT(*) FROM "MissingTable"\n'
        "sparql=PREFIX : <http://ex#> SELECT ?x WHERE { ?x :missing ?y }\n"
        "categories=class\n"
        "disabled=true\n",
        encoding="utf-8",
    )
    graph = Graph()
    graph.add((URIRef("urn:p1"), URIRef("http://ex#email"), Literal("ada@example.org")))
    graph_path = tmp_path / "graph.ttl"
    graph.serialize(destination=str(graph_path), format="turtle")

    sql_calls = 0

    def sql_exec(sql: str) -> list[object]:
        nonlocal sql_calls
        sql_calls += 1
        assert "MissingTable" not in sql
        return ["ada@example.org"]

    result = evaluate_graph(graph_path, scenario / "queries", sql_exec, tmp_path / "eval")
    assert result["f1"] == 1.0
    assert result["count"] == 1
    assert sql_calls == 1
    assert json.loads((tmp_path / "eval" / "metrics_details.json").read_text(encoding="utf-8"))[0]["f1"] == 1.0


def test_parse_qpair_accepts_indented_spaced_keys() -> None:
    tmp_path = work_path("qpair_spacing")
    qpair = tmp_path / "Q.qpair"
    qpair.write_text(
        "     name = 001-simple-country-count\n"
        "     sql =   Select COUNT(*) AS cnt From mondial_rdf2sql_standard.\"Country\"   \n\n"
        "     sparql =     prefix : <http://example#> SELECT (count(?C) AS ?cnt) WHERE { ?C a :Country } \n"
        "     categories = class,1-1\n",
        encoding="utf-8",
    )
    parsed = parse_qpair(qpair)
    assert parsed["name"] == "001-simple-country-count"
    assert parsed["sql"] == 'Select COUNT(*) AS cnt From mondial_rdf2sql_standard."Country"'
    assert parsed["sparql"].startswith("prefix :")
    assert parsed["categories"] == ["class", "1-1"]


def test_parse_qpair_stops_sparql_before_entity_metadata() -> None:
    tmp_path = work_path("qpair_entity_metadata")
    qpair = tmp_path / "Q.qpair"
    qpair.write_text(
        "name=Q\n"
        "orderNum=1\n"
        "sql=SELECT 1\n"
        "sparql=SELECT (COUNT(*) AS ?count) { ?x a <http://ex#T> }\n"
        "entityIdCols=0\n"
        "entityIdVars=x\n"
        "categories=class\n",
        encoding="utf-8",
    )

    parsed = parse_qpair(qpair)

    assert parsed["sql"] == "SELECT 1"
    assert parsed["sparql"] == "SELECT (COUNT(*) AS ?count) { ?x a <http://ex#T> }"


def _pattern_fixture() -> tuple[dict[str, Table], SqlData, list[dict[str, object]], list[dict[str, object]]]:
    tables = {
        "papers": Table("papers", [Column("id"), Column("title"), Column("abstract_text"), Column("isbn")], ["id"]),
        "persons": Table("persons", [Column("id"), Column("name"), Column("email")], ["id"]),
        "reviews": Table(
            "reviews",
            [Column("id"), Column("paper_id"), Column("review_text")],
            ["id"],
            [ForeignKey(["paper_id"], "papers", ["id"])],
        ),
        "paper_author": Table(
            "paper_author",
            [Column("paper_id"), Column("person_id")],
            ["paper_id", "person_id"],
            [ForeignKey(["paper_id"], "papers", ["id"]), ForeignKey(["person_id"], "persons", ["id"])],
        ),
        "paper_award": Table(
            "paper_award",
            [Column("paper_id"), Column("award_id"), Column("rank")],
            ["paper_id", "award_id"],
            [ForeignKey(["paper_id"], "papers", ["id"]), ForeignKey(["award_id"], "papers", ["id"])],
        ),
        "accepted_papers": Table(
            "accepted_papers",
            [Column("id"), Column("decision")],
            ["id"],
            [ForeignKey(["id"], "papers", ["id"])],
        ),
    }
    data = SqlData(
        rows={
            "papers": [
                {"id": "p1", "title": "A Pattern Paper", "abstract_text": "Long abstract text", "isbn": "978-1-234-56789-0"},
                {"id": "p2", "title": "Another Paper", "abstract_text": "More text", "isbn": "978-1-234-56789-1"},
            ],
            "persons": [{"id": "a1", "name": "Ada", "email": "ada@example.org"}],
            "reviews": [{"id": "r1", "paper_id": "p1", "review_text": "Strong review"}],
            "paper_author": [{"paper_id": "p1", "person_id": "a1"}],
            "paper_award": [{"paper_id": "p1", "award_id": "p2", "rank": "1"}],
            "accepted_papers": [{"id": "p1", "decision": "accept"}],
        }
    )
    matches: list[dict[str, object]] = [
        {"source_id": "source-class:papers", "target_uri": "http://ex#Paper", "target_kind": "class", "target_local_name": "Paper"},
        {"source_id": "source-class:persons", "target_uri": "http://ex#Person", "target_kind": "class", "target_local_name": "Person"},
        {"source_id": "source-class:reviews", "target_uri": "http://ex#Review", "target_kind": "class", "target_local_name": "Review"},
        {"source_id": "source-class:accepted_papers", "target_uri": "http://ex#AcceptedPaper", "target_kind": "class", "target_local_name": "AcceptedPaper"},
        {"source_id": "source-data:papers.title", "target_uri": "http://ex#title", "target_kind": "datatype_property", "target_local_name": "title", "target_domain": ["http://ex#Paper"]},
        {"source_id": "source-data:papers.abstract_text", "target_uri": "http://ex#abstract", "target_kind": "datatype_property", "target_local_name": "abstract", "target_domain": ["http://ex#Paper"]},
        {"source_id": "source-data:papers.id", "target_uri": "http://ex#title", "target_kind": "datatype_property", "target_local_name": "title", "target_domain": ["http://ex#Paper"]},
        {"source_id": "source-object:reviews.paper_id", "target_uri": "http://ex#reviewsPaper", "target_kind": "object_property", "target_domain": ["http://ex#Review"], "target_range": ["http://ex#Paper"]},
        {"source_id": "source-object:paper_author.paper_id", "target_uri": "http://ex#hasAuthor", "target_kind": "object_property", "target_domain": ["http://ex#Paper"], "target_range": ["http://ex#Person"]},
        {"source_id": "source-object:paper_award.paper_id", "target_uri": "http://ex#relatedAward", "target_kind": "object_property", "target_domain": ["http://ex#Paper"], "target_range": ["http://ex#Paper"]},
    ]
    candidate_rows: list[dict[str, object]] = [
        {
            "source": {"id": "source-object:papers.id"},
            "candidates": [{"uri": "http://ex#unsupportedObject", "kind": "object_property", "local_name": "unsupportedObject"}],
        }
    ]
    return tables, data, matches, candidate_rows


def test_pattern_schema_graph_profiles_and_edges() -> None:
    tables, data, _matches, _candidate_rows = _pattern_fixture()
    graph = build_schema_graph(tables, data)
    edge_kinds = {edge["kind"] for edge in graph["edges"]}
    assert {"contains_column", "primary_key", "foreign_key", "reverse_foreign_key", "primary_key_as_foreign_key"} <= edge_kinds
    title_profile = graph["value_profiles"]["papers.title"]
    assert title_profile["title_like"]
    assert title_profile["distinct_count"] == 2
    assert graph["value_profiles"]["papers.id"]["identifier_likeness"] >= 0.8


def test_pattern_extraction_covers_required_pattern_shapes() -> None:
    tables, data, matches, candidate_rows = _pattern_fixture()
    graph = build_schema_graph(tables, data)
    patterns = extract_pattern_candidates(tables, data, matches, candidate_rows, graph)
    types = {pattern["pattern_type"] for pattern in patterns}
    assert {"SCHEMA_ENTITY", "FK_OBJECT", "JOIN_TABLE_OBJECT", "ASSOCIATION_ENTITY", "SUBCLASS_PKFK", "LITERAL_ATTRIBUTE", "IDENTIFIER_AS_URI"} <= types
    assert not any(pattern["source_id"] == "source-object:papers.id" for pattern in patterns)
    assert any(pattern["source_id"] == "source-data:papers.title" and pattern["pattern_type"] == "LITERAL_ATTRIBUTE" for pattern in patterns)
    assert not any(pattern["source_id"] == "source-data:papers.id" and pattern["target_uri"] == "http://ex#title" for pattern in patterns)


def test_pattern_prompts_are_budgeted_and_leakage_safe() -> None:
    tables, data, matches, candidate_rows = _pattern_fixture()
    patterns = extract_pattern_candidates(tables, data, matches, candidate_rows, build_schema_graph(tables, data))
    prompts = build_pattern_selection_prompts(patterns * 8, budget_chars=1800)
    assert len(prompts) > 1
    assert all(prompt["char_count"] <= 1800 for prompt in prompts)
    assert all(validate_prompt_no_leakage(prompt["prompt"]) for prompt in prompts)


def test_pattern_selection_and_uri_key_inference() -> None:
    tables, data, matches, candidate_rows = _pattern_fixture()
    patterns = extract_pattern_candidates(tables, data, matches, candidate_rows, build_schema_graph(tables, data))
    selected, report = select_patterns_from_llm(patterns, [{"selected_pattern_ids": [patterns[0]["pattern_id"], "invented-pattern"]}])
    assert selected == [patterns[0]]
    assert report["invalid_pattern_ids"] == ["invented-pattern"]
    keys = infer_uri_keys(patterns, tables, data)
    assert keys["templates"]["papers"]["id_columns"] == ["id"]
    assert keys["templates"]["accepted_papers"]["reason"] == "subclass_pkfk_parent_identity"


def test_deterministic_pattern_compiler_rejects_invented_fields_and_keeps_uri_consistency() -> None:
    tables, data, matches, candidate_rows = _pattern_fixture()
    patterns = extract_pattern_candidates(tables, data, matches, candidate_rows, build_schema_graph(tables, data))
    selected = [
        pattern
        for pattern in patterns
        if pattern["source_id"] in {"source-class:papers", "source-data:papers.title", "source-object:reviews.paper_id"}
    ]
    invented = dict(selected[0])
    invented["pattern_id"] = "invented"
    invented["rule_fields"] = {"source_table": "missing", "target_class": "http://ex#Paper", "match_ids": ["source-class:missing"]}
    selected.append(invented)
    keys = infer_uri_keys(selected, tables, data)
    fol, report = compile_patterns_to_fol(selected, tables, keys, safe_domain_range_type_completion=True)
    assert any(rule["source_table"] == "papers" and rule["id_columns"] == ["id"] for rule in fol["rules"]["class"])
    assert any(rule["source_table"] == "papers" and rule["source_column"] == "title" for rule in fol["rules"]["data"])
    assert any(rule["source_table"] == "reviews" and rule["target_table"] == "papers" for rule in fol["rules"]["object"])
    assert any(item["reason"] == "unknown_source_table" for item in report["rejected"])
    assert not any(rule.get("target_class") == "http://ex#Invented" for rule in fol["rules"]["class"])


def test_pattern_cli_flags_and_default_off() -> None:
    parser = build_parser()
    default_args = parser.parse_args(["run-paper-compare"])
    assert not default_args.pattern_first_fgf
    args = parser.parse_args(
        [
            "run-paper-compare",
            "--pattern-first-fgf",
            "--pattern-schema-graph",
            "--pattern-candidate-expansion",
            "--uri-key-inference",
            "--deterministic-pattern-compiler",
            "--pattern-candidates",
            "3",
            "--internal-rerank-pattern-candidates",
            "--pattern-selector-context",
            "batched",
            "--safe-domain-range-type-completion",
            "--pattern-materialization-validation",
        ]
    )
    assert args.pattern_first_fgf
    assert args.pattern_candidates == 3
    assert args.pattern_selector_context == "batched"
    assert args.internal_rerank_pattern_candidates


def test_pattern_rerank_score_uses_internal_diagnostics_only() -> None:
    score = pattern_selection_internal_score(
        {"compiler_report": {"rejected": [], "rule_counts": {"class": 1, "data": 2, "object": 1}}, "uri_key_inference": {"consistency_issues": []}},
        {"issues": [], "target_coverage": []},
    )
    assert isinstance(score, tuple)
    assert len(score) == 6
