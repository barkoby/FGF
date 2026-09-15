"""Regressions from release validation, using synthetic identifiers only."""
import copy
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from coding_fgf import cli, llm, providers


@pytest.fixture
def candidate():
    return {"source": {"id": "source-object:items.owner_id", "kind": "object_property",
                       "uri": "urn:source:items#owner", "text": "Item owner"},
            "candidates": [{"id": "object_property:urn:target:ownedBy", "uri": "urn:target:ownedBy",
                            "kind": "object_property", "text": "Item owned by agent"}]}


def decision(candidate, **changes):
    row = {"source_id": candidate["source"]["id"], "target_uri": "urn:target:ownedBy",
           "target_id": "object_property:urn:target:ownedBy", "confidence": .8, "reason": "Owner relation"}
    return {"matches": [{**row, **changes}]}


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setenv("CODING_FGF_MODEL_OUTPUT_MAX_ATTEMPTS", "2")
    monkeypatch.setattr(llm, "_sleep_before_retry", Mock())


@pytest.mark.parametrize("provider", ["openai", "google"])
def test_wrong_source_id_gets_specific_feedback_and_request_schema(monkeypatch, candidate, provider):
    call = Mock(side_effect=[decision(candidate, source_id="object_property:urn:source:items#owner"), decision(candidate)])
    monkeypatch.setattr(llm, "call_structured_json", call)
    rows, _ = llm._match_one_live(candidate, provider=provider)
    assert rows[0].source_id == candidate["source"]["id"]
    assert call.call_count == 2
    assert "source_id_mismatch" in call.call_args.args[0]
    schema = call.call_args.kwargs["output_schema"]
    item = schema["properties"]["matches"]["items"]
    assert item["properties"]["source_id"]["enum"] == [candidate["source"]["id"]]
    assert item["properties"]["target_uri"]["enum"] == ["urn:target:ownedBy", None]


@pytest.mark.parametrize("bad", [
    {"target_id": "object_property:urn:target:other"},
    {"confidence": float("nan")}, {"confidence": float("inf")}, {"confidence": True},
    {"confidence": 1.2}, {"target_uri": None, "target_id": "object_property:urn:target:ownedBy"},
])
def test_inconsistent_match_is_not_accepted(monkeypatch, candidate, bad):
    call = Mock(return_value=decision(candidate, **bad))
    monkeypatch.setattr(llm, "call_structured_json", call)
    with pytest.raises(RuntimeError, match="no valid match"):
        llm._match_one_live(candidate)
    assert call.call_count == 2


@pytest.mark.parametrize("shape", ["duplicate", "missing", "wrong_type"])
def test_exactly_one_decision_required(monkeypatch, candidate, shape):
    data = decision(candidate)
    if shape == "duplicate": data["matches"] *= 2
    if shape == "missing": data["matches"] = []
    if shape == "wrong_type": data["matches"] = ["not a match"]
    call = Mock(return_value=data)
    monkeypatch.setattr(llm, "call_structured_json", call)
    with pytest.raises(RuntimeError, match="no valid match"):
        llm._match_one_live(candidate)
    assert call.call_count == 2


def test_valid_abstention_is_one_successful_request(monkeypatch, candidate):
    call = Mock(return_value=decision(candidate, target_uri=None, target_id=None))
    monkeypatch.setattr(llm, "call_structured_json", call)
    rows, _ = llm._match_one_live(candidate)
    assert rows[0].target_uri is None and call.call_count == 1


def test_provider_sends_strict_schema_and_preserves_usage(monkeypatch):
    import openai
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    call = Mock(return_value=NS(output_text='{"ok":true}', status="completed", model="requested",
                               usage=NS(input_tokens=3, output_tokens=2, total_tokens=5)))
    monkeypatch.setattr(openai, "OpenAI", lambda **_: NS(responses=NS(create=call)))
    schema = {"type":"object", "properties":{"ok":{"type":"boolean"}},
              "required":["ok"], "additionalProperties":False}
    result = providers.structured_generate("Return JSON", "ready", "requested", output_schema=schema)
    assert call.call_args.kwargs["text"]["format"] == {
        "type":"json_schema", "name":"ready", "strict":True, "schema":schema}
    assert result.usage["total_tokens"] == 5 and result.model_used == "requested"


@pytest.mark.parametrize("status", ["refusal", "incomplete"])
def test_provider_does_not_accept_refused_or_incomplete_output(monkeypatch, status):
    import openai
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    response = NS(output_text='{"ok":true}', usage={}, status="incomplete" if status=="incomplete" else "completed",
                  output=[NS(content=[NS(type="refusal", refusal="declined")])] if status=="refusal" else [])
    call = Mock(return_value=response)
    monkeypatch.setattr(openai, "OpenAI", lambda **_: NS(responses=NS(create=call)))
    with pytest.raises(RuntimeError, match=status):
        providers.structured_generate("Return JSON", "ready", "requested")


@pytest.mark.parametrize("kind,field,prefix", [("class","target_class","class"),
    ("data","target_property","data_property"),("object","target_property","object_property")])
def test_prefix_detector_uses_actual_fol_fields(kind, field, prefix):
    assert cli._role_prefixed_target_count({"rules":{kind:[{field:prefix+":urn:target:Thing"}]}}) == 1


def test_target_normalization_requires_accepted_same_kind_match():
    from coding_fgf.fol import finalize_target_uris
    matches = [{"source_id":"source-class:things", "target_uri":"urn:target:Thing",
                "target_id":"class:urn:target:Thing", "target_kind":"class"}]
    fol = {"rules":{"class":[{"target_class":"class:urn:target:Thing"}],
                    "data":[{"target_property":"data_property:urn:target:Thing"}],
                    "object":[{"target_property":"object_property:urn:target:Unknown"}]}}
    original = copy.deepcopy(fol)
    cleaned, report = finalize_target_uris(fol, matches)
    assert cleaned["rules"]["class"][0]["target_class"] == "urn:target:Thing"
    assert cleaned["rules"]["data"][0]["target_property"] == "data_property:urn:target:Thing"
    assert cleaned["rules"]["object"][0]["target_property"] == "object_property:urn:target:Unknown"
    assert report["canonicalized_target_uri_count"] == 1 and fol == original


def test_model_defaults_are_consistent():
    from coding_fgf.constants import REQUESTED_CODE_MODEL, REQUESTED_MATCH_MODEL, FALLBACK_CODE_MODEL, FALLBACK_MATCH_MODEL
    assert REQUESTED_CODE_MODEL == REQUESTED_MATCH_MODEL == "gpt-5.4-nano"
    assert FALLBACK_CODE_MODEL is None and FALLBACK_MATCH_MODEL is None




@pytest.mark.parametrize("provider", ["openai", "google"])
def test_semantic_review_repairs_contract_errors(monkeypatch, candidate, provider):
    source = decision(candidate)["matches"][0]
    call = Mock(side_effect=[decision(candidate, source_id="wrong"), decision(candidate)])
    monkeypatch.setattr(llm, "call_structured_json", call)
    reviewed, report = llm.review_all_matches([source], [candidate], {}, provider=provider)
    assert reviewed[0]["source_id"] == source["source_id"] and call.call_count == 2
    assert "source_id_mismatch" in call.call_args.args[0]
    assert report[0]["status"] == "reviewed"


@pytest.mark.parametrize("arm", ["full9_default", "stage2_hybrid", "stage2c_round2_only"])
def test_every_portfolio_arm_finalizes_target_ids(tmp_path, monkeypatch, arm):
    from coding_fgf.schema import Table, Column, SqlData
    tables = {"items": Table("items", [Column("id", "integer")], ["id"])}
    matches = [{"source_id":"source-class:items", "target_uri":"urn:target:Item",
                "target_id":"class:urn:target:Item", "target_kind":"class"}]
    raw = {"rules":{"class":[{"source_table":"items", "id_columns":["id"],
            "target_class":"class:urn:target:Item", "match_ids":["source-class:items"]}], "data":[], "object":[]}}
    monkeypatch.setattr(cli, "llm_fol", lambda *a, **k: copy.deepcopy(raw))
    monkeypatch.setattr(cli, "_copy_frozen_upstream_artifacts", lambda *a: None)
    monkeypatch.setattr(cli, "_run_fol_single_round2_style_repair", lambda **k: (k["fol"],[],{}))
    args = cli.build_parser().parse_args(["run-paper-compare", "--llm-model", "requested"])
    work, fol, issues, _ = cli._generate_fol_portfolio_arm(scenario="synthetic", dev_root=tmp_path,
        scenario_work=tmp_path, arm=arm, matches=matches, tables=tables, data=SqlData(rows={}), args=args, offline=False)
    assert fol["rules"]["class"][0]["target_class"] == "urn:target:Item"
    assert cli._role_prefixed_target_count(fol) == 0
    assert json.loads((work/"fol_rules_raw.json").read_text()) == raw


def test_explicit_matching_settings_roundtrip_yaml_command():
    from scripts._config import pipeline_command
    settings = {"retrieval_metric":"cosine", "match_candidate_limit":16,
                "match_candidate_context":"full", "match_validation":"all"}
    command = pipeline_command(settings)
    args = cli.build_parser().parse_args(command[3:])
    assert all(getattr(args,key) == value for key,value in settings.items())


@pytest.mark.parametrize("provider", ["openai", "google"])
def test_semantic_heuristic_is_recorded_without_rejecting_valid_contract(monkeypatch,provider):
    # Renaming a source table does not invalidate an otherwise grounded identifier.
    from coding_fgf.schema import Table,Column,ForeignKey
    tables={"source_rows":Table("source_rows",[Column("entity_code")],[],
        [ForeignKey(["entity_code"],"entities",["code"])])}
    row={"source":{"id":"source-data:source_rows.entity_code","kind":"data_property"},
         "candidates":[{"id":"data_property:urn:identifier","uri":"urn:identifier",
             "kind":"data_property","local_name":"identifier","domain":["urn:Entity"]}]}
    response={"matches":[{"source_id":row["source"]["id"],"target_uri":"urn:identifier",
        "target_id":"data_property:urn:identifier","confidence":.9,"reason":"Supplied entity identifier."}]}
    initial=llm.validate_matches(response["matches"],[row])[0]
    from dataclasses import asdict
    initial=asdict(initial)
    assert "fk_column_mapped_to_context_mismatched_identifier_property" in llm.match_validation_issues(initial,tables)
    call=Mock(return_value=response);monkeypatch.setattr(llm,"call_structured_json",call)
    reviewed,report=llm.review_all_matches([initial],[row],tables,provider=provider)
    assert reviewed[0]["target_uri"]=="urn:identifier" and call.call_count==1
    assert "fk_column_mapped_to_context_mismatched_identifier_property" in report[0]["semantic_warnings"]


def test_semantic_review_still_rejects_structural_kind_mismatch(monkeypatch):
    row={"source":{"id":"source-class:items","kind":"class"},"candidates":[
        {"id":"object_property:urn:related","uri":"urn:related","kind":"object_property"}]}
    response={"matches":[{"source_id":"source-class:items","target_uri":"urn:related",
        "target_id":"object_property:urn:related","confidence":.8,"reason":"test"}]}
    call=Mock(return_value=response);monkeypatch.setattr(llm,"call_structured_json",call)
    with pytest.raises(RuntimeError,match="Semantic validation"):
        llm.review_all_matches(response["matches"],[row],{})
    assert call.call_count==2


def test_dense_cosine_ignores_norm_and_lexical_bonus(tmp_path, monkeypatch):
    from coding_fgf.retrieval import build_index, retrieve_candidates
    monkeypatch.setenv("CODING_FGF_RETRIEVAL_METRIC", "cosine")
    build_index([{"id": "b", "uri": "urn:b", "kind": "class", "label": "needle", "embedding": [1, 1]},
                 {"id": "a", "uri": "urn:a", "kind": "class", "label": "other", "embedding": [10, 0]}], tmp_path)
    result = retrieve_candidates([{"id": "x", "kind": "class", "label": "needle", "embedding": [1, 0]}], tmp_path, k=2)
    assert [c["id"] for c in result[0]["candidates"]] == ["a", "b"]



def test_all_sixteen_candidate_descriptions_reach_prompt(monkeypatch):
    from coding_fgf.llm import _match_prompt
    monkeypatch.setenv("CODING_FGF_MATCH_CANDIDATE_LIMIT", "16")
    row = {"source": {"id": "x", "kind": "class"}, "candidates": [
        {"id": str(i), "uri": f"urn:c{i}", "kind": "class", "text": f"meaning_{i}", "domain": ["urn:Group"]}
        for i in range(16)]}
    prompt = _match_prompt(row)
    assert "meaning_15" in prompt
    assert '"domain": ["urn:Group"]' in prompt



def test_full_fraction_preserves_duplicate_and_dangling_rows():
    from coding_fgf.devset import select_dev_rows
    from coding_fgf.schema import Table, Column, ForeignKey, SqlData
    tables={"entities":Table("entities",[Column("id")],["id"]),
            "links":Table("links",[Column("a"),Column("b")],["a","b"],
                          [ForeignKey(["a"],"entities",["id"]),ForeignKey(["b"],"entities",["id"])])}
    data=SqlData({"entities":[{"id":"1"},{"id":"1"}],"links":[{"a":"1","b":"missing"}]})
    assert select_dev_rows(tables,data,fraction=1)==data.rows



def test_semantic_validation_reviews_a_null_decision(monkeypatch):
    import coding_fgf.llm as llm
    monkeypatch.setenv("CODING_FGF_MATCH_VALIDATION","all")
    row={"source":{"id":"source-class:collectives","kind":"class","uri":"urn:s"},
         "candidates":[{"id":"c","uri":"urn:Collective","kind":"class","label":"Collective"}]}
    calls=[]
    def call(*args,**kwargs):
        calls.append(args)
        return {"matches":[{"source_id":"source-class:collectives","target_uri":"urn:Collective","confidence":.8,"reason":"Supplied group label"}]}
    monkeypatch.setattr(llm,"_call_json_for_provider",call)
    out,report=llm.reask_suspicious_matches([{"source_id":"source-class:collectives","target_uri":None}], [row], {})
    assert len(calls)==1 and out[0]["target_uri"]=="urn:Collective"
    assert report[0]["before"]["target_uri"] is None



def test_missing_baseline_does_not_fall_back_to_embedded_scores():
    from coding_fgf.compare import decide_promotion
    decision = decide_promotion("synthetic", coding_f1=0.6, llm4vkg_f1=None, invalid_rules=0)
    assert not decision.promoted
    assert decision.fallback_f1 is None
    assert decision.reason == "missing baseline F1"


def test_benchmark_without_reference_is_uncompared():
    from coding_fgf.benchmark import _scenario_summary_row
    row = _scenario_summary_row("synthetic", [{"f1": "0.6", "artifact_path": "missing-fixture"}])
    assert row["status"] == "uncompared"
    assert row["target_f1"] is None and not row["passed"]
