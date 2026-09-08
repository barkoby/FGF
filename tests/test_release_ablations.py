import json
import sys
from pathlib import Path
import pytest
import yaml
from coding_fgf import cli, providers, candidate_methods, eval_candidates
from coding_fgf.analysis import matching_analysis as matching
from coding_fgf.analysis.matching_prompts import MATCHING_METHODS
from coding_fgf.constants import PAPER_SCENARIOS
from test_matching_analysis import candidate_row, write_minimal_rodi_scenario, write_candidate_artifact
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from _config import ablation_command, load_config, pipeline_command

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("provider",["openai","google"])
@pytest.mark.parametrize("method",candidate_methods.DEFAULT_METHODS)
def test_all_candidate_methods_route_and_record_provider(monkeypatch,tmp_path,provider,method):
    calls=[]
    def embed(self,texts):
        calls.append((self.provider,self.model,self.google_project))
        return [[float(len(text)),1.0] for text in texts]
    monkeypatch.setattr(candidate_methods.EmbeddingClient,"embed",embed)
    target=[{"id":"t","uri":"http://ex#email","kind":"data_property","text":"email","label":"email"}]
    context=candidate_methods.CandidateMethodContext([method],target,tmp_path,
        embedding_model="configured-embedding",embedding_provider=provider,google_project="project")
    ranked,_=candidate_methods.rank_method(context,method,candidate_row()["source"])
    assert ranked[0].uri=="http://ex#email"
    assert context.method_configs([1],1)["embedding_provider"]==provider
    if "dense" in method:
        assert calls and all(c==(provider,"configured-embedding","project") for c in calls)
    else:
        assert calls==[]

@pytest.mark.parametrize("provider",["openai","google"])
@pytest.mark.parametrize("method",MATCHING_METHODS)
def test_all_matching_methods_preserve_provider_prompt_temperature_usage(monkeypatch,provider,method):
    row=candidate_row()
    calls=[]
    def fake(prompt,schema,model,**kwargs):
        calls.append((prompt,schema,model,kwargs))
        if schema=="match_cov_stage1":
            data={"source_id":row["source"]["id"],"shortlist":[{"target_uri":"http://ex#email",
                "confidence":0.9,"reasons":["name","domain","sample"]}],
                "null_option":{"allowed":True,"reasons":["a","b","c"]}}
        else:
            data={"matches":[{"source_id":row["source"]["id"],"target_uri":"http://ex#email",
                "confidence":0.9,"decision":"selected"}]}
        return providers.StructuredResult(data,model,{"input_tokens":2,"output_tokens":3},provider)
    monkeypatch.setattr(matching,"structured_generate",fake)
    args=matching.build_parser().parse_args(["--llm-provider",provider,"--model","configured-model",
        "--google-project","project","--google-location","global","--google-credentials","credential-path",
        "--self-consistency-samples","2","--temperature","0.1","--self-consistency-temperature","0.7"])
    usage={}
    results=matching.run_method([row],{row["source"]["id"]:["http://ex#email"]},method,args,usage)
    assert results[0]["correct"] is True
    assert results[0]["llm_provider"]==provider
    assert results[0]["models_used"]==["configured-model"]
    assert usage=={"input_tokens":2*len(calls),"output_tokens":3*len(calls)}
    assert len(calls)==(2 if method in {"chain_of_verification","self_consistency"} else 1)
    for prompt,schema,model,options in calls:
        assert row["source"]["id"] in prompt and "http://ex#email" in prompt
        assert model=="configured-model" and options["provider"]==provider
        assert options["temperature"]==(0.7 if method=="self_consistency" else 0.1)
        if provider=="google":
            assert options["google_project"]=="project"
            assert options["google_credentials"]=="credential-path"

@pytest.mark.parametrize("filename",["openai_best_fol_default.yaml","gemini_default.yaml","gemma4_default.yaml"])
def test_yaml_commands_parse_and_ablation_outputs_align(monkeypatch,filename):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT","configured-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION","global")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS","/gcloud/adc.json")
    monkeypatch.setenv("DB_HOST","postgres")
    cfg=load_config(ROOT/"configs"/filename)
    parsed=cli.build_parser().parse_args(pipeline_command(cfg)[3:])
    assert parsed.scenarios=="cmt_renamed" and parsed.db_host=="postgres"
    candidate=eval_candidates.build_parser().parse_args(ablation_command(cfg,"candidate_generation")[3:])
    match=matching.build_parser().parse_args(ablation_command(cfg,"matching")[3:])
    fol=cli.build_parser().parse_args(ablation_command(cfg,"fol_selection")[3:])
    assert candidate.embedding_provider==cfg["embedding_provider"]
    assert match.llm_provider==cfg["llm_provider"] and match.model==cfg["llm_model"]
    assert match.embedding_model==candidate.embedding_model==cfg["embedding_model"]
    assert match.max_workers==cfg["match_workers"] and match.top_k==cfg["k"]
    assert Path(match.candidate_artifact)==Path(candidate.output_dir)/"candidates_by_method.jsonl"
    assert Path(candidate.output_dir).is_relative_to(Path(cfg["work"]))
    assert fol.fol_portfolio and fol.fol_ablation_report
    assert len({str(candidate.output_dir),str(match.output_dir),fol.work})==3
    if cfg["llm_provider"]=="google":
        assert candidate.google_project==match.google_project=="configured-project"
        assert candidate.google_credentials==match.google_credentials=="/gcloud/adc.json"

def test_explicit_output_and_candidate_overrides(tmp_path):
    cfg={"work":str(tmp_path)}
    args=matching.build_parser().parse_args(ablation_command(cfg,"matching",
        output_dir=tmp_path/"custom",candidate_artifact=tmp_path/"chosen.jsonl")[3:])
    assert Path(args.output_dir)==tmp_path/"custom"
    assert Path(args.candidate_artifact)==tmp_path/"chosen.jsonl"

def test_compose_uses_dedicated_healthy_database_and_host_mounts():
    compose=yaml.safe_load((ROOT/"docker-compose.yml").read_text())
    app=compose["services"]["fgf-pipeline"]
    assert app["environment"]["DB_HOST"]=="postgres"
    assert app["depends_on"]["postgres"]["condition"]=="service_healthy"
    assert compose["services"]["postgres"]["healthcheck"]
    volumes=app["volumes"]
    assert any("RODI_DIR" in v and v.endswith(":/data:ro") for v in volumes)
    assert any("OUTPUT_DIR" in v and v.endswith(":/outputs") for v in volumes)
    assert any("GOOGLE_GCLOUD_CONFIG_DIR" in v and v.endswith(":/gcloud:ro") for v in volumes)

def test_defaults_nine_and_explicit_extra_scenarios(tmp_path):
    from coding_fgf.candidate_gold import resolve_scenarios
    for name in PAPER_SCENARIOS:
        write_minimal_rodi_scenario(tmp_path/"data",name)
    write_minimal_rodi_scenario(tmp_path/"data","mondial_rel")
    assert cli.build_parser().parse_args(["run-paper-compare"]).scenarios=="cmt_renamed"
    assert resolve_scenarios(tmp_path,True,None)==list(PAPER_SCENARIOS)
    assert matching.resolve_matching_scenarios(tmp_path,True,None,False)==list(PAPER_SCENARIOS)
    assert resolve_scenarios(tmp_path,False,["mondial_rel"])==["mondial_rel"]
    assert cli.build_parser().parse_args(["run-paper-compare","--scenarios","mondial_rel"]).scenarios=="mondial_rel"

def test_legacy_dense_alias_is_only_openai():
    assert candidate_methods.validate_methods(["openai_small","dense"],"openai")==["dense"]
    with pytest.raises(ValueError,match="OpenAI"):
        candidate_methods.validate_methods(["openai_small"],"google")

@pytest.mark.parametrize("field,value",[
    ("embedding_provider","google"),("embedding_model","different"),
    ("embedding_mode","offline"),("embedding_provider",None)])
def test_candidate_artifacts_reject_incompatible_provenance(tmp_path,field,value):
    write_minimal_rodi_scenario(tmp_path/"data","mini")
    artifact=tmp_path/"candidates.jsonl"
    write_candidate_artifact(artifact,"mini")
    rows=[json.loads(line) for line in artifact.read_text().splitlines()]
    for row in rows:
        row.update(method="dense",embedding_provider="openai",
            embedding_model="text-embedding-3-small",embedding_mode="live")
        row[field]=value
    artifact.write_text("\n".join(map(json.dumps,rows)))
    with pytest.raises(ValueError,match="provenance|incompatible|live"):
        matching.load_candidate_artifact_rows(artifact,tmp_path,["mini"],"dense",16,
            embedding_provider="openai",embedding_model="text-embedding-3-small")

@pytest.mark.parametrize("provider",["openai","google"])
@pytest.mark.parametrize("failed_stage",["match_current_review","match_current_referee"])
def test_review_and_referee_exhaustion_are_recorded(monkeypatch,provider,failed_stage):
    row=candidate_row()
    calls=[]
    def fake(prompt,schema,model,**kwargs):
        calls.append(schema)
        if schema==failed_stage:
            raise providers.RetryExhausted("provider request budget exhausted")
        return providers.StructuredResult({"matches":[{"source_id":row["source"]["id"],
            "target_uri":None,"decision":"no_match","confidence":0.2}]},model,{},provider)
    monkeypatch.delenv("CODING_FGF_NO_API_ERROR_ROWS",raising=False)
    monkeypatch.setattr(matching,"structured_generate",fake)
    args=matching.build_parser().parse_args(["--llm-provider",provider,"--model","configured"])
    result=matching.run_method([row],{},"current_validated",args,{})
    assert result[0]["api_error"] is True
    assert calls.count(failed_stage)==1

def test_legacy_artifact_without_model_provenance_is_rejected(tmp_path):
    write_minimal_rodi_scenario(tmp_path/"data","mini")
    artifact=tmp_path/"candidates.jsonl"
    write_candidate_artifact(artifact,"mini")
    (tmp_path/"method_configs.json").unlink(missing_ok=True)
    with pytest.raises(ValueError,match="model"):
        matching.load_candidate_artifact_rows(artifact,tmp_path,["mini"],"dense",16,
            embedding_model="text-embedding-3-small")

def test_legacy_artifact_uses_saved_model_instead_of_assuming_default(tmp_path):
    write_minimal_rodi_scenario(tmp_path/"data","mini")
    artifact=tmp_path/"candidates.jsonl"
    write_candidate_artifact(artifact,"mini")
    (tmp_path/"method_configs.json").write_text(json.dumps({"embedding_model":"different-model"}))
    with pytest.raises(ValueError,match="model"):
        matching.load_candidate_artifact_rows(artifact,tmp_path,["mini"],"dense",16,
            embedding_model="text-embedding-3-small")
