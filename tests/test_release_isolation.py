import csv
import json
import multiprocessing
import os
import sys
from pathlib import Path
import pytest
from coding_fgf import cli, fol_ablation, sandbox
from coding_fgf.llm import default_codegen
from coding_fgf.schema import Table, Column, SqlData
from coding_fgf.materialize import materialize_graph_with_log, materialize_to_file

def materializer_inputs():
    tables={"people":Table("people",[Column("id"),Column("email")],["id"])}
    data=SqlData(rows={"people":[{"id":"1","email":"a@b"},{"id":"2","email":""}]})
    fol={"rules":{"class":[{"source_table":"people","target_class":"http://ex#Person","id_columns":["id"]}],
        "data":[{"source_table":"people","source_column":"email","target_property":"http://ex#email"}],"object":[]}}
    return "mini",tables,data,fol

def crash_worker(*args):
    os._exit(17)

@pytest.fixture
def child_pids():
    before={p.pid for p in multiprocessing.active_children()}
    yield before
    assert {p.pid for p in multiprocessing.active_children()}==before

@pytest.mark.skipif(sys.platform!="linux",reason="Linux resource limits exercised in Docker")
def test_spawned_worker_preserves_triples_and_complete_diagnostics(child_pids):
    inputs=materializer_inputs()
    expected,expected_log=materialize_graph_with_log(*inputs)
    graph,log=materialize_graph_with_log(*inputs,code=default_codegen())
    assert set(graph)==set(expected)
    assert log==expected_log
    assert log["rule_stats"]["data:0"]["failure_counts"]=={"empty_source_value":1}

@pytest.mark.skipif(sys.platform!="linux",reason="Linux resource limits exercised in Docker")
@pytest.mark.parametrize("code,status,timeout,memory",[
    ("def materialize(context):\n    for i in range(10**12):\n        pass\n","timeout","0.5","2048"),
    ("def materialize(context):\n    x = 'x' * (1024**3)\n","memory_exhausted","10","128"),
    ("def materialize(context):\n    x = 1/0\n","worker_error","10","2048"),
])
def test_worker_failures_reap_process_without_publishing_partial_rdf(tmp_path,monkeypatch,child_pids,code,status,timeout,memory):
    monkeypatch.setenv("CODING_FGF_SANDBOX_TIMEOUT_SECONDS",timeout)
    monkeypatch.setenv("CODING_FGF_SANDBOX_MEMORY_MB",memory)
    output=tmp_path/"import.ttl"
    output.write_text("stale RDF must not survive")
    codepath=tmp_path/"generated.py";codepath.write_text(code)
    scenario,tables,data,fol=materializer_inputs()
    folpath=tmp_path/"fol.json";folpath.write_text(json.dumps(fol))
    with pytest.raises(sandbox.SandboxExecutionError) as error:
        materialize_to_file(scenario,tables,data,folpath,output,code_path=codepath)
    assert error.value.status==status
    assert not output.exists()
    log=json.loads((tmp_path/"import.materialization_log.json").read_text())
    assert log["status"]==status

@pytest.mark.skipif(sys.platform!="linux",reason="Linux resource limits exercised in Docker")
def test_worker_crash_reaps_child(monkeypatch,child_pids):
    monkeypatch.setattr(sandbox,"_materializer_worker",crash_worker)
    with pytest.raises(sandbox.SandboxExecutionError,match="code 17"):
        sandbox.run_isolated_materializer(*materializer_inputs(),default_codegen())

@pytest.mark.skipif(sys.platform!="linux",reason="Linux resource limits exercised in Docker")
def test_large_result_transfer_does_not_deadlock(monkeypatch,child_pids):
    monkeypatch.setenv("CODING_FGF_SANDBOX_TIMEOUT_SECONDS","15")
    scenario,tables,data,fol=materializer_inputs()
    data.rows["people"]=[{"id":str(i),"email":"x"*80} for i in range(15000)]
    triples,log=sandbox.run_isolated_materializer(scenario,tables,data,fol,default_codegen())
    assert len(triples)==30000
    assert log["rule_stats"]["class:0"]["helper_calls"]==15000
    assert log["rule_stats"]["data:0"]["emitted_triples"]==15000

@pytest.mark.parametrize("code",[
    "import os\ndef materialize(context): pass",
    "def materialize(context):\n    open('file')",
    "def materialize(context):\n    x = context.__class__",
    "def materialize(context):\n    while True: pass",
])
def test_unsafe_syntax_rejected_before_worker_start(monkeypatch,code):
    def forbidden(*a,**k): raise AssertionError("worker must not start")
    monkeypatch.setattr(multiprocessing,"get_context",forbidden)
    with pytest.raises(sandbox.SandboxViolation):
        sandbox.run_isolated_materializer(*materializer_inputs(),code)

def test_unsupported_platform_has_docker_guidance(monkeypatch):
    monkeypatch.setattr(sys,"platform","win32")
    with pytest.raises(sandbox.SandboxExecutionError,match="Docker"):
        sandbox.run_isolated_materializer(*materializer_inputs(),default_codegen())

@pytest.mark.parametrize("scores",[(0.1,0.9),(0.9,0.1)])
def test_portfolio_selection_is_behaviorally_isolated_from_evaluation(monkeypatch,tmp_path,scores):
    work=tmp_path/"runs"/"mini";work.mkdir(parents=True)
    queries=tmp_path/"data"/"mini"/"queries";queries.mkdir(parents=True)
    secret=queries/"secret.qpair";secret.write_text("evaluation-only content")
    arms=["full9_default","stage2_hybrid","stage2c_round2_only"]
    generated=[];materialized=[];evaluated=[]
    original_open=Path.open
    def guarded_open(path,*args,**kwargs):
        mode=args[0] if args else kwargs.get("mode","r")
        if "r" in mode and (path.suffix==".qpair" or "eval" in path.parts or path.name.startswith("metrics")):
            assert (work/"selected_fol_arm.json").exists(), "evaluation artifact accessed before selection"
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guarded_open)
    monkeypatch.setattr(cli,"_fol_portfolio_prepare_upstream",lambda **_: ([],{},SqlData(rows={})))
    def generate(**kwargs):
        arm=kwargs["arm"];generated.append(arm)
        if arm==arms[2]: raise RuntimeError("synthetic failed arm")
        path=work/"fol_portfolio"/arm;path.mkdir(parents=True)
        (path/"fol.json").write_text('{"rules":{}}')
        return path,{"rules":{}},[],{}
    def materialize(**kwargs):
        arm=kwargs["scenario_work"].name;materialized.append(arm)
        kwargs["output"].write_text("<urn:"+arm+"> <urn:p> <urn:o> .")
        return {}
    def record(**kwargs):
        return {"arm":kwargs["arm"],"work":str(kwargs["arm_work"]),"generated_triples":100,
            "import_exists":True,"materialization_status":"success","issues_after":0,
            "selected_targets_with_emission":2 if kwargs["arm"]==arms[0] else 1}
    def evaluate(graph,qpair_dir,executor,output):
        saved=json.loads((work/"selected_fol_arm.json").read_text())
        assert saved["selected_arm"]==arms[0]
        assert secret.read_text()=="evaluation-only content"
        arm=graph.parent.name;evaluated.append(arm)
        assert arm in graph.read_text()
        score=scores[arms.index(arm)]
        return {"precision":score,"recall":score,"f1":score,"count":7}
    monkeypatch.setattr(cli,"_generate_fol_portfolio_arm",generate)
    monkeypatch.setattr(cli,"_codegen_and_materialize_with_round2_only_fallback",materialize)
    monkeypatch.setattr(cli,"_fol_portfolio_record",record)
    monkeypatch.setattr(cli,"fol_validation_issues",lambda *_: [])
    monkeypatch.setattr(fol_ablation,"evaluate_graph",evaluate)
    args=cli.build_parser().parse_args(["run-paper-compare","--fol-portfolio","--fol-ablation-report"])
    result=cli._run_fol_portfolio(scenario="mini",dev_root=tmp_path/"data",scenario_work=work,args=args,offline=True)
    assert result["selected_arm"]==arms[0]
    assert generated==arms and materialized==arms[:2] and evaluated==arms[:2]
    rows=list(csv.DictReader((work/"fol_ablation"/"fol_selection_comparison.csv").open()))
    by_arm={row["arm"]:row for row in rows}
    assert by_arm[arms[2]]["status"]=="failed" and by_arm[arms[2]]["f1"]==""
    assert by_arm["portfolio_selected"]["f1"]==by_arm[arms[0]]["f1"]==str(scores[0])
    assert by_arm["portfolio_selected"]["query_count"]=="7"
    assert (work/"import.ttl").read_bytes()==(work/"fol_portfolio"/arms[0]/"import.ttl").read_bytes()
    # Any final refinement changes the main graph, never the frozen selection result.
    (work/"import.ttl").write_text("later refinement")
    assert (work/"fol_portfolio"/arms[0]/"import.ttl").read_text().startswith("<urn:")

def test_empty_generated_file_never_uses_deterministic_fallback(tmp_path):
    scenario,tables,data,fol=materializer_inputs()
    folpath=tmp_path/"fol.json";folpath.write_text(json.dumps(fol))
    code=tmp_path/"generated.py";code.write_text("")
    output=tmp_path/"import.ttl"
    with pytest.raises(sandbox.SandboxViolation):
        materialize_to_file(scenario,tables,data,folpath,output,code_path=code)
    assert not output.exists()

def test_offline_portfolio_runs_without_live_repair_calls(tmp_path,monkeypatch):
    from test_core import write_fixture
    from coding_fgf import llm, providers
    write_fixture(tmp_path/"data")
    def forbidden(*a,**k):
        raise AssertionError("offline execution attempted a live provider call")
    monkeypatch.setattr(llm,"call_structured_json",forbidden)
    monkeypatch.setattr(providers,"structured_generate",forbidden)
    work=tmp_path/"out"
    cli.main(["run-paper-compare","--rodi-root",str(tmp_path),"--work",str(work),
        "--scenarios","mini","--fraction","1","--offline","--dry-run-db","--fol-portfolio","--fol-ablation-report"])
    assert (work/"runs"/"mini"/"import.ttl").exists()
    assert (work/"runs"/"mini"/"selected_fol_arm.json").exists()
