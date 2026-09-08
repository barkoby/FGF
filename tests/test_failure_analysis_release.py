import csv
import json
import subprocess
import sys
from pathlib import Path
import pytest
from coding_fgf.evaluate import calculate_precision_recall_f1
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from run_failure_analysis import summarize

SCRIPT=Path(__file__).resolve().parents[1]/"scripts"/"run_failure_analysis.py"

def evaluator_row(sql,sparql,categories="attrib"):
    p,r,f=calculate_precision_recall_f1(sparql,sql)
    return {"name":"Q","sql":"SELECT x","sparql":"SELECT ?x WHERE {}","categories":categories,
        "sql_results":sql,"sparql_results":sparql,"precision":p,"recall":r,"f1":f}

@pytest.mark.parametrize("row,expected",[
    (evaluator_row(["x"],[]),"zero_sparql_answers"),
    (evaluator_row(["x","y"],["x"],"object"),"missing_or_wrong_link_path"),
    (evaluator_row(["x","y"],["x"],"ref"),"missing_or_wrong_link_path"),
    (evaluator_row(["x","y"],["x"]),"missing_attribute_or_class"),
    (evaluator_row(["x"],["x","y"]),"overgenerated_sparql_answers"),
    ({"precision":0.5,"recall":1,"sql_results":["x","y"],"sparql_results":["x","z"],"categories":"class"},"other"),
    (evaluator_row([],[]),None),
    (evaluator_row(["x"],["x"]),None),
    (evaluator_row([],["x"]),"overgenerated_sparql_answers"),
])
def test_evaluator_json_classification_branches(tmp_path,row,expected):
    metrics=tmp_path/"runs"/"mini"/"eval"/"metrics_details.json";metrics.parent.mkdir(parents=True)
    metrics.write_text(json.dumps([row]))
    output=tmp_path/"summary.csv"
    summarize(metrics,output)
    rows=list(csv.DictReader(output.open()))
    assert [r["failure_category"] for r in rows]==([expected] if expected else [])

def test_actual_evaluator_shape_through_cli(tmp_path):
    metrics=tmp_path/"metrics.json";metrics.write_text(json.dumps([evaluator_row(["x"],[])]))
    output=tmp_path/"summary.csv"
    result=subprocess.run([sys.executable,str(SCRIPT),"--metrics",str(metrics),"--scenario","mini","--output",str(output)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert list(csv.DictReader(output.open()))==[{"scenario":"mini","failure_category":"zero_sparql_answers","count":"1"}]

def test_metrics_flag_is_required():
    result=subprocess.run([sys.executable,str(SCRIPT)],capture_output=True,text=True)
    assert result.returncode==2 and "--metrics" in result.stderr

def test_csv_category_alias_and_scenario_filter(tmp_path):
    path=tmp_path/"metrics.csv"
    path.write_text("scenario,precision,recall,sql_count,sparql_count,category\nmini,0,0,1,0,object\nother,0,0,1,0,class\nmini,1,1,0,0,class\n")
    output=tmp_path/"summary.csv";summarize(path,output,"mini")
    assert list(csv.DictReader(output.open()))==[{"scenario":"mini","failure_category":"zero_sparql_answers","count":"1"}]

@pytest.mark.parametrize("payload",[
    [],[evaluator_row(["x"],[])],
])
def test_json_requires_scenario_outside_canonical_path(tmp_path,payload):
    path=tmp_path/"metrics.json";path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="scenario"):
        summarize(path,tmp_path/"out.csv")

@pytest.mark.parametrize("missing",["precision","recall","sql_results","sparql_results","categories"])
def test_missing_json_fields_fail_without_output(tmp_path,missing):
    row=evaluator_row(["x"],[]);row.pop(missing)
    path=tmp_path/"metrics.json";path.write_text(json.dumps([row]))
    with pytest.raises(ValueError,match="Missing"):
        summarize(path,tmp_path/"out.csv","mini")
    assert not (tmp_path/"out.csv").exists()

def test_csv_required_field_validation(tmp_path):
    path=tmp_path/"metrics.csv";path.write_text("scenario,precision,recall\nmini,0,0\n")
    with pytest.raises(ValueError,match="required"):
        summarize(path,tmp_path/"out.csv","mini")

def test_scenario_conflict_fails(tmp_path):
    path=tmp_path/"runs"/"mini"/"eval"/"metrics_details.json";path.parent.mkdir(parents=True);path.write_text("[]")
    with pytest.raises(ValueError,match="conflicts"):
        summarize(path,tmp_path/"out.csv","different")
