"""Provider contracts use mocked HTTP/SDK clients; no network is needed."""
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from coding_fgf import providers, google_vertex, embeddings
from coding_fgf.analysis import matching_analysis as matching

@pytest.fixture(autouse=True)
def isolated_retry_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("CODING_FGF_API_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("CODING_FGF_MODEL_OUTPUT_MAX_ATTEMPTS", "2")
    monkeypatch.setattr(providers.time, "sleep", Mock())
    monkeypatch.setattr(google_vertex, "_sleep", Mock())
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("CODING_FGF_NO_API_ERROR_ROWS", raising=False)

class HttpFailure(Exception):
    def __init__(self, status):
        self.status_code = status

def install_transport(monkeypatch, provider, effects):
    request = Mock(side_effect=effects)
    if provider == "openai":
        import openai
        monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: NS(responses=NS(create=request)))
    else:
        monkeypatch.setattr(google_vertex, "_access_token", lambda _: "test-token")
        def post(*args, **kwargs):
            effect = request(*args, **kwargs)
            return NS(status_code=200, raise_for_status=lambda: None, json=lambda: effect)
        monkeypatch.setattr(google_vertex.requests, "post", post)
    return request

def response(provider, text='{"ok":true}'):
    if provider == "openai":
        return NS(output_text=text, usage=NS(input_tokens=2, output_tokens=3, total_tokens=5))
    return {"candidates":[{"content":{"parts":[{"text":text}]}}],
            "usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":3,"totalTokenCount":5}}

def generate(provider):
    return providers.structured_generate("Return JSON", "test", "requested-model",
        provider=provider, google_project="test-project", temperature=0.4)

@pytest.mark.parametrize("provider", ["openai", "google"])
@pytest.mark.parametrize("status", [400,401,403,404,422])
def test_permanent_provider_errors_make_one_request(monkeypatch, provider, status):
    request = install_transport(monkeypatch, provider, [HttpFailure(status)])
    with pytest.raises(HttpFailure):
        generate(provider)
    assert request.call_count == 1
    providers.time.sleep.assert_not_called()

@pytest.mark.parametrize("provider", ["openai", "google"])
@pytest.mark.parametrize("failure", [HttpFailure(429), HttpFailure(500), TimeoutError(), ConnectionError()])
def test_transient_provider_errors_recover_with_backoff(monkeypatch, provider, failure):
    request = install_transport(monkeypatch, provider, [failure, failure, response(provider)])
    result = generate(provider)
    assert request.call_count == 3 and result.data == {"ok":True}
    assert result.model_used == "requested-model" and result.provider == provider
    assert result.usage == {"input_tokens":2,"output_tokens":3,"total_tokens":5}
    assert providers.time.sleep.call_count == 2
    if provider == "openai":
        assert request.call_args.kwargs["temperature"] == 0.4
    else:
        assert request.call_args.kwargs["json"]["generationConfig"]["temperature"] == 0.4

@pytest.mark.parametrize("provider", ["openai", "google"])
@pytest.mark.parametrize("no_error_rows", [False, True])
def test_nested_matching_retries_do_not_restart_exhausted_requests(monkeypatch, provider, no_error_rows):
    request = install_transport(monkeypatch, provider, [HttpFailure(429)] * 10)
    monkeypatch.setenv("CODING_FGF_NO_API_ERROR_ROWS", str(int(no_error_rows)))
    def run():
        return matching.adaptive_map([1], lambda *_: generate(provider),
            lambda task, error, attempts, workers: {"error":error}, api_retries=5)
    if no_error_rows:
        with pytest.raises(RuntimeError, match="API task failed"):
            run()
    else:
        assert "RetryExhausted" in run()[0]["error"]
    assert request.call_count == 3

@pytest.mark.parametrize("provider", ["openai", "google"])
def test_malformed_output_has_separate_budget_and_accumulated_usage(monkeypatch, provider):
    request = install_transport(monkeypatch, provider, [response(provider, "invalid"), response(provider)])
    result = generate(provider)
    assert request.call_count == 2
    assert result.usage == {"input_tokens":4,"output_tokens":6,"total_tokens":10}
    request = install_transport(monkeypatch, provider, [response(provider, "invalid")] * 3)
    with pytest.raises(RuntimeError, match="model-output attempts"):
        generate(provider)
    assert request.call_count == 2

@pytest.mark.parametrize("provider", ["openai", "google"])
def test_explicit_model_fallback_preserves_provider_and_records_model(monkeypatch, provider):
    calls = []
    def fake(prompt, schema, model, **kwargs):
        calls.append((model, kwargs["provider"]))
        if model == "missing":
            raise HttpFailure(404)
        return providers.StructuredResult({}, model, {"total_tokens":7}, provider)
    monkeypatch.setattr(matching, "structured_generate", fake)
    with pytest.raises(HttpFailure):
        matching.ProviderJsonCaller("missing", provider=provider).call("p","s")
    assert calls == [("missing",provider)]
    calls.clear()
    result = matching.ProviderJsonCaller("missing", "explicit", provider=provider).call("p","s")
    assert calls == [("missing",provider),("explicit",provider)]
    assert result.model_used == "explicit"

@pytest.mark.parametrize("name", ["CODING_FGF_API_MAX_ATTEMPTS", "CODING_FGF_MODEL_OUTPUT_MAX_ATTEMPTS"])
@pytest.mark.parametrize("value", ["0","-1"])
def test_invalid_budgets_fail_before_request(monkeypatch, name, value):
    monkeypatch.setenv(name,value)
    request=install_transport(monkeypatch,"openai",[])
    with pytest.raises(ValueError, match="positive"):
        generate("openai")
    assert request.call_count == 0

def test_google_env_credential_path_is_validated(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path/"missing.json"))
    with pytest.raises(ValueError, match="credentials"):
        google_vertex.google_config("test")

def test_missing_google_adc_has_actionable_error(monkeypatch):
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError
    monkeypatch.setattr(google.auth, "default", Mock(side_effect=DefaultCredentialsError("missing")))
    with pytest.raises(google_vertex.GoogleVertexPermanentError, match="credentials|ADC"):
        google_vertex._access_token(google_vertex.GoogleVertexConfig("test"))

def test_legacy_cache_is_preserved_but_not_used(tmp_path):
    cache=tmp_path/"cache.jsonl"
    original=json.dumps({"hash":embeddings.embedding_key("m","x"),"embedding":[999]})+"\n"
    cache.write_text(original)
    embed=Mock(return_value=[[1,2]])
    assert embeddings.cached_embeddings(["x"],cache,embed,model="m")==[[1,2]]
    embed.assert_called_once_with(["x"])
    assert cache.read_text().startswith(original)

@pytest.mark.parametrize("field,value", [("provider","google"),("model","other"),("verbalization_version","other")])
def test_cache_provenance_partitions_requests(tmp_path, field, value):
    cache=tmp_path/"cache.jsonl"
    embeddings.cached_embeddings(["x"],cache,lambda _: [[1,2]],model="m")
    options={"model":"m",field:value}
    embed=Mock(return_value=[[3,4]])
    assert embeddings.cached_embeddings(["x"],cache,embed,**options)==[[3,4]]
    assert embed.call_count==1

def test_cache_dimension_mismatch_does_not_append_partial_entries(tmp_path):
    cache=tmp_path/"cache.jsonl"
    embeddings.cached_embeddings(["a"],cache,lambda _: [[1,2]],model="m")
    before=cache.read_bytes()
    with pytest.raises(ValueError,match="dimension"):
        embeddings.cached_embeddings(["a","b"],cache,lambda _: [[3]],model="m")
    assert cache.read_bytes()==before

@pytest.mark.parametrize("values", [[float("inf")],[True],["1"],[]])
def test_google_vectors_are_validated_before_return(monkeypatch, values):
    monkeypatch.setattr(google_vertex,"_post_vertex",lambda *a,**k: {"predictions":[{"embeddings":{"values":values}}]})
    with pytest.raises(ValueError):
        embeddings.EmbeddingClient(provider="google",model="text-embedding-005",google_project="test").embed(["x"])

def test_openai_response_indices_preserve_input_order(monkeypatch):
    import openai
    monkeypatch.setattr(openai,"OpenAI",lambda **_: NS(embeddings=NS(create=lambda **_: NS(
        data=[NS(index=1,embedding=[2,0]),NS(index=0,embedding=[1,0])]))))
    assert embeddings.EmbeddingClient().embed(["a","b"])==[[1,0],[2,0]]

def test_validation_retries_are_independent_of_transport_budget(monkeypatch):
    from coding_fgf import llm
    monkeypatch.setenv("CODING_FGF_API_MAX_ATTEMPTS","1")
    monkeypatch.setenv("CODING_FGF_MODEL_OUTPUT_MAX_ATTEMPTS","3")
    monkeypatch.setattr(llm,"_sleep_before_retry",lambda *a: None)
    request=Mock(side_effect=[{}, {}, {"code":"def materialize(context): pass"}])
    monkeypatch.setattr(llm,"call_structured_json",request)
    assert llm.llm_codegen({"rules":{}})=="def materialize(context): pass"
    assert request.call_count==3

@pytest.mark.parametrize("vector",[[1.0],[float("nan"),0.0]])
def test_retrieval_rejects_bad_queries_before_ranking(tmp_path,vector):
    from coding_fgf.retrieval import build_index,retrieve_candidates
    build_index([{"id":"target","kind":"class","embedding":[1.0,0.0]}],tmp_path)
    with pytest.raises(ValueError):
        retrieve_candidates([{"id":"source","kind":"class","embedding":vector}],tmp_path)

def test_retrieval_rejects_corrupted_saved_vectors(tmp_path):
    from coding_fgf.retrieval import build_index,retrieve_candidates
    build_index([{"id":"target","kind":"class","embedding":[1.0,0.0]}],tmp_path)
    (tmp_path/"index_vectors.json").write_text("[[1.0], [2.0, 3.0]]")
    with pytest.raises(ValueError,match="Expected"):
        retrieve_candidates([{"id":"source","kind":"class","embedding":[1.0,0.0]}],tmp_path)
