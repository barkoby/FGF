from __future__ import annotations

import json

import pytest

from coding_fgf import google_vertex
from coding_fgf.cli import build_parser
from coding_fgf.embeddings import embed_records


def test_google_generate_response_text_and_json_parsing() -> None:
    response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": '```json\n{"matches":[{"source_id":"s","target_uri":null}]}\n```'},
                    ]
                }
            }
        ]
    }
    text = google_vertex.extract_generate_text(response)
    parsed = google_vertex.parse_json_text(text)
    assert parsed["matches"][0]["source_id"] == "s"
    assert parsed["matches"][0]["target_uri"] is None


def test_google_embedding_response_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(google_vertex, "_access_token", lambda config: "token")

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"predictions": [{"embeddings": {"values": [0.1, 0.2, 0.3]}}]}

    monkeypatch.setattr(google_vertex.requests, "post", lambda *args, **kwargs: FakeResponse())
    vectors = google_vertex.embed_texts(["hello"], "text-embedding-005", google_vertex.GoogleVertexConfig("p"))
    assert vectors == [[0.1, 0.2, 0.3]]


def test_google_json_invalid_response_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODING_FGF_API_MAX_ATTEMPTS", "2")
    monkeypatch.setattr(google_vertex, "_sleep", lambda delay, jitter: None)
    responses = [
        {"candidates": [{"content": {"parts": [{"text": "not json"}]}}]},
        {"candidates": [{"content": {"parts": [{"text": json.dumps({"ok": True})}]}}]},
    ]
    monkeypatch.setattr(google_vertex, "_post_vertex", lambda *args, **kwargs: responses.pop(0))
    data = google_vertex.generate_json("prompt", "schema", "gemini-3.1-flash-lite", google_vertex.GoogleVertexConfig("p"))
    assert data == {"ok": True}


def test_google_json_invalid_response_exhausts_model_output_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODING_FGF_MODEL_OUTPUT_MAX_ATTEMPTS", "2")
    monkeypatch.setattr(google_vertex, "_sleep", lambda delay, jitter: None)
    monkeypatch.setattr(
        google_vertex,
        "_post_vertex",
        lambda *args, **kwargs: {"candidates": [{"content": {"parts": [{"text": "not json"}]}}]},
    )
    with pytest.raises(google_vertex.GoogleVertexError, match="model-output attempts"):
        google_vertex.generate_json("prompt", "schema", "gemini-3.1-flash-lite", google_vertex.GoogleVertexConfig("p"))


def test_provider_args_preserve_openai_defaults() -> None:
    args = build_parser().parse_args(["run-paper-compare", "--scenarios", "cmt_structured"])
    assert args.llm_provider == "openai"
    assert args.embedding_provider == "openai"
    assert args.llm_model == ""


def test_google_embedding_live_mode_does_not_use_deterministic_fallback(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        google_vertex,
        "embed_texts",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("vertex unavailable")),
    )
    records = [{"id": "x", "text": "hello"}]
    with pytest.raises(RuntimeError, match="vertex unavailable"):
        embed_records(records, tmp_path / "cache.jsonl", provider="google", model="text-embedding-005", offline=False, google_project="test-project")
