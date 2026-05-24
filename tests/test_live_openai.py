from __future__ import annotations

import os
from pathlib import Path

import pytest

from coding_fgf.candidate_methods import OpenAIEmbeddingCache
from coding_fgf.embeddings import EmbeddingClient
from coding_fgf.llm import llm_codegen, llm_match


pytestmark = pytest.mark.skipif(os.getenv("RUN_LIVE_OPENAI") != "1", reason="set RUN_LIVE_OPENAI=1 for live OpenAI smoke tests")


def test_live_embedding_smoke() -> None:
    vectors = EmbeddingClient(offline=False).embed(["class: Person with email"])
    assert len(vectors) == 1
    assert len(vectors[0]) > 100


def test_live_candidate_embedding_cache_smoke(tmp_path: Path) -> None:
    vectors = OpenAIEmbeddingCache(tmp_path / "candidate_embeddings.jsonl", model="text-embedding-3-small").embed_texts(
        ["class: Person with email"]
    )
    assert len(vectors) == 1
    assert len(vectors[0]) > 100


def test_live_matching_fol_and_codegen_smoke() -> None:
    candidates = [
        {
            "source": {
                "id": "source-class:people",
                "kind": "class",
                "uri": "urn:source#people",
                "text": "table people with names and emails",
            },
            "candidates": [
                {
                    "id": "class:http://ex#Person",
                    "kind": "class",
                    "uri": "http://ex#Person",
                    "text": "class Person",
                }
            ],
        }
    ]
    matches = llm_match(candidates, offline=False)
    assert matches
    fol = {"rules": {"class": [{"source_table": "people", "target_class": "http://ex#Person", "id_columns": ["id"]}], "data": [], "object": []}}
    code = llm_codegen(fol, offline=False)
    assert "def materialize" in code
