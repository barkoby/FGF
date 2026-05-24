from __future__ import annotations

from pathlib import Path

from coding_fgf.candidate_gold import (
    GoldIssue,
    GoldMapping,
    load_scenario_data,
    normalize_uri,
)
from coding_fgf.candidate_methods import (
    Bm25Index,
    OpenAIEmbeddingCache,
    RankedCandidate,
    embedding_cache_key,
    normalized_candidate_artifact,
    normalized_levenshtein_similarity,
    reciprocal_rank_fusion,
)
from coding_fgf.candidate_metrics import (
    assign_failure_category,
    evaluate_mapping,
    failure_row_from_issue,
    first_correct_rank,
)


K_VALUES = [1, 3, 5, 8, 10, 16, 20]


def write_candidate_fixture(root: Path) -> Path:
    scenario = root / "mini"
    queries = scenario / "queries"
    queries.mkdir(parents=True)
    (scenario / "dump.sql").write_text(
        "\n".join(
            [
                "CREATE TABLE people (",
                "  id integer NOT NULL,",
                "  email text",
                ");",
                "CREATE TABLE papers (",
                "  id integer NOT NULL,",
                "  author_id integer,",
                "  title text",
                ");",
                "ALTER TABLE ONLY people ADD CONSTRAINT people_pkey PRIMARY KEY (id);",
                "ALTER TABLE ONLY papers ADD CONSTRAINT papers_pkey PRIMARY KEY (id);",
                "ALTER TABLE ONLY papers ADD CONSTRAINT papers_author FOREIGN KEY (author_id) REFERENCES people(id);",
                "COPY people (id, email) FROM stdin;",
                "1\tada@example.org",
                r"\.",
                "COPY papers (id, author_id, title) FROM stdin;",
                "10\t1\tEngines",
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
                ":Person rdf:type owl:Class ; rdfs:label \"Person\" .",
                ":Paper rdf:type owl:Class ; rdfs:label \"Paper\" .",
                ":email rdf:type owl:DatatypeProperty ; rdfs:domain :Person .",
                ":title rdf:type owl:DatatypeProperty ; rdfs:domain :Paper .",
                ":hasAuthor rdf:type owl:ObjectProperty ; rdfs:domain :Paper ; rdfs:range :Person .",
            ]
        ),
        encoding="utf-8",
    )
    (queries / "Q01.qpair").write_text(
        "name=People\n"
        "sql=SELECT COUNT(*) FROM people\n"
        "sparql=PREFIX : <http://ex#> SELECT (COUNT(?x) AS ?count) WHERE { ?x a :Person }\n"
        "categories=class\n",
        encoding="utf-8",
    )
    (queries / "Q02.qpair").write_text(
        "name=Email\n"
        "sql=SELECT email FROM people\n"
        "sparql=PREFIX : <http://ex#> SELECT ?email WHERE { ?p :email ?email }\n"
        "categories=attrib\n",
        encoding="utf-8",
    )
    (queries / "Q03.qpair").write_text(
        "name=Authors\n"
        "sql=SELECT papers.author_id FROM papers JOIN people ON papers.author_id = people.id\n"
        "sparql=PREFIX : <http://ex#> SELECT ?paper ?person WHERE { ?paper :hasAuthor ?person }\n"
        "categories=ref\n",
        encoding="utf-8",
    )
    (queries / "Q04.qpair").write_text(
        "name=No source\n"
        "sql=SELECT COUNT(*)\n"
        "sparql=PREFIX : <http://ex#> SELECT (COUNT(?x) AS ?count) WHERE { ?x a :Paper }\n"
        "categories=class\n",
        encoding="utf-8",
    )
    return scenario


def test_gold_loader_aligns_qpair_sources_and_targets(tmp_path: Path) -> None:
    write_candidate_fixture(tmp_path)
    data = load_scenario_data(tmp_path, "mini")
    by_source = {mapping.source_id: set(mapping.gold_target_uris) for mapping in data.gold_mappings}
    assert "http://ex#Person" in by_source["source-class:people"]
    assert "http://ex#email" in by_source["source-data:people.email"]
    assert "http://ex#hasAuthor" in by_source["source-object:papers.author_id"]
    assert any(issue.failure_category == "source_not_in_gold" and issue.target_uri == "http://ex#Paper" for issue in data.gold_issues)


def test_uri_normalization_expands_prefixes_and_wrappers() -> None:
    assert normalize_uri("<http://ex#Person>") == "http://ex#Person"
    assert normalize_uri(":Person", {"": "http://ex#"}) == "http://ex#Person"
    assert normalize_uri("URIRef('http://ex#Person')") == "http://ex#Person"


def test_recall_and_mrr_for_required_k_values() -> None:
    mapping = GoldMapping("s", "source-data:t.c", "data_property", "t", "c", "entity_table", "attribute", ("u5",))
    candidates = [RankedCandidate(i, f"u{i}", {"uri": f"u{i}", "kind": "data_property"}, 1.0 / i) for i in range(1, 7)]
    unit, _ = evaluate_mapping(mapping, "mock", candidates[:5], {candidate.uri: candidate.rank for candidate in candidates}, K_VALUES)
    assert [unit[f"found_at_{k}"] for k in K_VALUES] == [0, 0, 1, 1, 1, 1, 1]
    assert unit["rank_of_first_gold"] == 5
    assert unit["reciprocal_rank"] == 0.2


def test_multi_gold_uses_first_retrieved_target() -> None:
    mapping = GoldMapping("s", "source-class:t", "class", "t", "", "entity_table", "", ("missing", "u2"))
    candidates = [
        RankedCandidate(1, "u1", {"uri": "u1", "kind": "class"}, 0.9),
        RankedCandidate(2, "u2", {"uri": "u2", "kind": "class"}, 0.8),
    ]
    unit, per_gold = evaluate_mapping(mapping, "mock", candidates, {"u1": 1, "u2": 2}, K_VALUES)
    assert unit["rank_of_first_gold"] == 2
    assert unit["found_at_1"] == 0
    assert unit["found_at_3"] == 1
    assert {row["gold_target_uri"] for row in per_gold} == {"missing", "u2"}


def test_no_match_and_null_source_failure_rows() -> None:
    assert first_correct_rank(["u"], {}, max_rank=20) is None
    issue = GoldIssue("s", "Q1", "u", "class", "source_not_in_gold", "no source")
    row = failure_row_from_issue(issue, "levenshtein")
    assert row["source_id"] == ""
    assert row["failure_category"] == "source_not_in_gold"


def test_normalized_levenshtein_ranking_prefers_closest_name() -> None:
    email = normalized_levenshtein_similarity("people email", "email")
    title = normalized_levenshtein_similarity("people email", "paper title")
    assert email > title


def test_bm25_index_ranking_prefers_token_match() -> None:
    records = [
        {"uri": "email", "text": "electronic mail email address"},
        {"uri": "title", "text": "paper title heading"},
    ]
    ranked = Bm25Index(records).rank("email address")
    assert ranked[0][0]["uri"] == "email"


def test_reciprocal_rank_fusion_combines_rankings() -> None:
    a = {"uri": "a", "kind": "class"}
    b = {"uri": "b", "kind": "class"}
    fused = reciprocal_rank_fusion(
        [
            [RankedCandidate(1, "a", a, 1.0), RankedCandidate(2, "b", b, 0.5)],
            [RankedCandidate(1, "b", b, 1.0), RankedCandidate(2, "a", a, 0.5)],
        ],
        rrf_k=60,
    )
    assert {candidate.uri for candidate in fused[:2]} == {"a", "b"}
    assert fused[0].rank == 1


def test_candidate_artifact_normalization() -> None:
    source = {"id": "source-class:people", "kind": "class", "source_table": "people", "source_table_role": "entity_table"}
    candidate = RankedCandidate(1, "http://ex#Person", {"id": "class:http://ex#Person", "uri": "http://ex#Person", "kind": "class"}, 0.99)
    artifact = normalized_candidate_artifact("mini", "mock", source, [candidate])
    assert artifact["source_id"] == "source-class:people"
    assert artifact["candidates"][0]["rank"] == 1
    assert artifact["candidates"][0]["uri"] == "http://ex#Person"


def test_openai_embedding_cache_key_and_mocked_embedding(tmp_path: Path) -> None:
    key1 = embedding_cache_key("text-embedding-3-small", "hello")
    key2 = embedding_cache_key("text-embedding-3-small", "hello")
    assert key1 == key2
    calls = {"count": 0}

    def embedder(model: str, texts: list[str]) -> list[list[float]]:
        calls["count"] += 1
        return [[float(len(text)), 1.0] for text in texts]

    cache = OpenAIEmbeddingCache(tmp_path / "embeddings.jsonl", embedder=embedder)
    assert cache.embed_texts(["hello"]) == [[5.0, 1.0]]
    assert cache.embed_texts(["hello"]) == [[5.0, 1.0]]
    assert calls["count"] == 1


def test_failure_category_assignment() -> None:
    candidate = RankedCandidate(1, "wrong", {"uri": "wrong", "kind": "class"}, 0.9)
    assert assign_failure_category("openai_small", 0, None, [], ["gold"], {}, "class") == "zero_candidates"
    assert assign_failure_category("levenshtein", 1, 25, [candidate], ["gold"], {}, "class") == "correct_target_rank_too_low"
    assert assign_failure_category("bm25", 1, None, [candidate], ["gold"], {}, "class") == "bm25_tokenization_failure"
    assert assign_failure_category("openai_small", 1, None, [candidate], ["gold"], {}, "class") == "dense_semantic_failure"
    assert assign_failure_category("hybrid_bm25_dense", 1, None, [candidate], ["gold"], {}, "class") == "hybrid_fusion_failure"


def test_candidate_eval_modules_do_not_import_llm() -> None:
    package_dir = Path(__file__).resolve().parents[1] / "coding_fgf"
    for name in ["candidate_gold.py", "candidate_methods.py", "candidate_metrics.py", "candidate_report.py", "eval_candidates.py"]:
        text = (package_dir / name).read_text(encoding="utf-8")
        assert "from .llm" not in text
        assert "import llm" not in text
        assert "coding_fgf.llm" not in text
