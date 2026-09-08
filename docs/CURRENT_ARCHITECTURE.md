# Current FGF / RODI Architecture

This document describes the FGF pipeline for the RODI benchmark, including its OpenAI, Google Gemini, and Google Vertex Gemma configurations. It explains the processing stages, validation, and evaluation boundaries.

## Architectural Goal

The system evaluates whether an LLM-assisted mapping pipeline can transform a relational RODI scenario into a target RDF graph that supports the official benchmark queries. The final benchmark score is based on the RDF graph produced by executing generated Python FGF code. Intermediate artifacts such as candidate sets, matches, FOL rules, or prompts are logged and analyzed, but they are not used as final evaluation outputs.

The implementation is organized around a reproducible end-to-end workflow:

```text
RODI scenario
  -> devset preparation and SQL loading
  -> schema and ontology verbalization
  -> embedding-based candidate retrieval
  -> LLM match selection and validation
  -> LLM FOL rule generation and repair
  -> LLM Python FGF code generation
  -> sandboxed RDF materialization
  -> SQL-vs-SPARQL qpair evaluation
```

## Pipeline Overview

### 1. Scenario Preparation

The runner prepares one or more RODI scenarios from the source benchmark directory. It copies or samples the required scenario files, including the SQL dump, ontology files, and query pairs. The SQL dump is loaded into PostgreSQL for query-pair evaluation and for extracting schema, rows, keys, foreign keys, and sample values.

### 2. Verbalization

The pipeline converts relational and ontology structures into textual records used by retrieval and prompting. Source records represent SQL tables, columns, foreign-key relations, discriminator-like structures, and optional source-context variants. Target records represent ontology classes, data properties, and object properties. Downstream stages share the same source and target record identifiers and context fields.

### 3. Candidate Generation

Candidate generation embeds source and target verbalizations with the selected embedding provider. The current architecture supports both OpenAI and Google Vertex embedding models. Embeddings are cached by provider, model, and text-derived keys. Pipeline retrieval ranks squared L2 distance with a lexical-overlap bonus, filters by source kind, and skips class retrieval for join tables. The default k is 16. Candidate ablations separately compare dense cosine similarity, lexical methods, and reciprocal-rank fusion.

Candidate outputs are saved as diagnostic artifacts. Candidate metrics can be evaluated separately when gold target alignments are available, but final pipeline scoring is still based on generated RDF and query-pair evaluation.

### 4. Matching

The matching stage asks the configured LLM to select the best target candidate, or no match, for each source element. The prompt receives source context, target candidate context, source kind, target kind, schema evidence, and table/column information. The selected match must refer to a supplied candidate URI rather than inventing a target.

After the initial LLM decision, the pipeline applies generic validation. Validation checks include candidate membership, source/target kind compatibility, suspicious identifier mappings, foreign-key datatype mismatches, and domain/range consistency where available. When enabled by the current pipeline mode, suspicious LLM outputs may be re-asked using the same provider and model. Invalid live outputs are recorded rather than silently converted into deterministic target substitutions.

### 5. FOL Rule Generation

Accepted matches are converted into FOL-style mapping rules using the configured LLM. The rules describe how source tables, columns, values, and relationships should produce target RDF classes, datatype assertions, and object-property assertions. Rule records keep provenance back to the originating match identifiers.

The FOL stage validates generated rules against schema and ontology evidence. It checks table and column references, target URI provenance, rule kind, row filters, foreign-key direction, and object-link plans where available. A repair prompt may be used to correct invalid rules, but repair decisions are based only on internal schema, match, FOL, and diagnostic evidence. The system does not use qpair failures, SQL answers, SPARQL answers, or paper baseline scores to accept or reject rules.

### 6. Python FGF Code Generation

The code-generation stage asks the configured LLM to produce executable Python code that implements the validated FOL rules. The generated module defines a materialization function that receives a controlled runtime context and emits RDF-compatible triples. The code is validated before execution for syntax and sandbox constraints.

The current implementation can generate multiple code candidates and select among them using internal runtime evidence, such as execution success, rule-kind coverage, helper usage, invalid triple count, and emitted triple count. One repair attempt may be made when code fails validation or execution diagnostics indicate a recoverable internal problem. The repair prompt is constrained to the supplied FOL rules and diagnostics; it must not invent mappings or target ontology terms.

### 7. RDF Materialization

Materialization executes the generated Python FGF code in the benchmark container. The generated code reads source data through the prepared runtime context and emits target RDF triples through approved helpers. The output graph is serialized to Turtle and related RDF artifacts. The materialization log records status, generated triple counts, invalid triple counts, runtime diagnostics, and any repair or candidate-selection decisions.

This stage is mandatory: final benchmark results are valid only when they come from RDF generated by executing the generated code.

### 8. Query-Pair Evaluation

The evaluation stage uses RODI SQL/SPARQL query pairs with the comparison conventions documented in evaluation.md. For each query pair, the SQL query is run against the original relational database, and the SPARQL query is run against the generated RDF graph. Precision, recall, and F1 are computed from the result-set comparison. Scenario-level and comparison reports are generated from these query-pair scores.

## Model Stack Architecture

The pipeline separates the LLM provider from the embedding provider. Both are selected through CLI arguments or environment-backed configuration. This keeps the same method logic available across several model stacks.

### OpenAI / GPT Suite

The OpenAI suite uses the OpenAI provider for generation and OpenAI embeddings for retrieval. In the current experiments, this stack is represented by GPT-family LLMs for matching, FOL generation, and code generation, paired with `text-embedding-3-small` for candidate generation. OpenAI API access is configured through the standard OpenAI environment variables.

### Google Gemini Suite

The Gemini suite uses Google Vertex AI as the provider path. Generation calls are sent to `gemini-3.1-flash-lite`, and candidate embeddings use `text-embedding-005`. Authentication uses Google Application Default Credentials, with project and location supplied through configuration. The project must be configured by the user; the default location is global.

### Google Vertex Gemma Suite

The Gemma suite also uses the Google Vertex provider path. It selects Gemma model identifiers for generation while keeping the Google embedding path for candidate generation. From the pipeline perspective, Gemma is another configured Google generation model: the downstream matching, FOL, code-generation, materialization, and evaluation stages are unchanged.

## Provider Layer

The provider layer exposes a common structured-generation interface to the pipeline. OpenAI calls use the OpenAI client path, while Google calls use Vertex `generateContent` for JSON-oriented LLM outputs and Vertex prediction endpoints for embeddings. Provider calls include retry and parsing logic appropriate to the live mode being used.

The embedding layer similarly abstracts over OpenAI and Google embeddings. It stores provider and model metadata with cached vectors so that artifacts remain tied to the model stack that produced them.

## Academic Safeguards

The implementation is designed to separate method execution from evaluation feedback.

- Final reported scores come only from the query-pair evaluation of materialized RDF.
- Matching, FOL repair, code repair, and output selection do not use SQL answers, SPARQL answers, qpair failures, or paper baseline scores.
- Dataset-specific target URI allowlists, hand-written mappings, and expected-output shortcuts are excluded from the method.
- Live strict runs use configured LLM and embedding providers rather than silent deterministic substitutes.
- Deterministic or offline behavior is reserved for tests, smoke runs, or explicit offline modes.
- Every accepted mapping artifact is traceable to source/schema evidence, retrieved candidates, LLM outputs, and internal validation diagnostics.

## Main Artifacts

Each scenario run produces a structured set of artifacts:

- verbalized source and target records;
- embedding caches and candidate retrieval outputs;
- LLM-selected matches and validation reports;
- FOL rules, validation reports, and repair diagnostics;
- generated Python FGF code and candidate-code scores;
- materialized RDF output and materialization logs;
- SQL-vs-SPARQL evaluation summaries;
- comparison reports against benchmark reference tables;
- diagnostic files for hard cases, suspicious mappings, and stage-level failures.

These artifacts make it possible to analyze candidate retrieval, matching, rule quality, code generation, and RDF materialization separately while preserving one final end-to-end evaluation path.

## Solution Summary

The current solution is an executable LLM-assisted FGF pipeline for relational-to-RDF mapping. It uses embeddings to propose ontology candidates, LLMs to choose matches and synthesize rules, LLM-generated Python to implement the mapping, and an RDF materialization step to create the final graph. The same pipeline logic is used across OpenAI/GPT, Google Gemini, and Google Vertex Gemma configurations by swapping provider and model parameters rather than changing the benchmark method.


## FOL portfolio ordering

Each arm generates FOL, generates Python code, and materializes RDF before selection. The selector uses internal diagnostics and persists its decision. Query-pair evaluation, including the optional arm comparison, happens afterward.
