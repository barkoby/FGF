# From Ontology Alignments to Executable Code: Automating Data Integration with Fact Generating Functions

This repository contains the research implementation for the paper **"From Ontology Alignments to Executable Code: Automating Data Integration with Fact Generating Functions"**.

Fact Generating Functions (FGFs) are executable triple-conversion functions that transform facts expressed under source semantics into equivalent facts under target semantics. FGFs bridge ontology alignment and RDF materialization by making the transformation executable. This repository provides an end-to-end pipeline for transforming relational data into RDF under a target ontology and evaluating the generated graph with SQL--SPARQL query pairs when those benchmark queries are available.

## Overview

The pipeline prepares RODI scenarios, verbalizes relational schemas and ontologies, retrieves candidate ontology correspondences, selects matches with an LLM, generates and validates FOL-style mapping rules, selects the best FOL method with a gold-blind portfolio selector, generates executable Python FGF code, materializes RDF in a sandbox, and optionally evaluates generated RDF with SQL--SPARQL query-pair evaluation.

The default reproducible configuration uses the best-FOL-method selection setup. It runs a portfolio of FOL-generation methods and selects a candidate using internal materialization diagnostics before downstream code generation/evaluation.

## Repository Structure

```text
coding_fgf/      Core Python package and CLI
configs/         Reproducible provider/configuration YAML files
scripts/         Docker-oriented wrapper commands for runs and ablations
tests/           Unit and regression tests
docs/            High-level method documentation
prompts/         Prompt examples/templates for paper artifacts
Dockerfile       Docker image definition
docker-compose.yml  Optional Docker Compose workflow
.env.example     Environment-variable template with placeholders only
```

## Requirements

The intended installation path is Docker-first. The host machine only needs Docker. Python dependencies are installed inside the image.

## Data Setup

Download the RODI benchmark from:

<https://github.com/chrpin/rodi/tree/master>

Mount the local RODI checkout into the container at `/data`. A typical layout is:

```text
/local/path/to/rodi/
  data/
  queries/
  ...
/local/path/to/fgf-outputs/
```

Inside Docker, the pipeline expects:

```text
/data      RODI benchmark checkout
/outputs   generated run artifacts, RDF, diagnostics, and evaluation results
```

## Environment Variables

Copy `.env.example` to `.env` and fill only the credentials you need locally. Do not commit `.env`.

Important variables:

```text
OPENAI_API_KEY=
GOOGLE_APPLICATION_CREDENTIALS=
GOOGLE_CLOUD_PROJECT=
GOOGLE_CLOUD_LOCATION=global
RODI_DIR=/local/path/to/rodi
OUTPUT_DIR=/local/path/to/outputs
```

## Docker-Based Installation

Build the image:

```bash
docker build -t fgf-pipeline .
```

Run help:

```bash
docker run --rm fgf-pipeline python -m coding_fgf --help
```

With Compose:

```bash
docker compose build fgf-pipeline
```

## Default Configuration

The default release configuration is `configs/openai_best_fol_default.yaml`. It uses:

- LLM provider: `openai`
- LLM model: `gpt-5.4-nano`
- embedding provider: `openai`
- embedding model: `text-embedding-3-small`
- FOL portfolio: enabled
- FOL portfolio arms: `full9_default`, `stage2_hybrid`, `stage2c_round2_only`
- selector: `internal_materialization`
- candidate retrieval `k`: `16`
- code-generation self-consistency: `3`
- random seed: `coding-fgf-dev10-v1`

Selection is gold-blind: SQL/SPARQL answers, gold mappings, paper scores, and query-pair failures are not used to select the FOL candidate.

## Run the Full Pipeline

```bash
docker run --rm \
  --env-file .env \
  -v /local/path/to/rodi:/data:ro \
  -v /local/path/to/outputs:/outputs \
  fgf-pipeline \
  python scripts/run_pipeline.py --config configs/openai_best_fol_default.yaml --scenario cmt_renamed
```

Run all nine paper scenarios by editing the `scenarios` list in the config or passing a comma-separated override:

```bash
docker run --rm --env-file .env \
  -v /local/path/to/rodi:/data:ro \
  -v /local/path/to/outputs:/outputs \
  fgf-pipeline \
  python scripts/run_pipeline.py --config configs/openai_best_fol_default.yaml \
  --scenario cmt_renamed,conference_renamed,sigkdd_renamed,cmt_structured,conference_structured,sigkdd_structured,sigkdd_mixed,conference_nofks,cmt_denormalized
```

## Provider Selection

### GPT/OpenAI

```bash
docker run --rm --env-file .env \
  -v /local/path/to/rodi:/data:ro \
  -v /local/path/to/outputs:/outputs \
  fgf-pipeline \
  python scripts/run_pipeline.py --config configs/openai_best_fol_default.yaml --scenario cmt_renamed
```

### Gemini

```bash
docker run --rm --env-file .env \
  -v /local/path/to/rodi:/data:ro \
  -v /local/path/to/outputs:/outputs \
  -v /local/path/to/gcloud:/gcloud:ro \
  fgf-pipeline \
  python scripts/run_pipeline.py --config configs/gemini_default.yaml --scenario cmt_renamed
```

### Gemma 4 via Google Vertex

```bash
docker run --rm --env-file .env \
  -v /local/path/to/rodi:/data:ro \
  -v /local/path/to/outputs:/outputs \
  -v /local/path/to/gcloud:/gcloud:ro \
  fgf-pipeline \
  python scripts/run_pipeline.py --config configs/gemma4_default.yaml --scenario cmt_renamed
```

## Main Hyperparameters

The main reproducibility parameters live in `configs/*.yaml`:

- `llm_provider`, `llm_model`
- `embedding_provider`, `embedding_model`
- `fol_portfolio`, `fol_portfolio_arms`, `fol_portfolio_selector`
- `k`, `match_workers`, `codegen_self_consistency`
- `fraction`, `seed`, `scenarios`
- `rodi_root`, `work`
- provider-specific Google credentials/project/location fields

Change hyperparameters by editing a YAML config or passing extra CLI arguments after the wrapper command.

## Ablation Studies

Candidate generation ablation:

```bash
docker run --rm --env-file .env \
  -v /local/path/to/rodi:/data:ro \
  -v /local/path/to/outputs:/outputs \
  fgf-pipeline \
  python scripts/run_ablation_candidate_generation.py --config configs/openai_best_fol_default.yaml
```

Matching ablation:

```bash
docker run --rm --env-file .env \
  -v /local/path/to/rodi:/data:ro \
  -v /local/path/to/outputs:/outputs \
  fgf-pipeline \
  python scripts/run_ablation_matching.py --config configs/openai_best_fol_default.yaml
```

FOL-method selection ablation:

```bash
docker run --rm --env-file .env \
  -v /local/path/to/rodi:/data:ro \
  -v /local/path/to/outputs:/outputs \
  fgf-pipeline \
  python scripts/run_ablation_fol_selection.py --config configs/openai_best_fol_default.yaml --scenario cmt_renamed
```

Failure analysis of missed SQL answers:

```bash
docker run --rm \
  -v /local/path/to/outputs:/outputs \
  fgf-pipeline \
  python scripts/run_failure_analysis.py --metrics /outputs/per_qpair_metrics.csv --scenario cmt_renamed
```

## Tests

Run the test suite inside Docker:

```bash
docker run --rm fgf-pipeline pytest
```

Live API tests are skipped automatically unless credentials are available in the environment.


## Reproducibility Notes

All final qpair scores should be computed only after generated Python FGF code materializes RDF. The portfolio selector uses internal diagnostics only and does not inspect SQL/SPARQL answers, query-pair failures, paper baselines, or gold mappings before output selection.
