# From Ontology Alignments to Executable Code: Automating Data Integration with Fact Generating Functions

Research implementation for **“From Ontology Alignments to Executable Code: Automating Data Integration with Fact Generating Functions.”**

Fact Generating Functions (FGFs) transform source facts into target RDF facts. The pipeline retrieves ontology candidates, selects matches, validates mapping rules, generates Python materializers, and evaluates their RDF using SQL–SPARQL query pairs.

## Supported workflow

Use Docker with Docker Compose on Linux, or Docker Desktop with Linux containers. The image supplies Python and dependencies. Generated code requires Linux process resource controls; run it inside the image.

- `coding_fgf/`: pipeline, providers, materialization, evaluation
- `configs/`: OpenAI, Gemini, and Gemma configurations
- `scripts/`: pipeline, ablation, failure-analysis, and smoke commands
- `tests/`: offline unit and regression tests
- `docs/`: method, evaluation, and artifact documentation

## Quick start

1. Obtain the [RODI benchmark](https://github.com/chrpin/rodi).
2. Copy `.env.example` to `.env`. Set `RODI_DIR` to the host directory containing the benchmark's `data/` directory, and `OUTPUT_DIR` to a host output directory. Supply `OPENAI_API_KEY` for the default live configuration.
3. Build and run:

```bash
docker compose build fgf-pipeline
docker compose run --rm fgf-pipeline \
  python scripts/run_pipeline.py \
  --config configs/openai_best_fol_default.yaml --scenario cmt_renamed
```

Compose starts PostgreSQL and waits for its health check. The application connects to `postgres:5432`. A standalone application container needs an explicitly configured database connection.

The loader **drops and recreates databases named after the selected scenarios**. Use the dedicated Compose PostgreSQL service, not a database server holding other work. PostgreSQL is not published on a host port.

The benchmark is mounted read-only at `/data`; the host output directory is mounted at `/outputs`. Default outputs appear under `/outputs/openai_best_fol_default`. See [artifact locations](docs/artifacts.md).

For help and a configuration-only check:

```bash
docker compose run --rm --no-deps fgf-pipeline python -m coding_fgf --help
docker compose run --rm --no-deps fgf-pipeline python scripts/run_smoke_test.py
```

## Providers and settings

- OpenAI: `configs/openai_best_fol_default.yaml`
- Gemini: `configs/gemini_default.yaml`
- Vertex Gemma: `configs/gemma4_default.yaml`

For Google, set `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` in `.env`. Set `GOOGLE_GCLOUD_CONFIG_DIR` to the **host** directory containing Google Application Default Credentials. Set `GOOGLE_APPLICATION_CREDENTIALS` to the **container** path, normally `/gcloud/application_default_credentials.json`. No personal project is assumed.

```bash
docker compose run --rm fgf-pipeline python scripts/run_pipeline.py \
  --config configs/gemini_default.yaml --scenario cmt_renamed
```

For Gemma, substitute its config. Provider/model access must be available in your account.

YAML controls models, candidate count `k`, workers, code-generation candidates, scenarios, and FOL features. Extra wrapper arguments override generated CLI arguments. Live runs never substitute deterministic embeddings. Explicit `--offline` mode is for fixture tests and demonstrations.

## Ablations

All three configurations support candidate generation, matching, and FOL selection:

```bash
docker compose run --rm fgf-pipeline python scripts/run_ablation_candidate_generation.py \
  --config configs/openai_best_fol_default.yaml
docker compose run --rm fgf-pipeline python scripts/run_ablation_matching.py \
  --config configs/openai_best_fol_default.yaml
docker compose run --rm fgf-pipeline python scripts/run_ablation_fol_selection.py \
  --config configs/openai_best_fol_default.yaml
```

Substitute the Gemini or Gemma config as needed. Run candidate generation before matching. Outputs are separated beneath the config's work directory in `ablations/candidate_generation`, `ablations/matching`, and `ablations/fol_selection`. Matching locates the corresponding candidate artifact automatically. Explicit `--candidate-artifact` and `--output-dir` override these defaults. `--dry-run` prints commands without provider calls.

`dense` is the provider-independent retrieval method; `openai_small` is an OpenAI-only legacy alias. Artifacts identify the embedding provider/model, and incompatible inputs are rejected. Keep `method_configs.json` beside legacy artifacts so their actual embedding model can be checked.

FOL selection generates each arm once and fixes the choice using internal diagnostics. Only afterward does the ablation evaluate each arm's saved RDF. The comparison retains failed arms and adds a `portfolio_selected` row with the selected arm's metrics. Evaluation cannot change selection. Later refinements have a separate final evaluation.

## Failure analysis

Read the pipeline's actual evaluation JSON:

```bash
docker compose run --rm --no-deps fgf-pipeline python scripts/run_failure_analysis.py \
  --metrics /outputs/openai_best_fol_default/runs/cmt_renamed/eval/metrics_details.json \
  --output /outputs/failure_modes_summary.csv
```

Successful queries are excluded. Labels are diagnostic heuristics, not proven causes. CSV input requires `precision`, `recall`, `sql_count`, `sparql_count`, and `categories` (or legacy `category`). Supply `scenario` in CSV rows or pass `--scenario`. For JSON outside its canonical scenario directory, pass `--scenario`.

## Failure and execution limits

Transient provider failures receive three total request attempts by default; permanent configuration errors fail immediately. `CODING_FGF_API_MAX_ATTEMPTS` must be positive. Malformed output has a separate bounded `CODING_FGF_MODEL_OUTPUT_MAX_ATTEMPTS` limit (default six). Matching has no implicit model fallback; an explicitly configured fallback stays within the selected provider and is recorded.

Embedding cache version 2 separates provider, model, live/offline mode, and verbalization version. Legacy entries remain on disk but are not reused automatically; the next live run may make new embedding requests.

Generated code is AST-validated and runs in a separate worker. Defaults are 120 seconds and 2048 MiB of worker address space, controlled by `CODING_FGF_SANDBOX_TIMEOUT_SECONDS` and `CODING_FGF_SANDBOX_MEMORY_MB`. Limit failures reject the candidate without partial RDF. These controls are not a general-purpose hostile-code isolation system.

## Tests

```bash
docker compose run --rm --no-deps fgf-pipeline python -m pytest -q -p no:cacheprovider
docker compose -f docker-compose.smoke.yml up --build --abort-on-container-exit --exit-code-from smoke
docker compose -f docker-compose.smoke.yml down -v
```

The smoke workflow uses a synthetic fixture and a separate disposable PostgreSQL service. It makes no live provider calls.

Live OpenAI tests require `RUN_LIVE_OPENAI=1`; the matching smoke requires `RUN_LIVE_LLM=1` and `OPENAI_API_KEY`. Credentials alone do not enable tests. CI runs offline tests and the container smoke without credentials.

See [method architecture](docs/CURRENT_ARCHITECTURE.md), [evaluation conventions](docs/evaluation.md), and [artifacts](docs/artifacts.md).
