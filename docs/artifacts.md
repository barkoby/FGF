# Run artifacts

For the default work directory `/outputs/openai_best_fol_default`:

- `devset/`: prepared inputs and row counts.
- `runs/<scenario>/source_records.jsonl` and `target_records.jsonl`: verbalized inputs.
- `runs/<scenario>/candidates.jsonl`, `matches.json`, and `fol.json`: decisions.
- `runs/<scenario>/generated_fgf.py`: selected materializer.
- `runs/<scenario>/import.ttl` and `import.materialization_log.json`: RDF and diagnostics.
- `runs/<scenario>/eval/metrics_details.json`: query-level answers and scores.
- `runs/<scenario>/eval/summary.csv`: scenario averages.
- `runs/<scenario>/fol_portfolio_selection_report.json`: internal selection evidence.
- `runs/<scenario>/fol_ablation/fol_selection_comparison.csv`: optional frozen-arm comparison.
- `comparison/`: stored reference-table comparisons.
- `run_metadata.json`: configuration and scenario status.

Ablations use `ablations/<ablation-name>` beneath the config's work directory. FOL ablations retain the same nested `runs/<scenario>` structure.

Use the `evaluate` subcommand with saved RDF and the corresponding scenario database to reevaluate a graph without generating code again. Failure analysis reads evaluation JSON without provider calls. Generated artifacts may contain source values and prompts.
