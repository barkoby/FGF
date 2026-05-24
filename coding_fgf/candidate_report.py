from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .candidate_methods import (
    METHOD_BM25,
    METHOD_HYBRID_BM25_DENSE,
    METHOD_HYBRID_LEVENSHTEIN_DENSE,
    METHOD_LEVENSHTEIN,
    METHOD_OPENAI_SMALL,
)
from .io import ensure_dir


def write_report(
    output_dir: Path,
    metrics_by_scenario: Sequence[dict[str, Any]],
    metrics_overall: Sequence[dict[str, Any]],
    methods: Sequence[str],
    k_values: Sequence[int],
) -> None:
    max_k = max(k_values)
    recall_rows = _method_rows(metrics_overall, source_kind="all", k=max_k)
    best = _best_method(recall_rows, metrics_overall)
    lines = [
        "# Candidate Generation Evaluation",
        "",
        "Gold mappings are derived from official RODI qpair SQL/SPARQL pairs. This is suitable for retrieval diagnosis, but it is an approximation rather than a hand-authored correspondence file.",
        "",
        "## Best Method Overall",
        "",
        _method_sentence(best, max_k),
        "",
        "## Overall Metrics",
        "",
        _overall_table(metrics_overall, methods, k_values),
        "",
        "## Best Method Per Scenario",
        "",
        _best_per_scenario_table(metrics_by_scenario, metrics_overall, max_k),
        "",
        "## Best Method Per Source Kind",
        "",
        _best_per_kind_table(metrics_overall, max_k),
        "",
        "## Method Comparisons",
        "",
        *comparison_lines(metrics_overall, max_k),
        "",
        "## Difficult Scenarios",
        "",
        *_difficult_scenarios(metrics_by_scenario, max_k),
        "",
        "## Recommendation",
        "",
        _recommendation(best, metrics_overall, max_k),
    ]
    ensure_dir(output_dir)
    (output_dir / "report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_plots(
    output_dir: Path,
    metrics_by_scenario: Sequence[dict[str, Any]],
    metrics_overall: Sequence[dict[str, Any]],
    methods: Sequence[str],
    k_values: Sequence[int],
) -> None:
    plots_dir = ensure_dir(output_dir / "plots")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        (plots_dir / "README.md").write_text(f"Matplotlib plots were not generated: {exc}\n", encoding="utf-8")
        return

    _plot_recall_curves(plt, plots_dir, metrics_overall, methods, k_values)
    _plot_bar_metric(plt, plots_dir, metrics_overall, methods, "mrr", "MRR", "mrr_by_method.png")
    _plot_bar_metric(plt, plots_dir, metrics_overall, methods, "mean_rank_first_gold", "Mean Rank", "mean_rank_by_method.png")
    _plot_bar_metric(plt, plots_dir, metrics_overall, methods, "candidate_coverage", "Candidate Coverage", "candidate_coverage_by_method.png")
    _plot_bar_metric(plt, plots_dir, metrics_overall, methods, "zero_candidate_rate", "Zero-Candidate Rate", "zero_candidate_rate_by_method.png")
    _plot_scenario_heatmap(plt, plots_dir, metrics_by_scenario, methods)


def _method_rows(rows: Sequence[dict[str, Any]], source_kind: str, k: int) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("source_kind") == source_kind and int(row.get("k", 0)) == k]


def _best_method(rows: Sequence[dict[str, Any]], metrics_overall: Sequence[dict[str, Any]]) -> dict[str, Any]:
    recall_at_1 = {row["method"]: float(row.get("recall_at_k", 0.0)) for row in _method_rows(metrics_overall, "all", 1)}
    return max(
        rows,
        key=lambda row: (
            float(row.get("recall_at_k", 0.0)),
            float(row.get("mrr", 0.0)),
            recall_at_1.get(row.get("method"), 0.0),
        ),
        default={},
    )


def _method_sentence(row: dict[str, Any], k: int) -> str:
    if not row:
        return "No evaluated gold-mapped sources were available."
    return (
        f"`{row['method']}` is best overall by Recall@{k} "
        f"({float(row.get('recall_at_k', 0.0)):.3f}) with MRR {float(row.get('mrr', 0.0)):.3f}."
    )


def _overall_table(rows: Sequence[dict[str, Any]], methods: Sequence[str], k_values: Sequence[int]) -> str:
    by_method_k = {
        (row.get("method"), int(row.get("k", 0))): row
        for row in rows
        if row.get("source_kind") == "all"
    }
    header = ["Method", *[f"R@{k}" for k in k_values], "MRR"]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---", *["---:" for _ in k_values], "---:"]) + " |",
    ]
    for method in methods:
        metric_values = []
        for k in k_values:
            metric_values.append(f"{float(by_method_k.get((method, k), {}).get('recall_at_k', 0.0)):.3f}")
        mrr = float(by_method_k.get((method, max(k_values)), {}).get("mrr", 0.0))
        lines.append(f"| `{method}` | " + " | ".join(metric_values) + f" | {mrr:.3f} |")
    return "\n".join(lines)


def _best_per_scenario_table(
    scenario_rows: Sequence[dict[str, Any]],
    overall_rows: Sequence[dict[str, Any]],
    k: int,
) -> str:
    rows = [row for row in scenario_rows if row.get("source_kind") == "all" and int(row.get("k", 0)) == k]
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(str(row.get("scenario", "")), []).append(row)
    lines = ["| Scenario | Best method | Recall@{} | MRR |".format(k), "| --- | --- | ---: | ---: |"]
    for scenario, vals in sorted(by_scenario.items()):
        best = _best_method(vals, overall_rows)
        lines.append(f"| `{scenario}` | `{best.get('method', '')}` | {float(best.get('recall_at_k', 0.0)):.3f} | {float(best.get('mrr', 0.0)):.3f} |")
    return "\n".join(lines)


def _best_per_kind_table(rows: Sequence[dict[str, Any]], k: int) -> str:
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if int(row.get("k", 0)) == k and row.get("source_kind") != "all":
            by_kind.setdefault(str(row.get("source_kind", "")), []).append(row)
    lines = ["| Source kind | Best method | Recall@{} | MRR |".format(k), "| --- | --- | ---: | ---: |"]
    for kind, vals in sorted(by_kind.items()):
        best = max(vals, key=lambda row: (float(row.get("recall_at_k", 0.0)), float(row.get("mrr", 0.0))))
        lines.append(f"| `{kind}` | `{best.get('method', '')}` | {float(best.get('recall_at_k', 0.0)):.3f} | {float(best.get('mrr', 0.0)):.3f} |")
    return "\n".join(lines)


def comparison_lines(rows: Sequence[dict[str, Any]], k: int) -> list[str]:
    by_method = {row.get("method"): row for row in rows if row.get("source_kind") == "all" and int(row.get("k", 0)) == k}
    dense = by_method.get(METHOD_OPENAI_SMALL, {})
    bm25 = by_method.get(METHOD_BM25, {})
    levenshtein = by_method.get(METHOD_LEVENSHTEIN, {})
    hybrid_lev = by_method.get(METHOD_HYBRID_LEVENSHTEIN_DENSE, {})
    hybrid_bm25 = by_method.get(METHOD_HYBRID_BM25_DENSE, {})
    lines = [
        f"- Hybrid Levenshtein+dense vs dense-only: {_delta_sentence(hybrid_lev, dense, k)}",
        f"- Hybrid BM25+dense vs dense-only: {_delta_sentence(hybrid_bm25, dense, k)}",
        f"- BM25 vs normalized Levenshtein: {_delta_sentence(bm25, levenshtein, k)}",
        f"- Better hybrid: `{_better(hybrid_lev, hybrid_bm25)}`.",
    ]
    best = _best_method(list(by_method.values()), rows)
    bottleneck = "candidate generation appears to remain a bottleneck" if float(best.get("recall_at_k", 0.0)) < 0.95 else "downstream LLM match selection is more likely to be the remaining bottleneck"
    lines.append(f"- Bottleneck signal: {bottleneck} at Recall@{k}.")
    return lines


def _delta_sentence(left: dict[str, Any], right: dict[str, Any], k: int) -> str:
    if not left or not right:
        return "not available."
    delta_recall = float(left.get("recall_at_k", 0.0)) - float(right.get("recall_at_k", 0.0))
    delta_mrr = float(left.get("mrr", 0.0)) - float(right.get("mrr", 0.0))
    return f"Delta Recall@{k} {delta_recall:+.3f}, Delta MRR {delta_mrr:+.3f}."


def _better(left: dict[str, Any], right: dict[str, Any]) -> str:
    if not left:
        return str(right.get("method", "not available"))
    if not right:
        return str(left.get("method", "not available"))
    winner = max([left, right], key=lambda row: (float(row.get("recall_at_k", 0.0)), float(row.get("mrr", 0.0))))
    return str(winner.get("method", "not available"))


def _difficult_scenarios(rows: Sequence[dict[str, Any]], k: int) -> list[str]:
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("source_kind") == "all" and int(row.get("k", 0)) == k:
            by_scenario.setdefault(str(row.get("scenario", "")), []).append(row)
    difficult = []
    for scenario, vals in sorted(by_scenario.items()):
        best = max(vals, key=lambda row: (float(row.get("recall_at_k", 0.0)), float(row.get("mrr", 0.0))))
        if float(best.get("recall_at_k", 0.0)) < 0.95:
            difficult.append(f"- `{scenario}` remains difficult: best Recall@{k} is {float(best.get('recall_at_k', 0.0)):.3f} with `{best.get('method', '')}`.")
    return difficult or ["- No scenario falls below the 0.95 Recall@{} bottleneck threshold.".format(k)]


def _recommendation(best: dict[str, Any], rows: Sequence[dict[str, Any]], k: int) -> str:
    if not best:
        return "No recommendation is available because no metrics were produced."
    recall = float(best.get("recall_at_k", 0.0))
    if recall < 0.95:
        return f"Use `{best.get('method')}` for the final FGF pipeline, but treat retrieval as an active bottleneck because Recall@{k} is {recall:.3f}."
    return f"Use `{best.get('method')}` for the final FGF pipeline; retrieval coverage is high enough that downstream selection should receive the next optimization pass."


def _plot_recall_curves(plt: Any, plots_dir: Path, rows: Sequence[dict[str, Any]], methods: Sequence[str], k_values: Sequence[int]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method in methods:
        values = [
            float(next((row for row in rows if row.get("method") == method and row.get("source_kind") == "all" and int(row.get("k", 0)) == k), {}).get("recall_at_k", 0.0))
            for k in k_values
        ]
        ax.plot(k_values, values, marker="o", label=method)
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "recall_at_k_by_method.png", dpi=160)
    plt.close(fig)


def _plot_bar_metric(plt: Any, plots_dir: Path, rows: Sequence[dict[str, Any]], methods: Sequence[str], metric: str, title: str, filename: str) -> None:
    max_k = max(int(row.get("k", 0)) for row in rows) if rows else 20
    values = [
        float(next((row for row in rows if row.get("method") == method and row.get("source_kind") == "all" and int(row.get("k", 0)) == max_k), {}).get(metric) or 0.0)
        for method in methods
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(methods)), values)
    ax.set_xticks(range(len(methods)), methods, rotation=25, ha="right")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / filename, dpi=160)
    plt.close(fig)


def _plot_scenario_heatmap(plt: Any, plots_dir: Path, rows: Sequence[dict[str, Any]], methods: Sequence[str]) -> None:
    max_k = max((int(row.get("k", 0)) for row in rows), default=20)
    scenarios = sorted({str(row.get("scenario", "")) for row in rows if row.get("source_kind") == "all" and int(row.get("k", 0)) == max_k})
    if not scenarios:
        return
    matrix = []
    for scenario in scenarios:
        matrix.append(
            [
                float(
                    next(
                        (
                            row
                            for row in rows
                            if row.get("scenario") == scenario
                            and row.get("method") == method
                            and row.get("source_kind") == "all"
                            and int(row.get("k", 0)) == max_k
                        ),
                        {},
                    ).get("recall_at_k", 0.0)
                )
                for method in methods
            ]
        )
    fig, ax = plt.subplots(figsize=(9, max(4, len(scenarios) * 0.35)))
    image = ax.imshow(matrix, vmin=0, vmax=1, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(methods)), methods, rotation=25, ha="right")
    ax.set_yticks(range(len(scenarios)), scenarios)
    ax.set_title(f"Scenario Recall@{max_k}")
    fig.colorbar(image, ax=ax, label=f"Recall@{max_k}")
    fig.tight_layout()
    fig.savefig(plots_dir / "scenario_recall_heatmap.png", dpi=160)
    plt.close(fig)
