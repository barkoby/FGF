from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..io import ensure_dir


HEATMAP_METHODS = ["current_validated", "cot_prompt", "self_consistency", "chain_of_verification"]


def write_heatmaps(output_dir: Path, metrics_by_dataset: Sequence[Mapping[str, Any]], methods: Sequence[str] = HEATMAP_METHODS) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heatmap_dir = ensure_dir(output_dir / "heatmaps")
    scenarios = sorted({str(row.get("scenario", "")) for row in metrics_by_dataset if row.get("scenario")})
    for metric, filename, title in (
        ("f1", "f1_by_dataset_method.png", "F1 by Dataset and Method"),
        ("precision", "precision_by_dataset_method.png", "Precision by Dataset and Method"),
        ("recall", "recall_by_dataset_method.png", "Recall by Dataset and Method"),
        ("error_rate", "error_rate_by_dataset_method.png", "Error Rate by Dataset and Method"),
    ):
        matrix = _matrix(metrics_by_dataset, scenarios, methods, metric)
        _draw_heatmap(plt, matrix, scenarios, list(methods), title, heatmap_dir / filename, value_format=".3f")

    dataset_metrics = ["precision", "recall", "f1", "invalid_selection_rate", "api_error_rate", "avg_candidate_rank"]
    enriched = [metric_row for row in metrics_by_dataset for metric_row in _with_rates(row)]
    for scenario in scenarios:
        rows = [row for row in enriched if row.get("scenario") == scenario]
        matrix = _matrix(rows, dataset_metrics, methods, "value", row_field="metric")
        _draw_heatmap(plt, matrix, dataset_metrics, list(methods), f"{scenario} Method Metrics", heatmap_dir / f"{scenario}_method_metrics.png", value_format=".3f")


def _with_rates(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    evaluated = _float(row.get("evaluated_sources"))
    return [
        {"scenario": row.get("scenario"), "method": row.get("method"), "metric": "precision", "value": _float(row.get("precision"))},
        {"scenario": row.get("scenario"), "method": row.get("method"), "metric": "recall", "value": _float(row.get("recall"))},
        {"scenario": row.get("scenario"), "method": row.get("method"), "metric": "f1", "value": _float(row.get("f1"))},
        {
            "scenario": row.get("scenario"),
            "method": row.get("method"),
            "metric": "invalid_selection_rate",
            "value": _float(row.get("invalid_selections")) / evaluated if evaluated else 0.0,
        },
        {
            "scenario": row.get("scenario"),
            "method": row.get("method"),
            "metric": "api_error_rate",
            "value": _float(row.get("api_errors")) / evaluated if evaluated else 0.0,
        },
        {"scenario": row.get("scenario"), "method": row.get("method"), "metric": "avg_candidate_rank", "value": _float(row.get("avg_candidate_rank"))},
    ]


def _matrix(
    rows: Sequence[Mapping[str, Any]] | Sequence[Sequence[Mapping[str, Any]]],
    row_labels: Sequence[str],
    methods: Sequence[str],
    metric: str,
    row_field: str = "scenario",
) -> list[list[float]]:
    flat_rows: list[Mapping[str, Any]] = []
    for row in rows:
        if isinstance(row, list):
            flat_rows.extend(row)
        else:
            flat_rows.append(row)
    by_key = {(str(row.get(row_field, "")), str(row.get("method", ""))): row for row in flat_rows}
    return [[_float(by_key.get((label, method), {}).get(metric)) for method in methods] for label in row_labels]


def _draw_heatmap(plt: Any, matrix: list[list[float]], row_labels: list[str], col_labels: list[str], title: str, path: Path, value_format: str = ".3f") -> None:
    width = max(7.0, 1.8 * len(col_labels))
    height = max(4.0, 0.45 * len(row_labels) + 1.8)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            color = "white" if value < 0.55 else "black"
            ax.text(j, i, format(value, value_format), ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _float(value: Any) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0
