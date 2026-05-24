from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .constants import PAPER_TARGET_F1
from .io import ensure_dir, write_csv


@dataclass(frozen=True)
class PromotionDecision:
    scenario: str
    coding_f1: float | None
    llm4vkg_f1: float | None
    fallback_f1: float | None
    invalid_rules: int
    threshold: float
    promoted: bool
    reason: str


def read_summary_csv(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    row = rows[0]
    return {key: float(row[key]) for key in ("precision", "recall", "f1") if row.get(key) not in (None, "")}


def invalid_rule_count(work_dir: Path) -> int:
    matches_path = work_dir / "matches.json"
    fol_path = work_dir / "fol.json"
    if not matches_path.exists() or not fol_path.exists():
        return 0
    matches = json.loads(matches_path.read_text(encoding="utf-8")).get("matches", [])
    accepted = sum(1 for match in matches if match.get("target_uri"))
    fol = json.loads(fol_path.read_text(encoding="utf-8")).get("rules", {})
    rules = sum(len(fol.get(kind, [])) for kind in ("class", "data", "object"))
    return max(0, accepted - rules)


def no_match_rate(work_dir: Path) -> float | None:
    matches_path = work_dir / "matches.json"
    if not matches_path.exists():
        return None
    matches = json.loads(matches_path.read_text(encoding="utf-8")).get("matches", [])
    if not matches:
        return None
    return sum(1 for match in matches if not match.get("target_uri")) / len(matches)


def emitted_triple_count(graph_path: Path) -> int | None:
    if not graph_path.exists():
        return None
    count = 0
    for line in graph_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.endswith(".") or stripped.endswith(";"):
            count += 1
    return count


def decide_promotion(
    scenario: str,
    coding_f1: float | None,
    llm4vkg_f1: float | None,
    invalid_rules: int,
    similarity_margin: float = 0.05,
) -> PromotionDecision:
    fallback_f1 = PAPER_TARGET_F1.get(scenario)
    baseline = llm4vkg_f1 if llm4vkg_f1 is not None else fallback_f1
    if coding_f1 is None:
        return PromotionDecision(scenario, coding_f1, llm4vkg_f1, fallback_f1, invalid_rules, similarity_margin, False, "missing coding_fgf F1")
    if baseline is None:
        return PromotionDecision(scenario, coding_f1, llm4vkg_f1, fallback_f1, invalid_rules, similarity_margin, False, "missing baseline F1")
    if invalid_rules:
        return PromotionDecision(scenario, coding_f1, llm4vkg_f1, fallback_f1, invalid_rules, similarity_margin, False, "invalid accepted rules")
    promoted = coding_f1 >= baseline - similarity_margin
    reason = "within similarity margin" if promoted else "below similarity margin"
    return PromotionDecision(scenario, coding_f1, llm4vkg_f1, fallback_f1, invalid_rules, similarity_margin, promoted, reason)


def compare_runs(
    coding_root: Path,
    baseline_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    scenarios: Sequence[str],
    similarity_margin: float = 0.05,
) -> list[dict[str, Any]]:
    ensure_dir(output_dir)
    baseline_by_scenario = {str(row.get("scenario")): row for row in baseline_rows}
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        work_dir = coding_root / scenario
        summary = read_summary_csv(work_dir / "eval" / "summary.csv") or read_summary_csv(work_dir / "summary.csv") or {}
        coding_f1 = summary.get("f1")
        baseline_f1 = _optional_float(baseline_by_scenario.get(scenario, {}).get("f1"))
        invalid_rules = invalid_rule_count(work_dir)
        decision = decide_promotion(scenario, coding_f1, baseline_f1, invalid_rules, similarity_margin=similarity_margin)
        rows.append(
            {
                "scenario": scenario,
                "coding_f1": coding_f1,
                "llm4vkg_f1": baseline_f1,
                "paper_f1": PAPER_TARGET_F1.get(scenario),
                "delta_vs_llm4vkg": None if coding_f1 is None or baseline_f1 is None else coding_f1 - baseline_f1,
                "invalid_rules": invalid_rules,
                "no_match_rate": no_match_rate(work_dir),
                "emitted_triples": emitted_triple_count(work_dir / "import.ttl"),
                "promoted": decision.promoted,
                "reason": decision.reason,
            }
        )
    write_csv(
        output_dir / "comparison.csv",
        rows,
        [
            "scenario",
            "coding_f1",
            "llm4vkg_f1",
            "paper_f1",
            "delta_vs_llm4vkg",
            "invalid_rules",
            "no_match_rate",
            "emitted_triples",
            "promoted",
            "reason",
        ],
    )
    _write_markdown(output_dir / "comparison.md", rows)
    return rows


def compare_to_paper(coding_root: Path, output_dir: Path, scenarios: Sequence[str]) -> list[dict[str, Any]]:
    ensure_dir(output_dir)
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        work_dir = coding_root / scenario
        summary = read_summary_csv(work_dir / "eval" / "summary.csv") or read_summary_csv(work_dir / "summary.csv") or {}
        coding_f1 = summary.get("f1")
        paper_f1 = PAPER_TARGET_F1.get(scenario)
        rows.append(
            {
                "scenario": scenario,
                "precision": summary.get("precision"),
                "recall": summary.get("recall"),
                "f1": coding_f1,
                "paper_f1": paper_f1,
                "delta_vs_paper": None if coding_f1 is None or paper_f1 is None else coding_f1 - paper_f1,
                "beats_paper_by_0_01": bool(coding_f1 is not None and paper_f1 is not None and coding_f1 >= paper_f1 + 0.01),
                "invalid_rules": invalid_rule_count(work_dir),
                "no_match_rate": no_match_rate(work_dir),
                "emitted_triples": emitted_triple_count(work_dir / "import.ttl"),
            }
        )
    write_csv(
        output_dir / "paper_comparison.csv",
        rows,
        [
            "scenario",
            "precision",
            "recall",
            "f1",
            "paper_f1",
            "delta_vs_paper",
            "beats_paper_by_0_01",
            "invalid_rules",
            "no_match_rate",
            "emitted_triples",
        ],
    )
    _write_paper_markdown(output_dir / "paper_comparison.md", rows)
    return rows


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Dev10 Comparison",
        "",
        "| Scenario | coding_fgf F1 | LLM4VKG F1 | Paper F1 | Invalid Rules | Promoted | Reason |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {coding_f1} | {llm4vkg_f1} | {paper_f1} | {invalid_rules} | {promoted} | {reason} |".format(
                **{key: "" if value is None else value for key, value in row.items()}
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_paper_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Paper Comparison",
        "",
        "| Scenario | Precision | Recall | F1 | Paper F1 | Delta | Beats Paper +0.01 | Invalid Rules |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        values = {key: "" if value is None else value for key, value in row.items()}
        lines.append(
            "| {scenario} | {precision} | {recall} | {f1} | {paper_f1} | {delta_vs_paper} | {beats_paper_by_0_01} | {invalid_rules} |".format(
                **values
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
