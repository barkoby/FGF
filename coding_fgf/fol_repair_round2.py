from __future__ import annotations

import copy
from collections import Counter
from typing import Any


def issue_type_counts(issues: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(issue.get("issue", "unknown")) for issue in issues))


def parse_rule_id(rule_id: str) -> tuple[str, int] | None:
    if ":" not in rule_id:
        return None
    kind, raw_index = rule_id.split(":", 1)
    if kind not in {"class", "data", "object"} or not raw_index.isdigit():
        return None
    return kind, int(raw_index)


def infer_rule_kind(rule: dict[str, Any], fallback: str = "") -> str:
    kind = str(rule.get("kind", "") or fallback)
    if kind in {"class", "data", "object"}:
        return kind
    if rule.get("target_class"):
        return "class"
    if rule.get("target_property") and rule.get("source_column"):
        return "data"
    if rule.get("target_property"):
        return "object"
    return ""


def clean_rule(rule: dict[str, Any]) -> dict[str, Any]:
    out = dict(rule)
    out.pop("kind", None)
    return out


def apply_round2_repairs(
    fol: dict[str, Any],
    repairs: list[dict[str, Any]],
    allow_drop: bool = True,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Apply action-list repairs without consulting evaluation answers."""
    rules = copy.deepcopy(fol.get("rules", {}) or {})
    for kind in ("class", "data", "object"):
        rules.setdefault(kind, [])

    replacements: dict[tuple[str, int], dict[str, Any]] = {}
    drops: dict[str, set[int]] = {"class": set(), "data": set(), "object": set()}
    additions: dict[str, list[dict[str, Any]]] = {"class": [], "data": [], "object": []}
    counts: Counter[str] = Counter()

    for repair in repairs:
        action = str(repair.get("action", ""))
        counts[action] += 1
        rule_id = str(repair.get("rule_id") or repair.get("issue_id") or "")
        parsed = parse_rule_id(rule_id)
        raw_rule = repair.get("rule")
        rule = clean_rule(raw_rule) if isinstance(raw_rule, dict) else None

        if action == "drop" and allow_drop and parsed:
            kind, index = parsed
            drops[kind].add(index)
        elif action == "repair" and rule:
            if parsed:
                kind, index = parsed
                replacements[(kind, index)] = rule
            else:
                kind = infer_rule_kind(rule)
                if kind:
                    additions[kind].append(rule)
        elif action == "add_class_rule" and rule:
            additions["class"].append(rule)
        elif action == "add_data_rule" and rule:
            additions["data"].append(rule)
        elif action in {"keep_with_justification", "explain_no_rule_needed"}:
            continue

    for (kind, index), rule in replacements.items():
        if 0 <= index < len(rules.get(kind, [])):
            rules[kind][index] = rule
    for kind, indexes in drops.items():
        if indexes:
            rules[kind] = [rule for index, rule in enumerate(rules.get(kind, [])) if index not in indexes]
    for kind, new_rules in additions.items():
        rules[kind].extend(new_rules)

    return {"rules": {kind: list(rules.get(kind, []) or []) for kind in ("class", "data", "object")}}, dict(counts)


def fol_repair_round2_summary(
    issues_before: list[dict[str, Any]],
    issues_after_round1: list[dict[str, Any]],
    issues_after_round2: list[dict[str, Any]],
    action_counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "fol_issues_before": len(issues_before),
        "fol_issues_after_round1": len(issues_after_round1),
        "fol_issues_after_round2": len(issues_after_round2),
        "issues_by_type_before": issue_type_counts(issues_before),
        "issues_by_type_after_round1": issue_type_counts(issues_after_round1),
        "issues_by_type_after_round2": issue_type_counts(issues_after_round2),
        "rules_added": int(action_counts.get("add_class_rule", 0)) + int(action_counts.get("add_data_rule", 0)),
        "rules_repaired": int(action_counts.get("repair", 0)),
        "rules_dropped": int(action_counts.get("drop", 0)),
        "rules_kept_with_justification": int(action_counts.get("keep_with_justification", 0)),
        "explain_no_rule_needed": int(action_counts.get("explain_no_rule_needed", 0)),
    }
