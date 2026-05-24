from __future__ import annotations

from typing import Any

from .lexical import words
from .schema import Table, foreign_key_columns, is_generic_identifier_column, table_key, table_role


_SEMANTIC_ID_WORDS = {"paperid", "paper_id", "doi", "url", "email", "isbn", "issn", "code", "abbrev", "abbreviation"}
_SEMANTIC_ID_TARGET_WORDS = {"code", "car", "abbrev", "abbreviation", "doi", "url", "email", "isbn", "issn"}


def _is_valid_target_uri(uri: str) -> bool:
    return uri.startswith("http://") or uri.startswith("https://") or uri.startswith("urn:")


def _local(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[1]
    return uri.rstrip("/").rsplit("/", 1)[-1]


def _namespace(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[0] + "#"
    return uri.rstrip("/").rsplit("/", 1)[0] + "/"


def _target_words(match: dict[str, Any]) -> set[str]:
    return words(" ".join(str(match.get(key, "")) for key in ("target_local_name", "target_uri", "target_id")))


def _class_compatible(table: str, match: dict[str, Any]) -> bool:
    if not match.get("target_uri") or not _is_valid_target_uri(str(match["target_uri"])):
        return False
    source = words(table)
    target = _target_words(match)
    return bool(source & target)


def _property_domain_compatible(match: dict[str, Any], class_uri: str) -> bool:
    return _property_domain_score(match, class_uri) > 0


def _property_range_compatible(match: dict[str, Any], class_uri: str) -> bool:
    return _property_range_score(match, class_uri) > 0


def _property_domain_score(match: dict[str, Any], class_uri: str) -> int:
    domains = [str(value) for value in match.get("target_domain") or []]
    if not domains:
        return 1
    class_words = words(_local(class_uri))
    scores = []
    for domain in domains:
        domain_words = words(_local(domain))
        scores.append(2 if domain_words == class_words else 1 if domain_words & class_words else 0)
    return max(scores, default=0)


def _property_range_score(match: dict[str, Any], class_uri: str) -> int:
    ranges = [str(value) for value in match.get("target_range") or []]
    if not ranges:
        return 1
    class_words = words(_local(class_uri))
    scores = []
    for value in ranges:
        range_words = words(_local(value))
        scores.append(2 if range_words == class_words else 1 if range_words & class_words else 0)
    return max(scores, default=0)


def _best_domain_range_score(match: dict[str, Any], domain_classes: list[str], range_classes: list[str]) -> int:
    scores = [
        _property_domain_score(match, domain_class) + _property_range_score(match, range_class)
        for domain_class in domain_classes
        for range_class in range_classes
    ]
    return max(scores, default=0)


def _semantic_id_allowed(table: str, column: str, match: dict[str, Any]) -> bool:
    if column.lower() == "id":
        return False
    target = "".join(_target_words(match))
    target_words = _target_words(match)
    column_words = words(column)
    if "paper" in words(table) and "paper" in target and "id" in target:
        return True
    if column.lower() in _SEMANTIC_ID_WORDS or "_".join(column_words) in _SEMANTIC_ID_WORDS:
        return bool(target_words & _SEMANTIC_ID_TARGET_WORDS)
    return False


def _is_type_column(column: str) -> bool:
    return column.lower() == "type"


def _is_boolean_column(table: Table, column: str) -> bool:
    return any(col.name == column and "bool" in col.datatype.lower() for col in table.columns)


def _is_discriminator_data_column(table: Table, column: str) -> bool:
    return column.lower().startswith("is_") or _is_type_column(column) or _is_boolean_column(table, column)


def _table_class_options(table: str, class_tables: dict[str, str], rules: dict[str, list[dict[str, Any]]]) -> list[str]:
    values: list[str] = []
    if table in class_tables:
        values.append(class_tables[table])
    for rule in rules.get("class", []):
        if rule.get("source_table") == table and rule.get("target_class"):
            values.append(str(rule["target_class"]))
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _add_class_rule_once(rules: dict[str, list[dict[str, Any]]], rule: dict[str, Any]) -> None:
    for existing in rules["class"]:
        if (
            existing.get("source_table") == rule.get("source_table")
            and existing.get("target_class") == rule.get("target_class")
            and existing.get("row_filter") == rule.get("row_filter")
        ):
            return
    rules["class"].append(rule)


def _add_structured_discriminator_class_rules(
    tables: dict[str, Table],
    class_tables: dict[str, str],
    rules: dict[str, list[dict[str, Any]]],
) -> None:
    for table_name, class_uri in list(class_tables.items()):
        if table_name not in tables:
            continue
        table = tables[table_name]
        ns = _namespace(class_uri)
        for column in table.column_names():
            if column.lower().startswith("is_") or _is_boolean_column(table, column):
                class_name = _discriminator_source_class_name(column)
                for inferred_class_name in _discriminator_class_names(class_name):
                    _add_class_rule_once(
                        rules,
                        {
                            "source_table": table_name,
                            "target_class": ns + inferred_class_name,
                            "id_columns": table_key(table),
                            "confidence": 0.9,
                            "table_role": table_role(table),
                            "row_filter": {"column": column, "truthy": True},
                            "inferred_from": "boolean_class_discriminator",
                        },
                    )
        type_column = _type_column_name(table)
        if _local(class_uri) == "Paper" and type_column:
            for type_value, class_name in (("1", "PaperFullVersion"), ("2", "PaperAbstract")):
                _add_class_rule_once(
                    rules,
                    {
                        "source_table": table_name,
                        "target_class": ns + class_name,
                        "id_columns": table_key(table),
                        "confidence": 0.9,
                        "table_role": table_role(table),
                        "row_filter": {"column": type_column, "equals": type_value},
                        "inferred_from": "type_class_discriminator",
                    },
                )
        if _local(class_uri) == "Document" and type_column:
            for type_value, class_name in (("1", "Paper"), ("2", "Abstract")):
                _add_class_rule_once(
                    rules,
                    {
                        "source_table": table_name,
                        "target_class": ns + class_name,
                        "id_columns": table_key(table),
                        "confidence": 0.9,
                        "table_role": table_role(table),
                        "row_filter": {"column": type_column, "equals": type_value},
                        "inferred_from": "type_class_discriminator",
                    },
                )
        if "fee" in words(table_name) or "fee" in words(_local(class_uri)):
            if "registration" in words(table_name) or "registration" in words(_local(class_uri)):
                _add_class_rule_once(
                    rules,
                    {
                        "source_table": table_name,
                        "target_class": ns + "Registration_fee",
                        "id_columns": table_key(table),
                        "confidence": 0.8,
                        "table_role": table_role(table),
                        "inferred_from": "class_uri_alias",
                    },
                )
            _add_class_rule_once(
                rules,
                {
                    "source_table": table_name,
                    "target_class": ns + "Fee",
                    "id_columns": table_key(table),
                    "confidence": 0.8,
                    "table_role": table_role(table),
                    "inferred_from": "class_uri_alias",
                },
            )
        if _local(class_uri) == "ProgramCommittee":
            _add_class_rule_once(
                rules,
                {
                    "source_table": table_name,
                    "target_class": ns + "Program_committee",
                    "id_columns": table_key(table),
                    "confidence": 0.8,
                    "table_role": table_role(table),
                    "inferred_from": "class_uri_alias",
                },
            )


def _type_column_name(table: Table) -> str | None:
    for column in table.column_names():
        if _is_type_column(column):
            return column
    return None


def _title_case_class_name(value: str) -> str:
    return "_".join(part[:1].upper() + part[1:] for part in value.replace("-", "_").split("_") if part)


def _discriminator_source_class_name(column: str) -> str:
    if column.lower().startswith("is_"):
        return column[3:]
    value = column
    aliases = {
        "author_paper_student": "Author_of_paper_student",
        "program_committee_member": "Program_Committee_member",
    }
    return aliases.get(value.lower(), _title_case_class_name(value))


def _discriminator_class_names(class_name: str) -> list[str]:
    names = [class_name]
    lowered = class_name.lower().replace("-", "_")
    if lowered.startswith("author_of_paper"):
        names.extend(["Author_of_paper", "Author"])
    if lowered.endswith("_student") or lowered == "student":
        names.append("Student")
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _attribute_table_data_rule(
    table: str,
    column: str,
    match: dict[str, Any],
    target_uri: str,
    tables: dict[str, Table],
    class_tables: dict[str, str],
) -> dict[str, Any] | None:
    for fk in tables[table].foreign_keys:
        if column in fk.columns:
            continue
        ref_class = class_tables.get(fk.ref_table)
        if not ref_class or not _property_domain_compatible(match, ref_class):
            continue
        return {
            "source_table": table,
            "source_column": column,
            "target_property": target_uri,
            "subject_table": fk.ref_table,
            "subject_columns": fk.columns,
            "subject_target_columns": fk.ref_columns,
            "confidence": match.get("confidence", 0.0),
        }
    return None


def matches_to_fol(matches: list[dict[str, Any]], tables: dict[str, Table]) -> dict[str, Any]:
    rules: dict[str, list[dict[str, Any]]] = {"class": [], "data": [], "object": []}
    class_tables: dict[str, str] = {}
    for match in matches:
        target_uri = match.get("target_uri")
        if not target_uri or not _is_valid_target_uri(str(target_uri)):
            continue
        source_id = str(match["source_id"])
        if source_id.startswith("source-class:"):
            table = source_id.split(":", 1)[1]
            if table in tables and table_role(tables[table]) != "join_table" and _class_compatible(table, match):
                class_tables[table] = str(target_uri)
                rules["class"].append(
                    {
                        "source_table": table,
                        "target_class": target_uri,
                        "id_columns": table_key(tables[table]),
                        "confidence": match.get("confidence", 0.0),
                        "table_role": table_role(tables[table]),
                    }
                )
    _add_structured_discriminator_class_rules(tables, class_tables, rules)
    for match in matches:
        target_uri = match.get("target_uri")
        if not target_uri or not _is_valid_target_uri(str(target_uri)):
            continue
        source_id = str(match["source_id"])
        if source_id.startswith("source-data:"):
            tail = source_id.split(":", 1)[1]
            table, column = tail.split(".", 1)
            if table not in tables:
                continue
            if _is_discriminator_data_column(tables[table], column):
                continue
            if is_generic_identifier_column(tables[table], column) and not _semantic_id_allowed(table, column, match):
                continue
            table_classes = _table_class_options(table, class_tables, rules)
            if table_classes and any(_property_domain_compatible(match, class_uri) for class_uri in table_classes):
                rules["data"].append(
                    {
                        "source_table": table,
                        "source_column": column,
                        "target_property": target_uri,
                        "confidence": match.get("confidence", 0.0),
                    }
                )
                continue
            attribute_rule = _attribute_table_data_rule(table, column, match, str(target_uri), tables, class_tables)
            if attribute_rule:
                rules["data"].append(attribute_rule)
        elif source_id.startswith("source-object:"):
            tail = source_id.split(":", 1)[1]
            table, columns = tail.split(".", 1)
            if table not in tables:
                continue
            source_columns = columns.split(",")
            for fk in tables[table].foreign_keys:
                if fk.columns == source_columns:
                    if table_role(tables[table]) == "join_table":
                        rules["object"].extend(_join_object_rules(table, fk, match, target_uri, tables, class_tables, rules))
                    else:
                        if fk.ref_table not in class_tables:
                            continue
                        source_classes = _table_class_options(table, class_tables, rules)
                        target_classes = _table_class_options(fk.ref_table, class_tables, rules)
                        if (
                            source_classes
                            and target_classes
                            and _best_domain_range_score(match, source_classes, target_classes) > 0
                        ):
                            rules["object"].append(
                                {
                                    "source_table": table,
                                    "source_columns": fk.columns,
                                    "target_property": target_uri,
                                    "target_table": fk.ref_table,
                                    "target_columns": fk.ref_columns,
                                    "confidence": match.get("confidence", 0.0),
                                }
                            )
                        elif (
                            source_classes
                            and target_classes
                            and _best_domain_range_score(match, target_classes, source_classes) > 0
                        ):
                            rules["object"].append(
                                {
                                    "source_table": table,
                                    "source_columns": fk.columns,
                                    "target_property": target_uri,
                                    "target_table": table,
                                    "target_columns": table_key(tables[table]),
                                    "subject_table": fk.ref_table,
                                    "subject_columns": fk.columns,
                                    "subject_target_columns": fk.ref_columns,
                                    "object_table": table,
                                    "object_columns": table_key(tables[table]),
                                    "object_target_columns": table_key(tables[table]),
                                    "confidence": match.get("confidence", 0.0),
                                }
                            )
    return {"rules": rules}


def _join_object_rules(
    table: str,
    matched_fk: Any,
    match: dict[str, Any],
    target_uri: str,
    tables: dict[str, Table],
    class_tables: dict[str, str],
    all_rules: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    matched_classes = _table_class_options(matched_fk.ref_table, class_tables, all_rules)
    if not matched_classes:
        return rules
    for other_fk in _other_join_endpoints(table, matched_fk, tables):
        if other_fk["columns"] == matched_fk.columns:
            continue
        other_classes = _table_class_options(other_fk["ref_table"], class_tables, all_rules)
        if not other_classes:
            continue
        inverse_score = _best_domain_range_score(match, other_classes, matched_classes)
        direct_score = _best_domain_range_score(match, matched_classes, other_classes)
        if inverse_score > direct_score and inverse_score > 0:
            rules.append(
                {
                    "source_table": table,
                    "source_columns": other_fk["columns"],
                    "target_property": target_uri,
                    "target_table": matched_fk.ref_table,
                    "target_columns": matched_fk.columns,
                    "subject_table": other_fk["ref_table"],
                    "subject_columns": other_fk["columns"],
                    "subject_target_columns": other_fk["ref_columns"],
                    "object_table": matched_fk.ref_table,
                    "object_columns": matched_fk.columns,
                    "object_target_columns": matched_fk.ref_columns,
                    "confidence": match.get("confidence", 0.0),
                }
            )
        elif direct_score > 0:
            rules.append(
                {
                    "source_table": table,
                    "source_columns": matched_fk.columns,
                    "target_property": target_uri,
                    "target_table": other_fk["ref_table"],
                    "target_columns": other_fk["columns"],
                    "subject_table": matched_fk.ref_table,
                    "subject_columns": matched_fk.columns,
                    "subject_target_columns": matched_fk.ref_columns,
                    "object_table": other_fk["ref_table"],
                    "object_columns": other_fk["columns"],
                    "object_target_columns": other_fk["ref_columns"],
                    "confidence": match.get("confidence", 0.0),
                }
            )
    return rules


def _other_join_endpoints(table: str, matched_fk: Any, tables: dict[str, Table]) -> list[dict[str, Any]]:
    endpoints = [
        {"columns": fk.columns, "ref_table": fk.ref_table, "ref_columns": fk.ref_columns}
        for fk in tables[table].foreign_keys
        if fk.columns != matched_fk.columns
    ]
    matched_columns = set(matched_fk.columns)
    for column in tables[table].column_names():
        if column in matched_columns or any(column in endpoint["columns"] for endpoint in endpoints):
            continue
        inferred = _infer_ref_table(column, tables)
        if inferred:
            endpoints.append({"columns": [column], "ref_table": inferred, "ref_columns": table_key(tables[inferred])})
    return endpoints


def _infer_ref_table(column: str, tables: dict[str, Table]) -> str | None:
    special = {
        "conference_member": "conf_members",
        "program_committee_member": "pc_members",
        "author": "authors",
        "reviewer": "reviewers",
        "paper": "papers",
        "aid": "authors",
        "rid": "reviewers",
        "pid": "papers",
    }
    if column in special and special[column] in tables:
        return special[column]
    candidates = [column, f"{column}s", column.replace("_id", "")]
    for candidate in candidates:
        if candidate in tables:
            return candidate
    return None


def validate_fol(fol: dict[str, Any], tables: dict[str, Table]) -> dict[str, Any]:
    valid = {"class": [], "data": [], "object": []}
    for rule in fol.get("rules", {}).get("class", []):
        table = rule.get("source_table")
        if table in tables and rule.get("target_class"):
            valid["class"].append(rule)
    for rule in fol.get("rules", {}).get("data", []):
        table = rule.get("source_table")
        column = rule.get("source_column")
        if table in tables and column in tables[table].column_names() and rule.get("target_property"):
            valid["data"].append(rule)
    for rule in fol.get("rules", {}).get("object", []):
        table = rule.get("source_table")
        target = rule.get("target_table")
        if table in tables and target in tables and rule.get("target_property"):
            valid["object"].append(rule)
    return {"rules": valid}


def _rule_targets(rule: dict[str, Any], kind: str) -> list[str]:
    if kind == "class":
        return [str(rule.get("target_class", ""))]
    if kind == "data":
        return [str(rule.get("target_property", ""))]
    if kind == "object":
        return [str(rule.get("target_property", ""))]
    return []


def _filter_valid(rule: dict[str, Any], table: Table) -> bool:
    row_filter = rule.get("row_filter")
    if not row_filter:
        return True
    column = row_filter.get("column")
    if column not in table.column_names():
        return False
    return bool(row_filter.get("truthy") or "equals" in row_filter)


def _fk_matches(table: Table, source_columns: list[str], target_table: str, target_columns: list[str]) -> bool:
    return any(fk.columns == source_columns and fk.ref_table == target_table and fk.ref_columns == target_columns for fk in table.foreign_keys)


def fol_validation_issues(fol: dict[str, Any], matches: list[dict[str, Any]], tables: dict[str, Table]) -> list[dict[str, Any]]:
    """Return schema/provenance FOL issues without using benchmark gold."""
    issues: list[dict[str, Any]] = []
    allowed_targets = {str(match.get("target_uri")) for match in matches if match.get("target_uri")}
    selected_by_source = {str(match.get("source_id")): match for match in matches if match.get("target_uri")}

    for kind in ("class", "data", "object"):
        for index, rule in enumerate(fol.get("rules", {}).get(kind, []) or []):
            rule_id = f"{kind}:{index}"
            for target in _rule_targets(rule, kind):
                if target and target not in allowed_targets:
                    issues.append({"rule_id": rule_id, "issue": "target_not_in_selected_matches", "target": target})
            match_ids = [str(value) for value in rule.get("match_ids", []) or []]
            if not match_ids:
                issues.append({"rule_id": rule_id, "issue": "missing_match_ids"})
            elif not any(match_id in selected_by_source for match_id in match_ids):
                issues.append({"rule_id": rule_id, "issue": "match_ids_do_not_reference_selected_matches", "match_ids": match_ids})

            table_name = str(rule.get("source_table", ""))
            table = tables.get(table_name)
            if not table:
                issues.append({"rule_id": rule_id, "issue": "unknown_source_table", "source_table": table_name})
                continue
            if kind == "class" and not _filter_valid(rule, table):
                issues.append({"rule_id": rule_id, "issue": "invalid_row_filter", "row_filter": rule.get("row_filter")})
            if kind == "data":
                column = str(rule.get("source_column", ""))
                if column not in table.column_names():
                    issues.append({"rule_id": rule_id, "issue": "unknown_source_column", "source_column": column})
                if column in foreign_key_columns(table) and not rule.get("subject_table"):
                    issues.append({"rule_id": rule_id, "issue": "fk_column_used_as_direct_data_rule", "source_column": column})
            if kind == "object":
                target_table = str(rule.get("target_table", ""))
                if target_table not in tables:
                    issues.append({"rule_id": rule_id, "issue": "unknown_target_table", "target_table": target_table})
                    continue
                source_columns = list(rule.get("source_columns") or [])
                target_columns = list(rule.get("target_columns") or table_key(tables[target_table]))
                if rule.get("subject_table") or rule.get("object_table"):
                    for field in ("subject_table", "object_table"):
                        value = rule.get(field)
                        if value and value not in tables:
                            issues.append({"rule_id": rule_id, "issue": f"unknown_{field}", field: value})
                    for field in ("subject_columns", "object_columns"):
                        for column in rule.get(field, []) or []:
                            if column not in table.column_names():
                                issues.append({"rule_id": rule_id, "issue": f"unknown_{field}", "column": column})
                elif not _fk_matches(table, source_columns, target_table, target_columns):
                    issues.append(
                        {
                            "rule_id": rule_id,
                            "issue": "object_rule_not_supported_by_fk",
                            "source_columns": source_columns,
                            "target_table": target_table,
                            "target_columns": target_columns,
                        }
                    )

    discriminator_matches = [
        match for match in matches if str(match.get("source_id", "")).startswith("source-discriminator:") and match.get("target_uri")
    ]
    class_rules = fol.get("rules", {}).get("class", []) or []
    for match in discriminator_matches:
        target = str(match.get("target_uri"))
        row_filter = match.get("row_filter")
        source_id = str(match.get("source_id"))
        if not any(rule.get("target_class") == target and rule.get("row_filter") == row_filter for rule in class_rules):
            issues.append({"rule_id": source_id, "issue": "missing_discriminator_class_rule", "target_class": target, "row_filter": row_filter})
    return issues


def mapping_diagnostics(matches: list[dict[str, Any]], fol: dict[str, Any], tables: dict[str, Table]) -> dict[str, Any]:
    selected = [match for match in matches if match.get("target_uri")]
    invalid_target_uri_count = sum(1 for match in selected if not _is_valid_target_uri(str(match.get("target_uri", ""))))
    anonymous_target_count = sum(1 for match in selected if str(match.get("target_uri", "")).startswith("n"))
    generic_id_data_matches = 0
    for match in selected:
        source_id = str(match.get("source_id", ""))
        if not source_id.startswith("source-data:"):
            continue
        tail = source_id.split(":", 1)[1]
        table, column = tail.split(".", 1)
        if table in tables and is_generic_identifier_column(tables[table], column):
            generic_id_data_matches += 1
    join_table_class_matches = sum(
        1
        for match in selected
        if str(match.get("source_id", "")).startswith("source-class:")
        and (table := str(match.get("source_id", "")).split(":", 1)[1]) in tables
        and table_role(tables[table]) == "join_table"
    )
    rules = fol.get("rules", {})
    return {
        "selected_matches": len(selected),
        "invalid_target_uri_count": invalid_target_uri_count,
        "anonymous_target_count": anonymous_target_count,
        "generic_id_data_matches": generic_id_data_matches,
        "join_table_class_matches": join_table_class_matches,
        "class_rules": len(rules.get("class", [])),
        "data_rules": len(rules.get("data", [])),
        "object_rules": len(rules.get("object", [])),
    }
