from __future__ import annotations

import ast
from typing import Any


ALLOWED_CALLS = {"rows", "emit_type", "emit_data", "emit_object", "all_rules", "range", "len", "str"}


class SandboxViolation(ValueError):
    pass


def validate_generated_code(code: str) -> None:
    tree = ast.parse(code)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "materialize":
        raise SandboxViolation("Generated code must define exactly materialize(context)")
    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.With,
                ast.AsyncWith,
                ast.Try,
                ast.Lambda,
                ast.ClassDef,
                ast.Global,
                ast.Nonlocal,
                ast.While,
            ),
        ):
            raise SandboxViolation(f"Disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SandboxViolation("Dunder attribute access is forbidden")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise SandboxViolation("Dunder names are forbidden")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in ALLOWED_CALLS:
                    raise SandboxViolation(f"Disallowed call: {node.func.id}")
            elif isinstance(node.func, ast.Attribute):
                raise SandboxViolation("Method calls are forbidden")
            else:
                raise SandboxViolation("Dynamic calls are forbidden")


def run_generated_materializer(code: str, context: dict[str, Any]) -> list[tuple[str, str, str]]:
    validate_generated_code(code)
    triples: list[tuple[str, str, str]] = []

    def rows(table: str) -> list[dict[str, Any]]:
        return context["rows"].get(table, [])

    def all_rules(kind: str) -> list[dict[str, Any]]:
        return context["rules"].get(kind, [])

    def emit_type(row: dict[str, Any], rule: dict[str, Any]) -> None:
        triple = context["emit_type"](row, rule)
        if triple:
            triples.append(triple)

    def emit_data(row: dict[str, Any], rule: dict[str, Any]) -> None:
        triple = context["emit_data"](row, rule)
        if triple:
            triples.append(triple)

    def emit_object(row: dict[str, Any], rule: dict[str, Any]) -> None:
        triple = context["emit_object"](row, rule)
        if triple:
            triples.append(triple)

    safe_globals = {
        "__builtins__": {"range": range, "len": len, "str": str},
        "rows": rows,
        "all_rules": all_rules,
        "emit_type": emit_type,
        "emit_data": emit_data,
        "emit_object": emit_object,
    }
    local_ns: dict[str, Any] = {}
    exec(compile(ast.parse(code), "<generated_fgf>", "exec"), safe_globals, local_ns)
    local_ns["materialize"](context)
    return triples
