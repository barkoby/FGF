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


class SandboxExecutionError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(f"{status}: {message}")
        self.status = status

def _materializer_worker(scenario, tables, data, fol, code, result_path, memory_mb):
    """Only this worker constructs and executes the generated-code context."""
    import json
    import resource
    from pathlib import Path
    result = Path(result_path)
    try:
        limit = memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except (OSError, ValueError) as exc:
            raise SandboxExecutionError("unsupported_platform",
                "Cannot enforce worker memory limits; use the supported Linux Docker environment") from exc
        from .materialize import build_context
        context = build_context(scenario, tables, data, fol)
        triples = run_generated_materializer(code, context)
        payload = {"ok": True, "triples": triples, "runtime_log": context["runtime_log"]}
        with result.with_suffix(".tmp").open("w") as fh:
            json.dump(payload, fh)
        result.with_suffix(".tmp").replace(result)
    except BaseException as exc:
        status = "memory_exhausted" if isinstance(exc, MemoryError) else getattr(exc, "status", "worker_error")
        # Do not retain a large result while reporting allocation failure.
        payload = None
        triples = None
        context = None
        result.write_text(json.dumps({"ok": False, "status": status, "error": type(exc).__name__ + ": " + str(exc)}))

def run_isolated_materializer(scenario, tables, data, fol, code):
    import json
    import math
    import multiprocessing
    import os
    import sys
    import tempfile
    from pathlib import Path
    if sys.platform != "linux":
        raise SandboxExecutionError("unsupported_platform", "Generated materialization requires Linux resource controls; use Docker")
    try:
        import resource
        resource.RLIMIT_AS
    except (ImportError, AttributeError) as exc:
        raise SandboxExecutionError("unsupported_platform", "Use the Linux Docker image for resource-limited execution") from exc
    timeout = float(os.getenv("CODING_FGF_SANDBOX_TIMEOUT_SECONDS", "120"))
    memory_mb = int(os.getenv("CODING_FGF_SANDBOX_MEMORY_MB", "2048"))
    if not math.isfinite(timeout) or timeout <= 0 or memory_mb <= 0:
        raise ValueError("Sandbox timeout and memory limits must be positive")
    validate_generated_code(code)
    with tempfile.TemporaryDirectory(prefix="fgf-materializer-") as scratch:
        result_path = Path(scratch) / "result.json"
        process = multiprocessing.get_context("spawn").Process(
            target=_materializer_worker,
            args=(scenario, tables, data, fol, code, str(result_path), memory_mb))
        process.start()
        try:
            # Results use a file rather than a pipe, so a large result cannot deadlock join().
            process.join(timeout)
            if process.is_alive():
                raise SandboxExecutionError("timeout", f"Materializer exceeded {timeout:g} seconds")
            if process.exitcode != 0 or not result_path.exists():
                raise SandboxExecutionError("worker_error", f"Materializer worker exited with code {process.exitcode}")
            result = json.loads(result_path.read_text())
            if not result.get("ok"):
                raise SandboxExecutionError(result["status"], result["error"])
            return [tuple(triple) for triple in result["triples"]], result["runtime_log"]
        finally:
            if process.is_alive():
                process.terminate()
                process.join(1)
            if process.is_alive():
                process.kill()
            process.join()
            process.close()
