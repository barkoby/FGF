# Code Generation and Validation

This document describes the current high-level design of the FGF code-generation stage. The stage converts validated FOL-style mapping rules into executable Python code, validates that code, selects among multiple generated candidates using internal execution evidence, and materializes RDF by executing the selected program.

The description is implementation-grounded but intentionally independent of any particular benchmark result. It does not describe evaluation feedback, run logs, or paper-score comparisons.

## Purpose

The code-generation stage is responsible for turning validated mapping rules into an executable transformation program. Its input is a set of FOL-style rules produced earlier in the pipeline. Its output is a Python program that defines a single function:

```python
def materialize(context):
    ...
```

The final RDF graph is produced only by executing this generated function inside the controlled FGF runtime. The generated program is therefore part of the evaluated method, not an auxiliary explanation of the mapping.

## Inputs

The stage receives:

- validated FOL-style rules, grouped by rule kind: class, data, and object;
- relational source schema and source rows;
- a configured LLM provider and model;
- the code-generation prompt version;
- a sandbox runtime context exposing approved helper functions.

The code generator does not receive query-pair failures, gold SPARQL answers, paper baseline scores, or dataset-specific target outputs. Repair and selection decisions are based on internal validation and execution evidence only.

## Prompt Contract

The code-generation prompt asks the LLM to return JSON containing Python source code. The generated code must define exactly one callable entry point, `materialize(context)`.

The sandbox exposes only the following helper functions:

- `rows(table)`: iterate source rows for a table;
- `all_rules(kind)`: retrieve rules of kind `class`, `data`, or `object`;
- `emit_type(row, rule)`: emit an RDF type triple for a class rule;
- `emit_data(row, rule)`: emit a datatype-property triple for a data rule;
- `emit_object(row, rule)`: emit an object-property triple for an object rule.

The intended generated code is helper-driven. It should iterate the applicable rule groups and source rows, then delegate RDF construction, row filters, key checks, and object lookup to the approved helper functions. It must not invent target URIs, constants, mappings, or benchmark-specific fixes.

## Self-Consistency over Code Candidates

Code generation uses a self-consistency strategy. Instead of relying on a single generated program, the pipeline requests multiple independent code candidates from the configured LLM. The number of candidates is controlled by the `codegen-self-consistency` setting. When runtime selection is available, the implementation uses at least three candidates.

Each candidate is treated as a complete possible implementation of the FOL rules. Candidates are not selected by comparing against benchmark answers. They are selected by static validity and internal execution behavior.

## Static Validation

Each generated candidate is parsed with the Python AST before execution. Static validation enforces a small and auditable subset of Python.

The validator requires:

- exactly one top-level function named `materialize`;
- no imports;
- no class definitions;
- no lambda expressions;
- no `while` loops;
- no `try` blocks;
- no global or nonlocal declarations;
- no dunder names or dunder attribute access;
- no method calls such as `row.get(...)`, `items()`, `append()`, or other object-method invocation;
- no dynamic function calls;
- only approved helper calls and a small set of safe builtins.

Candidates that fail static validation are rejected as executable materializers, but they may still be represented in the candidate-selection process as failed candidates.

## Execution Validation

Every static-valid candidate is executed in the sandbox against the scenario's source data and validated FOL rules. The sandbox supplies only the approved helper functions and a restricted builtin environment. The generated function cannot import modules, access files, open network connections, or call arbitrary methods.

Execution produces RDF-compatible triples through the helper functions. The runtime checks whether emitted triples have valid RDF shapes, whether expected rule kinds were attempted, whether reachable rules were skipped, and whether reachable rules produced no triples. This evidence is internal to the transformation process and is available before query-pair evaluation.

## Candidate Scoring

Each candidate receives an internal score. The score prioritizes:

1. safe execution with no invalid triples and no skipped reachable rules;
2. coverage of every rule kind present in the FOL input;
3. larger generated triple count;
4. fewer reachable rules with zero output;
5. fewer invalid triples;
6. shorter generated code as a final tie-breaker.

The selected program is the candidate with the highest internal score. This score is an execution-health criterion, not a benchmark-answer criterion.

## Repair

If the selected candidate fails static validation, fails execution validation, misses required rule-kind coverage, skips reachable rules, or has reachable zero-output rules, the pipeline may request one repair from the same configured LLM.

The repair prompt includes the previous code and internal diagnostics. The repair is constrained to the supplied FOL rules and approved helper interface. It may adjust iteration strategy, rule grouping, or helper use, but it must not introduce new ontology targets, constants, mappings, or benchmark-specific logic.

The repaired program is executed and scored like any other candidate. It is accepted only if it satisfies the internal safety criteria and does not worsen the selected candidate according to the acceptance checks. If the repair is not accepted, the pipeline keeps the original selected candidate.

## Materialization

After candidate selection and optional repair, the chosen program is saved as the final generated FGF code. RDF materialization then executes this generated code over the runtime context.

The materializer returns triples as subject-predicate-object tuples. Literal values are represented with a literal marker and converted into RDF literals during graph construction; other objects are interpreted as URI references. The resulting graph is serialized as the generated RDF output.

When generated code is required, deterministic materialization fallback is disallowed. This ensures that the final RDF graph is produced by executing the selected generated program.

## Worker isolation

Generated materializers execute in spawned Linux workers with wall-clock and address-space limits. Helpers are reconstructed in the worker. Triples and complete runtime diagnostics return to the parent. Limit failures reject candidates without partial RDF or deterministic replacement. These controls are not a general-purpose hostile-code isolation boundary.
