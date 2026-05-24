# FOL Generation and Validation

This document describes the current high-level design of the FOL generation and validation stage. The stage converts LLM-selected ontology matches into grounded FOL-style mapping rules that can later be translated into executable FGF code.

The description is implementation-grounded but independent of any particular benchmark result. It does not describe logs, run-specific F1 scores, or paper-score comparisons.

## Purpose

The FOL stage is the bridge between match selection and code generation. Matching produces source-to-ontology decisions, while code generation requires explicit transformation rules. The FOL stage translates accepted matches into structured rules for:

- class assertions;
- datatype-property assertions;
- object-property assertions.

These rules describe which source tables and columns should produce target RDF facts. The generated rules are validated before they are passed to the code-generation stage.

## Inputs

The stage receives:

- selected non-null source-target matches;
- the relational schema, including tables, columns, keys, and foreign keys;
- source table roles and source column structure;
- target URIs selected during matching;
- optional discriminator matches for boolean or type-like class distinctions;
- optional object-link evidence for foreign-key and join-table relations;
- the configured LLM provider and model.

The stage does not use query-pair failures, gold answers, paper baselines, or evaluation feedback when generating, repairing, or accepting FOL rules.

## Rule Generation

The LLM receives compact schema evidence and selected matches. When the number of selected matches is large, the implementation groups them into chunks so that each prompt remains bounded. Each chunk asks the LLM to return JSON with three rule arrays:

- `class`;
- `data`;
- `object`.

Each rule must cite the selected match or matches that justify it through `match_ids`. This provenance is important because it grounds the generated rule in a prior source-target decision. The LLM is instructed to use only selected target URIs, known source tables, and known source columns. If the supplied matches or schema do not justify a rule, the LLM should omit the rule rather than guess.

Class rules map source tables or discriminator-filtered rows to target classes. Data rules map source columns to target datatype properties. Object rules map source foreign-key or association-table structure to target object properties.

## Validation as Grounding

Validation grounds the generated FOL rules in the available schema and selected matches. The validator first normalizes the LLM output into the expected rule groups and discards rules that do not reference known tables, columns, or targets at the basic structural level. It then reports detailed validation issues that are used for repair and acceptance decisions.

The validation checks include:

- whether each target URI appears in the selected matches;
- whether each rule includes `match_ids`;
- whether `match_ids` refer to selected non-null matches;
- whether source tables and source columns exist in the SQL schema;
- whether row filters refer to valid source columns and supported filter forms;
- whether foreign-key columns are incorrectly used as direct datatype rules;
- whether object rules are supported by an actual foreign-key direction or explicit association-table structure;
- whether discriminator matches have corresponding class rules with the expected row filter.

These checks do not determine whether a generated RDF graph answers benchmark queries correctly. They only verify that rules are structurally grounded in the selected matches and source schema.

## Object-Link Evidence

When object-link evidence is enabled, the pipeline constructs legal object-rule plans from internal schema evidence and selected matches. A plan describes an allowed object-property realization, including source table, source columns, target table, target columns, subject/object fields for join tables, compatible class context, and reachability evidence from the source rows.

In this mode, object rules are constrained to the supplied plans. A generated object rule must reference a legal `plan_id` and copy the plan's rule fields exactly. This makes object-property FOL generation less free-form: the LLM chooses among schema-supported alternatives rather than inventing an object path.

Validation then checks that object rules reference known plans, match the selected plan fields, and avoid zero-reachability plans. This provides additional grounding for foreign-key and join-table mappings.

## Repair

If validation issues remain after initial FOL generation, the pipeline may ask the configured LLM for one repair. There are two repair modes:

- **General FOL repair**, which can repair class, data, and object rules using the current FOL, selected matches, schema, and validation issues.
- **Targeted object-rule repair**, which focuses on invalid object rules when object-link evidence is available.

Repair prompts may use only internal information: schema, selected matches, current FOL rules, validation issues, and optional object-link evidence. They must not use qpair names, gold answers, paper baselines, evaluation failures, or dataset-specific fixes. The repaired FOL is validated again before it can be accepted.

Repair acceptance is conservative. A repaired FOL is rejected if validation issues worsen, if all rules are removed, if selected matches remain but no rules are produced, if the total rule count collapses below half of the original rule count, or if any rule kind present in the original FOL disappears. Targeted object repair has an additional constraint: it must not modify unrelated valid class or data rules.

If repair is rejected or fails, the original validated LLM FOL is kept.

## Output

The stage returns grounded FOL rules grouped into `class`, `data`, and `object` arrays. These rules are the input to the code-generation stage. The FOL rules preserve target URI provenance through `match_ids`, and their structure is limited to source tables, columns, relationships, and object plans that are available in the current scenario.