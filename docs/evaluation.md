# Evaluation conventions

Enabled `.qpair` files are processed in sorted filename order. SQL runs against the scenario database; SPARQL runs against the materialized graph. Disabled queries are omitted.

Values are stringified. SPARQL IRIs become `##iri##`. For multicolumn results, non-IRI positions in the first returned SPARQL row determine the compared columns. Membership comparison counts occurrences in each result list; it is not a strict multiset join. Both lists empty gives precision, recall, and F1 of one; one empty list gives zero scores.

Scenario metrics are arithmetic means of per-query precision, recall, and F1. Scenario F1 is not recomputed from the average precision and recall. With no enabled pairs, scores and query count are zero. Query errors abort graph evaluation rather than silently exclude queries.

These describe this implementation's conventions and do not assert equivalence to every upstream RODI evaluator mode.

The FOL comparison evaluates frozen graphs only after selection is saved. Diagnostic failure labels guide inspection; they do not establish causes.
