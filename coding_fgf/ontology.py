from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from .constants import SOURCE_BASE
from .io import write_jsonl
from .schema import Table, table_role


@dataclass
class OntologyRecord:
    id: str
    kind: str
    uri: str
    local_name: str
    label: str = ""
    comment: str = ""
    domain: list[str] = field(default_factory=list)
    range: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
    neighbors: list[str] = field(default_factory=list)
    source_table: str = ""
    source_column: str = ""
    table_role: str = ""
    primary_key: list[str] = field(default_factory=list)
    foreign_keys: list[str] = field(default_factory=list)
    sample_values: list[str] = field(default_factory=list)
    text: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def uri_local_name(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[1]
    return uri.rstrip("/").rsplit("/", 1)[-1]


def split_words(value: str) -> list[str]:
    value = re.sub(r"[_\-]+", " ", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    return [part.lower() for part in re.split(r"\W+", value) if part]


def _string_objects(graph: Graph, subject: URIRef, predicate: URIRef) -> list[str]:
    values: list[str] = []
    for obj in graph.objects(subject, predicate):
        if isinstance(obj, Literal):
            values.append(str(obj))
        elif isinstance(obj, URIRef):
            values.append(str(obj))
    return sorted(set(values))


def parse_ontology_records(path: Path) -> list[OntologyRecord]:
    graph = Graph()
    graph.parse(str(path))
    classes = {str(s) for s in graph.subjects(RDF.type, OWL.Class) if isinstance(s, URIRef)} | {
        str(s) for s in graph.subjects(RDF.type, RDFS.Class) if isinstance(s, URIRef)
    }
    object_props = {str(s) for s in graph.subjects(RDF.type, OWL.ObjectProperty) if isinstance(s, URIRef)}
    data_props = {str(s) for s in graph.subjects(RDF.type, OWL.DatatypeProperty) if isinstance(s, URIRef)}
    generic_props = {str(s) for s in graph.subjects(RDF.type, RDF.Property) if isinstance(s, URIRef)}

    records: list[OntologyRecord] = []
    for uri in sorted(classes):
        subject = URIRef(uri)
        record = OntologyRecord(
            id=f"class:{uri}",
            kind="class",
            uri=uri,
            local_name=uri_local_name(uri),
            label="; ".join(_string_objects(graph, subject, RDFS.label)),
            comment="; ".join(_string_objects(graph, subject, RDFS.comment)),
            parents=_string_objects(graph, subject, RDFS.subClassOf),
        )
        records.append(_with_text(record))

    for uri in sorted(object_props | data_props | generic_props):
        if uri in classes:
            continue
        subject = URIRef(uri)
        kind = "object_property" if uri in object_props else "data_property"
        ranges = _string_objects(graph, subject, RDFS.range)
        if kind == "data_property" and any(value in classes for value in ranges):
            kind = "object_property"
        record = OntologyRecord(
            id=f"{kind}:{uri}",
            kind=kind,
            uri=uri,
            local_name=uri_local_name(uri),
            label="; ".join(_string_objects(graph, subject, RDFS.label)),
            comment="; ".join(_string_objects(graph, subject, RDFS.comment)),
            domain=_string_objects(graph, subject, RDFS.domain),
            range=ranges,
        )
        records.append(_with_text(record))
    records.extend(_rdfs_annotation_records({record.uri for record in records}))
    return records


def _rdfs_annotation_records(existing_uris: set[str]) -> list[OntologyRecord]:
    records: list[OntologyRecord] = []
    for uri, label in ((str(RDFS.label), "label"), (str(RDFS.comment), "comment")):
        if uri in existing_uris:
            continue
        records.append(
            _with_text(
                OntologyRecord(
                    id=f"data_property:{uri}",
                    kind="data_property",
                    uri=uri,
                    local_name=label,
                    label=label,
                    comment=f"RDFS {label} annotation",
                )
            )
        )
    return records


def _with_text(record: OntologyRecord) -> OntologyRecord:
    bits = [
        f"kind: {record.kind}",
        f"uri: {record.uri}",
        f"local name: {' '.join(split_words(record.local_name))}",
    ]
    if record.label:
        bits.append(f"label: {record.label}")
    if record.comment:
        bits.append(f"comment: {record.comment}")
    if record.domain:
        bits.append("domain: " + ", ".join(uri_local_name(v) for v in record.domain))
    if record.range:
        bits.append("range: " + ", ".join(uri_local_name(v) for v in record.range))
    if record.parents:
        bits.append("parents: " + ", ".join(uri_local_name(v) for v in record.parents))
    if record.neighbors:
        bits.append("neighbors: " + ", ".join(record.neighbors))
    if record.source_table:
        bits.append(f"source table: {record.source_table}")
    if record.source_column:
        bits.append(f"source column: {record.source_column}")
    if record.table_role:
        bits.append(f"source table role: {record.table_role}")
    if record.primary_key:
        bits.append("primary key: " + ", ".join(record.primary_key))
    if record.foreign_keys:
        bits.append("foreign keys: " + "; ".join(record.foreign_keys))
    if record.sample_values:
        bits.append("sample values: " + ", ".join(record.sample_values[:5]))
    record.text = "\n".join(bits)
    return record


def source_schema_records(tables: dict[str, Table], rows: dict[str, list[dict[str, Optional[str]]]] | None = None) -> list[OntologyRecord]:
    rows = rows or {}
    records: list[OntologyRecord] = []
    for table in sorted(tables.values(), key=lambda t: t.name):
        class_uri = f"{SOURCE_BASE}{table.name}#Class"
        role = table_role(table)
        fk_text = [f"{','.join(fk.columns)} -> {fk.ref_table}({','.join(fk.ref_columns)})" for fk in table.foreign_keys]
        records.append(
            _with_text(
                OntologyRecord(
                    id=f"source-class:{table.name}",
                    kind="class",
                    uri=class_uri,
                    local_name=table.name,
                    label=table.name,
                    source_table=table.name,
                    table_role=role,
                    primary_key=table.primary_key,
                    foreign_keys=fk_text,
                    neighbors=[f"column:{c.name}" for c in table.columns] + [f"fk:{','.join(fk.columns)}->{fk.ref_table}" for fk in table.foreign_keys],
                )
            )
        )
        table_rows = rows.get(table.name, [])
        for col in table.columns:
            samples = []
            for row in table_rows:
                value = row.get(col.name)
                if value and value not in samples:
                    samples.append(value)
                if len(samples) >= 5:
                    break
            records.append(
                _with_text(
                    OntologyRecord(
                        id=f"source-data:{table.name}.{col.name}",
                        kind="data_property",
                        uri=f"{SOURCE_BASE}{table.name}#{col.name}",
                        local_name=f"{table.name} {col.name}",
                        label=col.name,
                        source_table=table.name,
                        source_column=col.name,
                        table_role=role,
                        primary_key=table.primary_key,
                        foreign_keys=fk_text,
                        range=[col.datatype],
                        sample_values=samples,
                    )
                )
            )
        for fk in table.foreign_keys:
            records.append(
                _with_text(
                    OntologyRecord(
                        id=f"source-object:{table.name}.{','.join(fk.columns)}",
                        kind="object_property",
                        uri=f"{SOURCE_BASE}{table.name}#{'_'.join(fk.columns)}_to_{fk.ref_table}",
                        local_name=f"{table.name} {' '.join(fk.columns)} to {fk.ref_table}",
                        label=f"{table.name} to {fk.ref_table}",
                        domain=[table.name],
                        range=[fk.ref_table],
                        source_table=table.name,
                        source_column=",".join(fk.columns),
                        table_role=role,
                        primary_key=table.primary_key,
                        foreign_keys=fk_text,
                    )
                )
            )
    return records


def enrich_source_records_from_morphkgc(
    records: list[OntologyRecord],
    tables: dict[str, Table],
    scenario: str,
    source_graph_path: Path,
) -> list[OntologyRecord]:
    """Append Morph-KGC source-RDF evidence while preserving current source IDs."""
    graph = Graph()
    graph.parse(str(source_graph_path))
    class_counts: dict[str, int] = {}
    predicate_counts: dict[str, int] = {}
    predicate_samples: dict[str, list[str]] = {}

    for table in tables.values():
        class_uri = f"{SOURCE_BASE}{scenario}#{table.name}"
        class_counts[table.name] = len({str(subject) for subject in graph.subjects(RDF.type, URIRef(class_uri))})
        for column in table.columns:
            predicate_uri = f"{SOURCE_BASE}{scenario}#{table.name}_{column.name}"
            values = [str(obj) for obj in graph.objects(predicate=URIRef(predicate_uri))]
            predicate_counts[predicate_uri] = len(values)
            predicate_samples[predicate_uri] = sorted(set(values))[:5]
        for fk in table.foreign_keys:
            predicate_uri = f"{SOURCE_BASE}{scenario}#{table.name}_{'_'.join(fk.columns)}_to_{fk.ref_table}"
            values = [str(obj) for obj in graph.objects(predicate=URIRef(predicate_uri))]
            predicate_counts[predicate_uri] = len(values)
            predicate_samples[predicate_uri] = sorted(set(values))[:5]

    enriched: list[OntologyRecord] = []
    for record in records:
        extra: list[str] = []
        if record.id.startswith("source-class:") and record.source_table:
            extra.append(f"morphkgc source rdf class: {SOURCE_BASE}{scenario}#{record.source_table}")
            extra.append(f"morphkgc instance count: {class_counts.get(record.source_table, 0)}")
        elif record.id.startswith("source-data:") and record.source_table and record.source_column:
            predicate_uri = f"{SOURCE_BASE}{scenario}#{record.source_table}_{record.source_column}"
            extra.append(f"morphkgc source rdf predicate: {predicate_uri}")
            extra.append(f"morphkgc predicate count: {predicate_counts.get(predicate_uri, 0)}")
            samples = predicate_samples.get(predicate_uri, [])
            if samples:
                extra.append("morphkgc object samples: " + ", ".join(samples[:5]))
        elif record.id.startswith("source-object:") and record.source_table and record.source_column:
            table = tables.get(record.source_table)
            source_columns = record.source_column.split(",")
            fk = None
            if table is not None:
                for candidate in table.foreign_keys:
                    if candidate.columns == source_columns:
                        fk = candidate
                        break
            if fk is not None:
                predicate_uri = f"{SOURCE_BASE}{scenario}#{record.source_table}_{'_'.join(fk.columns)}_to_{fk.ref_table}"
                extra.append(f"morphkgc source rdf predicate: {predicate_uri}")
                extra.append(f"morphkgc edge count: {predicate_counts.get(predicate_uri, 0)}")
                samples = predicate_samples.get(predicate_uri, [])
                if samples:
                    extra.append("morphkgc object uri samples: " + ", ".join(samples[:5]))
        if extra:
            record.text = record.text + "\n" + "\n".join(extra)
        enriched.append(record)
    return enriched


def write_records(work_dir: Path, target_records: Iterable[OntologyRecord], source_records: Iterable[OntologyRecord]) -> None:
    target = list(target_records)
    source = list(source_records)
    write_jsonl(work_dir / "target_records.jsonl", [r.to_dict() for r in target])
    write_jsonl(work_dir / "source_records.jsonl", [r.to_dict() for r in source])
    (work_dir / "target_records.md").write_text(records_markdown(target), encoding="utf-8")
    (work_dir / "source_records.md").write_text(records_markdown(source), encoding="utf-8")


def records_markdown(records: Iterable[OntologyRecord]) -> str:
    lines = ["# Verbalized Records", ""]
    for record in records:
        lines.append(f"## {record.kind}: {record.local_name}")
        lines.append("")
        lines.append(record.text)
        lines.append("")
    return "\n".join(lines)
