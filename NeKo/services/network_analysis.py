"""Pure normalization and graph-analysis helpers for NeKo network data."""

from __future__ import annotations

import re

import pandas as pd

from ..src.structured_outputs import (
    NeKoInteractionRecord,
    NeKoNodeRecord,
    NeKoReferencedInteractionRecord,
)


def optional_text(value: object) -> str | None:
    """Convert one dataframe scalar to a JSON-safe optional string."""
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def required_text(value: object, *, field_name: str) -> str:
    """Convert a required dataframe scalar or report malformed network data."""
    text = optional_text(value)
    if text is None:
        raise ValueError(f"Network data contains an empty {field_name}.")
    return text


def node_records(network: object) -> list[NeKoNodeRecord]:
    """Return all network nodes in their stored order."""
    nodes = getattr(network, "nodes", None)
    if nodes is None or nodes.empty:
        return []

    records = []
    for _, row in nodes.iterrows():
        records.append(
            NeKoNodeRecord(
                gene_symbol=optional_text(row.get("Genesymbol")),
                uniprot=optional_text(row.get("Uniprot")),
                node_type=optional_text(row.get("Type")),
            )
        )
    return records


def node_record_index(network: object) -> dict[str, NeKoNodeRecord]:
    """Index node records by both UniProt identifier and gene symbol."""
    index = {}
    for record in node_records(network):
        if record.uniprot is not None:
            index[record.uniprot] = record
        if record.gene_symbol is not None:
            index.setdefault(record.gene_symbol, record)
    return index


def node_record_for_identifier(
    identifier: object,
    *,
    node_index: dict[str, NeKoNodeRecord],
    uniprot_to_symbol: dict[object, object],
) -> NeKoNodeRecord:
    """Return a complete node record for a backend graph identifier."""
    identifier_text = required_text(identifier, field_name="node identifier")
    stored = node_index.get(identifier_text)
    if stored is not None:
        return stored
    return NeKoNodeRecord(
        gene_symbol=optional_text(
            uniprot_to_symbol.get(
                identifier,
                uniprot_to_symbol.get(identifier_text),
            )
        ),
        uniprot=identifier_text,
    )


def interaction_records(df: pd.DataFrame) -> list[NeKoInteractionRecord]:
    """Convert an edge dataframe into strict JSON-safe interaction records."""
    records = []
    for _, row in df.iterrows():
        records.append(
            NeKoInteractionRecord(
                source=required_text(row.get("source"), field_name="edge source"),
                target=required_text(row.get("target"), field_name="edge target"),
                effect=optional_text(row.get("Effect")),
            )
        )
    return records


def edge_degrees(edges: pd.DataFrame) -> dict[str, int]:
    """Count incident edges per backend node after validating edge columns."""
    missing = {"source", "target"} - set(edges.columns)
    if missing:
        raise ValueError(
            "Network edge data are missing required columns: "
            + ", ".join(sorted(missing))
        )

    degrees: dict[str, int] = {}
    for source, target in edges[["source", "target"]].itertuples(
        index=False,
        name=None,
    ):
        if pd.notna(source) and source != "":
            source_text = str(source)
            degrees[source_text] = degrees.get(source_text, 0) + 1
        if pd.notna(target) and target != "":
            target_text = str(target)
            degrees[target_text] = degrees.get(target_text, 0) + 1
    return degrees


def resolve_requested_genes(
    network: object,
    edges: pd.DataFrame,
    requested_genes: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Resolve gene symbols or UniProt IDs to edge-list identifiers."""
    requested = list(dict.fromkeys(gene.strip() for gene in requested_genes))
    exact: dict[str, str] = {}
    folded: dict[str, set[str]] = {}

    def register(alias: str | None, canonical: str | None) -> None:
        if alias is None or canonical is None:
            return
        exact.setdefault(alias, canonical)
        folded.setdefault(alias.casefold(), set()).add(canonical)

    for record in node_records(network):
        canonical = record.gene_symbol or record.uniprot
        register(record.gene_symbol, canonical)
        register(record.uniprot, canonical)

    for column in ("source", "target"):
        if column not in edges.columns:
            continue
        for value in edges[column].tolist():
            identifier = optional_text(value)
            register(identifier, identifier)

    resolved = []
    missing = []
    seen_resolved = set()
    for gene in requested:
        canonical = exact.get(gene)
        if canonical is None:
            candidates = folded.get(gene.casefold(), set())
            if len(candidates) == 1:
                canonical = next(iter(candidates))
        if canonical is None:
            missing.append(gene)
        elif canonical not in seen_resolved:
            resolved.append(canonical)
            seen_resolved.add(canonical)
    return requested, resolved, missing


def partition_gene_set_edges(
    edges: pd.DataFrame,
    genes: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition edges touching, within, and crossing a requested gene set."""
    if edges.empty or not {"source", "target"}.issubset(edges.columns):
        empty = pd.DataFrame(columns=["source", "target", "Effect"])
        return empty.copy(), empty.copy(), empty.copy()
    source_inside = edges["source"].isin(genes)
    target_inside = edges["target"].isin(genes)
    internal = edges[source_inside & target_inside].copy()
    boundary = edges[source_inside ^ target_inside].copy()
    incident = edges[source_inside | target_inside].copy()
    return incident, internal, boundary


def gene_set_components(
    genes: list[str],
    internal_edges: pd.DataFrame,
    connectivity: str,
) -> list[list[str]]:
    """Compute weak or strong components of a gene-induced subgraph."""
    adjacency = {gene: set() for gene in genes}
    reverse = {gene: set() for gene in genes}
    for _, row in internal_edges.iterrows():
        source = required_text(row.get("source"), field_name="edge source")
        target = required_text(row.get("target"), field_name="edge target")
        adjacency[source].add(target)
        reverse[target].add(source)
        if connectivity == "weak":
            adjacency[target].add(source)
            reverse[source].add(target)

    if connectivity == "weak":
        remaining = set(genes)
        components = []
        while remaining:
            start = min(remaining)
            stack = [start]
            remaining.remove(start)
            component = []
            while stack:
                node = stack.pop()
                component.append(node)
                for neighbour in sorted(adjacency[node], reverse=True):
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        stack.append(neighbour)
            components.append(sorted(component))
    else:
        visited = set()
        finish_order = []
        for start in sorted(genes):
            if start in visited:
                continue
            stack = [(start, False)]
            while stack:
                node, expanded = stack.pop()
                if expanded:
                    finish_order.append(node)
                    continue
                if node in visited:
                    continue
                visited.add(node)
                stack.append((node, True))
                for neighbour in sorted(adjacency[node], reverse=True):
                    if neighbour not in visited:
                        stack.append((neighbour, False))

        visited.clear()
        components = []
        for start in reversed(finish_order):
            if start in visited:
                continue
            stack = [start]
            visited.add(start)
            component = []
            while stack:
                node = stack.pop()
                component.append(node)
                for neighbour in sorted(reverse[node], reverse=True):
                    if neighbour not in visited:
                        visited.add(neighbour)
                        stack.append(neighbour)
            components.append(sorted(component))

    components.sort(key=lambda component: (-len(component), component))
    return components


def reference_list(value: object) -> list[str]:
    """Normalize NeKo's comma/semicolon-delimited reference values."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_references = value
    else:
        text = optional_text(value)
        if text is None or text.lower() == "none":
            return []
        raw_references = re.split(r"[;,]", text)

    references = []
    seen = set()
    for value in raw_references:
        reference = optional_text(value)
        if reference is not None and reference not in seen:
            references.append(reference)
            seen.add(reference)
    return references


def referenced_interaction_records(
    df: pd.DataFrame,
) -> list[NeKoReferencedInteractionRecord]:
    """Convert an edge dataframe while retaining its complete evidence list."""
    records = []
    for _, row in df.iterrows():
        references = reference_list(row.get("References"))
        records.append(
            NeKoReferencedInteractionRecord(
                source=required_text(row.get("source"), field_name="edge source"),
                target=required_text(row.get("target"), field_name="edge target"),
                effect=optional_text(row.get("Effect")),
                reference_count=len(references),
                references=references,
            )
        )
    return records
