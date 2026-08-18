"""Structured scientific-analysis MCP tool registrations for NeKo."""

import json
from typing import Annotated

import pandas as pd
from mcp.types import CallToolResult
from pydantic import Field

from mcp_biomodelling_servers.structured_outputs import structured_report

from ..app import mcp
from ..contracts import (
    READ_ONLY_CLOSED,
    NonEmptyString,
    NonEmptyStringList,
    NormalizedConnectivityMode,
    NormalizedVerbosity,
    OutputFormat,
)
from ..services.network_analysis import (
    edge_degrees,
    gene_set_components,
    interaction_records,
    node_record_for_identifier,
    node_record_index,
    optional_text,
    partition_gene_set_edges,
    required_text,
    resolve_requested_genes,
)
from ..session_manager import DEFAULT_VERBOSITY, normalize_verbosity
from ..src.helpers import (
    E_NO_NET,
    SUMMARY_HINT,
    _compute_components,
    _get_translators,
    _session_network,
    session_locked,
)
from ..src.structured_outputs import (
    NeKoComponentRecord,
    NeKoConnectivityResult,
    NeKoGeneSetAnalysisResult,
    NeKoGeneSetComponent,
)
from ..utils import clean_for_markdown


@mcp.tool(
    title="Analyze gene set",
    annotations=READ_ONLY_CLOSED,
    structured_output=True,
)
@session_locked
def analyze_gene_set(
    genes: Annotated[
        NonEmptyStringList,
        Field(
            description=(
                "Requested gene symbols or UniProt identifiers to audit "
                "against the current network."
            )
        ),
    ],
    connectivity: Annotated[
        NormalizedConnectivityMode,
        Field(
            description=(
                "Connectivity of the induced subgraph: 'weak' ignores edge "
                "direction; 'strong' requires directed mutual reachability."
            )
        ),
    ] = "weak",
    session_id: Annotated[
        NonEmptyString | None,
        Field(description="Session ID; omit to use the active/default session."),
    ] = None,
    verbosity: Annotated[
        NormalizedVerbosity,
        Field(
            description=(
                "Output detail level: 'summary' (counts), 'preview' (bounded "
                "edge tables), or 'full' (up to 500 edges per table)."
            )
        ),
    ] = DEFAULT_VERBOSITY,
    format: Annotated[
        OutputFormat,
        Field(description="Output format: 'markdown' (default) or 'json'."),
    ] = "markdown",
    max_rows: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Maximum internal and boundary edges returned in preview mode."
            ),
        ),
    ] = 50,
) -> Annotated[CallToolResult, NeKoGeneSetAnalysisResult]:
    """Audit a requested gene set without exporting or modifying the network.

    Reports identifier resolution, internal and boundary edges, induced
    isolates, and weak or strong components of the requested-gene subgraph.
    """
    verbosity = normalize_verbosity(verbosity)
    sess, network = _session_network(session_id)
    if network is None:
        raise RuntimeError(E_NO_NET)
    edges = sess.get_edges_df()
    if edges is None:
        edges = pd.DataFrame(columns=["source", "target", "Effect"])

    requested, resolved, missing = resolve_requested_genes(
        network,
        edges,
        genes,
    )
    _, internal, boundary = partition_gene_set_edges(edges, set(resolved))
    components_raw = gene_set_components(resolved, internal, connectivity)
    components = [
        NeKoGeneSetComponent(
            component_id=index,
            size=len(component),
            genes=component,
        )
        for index, component in enumerate(components_raw)
    ]
    internally_connected = set()
    if not internal.empty:
        internally_connected.update(
            required_text(value, field_name="edge source")
            for value in internal["source"].tolist()
        )
        internally_connected.update(
            required_text(value, field_name="edge target")
            for value in internal["target"].tolist()
        )
    induced_isolates = sorted(set(resolved) - internally_connected)

    if verbosity == "summary" and format != "json":
        internal_returned = internal.head(0)
        boundary_returned = boundary.head(0)
    elif verbosity == "full":
        internal_returned = internal.head(500)
        boundary_returned = boundary.head(500)
    else:
        internal_returned = internal.head(max_rows)
        boundary_returned = boundary.head(max_rows)

    internal_records = interaction_records(internal_returned)
    boundary_records = interaction_records(boundary_returned)
    truncated = (
        len(internal_records) < len(internal)
        or len(boundary_records) < len(boundary)
    )
    payload = NeKoGeneSetAnalysisResult(
        server="NeKo",
        session_id=sess.session_id,
        connectivity=connectivity,
        requested_genes=requested,
        resolved_genes=resolved,
        missing_genes=missing,
        requested_count=len(requested),
        resolved_count=len(resolved),
        missing_count=len(missing),
        internal_edge_count=len(internal),
        boundary_edge_count=len(boundary),
        induced_isolate_count=len(induced_isolates),
        induced_isolates=induced_isolates,
        component_count=len(components),
        largest_component_size=max(
            (component.size for component in components),
            default=0,
        ),
        components=components,
        returned_internal_edge_count=len(internal_records),
        returned_boundary_edge_count=len(boundary_records),
        truncated=truncated,
        internal_edges=internal_records,
        boundary_edges=boundary_records,
    )

    if format == "json":
        return structured_report(
            json.dumps(payload.model_dump(mode="json"), separators=(",", ":")),
            payload,
        )

    summary = (
        f"Gene-set audit: requested={len(requested)} resolved={len(resolved)} "
        f"missing={len(missing)}. Internal edges={len(internal)}; "
        f"boundary edges={len(boundary)}; induced {connectivity} "
        f"components={len(components)}; induced isolates="
        f"{len(induced_isolates)}."
    )
    if verbosity == "summary":
        return structured_report(f"{summary} {SUMMARY_HINT}", payload)

    component_lines = [
        f"- {component.component_id}: size={component.size} "
        f"genes={component.genes}"
        for component in components
    ] or ["- none"]
    internal_table = (
        "_No internal edges._"
        if internal_returned.empty
        else clean_for_markdown(internal_returned).to_markdown(
            index=False,
            tablefmt="plain",
        )
    )
    boundary_table = (
        "_No boundary edges._"
        if boundary_returned.empty
        else clean_for_markdown(boundary_returned).to_markdown(
            index=False,
            tablefmt="plain",
        )
    )
    details = [
        summary,
        f"Missing genes: {missing or 'none'}",
        f"Induced isolates: {induced_isolates or 'none'}",
        "Components:\n" + "\n".join(component_lines),
        "Internal edges:\n" + internal_table,
        "Boundary edges:\n" + boundary_table,
    ]
    if truncated:
        details.append("_Edge tables truncated._")
    return structured_report("\n\n".join(details), payload)


@mcp.tool(
    title="Analyze network connectivity",
    annotations=READ_ONLY_CLOSED,
    structured_output=True,
)
@session_locked
def analyze_connectivity(
    session_id: Annotated[
        NonEmptyString | None,
        Field(description="Session ID; omit to use the active/default session."),
    ] = None,
    verbosity: Annotated[
        NormalizedVerbosity,
        Field(
            description=(
                "Output detail level: 'summary' (counts only), 'preview'/'full' "
                "(isolated node list and per-component stats)."
            )
        ),
    ] = DEFAULT_VERBOSITY,
    format: Annotated[
        OutputFormat,
        Field(description="Output format: 'markdown' (default) or 'json'."),
    ] = "markdown",
) -> Annotated[CallToolResult, NeKoConnectivityResult]:
    """Report isolated (0-edge) nodes AND the full connected-component partition.

    A network can have zero isolated nodes yet still be fragmented into
    several disconnected multi-node clusters (e.g. two unrelated 10-node
    islands). This tool reports both facets together, so "all_nodes_have_interactions"
    plus "component_count == 1" together confirm the network is fully connected.
    Use before choosing a connection strategy (see preview_connection_impact()).
    Each component's `component_id` is a report label only - to bridge two
    components with bridge_components(), pass the Gene Symbols listed in
    their `nodes`, not the `component_id` integers.
    """
    verbosity = normalize_verbosity(verbosity)
    sess, network = _session_network(session_id)
    if network is None:
        raise RuntimeError(E_NO_NET)

    uniprot_to_symbol, _ = _get_translators(network)
    node_index = node_record_index(network)

    all_nodes = {
        node
        for node in network.nodes["Uniprot"].tolist()
        if optional_text(node) is not None
    }
    connected_nodes = (
        set(network.edges["source"].tolist())
        | set(network.edges["target"].tolist())
    )
    disconnected_uniprot = all_nodes - connected_nodes
    disconnected_nodes = [
        node_record_for_identifier(
            node,
            node_index=node_index,
            uniprot_to_symbol=uniprot_to_symbol,
        )
        for node in disconnected_uniprot
    ]
    disconnected_nodes.sort(
        key=lambda record: record.gene_symbol or record.uniprot or ""
    )

    components_raw = _compute_components(network)
    degrees = edge_degrees(network.edges)

    components = []
    for index, component in enumerate(components_raw):
        component_degrees = [degrees.get(node, 0) for node in component]
        average_degree = (
            round(sum(component_degrees) / len(component_degrees), 2)
            if component_degrees
            else 0
        )
        components.append(
            NeKoComponentRecord(
                component_id=index,
                size=len(component),
                average_degree=average_degree,
                nodes=[
                    node_record_for_identifier(
                        node,
                        node_index=node_index,
                        uniprot_to_symbol=uniprot_to_symbol,
                    )
                    for node in component
                ],
            )
        )
    largest_component_size = max(
        (component.size for component in components),
        default=0,
    )
    payload = NeKoConnectivityResult(
        server="NeKo",
        session_id=sess.session_id,
        total_node_count=len(all_nodes),
        disconnected_count=len(disconnected_nodes),
        all_nodes_have_interactions=not disconnected_nodes,
        disconnected_nodes=disconnected_nodes,
        component_count=len(components),
        largest_component_size=largest_component_size,
        components=components,
    )

    if format == "json":
        text = json.dumps(payload.model_dump(mode="json"), separators=(",", ":"))
        return structured_report(text, payload)
    if verbosity == "summary":
        text = (
            f"Isolated nodes: {len(disconnected_nodes)}/{len(all_nodes)}. "
            f"Components={len(components)} largest={largest_component_size}. "
            f"{SUMMARY_HINT}"
        )
        return structured_report(text, payload)

    lines = []
    if disconnected_nodes:
        labels = [
            record.gene_symbol or record.uniprot or "(unknown)"
            for record in disconnected_nodes
        ]
        lines.append("Isolated nodes (Gene Symbols):\n" + "\n".join(labels))
    else:
        lines.append("No isolated nodes.")

    if not components:
        lines.append("No components (empty network).")
    else:
        component_lines = ["Components:"]
        for component in components:
            visible_nodes = (
                component.nodes[:5]
                if verbosity == "preview"
                else component.nodes
            )
            node_labels = [
                node.gene_symbol or node.uniprot or "(unknown)"
                for node in visible_nodes
            ]
            label = "sample" if verbosity == "preview" else "nodes"
            component_lines.append(
                f"- {component.component_id}: size={component.size} "
                f"avg_deg={component.average_degree} {label}={node_labels}"
            )
        lines.append("\n".join(component_lines))

    return structured_report("\n\n".join(lines), payload)
