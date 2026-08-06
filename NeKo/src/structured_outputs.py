"""Validated scientific output contracts for NeKo MCP tools."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from mcp_biomodelling_servers.structured_outputs import StructuredOutputModel

Verbosity = Literal["summary", "preview", "full"]
ConnectorMethod = Literal["hubs", "relax_max_len", "unsigned"]


class NeKoScientificResult(StructuredOutputModel):
    """Fields shared by NeKo scientific query results."""

    server: Literal["NeKo"]
    session_id: str = Field(min_length=1)


class NeKoNodeRecord(StructuredOutputModel):
    """Stable biological identifiers for one network node."""

    gene_symbol: str | None = None
    uniprot: str | None = None
    node_type: str | None = None


class NeKoInteractionRecord(StructuredOutputModel):
    """One directed signalling interaction."""

    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    effect: str | None = None


class NeKoReferencedInteractionRecord(NeKoInteractionRecord):
    """One interaction and its complete normalized reference list."""

    reference_count: int = Field(ge=0)
    references: list[str]


class NeKoNetworkStatusResult(NeKoScientificResult):
    """Machine-readable status for one NeKo network session."""

    has_network: bool
    node_count: int = Field(ge=0)
    interaction_count: int = Field(ge=0)


class NeKoNetworkInventoryResult(NeKoScientificResult):
    """Nodes and interactions returned by ``list_genes_and_interactions``."""

    verbosity: Verbosity
    total_node_count: int = Field(ge=0)
    total_interaction_count: int = Field(ge=0)
    returned_node_count: int = Field(ge=0)
    returned_interaction_count: int = Field(ge=0)
    truncated: bool
    nodes: list[NeKoNodeRecord]
    interactions: list[NeKoInteractionRecord]


class NeKoPathSearchResult(NeKoScientificResult):
    """Captured output from NeKo's print-based path search API."""

    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    max_length: int = Field(ge=1)
    has_output: bool
    output_line_count: int = Field(ge=0)
    path_output_lines: list[str]


class NeKoReferenceQueryResult(NeKoScientificResult):
    """Literature evidence for interactions matching one or two nodes."""

    node1: str = Field(min_length=1)
    node2: str | None = None
    interaction_count: int = Field(ge=0)
    interactions: list[NeKoReferencedInteractionRecord]


class NeKoInteractionFilterResult(NeKoScientificResult):
    """Interactions matching the requested effect and endpoint filters."""

    verbosity: Verbosity
    effect_filter: list[str] | None = None
    source_filter: str | None = None
    target_filter: str | None = None
    total_match_count: int = Field(ge=0)
    returned_count: int = Field(ge=0)
    truncated: bool
    interactions: list[NeKoInteractionRecord]


class NeKoComponentRecord(StructuredOutputModel):
    """One undirected connected component of the current network."""

    component_id: int = Field(ge=0)
    size: int = Field(ge=1)
    average_degree: float = Field(ge=0)
    nodes: list[NeKoNodeRecord]


class NeKoConnectivityResult(NeKoScientificResult):
    """Isolated nodes and full connected-component membership for the network."""

    total_node_count: int = Field(ge=0)
    disconnected_count: int = Field(ge=0)
    all_nodes_have_interactions: bool
    disconnected_nodes: list[NeKoNodeRecord]
    component_count: int = Field(ge=0)
    largest_component_size: int = Field(ge=0)
    components: list[NeKoComponentRecord]


class NeKoHubCandidate(StructuredOutputModel):
    """A high-degree node suggested as a possible connector."""

    gene_symbol: str | None = None
    uniprot: str = Field(min_length=1)
    relative_score: float = Field(ge=0, le=1)
    degree: int = Field(ge=0)


class NeKoConnectorSimulation(StructuredOutputModel):
    """Predicted effect of a non-mutating connection-parameter simulation."""

    predicted_new_edges: int = Field(ge=0)
    simulated_max_length: int | None = Field(default=None, ge=1)
    simulated_only_signed: bool | None = None


class NeKoConnectionPreviewResult(NeKoScientificResult):
    """Candidates or simulation results returned by ``preview_connection_impact``."""

    method: ConnectorMethod
    rationale: str
    suggestion_count: int = Field(ge=0)
    hub_candidates: list[NeKoHubCandidate]
    simulation: NeKoConnectorSimulation | None = None
