"""Scientific outputs and durable session documents."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue

from mcp_biomodelling_servers.ode_handoff import (
    NeKoToBioMASSHandoffManifest,
    ODENetwork,
)
from mcp_biomodelling_servers.structured_outputs import (
    ArtifactFileSummary,
    ArtifactSessionSummary,
    StructuredOutputModel,
)

from .contracts import EvidenceRecord, LineEvidence, ModelConfiguration, ReactionRecord


class ModelDocument(StructuredOutputModel):
    mode: Literal["empty", "records", "document"] = "empty"
    version: int = 0
    biological_context: str | None = None
    network: ODENetwork | None = None
    upstream: NeKoToBioMASSHandoffManifest | None = None
    evidence: dict[str, EvidenceRecord] = Field(default_factory=dict)
    reactions: list[ReactionRecord] = Field(default_factory=list)
    text: str | None = None
    line_evidence: list[LineEvidence] = Field(default_factory=list)
    configuration: ModelConfiguration = Field(default_factory=ModelConfiguration)
    current_revision: str | None = None
    revisions: list[str] = Field(default_factory=list)
    reaction_lines: dict[str, int] = Field(default_factory=dict)
    record_line_count: int = 0
    species_mapping: dict[str, list[str]] = Field(default_factory=dict)
    source_file: dict[str, str] | None = None


class Coverage(StructuredOutputModel):
    total_edges: int = 0
    supported_edges: list[str] = Field(default_factory=list)
    assumed_edges: list[str] = Field(default_factory=list)
    unresolved_edges: list[str] = Field(default_factory=list)
    conflicting_evidence: list[str] = Field(default_factory=list)
    assumed_reactions: list[str] = Field(default_factory=list)
    unreviewed_edges: list[str] = Field(default_factory=list)


class ReactionInventoryItem(StructuredOutputModel):
    reaction_id: str
    line_number: int
    statement: str
    parameters: list[str] = Field(default_factory=list)


class BioMASSInventoryResult(StructuredOutputModel):
    server: Literal["BioMASS"] = "BioMASS"
    session_id: str
    document_version: int
    reactions: list[ReactionInventoryItem]
    species: list[str] = Field(default_factory=list)
    parameters: list[str] = Field(default_factory=list)
    species_mapping: dict[str, list[str]] = Field(default_factory=dict)
    generation_valid: bool | None = None
    issues: list[str] = Field(default_factory=list)


class BioMASSBuildResult(StructuredOutputModel):
    server: Literal["BioMASS"] = "BioMASS"
    session_id: str
    base_version: int
    document_version: int
    applied: bool
    can_apply: bool
    changes: list[ReactionInventoryItem]
    removed_reaction_ids: list[str]
    added_species: list[str] = Field(default_factory=list)
    removed_species: list[str] = Field(default_factory=list)
    added_parameters: list[str] = Field(default_factory=list)
    removed_parameters: list[str] = Field(default_factory=list)
    species_mapping: dict[str, list[str]] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class BioMASSStateResult(StructuredOutputModel):
    server: Literal["BioMASS"] = "BioMASS"
    session_id: str
    document: ModelDocument
    coverage: Coverage


class BioMASSSessionSummary(StructuredOutputModel):
    session_id: str
    created_at: float
    last_accessed: float
    is_default: bool
    mode: str
    current_revision: str | None


class BioMASSSessionListResult(StructuredOutputModel):
    server: Literal["BioMASS"] = "BioMASS"
    count: int
    sessions: list[BioMASSSessionSummary]


class BioMASSArtifactSessionListResult(StructuredOutputModel):
    server: Literal["BioMASS"] = "BioMASS"
    count: int
    sessions: list[ArtifactSessionSummary]


class BioMASSArtifactFileListResult(StructuredOutputModel):
    server: Literal["BioMASS"] = "BioMASS"
    scope: Literal["session"] = "session"
    session_id: str
    count: int
    files: list[ArtifactFileSummary]


class BioMASSArtifactCleanupResult(StructuredOutputModel):
    server: Literal["BioMASS"] = "BioMASS"
    session_id: str
    removed_count: int


class BioMASSValidationResult(StructuredOutputModel):
    server: Literal["BioMASS"] = "BioMASS"
    session_id: str
    syntax_valid: bool
    generation_valid: bool | None = None
    numerical_valid: bool | None = None
    issues: list[str]
    coverage: Coverage


class BioMASSJobResult(StructuredOutputModel):
    server: Literal["BioMASS"] = "BioMASS"
    session_id: str
    revision: str
    operation: Literal["generate", "simulate", "graph", "export"]
    files: list[ArtifactFileSummary]
    details: dict[str, JsonValue]
    coverage: Coverage
