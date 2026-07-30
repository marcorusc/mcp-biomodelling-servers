"""Shared validated output models for MCP bio-modelling tools."""

from __future__ import annotations

from typing import Literal

from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel, ConfigDict, Field


class StructuredOutputModel(BaseModel):
    """Base class for strict structured tool output contracts."""

    model_config = ConfigDict(extra="forbid")


class ActiveSessionSummary(StructuredOutputModel):
    """Fields shared by active in-memory modelling sessions."""

    session_id: str = Field(
        min_length=1,
        description="Complete server-generated session identifier.",
    )
    created_at: float = Field(
        ge=0,
        description="Session creation time as Unix seconds.",
    )
    last_accessed: float = Field(
        ge=0,
        description="Most recent session access time as Unix seconds.",
    )
    is_default: bool = Field(
        description="Whether this session is the server's active default.",
    )


class MaBoSSSessionSummary(ActiveSessionSummary):
    """Machine-readable status for one active MaBoSS session."""

    has_simulation: bool
    has_result: bool
    bnd_path: str | None = None
    cfg_path: str | None = None


class NeKoSessionSummary(ActiveSessionSummary):
    """Machine-readable status for one active NeKo session."""

    has_network: bool
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    history_max_states: int | None = Field(default=None, ge=2)


class PhysiCellSessionSummary(ActiveSessionSummary):
    """Machine-readable status for one active PhysiCell session."""

    session_name: str | None = None
    has_configuration: bool
    progress: float = Field(ge=0, le=100)
    scenario_context: str | None = None
    substrates_count: int = Field(ge=0)
    cell_types_count: int = Field(ge=0)
    rules_count: int = Field(ge=0)
    physiboss_models_count: int = Field(ge=0)
    physiboss_settings_count: int = Field(ge=0)
    physiboss_input_links_count: int = Field(ge=0)
    physiboss_output_links_count: int = Field(ge=0)
    physiboss_mutations_count: int = Field(ge=0)
    loaded_from_xml: bool
    xml_modification_count: int = Field(ge=0)


class MaBoSSSessionListResult(StructuredOutputModel):
    """Structured result returned by MaBoSS ``list_sessions``."""

    server: Literal["MaBoSS"]
    count: int = Field(ge=0)
    sessions: list[MaBoSSSessionSummary]


class NeKoSessionListResult(StructuredOutputModel):
    """Structured result returned by NeKo ``list_sessions``."""

    server: Literal["NeKo"]
    count: int = Field(ge=0)
    sessions: list[NeKoSessionSummary]


class PhysiCellSessionListResult(StructuredOutputModel):
    """Structured result returned by PhysiCell ``list_sessions``."""

    server: Literal["PhysiCell"]
    count: int = Field(ge=0)
    sessions: list[PhysiCellSessionSummary]


class ArtifactSessionSummary(StructuredOutputModel):
    """Metadata for one session artifact directory."""

    session_id: str = Field(
        min_length=1,
        description="Complete session identifier read from artifact metadata.",
    )
    server: str = Field(
        min_length=1,
        description="Server recorded in the artifact metadata.",
    )
    label: str | None = None
    created_at: str | None = Field(
        default=None,
        description="UTC ISO-8601 creation timestamp when metadata is available.",
    )
    files: list[str] = Field(
        description="Artifact basenames stored for this session.",
    )


class MaBoSSArtifactSessionListResult(StructuredOutputModel):
    """Structured result returned by MaBoSS ``list_artifact_sessions``."""

    server: Literal["MaBoSS"]
    count: int = Field(ge=0)
    sessions: list[ArtifactSessionSummary]


class NeKoArtifactSessionListResult(StructuredOutputModel):
    """Structured result returned by NeKo ``list_artifact_sessions``."""

    server: Literal["NeKo"]
    count: int = Field(ge=0)
    sessions: list[ArtifactSessionSummary]


class PhysiCellArtifactSessionListResult(StructuredOutputModel):
    """Structured result returned by PhysiCell ``list_artifact_sessions``."""

    server: Literal["PhysiCell"]
    count: int = Field(ge=0)
    sessions: list[ArtifactSessionSummary]


def structured_report(
    text: str,
    payload: StructuredOutputModel,
) -> CallToolResult:
    """Return model-readable text plus validated application data."""
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=payload.model_dump(mode="json"),
    )
