"""Shared validated output models for MCP bio-modelling tools."""

from __future__ import annotations

from pathlib import Path
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


class ArtifactFileSummary(StructuredOutputModel):
    """Machine-readable metadata for one generated artifact file."""

    session_id: str = Field(
        min_length=1,
        description="Complete session identifier that owns the artifact.",
    )
    name: str = Field(min_length=1, description="Artifact basename.")
    path: str = Field(min_length=1, description="Absolute artifact file path.")
    suffix: str = Field(description="Lowercase filename suffix, including the dot.")
    media_type: str | None = Field(
        default=None,
        description="Known Internet media type, or null for an unknown format.",
    )
    size_bytes: int | None = Field(
        default=None,
        ge=0,
        description="File size in bytes when the artifact remains readable.",
    )


class ArtifactFileListResult(StructuredOutputModel):
    """Fields shared by generated-artifact listing results."""

    scope: Literal["session", "all"]
    session_id: str | None = Field(
        default=None,
        description="Resolved full session identifier for session-scoped listings.",
    )
    count: int = Field(ge=0)
    files: list[ArtifactFileSummary]


class MaBoSSArtifactFileListResult(ArtifactFileListResult):
    """Structured result returned by MaBoSS ``list_generated_files``."""

    server: Literal["MaBoSS"]


class NeKoArtifactFileListResult(ArtifactFileListResult):
    """Structured result returned by NeKo ``list_bnet_files``."""

    server: Literal["NeKo"]


class PhysiCellArtifactFileListResult(ArtifactFileListResult):
    """Structured result returned by PhysiCell ``list_generated_files``."""

    server: Literal["PhysiCell"]


class ArtifactCleanupResult(StructuredOutputModel):
    """Fields shared by generated-artifact cleanup results."""

    session_id: str = Field(min_length=1)
    removed_count: int = Field(ge=0)


class MaBoSSArtifactCleanupResult(ArtifactCleanupResult):
    """Structured result returned by MaBoSS ``clean_generated_files``."""

    server: Literal["MaBoSS"]


class NeKoArtifactCleanupResult(ArtifactCleanupResult):
    """Structured result returned by NeKo ``clean_generated_files``."""

    server: Literal["NeKo"]


class PhysiCellArtifactCleanupResult(ArtifactCleanupResult):
    """Structured result returned by PhysiCell ``clean_generated_files``."""

    server: Literal["PhysiCell"]


class MaBoSSBnetConversionResult(StructuredOutputModel):
    """Structured result returned by ``bnet_to_bnd_and_cfg``."""

    server: Literal["MaBoSS"]
    session_id: str = Field(min_length=1)
    input_bnet_path: str = Field(min_length=1)
    bnd_file: ArtifactFileSummary
    cfg_file: ArtifactFileSummary


class MaBoSSModelExportResult(StructuredOutputModel):
    """Structured result returned by ``export_maboss_bnd_cfg``."""

    server: Literal["MaBoSS"]
    session_id: str = Field(min_length=1)
    prefix: str = Field(min_length=1)
    overwrite: bool
    bnd_file: ArtifactFileSummary
    cfg_file: ArtifactFileSummary


class NeKoNetworkExportResult(StructuredOutputModel):
    """Structured result returned by NeKo ``export_network``."""

    server: Literal["NeKo"]
    session_id: str = Field(min_length=1)
    format: Literal["sif", "bnet"]
    file: ArtifactFileSummary
    renamed_nodes: list[str]
    duplicate_rules_removed: list[str]


class PhysiCellXmlExportResult(StructuredOutputModel):
    """Structured result returned by ``export_xml_configuration``."""

    server: Literal["PhysiCell"]
    session_id: str = Field(min_length=1)
    file: ArtifactFileSummary
    source: Literal["created", "loaded"]
    source_filename: str | None = None
    modification_count: int = Field(ge=0)
    substrates: list[str]
    cell_types: list[str]
    progress: float = Field(ge=0, le=100)


class PhysiCellRulesExportResult(StructuredOutputModel):
    """Structured result returned by ``export_cell_rules_csv``."""

    server: Literal["PhysiCell"]
    session_id: str = Field(min_length=1)
    file: ArtifactFileSummary
    xml_reference: str = Field(min_length=1)
    enabled: bool
    rule_count: int = Field(ge=0)
    progress: float = Field(ge=0, le=100)


_ARTIFACT_MEDIA_TYPES = {
    ".bnd": "text/plain",
    ".bnet": "text/plain",
    ".cfg": "text/plain",
    ".csv": "text/csv",
    ".json": "application/json",
    ".png": "image/png",
    ".sif": "text/tab-separated-values",
    ".xml": "application/xml",
}


def artifact_file_summary(
    path: str | Path,
    *,
    session_id: str,
) -> ArtifactFileSummary:
    """Build serialization-safe metadata for an artifact path."""
    artifact_path = Path(path)
    try:
        size_bytes = artifact_path.stat().st_size
    except OSError:
        size_bytes = None
    suffix = artifact_path.suffix.lower()
    return ArtifactFileSummary(
        session_id=session_id,
        name=artifact_path.name,
        path=str(artifact_path),
        suffix=suffix,
        media_type=_ARTIFACT_MEDIA_TYPES.get(suffix),
        size_bytes=size_bytes,
    )


def structured_report(
    text: str,
    payload: StructuredOutputModel,
) -> CallToolResult:
    """Return model-readable text plus validated application data."""
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=payload.model_dump(mode="json"),
    )
