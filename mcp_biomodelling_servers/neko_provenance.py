"""Typed binding of a NeKo history to a published model, independent of approval."""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import Field, model_validator

from .handoff import HandoffArtifact, verify_handoff_artifact
from .structured_outputs import StructuredOutputModel

MAX_NEKO_HISTORY_BYTES = 16 * 1024 * 1024


class NeKoStateIdentity(StructuredOutputModel):
    session_id: str = Field(min_length=1)
    network_id: str = Field(min_length=1)
    state_id: int = Field(ge=0)
    database_name: str | None


class NeKoExportBinding(StructuredOutputModel):
    artifact: HandoffArtifact
    node_renames: dict[str, str] = Field(default_factory=dict)
    library_projection: dict[str, Any] | None = None


class NeKoExportEvidence(StructuredOutputModel):
    # NeKo.from_history accepts this companion and restores its history payload.
    format: Literal["neko-export"] = "neko-export"
    schema_version: Literal[1] = 1
    mcp: NeKoStateIdentity
    export: NeKoExportBinding
    history: dict[str, Any]

    @model_validator(mode="after")
    def check_binding(self):
        if (self.export.artifact.server != "NeKo"
                or self.export.artifact.session_id != self.mcp.session_id
                or self.export.artifact.role not in {"neko_bnet", "neko_sif", "neko_ode_network"}):
            raise ValueError("Export/history ownership mismatch.")
        if (self.history.get("format") != "neko-history"
                or self.history.get("schema_version") != 1
                or self.history.get("current_state_id") != self.mcp.state_id
                or self.history.get("database_name") != self.mcp.database_name):
            raise ValueError("Export/history state or database identity mismatch.")
        if self.history.get("working_state_matches_current") is not True:
            raise ValueError("Export requires a saved state matching the working network.")
        projection = self.export.library_projection
        if projection is not None and (
            projection.get("sha256") != self.export.artifact.sha256
            or projection.get("file") != self.export.artifact.name
            or projection.get("state_id") != self.mcp.state_id
        ):
            raise ValueError("Library projection differs from the published artifact/state.")
        return self


def parse_neko_provenance(raw: bytes, reference: HandoffArtifact,
                          artifact: HandoffArtifact) -> NeKoExportEvidence:
    """Validate live or snapshotted bytes without opening historical paths."""
    if len(raw) > MAX_NEKO_HISTORY_BYTES:
        raise ValueError("NeKo provenance exceeds 16 MiB.")
    if (len(raw) != reference.size_bytes
            or hashlib.sha256(raw).hexdigest() != reference.sha256):
        raise ValueError("NeKo provenance digest/size mismatch.")
    evidence = NeKoExportEvidence.model_validate_json(raw)
    if (reference.role != "neko_history" or reference.server != "NeKo"
            or reference.session_id != evidence.mcp.session_id
            or evidence.export.artifact != artifact):
        raise ValueError("NeKo provenance is not bound to this model artifact.")
    return evidence


def verify_neko_provenance(reference: HandoffArtifact,
                           artifact: HandoffArtifact) -> NeKoExportEvidence:
    path = verify_handoff_artifact(reference)
    verify_handoff_artifact(artifact)
    return parse_neko_provenance(path.read_bytes(), reference, artifact)
