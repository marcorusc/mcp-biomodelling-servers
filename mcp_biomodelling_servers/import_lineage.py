"""Mutable source provenance, deliberately separate from accepted handoff v1.

Sidecars record byte identity, not biological equivalence or approval. Historical
source digests are evidence at import time; only the sidecar's exported artifact
is verified when it is used as a new parent.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from .handoff import (
    MAX_HANDOFF_MANIFEST_BYTES,
    ArtifactRole,
    HandoffArtifact,
    handoff_artifact,
    load_handoff_manifest,
    manifest_artifacts,
    sha256_file,
    verify_handoff_artifact,
)
from .structured_outputs import StructuredOutputModel
from .timestamps import utc_now


class SourceImport(StructuredOutputModel):
    source: HandoffArtifact
    recorded_at: AwareDatetime = Field(default_factory=utc_now)
    relationship: Literal["standalone", "byte-identical", "edited-derivative"]
    parent_manifest: HandoffArtifact | None = None
    derived_from: HandoffArtifact | None = None

    @model_validator(mode="after")
    def validate_relationship(self) -> SourceImport:
        if self.source.role not in ("neko_sif", "neko_bnet"):
            raise ValueError("Source lineage requires a SIF or BNET source.")
        if self.relationship == "standalone":
            if self.parent_manifest is not None or self.derived_from is not None:
                raise ValueError("Standalone imports cannot declare a parent.")
        else:
            if self.parent_manifest is None or self.derived_from is None:
                raise ValueError("Derived imports require both parent references.")
            if self.parent_manifest.role != "parent_manifest":
                raise ValueError("Invalid parent manifest role.")
            if (self.parent_manifest.server, self.parent_manifest.session_id) != (
                self.derived_from.server, self.derived_from.session_id
            ):
                raise ValueError("Parent manifest and ancestor ownership differ.")
            if self.derived_from.suffix != self.source.suffix:
                raise ValueError("Derived imports require the same source format.")
            identical = self.source.sha256 == self.derived_from.sha256
            if identical != (self.relationship == "byte-identical"):
                raise ValueError("Import relationship does not match source digests.")
        return self


class SourceExport(StructuredOutputModel):
    schema_name: Literal["mcp-biomodelling-source-lineage"] = "mcp-biomodelling-source-lineage"
    schema_version: Literal["1.0"] = "1.0"
    created_at: AwareDatetime = Field(default_factory=utc_now)
    artifact: HandoffArtifact
    imports: list[SourceImport] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_import_ownership(self) -> SourceExport:
        for imported in self.imports:
            if (imported.source.server, imported.source.session_id) != (
                self.artifact.server, self.artifact.session_id
            ):
                raise ValueError("Export and active import ownership differ.")
        return self


class SourceImportRecord(StructuredOutputModel):
    """Durable historical import evidence, independent of later mutable exports."""

    schema_name: Literal["mcp-biomodelling-source-import"] = "mcp-biomodelling-source-import"
    schema_version: Literal["1.0"] = "1.0"
    imported: SourceImport


def _write_record(destination: Path, record: StructuredOutputModel, *, overwrite: bool) -> Path:
    """Publish complete JSON atomically; immutable records use no-overwrite links."""
    data = record.model_dump_json(indent=2) + "\n"
    descriptor, name = tempfile.mkstemp(prefix=".source-record-", suffix=".tmp", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_source_import(directory: Path, imported: SourceImport) -> Path:
    """Persist a unique import event before acknowledging an imported model."""
    return _write_record(
        directory / f"source-import.{uuid.uuid4().hex}.json",
        SourceImportRecord(imported=imported), overwrite=False,
    )


def capture_import(
    path: str | Path, *, server: Literal["NeKo", "MaBoSS"], session_id: str,
    parent_manifest: str | None = None, derived_from: str | None = None,
) -> SourceImport:
    """Verify a parent before recording identity or explicitly declared derivation.

    With a parent alone, imported bytes must match a same-format parent artifact.
    For edited bytes, derived_from must name that exact parent artifact path.
    """
    path = Path(path).resolve()
    roles: dict[str, ArtifactRole] = {".sif": "neko_sif", ".bnet": "neko_bnet"}
    if path.suffix.lower() not in roles:
        raise ValueError("Source imports support SIF and BNET files only.")
    source = handoff_artifact(path, server=server, session_id=session_id,
                              role=roles[path.suffix.lower()])
    if parent_manifest is None:
        if derived_from is not None:
            raise ValueError("derived_from requires a verified parent_manifest.")
        return SourceImport(source=source, relationship="standalone")
    parent = Path(parent_manifest).resolve()
    if parent.stat().st_size > MAX_HANDOFF_MANIFEST_BYTES:
        raise ValueError("Parent manifest exceeds size limit.")
    parent_digest = sha256_file(parent)
    data = json.loads(parent.read_bytes())
    if not isinstance(data, dict):
        raise ValueError("Parent manifest must be a JSON object.")
    if data.get("schema_name") == "mcp-biomodelling-source-lineage":
        exported = SourceExport.model_validate(data)
        verify_handoff_artifact(exported.artifact)
        candidates = [exported.artifact]
        parent_server, parent_session = exported.artifact.server, exported.artifact.session_id
    else:
        manifest = load_handoff_manifest(parent, verify_artifacts=True)
        candidates = list(manifest_artifacts(manifest))
        parent_server, parent_session = manifest.source.server, manifest.source.session_id
    candidates = [a for a in candidates if a.suffix == source.suffix]
    if derived_from is not None:
        candidates = [a for a in candidates if Path(a.path) == Path(derived_from).resolve()]
    else:
        candidates = [a for a in candidates if a.sha256 == source.sha256]
    if len(candidates) != 1:
        raise ValueError("Import does not match a unique parent artifact; edited bytes require derived_from.")
    reference = handoff_artifact(parent, server=parent_server,
                                session_id=parent_session, role="parent_manifest")
    if reference.sha256 != parent_digest:
        raise ValueError("Parent manifest changed during verification.")
    ancestor = candidates[0]
    return SourceImport(source=source, parent_manifest=reference, derived_from=ancestor,
                        relationship="byte-identical" if source.sha256 == ancestor.sha256
                        else "edited-derivative")


def write_source_export(path: str | Path, *, server: Literal["NeKo", "MaBoSS"],
                        session_id: str, imports: list[SourceImport],
                        overwrite: bool = True) -> Path:
    """Write a sibling provenance sidecar without modifying handoff contracts."""
    path = Path(path)
    roles: dict[str, ArtifactRole] = {
        ".sif": "neko_sif", ".bnet": "neko_bnet", ".bnd": "maboss_bnd",
        ".cfg": "maboss_cfg", ".json": "parent_manifest",
    }
    role = roles[path.suffix.lower()]
    record = SourceExport(artifact=handoff_artifact(path, server=server,
                          session_id=session_id, role=role), imports=imports)
    destination = path.with_name(path.name + ".lineage.json")
    return _write_record(destination, record, overwrite=overwrite)
