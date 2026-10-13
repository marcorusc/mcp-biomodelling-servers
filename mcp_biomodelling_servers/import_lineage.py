"""Portable Boolean source ancestry, separate from current-stage approval.

V2 embeds integrity-checked original bytes; v1 references can be snapshotted
while available. Historical paths never authorize reads during bundle validation.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, Field, ValidationInfo, model_validator

from .handoff import (
    MAX_HANDOFF_MANIFEST_BYTES,
    MAX_V2_HANDOFF_MANIFEST_BYTES,
    SourceImport,
    AncestryBundle,
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


class SourceExport(StructuredOutputModel):
    schema_name: Literal["mcp-biomodelling-source-lineage"] = "mcp-biomodelling-source-lineage"
    schema_version: Literal["1.0", "2.0"] = "1.0"
    created_at: AwareDatetime = Field(default_factory=utc_now)
    artifact: HandoffArtifact
    imports: list[SourceImport] = Field(default_factory=list, max_length=128)

    @model_validator(mode="after")
    def validate_import_ownership(self, info: ValidationInfo) -> SourceExport:
        for imported in self.imports:
            if (imported.source.server, imported.source.session_id) != (
                self.artifact.server, self.artifact.session_id
            ):
                raise ValueError("Export and active import ownership differ.")
        if self.schema_version == "2.0":
            if "imports" not in self.model_fields_set:
                raise ValueError("v2 source exports require explicit imports.")
            if any(imported.evidence is None for imported in self.imports):
                raise ValueError("v2 source exports require ancestry evidence.")
            if not (info.context or {}).get("ancestry_snapshot"):
                verify_source_ancestry(self.imports)
        return self


class SourceImportRecord(StructuredOutputModel):
    """Durable historical import evidence, independent of later mutable exports."""

    schema_name: Literal["mcp-biomodelling-source-import"] = "mcp-biomodelling-source-import"
    schema_version: Literal["1.0", "2.0"] = "1.0"
    imported: SourceImport

    @model_validator(mode="after")
    def validate_evidence(self) -> SourceImportRecord:
        if self.schema_version == "2.0":
            verify_source_ancestry([self.imported])
        return self


def _write_record(destination: Path, record: StructuredOutputModel, *, overwrite: bool) -> Path:
    """Publish complete JSON atomically; immutable records use no-overwrite links."""
    data = record.model_dump_json(indent=2) + "\n"
    if len(data.encode("utf-8")) > MAX_V2_HANDOFF_MANIFEST_BYTES:
        raise ValueError("Source record exceeds size limit.")
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
        SourceImportRecord(schema_version="2.0", imported=snapshot_source_import(imported)),
        overwrite=False,
    )


def capture_import(
    path: str | Path, *, server: Literal["NeKo", "MaBoSS"], session_id: str,
    parent_manifest: str | None = None, derived_from: str | None = None,
    trusted_root: str | Path | None = None,
) -> SourceImport:
    """Verify a parent before recording identity or explicitly declared derivation.

    With a parent alone, imported bytes must match a same-format parent artifact.
    For edited bytes, derived_from must name that exact parent artifact path.
    """
    path = Path(path).resolve()
    root = Path(trusted_root).resolve() if trusted_root is not None else None
    if root is not None and not path.is_relative_to(root):
        raise ValueError("Source path is outside trusted root.")
    roles: dict[str, ArtifactRole] = {".sif": "neko_sif", ".bnet": "neko_bnet"}
    if path.suffix.lower() not in roles:
        raise ValueError("Source imports support SIF and BNET files only.")
    if path.stat().st_size > MAX_ANCESTRY_BYTES:
        raise ValueError("Source exceeds ancestry byte limit.")
    source = handoff_artifact(path, server=server, session_id=session_id,
                              role=roles[path.suffix.lower()])
    if parent_manifest is None:
        if derived_from is not None:
            raise ValueError("derived_from requires a verified parent_manifest.")
        return snapshot_source_import(SourceImport(source=source, relationship="standalone"), trusted_root=root)
    parent = Path(parent_manifest).resolve()
    if root is not None and not parent.is_relative_to(root):
        raise ValueError("Parent path is outside trusted root.")
    if parent.stat().st_size > MAX_V2_HANDOFF_MANIFEST_BYTES:
        raise ValueError("Parent manifest exceeds size limit.")
    parent_digest = sha256_file(parent)
    data = json.loads(parent.read_bytes())
    if not isinstance(data, dict):
        raise ValueError("Parent manifest must be a JSON object.")
    if data.get("schema_name") == "mcp-biomodelling-source-lineage":
        exported = SourceExport.model_validate(data)
        # The sidecar describes a currently exported artifact. Historical
        # snapshots do not excuse changes to that live parent at a new import.
        if root is not None and not Path(exported.artifact.path).resolve().is_relative_to(root):
            raise ValueError("Parent artifact is outside trusted root.")
        verify_handoff_artifact(exported.artifact)
        candidates = [exported.artifact]
        parent_server, parent_session = exported.artifact.server, exported.artifact.session_id
    else:
        manifest = load_handoff_manifest(parent, verify_artifacts=False)
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
    imported = SourceImport(source=source, parent_manifest=reference, derived_from=ancestor,
                        relationship="byte-identical" if source.sha256 == ancestor.sha256
                        else "edited-derivative")
    return snapshot_source_import(imported, trusted_root=root)


def write_source_export(path: str | Path, *, server: Literal["NeKo", "MaBoSS"],
                        session_id: str, imports: list[SourceImport],
                        overwrite: bool = True) -> Path:
    """Write a v2 sibling sidecar with durable, verified source snapshots."""
    path = Path(path)
    roles: dict[str, ArtifactRole] = {
        ".sif": "neko_sif", ".bnet": "neko_bnet", ".bnd": "maboss_bnd",
        ".cfg": "maboss_cfg", ".json": "parent_manifest",
    }
    role = roles[path.suffix.lower()]
    imports = [snapshot_source_import(imported) for imported in imports]
    record = SourceExport(schema_version="2.0", artifact=handoff_artifact(path, server=server,
                          session_id=session_id, role=role), imports=imports)
    destination = path.with_name(path.name + ".lineage.json")
    return _write_record(destination, record, overwrite=overwrite)


MAX_ANCESTRY_BYTES = 16 * 1024 * 1024
MAX_ANCESTRY_DEPTH = 16
MAX_ANCESTRY_BLOBS = 128


def _decode_blobs(bundle: AncestryBundle) -> dict[str, bytes]:
    result = {}
    total = 0
    for digest, encoded in bundle.blobs.items():
        if len(encoded) > (MAX_ANCESTRY_BYTES * 4 // 3 + 4):
            raise ValueError("Ancestry exceeds byte limit.")
        try:
            data = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise ValueError("Invalid ancestry base64 bytes.") from exc
        total += len(data)
        if total > MAX_ANCESTRY_BYTES:
            raise ValueError("Ancestry exceeds byte limit.")
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Ancestry blob digest mismatch.")
        result[digest] = data
    return result


class _Evidence:
    """Bounded graph traversal. File reads exist only in explicit import capture."""

    def __init__(self, *, live: bool = False, trusted_root: Path | None = None):
        self.blobs: dict[str, bytes] = {}
        self.live = live
        self.root = trusted_root.resolve() if trusted_root is not None else None
        self.visiting: set[str] = set()
        self.used: set[str] = set()
        self.steps = 0

    def add(self, digest: str, data: bytes) -> None:
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Ancestry digest mismatch.")
        self.blobs[digest] = data
        if len(self.blobs) > MAX_ANCESTRY_BLOBS or sum(map(len, self.blobs.values())) > MAX_ANCESTRY_BYTES:
            raise ValueError("Ancestry exceeds blob/byte limit.")

    def read(self, artifact: HandoffArtifact, *, allow_live: bool) -> bytes:
        self.used.add(artifact.sha256)
        data = self.blobs.get(artifact.sha256)
        # Initial live imports must verify each declared path even when another
        # file has identical bytes. Digest reuse is valid for saved snapshots,
        # but cannot establish that a newly supplied live parent still exists.
        if data is None or (self.live and allow_live):
            if not (self.live and allow_live):
                raise ValueError("Missing ancestry bytes: " + artifact.sha256)
            path = Path(artifact.path).resolve()
            if self.root is not None and not path.is_relative_to(self.root):
                raise ValueError("Ancestry path is outside trusted root.")
            if artifact.size_bytes > MAX_ANCESTRY_BYTES or path.stat().st_size > MAX_ANCESTRY_BYTES:
                raise ValueError("Ancestry exceeds byte limit.")
            with path.open("rb") as source:
                data = source.read(MAX_ANCESTRY_BYTES + 1)
            if len(data) > MAX_ANCESTRY_BYTES:
                raise ValueError("Ancestry exceeds byte limit.")
            self.add(artifact.sha256, data)
        if len(data) != artifact.size_bytes:
            raise ValueError("Ancestry artifact size mismatch.")
        return data

    def imported(self, imported: SourceImport, depth: int = 0, *, require_bundle: bool = True) -> None:
        self.steps += 1
        if depth > MAX_ANCESTRY_DEPTH or self.steps > MAX_ANCESTRY_BLOBS:
            raise ValueError("Ancestry traversal limit exceeded.")
        # Revalidate even mutable Pydantic objects passed by callers.
        imported = SourceImport.model_validate(imported.model_dump())
        if imported.evidence is None:
            if require_bundle:
                raise ValueError("Missing ancestry evidence bundle.")
        else:
            for digest, data in _decode_blobs(imported.evidence).items():
                self.add(digest, data)
        allow_live = imported.evidence is None and not require_bundle
        self.read(imported.source, allow_live=allow_live)
        if imported.parent_manifest is not None:
            ancestor = imported.derived_from
            assert ancestor is not None
            self.read(ancestor, allow_live=allow_live)
            candidates, owner, _ = self.parent(imported.parent_manifest, depth + 1, allow_live=allow_live)
            if ancestor not in candidates:
                raise ValueError("Ancestor is not an exact artifact of its parent manifest.")
            if (ancestor.server, ancestor.session_id) != owner:
                raise ValueError("Ancestor owner differs from parent identity.")

    def parent(self, reference: HandoffArtifact, depth: int, *, allow_live: bool):
        from .handoff import _HANDOFF_ADAPTER
        self.steps += 1
        if depth > MAX_ANCESTRY_DEPTH or self.steps > MAX_ANCESTRY_BLOBS:
            raise ValueError("Ancestry traversal/depth limit exceeded.")
        if reference.sha256 in self.visiting:
            raise ValueError("Ancestry cycle detected.")
        self.visiting.add(reference.sha256)
        raw = self.read(reference, allow_live=allow_live)
        if len(raw) > MAX_V2_HANDOFF_MANIFEST_BYTES:
            raise ValueError("Ancestry manifest exceeds size limit.")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Ancestry manifest must be an object.")
        if data.get("schema_version") == "1.0" and len(raw) > MAX_HANDOFF_MANIFEST_BYTES:
            raise ValueError("v1 ancestry manifest exceeds size limit.")
        if data.get("schema_name") == "mcp-biomodelling-source-lineage":
            parsed = SourceExport.model_validate(data, context={"ancestry_snapshot": True})
            candidates = [parsed.artifact]
            owner = (parsed.artifact.server, parsed.artifact.session_id)
            imports = parsed.imports
        else:
            parsed = _HANDOFF_ADAPTER.validate_python(data, context={"ancestry_snapshot": True})
            candidates = list(manifest_artifacts(parsed))
            owner = (parsed.source.server, parsed.source.session_id)
            imports = parsed.source_ancestry or []
        if (reference.server, reference.session_id) != owner:
            raise ValueError("Parent manifest owner differs from snapshot identity.")
        # v2 parents must never repair missing evidence using live references.
        for imported in imports:
            self.imported(imported, depth + 1, require_bundle=parsed.schema_version == "2.0")
        for artifact in candidates:
            self.read(artifact, allow_live=allow_live)
            if artifact.role == "parent_manifest":
                _, _, stage_parent = self.parent(artifact, depth + 1, allow_live=allow_live)
                if getattr(parsed, "handoff_type", None) == "maboss-to-physicell":
                    if getattr(stage_parent, "handoff_type", None) != "neko-to-maboss":
                        raise ValueError("Invalid ancestry stage-parent type.")
                    if parsed.lineage != [stage_parent.source]:
                        raise ValueError("Ancestry stage-parent identity mismatch.")
                    if (set(parsed.network.nodes) != set(stage_parent.network.nodes)
                            or set(parsed.network.output_nodes) != set(stage_parent.network.output_nodes)):
                        raise ValueError("Ancestry stage-parent node/output mismatch.")
                    if parsed.schema_version == "2.0":
                        inherited = [entry for entry in parsed.source_ancestry or []
                                     if entry.source.server == "NeKo"]
                        if inherited != (stage_parent.source_ancestry or []):
                            raise ValueError("Ancestry stage-parent source ancestry mismatch.")
        self.visiting.remove(reference.sha256)
        return candidates, owner, parsed

    def bundle(self) -> AncestryBundle:
        return AncestryBundle(blobs={digest: base64.b64encode(data).decode("ascii")
                                     for digest, data in self.blobs.items()})


def _verified_evidence(imports: list[SourceImport]) -> _Evidence:
    if len(imports) > MAX_ANCESTRY_BLOBS:
        raise ValueError("Ancestry import count exceeds limit.")
    combined = _Evidence()
    for imported in imports:
        evidence = _Evidence()
        evidence.imported(imported)
        if set(evidence.blobs) != evidence.used:
            raise ValueError("Ancestry contains unrelated historical bytes.")
        for digest, data in evidence.blobs.items():
            combined.add(digest, data)
    return combined


def verify_source_ancestry(imports: list[SourceImport]) -> list[SourceImport]:
    """Validate complete portable ancestry without opening any original path.

    Integrity and ancestry are not approval, authentication, or equivalence of
    the currently exported network. Gate approval must bind its current bytes.
    """
    _verified_evidence(imports)
    return imports


def copy_source_ancestry(imports: list[SourceImport], destination: str | Path) -> dict[str, Path]:
    """Materialize verified evidence using digest-only names under destination.

    Original parent paths are never read. Existing files must be identical;
    symlinks are rejected. Preserve the original records alongside these bytes.
    """
    evidence = _verified_evidence(imports)
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for digest, data in evidence.blobs.items():
        path = root / digest
        if path.is_symlink():
            raise ValueError("Ancestry destination cannot be a symlink.")
        try:
            with path.open("xb") as output:
                output.write(data)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError("Ancestry destination digest mismatch.")
        paths[digest] = path
    return paths


def snapshot_source_import(imported: SourceImport, *, trusted_root: Path | None = None) -> SourceImport:
    """Upgrade legacy references while available, or verify existing snapshots."""
    evidence = _Evidence(live=True, trusted_root=trusted_root)
    evidence.imported(imported, require_bundle=imported.evidence is not None)
    result = imported.model_copy(update={"evidence": evidence.bundle()})
    verify_source_ancestry([result])
    return result
