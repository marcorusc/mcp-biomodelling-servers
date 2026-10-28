"""Portable handoff bundles for servers that do not share a filesystem.

A bundle carries the exact bytes of a handoff manifest and of every artifact it
references, transitively, keyed by SHA-256. Manifests are never rewritten: the
importer stores each verified blob under its digest and the consuming server
resolves recorded artifact paths it cannot see to those local copies.
"""

from __future__ import annotations

import base64
import binascii
import gzip
import hashlib
import json
import os
import tempfile
import zlib
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, StringConstraints, ValidationError

from .handoff import (
    _HANDOFF_ADAPTER,
    MAX_V2_HANDOFF_MANIFEST_BYTES,
    HandoffArtifact,
    MaBoSSToPhysiCellHandoffManifest,
    NeKoToMaBoSSHandoffManifest,
    load_handoff_manifest,
    local_handoff_blob_path,
    manifest_artifacts,
    verify_handoff_artifact,
)
from .ode_handoff import NeKoToBioMASSHandoffManifest, read_ode_handoff
from .structured_outputs import StructuredOutputModel

HANDOFF_BUNDLE_SCHEMA_NAME: Literal[
    "mcp-biomodelling-handoff-bundle"
] = "mcp-biomodelling-handoff-bundle"
MAX_HANDOFF_BUNDLE_BYTES = 64 * 1024 * 1024
MAX_HANDOFF_BUNDLE_BLOBS = 64

BundleHandoffType: TypeAlias = Literal[
    "neko-to-maboss", "maboss-to-physicell", "neko-to-biomass"
]
Sha256: TypeAlias = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
AnyHandoffManifest: TypeAlias = (
    NeKoToMaBoSSHandoffManifest
    | MaBoSSToPhysiCellHandoffManifest
    | NeKoToBioMASSHandoffManifest
)


class HandoffBundle(StructuredOutputModel):
    """Self-contained, content-addressed transfer form of one typed handoff."""

    schema_name: Literal["mcp-biomodelling-handoff-bundle"] = (
        HANDOFF_BUNDLE_SCHEMA_NAME
    )
    schema_version: Literal["1.0"] = "1.0"
    server: Literal["NeKo", "MaBoSS"] = Field(description="Server that produced the bundle.")
    handoff_type: BundleHandoffType
    manifest_name: str = Field(
        min_length=6,
        max_length=255,
        pattern=r"^[^/\\]+\.json$",
        description="Basename of the producer's manifest file.",
    )
    manifest_sha256: Sha256 = Field(
        description="SHA-256 of the exact manifest bytes carried in blobs.",
    )
    encoding: Literal["gzip+base64"] = "gzip+base64"
    blobs: dict[Sha256, str] = Field(
        min_length=1,
        max_length=MAX_HANDOFF_BUNDLE_BLOBS,
        description=(
            "Manifest and referenced artifact bytes, gzip-compressed and "
            "base64-encoded, keyed by the SHA-256 of the uncompressed bytes."
        ),
    )


class NeKoHandoffBundle(HandoffBundle):
    """Handoff bundle produced by NeKo."""

    server: Literal["NeKo"] = "NeKo"


class MaBoSSHandoffBundle(HandoffBundle):
    """Handoff bundle produced by MaBoSS."""

    server: Literal["MaBoSS"] = "MaBoSS"


_PRODUCERS = {
    "neko-to-maboss": "NeKo",
    "neko-to-biomass": "NeKo",
    "maboss-to-physicell": "MaBoSS",
}


def _parse_manifest(raw: bytes) -> AnyHandoffManifest:
    try:
        envelope = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"Handoff manifest is not valid JSON: {exc}") from exc
    if not isinstance(envelope, dict):
        raise ValueError("Handoff manifest must be a JSON object.")
    try:
        if envelope.get("handoff_type") == "neko-to-biomass":
            return NeKoToBioMASSHandoffManifest.model_validate_json(raw)
        return _HANDOFF_ADAPTER.validate_json(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid handoff manifest: {exc}") from exc


def _direct_artifacts(manifest: AnyHandoffManifest) -> tuple[HandoffArtifact, ...]:
    if isinstance(manifest, NeKoToBioMASSHandoffManifest):
        if manifest.curation_file is None:
            return (manifest.network_file,)
        return (manifest.network_file, manifest.curation_file)
    return manifest_artifacts(manifest)


def _parent_reference(manifest: AnyHandoffManifest) -> HandoffArtifact | None:
    if isinstance(manifest, MaBoSSToPhysiCellHandoffManifest):
        return manifest.parent_manifest
    return None


def _read_verified(artifact: HandoffArtifact) -> bytes:
    path = verify_handoff_artifact(artifact)
    data = path.read_bytes()
    if len(data) != artifact.size_bytes or hashlib.sha256(data).hexdigest() != artifact.sha256:
        raise ValueError(f"Handoff artifact changed while it was bundled: {path}")
    return data


def build_handoff_bundle(
    manifest_path: str | Path,
    *,
    allowed_roots: tuple[Path, ...],
) -> HandoffBundle:
    """Verify one manifest under allowed_roots and pack its complete closure."""
    path = Path(manifest_path).resolve()
    roots = tuple(Path(root).resolve() for root in allowed_roots)
    if not any(path.is_relative_to(root) for root in roots):
        raise ValueError(
            "Bundle only handoff manifests stored in this server's artifact "
            f"directory: {path}"
        )
    if path.suffix.lower() != ".json":
        raise ValueError("Handoff manifests must use the .json suffix.")
    if not path.is_file():
        raise FileNotFoundError(f"Handoff manifest does not exist: {path}")
    if path.stat().st_size > MAX_V2_HANDOFF_MANIFEST_BYTES:
        raise ValueError("Handoff manifest exceeds the size limit.")

    raw = path.read_bytes()
    manifest = _parse_manifest(raw)
    # Full verification, including curation provenance, uses the existing loaders.
    if isinstance(manifest, NeKoToBioMASSHandoffManifest):
        read_ode_handoff(str(path))
    else:
        load_handoff_manifest(path, verify_artifacts=True)
    if path.read_bytes() != raw:
        raise ValueError("Handoff manifest changed while it was bundled.")

    contents: dict[str, bytes] = {hashlib.sha256(raw).hexdigest(): raw}
    visited = set(contents)
    pending: list[AnyHandoffManifest] = [manifest]
    while pending:
        current = pending.pop()
        for artifact in _direct_artifacts(current):
            if artifact.sha256 not in contents:
                contents[artifact.sha256] = _read_verified(artifact)
        parent = _parent_reference(current)
        if parent is not None and parent.sha256 not in visited:
            visited.add(parent.sha256)
            pending.append(_parse_manifest(contents[parent.sha256]))

    total = sum(len(data) for data in contents.values())
    if total > MAX_HANDOFF_BUNDLE_BYTES:
        raise ValueError(
            f"Handoff closure is {total} bytes; bundles are limited to "
            f"{MAX_HANDOFF_BUNDLE_BYTES} bytes."
        )
    model = (
        NeKoHandoffBundle
        if _PRODUCERS[manifest.handoff_type] == "NeKo"
        else MaBoSSHandoffBundle
    )
    return model(
        handoff_type=manifest.handoff_type,
        manifest_name=path.name,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        blobs={
            digest: base64.b64encode(gzip.compress(data, mtime=0)).decode("ascii")
            for digest, data in contents.items()
        },
    )


def _decode_blobs(bundle: HandoffBundle) -> dict[str, bytes]:
    decoded: dict[str, bytes] = {}
    remaining = MAX_HANDOFF_BUNDLE_BYTES
    for digest, encoded in bundle.blobs.items():
        try:
            compressed = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"Bundle blob {digest} is not valid base64.") from exc
        inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            data = inflater.decompress(compressed, remaining + 1)
        except zlib.error as exc:
            raise ValueError(f"Bundle blob {digest} is not valid gzip data.") from exc
        if len(data) > remaining or inflater.unconsumed_tail:
            raise ValueError(
                f"Bundle content exceeds {MAX_HANDOFF_BUNDLE_BYTES} bytes."
            )
        if not inflater.eof or inflater.unused_data:
            raise ValueError(f"Bundle blob {digest} is not one complete gzip member.")
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Bundle blob {digest} does not match its digest.")
        remaining -= len(data)
        decoded[digest] = data
    return decoded


def _store_blob(digest: str, name: str, data: bytes) -> Path:
    destination = local_handoff_blob_path(digest, name)
    directory = destination.parent
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or destination.is_symlink():
        raise ValueError("Handoff blob locations cannot be symlinks.")
    if destination.is_file():
        if hashlib.sha256(destination.read_bytes()).hexdigest() == digest:
            return destination
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".blob-", suffix=".tmp", dir=directory
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        # A concurrent import of the same digest writes identical bytes.
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def materialize_handoff_bundle(
    bundle: HandoffBundle,
    *,
    expected_handoff_type: BundleHandoffType,
) -> Path:
    """Verify every blob and store the closure; return the local manifest path.

    The returned manifest is byte-identical to the producer's. Callers load it
    with the normal path-based loaders, which re-verify every artifact.
    """
    if bundle.handoff_type != expected_handoff_type:
        raise ValueError(
            f"Expected a {expected_handoff_type!r} bundle, found "
            f"{bundle.handoff_type!r}."
        )
    if bundle.server != _PRODUCERS[bundle.handoff_type]:
        raise ValueError(
            f"A {bundle.handoff_type!r} bundle must come from "
            f"{_PRODUCERS[bundle.handoff_type]}, not {bundle.server}."
        )
    blobs = _decode_blobs(bundle)
    raw = blobs.get(bundle.manifest_sha256)
    if raw is None:
        raise ValueError("Bundle does not carry its manifest bytes.")
    manifest = _parse_manifest(raw)
    if manifest.handoff_type != bundle.handoff_type:
        raise ValueError("Bundle handoff type does not match its manifest.")

    placements: dict[tuple[str, str], bytes] = {
        (bundle.manifest_sha256, bundle.manifest_name): raw
    }
    visited = {bundle.manifest_sha256}
    pending: list[AnyHandoffManifest] = [manifest]
    while pending:
        current = pending.pop()
        for artifact in _direct_artifacts(current):
            data = blobs.get(artifact.sha256)
            if data is None:
                raise ValueError(
                    f"Bundle is missing referenced artifact {artifact.name} "
                    f"({artifact.sha256})."
                )
            if len(data) != artifact.size_bytes:
                raise ValueError(f"Bundle artifact {artifact.name} has the wrong size.")
            placements[(artifact.sha256, artifact.name)] = data
        parent = _parent_reference(current)
        if parent is not None and parent.sha256 not in visited:
            visited.add(parent.sha256)
            pending.append(_parse_manifest(blobs[parent.sha256]))

    unused = set(blobs) - {digest for digest, _ in placements}
    if unused:
        raise ValueError(
            "Bundle carries blobs its manifest does not reference: "
            + ", ".join(sorted(unused))
        )
    for (digest, name), data in placements.items():
        _store_blob(digest, name, data)
    return local_handoff_blob_path(bundle.manifest_sha256, bundle.manifest_name)


def resolve_handoff_source(
    manifest_path: str | None,
    handoff_bundle: HandoffBundle | None,
    *,
    expected_handoff_type: BundleHandoffType,
) -> str:
    """Return the manifest path to load from exactly one path or bundle argument."""
    if (manifest_path is None) == (handoff_bundle is None):
        raise ValueError("Provide exactly one of manifest_path or handoff_bundle.")
    if handoff_bundle is None:
        assert manifest_path is not None
        return manifest_path
    return str(materialize_handoff_bundle(
        handoff_bundle, expected_handoff_type=expected_handoff_type,
    ))
