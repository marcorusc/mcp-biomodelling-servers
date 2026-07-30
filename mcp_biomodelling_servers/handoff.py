"""Versioned cross-server model handoff contracts and integrity helpers.

The handoff documents defined here are deliberately independent of MCP wire
protocol versions.  They describe modelling artifacts passed between the NeKo,
MaBoSS, and PhysiCell servers and retain enough provenance to reject stale or
incompatible files at each boundary.
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    AwareDatetime,
    Field,
    StringConstraints,
    TypeAdapter,
    ValidationInfo,
    field_validator,
    model_validator,
)

from mcp_biomodelling_servers.structured_outputs import (
    ArtifactFileSummary,
    StructuredOutputModel,
    artifact_file_summary,
)

HANDOFF_SCHEMA_NAME: Literal[
    "mcp-biomodelling-handoff"
] = "mcp-biomodelling-handoff"
HANDOFF_SCHEMA_VERSION: Literal["1.0"] = "1.0"
MAX_HANDOFF_MANIFEST_BYTES = 1024 * 1024

ServerName: TypeAlias = Literal["NeKo", "MaBoSS", "PhysiCell"]
HandoffType: TypeAlias = Literal["neko-to-maboss", "maboss-to-physicell"]
ArtifactRole: TypeAlias = Literal[
    "neko_bnet",
    "maboss_bnd",
    "maboss_cfg",
    "maboss_result",
    "parent_manifest",
]
NonEmptyString: TypeAlias = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
JsonScalar: TypeAlias = bool | int | float | str | None

_ROLE_SUFFIXES: dict[ArtifactRole, str] = {
    "neko_bnet": ".bnet",
    "maboss_bnd": ".bnd",
    "maboss_cfg": ".cfg",
    "maboss_result": ".csv",
    "parent_manifest": ".json",
}
_ROLE_MEDIA_TYPES: dict[ArtifactRole, str] = {
    "neko_bnet": "text/plain",
    "maboss_bnd": "text/plain",
    "maboss_cfg": "text/plain",
    "maboss_result": "text/csv",
    "parent_manifest": "application/json",
}
_MODELLING_PACKAGES: dict[ServerName, str] = {
    "NeKo": "nekomata",
    "MaBoSS": "maboss",
    "PhysiCell": "physicell-settings",
}


def _utc_now() -> datetime:
    """Return an aware UTC timestamp for manifest defaults."""
    return datetime.now(timezone.utc)


def _require_unique(values: list[str], field_name: str) -> list[str]:
    """Reject duplicate values in an ordered manifest collection."""
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicate values.")
    return values


class HandoffPackage(StructuredOutputModel):
    """One package and the exact version used for a modelling operation."""

    name: NonEmptyString
    version: NonEmptyString


class HandoffProvenance(StructuredOutputModel):
    """Server, session, package, and operation provenance for one handoff step."""

    server: ServerName
    session_id: NonEmptyString
    mcp_package: HandoffPackage
    modelling_package: HandoffPackage
    operation: NonEmptyString
    recorded_at: AwareDatetime = Field(default_factory=_utc_now)

    @field_validator("recorded_at")
    @classmethod
    def validate_recorded_at_utc(cls, value: datetime) -> datetime:
        """Require provenance timestamps to use UTC explicitly."""
        if value.utcoffset() != timedelta(0):
            raise ValueError("recorded_at must use the UTC timezone.")
        return value

    @model_validator(mode="after")
    def validate_package_identity(self) -> HandoffProvenance:
        """Keep package identity consistent with the declared source server."""
        if self.mcp_package.name != "mcp-biomodelling-servers":
            raise ValueError(
                "mcp_package.name must be 'mcp-biomodelling-servers'."
            )
        expected_package = _MODELLING_PACKAGES[self.server]
        if self.modelling_package.name != expected_package:
            raise ValueError(
                f"{self.server} provenance requires modelling package "
                f"{expected_package!r}."
            )
        return self


class HandoffArtifact(ArtifactFileSummary):
    """Integrity-protected reference to an artifact owned by one server session."""

    server: ServerName
    role: ArtifactRole
    media_type: NonEmptyString
    size_bytes: int = Field(ge=0)
    sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Lowercase SHA-256 digest of the complete artifact.",
    )

    @model_validator(mode="after")
    def validate_path_metadata(self) -> HandoffArtifact:
        """Require self-consistent absolute paths, names, suffixes, and roles."""
        artifact_path = Path(self.path)
        if not artifact_path.is_absolute():
            raise ValueError("Handoff artifact paths must be absolute.")
        if self.name != artifact_path.name:
            raise ValueError("Artifact name does not match its path basename.")

        actual_suffix = artifact_path.suffix.lower()
        if self.suffix != actual_suffix:
            raise ValueError("Artifact suffix does not match its path.")

        expected_suffix = _ROLE_SUFFIXES[self.role]
        if self.suffix != expected_suffix:
            raise ValueError(
                f"Artifact role {self.role!r} requires suffix "
                f"{expected_suffix!r}."
            )
        expected_media_type = _ROLE_MEDIA_TYPES[self.role]
        if self.media_type != expected_media_type:
            raise ValueError(
                f"Artifact role {self.role!r} requires media type "
                f"{expected_media_type!r}."
            )
        return self


class HandoffNetwork(StructuredOutputModel):
    """Boolean-network identifiers shared across modelling servers."""

    nodes: list[NonEmptyString] = Field(min_length=1)
    output_nodes: list[NonEmptyString] = Field(default_factory=list)
    renamed_nodes: list[NonEmptyString] = Field(default_factory=list)
    duplicate_rules_removed: list[NonEmptyString] = Field(default_factory=list)

    @field_validator(
        "nodes",
        "output_nodes",
        "renamed_nodes",
        "duplicate_rules_removed",
    )
    @classmethod
    def validate_unique_names(
        cls,
        value: list[str],
        info: ValidationInfo,
    ) -> list[str]:
        """Keep every ordered identifier collection unambiguous."""
        return _require_unique(value, info.field_name)

    @model_validator(mode="after")
    def validate_output_nodes(self) -> HandoffNetwork:
        """Require every declared output to exist in the network."""
        unknown = sorted(set(self.output_nodes) - set(self.nodes))
        if unknown:
            raise ValueError(
                "Output nodes are absent from the network: "
                + ", ".join(unknown)
            )
        return self


class MaBoSSSimulationHandoff(StructuredOutputModel):
    """MaBoSS settings and concise results retained for PhysiCell integration."""

    parameters: dict[NonEmptyString, JsonScalar] = Field(default_factory=dict)
    simulation_summary: NonEmptyString | None = None
    result_file: HandoffArtifact | None = None

    @field_validator("parameters")
    @classmethod
    def validate_finite_parameters(
        cls,
        value: dict[str, JsonScalar],
    ) -> dict[str, JsonScalar]:
        """Reject non-finite numbers that are not portable JSON values."""
        invalid = sorted(
            key
            for key, parameter in value.items()
            if isinstance(parameter, float) and not math.isfinite(parameter)
        )
        if invalid:
            raise ValueError(
                "MaBoSS parameters must be finite JSON values: "
                + ", ".join(invalid)
            )
        return value

    @model_validator(mode="after")
    def validate_result_artifact(self) -> MaBoSSSimulationHandoff:
        """Accept only a MaBoSS result artifact in the result slot."""
        if self.result_file is not None and self.result_file.role != "maboss_result":
            raise ValueError("result_file must have role 'maboss_result'.")
        return self


class PhysiCellTarget(StructuredOutputModel):
    """Destination selected for a MaBoSS model in a PhysiCell configuration."""

    cell_type: NonEmptyString


class HandoffManifestBase(StructuredOutputModel):
    """Fields shared by every version 1.0 model handoff document."""

    schema_name: Literal["mcp-biomodelling-handoff"] = HANDOFF_SCHEMA_NAME
    schema_version: Literal["1.0"] = HANDOFF_SCHEMA_VERSION
    handoff_type: HandoffType
    created_at: AwareDatetime = Field(default_factory=_utc_now)
    source: HandoffProvenance
    lineage: list[HandoffProvenance] = Field(default_factory=list)
    biological_context: NonEmptyString | None = None
    network: HandoffNetwork

    @field_validator("created_at")
    @classmethod
    def validate_created_at_utc(cls, value: datetime) -> datetime:
        """Require manifest timestamps to use UTC explicitly."""
        if value.utcoffset() != timedelta(0):
            raise ValueError("created_at must use the UTC timezone.")
        return value


class NeKoToMaBoSSHandoffManifest(HandoffManifestBase):
    """Manifest produced from a NeKo network for conversion by MaBoSS."""

    handoff_type: Literal["neko-to-maboss"] = "neko-to-maboss"
    bnet_file: HandoffArtifact

    @model_validator(mode="after")
    def validate_neko_source(self) -> NeKoToMaBoSSHandoffManifest:
        """Require the BNET artifact to belong to the declared NeKo source."""
        if self.source.server != "NeKo":
            raise ValueError("A NeKo handoff must declare NeKo as its source.")
        if self.bnet_file.server != "NeKo":
            raise ValueError("A NeKo BNET artifact must be owned by NeKo.")
        if self.bnet_file.role != "neko_bnet":
            raise ValueError("bnet_file must have role 'neko_bnet'.")
        if self.bnet_file.session_id != self.source.session_id:
            raise ValueError(
                "The BNET artifact session does not match the NeKo source."
            )
        if self.lineage:
            raise ValueError(
                "A NeKo-to-MaBoSS manifest cannot declare upstream lineage."
            )
        return self


class MaBoSSToPhysiCellHandoffManifest(HandoffManifestBase):
    """Manifest produced from a MaBoSS model for PhysiCell integration."""

    handoff_type: Literal["maboss-to-physicell"] = "maboss-to-physicell"
    bnd_file: HandoffArtifact
    cfg_file: HandoffArtifact
    parent_manifest: HandoffArtifact | None = None
    simulation: MaBoSSSimulationHandoff
    target: PhysiCellTarget

    @model_validator(mode="after")
    def validate_maboss_source(self) -> MaBoSSToPhysiCellHandoffManifest:
        """Require a consistent MaBoSS pair and optional NeKo parent lineage."""
        if self.source.server != "MaBoSS":
            raise ValueError(
                "A MaBoSS handoff must declare MaBoSS as its source."
            )

        for artifact, role in (
            (self.bnd_file, "maboss_bnd"),
            (self.cfg_file, "maboss_cfg"),
        ):
            if artifact.server != "MaBoSS" or artifact.role != role:
                raise ValueError(
                    f"{role} must be owned by the MaBoSS source."
                )
            if artifact.session_id != self.source.session_id:
                raise ValueError(
                    f"{role} session does not match the MaBoSS source."
                )

        if self.bnd_file.path == self.cfg_file.path:
            raise ValueError("BND and CFG artifacts must be different files.")

        if self.simulation.result_file is not None:
            result_file = self.simulation.result_file
            if result_file.server != "MaBoSS":
                raise ValueError(
                    "The simulation result must be owned by MaBoSS."
                )
            if result_file.session_id != self.source.session_id:
                raise ValueError(
                    "The simulation result session does not match the source."
                )

        if self.parent_manifest is None:
            if self.lineage:
                raise ValueError(
                    "Upstream lineage requires a parent_manifest reference."
                )
        else:
            if self.parent_manifest.server != "NeKo":
                raise ValueError(
                    "The parent manifest must be owned by NeKo."
                )
            if self.parent_manifest.role != "parent_manifest":
                raise ValueError(
                    "parent_manifest must have role 'parent_manifest'."
                )
            if len(self.lineage) != 1 or self.lineage[0].server != "NeKo":
                raise ValueError(
                    "A NeKo parent requires exactly one NeKo lineage entry."
                )
            if self.parent_manifest.session_id != self.lineage[0].session_id:
                raise ValueError(
                    "The parent manifest session does not match NeKo lineage."
                )
        return self


ModelHandoffManifest: TypeAlias = Annotated[
    NeKoToMaBoSSHandoffManifest | MaBoSSToPhysiCellHandoffManifest,
    Field(discriminator="handoff_type"),
]
_HANDOFF_ADAPTER = TypeAdapter(ModelHandoffManifest)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a regular file without loading it in memory."""
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact does not exist: {artifact_path}")
    if not artifact_path.is_file():
        raise ValueError(f"Artifact is not a regular file: {artifact_path}")

    before = artifact_path.stat()
    digest = hashlib.sha256()
    with artifact_path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    after = artifact_path.stat()

    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise RuntimeError(
            f"Artifact changed while its digest was calculated: {artifact_path}"
        )
    return digest.hexdigest()


def handoff_artifact(
    path: str | Path,
    *,
    server: ServerName,
    session_id: str,
    role: ArtifactRole,
) -> HandoffArtifact:
    """Build an absolute, integrity-protected handoff artifact reference."""
    artifact_path = Path(path).resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact does not exist: {artifact_path}")
    if not artifact_path.is_file():
        raise ValueError(f"Artifact is not a regular file: {artifact_path}")

    summary = artifact_file_summary(
        artifact_path,
        session_id=session_id,
    )
    if summary.size_bytes is None or summary.media_type is None:
        raise ValueError(
            f"Artifact metadata is unavailable or unsupported: {artifact_path}"
        )
    digest = sha256_file(artifact_path)
    if artifact_path.stat().st_size != summary.size_bytes:
        raise RuntimeError(
            f"Artifact changed while its handoff metadata was created: "
            f"{artifact_path}"
        )
    return HandoffArtifact(
        **summary.model_dump(),
        server=server,
        role=role,
        sha256=digest,
    )


def verify_handoff_artifact(artifact: HandoffArtifact) -> Path:
    """Verify that a referenced artifact still matches its recorded metadata."""
    artifact_path = Path(artifact.path)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Handoff artifact no longer exists: {artifact_path}"
        )
    if not artifact_path.is_file():
        raise ValueError(
            f"Handoff artifact is not a regular file: {artifact_path}"
        )

    current_size = artifact_path.stat().st_size
    if current_size != artifact.size_bytes:
        raise ValueError(
            f"Handoff artifact size changed for {artifact_path}: "
            f"expected {artifact.size_bytes}, found {current_size}."
        )

    current_digest = sha256_file(artifact_path)
    if current_digest != artifact.sha256:
        raise ValueError(
            f"Handoff artifact digest changed for {artifact_path}."
        )
    return artifact_path


def manifest_artifacts(
    manifest: ModelHandoffManifest,
) -> tuple[HandoffArtifact, ...]:
    """Return every artifact reference carried by a handoff manifest."""
    if isinstance(manifest, NeKoToMaBoSSHandoffManifest):
        return (manifest.bnet_file,)

    artifacts = [manifest.bnd_file, manifest.cfg_file]
    if manifest.parent_manifest is not None:
        artifacts.append(manifest.parent_manifest)
    if manifest.simulation.result_file is not None:
        artifacts.append(manifest.simulation.result_file)
    return tuple(artifacts)


def verify_handoff_manifest(
    manifest: ModelHandoffManifest,
) -> ModelHandoffManifest:
    """Verify every artifact referenced by an already parsed manifest."""
    for artifact in manifest_artifacts(manifest):
        verify_handoff_artifact(artifact)
    return manifest


def load_handoff_manifest(
    path: str | Path,
    *,
    expected_handoff_type: HandoffType | None = None,
    verify_artifacts: bool = True,
) -> ModelHandoffManifest:
    """Load, validate, and optionally verify a handoff manifest from JSON."""
    manifest_path = Path(path).resolve()
    if manifest_path.suffix.lower() != ".json":
        raise ValueError("Handoff manifests must use the .json suffix.")
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Handoff manifest does not exist: {manifest_path}"
        )
    if not manifest_path.is_file():
        raise ValueError(
            f"Handoff manifest is not a regular file: {manifest_path}"
        )

    size_bytes = manifest_path.stat().st_size
    if size_bytes > MAX_HANDOFF_MANIFEST_BYTES:
        raise ValueError(
            f"Handoff manifest exceeds the "
            f"{MAX_HANDOFF_MANIFEST_BYTES}-byte limit."
        )

    try:
        manifest = _HANDOFF_ADAPTER.validate_json(manifest_path.read_bytes())
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(
            f"Invalid handoff manifest {manifest_path}: {exc}"
        ) from exc

    if (
        expected_handoff_type is not None
        and manifest.handoff_type != expected_handoff_type
    ):
        raise ValueError(
            f"Expected handoff type {expected_handoff_type!r}, found "
            f"{manifest.handoff_type!r}."
        )

    if verify_artifacts:
        verify_handoff_manifest(manifest)
    return manifest


def write_handoff_manifest(
    path: str | Path,
    manifest: ModelHandoffManifest,
    *,
    overwrite: bool = False,
) -> Path:
    """Write one validated manifest as readable JSON."""
    manifest_path = Path(path).resolve()
    if manifest_path.suffix.lower() != ".json":
        raise ValueError("Handoff manifests must use the .json suffix.")
    if not manifest_path.parent.is_dir():
        raise FileNotFoundError(
            f"Handoff manifest directory does not exist: "
            f"{manifest_path.parent}"
        )
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing handoff manifest: {manifest_path}"
        )

    serialized = manifest.model_dump_json(indent=2) + "\n"
    if len(serialized.encode("utf-8")) > MAX_HANDOFF_MANIFEST_BYTES:
        raise ValueError(
            f"Handoff manifest exceeds the "
            f"{MAX_HANDOFF_MANIFEST_BYTES}-byte limit."
        )
    manifest_path.write_text(serialized, encoding="utf-8")
    return manifest_path
