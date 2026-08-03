"""Versioned cross-server model handoff contracts and integrity helpers.

The handoff documents defined here are deliberately independent of MCP wire
protocol versions.  They describe modelling artifacts passed between the NeKo,
MaBoSS, and PhysiCell servers and retain enough provenance to reject stale or
incompatible files at each boundary.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
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
_BND_NODE_DECLARATION = re.compile(
    r"(?im)^[ \t]*Node[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]*\{"
)
_BND_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


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
    node_renames: dict[NonEmptyString, NonEmptyString] = Field(
        default_factory=dict,
        description="Original NeKo node name to sanitized MaBoSS node name.",
    )
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
        return _require_unique(value, info.field_name or "collection")

    @model_validator(mode="after")
    def validate_output_nodes(self) -> HandoffNetwork:
        """Require outputs and sanitizer metadata to match the final network."""
        unknown = sorted(set(self.output_nodes) - set(self.nodes))
        if unknown:
            raise ValueError(
                "Output nodes are absent from the network: "
                + ", ".join(unknown)
            )

        unknown_renames = sorted(
            set(self.node_renames.values()) - set(self.nodes)
        )
        if unknown_renames:
            raise ValueError(
                "Renamed targets are absent from the network: "
                + ", ".join(unknown_renames)
            )
        identity_renames = sorted(
            original
            for original, renamed in self.node_renames.items()
            if original == renamed
        )
        if identity_renames:
            raise ValueError(
                "Node rename mappings must change the original name: "
                + ", ".join(identity_renames)
            )
        undeclared_renames = sorted(
            set(self.node_renames) - set(self.renamed_nodes)
        )
        if undeclared_renames:
            raise ValueError(
                "Node rename mappings are missing from renamed_nodes: "
                + ", ".join(undeclared_renames)
            )
        unknown_duplicates = sorted(
            set(self.duplicate_rules_removed) - set(self.nodes)
        )
        if unknown_duplicates:
            raise ValueError(
                "Removed duplicate rules are absent from the final network: "
                + ", ".join(unknown_duplicates)
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
    history_state_id: int | None = Field(
        default=None,
        ge=0,
        description="Current NeKo history state when the BNET was exported.",
    )
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


class NeKoHandoffExportResult(StructuredOutputModel):
    """Structured result returned by NeKo ``export_neko_handoff``."""

    server: Literal["NeKo"]
    session_id: NonEmptyString
    manifest_file: HandoffArtifact
    manifest: NeKoToMaBoSSHandoffManifest

    @model_validator(mode="after")
    def validate_export_result(self) -> NeKoHandoffExportResult:
        """Keep the returned manifest artifact and source session aligned."""
        if self.manifest_file.server != "NeKo":
            raise ValueError("The handoff manifest file must be owned by NeKo.")
        if self.manifest_file.role != "parent_manifest":
            raise ValueError(
                "The handoff manifest file must have role 'parent_manifest'."
            )
        if self.manifest_file.session_id != self.session_id:
            raise ValueError(
                "The handoff manifest file session does not match the result."
            )
        if self.manifest.source.session_id != self.session_id:
            raise ValueError(
                "The NeKo manifest source session does not match the result."
            )
        return self


class MaBoSSHandoffImportResult(StructuredOutputModel):
    """Structured result returned by MaBoSS ``import_neko_handoff``."""

    server: Literal["MaBoSS"]
    session_id: NonEmptyString
    source_manifest_file: HandoffArtifact
    source_manifest: NeKoToMaBoSSHandoffManifest
    bnd_file: HandoffArtifact
    cfg_file: HandoffArtifact
    nodes: list[NonEmptyString] = Field(min_length=1)
    output_nodes: list[NonEmptyString] = Field(default_factory=list)
    requires_output_selection: bool

    @field_validator("nodes", "output_nodes")
    @classmethod
    def validate_unique_names(
        cls,
        value: list[str],
        info: ValidationInfo,
    ) -> list[str]:
        """Keep imported node collections unambiguous."""
        return _require_unique(value, info.field_name or "collection")

    @model_validator(mode="after")
    def validate_import_result(self) -> MaBoSSHandoffImportResult:
        """Align the verified NeKo parent and generated MaBoSS artifacts."""
        source_session = self.source_manifest.source.session_id
        if (
            self.source_manifest_file.server != "NeKo"
            or self.source_manifest_file.role != "parent_manifest"
        ):
            raise ValueError(
                "The source manifest file must be a NeKo parent manifest."
            )
        if self.source_manifest_file.session_id != source_session:
            raise ValueError(
                "The source manifest file session does not match NeKo provenance."
            )
        for artifact, role in (
            (self.bnd_file, "maboss_bnd"),
            (self.cfg_file, "maboss_cfg"),
        ):
            if artifact.server != "MaBoSS" or artifact.role != role:
                raise ValueError(
                    f"The imported {role} artifact must be owned by MaBoSS."
                )
            if artifact.session_id != self.session_id:
                raise ValueError(
                    f"The imported {role} session does not match the result."
                )
        if self.nodes != self.source_manifest.network.nodes:
            raise ValueError(
                "Imported MaBoSS nodes do not match the NeKo manifest."
            )
        if self.output_nodes != self.source_manifest.network.output_nodes:
            raise ValueError(
                "Applied MaBoSS outputs do not match the NeKo manifest."
            )
        if self.requires_output_selection is bool(self.output_nodes):
            raise ValueError(
                "requires_output_selection must be true exactly when no "
                "output nodes were declared."
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


class MaBoSSHandoffExportResult(StructuredOutputModel):
    """Structured result returned by MaBoSS ``export_maboss_handoff``."""

    server: Literal["MaBoSS"]
    session_id: NonEmptyString
    manifest_file: HandoffArtifact
    manifest: MaBoSSToPhysiCellHandoffManifest

    @model_validator(mode="after")
    def validate_export_result(self) -> MaBoSSHandoffExportResult:
        """Keep the returned manifest artifact and source session aligned."""
        if (
            self.manifest_file.server != "MaBoSS"
            or self.manifest_file.role != "parent_manifest"
        ):
            raise ValueError(
                "The handoff manifest file must be a MaBoSS parent manifest."
            )
        if self.manifest_file.session_id != self.session_id:
            raise ValueError(
                "The handoff manifest file session does not match the result."
            )
        if self.manifest.source.session_id != self.session_id:
            raise ValueError(
                "The MaBoSS manifest source session does not match the result."
            )
        return self


class PhysiCellHandoffImportResult(StructuredOutputModel):
    """Structured result returned by PhysiCell ``import_maboss_handoff``."""

    server: Literal["PhysiCell"]
    session_id: NonEmptyString
    source_manifest_file: HandoffArtifact
    source_manifest: MaBoSSToPhysiCellHandoffManifest
    manifest_snapshot_file: HandoffArtifact
    bnd_file: HandoffArtifact
    cfg_file: HandoffArtifact
    result_file: HandoffArtifact | None = None
    neko_manifest: NeKoToMaBoSSHandoffManifest | None = None
    neko_manifest_file: HandoffArtifact | None = None
    bnet_file: HandoffArtifact | None = None
    target_cell_type: NonEmptyString
    nodes: list[NonEmptyString] = Field(min_length=1)
    output_nodes: list[NonEmptyString] = Field(min_length=1)
    replaced_existing: bool
    context_count: int = Field(ge=1)

    @field_validator("nodes", "output_nodes")
    @classmethod
    def validate_unique_names(
        cls,
        value: list[str],
        info: ValidationInfo,
    ) -> list[str]:
        """Keep imported node collections unambiguous."""
        return _require_unique(value, info.field_name or "collection")

    @staticmethod
    def _validate_copy(
        copied: HandoffArtifact,
        source: HandoffArtifact,
        *,
        session_id: str,
        role: ArtifactRole,
    ) -> None:
        """Require a PhysiCell-owned copy to match its source byte for byte."""
        if copied.server != "PhysiCell" or copied.role != role:
            raise ValueError(
                f"The copied {role} artifact must be owned by PhysiCell."
            )
        if copied.session_id != session_id:
            raise ValueError(
                f"The copied {role} session does not match the result."
            )
        if (
            copied.size_bytes != source.size_bytes
            or copied.sha256 != source.sha256
        ):
            raise ValueError(
                f"The copied {role} artifact does not match its source."
            )

    @model_validator(mode="after")
    def validate_import_result(self) -> PhysiCellHandoffImportResult:
        """Align source provenance, local copies, and the applied cell target."""
        source_session = self.source_manifest.source.session_id
        if (
            self.source_manifest_file.server != "MaBoSS"
            or self.source_manifest_file.role != "parent_manifest"
        ):
            raise ValueError(
                "The source manifest file must be a MaBoSS parent manifest."
            )
        if self.source_manifest_file.session_id != source_session:
            raise ValueError(
                "The source manifest file session does not match MaBoSS "
                "provenance."
            )

        self._validate_copy(
            self.manifest_snapshot_file,
            self.source_manifest_file,
            session_id=self.session_id,
            role="parent_manifest",
        )
        self._validate_copy(
            self.bnd_file,
            self.source_manifest.bnd_file,
            session_id=self.session_id,
            role="maboss_bnd",
        )
        self._validate_copy(
            self.cfg_file,
            self.source_manifest.cfg_file,
            session_id=self.session_id,
            role="maboss_cfg",
        )

        source_result = self.source_manifest.simulation.result_file
        if (self.result_file is None) != (source_result is None):
            raise ValueError(
                "The copied MaBoSS result must match source result availability."
            )
        if self.result_file is not None and source_result is not None:
            self._validate_copy(
                self.result_file,
                source_result,
                session_id=self.session_id,
                role="maboss_result",
            )

        source_parent = self.source_manifest.parent_manifest
        lineage_values = (
            self.neko_manifest,
            self.neko_manifest_file,
            self.bnet_file,
        )
        if source_parent is None:
            if any(value is not None for value in lineage_values):
                raise ValueError(
                    "Standalone MaBoSS imports cannot include NeKo copies."
                )
        else:
            if any(value is None for value in lineage_values):
                raise ValueError(
                    "A NeKo parent requires copied manifest and BNET artifacts."
                )
            assert self.neko_manifest is not None
            assert self.neko_manifest_file is not None
            assert self.bnet_file is not None
            self._validate_copy(
                self.neko_manifest_file,
                source_parent,
                session_id=self.session_id,
                role="parent_manifest",
            )
            self._validate_copy(
                self.bnet_file,
                self.neko_manifest.bnet_file,
                session_id=self.session_id,
                role="neko_bnet",
            )
            if (
                self.neko_manifest.source.session_id
                != source_parent.session_id
            ):
                raise ValueError(
                    "The copied NeKo manifest does not match parent provenance."
                )

        if self.target_cell_type != self.source_manifest.target.cell_type:
            raise ValueError(
                "The imported target cell type does not match the manifest."
            )
        if self.nodes != self.source_manifest.network.nodes:
            raise ValueError(
                "Imported PhysiBoSS nodes do not match the MaBoSS manifest."
            )
        if self.output_nodes != self.source_manifest.network.output_nodes:
            raise ValueError(
                "Imported PhysiBoSS outputs do not match the MaBoSS manifest."
            )
        return self


ModelHandoffManifest: TypeAlias = Annotated[
    NeKoToMaBoSSHandoffManifest | MaBoSSToPhysiCellHandoffManifest,
    Field(discriminator="handoff_type"),
]
_HANDOFF_ADAPTER: TypeAdapter[ModelHandoffManifest] = TypeAdapter(
    ModelHandoffManifest
)


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


def bnet_node_names(path: str | Path) -> list[str]:
    """Read unique Boolean target names from a BNET file in stored order."""
    bnet_path = Path(path)
    if bnet_path.suffix.lower() != ".bnet":
        raise ValueError("Boolean network files must use the .bnet suffix.")
    if not bnet_path.exists():
        raise FileNotFoundError(f"BNET file does not exist: {bnet_path}")
    if not bnet_path.is_file():
        raise ValueError(f"BNET path is not a regular file: {bnet_path}")

    nodes: list[str] = []
    seen: set[str] = set()
    with bnet_path.open("r", encoding="utf-8") as bnet_file:
        for line_number, raw_line in enumerate(bnet_file, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower() == "targets, factors":
                continue
            if "," not in line:
                raise ValueError(
                    f"Invalid BNET rule on line {line_number}: missing comma."
                )
            node = line.split(",", 1)[0].strip()
            if not node:
                raise ValueError(
                    f"Invalid BNET rule on line {line_number}: empty target."
                )
            if node in seen:
                raise ValueError(
                    f"Duplicate BNET target {node!r} on line {line_number}."
                )
            nodes.append(node)
            seen.add(node)

    if not nodes:
        raise ValueError(f"BNET file contains no Boolean rules: {bnet_path}")
    return nodes


def bnd_node_names(path: str | Path) -> list[str]:
    """Read unique MaBoSS node declarations from a BND file in stored order."""
    bnd_path = Path(path)
    if bnd_path.suffix.lower() != ".bnd":
        raise ValueError("MaBoSS network files must use the .bnd suffix.")
    if not bnd_path.exists():
        raise FileNotFoundError(f"BND file does not exist: {bnd_path}")
    if not bnd_path.is_file():
        raise ValueError(f"BND path is not a regular file: {bnd_path}")

    try:
        text = bnd_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Could not read MaBoSS BND file {bnd_path}: {exc}") from exc

    without_blocks = _BND_BLOCK_COMMENT.sub("", text)
    without_comments = "\n".join(
        line.split("//", 1)[0].split("#", 1)[0]
        for line in without_blocks.splitlines()
    )
    nodes = [
        match.group(1)
        for match in _BND_NODE_DECLARATION.finditer(without_comments)
    ]
    if not nodes:
        raise ValueError(f"BND file contains no node declarations: {bnd_path}")
    if len(nodes) != len(set(nodes)):
        duplicates = sorted(
            node for node in set(nodes) if nodes.count(node) > 1
        )
        raise ValueError(
            "BND file contains duplicate node declarations: "
            + ", ".join(duplicates)
        )
    return nodes


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
    temporary_path: Path | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=manifest_path.parent,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as temporary:
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
        if overwrite:
            temporary_path.replace(manifest_path)
        else:
            try:
                os.link(temporary_path, manifest_path)
            except FileExistsError as exc:
                raise FileExistsError(
                    "Refusing to overwrite handoff manifest created "
                    f"concurrently: {manifest_path}"
                ) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return manifest_path
