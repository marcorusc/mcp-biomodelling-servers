"""NeKo-to-BioMASS graph handoff, independent of Boolean-network contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .handoff import (
    MAX_HANDOFF_MANIFEST_BYTES,
    HandoffArtifact,
    HandoffProvenance,
    handoff_artifact,
    verify_handoff_artifact,
)
from .structured_outputs import StructuredOutputModel


class ODENode(StructuredOutputModel):
    node_id: str = Field(min_length=1)
    gene_symbol: str | None = None
    uniprot: str | None = None
    node_type: str | None = None


class ODEEdge(StructuredOutputModel):
    edge_id: str = Field(min_length=1)
    source: str
    target: str
    effect: str | None = None
    references: list[str] = Field(default_factory=list)
    metadata: dict[str, str | list[str] | None] = Field(default_factory=dict)


class ODENetwork(StructuredOutputModel):
    nodes: list[ODENode] = Field(min_length=1)
    edges: list[ODEEdge]

    @model_validator(mode="after")
    def check_identifiers(self) -> ODENetwork:
        nodes = [n.node_id for n in self.nodes]
        edges = [e.edge_id for e in self.edges]
        if len(nodes) != len(set(nodes)) or len(edges) != len(set(edges)):
            raise ValueError("Network node and edge IDs must be unique.")
        for edge in self.edges:
            if edge.source not in nodes or edge.target not in nodes:
                raise ValueError("Edge endpoint is missing from network nodes.")
        return self


class NeKoToBioMASSHandoffManifest(StructuredOutputModel):
    schema_name: Literal["mcp-biomodelling-handoff"] = "mcp-biomodelling-handoff"
    schema_version: Literal["1.0"] = "1.0"
    handoff_type: Literal["neko-to-biomass"] = "neko-to-biomass"
    source: HandoffProvenance
    biological_context: str = Field(min_length=1)
    history_state_id: int | None = Field(default=None, ge=0)
    network_file: HandoffArtifact

    @model_validator(mode="after")
    def ownership(self) -> NeKoToBioMASSHandoffManifest:
        if self.source.server != "NeKo" or self.network_file.server != "NeKo":
            raise ValueError("The ODE network handoff must originate in NeKo.")
        if (
            self.network_file.role != "neko_ode_network"
            or self.source.session_id != self.network_file.session_id
        ):
            raise ValueError(
                "Network artifact role/session does not match source provenance."
            )
        return self


class NeKoBioMASSHandoffExportResult(StructuredOutputModel):
    server: Literal["NeKo"] = "NeKo"
    session_id: str
    manifest_file: HandoffArtifact
    manifest: NeKoToBioMASSHandoffManifest


def read_ode_handoff(path: str) -> tuple[NeKoToBioMASSHandoffManifest, ODENetwork]:
    manifest_path = Path(path)
    if manifest_path.stat().st_size > MAX_HANDOFF_MANIFEST_BYTES:
        raise ValueError("Handoff manifest exceeds 1 MiB.")
    manifest = NeKoToBioMASSHandoffManifest.model_validate_json(
        manifest_path.read_bytes()
    )
    network_path = verify_handoff_artifact(manifest.network_file)
    if network_path.stat().st_size > 10 * MAX_HANDOFF_MANIFEST_BYTES:
        raise ValueError("Network artifact exceeds 10 MiB.")
    content = network_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != manifest.network_file.sha256:
        raise ValueError("Network artifact changed while importing.")
    return manifest, ODENetwork.model_validate_json(content)


def write_ode_handoff(
    directory: Path,
    prefix: str,
    network: ODENetwork,
    source: HandoffProvenance,
    biological_context: str,
    history_state_id: int | None,
) -> NeKoBioMASSHandoffExportResult:
    from .artifact_manager import safe_artifact_path

    network_path = safe_artifact_path(directory, prefix + ".network.json")
    manifest_path = safe_artifact_path(directory, prefix + ".handoff.json")
    if network_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "Choose a new prefix; existing handoffs cannot be overwritten."
        )
    created: list[Path] = []
    try:
        with network_path.open("x", encoding="utf-8") as handle:
            created.append(network_path)
            handle.write(network.model_dump_json(indent=2))
        manifest = NeKoToBioMASSHandoffManifest(
            source=source,
            biological_context=biological_context,
            history_state_id=history_state_id,
            network_file=handoff_artifact(
                network_path,
                server="NeKo",
                session_id=source.session_id,
                role="neko_ode_network",
            ),
        )
        with manifest_path.open("x", encoding="utf-8") as handle:
            created.append(manifest_path)
            json.dump(manifest.model_dump(mode="json"), handle, indent=2)
        return NeKoBioMASSHandoffExportResult(
            session_id=source.session_id,
            manifest=manifest,
            manifest_file=handoff_artifact(
                manifest_path,
                server="NeKo",
                session_id=source.session_id,
                role="parent_manifest",
            ),
        )
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
