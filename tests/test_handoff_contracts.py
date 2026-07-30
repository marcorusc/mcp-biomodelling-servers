"""Tests for versioned cross-server handoff contracts and file integrity."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

import mcp_biomodelling_servers.handoff as handoff_contracts
from mcp_biomodelling_servers.handoff import (
    HANDOFF_SCHEMA_VERSION,
    MAX_HANDOFF_MANIFEST_BYTES,
    HandoffArtifact,
    HandoffNetwork,
    HandoffPackage,
    HandoffProvenance,
    MaBoSSHandoffExportResult,
    MaBoSSHandoffImportResult,
    MaBoSSSimulationHandoff,
    MaBoSSToPhysiCellHandoffManifest,
    NeKoHandoffExportResult,
    NeKoToMaBoSSHandoffManifest,
    PhysiCellHandoffImportResult,
    PhysiCellTarget,
    bnd_node_names,
    bnet_node_names,
    handoff_artifact,
    load_handoff_manifest,
    manifest_artifacts,
    sha256_file,
    verify_handoff_artifact,
    verify_handoff_manifest,
    write_handoff_manifest,
)


def _package(name: str, version: str = "1.2.3") -> HandoffPackage:
    return HandoffPackage(name=name, version=version)


def _provenance(
    server: Literal["NeKo", "MaBoSS", "PhysiCell"],
    session_id: str,
) -> HandoffProvenance:
    package_names = {
        "NeKo": "nekomata",
        "MaBoSS": "maboss",
        "PhysiCell": "physicell-settings",
    }
    return HandoffProvenance(
        server=server,
        session_id=session_id,
        mcp_package=_package("mcp-biomodelling-servers", "1.0.0"),
        modelling_package=_package(package_names[server]),
        operation=f"export-from-{server.lower()}",
    )


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _neko_manifest(tmp_path: Path) -> NeKoToMaBoSSHandoffManifest:
    session_id = "neko-session"
    bnet_path = _write(
        tmp_path / "Network.bnet",
        "targets, factors\nA, B\nB, A\n",
    )
    return NeKoToMaBoSSHandoffManifest(
        source=_provenance("NeKo", session_id),
        biological_context="Investigate reciprocal A/B signalling.",
        network=HandoffNetwork(
            nodes=["A", "B"],
            output_nodes=["B"],
            renamed_nodes=["A"],
        ),
        bnet_file=handoff_artifact(
            bnet_path,
            server="NeKo",
            session_id=session_id,
            role="neko_bnet",
        ),
    )


def _maboss_manifest(
    tmp_path: Path,
    *,
    with_neko_parent: bool,
) -> MaBoSSToPhysiCellHandoffManifest:
    maboss_session = "maboss-session"
    bnd_file = handoff_artifact(
        _write(
            tmp_path / "model.bnd",
            "Node A { logic = B; }\nNode B { logic = A; }\n",
        ),
        server="MaBoSS",
        session_id=maboss_session,
        role="maboss_bnd",
    )
    cfg_file = handoff_artifact(
        _write(tmp_path / "model.cfg", "max_time = 100;\n"),
        server="MaBoSS",
        session_id=maboss_session,
        role="maboss_cfg",
    )
    result_file = handoff_artifact(
        _write(tmp_path / "result.csv", "Time,A\n100,0.75\n"),
        server="MaBoSS",
        session_id=maboss_session,
        role="maboss_result",
    )

    lineage: list[HandoffProvenance] = []
    parent_manifest: HandoffArtifact | None = None
    if with_neko_parent:
        neko_manifest = _neko_manifest(tmp_path)
        parent_path = write_handoff_manifest(
            tmp_path / "neko-handoff.json",
            neko_manifest,
        )
        lineage = [neko_manifest.source]
        parent_manifest = handoff_artifact(
            parent_path,
            server="NeKo",
            session_id=neko_manifest.source.session_id,
            role="parent_manifest",
        )

    return MaBoSSToPhysiCellHandoffManifest(
        source=_provenance("MaBoSS", maboss_session),
        lineage=lineage,
        biological_context="Connect Boolean activation to an epithelial cell.",
        network=HandoffNetwork(nodes=["A", "B"], output_nodes=["A"]),
        bnd_file=bnd_file,
        cfg_file=cfg_file,
        parent_manifest=parent_manifest,
        simulation=MaBoSSSimulationHandoff(
            parameters={"max_time": 100.0, "sample_count": 10000},
            simulation_summary="A is active with probability 0.75.",
            result_file=result_file,
        ),
        target=PhysiCellTarget(cell_type="epithelial"),
    )


def test_neko_manifest_round_trip_and_integrity(tmp_path: Path) -> None:
    manifest = _neko_manifest(tmp_path)
    manifest_path = write_handoff_manifest(
        tmp_path / "handoff.json",
        manifest,
    )

    loaded = load_handoff_manifest(
        manifest_path,
        expected_handoff_type="neko-to-maboss",
    )

    assert loaded == manifest
    assert loaded.schema_version == HANDOFF_SCHEMA_VERSION
    assert manifest_path.read_text(encoding="utf-8").endswith("\n")
    assert manifest_artifacts(loaded) == (manifest.bnet_file,)


def test_maboss_manifest_supports_standalone_models(tmp_path: Path) -> None:
    manifest = _maboss_manifest(tmp_path, with_neko_parent=False)
    manifest_path = write_handoff_manifest(
        tmp_path / "standalone.json",
        manifest,
    )

    loaded = load_handoff_manifest(
        manifest_path,
        expected_handoff_type="maboss-to-physicell",
    )

    assert isinstance(loaded, MaBoSSToPhysiCellHandoffManifest)
    assert loaded.lineage == []
    assert loaded.parent_manifest is None
    assert len(manifest_artifacts(loaded)) == 3


def test_maboss_manifest_preserves_verified_neko_lineage(
    tmp_path: Path,
) -> None:
    manifest = _maboss_manifest(tmp_path, with_neko_parent=True)
    manifest_path = write_handoff_manifest(
        tmp_path / "maboss-handoff.json",
        manifest,
    )

    loaded = load_handoff_manifest(manifest_path)

    assert isinstance(loaded, MaBoSSToPhysiCellHandoffManifest)
    assert loaded.lineage[0].server == "NeKo"
    assert loaded.parent_manifest is not None
    assert len(manifest_artifacts(loaded)) == 4


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    artifact_path = _write(tmp_path / "known.bnet", "A, B\n")

    assert sha256_file(artifact_path) == hashlib.sha256(b"A, B\n").hexdigest()


def test_bnet_node_names_reads_sanitized_targets_in_stored_order(
    tmp_path: Path,
) -> None:
    bnet_path = _write(
        tmp_path / "Network.bnet",
        "# comment\n"
        "targets, factors\n"
        "A_1, B\n"
        "\n"
        "B, A_1\n",
    )

    assert bnet_node_names(bnet_path) == ["A_1", "B"]


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("not-a-rule\n", "missing comma"),
        (", A\n", "empty target"),
        ("A, A\nA, A\n", "Duplicate BNET target"),
        ("# comments only\n", "contains no Boolean rules"),
    ],
)
def test_bnet_node_names_rejects_invalid_models(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    bnet_path = _write(tmp_path / "invalid.bnet", contents)

    with pytest.raises(ValueError, match=message):
        bnet_node_names(bnet_path)


def test_bnd_node_names_reads_declarations_and_ignores_comments(
    tmp_path: Path,
) -> None:
    bnd_path = _write(
        tmp_path / "model.bnd",
        "/* Node Fake { logic = 1; } */\n"
        "Node A_1 { logic = B; } // Node Hidden { logic = 1; }\n"
        "# Node AlsoHidden { logic = 1; }\n"
        "node B { logic = A_1; }\n",
    )

    assert bnd_node_names(bnd_path) == ["A_1", "B"]


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("// comments only\n", "contains no node declarations"),
        (
            "Node A { logic = 1; }\nNode A { logic = 0; }\n",
            "duplicate node declarations",
        ),
    ],
)
def test_bnd_node_names_rejects_invalid_models(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    bnd_path = _write(tmp_path / "invalid.bnd", contents)

    with pytest.raises(ValueError, match=message):
        bnd_node_names(bnd_path)


def test_network_validates_node_rename_and_duplicate_rule_metadata() -> None:
    network = HandoffNetwork(
        nodes=["A_1", "B"],
        renamed_nodes=["A-1"],
        node_renames={"A-1": "A_1"},
        duplicate_rules_removed=["A_1"],
    )
    assert network.node_renames == {"A-1": "A_1"}

    with pytest.raises(ValidationError, match="absent from the network"):
        HandoffNetwork(
            nodes=["A"],
            renamed_nodes=["B-1"],
            node_renames={"B-1": "B_1"},
        )
    with pytest.raises(ValidationError, match="must change"):
        HandoffNetwork(
            nodes=["A"],
            renamed_nodes=["A"],
            node_renames={"A": "A"},
        )


def test_neko_handoff_export_result_aligns_manifest_and_file(
    tmp_path: Path,
) -> None:
    manifest = _neko_manifest(tmp_path)
    manifest.history_state_id = 4
    manifest_path = write_handoff_manifest(
        tmp_path / "neko-handoff.json",
        manifest,
    )
    manifest_file = handoff_artifact(
        manifest_path,
        server="NeKo",
        session_id=manifest.source.session_id,
        role="parent_manifest",
    )

    result = NeKoHandoffExportResult(
        server="NeKo",
        session_id=manifest.source.session_id,
        manifest_file=manifest_file,
        manifest=manifest,
    )

    assert result.manifest.history_state_id == 4
    assert result.manifest_file.sha256 == sha256_file(manifest_path)


def test_maboss_handoff_import_result_aligns_parent_and_generated_pair(
    tmp_path: Path,
) -> None:
    source_manifest = _neko_manifest(tmp_path)
    source_path = write_handoff_manifest(
        tmp_path / "neko.handoff.json",
        source_manifest,
    )
    source_file = handoff_artifact(
        source_path,
        server="NeKo",
        session_id=source_manifest.source.session_id,
        role="parent_manifest",
    )
    bnd_file = handoff_artifact(
        _write(tmp_path / "imported.bnd", "node A { logic = B; }\n"),
        server="MaBoSS",
        session_id="maboss-session",
        role="maboss_bnd",
    )
    cfg_file = handoff_artifact(
        _write(tmp_path / "imported.cfg", "max_time = 100;\n"),
        server="MaBoSS",
        session_id="maboss-session",
        role="maboss_cfg",
    )

    result = MaBoSSHandoffImportResult(
        server="MaBoSS",
        session_id="maboss-session",
        source_manifest_file=source_file,
        source_manifest=source_manifest,
        bnd_file=bnd_file,
        cfg_file=cfg_file,
        nodes=["A", "B"],
        output_nodes=["B"],
        requires_output_selection=False,
    )
    assert result.source_manifest.handoff_type == "neko-to-maboss"

    with pytest.raises(ValidationError, match="output nodes were declared"):
        MaBoSSHandoffImportResult(
            **result.model_dump(exclude={"requires_output_selection"}),
            requires_output_selection=True,
        )


def test_maboss_handoff_export_result_aligns_manifest_and_file(
    tmp_path: Path,
) -> None:
    manifest = _maboss_manifest(tmp_path, with_neko_parent=True)
    manifest_path = write_handoff_manifest(
        tmp_path / "maboss.handoff.json",
        manifest,
    )
    manifest_file = handoff_artifact(
        manifest_path,
        server="MaBoSS",
        session_id=manifest.source.session_id,
        role="parent_manifest",
    )

    result = MaBoSSHandoffExportResult(
        server="MaBoSS",
        session_id=manifest.source.session_id,
        manifest_file=manifest_file,
        manifest=manifest,
    )

    assert result.manifest.target.cell_type == "epithelial"
    assert result.manifest_file.sha256 == sha256_file(manifest_path)


def test_physicell_handoff_import_result_aligns_complete_copied_lineage(
    tmp_path: Path,
) -> None:
    manifest = _maboss_manifest(tmp_path, with_neko_parent=True)
    manifest_path = write_handoff_manifest(
        tmp_path / "maboss.handoff.json",
        manifest,
    )
    source_manifest_file = handoff_artifact(
        manifest_path,
        server="MaBoSS",
        session_id=manifest.source.session_id,
        role="parent_manifest",
    )
    assert manifest.parent_manifest is not None
    neko_manifest = load_handoff_manifest(manifest.parent_manifest.path)
    assert isinstance(neko_manifest, NeKoToMaBoSSHandoffManifest)

    copied_root = tmp_path / "physicell"
    copied_root.mkdir()

    def copied_artifact(
        source: HandoffArtifact,
        name: str,
        role: Literal[
            "maboss_bnd",
            "maboss_cfg",
            "maboss_result",
            "parent_manifest",
            "neko_bnet",
        ],
    ) -> HandoffArtifact:
        destination = copied_root / name
        destination.write_bytes(Path(source.path).read_bytes())
        return handoff_artifact(
            destination,
            server="PhysiCell",
            session_id="physicell-session",
            role=role,
        )

    result = PhysiCellHandoffImportResult(
        server="PhysiCell",
        session_id="physicell-session",
        source_manifest_file=source_manifest_file,
        source_manifest=manifest,
        manifest_snapshot_file=copied_artifact(
            source_manifest_file,
            "source.handoff.json",
            "parent_manifest",
        ),
        bnd_file=copied_artifact(
            manifest.bnd_file,
            "model.bnd",
            "maboss_bnd",
        ),
        cfg_file=copied_artifact(
            manifest.cfg_file,
            "model.cfg",
            "maboss_cfg",
        ),
        result_file=copied_artifact(
            manifest.simulation.result_file,
            "result.csv",
            "maboss_result",
        ),
        neko_manifest=neko_manifest,
        neko_manifest_file=copied_artifact(
            manifest.parent_manifest,
            "neko.handoff.json",
            "parent_manifest",
        ),
        bnet_file=copied_artifact(
            neko_manifest.bnet_file,
            "network.bnet",
            "neko_bnet",
        ),
        target_cell_type="epithelial",
        nodes=["A", "B"],
        output_nodes=["A"],
        replaced_existing=False,
        context_count=1,
    )

    assert result.manifest_snapshot_file.sha256 == source_manifest_file.sha256
    assert result.bnet_file is not None
    assert result.bnet_file.sha256 == neko_manifest.bnet_file.sha256

    with pytest.raises(ValidationError, match="context_count"):
        PhysiCellHandoffImportResult(
            **result.model_dump(exclude={"context_count"}),
            context_count=0,
        )


def test_handoff_artifact_requires_existing_regular_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        handoff_artifact(
            tmp_path / "missing.bnet",
            server="NeKo",
            session_id="session",
            role="neko_bnet",
        )

    directory = tmp_path / "directory.bnet"
    directory.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        handoff_artifact(
            directory,
            server="NeKo",
            session_id="session",
            role="neko_bnet",
        )


def test_artifact_contract_rejects_wrong_role_suffix_and_media_type(
    tmp_path: Path,
) -> None:
    cfg_path = _write(tmp_path / "model.cfg", "max_time = 10;\n")

    with pytest.raises(ValidationError, match="requires suffix"):
        handoff_artifact(
            cfg_path,
            server="MaBoSS",
            session_id="session",
            role="maboss_bnd",
        )

    valid = handoff_artifact(
        cfg_path,
        server="MaBoSS",
        session_id="session",
        role="maboss_cfg",
    )
    invalid_payload = valid.model_dump()
    invalid_payload["media_type"] = "image/png"
    with pytest.raises(ValidationError, match="requires media type"):
        HandoffArtifact.model_validate(invalid_payload)


def test_artifact_contract_requires_absolute_self_consistent_path(
    tmp_path: Path,
) -> None:
    artifact = handoff_artifact(
        _write(tmp_path / "Network.bnet", "A, A\n"),
        server="NeKo",
        session_id="session",
        role="neko_bnet",
    )
    payload = artifact.model_dump()
    payload["path"] = "Network.bnet"
    with pytest.raises(ValidationError, match="must be absolute"):
        HandoffArtifact.model_validate(payload)

    payload = artifact.model_dump()
    payload["name"] = "other.bnet"
    with pytest.raises(ValidationError, match="basename"):
        HandoffArtifact.model_validate(payload)


def test_network_rejects_duplicates_and_unknown_outputs() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        HandoffNetwork(nodes=["A", "A"])

    with pytest.raises(ValidationError, match="absent from the network"):
        HandoffNetwork(nodes=["A"], output_nodes=["B"])


def test_manifest_models_reject_extra_fields_and_unsupported_versions(
    tmp_path: Path,
) -> None:
    manifest = _neko_manifest(tmp_path)
    payload = manifest.model_dump(mode="json")
    payload["unexpected"] = "value"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        NeKoToMaBoSSHandoffManifest.model_validate(payload)

    payload = manifest.model_dump(mode="json")
    payload["schema_version"] = "2.0"
    with pytest.raises(ValidationError, match="Input should be '1.0'"):
        NeKoToMaBoSSHandoffManifest.model_validate(payload)


def test_provenance_rejects_inconsistent_packages_and_non_utc_time() -> None:
    with pytest.raises(ValidationError, match="requires modelling package"):
        HandoffProvenance(
            server="MaBoSS",
            session_id="session",
            mcp_package=_package("mcp-biomodelling-servers"),
            modelling_package=_package("nekomata"),
            operation="export",
        )

    with pytest.raises(ValidationError, match="UTC timezone"):
        HandoffProvenance(
            server="NeKo",
            session_id="session",
            mcp_package=_package("mcp-biomodelling-servers"),
            modelling_package=_package("nekomata"),
            operation="export",
            recorded_at=datetime.now(timezone(timedelta(hours=2))),
        )


def test_maboss_manifest_rejects_inconsistent_parent_lineage(
    tmp_path: Path,
) -> None:
    manifest = _maboss_manifest(tmp_path, with_neko_parent=True)
    payload = manifest.model_dump()
    payload["lineage"] = []

    with pytest.raises(ValidationError, match="exactly one NeKo lineage"):
        MaBoSSToPhysiCellHandoffManifest.model_validate(payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_maboss_parameters_reject_non_finite_values(value: float) -> None:
    with pytest.raises(ValidationError, match="finite JSON values"):
        MaBoSSSimulationHandoff(parameters={"max_time": value})


def test_verify_detects_size_digest_and_missing_file_changes(
    tmp_path: Path,
) -> None:
    artifact_path = _write(tmp_path / "Network.bnet", "A, B\n")
    artifact = handoff_artifact(
        artifact_path,
        server="NeKo",
        session_id="session",
        role="neko_bnet",
    )

    artifact_path.write_text("A, B\nB, A\n", encoding="utf-8")
    with pytest.raises(ValueError, match="size changed"):
        verify_handoff_artifact(artifact)

    artifact_path.write_text("B, A\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest changed"):
        verify_handoff_artifact(artifact)

    artifact_path.unlink()
    with pytest.raises(FileNotFoundError, match="no longer exists"):
        verify_handoff_artifact(artifact)


def test_verify_manifest_checks_every_referenced_artifact(
    tmp_path: Path,
) -> None:
    manifest = _maboss_manifest(tmp_path, with_neko_parent=False)
    Path(manifest.cfg_file.path).write_text(
        "max_time = 999;\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="changed"):
        verify_handoff_manifest(manifest)


def test_load_rejects_wrong_expected_type_malformed_and_oversized_json(
    tmp_path: Path,
) -> None:
    manifest_path = write_handoff_manifest(
        tmp_path / "handoff.json",
        _neko_manifest(tmp_path),
    )
    with pytest.raises(ValueError, match="Expected handoff type"):
        load_handoff_manifest(
            manifest_path,
            expected_handoff_type="maboss-to-physicell",
        )

    malformed = _write(tmp_path / "malformed.json", "{not-json")
    with pytest.raises(ValueError, match="Invalid handoff manifest"):
        load_handoff_manifest(malformed)

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (MAX_HANDOFF_MANIFEST_BYTES + 1))
    with pytest.raises(ValueError, match="exceeds"):
        load_handoff_manifest(oversized)


def test_load_can_parse_without_verifying_artifacts(tmp_path: Path) -> None:
    manifest = _neko_manifest(tmp_path)
    manifest_path = write_handoff_manifest(
        tmp_path / "handoff.json",
        manifest,
    )
    Path(manifest.bnet_file.path).unlink()

    loaded = load_handoff_manifest(
        manifest_path,
        verify_artifacts=False,
    )

    assert loaded == manifest


def test_manifest_io_rejects_invalid_suffix_and_accidental_overwrite(
    tmp_path: Path,
) -> None:
    manifest = _neko_manifest(tmp_path)
    with pytest.raises(ValueError, match=r"\.json suffix"):
        write_handoff_manifest(tmp_path / "handoff.txt", manifest)

    manifest_path = write_handoff_manifest(
        tmp_path / "handoff.json",
        manifest,
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_handoff_manifest(manifest_path, manifest)

    assert write_handoff_manifest(
        manifest_path,
        manifest,
        overwrite=True,
    ) == manifest_path


def test_manifest_io_preserves_a_concurrently_created_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _neko_manifest(tmp_path)
    manifest_path = tmp_path / "handoff.json"

    def create_competing_file(_source: Path, destination: Path) -> None:
        Path(destination).write_text("concurrent\n", encoding="utf-8")
        raise FileExistsError

    monkeypatch.setattr(handoff_contracts.os, "link", create_competing_file)

    with pytest.raises(FileExistsError, match="created concurrently"):
        write_handoff_manifest(manifest_path, manifest)

    assert manifest_path.read_text(encoding="utf-8") == "concurrent\n"
    assert list(tmp_path.glob(".handoff.json.*.tmp")) == []


def test_manifest_io_requires_existing_directory_and_json_input(
    tmp_path: Path,
) -> None:
    manifest = _neko_manifest(tmp_path)
    with pytest.raises(FileNotFoundError, match="directory does not exist"):
        write_handoff_manifest(
            tmp_path / "missing" / "handoff.json",
            manifest,
        )

    text_path = _write(tmp_path / "handoff.txt", "{}")
    with pytest.raises(ValueError, match=r"\.json suffix"):
        load_handoff_manifest(text_path)
