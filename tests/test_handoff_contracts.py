"""Tests for versioned cross-server handoff contracts and file integrity."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from mcp_biomodelling_servers.handoff import (
    HANDOFF_SCHEMA_VERSION,
    MAX_HANDOFF_MANIFEST_BYTES,
    HandoffArtifact,
    HandoffNetwork,
    HandoffPackage,
    HandoffProvenance,
    MaBoSSSimulationHandoff,
    MaBoSSToPhysiCellHandoffManifest,
    NeKoToMaBoSSHandoffManifest,
    PhysiCellTarget,
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
        _write(tmp_path / "model.bnd", "Node A { logic = B; }\n"),
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
