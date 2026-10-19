"""Offline regressions for storage, mutable provenance, and UTC metadata."""
import json
from datetime import datetime, timedelta

import pytest

import artifact_manager as compatibility
from mcp_biomodelling_servers import artifact_manager as artifacts
from mcp_biomodelling_servers.handoff import (
    HandoffNetwork,
    HandoffPackage,
    HandoffProvenance,
    NeKoToMaBoSSHandoffManifest,
    handoff_artifact,
    load_handoff_manifest,
    write_handoff_manifest,
)
from mcp_biomodelling_servers.import_lineage import (
    SourceExport,
    capture_import,
    write_source_export,
)
from mcp_biomodelling_servers.timestamps import utc_now


@pytest.mark.parametrize("server", ["NeKo", "MaBoSS", "PhysiCell", "BioMASS"])
def test_storage_and_legacy_collisions(tmp_path, server):
    root = tmp_path / "installation" / server
    legacy = root / "artifacts" / "same-id"
    legacy.mkdir(parents=True)
    (legacy / "accepted.bnet").write_text("A, A\n")
    current = artifacts.get_artifact_dir(root, "same-id")
    assert current == tmp_path / "artifact-cache" / server / "same-id"
    (current / "mutable.bnet").write_text("A, !A\n")
    artifacts.write_session_meta(root, "same-id", server)
    listing = artifacts.list_artifact_sessions(root, server)
    assert len(listing) == 2
    assert {row["artifact_directory"] for row in listing} == {str(current), str(legacy)}
    assert {row["legacy"] for row in listing} == {True, False}
    assert len(artifacts.list_artifacts(root, "same-id")) == 3
    assert len(artifacts.list_artifacts(root)) == 3
    artifacts.clean_artifacts(root, "same-id")
    assert (legacy / "accepted.bnet").read_text() == "A, A\n"
    assert artifacts.list_artifact_sessions(root)[0]["legacy"]


def test_default_cache_and_shim(tmp_path, monkeypatch):
    monkeypatch.delenv("MCP_BIOMODELLING_ARTIFACT_ROOT")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert compatibility.get_artifact_dir is artifacts.get_artifact_dir
    directories = [artifacts.get_artifact_dir(tmp_path / name, "same")
                   for name in ["NeKo", "MaBoSS", "PhysiCell", "BioMASS"]]
    assert len(set(directories)) == 4
    assert all(p.is_relative_to(tmp_path / "cache") for p in directories)
    monkeypatch.setenv("MCP_BIOMODELLING_ARTIFACT_ROOT", "relative")
    with pytest.raises(ValueError, match="absolute"):
        artifacts.get_artifact_dir(tmp_path / "NeKo", "same")


@pytest.mark.parametrize("operation", [artifacts.list_artifacts, artifacts.clean_artifacts,
                                      artifacts.read_session_meta])
def test_session_traversal_rejected(tmp_path, operation):
    with pytest.raises(ValueError):
        operation(tmp_path, "../outside")


@pytest.mark.parametrize("suffix,text,edited", [(".sif", "A\t1\tB\n", "A\t-1\tB\n"),
                                               (".bnet", "A, A\n", "A, !A\n")])
def test_import_parent_identity_derivation_and_export(tmp_path, suffix, text, edited):
    source = tmp_path / ("original" + suffix)
    source.write_text(text)
    parent = write_source_export(source, server="NeKo", session_id="parent", imports=[])
    imported = tmp_path / ("copy" + suffix)
    imported.write_text(text)
    old_call = capture_import(imported, server="NeKo", session_id="child")
    assert old_call.relationship == "standalone"
    identical = capture_import(imported, server="NeKo", session_id="child",
                               parent_manifest=str(parent))
    assert identical.relationship == "byte-identical"
    imported.write_text(edited)
    with pytest.raises(ValueError, match="match"):
        capture_import(imported, server="NeKo", session_id="child", parent_manifest=str(parent))
    derivative = capture_import(imported, server="NeKo", session_id="child",
                                parent_manifest=str(parent), derived_from=str(source))
    assert derivative.relationship == "edited-derivative"
    assert derivative.source.sha256 != derivative.derived_from.sha256
    exported = write_source_export(imported, server="NeKo", session_id="child", imports=[derivative])
    record = SourceExport.model_validate_json(exported.read_bytes())
    assert record.imports[0] == derivative
    assert record.created_at.utcoffset() == timedelta(0)
    # Changing an old mutable input cannot silently rewrite retained evidence.
    imported.write_text(text)
    assert SourceExport.model_validate_json(exported.read_bytes()).imports[0] == derivative
    with pytest.raises(ValueError, match="changed"):
        capture_import(imported, server="NeKo", session_id="next", parent_manifest=str(exported))
    with pytest.raises(ValueError, match="requires"):
        capture_import(imported, server="NeKo", session_id="child", derived_from=str(source))
    with pytest.raises(ValueError, match="match"):
        capture_import(imported, server="NeKo", session_id="child",
                       parent_manifest=str(parent), derived_from=str(tmp_path / "arbitrary"))
    source.write_text(text.replace("A", "X"))  # same size, different hash
    with pytest.raises(ValueError, match="digest changed"):
        capture_import(imported, server="NeKo", session_id="child", parent_manifest=str(parent))


def test_verified_handoff_parent_and_unchanged_v1_contract(tmp_path):
    source = tmp_path / "model.bnet"
    source.write_text("A, A\n")
    manifest = NeKoToMaBoSSHandoffManifest(
        source=HandoffProvenance(server="NeKo", session_id="parent",
            mcp_package=HandoffPackage(name="mcp-biomodelling-servers", version="2.4.0"),
            modelling_package=HandoffPackage(name="nekomata", version="1.10.1"), operation="export"),
        network=HandoffNetwork(nodes=["A"]),
        bnet_file=handoff_artifact(source, server="NeKo", session_id="parent", role="neko_bnet"))
    parent = write_handoff_manifest(tmp_path / "parent.json", manifest)
    imported = capture_import(source, server="MaBoSS", session_id="child", parent_manifest=str(parent))
    assert imported.parent_manifest.sha256
    assert imported.relationship == "byte-identical"
    assert load_handoff_manifest(parent).lineage == []
    data = json.loads(parent.read_text())
    data["lineage"] = [data["source"]]
    parent.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="upstream lineage"):
        capture_import(source, server="MaBoSS", session_id="child", parent_manifest=str(parent))
    parent.write_text('{"arbitrary": "untrusted"}')
    with pytest.raises(ValueError, match="Invalid handoff"):
        capture_import(source, server="MaBoSS", session_id="child", parent_manifest=str(parent))


def test_utc_metadata_and_biomass_restore(tmp_path):
    from BioMASS.session_manager import BioMASSSessionManager
    manager = BioMASSSessionManager(tmp_path / "BioMASS")
    sess = manager.create()
    directory = manager.directory(sess.session_id)
    meta = artifacts.read_session_meta(manager.root, sess.session_id)
    assert datetime.fromisoformat(meta["created_at"]).utcoffset() == timedelta(0)
    assert utc_now().utcoffset() == timedelta(0)
    manager.close(sess.session_id)
    assert manager.restore(sess.session_id).document == sess.document
    assert directory.is_relative_to(tmp_path / "artifact-cache")
    assert not (manager.root / "artifacts").exists()


def test_current_orphan_does_not_inherit_legacy_identity(tmp_path):
    root = tmp_path / "NeKo"
    legacy = root / "artifacts" / "collision"
    legacy.mkdir(parents=True)
    (legacy / "session_meta.json").write_text(json.dumps({
        "session_id": "collision", "server": "NeKo", "label": "accepted legacy"
    }))
    assert artifacts.read_session_meta(root, "collision")["label"] == "accepted legacy"
    artifacts.get_artifact_dir(root, "collision")
    assert artifacts.read_session_meta(root, "collision") is None
    entries = artifacts.list_artifact_sessions(root)
    assert len(entries) == 2
    assert next(e for e in entries if not e["legacy"])["label"] == ""
