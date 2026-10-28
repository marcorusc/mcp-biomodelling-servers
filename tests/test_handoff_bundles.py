"""Portable handoff bundles across servers that share no filesystem."""

import base64
import gzip
import hashlib
import inspect
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from mcp_biomodelling_servers import handoff as handoff_module
from mcp_biomodelling_servers import handoff_bundle as bundles
from mcp_biomodelling_servers.handoff import (
    HandoffPackage,
    HandoffProvenance,
    handoff_artifact,
    load_handoff_manifest,
    verify_handoff_artifact,
)
from mcp_biomodelling_servers.handoff_bundle import (
    HandoffBundle,
    build_handoff_bundle,
    materialize_handoff_bundle,
)
from mcp_biomodelling_servers.neko_provenance import (
    NeKoExportBinding,
    NeKoExportEvidence,
    NeKoStateIdentity,
)
from mcp_biomodelling_servers.ode_handoff import write_ode_handoff


def _sandbox(monkeypatch, tmp_path, name):
    """Give the next server its own artifact root, as sandboxed hosts do."""
    root = tmp_path / name
    root.mkdir()
    monkeypatch.setenv("MCP_BIOMODELLING_ARTIFACT_ROOT", str(root))
    return root


def _hide(root: Path) -> None:
    """Make every file of a previous server invisible to the next one."""
    shutil.rmtree(root)


def _neko_boolean_export(monkeypatch, neko_root):
    from NeKo.session_manager import NeKoSessionManager
    from tests.test_neko_mcp_errors import neko_server as neko

    manager = NeKoSessionManager()
    ns = manager.get_session(manager.create_session())

    class Network:
        nodes = pd.DataFrame({"Genesymbol": ["A", "C"]})

    def export_bnet(network, path, **kwargs):
        Path(path).write_text("A, !C\nC, A\n")
        return {}

    def export_curation(network, sess, bnet_reference, **kwargs):
        path = Path(bnet_reference.path).with_name("current.history.json")
        evidence = NeKoExportEvidence(
            mcp=NeKoStateIdentity(session_id=sess.session_id, network_id="net",
                                  state_id=3, database_name="SIGNOR"),
            export=NeKoExportBinding(artifact=bnet_reference),
            history={"format": "neko-history", "schema_version": 1,
                     "current_state_id": 3, "database_name": "SIGNOR",
                     "working_state_matches_current": True},
        )
        path.write_text(evidence.model_dump_json())
        return handoff_artifact(path, server="NeKo", session_id=sess.session_id,
                                role="neko_history")

    monkeypatch.setattr(neko, "_session_network", lambda _: (ns, Network()))
    monkeypatch.setattr(neko, "is_connected", lambda _: True)
    monkeypatch.setattr(neko, "_neko_package_version", lambda: "offline")
    monkeypatch.setattr(neko, "_export_sanitized_bnet", export_bnet)
    monkeypatch.setattr(neko, "export_provenance", export_curation)
    monkeypatch.setattr(neko, "_network_history_state_id", lambda _: 3)
    result = inspect.unwrap(neko.export_neko_handoff)(
        biological_context="isolated sandboxes",
        output_nodes=["C"],
        artifact_prefix="current",
        session_id=ns.session_id,
    )
    path = result.structured_content["manifest_file"]["path"]
    assert Path(path).is_relative_to(neko_root)
    assert load_handoff_manifest(path).curation_file is not None
    return path


def _round_trip(bundle: HandoffBundle) -> HandoffBundle:
    """Cross the host boundary as JSON, the way MCP clients pass values."""
    return HandoffBundle.model_validate(json.loads(bundle.model_dump_json()))


def test_bundles_carry_boolean_pipeline_across_isolated_sandboxes(tmp_path, monkeypatch):
    from MaBoSS import server as maboss
    from MaBoSS.session_manager import MaBoSSSession
    from MaBoSS.tools.handoff_bundle import export_handoff_bundle as maboss_bundle
    from NeKo.tools.handoff_bundle import export_handoff_bundle as neko_bundle
    from PhysiCell.session_manager import SessionState
    from tests.test_physicell_mcp_errors import _handoff_config
    from tests.test_physicell_mcp_errors import physicell_server as physicell

    neko_root = _sandbox(monkeypatch, tmp_path, "neko")
    neko_path = _neko_boolean_export(monkeypatch, neko_root)
    original = load_handoff_manifest(neko_path)
    neko_transfer = _round_trip(neko_bundle(manifest_path=neko_path))
    assert neko_transfer.handoff_type == "neko-to-maboss"
    assert len(neko_transfer.blobs) == 3  # manifest, BNET, curation history
    _hide(neko_root)

    maboss_root = _sandbox(monkeypatch, tmp_path, "maboss")
    with pytest.raises(FileNotFoundError, match="export_handoff_bundle"):
        load_handoff_manifest(neko_path)

    class BooleanNetwork(dict):
        outputs = ["C"]

        def get_output(self):
            return self.outputs

        def set_output(self, value):
            self.outputs = value

    sim = SimpleNamespace(
        network=BooleanNetwork(A=None, C=None),
        param={"max_time": 100},
        print_bnd=lambda out: out.write("Node A {logic = !C;}\nNode C {logic = A;}\n"),
        print_cfg=lambda out: out.write("max_time = 100;\n"),
    )

    def convert(source, bnd, cfg):
        assert Path(source).is_relative_to(maboss_root)
        with Path(bnd).open("w") as out:
            sim.print_bnd(out)
        with Path(cfg).open("w") as out:
            sim.print_cfg(out)

    ms = MaBoSSSession("maboss-child")
    monkeypatch.setattr(maboss, "ensure_session", lambda _: ms)
    monkeypatch.setattr(maboss.maboss, "bnet_to_bnd_and_cfg", convert)
    monkeypatch.setattr(maboss.maboss, "load", lambda *args: sim)
    monkeypatch.setattr(maboss, "_maboss_package_version", lambda: "offline")
    imported = inspect.unwrap(maboss.import_neko_handoff)(
        artifact_prefix="imported", session_id=ms.session_id,
        handoff_bundle=neko_transfer,
    ).structured_content
    # The manifest travels byte-identical: same digest, same recorded paths.
    assert imported["source_manifest_file"]["sha256"] == neko_transfer.manifest_sha256
    assert imported["source_manifest"]["bnet_file"]["path"] == original.bnet_file.path
    assert Path(imported["source_manifest_file"]["path"]).is_relative_to(maboss_root)

    exported = inspect.unwrap(maboss.export_maboss_handoff)(
        target_cell_type="tumour",
        biological_context=None,
        simulation_summary=None,
        include_result=False,
        artifact_prefix="exported",
        session_id=ms.session_id,
    )
    maboss_path = exported.structured_content["manifest_file"]["path"]
    downstream = load_handoff_manifest(maboss_path)
    assert downstream.parent_manifest.sha256 == neko_transfer.manifest_sha256
    assert downstream.lineage == [original.source]
    maboss_transfer = _round_trip(maboss_bundle(manifest_path=maboss_path))
    # MaBoSS manifest, BND, CFG, then the NeKo parent with its BNET and history.
    assert len(maboss_transfer.blobs) == 6
    _hide(maboss_root)

    physicell_root = _sandbox(monkeypatch, tmp_path, "physicell")
    ps = SessionState(session_id="physicell-child")
    ps.config = _handoff_config("tumour")
    monkeypatch.setattr(physicell, "_require_configuration", lambda _: ps)
    monkeypatch.setattr(physicell, "PHYSIBOSS_AVAILABLE", True)
    final = inspect.unwrap(physicell.import_maboss_handoff)(
        artifact_prefix="attached",
        replace_existing=False,
        session_id=ps.session_id,
        handoff_bundle=maboss_transfer,
    ).structured_content
    assert final["neko_manifest"]["curation_file"]["sha256"] == original.curation_file.sha256
    assert Path(final["bnd_file"]["path"]).is_relative_to(physicell_root)
    assert final["source_manifest_file"]["sha256"] == maboss_transfer.manifest_sha256


def _ode_export(directory: Path):
    from NeKo.services.ode_exporting import ode_network
    from tests.test_biomass_workflow import network_fixture

    source = HandoffProvenance(
        server="NeKo",
        session_id="source-session",
        mcp_package=HandoffPackage(name="mcp-biomodelling-servers", version="2.5.1"),
        modelling_package=HandoffPackage(name="nekomata", version="1.10.0"),
        operation="export_biomass_handoff",
    )
    return write_ode_handoff(directory, "enzyme", ode_network(network_fixture()), source,
                             "Synthetic enzyme fixture.", 0)


def test_bundle_carries_ode_handoff_with_identical_upstream(tmp_path, monkeypatch):
    from BioMASS import server as biomass
    from BioMASS.session_manager import BioMASSSessionManager
    from mcp_biomodelling_servers.artifact_manager import get_artifact_dir
    from NeKo.src.helpers import _SERVER_ROOT
    from NeKo.tools.handoff_bundle import export_handoff_bundle as neko_bundle

    neko_root = _sandbox(monkeypatch, tmp_path, "neko")
    export = _ode_export(get_artifact_dir(_SERVER_ROOT, "source-session"))
    transfer = _round_trip(neko_bundle(manifest_path=export.manifest_file.path))
    assert transfer.handoff_type == "neko-to-biomass"
    _hide(neko_root)

    biomass_root = _sandbox(monkeypatch, tmp_path, "biomass")
    manager = BioMASSSessionManager(biomass_root / "sessions")
    monkeypatch.setattr(biomass, "session_manager", manager)
    sid = biomass.create_session().session_id
    state = biomass.import_neko_handoff(session_id=sid, handoff_bundle=transfer)
    assert state.document.upstream == export.manifest
    assert state.document.network.nodes


@pytest.fixture
def neko_manifest(tmp_path, monkeypatch):
    root = _sandbox(monkeypatch, tmp_path, "producer")
    (root / "session").mkdir()
    return root, Path(_ode_export(root / "session").manifest_file.path)


def test_bundle_export_refuses_manifests_outside_its_artifact_root(neko_manifest, tmp_path):
    root, manifest = neko_manifest
    with pytest.raises(ValueError, match="artifact directory"):
        build_handoff_bundle(manifest, allowed_roots=(tmp_path / "elsewhere",))
    assert build_handoff_bundle(manifest, allowed_roots=(root,)).manifest_name == manifest.name


def _bundle(neko_manifest):
    root, manifest = neko_manifest
    return build_handoff_bundle(manifest, allowed_roots=(root,))


def _replace_blob(bundle, digest, data):
    blobs = dict(bundle.blobs)
    blobs[digest] = base64.b64encode(gzip.compress(data)).decode()
    return bundle.model_copy(update={"blobs": blobs})


def test_tampered_blob_is_rejected(neko_manifest):
    bundle = _bundle(neko_manifest)
    digest = next(d for d in bundle.blobs if d != bundle.manifest_sha256)
    with pytest.raises(ValueError, match="does not match its digest"):
        materialize_handoff_bundle(
            _replace_blob(bundle, digest, b"tampered"),
            expected_handoff_type="neko-to-biomass",
        )


def test_missing_and_unreferenced_blobs_are_rejected(neko_manifest):
    bundle = _bundle(neko_manifest)
    network = next(d for d in bundle.blobs if d != bundle.manifest_sha256)
    missing = bundle.model_copy(update={"blobs": {bundle.manifest_sha256: bundle.blobs[bundle.manifest_sha256]}})
    with pytest.raises(ValueError, match="missing referenced artifact"):
        materialize_handoff_bundle(missing, expected_handoff_type="neko-to-biomass")
    extra_data = b"unrelated"
    extra = _replace_blob(bundle, hashlib.sha256(extra_data).hexdigest(), extra_data)
    with pytest.raises(ValueError, match="does not reference"):
        materialize_handoff_bundle(extra, expected_handoff_type="neko-to-biomass")
    assert network in bundle.blobs


def test_wrong_handoff_type_and_oversized_content_are_rejected(neko_manifest, monkeypatch):
    bundle = _bundle(neko_manifest)
    with pytest.raises(ValueError, match="Expected a 'neko-to-maboss' bundle"):
        materialize_handoff_bundle(bundle, expected_handoff_type="neko-to-maboss")
    forged = HandoffBundle.model_validate({**bundle.model_dump(), "server": "MaBoSS"})
    with pytest.raises(ValueError, match="must come from NeKo"):
        materialize_handoff_bundle(forged, expected_handoff_type="neko-to-biomass")
    monkeypatch.setattr(bundles, "MAX_HANDOFF_BUNDLE_BYTES", 64)
    with pytest.raises(ValueError, match="exceeds"):
        materialize_handoff_bundle(bundle, expected_handoff_type="neko-to-biomass")


def test_local_copies_are_ignored_unless_the_process_enables_them(neko_manifest, monkeypatch):
    root, manifest = neko_manifest
    bundle = build_handoff_bundle(manifest, allowed_roots=(root,))
    from mcp_biomodelling_servers.ode_handoff import NeKoToBioMASSHandoffManifest

    reference = NeKoToBioMASSHandoffManifest.model_validate_json(manifest.read_bytes()).network_file
    materialize_handoff_bundle(bundle, expected_handoff_type="neko-to-biomass")
    Path(reference.path).unlink()
    monkeypatch.setattr(handoff_module, "_local_handoff_blobs_enabled", False)
    with pytest.raises(FileNotFoundError, match="no longer exists"):
        verify_handoff_artifact(reference)
    monkeypatch.setattr(handoff_module, "_local_handoff_blobs_enabled", True)
    assert verify_handoff_artifact(reference).is_relative_to(handoff_module.handoff_blob_store())


def test_importers_require_exactly_one_source(neko_manifest, tmp_path, monkeypatch):
    from BioMASS import server as biomass
    from BioMASS.session_manager import BioMASSSessionManager

    _, manifest = neko_manifest
    bundle = _bundle(neko_manifest)
    monkeypatch.setattr(biomass, "session_manager", BioMASSSessionManager(tmp_path / "s"))
    sid = biomass.create_session().session_id
    with pytest.raises(ValueError, match="exactly one"):
        biomass.import_neko_handoff(session_id=sid)
    with pytest.raises(ValueError, match="exactly one"):
        biomass.import_neko_handoff(str(manifest), sid, handoff_bundle=bundle)
