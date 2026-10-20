"""Offline ancestry integrity and portability, with no scientific initialization."""

import base64
from pathlib import Path

import pytest

from mcp_biomodelling_servers import import_lineage as lineage
from mcp_biomodelling_servers.handoff import (
    NeKoToMaBoSSHandoffManifest,
    SourceImport,
    load_handoff_manifest,
    verify_handoff_artifact,
    write_handoff_manifest,
)
from tests.test_handoff_contracts import _neko_manifest


def derivative(tmp_path, suffix=".sif", server="NeKo", session="neko-session"):
    original = tmp_path / ("original" + suffix)
    original.write_text("A\t1\tB\n" if suffix == ".sif" else "A, A\n")
    parent = lineage.write_source_export(
        original, server="NeKo", session_id="ancestor", imports=[]
    )
    edited = tmp_path / ("edited" + suffix)
    edited.write_text("A\t-1\tC\n" if suffix == ".sif" else "A, !A\nC, A\n")
    imported = lineage.capture_import(
        edited,
        server=server,
        session_id=session,
        parent_manifest=str(parent),
        derived_from=str(original),
    )
    return imported, original, parent, edited


def test_v1_load_and_v2_required_ancestry(tmp_path):
    original = _neko_manifest(tmp_path)
    path = write_handoff_manifest(tmp_path / "v1.json", original)
    assert load_handoff_manifest(path).schema_version == "1.0"
    data = original.model_dump()
    data["schema_version"] = "2.0"
    with pytest.raises(ValueError, match="explicit source_ancestry"):
        NeKoToMaBoSSHandoffManifest.model_validate(data)
    data["source_ancestry"] = []
    assert NeKoToMaBoSSHandoffManifest.model_validate(data).schema_version == "2.0"
    data["schema_version"] = "99.0"
    with pytest.raises(ValueError):
        NeKoToMaBoSSHandoffManifest.model_validate(data)


@pytest.mark.parametrize("version", ["1.0", "2.0"])
@pytest.mark.parametrize("change", ["deleted", "modified", "outside-root"])
def test_initial_typed_parent_requires_live_artifact_even_for_identical_copy(
    tmp_path, version, change
):
    parent_model = _neko_manifest(tmp_path)
    if version == "2.0":
        parent_model = NeKoToMaBoSSHandoffManifest.model_validate(
            {
                **parent_model.model_dump(),
                "schema_version": "2.0",
                "source_ancestry": [],
            }
        )
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    ancestor = Path(parent_model.bnet_file.path)
    source_copy = trusted / "copy.bnet"
    source_copy.write_bytes(ancestor.read_bytes())
    parent = write_handoff_manifest(trusted / "parent.json", parent_model)
    if change == "deleted":
        ancestor.unlink()
    elif change == "modified":
        # Keep the byte count, so a digest check is required.
        ancestor.write_bytes(ancestor.read_bytes().replace(b"A", b"X"))
    with pytest.raises((ValueError, FileNotFoundError)):
        lineage.capture_import(
            source_copy,
            server="MaBoSS",
            session_id="child",
            parent_manifest=str(parent),
            trusted_root=trusted if change == "outside-root" else tmp_path,
        )


def test_derivative_is_ancestry_not_stage_parent_or_current_approval(tmp_path):
    imported, _, _, _ = derivative(tmp_path)
    manifest = _neko_manifest(tmp_path)
    data = manifest.model_dump()
    data.update(schema_version="2.0", source_ancestry=[imported.model_dump()])
    v2 = NeKoToMaBoSSHandoffManifest.model_validate(data)
    assert v2.lineage == []
    assert v2.network.nodes == ["A", "B"]  # ancestor/derivative can have other nodes
    assert imported.relationship == "edited-derivative"
    assert imported.source.sha256 != imported.derived_from.sha256
    path = write_handoff_manifest(tmp_path / "v2.json", v2)
    Path(v2.bnet_file.path).write_text("A, !B\nB, A\n")
    with pytest.raises(ValueError, match="changed"):
        load_handoff_manifest(path)  # ancestry never approves altered current bytes


def test_portable_copy_after_all_historical_files_disappear(tmp_path, monkeypatch):
    imported, original, parent, edited = derivative(tmp_path)
    original.unlink()
    parent.unlink()
    edited.unlink()

    # Enforce the API guarantee instead of merely relying on missing files.
    def no_reads(*args, **kwargs):
        raise AssertionError("Historical filesystem read attempted")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", no_reads)
        lineage.verify_source_ancestry([imported])
        copied = lineage.copy_source_ancestry([imported], tmp_path / "portable")
    assert copied[imported.source.sha256].read_bytes() == b"A\t-1\tC\n"
    assert copied[imported.parent_manifest.sha256].read_bytes().startswith(b"{")
    assert lineage.copy_source_ancestry([imported], tmp_path / "portable") == copied


@pytest.mark.parametrize("failure", ["flush", "link"])
def test_ancestry_publication_failure_is_retryable(tmp_path, monkeypatch, failure):
    imported, _, _, _ = derivative(tmp_path)
    destination = tmp_path / "portable"

    def fail(*args):
        raise OSError("interrupted publication")

    with monkeypatch.context() as patch:
        patch.setattr(lineage.os, "fsync" if failure == "flush" else "link", fail)
        with pytest.raises(OSError, match="interrupted publication"):
            lineage.copy_source_ancestry([imported], destination)
    assert list(destination.iterdir()) == []
    copied = lineage.copy_source_ancestry([imported], destination)
    assert copied[imported.source.sha256].read_bytes() == b"A\t-1\tC\n"


@pytest.mark.parametrize("competing", ["identical", "different", "symlink"])
def test_ancestry_concurrent_publication_never_overwrites(tmp_path, monkeypatch, competing):
    imported, _, _, _ = derivative(tmp_path)
    destination = tmp_path / "portable"
    link = lineage.os.link
    collided = []

    def concurrent_link(source, target):
        source, target = Path(source), Path(target)
        # The final name must remain absent until all bytes are flushed.
        assert not target.exists()
        assert source.read_bytes() == base64.b64decode(imported.evidence.blobs[target.name])
        if not collided:
            collided.append(target)
            if competing == "symlink":
                other = tmp_path / "other"
                other.write_bytes(source.read_bytes())
                target.symlink_to(other)
            else:
                target.write_bytes(source.read_bytes() if competing == "identical" else b"other")
        return link(source, target)

    monkeypatch.setattr(lineage.os, "link", concurrent_link)
    if competing == "identical":
        lineage.copy_source_ancestry([imported], destination)
    else:
        with pytest.raises(ValueError, match="symlink" if competing == "symlink" else "digest mismatch"):
            lineage.copy_source_ancestry([imported], destination)
        if competing == "symlink":
            assert collided[0].is_symlink()
        else:
            assert collided[0].read_bytes() == b"other"
    assert not list(destination.glob(".ancestry-*"))


@pytest.mark.parametrize(
    "mutation", ["missing", "tamper", "owner", "ancestor", "relationship", "version"]
)
def test_tampered_evidence_rejected(tmp_path, mutation):
    imported, _, _, _ = derivative(tmp_path)
    data = imported.model_dump(mode="json")
    if mutation == "missing":
        del data["evidence"]["blobs"][imported.derived_from.sha256]
    elif mutation == "tamper":
        data["evidence"]["blobs"][imported.source.sha256] = base64.b64encode(
            b"tampered"
        ).decode()
    elif mutation == "owner":
        data["parent_manifest"]["session_id"] = data["derived_from"]["session_id"] = (
            "forged"
        )
    elif mutation == "ancestor":
        data["derived_from"]["path"] = str(tmp_path / "not-an-ancestor.sif")
        data["derived_from"]["name"] = "not-an-ancestor.sif"
    elif mutation == "relationship":
        data["relationship"] = "byte-identical"
    else:
        data["evidence"]["schema_version"] = "2.0"
    with pytest.raises(ValueError):
        lineage.verify_source_ancestry([SourceImport.model_validate(data)])


def test_stale_parent_and_mismatch_fail_before_capture(tmp_path):
    imported, original, parent, edited = derivative(tmp_path)
    with pytest.raises(ValueError, match="match"):
        lineage.capture_import(
            edited, server="NeKo", session_id="child", parent_manifest=str(parent)
        )
    original.write_text("corrupted parent source")
    with pytest.raises(ValueError, match="size changed"):
        lineage.capture_import(
            edited,
            server="NeKo",
            session_id="child",
            parent_manifest=str(parent),
            derived_from=str(original),
        )
    lineage.verify_source_ancestry([imported])  # immutable history remains valid
    with pytest.raises(ValueError, match="changed"):
        verify_handoff_artifact(imported.derived_from)


def test_trusted_root_rejects_parent_artifact_escape(tmp_path):
    imported, original, parent, edited = derivative(tmp_path)
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    local = trusted / edited.name
    local.write_bytes(edited.read_bytes())
    local_parent = trusted / parent.name
    local_parent.write_bytes(parent.read_bytes())
    with pytest.raises(ValueError, match="outside trusted root"):
        lineage.capture_import(
            local,
            server="NeKo",
            session_id="child",
            parent_manifest=str(local_parent),
            derived_from=str(original),
            trusted_root=trusted,
        )


def test_multigeneration_survives_deleted_grandparents(tmp_path):
    imported, original, parent, edited = derivative(tmp_path)
    next_parent = lineage.write_source_export(
        edited, server="NeKo", session_id="neko-session", imports=[imported]
    )
    original.unlink()
    parent.unlink()
    child = lineage.capture_import(
        edited, server="MaBoSS", session_id="child", parent_manifest=str(next_parent)
    )
    assert child.relationship == "byte-identical"
    edited.unlink()
    next_parent.unlink()
    lineage.verify_source_ancestry([child])
    assert imported.parent_manifest.sha256 in child.evidence.blobs


def test_legacy_sidecar_reference_imports_are_snapshotted(tmp_path):
    source = tmp_path / "legacy.bnet"
    source.write_text("A, A\n")
    imported = lineage.capture_import(source, server="NeKo", session_id="legacy")
    legacy = imported.model_copy(update={"evidence": None})
    record = lineage.SourceExport(artifact=imported.source, imports=[legacy])
    parent = tmp_path / "legacy.json"
    parent.write_text(record.model_dump_json())
    child = lineage.capture_import(
        source, server="MaBoSS", session_id="child", parent_manifest=str(parent)
    )
    source.unlink()
    parent.unlink()
    lineage.verify_source_ancestry([child])


def test_owner_and_missing_v2_bundle_rejected(tmp_path):
    imported, _, _, _ = derivative(tmp_path)
    data = _neko_manifest(tmp_path).model_dump()
    data.update(schema_version="2.0", source_ancestry=[imported.model_dump()])
    data["source_ancestry"][0]["source"]["session_id"] = "unrelated-history"
    with pytest.raises(ValueError, match="owner"):
        NeKoToMaBoSSHandoffManifest.model_validate(data)
    data["source_ancestry"][0] = imported.model_dump()
    data["source_ancestry"][0]["evidence"] = None
    with pytest.raises(ValueError, match="evidence"):
        NeKoToMaBoSSHandoffManifest.model_validate(data)


def test_limits_and_copy_symlink_rejected(tmp_path, monkeypatch):
    imported, _, _, _ = derivative(tmp_path)
    monkeypatch.setattr(lineage, "MAX_ANCESTRY_DEPTH", 0)
    with pytest.raises(ValueError, match="limit"):
        lineage.verify_source_ancestry([imported])
    monkeypatch.setattr(lineage, "MAX_ANCESTRY_DEPTH", 16)
    monkeypatch.setattr(lineage, "MAX_ANCESTRY_BYTES", 5)
    with pytest.raises(ValueError, match="limit"):
        lineage.verify_source_ancestry([imported])
    monkeypatch.setattr(lineage, "MAX_ANCESTRY_BYTES", 16 * 1024 * 1024)
    dest = tmp_path / "copy"
    dest.mkdir()
    (dest / imported.source.sha256).symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="symlink"):
        lineage.copy_source_ancestry([imported], dest)


def test_offline_sif_derivative_neko_maboss_physicell_pipeline(tmp_path, monkeypatch):
    import inspect
    from types import SimpleNamespace

    import pandas as pd

    from MaBoSS import server as maboss
    from MaBoSS.session_manager import MaBoSSSession
    from NeKo.session_manager import NeKoSessionManager
    from PhysiCell.session_manager import SessionState
    from tests.test_neko_mcp_errors import neko_server as neko
    from tests.test_physicell_mcp_errors import _handoff_config
    from tests.test_physicell_mcp_errors import physicell_server as physicell

    manager = NeKoSessionManager()
    ns = manager.get_session(manager.create_session())
    monkeypatch.setattr(neko, "session_manager", manager)
    monkeypatch.setattr(neko, "ensure_session", lambda _: ns)
    imported, original, parent, edited = derivative(tmp_path, session=ns.session_id)

    class Network:
        def __init__(self, **kwargs):
            self.nodes = pd.DataFrame({"Genesymbol": ["A", "C"]})
            self.edges = self.convert_edgelist_into_genesymbol()

        def convert_edgelist_into_genesymbol(self):
            return pd.DataFrame(
                [{"source": "A", "target": "C", "Effect": "inhibition"}]
            )

    async def immediate(function):
        return function()

    async def progress(*args):
        pass

    monkeypatch.setattr(neko, "Network", Network)
    monkeypatch.setattr(neko.anyio.to_thread, "run_sync", immediate)
    operation = neko.create_network(
        list_of_initial_genes=[],
        ctx=SimpleNamespace(report_progress=progress),
        database="omnipath",
        sif_file=str(edited),
        max_len=2,
        path_policy="one_shortest",
        reuse_policy="discovered_paths",
        only_signed=True,
        consensus=True,
        session_id=ns.session_id,
        verbosity="summary",
        parent_manifest=str(parent),
        derived_from=str(original),
    )
    # Offline coroutine has only immediate mock awaits; no event loop/socketpair.
    with pytest.raises(StopIteration):
        operation.send(None)
    assert ns.source_import.relationship == "edited-derivative"
    original.unlink()
    parent.unlink()
    edited.unlink()
    export_dir = tmp_path / "neko-export"
    export_dir.mkdir()
    monkeypatch.setattr(neko, "_export_dir", lambda _: export_dir)
    monkeypatch.setattr(neko, "_session_network", lambda _: (ns, ns.network))
    monkeypatch.setattr(neko, "is_connected", lambda _: True)
    monkeypatch.setattr(neko, "_neko_package_version", lambda: "offline")

    def export_bnet(network, path, **kwargs):
        Path(path).write_text("A, !C\nC, A\n")
        return {}

    monkeypatch.setattr(neko, "_export_sanitized_bnet", export_bnet)
    result = inspect.unwrap(neko.export_neko_handoff)(
        biological_context="offline derivative",
        output_nodes=["C"],
        artifact_prefix="current",
        session_id=ns.session_id,
    )
    neko_path = result.structured_content["manifest_file"]["path"]
    current = load_handoff_manifest(neko_path)
    assert current.schema_version == "2.0"
    assert current.network.nodes == ["A", "C"]
    assert current.source_ancestry == [ns.source_import]

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
        with Path(bnd).open("w") as out:
            sim.print_bnd(out)
        with Path(cfg).open("w") as out:
            sim.print_cfg(out)

    ms = MaBoSSSession("maboss-child")
    monkeypatch.setattr(maboss, "ensure_session", lambda _: ms)
    monkeypatch.setattr(maboss.maboss, "bnet_to_bnd_and_cfg", convert)
    monkeypatch.setattr(maboss.maboss, "load", lambda *args: sim)
    monkeypatch.setattr(maboss, "_maboss_package_version", lambda: "offline")
    inspect.unwrap(maboss.import_neko_handoff)(
        manifest_path=neko_path, artifact_prefix="imported", session_id=ms.session_id
    )
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
    assert downstream.source_ancestry == current.source_ancestry
    assert downstream.parent_manifest.sha256  # current stage identity, not SIF parent
    assert downstream.lineage == [current.source]

    ps = SessionState(session_id="physicell-child")
    ps.config = _handoff_config("tumour")
    monkeypatch.setattr(physicell, "_require_configuration", lambda _: ps)
    monkeypatch.setattr(physicell, "PHYSIBOSS_AVAILABLE", True)
    final = inspect.unwrap(physicell.import_maboss_handoff)(
        manifest_path=maboss_path,
        artifact_prefix="attached",
        replace_existing=False,
        session_id=ps.session_id,
    )
    payload = final.structured_content
    assert (
        payload["source_manifest"]["source_ancestry"][0]["relationship"]
        == "edited-derivative"
    )
    snapshot = load_handoff_manifest(payload["manifest_snapshot_file"]["path"])
    assert snapshot.source_ancestry == current.source_ancestry
    lineage.copy_source_ancestry(
        snapshot.source_ancestry, tmp_path / "portable-capture"
    )
    removed = downstream.model_dump()
    removed["source_ancestry"] = []
    bad = type(downstream).model_validate(removed)
    bad_path = write_handoff_manifest(tmp_path / "missing-inherited.json", bad)
    previous_config = ps.config
    with pytest.raises(ValueError, match="source ancestry"):
        inspect.unwrap(physicell.import_maboss_handoff)(
            manifest_path=str(bad_path),
            artifact_prefix="rejected",
            replace_existing=True,
            session_id=ps.session_id,
        )
    assert ps.config is previous_config


def test_ordinary_bnet_derivative_export_has_source_parent_only(tmp_path, monkeypatch):
    import inspect
    from types import SimpleNamespace

    from MaBoSS import server
    from MaBoSS.session_manager import MaBoSSSession

    imported, original, parent, edited = derivative(
        tmp_path, suffix=".bnet", server="MaBoSS", session="child"
    )
    sess = MaBoSSSession("child")
    monkeypatch.setattr(server, "ensure_session", lambda _: sess)

    class Network(dict):
        def get_output(self):
            return ["C"]

    sim = SimpleNamespace(
        network=Network(A=None, C=None),
        param={},
        print_bnd=lambda out: out.write("Node A {logic = !A;}\nNode C {logic = A;}\n"),
        print_cfg=lambda out: out.write("max_time = 100;\n"),
    )

    def convert(source, bnd, cfg):
        with Path(bnd).open("w") as out:
            sim.print_bnd(out)
        with Path(cfg).open("w") as out:
            sim.print_cfg(out)

    monkeypatch.setattr(server.maboss, "bnet_to_bnd_and_cfg", convert)
    monkeypatch.setattr(server.maboss, "load", lambda *args: sim)
    monkeypatch.setattr(server, "_maboss_package_version", lambda: "offline")
    inspect.unwrap(server.bnet_to_bnd_and_cfg)(
        str(edited),
        session_id="child",
        parent_manifest=str(parent),
        derived_from=str(original),
    )
    inspect.unwrap(server.build_simulation)(
        session_id="child", bnd_path=None, cfg_path=None
    )
    original.unlink()
    parent.unlink()
    edited.unlink()
    result = inspect.unwrap(server.export_maboss_handoff)(
        target_cell_type="tumour",
        biological_context="ordinary edited import",
        simulation_summary=None,
        include_result=False,
        artifact_prefix="current",
        session_id="child",
    )
    manifest = load_handoff_manifest(result.structured_content["manifest_file"]["path"])
    assert manifest.schema_version == "2.1"
    assert manifest.model_revision and manifest.readouts == {}
    assert manifest.parent_manifest is None
    assert manifest.lineage == []
    assert manifest.source_ancestry[0].relationship == "edited-derivative"
    assert manifest.network.nodes == ["A", "C"]


@pytest.mark.parametrize("version", ["1.0", "2.0"])
@pytest.mark.parametrize("mismatch", ["nodes", "outputs"])
def test_physicell_stage_checks_remain_exact(tmp_path, version, mismatch):
    from types import SimpleNamespace

    from PhysiCell.services.handoff import import_maboss_handoff_transaction
    from tests.test_handoff_contracts import _maboss_manifest

    manifest = _maboss_manifest(tmp_path, with_neko_parent=True)
    data = manifest.model_dump()
    data["schema_version"] = version
    if version == "2.0":
        data["source_ancestry"] = []
    if mismatch == "nodes":
        data["network"]["nodes"].append("C")
        bnd = Path(manifest.bnd_file.path)
        bnd.write_text(bnd.read_text() + "Node C { logic = A; }\n")
        from mcp_biomodelling_servers.handoff import handoff_artifact

        data["bnd_file"] = handoff_artifact(
            bnd,
            server="MaBoSS",
            session_id=manifest.source.session_id,
            role="maboss_bnd",
        ).model_dump()
    else:
        data["network"]["output_nodes"] = ["A"]
    current = type(manifest).model_validate(data)
    path = write_handoff_manifest(tmp_path / "mismatch.json", current)
    with pytest.raises(ValueError, match="node sets|output-node"):
        import_maboss_handoff_transaction(
            session=SimpleNamespace(),
            manifest_path=str(path),
            artifact_prefix="unused",
            replace_existing=False,
            server_root=tmp_path,
            require_unused_paths=None,
            copy_verified_artifact=None,
            link_artifact_without_overwrite=None,
            rollback_artifacts=None,
            physiboss_tracking=None,
        )


def test_missing_sidecar_ancestry_and_unrelated_bytes_fail(tmp_path):
    import hashlib

    imported, _, parent, _ = derivative(tmp_path)
    record = lineage.SourceExport.model_validate_json(parent.read_bytes()).model_dump()
    del record["imports"]
    with pytest.raises(ValueError, match="explicit imports"):
        lineage.SourceExport.model_validate(record)
    data = imported.model_dump()
    data["evidence"]["blobs"][hashlib.sha256(b"unrelated history").hexdigest()] = (
        base64.b64encode(b"unrelated history").decode()
    )
    with pytest.raises(ValueError, match="unrelated"):
        lineage.verify_source_ancestry([SourceImport.model_validate(data)])
