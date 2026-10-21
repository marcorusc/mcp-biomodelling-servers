"""Direct handler regressions using offline backend doubles; no MCP calls."""
import asyncio
import inspect
import threading
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from mcp_biomodelling_servers.import_lineage import SourceExport, write_source_export


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["backend", "bnd-sidecar", "cfg-sidecar", "record",
                                     "publish-bnd", "publish-cfg", "publish-bnd-sidecar",
                                     "publish-cfg-sidecar", "publish-record"])
def test_bnet_failed_conversion_preserves_files_and_session(tmp_path, monkeypatch, existing, failure):
    from MaBoSS import server
    from MaBoSS.services import artifacts
    from MaBoSS.session_manager import MaBoSSSession

    sess = MaBoSSSession("offline")
    monkeypatch.setattr(server, "ensure_session", lambda _: sess)
    source = tmp_path / "source.bnet"
    source.write_text("A, A\n")

    def convert(source, bnd, cfg):
        Path(bnd).write_text(Path(source).read_text())
        Path(cfg).write_text("max_time = 100;\n")

    monkeypatch.setattr(server.maboss, "bnet_to_bnd_and_cfg", convert)
    sim = SimpleNamespace(param={})
    monkeypatch.setattr(server.maboss, "load", lambda *args: sim)
    handler = inspect.unwrap(server.bnet_to_bnd_and_cfg)
    build = inspect.unwrap(server.build_simulation)
    if existing:
        handler(str(source), session_id="offline")
        build(session_id="offline", bnd_path=None, cfg_path=None)
        # Preserve both the loaded model and a newer pending conversion.
        handler(str(source), session_id="offline")
    art_dir = tmp_path / "artifact-cache" / "MaBoSS" / "offline"

    def inventory():
        return {p.name: p.read_bytes() for p in art_dir.iterdir() if p.is_file()} if art_dir.exists() else {}

    before = inventory()
    pending, active, history = sess.pending_source_import, sess.source_import, list(sess.import_history)
    source.write_text("A, !A\n")
    export = server.write_source_export
    replace = artifacts.os.replace
    link = artifacts.os.link
    publication_names = ["output.bnd", "output.cfg", "output.bnd.lineage.json",
                         "output.cfg.lineage.json"]
    failed = False

    def fail(*args, **kwargs):
        raise OSError("injected failure")

    def sidecar(path, **kwargs):
        if failure == {".bnd": "bnd-sidecar", ".cfg": "cfg-sidecar"}[path.suffix]:
            fail()
        return export(path, **kwargs)

    def publish(src, dst):
        nonlocal failed
        dst = Path(dst)
        kind = ("publish-bnd", "publish-cfg", "publish-bnd-sidecar", "publish-cfg-sidecar")
        target = kind[publication_names.index(dst.name)] if dst.name in publication_names else "publish-record"
        if dst.parent == art_dir and target == failure and not failed:
            failed = True
            fail()
        return replace(src, dst)

    def publish_record(src, dst):
        if Path(dst).parent == art_dir and failure == "publish-record":
            fail()
        return link(src, dst)

    monkeypatch.setattr(server, "write_source_export", sidecar)
    monkeypatch.setattr(artifacts.os, "replace", publish)
    monkeypatch.setattr(artifacts.os, "link", publish_record)
    if failure == "backend":
        def partial_convert(source, bnd, cfg):
            Path(bnd).write_text("incomplete replacement")
            fail()
        monkeypatch.setattr(server.maboss, "bnet_to_bnd_and_cfg", partial_convert)
    if failure == "record":
        monkeypatch.setattr(server, "write_source_import", fail)
    with pytest.raises((OSError, RuntimeError), match="injected failure"):
        handler(str(source), session_id="offline")
    assert inventory() == before
    assert not list(art_dir.glob(".bnet-conversion-*"))
    assert sess.pending_source_import is pending
    assert sess.source_import is active
    assert sess.import_history == history
    if existing:
        assert sess.sim is sim
        build(session_id="offline", bnd_path=None, cfg_path=None)
        assert sess.source_import == pending


def test_bnet_old_call_parent_rejection_and_loaded_lineage(tmp_path, monkeypatch):
    from MaBoSS import server
    from MaBoSS.session_manager import MaBoSSSession
    sess = MaBoSSSession("offline")
    monkeypatch.setattr(server, "ensure_session", lambda _: sess)
    monkeypatch.setattr(server, "_SERVER_ROOT", tmp_path / "MaBoSS")
    calls = []

    def convert(source, bnd, cfg):
        calls.append(source)
        Path(bnd).write_text("Node A {logic = A;}\n")
        Path(cfg).write_text("max_time = 100;\n")

    monkeypatch.setattr(server.maboss, "bnet_to_bnd_and_cfg", convert)
    sim = SimpleNamespace(param={}, print_bnd=lambda out: out.write("Node A {logic = A;}\n"),
                          print_cfg=lambda out: out.write("max_time = 100;\n"))
    monkeypatch.setattr(server.maboss, "load", lambda *args: sim)
    handler = inspect.unwrap(server.bnet_to_bnd_and_cfg)
    source = tmp_path / "original.bnet"
    source.write_text("A, A\n")
    handler(str(source), session_id="offline")  # Existing signature remains usable.
    assert len(calls) == 1
    assert sess.pending_source_import.relationship == "standalone"
    assert sess.source_import is None  # Conversion did not replace the loaded model.
    build = inspect.unwrap(server.build_simulation)
    build(session_id="offline", bnd_path=None, cfg_path=None)
    assert sess.source_import.relationship == "standalone"
    build(session_id="offline", bnd_path=None, cfg_path=None)
    assert sess.source_import.relationship == "standalone"
    previous = sess.source_import
    parent = write_source_export(source, server="NeKo", session_id="ancestor", imports=[])
    derivative = tmp_path / "edited.bnet"
    derivative.write_text("A, !A\n")
    with pytest.raises(ValueError, match="match"):
        handler(str(derivative), session_id="offline", parent_manifest=str(parent))
    assert len(calls) == 1
    assert sess.source_import == previous
    assert len(sess.import_history) == 1
    handler(str(derivative), session_id="offline", parent_manifest=str(parent), derived_from=str(source))
    assert sess.source_import == previous
    assert sess.pending_source_import.relationship == "edited-derivative"
    converted = tmp_path / "artifact-cache" / "MaBoSS" / "offline" / "output.bnd"
    intact = converted.read_bytes()
    converted.write_bytes(intact.replace(b"logic = A", b"logic = B"))
    with pytest.raises(ValueError, match="digest changed"):
        build(session_id="offline", bnd_path=None, cfg_path=None)
    assert sess.source_import == previous
    converted.write_bytes(intact)
    build(session_id="offline", bnd_path=None, cfg_path=None)
    assert sess.source_import.relationship == "edited-derivative"
    inspect.unwrap(server.export_maboss_bnd_cfg)(prefix="review", overwrite=False, session_id="offline")
    sidecars = list((tmp_path / "artifact-cache" / "MaBoSS" / "offline").glob("review.*.lineage.json"))
    assert len(sidecars) == 2
    assert all(SourceExport.model_validate_json(p.read_bytes()).imports[0] == sess.source_import
               for p in sidecars)
    sess.set_simulation(sim, "unrelated.bnd", "unrelated.cfg")
    assert sess.source_import is None
    assert sess.pending_source_import is None
    assert len(sess.import_history) == 2


def test_sif_handler_rejects_before_backend_and_retains_export(tmp_path, monkeypatch):
    # This module installs NeKo import stubs before importing the server. Merely
    # importing it runs no tests, protocol calls, or scientific initialization.
    from NeKo.session_manager import NeKoSessionManager
    from tests.test_neko_mcp_errors import neko_server as server
    manager = NeKoSessionManager()
    sess = manager.get_session(manager.create_session())
    monkeypatch.setattr(server, "session_manager", manager)
    monkeypatch.setattr(server, "ensure_session", lambda _: sess)
    edges = pd.DataFrame([{"source": "A", "target": "B", "Effect": "stimulation"}])
    calls = []

    class Network:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.nodes = pd.DataFrame({"Genesymbol": ["A", "B"]})
            self.edges = edges
        def convert_edgelist_into_genesymbol(self):
            return edges

    async def progress(*args):
        pass

    event_loop_thread = threading.get_ident()
    original_capture = server.capture_import
    capture_threads = []

    def capture(*args, **kwargs):
        capture_threads.append(threading.get_ident())
        assert capture_threads[-1] != event_loop_thread
        return original_capture(*args, **kwargs)

    monkeypatch.setattr(server, "Network", Network)
    monkeypatch.setattr(server, "capture_import", capture)
    source = tmp_path / "parent.sif"
    source.write_text("A\t1\tB\n")
    parent = write_source_export(source, server="NeKo", session_id="parent", imports=[])
    edited = tmp_path / "edited.sif"
    edited.write_text("A\t-1\tB\n")
    arguments = {
        "list_of_initial_genes": [], "ctx": SimpleNamespace(report_progress=progress),
        "database": "omnipath", "sif_file": str(edited), "max_len": 2,
        "path_policy": "one_shortest", "reuse_policy": "discovered_paths",
        "only_signed": True, "consensus": True, "session_id": sess.session_id,
        "verbosity": "summary",
    }

    with pytest.raises(ValueError, match="match"):
        asyncio.run(server.create_network(**arguments, parent_manifest=str(parent)))
    assert calls == []
    assert sess.network is None
    asyncio.run(server.create_network(**arguments))
    assert sess.source_import.relationship == "standalone"
    asyncio.run(server.create_network(**arguments, parent_manifest=str(parent), derived_from=str(source)))
    assert len(capture_threads) == 3
    assert sess.source_import.relationship == "edited-derivative"
    assert len(sess.import_history) == 2
    output_dir = tmp_path / "export"
    output_dir.mkdir()
    monkeypatch.setattr(server, "_export_dir", lambda _: output_dir)
    monkeypatch.setattr(server, "_session_network", lambda _: (sess, sess.network))
    monkeypatch.setattr(server, "Exports", lambda _: SimpleNamespace(export_sif=lambda p: Path(p).write_text("A\t-1\tB\n")))
    inspect.unwrap(server.export_network)(format="sif", session_id=sess.session_id, verbosity="summary")
    record = SourceExport.model_validate_json((output_dir / "Network.sif.lineage.json").read_bytes())
    assert record.imports == [sess.source_import]
    sess.set_network(None)
    assert sess.source_import is None
    assert len(sess.import_history) == 2
