"""Direct handler regressions using offline backend doubles; no MCP calls."""
import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from mcp_biomodelling_servers.import_lineage import SourceExport, write_source_export


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
        def convert_edgelist_into_genesymbol(self):
            return edges

    async def progress(*args):
        pass

    # Run synchronously within the async test to avoid sandbox thread shutdown.
    async def run_sync(function):
        return function()

    monkeypatch.setattr(server, "Network", Network)
    monkeypatch.setattr(server.anyio.to_thread, "run_sync", run_sync)
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
