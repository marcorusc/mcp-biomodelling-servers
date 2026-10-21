"""Exact input preservation, restart reads, failure isolation and immutable plots."""
import json
from pathlib import Path

import pytest

from MaBoSS.external import maboss_executable
from MaBoSS.services import artifacts
from MaBoSS.services.model import model_revision
from MaBoSS.services.runs import load_run
from tests.test_maboss_mcp_errors import (
    _call_tool,
    _clear_sessions,
    _create_simulation_session,
    _run,
    maboss_server,
    session_manager,
)
from tests.test_maboss_preservation import FakeResult
from tests.test_maboss_wave2_model import session as model_session


@pytest.fixture(autouse=True)
def sessions():
    _clear_sessions()
    yield
    _clear_sessions()


def call(name, **arguments):
    result = _run(_call_tool(name, arguments))
    assert not result.is_error, result.content
    return result.structured_content


def test_two_ordinary_two_mutants_survive_restart_and_never_relabel(tmp_path, monkeypatch):
    monkeypatch.setattr(maboss_server, "_SERVER_ROOT", tmp_path / "MaBoSS")
    sim = model_session().sim
    calls = []
    def run_for(model):
        calls.append(model_revision(model))
        result = FakeResult(tmp_path / f"engine{len(calls)}")
        with Path(result._bnd).open("w") as out:
            model.print_bnd(out=out)
        with Path(result._cfg).open("w") as out:
            model.print_cfg(out=out)
        return result
    sim.run = lambda: run_for(sim)
    original_copy = sim.copy
    def copy():
        mutant = original_copy()
        mutant.run = lambda: run_for(mutant)
        return mutant
    sim.copy = copy
    sid = _create_simulation_session(sim)
    first = call("run_simulation", session_id=sid)
    sim.param["max_time"] = 8
    second = call("run_simulation", session_id=sid)
    mutants = [call("simulate_mutation", session_id=sid, nodes="A", state=state) for state in ("OFF", "ON")]
    assert session_manager.get_session(sid).latest_run_id == second["run_id"]
    assert model_revision(sim) == second["model_revision"]
    results = [first, second, *mutants]
    assert len({item["run_id"] for item in results}) == 4
    assert first["model_revision"] != second["model_revision"]
    snapshots = []
    for item in results:
        manifest, path = load_run(maboss_server._SERVER_ROOT, item["run_id"], sid)
        inputs = [path.parent / f["name"] for f in manifest["files"] if f["role"] == "engine_input"]
        snapshots.append([p.read_bytes() for p in inputs])
    assert snapshots[2] != snapshots[3] != snapshots[1]
    _clear_sessions()  # restart: there is no live model or result
    for item in results:
        read = call("get_simulation_result", session_id=sid, run_id=item["run_id"])
        assert read["model_revision"] == item["model_revision"]
        plot1 = call("visualize_network_trajectories", session_id=sid, run_id=item["run_id"])
        plot2 = call("visualize_network_trajectories", session_id=sid, run_id=item["run_id"], until=1)
        assert plot1["image_file"]["path"] != plot2["image_file"]["path"]
        assert Path(plot1["image_file"]["path"]).is_file()
    assert len(calls) == 4  # reads and plots never execute an engine
    wrong = _run(_call_tool("get_simulation_result", {"session_id": "another-session", "run_id": first["run_id"]}))
    assert wrong.is_error
    manifest = json.loads(Path(first["run_manifest"]).read_text())
    file = Path(first["run_manifest"]).parent / manifest["files"][0]["name"]
    file.write_text("tampered")
    assert _run(_call_tool("get_simulation_result", {"run_id": first["run_id"]})).is_error


def test_failed_save_keeps_previous_selection_and_base_model(tmp_path, monkeypatch):
    monkeypatch.setattr(maboss_server, "_SERVER_ROOT", tmp_path / "MaBoSS")
    sim = model_session().sim
    calls = []
    def run():
        calls.append(1)
        return FakeResult(tmp_path / f"engine{len(calls)}")
    sim.run = run
    sid = _create_simulation_session(sim)
    first = call("run_simulation", session_id=sid)
    session = session_manager.get_session(sid)
    previous_result = session.result
    def fail(*args, **kwargs):
        raise OSError("injected disk failure")
    monkeypatch.setattr(artifacts.shutil, "copyfile", fail)
    failed = _run(_call_tool("run_simulation", {"session_id": sid}))
    assert failed.is_error and "persistence failed" in failed.content[0].text
    assert session.latest_run_id == first["run_id"] and session.result is previous_result
    assert len(calls) == 2 and session.sim is sim
    assert Path(first["run_manifest"]).is_file()


def test_failed_engine_preserves_diagnostics_and_does_not_select_run(tmp_path, monkeypatch):
    monkeypatch.setattr(maboss_server, "_SERVER_ROOT", tmp_path / "MaBoSS")
    sim = model_session().sim
    failed_result = FakeResult(tmp_path / "engine")
    failed_result._err = 1
    sim.run = lambda: failed_result
    sid = _create_simulation_session(sim)
    failed = _run(_call_tool("run_simulation", {"session_id": sid}))
    assert failed.is_error and "engine failed" in failed.content[0].text
    assert session_manager.get_session(sid).latest_run_id is None
    records = list(tmp_path.rglob("failed_*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["phase"] == "engine" and record["status"] == "incomplete"
    assert record["diagnostic_files"]
    assert not list(tmp_path.rglob("run_*.json"))


def test_diagnostic_storage_failure_does_not_mask_original_phase(tmp_path):
    from MaBoSS.services.runs import record_failure
    result = record_failure(tmp_path / "missing", "fixture", "engine", "exit 1", "revision")
    assert "diagnostic storage also failed" in str(result)


def test_missing_maboss_executable_is_named(tmp_path, monkeypatch):
    monkeypatch.setattr(maboss_server, "_SERVER_ROOT", tmp_path / "MaBoSS")
    sid = _create_simulation_session(model_session().sim)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    failed = _run(_call_tool("run_simulation", {"session_id": sid}))

    assert failed.is_error
    assert "The MaBoSS executable 'MaBoSS' is not on PATH" in failed.content[0].text
    assert session_manager.get_session(sid).latest_run_id is None


def test_executable_guard_only_translates_maboss_executables():
    with pytest.raises(RuntimeError, match="'MaBoSS_128n' is not on PATH"):
        with maboss_executable():
            raise FileNotFoundError(2, "No such file or directory", "MaBoSS_128n")
    with pytest.raises(FileNotFoundError):
        with maboss_executable():
            raise FileNotFoundError(2, "No such file or directory", "model.bnd")
