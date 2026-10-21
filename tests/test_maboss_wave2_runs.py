"""Exact input preservation, restart reads, failure isolation and immutable plots."""
import json
from pathlib import Path

import pytest

from MaBoSS.external import pymaboss as maboss
from MaBoSS.services import artifacts, engine
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


def test_real_runs_use_cmaboss_without_a_maboss_executable(tmp_path, monkeypatch):
    monkeypatch.setattr(maboss_server, "_SERVER_ROOT", tmp_path / "MaBoSS")
    sim = model_session().sim
    sim.network.set_output(["A", "B"])
    sim.update_parameters(sample_count=200, max_time=2, time_tick=0.5, thread_count=1)
    sid = _create_simulation_session(sim)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    ordinary = call("run_simulation", session_id=sid)
    mutant = call("simulate_mutation", session_id=sid, nodes="A", state="OFF")

    manifest, path = load_run(maboss_server._SERVER_ROOT, ordinary["run_id"], sid)
    inputs = {entry["source_name"]: (path.parent / entry["name"]).read_text()
              for entry in manifest["files"] if entry["role"] == "engine_input"}
    assert inputs == {"input.bnd": str(sim.network), "input.cfg": sim.str_cfg()}
    outputs = {entry["source_name"] for entry in manifest["files"] if entry["role"] == "engine_output"}
    assert {"res_probtraj.csv", "res_fp.csv", "res_statdist.csv", "res_run.txt"} <= outputs
    assert manifest["state_timepoints"][-1] == pytest.approx(1.5)
    mutant_result = call("get_simulation_result", session_id=sid, run_id=mutant["run_id"])
    assert mutant_result["run_id"] == mutant["run_id"]
    assert session_manager.get_session(sid).latest_run_id == ordinary["run_id"]


def test_engine_crash_is_a_recorded_failure_not_a_server_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(maboss_server, "_SERVER_ROOT", tmp_path / "MaBoSS")
    sid = _create_simulation_session(model_session().sim)
    crash = tmp_path / "crash.py"
    crash.write_text("import os, sys\nprint('engine exploded', file=sys.stderr)\nsys.stderr.flush()\nos.abort()\n")
    monkeypatch.setattr(engine, "WORKER", crash)

    failed = _run(_call_tool("run_simulation", {"session_id": sid}))

    assert failed.is_error
    assert "engine exploded" in failed.content[0].text
    (record,) = list(tmp_path.rglob("failed_*.json"))
    assert json.loads(record.read_text())["phase"] == "engine"
    assert session_manager.get_session(sid).latest_run_id is None


def test_rejected_model_reports_the_engine_message():
    node = maboss.Node("A", "A", "@logic ? $missing : 0", "@logic ? 0 : 1")
    with pytest.raises(ValueError, match=r"MaBoSS rejected the model: .*\$missing"):
        maboss.Simulation(maboss.Network([node]))
