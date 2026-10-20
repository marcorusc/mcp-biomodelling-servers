"""Real Settings object and MCP boundary checks for the approved Wave 2 surface."""
import hashlib
import json
import zipfile
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from physicell_config import PhysiCellConfig

from PhysiCell.services.bundle import export_bundle
from PhysiCell.session_manager import SessionState
from tests.test_physicell_mcp_errors import (
    _call_tool,
    _clear_sessions,
    _create_session,
    _run,
    session_manager,
)


@pytest.fixture(autouse=True)
def sessions(monkeypatch):
    from PhysiCell import server
    monkeypatch.setattr(server, "PhysiCellConfig", PhysiCellConfig)
    _clear_sessions()
    yield
    _clear_sessions()


def config():
    value = PhysiCellConfig()
    value.cell_types.add_cell_type("tumour")
    return value


def call(tool, sid, **kwargs):
    result = _run(_call_tool(tool, {"session_id": sid, **kwargs}))
    assert not result.is_error, result.content
    return result.structured_content


def test_stored_zero_false_cycle_readback():
    sid = _create_session(config())
    result = call("configure_cell_parameters", sid, cell_type="tumour", motility_speed=0,
                  motility_enabled=False, cycle_transitions=[{"from_phase": 0, "to_phase": 1, "rate": 0, "fixed_duration": False}])
    stored = result["stored"]
    assert stored["motility"]["speed"] == 0
    assert stored["cycle"]["transition_rates"][0]["rate"] == 0
    assert stored["cycle"]["transition_rates"][0]["fixed_duration"] is False
    assert call("configure_cell_parameters", sid, cell_type="tumour")["stored"] == stored


def test_typed_inputs_csv_and_bundle(tmp_path):
    sid = _create_session(config())
    csv = tmp_path / "cells.csv"
    csv.write_text("x,y,z,type\n0,1,0,tumour\n")
    result = call("configure_simulation_inputs", sid, parameters=[
        {"name": "number_of_cells", "type": "int", "value": 0, "units": "cells"},
        {"name": "flag", "type": "bool", "value": False, "units": "dimensionless"}],
        cells_csv=str(csv), csv_enabled=True)
    assert result["parameters"]["flag"]["value"] is False
    assert result["number_of_cells"]["value"] == 0
    csv.unlink()  # preserved input no longer depends on the original
    first = call("export_configuration_bundle", sid)
    second = call("export_configuration_bundle", sid)
    assert first["bundle"] != second["bundle"]
    with zipfile.ZipFile(first["bundle"]) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for entry in manifest["files"]:
            assert hashlib.sha256(archive.read(entry["path"])).hexdigest() == entry["sha256"]
        xml = ET.fromstring(archive.read("config/PhysiCell_settings.xml"))
        assert xml.findtext("./initial_conditions/cell_positions/folder") == "config"
        assert archive.read("config/cells.csv") == b"x,y,z,type\n0,1,0,tumour\n"
    stored = session_manager.get_session(sid).config.initial_conditions.csv_config
    assert Path(stored["folder"]).is_absolute()


def test_bundle_two_models_same_names_and_active_config_unchanged(tmp_path):
    cfg = config()
    cfg.cell_types.add_cell_type("immune")
    for i, cell in enumerate(("tumour", "immune")):
        folder = tmp_path / str(i)
        folder.mkdir()
        (folder / "model.bnd").write_text(f"node N{i} {{logic = 1;}}")
        (folder / "model.cfg").write_text("max_time=1;")
        cfg.physiboss.add_intracellular_model(cell, bnd_filename=str(folder / "model.bnd"), cfg_filename=str(folder / "model.cfg"))
    before = deepcopy(cfg.cell_types.get_cell_types())
    session = SessionState("fixture", config=cfg)
    result = export_bundle(session, tmp_path / "PhysiCell")
    assert cfg.cell_types.get_cell_types() == before
    with zipfile.ZipFile(result["bundle"]) as archive:
        assert archive.read("config/boolean_models/cell_0.bnd") != archive.read("config/boolean_models/cell_1.bnd")


def test_default_visible_before_export():
    sid = _create_session()
    call("create_simulation_domain", sid, domain_x=100, domain_y=100)
    result = call("list_loaded_components", sid, component_type="substrates")
    assert result["substrate_count"] == 1
    assert result["substrates"][0]["origin"] == "library_default"
    assert result["substrates"][0]["stored"]["initial_units"] == "mmHg"
    session = session_manager.get_session(sid)
    before = deepcopy(session.config.substrates.get_substrates())
    duplicate = _run(_call_tool("add_single_substrate", {
        "session_id": sid, "substrate_name": next(iter(before)),
        "diffusion_coefficient": 1, "decay_rate": 0, "initial_condition": 0,
    }))
    assert duplicate.is_error and "already exists" in duplicate.content[0].text
    assert session.config.substrates.get_substrates() == before


def test_invalid_inputs_atomic_and_missing_bundle_not_published(tmp_path):
    sid = _create_session(config())
    result = _run(_call_tool("configure_simulation_inputs", {"session_id": sid, "parameters": [
        {"name": "number_of_cells", "type": "int", "value": False, "units": "cells"}]}))
    assert result.is_error
    session = session_manager.get_session(sid)
    session.config.initial_conditions.add_csv_file("missing.csv", folder=str(tmp_path), enabled=True)
    with pytest.raises(ValueError, match="Missing"):
        export_bundle(session, tmp_path / "PhysiCell")
    assert not list(tmp_path.rglob("bundle_*.zip"))


def test_fractional_timing_and_integer_steepness_are_stored(tmp_path):
    cfg = config()
    cfg.physiboss.add_intracellular_model("tumour", bnd_filename="fixture.bnd", cfg_filename="fixture.cfg")
    sid = _create_session(cfg)
    result = call("configure_physiboss_settings", sid, cell_type="tumour", time_stochasticity=0.25)
    assert result["stored"]["time_stochasticity"] == 0.25
    mapped = call("add_physiboss_output_link", sid, cell_type="tumour", boolean_node="A",
                  physicell_behavior="cycle entry", steepness=0, value=0, base_value=0)
    assert mapped["stored"]["settings"]["steepness"] == 0
    assert mapped["stored"]["settings"]["value"] == 0
    invalid = _run(_call_tool("add_physiboss_output_link", {"session_id": sid, "cell_type": "tumour",
                   "boolean_node": "A", "physicell_behavior": "cycle entry", "steepness": 0.5}))
    assert invalid.is_error


def test_portable_rules_use_stored_updates_and_keep_original_export(tmp_path):
    cfg = config()
    cfg.cell_rules.add_rule("tumour", "pressure", "decreases", "cycle entry", saturation_value=0, half_max=1, hill_power=2)
    sid = _create_session(cfg)
    old = call("export_cell_rules_csv", sid)
    original = Path(old["file"]["path"]).read_bytes()
    session = session_manager.get_session(sid)
    assert session.config.cell_rules.get_rulesets()["default"]["folder"] == str(Path(old["file"]["path"]).parent)
    session.config.cell_rules.add_rule("tumour", "pressure", "increases", "apoptosis", saturation_value=1, half_max=1, hill_power=2)
    before = deepcopy(session.config.cell_rules.get_rulesets())
    result = call("export_configuration_bundle", sid)
    assert session.config.cell_rules.get_rulesets() == before
    assert Path(old["file"]["path"]).read_bytes() == original
    with zipfile.ZipFile(result["bundle"]) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        generated = [item for item in manifest["files"] if item.get("source") == "stored_cell_rules"]
        assert len(generated) == 1
        assert b"apoptosis" in archive.read(generated[0]["path"])
