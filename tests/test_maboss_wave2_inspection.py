"""MCP-level Wave 2 inspection of real pyMaBoSS models (B-MB-5/7/8/9)."""
import pytest

from MaBoSS.external import pymaboss as maboss
from MaBoSS.services.model import model_revision
from tests.test_maboss_mcp_errors import (
    _call_tool,
    _clear_sessions,
    _create_simulation_session,
    _run,
    session_manager,
)
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


def test_rules_flag_custom_rates_without_replacing_the_rule():
    sim = model_session().sim
    sim.network["B"].rt_up = "A * 3"
    sid = _create_simulation_session(sim)

    rules = {item["node"]: item for item in call("get_maboss_logical_rules", session_id=sid)["rules"]}

    assert rules["A"] == {"node": "A", "rule": "B"}
    assert rules["B"]["rule"] == "A"
    assert "Custom up rate expression" in rules["B"]["rule_limitation"]


def test_rules_without_logic_report_null_rule_and_limitation():
    node = maboss.Node("A", "", "1.0", "1.0")
    sid = _create_simulation_session(maboss.Simulation(maboss.Network([node])))

    (rule,) = call("get_maboss_logical_rules", session_id=sid)["rules"]

    assert rule["rule"] is None
    assert rule["rule_limitation"].startswith("No explicit Boolean logic")


def test_rules_report_truncation_beyond_default_bound():
    nodes = [maboss.Node(f"N{i}", f"N{(i + 1) % 101}") for i in range(101)]
    sid = _create_simulation_session(maboss.Simulation(maboss.Network(nodes)))

    result = _run(_call_tool("get_maboss_logical_rules", {"session_id": sid}))
    payload = result.structured_content

    assert payload["rule_count"] == 100
    assert payload["total_node_count"] == 101
    assert payload["truncated"] is True
    assert "Showing 100 of 101 nodes" in result.content[0].text
    selected = call("get_maboss_logical_rules", session_id=sid, selected_nodes=["N100"])
    assert [item["node"] for item in selected["rules"]] == ["N100"]
    assert "truncated" not in selected


def test_rate_edit_through_tool_reads_back_and_rejects_stale_revision():
    sim = model_session().sim
    sid = _create_simulation_session(sim)
    revision = call("get_maboss_nodes", session_id=sid, selected_nodes=["A"])["model_revision"]

    result = call("update_maboss_parameters", session_id=sid, expected_revision=revision,
                  rates=[{"node": "A", "down": 0.5}])

    changes = result["rate_changes"]
    assert changes["previous_revision"] == revision
    assert changes["after"][0]["rate_expressions"] == {"up": "@logic ? $shared : 0", "down": "@logic ? 0 : 0.5"}
    session = session_manager.get_session(sid)
    assert session.sim.network["B"].rt_down == "@logic ? 0 : $shared"
    assert result["model_revision"] == model_revision(session.sim, session.readouts)
    stale = _run(_call_tool("update_maboss_parameters", {
        "session_id": sid, "expected_revision": revision, "rates": [{"node": "A", "down": 1}]}))
    assert stale.is_error and "stale_model_revision" in stale.content[0].text
    assert session.sim.network["A"].rt_down == "@logic ? 0 : 0.5"


def test_effective_configuration_and_output_visibility(tmp_path):
    sim = model_session().sim
    sim.network.set_output(["A"])
    sim.update_parameters(statdist_traj_count=0)
    sid = _create_simulation_session(sim)

    parameters = call("update_maboss_parameters", session_id=sid)
    nodes = call("get_maboss_nodes", session_id=sid, selected_nodes=["A", "B"])

    assert parameters["stationary_distribution_enabled"] is False
    assert parameters["mode"] == "inspect"
    assert nodes["output_nodes"] == ["A"]
    assert {item["node"]: item["output"] for item in nodes["details"]} == {"A": True, "B": False}
