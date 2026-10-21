"""Real pyMaBoSS object contracts without running a stochastic engine."""
from types import SimpleNamespace

import pytest

from MaBoSS.external import pymaboss as maboss
from MaBoSS.services.model import (
    RateEdit,
    ReadoutEdit,
    edit_rates,
    edit_readouts,
    inspect_nodes,
    model_revision,
    print_bnd_with_readouts,
    read_readouts,
)


def session():
    a = maboss.Node("A", "B", "@logic ? $shared : 0", "@logic ? 0 : $shared")
    b = maboss.Node("B", "A", "@logic ? $shared : 0", "@logic ? 0 : $shared")
    sim = maboss.Simulation(maboss.Network([a, b]), parameters={"$shared": 2})
    return SimpleNamespace(sim=sim, readouts={}, session_id="fixture")


def request(**kw):
    return ReadoutEdit(operation="add", node="Readout", rule="A & !B", explanation="Explicit fixture readout", rate_up=1, rate_down=2, initial_probability_on=0, **kw)


def test_rates_preserve_logic_other_nodes_and_shared_parameter():
    sess = session()
    old = model_revision(sess.sim)
    result = edit_rates(sess, [RateEdit(node="A", up=0)], old)
    assert sess.sim.network["A"].rt_up == "@logic ? 0.0 : 0"
    assert sess.sim.network["B"].rt_up == "@logic ? $shared : 0"
    assert sess.sim.param["$shared"] == 2
    assert result["model_revision"] != old
    with pytest.raises(ValueError, match="stale"):
        edit_rates(sess, [RateEdit(node="A", up=1)], old)


def test_ambiguous_batch_is_atomic():
    sess = session()
    sess.sim.network["B"].rt_up = "A * 3"
    old = model_revision(sess.sim)
    with pytest.raises(ValueError, match="unambiguously"):
        edit_rates(sess, [RateEdit(node="A", up=1), RateEdit(node="B", up=3)], old)
    assert model_revision(sess.sim) == old


def test_readout_core_unchanged_and_provenance_roundtrip(tmp_path):
    sess = session()
    core = {n: str(node) for n, node in sess.sim.network.items()}
    edit_readouts(sess, [request()], model_revision(sess.sim))
    assert all(str(sess.sim.network[n]) == rule for n, rule in core.items())
    assert "Readout" not in sess.sim.network.get_output()
    old = model_revision(sess.sim, sess.readouts)
    edit_readouts(sess, [ReadoutEdit(operation="set_rule", node="Readout", rule="A | B", explanation="Revised fixture")], old)
    bnd, cfg = tmp_path / "model.bnd", tmp_path / "model.cfg"
    with bnd.open("w") as out:
        print_bnd_with_readouts(sess, out)
    with cfg.open("w") as out:
        sess.sim.print_cfg(out=out)
    loaded = maboss.load(str(bnd), str(cfg))
    assert read_readouts(bnd, loaded) == sess.readouts
    assert inspect_nodes(loaded, ["Readout"])[0]["rate_expressions"] == inspect_nodes(sess.sim, ["Readout"])[0]["rate_expressions"]


@pytest.mark.parametrize("rule", ["Readout", "Unknown", "A &", "A B", "A; B", "(A"])
def test_invalid_readout_has_no_partial_mutation(rule):
    sess = session()
    old = model_revision(sess.sim)
    edit = request().model_copy(update={"rule": rule})
    with pytest.raises(ValueError):
        edit_readouts(sess, [edit], old)
    assert model_revision(sess.sim) == old
    assert not sess.readouts


@pytest.mark.parametrize("name", ["true", "Node", "and", "_invalid"])
def test_reserved_readout_names_rejected(name):
    with pytest.raises(ValueError):
        ReadoutEdit(operation="add", node=name, rule="A", explanation="Explicit fixture", rate_up=1, rate_down=1, initial_probability_on=0)


def test_handoff_readout_rules_must_match_and_cannot_feed_core(tmp_path):
    from mcp_biomodelling_servers.boolean_readouts import verify_bnd_readouts
    definitions = {"R": {"origin": "readout", "rule": "A", "explanation": "Fixture"}}
    path = tmp_path / "model.bnd"
    path.write_text("Node A { logic = 1; }\nNode R { logic = A; }")
    verify_bnd_readouts(path, definitions, ["A", "R"])
    path.write_text("Node A { logic = R; }\nNode R { logic = A; }")
    with pytest.raises(ValueError, match="feedback"):
        verify_bnd_readouts(path, definitions, ["A", "R"])
    path.write_text("Node A { logic = 1; }\nNode R { logic = !A; }")
    with pytest.raises(ValueError, match="differs"):
        verify_bnd_readouts(path, definitions, ["A", "R"])


def test_constant_rate_edit_remains_unconditional():
    sess = session()
    sess.sim.network["A"].rt_up = "$shared"
    old = model_revision(sess.sim)
    edit_rates(sess, [RateEdit(node="A", up=3)], old)
    assert sess.sim.network["A"].rt_up == "3.0"
    assert sess.sim.param["$shared"] == 2
    assert sess.sim.network["B"].rt_up == "@logic ? $shared : 0"
