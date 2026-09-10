"""Real ODE, graph, evidence, and isolated-worker acceptance tests."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
import subprocess
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from mcp import Client
from pydantic import ValidationError

from BioMASS import server
from BioMASS.contracts import (
    EvidenceRecord,
    GraphOptions,
    ModelConfiguration,
    Quantity,
    ReactionRecord,
    SimulationScenario,
)
from BioMASS.services import artifacts
from BioMASS.services.authoring import render_records, validate_text
from BioMASS.session_manager import BioMASSSessionManager
from mcp_biomodelling_servers.handoff import HandoffPackage, HandoffProvenance
from mcp_biomodelling_servers.ode_handoff import read_ode_handoff, write_ode_handoff
from NeKo.services.ode_exporting import ode_network

ENZYME = (Path(__file__).parents[1] / "BioMASS/examples/enzyme.txt").read_text()


@pytest.fixture
def manager(tmp_path, monkeypatch):
    manager = BioMASSSessionManager(tmp_path)
    monkeypatch.setattr(server, "session_manager", manager)
    return manager


def network_fixture():
    return SimpleNamespace(
        nodes=pd.DataFrame(
            [
                {"Uniprot": "E", "Genesymbol": "Enzyme", "Type": "protein"},
                {"Uniprot": "S", "Genesymbol": "Substrate", "Type": "protein"},
                {"Uniprot": "X", "Genesymbol": "Unknown", "Type": "protein"},
            ]
        ),
        edges=pd.DataFrame(
            [
                {
                    "source": "E",
                    "target": "S",
                    "Effect": "stimulation",
                    "References": ["PMID:1", "10.1234/example", "unresolved"],
                    "Mechanism": "binding",
                    "Database": "SIGNOR",
                },
                {
                    "source": "X",
                    "target": "S",
                    "Effect": None,
                    "References": None,
                    "Mechanism": None,
                    "Database": "SIGNOR",
                },
            ]
        ),
    )


def handoff(tmp_path):
    graph = ode_network(network_fixture())
    source = HandoffProvenance(
        server="NeKo",
        session_id="source-session",
        mcp_package=HandoffPackage(name="mcp-biomodelling-servers", version="2.3.0"),
        modelling_package=HandoffPackage(name="nekomata", version="1.10.0"),
        operation="export_biomass_handoff",
    )
    return write_ode_handoff(
        tmp_path,
        "enzyme",
        graph,
        source,
        "Synthetic enzyme fixture; references are test identifiers.",
        0,
    )


def test_handoff_stability_references_and_integrity(tmp_path):
    export = handoff(tmp_path)
    _, network = read_ode_handoff(export.manifest_file.path)
    edge = next(e for e in network.edges if e.source == "E")
    assert edge.references == ["10.1234/example", "PMID:1", "unresolved"]
    assert edge.metadata["Mechanism"] == "binding"
    reverse = network_fixture()
    reverse.edges = reverse.edges.iloc[::-1]
    assert ode_network(reverse) == network
    with pytest.raises(FileExistsError):
        handoff(tmp_path)
    path = Path(export.manifest.network_file.path)
    path.write_text(path.read_text().replace("binding", "changed"))
    with pytest.raises(ValueError, match="digest changed"):
        read_ode_handoff(export.manifest_file.path)


def test_evidence_coverage_and_atomic_invalid_update(manager, tmp_path):
    sid = server.create_session().session_id
    export = handoff(tmp_path)
    imported = server.import_neko_handoff(export.manifest_file.path, sid)
    edge = next(e for e in imported.document.network.edges if e.source == "E")
    assert len(imported.coverage.unresolved_edges) == 2
    server.set_evidence(
        [
            EvidenceRecord(
                evidence_id="paper",
                source_identifiers=["PMID:1"],
                summary="Synthetic fixture",
                edge_ids=[edge.edge_id],
            )
        ],
        sid,
    )
    record = ReactionRecord(
        reaction_id="binding",
        statement="E + S <--> ES",
        status="supported",
        evidence_ids=["paper"],
        edge_ids=[edge.edge_id],
    )
    result = server.set_reactions([record], sid)
    assert result.coverage.supported_edges == [edge.edge_id]
    assert len(result.coverage.unresolved_edges) == 1
    prior = server.inspect_model(sid)
    with pytest.raises(ValueError, match="unknown"):
        server.set_reactions(
            [record.model_copy(update={"evidence_ids": ["missing"]})], sid
        )
    assert server.inspect_model(sid).document == prior.document
    with pytest.raises(ValidationError, match="rationale"):
        ReactionRecord(reaction_id="missing", statement="X --> S", status="assumed")
    conflict = EvidenceRecord(
        evidence_id="conflict",
        source_identifiers=["PMID:2"],
        summary="Conflicting context",
        stance="contradicts",
    )
    server.set_evidence([conflict], sid)
    assert server.inspect_model(sid).coverage.conflicting_evidence == ["conflict"]


def test_parameter_sharing_tracks_stable_reaction_ids():
    records = [
        ReactionRecord(
            reaction_id="first",
            statement="A --> B | kf=0.1",
            status="assumed",
            assumption="test",
        ),
        ReactionRecord(
            reaction_id="second",
            statement="B --> C",
            share_parameters_with="first",
            status="assumed",
            assumption="test",
        ),
    ]
    text, mapping = render_records(records, ModelConfiguration())
    assert mapping == {"first": 1, "second": 2}
    assert text.splitlines()[1] == "B --> C | 1"
    prefix = ReactionRecord(
        reaction_id="prefix", statement="X --> Y", status="assumed", assumption="test"
    )
    text, mapping = render_records([prefix, *records], ModelConfiguration())
    assert mapping["first"] == 2 and text.splitlines()[2].endswith("| 2")
    with pytest.raises(ValueError, match="earlier"):
        render_records(records[::-1], ModelConfiguration())


@pytest.mark.parametrize(
    "text",
    [
        "A --> B\n@obs unsafe: __import__('os').system('echo unsafe')",
        "A --> B\n@sim condition unsafe: init[A] = 1; import os",
        "@rxn A --> B: u[A].__class__",
        "A --> B | kf=1e999",
        "A --> B\n@sim tspan: [0, 100000000]",
        "A --> B\n@obs broken: [u[A] for x in p[B]]",
        "@add species A\n@obs a: u[A]",
        "A --> B | 1",
        "A --> B\n@obs a: u[A]\n@obs a: u[B]",
    ],
)
def test_executable_or_unbounded_text_is_rejected(text):
    with pytest.raises((ValueError, SyntaxError)):
        validate_text(text)


def test_standalone_preservation_restart_and_cleanup(manager):
    sid = server.create_session(label="retained").session_id
    text = "# exact comments and whitespace\n" + ENZYME + "\n"
    server.import_text(text, session_id=sid)
    server.close_session(sid)
    restored = server.restore_session(sid)
    assert restored.document.text == text
    directory = manager.directory(sid)
    (directory / "nested").mkdir()
    (directory / "nested/file.txt").write_text("generated")
    assert server.clean_generated_files(sid).removed_count == 1
    assert server.inspect_model(sid).document.text == text
    assert server.list_artifact_sessions().count == 1


def test_symlinks_and_revision_traversal_rejected(manager, tmp_path):
    sid = server.create_session().session_id
    directory = manager.directory(sid)
    outside = tmp_path / "outside.txt"
    outside.write_text("retain")
    (directory / "link").symlink_to(outside)
    with pytest.raises(ValueError, match="Symlink"):
        server.clean_generated_files(sid)
    assert outside.read_text() == "retain"
    with pytest.raises(ValueError, match="Unknown"):
        artifacts.revision_path(directory, "../outside", ["../outside"])


@pytest.mark.parametrize("filename", ["session.json", "session.json.tmp"])
def test_snapshot_symlinks_cannot_write_outside_artifacts(manager, tmp_path, filename):
    sid = server.create_session().session_id
    outside = tmp_path / "outside.json"
    outside.write_text("preserve")
    snapshot = manager.directory(sid) / filename
    snapshot.unlink(missing_ok=True)
    snapshot.symlink_to(outside)
    with pytest.raises(ValueError, match="Symlinked"):
        server.import_text(ENZYME, session_id=sid)
    assert outside.read_text() == "preserve"
    assert manager.sessions[sid].document.mode == "empty"


def test_protocol_errors_resources_and_guidance(manager):
    async def check():
        async with Client(server.mcp) as client:
            result = await client.call_tool("create_session", {})
            sid = result.structured_content["session_id"]
            invalid = await client.call_tool(
                "import_text", {"text": "A --> B\n@obs x: eval('1')", "session_id": sid}
            )
            assert invalid.is_error
            assert "Expressions allow only" in invalid.content[0].text
            missing = await client.call_tool(
                "generate_model", {"session_id": "missing"}
            )
            assert missing.is_error
            resource = await client.read_resource(f"biomass://session/{sid}/model")
            assert resource.contents
            manual = await client.read_resource("docs://biomass/agent_manual")
            assert "placeholder" in manual.contents[0].text
            assert (await client.get_prompt("biomass_workflow_prompt")).messages

    asyncio.run(check())


def test_session_operations_serialize_and_close_waits(manager):
    first = server.create_session().session_id
    second = server.create_session().session_id
    entered = threading.Event()
    release = threading.Event()

    def hold():
        with manager.use(first):
            entered.set()
            assert release.wait(5)

    with ThreadPoolExecutor(max_workers=3) as pool:
        held = pool.submit(hold)
        assert entered.wait(5)
        closing = pool.submit(server.close_session, first)
        assert server.inspect_model(second).session_id == second
        assert not closing.done()
        release.set()
        held.result(timeout=5)
        closing.result(timeout=5)
    with pytest.raises(ValueError):
        server.inspect_model(first)


def test_timeout_kills_worker_and_preserves_only_log(tmp_path, monkeypatch):
    sleeper = tmp_path / "sleeper.py"
    sleeper.write_text(
        "import time\nfrom pathlib import Path\nPath('partial.py').write_text('incomplete')\ntime.sleep(30)\n"
    )
    monkeypatch.setattr(artifacts, "WORKER", sleeper)
    directory = tmp_path / "jobs"
    directory.mkdir()
    with pytest.raises(TimeoutError, match="terminated"):
        artifacts.job(directory, {"operation": "generate"}, 1)
    assert not list(directory.glob(".pending*"))
    assert not any(path.is_dir() for path in directory.glob("rev_*"))
    assert len(list(directory.glob("*.failed.log"))) == 1


def test_graph_dependency_failure_is_actionable(monkeypatch):
    from BioMASS.worker import graph

    original = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None if name == "pygraphviz" else original(name),
    )
    with pytest.raises(RuntimeError, match="biomass-graph"):
        graph({"format": "png", "graph_options": {}}, None)


def test_html_uses_upstream_without_opening_browser(monkeypatch, tmp_path):
    import pyvis.network

    from BioMASS.worker import graph

    class Description:
        species = ["A", "B"]
        graph = SimpleNamespace(edges=lambda: [("A", "B")])

        def dynamic_plot(self, **kwargs):
            assert kwargs["show"] is False
            assert kwargs["show_controls"] is True

    monkeypatch.setattr(pyvis.network, "Network", lambda **kwargs: None)
    result = graph(
        {
            "format": "html",
            "graph_options": GraphOptions(show_controls=True).model_dump(),
        },
        Description(),
    )
    assert result["edges"] == [["A", "B"]]


def test_real_standalone_generation_simulation_graphs_and_export(manager, tmp_path):
    sid = server.create_session().session_id
    server.import_text(ENZYME, session_id=sid)
    generated = server.generate_model(session_id=sid)
    assert generated.details["species"] == ["E", "S", "ES", "P"]
    assert len(generated.details["reactions"]) == 2
    assert len(generated.details["parameters"]) == 3
    simulation = server.run_simulation(
        SimulationScenario(name="enzyme"), session_id=sid
    )
    csv = next(Path(f.path) for f in simulation.files if f.name == "species.csv")
    frame = pd.read_csv(csv)
    np.testing.assert_allclose(frame.E + frame.ES, 100, atol=1e-6)
    np.testing.assert_allclose(frame.S + frame.ES + frame.P, 50, atol=1e-6)
    assert frame.P.iloc[-1] > 0
    for fmt in ("png", "svg", "html"):
        response = server.visualize_model(format=fmt, session_id=sid)
        payload = response.structured_content
        assert payload["revision"] == generated.revision
        artifact = next(f for f in payload["files"] if f["name"] == "graph." + fmt)
        assert Path(artifact["path"]).stat().st_size > 100
        assert ["E", "ES"] in payload["details"]["edges"]
    graph = server.export_model_graph(session_id=sid)
    assert {tuple(e) for e in graph.details["edges"]} == {
        ("E", "ES"),
        ("S", "ES"),
        ("ES", "E"),
        ("ES", "S"),
        ("ES", "P"),
    }
    bundle = server.export_model_bundle(session_id=sid)
    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(bundle.details["bundle"]) as archive:
        assert "model/snapshot.json" in archive.namelist()
        assert any(name.endswith("graph.html") for name in archive.namelist())
        archive.extractall(unpacked)
    run = next(unpacked.glob("simulate_*"))
    completed = subprocess.run(
        [sys.executable, str(unpacked / "run_simulation.py"), run.name],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    reproduced = pd.read_csv(run / "control_reproduced.csv")
    np.testing.assert_allclose(reproduced.E, frame.E, rtol=1e-8)
    # Editing invalidates the current pointer, but prior graphs retain their revision.
    server.configure_model(
        ModelConfiguration(parameters={"kf1": Quantity(value=0.004)}), sid
    )
    assert server.inspect_model(sid).document.current_revision is None
    with pytest.raises(ValueError, match="Generate"):
        server.export_model_bundle(session_id=sid)
    assert (
        server.export_model_bundle(revision=generated.revision, session_id=sid).revision
        == generated.revision
    )
    old_text = manager.directory(sid) / generated.revision / "model.txt"
    old_text.write_text(old_text.read_text() + "# changed")
    with pytest.raises(ValueError, match="changed"):
        server.export_model_bundle(revision=generated.revision, session_id=sid)


def test_real_neko_handoff_to_ode_with_assumptions(manager, tmp_path):
    export = handoff(tmp_path)
    sid = server.create_session().session_id
    imported = server.import_neko_handoff(export.manifest_file.path, sid)
    edge = next(e for e in imported.document.network.edges if e.source == "E")
    evidence = EvidenceRecord(
        evidence_id="binding_evidence",
        source_identifiers=edge.references,
        summary="Synthetic fixture supporting binding",
        edge_ids=[edge.edge_id],
    )
    server.set_evidence([evidence], sid)
    server.set_reactions(
        [
            ReactionRecord(
                reaction_id="bind",
                statement=ENZYME.splitlines()[0],
                status="supported",
                evidence_ids=[evidence.evidence_id],
                edge_ids=[edge.edge_id],
            ),
            ReactionRecord(
                reaction_id="catalysis",
                statement=ENZYME.splitlines()[1],
                status="assumed",
                assumption="Illustrative catalytic mechanism; not an evidence claim.",
                edge_ids=[edge.edge_id],
            ),
        ],
        sid,
    )
    server.configure_model(
        ModelConfiguration(
            observables={"Enzyme_total": "u[E] + u[ES]"}, time_span=(0, 100)
        ),
        sid,
    )
    generated = server.generate_model(session_id=sid)
    assert len(generated.coverage.unresolved_edges) == 1
    assert generated.coverage.assumed_reactions == ["catalysis"]
    assert generated.details["line_mapping"] == {"bind": 1, "catalysis": 2}
    server.run_simulation(SimulationScenario(name="illustrative"), session_id=sid)
    server.visualize_model(session_id=sid)
    bundle = server.export_model_bundle(session_id=sid)
    with zipfile.ZipFile(bundle.details["bundle"]) as archive:
        snapshot = json.loads(archive.read("model/snapshot.json"))
        assert snapshot["document"]["upstream"]["handoff_type"] == "neko-to-biomass"
        assert (
            snapshot["document"]["evidence"]["binding_evidence"]["source_identifiers"]
            == edge.references
        )


def test_real_placeholders_require_opt_in_and_unknown_quantities_fail(manager):
    sid = server.create_session().session_id
    server.import_text("A --> B\n@obs B: u[B]\n@sim tspan: [0, 2]\n", session_id=sid)
    server.generate_model(session_id=sid)
    with pytest.raises(RuntimeError, match="placeholder"):
        server.run_simulation(SimulationScenario(name="unset"), session_id=sid)
    scenario = SimulationScenario(
        name="test", allow_placeholders=True, initials={"A": Quantity(value=1)}
    )
    result = server.run_simulation(scenario, session_id=sid)
    assert result.details["parameters"]["kf1"]["origin"] == "placeholder"
    assert result.details["conditions"][0]["initials"]["A"] == 1
    with pytest.raises(RuntimeError, match="Unknown numerical"):
        server.run_simulation(
            SimulationScenario(name="bad", parameters={"unknown": Quantity(value=1)}),
            session_id=sid,
        )


def test_record_cannot_claim_coverage_with_a_comment():
    record = ReactionRecord(
        reaction_id="comment",
        statement="# no mechanism",
        status="assumed",
        assumption="test",
    )
    with pytest.raises(ValueError, match="comment"):
        render_records([record], ModelConfiguration())


def test_real_parameter_sharing_survives_scenarios_and_conditions(manager):
    sid = server.create_session().session_id
    text = """A --> B | kf=0.1 | A=1, B=0
B --> C | 1 | C=0
@obs sum: u[A] + u[B] + u[C]
@obs shared_rate: p[kf2] + 0 * u[A]
@sim tspan: [0, 5]
@sim condition fast: p[kf1] = 0.2
@sim condition reset: init[A] = 2
"""
    server.import_text(text, session_id=sid)
    server.generate_model(session_id=sid)
    result = server.run_simulation(
        SimulationScenario(name="shared", parameters={"kf1": Quantity(value=0.3)}),
        session_id=sid,
    )
    fast, reset = result.details["conditions"]
    assert fast["parameters"] == {"kf1": 0.2, "kf2": 0.2}
    assert reset["parameters"] == {"kf1": 0.3, "kf2": 0.3}
    assert fast["initials"]["A"] == 1 and reset["initials"]["A"] == 2
    obs = pd.read_csv(next(f.path for f in result.files if f.name == "observables.csv"))
    np.testing.assert_allclose(obs.loc[obs.condition == "fast", "shared_rate"], 0.2)
    np.testing.assert_allclose(obs.loc[obs.condition == "reset", "shared_rate"], 0.3)


def test_worker_in_wheel_namespace_does_not_shadow_upstream_biomass(
    tmp_path, monkeypatch
):
    namespace = tmp_path / "installed" / "mcp_biomodelling_servers"
    server_source = Path(server.__file__).parent
    shutil.copytree(
        server_source,
        namespace / "BioMASS",
        ignore=shutil.ignore_patterns("artifacts", "__pycache__"),
    )
    (namespace / "__init__.py").write_text("")
    (namespace / "biomass.py").write_text(
        "raise AssertionError('entrypoint shadowed upstream')\n"
    )
    monkeypatch.setattr(artifacts, "WORKER", namespace / "BioMASS" / "worker.py")
    result = artifacts.run_worker(
        tmp_path / "job",
        {"operation": "generate", "text": ENZYME, "configuration": {}},
    )
    assert set(result["species"]) == {"E", "S", "ES", "P"}
