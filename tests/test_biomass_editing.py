"""End-to-end conversational construction and lossless document edits."""

import asyncio
import json
import re
import zipfile

import numpy as np
import pandas as pd
import pytest
from mcp import Client

from BioMASS import server
from BioMASS.contracts import (
    DocumentLineEdit,
    ModelConfiguration,
    Quantity,
    ReactionEdit,
    ReactionRecord,
    SimulationScenario,
)
from BioMASS.services import artifacts
from BioMASS.services.models import render
from BioMASS.services.templates import TEMPLATES, template_reference, template_statement
from BioMASS.session_manager import BioMASSSessionManager
from BioMASS.tools.guidance import DOCS_DIRECTORY
from tests.test_biomass_workflow import handoff


@pytest.fixture
def manager(tmp_path, monkeypatch):
    manager = BioMASSSessionManager(tmp_path / "sessions")
    monkeypatch.setattr(server, "session_manager", manager)
    return manager


@pytest.mark.parametrize("name", TEMPLATES)
def test_templates_convert(name, tmp_path):
    template = TEMPLATES[name]
    edit = ReactionEdit(
        action="add",
        reaction_id="test",
        template=name,
        participants={role: "S_" + role for role in template.required},
        reversible=template.reversible,
    )
    text = template_statement(edit, {}) + "\n"
    _, summary = artifacts.job(
        tmp_path, {"operation": "generate", "text": text, "configuration": {}}, 60
    )
    expected = {p + "1" for p in template.parameters}
    if template.reversible:
        expected.add("kr1")
    assert set(summary["parameters"]) == expected


def build(sid, edits, **kwargs):
    return server.build_reactions(
        server.inspect_model(sid).document.version, edits, session_id=sid, **kwargs
    )


def test_empty_model_preview_apply_and_stale_edits(manager):
    sid = server.create_session().session_id
    edit = ReactionEdit(
        action="add",
        reaction_id="binding",
        template="binding",
        participants={"left": "E", "right": "S", "complex": "ES"},
        reversible=True,
        parameters={"kf": Quantity(value=0.003), "kr": Quantity(value=0.001)},
        initials={
            "E": Quantity(value=100),
            "S": Quantity(value=50),
            "ES": Quantity(value=0),
        },
    )
    cfg = ModelConfiguration(observables={"Enzyme": "u[E] + u[ES]"}, time_span=(0, 5))
    preview = build(sid, [edit], configuration=cfg)
    assert preview.can_apply and not preview.applied, preview.issues
    assert preview.changes[0].parameters == ["kf1", "kr1"]
    assert server.inspect_model(sid).document.mode == "empty"
    applied = build(sid, [edit], configuration=cfg, preview=False)
    assert applied.applied, applied.issues
    assert server.inspect_model(sid).document.reactions[0].status == "unreviewed"
    with pytest.raises(ValueError, match="Stale"):
        server.build_reactions(0, [edit], preview=False, session_id=sid)
    generated = server.generate_model(session_id=sid)
    assert generated.details["parameters"]["kf1"]["value"] == 0.003
    run = server.run_simulation(SimulationScenario(name="binding"), session_id=sid)
    values = pd.read_csv(next(f.path for f in run.files if f.name == "species.csv"))
    assert np.isfinite(values[["E", "S", "ES"]]).all().all()
    np.testing.assert_allclose(values.E + values.ES, 100, rtol=1e-7)
    before = server.inspect_model(sid).document
    bad = build(
        sid,
        [
            ReactionEdit(
                action="add", reaction_id="bad", statement="@rxn E --> S: open('bad')"
            )
        ],
        preview=False,
    )
    assert not bad.applied and bad.issues
    assert server.inspect_model(sid).document == before
    valid = build(
        sid,
        [
            ReactionEdit(
                action="add",
                reaction_id="decay",
                template="degradation",
                participants={"substrate": "S"},
            )
        ],
        preview=False,
    )
    assert valid.applied, valid.issues
    state = server.inspect_model(sid).document
    assert state.current_revision is None and generated.revision in state.revisions


def test_record_slots_preserve_sharing_and_configuration(manager):
    sid = server.create_session().session_id
    server.set_reactions(
        [
            ReactionRecord(
                reaction_id="unused", statement="X --> Y | kf=0.3 | X=1, Y=0"
            ),
            ReactionRecord(
                reaction_id="source", statement="A --> B | kf=0.1 | A=1, B=0"
            ),
            ReactionRecord(
                reaction_id="shared",
                statement="B --> C | | C=0",
                share_parameters_with="source",
            ),
        ],
        sid,
    )
    cfg = ModelConfiguration(
        parameters={"kf2": Quantity(value=0.7)},
        observables={"total": "u[A] + u[B] + u[C]"},
        time_span=(0, 5),
    )
    server.configure_model(cfg, sid)
    result = build(
        sid, [ReactionEdit(action="remove", reaction_id="unused")], preview=False
    )
    assert result.applied, result.issues
    text, mapping = render(server.inspect_model(sid).document)
    assert text.startswith("# Removed reaction\n")
    assert mapping == {"source": 2, "shared": 3}
    assert text.splitlines()[2].split("|")[1].strip() == "2"
    generated = server.generate_model(session_id=sid)
    assert generated.details["parameters"]["kf2"]["value"] == 0.7
    assert generated.details["parameters"]["kf3"]["value"] == 0.7
    old = server.inspect_model(sid).document
    rejected = build(
        sid, [ReactionEdit(action="remove", reaction_id="source")], preview=False
    )
    assert not rejected.applied and rejected.issues
    assert server.inspect_model(sid).document == old


def test_file_inventory_editing_and_dependency_repairs(manager, tmp_path):
    source = tmp_path / "existing.txt"
    original = b"# Keep this comment\r\nA --> B | kf=0.1 | A=1, B=0\r\nB --> C | 2 | C=0\r\n@obs Total: u[A] + u[B] + u[C]\r\n@sim tspan: [0, 5]"
    source.write_bytes(original)
    sid = server.create_session().session_id
    loaded = server.import_text_file(str(source), session_id=sid)
    assert loaded.document.text.encode() == original
    assert loaded.document.source_file["path"] == str(source)
    inventory = server.inspect_reactions(session_id=sid)
    assert inventory.generation_valid is True, inventory.issues
    assert [r.reaction_id for r in inventory.reactions] == ["line_2", "line_3"]
    assert inventory.reactions[1].parameters == ["kf3"]
    before = loaded.document
    failure = build(
        sid, [ReactionEdit(action="remove", reaction_id="line_3")], preview=False
    )
    assert not failure.can_apply and any("u[C]" in e for e in failure.issues)
    assert server.inspect_model(sid).document == before
    repaired = build(
        sid,
        [ReactionEdit(action="remove", reaction_id="line_3")],
        line_edits=[DocumentLineEdit(line_number=4, text="@obs Total: u[A] + u[B]")],
        preview=False,
    )
    assert repaired.applied, repaired.issues
    text = server.inspect_model(sid).document.text
    assert (
        text.splitlines(keepends=True)[:2]
        == original.decode().splitlines(keepends=True)[:2]
    )
    assert text.endswith("@sim tspan: [0, 5]")
    extra = build(
        sid,
        [
            ReactionEdit(
                action="add",
                reaction_id="decay",
                template="degradation",
                participants={"substrate": "B"},
                parameters={"kf": Quantity(value=0.02)},
            )
        ],
        preview=False,
    )
    assert extra.applied and extra.changes[0].line_number == 6, extra.issues
    assert source.read_bytes() == original
    generated = server.generate_model(session_id=sid)
    assert generated.details["line_mapping"] == {"line_2": 2, "decay": 6}
    bundle = server.export_model_bundle(session_id=sid)
    with zipfile.ZipFile(bundle.details["bundle"]) as archive:
        exported = json.loads(archive.read("model/snapshot.json"))["document"]
        assert exported["text"] == server.inspect_model(sid).document.text
        assert exported["source_file"] == loaded.document.source_file


def test_network_mapping_and_optional_metadata(manager, tmp_path):
    sid = server.create_session().session_id
    export = handoff(tmp_path)
    imported = server.import_neko_handoff(export.manifest_file.path, sid)
    edge = next(e for e in imported.document.network.edges if e.source == "E")
    edit = ReactionEdit.model_validate(
        {
            "action": "add",
            "reaction_id": "edge_binding",
            "template": "binding",
            "participants": {"left": "E", "right": "S", "complex": "ES"},
            "metadata": {"edge_ids": [edge.edge_id]},
        }
    )
    result = build(sid, [edit], preview=False)
    assert result.applied, result.issues
    assert result.changes[0].statement == "Enzyme binds Substrate --> ES"
    state = server.inspect_model(sid)
    assert state.coverage.unreviewed_edges == [edge.edge_id]
    assert edge.edge_id in state.coverage.unresolved_edges
    assert state.document.species_mapping["E"] == ["Enzyme"]
    extended = build(
        sid, [], species_mapping={"E": ["Enzyme", "Enzyme_active"]}, preview=False
    )
    assert extended.applied, extended.issues


def test_mapping_collision_and_ambiguous_states_are_rejected(manager, tmp_path):
    sid = server.create_session().session_id
    export = handoff(tmp_path)
    server.import_neko_handoff(export.manifest_file.path, sid)
    edit = ReactionEdit(
        action="add",
        reaction_id="bind",
        template="binding",
        participants={"left": "E", "right": "S", "complex": "ES"},
    )
    collision = build(sid, [edit], species_mapping={"E": ["A"], "S": ["A"]})
    assert not collision.can_apply and "collision" in collision.issues[0]
    ambiguity = build(sid, [edit], species_mapping={"E": ["E_inactive", "E_active"]})
    assert not ambiguity.can_apply and "multiple states" in ambiguity.issues[0]


def test_custom_parameters_prune_and_conditions(manager):
    sid = server.create_session().session_id
    edit = ReactionEdit(
        action="add",
        reaction_id="source",
        statement="@rxn A --> B: p[rate] * u[A]",
        parameters={"rate": Quantity(value=0.1)},
        initials={"A": Quantity(value=1), "B": Quantity(value=0)},
    )
    assert build(sid, [edit], preview=False).applied
    update = ReactionEdit(
        action="update",
        reaction_id="source",
        statement="@rxn A --> B: p[new_rate] * u[A]",
        parameters={"new_rate": Quantity(value=0.3)},
    )
    failed = build(sid, [update], preview=False)
    assert not failed.applied and any("rate" in e for e in failed.issues)
    result = build(sid, [update], prune_unused_values=True, preview=False)
    assert result.applied, result.issues
    doc = server.inspect_model(sid).document
    assert list(doc.configuration.parameters) == ["new_rate"]
    assert doc.configuration.initials["A"].value == 1


def test_file_limits_and_mcp_edit_contract(manager, tmp_path):
    oversized = tmp_path / "large.txt"
    oversized.write_bytes(b"A" * (1024 * 1024 + 1))
    sid = server.create_session().session_id
    with pytest.raises(ValueError, match="1 MiB"):
        server.import_text_file(str(oversized), session_id=sid)

    async def check():
        async with Client(server.mcp) as client:
            result = await client.call_tool(
                "build_reactions",
                {
                    "session_id": sid,
                    "expected_version": 0,
                    "preview": False,
                    "edits": [
                        {
                            "action": "add",
                            "reaction_id": "decay",
                            "template": "degradation",
                            "participants": {"substrate": "A"},
                        }
                    ],
                },
            )
            assert not result.is_error, result.content
            assert result.structured_content["applied"]
            stale = await client.call_tool(
                "build_reactions",
                {"session_id": sid, "expected_version": 0, "edits": []},
            )
            assert stale.is_error and "Stale" in stale.content[0].text

    asyncio.run(check())


def test_editing_documentation_matches_templates_and_executes(manager):
    text = (DOCS_DIRECTORY / "model_editing.md").read_text()
    assert template_reference() in text
    examples = [json.loads(s) for s in re.findall(r"```json\n(.*?)```", text, re.S)]
    sid = server.create_session().session_id
    first, custom = examples
    first["session_id"] = sid
    first["edits"] = [ReactionEdit.model_validate(e) for e in first["edits"]]
    first["configuration"] = ModelConfiguration.model_validate(first["configuration"])
    first["preview"] = False
    result = server.build_reactions(**first)
    assert result.applied, result.issues
    second = build(sid, [ReactionEdit.model_validate(custom)], preview=False)
    assert second.applied, second.issues


def test_edit_timeout_is_atomic_and_cleans_worker(manager, monkeypatch, tmp_path):
    worker = tmp_path / "slow.py"
    worker.write_text("import time\nprint('started', flush=True)\ntime.sleep(10)\n")
    monkeypatch.setattr(artifacts, "WORKER", worker)
    sid = server.create_session().session_id
    before = server.inspect_model(sid).document
    result = build(
        sid,
        [ReactionEdit(action="add", reaction_id="slow", statement="A --> B")],
        timeout_seconds=1,
        preview=False,
    )
    assert not result.applied and "terminated" in result.issues[0]
    assert server.inspect_model(sid).document == before
    directory = manager.directory(sid)
    assert not list(directory.glob(".pending*"))
    assert not any(p.is_dir() for p in directory.glob("rev_*"))
    assert list(directory.glob("*.failed.log"))
