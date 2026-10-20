"""Integrity failures for versioned NeKo model/provenance bindings."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mcp_biomodelling_servers.handoff import handoff_artifact
from mcp_biomodelling_servers.neko_provenance import (
    NeKoExportBinding,
    NeKoExportEvidence,
    NeKoStateIdentity,
    parse_neko_provenance,
    verify_neko_provenance,
)


@pytest.fixture
def evidence(tmp_path):
    model = tmp_path / "fixture.bnet"
    model.write_text("A, A\n")
    artifact = handoff_artifact(model, server="NeKo", session_id="session", role="neko_bnet")
    content = NeKoExportEvidence(
        mcp=NeKoStateIdentity(session_id="session", network_id="network", state_id=4, database_name="SIGNOR"),
        export=NeKoExportBinding(artifact=artifact),
        history={"format": "neko-history", "schema_version": 1, "current_state_id": 4,
                 "database_name": "SIGNOR", "working_state_matches_current": True},
    )
    path = tmp_path / "fixture.history.json"
    path.write_text(content.model_dump_json())
    reference = handoff_artifact(path, server="NeKo", session_id="session", role="neko_history")
    return artifact, content, reference


@pytest.mark.parametrize("field,value", [("state_id", 5), ("session_id", "wrong"), ("database_name", "OmniPath")])
def test_rehashed_wrong_identity_is_rejected(evidence, field, value):
    artifact, content, reference = evidence
    data = content.model_dump()
    data["mcp"][field] = value
    Path(reference.path).write_text(json.dumps(data))
    changed = handoff_artifact(reference.path, server="NeKo", session_id="session", role="neko_history")
    with pytest.raises(ValueError):
        verify_neko_provenance(changed, artifact)


def test_rehashed_wrong_model_binding_is_rejected(evidence, tmp_path):
    artifact, _, reference = evidence
    other = tmp_path / "other.bnet"
    other.write_text("B, B\n")
    wrong = handoff_artifact(other, server="NeKo", session_id="session", role="neko_bnet")
    with pytest.raises(ValueError, match="bound"):
        verify_neko_provenance(reference, wrong)
    assert verify_neko_provenance(reference, artifact).mcp.state_id == 4


def test_snapshot_verification_does_not_read_historical_paths(evidence):
    artifact, _, reference = evidence
    raw = Path(reference.path).read_bytes()
    Path(reference.path).unlink()
    Path(artifact.path).unlink()
    assert parse_neko_provenance(raw, reference, artifact).mcp.state_id == 4
    with pytest.raises(ValueError, match="digest/size"):
        parse_neko_provenance(raw + b" ", reference, artifact)


def test_unrecorded_working_changes_cannot_claim_saved_state(evidence):
    _, content, _ = evidence
    data = content.model_dump()
    data["history"]["working_state_matches_current"] = False
    with pytest.raises(ValueError, match="saved state"):
        NeKoExportEvidence.model_validate(data)


@pytest.mark.parametrize("saved_state", [4, 5])
def test_reuse_exported_history_without_weakening_state_validation(evidence, saved_state):
    from NeKo.services.provenance import export_provenance

    artifact, content, _ = evidence
    def unexpected_traversal():
        pytest.fail("Must reuse the history already serialized by the library export")
    network = SimpleNamespace(history_dict=unexpected_traversal,
                              current_state_id=saved_state, database_name="SIGNOR")
    session = SimpleNamespace(session_id="session", network_id="network")
    if saved_state != content.mcp.state_id:
        with pytest.raises(ValueError, match="identity mismatch"):
            export_provenance(network, session, artifact, history_payload=content.history)
    else:
        reference = export_provenance(network, session, artifact, history_payload=content.history)
        assert verify_neko_provenance(reference, artifact).history == content.history


def test_sanitized_bnet_rebinds_projection_hash_filename_and_identifiers(tmp_path):
    from NeKo.services.exporting import export_sanitized_bnet
    from NeKo.src.helpers import sanitize_bnet_file

    class Exporter:
        def __init__(self, network):
            pass

        def export_bnet(self, prefix):
            path = Path(prefix + "_1.bnet")
            path.write_text("Gene.name, Gene.name\n")
            path.with_name(path.name + ".neko.json").write_text(json.dumps({
                "history": {"format": "neko-history", "schema_version": 1},
                "export": {"file": path.name,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "identifiers": [{"identifier": "ID:gene", "label": "Gene.name",
                                            "exported_identifier": "Gene.name"}]},
            }))

    output = tmp_path / "final.bnet"
    result = export_sanitized_bnet(object(), output, overwrite=False,
                                   export_factory=Exporter, sanitizer=sanitize_bnet_file)
    assert output.read_text() == "Gene_name, Gene_name\n"
    projection = result["library_projection"]
    assert projection["file"] == "final.bnet"
    assert projection["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert projection["identifiers"][0]["exported_identifier"] == "Gene_name"
    assert result["name_mapping"] == {"Gene.name": "Gene_name"}
    assert result["library_history"] == {"format": "neko-history", "schema_version": 1}
