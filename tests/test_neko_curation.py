"""MCP boundary tests for bounded, authorized-scope interaction requests."""

import copy
import json

import pytest

from tests.test_neko_mcp_errors import (
    _call_tool,
    _clear_sessions,
    _create_session,
    _HistoryNetwork,
    _list_tools,
    _network_stub,
    _read_resource,
    _run,
    session_manager,
)


@pytest.fixture(autouse=True)
def sessions():
    _clear_sessions()
    yield
    _clear_sessions()


def edit():
    return {"operation": "add", "source": "A", "target": "B", "effect": "inhibition",
            "rationale": "Approved fixture correction", "evidence": {
                "summary": "Fixture publication", "pmids": [], "citations": ["Fixture citation"],
            }}


def request(network):
    sid = _create_session(network)
    return {"session_id": sid, "expected_network_id": session_manager.get_session(sid).network_id,
            "expected_state_id": network.current_state_id, "edits": [edit()]}


@pytest.mark.parametrize("change", [True, False])
def test_actual_library_result_and_cache_invalidation(change):
    calls = []

    def mutate(edits, expected_state_id):
        calls.append((edits, expected_state_id))
        return {"changed": change, "previous_state_id": 3, "state_id": 4 if change else 3,
                "changes": [{"operation": "add", "source": "canonical:A", "target": "canonical:B",
                             "before": None, "after": {"Effect": "inhibition", "Provenance": {
                                 "origin": "manual", "edits": [{"rationale": "x" * 100000}],
                             }}}] if change else []}

    network = _network_stub(current_state_id=3, edit_interactions=mutate)
    args = request(network)
    sess = session_manager.get_session(args["session_id"])
    sess._edges_cache_dirty = False
    result = _run(_call_tool("edit_interactions", args))
    assert not result.is_error
    assert calls == [(args["edits"], 3)]
    assert sess._edges_cache_dirty == change
    data = result.structured_content
    assert data["changed"] == change
    assert data["unchanged_count"] == int(not change)
    assert len(json.dumps(data)) < 2000
    assert args["expected_network_id"] in data["details_uri"]
    if change:
        assert data["changes"][0]["source"] == "canonical:A"
        assert data["changes"][0]["edit_origin"] == "manual"


@pytest.mark.parametrize("alteration", [
    {"expected_network_id": "replaced"}, {"expected_state_id": 2},
    {"expected_state_id": True}, {"edits": []}, {"edits": [edit()] * 21},
])
def test_invalid_or_stale_request_never_calls_library(alteration):
    def forbidden(*args, **kwargs):
        pytest.fail("Must validate before mutation")
    args = request(_network_stub(current_state_id=3, edit_interactions=forbidden))
    args.update(alteration)
    assert _run(_call_tool("edit_interactions", args)).is_error


@pytest.mark.parametrize("field,value", [
    ("operation", "remove"), ("effect", "+"), ("rationale", " "),
    ("evidence", {"summary": "no citation"}),
    ("evidence", {"summary": "bad PMID", "pmids": ["PMID:12"]}),
    ("evidence", {"summary": "bad PMID", "pmids": [12]}),
    ("unknown", "not allowed"),
])
def test_operation_schema_rejects_invalid_evidence_and_extra_fields(field, value):
    args = request(_network_stub(current_state_id=3))
    args["edits"][0][field] = value
    result = _run(_call_tool("edit_interactions", args))
    assert result.is_error
    assert "validation" in result.content[0].text.lower()


def test_library_refusal_preserves_code_and_does_not_invalidate_cache():
    def refuse(*args, **kwargs):
        error = ValueError("No existing endpoint")
        error.code = "endpoint_not_found"
        error.edit_index = 0
        raise error
    args = request(_network_stub(current_state_id=3, edit_interactions=refuse))
    sess = session_manager.get_session(args["session_id"])
    sess._edges_cache_dirty = False
    result = _run(_call_tool("edit_interactions", args))
    assert result.is_error
    assert "endpoint_not_found (edit 0)" in result.content[0].text
    assert sess._edges_cache_dirty is False


def test_replacing_network_rejects_same_numbered_state():
    network = _network_stub(current_state_id=3)
    args = request(network)
    session_manager.get_session(args["session_id"]).set_network(network)
    assert "stale_network" in _run(_call_tool("edit_interactions", args)).content[0].text


def test_history_page_is_bounded_but_full_state_is_available():
    network = _HistoryNetwork()
    network._metadata[4]["explanation"] = "evidence" * 10000
    network.get_state = lambda state: {"id": state, "metadata": copy.deepcopy(network._metadata[state])}
    sid = _create_session(network)
    result = _run(_call_tool("list_network_history", {"session_id": sid, "offset": 2, "limit": 1}))
    data = result.structured_content
    assert data["state_count"] == 4 and len(data["states"]) == 1
    assert data["next_offset"] == 3
    assert data["states"][0]["metadata"]["details_omitted"]
    assert len(json.dumps(data)) < 2000
    resource = _run(_read_resource(data["states"][0]["details_uri"]))
    assert json.loads(resource.contents[0].text)["state"]["metadata"]["explanation"] == "evidence" * 10000
    assert network.current_state_id == 4


def test_editor_schema_is_typed_and_bounded():
    tools = {t.name: t for t in _run(_list_tools()).tools}
    schema = tools["edit_interactions"].input_schema
    assert set(schema["required"]) == {"edits", "expected_network_id", "expected_state_id", "session_id"}
    assert schema["properties"]["edits"]["maxItems"] == 20
    assert tools["edit_interactions"].annotations.destructive_hint


def test_exact_comparison_reports_truncation_even_when_legacy_summary_is_empty():
    network = _HistoryNetwork()
    edges = [{"source": "A", "target": str(i), "Effect": "inhibition",
              "References": "x" * 5000, "Provenance": None} for i in range(3)]
    network.compare_states = lambda a, b: {
        "edge_changes": {"added": edges, "removed": [], "changed": []},
        "node_changes": {"added": [{"Uniprot": str(i)} for i in range(3)]},
    }
    network.get_state = lambda state: {"edges": {"columns": list(edges[0])}}
    result = _run(_call_tool("compare_network_states", {
        "session_id": _create_session(network), "state_a": 0, "state_b": 2, "limit": 1,
    }))
    assert not result.is_error
    data = result.structured_content
    assert data["truncated"]
    assert len(data["added_edges"]) == len(data["node_changes"]["added"]) == 1
    assert data["added_edges"][0]["origin"] == "unknown"
    assert data["added_edges"][0]["details_omitted"]
