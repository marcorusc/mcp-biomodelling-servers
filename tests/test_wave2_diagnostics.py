"""Regression checks for concise diagnostics; no network or engine execution."""
import pytest

from tests.test_neko_mcp_errors import (
    _call_tool,
    _clear_sessions,
    _create_session,
    _network_stub,
    _run,
)


@pytest.mark.parametrize("printed,status,count", [
    ("Warning: No paths found between source: A and target: B", "no_path_within_bound", 0),
    ("['A', 'B']", "paths_found", 1),
])
def test_path_warnings_are_not_paths(printed, status, count):
    _clear_sessions()
    sid = _create_session(_network_stub(print_my_paths=lambda *a, **k: print(printed)))
    result = _run(_call_tool("find_paths", {"source": "A", "target": "B", "session_id": sid}))
    assert not result.is_error
    assert result.structured_content["status"] == status
    assert result.structured_content["output_line_count"] == count
    assert result.structured_content["path_output_lines"] == []
    _clear_sessions()


def test_invalid_endpoints_are_errors():
    _clear_sessions()
    sid = _create_session(_network_stub(print_my_paths=lambda *a, **k: print("Error: endpoints missing")))
    result = _run(_call_tool("find_paths", {"source": "A", "target": "B", "session_id": sid}))
    assert result.is_error
    assert "Invalid path endpoints" in result.content[0].text
    _clear_sessions()


def test_path_identity_keeps_missing_and_ambiguous_labels_explicit():
    import pandas as pd

    from NeKo.services.path_reporting import path_identities
    network = _network_stub(nodes=pd.DataFrame([
        {"Genesymbol": "A", "Uniprot": "ID:A"},
        {"Genesymbol": "same", "Uniprot": "ID:1"},
        {"Genesymbol": "same", "Uniprot": "ID:2"},
        {"Genesymbol": "label_only", "Uniprot": None},
    ]))
    paths = path_identities(network, [["ID:A", "same", "label_only"]])
    assert paths[0][0] == {"label": "A", "uniprot": "ID:A"}
    assert paths[0][1]["identity"] == "ambiguous"
    assert paths[0][2] == {"label": "label_only", "uniprot": None}


def test_lightweight_history_status_avoids_full_serialization():
    from NeKo.src.history import summarize_history
    from tests.test_neko_mcp_errors import _HistoryNetwork
    network = _HistoryNetwork()
    network.history_status = lambda: {"pruned_state_count": 2, "working_state_matches_current": False}
    network.history_dict = lambda: pytest.fail("Do not serialize all history for two status flags")
    summary = summarize_history(network, session_id="fixture", max_states=None)
    assert summary.pruned_state_count == 2
    assert summary.working_state_matches_current is False


def test_export_diagnostic_names_invalid_source_node(tmp_path):
    import pandas as pd

    from NeKo.services.exporting import export_sanitized_bnet
    network = _network_stub(nodes=pd.DataFrame([{"Uniprot": "P_BAD", "Genesymbol": None}]))
    with pytest.raises(ValueError, match="P_BAD"):
        export_sanitized_bnet(network, tmp_path / "model.bnet", overwrite=False,
                              export_factory=lambda _: pytest.fail("Must fail before export"), sanitizer=lambda _: {})
    assert not list(tmp_path.iterdir())


def test_lightweight_history_pages_at_library_boundary_and_keeps_unknown_counts():
    from types import SimpleNamespace

    from NeKo.src.history import summarize_history
    calls = []
    def page(**kwargs):
        calls.append(kwargs)
        return [{"id": 50, "parent_ids": [49], "children_ids": [51],
                 "metadata": {"transition": {"explanation_status": "unavailable",
                                             "counts": {"edges": None}}}}]
    network = SimpleNamespace(
        state_count=1000, current_state_id=999, root_state_id=0,
        list_states=page,
        history_status=lambda: {"pruned_state_count": 0, "working_state_matches_current": True},
        history_graph=lambda: pytest.fail("Routine summary must not build the full graph"),
        history_dict=lambda: pytest.fail("Routine summary must not copy full history"),
    )
    summary = summarize_history(network, session_id="fixture", max_states=None, offset=50, limit=1)
    assert calls == [{"offset": 50, "limit": 1, "detail": "summary"}]
    assert summary.state_count == 1000 and summary.next_offset == 51
    assert summary.states[0].parent_state_ids == [49]
    assert summary.states[0].metadata["edge_changes"] is None
