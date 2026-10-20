"""Regression checks for concise diagnostics; no network or engine execution."""
import pytest

from tests.test_neko_mcp_errors import (
    _call_tool, _clear_sessions, _create_session, _network_stub, _run,
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
