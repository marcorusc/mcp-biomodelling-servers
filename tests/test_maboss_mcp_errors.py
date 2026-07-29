"""Protocol-level tests for MaBoSS tool and resource failures."""

import asyncio
import sys
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import pytest
from mcp import Client, MCPError

MABOSS_DIR = Path(__file__).parent.parent / "MaBoSS"
sys.path.insert(0, str(MABOSS_DIR))

# These imports intentionally follow the launcher-compatible sys.path setup.
from session_manager import session_manager  # noqa: E402

from MaBoSS.server import mcp  # noqa: E402


def _run(coroutine: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coroutine)


async def _call_tool(name: str, arguments: dict[str, Any] | None = None) -> Any:
    async with Client(mcp) as client:
        return await client.call_tool(name, arguments or {})


async def _read_resource_error(uri: str) -> MCPError:
    async with Client(mcp) as client:
        try:
            await client.read_resource(uri)
        except MCPError as error:
            return error
    raise AssertionError("Expected the resource read to raise MCPError.")


def _clear_sessions() -> None:
    for session_id in list(session_manager.list_sessions()):
        session_manager.delete_session(session_id)


@pytest.fixture(autouse=True)
def isolated_sessions() -> None:
    _clear_sessions()
    yield
    _clear_sessions()


def test_create_session_is_successful() -> None:
    result = _run(_call_tool("create_session"))

    assert result.is_error is False


def test_unknown_default_session_is_tool_error() -> None:
    result = _run(
        _call_tool("set_default_session", {"session_id": "missing-session"})
    )

    assert result.is_error is True
    assert "Session not found: missing-session" in result.content[0].text


def test_run_without_simulation_is_tool_error() -> None:
    result = _run(_call_tool("run_simulation"))

    assert result.is_error is True
    assert "No MaBoSS simulation has been built yet" in result.content[0].text


def test_visualize_without_result_is_tool_error() -> None:
    result = _run(_call_tool("visualize_network_trajectories"))

    assert result.is_error is True
    assert "No simulation has been run yet" in result.content[0].text


def test_missing_simulation_resource_is_protocol_error() -> None:
    error = _run(
        _read_resource_error("maboss://session/missing-session/parameters")
    )

    assert "No simulation loaded" in str(error)
