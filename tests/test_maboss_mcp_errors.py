"""Protocol-level tests for MaBoSS tool and resource failures."""

import asyncio
import base64
import inspect
import sys
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest
from maboss.results.baseresult import BaseResult
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mcp import Client, MCPError
from mcp.types import ImageContent, TextContent

MABOSS_DIR = Path(__file__).parent.parent / "MaBoSS"
sys.path.insert(0, str(MABOSS_DIR))

# These imports intentionally follow the launcher-compatible sys.path setup.
from session_manager import session_manager  # noqa: E402

from MaBoSS import server as maboss_server  # noqa: E402

mcp = maboss_server.mcp


class FakeTrajectoryResult:
    """Minimal test double matching pyMaBoSS's plotting contract."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.until: float | None = None

    def plot_trajectory(
        self,
        legend: bool = True,
        until: float | None = None,
        error: bool = False,
        prob_cutoff: float = 0.01,
        axes: Axes | None = None,
    ) -> None:
        del error, prob_cutoff
        self.until = until
        if axes is None:
            raise AssertionError("The server must provide explicit plot axes.")
        if self.fail:
            raise RuntimeError("plot failed")

        axes.plot(
            [0.0, 1.0, 2.0],
            [0.2, 0.7, 0.9],
            label="A deliberately long trajectory state label",
        )
        if legend:
            axes.legend(loc=(1.1, 0))


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


def _create_result_session(result: FakeTrajectoryResult) -> str:
    session_id = session_manager.create_session()
    session = session_manager.get_session(session_id)
    assert session is not None
    session.set_result(result)
    return session_id


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


def test_visualize_returns_uncropped_png_and_persists_same_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_result = FakeTrajectoryResult()
    session_id = _create_result_session(fake_result)
    monkeypatch.setattr(maboss_server, "_SERVER_ROOT", tmp_path)

    save_calls: list[dict[str, Any]] = []
    original_savefig = Figure.savefig

    def recording_savefig(
        figure: Figure,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        save_calls.append(kwargs.copy())
        original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", recording_savefig)
    figures_before = set(plt.get_fignums())

    result = _run(
        _call_tool(
            "visualize_network_trajectories",
            {"session_id": session_id, "until": 1.5},
        )
    )

    assert result.is_error is False
    assert result.structured_content is None
    assert len(result.content) == 2
    assert isinstance(result.content[0], TextContent)
    assert "simulation time <= 1.5" in result.content[0].text
    assert isinstance(result.content[1], ImageContent)
    assert result.content[1].mime_type == "image/png"

    returned_png = base64.b64decode(result.content[1].data)
    assert returned_png.startswith(b"\x89PNG\r\n\x1a\n")
    artifact_path = (
        tmp_path / "artifacts" / session_id / "network_trajectory.png"
    )
    assert artifact_path.read_bytes() == returned_png
    assert fake_result.until == 1.5
    assert save_calls == [
        {
            "format": "png",
            "dpi": 150,
            "bbox_inches": "tight",
            "pad_inches": 0.2,
        }
    ]
    assert set(plt.get_fignums()) == figures_before


def test_visualize_rejects_non_positive_until() -> None:
    fake_result = FakeTrajectoryResult()
    session_id = _create_result_session(fake_result)

    result = _run(
        _call_tool(
            "visualize_network_trajectories",
            {"session_id": session_id, "until": -1},
        )
    )

    assert result.is_error is True
    assert "greater than 0" in result.content[0].text
    assert fake_result.until is None


def test_visualize_closes_figure_when_plotting_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_result = FakeTrajectoryResult(fail=True)
    session_id = _create_result_session(fake_result)
    monkeypatch.setattr(maboss_server, "_SERVER_ROOT", tmp_path)
    figures_before = set(plt.get_fignums())

    result = _run(
        _call_tool(
            "visualize_network_trajectories",
            {"session_id": session_id},
        )
    )

    assert result.is_error is True
    assert "plot failed" in result.content[0].text
    assert set(plt.get_fignums()) == figures_before
    assert not (
        tmp_path / "artifacts" / session_id / "network_trajectory.png"
    ).exists()


def test_pymaboss_plot_contract_supports_until_and_axes() -> None:
    parameters = inspect.signature(BaseResult.plot_trajectory).parameters

    assert "until" in parameters
    assert "axes" in parameters
