"""Protocol-level tests for MaBoSS tool and resource failures."""

import asyncio
import base64
import inspect
import sys
from collections.abc import Coroutine
from pathlib import Path
from threading import Event
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
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


class BlockingSimulation:
    """Test double that exposes deterministic simulation/resource ordering."""

    def __init__(self) -> None:
        self.run_started = Event()
        self.release_run = Event()
        self.nodes_read = Event()
        self.network = self

    def run(self) -> "BlockingSimulationResult":
        self.run_started.set()
        if not self.release_run.wait(timeout=2):
            raise RuntimeError("test did not release simulation")
        return BlockingSimulationResult()

    def keys(self) -> list[str]:
        self.nodes_read.set()
        return ["A"]


class BlockingSimulationResult:
    def get_last_states_probtraj(self) -> pd.DataFrame:
        return pd.DataFrame()


def _run(coroutine: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coroutine)


async def _call_tool(name: str, arguments: dict[str, Any] | None = None) -> Any:
    async with Client(mcp) as client:
        return await client.call_tool(name, arguments or {})


async def _list_tools() -> Any:
    async with Client(mcp) as client:
        return await client.list_tools()


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


def _create_simulation_session(simulation: object) -> str:
    session_id = session_manager.create_session()
    session = session_manager.get_session(session_id)
    assert session is not None
    session.set_simulation(simulation, "/model.bnd", "/model.cfg")
    return session_id


def _wait_for_lease_count(session_id: str, expected: int) -> bool:
    session = session_manager.get_session(session_id)
    assert session is not None
    with session_manager._condition:
        return session_manager._condition.wait_for(
            lambda: session._lease_count == expected,
            timeout=2,
        )


@pytest.fixture(autouse=True)
def isolated_sessions() -> None:
    _clear_sessions()
    yield
    _clear_sessions()


def test_create_session_is_successful() -> None:
    result = _run(_call_tool("create_session"))

    assert result.is_error is False


def test_unknown_default_session_is_tool_error() -> None:
    result = _run(_call_tool("set_default_session", {"session_id": "missing-session"}))

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
    error = _run(_read_resource_error("maboss://session/missing-session/parameters"))

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
    artifact_path = tmp_path / "artifacts" / session_id / "network_trajectory.png"
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
    assert not (tmp_path / "artifacts" / session_id / "network_trajectory.png").exists()


def test_pymaboss_plot_contract_supports_until_and_axes() -> None:
    parameters = inspect.signature(BaseResult.plot_trajectory).parameters

    assert "until" in parameters
    assert "axes" in parameters


@pytest.mark.parametrize(
    "handler_name",
    [
        "bnet_to_bnd_and_cfg",
        "build_simulation",
        "export_maboss_bnd_cfg",
        "change_maboss_rule",
        "update_maboss_parameters",
        "set_maboss_output_nodes",
        "set_maboss_initial_state",
        "visualize_network_trajectories",
        "clean_generated_files",
    ],
)
def test_blocking_handlers_are_synchronous(handler_name: str) -> None:
    handler = getattr(maboss_server, handler_name)

    assert inspect.iscoroutinefunction(handler) is False


@pytest.mark.parametrize(
    "handler_name",
    ["run_simulation", "simulate_mutation"],
)
def test_progress_handlers_remain_asynchronous(handler_name: str) -> None:
    handler = getattr(maboss_server, handler_name)

    assert inspect.iscoroutinefunction(handler) is True


def test_session_locking_preserves_public_tool_schemas() -> None:
    listed_tools = _run(_list_tools())
    tools = {tool.name: tool for tool in listed_tools.tools}
    session_backed_tools = {
        "bnet_to_bnd_and_cfg",
        "build_simulation",
        "run_simulation",
        "export_maboss_bnd_cfg",
        "change_maboss_rule",
        "update_maboss_parameters",
        "set_maboss_output_nodes",
        "set_maboss_initial_state",
        "simulate_mutation",
        "visualize_network_trajectories",
        "get_simulation_result",
        "clean_generated_files",
    }

    for tool_name in session_backed_tools:
        properties = tools[tool_name].input_schema["properties"]
        assert "ctx" not in properties
        assert "session_id" in properties

    trajectory_properties = tools[
        "visualize_network_trajectories"
    ].input_schema["properties"]
    assert "until" in trajectory_properties


def test_resource_waits_for_same_session_simulation() -> None:
    simulation = BlockingSimulation()
    session_id = _create_simulation_session(simulation)

    async def run_concurrently() -> tuple[Any, Any]:
        async with (
            Client(mcp) as run_client,
            Client(mcp) as resource_client,
        ):
            run_task = asyncio.create_task(
                run_client.call_tool(
                    "run_simulation",
                    {"session_id": session_id},
                )
            )
            assert await asyncio.to_thread(
                simulation.run_started.wait,
                2,
            )

            resource_task = asyncio.create_task(
                resource_client.read_resource(f"maboss://session/{session_id}/nodes")
            )
            assert await asyncio.to_thread(
                _wait_for_lease_count,
                session_id,
                2,
            )
            assert simulation.nodes_read.is_set() is False

            simulation.release_run.set()
            return await asyncio.gather(run_task, resource_task)

    tool_result, _resource_result = _run(run_concurrently())

    assert tool_result.is_error is False
    assert simulation.nodes_read.is_set() is True
