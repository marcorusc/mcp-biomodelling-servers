"""Protocol-level tests for MaBoSS tool and resource failures."""

import asyncio
import base64
import inspect
import sys
from collections.abc import Coroutine
from pathlib import Path
from threading import Event
from types import SimpleNamespace
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


def test_clean_for_markdown_handles_missing_values_across_pandas_versions() -> None:
    with pd.option_context("future.infer_string", True):
        frame = pd.DataFrame(
            {
                "state": ["  A\nB  ", None],
                "probability": [0.75, float("nan")],
            }
        )
        cleaned = maboss_server.clean_for_markdown(frame)

    assert cleaned.to_dict(orient="records") == [
        {"state": "A B", "probability": "0.75"},
    ]


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


def test_session_discovery_tools_publish_structured_output_schemas() -> None:
    listed_tools = _run(_list_tools())
    tools = {tool.name: tool for tool in listed_tools.tools}

    for tool_name, expected_title in {
        "list_sessions": "MaBoSSSessionListResult",
        "list_artifact_sessions": "MaBoSSArtifactSessionListResult",
    }.items():
        schema = tools[tool_name].output_schema
        assert schema is not None
        assert schema["title"] == expected_title
        assert schema["required"] == ["server", "count", "sessions"]
        assert set(schema["properties"]) == {"server", "count", "sessions"}
        assert "result" not in schema["properties"]


def test_list_sessions_returns_structured_empty_result() -> None:
    result = _run(_call_tool("list_sessions"))

    assert result.is_error is False
    assert result.content[0].text == (
        "No active sessions. Call create_session() to start one."
    )
    assert result.structured_content == {
        "server": "MaBoSS",
        "count": 0,
        "sessions": [],
    }


def test_list_sessions_returns_structured_maboss_state() -> None:
    session_id = _create_simulation_session(object())

    result = _run(_call_tool("list_sessions"))

    assert result.is_error is False
    assert f"**{session_id}**" in result.content[0].text
    assert result.structured_content is not None
    structured_session = result.structured_content["sessions"][0]
    assert structured_session["session_id"] == session_id
    assert structured_session["created_at"] >= 0
    assert structured_session["last_accessed"] >= structured_session["created_at"]
    assert structured_session["is_default"] is True
    assert structured_session["has_simulation"] is True
    assert structured_session["has_result"] is False
    assert structured_session["bnd_path"] == "/model.bnd"
    assert structured_session["cfg_path"] == "/model.cfg"
    assert result.structured_content["server"] == "MaBoSS"
    assert result.structured_content["count"] == 1


def test_list_artifact_sessions_returns_structured_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        maboss_server,
        "_list_artifact_sessions_on_disk",
        lambda *_args, **_kwargs: [
            {
                "session_id": "maboss-artifact-session",
                "server": "MaBoSS",
                "label": "baseline",
                "created_at": "2026-07-30T10:20:30+00:00",
                "files": ["model.bnd", "model.cfg"],
            }
        ],
    )

    result = _run(_call_tool("list_artifact_sessions"))

    assert result.is_error is False
    assert "**maboss-artifact-session** (baseline)" in result.content[0].text
    assert result.structured_content == {
        "server": "MaBoSS",
        "count": 1,
        "sessions": [
            {
                "session_id": "maboss-artifact-session",
                "server": "MaBoSS",
                "label": "baseline",
                "created_at": "2026-07-30T10:20:30+00:00",
                "files": ["model.bnd", "model.cfg"],
            }
        ],
    }


def test_artifact_tools_publish_structured_output_schemas() -> None:
    listed_tools = _run(_list_tools())
    tools = {tool.name: tool for tool in listed_tools.tools}

    for tool_name, expected_title in {
        "bnet_to_bnd_and_cfg": "MaBoSSBnetConversionResult",
        "export_maboss_bnd_cfg": "MaBoSSModelExportResult",
        "list_generated_files": "MaBoSSArtifactFileListResult",
        "clean_generated_files": "MaBoSSArtifactCleanupResult",
    }.items():
        schema = tools[tool_name].output_schema
        assert schema is not None
        assert schema["title"] == expected_title
        assert "result" not in schema["properties"]


def test_scientific_tools_publish_structured_output_schemas() -> None:
    listed_tools = _run(_list_tools())
    tools = {tool.name: tool for tool in listed_tools.tools}

    expected_models = {
        "run_simulation": "MaBoSSSimulationRunResult",
        "get_maboss_nodes": "MaBoSSNodeListResult",
        "get_maboss_initial_state": "MaBoSSInitialStateResult",
        "get_maboss_logical_rules": "MaBoSSLogicalRulesResult",
        "get_maboss_mutations": "MaBoSSMutationListResult",
        "update_maboss_parameters": "MaBoSSParameterResult",
        "simulate_mutation": "MaBoSSMutationSimulationResult",
        "get_simulation_result": "MaBoSSSimulationResult",
        "visualize_network_trajectories": "MaBoSSTrajectoryPlotResult",
    }

    for tool_name, expected_title in expected_models.items():
        schema = tools[tool_name].output_schema
        assert schema is not None
        assert schema["title"] == expected_title
        assert "result" not in schema["properties"]


def test_maboss_inspection_and_parameter_tools_return_scientific_data() -> None:
    network = SimpleNamespace(
        keys=lambda: ["A", "B", "C"],
        get_istate=lambda: {
            "A": {0: 0.25, 1: 0.75},
            ("B", "C"): {
                (0, 0): 0.4,
                (1, 1): "$joint_probability",
            },
        },
    )
    simulation = SimpleNamespace(
        network=network,
        get_logical_rules=lambda: {"A": "B | C", "B": "!A"},
        get_mutations=lambda: {"C": "OFF"},
        param={"sample_count": 100, "max_time": 10.0},
    )
    session_id = _create_simulation_session(simulation)

    nodes_result = _run(
        _call_tool("get_maboss_nodes", {"session_id": session_id})
    )
    initial_state_result = _run(
        _call_tool("get_maboss_initial_state", {"session_id": session_id})
    )
    rules_result = _run(
        _call_tool("get_maboss_logical_rules", {"session_id": session_id})
    )
    mutations_result = _run(
        _call_tool("get_maboss_mutations", {"session_id": session_id})
    )
    parameter_result = _run(
        _call_tool("update_maboss_parameters", {"session_id": session_id})
    )
    update_result = _run(
        _call_tool(
            "update_maboss_parameters",
            {
                "session_id": session_id,
                "parameters": {"sample_count": 250},
            },
        )
    )

    assert nodes_result.structured_content == {
        "server": "MaBoSS",
        "session_id": session_id,
        "node_count": 3,
        "nodes": ["A", "B", "C"],
    }
    assert initial_state_result.structured_content is not None
    assert initial_state_result.structured_content["group_count"] == 2
    assert initial_state_result.structured_content["groups"][1] == {
        "nodes": ["B", "C"],
        "probabilities": [
            {"state": [0, 0], "probability": 0.4},
            {"state": [1, 1], "probability": "$joint_probability"},
        ],
    }
    assert rules_result.structured_content is not None
    assert rules_result.structured_content["rules"] == [
        {"node": "A", "rule": "B | C"},
        {"node": "B", "rule": "!A"},
    ]
    assert mutations_result.structured_content is not None
    assert mutations_result.structured_content["mutations"] == [
        {"node": "C", "state": "OFF"}
    ]
    assert parameter_result.structured_content is not None
    assert parameter_result.structured_content["mode"] == "inspect"
    assert parameter_result.structured_content["parameters"] == [
        {"name": "sample_count", "value": 100},
        {"name": "max_time", "value": 10.0},
    ]
    assert update_result.structured_content is not None
    assert update_result.structured_content["mode"] == "update"
    assert update_result.structured_content["updated_parameters"] == [
        "sample_count"
    ]
    assert update_result.structured_content["parameters"][0]["value"] == 250


def test_run_and_read_simulation_preserve_numeric_trajectory_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trajectory = pd.DataFrame(
        {"<nil>": [0.1], "A": [0.9]},
        index=pd.Index([10.0], name="Time"),
    )
    simulation_result = SimpleNamespace(
        get_last_states_probtraj=lambda: trajectory,
    )
    simulation = SimpleNamespace(run=lambda: simulation_result)
    session_id = _create_simulation_session(simulation)
    monkeypatch.setattr(maboss_server, "_SERVER_ROOT", tmp_path)

    run_result = _run(
        _call_tool("run_simulation", {"session_id": session_id})
    )
    read_result = _run(
        _call_tool("get_simulation_result", {"session_id": session_id})
    )

    csv_path = tmp_path / "artifacts" / session_id / "result.csv"
    assert run_result.is_error is False
    assert run_result.structured_content is not None
    assert run_result.structured_content["result_available"] is True
    assert run_result.structured_content["trajectory_row_count"] == 1
    assert run_result.structured_content["trajectory_column_count"] == 2
    assert run_result.structured_content["result_file"]["path"] == str(csv_path)
    assert csv_path.exists()
    assert read_result.structured_content is not None
    assert read_result.structured_content["has_trajectory_data"] is True
    assert read_result.structured_content["trajectory"] == {
        "columns": ["<nil>", "A"],
        "index_name": "Time",
        "index": [10.0],
        "row_count": 1,
        "column_count": 2,
        "rows": [[0.1, 0.9]],
    }


def test_mutant_simulation_returns_mutations_and_numeric_trajectory() -> None:
    trajectory = pd.DataFrame(
        {"A": [0.0], "B": [1.0]},
        index=pd.Index([20.0], name="Time"),
    )

    class MutantSimulation:
        def __init__(self) -> None:
            self.applied: list[tuple[str, str]] = []

        def mutate(self, node: str, state: str) -> None:
            self.applied.append((node, state))

        def run(self) -> object:
            return SimpleNamespace(
                get_last_states_probtraj=lambda: trajectory,
            )

    mutant = MutantSimulation()
    simulation = SimpleNamespace(copy=lambda: mutant)
    session_id = _create_simulation_session(simulation)

    result = _run(
        _call_tool(
            "simulate_mutation",
            {
                "session_id": session_id,
                "nodes": ["A", "B"],
                "state": ["OFF", "ON"],
            },
        )
    )

    assert result.is_error is False
    assert mutant.applied == [("A", "OFF"), ("B", "ON")]
    assert result.structured_content is not None
    assert result.structured_content["mutations"] == [
        {"node": "A", "state": "OFF"},
        {"node": "B", "state": "ON"},
    ]
    assert result.structured_content["has_trajectory_data"] is True
    assert result.structured_content["trajectory"]["rows"] == [[0.0, 1.0]]


def test_empty_simulation_result_remains_a_structured_success() -> None:
    empty_result = SimpleNamespace(
        get_last_states_probtraj=lambda: pd.DataFrame(columns=["A"]),
    )
    session_id = _create_result_session(empty_result)

    result = _run(
        _call_tool("get_simulation_result", {"session_id": session_id})
    )

    assert result.is_error is False
    assert result.content[0].text == (
        "_Simulation completed but returned no trajectory data._"
    )
    assert result.structured_content is not None
    assert result.structured_content["has_trajectory_data"] is False
    assert result.structured_content["trajectory"] == {
        "columns": ["A"],
        "index_name": None,
        "index": [],
        "row_count": 0,
        "column_count": 1,
        "rows": [],
    }


def test_bnet_conversion_returns_structured_artifact_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    session_id = session_manager.create_session()
    monkeypatch.setattr(maboss_server, "_SERVER_ROOT", tmp_path)

    def convert(_input: str, bnd_path: str, cfg_path: str) -> None:
        Path(bnd_path).write_text("node A\n", encoding="utf-8")
        Path(cfg_path).write_text("max_time = 10;\n", encoding="utf-8")

    monkeypatch.setattr(
        maboss_server.maboss,
        "bnet_to_bnd_and_cfg",
        convert,
    )

    result = _run(
        _call_tool(
            "bnet_to_bnd_and_cfg",
            {
                "bnet_path": "/models/network.bnet",
                "session_id": session_id,
            },
        )
    )

    artifact_dir = tmp_path / "artifacts" / session_id
    assert result.is_error is False
    assert "MaBoSS .bnd and .cfg files created successfully" in result.content[0].text
    assert result.structured_content is not None
    assert result.structured_content["server"] == "MaBoSS"
    assert result.structured_content["session_id"] == session_id
    assert result.structured_content["input_bnet_path"] == "/models/network.bnet"
    assert result.structured_content["bnd_file"] == {
        "session_id": session_id,
        "name": "output.bnd",
        "path": str(artifact_dir / "output.bnd"),
        "suffix": ".bnd",
        "media_type": "text/plain",
        "size_bytes": 7,
    }
    assert result.structured_content["cfg_file"]["name"] == "output.cfg"
    assert result.structured_content["cfg_file"]["size_bytes"] == 15


def test_maboss_export_returns_normalized_structured_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    simulation = SimpleNamespace(
        print_bnd=lambda out: out.write("node A\n"),
        print_cfg=lambda out: out.write("max_time = 10;\n"),
    )
    session_id = _create_simulation_session(simulation)
    monkeypatch.setattr(maboss_server, "_SERVER_ROOT", tmp_path)

    result = _run(
        _call_tool(
            "export_maboss_bnd_cfg",
            {
                "prefix": "run 2",
                "overwrite": True,
                "session_id": session_id,
            },
        )
    )

    assert result.is_error is False
    assert "Exported current MaBoSS model successfully" in result.content[0].text
    assert result.structured_content is not None
    assert result.structured_content["session_id"] == session_id
    assert result.structured_content["prefix"] == "run_2"
    assert result.structured_content["overwrite"] is True
    assert result.structured_content["bnd_file"]["name"] == "run_2.bnd"
    assert result.structured_content["cfg_file"]["name"] == "run_2.cfg"


def test_maboss_artifact_listing_and_cleanup_are_structured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first_id = session_manager.create_session()
    second_id = session_manager.create_session()
    monkeypatch.setattr(maboss_server, "_SERVER_ROOT", tmp_path)
    first_dir = tmp_path / "artifacts" / first_id
    second_dir = tmp_path / "artifacts" / second_id
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    (first_dir / "model.bnd").write_text("A\n", encoding="utf-8")
    (second_dir / "result.csv").write_text("state,probability\n", encoding="utf-8")

    session_result = _run(
        _call_tool("list_generated_files", {"session_id": first_id})
    )
    all_result = _run(
        _call_tool("list_generated_files", {"session_id": "all"})
    )
    cleanup_result = _run(
        _call_tool("clean_generated_files", {"session_id": first_id})
    )

    assert session_result.is_error is False
    assert session_result.structured_content is not None
    assert session_result.structured_content["scope"] == "session"
    assert session_result.structured_content["session_id"] == first_id
    assert session_result.structured_content["count"] == 1
    assert all_result.structured_content is not None
    assert all_result.structured_content["scope"] == "all"
    assert all_result.structured_content["session_id"] is None
    assert {
        file_record["session_id"]
        for file_record in all_result.structured_content["files"]
    } == {first_id, second_id}
    assert cleanup_result.structured_content == {
        "session_id": first_id,
        "removed_count": 1,
        "server": "MaBoSS",
    }
    assert not (first_dir / "model.bnd").exists()


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
    assert result.structured_content is not None
    assert result.structured_content["server"] == "MaBoSS"
    assert result.structured_content["session_id"] == session_id
    assert result.structured_content["until"] == 1.5
    assert result.structured_content["time_window"] == "bounded"
    assert result.structured_content["image_file"]["name"] == (
        "network_trajectory.png"
    )
    assert result.structured_content["image_file"]["media_type"] == "image/png"
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


def test_all_maboss_tools_publish_safety_annotations() -> None:
    listed_tools = _run(_list_tools())
    tools = {tool.name: tool for tool in listed_tools.tools}

    read_only = {
        "list_sessions",
        "list_artifact_sessions",
        "get_maboss_nodes",
        "get_maboss_initial_state",
        "get_maboss_logical_rules",
        "get_maboss_mutations",
        "simulate_mutation",
        "get_simulation_result",
        "list_generated_files",
    }
    idempotent = {
        "set_default_session",
        "bnet_to_bnd_and_cfg",
        "build_simulation",
        "change_maboss_rule",
        "update_maboss_parameters",
        "set_maboss_output_nodes",
        "set_maboss_initial_state",
        "visualize_network_trajectories",
    }
    non_idempotent = {
        "create_session",
        "run_simulation",
        "export_maboss_bnd_cfg",
    }
    destructive = {"delete_session"}
    idempotent_destructive = {"clean_generated_files"}

    assert set(tools) == (
        read_only
        | idempotent
        | non_idempotent
        | destructive
        | idempotent_destructive
    )

    for tool_name, tool in tools.items():
        annotations = tool.annotations
        assert annotations is not None
        assert annotations.open_world_hint is False
        assert annotations.read_only_hint is (tool_name in read_only)
        assert annotations.destructive_hint is (
            tool_name in destructive | idempotent_destructive
        )
        assert annotations.idempotent_hint is (
            tool_name in read_only | idempotent | idempotent_destructive
        )


def test_maboss_tool_schemas_constrain_common_inputs() -> None:
    listed_tools = _run(_list_tools())
    tools = {tool.name: tool for tool in listed_tools.tools}

    mutation_properties = tools["simulate_mutation"].input_schema["properties"]
    state_schema = mutation_properties["state"]["anyOf"]
    assert state_schema[0]["enum"] == ["ON", "OFF", "WT"]
    assert state_schema[1]["items"]["enum"] == ["ON", "OFF", "WT"]
    assert mutation_properties["nodes"]["anyOf"][1]["minItems"] == 1

    output_schema = tools["set_maboss_output_nodes"].input_schema["properties"][
        "output_nodes"
    ]
    assert output_schema["minItems"] == 1
    assert output_schema["items"]["minLength"] == 1

    parameter_schema = tools["update_maboss_parameters"].input_schema
    parameter_definition = parameter_schema["$defs"]["MaBoSSParameterUpdates"]
    assert parameter_definition["additionalProperties"] is True
    assert parameter_definition["properties"]["sample_count"]["anyOf"][0][
        "minimum"
    ] == 1
    assert parameter_definition["properties"]["max_time"]["anyOf"][0][
        "exclusiveMinimum"
    ] == 0
    assert parameter_definition["properties"]["time_tick"]["anyOf"][0][
        "exclusiveMinimum"
    ] == 0
    assert parameter_definition["properties"]["discrete_time"]["anyOf"][0][
        "enum"
    ] == [0, 1]
    assert parameter_definition["properties"]["thread_count"]["anyOf"][0][
        "minimum"
    ] == 1


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("set_default_session", {"session_id": ""}),
        ("set_default_session", {"session_id": "   "}),
        ("bnet_to_bnd_and_cfg", {"bnet_path": ""}),
        ("change_maboss_rule", {"node": "", "new_rule": "A"}),
        ("change_maboss_rule", {"node": "A", "new_rule": ""}),
        ("set_maboss_output_nodes", {"output_nodes": []}),
        ("set_maboss_initial_state", {"nodes": [], "probDict": {}}),
        ("simulate_mutation", {"nodes": []}),
        ("simulate_mutation", {"nodes": "A", "state": "INVALID"}),
    ],
)
def test_invalid_common_inputs_are_rejected_before_execution(
    tool_name: str,
    arguments: dict[str, Any],
) -> None:
    result = _run(_call_tool(tool_name, arguments))

    assert result.is_error is True
    assert "validation error" in result.content[0].text.lower()


def test_parameter_schema_allows_backend_extensions_but_runtime_checks_keys() -> None:
    simulation = SimpleNamespace(param={"sample_count": 100})
    session_id = _create_simulation_session(simulation)

    result = _run(
        _call_tool(
            "update_maboss_parameters",
            {
                "session_id": session_id,
                "parameters": {"future_parameter": 2},
            },
        )
    )

    assert result.is_error is True
    assert "Unsupported parameter(s): future_parameter" in result.content[0].text
    assert simulation.param == {"sample_count": 100}


def test_constrained_parameter_model_preserves_update_behavior() -> None:
    simulation = SimpleNamespace(
        param={
            "sample_count": 100,
            "max_time": 10.0,
            "thread_count": 1,
        }
    )
    session_id = _create_simulation_session(simulation)

    result = _run(
        _call_tool(
            "update_maboss_parameters",
            {
                "session_id": session_id,
                "parameters": {
                    "sample_count": 250,
                    "max_time": 25.0,
                    "thread_count": 4,
                },
            },
        )
    )

    assert result.is_error is False
    assert simulation.param == {
        "sample_count": 250,
        "max_time": 25.0,
        "thread_count": 4,
    }


@pytest.mark.parametrize(
    "parameters",
    [
        {"sample_count": 0},
        {"max_time": 0},
        {"time_tick": -1},
        {"discrete_time": 2},
        {"thread_count": 0},
    ],
)
def test_invalid_parameter_bounds_are_rejected_before_mutation(
    parameters: dict[str, Any],
) -> None:
    simulation = SimpleNamespace(
        param={
            "sample_count": 100,
            "max_time": 10.0,
            "time_tick": 0.1,
            "discrete_time": 0,
            "thread_count": 1,
        }
    )
    original_parameters = simulation.param.copy()
    session_id = _create_simulation_session(simulation)

    result = _run(
        _call_tool(
            "update_maboss_parameters",
            {"session_id": session_id, "parameters": parameters},
        )
    )

    assert result.is_error is True
    assert "validation error" in result.content[0].text.lower()
    assert simulation.param == original_parameters


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
