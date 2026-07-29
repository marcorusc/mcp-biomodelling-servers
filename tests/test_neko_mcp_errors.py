"""Protocol-level tests for NeKo tool semantics and concurrency."""

import asyncio
import inspect
import sys
import threading
from collections.abc import Coroutine
from pathlib import Path
from threading import Event
from types import ModuleType, SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from mcp import Client


def _install_neko_import_stubs() -> None:
    """Provide the unreleased NeKo import surface needed to load the server."""
    neko = ModuleType("neko")
    neko.__path__ = []  # type: ignore[attr-defined]
    core = ModuleType("neko.core")
    core.__path__ = []  # type: ignore[attr-defined]
    outputs = ModuleType("neko._outputs")
    outputs.__path__ = []  # type: ignore[attr-defined]

    network_module = ModuleType("neko.core.network")
    exports_module = ModuleType("neko._outputs.exports")
    inputs_module = ModuleType("neko.inputs")
    tools_module = ModuleType("neko.core.tools")
    strategies_module = ModuleType("neko.core.strategies")

    class StubNetwork:
        pass

    class StubExports:
        def __init__(self, network: object) -> None:
            self.network = network

    def no_op(*args: Any, **kwargs: Any) -> None:
        del args, kwargs

    network_module.Network = StubNetwork  # type: ignore[attr-defined]
    exports_module.Exports = StubExports  # type: ignore[attr-defined]
    inputs_module.Universe = object  # type: ignore[attr-defined]
    inputs_module.signor = no_op  # type: ignore[attr-defined]
    tools_module.is_connected = lambda network: True  # type: ignore[attr-defined]
    strategies_module.connect_as_atopo = no_op  # type: ignore[attr-defined]
    strategies_module.connect_component = no_op  # type: ignore[attr-defined]
    strategies_module.complete_connection = no_op  # type: ignore[attr-defined]
    strategies_module.connect_network_radially = no_op  # type: ignore[attr-defined]
    strategies_module.connect_to_upstream_nodes = no_op  # type: ignore[attr-defined]
    strategies_module.connect_subgroup = no_op  # type: ignore[attr-defined]

    neko.core = core  # type: ignore[attr-defined]
    neko._outputs = outputs  # type: ignore[attr-defined]
    core.network = network_module  # type: ignore[attr-defined]
    outputs.exports = exports_module  # type: ignore[attr-defined]

    sys.modules.update(
        {
            "neko": neko,
            "neko.core": core,
            "neko.core.network": network_module,
            "neko._outputs": outputs,
            "neko._outputs.exports": exports_module,
            "neko.inputs": inputs_module,
            "neko.core.tools": tools_module,
            "neko.core.strategies": strategies_module,
        }
    )


NEKO_DIR = Path(__file__).parent.parent / "NeKo"
sys.path.insert(0, str(NEKO_DIR))
_install_neko_import_stubs()

# Other in-memory server tests use the same launcher-style module name.
# Remove any previously collected server's alias before importing NeKo.
for module_name in ("session_manager", "utils", "src", "src.helpers"):
    sys.modules.pop(module_name, None)

# These imports intentionally follow the launcher-compatible sys.path setup.
from session_manager import session_manager  # noqa: E402
from src import helpers as neko_helpers  # noqa: E402

from NeKo import server as neko_server  # noqa: E402

mcp = neko_server.mcp


def _run(coroutine: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coroutine)


async def _call_tool(name: str, arguments: dict[str, Any] | None = None) -> Any:
    async with Client(mcp) as client:
        return await client.call_tool(name, arguments or {})


async def _list_tools() -> Any:
    async with Client(mcp) as client:
        return await client.list_tools()


def _clear_sessions() -> None:
    for session_id in list(session_manager.list_sessions()):
        session_manager.delete_session(session_id)


def _create_session(network: object | None = None) -> str:
    session_id = session_manager.create_session()
    session = session_manager.get_session(session_id)
    assert session is not None
    if network is not None:
        session.set_network(network)
    return session_id


def _wait_for_lease_count(session_id: str, expected: int) -> bool:
    session = session_manager.get_session(session_id)
    assert session is not None
    with session_manager._condition:
        return session_manager._condition.wait_for(
            lambda: session._lease_count == expected,
            timeout=2,
        )


def _network_stub(**attributes: Any) -> SimpleNamespace:
    defaults: dict[str, Any] = {
        "nodes": pd.DataFrame(columns=["Uniprot", "Genesymbol"]),
        "edges": pd.DataFrame(columns=["source", "target", "Effect"]),
    }
    defaults.update(attributes)
    return SimpleNamespace(**defaults)


@pytest.fixture(autouse=True)
def isolated_sessions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(neko_server, "_SERVER_ROOT", tmp_path)
    monkeypatch.setattr(neko_helpers, "_SERVER_ROOT", tmp_path)
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


def test_network_guard_is_tool_error() -> None:
    session_id = _create_session()

    result = _run(
        _call_tool("add_gene", {"gene": "TP53", "session_id": session_id})
    )

    assert result.is_error is True
    assert "E_NO_NET" in result.content[0].text
    assert "create_network" in result.content[0].text


def test_invalid_database_is_tool_error() -> None:
    result = _run(
        _call_tool(
            "create_network",
            {"list_of_initial_genes": ["TP53"], "database": "invalid"},
        )
    )

    assert result.is_error is True
    assert "Unsupported database" in result.content[0].text
    assert session_manager.list_sessions() == {}


def test_missing_sif_file_is_tool_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.sif"

    result = _run(
        _call_tool(
            "create_network",
            {
                "list_of_initial_genes": [],
                "sif_file": str(missing_path),
            },
        )
    )

    assert result.is_error is True
    assert f"SIF file not found: {missing_path}" in result.content[0].text
    assert session_manager.list_sessions() == {}


def test_invalid_export_format_is_tool_error() -> None:
    session_id = _create_session(_network_stub())

    result = _run(
        _call_tool(
            "export_network",
            {"session_id": session_id, "format": "graphml"},
        )
    )

    assert result.is_error is True
    assert "Export Format Not Supported" in result.content[0].text


def test_invalid_targeted_strategy_is_tool_error() -> None:
    session_id = _create_session(_network_stub())

    result = _run(
        _call_tool(
            "connect_targeted_nodes",
            {
                "session_id": session_id,
                "strategy": "invalid",
                "nodes": ["TP53"],
            },
        )
    )

    assert result.is_error is True
    assert "Unsupported targeted strategy" in result.content[0].text


def test_partial_add_gene_failure_is_reported_as_tool_error() -> None:
    added_genes: list[str] = []

    def add_node(gene: str) -> None:
        added_genes.append(gene)

    def fail_autoconnect(**kwargs: Any) -> None:
        del kwargs
        raise RuntimeError("database connection failed")

    network = _network_stub(
        add_node=add_node,
        complete_connection=fail_autoconnect,
    )
    session_id = _create_session(network)

    result = _run(
        _call_tool(
            "add_gene",
            {
                "session_id": session_id,
                "gene": "TP53",
                "autoconnect": True,
            },
        )
    )

    assert result.is_error is True
    assert "TP53 was added, but autoconnect failed" in result.content[0].text
    assert added_genes == ["TP53"]


def test_empty_path_query_remains_successful() -> None:
    network = _network_stub(print_my_paths=lambda *args, **kwargs: None)
    session_id = _create_session(network)

    result = _run(
        _call_tool(
            "find_paths",
            {
                "session_id": session_id,
                "source": "TP53",
                "target": "MDM2",
            },
        )
    )

    assert result.is_error is False
    assert result.content[0].text == "No paths found."


def test_find_paths_restores_stdout_after_failure() -> None:
    def fail_path_search(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("path search failed")

    network = _network_stub(print_my_paths=fail_path_search)
    session_id = _create_session(network)
    stdout_before = sys.stdout

    result = _run(
        _call_tool(
            "find_paths",
            {
                "session_id": session_id,
                "source": "TP53",
                "target": "MDM2",
            },
        )
    )

    assert result.is_error is True
    assert "path search failed" in result.content[0].text
    assert sys.stdout is stdout_before


def test_session_locking_preserves_public_tool_schemas() -> None:
    listed_tools = _run(_list_tools())
    tools = {tool.name: tool for tool in listed_tools.tools}
    session_backed_tools = {
        "create_network",
        "add_gene",
        "remove_gene",
        "remove_interaction",
        "export_network",
        "list_genes_and_interactions",
        "find_paths",
        "reset_network",
        "clean_generated_files",
        "remove_bimodal_interactions",
        "remove_undefined_interactions",
        "list_bnet_files",
        "check_disconnected_nodes",
        "get_references",
        "extend_network",
        "set_default_params",
        "filter_interactions",
        "status",
        "list_components",
        "candidate_connectors",
        "bridge_components",
        "connect_targeted_nodes",
        "apply_global_connection",
    }

    for tool_name in session_backed_tools:
        properties = tools[tool_name].input_schema["properties"]
        assert "ctx" not in properties
        assert "sess" not in properties
        assert "network" not in properties
        assert "session_id" in properties


@pytest.mark.parametrize(
    "handler_name",
    [
        "add_gene",
        "remove_gene",
        "remove_interaction",
        "export_network",
        "list_genes_and_interactions",
        "find_paths",
        "reset_network",
        "clean_generated_files",
        "remove_bimodal_interactions",
        "remove_undefined_interactions",
        "list_bnet_files",
        "check_disconnected_nodes",
        "get_references",
        "extend_network",
        "set_default_params",
        "filter_interactions",
        "status",
        "list_components",
        "candidate_connectors",
        "bridge_components",
        "connect_targeted_nodes",
        "apply_global_connection",
    ],
)
def test_blocking_session_handlers_are_synchronous(
    handler_name: str,
) -> None:
    assert inspect.iscoroutinefunction(
        getattr(neko_server, handler_name)
    ) is False


def test_create_network_remains_asynchronous() -> None:
    assert inspect.iscoroutinefunction(neko_server.create_network) is True


def test_same_session_status_waits_for_mutation() -> None:
    mutation_started = Event()
    release_mutation = Event()

    def add_node(gene: str) -> None:
        del gene
        mutation_started.set()
        assert release_mutation.wait(timeout=2)

    network = _network_stub(
        add_node=add_node,
        convert_edgelist_into_genesymbol=lambda: pd.DataFrame(
            [{"source": "TP53", "target": "MDM2", "Effect": "stimulation"}]
        ),
    )
    session_id = _create_session(network)

    async def run_concurrently() -> tuple[Any, Any]:
        async with Client(mcp) as mutation_client, Client(mcp) as status_client:
            mutation_task = asyncio.create_task(
                mutation_client.call_tool(
                    "add_gene",
                    {"session_id": session_id, "gene": "EGFR"},
                )
            )
            assert await asyncio.to_thread(mutation_started.wait, 2)

            status_task = asyncio.create_task(
                status_client.call_tool(
                    "status",
                    {"session_id": session_id},
                )
            )
            assert await asyncio.to_thread(
                _wait_for_lease_count,
                session_id,
                2,
            )
            assert status_task.done() is False

            release_mutation.set()
            return await asyncio.gather(mutation_task, status_task)

    mutation_result, status_result = _run(run_concurrently())

    assert mutation_result.is_error is False
    assert status_result.is_error is False
    assert "edges=1" in status_result.content[0].text


def test_create_network_runs_backend_in_worker_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caller_thread = threading.get_ident()
    constructor_threads: list[int] = []

    class WorkerNetwork:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            constructor_threads.append(threading.get_ident())
            self.nodes = pd.DataFrame(
                [{"Uniprot": "P04637", "Genesymbol": "TP53"}]
            )
            self.edges = pd.DataFrame(
                [{"source": "P04637", "target": "Q00987"}]
            )

        def complete_connection(self, **kwargs: Any) -> None:
            del kwargs

        def convert_edgelist_into_genesymbol(self) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "source": "TP53",
                        "target": "MDM2",
                        "Effect": "stimulation",
                    }
                ]
            )

    monkeypatch.setattr(neko_server, "Network", WorkerNetwork)
    session_id = _create_session()

    result = _run(
        _call_tool(
            "create_network",
            {
                "session_id": session_id,
                "list_of_initial_genes": ["TP53"],
            },
        )
    )

    assert result.is_error is False
    assert constructor_threads
    assert constructor_threads[0] != caller_thread


def test_failed_rebuild_preserves_existing_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_network = _network_stub()
    session_id = _create_session(original_network)

    class FailingNetwork:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def complete_connection(self, **kwargs: Any) -> None:
            del kwargs
            raise RuntimeError("completion failed")

    monkeypatch.setattr(neko_server, "Network", FailingNetwork)

    result = _run(
        _call_tool(
            "create_network",
            {
                "session_id": session_id,
                "list_of_initial_genes": ["TP53"],
            },
        )
    )

    assert result.is_error is True
    assert "completion failed" in result.content[0].text
    session = session_manager.get_session(session_id)
    assert session is not None
    assert session.network is original_network


@pytest.mark.parametrize(
    ("tool_name", "arguments", "expected_method"),
    [
        (
            "bridge_components",
            {"comp_a": ["TP53"], "comp_b": ["MDM2"]},
            "connect_component",
        ),
        (
            "connect_targeted_nodes",
            {"strategy": "connect_to_upstream_nodes", "nodes": ["TP53"]},
            "connect_to_upstream_nodes",
        ),
        (
            "connect_targeted_nodes",
            {"strategy": "connect_subgroup", "nodes": ["TP53"]},
            "connect_subgroup",
        ),
        (
            "connect_targeted_nodes",
            {"strategy": "connect_as_atopo", "nodes": ["TP53"]},
            "connect_as_atopo",
        ),
        (
            "apply_global_connection",
            {"strategy": "complete_connection"},
            "complete_connection",
        ),
        (
            "apply_global_connection",
            {"strategy": "connect_network_radially"},
            "connect_network_radially",
        ),
    ],
)
def test_connection_tools_use_history_aware_network_methods(
    tool_name: str,
    arguments: dict[str, Any],
    expected_method: str,
) -> None:
    calls: list[str] = []

    def record(method_name: str):
        def call(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            calls.append(method_name)

        return call

    network = _network_stub(
        convert_edgelist_into_genesymbol=lambda: pd.DataFrame(
            columns=["source", "target", "Effect"]
        ),
        connect_component=record("connect_component"),
        connect_to_upstream_nodes=record("connect_to_upstream_nodes"),
        connect_subgroup=record("connect_subgroup"),
        connect_as_atopo=record("connect_as_atopo"),
        complete_connection=record("complete_connection"),
        connect_network_radially=record("connect_network_radially"),
    )
    session_id = _create_session(network)
    arguments["session_id"] = session_id

    result = _run(_call_tool(tool_name, arguments))

    assert result.is_error is False
    assert calls == [expected_method]
