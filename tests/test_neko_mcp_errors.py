"""Protocol-level tests for NeKo tool failure semantics."""

import asyncio
import sys
from collections.abc import Coroutine
from pathlib import Path
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
