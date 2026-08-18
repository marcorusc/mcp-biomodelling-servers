"""Protocol-level tests for NeKo tool semantics and concurrency."""

import asyncio
import inspect
import json
import sys
import threading
from collections.abc import Coroutine
from pathlib import Path
from threading import Event
from types import ModuleType, SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from mcp import Client, MCPError


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


_install_neko_import_stubs()

from NeKo import server as neko_server  # noqa: E402
from NeKo import session_manager as neko_session_manager  # noqa: E402
from NeKo.src import helpers as neko_helpers  # noqa: E402

mcp = neko_server.mcp
session_manager = neko_session_manager.session_manager


def _run(coroutine: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coroutine)


async def _call_tool(name: str, arguments: dict[str, Any] | None = None) -> Any:
    async with Client(mcp) as client:
        return await client.call_tool(name, arguments or {})


async def _list_tools() -> Any:
    async with Client(mcp) as client:
        return await client.list_tools()


async def _list_prompts() -> Any:
    async with Client(mcp) as client:
        return await client.list_prompts()


async def _get_prompt(name: str) -> Any:
    async with Client(mcp) as client:
        return await client.get_prompt(name)


async def _list_resources() -> Any:
    async with Client(mcp) as client:
        return await client.list_resources()


async def _list_resource_templates() -> Any:
    async with Client(mcp) as client:
        return await client.list_resource_templates()


async def _read_resource(uri: str) -> Any:
    async with Client(mcp) as client:
        return await client.read_resource(uri)


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


class _HistoryGraph:
    def __init__(
        self,
        parents: dict[int, list[int]],
        children: dict[int, list[int]],
    ) -> None:
        self.parents = parents
        self.children = children

    def predecessors(self, state_id: int) -> list[int]:
        return self.parents[state_id]

    def successors(self, state_id: int) -> list[int]:
        return self.children[state_id]


class _HistoryNetwork:
    def __init__(self) -> None:
        self._state_order = [0, 2, 4, 6]
        self._metadata = {
            0: {"label": "initial"},
            2: {"method": "add_node", "args": ["AKT1"], "kwargs": {}},
            4: {"method": "connect_nodes", "args": [], "kwargs": {}},
            6: {"method": "remove_node", "args": ["AKT1"], "kwargs": {}},
        }
        self._parents = {0: [], 2: [0], 4: [2], 6: [2]}
        self._children = {0: [2], 2: [4, 6], 4: [], 6: []}
        self._snapshots = {
            0: (
                pd.DataFrame(
                    [["P04637", "TP53"]],
                    columns=["Uniprot", "Genesymbol"],
                ),
                pd.DataFrame(columns=["source", "target", "Effect"]),
            ),
            2: (
                pd.DataFrame(
                    [["P04637", "TP53"], ["P31749", "AKT1"]],
                    columns=["Uniprot", "Genesymbol"],
                ),
                pd.DataFrame(
                    [["TP53", "AKT1", "stimulation"]],
                    columns=["source", "target", "Effect"],
                ),
            ),
            4: (
                pd.DataFrame(
                    [
                        ["P04637", "TP53"],
                        ["P31749", "AKT1"],
                        ["Q00987", "MDM2"],
                    ],
                    columns=["Uniprot", "Genesymbol"],
                ),
                pd.DataFrame(
                    [
                        ["TP53", "AKT1", "stimulation"],
                        ["AKT1", "MDM2", "inhibition"],
                    ],
                    columns=["source", "target", "Effect"],
                ),
            ),
            6: (
                pd.DataFrame(
                    [["P04637", "TP53"]],
                    columns=["Uniprot", "Genesymbol"],
                ),
                pd.DataFrame(columns=["source", "target", "Effect"]),
            ),
        }
        self.current_state_id = 4
        self.root_state_id = 0
        self.max_history_calls: list[int | None] = []
        self.nodes, self.edges = self._copy_snapshot(4)

    def _copy_snapshot(
        self,
        state_id: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        nodes, edges = self._snapshots[state_id]
        return nodes.copy(deep=True), edges.copy(deep=True)

    def list_states(self) -> list[dict[str, Any]]:
        return [
            {
                "id": state_id,
                "metadata": self._metadata[state_id],
            }
            for state_id in self._state_order
        ]

    def history_graph(self) -> _HistoryGraph:
        return _HistoryGraph(self._parents, self._children)

    def checkout(self, state_id: int) -> None:
        if state_id not in self._state_order:
            raise ValueError(f"Unknown state id: {state_id}")
        self.current_state_id = state_id
        self.nodes, self.edges = self._copy_snapshot(state_id)

    def undo(self) -> None:
        parents = self._parents[self.current_state_id]
        if parents:
            self.checkout(parents[-1])

    def redo(self, state_id: int | None = None) -> None:
        children = self._children[self.current_state_id]
        if not children:
            return
        if state_id is None:
            if len(children) > 1:
                raise ValueError(
                    "Multiple branches available; specify a target state id."
                )
            state_id = children[0]
        elif state_id not in children:
            raise ValueError(
                f"State {state_id} is not a child of "
                f"{self.current_state_id}."
            )
        self.checkout(state_id)

    def compare_states(self, state_a: int, state_b: int) -> dict[str, Any]:
        del state_a, state_b
        return {
            "added_nodes": ["MDM2", "AKT1"],
            "removed_nodes": ["EGFR"],
            "added_edges": [
                ("TP53", "AKT1", "stimulation"),
                ("AKT1", "MDM2", pd.NA),
            ],
            "removed_edges": [("EGFR", "TP53", "inhibition")],
        }

    def set_max_history(self, max_states: int | None) -> None:
        self.max_history_calls.append(max_states)
        if max_states is not None and len(self._state_order) > max_states:
            retained = [self.root_state_id, self.current_state_id]
            self._state_order = retained
            self._parents = {
                self.root_state_id: [],
                self.current_state_id: [self.root_state_id],
            }
            self._children = {
                self.root_state_id: [self.current_state_id],
                self.current_state_id: [],
            }

    def history_html(self) -> str:
        return '<div class="neko-history-graph"><svg>history</svg></div>'


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


def test_session_discovery_tools_publish_structured_output_schemas() -> None:
    listed_tools = _run(_list_tools())
    tools = {tool.name: tool for tool in listed_tools.tools}

    for tool_name, expected_title in {
        "list_sessions": "NeKoSessionListResult",
        "list_artifact_sessions": "NeKoArtifactSessionListResult",
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
    assert result.content[0].text == "No sessions."
    assert result.structured_content == {
        "server": "NeKo",
        "count": 0,
        "sessions": [],
    }


def test_list_sessions_returns_structured_neko_state() -> None:
    network = _network_stub(
        nodes=pd.DataFrame([{"Genesymbol": "TP53"}, {"Genesymbol": "MDM2"}]),
        edges=pd.DataFrame([{"source": "TP53", "target": "MDM2"}]),
    )
    session_id = _create_session(network)

    result = _run(_call_tool("list_sessions"))

    assert result.is_error is False
    assert session_id in result.content[0].text
    assert result.structured_content is not None
    structured_session = result.structured_content["sessions"][0]
    assert structured_session["session_id"] == session_id
    assert structured_session["created_at"] >= 0
    assert structured_session["last_accessed"] >= structured_session["created_at"]
    assert structured_session["is_default"] is True
    assert structured_session["has_network"] is True
    assert structured_session["node_count"] == 2
    assert structured_session["edge_count"] == 1
    assert structured_session["history_max_states"] is None
    assert result.structured_content["server"] == "NeKo"
    assert result.structured_content["count"] == 1


def test_list_artifact_sessions_returns_structured_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        neko_server,
        "_list_artifact_sessions_on_disk",
        lambda *_args, **_kwargs: [
            {
                "session_id": "neko-artifact-session",
                "server": "NeKo",
                "label": "connected network",
                "created_at": "2026-07-30T10:20:30+00:00",
                "files": ["Network.sif"],
            }
        ],
    )

    result = _run(_call_tool("list_artifact_sessions"))

    assert result.is_error is False
    assert "**neko-artifact-session** (connected network)" in result.content[0].text
    assert result.structured_content == {
        "server": "NeKo",
        "count": 1,
        "sessions": [
            {
                "session_id": "neko-artifact-session",
                "server": "NeKo",
                "label": "connected network",
                "created_at": "2026-07-30T10:20:30+00:00",
                "files": ["Network.sif"],
            }
        ],
    }


def test_artifact_tools_publish_structured_output_schemas() -> None:
    listed_tools = _run(_list_tools())
    tools = {tool.name: tool for tool in listed_tools.tools}

    for tool_name, expected_title in {
        "export_network": "NeKoNetworkExportResult",
        "export_neko_handoff": "NeKoHandoffExportResult",
        "list_bnet_files": "NeKoArtifactFileListResult",
        "clean_generated_files": "NeKoArtifactCleanupResult",
    }.items():
        schema = tools[tool_name].output_schema
        assert schema is not None
        assert schema["title"] == expected_title
        assert "result" not in schema["properties"]


def test_neko_guidance_prefers_typed_handoff_for_maboss_transfer() -> None:
    assert "export_neko_handoff" in neko_server.NEKO_SERVER_INSTRUCTIONS
    assert "export_neko_handoff" in neko_server.NEKO_AGENT_MANUAL
    assert "output nodes" in neko_server.NEKO_AGENT_MANUAL
    assert "export_network(format='bnet')" in neko_server.NEKO_AGENT_MANUAL


def test_neko_prompt_and_resources_publish_exact_contracts() -> None:
    prompts = _run(_list_prompts()).prompts
    resources = _run(_list_resources()).resources
    templates = _run(_list_resource_templates()).resource_templates
    rendered_prompt = _run(_get_prompt("neko_workflow_prompt"))
    manual = _run(_read_resource("docs://neko/agent_manual"))

    assert len(prompts) == 1
    assert prompts[0].name == "neko_workflow_prompt"
    assert prompts[0].title == "NeKo modelling workflow"
    assert prompts[0].arguments == []

    assert len(resources) == 1
    assert str(resources[0].uri) == "docs://neko/agent_manual"
    assert resources[0].name == "NeKo Agent Operations Manual"
    assert resources[0].title == "NeKo agent operations manual"
    assert resources[0].mime_type == "text/markdown"

    assert len(templates) == 1
    assert str(templates[0].uri_template) == (
        "neko://session/{session_id}/history"
    )
    assert templates[0].name == "Network History"
    assert templates[0].title == "NeKo network history"
    assert templates[0].mime_type == "text/html"

    expected = neko_server.NEKO_AGENT_MANUAL
    assert rendered_prompt.messages[0].content.text == expected
    assert manual.contents[0].mime_type == "text/markdown"
    assert manual.contents[0].text == expected


def test_scientific_tools_publish_named_structured_output_schemas() -> None:
    listed_tools = _run(_list_tools())
    tools = {tool.name: tool for tool in listed_tools.tools}

    for tool_name, expected_title in {
        "status": "NeKoNetworkStatusResult",
        "list_genes_and_interactions": "NeKoNetworkInventoryResult",
        "find_paths": "NeKoPathSearchResult",
        "analyze_connectivity": "NeKoConnectivityResult",
        "analyze_gene_set": "NeKoGeneSetAnalysisResult",
        "get_references": "NeKoReferenceQueryResult",
        "filter_interactions": "NeKoInteractionFilterResult",
        "preview_connection_impact": "NeKoConnectionPreviewResult",
    }.items():
        schema = tools[tool_name].output_schema
        assert schema is not None
        assert schema["title"] == expected_title
        assert "result" not in schema["properties"]
        assert "server" in schema["properties"]
        assert "session_id" in schema["properties"]


def test_status_returns_structured_network_presence_and_counts() -> None:
    empty_session_id = _create_session()
    empty_result = _run(
        _call_tool("status", {"session_id": empty_session_id})
    )

    assert empty_result.is_error is False
    assert empty_result.structured_content == {
        "server": "NeKo",
        "session_id": empty_session_id,
        "has_network": False,
        "node_count": 0,
        "interaction_count": 0,
    }

    edges = pd.DataFrame(
        [{"source": "TP53", "target": "MDM2", "Effect": "inhibition"}]
    )
    network = _network_stub(
        nodes=pd.DataFrame(
            [
                {"Uniprot": "P04637", "Genesymbol": "TP53"},
                {"Uniprot": "Q00987", "Genesymbol": "MDM2"},
            ]
        ),
        convert_edgelist_into_genesymbol=lambda: edges.copy(),
    )
    session_id = _create_session(network)
    result = _run(_call_tool("status", {"session_id": session_id}))

    assert result.is_error is False
    assert "nodes=2 edges=1" in result.content[0].text
    assert result.structured_content["has_network"] is True
    assert result.structured_content["node_count"] == 2
    assert result.structured_content["interaction_count"] == 1


def test_network_inventory_returns_nodes_interactions_and_truncation() -> None:
    edges = pd.DataFrame(
        [
            {
                "source": "TP53",
                "target": "MDM2",
                "Effect": "inhibition",
            },
            {
                "source": "MDM2",
                "target": "AKT1",
                "Effect": pd.NA,
            },
        ]
    )
    network = _network_stub(
        nodes=pd.DataFrame(
            [
                {
                    "Uniprot": "P04637",
                    "Genesymbol": "TP53",
                    "Type": "protein",
                },
                {
                    "Uniprot": "Q00987",
                    "Genesymbol": "MDM2",
                    "Type": pd.NA,
                },
            ]
        ),
        convert_edgelist_into_genesymbol=lambda: edges.copy(),
    )
    session_id = _create_session(network)

    result = _run(
        _call_tool(
            "list_genes_and_interactions",
            {
                "session_id": session_id,
                "verbosity": "preview",
                "max_rows": 1,
            },
        )
    )

    assert result.is_error is False
    assert "Nodes (2 total)" in result.content[0].text
    assert "Interactions (2 total)" in result.content[0].text
    assert result.structured_content["total_node_count"] == 2
    assert result.structured_content["total_interaction_count"] == 2
    assert result.structured_content["returned_node_count"] == 1
    assert result.structured_content["returned_interaction_count"] == 1
    assert result.structured_content["truncated"] is True
    assert result.structured_content["nodes"] == [
        {
            "gene_symbol": "TP53",
            "uniprot": "P04637",
            "node_type": "protein",
        }
    ]
    assert result.structured_content["interactions"] == [
        {
            "source": "TP53",
            "target": "MDM2",
            "effect": "inhibition",
        }
    ]


def test_find_paths_returns_captured_structured_lines() -> None:
    def print_paths(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        print("TP53 -> MDM2")
        print("TP53 -> AKT1 -> MDM2")

    network = _network_stub(print_my_paths=print_paths)
    session_id = _create_session(network)

    result = _run(
        _call_tool(
            "find_paths",
            {
                "session_id": session_id,
                "source": "TP53",
                "target": "MDM2",
                "maxlen": 3,
                "verbosity": "full",
            },
        )
    )

    assert result.is_error is False
    assert "Paths (full output)" in result.content[0].text
    assert result.structured_content == {
        "server": "NeKo",
        "session_id": session_id,
        "source": "TP53",
        "target": "MDM2",
        "max_length": 3,
        "has_output": True,
        "output_line_count": 2,
        "path_output_lines": [
            "TP53 -> MDM2",
            "TP53 -> AKT1 -> MDM2",
        ],
    }


def test_disconnected_nodes_return_both_biological_identifiers() -> None:
    network = _network_stub(
        nodes=pd.DataFrame(
            [
                {"Uniprot": "P04637", "Genesymbol": "TP53"},
                {"Uniprot": "Q00987", "Genesymbol": "MDM2"},
                {"Uniprot": "P31749", "Genesymbol": "AKT1"},
            ]
        ),
        edges=pd.DataFrame(
            [
                {
                    "source": "P04637",
                    "target": "Q00987",
                    "Effect": "inhibition",
                }
            ]
        ),
    )
    session_id = _create_session(network)

    result = _run(
        _call_tool(
            "analyze_connectivity",
            {"session_id": session_id, "verbosity": "full"},
        )
    )

    assert result.is_error is False
    assert "AKT1" in result.content[0].text
    assert result.structured_content["total_node_count"] == 3
    assert result.structured_content["disconnected_count"] == 1
    assert result.structured_content["all_nodes_have_interactions"] is False
    assert result.structured_content["disconnected_nodes"] == [
        {
            "gene_symbol": "AKT1",
            "uniprot": "P31749",
            "node_type": None,
        }
    ]

    network.edges.loc[len(network.edges)] = {
        "source": "Q00987",
        "target": "P31749",
        "Effect": "stimulation",
    }
    connected_result = _run(
        _call_tool(
            "analyze_connectivity",
            {"session_id": session_id, "verbosity": "full"},
        )
    )
    assert connected_result.is_error is False
    assert "No isolated nodes." in connected_result.content[0].text
    assert connected_result.structured_content["disconnected_count"] == 0
    assert connected_result.structured_content["all_nodes_have_interactions"] is True
    assert connected_result.structured_content["disconnected_nodes"] == []


def test_reference_query_preserves_complete_normalized_evidence() -> None:
    edges = pd.DataFrame(
        [
            {
                "source": "TP53",
                "target": "MDM2",
                "Effect": "inhibition",
                "References": (
                    "PMID:1; PMID:2, PMID:3; PMID:4; "
                    "PMID:5; PMID:6; PMID:1"
                ),
            }
        ]
    )
    network = _network_stub(
        convert_edgelist_into_genesymbol=lambda: edges.copy()
    )
    session_id = _create_session(network)

    result = _run(
        _call_tool(
            "get_references",
            {
                "session_id": session_id,
                "node1": "TP53",
                "node2": "MDM2",
                "verbosity": "full",
            },
        )
    )

    assert result.is_error is False
    assert "(+1 more)" in result.content[0].text
    interaction = result.structured_content["interactions"][0]
    assert interaction["reference_count"] == 6
    assert interaction["references"] == [
        "PMID:1",
        "PMID:2",
        "PMID:3",
        "PMID:4",
        "PMID:5",
        "PMID:6",
    ]


def test_filter_interactions_returns_typed_records_and_valid_json() -> None:
    edges = pd.DataFrame(
        [
            {
                "source": "TP53",
                "target": "MDM2",
                "Effect": "inhibition",
            },
            {
                "source": "AKT1",
                "target": "MDM2",
                "Effect": "stimulation",
            },
        ]
    )
    network = _network_stub(
        convert_edgelist_into_genesymbol=lambda: edges.copy()
    )
    session_id = _create_session(network)

    result = _run(
        _call_tool(
            "filter_interactions",
            {
                "session_id": session_id,
                "effect": ["inhibition"],
                "format": "json",
                "verbosity": "full",
            },
        )
    )

    assert result.is_error is False
    assert json.loads(result.content[0].text) == [
        {
            "source": "TP53",
            "target": "MDM2",
            "effect": "inhibition",
        }
    ]
    assert result.structured_content["effect_filter"] == ["inhibition"]
    assert result.structured_content["total_match_count"] == 1
    assert result.structured_content["returned_count"] == 1
    assert result.structured_content["truncated"] is False

    empty_result = _run(
        _call_tool(
            "filter_interactions",
            {
                "session_id": session_id,
                "source": "EGFR",
                "format": "json",
            },
        )
    )
    assert empty_result.is_error is False
    assert json.loads(empty_result.content[0].text) == []
    assert empty_result.structured_content["total_match_count"] == 0
    assert empty_result.structured_content["interactions"] == []


def test_filter_interactions_supports_internal_and_boundary_node_sets() -> None:
    edges = pd.DataFrame(
        [
            {"source": "TP53", "target": "MDM2", "Effect": "inhibition"},
            {"source": "MDM2", "target": "AKT1", "Effect": "stimulation"},
            {"source": "EGFR", "target": "AKT1", "Effect": "stimulation"},
        ]
    )
    network = _network_stub(
        nodes=pd.DataFrame(
            [
                {"Uniprot": "P04637", "Genesymbol": "TP53"},
                {"Uniprot": "Q00987", "Genesymbol": "MDM2"},
                {"Uniprot": "P31749", "Genesymbol": "AKT1"},
                {"Uniprot": "P00533", "Genesymbol": "EGFR"},
            ]
        ),
        convert_edgelist_into_genesymbol=lambda: edges.copy(),
    )
    session_id = _create_session(network)

    internal = _run(
        _call_tool(
            "filter_interactions",
            {
                "session_id": session_id,
                "nodes": ["TP53", "MDM2"],
                "node_scope": "internal",
                "format": "json",
            },
        )
    )
    boundary = _run(
        _call_tool(
            "filter_interactions",
            {
                "session_id": session_id,
                "nodes": ["P04637", "MDM2"],
                "node_scope": "boundary",
                "format": "json",
            },
        )
    )

    assert internal.is_error is False
    assert internal.structured_content["node_scope"] == "internal"
    assert internal.structured_content["total_match_count"] == 1
    assert internal.structured_content["interactions"] == [
        {"source": "TP53", "target": "MDM2", "effect": "inhibition"}
    ]
    assert boundary.is_error is False
    assert boundary.structured_content["node_scope"] == "boundary"
    assert boundary.structured_content["total_match_count"] == 1
    assert boundary.structured_content["interactions"] == [
        {"source": "MDM2", "target": "AKT1", "effect": "stimulation"}
    ]


def test_analyze_gene_set_reports_induced_components_and_boundary_edges() -> None:
    edges = pd.DataFrame(
        [
            {"source": "TP53", "target": "MDM2", "Effect": "inhibition"},
            {"source": "MDM2", "target": "COMPLEX:X", "Effect": "stimulation"},
            {"source": "COMPLEX:Y", "target": "AKT1", "Effect": "stimulation"},
            {"source": "COMPLEX:X", "target": "COMPLEX:Y", "Effect": "stimulation"},
        ]
    )
    network = _network_stub(
        nodes=pd.DataFrame(
            [
                {"Uniprot": "P04637", "Genesymbol": "TP53"},
                {"Uniprot": "Q00987", "Genesymbol": "MDM2"},
                {"Uniprot": "P31749", "Genesymbol": "AKT1"},
                {"Uniprot": "COMPLEX:X", "Genesymbol": "COMPLEX:X"},
                {"Uniprot": "COMPLEX:Y", "Genesymbol": "COMPLEX:Y"},
            ]
        ),
        convert_edgelist_into_genesymbol=lambda: edges.copy(),
    )
    session_id = _create_session(network)

    result = _run(
        _call_tool(
            "analyze_gene_set",
            {
                "session_id": session_id,
                "genes": ["p04637", "MDM2", "AKT1", "MISSING"],
                "connectivity": "weak",
                "verbosity": "full",
                "format": "json",
            },
        )
    )

    assert result.is_error is False
    payload = result.structured_content
    assert json.loads(result.content[0].text) == payload
    assert payload["resolved_genes"] == ["TP53", "MDM2", "AKT1"]
    assert payload["missing_genes"] == ["MISSING"]
    assert payload["internal_edge_count"] == 1
    assert payload["boundary_edge_count"] == 2
    assert payload["induced_isolates"] == ["AKT1"]
    assert payload["component_count"] == 2
    assert payload["largest_component_size"] == 2
    assert payload["components"] == [
        {"component_id": 0, "size": 2, "genes": ["MDM2", "TP53"]},
        {"component_id": 1, "size": 1, "genes": ["AKT1"]},
    ]
    assert payload["boundary_edges"] == [
        {"source": "MDM2", "target": "COMPLEX:X", "effect": "stimulation"},
        {"source": "COMPLEX:Y", "target": "AKT1", "effect": "stimulation"},
    ]

    strong_result = _run(
        _call_tool(
            "analyze_gene_set",
            {
                "session_id": session_id,
                "genes": ["TP53", "MDM2", "AKT1"],
                "connectivity": "strong",
            },
        )
    )
    assert strong_result.is_error is False
    assert strong_result.structured_content["component_count"] == 3
    assert strong_result.structured_content["largest_component_size"] == 1


def test_component_output_contains_complete_gene_symbol_membership() -> None:
    network = _network_stub(
        nodes=pd.DataFrame(
            [
                {"Uniprot": "P04637", "Genesymbol": "TP53"},
                {"Uniprot": "Q00987", "Genesymbol": "MDM2"},
                {"Uniprot": "P31749", "Genesymbol": "AKT1"},
            ]
        ),
        edges=pd.DataFrame(
            [
                {
                    "source": "P04637",
                    "target": "Q00987",
                    "Effect": "inhibition",
                }
            ]
        ),
    )
    session_id = _create_session(network)

    result = _run(
        _call_tool(
            "analyze_connectivity",
            {
                "session_id": session_id,
                "verbosity": "full",
                "format": "json",
            },
        )
    )

    assert result.is_error is False
    assert isinstance(json.loads(result.content[0].text), dict)
    assert result.structured_content["component_count"] == 2
    assert result.structured_content["largest_component_size"] == 2
    memberships = [
        {
            node["gene_symbol"]
            for node in component["nodes"]
        }
        for component in result.structured_content["components"]
    ]
    assert {"TP53", "MDM2"} in memberships
    assert {"AKT1"} in memberships


def test_hub_candidates_return_typed_identifiers_and_scores() -> None:
    network = _network_stub(
        nodes=pd.DataFrame(
            [
                {"Uniprot": "P04637", "Genesymbol": "TP53"},
                {"Uniprot": "Q00987", "Genesymbol": "MDM2"},
            ]
        ),
        edges=pd.DataFrame(
            [
                {
                    "source": "P04637",
                    "target": "Q00987",
                    "Effect": "inhibition",
                }
            ]
        ),
    )
    session_id = _create_session(network)

    result = _run(
        _call_tool(
            "preview_connection_impact",
            {
                "session_id": session_id,
                "method": "hubs",
                "top_k": 1,
                "verbosity": "full",
            },
        )
    )

    assert result.is_error is False
    assert result.structured_content["method"] == "hubs"
    assert result.structured_content["suggestion_count"] == 1
    assert result.structured_content["simulation"] is None
    assert result.structured_content["hub_candidates"] == [
        {
            "gene_symbol": "TP53",
            "uniprot": "P04637",
            "relative_score": 1.0,
            "degree": 1,
        }
    ]


@pytest.mark.parametrize(
    ("method", "expected_max_length", "expected_only_signed"),
    [
        ("relax_max_len", 3, True),
        ("unsigned", 2, False),
    ],
)
def test_connector_simulations_return_typed_predictions(
    method: str,
    expected_max_length: int,
    expected_only_signed: bool,
) -> None:
    completion_calls: list[dict[str, Any]] = []

    class SimulatedNetwork:
        def __init__(self) -> None:
            self.nodes = pd.DataFrame(
                [
                    {"Uniprot": "P04637", "Genesymbol": "TP53"},
                    {"Uniprot": "Q00987", "Genesymbol": "MDM2"},
                ]
            )
            self.edges = pd.DataFrame(
                [{"source": "P04637", "target": "Q00987"}]
            )

        def complete_connection(self, **kwargs: Any) -> None:
            completion_calls.append(kwargs)
            self.edges.loc[len(self.edges)] = {
                "source": "Q00987",
                "target": "P04637",
            }

    session_id = _create_session(SimulatedNetwork())

    result = _run(
        _call_tool(
            "preview_connection_impact",
            {
                "session_id": session_id,
                "method": method,
                "format": "json",
            },
        )
    )

    assert result.is_error is False
    assert completion_calls == [
        {
            "maxlen": expected_max_length,
            "path_policy": "one_shortest",
            "reuse_policy": "discovered_paths",
            "only_signed": expected_only_signed,
            "consensus": True,
        }
    ]
    assert json.loads(result.content[0].text)["method"] == method
    assert result.structured_content["hub_candidates"] == []
    assert result.structured_content["simulation"] == {
        "predicted_new_edges": 1,
        "simulated_max_length": expected_max_length,
        "simulated_only_signed": expected_only_signed,
        "simulated_path_policy": "one_shortest",
        "simulated_reuse_policy": "discovered_paths",
    }


def test_bnet_export_returns_structured_sanitization_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class RecordingExports:
        def __init__(self, network: object) -> None:
            del network

        def export_bnet(self, prefix: str) -> None:
            Path(f"{prefix}.bnet").write_text(
                "# model in BoolNet format\n"
                "targets, factors\n"
                "A-1, A-1\n"
                "A_1, A_1\n",
                encoding="utf-8",
            )

    network = _network_stub()
    session_id = _create_session(network)
    artifact_dir = tmp_path / "artifacts" / session_id
    artifact_dir.mkdir(parents=True)
    stale_path = artifact_dir / "AAA_previous.bnet"
    stale_path.write_text("STALE, STALE\n", encoding="utf-8")
    monkeypatch.setattr(neko_server, "Exports", RecordingExports)
    monkeypatch.setattr(neko_server, "is_connected", lambda _network: True)
    monkeypatch.setattr(
        neko_server,
        "_export_dir",
        lambda _session_id: artifact_dir,
    )

    result = _run(
        _call_tool(
            "export_network",
            {
                "format": "bnet",
                "session_id": session_id,
                "verbosity": "full",
            },
        )
    )

    output_path = artifact_dir / "Network.bnet"
    assert result.is_error is False
    assert f"BNET exported: `{output_path}`" in result.content[0].text
    assert result.structured_content is not None
    assert result.structured_content["server"] == "NeKo"
    assert result.structured_content["session_id"] == session_id
    assert result.structured_content["format"] == "bnet"
    assert result.structured_content["renamed_nodes"] == ["A-1"]
    assert result.structured_content["duplicate_rules_removed"] == ["A_1"]
    assert result.structured_content["file"]["name"] == "Network.bnet"
    assert result.structured_content["file"]["media_type"] == "text/plain"
    assert result.structured_content["file"]["size_bytes"] == output_path.stat().st_size
    assert stale_path.read_text(encoding="utf-8") == "STALE, STALE\n"


def test_neko_handoff_exports_typed_provenance_and_translated_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class RecordingExports:
        def __init__(self, network: object) -> None:
            del network

        def export_bnet(self, prefix: str) -> None:
            Path(f"{prefix}_1.bnet").write_text(
                "# model in BoolNet format\n"
                "targets, factors\n"
                "A-1, B\n"
                "A_1, A_1\n"
                "B, A-1\n",
                encoding="utf-8",
            )

    network = _network_stub(current_state_id=7)
    session_id = _create_session(network)
    artifact_dir = tmp_path / "artifacts" / session_id
    artifact_dir.mkdir(parents=True)
    monkeypatch.setattr(neko_server, "Exports", RecordingExports)
    monkeypatch.setattr(neko_server, "is_connected", lambda _network: True)
    monkeypatch.setattr(neko_server, "_neko_package_version", lambda: "2.0.0")
    monkeypatch.setattr(
        neko_server,
        "_export_dir",
        lambda _session_id: artifact_dir,
    )

    result = _run(
        _call_tool(
            "export_neko_handoff",
            {
                "biological_context": "Test A/B signalling.",
                "output_nodes": ["A-1"],
                "artifact_prefix": "a_b_model",
                "session_id": session_id,
            },
        )
    )

    bnet_path = artifact_dir / "a_b_model.bnet"
    manifest_path = artifact_dir / "a_b_model.handoff.json"
    assert result.is_error is False
    assert bnet_path.read_text(encoding="utf-8") == "A_1, B\nB, A_1\n"
    assert manifest_path.is_file()
    assert result.structured_content is not None
    assert result.structured_content["server"] == "NeKo"
    assert result.structured_content["session_id"] == session_id
    assert result.structured_content["manifest_file"]["path"] == str(
        manifest_path
    )
    manifest = result.structured_content["manifest"]
    assert manifest["schema_version"] == "1.0"
    assert manifest["handoff_type"] == "neko-to-maboss"
    assert manifest["history_state_id"] == 7
    assert manifest["biological_context"] == "Test A/B signalling."
    assert manifest["source"]["mcp_package"] == {
        "name": "mcp-biomodelling-servers",
        "version": neko_server.__version__,
    }
    assert manifest["source"]["modelling_package"] == {
        "name": "nekomata",
        "version": "2.0.0",
    }
    assert manifest["network"] == {
        "nodes": ["A_1", "B"],
        "output_nodes": ["A_1"],
        "renamed_nodes": ["A-1"],
        "node_renames": {"A-1": "A_1"},
        "duplicate_rules_removed": ["A_1"],
    }
    assert manifest["bnet_file"]["path"] == str(bnet_path)
    assert len(manifest["bnet_file"]["sha256"]) == 64
    assert len(result.structured_content["manifest_file"]["sha256"]) == 64
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert "Next: pass the manifest path" in result.content[0].text


def test_neko_handoff_rejects_multiple_generated_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class MultipleModelExports:
        def __init__(self, network: object) -> None:
            del network

        def export_bnet(self, prefix: str) -> None:
            Path(f"{prefix}_1.bnet").write_text("A, A\n", encoding="utf-8")
            Path(f"{prefix}_2.bnet").write_text("B, B\n", encoding="utf-8")

    session_id = _create_session(_network_stub())
    artifact_dir = tmp_path / "artifacts" / session_id
    artifact_dir.mkdir(parents=True)
    monkeypatch.setattr(neko_server, "Exports", MultipleModelExports)
    monkeypatch.setattr(neko_server, "is_connected", lambda _network: True)
    monkeypatch.setattr(neko_server, "_neko_package_version", lambda: "2.0.0")
    monkeypatch.setattr(
        neko_server,
        "_export_dir",
        lambda _session_id: artifact_dir,
    )

    result = _run(
        _call_tool(
            "export_neko_handoff",
            {
                "biological_context": "Test context.",
                "artifact_prefix": "ambiguous",
                "session_id": session_id,
            },
        )
    )

    assert result.is_error is True
    assert "generated multiple BNET models" in result.content[0].text
    assert "requires exactly one model" in result.content[0].text
    assert list(artifact_dir.iterdir()) == []


@pytest.mark.parametrize(
    ("output_nodes", "message"),
    [
        (["missing"], "absent from the sanitized BNET"),
        (
            ["A-1", "A_1"],
            "collapse onto the same sanitized MaBoSS node",
        ),
        (["B", "B"], "must not contain duplicate names"),
    ],
)
def test_neko_handoff_rejects_invalid_outputs_without_partial_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    output_nodes: list[str],
    message: str,
) -> None:
    class RecordingExports:
        def __init__(self, network: object) -> None:
            del network

        def export_bnet(self, prefix: str) -> None:
            Path(f"{prefix}.bnet").write_text(
                "A-1, B\nA_1, A_1\nB, A-1\n",
                encoding="utf-8",
            )

    session_id = _create_session(_network_stub())
    artifact_dir = tmp_path / "artifacts" / session_id
    artifact_dir.mkdir(parents=True)
    monkeypatch.setattr(neko_server, "Exports", RecordingExports)
    monkeypatch.setattr(neko_server, "is_connected", lambda _network: True)
    monkeypatch.setattr(neko_server, "_neko_package_version", lambda: "2.0.0")
    monkeypatch.setattr(
        neko_server,
        "_export_dir",
        lambda _session_id: artifact_dir,
    )

    result = _run(
        _call_tool(
            "export_neko_handoff",
            {
                "biological_context": "Test context.",
                "output_nodes": output_nodes,
                "artifact_prefix": "invalid_outputs",
                "session_id": session_id,
            },
        )
    )

    assert result.is_error is True
    assert message in result.content[0].text
    assert list(artifact_dir.iterdir()) == []


def test_neko_handoff_refuses_existing_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    session_id = _create_session(_network_stub())
    artifact_dir = tmp_path / "artifacts" / session_id
    artifact_dir.mkdir(parents=True)
    existing_bnet = artifact_dir / "retained.bnet"
    existing_bnet.write_text("original\n", encoding="utf-8")
    monkeypatch.setattr(neko_server, "is_connected", lambda _network: True)
    monkeypatch.setattr(
        neko_server,
        "_export_dir",
        lambda _session_id: artifact_dir,
    )

    result = _run(
        _call_tool(
            "export_neko_handoff",
            {
                "biological_context": "Test context.",
                "artifact_prefix": "retained",
                "session_id": session_id,
            },
        )
    )

    assert result.is_error is True
    assert "Refusing to overwrite" in result.content[0].text
    assert "different artifact_prefix" in result.content[0].text
    assert existing_bnet.read_text(encoding="utf-8") == "original\n"
    assert not (artifact_dir / "retained.handoff.json").exists()


def test_neko_handoff_allows_empty_declared_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class RecordingExports:
        def __init__(self, network: object) -> None:
            del network

        def export_bnet(self, prefix: str) -> None:
            Path(f"{prefix}.bnet").write_text("A, A\n", encoding="utf-8")

    session_id = _create_session(_network_stub())
    artifact_dir = tmp_path / "artifacts" / session_id
    artifact_dir.mkdir(parents=True)
    monkeypatch.setattr(neko_server, "Exports", RecordingExports)
    monkeypatch.setattr(neko_server, "is_connected", lambda _network: True)
    monkeypatch.setattr(neko_server, "_neko_package_version", lambda: "2.0.0")
    monkeypatch.setattr(
        neko_server,
        "_export_dir",
        lambda _session_id: artifact_dir,
    )

    result = _run(
        _call_tool(
            "export_neko_handoff",
            {
                "biological_context": "Test context.",
                "session_id": session_id,
            },
        )
    )

    assert result.is_error is False
    assert result.structured_content["manifest"]["network"]["output_nodes"] == []
    assert "select a small output set in MaBoSS" in result.content[0].text


def test_neko_bnet_listing_and_cleanup_are_structured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    session_id = _create_session()
    monkeypatch.setattr(neko_server, "_SERVER_ROOT", tmp_path)
    artifact_dir = tmp_path / "artifacts" / session_id
    artifact_dir.mkdir(parents=True)
    bnet_path = artifact_dir / "Network.bnet"
    bnet_path.write_text("A, A\n", encoding="utf-8")
    (artifact_dir / "Network.sif").write_text("A\t1\tA\n", encoding="utf-8")

    listing_result = _run(
        _call_tool("list_bnet_files", {"session_id": session_id})
    )
    cleanup_result = _run(
        _call_tool("clean_generated_files", {"session_id": session_id})
    )

    assert listing_result.is_error is False
    assert listing_result.content[0].text == "Network.bnet"
    assert listing_result.structured_content is not None
    assert listing_result.structured_content["scope"] == "session"
    assert listing_result.structured_content["session_id"] == session_id
    assert listing_result.structured_content["count"] == 1
    assert listing_result.structured_content["files"][0]["path"] == str(bnet_path)
    assert cleanup_result.structured_content == {
        "session_id": session_id,
        "removed_count": 2,
        "server": "NeKo",
    }
    assert not artifact_dir.exists()


def test_unknown_default_session_is_tool_error() -> None:
    result = _run(
        _call_tool("set_default_session", {"session_id": "missing-session"})
    )

    assert result.is_error is True
    assert "Session not found: missing-session" in result.content[0].text


def test_unknown_explicit_tool_session_does_not_create_fallback() -> None:
    result = _run(
        _call_tool("status", {"session_id": "missing-session"})
    )

    assert result.is_error is True
    assert "Unknown NeKo session: missing-session" in result.content[0].text
    assert session_manager.list_sessions() == {}


def test_network_guard_is_tool_error() -> None:
    session_id = _create_session()

    result = _run(
        _call_tool("add_nodes", {"genes": ["TP53"], "session_id": session_id})
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
    assert "validation error" in result.content[0].text.lower()
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
    assert "validation error" in result.content[0].text.lower()


def test_neko_handoff_requires_network_and_connectivity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_session = _create_session()
    missing_network = _run(
        _call_tool(
            "export_neko_handoff",
            {
                "biological_context": "Test context.",
                "session_id": empty_session,
            },
        )
    )
    assert missing_network.is_error is True
    assert "No network" in missing_network.content[0].text

    disconnected_session = _create_session(_network_stub())
    monkeypatch.setattr(neko_server, "is_connected", lambda _network: False)
    disconnected = _run(
        _call_tool(
            "export_neko_handoff",
            {
                "biological_context": "Test context.",
                "session_id": disconnected_session,
            },
        )
    )
    assert disconnected.is_error is True
    assert "connected" in disconnected.content[0].text.lower()


@pytest.mark.parametrize(
    "artifact_prefix",
    ["../escape", "/absolute", ".hidden", "trailing.", "with space"],
)
def test_neko_handoff_rejects_unsafe_artifact_prefixes(
    artifact_prefix: str,
) -> None:
    session_id = _create_session(_network_stub())

    result = _run(
        _call_tool(
            "export_neko_handoff",
            {
                "biological_context": "Test context.",
                "artifact_prefix": artifact_prefix,
                "session_id": session_id,
            },
        )
    )

    assert result.is_error is True
    assert "validation error" in result.content[0].text.lower()


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
    assert "validation error" in result.content[0].text.lower()


def test_partial_add_nodes_failure_is_reported_as_tool_error() -> None:
    added_genes: list[str] = []

    def add_node(gene: str) -> None:
        added_genes.append(gene)

    def fail_autoconnect(**kwargs: Any) -> None:
        del kwargs
        raise RuntimeError("database connection failed")

    network = _network_stub(
        add_node=add_node,
        connect_nodes=fail_autoconnect,
    )
    session_id = _create_session(network)

    result = _run(
        _call_tool(
            "add_nodes",
            {
                "session_id": session_id,
                "genes": ["TP53"],
                "autoconnect": True,
            },
        )
    )

    assert result.is_error is True
    assert "autoconnect failed" in result.content[0].text
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
    assert result.structured_content == {
        "server": "NeKo",
        "session_id": session_id,
        "source": "TP53",
        "target": "MDM2",
        "max_length": 3,
        "has_output": False,
        "output_line_count": 0,
        "path_output_lines": [],
    }


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


def test_list_network_history_returns_branching_structured_output() -> None:
    network = _HistoryNetwork()
    session_id = _create_session(network)

    result = _run(
        _call_tool(
            "list_network_history",
            {"session_id": session_id},
        )
    )

    assert result.is_error is False
    assert result.structured_content is not None
    assert result.structured_content["server"] == "NeKo"
    assert result.structured_content["session_id"] == session_id
    assert result.structured_content["current_state_id"] == 4
    assert result.structured_content["root_state_id"] == 0
    assert result.structured_content["max_states"] is None
    assert [
        state["state_id"]
        for state in result.structured_content["states"]
    ] == [0, 2, 4, 6]
    branch = result.structured_content["states"][1]
    assert branch["parent_state_ids"] == [0]
    assert branch["child_state_ids"] == [4, 6]


def test_navigate_network_history_moves_and_invalidates_cache() -> None:
    network = _HistoryNetwork()
    session_id = _create_session(network)
    session = session_manager.get_session(session_id)
    assert session is not None
    session._edges_cache_dirty = False

    result = _run(
        _call_tool(
            "navigate_network_history",
            {"action": "undo", "session_id": session_id},
        )
    )

    assert result.is_error is False
    assert result.structured_content["server"] == "NeKo"
    assert result.structured_content["previous_state_id"] == 4
    assert result.structured_content["current_state_id"] == 2
    assert result.structured_content["moved"] is True
    assert network.current_state_id == 2
    assert session._edges_cache_dirty is True


def test_navigate_network_history_reports_root_noop_as_success() -> None:
    network = _HistoryNetwork()
    network.checkout(0)
    session_id = _create_session(network)
    session = session_manager.get_session(session_id)
    assert session is not None
    session._edges_cache_dirty = False

    result = _run(
        _call_tool(
            "navigate_network_history",
            {"action": "undo", "session_id": session_id},
        )
    )

    assert result.is_error is False
    assert result.structured_content["moved"] is False
    assert "root state" in result.structured_content["message"]
    assert session._edges_cache_dirty is False


def test_navigate_network_history_requires_checkout_state_id() -> None:
    session_id = _create_session(_HistoryNetwork())

    result = _run(
        _call_tool(
            "navigate_network_history",
            {"action": "checkout", "session_id": session_id},
        )
    )

    assert result.is_error is True
    assert "checkout requires an exact state_id" in result.content[0].text


def test_navigate_network_history_requires_branch_target_for_redo() -> None:
    network = _HistoryNetwork()
    network.checkout(2)
    session_id = _create_session(network)

    result = _run(
        _call_tool(
            "navigate_network_history",
            {"action": "redo", "session_id": session_id},
        )
    )

    assert result.is_error is True
    assert "Multiple branches available" in result.content[0].text
    assert network.current_state_id == 2


def test_compare_network_states_is_deterministic_and_non_mutating() -> None:
    network = _HistoryNetwork()
    session_id = _create_session(network)

    result = _run(
        _call_tool(
            "compare_network_states",
            {
                "state_a": 0,
                "state_b": 4,
                "session_id": session_id,
            },
        )
    )

    assert result.is_error is False
    assert result.structured_content["server"] == "NeKo"
    assert result.structured_content["added_nodes"] == ["AKT1", "MDM2"]
    assert result.structured_content["edge_columns"] == [
        "source",
        "target",
        "Effect",
    ]
    assert result.structured_content["added_edges"] == [
        {"source": "AKT1", "target": "MDM2", "Effect": None},
        {
            "source": "TP53",
            "target": "AKT1",
            "Effect": "stimulation",
        },
    ]
    assert network.current_state_id == 4


def test_compare_network_states_rejects_positional_fallback() -> None:
    network = _HistoryNetwork()
    session_id = _create_session(network)

    result = _run(
        _call_tool(
            "compare_network_states",
            {
                "state_a": 1,
                "state_b": 4,
                "session_id": session_id,
            },
        )
    )

    assert result.is_error is True
    assert "Unknown history state ID(s): 1" in result.content[0].text
    assert "Available state IDs: 0, 2, 4, 6" in result.content[0].text


def test_set_network_history_limit_prunes_and_can_restore_unbounded_policy() -> None:
    network = _HistoryNetwork()
    session_id = _create_session(network)

    bounded = _run(
        _call_tool(
            "set_network_history_limit",
            {"max_states": 2, "session_id": session_id},
        )
    )

    assert bounded.is_error is False
    assert bounded.structured_content["server"] == "NeKo"
    assert bounded.structured_content["max_states"] == 2
    assert bounded.structured_content["pruned_state_ids"] == [2, 6]
    assert bounded.structured_content["retained_state_ids"] == [0, 4]
    session = session_manager.get_session(session_id)
    assert session is not None
    assert session.get_history_max_states() == 2

    unbounded = _run(
        _call_tool(
            "set_network_history_limit",
            {"max_states": None, "session_id": session_id},
        )
    )

    assert unbounded.is_error is False
    assert unbounded.structured_content["max_states"] is None
    assert network.max_history_calls == [2, None]
    assert session.get_history_max_states() is None


def test_set_network_history_limit_can_be_configured_before_network() -> None:
    session_id = _create_session()

    result = _run(
        _call_tool(
            "set_network_history_limit",
            {"max_states": 20, "session_id": session_id},
        )
    )

    assert result.is_error is False
    assert result.structured_content["applies_to_current_network"] is False
    session = session_manager.get_session(session_id)
    assert session is not None
    assert session.get_history_max_states() == 20


def test_network_history_resource_returns_inline_svg_html() -> None:
    session_id = _create_session(_HistoryNetwork())

    result = _run(
        _read_resource(f"neko://session/{session_id}/history")
    )

    assert len(result.contents) == 1
    assert result.contents[0].mime_type == "text/html"
    assert result.contents[0].text == (
        '<div class="neko-history-graph"><svg>history</svg></div>'
    )


def test_missing_network_history_resource_does_not_create_session() -> None:
    assert session_manager.list_sessions() == {}

    error = _run(
        _read_resource_error(
            "neko://session/missing-session/history"
        )
    )

    assert "NeKo session not found: missing-session" in str(error)
    assert session_manager.list_sessions() == {}


def test_network_history_resource_requires_network() -> None:
    session_id = _create_session()

    error = _run(
        _read_resource_error(
            f"neko://session/{session_id}/history"
        )
    )

    assert "No network in this session" in str(error)


def test_session_locking_preserves_public_tool_schemas() -> None:
    listed_tools = _run(_list_tools())
    tools = {tool.name: tool for tool in listed_tools.tools}
    session_backed_tools = {
        "create_network",
        "add_nodes",
        "remove_gene",
        "remove_interaction",
        "export_network",
        "export_neko_handoff",
        "list_genes_and_interactions",
        "find_paths",
        "reset_network",
        "list_network_history",
        "navigate_network_history",
        "compare_network_states",
        "set_network_history_limit",
        "clean_generated_files",
        "remove_bimodal_interactions",
        "remove_undefined_interactions",
        "list_bnet_files",
        "analyze_connectivity",
        "analyze_gene_set",
        "get_references",
        "set_default_params",
        "filter_interactions",
        "status",
        "preview_connection_impact",
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

    navigation_schema = tools[
        "navigate_network_history"
    ].input_schema["properties"]
    assert navigation_schema["action"]["enum"] == [
        "undo",
        "redo",
        "checkout",
    ]

    retention_schema = tools[
        "set_network_history_limit"
    ].input_schema["properties"]["max_states"]
    integer_schema = next(
        option
        for option in retention_schema["anyOf"]
        if option.get("type") == "integer"
    )
    assert integer_schema["minimum"] == 2

    for tool_name in {
        "list_network_history",
        "navigate_network_history",
        "compare_network_states",
        "set_network_history_limit",
    }:
        assert tools[tool_name].output_schema is not None


def test_all_neko_tools_publish_safety_annotations() -> None:
    listed_tools = _run(_list_tools())
    tools = {tool.name: tool for tool in listed_tools.tools}

    read_only_closed = {
        "list_genes_and_interactions",
        "find_paths",
        "list_network_history",
        "compare_network_states",
        "list_bnet_files",
        "analyze_connectivity",
        "analyze_gene_set",
        "get_references",
        "filter_interactions",
        "list_sessions",
        "list_artifact_sessions",
        "status",
    }
    read_only_open = {"preview_connection_impact"}
    idempotent_closed = {
        "export_network",
        "set_default_params",
        "set_default_session",
    }
    non_idempotent_closed = {
        "create_session",
        "export_neko_handoff",
        "navigate_network_history",
    }
    non_idempotent_open = {
        "add_nodes",
        "bridge_components",
        "connect_targeted_nodes",
        "apply_global_connection",
    }
    destructive_idempotent = {
        "reset_network",
        "set_network_history_limit",
        "clean_generated_files",
        "remove_bimodal_interactions",
        "remove_undefined_interactions",
    }
    destructive_non_idempotent = {
        "remove_gene",
        "remove_interaction",
        "delete_session",
    }
    destructive_open = {"create_network"}

    assert set(tools) == (
        read_only_closed
        | read_only_open
        | idempotent_closed
        | non_idempotent_closed
        | non_idempotent_open
        | destructive_idempotent
        | destructive_non_idempotent
        | destructive_open
    )

    read_only = read_only_closed | read_only_open
    destructive = (
        destructive_idempotent
        | destructive_non_idempotent
        | destructive_open
    )
    idempotent = read_only | idempotent_closed | destructive_idempotent
    open_world = read_only_open | non_idempotent_open | destructive_open

    for tool_name, tool in tools.items():
        assert tool.title is not None
        assert tool.output_schema is not None
        annotations = tool.annotations
        assert annotations is not None
        assert annotations.read_only_hint is (tool_name in read_only)
        assert annotations.destructive_hint is (tool_name in destructive)
        assert annotations.idempotent_hint is (tool_name in idempotent)
        assert annotations.open_world_hint is (tool_name in open_world)


def test_neko_tool_schemas_publish_stable_enums_and_bounds() -> None:
    listed_tools = _run(_list_tools())
    tools = {tool.name: tool for tool in listed_tools.tools}

    create_schema = tools["create_network"].input_schema["properties"]
    assert create_schema["database"]["enum"] == ["omnipath", "signor"]
    assert create_schema["path_policy"]["enum"] == [
        "one_shortest",
        "all_shortest",
        "all_bounded",
    ]
    assert create_schema["reuse_policy"]["enum"] == [
        "none",
        "discovered_paths",
        "induced_subgraph",
    ]
    assert "algorithm" not in create_schema
    assert "connect_with_bias" not in create_schema
    assert create_schema["verbosity"]["enum"] == [
        "summary",
        "preview",
        "full",
    ]
    assert create_schema["max_len"]["minimum"] == 1
    assert create_schema["max_len"]["maximum"] == 4
    assert create_schema["list_of_initial_genes"]["items"]["minLength"] == 1

    export_schema = tools["export_network"].input_schema["properties"]
    assert export_schema["format"]["enum"] == ["sif", "bnet"]

    handoff_schema = tools["export_neko_handoff"].input_schema
    assert handoff_schema["required"] == ["biological_context"]
    handoff_properties = handoff_schema["properties"]
    assert handoff_properties["artifact_prefix"]["minLength"] == 1
    assert handoff_properties["artifact_prefix"]["maxLength"] == 128
    assert handoff_properties["artifact_prefix"]["pattern"].startswith("^")
    assert (
        handoff_properties["biological_context"]["minLength"] == 1
    )

    path_schema = tools["find_paths"].input_schema["properties"]["maxlen"]
    assert path_schema["minimum"] == 1
    assert path_schema["maximum"] == 5

    parameter_schema = tools["set_default_params"].input_schema["properties"]
    max_len_schema = next(
        option
        for option in parameter_schema["max_len"]["anyOf"]
        if option.get("type") == "integer"
    )
    assert max_len_schema["minimum"] == 1
    assert max_len_schema["maximum"] == 4
    path_policy_schema = next(
        option
        for option in parameter_schema["path_policy"]["anyOf"]
        if option.get("type") == "string"
    )
    assert path_policy_schema["enum"] == [
        "one_shortest",
        "all_shortest",
        "all_bounded",
    ]
    reuse_policy_schema = next(
        option
        for option in parameter_schema["reuse_policy"]["anyOf"]
        if option.get("type") == "string"
    )
    assert reuse_policy_schema["enum"] == [
        "none",
        "discovered_paths",
        "induced_subgraph",
    ]
    assert "algorithm" not in parameter_schema
    assert "connect_with_bias" not in parameter_schema

    filter_schema = tools["filter_interactions"].input_schema["properties"]
    assert filter_schema["format"]["enum"] == ["markdown", "json"]
    assert filter_schema["max_rows"]["minimum"] == 1
    assert filter_schema["nodes"]["anyOf"][0]["minItems"] == 1
    assert filter_schema["node_scope"]["enum"] == [
        "incident",
        "internal",
        "boundary",
    ]

    gene_set_schema = tools["analyze_gene_set"].input_schema["properties"]
    assert gene_set_schema["genes"]["minItems"] == 1
    assert gene_set_schema["connectivity"]["enum"] == ["weak", "strong"]
    assert gene_set_schema["max_rows"]["minimum"] == 1

    candidate_schema = tools["preview_connection_impact"].input_schema["properties"]
    assert candidate_schema["method"]["enum"] == [
        "hubs",
        "relax_max_len",
        "unsigned",
    ]
    assert candidate_schema["top_k"]["minimum"] == 1

    bridge_schema = tools["bridge_components"].input_schema["properties"]
    assert bridge_schema["mode"]["enum"] == ["OUT", "IN", "ALL"]
    assert bridge_schema["comp_a"]["minItems"] == 1
    assert bridge_schema["comp_b"]["minItems"] == 1
    assert bridge_schema["max_len"]["minimum"] == 1

    targeted_schema = tools["connect_targeted_nodes"].input_schema["properties"]
    assert targeted_schema["strategy"]["enum"] == [
        "connect_to_upstream_nodes",
        "connect_subgroup",
    ]
    assert targeted_schema["nodes"]["minItems"] == 1
    assert "strategy_mode" not in targeted_schema
    assert "outputs" not in targeted_schema

    global_schema = tools["apply_global_connection"].input_schema["properties"]
    assert global_schema["strategy"]["enum"] == [
        "complete_connection",
        "connect_network_radially",
        "connect_as_atopo",
    ]
    global_path_policy = next(
        option
        for option in global_schema["path_policy"]["anyOf"]
        if option.get("type") == "string"
    )
    assert global_path_policy["enum"] == [
        "one_shortest",
        "all_shortest",
        "all_bounded",
    ]
    global_reuse_policy = next(
        option
        for option in global_schema["reuse_policy"]["anyOf"]
        if option.get("type") == "string"
    )
    assert global_reuse_policy["enum"] == [
        "none",
        "discovered_paths",
        "induced_subgraph",
    ]
    assert "algorithm" not in global_schema
    assert "minimal" not in global_schema
    assert global_schema["direction"]["enum"] == ["OUT", "IN"]
    global_max_len = next(
        option
        for option in global_schema["max_len"]["anyOf"]
        if option.get("type") == "integer"
    )
    assert global_max_len["minimum"] == 1
    atopo_schema = next(
        option
        for option in global_schema["strategy_mode"]["anyOf"]
        if option.get("type") == "string"
    )
    assert atopo_schema["enum"] == ["radial", "complete"]
    assert global_schema["outputs"]["anyOf"][0]["minItems"] == 1

    navigation_schema = tools[
        "navigate_network_history"
    ].input_schema["properties"]["state_id"]
    state_id_schema = next(
        option
        for option in navigation_schema["anyOf"]
        if option.get("type") == "integer"
    )
    assert state_id_schema["minimum"] == 0
    comparison_schema = tools["compare_network_states"].input_schema["properties"]
    assert comparison_schema["state_a"]["minimum"] == 0
    assert comparison_schema["state_b"]["minimum"] == 0


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("set_default_session", {"session_id": ""}),
        ("set_default_session", {"session_id": "   "}),
        ("create_network", {"list_of_initial_genes": [""]}),
        ("create_network", {"list_of_initial_genes": ["TP53"], "max_len": 0}),
        ("create_network", {"list_of_initial_genes": ["TP53"], "max_len": 5}),
        (
            "create_network",
            {"list_of_initial_genes": ["TP53"], "path_policy": "astar"},
        ),
        (
            "create_network",
            {"list_of_initial_genes": ["TP53"], "reuse_policy": "all"},
        ),
        ("list_genes_and_interactions", {"max_rows": 0}),
        (
            "find_paths",
            {"source": "TP53", "target": "MDM2", "maxlen": 6},
        ),
        (
            "navigate_network_history",
            {"action": "checkout", "state_id": -1},
        ),
        ("compare_network_states", {"state_a": -1, "state_b": 0}),
        ("add_nodes", {"genes": []}),
        ("set_default_params", {"max_len": 0}),
        ("set_default_params", {"path_policy": "astar"}),
        ("set_default_params", {"reuse_policy": "all"}),
        ("filter_interactions", {"effect": [""]}),
        ("filter_interactions", {"format": "xml"}),
        ("preview_connection_impact", {"top_k": 0}),
        ("bridge_components", {"comp_a": [], "comp_b": ["MDM2"]}),
        (
            "bridge_components",
            {"comp_a": ["TP53"], "comp_b": ["MDM2"], "mode": "SIDEWAYS"},
        ),
        (
            "connect_targeted_nodes",
            {"strategy": "connect_subgroup", "nodes": []},
        ),
        (
            "apply_global_connection",
            {
                "strategy": "connect_as_atopo",
                "outputs": [],
            },
        ),
        (
            "apply_global_connection",
            {
                "strategy": "connect_as_atopo",
                "strategy_mode": "hierarchy",
            },
        ),
        (
            "apply_global_connection",
            {"strategy": "complete_connection", "direction": "SIDEWAYS"},
        ),
        (
            "apply_global_connection",
            {"strategy": "complete_connection", "path_policy": "astar"},
        ),
        (
            "apply_global_connection",
            {"strategy": "complete_connection", "reuse_policy": "all"},
        ),
    ],
)
def test_invalid_neko_inputs_are_rejected_before_execution(
    tool_name: str,
    arguments: dict[str, Any],
) -> None:
    result = _run(_call_tool(tool_name, arguments))

    assert result.is_error is True
    assert "validation error" in result.content[0].text.lower()
    assert session_manager.list_sessions() == {}


def test_case_insensitive_public_options_remain_compatible(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class RecordingExports:
        paths: list[str] = []

        def __init__(self, network: object) -> None:
            del network

        def export_sif(self, path: str) -> None:
            self.paths.append(path)

    network = _network_stub(
        nodes=pd.DataFrame(
            [{"Uniprot": "P04637", "Genesymbol": "TP53"}]
        ),
        edges=pd.DataFrame(
            [{"source": "P04637", "target": "P04637", "Effect": "stimulation"}]
        ),
        convert_edgelist_into_genesymbol=lambda: pd.DataFrame(
            [{"source": "TP53", "target": "TP53", "Effect": "stimulation"}]
        ),
    )
    session_id = _create_session(network)
    monkeypatch.setattr(neko_server, "Exports", RecordingExports)
    monkeypatch.setattr(
        neko_server,
        "_export_dir",
        lambda _session_id: tmp_path,
    )

    export_result = _run(
        _call_tool(
            "export_network",
            {
                "format": "SIF",
                "session_id": session_id,
                "verbosity": "SUMMARY",
            },
        )
    )
    with pd.option_context("future.infer_string", True):
        listing_result = _run(
            _call_tool(
                "list_genes_and_interactions",
                {"session_id": session_id, "verbosity": "FULL"},
            )
        )
    connector_result = _run(
        _call_tool(
            "preview_connection_impact",
            {
                "session_id": session_id,
                "method": "HUBS",
                "verbosity": "SUMMARY",
            },
        )
    )

    assert export_result.is_error is False
    assert RecordingExports.paths == [str(tmp_path / "Network.sif")]
    assert export_result.structured_content is not None
    assert export_result.structured_content["session_id"] == session_id
    assert export_result.structured_content["format"] == "sif"
    assert export_result.structured_content["renamed_nodes"] == []
    assert export_result.structured_content["duplicate_rules_removed"] == []
    assert listing_result.is_error is False
    assert listing_result.structured_content is not None
    assert listing_result.structured_content["nodes"] == [
        {
            "gene_symbol": "TP53",
            "uniprot": "P04637",
            "node_type": None,
        }
    ]
    assert connector_result.is_error is False


def test_list_bnet_files_does_not_create_artifact_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    session_id = _create_session()
    monkeypatch.setattr(neko_server, "_SERVER_ROOT", tmp_path)

    result = _run(
        _call_tool("list_bnet_files", {"session_id": session_id})
    )

    assert result.is_error is False
    assert "No .bnet files found" in result.content[0].text
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize(
    "handler_name",
    [
        "add_nodes",
        "remove_gene",
        "remove_interaction",
        "export_network",
        "export_neko_handoff",
        "list_genes_and_interactions",
        "find_paths",
        "reset_network",
        "list_network_history",
        "navigate_network_history",
        "compare_network_states",
        "set_network_history_limit",
        "clean_generated_files",
        "remove_bimodal_interactions",
        "remove_undefined_interactions",
        "list_bnet_files",
        "analyze_connectivity",
        "analyze_gene_set",
        "get_references",
        "set_default_params",
        "filter_interactions",
        "status",
        "preview_connection_impact",
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
                    "add_nodes",
                    {
                        "session_id": session_id,
                        "genes": ["EGFR"],
                        "autoconnect": False,
                    },
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
    completion_calls: list[dict[str, Any]] = []

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
            completion_calls.append(kwargs)

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
    assert completion_calls == [
        {
            "maxlen": 2,
            "path_policy": "one_shortest",
            "reuse_policy": "discovered_paths",
            "only_signed": True,
            "consensus": True,
        }
    ]


def test_create_network_applies_session_history_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_limits: list[int | None] = []

    class LimitedNetwork:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            self.nodes = pd.DataFrame(
                [{"Uniprot": "P04637", "Genesymbol": "TP53"}]
            )
            self.edges = pd.DataFrame(
                [{"source": "P04637", "target": "Q00987"}]
            )

        def complete_connection(self, **kwargs: Any) -> None:
            del kwargs

        def set_max_history(self, max_states: int | None) -> None:
            configured_limits.append(max_states)

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

    monkeypatch.setattr(neko_server, "Network", LimitedNetwork)
    session_id = _create_session()
    configured = _run(
        _call_tool(
            "set_network_history_limit",
            {"max_states": 20, "session_id": session_id},
        )
    )
    assert configured.is_error is False

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
    assert configured_limits == [20]


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
            {
                "comp_a": ["TP53"],
                "comp_b": ["MDM2"],
                "mode": "ALL",
            },
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
            "apply_global_connection",
            {
                "strategy": "connect_as_atopo",
                "strategy_mode": "radial",
            },
            "connect_as_atopo",
        ),
        (
            "apply_global_connection",
            {"strategy": "complete_connection"},
            "complete_connection",
        ),
        (
            "apply_global_connection",
            {
                "strategy": "connect_network_radially",
                "direction": "IN",
            },
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


def test_global_completion_uses_session_policies_and_explicit_overrides() -> None:
    calls: list[dict[str, Any]] = []

    def complete_connection(**kwargs: Any) -> None:
        calls.append(kwargs)

    network = _network_stub(
        convert_edgelist_into_genesymbol=lambda: pd.DataFrame(
            columns=["source", "target", "Effect"]
        ),
        complete_connection=complete_connection,
    )
    session_id = _create_session(network)

    configured = _run(
        _call_tool(
            "set_default_params",
            {
                "session_id": session_id,
                "max_len": 4,
                "path_policy": "all_shortest",
                "reuse_policy": "none",
                "only_signed": False,
                "consensus": False,
            },
        )
    )
    assert configured.is_error is False

    inherited = _run(
        _call_tool(
            "apply_global_connection",
            {
                "session_id": session_id,
                "strategy": "complete_connection",
            },
        )
    )
    overridden = _run(
        _call_tool(
            "apply_global_connection",
            {
                "session_id": session_id,
                "strategy": "complete_connection",
                "max_len": 3,
                "path_policy": "all_bounded",
                "reuse_policy": "induced_subgraph",
                "only_signed": True,
                "consensus": True,
            },
        )
    )

    assert inherited.is_error is False
    assert overridden.is_error is False
    assert calls == [
        {
            "maxlen": 4,
            "path_policy": "all_shortest",
            "reuse_policy": "none",
            "only_signed": False,
            "consensus": False,
        },
        {
            "maxlen": 3,
            "path_policy": "all_bounded",
            "reuse_policy": "induced_subgraph",
            "only_signed": True,
            "consensus": True,
        },
    ]
