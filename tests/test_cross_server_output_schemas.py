"""Cross-server regression audit for MCP tool input and output contracts."""

import asyncio
import re
from collections import Counter
from typing import Any

from mcp import Client

EXPECTED_NAMED_OUTPUTS = {
    "MaBoSS": {
        "list_sessions": "MaBoSSSessionListResult",
        "list_artifact_sessions": "MaBoSSArtifactSessionListResult",
        "import_neko_handoff": "MaBoSSHandoffImportResult",
        "bnet_to_bnd_and_cfg": "MaBoSSBnetConversionResult",
        "run_simulation": "MaBoSSSimulationRunResult",
        "export_maboss_bnd_cfg": "MaBoSSModelExportResult",
        "export_maboss_handoff": "MaBoSSHandoffExportResult",
        "get_maboss_nodes": "MaBoSSNodeListResult",
        "get_maboss_initial_state": "MaBoSSInitialStateResult",
        "get_maboss_logical_rules": "MaBoSSLogicalRulesResult",
        "get_maboss_mutations": "MaBoSSMutationListResult",
        "update_maboss_parameters": "MaBoSSParameterResult",
        "simulate_mutation": "MaBoSSMutationSimulationResult",
        "visualize_network_trajectories": "MaBoSSTrajectoryPlotResult",
        "get_simulation_result": "MaBoSSSimulationResult",
        "list_generated_files": "MaBoSSArtifactFileListResult",
        "clean_generated_files": "MaBoSSArtifactCleanupResult",
    },
    "NeKo": {
        "export_network": "NeKoNetworkExportResult",
        "export_neko_handoff": "NeKoHandoffExportResult",
        "list_genes_and_interactions": "NeKoNetworkInventoryResult",
        "find_paths": "NeKoPathSearchResult",
        "list_network_history": "NetworkHistorySummary",
        "navigate_network_history": "HistoryNavigationResult",
        "compare_network_states": "NetworkStateComparison",
        "set_network_history_limit": "HistoryRetentionResult",
        "clean_generated_files": "NeKoArtifactCleanupResult",
        "list_bnet_files": "NeKoArtifactFileListResult",
        "analyze_connectivity": "NeKoConnectivityResult",
        "get_references": "NeKoReferenceQueryResult",
        "filter_interactions": "NeKoInteractionFilterResult",
        "list_sessions": "NeKoSessionListResult",
        "list_artifact_sessions": "NeKoArtifactSessionListResult",
        "status": "NeKoNetworkStatusResult",
        "preview_connection_impact": "NeKoConnectionPreviewResult",
    },
    "PhysiCell": {
        "list_sessions": "PhysiCellSessionListResult",
        "list_artifact_sessions": "PhysiCellArtifactSessionListResult",
        "import_maboss_handoff": "PhysiCellHandoffImportResult",
        "get_workflow_status": "PhysiCellWorkflowStatusResult",
        "get_maboss_context": "PhysiCellMaBoSSContextResult",
        "validate_xml_file": "PhysiCellXmlValidationResult",
        "analyze_loaded_configuration": (
            "PhysiCellLoadedConfigurationResult"
        ),
        "list_loaded_components": "PhysiCellLoadedComponentsResult",
        "get_available_cycle_models": "PhysiCellCycleModelListResult",
        "list_all_available_signals": "PhysiCellSignalListResult",
        "list_all_available_behaviors": "PhysiCellBehaviorListResult",
        "get_simulation_summary": "PhysiCellWorkflowStatusResult",
        "export_xml_configuration": "PhysiCellXmlExportResult",
        "export_cell_rules_csv": "PhysiCellRulesExportResult",
        "list_generated_files": "PhysiCellArtifactFileListResult",
        "clean_generated_files": "PhysiCellArtifactCleanupResult",
    },
}

EXPECTED_TOOL_COUNTS = {
    "MaBoSS": 24,
    "NeKo": 31,
    "PhysiCell": 34,
}
TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
INTERNAL_PARAMETER_NAMES = {
    "config",
    "context",
    "ctx",
    "network",
    "sess",
    "session",
}


async def _list_all_tools() -> dict[str, dict[str, Any]]:
    # Import all three source servers into this process before inspecting any
    # schemas. This catches collisions between launcher-compatible bare module
    # names in addition to auditing the individual tool surfaces.
    from tests import test_maboss_mcp_errors as maboss_tests
    from tests import test_neko_mcp_errors as neko_tests
    from tests import test_physicell_mcp_errors as physicell_tests

    servers = {
        "MaBoSS": maboss_tests.mcp,
        "NeKo": neko_tests.mcp,
        "PhysiCell": physicell_tests.mcp,
    }
    tool_maps = {}
    for server_name, server in servers.items():
        async with Client(server) as client:
            listed = await client.list_tools()
        tool_maps[server_name] = {
            tool.name: tool for tool in listed.tools
        }
    return tool_maps


def test_all_servers_publish_complete_input_and_annotation_contracts() -> None:
    tool_maps = asyncio.run(_list_all_tools())

    assert sum(len(tools) for tools in tool_maps.values()) == 89
    for server_name, tools in tool_maps.items():
        assert len(tools) == EXPECTED_TOOL_COUNTS[server_name]
        for tool_name, tool in tools.items():
            assert TOOL_NAME_PATTERN.fullmatch(tool_name)
            assert tool.description is not None
            assert tool.description.strip()

            schema = tool.input_schema
            assert schema["type"] == "object"
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            assert isinstance(properties, dict)
            assert isinstance(required, list)
            assert len(required) == len(set(required))
            assert set(required) <= set(properties)
            assert not (set(properties) & INTERNAL_PARAMETER_NAMES)

            for parameter_name, parameter_schema in properties.items():
                assert isinstance(parameter_schema, dict), (
                    f"{server_name}.{tool_name}.{parameter_name}"
                )
                description = parameter_schema.get("description")
                assert isinstance(description, str), (
                    f"{server_name}.{tool_name}.{parameter_name}"
                )
                assert description.strip(), (
                    f"{server_name}.{tool_name}.{parameter_name}"
                )

            annotations = tool.annotations
            assert annotations is not None
            for hint_name in (
                "read_only_hint",
                "destructive_hint",
                "idempotent_hint",
                "open_world_hint",
            ):
                assert isinstance(getattr(annotations, hint_name), bool)
            if annotations.read_only_hint:
                assert annotations.destructive_hint is False
                assert annotations.idempotent_hint is True


def test_all_servers_publish_expected_strict_output_contracts() -> None:
    tool_maps = asyncio.run(_list_all_tools())

    for server_name, tools in tool_maps.items():
        assert len(tools) == EXPECTED_TOOL_COUNTS[server_name]
        expected_named = EXPECTED_NAMED_OUTPUTS[server_name]

        for tool_name, tool in tools.items():
            schema = tool.output_schema
            assert schema is not None
            assert schema["type"] == "object"
            properties = schema.get("properties", {})

            if tool_name in expected_named:
                assert schema["title"] == expected_named[tool_name]
                assert set(properties) != {"result"}
                assert schema["additionalProperties"] is False
                assert properties["server"]["const"] == server_name
            else:
                assert set(properties) == {"result"}
                assert properties["result"]["type"] == "string"


def test_shared_output_contract_fields_are_aligned() -> None:
    tool_maps = asyncio.run(_list_all_tools())
    expected_properties = {
        "list_sessions": {"server", "count", "sessions"},
        "list_artifact_sessions": {"server", "count", "sessions"},
        "file_listing": {"server", "scope", "session_id", "count", "files"},
        "clean_generated_files": {
            "server",
            "session_id",
            "removed_count",
        },
    }
    file_listing_names = {
        "MaBoSS": "list_generated_files",
        "NeKo": "list_bnet_files",
        "PhysiCell": "list_generated_files",
    }

    for server_name, tools in tool_maps.items():
        named_tools = {
            "list_sessions": "list_sessions",
            "list_artifact_sessions": "list_artifact_sessions",
            "file_listing": file_listing_names[server_name],
            "clean_generated_files": "clean_generated_files",
        }
        for contract_name, tool_name in named_tools.items():
            schema = tools[tool_name].output_schema
            assert schema is not None
            assert set(schema["properties"]) == expected_properties[
                contract_name
            ]


def test_schema_titles_are_unique_except_for_intentional_aliases() -> None:
    tool_maps = asyncio.run(_list_all_tools())
    uses_by_title: dict[str, list[str]] = {}

    for server_name, tools in tool_maps.items():
        for tool_name, tool in tools.items():
            schema = tool.output_schema
            assert schema is not None
            uses_by_title.setdefault(schema["title"], []).append(
                f"{server_name}.{tool_name}"
            )

    duplicated_titles = {
        title: sorted(uses)
        for title, uses in uses_by_title.items()
        if len(uses) > 1
    }
    assert Counter(map(len, duplicated_titles.values())) == Counter({3: 3, 2: 1})
    assert duplicated_titles == {
        "PhysiCellWorkflowStatusResult": [
            "PhysiCell.get_simulation_summary",
            "PhysiCell.get_workflow_status",
        ],
        "create_sessionOutput": [
            "MaBoSS.create_session",
            "NeKo.create_session",
            "PhysiCell.create_session",
        ],
        "delete_sessionOutput": [
            "MaBoSS.delete_session",
            "NeKo.delete_session",
            "PhysiCell.delete_session",
        ],
        "set_default_sessionOutput": [
            "MaBoSS.set_default_session",
            "NeKo.set_default_session",
            "PhysiCell.set_default_session",
        ],
    }
