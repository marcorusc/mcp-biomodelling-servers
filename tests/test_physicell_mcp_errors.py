"""Protocol-level tests for PhysiCell tool failure semantics."""

import asyncio
import inspect
import sys
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, local
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from mcp import Client


def _install_physicell_import_stubs() -> None:
    """Provide the PhysiCell-settings import surface needed to load the server."""
    physicell_config = ModuleType("physicell_config")
    physicell_config.__path__ = []  # type: ignore[attr-defined]
    config_package = ModuleType("physicell_config.config")
    config_package.__path__ = []  # type: ignore[attr-defined]

    signals_module = ModuleType(
        "physicell_config.config.embedded_signals_behaviors"
    )
    defaults_module = ModuleType("physicell_config.config.embedded_defaults")

    class StubPhysiCellConfig:
        pass

    def get_signals_behaviors() -> dict[str, dict[str, Any]]:
        return {"signals": {}, "behaviors": {}}

    def no_op(*args: Any, **kwargs: Any) -> None:
        del args, kwargs

    physicell_config.PhysiCellConfig = StubPhysiCellConfig  # type: ignore[attr-defined]
    signals_module.get_signals_behaviors = get_signals_behaviors  # type: ignore[attr-defined]
    signals_module.get_signal_by_name = no_op  # type: ignore[attr-defined]
    signals_module.get_behavior_by_name = no_op  # type: ignore[attr-defined]
    signals_module.update_signals_behaviors_context_from_config = no_op  # type: ignore[attr-defined]
    signals_module.get_expanded_signals = lambda: []  # type: ignore[attr-defined]
    signals_module.get_expanded_behaviors = lambda: []  # type: ignore[attr-defined]
    defaults_module.get_default_parameters = lambda: {  # type: ignore[attr-defined]
        "cell_cycle_models": {}
    }

    physicell_config.config = config_package  # type: ignore[attr-defined]
    config_package.embedded_signals_behaviors = signals_module  # type: ignore[attr-defined]
    config_package.embedded_defaults = defaults_module  # type: ignore[attr-defined]

    sys.modules.update(
        {
            "physicell_config": physicell_config,
            "physicell_config.config": config_package,
            "physicell_config.config.embedded_signals_behaviors": signals_module,
            "physicell_config.config.embedded_defaults": defaults_module,
        }
    )


PHYSICELL_DIR = Path(__file__).parent.parent / "PhysiCell"
sys.path.insert(0, str(PHYSICELL_DIR))
_install_physicell_import_stubs()

# The launchers import their local session manager as the top-level module.
sys.modules.pop("session_manager", None)

from PhysiCell import server as physicell_server  # noqa: E402

mcp = physicell_server.mcp
session_manager = physicell_server.session_manager


def _run(coroutine: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coroutine)


async def _call_tool(name: str, arguments: dict[str, Any] | None = None) -> Any:
    async with Client(mcp) as client:
        return await client.call_tool(name, arguments or {})


async def _list_tools() -> Any:
    async with Client(mcp) as client:
        return await client.list_tools()


def _clear_sessions() -> None:
    for session in list(session_manager.list_sessions()):
        session_manager.delete_session(session.session_id)


def _create_session(config: object | None = None) -> str:
    session_id = session_manager.create_session()
    session = session_manager.get_session(session_id)
    assert session is not None
    session.config = config
    return session_id


def _xml_export_config(generate_calls: list[str]) -> SimpleNamespace:
    def generate_xml() -> str:
        generate_calls.append("generate_xml")
        return "<PhysiCell_settings/>"

    return SimpleNamespace(generate_xml=generate_xml)


def _cell_rules_export_config(
    generate_calls: list[str],
    ruleset_calls: list[dict[str, Any]],
) -> SimpleNamespace:
    def generate_csv(path: str) -> None:
        generate_calls.append(path)
        Path(path).write_text("cell_type,signal,behavior\n", encoding="utf-8")

    def add_ruleset(**kwargs: Any) -> None:
        ruleset_calls.append(kwargs)

    cell_rules = SimpleNamespace(
        get_rules=lambda: [object()],
        generate_csv=generate_csv,
        add_ruleset=add_ruleset,
    )
    return SimpleNamespace(cell_rules=cell_rules)


@pytest.fixture(autouse=True)
def isolated_sessions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(physicell_server, "_SERVER_ROOT", tmp_path)
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


def test_missing_xml_is_tool_error_without_creating_session(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.xml"

    result = _run(
        _call_tool("load_xml_configuration", {"filepath": str(missing_path)})
    )

    assert result.is_error is True
    assert f"PhysiCell XML file not found: {missing_path}" in result.content[0].text
    assert session_manager.list_sessions() == []


def test_invalid_existing_xml_is_successful_validation_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    invalid_path = tmp_path / "invalid.xml"
    invalid_path.write_text("<invalid/>", encoding="utf-8")

    class InvalidConfig:
        def validate_xml_file(self, filepath: str) -> tuple[bool, str]:
            assert filepath == str(invalid_path)
            return False, "missing PhysiCell_settings root"

    monkeypatch.setattr(physicell_server, "PhysiCellConfig", InvalidConfig)

    result = _run(
        _call_tool("validate_xml_file", {"filepath": str(invalid_path)})
    )

    assert result.is_error is False
    assert "Invalid: missing PhysiCell_settings root" in result.content[0].text


def test_analysis_without_loaded_xml_is_tool_error() -> None:
    session_id = _create_session()

    result = _run(
        _call_tool(
            "analyze_loaded_configuration",
            {"session_id": session_id},
        )
    )

    assert result.is_error is True
    assert "No XML configuration loaded" in result.content[0].text


def test_invalid_domain_is_tool_error_without_creating_session() -> None:
    result = _run(
        _call_tool(
            "create_simulation_domain",
            {"domain_x": 0, "domain_y": 100},
        )
    )

    assert result.is_error is True
    assert "validation error" in result.content[0].text.lower()
    assert session_manager.list_sessions() == []


def test_missing_configuration_prerequisite_is_tool_error() -> None:
    session_id = _create_session()

    result = _run(
        _call_tool(
            "add_single_substrate",
            {
                "substrate_name": "oxygen",
                "diffusion_coefficient": 100000,
                "decay_rate": 0.01,
                "initial_condition": 38,
                "session_id": session_id,
            },
        )
    )

    assert result.is_error is True
    assert "No simulation configured" in result.content[0].text


def test_backend_configuration_failure_is_tool_error() -> None:
    def fail_volume_configuration(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise LookupError("unknown cell type")

    config = SimpleNamespace(
        cell_types=SimpleNamespace(
            set_volume_parameters=fail_volume_configuration,
        )
    )
    session_id = _create_session(config)

    result = _run(
        _call_tool(
            "configure_cell_parameters",
            {"cell_type": "missing", "session_id": session_id},
        )
    )

    assert result.is_error is True
    assert "Could not configure cell type 'missing'" in result.content[0].text
    assert "unknown cell type" in result.content[0].text


def test_unavailable_physiboss_support_is_tool_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = _create_session(SimpleNamespace())
    monkeypatch.setattr(physicell_server, "PHYSIBOSS_AVAILABLE", False)

    result = _run(
        _call_tool(
            "add_physiboss_model",
            {
                "cell_type": "tumour",
                "bnd_file": "/model.bnd",
                "cfg_file": "/model.cfg",
                "session_id": session_id,
            },
        )
    )

    assert result.is_error is True
    assert "PhysiBoSS support is not available" in result.content[0].text


def test_export_without_rules_is_tool_error() -> None:
    config = SimpleNamespace(
        cell_rules=SimpleNamespace(get_rules=lambda: []),
    )
    session_id = _create_session(config)

    result = _run(
        _call_tool("export_cell_rules_csv", {"session_id": session_id})
    )

    assert result.is_error is True
    assert "No cell rules are available to export" in result.content[0].text


@pytest.mark.parametrize(
    "filename",
    [
        "../outside.xml",
        "nested/settings.xml",
        r"..\outside.xml",
        "settings.csv",
        " settings.xml ",
        "bad\x00.xml",
    ],
)
def test_unsafe_xml_export_filename_is_tool_error(filename: str) -> None:
    generate_calls: list[str] = []
    session_id = _create_session(_xml_export_config(generate_calls))

    result = _run(
        _call_tool(
            "export_xml_configuration",
            {"filename": filename, "session_id": session_id},
        )
    )

    assert result.is_error is True
    assert "Export filename" in result.content[0].text
    assert generate_calls == []


def test_absolute_xml_export_path_cannot_escape_artifacts(tmp_path: Path) -> None:
    generate_calls: list[str] = []
    session_id = _create_session(_xml_export_config(generate_calls))
    outside_path = tmp_path / "outside.xml"

    result = _run(
        _call_tool(
            "export_xml_configuration",
            {"filename": str(outside_path), "session_id": session_id},
        )
    )

    assert result.is_error is True
    assert "without directory components" in result.content[0].text
    assert generate_calls == []
    assert not outside_path.exists()


@pytest.mark.parametrize(
    "filename",
    [
        "../outside.csv",
        "nested/rules.csv",
        r"..\outside.csv",
        "rules.xml",
    ],
)
def test_unsafe_csv_export_filename_is_tool_error(filename: str) -> None:
    generate_calls: list[str] = []
    ruleset_calls: list[dict[str, Any]] = []
    config = _cell_rules_export_config(generate_calls, ruleset_calls)
    session_id = _create_session(config)

    result = _run(
        _call_tool(
            "export_cell_rules_csv",
            {"filename": filename, "session_id": session_id},
        )
    )

    assert result.is_error is True
    assert "Export filename" in result.content[0].text
    assert generate_calls == []
    assert ruleset_calls == []


def test_absolute_csv_export_path_cannot_escape_artifacts(tmp_path: Path) -> None:
    generate_calls: list[str] = []
    ruleset_calls: list[dict[str, Any]] = []
    config = _cell_rules_export_config(generate_calls, ruleset_calls)
    session_id = _create_session(config)
    outside_path = tmp_path / "outside.csv"

    result = _run(
        _call_tool(
            "export_cell_rules_csv",
            {"filename": str(outside_path), "session_id": session_id},
        )
    )

    assert result.is_error is True
    assert "without directory components" in result.content[0].text
    assert generate_calls == []
    assert ruleset_calls == []
    assert not outside_path.exists()


@pytest.mark.parametrize(
    ("filename", "expected_name"),
    [
        (None, "PhysiCell_settings.xml"),
        ("custom-settings.xml", "custom-settings.xml"),
        ("CUSTOM.XML", "CUSTOM.XML"),
    ],
)
def test_valid_xml_export_stays_in_session_artifacts(
    tmp_path: Path,
    filename: str | None,
    expected_name: str,
) -> None:
    generate_calls: list[str] = []
    session_id = _create_session(_xml_export_config(generate_calls))
    arguments = {"session_id": session_id}
    if filename is not None:
        arguments["filename"] = filename

    result = _run(_call_tool("export_xml_configuration", arguments))

    output_path = tmp_path / "artifacts" / session_id / expected_name
    assert result.is_error is False
    assert generate_calls == ["generate_xml"]
    assert output_path.read_text(encoding="utf-8") == "<PhysiCell_settings/>"
    assert str(output_path) in result.content[0].text


@pytest.mark.parametrize(
    ("filename", "expected_name"),
    [
        (None, "cell_rules.csv"),
        ("custom-rules.csv", "custom-rules.csv"),
        ("RULES.CSV", "RULES.CSV"),
    ],
)
def test_valid_csv_export_stays_in_session_artifacts(
    tmp_path: Path,
    filename: str | None,
    expected_name: str,
) -> None:
    generate_calls: list[str] = []
    ruleset_calls: list[dict[str, Any]] = []
    config = _cell_rules_export_config(generate_calls, ruleset_calls)
    session_id = _create_session(config)
    arguments = {"session_id": session_id}
    if filename is not None:
        arguments["filename"] = filename

    result = _run(_call_tool("export_cell_rules_csv", arguments))

    output_path = tmp_path / "artifacts" / session_id / expected_name
    assert result.is_error is False
    assert generate_calls == [str(output_path)]
    assert output_path.read_text(encoding="utf-8").startswith("cell_type")
    assert ruleset_calls == [
        {
            "name": "default",
            "folder": str(output_path),
            "filename": expected_name,
            "enabled": True,
        }
    ]
    assert str(output_path) in result.content[0].text


def test_empty_status_results_remain_successful() -> None:
    summary = _run(_call_tool("get_simulation_summary"))

    assert summary.is_error is False
    assert "No active session" in summary.content[0].text

    session_id = _create_session()
    files = _run(
        _call_tool("list_generated_files", {"session_id": session_id})
    )
    context = _run(_call_tool("get_maboss_context", {"session_id": session_id}))

    assert files.is_error is False
    assert "No PhysiCell artifact files found" in files.content[0].text
    assert context.is_error is False
    assert "No MaBoSS context available" in context.content[0].text


@pytest.mark.parametrize(
    "handler_name",
    [
        "load_xml_configuration",
        "analyze_biological_scenario",
        "create_simulation_domain",
        "add_single_substrate",
        "add_single_cell_type",
        "configure_cell_parameters",
        "set_substrate_interaction",
        "list_all_available_signals",
        "list_all_available_behaviors",
        "add_single_cell_rule",
        "add_physiboss_model",
        "configure_physiboss_settings",
        "add_physiboss_input_link",
        "add_physiboss_output_link",
        "apply_physiboss_mutation",
        "get_simulation_summary",
        "export_xml_configuration",
        "export_cell_rules_csv",
        "list_generated_files",
        "clean_generated_files",
    ],
)
def test_blocking_handlers_remain_synchronous(handler_name: str) -> None:
    handler = getattr(physicell_server, handler_name)

    assert inspect.iscoroutinefunction(handler) is False


def test_session_locking_preserves_public_tool_schemas() -> None:
    listed_tools = _run(_list_tools())
    tools = {tool.name: tool for tool in listed_tools.tools}
    session_backed_tools = {
        "set_default_session",
        "get_workflow_status",
        "set_maboss_context",
        "get_maboss_context",
        "load_xml_configuration",
        "analyze_loaded_configuration",
        "list_loaded_components",
        "analyze_biological_scenario",
        "create_simulation_domain",
        "add_single_substrate",
        "add_single_cell_type",
        "configure_cell_parameters",
        "set_substrate_interaction",
        "list_all_available_signals",
        "list_all_available_behaviors",
        "add_single_cell_rule",
        "add_physiboss_model",
        "configure_physiboss_settings",
        "add_physiboss_input_link",
        "add_physiboss_output_link",
        "apply_physiboss_mutation",
        "get_simulation_summary",
        "export_xml_configuration",
        "export_cell_rules_csv",
        "list_generated_files",
        "clean_generated_files",
    }

    for tool_name in session_backed_tools:
        properties = tools[tool_name].input_schema["properties"]
        assert "session" not in properties
        assert "ctx" not in properties
        assert "session_id" in properties


def test_all_physicell_tools_publish_safety_annotations() -> None:
    listed_tools = _run(_list_tools())
    tools = {tool.name: tool for tool in listed_tools.tools}

    read_only = {
        "list_sessions",
        "list_artifact_sessions",
        "get_workflow_status",
        "get_maboss_context",
        "validate_xml_file",
        "list_loaded_components",
        "get_available_cycle_models",
        "list_all_available_signals",
        "list_all_available_behaviors",
        "get_simulation_summary",
        "list_generated_files",
        "get_help",
    }
    idempotent = {
        "set_default_session",
        "set_maboss_context",
        "analyze_loaded_configuration",
        "analyze_biological_scenario",
        "configure_cell_parameters",
        "set_substrate_interaction",
        "configure_physiboss_settings",
        "export_xml_configuration",
        "export_cell_rules_csv",
    }
    non_idempotent = {
        "create_session",
        "add_single_cell_rule",
        "add_physiboss_input_link",
        "add_physiboss_output_link",
        "apply_physiboss_mutation",
    }
    destructive = {
        "delete_session",
        "add_single_substrate",
        "add_single_cell_type",
        "add_physiboss_model",
    }
    idempotent_destructive = {
        "load_xml_configuration",
        "create_simulation_domain",
        "clean_generated_files",
    }

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
        assert tool.output_schema is not None


def test_physicell_tool_schemas_publish_stable_enums_and_bounds() -> None:
    listed_tools = _run(_list_tools())
    tools = {tool.name: tool for tool in listed_tools.tools}

    component_schema = tools[
        "list_loaded_components"
    ].input_schema["properties"]["component_type"]
    assert component_schema["enum"] == [
        "substrates",
        "cell_types",
        "physiboss",
        "all",
    ]

    domain_schema = tools[
        "create_simulation_domain"
    ].input_schema["properties"]
    assert domain_schema["domain_x"]["exclusiveMinimum"] == 0
    assert domain_schema["domain_y"]["exclusiveMinimum"] == 0
    assert domain_schema["domain_z"]["anyOf"][0]["exclusiveMinimum"] == 0
    assert domain_schema["dx"]["exclusiveMinimum"] == 0
    assert domain_schema["max_time"]["exclusiveMinimum"] == 0

    substrate_schema = tools[
        "add_single_substrate"
    ].input_schema["properties"]
    assert substrate_schema["substrate_name"]["minLength"] == 1
    assert substrate_schema["diffusion_coefficient"]["minimum"] == 0
    assert substrate_schema["decay_rate"]["minimum"] == 0
    assert substrate_schema["units"]["minLength"] == 1

    cell_schema = tools[
        "configure_cell_parameters"
    ].input_schema["properties"]
    assert cell_schema["cell_type"]["minLength"] == 1
    assert cell_schema["volume_total"]["exclusiveMinimum"] == 0
    assert cell_schema["volume_nuclear"]["exclusiveMinimum"] == 0
    assert cell_schema["fluid_fraction"]["minimum"] == 0
    assert cell_schema["fluid_fraction"]["maximum"] == 1
    assert cell_schema["motility_speed"]["minimum"] == 0
    assert cell_schema["persistence_time"]["minimum"] == 0
    assert cell_schema["apoptosis_rate"]["minimum"] == 0
    assert cell_schema["necrosis_rate"]["minimum"] == 0

    interaction_schema = tools[
        "set_substrate_interaction"
    ].input_schema["properties"]
    assert interaction_schema["secretion_rate"]["minimum"] == 0
    assert interaction_schema["uptake_rate"]["minimum"] == 0

    rule_schema = tools[
        "add_single_cell_rule"
    ].input_schema["properties"]
    assert rule_schema["direction"]["enum"] == [
        "increases",
        "decreases",
    ]
    assert rule_schema["signal"]["minLength"] == 1
    assert rule_schema["behavior"]["minLength"] == 1
    assert rule_schema["half_max"]["exclusiveMinimum"] == 0
    assert rule_schema["hill_power"]["exclusiveMinimum"] == 0

    settings_schema = tools[
        "configure_physiboss_settings"
    ].input_schema["properties"]
    assert settings_schema["intracellular_dt"]["exclusiveMinimum"] == 0
    assert settings_schema["time_stochasticity"]["minimum"] == 0
    assert settings_schema["scaling"]["exclusiveMinimum"] == 0
    assert settings_schema["start_time"]["minimum"] == 0

    input_schema = tools[
        "add_physiboss_input_link"
    ].input_schema["properties"]
    assert input_schema["action"]["enum"] == [
        "activation",
        "inhibition",
    ]
    assert input_schema["smoothing"]["minimum"] == 0

    output_schema = tools[
        "add_physiboss_output_link"
    ].input_schema["properties"]
    assert output_schema["action"]["enum"] == [
        "activation",
        "inhibition",
    ]
    assert output_schema["smoothing"]["minimum"] == 0

    mutation_schema = tools[
        "apply_physiboss_mutation"
    ].input_schema["properties"]
    assert mutation_schema["fixed_value"]["enum"] == [0, 1]
    assert mutation_schema["node_name"]["minLength"] == 1

    load_schema = tools[
        "load_xml_configuration"
    ].input_schema["properties"]
    assert load_schema["filepath"]["minLength"] == 1


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("set_default_session", {"session_id": ""}),
        ("load_xml_configuration", {"filepath": "   "}),
        (
            "create_simulation_domain",
            {"domain_x": 0, "domain_y": 100},
        ),
        (
            "add_single_substrate",
            {
                "substrate_name": "",
                "diffusion_coefficient": 1,
                "decay_rate": 0,
                "initial_condition": 0,
            },
        ),
        (
            "add_single_substrate",
            {
                "substrate_name": "oxygen",
                "diffusion_coefficient": -1,
                "decay_rate": 0,
                "initial_condition": 0,
            },
        ),
        ("add_single_cell_type", {"cell_type_name": "   "}),
        (
            "configure_cell_parameters",
            {"cell_type": "tumour", "volume_total": 0},
        ),
        (
            "set_substrate_interaction",
            {
                "cell_type": "tumour",
                "substrate": "oxygen",
                "secretion_rate": -1,
            },
        ),
        ("list_loaded_components", {"component_type": "unknown"}),
        (
            "add_single_cell_rule",
            {
                "cell_type": "tumour",
                "signal": "oxygen",
                "direction": "promotes",
                "behavior": "cycle entry",
            },
        ),
        (
            "add_single_cell_rule",
            {
                "cell_type": "tumour",
                "signal": "oxygen",
                "direction": "increases",
                "behavior": "cycle entry",
                "half_max": 0,
            },
        ),
        (
            "configure_physiboss_settings",
            {"cell_type": "tumour", "intracellular_dt": 0},
        ),
        (
            "add_physiboss_input_link",
            {
                "cell_type": "tumour",
                "physicell_signal": "oxygen",
                "boolean_node": "HIF1",
                "action": "invalid",
            },
        ),
        (
            "add_physiboss_output_link",
            {
                "cell_type": "tumour",
                "boolean_node": "Apoptosis",
                "physicell_behavior": "apoptosis",
                "smoothing": -1,
            },
        ),
        (
            "apply_physiboss_mutation",
            {
                "cell_type": "tumour",
                "node_name": "TP53",
                "fixed_value": 2,
            },
        ),
        ("export_xml_configuration", {"filename": ""}),
    ],
)
def test_invalid_common_inputs_are_rejected_before_execution(
    tool_name: str,
    arguments: dict[str, Any],
) -> None:
    result = _run(_call_tool(tool_name, arguments))

    assert result.is_error is True
    assert "validation error" in result.content[0].text.lower()


def test_workflow_status_accepts_session_and_matches_summary() -> None:
    config = SimpleNamespace(
        substrates=SimpleNamespace(get_substrates=lambda: {}),
        cell_types=SimpleNamespace(get_cell_types=lambda: {}),
        cell_rules=SimpleNamespace(get_rules=lambda: []),
    )
    session_id = _create_session(config)

    summary = _run(
        _call_tool("get_simulation_summary", {"session_id": session_id})
    )
    status = _run(
        _call_tool("get_workflow_status", {"session_id": session_id})
    )

    assert summary.is_error is False
    assert status.is_error is False
    assert summary.content[0].text == status.content[0].text


def test_failed_domain_rebuild_preserves_existing_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing_config = object()
    session_id = _create_session(existing_config)

    class FailingDomain:
        def set_bounds(self, *args: Any) -> None:
            del args
            raise RuntimeError("domain backend failed")

    candidate = SimpleNamespace(domain=FailingDomain())
    monkeypatch.setattr(
        physicell_server,
        "PhysiCellConfig",
        lambda: candidate,
    )

    result = _run(
        _call_tool(
            "create_simulation_domain",
            {
                "domain_x": 100,
                "domain_y": 100,
                "session_id": session_id,
            },
        )
    )

    session = session_manager.get_session(session_id)
    assert result.is_error is True
    assert "domain backend failed" in result.content[0].text
    assert session is not None
    assert session.config is existing_config


def test_same_session_summary_waits_for_signal_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = _create_session(SimpleNamespace())
    session = session_manager.get_session(session_id)
    assert session is not None
    expansion_entered = Event()
    release_expansion = Event()

    def blocking_update(config: object) -> None:
        del config
        expansion_entered.set()
        assert release_expansion.wait(timeout=2)

    monkeypatch.setattr(
        physicell_server,
        "update_signals_behaviors_context_from_config",
        blocking_update,
    )
    monkeypatch.setattr(
        physicell_server,
        "get_expanded_signals",
        lambda: [],
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        signals_future = executor.submit(
            physicell_server.list_all_available_signals,
            session_id=session_id,
        )
        assert expansion_entered.wait(timeout=2)
        summary_future = executor.submit(
            physicell_server.get_simulation_summary,
            session_id=session_id,
        )

        with session_manager._condition:
            assert session_manager._condition.wait_for(
                lambda: session._lease_count == 2,
                timeout=2,
            )
        assert not summary_future.done()

        release_expansion.set()
        signals_future.result(timeout=2)
        summary_future.result(timeout=2)


def test_artifact_cleanup_waits_for_same_session_export(
    tmp_path: Path,
) -> None:
    export_entered = Event()
    release_export = Event()

    def generate_xml() -> str:
        export_entered.set()
        assert release_export.wait(timeout=2)
        return "<PhysiCell_settings/>"

    session_id = _create_session(
        SimpleNamespace(generate_xml=generate_xml)
    )
    session = session_manager.get_session(session_id)
    assert session is not None

    with ThreadPoolExecutor(max_workers=2) as executor:
        export_future = executor.submit(
            physicell_server.export_xml_configuration,
            filename="concurrent.xml",
            session_id=session_id,
        )
        assert export_entered.wait(timeout=2)
        cleanup_future = executor.submit(
            physicell_server.clean_generated_files,
            session_id=session_id,
        )

        with session_manager._condition:
            assert session_manager._condition.wait_for(
                lambda: session._lease_count == 2,
                timeout=2,
            )
        assert not cleanup_future.done()

        release_export.set()
        export_result = export_future.result(timeout=2)
        cleanup_result = cleanup_future.result(timeout=2)

    assert "concurrent.xml" in export_result
    assert "Cleaned 1 artifact file" in cleanup_result
    assert not (
        tmp_path / "artifacts" / session_id / "concurrent.xml"
    ).exists()


def test_signal_context_is_isolated_between_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_id = _create_session(SimpleNamespace(marker="first"))
    second_id = _create_session(SimpleNamespace(marker="second"))
    thread_context = local()
    shared_context = {"marker": ""}
    first_updated = Event()
    second_updated = Event()

    def update_context(config: Any) -> None:
        thread_context.expected = config.marker
        shared_context["marker"] = config.marker
        if config.marker == "first":
            first_updated.set()
        else:
            second_updated.set()

    def expand_signals() -> list[dict[str, Any]]:
        expected = thread_context.expected
        if expected == "first":
            second_updated.wait(timeout=0.2)
        marker = shared_context["marker"]
        assert marker == expected
        return [
            {
                "name": marker,
                "type": "test",
                "requires": [],
                "description": marker,
            }
        ]

    monkeypatch.setattr(
        physicell_server,
        "update_signals_behaviors_context_from_config",
        update_context,
    )
    monkeypatch.setattr(
        physicell_server,
        "get_expanded_signals",
        expand_signals,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(
            physicell_server.list_all_available_signals,
            session_id=first_id,
        )
        assert first_updated.wait(timeout=2)
        second_future = executor.submit(
            physicell_server.list_all_available_signals,
            session_id=second_id,
        )
        first_result = first_future.result(timeout=2)
        second_result = second_future.result(timeout=2)

    assert second_updated.is_set()
    assert "**first**" in first_result
    assert "**second**" in second_result
