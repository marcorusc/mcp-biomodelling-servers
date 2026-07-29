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
    assert "Domain dimensions must be positive" in result.content[0].text
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
