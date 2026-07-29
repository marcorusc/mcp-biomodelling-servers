"""Protocol-level tests for PhysiCell tool failure semantics."""

import asyncio
import sys
from collections.abc import Coroutine
from pathlib import Path
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


def _clear_sessions() -> None:
    for session in list(session_manager.list_sessions()):
        session_manager.delete_session(session.session_id)


def _create_session(config: object | None = None) -> str:
    session_id = session_manager.create_session()
    session = session_manager.get_session(session_id)
    assert session is not None
    session.config = config
    return session_id


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
