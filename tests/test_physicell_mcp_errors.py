"""Protocol-level tests for PhysiCell tool failure semantics."""

from __future__ import annotations

import asyncio
import copy
import inspect
import sys
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, local
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from mcp import Client, MCPError


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

from mcp_biomodelling_servers.handoff import (  # noqa: E402
    HandoffNetwork,
    HandoffPackage,
    HandoffProvenance,
    MaBoSSSimulationHandoff,
    MaBoSSToPhysiCellHandoffManifest,
    NeKoToMaBoSSHandoffManifest,
    PhysiCellTarget,
    handoff_artifact,
    write_handoff_manifest,
)
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


def _session_resource_config() -> SimpleNamespace:
    intracellular = {
        "type": "maboss",
        "bnd_filename": "tumour.bnd",
        "cfg_filename": "tumour.cfg",
        "settings": {
            "intracellular_dt": 6.0,
            "mutations": [{"intracellular_name": "TP53", "value": "1"}],
        },
        "mapping": {
            "inputs": [{"intracellular_name": "Hypoxia"}],
            "outputs": [{"intracellular_name": "Apoptosis"}],
        },
        "initial_values": [{"intracellular_name": "TP53", "value": 0}],
    }
    cell_type = {
        "phenotype": {
            "cycle": {"model": "Ki67_basic"},
            "volume": {"total": 2500.0, "nuclear": 500.0},
            "motility": {"speed": 0.5, "persistence_time": 5.0},
            "death": {
                "apoptosis": {"default_rate": 0.0001},
                "necrosis": {"default_rate": 0.0002},
            },
            "intracellular": intracellular,
        }
    }
    return SimpleNamespace(
        domain=SimpleNamespace(
            get_info=lambda: {
                "x_min": -500.0,
                "x_max": 500.0,
                "y_min": -250.0,
                "y_max": 250.0,
                "z_min": -10.0,
                "z_max": 10.0,
                "dx": 20.0,
                "dy": 20.0,
                "dz": 20.0,
                "use_2D": True,
            }
        ),
        options=SimpleNamespace(
            get_options=lambda: {
                "max_time": 7200.0,
                "time_units": "min",
                "space_units": "micron",
                "dt_diffusion": 0.01,
                "dt_mechanics": 0.1,
                "dt_phenotype": 6.0,
            }
        ),
        substrates=SimpleNamespace(
            get_substrates=lambda: {
                "oxygen": {
                    "diffusion_coefficient": 100000.0,
                    "decay_rate": 0.01,
                    "initial_condition": 38.0,
                    "units": "mmHg",
                    "dirichlet_enabled": True,
                    "dirichlet_value": 38.0,
                }
            }
        ),
        cell_types=SimpleNamespace(
            get_cell_types=lambda: {"tumour": cell_type}
        ),
        cell_rules=SimpleNamespace(
            get_rules=lambda: [
                {
                    "cell_type": "tumour",
                    "signal": "oxygen",
                    "direction": "decreases",
                    "behavior": "apoptosis",
                    "saturation_value": 0.01,
                    "half_max": 5.0,
                    "hill_power": 4.0,
                    "apply_to_dead": 0,
                }
            ],
            get_rulesets=lambda: {
                "main": {
                    "enabled": True,
                    "folder": "./config",
                    "filename": "rules.csv",
                }
            },
        ),
    )


class HandoffPhysiBoSSStub:
    """Minimal candidate-config PhysiBoSS mutation surface."""

    def __init__(self, config: HandoffConfigStub) -> None:
        self._config = config

    def add_intracellular_model(
        self,
        *,
        cell_type_name: str,
        model_type: str,
        bnd_filename: str,
        cfg_filename: str,
    ) -> None:
        if self._config.fail_attach:
            raise RuntimeError("candidate attach failed")
        if cell_type_name not in self._config.cell_type_data:
            raise ValueError(f"unknown cell type {cell_type_name}")
        phenotype = self._config.cell_type_data[cell_type_name]["phenotype"]
        phenotype["intracellular"] = {
            "type": model_type,
            "bnd_filename": bnd_filename,
            "cfg_filename": cfg_filename,
            "settings": {},
            "mapping": {"inputs": [], "outputs": []},
            "initial_values": [],
        }


class HandoffConfigStub:
    """Copyable PhysiCell configuration for atomic handoff tests."""

    def __init__(
        self,
        cell_type_data: dict[str, dict[str, Any]],
        *,
        fail_copy: bool = False,
        fail_attach: bool = False,
    ) -> None:
        self.cell_type_data = copy.deepcopy(cell_type_data)
        self.fail_copy = fail_copy
        self.fail_attach = fail_attach
        self.cell_types = SimpleNamespace(
            get_cell_types=lambda: self.cell_type_data
        )
        self.physiboss = HandoffPhysiBoSSStub(self)

    def copy(self) -> HandoffConfigStub:
        if self.fail_copy:
            raise RuntimeError("candidate copy failed")
        return HandoffConfigStub(
            self.cell_type_data,
            fail_attach=self.fail_attach,
        )


def _handoff_config(
    *cell_types: str,
    existing_target: str | None = None,
    fail_copy: bool = False,
    fail_attach: bool = False,
) -> HandoffConfigStub:
    data = {
        name: {"phenotype": {}}
        for name in cell_types
    }
    if existing_target is not None:
        data[existing_target]["phenotype"]["intracellular"] = {
            "type": "maboss",
            "bnd_filename": "old.bnd",
            "cfg_filename": "old.cfg",
            "settings": {"intracellular_dt": 6.0},
            "mapping": {
                "inputs": [{"intracellular_name": "OldInput"}],
                "outputs": [{"intracellular_name": "OldOutput"}],
            },
            "initial_values": [],
        }
    return HandoffConfigStub(
        data,
        fail_copy=fail_copy,
        fail_attach=fail_attach,
    )


def _handoff_provenance(
    server: str,
    session_id: str,
) -> HandoffProvenance:
    modelling_package = {
        "NeKo": "nekomata",
        "MaBoSS": "maboss",
    }[server]
    return HandoffProvenance(
        server=server,
        session_id=session_id,
        mcp_package=HandoffPackage(
            name="mcp-biomodelling-servers",
            version="1.0.0",
        ),
        modelling_package=HandoffPackage(
            name=modelling_package,
            version="1.2.3",
        ),
        operation=f"export-{server.lower()}-handoff",
    )


def _create_maboss_handoff(
    tmp_path: Path,
    *,
    target_cell_type: str = "tumour",
    prefix: str = "source",
    with_neko_parent: bool = False,
    include_result: bool = True,
) -> Path:
    source_root = tmp_path / prefix
    source_root.mkdir()
    maboss_session = f"{prefix}-maboss-session"
    bnd_path = source_root / "model.bnd"
    bnd_path.write_text(
        "Node A { logic = B; }\nNode B { logic = A; }\n",
        encoding="utf-8",
    )
    cfg_path = source_root / "model.cfg"
    cfg_path.write_text("max_time = 100;\n", encoding="utf-8")

    lineage = []
    parent_manifest_file = None
    if with_neko_parent:
        neko_session = f"{prefix}-neko-session"
        bnet_path = source_root / "network.bnet"
        bnet_path.write_text(
            "targets, factors\nA, B\nB, A\n",
            encoding="utf-8",
        )
        neko_manifest = NeKoToMaBoSSHandoffManifest(
            source=_handoff_provenance("NeKo", neko_session),
            biological_context="drug response",
            network=HandoffNetwork(
                nodes=["A", "B"],
                output_nodes=["A"],
            ),
            bnet_file=handoff_artifact(
                bnet_path,
                server="NeKo",
                session_id=neko_session,
                role="neko_bnet",
            ),
        )
        parent_path = write_handoff_manifest(
            source_root / "neko.handoff.json",
            neko_manifest,
        )
        lineage = [neko_manifest.source]
        parent_manifest_file = handoff_artifact(
            parent_path,
            server="NeKo",
            session_id=neko_session,
            role="parent_manifest",
        )

    result_file = None
    if include_result:
        result_path = source_root / "result.csv"
        result_path.write_text("Time,A\n100,0.75\n", encoding="utf-8")
        result_file = handoff_artifact(
            result_path,
            server="MaBoSS",
            session_id=maboss_session,
            role="maboss_result",
        )

    manifest = MaBoSSToPhysiCellHandoffManifest(
        source=_handoff_provenance("MaBoSS", maboss_session),
        lineage=lineage,
        biological_context="drug response",
        network=HandoffNetwork(
            nodes=["A", "B"],
            output_nodes=["A"],
        ),
        bnd_file=handoff_artifact(
            bnd_path,
            server="MaBoSS",
            session_id=maboss_session,
            role="maboss_bnd",
        ),
        cfg_file=handoff_artifact(
            cfg_path,
            server="MaBoSS",
            session_id=maboss_session,
            role="maboss_cfg",
        ),
        parent_manifest=parent_manifest_file,
        simulation=MaBoSSSimulationHandoff(
            parameters={"max_time": 100.0, "sample_count": 1000},
            simulation_summary="A reaches probability 0.75.",
            result_file=result_file,
        ),
        target=PhysiCellTarget(cell_type=target_cell_type),
    )
    return write_handoff_manifest(
        source_root / "maboss.handoff.json",
        manifest,
    )


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


def test_workflow_prompt_manual_resource_and_help_share_guidance() -> None:
    prompts = _run(_list_prompts())
    resources = _run(_list_resources())
    templates = _run(_list_resource_templates())
    rendered_prompt = _run(_get_prompt("physicell_workflow_prompt"))
    manual = _run(_read_resource("docs://physicell/agent_manual"))
    help_result = _run(_call_tool("get_help"))

    assert len(prompts.prompts) == 1
    assert prompts.prompts[0].name == "physicell_workflow_prompt"
    assert prompts.prompts[0].title == (
        "Build or revise a PhysiCell configuration"
    )
    assert prompts.prompts[0].arguments == []
    assert len(resources.resources) == 1
    assert str(resources.resources[0].uri) == (
        "docs://physicell/agent_manual"
    )
    assert resources.resources[0].mime_type == "text/markdown"
    assert len(templates.resource_templates) == 7

    expected = physicell_server.PHYSICELL_AGENT_MANUAL
    assert rendered_prompt.messages[0].content.text == expected
    assert manual.contents[0].mime_type == "text/markdown"
    assert manual.contents[0].text == expected
    assert help_result.is_error is False
    assert help_result.content[0].text == expected
    assert help_result.structured_content == {"result": expected}


def test_session_resource_templates_publish_exact_contracts() -> None:
    templates = _run(_list_resource_templates()).resource_templates
    published = {
        str(template.uri_template): (template.name, template.mime_type)
        for template in templates
    }

    assert published == {
        "physicell://session/{session_id}/workflow": (
            "PhysiCell Workflow Status",
            "text/markdown",
        ),
        "physicell://session/{session_id}/domain": (
            "PhysiCell Domain",
            "text/markdown",
        ),
        "physicell://session/{session_id}/substrates": (
            "PhysiCell Substrates",
            "text/markdown",
        ),
        "physicell://session/{session_id}/cell_types": (
            "PhysiCell Cell Types",
            "text/markdown",
        ),
        "physicell://session/{session_id}/cell_rules": (
            "PhysiCell Cell Rules",
            "text/markdown",
        ),
        "physicell://session/{session_id}/physiboss": (
            "PhysiBoSS Integration",
            "text/markdown",
        ),
        "physicell://session/{session_id}/files": (
            "PhysiCell Artifact Files",
            "text/markdown",
        ),
    }


def test_session_resources_return_meaningful_configuration_snapshots(
    tmp_path: Path,
) -> None:
    session_id = _create_session(_session_resource_config())
    session = session_manager.get_session(session_id)
    assert session is not None
    session.scenario_context = "hypoxic tumour"
    session.maboss_context = physicell_server.MaBoSSContext(
        model_name="cell_fate",
        bnd_file_path="/models/cell_fate.bnd",
        cfg_file_path="/models/cell_fate.cfg",
        target_cell_type="tumour",
        available_nodes=["Hypoxia", "TP53", "Apoptosis"],
        output_nodes=["Apoptosis"],
    )
    session.loaded_physiboss_models = ["tumour"]
    session.physiboss_models_count = 1
    session.physiboss_settings_count = 1
    session.physiboss_input_links_count = 1
    session.physiboss_output_links_count = 1
    session.physiboss_mutations_count = 1
    session.completed_steps.update(
        {
            physicell_server.WorkflowStep.DOMAIN_SETUP,
            physicell_server.WorkflowStep.SUBSTRATES_ADDED,
            physicell_server.WorkflowStep.CELL_TYPES_ADDED,
        }
    )
    artifact_dir = tmp_path / "artifacts" / session_id
    artifact_dir.mkdir(parents=True)
    artifact = artifact_dir / "PhysiCell_settings.xml"
    artifact.write_text("<PhysiCell_settings/>", encoding="utf-8")

    rendered = {
        suffix: _run(
            _read_resource(
                f"physicell://session/{session_id}/{suffix}"
            )
        ).contents[0]
        for suffix in (
            "workflow",
            "domain",
            "substrates",
            "cell_types",
            "cell_rules",
            "physiboss",
            "files",
        )
    }

    assert all(
        content.mime_type == "text/markdown"
        for content in rendered.values()
    )
    assert "hypoxic tumour" in rendered["workflow"].text
    assert "Mode: 2D" in rendered["domain"].text
    assert "x=[-500, 500]" in rendered["domain"].text
    assert "Extent: x=1000, y=500, z=20 micron" in rendered["domain"].text
    assert "diffusion=0.01" in rendered["domain"].text
    assert "**oxygen**" in rendered["substrates"].text
    assert "diffusion=100000" in rendered["substrates"].text
    assert "**tumour**" in rendered["cell_types"].text
    assert "cycle=Ki67_basic" in rendered["cell_types"].text
    assert "apoptosis=0.0001" in rendered["cell_types"].text
    assert "oxygen decreases apoptosis" in rendered["cell_rules"].text
    assert "**main**" in rendered["cell_rules"].text
    assert "Model: cell_fate" in rendered["physiboss"].text
    assert "settings=[intracellular_dt=6]" in rendered["physiboss"].text
    assert "inputs=1" in rendered["physiboss"].text
    assert "mutations=1" in rendered["physiboss"].text
    assert str(artifact) in rendered["files"].text


@pytest.mark.parametrize(
    "suffix",
    [
        "workflow",
        "domain",
        "substrates",
        "cell_types",
        "cell_rules",
        "physiboss",
        "files",
    ],
)
def test_unknown_session_resources_return_typed_not_found(
    suffix: str,
) -> None:
    uri = f"physicell://session/missing-session/{suffix}"

    error = _run(_read_resource_error(uri))

    assert error.code == -32602
    assert "PhysiCell session not found: missing-session" in str(error)
    assert error.data == {"uri": uri}
    assert session_manager.list_sessions() == []


def test_resources_distinguish_absent_configuration_from_valid_session() -> None:
    session_id = _create_session()

    workflow = _run(
        _read_resource(f"physicell://session/{session_id}/workflow")
    )
    files = _run(
        _read_resource(f"physicell://session/{session_id}/files")
    )

    assert "No simulation configured yet" in workflow.contents[0].text
    assert "No artifact files found" in files.contents[0].text
    for suffix in (
        "domain",
        "substrates",
        "cell_types",
        "cell_rules",
        "physiboss",
    ):
        error = _run(
            _read_resource_error(
                f"physicell://session/{session_id}/{suffix}"
            )
        )
        assert error.code == -32602
        assert "No PhysiCell configuration in this session" in str(error)


def test_empty_configuration_resources_are_successful() -> None:
    config = _session_resource_config()
    config.substrates.get_substrates = lambda: {}
    config.cell_types.get_cell_types = lambda: {}
    config.cell_rules.get_rules = lambda: []
    config.cell_rules.get_rulesets = lambda: {}
    session_id = _create_session(config)

    expected_messages = {
        "substrates": "No substrates configured.",
        "cell_types": "No cell types configured.",
        "cell_rules": "No cell rules configured.",
        "physiboss": "No PhysiBoSS integration configured.",
    }
    for suffix, expected in expected_messages.items():
        result = _run(
            _read_resource(
                f"physicell://session/{session_id}/{suffix}"
            )
        )
        assert expected in result.contents[0].text


def test_files_resource_does_not_create_artifact_directory(
    tmp_path: Path,
) -> None:
    session_id = _create_session()
    artifacts_dir = tmp_path / "artifacts"
    assert not artifacts_dir.exists()

    result = _run(
        _read_resource(f"physicell://session/{session_id}/files")
    )

    assert "No artifact files found" in result.contents[0].text
    assert not artifacts_dir.exists()


def test_session_discovery_tools_publish_structured_output_schemas() -> None:
    listed_tools = _run(_list_tools())
    tools = {tool.name: tool for tool in listed_tools.tools}

    for tool_name, expected_title in {
        "list_sessions": "PhysiCellSessionListResult",
        "list_artifact_sessions": "PhysiCellArtifactSessionListResult",
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
        "No active sessions. Use `create_session()` to start."
    )
    assert result.structured_content == {
        "server": "PhysiCell",
        "count": 0,
        "sessions": [],
    }


def test_list_sessions_returns_structured_physicell_state() -> None:
    session_id = session_manager.create_session(session_name="tumour spheroid")
    session = session_manager.get_session(session_id)
    assert session is not None
    session.config = object()
    session.scenario_context = "hypoxic tumour spheroid"
    session.substrates_count = 2
    session.cell_types_count = 3
    session.rules_count = 1

    result = _run(_call_tool("list_sessions"))

    assert result.is_error is False
    assert session_id[:8] in result.content[0].text
    assert result.structured_content is not None
    structured_session = result.structured_content["sessions"][0]
    assert structured_session["session_id"] == session_id
    assert structured_session["session_name"] == "tumour spheroid"
    assert structured_session["created_at"] >= 0
    assert structured_session["last_accessed"] >= structured_session["created_at"]
    assert structured_session["is_default"] is True
    assert structured_session["has_configuration"] is True
    assert structured_session["scenario_context"] == "hypoxic tumour spheroid"
    assert structured_session["substrates_count"] == 2
    assert structured_session["cell_types_count"] == 3
    assert structured_session["rules_count"] == 1
    assert structured_session["loaded_from_xml"] is False
    assert structured_session["xml_modification_count"] == 0
    assert result.structured_content["server"] == "PhysiCell"
    assert result.structured_content["count"] == 1


def test_list_artifact_sessions_returns_structured_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        physicell_server,
        "_list_artifact_sessions_on_disk",
        lambda *_args, **_kwargs: [
            {
                "session_id": "physicell-artifact-session",
                "server": "PhysiCell",
                "label": "tumour spheroid",
                "created_at": "2026-07-30T10:20:30+00:00",
                "files": ["PhysiCell_settings.xml"],
            }
        ],
    )

    result = _run(_call_tool("list_artifact_sessions"))

    assert result.is_error is False
    assert "Full ID: `physicell-artifact-session`" in result.content[0].text
    assert result.structured_content == {
        "server": "PhysiCell",
        "count": 1,
        "sessions": [
            {
                "session_id": "physicell-artifact-session",
                "server": "PhysiCell",
                "label": "tumour spheroid",
                "created_at": "2026-07-30T10:20:30+00:00",
                "files": ["PhysiCell_settings.xml"],
            }
        ],
    }


def test_artifact_tools_publish_structured_output_schemas() -> None:
    listed_tools = _run(_list_tools())
    tools = {tool.name: tool for tool in listed_tools.tools}

    for tool_name, expected_title in {
        "export_xml_configuration": "PhysiCellXmlExportResult",
        "export_cell_rules_csv": "PhysiCellRulesExportResult",
        "list_generated_files": "PhysiCellArtifactFileListResult",
        "clean_generated_files": "PhysiCellArtifactCleanupResult",
    }.items():
        schema = tools[tool_name].output_schema
        assert schema is not None
        assert schema["title"] == expected_title
        assert "result" not in schema["properties"]


def test_scientific_tools_publish_structured_output_schemas() -> None:
    listed_tools = _run(_list_tools())
    tools = {tool.name: tool for tool in listed_tools.tools}

    expected_models = {
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
    }

    for tool_name, expected_title in expected_models.items():
        schema = tools[tool_name].output_schema
        assert schema is not None
        assert schema["title"] == expected_title
        assert "result" not in schema["properties"]


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
    assert result.structured_content == {
        "server": "PhysiCell",
        "filepath": str(invalid_path),
        "filename": "invalid.xml",
        "valid": False,
        "error_message": "missing PhysiCell_settings root",
    }


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


def test_import_maboss_handoff_copies_and_attaches_standalone_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(physicell_server, "PHYSIBOSS_AVAILABLE", True)
    manifest_path = _create_maboss_handoff(tmp_path)
    original_config = _handoff_config("tumour")
    session_id = _create_session(original_config)

    result = _run(
        _call_tool(
            "import_maboss_handoff",
            {
                "manifest_path": str(manifest_path),
                "artifact_prefix": "tumour_model",
                "session_id": session_id,
            },
        )
    )

    assert result.is_error is False
    payload = result.structured_content
    assert payload["target_cell_type"] == "tumour"
    assert payload["nodes"] == ["A", "B"]
    assert payload["output_nodes"] == ["A"]
    assert payload["replaced_existing"] is False
    assert payload["context_count"] == 1
    assert payload["result_file"]["name"] == "tumour_model.result.csv"
    assert payload["neko_manifest"] is None
    assert payload["neko_manifest_file"] is None
    assert payload["bnet_file"] is None

    session = session_manager.get_session(session_id)
    assert session is not None
    assert session.config is not original_config
    assert "intracellular" not in (
        original_config.cell_type_data["tumour"]["phenotype"]
    )
    intracellular = session.config.cell_type_data["tumour"]["phenotype"][
        "intracellular"
    ]
    assert intracellular["bnd_filename"].endswith("tumour_model.bnd")
    assert intracellular["cfg_filename"].endswith("tumour_model.cfg")
    assert session.physiboss_models_count == 1
    context = session.maboss_contexts["tumour"]
    assert context.source_manifest_path == str(manifest_path)
    assert context.simulation_parameters == {
        "max_time": 100.0,
        "sample_count": 1000,
    }
    assert context.result_file_path.endswith("tumour_model.result.csv")


def test_import_maboss_handoff_preserves_complete_neko_lineage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(physicell_server, "PHYSIBOSS_AVAILABLE", True)
    manifest_path = _create_maboss_handoff(
        tmp_path,
        with_neko_parent=True,
    )
    session_id = _create_session(_handoff_config("tumour"))

    result = _run(
        _call_tool(
            "import_maboss_handoff",
            {
                "manifest_path": str(manifest_path),
                "session_id": session_id,
            },
        )
    )

    assert result.is_error is False
    payload = result.structured_content
    assert payload["neko_manifest"]["handoff_type"] == "neko-to-maboss"
    assert payload["neko_manifest_file"]["server"] == "PhysiCell"
    assert payload["bnet_file"]["server"] == "PhysiCell"
    assert payload["bnet_file"]["name"] == "maboss_import.neko.bnet"
    session = session_manager.get_session(session_id)
    assert session is not None
    context = session.maboss_contexts["tumour"]
    assert context.neko_session_id == "source-neko-session"
    assert context.local_neko_manifest_path.endswith(
        "maboss_import.neko.handoff.json"
    )
    assert context.local_bnet_path.endswith("maboss_import.neko.bnet")


def test_import_maboss_handoffs_retain_context_per_target_cell(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(physicell_server, "PHYSIBOSS_AVAILABLE", True)
    tumour_manifest = _create_maboss_handoff(
        tmp_path,
        target_cell_type="tumour",
        prefix="tumour_source",
    )
    immune_manifest = _create_maboss_handoff(
        tmp_path,
        target_cell_type="immune",
        prefix="immune_source",
        include_result=False,
    )
    session_id = _create_session(_handoff_config("tumour", "immune"))

    tumour_result = _run(
        _call_tool(
            "import_maboss_handoff",
            {
                "manifest_path": str(tumour_manifest),
                "artifact_prefix": "tumour",
                "session_id": session_id,
            },
        )
    )
    immune_result = _run(
        _call_tool(
            "import_maboss_handoff",
            {
                "manifest_path": str(immune_manifest),
                "artifact_prefix": "immune",
                "session_id": session_id,
            },
        )
    )
    replacement_result = _run(
        _call_tool(
            "import_maboss_handoff",
            {
                "manifest_path": str(tumour_manifest),
                "artifact_prefix": "tumour_reimport",
                "replace_existing": True,
                "session_id": session_id,
            },
        )
    )
    contexts = _run(
        _call_tool(
            "get_maboss_context",
            {"cell_type": "immune", "session_id": session_id},
        )
    )
    latest_context = _run(
        _call_tool(
            "get_maboss_context",
            {"session_id": session_id},
        )
    )

    assert tumour_result.is_error is False
    assert immune_result.is_error is False
    assert replacement_result.is_error is False
    assert immune_result.structured_content["context_count"] == 2
    assert replacement_result.structured_content["context_count"] == 2
    assert contexts.structured_content["context_count"] == 2
    assert contexts.structured_content["selected_cell_type"] == "immune"
    assert contexts.structured_content["context"]["target_cell_type"] == "immune"
    assert latest_context.structured_content["selected_cell_type"] is None
    assert latest_context.structured_content["context"]["target_cell_type"] == (
        "tumour"
    )
    assert {
        context["target_cell_type"]
        for context in contexts.structured_content["contexts"]
    } == {"tumour", "immune"}


def test_import_maboss_handoff_requires_explicit_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(physicell_server, "PHYSIBOSS_AVAILABLE", True)
    manifest_path = _create_maboss_handoff(tmp_path)
    original_config = _handoff_config(
        "tumour",
        "immune",
        existing_target="tumour",
    )
    session_id = _create_session(original_config)

    refused = _run(
        _call_tool(
            "import_maboss_handoff",
            {
                "manifest_path": str(manifest_path),
                "session_id": session_id,
            },
        )
    )
    assert refused.is_error is True
    assert "replace_existing=true" in refused.content[0].text
    session = session_manager.get_session(session_id)
    assert session is not None
    assert session.config is original_config
    assert not (tmp_path / "artifacts" / session_id).exists()

    replaced = _run(
        _call_tool(
            "import_maboss_handoff",
            {
                "manifest_path": str(manifest_path),
                "replace_existing": True,
                "session_id": session_id,
            },
        )
    )
    assert replaced.is_error is False
    assert replaced.structured_content["replaced_existing"] is True
    session = session_manager.get_session(session_id)
    assert session is not None
    intracellular = session.config.cell_type_data["tumour"]["phenotype"][
        "intracellular"
    ]
    assert intracellular["settings"] == {}
    assert intracellular["mapping"] == {"inputs": [], "outputs": []}
    assert session.config.cell_type_data["immune"] == (
        original_config.cell_type_data["immune"]
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_handoff_config("immune"), "Target cell type 'tumour'"),
        (
            _handoff_config("tumour", fail_copy=True),
            "Could not copy the PhysiCell configuration",
        ),
        (
            _handoff_config("tumour", fail_attach=True),
            "Could not attach the MaBoSS model",
        ),
    ],
)
def test_import_maboss_handoff_candidate_failures_preserve_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config: HandoffConfigStub,
    message: str,
) -> None:
    monkeypatch.setattr(physicell_server, "PHYSIBOSS_AVAILABLE", True)
    manifest_path = _create_maboss_handoff(tmp_path)
    session_id = _create_session(config)

    result = _run(
        _call_tool(
            "import_maboss_handoff",
            {
                "manifest_path": str(manifest_path),
                "session_id": session_id,
            },
        )
    )

    assert result.is_error is True
    assert message in result.content[0].text
    session = session_manager.get_session(session_id)
    assert session is not None
    assert session.config is config
    assert session.maboss_contexts == {}
    artifact_dir = tmp_path / "artifacts" / session_id
    assert not artifact_dir.exists() or not list(artifact_dir.iterdir())


def test_import_maboss_handoff_rejects_changed_source_without_state_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(physicell_server, "PHYSIBOSS_AVAILABLE", True)
    manifest_path = _create_maboss_handoff(tmp_path)
    source_bnd = manifest_path.parent / "model.bnd"
    source_bnd.write_text(
        "Node A { logic = 0; }\nNode B { logic = 0; }\n",
        encoding="utf-8",
    )
    config = _handoff_config("tumour")
    session_id = _create_session(config)

    result = _run(
        _call_tool(
            "import_maboss_handoff",
            {
                "manifest_path": str(manifest_path),
                "session_id": session_id,
            },
        )
    )

    assert result.is_error is True
    assert "size changed" in result.content[0].text or (
        "digest changed" in result.content[0].text
    )
    session = session_manager.get_session(session_id)
    assert session is not None
    assert session.config is config
    assert session.maboss_contexts == {}


def test_import_maboss_handoff_rejects_stale_neko_lineage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(physicell_server, "PHYSIBOSS_AVAILABLE", True)
    manifest_path = _create_maboss_handoff(
        tmp_path,
        with_neko_parent=True,
    )
    source_bnet = manifest_path.parent / "network.bnet"
    source_bnet.write_text("targets, factors\nA, 0\nB, 0\n", encoding="utf-8")
    config = _handoff_config("tumour")
    session_id = _create_session(config)

    result = _run(
        _call_tool(
            "import_maboss_handoff",
            {
                "manifest_path": str(manifest_path),
                "session_id": session_id,
            },
        )
    )

    assert result.is_error is True
    assert "digest changed" in result.content[0].text
    session = session_manager.get_session(session_id)
    assert session is not None
    assert session.config is config
    assert session.maboss_contexts == {}


def test_import_maboss_handoff_rolls_back_partial_artifact_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(physicell_server, "PHYSIBOSS_AVAILABLE", True)
    manifest_path = _create_maboss_handoff(tmp_path)
    config = _handoff_config("tumour")
    session_id = _create_session(config)
    original_link = physicell_server._link_handoff_artifact_without_overwrite
    calls = 0

    def fail_second_link(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated publication failure")
        original_link(source, destination)

    monkeypatch.setattr(
        physicell_server,
        "_link_handoff_artifact_without_overwrite",
        fail_second_link,
    )
    result = _run(
        _call_tool(
            "import_maboss_handoff",
            {
                "manifest_path": str(manifest_path),
                "session_id": session_id,
            },
        )
    )

    assert result.is_error is True
    assert "simulated publication failure" in result.content[0].text
    session = session_manager.get_session(session_id)
    assert session is not None
    assert session.config is config
    assert session.maboss_contexts == {}
    artifact_dir = tmp_path / "artifacts" / session_id
    assert list(artifact_dir.glob("maboss_import*")) == []


def test_import_maboss_handoff_refuses_existing_artifact_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(physicell_server, "PHYSIBOSS_AVAILABLE", True)
    manifest_path = _create_maboss_handoff(tmp_path)
    config = _handoff_config("tumour")
    session_id = _create_session(config)
    artifact_dir = tmp_path / "artifacts" / session_id
    artifact_dir.mkdir(parents=True)
    existing = artifact_dir / "maboss_import.cfg"
    existing.write_text("keep\n", encoding="utf-8")

    result = _run(
        _call_tool(
            "import_maboss_handoff",
            {
                "manifest_path": str(manifest_path),
                "session_id": session_id,
            },
        )
    )

    assert result.is_error is True
    assert "Refusing to overwrite" in result.content[0].text
    assert existing.read_text(encoding="utf-8") == "keep\n"
    session = session_manager.get_session(session_id)
    assert session is not None
    assert session.config is config


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
    assert result.structured_content is not None
    assert result.structured_content["server"] == "PhysiCell"
    assert result.structured_content["session_id"] == session_id
    assert result.structured_content["source"] == "created"
    assert result.structured_content["source_filename"] is None
    assert result.structured_content["modification_count"] == 0
    assert result.structured_content["file"]["path"] == str(output_path)
    assert result.structured_content["file"]["media_type"] == "application/xml"
    assert result.structured_content["file"]["size_bytes"] == output_path.stat().st_size


def test_loaded_xml_export_returns_structured_source_provenance(
    tmp_path: Path,
) -> None:
    generate_calls: list[str] = []
    session_id = _create_session(_xml_export_config(generate_calls))
    session = session_manager.get_session(session_id)
    assert session is not None
    session.loaded_from_xml = True
    session.original_xml_path = "/models/baseline.xml"
    session.xml_modification_count = 2

    result = _run(
        _call_tool(
            "export_xml_configuration",
            {"filename": "modified.xml", "session_id": session_id},
        )
    )

    output_path = tmp_path / "artifacts" / session_id / "modified.xml"
    assert result.is_error is False
    assert "Modified 2 times from baseline.xml" in result.content[0].text
    assert result.structured_content is not None
    assert result.structured_content["source"] == "loaded"
    assert result.structured_content["source_filename"] == "baseline.xml"
    assert result.structured_content["modification_count"] == 2
    assert result.structured_content["file"]["path"] == str(output_path)


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
    assert result.structured_content is not None
    assert result.structured_content["server"] == "PhysiCell"
    assert result.structured_content["session_id"] == session_id
    assert result.structured_content["xml_reference"] == f"./config/{expected_name}"
    assert result.structured_content["enabled"] is True
    assert result.structured_content["rule_count"] == 1
    assert result.structured_content["file"]["path"] == str(output_path)
    assert result.structured_content["file"]["media_type"] == "text/csv"
    assert result.structured_content["file"]["size_bytes"] == output_path.stat().st_size


def test_physicell_artifact_listing_and_cleanup_are_structured(
    tmp_path: Path,
) -> None:
    first_id = _create_session()
    second_id = _create_session()
    first_dir = tmp_path / "artifacts" / first_id
    second_dir = tmp_path / "artifacts" / second_id
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    first_xml = first_dir / "PhysiCell_settings.xml"
    second_csv = second_dir / "cell_rules.csv"
    first_xml.write_text("<PhysiCell_settings/>", encoding="utf-8")
    second_csv.write_text("cell_type,signal\n", encoding="utf-8")
    (second_dir / "notes.txt").write_text("ignored", encoding="utf-8")

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
    assert session_result.structured_content["files"][0]["path"] == str(first_xml)
    assert all_result.structured_content is not None
    assert all_result.structured_content["scope"] == "all"
    assert all_result.structured_content["session_id"] is None
    assert all_result.structured_content["count"] == 2
    assert {
        file_record["session_id"]
        for file_record in all_result.structured_content["files"]
    } == {first_id, second_id}
    assert cleanup_result.structured_content == {
        "session_id": first_id,
        "removed_count": 1,
        "server": "PhysiCell",
    }
    assert not first_xml.exists()


def test_empty_status_results_remain_successful() -> None:
    summary = _run(_call_tool("get_simulation_summary"))

    assert summary.is_error is False
    assert "No active session" in summary.content[0].text
    assert summary.structured_content == {
        "server": "PhysiCell",
        "session_id": None,
        "has_active_session": False,
        "has_configuration": False,
        "progress": 0.0,
        "scenario_context": None,
        "substrates": [],
        "cell_types": [],
        "rules_count": 0,
        "physiboss_models_count": 0,
        "completed_steps": [],
        "next_steps": [],
        "ready_for_export": False,
        "loaded_from_xml": False,
        "original_xml_path": None,
        "xml_modification_count": 0,
    }

    session_id = _create_session()
    files = _run(
        _call_tool("list_generated_files", {"session_id": session_id})
    )
    context = _run(_call_tool("get_maboss_context", {"session_id": session_id}))

    assert files.is_error is False
    assert "No PhysiCell artifact files found" in files.content[0].text
    assert files.structured_content == {
        "scope": "session",
        "session_id": session_id,
        "count": 0,
        "files": [],
        "server": "PhysiCell",
    }
    assert context.is_error is False
    assert "No MaBoSS context available" in context.content[0].text
    assert context.structured_content == {
        "server": "PhysiCell",
        "session_id": session_id,
        "has_context": False,
        "context": None,
        "context_count": 0,
        "contexts": [],
        "selected_cell_type": None,
    }


def test_workflow_summary_returns_complete_configuration_state() -> None:
    config = SimpleNamespace(
        substrates=SimpleNamespace(
            get_substrates=lambda: {"oxygen": object(), "drug": object()}
        ),
        cell_types=SimpleNamespace(
            get_cell_types=lambda: {"tumour": object()}
        ),
        cell_rules=SimpleNamespace(get_rules=lambda: [object(), object()]),
    )
    session_id = _create_session(config)
    session = session_manager.get_session(session_id)
    assert session is not None
    session.scenario_context = "hypoxic treated tumour"
    session.physiboss_models_count = 1
    session.completed_steps.update(
        {
            physicell_server.WorkflowStep.DOMAIN_SETUP,
            physicell_server.WorkflowStep.SUBSTRATES_ADDED,
            physicell_server.WorkflowStep.CELL_TYPES_ADDED,
        }
    )

    summary = _run(
        _call_tool("get_simulation_summary", {"session_id": session_id})
    )
    status = _run(
        _call_tool("get_workflow_status", {"session_id": session_id})
    )

    assert summary.is_error is False
    assert summary.structured_content is not None
    assert summary.structured_content["session_id"] == session_id
    assert summary.structured_content["has_configuration"] is True
    assert summary.structured_content["scenario_context"] == (
        "hypoxic treated tumour"
    )
    assert summary.structured_content["substrates"] == ["oxygen", "drug"]
    assert summary.structured_content["cell_types"] == ["tumour"]
    assert summary.structured_content["rules_count"] == 2
    assert summary.structured_content["physiboss_models_count"] == 1
    assert summary.structured_content["ready_for_export"] is True
    assert summary.structured_content["completed_steps"] == [
        "cell_types_added",
        "domain_setup",
        "substrates_added",
    ]
    assert status.structured_content == summary.structured_content


def test_maboss_context_returns_cross_server_handoff_fields() -> None:
    session_id = _create_session()
    store_result = _run(
        _call_tool(
            "set_maboss_context",
            {
                "session_id": session_id,
                "model_name": "cell_fate",
                "bnd_file_path": "/models/cell_fate.bnd",
                "cfg_file_path": "/models/cell_fate.cfg",
                "target_cell_type": "tumour",
                "available_nodes": "Apoptosis, Proliferation",
                "output_nodes": "Apoptosis",
                "simulation_results": "Apoptosis reaches 0.7 probability.",
                "biological_context": "drug response",
            },
        )
    )
    context_result = _run(
        _call_tool("get_maboss_context", {"session_id": session_id})
    )

    assert store_result.is_error is False
    expected_context = {
        "model_name": "cell_fate",
        "bnd_file_path": "/models/cell_fate.bnd",
        "cfg_file_path": "/models/cell_fate.cfg",
        "available_nodes": ["Apoptosis", "Proliferation"],
        "output_nodes": ["Apoptosis"],
        "simulation_results": "Apoptosis reaches 0.7 probability.",
        "target_cell_type": "tumour",
        "biological_context": "drug response",
        "source_manifest_path": None,
        "local_manifest_path": None,
        "source_session_id": None,
        "result_file_path": None,
        "simulation_parameters": {},
        "neko_session_id": None,
        "neko_manifest_path": None,
        "local_neko_manifest_path": None,
        "local_bnet_path": None,
    }
    assert context_result.structured_content == {
        "server": "PhysiCell",
        "session_id": session_id,
        "has_context": True,
        "context": expected_context,
        "context_count": 1,
        "contexts": [expected_context],
        "selected_cell_type": None,
    }


def test_loaded_configuration_and_components_return_typed_properties() -> None:
    oxygen = SimpleNamespace(
        diffusion_coefficient=100000.0,
        decay_rate=0.01,
        initial_condition=38.0,
    )
    tumour = SimpleNamespace(
        cycle_model="live",
        phenotype=SimpleNamespace(
            volume=SimpleNamespace(total=2494.0),
            motility=SimpleNamespace(speed=1.2),
            intracellular=object(),
        ),
    )
    config = SimpleNamespace(
        domain=SimpleNamespace(
            x_min=-500.0,
            x_max=500.0,
            y_min=-250.0,
            y_max=250.0,
            z_min=-10.0,
            z_max=10.0,
        ),
        substrates=SimpleNamespace(
            get_substrate=lambda name: oxygen if name == "oxygen" else None,
        ),
        cell_types=SimpleNamespace(
            get_cell_type=lambda name: tumour if name == "tumour" else None,
        ),
    )
    session_id = _create_session(config)
    session = session_manager.get_session(session_id)
    assert session is not None
    session.loaded_from_xml = True
    session.original_xml_path = "/models/baseline.xml"
    session.xml_modification_count = 3
    session.loaded_substrates = ["oxygen", "inaccessible"]
    session.loaded_cell_types = ["tumour", "missing"]
    session.loaded_physiboss_models = ["tumour"]
    session.has_existing_rules = True

    analysis = _run(
        _call_tool(
            "analyze_loaded_configuration",
            {"session_id": session_id},
        )
    )
    components = _run(
        _call_tool(
            "list_loaded_components",
            {"session_id": session_id, "component_type": "all"},
        )
    )

    assert analysis.structured_content is not None
    assert analysis.structured_content["source_path"] == "/models/baseline.xml"
    assert analysis.structured_content["modification_count"] == 3
    assert analysis.structured_content["domain"] == {
        "x_size": 1000.0,
        "y_size": 500.0,
        "z_size": 20.0,
    }
    assert analysis.structured_content["substrates"] == [
        "oxygen",
        "inaccessible",
    ]
    assert analysis.structured_content["marked_analyzed"] is True
    assert components.structured_content is not None
    assert components.structured_content["component_type"] == "all"
    assert components.structured_content["substrate_count"] == 2
    assert components.structured_content["cell_type_count"] == 2
    assert components.structured_content["physiboss_model_count"] == 1
    assert components.structured_content["substrates"][0] == {
        "name": "oxygen",
        "properties_accessible": True,
        "diffusion_coefficient": 100000.0,
        "decay_rate": 0.01,
        "initial_condition": 38.0,
    }
    assert components.structured_content["substrates"][1][
        "properties_accessible"
    ] is False
    assert components.structured_content["cell_types"][0] == {
        "name": "tumour",
        "properties_accessible": True,
        "total_volume": 2494.0,
        "motility_speed": 1.2,
        "cycle_model": "live",
        "physiboss_enabled": True,
    }
    assert components.structured_content["cell_types"][1][
        "properties_accessible"
    ] is False


def test_cycle_signal_and_behavior_discovery_return_exact_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        physicell_server,
        "get_default_parameters",
        lambda: {
            "cell_cycle_models": {
                "live": {"name": "Live cells"},
                "Ki67_basic": {"name": "Ki67 basic"},
            }
        },
    )
    monkeypatch.setattr(
        physicell_server,
        "get_signals_behaviors",
        lambda: {
            "signals": {
                "oxygen": {
                    "name": "oxygen",
                    "type": "substrate",
                    "description": "Local oxygen concentration",
                    "requires": ["oxygen substrate"],
                }
            },
            "behaviors": {
                "cycle entry": {
                    "name": "cycle entry",
                    "type": "cycle",
                    "description": "Cell-cycle entry rate",
                    "requires": [],
                }
            },
        },
    )

    cycles = _run(_call_tool("get_available_cycle_models"))
    signals = _run(_call_tool("list_all_available_signals"))
    behaviors = _run(_call_tool("list_all_available_behaviors"))

    assert cycles.structured_content == {
        "server": "PhysiCell",
        "model_count": 2,
        "models": [
            {"key": "live", "name": "Live cells"},
            {"key": "Ki67_basic", "name": "Ki67 basic"},
        ],
    }
    assert signals.structured_content == {
        "server": "PhysiCell",
        "session_id": None,
        "scenario_context": None,
        "signal_count": 1,
        "signals": [
            {
                "name": "oxygen",
                "signal_type": "substrate",
                "description": "Local oxygen concentration",
                "requires": ["oxygen substrate"],
            }
        ],
    }
    assert behaviors.structured_content == {
        "server": "PhysiCell",
        "session_id": None,
        "scenario_context": None,
        "behavior_count": 1,
        "behaviors": [
            {
                "name": "cycle entry",
                "behavior_type": "cycle",
                "description": "Cell-cycle entry rate",
                "requires": [],
            }
        ],
    }


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
        "import_maboss_handoff",
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
        "import_maboss_handoff",
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
        "import_maboss_handoff",
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

    import_schema = tools[
        "import_maboss_handoff"
    ].input_schema
    import_properties = import_schema["properties"]
    assert "manifest_path" in import_schema["required"]
    assert import_properties["manifest_path"]["minLength"] == 1
    assert import_properties["artifact_prefix"]["default"] == "maboss_import"
    assert import_properties["artifact_prefix"]["pattern"] == (
        r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9_-])?$"
    )
    assert import_properties["replace_existing"]["default"] is False


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("set_default_session", {"session_id": ""}),
        ("load_xml_configuration", {"filepath": "   "}),
        ("import_maboss_handoff", {"manifest_path": "   "}),
        (
            "import_maboss_handoff",
            {
                "manifest_path": "/tmp/handoff.json",
                "artifact_prefix": "../unsafe",
            },
        ),
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

    assert "concurrent.xml" in export_result.content[0].text
    assert "Cleaned 1 artifact file" in cleanup_result.content[0].text
    assert not (
        tmp_path / "artifacts" / session_id / "concurrent.xml"
    ).exists()


def test_files_resource_waits_for_same_session_export_only(
    tmp_path: Path,
) -> None:
    export_entered = Event()
    release_export = Event()

    def generate_xml() -> str:
        export_entered.set()
        assert release_export.wait(timeout=2)
        return "<PhysiCell_settings/>"

    blocked_id = _create_session(
        SimpleNamespace(generate_xml=generate_xml)
    )
    independent_id = _create_session()
    blocked_session = session_manager.get_session(blocked_id)
    assert blocked_session is not None

    with ThreadPoolExecutor(max_workers=3) as executor:
        export_future = executor.submit(
            physicell_server.export_xml_configuration,
            filename="concurrent.xml",
            session_id=blocked_id,
        )
        assert export_entered.wait(timeout=2)
        blocked_resource = executor.submit(
            physicell_server.physicell_files_resource,
            session_id=blocked_id,
        )

        with session_manager._condition:
            assert session_manager._condition.wait_for(
                lambda: blocked_session._lease_count == 2,
                timeout=2,
            )
        assert not blocked_resource.done()

        independent_resource = executor.submit(
            physicell_server.physicell_files_resource,
            session_id=independent_id,
        )
        assert "No artifact files found" in independent_resource.result(
            timeout=2
        )

        release_export.set()
        export_future.result(timeout=2)
        resource_text = blocked_resource.result(timeout=2)

    expected_path = (
        tmp_path / "artifacts" / blocked_id / "concurrent.xml"
    )
    assert str(expected_path) in resource_text


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
    assert "**first**" in first_result.content[0].text
    assert "**second**" in second_result.content[0].text
    assert first_result.structured_content["signals"][0]["name"] == "first"
    assert second_result.structured_content["signals"][0]["name"] == "second"
