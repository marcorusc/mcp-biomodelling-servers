"""Validated scientific output contracts for PhysiCell MCP tools."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from mcp_biomodelling_servers.structured_outputs import StructuredOutputModel

ComponentType = Literal["substrates", "cell_types", "physiboss", "all"]


class PhysiCellScientificResult(StructuredOutputModel):
    """Base contract for PhysiCell scientific results."""

    server: Literal["PhysiCell"]


class PhysiCellDomainRecord(StructuredOutputModel):
    """Spatial dimensions read from a PhysiCell configuration."""

    x_size: float
    y_size: float
    z_size: float


class PhysiCellWorkflowStatusResult(PhysiCellScientificResult):
    """Machine-readable workflow status for an optional active session."""

    session_id: str | None = None
    has_active_session: bool
    has_configuration: bool
    progress: float = Field(ge=0, le=100)
    scenario_context: str | None = None
    substrates: list[str]
    cell_types: list[str]
    rules_count: int = Field(ge=0)
    physiboss_models_count: int = Field(ge=0)
    completed_steps: list[str]
    next_steps: list[str]
    ready_for_export: bool
    loaded_from_xml: bool
    original_xml_path: str | None = None
    xml_modification_count: int = Field(ge=0)


class PhysiCellMaBoSSContextRecord(StructuredOutputModel):
    """Cross-server MaBoSS context stored for PhysiBoSS integration."""

    model_name: str
    bnd_file_path: str
    cfg_file_path: str
    available_nodes: list[str]
    output_nodes: list[str]
    simulation_results: str | None = None
    target_cell_type: str
    biological_context: str | None = None


class PhysiCellMaBoSSContextResult(PhysiCellScientificResult):
    """Optional MaBoSS context associated with a PhysiCell session."""

    session_id: str = Field(min_length=1)
    has_context: bool
    context: PhysiCellMaBoSSContextRecord | None = None


class PhysiCellXmlValidationResult(PhysiCellScientificResult):
    """Validation outcome for an existing PhysiCell XML file."""

    filepath: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    valid: bool
    error_message: str | None = None


class PhysiCellLoadedConfigurationResult(PhysiCellScientificResult):
    """Overview of a configuration loaded from PhysiCell XML."""

    session_id: str = Field(min_length=1)
    source_path: str | None = None
    modification_count: int = Field(ge=0)
    domain: PhysiCellDomainRecord | None = None
    substrates: list[str]
    cell_types: list[str]
    physiboss_models: list[str]
    has_existing_rules: bool
    marked_analyzed: bool


class PhysiCellSubstrateRecord(StructuredOutputModel):
    """One substrate read from a loaded configuration."""

    name: str = Field(min_length=1)
    properties_accessible: bool
    diffusion_coefficient: float | None = None
    decay_rate: float | None = None
    initial_condition: float | None = None


class PhysiCellCellTypeRecord(StructuredOutputModel):
    """One cell definition read from a loaded configuration."""

    name: str = Field(min_length=1)
    properties_accessible: bool
    total_volume: float | None = None
    motility_speed: float | None = None
    cycle_model: str | None = None
    physiboss_enabled: bool


class PhysiCellLoadedComponentsResult(PhysiCellScientificResult):
    """Filtered components and accessible properties from loaded XML."""

    session_id: str = Field(min_length=1)
    component_type: ComponentType
    substrate_count: int = Field(ge=0)
    cell_type_count: int = Field(ge=0)
    physiboss_model_count: int = Field(ge=0)
    substrates: list[PhysiCellSubstrateRecord]
    cell_types: list[PhysiCellCellTypeRecord]
    physiboss_models: list[str]


class PhysiCellCycleModelRecord(StructuredOutputModel):
    """One cell-cycle model accepted by physicell-settings."""

    key: str = Field(min_length=1)
    name: str = Field(min_length=1)


class PhysiCellCycleModelListResult(PhysiCellScientificResult):
    """Available cell-cycle models."""

    model_count: int = Field(ge=0)
    models: list[PhysiCellCycleModelRecord]


class PhysiCellSignalRecord(StructuredOutputModel):
    """One signal accepted by PhysiCell cell rules."""

    name: str = Field(min_length=1)
    signal_type: str
    description: str
    requires: list[str]


class PhysiCellSignalListResult(PhysiCellScientificResult):
    """Available signals after optional configuration-specific expansion."""

    session_id: str | None = None
    scenario_context: str | None = None
    signal_count: int = Field(ge=0)
    signals: list[PhysiCellSignalRecord]


class PhysiCellBehaviorRecord(StructuredOutputModel):
    """One behavior controlled by PhysiCell cell rules."""

    name: str = Field(min_length=1)
    behavior_type: str
    description: str
    requires: list[str]


class PhysiCellBehaviorListResult(PhysiCellScientificResult):
    """Available behaviors after optional configuration-specific expansion."""

    session_id: str | None = None
    scenario_context: str | None = None
    behavior_count: int = Field(ge=0)
    behaviors: list[PhysiCellBehaviorRecord]
