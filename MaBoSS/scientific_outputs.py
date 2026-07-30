"""Validated scientific output contracts for MaBoSS MCP tools."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from mcp_biomodelling_servers.structured_outputs import (
    ArtifactFileSummary,
    StructuredOutputModel,
)

ScientificScalar = str | int | float | bool | None
MutationState = Literal["ON", "OFF", "WT"]


class MaBoSSScientificResult(StructuredOutputModel):
    """Fields shared by MaBoSS scientific and simulation results."""

    server: Literal["MaBoSS"]
    session_id: str = Field(min_length=1)


class MaBoSSNodeListResult(MaBoSSScientificResult):
    """Node names available in the loaded Boolean network."""

    node_count: int = Field(ge=0)
    nodes: list[str]


class MaBoSSStateProbabilityRecord(StructuredOutputModel):
    """Probability assigned to one Boolean state of a node group."""

    state: list[Literal[0, 1]]
    probability: float | str


class MaBoSSInitialStateGroup(StructuredOutputModel):
    """One independent or jointly configured initial-state group."""

    nodes: list[str] = Field(min_length=1)
    probabilities: list[MaBoSSStateProbabilityRecord]


class MaBoSSInitialStateResult(MaBoSSScientificResult):
    """Complete initial-state probability configuration."""

    group_count: int = Field(ge=0)
    groups: list[MaBoSSInitialStateGroup]


class MaBoSSLogicalRuleRecord(StructuredOutputModel):
    """Boolean update rule for one network node."""

    node: str = Field(min_length=1)
    rule: str


class MaBoSSLogicalRulesResult(MaBoSSScientificResult):
    """Logical rules returned by the loaded MaBoSS model."""

    rule_count: int = Field(ge=0)
    rules: list[MaBoSSLogicalRuleRecord]


class MaBoSSMutationRecord(StructuredOutputModel):
    """Mutation state applied to one Boolean node."""

    node: str = Field(min_length=1)
    state: MutationState


class MaBoSSMutationListResult(MaBoSSScientificResult):
    """Mutations currently configured on the loaded simulation."""

    mutation_count: int = Field(ge=0)
    mutations: list[MaBoSSMutationRecord]


class MaBoSSParameterRecord(StructuredOutputModel):
    """Current value of one MaBoSS simulation parameter."""

    name: str = Field(min_length=1)
    value: ScientificScalar


class MaBoSSParameterResult(MaBoSSScientificResult):
    """Parameter inspection or update result."""

    mode: Literal["inspect", "update"]
    parameter_count: int = Field(ge=0)
    parameters: list[MaBoSSParameterRecord]
    updated_parameters: list[str]


class MaBoSSScientificTable(StructuredOutputModel):
    """JSON-safe tabular data preserving numeric scientific values."""

    columns: list[str]
    index_name: str | None = None
    index: list[ScientificScalar]
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    rows: list[list[ScientificScalar]]


class MaBoSSSimulationRunResult(MaBoSSScientificResult):
    """Completion and persisted-artifact metadata for a simulation run."""

    result_available: bool
    trajectory_row_count: int | None = Field(default=None, ge=0)
    trajectory_column_count: int | None = Field(default=None, ge=0)
    result_file: ArtifactFileSummary | None = None


class MaBoSSMutationSimulationResult(MaBoSSScientificResult):
    """Final probability table for a one-off mutant simulation."""

    mutations: list[MaBoSSMutationRecord]
    has_trajectory_data: bool
    trajectory: MaBoSSScientificTable


class MaBoSSSimulationResult(MaBoSSScientificResult):
    """Final state-probability table from the stored simulation result."""

    has_trajectory_data: bool
    trajectory: MaBoSSScientificTable


class MaBoSSTrajectoryPlotResult(MaBoSSScientificResult):
    """Metadata for the PNG trajectory image returned to the client."""

    until: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    time_window: Literal["full", "bounded"]
    image_file: ArtifactFileSummary
