"""Validated authoring inputs shared by the server and worker boundary."""

from __future__ import annotations

from typing import Annotated, Literal

from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$", max_length=100)]
NonEmpty = Annotated[str, Field(min_length=1, pattern=r".*\S.*")]
Timeout = Annotated[int, Field(ge=1, le=300)]

READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
WRITE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=False,
)
DELETE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=True,
    open_world_hint=False,
)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class EvidenceRecord(Contract):
    evidence_id: Identifier
    source_identifiers: list[NonEmpty] = Field(min_length=1)
    summary: NonEmpty
    passage: str | None = None
    source_location: str | None = None
    biological_context: str | None = None
    access_limitations: str | None = None
    edge_ids: list[str] = Field(default_factory=list)
    stance: Literal["supports", "contradicts", "context"] = "supports"


class ReactionRecord(Contract):
    reaction_id: Identifier
    statement: NonEmpty
    evidence_ids: list[Identifier] = Field(default_factory=list)
    edge_ids: list[str] = Field(default_factory=list)
    status: Literal["supported", "assumed"]
    assumption: str | None = None
    share_parameters_with: Identifier | None = None

    @model_validator(mode="after")
    def require_justification(self) -> ReactionRecord:
        if self.status == "assumed" and not (
            self.assumption and self.assumption.strip()
        ):
            raise ValueError("Assumed reactions require an explicit rationale.")
        if self.status == "supported" and not self.evidence_ids:
            raise ValueError("Supported reactions require evidence links.")
        if "\n" in self.statement or "\r" in self.statement:
            raise ValueError(
                "A reaction record must contain one statement on one line."
            )
        return self


class LineEvidence(Contract):
    line_number: int = Field(ge=1)
    evidence_ids: list[Identifier] = Field(default_factory=list)
    assumption: str | None = None


class Quantity(Contract):
    value: float = Field(ge=0)
    units: str | None = None
    origin: Literal["supplied", "literature", "assumed", "placeholder"] = "supplied"
    evidence_ids: list[Identifier] = Field(default_factory=list)

    @model_validator(mode="after")
    def literature_source(self) -> Quantity:
        if self.origin == "literature" and not self.evidence_ids:
            raise ValueError("Literature-derived quantities require evidence links.")
        return self


class Condition(Contract):
    name: Identifier
    parameters: dict[Identifier, Quantity] = Field(default_factory=dict)
    initials: dict[Identifier, Quantity] = Field(default_factory=dict)


class ModelConfiguration(Contract):
    observables: dict[Identifier, NonEmpty] = Field(default_factory=dict)
    parameters: dict[Identifier, Quantity] = Field(default_factory=dict)
    initials: dict[Identifier, Quantity] = Field(default_factory=dict)
    conditions: list[Condition] = Field(default_factory=list)
    time_span: tuple[int, int] | None = None
    time_units: str | None = None

    @model_validator(mode="after")
    def bounded_time(self) -> ModelConfiguration:
        if self.time_span is not None:
            start, stop = self.time_span
            if not 0 <= start < stop or stop - start > 10000:
                raise ValueError(
                    "Time span must increase, start at >=0, and contain at most 10001 integer samples."
                )
        names = [c.name for c in self.conditions]
        if len(names) != len(set(names)) or len(names) > 20:
            raise ValueError("Conditions must have unique names (maximum 20).")
        return self


class SimulationScenario(Contract):
    name: Identifier
    parameters: dict[Identifier, Quantity] = Field(default_factory=dict)
    initials: dict[Identifier, Quantity] = Field(default_factory=dict)
    allow_placeholders: bool = False


class GraphOptions(Contract):
    layout: Literal["dot", "neato", "fdp", "circo", "twopi"] = "dot"
    show_controls: bool = False


GRAPH_LIMITATIONS = (
    "BioMASS projects reactants/modifiers onto products; it does not distinguish "
    "reactants from modifiers or activating from inhibiting modifiers. "
    "This is a species projection, not a complete reaction or signed causal graph. "
    "Interactive HTML is not a simulation animation. Unresolved NeKo edges are in the coverage report."
)
