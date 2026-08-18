"""Public input contracts and safety annotations for MaBoSS MCP tools."""

from typing import Annotated, Literal

from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

NonEmptyString = Annotated[
    str,
    Field(
        min_length=1,
        pattern=r".*\S.*",
        description=(
            "A non-empty string containing at least one non-whitespace character."
        ),
    ),
]
MutationState = Literal["ON", "OFF", "WT"]
HandoffArtifactPrefix = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9_-])?$",
        description=(
            "Safe basename prefix for handoff artifacts. Directory components "
            "and a trailing dot are forbidden."
        ),
    ),
]

READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
IDEMPOTENT = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
NON_IDEMPOTENT = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=False,
)
DESTRUCTIVE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
    open_world_hint=False,
)
IDEMPOTENT_DESTRUCTIVE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=True,
    open_world_hint=False,
)


class MaBoSSParameterUpdates(BaseModel):
    """Schema for common MaBoSS parameters with backend extensions."""

    model_config = ConfigDict(extra="allow")

    sample_count: int | None = Field(default=None, ge=1)
    max_time: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    time_tick: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    discrete_time: Literal[0, 1] | None = None
    thread_count: int | None = Field(default=None, ge=1)


InitialStateProbability = Annotated[
    float,
    Field(ge=0, le=1, allow_inf_nan=False),
]
SingleNodeProbabilityList = Annotated[
    list[InitialStateProbability],
    Field(min_length=2, max_length=2),
]
JointStateVector = Annotated[
    list[Literal[0, 1]],
    Field(min_length=1),
]


class MaBoSSJointStateProbability(BaseModel):
    """One JSON-native state/probability entry for a joint distribution."""

    model_config = ConfigDict(extra="forbid")

    state: JointStateVector = Field(
        description=(
            "Boolean state vector in the same order as the requested nodes."
        )
    )
    probability: InitialStateProbability = Field(
        description="Probability assigned to this joint Boolean state."
    )


JointStateProbabilityList = Annotated[
    list[MaBoSSJointStateProbability],
    Field(min_length=1),
]
InitialStateProbabilitySpecification = (
    SingleNodeProbabilityList
    | JointStateProbabilityList
    | dict
)
