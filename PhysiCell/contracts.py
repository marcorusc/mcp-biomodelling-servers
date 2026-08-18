"""Public input contracts and safety annotations for PhysiCell MCP tools."""

from typing import Annotated, Literal

from mcp.types import ToolAnnotations
from pydantic import Field

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
ComponentType = Literal["substrates", "cell_types", "physiboss", "all"]
RuleDirection = Literal["increases", "decreases"]
PhysiBoSSAction = Literal["activation", "inhibition"]
MutationState = Literal[0, 1]
HandoffArtifactPrefix = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9_-])?$",
        description=(
            "Safe basename prefix for copied handoff artifacts. Directory "
            "components and a trailing dot are forbidden."
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
