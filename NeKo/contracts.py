"""Public input contracts and behavioural annotations for NeKo MCP tools."""

from __future__ import annotations

from typing import Annotated, Literal

from mcp.types import ToolAnnotations
from pydantic import BeforeValidator, Field


def _lower_string(value: object) -> object:
    """Normalize values whose public contract is case-insensitive."""
    return value.lower() if isinstance(value, str) else value


NonEmptyString = Annotated[
    str,
    Field(
        min_length=1,
        pattern=r".*\S.*",
        description=(
            "A non-empty string containing at least one non-whitespace "
            "character."
        ),
    ),
]
NonEmptyStringList = Annotated[list[NonEmptyString], Field(min_length=1)]

Database = Literal["omnipath", "signor"]
PathPolicy = Literal["one_shortest", "all_shortest", "all_bounded"]
ReusePolicy = Literal["none", "discovered_paths", "induced_subgraph"]
NormalizedVerbosity = Annotated[
    Literal["summary", "preview", "full"],
    BeforeValidator(_lower_string),
]
NormalizedExportFormat = Annotated[
    Literal["sif", "bnet"],
    BeforeValidator(_lower_string),
]
HandoffArtifactPrefix = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9_-])?$",
        description=(
            "Safe basename prefix for the paired .bnet and .handoff.json "
            "artifacts. Directory components and a trailing dot are forbidden."
        ),
    ),
]
OutputFormat = Literal["markdown", "json"]
NormalizedConnectorMethod = Annotated[
    Literal["hubs", "relax_max_len", "unsigned"],
    BeforeValidator(_lower_string),
]
NormalizedInteractionNodeScope = Annotated[
    Literal["incident", "internal", "boundary"],
    BeforeValidator(_lower_string),
]
NormalizedConnectivityMode = Annotated[
    Literal["weak", "strong"],
    BeforeValidator(_lower_string),
]
BridgeMode = Literal["OUT", "IN", "ALL"]
TargetStrategy = Literal[
    "connect_to_upstream_nodes",
    "connect_subgroup",
]
AtopoStrategy = Literal["radial", "complete"]
GlobalStrategy = Literal[
    "complete_connection",
    "connect_network_radially",
    "connect_as_atopo",
]
RadialDirection = Literal["OUT", "IN"]


READ_ONLY_CLOSED = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
READ_ONLY_OPEN = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=True,
)
IDEMPOTENT_CLOSED = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
NON_IDEMPOTENT_CLOSED = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=False,
)
NON_IDEMPOTENT_OPEN = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=True,
)
DESTRUCTIVE_IDEMPOTENT_CLOSED = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=True,
    open_world_hint=False,
)
DESTRUCTIVE_NON_IDEMPOTENT_CLOSED = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
    open_world_hint=False,
)
DESTRUCTIVE_NON_IDEMPOTENT_OPEN = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
    open_world_hint=True,
)
