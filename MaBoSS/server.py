import inspect
import logging
import math
import os
import sys
import tempfile
from functools import wraps
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Annotated, Literal

# Make the repo root importable so we can use the shared artifact_manager
sys.path.insert(0, str(Path(__file__).parent.parent))

import io

import anyio
import maboss
import matplotlib.pyplot as plt
import pandas as pd
from mcp.server.mcpserver import Context, Image, MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field
from session_manager import ensure_session, session_manager

from artifact_manager import (
    clean_artifacts,
    get_artifact_dir,
    list_artifacts,
    safe_artifact_path,
    write_session_meta,
)
from artifact_manager import list_artifact_sessions as _list_artifact_sessions_on_disk
from mcp_biomodelling_servers import __version__
from mcp_biomodelling_servers.handoff import (
    HandoffNetwork,
    HandoffPackage,
    HandoffProvenance,
    MaBoSSHandoffExportResult,
    MaBoSSHandoffImportResult,
    MaBoSSSimulationHandoff,
    MaBoSSToPhysiCellHandoffManifest,
    NeKoToMaBoSSHandoffManifest,
    PhysiCellTarget,
    bnet_node_names,
    handoff_artifact,
    load_handoff_manifest,
    verify_handoff_artifact,
    verify_handoff_manifest,
    write_handoff_manifest,
)
from mcp_biomodelling_servers.structured_outputs import (
    ArtifactSessionSummary,
    MaBoSSArtifactCleanupResult,
    MaBoSSArtifactFileListResult,
    MaBoSSArtifactSessionListResult,
    MaBoSSBnetConversionResult,
    MaBoSSModelExportResult,
    MaBoSSSessionListResult,
    MaBoSSSessionSummary,
    artifact_file_summary,
    structured_report,
)
from scientific_outputs import (
    MaBoSSInitialStateGroup,
    MaBoSSInitialStateResult,
    MaBoSSLogicalRuleRecord,
    MaBoSSLogicalRulesResult,
    MaBoSSMutationListResult,
    MaBoSSMutationRecord,
    MaBoSSMutationSimulationResult,
    MaBoSSNodeListResult,
    MaBoSSParameterRecord,
    MaBoSSParameterResult,
    MaBoSSScientificTable,
    MaBoSSSimulationResult,
    MaBoSSSimulationRunResult,
    MaBoSSStateProbabilityRecord,
    MaBoSSTrajectoryPlotResult,
)

logger = logging.getLogger(__name__)

MABOSS_SERVER_INSTRUCTIONS = (
    "Create a session before loading or simulating a Boolean model, and pass "
    "`session_id` explicitly when working with multiple models. Use "
    "`import_neko_handoff` for a typed NeKo transfer, then inspect node names "
    "and restrict output nodes to the smallest biologically meaningful set "
    "before `run_simulation()` to control the exponential state space. Use "
    "`export_maboss_handoff` for a provenance-preserving PhysiCell transfer. "
    "Read `docs://maboss/agent_manual` or use `maboss_workflow_prompt` for the "
    "complete workflow."
)

mcp = MCPServer(
    "MaBoSS",
    title="MaBoSS Boolean Model Simulator",
    description=(
        "Configure, simulate, analyze, and visualize Boolean models with MaBoSS."
    ),
    instructions=MABOSS_SERVER_INSTRUCTIONS,
    version=__version__,
)

_SERVER_ROOT = Path(__file__).parent

NonEmptyString = Annotated[
    str,
    Field(
        min_length=1,
        pattern=r".*\S.*",
        description="A non-empty string containing at least one non-whitespace character.",
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

_READ_ONLY_TOOL = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
_IDEMPOTENT_TOOL = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
_NON_IDEMPOTENT_TOOL = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=False,
)
_DESTRUCTIVE_TOOL = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
    open_world_hint=False,
)
_IDEMPOTENT_DESTRUCTIVE_TOOL = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=True,
    open_world_hint=False,
)


class MaBoSSParameterUpdates(BaseModel):
    """Schema for common MaBoSS parameters with support for backend extensions."""

    model_config = ConfigDict(extra="allow")

    sample_count: int | None = Field(default=None, ge=1)
    max_time: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    time_tick: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    discrete_time: Literal[0, 1] | None = None
    thread_count: int | None = Field(default=None, ge=1)


def _scientific_scalar(value):
    """Convert a dataframe/backend scalar into a strict JSON-safe value."""
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return str(value)


def _scientific_table(df: pd.DataFrame) -> MaBoSSScientificTable:
    """Preserve dataframe values and index without embedding them in Markdown."""
    return MaBoSSScientificTable(
        columns=[str(column) for column in df.columns],
        index_name=(
            str(df.index.name)
            if df.index.name is not None
            else None
        ),
        index=[_scientific_scalar(value) for value in df.index.tolist()],
        row_count=len(df),
        column_count=len(df.columns),
        rows=[
            [_scientific_scalar(value) for value in row]
            for row in df.itertuples(index=False, name=None)
        ],
    )


def _initial_state_groups(initial_state) -> list[MaBoSSInitialStateGroup]:
    """Normalize tuple-keyed pyMaBoSS initial states into JSON records."""
    groups = []
    for binding, distribution in initial_state.items():
        nodes = (
            [str(node) for node in binding]
            if isinstance(binding, (list, tuple))
            else [str(binding)]
        )
        probabilities = []
        for state, probability in distribution.items():
            state_values = (
                [int(value) for value in state]
                if isinstance(state, (list, tuple))
                else [int(state)]
            )
            normalized_probability = _scientific_scalar(probability)
            if isinstance(normalized_probability, bool):
                probability_value = float(normalized_probability)
            elif isinstance(normalized_probability, (int, float)):
                probability_value = float(normalized_probability)
            elif isinstance(normalized_probability, str):
                probability_value = normalized_probability
            else:
                raise ValueError(
                    "Initial-state probabilities cannot be missing."
                )
            probabilities.append(
                MaBoSSStateProbabilityRecord(
                    state=state_values,
                    probability=probability_value,
                )
            )
        groups.append(
            MaBoSSInitialStateGroup(
                nodes=nodes,
                probabilities=probabilities,
            )
        )
    return groups


def _logical_rule_records(logical_rules) -> list[MaBoSSLogicalRuleRecord]:
    """Normalize pyMaBoSS's logical-rule mapping."""
    return [
        MaBoSSLogicalRuleRecord(node=str(node), rule=str(rule))
        for node, rule in logical_rules.items()
    ]


def _mutation_records(mutations) -> list[MaBoSSMutationRecord]:
    """Normalize pyMaBoSS's mutation mapping."""
    return [
        MaBoSSMutationRecord(node=str(node), state=str(state))
        for node, state in mutations.items()
    ]


def _parameter_records(parameters) -> list[MaBoSSParameterRecord]:
    """Normalize current MaBoSS parameters while preserving scalar types."""
    return [
        MaBoSSParameterRecord(
            name=str(name),
            value=_scientific_scalar(value),
        )
        for name, value in parameters.items()
    ]


def _maboss_package_version() -> str:
    """Return the installed pyMaBoSS distribution version for provenance."""
    try:
        return package_version("maboss")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Cannot export a handoff because the installed `maboss` package "
            "version is unavailable."
        ) from exc


def _handoff_parameters(parameters) -> dict[str, bool | int | float | str | None]:
    """Return exact portable scalar parameters for a handoff manifest."""
    normalized = {}
    for raw_name, raw_value in parameters.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError(
                "MaBoSS contains an empty parameter name that cannot be exported."
            )
        if name in normalized:
            raise ValueError(
                f"MaBoSS parameter names collapse to duplicate key {name!r}."
            )

        value = raw_value
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (TypeError, ValueError):
                pass
        if value is None or isinstance(value, (bool, int, str)):
            normalized[name] = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(
                    f"MaBoSS parameter {name!r} must be finite for handoff."
                )
            normalized[name] = value
        else:
            raise ValueError(
                f"MaBoSS parameter {name!r} has unsupported non-scalar type "
                f"{type(value).__name__!r}."
            )
    return normalized


def _require_unused_artifact_paths(paths: list[Path]) -> None:
    """Reject a handoff prefix when any destination already exists."""
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing MaBoSS handoff artifacts: "
            + ", ".join(str(path) for path in existing)
            + ". Choose a different artifact_prefix."
        )


def _link_artifact_without_overwrite(source: Path, destination: Path) -> None:
    """Atomically publish one complete temporary artifact if absent."""
    if not source.is_file():
        raise FileNotFoundError(
            f"Expected temporary handoff artifact was not created: {source}"
        )
    try:
        os.link(source, destination)
    except FileExistsError as exc:
        raise FileExistsError(
            "Refusing to overwrite a MaBoSS handoff artifact created "
            f"concurrently: {destination}"
        ) from exc


def _rollback_artifacts(paths: list[Path]) -> None:
    """Best-effort cleanup for an incomplete multi-file handoff."""
    for path in reversed(paths):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning(
                "Could not roll back incomplete handoff artifact %s",
                path,
                exc_info=True,
            )


def _session_locked(handler):
    """Run a synchronous handler under its session's exclusive lease."""
    signature = inspect.signature(handler)

    @wraps(handler)
    def locked_handler(*args, **kwargs):
        arguments = signature.bind_partial(*args, **kwargs).arguments
        with session_manager.session_scope(arguments.get("session_id")):
            return handler(*args, **kwargs)

    return locked_handler

# ---------------------------------------------------------------------------
# Agent manual — single source of truth for workflows and operating rules.
# Exposed as an MCP prompt (for smart clients) and as a resource (for readers).
# ---------------------------------------------------------------------------

MABOSS_AGENT_MANUAL = """
# MaBoSS Agent Operations Manual

## 1. Recommended Workflow (in order)
1. **Session:** `create_session()` — returns a session_id
2. **Load a model:** Prefer `import_neko_handoff(manifest_path)` for a typed
   NeKo transfer. For a standalone BNET, call
   `bnet_to_bnd_and_cfg(bnet_path)` followed by `build_simulation()`.
3. **Inspect nodes (MANDATORY):** `get_maboss_nodes()` — list ALL valid node names; always do this before any configuration step to avoid referencing non-existent nodes
4. **Inspect parameters:** `update_maboss_parameters()` (no args) — review current defaults
5. **Tune:** `update_maboss_parameters({"sample_count": 1000, "thread_count": 4})`
6. **Reduce output nodes (IMPORTANT):** `set_maboss_output_nodes(["Apoptosis", "Proliferation"])` — restricts the result to only the nodes you care about. Without this, MaBoSS enumerates ALL 2^N Boolean states, which becomes exponentially expensive for large networks (>20 nodes). Always set output nodes to the smallest biologically meaningful subset before running.
7. **Configure (optional):** `get_maboss_initial_state()` to inspect current state, then `set_maboss_initial_state(...)` if non-default probabilities are needed. Only use node names returned by `get_maboss_nodes()`.
8. **Run:** `run_simulation()` — executes the simulation and saves `result.csv` to the artifact directory
9. **Analyse:** `get_simulation_result()` — returns the state probability table as a Markdown table
10. **Visualise:** `visualize_network_trajectories()` — saves a PNG artifact
11. **Mutate:** `simulate_mutation(nodes, state)` — runs a one-off mutant copy
12. **PhysiCell handoff:** `export_maboss_handoff(target_cell_type=...)`
    snapshots the current model, parameters, outputs, optional result, and
    complete NeKo lineage.

> **State space warning:** A network with N nodes produces up to 2^N possible Boolean states.
> Always call `set_maboss_output_nodes` to restrict outputs before `run_simulation`.
> For a 30-node network this reduces the result from >1 billion states to only the states
> of the selected output nodes (typically 2-5 nodes).

## 2. Tool Categories
* **Session management:** `create_session`, `list_sessions`, `set_default_session`, `delete_session`
* **Pipeline:** `import_neko_handoff`, `bnet_to_bnd_and_cfg`, `build_simulation`, `run_simulation`
* **Handoff:** `import_neko_handoff`, `export_maboss_handoff`
* **Inspection (read, no side effects):** `get_maboss_nodes`, `get_maboss_initial_state`, `get_maboss_logical_rules`, `get_maboss_mutations`, `update_maboss_parameters` (no args)
* **Configuration:** `update_maboss_parameters`, `set_maboss_output_nodes`, `set_maboss_initial_state`
* **Analysis:** `get_simulation_result`, `simulate_mutation`, `visualize_network_trajectories`
* **Housekeeping:** `list_generated_files`, `clean_generated_files`

## 4. Key Parameters for `update_maboss_parameters`
| Parameter      | Type  | Description                                  |
| -------------- | ----- | -------------------------------------------- |
| `sample_count` | int   | Trajectories (larger = more precise, slower) |
| `max_time`     | float | Simulation time horizon                      |
| `time_tick`    | float | Discretisation step                          |
| `discrete_time`| int   | 0/1 toggle for discrete time mode            |
| `thread_count` | int   | Parallel threads (environment-dependent)     |

## 5. Critical Rules
* Always call `create_session()` before any simulation tool.
* All file I/O is scoped to `<server>/artifacts/<session_id>/`.
* Pass `session_id` explicitly when running multiple simulations in parallel.
* Call `update_maboss_parameters` with no args to list all valid keys.
* Set `thread_count` early to speed up iteration.
* Keep an imported NeKo manifest and its BNET artifact until the MaBoSS
  handoff has been exported; integrity is rechecked before lineage is emitted.
* `export_maboss_bnd_cfg` is a standalone file export. Use
  `export_maboss_handoff` when PhysiCell needs typed provenance and context.
"""

@mcp.prompt(name="maboss_workflow_prompt",
            description="System prompt and operating manual for the MaBoSS agent.")
def maboss_workflow_prompt() -> str:
    return MABOSS_AGENT_MANUAL


@mcp.resource(
    uri="docs://maboss/agent_manual",
    name="MaBoSS Agent Operations Manual",
    description="Single source of truth for MaBoSS workflows, resources, tool categories, and rules.",
    mime_type="text/markdown",
)
def maboss_agent_manual_resource() -> str:
    return MABOSS_AGENT_MANUAL


# ---------------------------------------------------------------------------
# Read-only resources (URI templates — no side effects)
# ---------------------------------------------------------------------------

@mcp.resource(
    uri="maboss://session/{session_id}/nodes",
    name="Network Nodes",
    description="Comma-separated list of node names in the loaded MaBoSS network.",
    mime_type="text/plain",
)
@_session_locked
def resource_network_nodes(session_id: str) -> str:
    """Return the node names for the given session."""
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise ResourceNotFoundError(
            "No simulation loaded. Call bnet_to_bnd_and_cfg then build_simulation first."
        )
    nodes_list = list(sess.sim.network.keys())
    if not nodes_list:
        return "No nodes found in the MaBoSS network."
    return f"Nodes: {', '.join(nodes_list)}"


@mcp.resource(
    uri="maboss://session/{session_id}/parameters",
    name="Simulation Parameters",
    description="Current MaBoSS simulation parameters as a Markdown table. Use update_maboss_parameters to modify.",
    mime_type="text/markdown",
)
@_session_locked
def resource_parameters(session_id: str) -> str:
    """Return current parameter table for the given session."""
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise ResourceNotFoundError(
            "No simulation loaded. Call bnet_to_bnd_and_cfg then build_simulation first."
        )
    df = pd.DataFrame(
        [[k, v] for k, v in sess.sim.param.items()],
        columns=["parameter", "value"],
    )
    return df.to_markdown(index=False, tablefmt="plain")


@mcp.resource(
    uri="maboss://session/{session_id}/initial_state",
    name="Initial State",
    description="Initial state probability configuration of the MaBoSS simulation.",
    mime_type="text/plain",
)
@_session_locked
def resource_initial_state(session_id: str) -> str:
    """Return the initial state for the given session."""
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise ResourceNotFoundError(
            "No simulation loaded. Call bnet_to_bnd_and_cfg then build_simulation first."
        )
    return str(sess.sim.network.get_istate())


@mcp.resource(
    uri="maboss://session/{session_id}/logical_rules",
    name="Logical Rules",
    description="Boolean logical rules of the MaBoSS network.",
    mime_type="text/plain",
)
@_session_locked
def resource_logical_rules(session_id: str) -> str:
    """Return the logical rules for the given session."""
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise ResourceNotFoundError(
            "No simulation loaded. Call bnet_to_bnd_and_cfg then build_simulation first."
        )
    return str(sess.sim.get_logical_rules())


@mcp.resource(
    uri="maboss://session/{session_id}/mutations",
    name="Mutations",
    description="Mutation settings currently applied to the MaBoSS network.",
    mime_type="text/plain",
)
@_session_locked
def resource_mutations(session_id: str) -> str:
    """Return mutation settings for the given session."""
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise ResourceNotFoundError(
            "No simulation loaded. Call bnet_to_bnd_and_cfg then build_simulation first."
        )
    return str(sess.sim.get_mutations())


@mcp.resource(
    uri="maboss://session/{session_id}/result",
    name="Simulation Result",
    description=(
        "Post-run state probability table as Markdown. "
        "Columns = Boolean states (ON nodes joined by '--'); "
        "rows = final timepoint snapshot (values sum to ~1). "
        "Available only after run_simulation has been called."
    ),
    mime_type="text/markdown",
)
@_session_locked
def resource_simulation_result(session_id: str) -> str:
    """Return the last simulation result for the given session."""
    sess = ensure_session(session_id)
    if sess.result is None:
        raise ResourceNotFoundError(
            "No simulation has been run yet. Call run_simulation first."
        )
    df_prob = sess.result.get_last_states_probtraj()
    if df_prob.empty:
        return "_Simulation completed but returned no trajectory data._"
    df_prob = clean_for_markdown(df_prob)
    md_table = df_prob.to_markdown(index=False, tablefmt="plain")
    return "\n".join([
        "**MaBoSS Simulation: State Probability Trajectory**",
        "",
        md_table,
    ])


@mcp.resource(
    uri="maboss://session/{session_id}/files",
    name="Artifact Files",
    description="List of artifact files (BND, CFG, PNG, …) generated for a session.",
    mime_type="text/markdown",
)
@_session_locked
def resource_generated_files(session_id: str) -> str:
    """Return a Markdown list of artifact files for the given session."""
    sess = ensure_session(session_id)
    files = list_artifacts(_SERVER_ROOT, session_id=sess.session_id)
    if not files:
        return "No artifact files found for this session."
    return "## Artifact files\n\n" + "\n".join(f"- {f}" for f in files)


# ---------------------------------------------------------------------------
# Session management tools
# ---------------------------------------------------------------------------

@mcp.tool(annotations=_NON_IDEMPOTENT_TOOL)
def create_session(
    set_as_default: bool = Field(
        default=True,
        description="When True (default), the new session becomes the active default for subsequent calls.",
    ),
    label: str | None = Field(
        default=None,
        description="Optional human-readable label (e.g. 'TP53-MYC Boolean run'). Stored on disk so the session can be rediscovered after a server restart.",
    ),
) -> str:
    """Create a new MaBoSS session.

    Returns the session ID (UUID) that must be passed to pipeline tools when running
    multiple independent simulations in parallel.
    """
    with session_manager.create_session_scope(
        set_as_default=set_as_default,
    ) as session:
        sid = session.session_id
        write_session_meta(_SERVER_ROOT, sid, server_name="MaBoSS", label=label)
    label_info = f" ({label})" if label else ""
    return f"Session created: {sid}{label_info}" + (" (set as default)" if set_as_default else "")


@mcp.tool(annotations=_READ_ONLY_TOOL)
def list_sessions() -> Annotated[CallToolResult, MaBoSSSessionListResult]:
    """List all active MaBoSS sessions with their simulation and result status."""
    sessions = session_manager.list_sessions()
    payload = MaBoSSSessionListResult(
        server="MaBoSS",
        count=len(sessions),
        sessions=[
            MaBoSSSessionSummary(
                session_id=sid,
                created_at=info["created_at"],
                last_accessed=info["last_accessed"],
                is_default=info["is_default"],
                has_simulation=info["has_simulation"],
                has_result=info["has_result"],
                bnd_path=info["bnd_path"],
                cfg_path=info["cfg_path"],
                upstream_neko_manifest_path=info[
                    "upstream_neko_manifest_path"
                ],
            )
            for sid, info in sessions.items()
        ],
    )
    if not sessions:
        return structured_report(
            "No active sessions. Call create_session() to start one.",
            payload,
        )
    lines = ["## MaBoSS Sessions\n"]
    for sid, info in sessions.items():
        default_marker = " **(default)**" if info["is_default"] else ""
        has_sim = "✓" if info["has_simulation"] else "✗"
        has_res = "✓" if info["has_result"] else "✗"
        lines.append(
            f"- **{sid}**{default_marker}: sim={has_sim}  result={has_res}  "
            f"bnd={info['bnd_path'] or '—'}  "
            "NeKo lineage="
            f"{info['upstream_neko_manifest_path'] or '—'}"
        )
    return structured_report("\n".join(lines), payload)


@mcp.tool(annotations=_READ_ONLY_TOOL)
def list_artifact_sessions(
) -> Annotated[CallToolResult, MaBoSSArtifactSessionListResult]:
    """List all MaBoSS sessions that have artifact files on disk (including past server runs).

    Unlike list_sessions() which only shows in-memory sessions, this scans the
    artifacts/ directory and reads session_meta.json files, so previously created
    sessions are visible even after a server restart.

    Use the returned session_id and file paths to resume earlier work, e.g.:
      build_simulation(bnd_path='/path/to/artifacts/<uuid>/output.bnd',
                       cfg_path='/path/to/artifacts/<uuid>/output.cfg')
    """
    sessions = _list_artifact_sessions_on_disk(_SERVER_ROOT, server_name="MaBoSS")
    payload = MaBoSSArtifactSessionListResult(
        server="MaBoSS",
        count=len(sessions),
        sessions=[
            ArtifactSessionSummary(
                session_id=str(session["session_id"]),
                server=str(session.get("server") or "unknown"),
                label=str(session["label"]) if session.get("label") else None,
                created_at=(
                    str(session["created_at"])
                    if session.get("created_at")
                    else None
                ),
                files=[str(filename) for filename in session.get("files", [])],
            )
            for session in sessions
        ],
    )
    if not sessions:
        return structured_report("No artifact sessions found on disk.", payload)
    lines = ["## MaBoSS Artifact Sessions (on disk)\n"]
    for s in sessions:
        sid = s["session_id"]
        label = s.get("label") or ""
        created = s.get("created_at", "")[:19].replace("T", " ")
        files = s.get("files", [])
        lines.append(f"- **{sid}**" + (f" ({label})" if label else ""))
        if created:
            lines.append(f"  Created: {created} UTC")
        if files:
            lines.append(f"  Files: {', '.join(files)}")
        else:
            lines.append("  Files: (none)")
    return structured_report("\n".join(lines), payload)


@mcp.tool(annotations=_IDEMPOTENT_TOOL)
def set_default_session(
    session_id: Annotated[
        NonEmptyString,
        Field(description="ID of the session to set as the active default."),
    ],
) -> str:
    """Set the default (active) MaBoSS session used when session_id is omitted in other tools."""
    if session_manager.set_default(session_id):
        return f"Default session set to: {session_id}"
    raise ValueError(f"Session not found: {session_id}")


@mcp.tool(annotations=_DESTRUCTIVE_TOOL)
def delete_session(
    session_id: Annotated[
        NonEmptyString,
        Field(description="ID of the session to delete."),
    ],
    clean_files: bool = Field(
        default=True,
        description="When True (default), also remove all artifact files for this session.",
    ),
) -> str:
    """Delete a MaBoSS session and optionally its artifact files."""
    session = session_manager.get_session(session_id)
    if session is None or not session_manager.delete_session(session_id):
        raise ValueError(f"Session not found: {session_id}")

    removed_files = (
        clean_artifacts(_SERVER_ROOT, session.session_id)
        if clean_files
        else 0
    )
    return f"Session {session.session_id} deleted." + (
        f" Removed {removed_files} artifact file(s)." if clean_files else ""
    )


# ---------------------------------------------------------------------------
# Pipeline tools
# ---------------------------------------------------------------------------

@mcp.tool(annotations=_NON_IDEMPOTENT_TOOL)
@_session_locked
def import_neko_handoff(
    manifest_path: Annotated[
        NonEmptyString,
        Field(
            description=(
                "Path to a NeKo `neko-to-maboss` handoff manifest. Its BNET "
                "artifact and integrity metadata are verified before import."
            )
        ),
    ],
    artifact_prefix: HandoffArtifactPrefix = Field(
        default="neko_import",
        description=(
            "Safe prefix for the imported MaBoSS BND and CFG artifacts. "
            "Choose a new prefix for every retained import."
        ),
    ),
    session_id: NonEmptyString | None = Field(
        default=None,
        description=(
            "Session to replace only after a complete successful import. "
            "Omit to use the active default session."
        ),
    ),
) -> Annotated[CallToolResult, MaBoSSHandoffImportResult]:
    """Verify, convert, and atomically load a typed NeKo handoff."""
    sess = ensure_session(session_id)
    loaded_manifest = load_handoff_manifest(
        manifest_path,
        expected_handoff_type="neko-to-maboss",
        verify_artifacts=True,
    )
    if not isinstance(loaded_manifest, NeKoToMaBoSSHandoffManifest):
        raise ValueError(
            "The supplied handoff is not a NeKo-to-MaBoSS manifest."
        )

    source_manifest_path = Path(manifest_path).resolve()
    source_manifest_file = handoff_artifact(
        source_manifest_path,
        server="NeKo",
        session_id=loaded_manifest.source.session_id,
        role="parent_manifest",
    )
    bnet_path = Path(loaded_manifest.bnet_file.path)
    stored_nodes = bnet_node_names(bnet_path)
    if stored_nodes != loaded_manifest.network.nodes:
        raise ValueError(
            "The BNET target order does not match the NeKo handoff manifest."
        )

    art_dir = get_artifact_dir(_SERVER_ROOT, sess.session_id)
    bnd_path = safe_artifact_path(art_dir, f"{artifact_prefix}.bnd")
    cfg_path = safe_artifact_path(art_dir, f"{artifact_prefix}.cfg")
    _require_unused_artifact_paths([bnd_path, cfg_path])

    created_paths: list[Path] = []
    try:
        with tempfile.TemporaryDirectory(
            dir=art_dir,
            prefix=".neko-handoff-import-",
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            temporary_bnd = temporary_root / "model.bnd"
            temporary_cfg = temporary_root / "model.cfg"
            try:
                maboss.bnet_to_bnd_and_cfg(
                    str(bnet_path),
                    str(temporary_bnd),
                    str(temporary_cfg),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Error converting the verified NeKo BNET: {exc}"
                ) from exc

            for path, label in (
                (temporary_bnd, "BND"),
                (temporary_cfg, "CFG"),
            ):
                if not path.is_file():
                    raise FileNotFoundError(
                        f"MaBoSS conversion did not create the {label} file."
                    )

            try:
                candidate_simulation = maboss.load(
                    str(temporary_bnd),
                    str(temporary_cfg),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Error loading the converted MaBoSS simulation: {exc}"
                ) from exc
            if not candidate_simulation:
                raise RuntimeError(
                    "maboss.load returned no simulation for the verified "
                    "NeKo handoff."
                )

            loaded_nodes = [
                str(node)
                for node in candidate_simulation.network.keys()
            ]
            if (
                len(loaded_nodes) != len(set(loaded_nodes))
                or set(loaded_nodes) != set(stored_nodes)
            ):
                raise ValueError(
                    "Converted MaBoSS nodes do not match the verified NeKo "
                    "BNET nodes."
                )

            output_nodes = list(loaded_manifest.network.output_nodes)
            try:
                candidate_simulation.network.set_output(output_nodes)
            except Exception as exc:
                raise ValueError(
                    "Could not apply the NeKo-declared MaBoSS output nodes: "
                    f"{exc}"
                ) from exc
            applied_outputs = [
                str(node)
                for node in candidate_simulation.network.get_output()
            ]
            if (
                len(applied_outputs) != len(set(applied_outputs))
                or set(applied_outputs) != set(output_nodes)
            ):
                raise ValueError(
                    "MaBoSS did not retain the output-node selection declared "
                    "by the NeKo manifest."
                )

            verify_handoff_artifact(source_manifest_file)
            verify_handoff_artifact(loaded_manifest.bnet_file)
            _link_artifact_without_overwrite(temporary_bnd, bnd_path)
            created_paths.append(bnd_path)
            _link_artifact_without_overwrite(temporary_cfg, cfg_path)
            created_paths.append(cfg_path)

        bnd_file = handoff_artifact(
            bnd_path,
            server="MaBoSS",
            session_id=sess.session_id,
            role="maboss_bnd",
        )
        cfg_file = handoff_artifact(
            cfg_path,
            server="MaBoSS",
            session_id=sess.session_id,
            role="maboss_cfg",
        )
        payload = MaBoSSHandoffImportResult(
            server="MaBoSS",
            session_id=sess.session_id,
            source_manifest_file=source_manifest_file,
            source_manifest=loaded_manifest,
            bnd_file=bnd_file,
            cfg_file=cfg_file,
            nodes=stored_nodes,
            output_nodes=output_nodes,
            requires_output_selection=not output_nodes,
        )
    except Exception:
        _rollback_artifacts(created_paths)
        raise

    sess.set_simulation(
        candidate_simulation,
        str(bnd_path),
        str(cfg_path),
        upstream_neko_manifest_path=str(source_manifest_path),
    )
    output_guidance = (
        "Applied outputs: " + ", ".join(output_nodes)
        if output_nodes
        else (
            "No outputs were declared. All nodes were marked internal; call "
            "set_maboss_output_nodes() before run_simulation()."
        )
    )
    text = (
        "NeKo handoff imported into MaBoSS successfully.\n"
        f"  Session: {sess.session_id}\n"
        f"  Source manifest: {source_manifest_path}\n"
        f"  BND: {bnd_path}\n"
        f"  CFG: {cfg_path}\n"
        f"  Boolean nodes: {len(stored_nodes)}\n"
        f"  {output_guidance}"
    )
    return structured_report(text, payload)


@mcp.tool(annotations=_IDEMPOTENT_TOOL)
@_session_locked
def bnet_to_bnd_and_cfg(
    bnet_path: Annotated[
        NonEmptyString,
        Field(description="Absolute or CWD-relative path to the .bnet file to convert."),
    ],
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to write the output files into. Omit to use the active default session.",
    ),
) -> Annotated[CallToolResult, MaBoSSBnetConversionResult]:
    """Convert a BNET file to MaBoSS BND and CFG files.

    Output files are written to the session artifact directory
    (<server>/artifacts/<session_id>/output.bnd and output.cfg).
    After conversion, call build_simulation() to load the simulation.
    """
    sess = ensure_session(session_id)
    art_dir = get_artifact_dir(_SERVER_ROOT, sess.session_id)
    bnd_out = str(safe_artifact_path(art_dir, "output.bnd"))
    cfg_out = str(safe_artifact_path(art_dir, "output.cfg"))

    logger.info("Converting %s -> %s, %s", bnet_path, bnd_out, cfg_out)
    try:
        maboss.bnet_to_bnd_and_cfg(bnet_path, bnd_out, cfg_out)
    except Exception as e:
        logger.exception("bnet_to_bnd_and_cfg failed")
        raise RuntimeError(f"Error converting .bnet file: {e}") from e

    for path, label in [(bnd_out, "BND"), (cfg_out, "CFG")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected {label} file was not created at {path}."
            )

    logger.info("BND and CFG files created: %s, %s", bnd_out, cfg_out)
    text = (
        f"MaBoSS .bnd and .cfg files created successfully.\n"
        f"  BND: {bnd_out}\n"
        f"  CFG: {cfg_out}\n\n"
        f"Next: call build_simulation(session_id='{sess.session_id}') to load the simulation."
    )
    payload = MaBoSSBnetConversionResult(
        server="MaBoSS",
        session_id=sess.session_id,
        input_bnet_path=bnet_path,
        bnd_file=artifact_file_summary(bnd_out, session_id=sess.session_id),
        cfg_file=artifact_file_summary(cfg_out, session_id=sess.session_id),
    )
    return structured_report(text, payload)


@mcp.tool(annotations=_IDEMPOTENT_TOOL)
@_session_locked
def build_simulation(
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to load the simulation into. Omit to use the active default session.",
    ),
    bnd_path: NonEmptyString | None = Field(
        default=None,
        description="Path to the .bnd file. Omit to use the file generated by bnet_to_bnd_and_cfg for this session.",
    ),
    cfg_path: NonEmptyString | None = Field(
        default=None,
        description="Path to the .cfg file. Omit to use the file generated by bnet_to_bnd_and_cfg for this session.",
    ),
) -> str:
    """Load a MaBoSS simulation from BND and CFG files into the session.

    When bnd_path/cfg_path are omitted, the files produced by the last
    bnet_to_bnd_and_cfg call for this session are used automatically.
    After loading, inspect parameters via the maboss://session/{id}/parameters
    resource, tune with update_maboss_parameters, then call run_simulation.
    """
    sess = ensure_session(session_id)
    art_dir = get_artifact_dir(_SERVER_ROOT, sess.session_id)

    if bnd_path is None:
        bnd_path = str(art_dir / "output.bnd")
    if cfg_path is None:
        cfg_path = str(art_dir / "output.cfg")

    logger.info("Loading MaBoSS simulation: BND=%s CFG=%s", bnd_path, cfg_path)

    try:
        loaded_sim = maboss.load(bnd_path, cfg_path)
    except Exception as e:
        logger.exception("Failed to load simulation")
        raise RuntimeError(f"Error loading MaBoSS simulation: {e}") from e

    if loaded_sim:
        sess.set_simulation(loaded_sim, bnd_path, cfg_path)
        logger.info("MaBoSS simulation loaded successfully")
        parameters_str = "\n".join(f"{k}: {v}" for k, v in loaded_sim.param.items())
        return (
            f"MaBoSS simulation loaded successfully.\n{parameters_str}\n\n"
            f"NEXT STEP: call get_maboss_nodes() to retrieve the list of valid node "
            f"names before calling set_maboss_output_nodes() or set_maboss_initial_state()."
        )
    else:
        logger.error("maboss.load returned None")
        raise RuntimeError(
            "maboss.load returned None. Check the BND and CFG files."
        )


@mcp.tool(annotations=_NON_IDEMPOTENT_TOOL)
async def run_simulation(
    ctx: Context,
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to run. Omit to use the active default session.",
    ),
) -> Annotated[CallToolResult, MaBoSSSimulationRunResult]:
    """Execute the loaded MaBoSS simulation and store the result in the session.

    IMPORTANT: Call set_maboss_output_nodes() before this tool. Without it, MaBoSS becomes exponentially expensive. 
    Restricting to a small set of output nodes keeps the run time and result size manageable.

    Tune performance via update_maboss_parameters (sample_count, thread_count)
    before running large simulations. After completion, read the result from
    maboss://session/{id}/result.
    """
    await ctx.report_progress(0, 2)

    def run_locked():
        with session_manager.session_scope(session_id):
            sess = ensure_session(session_id)
            if sess.sim is None:
                raise RuntimeError(
                    "No MaBoSS simulation has been built yet. "
                    "Call bnet_to_bnd_and_cfg then build_simulation first."
                )
            simulation_network = getattr(sess.sim, "network", None)
            get_output = getattr(simulation_network, "get_output", None)
            if callable(get_output) and not list(get_output()):
                raise RuntimeError(
                    "No MaBoSS output nodes are selected. Call "
                    "set_maboss_output_nodes() with a small biologically "
                    "meaningful set before run_simulation()."
                )
            try:
                logger.info("Running MaBoSS simulation")
                run_result = sess.sim.run()
            except Exception as e:
                logger.exception("Error during MaBoSS simulation run")
                raise RuntimeError(
                    f"Error during MaBoSS simulation run: {e}"
                ) from e
            sess.set_result(run_result)

            # Persist the result table so list_generated_files shows it.
            row_count = None
            column_count = None
            saved_csv_path = None
            try:
                art_dir = get_artifact_dir(_SERVER_ROOT, sess.session_id)
                csv_path = safe_artifact_path(art_dir, "result.csv")
                df_result = run_result.get_last_states_probtraj()
                row_count = len(df_result)
                column_count = len(df_result.columns)
                if not df_result.empty:
                    df_result.to_csv(csv_path, index=False)
                    saved_csv_path = csv_path
                    logger.info("Result saved to %s", csv_path)
            except Exception as csv_err:
                logger.warning(
                    "Could not save result CSV: %s",
                    csv_err,
                    exc_info=True,
                )
            return (
                sess.session_id,
                row_count,
                column_count,
                saved_csv_path,
            )

    resolved_session_id, row_count, column_count, csv_path = (
        await anyio.to_thread.run_sync(run_locked)
    )
    await ctx.report_progress(2, 2)
    logger.info("MaBoSS simulation completed successfully")
    text = (
        "MaBoSS simulation completed successfully.\n"
        "Call `get_simulation_result()` to read the state probability table.\n"
        + (
            "The result is also saved to the session artifact directory as result.csv."
            if csv_path is not None
            else "No non-empty result table was written to result.csv."
        )
    )
    payload = MaBoSSSimulationRunResult(
        server="MaBoSS",
        session_id=resolved_session_id,
        result_available=True,
        trajectory_row_count=row_count,
        trajectory_column_count=column_count,
        result_file=(
            artifact_file_summary(
                csv_path,
                session_id=resolved_session_id,
            )
            if csv_path is not None
            else None
        ),
    )
    return structured_report(text, payload)

@mcp.tool(annotations=_NON_IDEMPOTENT_TOOL)
@_session_locked
def export_maboss_bnd_cfg(
    prefix: Annotated[NonEmptyString, Field(
        default="updated",
        description=(
            "Prefix for output filenames written to the session artifact directory. "
            "Produces '<prefix>.bnd' and '<prefix>.cfg'. Example: 'run2' -> run2.bnd/run2.cfg."
        ),
    )] = "updated",
    overwrite: Annotated[bool, Field(
        default=False,
        description="If True, overwrite existing files with the same names in the artifact directory.",
    )] = False,
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to export from. Omit to use the active default session.",
    ),
) -> Annotated[CallToolResult, MaBoSSModelExportResult]:
    """Export the current in-memory MaBoSS simulation to .bnd and .cfg files.

    Writes files to: <server>/artifacts/<session_id>/<prefix>.bnd and <prefix>.cfg
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise RuntimeError(
            "No MaBoSS simulation has been built yet. Call build_simulation first."
        )

    try:
        prefix = (prefix or "").strip()
        if not prefix:
            raise ValueError("Invalid prefix. Must be a non-empty string.")

        # Optionally normalize prefix a bit (avoid spaces)
        prefix = prefix.replace(" ", "_")

        art_dir = get_artifact_dir(_SERVER_ROOT, sess.session_id)
        bnd_name = f"{prefix}.bnd"
        cfg_name = f"{prefix}.cfg"

        bnd_out = str(safe_artifact_path(art_dir, bnd_name))
        cfg_out = str(safe_artifact_path(art_dir, cfg_name))

        if not overwrite:
            for p in (bnd_out, cfg_out):
                if os.path.exists(p):
                    raise FileExistsError(
                        f"Refusing to overwrite existing file: {p}\n"
                        f"Choose a different prefix or set overwrite=True."
                    )

        logger.info("Exporting MaBoSS model -> %s, %s", bnd_out, cfg_out)

        # Write .bnd
        with open(bnd_out, "w") as fbnd:
            sess.sim.print_bnd(out=fbnd)

        # Write .cfg
        with open(cfg_out, "w") as fcfg:
            sess.sim.print_cfg(out=fcfg)

        # Sanity check
        for path, label in [(bnd_out, "BND"), (cfg_out, "CFG")]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Expected {label} file was not created at {path}."
                )

        logger.info("Export complete: %s, %s", bnd_out, cfg_out)
        text = (
            f"Exported current MaBoSS model successfully.\n"
            f"  BND: {bnd_out}\n"
            f"  CFG: {cfg_out}"
        )
        payload = MaBoSSModelExportResult(
            server="MaBoSS",
            session_id=sess.session_id,
            prefix=prefix,
            overwrite=overwrite,
            bnd_file=artifact_file_summary(
                bnd_out,
                session_id=sess.session_id,
            ),
            cfg_file=artifact_file_summary(
                cfg_out,
                session_id=sess.session_id,
            ),
        )
        return structured_report(text, payload)

    except (ValueError, FileExistsError, FileNotFoundError):
        raise
    except Exception as e:
        logger.exception("Error exporting MaBoSS model")
        raise RuntimeError(f"Error exporting MaBoSS model: {e}") from e


@mcp.tool(annotations=_NON_IDEMPOTENT_TOOL)
@_session_locked
def export_maboss_handoff(
    target_cell_type: Annotated[
        NonEmptyString,
        Field(
            description=(
                "PhysiCell cell type intended to receive this Boolean model."
            )
        ),
    ],
    biological_context: NonEmptyString | None = Field(
        default=None,
        description=(
            "Biological context for PhysiCell integration. Omit to inherit "
            "the context of an imported NeKo handoff; required for standalone "
            "MaBoSS models."
        ),
    ),
    simulation_summary: NonEmptyString | None = Field(
        default=None,
        description=(
            "Optional scientific interpretation of the MaBoSS result. When "
            "omitted, a concise table-availability summary is generated."
        ),
    ),
    include_result: bool = Field(
        default=True,
        description=(
            "Snapshot the stored state-probability table as a CSV artifact "
            "when non-empty simulation results are available."
        ),
    ),
    artifact_prefix: HandoffArtifactPrefix = Field(
        default="maboss_to_physicell",
        description=(
            "Safe prefix for the BND, CFG, optional result CSV, and manifest. "
            "Choose a new prefix for every retained handoff."
        ),
    ),
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to export; omit to use the active default session.",
    ),
) -> Annotated[CallToolResult, MaBoSSHandoffExportResult]:
    """Export an integrity-protected MaBoSS-to-PhysiCell handoff."""
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise RuntimeError(
            "No MaBoSS simulation has been built yet. Import a NeKo handoff "
            "or call build_simulation first."
        )

    nodes = [str(node) for node in sess.sim.network.keys()]
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValueError(
            "The loaded MaBoSS simulation must contain unique Boolean nodes."
        )
    output_nodes = [
        str(node)
        for node in sess.sim.network.get_output()
    ]
    if not output_nodes:
        raise RuntimeError(
            "No MaBoSS output nodes are selected. Call "
            "set_maboss_output_nodes() with a small biologically meaningful "
            "set before exporting a PhysiCell handoff."
        )
    if len(output_nodes) != len(set(output_nodes)):
        raise ValueError("MaBoSS output nodes contain duplicate names.")
    unknown_outputs = sorted(set(output_nodes) - set(nodes))
    if unknown_outputs:
        raise ValueError(
            "MaBoSS output nodes are absent from the loaded network: "
            + ", ".join(unknown_outputs)
        )

    parent_manifest = None
    parent_manifest_file = None
    lineage = []
    inherited_context = None
    renamed_nodes: list[str] = []
    node_renames: dict[str, str] = {}
    duplicate_rules_removed: list[str] = []
    if sess.upstream_neko_manifest_path is not None:
        loaded_parent = load_handoff_manifest(
            sess.upstream_neko_manifest_path,
            expected_handoff_type="neko-to-maboss",
            verify_artifacts=True,
        )
        if not isinstance(loaded_parent, NeKoToMaBoSSHandoffManifest):
            raise ValueError(
                "The session's upstream handoff is not a NeKo manifest."
            )
        parent_manifest = loaded_parent
        parent_manifest_file = handoff_artifact(
            sess.upstream_neko_manifest_path,
            server="NeKo",
            session_id=loaded_parent.source.session_id,
            role="parent_manifest",
        )
        lineage = [loaded_parent.source]
        inherited_context = loaded_parent.biological_context
        renamed_nodes = list(loaded_parent.network.renamed_nodes)
        node_renames = dict(loaded_parent.network.node_renames)
        duplicate_rules_removed = list(
            loaded_parent.network.duplicate_rules_removed
        )

    context = (
        biological_context.strip()
        if biological_context is not None
        else inherited_context
    )
    if not context:
        raise ValueError(
            "biological_context is required for a standalone MaBoSS model "
            "because no NeKo context is available to inherit."
        )

    parameters = _handoff_parameters(sess.sim.param)
    result_table = None
    result_row_count = 0
    result_column_count = 0
    if sess.result is not None:
        try:
            result_table = sess.result.get_last_states_probtraj()
        except Exception as exc:
            raise RuntimeError(
                f"Could not read the stored MaBoSS simulation result: {exc}"
            ) from exc
        if not isinstance(result_table, pd.DataFrame):
            raise ValueError(
                "The stored MaBoSS result did not return a pandas DataFrame."
            )
        result_row_count = len(result_table)
        result_column_count = len(result_table.columns)

    summary = (
        simulation_summary.strip()
        if simulation_summary is not None
        else (
            "Stored state-probability table contains "
            f"{result_row_count} row(s) and {result_column_count} column(s)."
            if result_table is not None
            else "No MaBoSS simulation result was stored at export time."
        )
    )

    art_dir = get_artifact_dir(_SERVER_ROOT, sess.session_id)
    bnd_path = safe_artifact_path(art_dir, f"{artifact_prefix}.bnd")
    cfg_path = safe_artifact_path(art_dir, f"{artifact_prefix}.cfg")
    result_path = safe_artifact_path(
        art_dir,
        f"{artifact_prefix}.result.csv",
    )
    manifest_path = safe_artifact_path(
        art_dir,
        f"{artifact_prefix}.handoff.json",
    )
    include_result_artifact = bool(
        include_result
        and result_table is not None
        and not result_table.empty
    )
    destinations = [bnd_path, cfg_path, manifest_path]
    if include_result_artifact:
        destinations.append(result_path)
    _require_unused_artifact_paths(destinations)

    created_paths: list[Path] = []
    try:
        with tempfile.TemporaryDirectory(
            dir=art_dir,
            prefix=".maboss-handoff-export-",
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            temporary_bnd = temporary_root / "model.bnd"
            temporary_cfg = temporary_root / "model.cfg"
            temporary_result = temporary_root / "result.csv"
            try:
                with temporary_bnd.open("w", encoding="utf-8") as bnd_file:
                    sess.sim.print_bnd(out=bnd_file)
                with temporary_cfg.open("w", encoding="utf-8") as cfg_file:
                    sess.sim.print_cfg(out=cfg_file)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not snapshot the current MaBoSS model: {exc}"
                ) from exc

            if include_result_artifact:
                try:
                    result_table.to_csv(temporary_result, index=False)
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not snapshot the MaBoSS result table: {exc}"
                    ) from exc

            _link_artifact_without_overwrite(temporary_bnd, bnd_path)
            created_paths.append(bnd_path)
            _link_artifact_without_overwrite(temporary_cfg, cfg_path)
            created_paths.append(cfg_path)
            if include_result_artifact:
                _link_artifact_without_overwrite(
                    temporary_result,
                    result_path,
                )
                created_paths.append(result_path)

        bnd_file = handoff_artifact(
            bnd_path,
            server="MaBoSS",
            session_id=sess.session_id,
            role="maboss_bnd",
        )
        cfg_file = handoff_artifact(
            cfg_path,
            server="MaBoSS",
            session_id=sess.session_id,
            role="maboss_cfg",
        )
        result_file = (
            handoff_artifact(
                result_path,
                server="MaBoSS",
                session_id=sess.session_id,
                role="maboss_result",
            )
            if include_result_artifact
            else None
        )
        manifest = MaBoSSToPhysiCellHandoffManifest(
            source=HandoffProvenance(
                server="MaBoSS",
                session_id=sess.session_id,
                mcp_package=HandoffPackage(
                    name="mcp-biomodelling-servers",
                    version=__version__,
                ),
                modelling_package=HandoffPackage(
                    name="maboss",
                    version=_maboss_package_version(),
                ),
                operation="export_maboss_handoff",
            ),
            lineage=lineage,
            biological_context=context,
            network=HandoffNetwork(
                nodes=nodes,
                output_nodes=output_nodes,
                renamed_nodes=renamed_nodes,
                node_renames=node_renames,
                duplicate_rules_removed=duplicate_rules_removed,
            ),
            bnd_file=bnd_file,
            cfg_file=cfg_file,
            parent_manifest=parent_manifest_file,
            simulation=MaBoSSSimulationHandoff(
                parameters=parameters,
                simulation_summary=summary,
                result_file=result_file,
            ),
            target=PhysiCellTarget(cell_type=target_cell_type),
        )
        if parent_manifest is not None:
            verify_handoff_artifact(parent_manifest_file)
            verify_handoff_manifest(parent_manifest)
        write_handoff_manifest(manifest_path, manifest)
        created_paths.append(manifest_path)
        manifest_file = handoff_artifact(
            manifest_path,
            server="MaBoSS",
            session_id=sess.session_id,
            role="parent_manifest",
        )
        payload = MaBoSSHandoffExportResult(
            server="MaBoSS",
            session_id=sess.session_id,
            manifest_file=manifest_file,
            manifest=manifest,
        )
    except Exception:
        _rollback_artifacts(created_paths)
        raise

    lineage_text = (
        f"NeKo session {parent_manifest.source.session_id}"
        if parent_manifest is not None
        else "standalone MaBoSS model"
    )
    result_text = (
        str(result_path)
        if include_result_artifact
        else "not included"
    )
    text = (
        "MaBoSS-to-PhysiCell handoff exported successfully.\n"
        f"  Manifest: {manifest_path}\n"
        f"  BND: {bnd_path}\n"
        f"  CFG: {cfg_path}\n"
        f"  Result CSV: {result_text}\n"
        f"  Boolean nodes: {len(nodes)}\n"
        f"  Output nodes: {', '.join(output_nodes)}\n"
        f"  Target cell type: {target_cell_type}\n"
        f"  Lineage: {lineage_text}\n\n"
        "Next: pass the manifest path to the PhysiCell handoff import tool."
    )
    return structured_report(text, payload)


# ---------------------------------------------------------------------------
# Inspection tools (read-only, no side effects)
# ---------------------------------------------------------------------------

@mcp.tool(annotations=_READ_ONLY_TOOL)
@_session_locked
def get_maboss_nodes(
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to query. Omit to use the active default session.",
    ),
) -> Annotated[CallToolResult, MaBoSSNodeListResult]:
    """Return the list of node names in the loaded MaBoSS network.

    Always call this after build_simulation() and before set_maboss_output_nodes()
    or set_maboss_initial_state() to avoid referencing non-existent nodes.
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise RuntimeError(
            "No simulation loaded. Call bnet_to_bnd_and_cfg then build_simulation first."
        )
    nodes_list = list(sess.sim.network.keys())
    text = (
        "No nodes found in the MaBoSS network."
        if not nodes_list
        else "Network nodes:\n" + "\n".join(f"- {n}" for n in nodes_list)
    )
    payload = MaBoSSNodeListResult(
        server="MaBoSS",
        session_id=sess.session_id,
        node_count=len(nodes_list),
        nodes=[str(node) for node in nodes_list],
    )
    return structured_report(text, payload)


@mcp.tool(annotations=_READ_ONLY_TOOL)
@_session_locked
def get_maboss_initial_state(
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to query. Omit to use the active default session.",
    ),
) -> Annotated[CallToolResult, MaBoSSInitialStateResult]:
    """Return the current initial state probability configuration of the MaBoSS simulation.

    Use this to inspect the state before calling set_maboss_initial_state().
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise RuntimeError(
            "No simulation loaded. Call bnet_to_bnd_and_cfg then build_simulation first."
        )
    try:
        initial_state = sess.sim.network.get_istate()
        groups = _initial_state_groups(initial_state)
        payload = MaBoSSInitialStateResult(
            server="MaBoSS",
            session_id=sess.session_id,
            group_count=len(groups),
            groups=groups,
        )
        return structured_report(f"Initial state:\n{initial_state}", payload)
    except Exception as e:
        raise RuntimeError(f"Error retrieving initial state: {e}") from e

@mcp.tool(annotations=_READ_ONLY_TOOL)
@_session_locked
def get_maboss_logical_rules(
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to query. Omit to use the active default session.",
    ),
) -> Annotated[CallToolResult, MaBoSSLogicalRulesResult]:
    """Return the Boolean logical rules of the loaded MaBoSS network."""
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise RuntimeError(
            "No simulation loaded. Call bnet_to_bnd_and_cfg then build_simulation first."
        )
    try:
        logical_rules = sess.sim.get_logical_rules()
        rules = _logical_rule_records(logical_rules)
        payload = MaBoSSLogicalRulesResult(
            server="MaBoSS",
            session_id=sess.session_id,
            rule_count=len(rules),
            rules=rules,
        )
        return structured_report(str(logical_rules), payload)
    except Exception as e:
        raise RuntimeError(f"Error retrieving logical rules: {e}") from e


@mcp.tool(annotations=_IDEMPOTENT_TOOL)
@_session_locked
def change_maboss_rule(
    node: Annotated[
        NonEmptyString,
        Field(description="Name of the node to change the rule for."),
    ],
    new_rule: Annotated[
        NonEmptyString,
        Field(description="New rule string to replace the existing rule."),
    ],
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to update. Omit to use the active default session.",
    ),
) -> str:
    """Change the Boolean rule for a specific node in the MaBoSS simulation network.

    The new rule is validated before being kept. If invalid, the previous rule is restored.
    Call get_maboss_logical_rules() first to inspect the current rules.
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise RuntimeError(
            "No MaBoSS simulation has been built yet. "
            "Call bnet_to_bnd_and_cfg then build_simulation first."
        )

    try:
        if not isinstance(node, str) or not node.strip():
            raise ValueError("Invalid node name. Must be a non-empty string.")

        if not isinstance(new_rule, str) or not new_rule.strip():
            raise ValueError("Invalid new_rule. Must be a non-empty string.")

        node = node.strip()
        new_rule = new_rule.strip()

        # Check node existence
        try:
            target_node = sess.sim.network[node]
        # pyMaBoSS does not expose a stable lookup exception contract.
        except Exception as node_error:  # noqa: BLE001
            try:
                available_nodes = list(sess.sim.network.keys())
            except Exception as list_error:  # noqa: BLE001
                raise ValueError(f"Unknown node '{node}'.") from list_error
            raise ValueError(
                f"Unknown node '{node}'. "
                f"Available nodes: {', '.join(available_nodes)}"
            ) from node_error

        # Save previous rule
        old_rule = target_node.logExp
        logger.info("Previous rule for %s: %s", node, old_rule)

        # Apply new rule
        target_node.logExp = new_rule

        # Validate the updated model
        try:
            check_result = sess.sim.check()
        except Exception as check_exc:
            target_node.logExp = old_rule
            logger.exception(
                "Validation failed unexpectedly after updating %s", node
            )
            raise RuntimeError(
                f"Rule change aborted for '{node}'. "
                f"Validation could not be completed: {check_exc}"
            ) from check_exc

        if check_result:
            # pyMaBoSS may return parser/check errors as a non-empty result
            target_node.logExp = old_rule
            logger.error(
                "Invalid rule for %s. Change reverted. Errors: %s",
                node,
                check_result,
            )
            raise ValueError(
                f"Rule change rejected for '{node}'. The previous rule has been restored.\n"
                f"Previous rule: {old_rule}\n"
                f"Proposed rule: {new_rule}\n"
                f"Validation errors: {check_result}"
            )

        logger.info("Updated rule for %s: %s", node, target_node.logExp)
        return (
            f"Rule changed successfully for '{node}'.\n"
            f"Previous rule: {old_rule}\n"
            f"New rule: {target_node.logExp}"
        )

    except (ValueError, RuntimeError):
        raise
    except Exception as e:
        logger.exception("Error changing MaBoSS rule")
        raise RuntimeError(f"Error changing MaBoSS rule: {e}") from e


@mcp.tool(annotations=_READ_ONLY_TOOL)
@_session_locked
def get_maboss_mutations(
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to query. Omit to use the active default session.",
    ),
) -> Annotated[CallToolResult, MaBoSSMutationListResult]:
    """Return the mutation settings currently applied to the MaBoSS network."""
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise RuntimeError(
            "No simulation loaded. Call bnet_to_bnd_and_cfg then build_simulation first."
        )
    try:
        mutation_values = sess.sim.get_mutations()
        mutations = _mutation_records(mutation_values)
        payload = MaBoSSMutationListResult(
            server="MaBoSS",
            session_id=sess.session_id,
            mutation_count=len(mutations),
            mutations=mutations,
        )
        return structured_report(str(mutation_values), payload)
    except Exception as e:
        raise RuntimeError(f"Error retrieving mutations: {e}") from e


# ---------------------------------------------------------------------------
# Configuration tools
# ---------------------------------------------------------------------------

@mcp.tool(annotations=_IDEMPOTENT_TOOL)
@_session_locked
def update_maboss_parameters(
    parameters: MaBoSSParameterUpdates | None = Field(  # noqa: B008
        default=None,
        description=(
            "Dict of {parameter_name: value} to update. "
            "Omit or pass null to list all current parameters and valid keys. "
            "Common keys: sample_count (int), max_time (float), time_tick (float), "
            "discrete_time (0|1), thread_count (int)."
        ),
    ),
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to update. Omit to use the active default session.",
    ),
) -> Annotated[CallToolResult, MaBoSSParameterResult]:
    """Update one or more MaBoSS simulation parameters, or list current values.

    Call with parameters=null (or omit it) to display all current parameter
    values and their valid keys before making changes.
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise RuntimeError(
            "No MaBoSS simulation has been built yet. "
            "Call bnet_to_bnd_and_cfg then build_simulation first."
        )
    try:
        parameter_updates = (
            None
            if parameters is None
            else parameters.model_dump(exclude_none=True)
        )
        if not parameter_updates:
            df = pd.DataFrame(
                [[k, v] for k, v in sess.sim.param.items()],
                columns=["parameter", "value"],
            )
            text = (
                "Current MaBoSS parameters "
                "(pass a parameters dict to update_maboss_parameters to modify):\n"
                + df.to_markdown(index=False, tablefmt="plain")
            )
            parameters_list = _parameter_records(sess.sim.param)
            payload = MaBoSSParameterResult(
                server="MaBoSS",
                session_id=sess.session_id,
                mode="inspect",
                parameter_count=len(parameters_list),
                parameters=parameters_list,
                updated_parameters=[],
            )
            return structured_report(text, payload)
        allowed = set(sess.sim.param.keys())
        unknown = [k for k in parameter_updates if k not in allowed]
        if unknown:
            raise ValueError(
                "Unsupported parameter(s): " + ", ".join(unknown) +
                "\nCall update_maboss_parameters with no arguments to list valid keys."
            )
        for key, value in parameter_updates.items():
            sess.sim.param[key] = value
        logger.info("MaBoSS parameters updated: %s", parameter_updates)
        summary = ", ".join(f"{k}={v}" for k, v in parameter_updates.items())
        parameters_list = _parameter_records(sess.sim.param)
        payload = MaBoSSParameterResult(
            server="MaBoSS",
            session_id=sess.session_id,
            mode="update",
            parameter_count=len(parameters_list),
            parameters=parameters_list,
            updated_parameters=list(parameter_updates),
        )
        return structured_report(f"Parameters updated: {summary}", payload)
    except ValueError:
        raise
    except Exception as e:
        logger.exception("Error updating MaBoSS parameters")
        raise RuntimeError(f"Error updating MaBoSS parameters: {e}") from e


@mcp.tool(annotations=_IDEMPOTENT_TOOL)
@_session_locked
def set_maboss_output_nodes(
    output_nodes: Annotated[
        list[NonEmptyString],
        Field(
            min_length=1,
            description="Non-empty list of node names to mark as output nodes (e.g. ['Apoptosis', 'Proliferation']).",
        ),
    ],
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to update. Omit to use the active default session.",
    ),
) -> str:
    """Set which nodes are treated as outputs in the MaBoSS simulation.

    Call get_maboss_nodes() first to obtain the exact valid node names.
    Limiting outputs reduces result size and speeds up large simulations.
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise RuntimeError(
            "No MaBoSS simulation has been built yet. "
            "Call bnet_to_bnd_and_cfg then build_simulation first."
        )
    try:
        logger.info("Previous output nodes: %s", sess.sim.network.get_output())
        sess.sim.network.set_output(output_nodes)
        logger.info("Updated output nodes: %s", sess.sim.network.get_output())
        return f"Output nodes set successfully: {sess.sim.network.get_output()}"
    except Exception as e:
        logger.exception("Error setting MaBoSS output nodes")
        raise RuntimeError(f"Error setting MaBoSS output nodes: {e}") from e


@mcp.tool(annotations=_IDEMPOTENT_TOOL)
@_session_locked
def set_maboss_initial_state(
    nodes: Annotated[
        NonEmptyString | Annotated[list[NonEmptyString], Field(min_length=1)],
        Field(
            description=(
                "Node name (str) or non-empty list of node names to set initial "
                "state for. E.g. 'node1' or ['node1', 'node2']."
            )
        ),
    ],
    probDict: Annotated[list[float] | dict, Field(
        description=(
            "Probability specification. "
            "Single node: list [P(OFF), P(ON)] or dict {0: P(OFF), 1: P(ON)}. "
            "Multiple nodes: dict mapping tuples of 0/1 to probabilities, "
            "e.g. {(0, 0): 0.4, (1, 0): 0.6}."
        )
    )],
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to update. Omit to use the active default session.",
    ),
) -> str:
    """Set initial state probabilities for one or more nodes in the MaBoSS simulation.

    Call get_maboss_nodes() first to obtain the exact valid node names.
    Call get_maboss_initial_state() to inspect the current state before modifying it.

    Examples:
        set_maboss_initial_state('node1', [0.3, 0.7])
        set_maboss_initial_state(['node1', 'node2'], {(0, 0): 0.4, (1, 0): 0.6, (0, 1): 0})
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise RuntimeError(
            "No MaBoSS simulation has been built yet. "
            "Call bnet_to_bnd_and_cfg then build_simulation first."
        )
    try:
        if isinstance(nodes, str):
            node_arg = nodes
        elif isinstance(nodes, (list, tuple)):
            node_arg = list(nodes)
        else:
            raise ValueError("Invalid type for 'nodes'. Must be str or list of str.")

        if isinstance(node_arg, str):
            if not isinstance(probDict, (list, dict)):
                raise ValueError(
                    "For a single node, probDict must be a list or dict."
                )
        elif isinstance(node_arg, list):
            if not isinstance(probDict, dict):
                raise ValueError(
                    "For multiple nodes, probDict must be a dict mapping "
                    "tuples to probabilities."
                )

        logger.info("Previous initial state: %s", sess.sim.network.get_istate())
        sess.sim.network.set_istate(node_arg, probDict)
        logger.info("Updated initial state: %s", sess.sim.network.get_istate())
        return f"Initial state set: {sess.sim.network.get_istate()}"
    except ValueError:
        raise
    except Exception as e:
        logger.exception("Error setting MaBoSS initial state")
        raise RuntimeError(f"Error setting MaBoSS initial state: {e}") from e


# ---------------------------------------------------------------------------
# Analysis tools
# ---------------------------------------------------------------------------

@mcp.tool(annotations=_READ_ONLY_TOOL)
async def simulate_mutation(
    ctx: Context,
    nodes: Annotated[
        NonEmptyString | Annotated[list[NonEmptyString], Field(min_length=1)],
        Field(
            description=(
                "Node name (str) or list of node names to mutate. "
                "E.g. 'FoxO3' or ['FoxO3', 'AKT']."
            )
        ),
    ],
    state: Annotated[
        MutationState | list[MutationState],
        Field(
            default="OFF",
            description=(
                "Mutation state(s): 'ON', 'OFF', or 'WT'. "
                "A single string applies to all nodes. "
                "A list must match the length of nodes, e.g. ['OFF', 'ON']."
            ),
        ),
    ] = "OFF",
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to use. Omit to use the active default session.",
    ),
) -> Annotated[CallToolResult, MaBoSSMutationSimulationResult]:
    """Run a one-off mutant simulation without modifying the session's base simulation.

    Creates an internal copy of the current simulation, applies the
    specified mutations, runs it, and returns the final state probability
    distribution as a Markdown table. The session state is unchanged.

    Examples:
        simulate_mutation('FoxO3', 'OFF')
        simulate_mutation(['FoxO3', 'AKT'], ['OFF', 'ON'])
    """
    node_list = [nodes] if isinstance(nodes, str) else list(nodes)
    if isinstance(state, str):
        state_list = [state] * len(node_list)
    else:
        state_list = list(state)
        if len(state_list) != len(node_list):
            raise ValueError("Length of 'state' must match length of 'nodes'.")

    valid_states = {"ON", "OFF", "WT"}
    for mutation_state in state_list:
        if mutation_state not in valid_states:
            raise ValueError(
                f"Invalid mutation state '{mutation_state}'. "
                f"Must be one of {valid_states}."
            )

    await ctx.report_progress(0, 3)
    await ctx.report_progress(1, 3)

    def run_mutation_locked():
        with session_manager.session_scope(session_id):
            sess = ensure_session(session_id)
            if sess.sim is None:
                raise RuntimeError(
                    "No MaBoSS simulation has been built yet. "
                    "Call bnet_to_bnd_and_cfg then build_simulation first."
                )
            try:
                logger.info("Running mutant simulation")
                mutated_simulation = sess.sim.copy()
                for node, mutation_state in zip(
                    node_list,
                    state_list,
                    strict=True,
                ):
                    mutated_simulation.mutate(node, mutation_state)
                    logger.info(
                        "Applied mutation: %s -> %s",
                        node,
                        mutation_state,
                    )
                mutation_result = mutated_simulation.run()
                return (
                    sess.session_id,
                    mutation_result.get_last_states_probtraj(),
                )
            except ValueError:
                raise
            except Exception as e:
                logger.exception("Error running mutant simulation")
                raise RuntimeError(
                    f"Error running mutant simulation: {e}"
                ) from e

    resolved_session_id, df_prob = await anyio.to_thread.run_sync(
        run_mutation_locked
    )
    await ctx.report_progress(2, 3)

    mutations = [
        MaBoSSMutationRecord(node=node, state=mutation_state)
        for node, mutation_state in zip(node_list, state_list, strict=True)
    ]
    trajectory = _scientific_table(df_prob)
    if df_prob.empty:
        await ctx.report_progress(3, 3)
        payload = MaBoSSMutationSimulationResult(
            server="MaBoSS",
            session_id=resolved_session_id,
            mutations=mutations,
            has_trajectory_data=False,
            trajectory=trajectory,
        )
        return structured_report(
            "_Simulation completed but returned no trajectory data._",
            payload,
        )

    display_df = clean_for_markdown(df_prob)
    md_table = display_df.to_markdown(index=False, tablefmt="plain")
    await ctx.report_progress(3, 3)
    text = "\n".join([
        "**MaBoSS Mutant Simulation: State Probability Trajectory**",
        "",
        f"_Mutations applied: {dict(zip(node_list, state_list, strict=True))}_",
        "",
        md_table,
    ])
    payload = MaBoSSMutationSimulationResult(
        server="MaBoSS",
        session_id=resolved_session_id,
        mutations=mutations,
        has_trajectory_data=True,
        trajectory=trajectory,
    )
    return structured_report(text, payload)


@mcp.tool(annotations=_IDEMPOTENT_TOOL)
@_session_locked
def visualize_network_trajectories(
    session_id: NonEmptyString | None = None,
    until: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Maximum MaBoSS simulation time to display. "
            "Omit to plot the full available trajectory."
        ),
    ),
) -> Annotated[CallToolResult, MaBoSSTrajectoryPlotResult]:
    """Plot network state trajectories and return an uncropped PNG image."""
    logger.info("Visualizing network trajectories")
    sess = ensure_session(session_id)

    if sess.result is None:
        raise RuntimeError(
            "No simulation has been run yet. Call run_simulation first."
        )

    fig = None
    try:
        # pyMaBoSS draws the legend outside the axes and returns None. Own the
        # figure explicitly so rendering and cleanup do not depend on plt.gcf().
        fig, axes = plt.subplots(figsize=(10, 6))
        sess.result.plot_trajectory(until=until, axes=axes)
        if not axes.has_data():
            raise RuntimeError("MaBoSS returned no trajectory data to plot.")
        fig.tight_layout()

        # Render once with a tight bounding box so the external legend is not
        # cropped. The exact same PNG bytes are saved and returned to the client.
        with io.BytesIO() as buffer:
            fig.savefig(
                buffer,
                format="png",
                dpi=150,
                bbox_inches="tight",
                pad_inches=0.2,
            )
            png_data = buffer.getvalue()

        art_dir = get_artifact_dir(_SERVER_ROOT, sess.session_id)
        output_path = safe_artifact_path(art_dir, "network_trajectory.png")
        output_path.write_bytes(png_data)

        logger.info("Trajectory plot saved to %s", output_path)

        time_window = (
            "the full available simulation time"
            if until is None
            else f"simulation time <= {until:g}"
        )
        payload = MaBoSSTrajectoryPlotResult(
            server="MaBoSS",
            session_id=sess.session_id,
            until=until,
            time_window="full" if until is None else "bounded",
            image_file=artifact_file_summary(
                output_path,
                session_id=sess.session_id,
            ),
        )
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=(
                        f"Network trajectory plot ({time_window}) saved to: "
                        f"{output_path}"
                    ),
                ),
                Image(data=png_data, format="png").to_image_content(),
            ],
            structured_content=payload.model_dump(mode="json"),
        )

    except Exception as e:
        logger.exception("Error saving trajectory plot")
        raise RuntimeError(f"Error saving trajectory plot: {e}") from e
    finally:
        if fig is not None:
            plt.close(fig)


@mcp.tool(annotations=_READ_ONLY_TOOL)
@_session_locked
def get_simulation_result(
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to query. Omit to use the active default session.",
    ),
) -> Annotated[CallToolResult, MaBoSSSimulationResult]:
    """Return the last simulation result as a Markdown table of state probabilities.

    Columns = distinct Boolean states (sets of ON nodes joined by '--').
    The single row is the final timepoint snapshot; values sum to ~1.
    Call run_simulation() first.
    """
    sess = ensure_session(session_id)
    if sess.result is None:
        raise RuntimeError(
            "No simulation has been run yet. Call run_simulation() first."
        )
    try:
        df_prob = sess.result.get_last_states_probtraj()
        trajectory = _scientific_table(df_prob)
        if df_prob.empty:
            payload = MaBoSSSimulationResult(
                server="MaBoSS",
                session_id=sess.session_id,
                has_trajectory_data=False,
                trajectory=trajectory,
            )
            return structured_report(
                "_Simulation completed but returned no trajectory data._",
                payload,
            )
        display_df = clean_for_markdown(df_prob)
        md_table = display_df.to_markdown(index=False, tablefmt="plain")
        text = "\n".join([
            "**MaBoSS Simulation: State Probability Trajectory**",
            "",
            md_table,
        ])
        payload = MaBoSSSimulationResult(
            server="MaBoSS",
            session_id=sess.session_id,
            has_trajectory_data=True,
            trajectory=trajectory,
        )
        return structured_report(text, payload)
    except Exception as e:
        raise RuntimeError(f"Error retrieving simulation result: {e}") from e


# ---------------------------------------------------------------------------
# Housekeeping tools
# ---------------------------------------------------------------------------

@mcp.tool(annotations=_READ_ONLY_TOOL)
def list_generated_files(
    session_id: NonEmptyString | None = Field(
        default=None,
        description=(
            "Session whose artifact files to list. "
            "Omit for the active default session. Pass 'all' to list every session."
        ),
    ),
) -> Annotated[CallToolResult, MaBoSSArtifactFileListResult]:
    """List all artifact files (BND, CFG, PNG, …) for a session or across all sessions."""
    if session_id == "all":
        files = list_artifacts(_SERVER_ROOT, session_id=None)
        resolved_session_id = None
        scope = "all"
    else:
        with session_manager.session_scope(session_id):
            sess = ensure_session(session_id)
            files = list_artifacts(_SERVER_ROOT, session_id=sess.session_id)
            resolved_session_id = sess.session_id
            scope = "session"

    if not files:
        text = "No artifact files found."
    else:
        text = "## Generated artifact files\n\n" + "\n".join(
            f"- {file_path}" for file_path in files
        )
    payload = MaBoSSArtifactFileListResult(
        server="MaBoSS",
        scope=scope,
        session_id=resolved_session_id,
        count=len(files),
        files=[
            artifact_file_summary(
                file_path,
                session_id=(
                    resolved_session_id
                    if resolved_session_id is not None
                    else file_path.parent.name
                ),
            )
            for file_path in files
        ],
    )
    return structured_report(text, payload)


@mcp.tool(annotations=_IDEMPOTENT_DESTRUCTIVE_TOOL)
@_session_locked
def clean_generated_files(
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session whose artifact files to remove. Omit to use the active default session.",
    ),
) -> Annotated[CallToolResult, MaBoSSArtifactCleanupResult]:
    """Remove all artifact files (BND, CFG, PNG, …) for the given session."""
    sess = ensure_session(session_id)
    try:
        count = clean_artifacts(_SERVER_ROOT, sess.session_id)
        logger.info(
            "Cleaned %s artifact file(s) for session %s",
            count,
            sess.session_id,
        )
        text = f"Removed {count} artifact file(s) for session {sess.session_id}."
        payload = MaBoSSArtifactCleanupResult(
            server="MaBoSS",
            session_id=sess.session_id,
            removed_count=count,
        )
        return structured_report(text, payload)
    except Exception as e:
        logger.exception("Error during cleanup")
        raise RuntimeError(f"Error during cleanup: {e}") from e


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def clean_for_markdown(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitise a DataFrame for safe Markdown rendering.

    Converts all cells to strings, collapses whitespace, removes 'nan' literals,
    and drops entirely-empty rows/columns.
    """
    # pandas 3 preserves missing values when casting to str, so normalize only
    # non-missing cells and replace the preserved sentinels explicitly.
    df_str = df.map(
        lambda val: " ".join(str(val).split()),
        na_action="ignore",
    ).fillna("")
    df_str = df_str.replace("nan", "", regex=False)
    df_str = df_str.dropna(axis=1, how="all")
    df_str = df_str.loc[:, (df_str != "").any(axis=0)]
    df_str = df_str.dropna(axis=0, how="all")
    df_str = df_str.loc[(df_str != "").any(axis=1), :]
    return df_str


if __name__ == "__main__":
    mcp.run()
