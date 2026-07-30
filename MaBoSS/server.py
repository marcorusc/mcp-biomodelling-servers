import inspect
import logging
import os
import sys
from functools import wraps
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

logger = logging.getLogger(__name__)

mcp = MCPServer(
    "MaBoSS",
    title="MaBoSS Boolean Model Simulator",
    description=(
        "Configure, simulate, analyze, and visualize Boolean models with MaBoSS."
    ),
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
2. **Convert:** `bnet_to_bnd_and_cfg(bnet_path)` — BNET → BND + CFG
3. **Load:** `build_simulation()` — loads BND/CFG into session
4. **Inspect nodes (MANDATORY):** `get_maboss_nodes()` — list ALL valid node names; always do this before any configuration step to avoid referencing non-existent nodes
5. **Inspect parameters:** `update_maboss_parameters()` (no args) — review current defaults
6. **Tune:** `update_maboss_parameters({"sample_count": 1000, "thread_count": 4})`
7. **Reduce output nodes (IMPORTANT):** `set_maboss_output_nodes(["Apoptosis", "Proliferation"])` — restricts the result to only the nodes you care about. Without this, MaBoSS enumerates ALL 2^N Boolean states, which becomes exponentially expensive for large networks (>20 nodes). Always set output nodes to the smallest biologically meaningful subset before running.
8. **Configure (optional):** `get_maboss_initial_state()` to inspect current state, then `set_maboss_initial_state(...)` if non-default probabilities are needed. Only use node names returned by `get_maboss_nodes()`.
9. **Run:** `run_simulation()` — executes the simulation and saves `result.csv` to the artifact directory
10. **Analyse:** `get_simulation_result()` — returns the state probability table as a Markdown table
11. **Visualise:** `visualize_network_trajectories()` — saves a PNG artifact
12. **Mutate:** `simulate_mutation(nodes, state)` — runs a one-off mutant copy

> **State space warning:** A network with N nodes produces up to 2^N possible Boolean states.
> Always call `set_maboss_output_nodes` to restrict outputs before `run_simulation`.
> For a 30-node network this reduces the result from >1 billion states to only the states
> of the selected output nodes (typically 2-5 nodes).

## 2. Tool Categories
* **Session management:** `create_session`, `list_sessions`, `set_default_session`, `delete_session`
* **Pipeline:** `bnet_to_bnd_and_cfg`, `build_simulation`, `run_simulation`
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
            f"- **{sid}**{default_marker}: sim={has_sim}  result={has_res}  bnd={info['bnd_path'] or '—'}"
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
) -> str:
    """Execute the loaded MaBoSS simulation and store the result in the session.

    IMPORTANT: Call set_maboss_output_nodes() before this tool. Without it, MaBoSS becomes exponentially expensive. 
    Restricting to a small set of output nodes keeps the run time and result size manageable.

    Tune performance via update_maboss_parameters (sample_count, thread_count)
    before running large simulations. After completion, read the result from
    maboss://session/{id}/result.
    """
    await ctx.report_progress(0, 2)

    def run_locked() -> None:
        with session_manager.session_scope(session_id):
            sess = ensure_session(session_id)
            if sess.sim is None:
                raise RuntimeError(
                    "No MaBoSS simulation has been built yet. "
                    "Call bnet_to_bnd_and_cfg then build_simulation first."
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
            try:
                art_dir = get_artifact_dir(_SERVER_ROOT, sess.session_id)
                csv_path = safe_artifact_path(art_dir, "result.csv")
                df_result = run_result.get_last_states_probtraj()
                if not df_result.empty:
                    df_result.to_csv(csv_path, index=False)
                    logger.info("Result saved to %s", csv_path)
            except Exception as csv_err:
                logger.warning(
                    "Could not save result CSV: %s",
                    csv_err,
                    exc_info=True,
                )

    await anyio.to_thread.run_sync(run_locked)
    await ctx.report_progress(2, 2)
    logger.info("MaBoSS simulation completed successfully")
    return (
        "MaBoSS simulation completed successfully.\n"
        "Call `get_simulation_result()` to read the state probability table.\n"
        "The result is also saved to the session artifact directory as result.csv."
    )

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
) -> str:
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
    if not nodes_list:
        return "No nodes found in the MaBoSS network."
    return "Network nodes:\n" + "\n".join(f"- {n}" for n in nodes_list)


@mcp.tool(annotations=_READ_ONLY_TOOL)
@_session_locked
def get_maboss_initial_state(
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to query. Omit to use the active default session.",
    ),
) -> str:
    """Return the current initial state probability configuration of the MaBoSS simulation.

    Use this to inspect the state before calling set_maboss_initial_state().
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise RuntimeError(
            "No simulation loaded. Call bnet_to_bnd_and_cfg then build_simulation first."
        )
    try:
        return f"Initial state:\n{sess.sim.network.get_istate()}"
    except Exception as e:
        raise RuntimeError(f"Error retrieving initial state: {e}") from e

@mcp.tool(annotations=_READ_ONLY_TOOL)
@_session_locked
def get_maboss_logical_rules(
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to query. Omit to use the active default session.",
    ),
) -> str:
    """Return the Boolean logical rules of the loaded MaBoSS network."""
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise RuntimeError(
            "No simulation loaded. Call bnet_to_bnd_and_cfg then build_simulation first."
        )
    try:
        return str(sess.sim.get_logical_rules())
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
) -> str:
    """Return the mutation settings currently applied to the MaBoSS network."""
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise RuntimeError(
            "No simulation loaded. Call bnet_to_bnd_and_cfg then build_simulation first."
        )
    try:
        return str(sess.sim.get_mutations())
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
) -> str:
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
            return (
                "Current MaBoSS parameters "
                "(pass a parameters dict to update_maboss_parameters to modify):\n"
                + df.to_markdown(index=False, tablefmt="plain")
            )
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
        return f"Parameters updated: {summary}"
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
) -> str:
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

    def run_mutation_locked() -> pd.DataFrame:
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
                return mutation_result.get_last_states_probtraj()
            except ValueError:
                raise
            except Exception as e:
                logger.exception("Error running mutant simulation")
                raise RuntimeError(
                    f"Error running mutant simulation: {e}"
                ) from e

    df_prob = await anyio.to_thread.run_sync(run_mutation_locked)
    await ctx.report_progress(2, 3)

    if df_prob.empty:
        return "_Simulation completed but returned no trajectory data._"

    df_prob = clean_for_markdown(df_prob)
    md_table = df_prob.to_markdown(index=False, tablefmt="plain")
    await ctx.report_progress(3, 3)
    return "\n".join([
        "**MaBoSS Mutant Simulation: State Probability Trajectory**",
        "",
        f"_Mutations applied: {dict(zip(node_list, state_list, strict=True))}_",
        "",
        md_table,
    ])


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
) -> CallToolResult:
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
            ]
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
) -> str:
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
        if df_prob.empty:
            return "_Simulation completed but returned no trajectory data._"
        df_prob = clean_for_markdown(df_prob)
        md_table = df_prob.to_markdown(index=False, tablefmt="plain")
        return "\n".join([
            "**MaBoSS Simulation: State Probability Trajectory**",
            "",
            md_table,
        ])
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
    df_str = df.astype(str)
    df_str = df_str.map(lambda val: " ".join(val.split()))
    df_str = df_str.replace("nan", "", regex=False)
    df_str = df_str.dropna(axis=1, how="all")
    df_str = df_str.loc[:, (df_str != "").any(axis=0)]
    df_str = df_str.dropna(axis=0, how="all")
    df_str = df_str.loc[(df_str != "").any(axis=1), :]
    return df_str


if __name__ == "__main__":
    mcp.run()
