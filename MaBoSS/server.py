import os
import sys
from pathlib import Path
from typing import Annotated, Optional, List, Union

# Make the repo root importable so we can use the shared artifact_manager
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import maboss
import io
import pandas as pd
from pydantic import Field

from mcp.server.fastmcp import Context, FastMCP, Image
from session_manager import session_manager, ensure_session, MaBoSSSession
from artifact_manager import get_artifact_dir, safe_artifact_path, list_artifacts, clean_artifacts, write_session_meta, list_artifact_sessions as _list_artifact_sessions_on_disk

mcp = FastMCP("MaBoSS")

_SERVER_ROOT = Path(__file__).parent

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
def resource_network_nodes(session_id: str) -> str:
    """Return the node names for the given session."""
    sess = ensure_session(session_id)
    if sess.sim is None:
        return "No simulation loaded. Call bnet_to_bnd_and_cfg then build_simulation first."
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
def resource_parameters(session_id: str) -> str:
    """Return current parameter table for the given session."""
    sess = ensure_session(session_id)
    if sess.sim is None:
        return "No simulation loaded. Call bnet_to_bnd_and_cfg then build_simulation first."
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
def resource_initial_state(session_id: str) -> str:
    """Return the initial state for the given session."""
    sess = ensure_session(session_id)
    if sess.sim is None:
        return "No simulation loaded. Call bnet_to_bnd_and_cfg then build_simulation first."
    try:
        return str(sess.sim.network.get_istate())
    except Exception as e:
        return f"Error retrieving initial state: {e}"


@mcp.resource(
    uri="maboss://session/{session_id}/logical_rules",
    name="Logical Rules",
    description="Boolean logical rules of the MaBoSS network.",
    mime_type="text/plain",
)
def resource_logical_rules(session_id: str) -> str:
    """Return the logical rules for the given session."""
    sess = ensure_session(session_id)
    if sess.sim is None:
        return "No simulation loaded. Call bnet_to_bnd_and_cfg then build_simulation first."
    try:
        return str(sess.sim.get_logical_rules())
    except Exception as e:
        return f"Error retrieving logical rules: {e}"


@mcp.resource(
    uri="maboss://session/{session_id}/mutations",
    name="Mutations",
    description="Mutation settings currently applied to the MaBoSS network.",
    mime_type="text/plain",
)
def resource_mutations(session_id: str) -> str:
    """Return mutation settings for the given session."""
    sess = ensure_session(session_id)
    if sess.sim is None:
        return "No simulation loaded. Call bnet_to_bnd_and_cfg then build_simulation first."
    try:
        return str(sess.sim.get_mutations())
    except Exception as e:
        return f"Error retrieving mutations: {e}"


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
def resource_simulation_result(session_id: str) -> str:
    """Return the last simulation result for the given session."""
    sess = ensure_session(session_id)
    if sess.result is None:
        return "_No simulation has been run yet. Call run_simulation first._"
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
        return f"**Error retrieving simulation result:** {e}"


@mcp.resource(
    uri="maboss://session/{session_id}/files",
    name="Artifact Files",
    description="List of artifact files (BND, CFG, PNG, …) generated for a session.",
    mime_type="text/markdown",
)
def resource_generated_files(session_id: str) -> str:
    """Return a Markdown list of artifact files for the given session."""
    files = list_artifacts(_SERVER_ROOT, session_id=session_id)
    if not files:
        return "No artifact files found for this session."
    return "## Artifact files\n\n" + "\n".join(f"- {f}" for f in files)


# ---------------------------------------------------------------------------
# Session management tools
# ---------------------------------------------------------------------------

@mcp.tool()
def create_session(
    set_as_default: bool = Field(
        default=True,
        description="When True (default), the new session becomes the active default for subsequent calls.",
    ),
    label: Optional[str] = Field(
        default=None,
        description="Optional human-readable label (e.g. 'TP53-MYC Boolean run'). Stored on disk so the session can be rediscovered after a server restart.",
    ),
) -> str:
    """Create a new MaBoSS session.

    Returns the session ID (UUID) that must be passed to pipeline tools when running
    multiple independent simulations in parallel.
    """
    sid = session_manager.create_session(set_as_default=set_as_default)
    write_session_meta(_SERVER_ROOT, sid, server_name="MaBoSS", label=label)
    label_info = f" ({label})" if label else ""
    return f"Session created: {sid}{label_info}" + (" (set as default)" if set_as_default else "")


@mcp.tool()
def list_sessions() -> str:
    """List all active MaBoSS sessions with their simulation and result status."""
    sessions = session_manager.list_sessions()
    if not sessions:
        return "No active sessions. Call create_session() to start one."
    lines = ["## MaBoSS Sessions\n"]
    for sid, info in sessions.items():
        default_marker = " **(default)**" if info["is_default"] else ""
        has_sim = "✓" if info["has_simulation"] else "✗"
        has_res = "✓" if info["has_result"] else "✗"
        lines.append(
            f"- **{sid}**{default_marker}: sim={has_sim}  result={has_res}  bnd={info['bnd_path'] or '—'}"
        )
    return "\n".join(lines)


@mcp.tool()
def list_artifact_sessions() -> str:
    """List all MaBoSS sessions that have artifact files on disk (including past server runs).

    Unlike list_sessions() which only shows in-memory sessions, this scans the
    artifacts/ directory and reads session_meta.json files, so previously created
    sessions are visible even after a server restart.

    Use the returned session_id and file paths to resume earlier work, e.g.:
      build_simulation(bnd_path='/path/to/artifacts/<uuid>/output.bnd',
                       cfg_path='/path/to/artifacts/<uuid>/output.cfg')
    """
    sessions = _list_artifact_sessions_on_disk(_SERVER_ROOT, server_name="MaBoSS")
    if not sessions:
        return "No artifact sessions found on disk."
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
    return "\n".join(lines)


@mcp.tool()
def set_default_session(
    session_id: Annotated[str, Field(description="ID of the session to set as the active default.")],
) -> str:
    """Set the default (active) MaBoSS session used when session_id is omitted in other tools."""
    if session_manager.set_default(session_id):
        return f"Default session set to: {session_id}"
    return f"Session not found: {session_id}"


@mcp.tool()
def delete_session(
    session_id: Annotated[str, Field(description="ID of the session to delete.")],
    clean_files: bool = Field(
        default=True,
        description="When True (default), also remove all artifact files for this session.",
    ),
) -> str:
    """Delete a MaBoSS session and optionally its artifact files."""
    removed_files = 0
    if clean_files:
        removed_files = clean_artifacts(_SERVER_ROOT, session_id)
    if session_manager.delete_session(session_id):
        return f"Session {session_id} deleted." + (
            f" Removed {removed_files} artifact file(s)." if clean_files else ""
        )
    return f"Session not found: {session_id}"


# ---------------------------------------------------------------------------
# Pipeline tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def bnet_to_bnd_and_cfg(
    bnet_path: Annotated[str, Field(description="Absolute or CWD-relative path to the .bnet file to convert.")],
    ctx: Context,
    session_id: Optional[str] = Field(
        default=None,
        description="Session to write the output files into. Omit to use the active default session.",
    ),
) -> str:
    """Convert a BNET file to MaBoSS BND and CFG files.

    Output files are written to the session artifact directory
    (<server>/artifacts/<session_id>/output.bnd and output.cfg).
    After conversion, call build_simulation() to load the simulation.
    """
    sess = ensure_session(session_id)
    art_dir = get_artifact_dir(_SERVER_ROOT, sess.session_id)
    bnd_out = str(safe_artifact_path(art_dir, "output.bnd"))
    cfg_out = str(safe_artifact_path(art_dir, "output.cfg"))

    await ctx.info(f"Converting {bnet_path} -> {bnd_out}, {cfg_out}")
    try:
        maboss.bnet_to_bnd_and_cfg(bnet_path, bnd_out, cfg_out)
    except Exception as e:
        await ctx.error(f"bnet_to_bnd_and_cfg failed: {e}")
        return f"Error converting .bnet file: {e}"

    for path, label in [(bnd_out, "BND"), (cfg_out, "CFG")]:
        if not os.path.exists(path):
            return f"Error: expected {label} file was not created at {path}."

    await ctx.info(f"BND and CFG files created: {bnd_out}, {cfg_out}")
    return (
        f"MaBoSS .bnd and .cfg files created successfully.\n"
        f"  BND: {bnd_out}\n"
        f"  CFG: {cfg_out}\n\n"
        f"Next: call build_simulation(session_id='{sess.session_id}') to load the simulation."
    )


@mcp.tool()
async def build_simulation(
    ctx: Context,
    session_id: Optional[str] = Field(
        default=None,
        description="Session to load the simulation into. Omit to use the active default session.",
    ),
    bnd_path: Optional[str] = Field(
        default=None,
        description="Path to the .bnd file. Omit to use the file generated by bnet_to_bnd_and_cfg for this session.",
    ),
    cfg_path: Optional[str] = Field(
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

    await ctx.info(f"Loading MaBoSS simulation: BND={bnd_path}  CFG={cfg_path}")

    try:
        loaded_sim = maboss.load(bnd_path, cfg_path)
    except Exception as e:
        await ctx.error(f"Failed to load simulation: {e}")
        return f"Error loading MaBoSS simulation: {e}"

    if loaded_sim:
        sess.set_simulation(loaded_sim, bnd_path, cfg_path)
        await ctx.info("MaBoSS simulation loaded successfully.")
        parameters_str = "\n".join(f"{k}: {v}" for k, v in loaded_sim.param.items())
        return (
            f"MaBoSS simulation loaded successfully.\n{parameters_str}\n\n"
            f"NEXT STEP: call get_maboss_nodes() to retrieve the list of valid node "
            f"names before calling set_maboss_output_nodes() or set_maboss_initial_state()."
        )
    else:
        await ctx.error("maboss.load returned None.")
        return "Error: maboss.load returned None. Check the BND and CFG files."


@mcp.tool()
async def run_simulation(
    ctx: Context,
    session_id: Optional[str] = Field(
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
    sess = ensure_session(session_id)
    if sess.sim is None:
        return "No MaBoSS simulation has been built yet. Call bnet_to_bnd_and_cfg then build_simulation first."
    try:
        await ctx.report_progress(0, 2)
        await ctx.info("Running MaBoSS simulation...")
        run_result = sess.sim.run()
        sess.set_result(run_result)

        # Persist the result table to disk so list_generated_files shows it
        try:
            art_dir = get_artifact_dir(_SERVER_ROOT, sess.session_id)
            csv_path = safe_artifact_path(art_dir, "result.csv")
            df_result = run_result.get_last_states_probtraj()
            if not df_result.empty:
                df_result.to_csv(csv_path, index=False)
                await ctx.info(f"Result saved to {csv_path}")
        except Exception as csv_err:
            await ctx.warning(f"Could not save result CSV: {csv_err}")

        await ctx.report_progress(2, 2)
        await ctx.info("MaBoSS simulation completed successfully.")
        return (
            f"MaBoSS simulation completed successfully.\n"
            f"Call `get_simulation_result()` to read the state probability table.\n"
            f"The result is also saved to the session artifact directory as result.csv."
        )
    except Exception as e:
        await ctx.error(f"Error during MaBoSS simulation run: {str(e)}")
        return f"Error during MaBoSS simulation run: {str(e)}"

@mcp.tool()
async def export_maboss_bnd_cfg(
    ctx: Context,
    prefix: Annotated[str, Field(
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
    session_id: Optional[str] = Field(
        default=None,
        description="Session to export from. Omit to use the active default session.",
    ),
) -> str:
    """Export the current in-memory MaBoSS simulation to .bnd and .cfg files.

    Writes files to: <server>/artifacts/<session_id>/<prefix>.bnd and <prefix>.cfg
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        return "No MaBoSS simulation has been built yet. Call build_simulation first."

    try:
        prefix = (prefix or "").strip()
        if not prefix:
            return "Invalid prefix. Must be a non-empty string."

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
                    return (
                        f"Refusing to overwrite existing file: {p}\n"
                        f"Choose a different prefix or set overwrite=True."
                    )

        await ctx.info(f"Exporting MaBoSS model -> {bnd_out}, {cfg_out}")

        # Write .bnd
        with open(bnd_out, "w") as fbnd:
            sess.sim.print_bnd(out=fbnd)

        # Write .cfg
        with open(cfg_out, "w") as fcfg:
            sess.sim.print_cfg(out=fcfg)

        # Sanity check
        for path, label in [(bnd_out, "BND"), (cfg_out, "CFG")]:
            if not os.path.exists(path):
                return f"Error: expected {label} file was not created at {path}."

        await ctx.info(f"Export complete: {bnd_out}, {cfg_out}")
        return (
            f"Exported current MaBoSS model successfully.\n"
            f"  BND: {bnd_out}\n"
            f"  CFG: {cfg_out}"
        )

    except Exception as e:
        await ctx.error(f"Error exporting MaBoSS model: {str(e)}")
        return f"Error exporting MaBoSS model: {str(e)}"


# ---------------------------------------------------------------------------
# Inspection tools (read-only, no side effects)
# ---------------------------------------------------------------------------

@mcp.tool()
def get_maboss_nodes(
    session_id: Optional[str] = Field(
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
        return "No simulation loaded. Call bnet_to_bnd_and_cfg then build_simulation first."
    nodes_list = list(sess.sim.network.keys())
    if not nodes_list:
        return "No nodes found in the MaBoSS network."
    return "Network nodes:\n" + "\n".join(f"- {n}" for n in nodes_list)


@mcp.tool()
def get_maboss_initial_state(
    session_id: Optional[str] = Field(
        default=None,
        description="Session to query. Omit to use the active default session.",
    ),
) -> str:
    """Return the current initial state probability configuration of the MaBoSS simulation.

    Use this to inspect the state before calling set_maboss_initial_state().
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        return "No simulation loaded. Call bnet_to_bnd_and_cfg then build_simulation first."
    try:
        return f"Initial state:\n{sess.sim.network.get_istate()}"
    except Exception as e:
        return f"Error retrieving initial state: {e}"

@mcp.tool()
def get_maboss_logical_rules(
    session_id: Optional[str] = Field(
        default=None,
        description="Session to query. Omit to use the active default session.",
    ),
) -> str:
    """Return the Boolean logical rules of the loaded MaBoSS network."""
    sess = ensure_session(session_id)
    if sess.sim is None:
        return "No simulation loaded. Call bnet_to_bnd_and_cfg then build_simulation first."
    try:
        return str(sess.sim.get_logical_rules())
    except Exception as e:
        return f"Error retrieving logical rules: {e}"


@mcp.tool()
async def change_maboss_rule(
    ctx: Context,
    node: Annotated[str, Field(description="Name of the node to change the rule for.")],
    new_rule: Annotated[str, Field(description="New rule string to replace the existing rule.")],
    session_id: Optional[str] = Field(
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
        return (
            "No MaBoSS simulation has been built yet. "
            "Call bnet_to_bnd_and_cfg then build_simulation first."
        )

    try:
        if not isinstance(node, str) or not node.strip():
            return "Invalid node name. Must be a non-empty string."

        if not isinstance(new_rule, str) or not new_rule.strip():
            return "Invalid new_rule. Must be a non-empty string."

        node = node.strip()
        new_rule = new_rule.strip()

        # Check node existence
        try:
            target_node = sess.sim.network[node]
        except Exception:
            try:
                available_nodes = list(sess.sim.network.keys())
                return (
                    f"Unknown node '{node}'. "
                    f"Available nodes: {', '.join(available_nodes)}"
                )
            except Exception:
                return f"Unknown node '{node}'."

        # Save previous rule
        old_rule = target_node.logExp
        await ctx.info(f"Previous rule for {node}: {old_rule}")

        # Apply new rule
        target_node.logExp = new_rule

        # Validate the updated model
        try:
            check_result = sess.sim.check()
        except Exception as check_exc:
            target_node.logExp = old_rule
            await ctx.error(
                f"Validation failed unexpectedly after updating {node}: {check_exc}"
            )
            return (
                f"Rule change aborted for '{node}'. "
                f"Validation could not be completed: {check_exc}"
            )

        if check_result:
            # pyMaBoSS may return parser/check errors as a non-empty result
            target_node.logExp = old_rule
            await ctx.error(
                f"Invalid rule for {node}. Change reverted. Errors: {check_result}"
            )
            return (
                f"Rule change rejected for '{node}'. The previous rule has been restored.\n"
                f"Previous rule: {old_rule}\n"
                f"Proposed rule: {new_rule}\n"
                f"Validation errors: {check_result}"
            )

        await ctx.info(f"Updated rule for {node}: {target_node.logExp}")
        return (
            f"Rule changed successfully for '{node}'.\n"
            f"Previous rule: {old_rule}\n"
            f"New rule: {target_node.logExp}"
        )

    except Exception as e:
        await ctx.error(f"Error changing MaBoSS rule: {str(e)}")
        return f"Error changing MaBoSS rule: {str(e)}"


@mcp.tool()
def get_maboss_mutations(
    session_id: Optional[str] = Field(
        default=None,
        description="Session to query. Omit to use the active default session.",
    ),
) -> str:
    """Return the mutation settings currently applied to the MaBoSS network."""
    sess = ensure_session(session_id)
    if sess.sim is None:
        return "No simulation loaded. Call bnet_to_bnd_and_cfg then build_simulation first."
    try:
        return str(sess.sim.get_mutations())
    except Exception as e:
        return f"Error retrieving mutations: {e}"


# ---------------------------------------------------------------------------
# Configuration tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def update_maboss_parameters(
    ctx: Context,
    parameters: Optional[dict] = Field(
        default=None,
        description=(
            "Dict of {parameter_name: value} to update. "
            "Omit or pass null to list all current parameters and valid keys. "
            "Common keys: sample_count (int), max_time (float), time_tick (float), "
            "discrete_time (0|1), thread_count (int)."
        ),
    ),
    session_id: Optional[str] = Field(
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
        return "No MaBoSS simulation has been built yet. Call bnet_to_bnd_and_cfg then build_simulation first."
    try:
        if not parameters:
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
        unknown = [k for k in parameters.keys() if k not in allowed]
        if unknown:
            return (
                "Unsupported parameter(s): " + ", ".join(unknown) +
                "\nCall update_maboss_parameters with no arguments to list valid keys."
            )
        for key, value in parameters.items():
            sess.sim.param[key] = value
        await ctx.info(f"MaBoSS parameters updated: {parameters}")
        summary = ", ".join(f"{k}={v}" for k, v in parameters.items())
        return f"Parameters updated: {summary}"
    except Exception as e:
        await ctx.error(f"Error updating MaBoSS parameters: {str(e)}")
        return f"Error updating MaBoSS parameters: {str(e)}"


@mcp.tool()
async def set_maboss_output_nodes(
    ctx: Context,
    output_nodes: Annotated[List[str], Field(description="List of node names to mark as output nodes (e.g. ['Apoptosis', 'Proliferation']).")],
    session_id: Optional[str] = Field(
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
        return "No MaBoSS simulation has been built yet. Call bnet_to_bnd_and_cfg then build_simulation first."
    try:
        await ctx.info(f"Previous output nodes: {sess.sim.network.get_output()}")
        sess.sim.network.set_output(output_nodes)
        await ctx.info(f"Updated output nodes: {sess.sim.network.get_output()}")
        return f"Output nodes set successfully: {sess.sim.network.get_output()}"
    except Exception as e:
        await ctx.error(f"Error setting MaBoSS output nodes: {str(e)}")
        return f"Error setting MaBoSS output nodes: {str(e)}"


@mcp.tool()
async def set_maboss_initial_state(
    ctx: Context,
    nodes: Annotated[Union[str, List[str]], Field(
        description=(
            "Node name (str) or list of node names to set initial state for. "
            "E.g. 'node1' or ['node1', 'node2']."
        )
    )],
    probDict: Annotated[Union[List[float], dict], Field(
        description=(
            "Probability specification. "
            "Single node: list [P(OFF), P(ON)] or dict {0: P(OFF), 1: P(ON)}. "
            "Multiple nodes: dict mapping tuples of 0/1 to probabilities, "
            "e.g. {(0, 0): 0.4, (1, 0): 0.6}."
        )
    )],
    session_id: Optional[str] = Field(
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
        return "No MaBoSS simulation has been built yet. Call bnet_to_bnd_and_cfg then build_simulation first."
    try:
        if isinstance(nodes, str):
            node_arg = nodes
        elif isinstance(nodes, (list, tuple)):
            node_arg = list(nodes)
        else:
            return "Invalid type for 'nodes'. Must be str or list of str."

        if isinstance(node_arg, str):
            if not isinstance(probDict, (list, dict)):
                return "For a single node, probDict must be a list or dict."
        elif isinstance(node_arg, list):
            if not isinstance(probDict, dict):
                return "For multiple nodes, probDict must be a dict mapping tuples to probabilities."

        await ctx.info(f"Previous initial state: {sess.sim.network.get_istate()}")
        sess.sim.network.set_istate(node_arg, probDict)
        await ctx.info(f"Updated initial state: {sess.sim.network.get_istate()}")
        return f"Initial state set: {sess.sim.network.get_istate()}"
    except Exception as e:
        await ctx.error(f"Error setting MaBoSS initial state: {str(e)}")
        return f"Error setting MaBoSS initial state: {str(e)}"


# ---------------------------------------------------------------------------
# Analysis tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def simulate_mutation(
    ctx: Context,
    nodes: Annotated[Union[str, List[str]], Field(
        description="Node name (str) or list of node names to mutate. E.g. 'FoxO3' or ['FoxO3', 'AKT']."
    )],
    state: Annotated[Union[str, List[str]], Field(
        default="OFF",
        description=(
            "Mutation state(s): 'ON', 'OFF', or 'WT'. "
            "A single string applies to all nodes. "
            "A list must match the length of nodes, e.g. ['OFF', 'ON']."
        ),
    )] = "OFF",
    session_id: Optional[str] = Field(
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
    sess = ensure_session(session_id)
    if sess.sim is None:
        return "No MaBoSS simulation has been built yet. Call bnet_to_bnd_and_cfg then build_simulation first."
    try:
        await ctx.report_progress(0, 3)
        await ctx.info("Running mutant simulation...")
        mutated_simulation = sess.sim.copy()

        node_list = [nodes] if isinstance(nodes, str) else list(nodes)

        if isinstance(state, str):
            state_list = [state] * len(node_list)
        else:
            state_list = list(state)
            if len(state_list) != len(node_list):
                return "Length of 'state' must match length of 'nodes'."

        valid_states = {"ON", "OFF", "WT"}
        for s in state_list:
            if s not in valid_states:
                return f"Invalid mutation state '{s}'. Must be one of {valid_states}."

        await ctx.report_progress(1, 3)
        for node, s in zip(node_list, state_list):
            mutated_simulation.mutate(node, s)
            await ctx.info(f"Applied mutation: {node} -> {s}")

        mut_result = mutated_simulation.run()
        await ctx.report_progress(2, 3)
        df_prob = mut_result.get_last_states_probtraj()

        if df_prob.empty:
            return "_Simulation completed but returned no trajectory data._"

        df_prob = clean_for_markdown(df_prob)
        md_table = df_prob.to_markdown(index=False, tablefmt="plain")
        await ctx.report_progress(3, 3)
        return "\n".join([
            "**MaBoSS Mutant Simulation: State Probability Trajectory**",
            "",
            f"_Mutations applied: {dict(zip(node_list, state_list))}_",
            "",
            md_table,
        ])
    except Exception as e:
        await ctx.error(f"Error running mutant simulation: {str(e)}")
        return f"Error running mutant simulation: {str(e)}"


@mcp.tool()
async def visualize_network_trajectories(
    ctx: Context,
    session_id: Optional[str] = None,
) -> list: # Changed return type to list to support multiple content types
    """Plot network state trajectories and return the image for visualization."""
    await ctx.info("Visualizing network trajectories...")
    sess = ensure_session(session_id)
    
    if sess.result is None:
        return ["No simulation has been run yet. Call run_simulation first."]
    
    try:
        fig = sess.result.plot_trajectory()
        if fig is None:
            fig = plt.gcf()

        # 1. Save to disk as usual (for your artifacts)
        art_dir = get_artifact_dir(_SERVER_ROOT, sess.session_id)
        output_path = str(safe_artifact_path(art_dir, "network_trajectory.png"))
        fig.savefig(output_path)
        
        # 2. Capture the figure in a buffer for the MCP client
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        
        await ctx.info(f"Trajectory plot saved to {output_path}")

        # 3. Return BOTH the text description and the Image object
        return [
            f"Network trajectory plot saved to: {output_path}",
            Image(data=buf.getvalue(), format="png")
        ]
        
    except Exception as e:
        await ctx.error(f"Error saving trajectory plot: {str(e)}")
        return [f"Error saving trajectory plot: {str(e)}"]


@mcp.tool()
def get_simulation_result(
    session_id: Optional[str] = Field(
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
        return "No simulation has been run yet. Call run_simulation() first."
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
        return f"**Error retrieving simulation result:** {e}"


# ---------------------------------------------------------------------------
# Housekeeping tools
# ---------------------------------------------------------------------------

@mcp.tool()
def list_generated_files(
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Session whose artifact files to list. "
            "Omit for the active default session. Pass 'all' to list every session."
        ),
    ),
) -> str:
    """List all artifact files (BND, CFG, PNG, …) for a session or across all sessions."""
    if session_id == "all":
        files = list_artifacts(_SERVER_ROOT, session_id=None)
    else:
        sess = ensure_session(session_id)
        files = list_artifacts(_SERVER_ROOT, session_id=sess.session_id)

    if not files:
        return "No artifact files found."
    return "## Generated artifact files\n\n" + "\n".join(f"- {f}" for f in files)


@mcp.tool()
async def clean_generated_files(
    ctx: Context,
    session_id: Optional[str] = Field(
        default=None,
        description="Session whose artifact files to remove. Omit to use the active default session.",
    ),
) -> str:
    """Remove all artifact files (BND, CFG, PNG, …) for the given session."""
    sess = ensure_session(session_id)
    try:
        count = clean_artifacts(_SERVER_ROOT, sess.session_id)
        await ctx.info(f"Cleaned {count} artifact file(s) for session {sess.session_id}.")
        return f"Removed {count} artifact file(s) for session {sess.session_id}."
    except Exception as e:
        await ctx.error(f"Error during cleanup: {str(e)}")
        return f"Error during cleanup: {str(e)}"


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
