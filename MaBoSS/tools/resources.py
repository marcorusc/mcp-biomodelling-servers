"""Read-only model-state resource registrations for MaBoSS."""

import pandas as pd
from mcp.server.mcpserver.exceptions import ResourceNotFoundError

from ..app import mcp
from ..locking import resource_session_locked
from ..services.formatting import clean_for_markdown
from ..session_manager import ensure_session


@mcp.resource(
    uri="maboss://session/{session_id}/nodes",
    name="Network Nodes",
    title="MaBoSS network nodes",
    description="Comma-separated list of node names in the loaded MaBoSS network.",
    mime_type="text/plain",
)
@resource_session_locked
def resource_network_nodes(session_id: str) -> str:
    """Return the node names for the given session."""
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise ResourceNotFoundError(
            "No simulation loaded. Call bnet_to_bnd_and_cfg then "
            "build_simulation first."
        )
    nodes_list = list(sess.sim.network.keys())
    if not nodes_list:
        return "No nodes found in the MaBoSS network."
    return f"Nodes: {', '.join(nodes_list)}"


@mcp.resource(
    uri="maboss://session/{session_id}/parameters",
    name="Simulation Parameters",
    title="MaBoSS simulation parameters",
    description=(
        "Current MaBoSS simulation parameters as a Markdown table. "
        "Use update_maboss_parameters to modify."
    ),
    mime_type="text/markdown",
)
@resource_session_locked
def resource_parameters(session_id: str) -> str:
    """Return current parameter table for the given session."""
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise ResourceNotFoundError(
            "No simulation loaded. Call bnet_to_bnd_and_cfg then "
            "build_simulation first."
        )
    frame = pd.DataFrame(
        [[key, value] for key, value in sess.sim.param.items()],
        columns=["parameter", "value"],
    )
    return frame.to_markdown(index=False, tablefmt="plain")


@mcp.resource(
    uri="maboss://session/{session_id}/initial_state",
    name="Initial State",
    title="MaBoSS initial state",
    description="Initial state probability configuration of the MaBoSS simulation.",
    mime_type="text/plain",
)
@resource_session_locked
def resource_initial_state(session_id: str) -> str:
    """Return the initial state for the given session."""
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise ResourceNotFoundError(
            "No simulation loaded. Call bnet_to_bnd_and_cfg then "
            "build_simulation first."
        )
    return str(sess.sim.network.get_istate())


@mcp.resource(
    uri="maboss://session/{session_id}/logical_rules",
    name="Logical Rules",
    title="MaBoSS logical rules",
    description="Boolean logical rules of the MaBoSS network.",
    mime_type="text/plain",
)
@resource_session_locked
def resource_logical_rules(session_id: str) -> str:
    """Return the logical rules for the given session."""
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise ResourceNotFoundError(
            "No simulation loaded. Call bnet_to_bnd_and_cfg then "
            "build_simulation first."
        )
    return str(sess.sim.get_logical_rules())


@mcp.resource(
    uri="maboss://session/{session_id}/mutations",
    name="Mutations",
    title="MaBoSS mutations",
    description="Mutation settings currently applied to the MaBoSS network.",
    mime_type="text/plain",
)
@resource_session_locked
def resource_mutations(session_id: str) -> str:
    """Return mutation settings for the given session."""
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise ResourceNotFoundError(
            "No simulation loaded. Call bnet_to_bnd_and_cfg then "
            "build_simulation first."
        )
    return str(sess.sim.get_mutations())


@mcp.resource(
    uri="maboss://session/{session_id}/result",
    name="Simulation Result",
    title="MaBoSS simulation result",
    description=(
        "Post-run state probability table as Markdown. "
        "Columns = Boolean states (ON nodes joined by '--'); "
        "rows = final timepoint snapshot (values sum to ~1). "
        "Available only after run_simulation has been called."
    ),
    mime_type="text/markdown",
)
@resource_session_locked
def resource_simulation_result(session_id: str) -> str:
    """Return the last simulation result for the given session."""
    sess = ensure_session(session_id)
    if sess.result is None:
        raise ResourceNotFoundError(
            "No simulation has been run yet. Call run_simulation first."
        )
    frame = sess.result.get_last_states_probtraj()
    if frame.empty:
        return "_Simulation completed but returned no trajectory data._"
    markdown = clean_for_markdown(frame).to_markdown(
        index=False,
        tablefmt="plain",
    )
    return "\n".join([
        "**MaBoSS Simulation: State Probability Trajectory**",
        "",
        markdown,
    ])
