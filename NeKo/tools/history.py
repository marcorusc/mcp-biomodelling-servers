"""NeKo network-history MCP resource and tool registrations."""

from typing import Annotated, Literal

from mcp.server.mcpserver.exceptions import ResourceNotFoundError
from pydantic import Field

from ..app import mcp
from ..contracts import (
    DESTRUCTIVE_IDEMPOTENT_CLOSED,
    NON_IDEMPOTENT_CLOSED,
    READ_ONLY_CLOSED,
    NonEmptyString,
)
from ..session_manager import ensure_session, session_manager
from ..src.helpers import _invalidate, requires_network, session_locked
from ..src.history import (
    HistoryNavigationResult,
    HistoryRetentionResult,
    NetworkHistorySummary,
    NetworkStateComparison,
    compare_history_states,
    state_ids,
    summarize_history,
)


@mcp.resource(
    uri="neko://session/{session_id}/history",
    name="Network History",
    title="NeKo network history",
    description=(
        "Branching NeKo network history rendered as inline SVG HTML. "
        "Use list_network_history to identify the current state."
    ),
    mime_type="text/html",
)
def network_history_resource(session_id: str) -> str:
    """Render the history of an existing NeKo session without mutating it."""
    try:
        with session_manager.existing_session_scope(session_id) as sess:
            if sess.network is None:
                raise ResourceNotFoundError(
                    "No network in this session. Call create_network first."
                )
            try:
                return sess.network.history_html()
            except Exception as exc:
                raise RuntimeError(
                    "Unable to render NeKo history as HTML. "
                    f"Check the Graphviz installation: {exc}"
                ) from exc
    except KeyError as exc:
        raise ResourceNotFoundError(
            f"NeKo session not found: {session_id}"
        ) from exc


@mcp.tool(
    title="List network history",
    annotations=READ_ONLY_CLOSED,
    structured_output=True,
)
@requires_network
def list_network_history(
    session_id: Annotated[
        NonEmptyString | None,
        Field(description="Session ID; omit to inspect the active/default session."),
    ] = None,
    *,
    sess=None,
    network=None,
) -> NetworkHistorySummary:
    """List exact state IDs and branch relationships for the current network."""
    return summarize_history(
        network,
        session_id=sess.session_id,
        max_states=sess.get_history_max_states(),
    )


@mcp.tool(
    title="Navigate network history",
    annotations=NON_IDEMPOTENT_CLOSED,
    structured_output=True,
)
@requires_network
def navigate_network_history(
    action: Annotated[
        Literal["undo", "redo", "checkout"],
        Field(
            description=(
                "History operation: undo to a parent, redo to a child, "
                "or checkout an exact state ID."
            )
        ),
    ],
    state_id: Annotated[
        int | None,
        Field(
            ge=0,
            description=(
                "Exact target state ID. Required for checkout; optional for "
                "redo when the current state has one child; invalid for undo."
            ),
        ),
    ] = None,
    session_id: Annotated[
        NonEmptyString | None,
        Field(description="Session ID; omit to use the active/default session."),
    ] = None,
    *,
    sess=None,
    network=None,
) -> HistoryNavigationResult:
    """Move through NeKo's branching history while preserving saved states."""
    if action == "checkout" and state_id is None:
        raise ValueError("checkout requires an exact state_id.")
    if action == "undo" and state_id is not None:
        raise ValueError("undo does not accept state_id.")

    previous_state_id = network.current_state_id
    if action == "undo":
        network.undo()
    elif action == "redo":
        network.redo(state_id)
    else:
        network.checkout(state_id)

    current_state_id = network.current_state_id
    moved = previous_state_id != current_state_id
    if moved:
        _invalidate(sess)

    if moved:
        message = (
            f"Moved from state {previous_state_id} to state "
            f"{current_state_id}."
        )
    elif action == "undo":
        message = "Already at a root state; no parent state is available."
    elif action == "redo":
        message = "No child state is available from the current state."
    else:
        message = f"State {current_state_id} is already checked out."

    return HistoryNavigationResult(
        server="NeKo",
        session_id=sess.session_id,
        action=action,
        requested_state_id=state_id,
        previous_state_id=previous_state_id,
        current_state_id=current_state_id,
        moved=moved,
        node_count=len(network.nodes),
        edge_count=len(network.edges),
        message=message,
    )


@mcp.tool(
    title="Compare network states",
    annotations=READ_ONLY_CLOSED,
    structured_output=True,
)
@requires_network
def compare_network_states(
    state_a: Annotated[
        int,
        Field(ge=0, description="Exact ID of the baseline history state."),
    ],
    state_b: Annotated[
        int,
        Field(ge=0, description="Exact ID of the comparison history state."),
    ],
    session_id: Annotated[
        NonEmptyString | None,
        Field(description="Session ID; omit to use the active/default session."),
    ] = None,
    *,
    sess=None,
    network=None,
) -> NetworkStateComparison:
    """Compare two saved topologies without changing the checked-out state."""
    return compare_history_states(
        network,
        session_id=sess.session_id,
        state_a=state_a,
        state_b=state_b,
    )


@mcp.tool(
    title="Set network history limit",
    annotations=DESTRUCTIVE_IDEMPOTENT_CLOSED,
    structured_output=True,
)
@session_locked
def set_network_history_limit(
    max_states: Annotated[
        int | None,
        Field(
            ge=2,
            description=(
                "Maximum retained states (minimum 2), or null for unbounded "
                "history. Lowering this value may immediately and "
                "irreversibly prune older snapshots."
            ),
        ),
    ],
    session_id: Annotated[
        NonEmptyString | None,
        Field(description="Session ID; omit to configure the active/default session."),
    ] = None,
) -> HistoryRetentionResult:
    """Configure optional NeKo history pruning for this session."""
    sess = ensure_session(session_id)
    network = sess.network
    before_ids = state_ids(network) if network is not None else []

    if network is not None:
        network.set_max_history(max_states)
    sess.set_history_max_states(max_states)

    after_ids = state_ids(network) if network is not None else []
    pruned_ids = sorted(set(before_ids) - set(after_ids))
    if max_states is None:
        policy = "History retention is now unbounded."
    else:
        policy = f"History retention is limited to {max_states} states."
    if network is None:
        policy += " The policy will apply when a network is created."
    elif pruned_ids:
        policy += (
            " Pruned state IDs: "
            + ", ".join(str(state_id) for state_id in pruned_ids)
            + "."
        )

    return HistoryRetentionResult(
        server="NeKo",
        session_id=sess.session_id,
        max_states=max_states,
        applies_to_current_network=network is not None,
        state_count_before=len(before_ids),
        state_count_after=len(after_ids),
        pruned_state_ids=pruned_ids,
        retained_state_ids=after_ids,
        message=policy,
    )
