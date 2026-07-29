"""Structured NeKo network-history responses and normalization helpers."""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


class HistoryStateSummary(BaseModel):
    """One state in NeKo's branching network history."""

    state_id: int = Field(description="Stable NeKo history state ID.")
    parent_state_ids: list[int] = Field(
        description="Direct parent state IDs.",
    )
    child_state_ids: list[int] = Field(
        description="Direct child state IDs.",
    )
    is_current: bool = Field(
        description="Whether this state is currently checked out.",
    )
    is_root: bool = Field(
        description="Whether this state is the history root.",
    )
    metadata: dict[str, Any] = Field(
        description="NeKo operation metadata recorded for this state.",
    )


class NetworkHistorySummary(BaseModel):
    """Creation-ordered summary of a session's branching history."""

    session_id: str
    current_state_id: int | None
    root_state_id: int | None
    max_states: int | None = Field(
        description="Retention limit; null means unbounded history.",
    )
    state_count: int
    states: list[HistoryStateSummary]


class HistoryNavigationResult(BaseModel):
    """Result of moving through NeKo network history."""

    session_id: str
    action: str
    requested_state_id: int | None
    previous_state_id: int | None
    current_state_id: int | None
    moved: bool
    node_count: int
    edge_count: int
    message: str


class NetworkStateComparison(BaseModel):
    """Deterministic topology difference between two NeKo states."""

    session_id: str
    state_a: int
    state_b: int
    edge_columns: list[str]
    added_nodes: list[str]
    removed_nodes: list[str]
    added_edges: list[dict[str, Any]]
    removed_edges: list[dict[str, Any]]


class HistoryRetentionResult(BaseModel):
    """Result of changing one session's history-retention policy."""

    session_id: str
    max_states: int | None
    applies_to_current_network: bool
    state_count_before: int
    state_count_after: int
    pruned_state_ids: list[int]
    retained_state_ids: list[int]
    message: str


def _json_safe(value: Any) -> Any:
    """Convert pandas/NumPy and arbitrary metadata values to JSON-safe data."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, bool) and missing:
        return None

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass
    return str(value)


def state_ids(network: Any) -> list[int]:
    """Return creation-ordered, exact NeKo state IDs."""
    return [int(state["id"]) for state in network.list_states()]


def summarize_history(
    network: Any,
    *,
    session_id: str,
    max_states: int | None,
) -> NetworkHistorySummary:
    """Build a structured history summary through NeKo's public graph API."""
    listed_states = network.list_states()
    graph = network.history_graph()
    current_state_id = network.current_state_id
    root_state_id = network.root_state_id
    states = [
        HistoryStateSummary(
            state_id=int(state["id"]),
            parent_state_ids=sorted(
                int(parent)
                for parent in graph.predecessors(state["id"])
            ),
            child_state_ids=sorted(
                int(child)
                for child in graph.successors(state["id"])
            ),
            is_current=state["id"] == current_state_id,
            is_root=state["id"] == root_state_id,
            metadata=_json_safe(state.get("metadata", {})),
        )
        for state in listed_states
    ]
    return NetworkHistorySummary(
        session_id=session_id,
        current_state_id=current_state_id,
        root_state_id=root_state_id,
        max_states=max_states,
        state_count=len(states),
        states=states,
    )


def _edge_record(
    signature: Any,
    edge_columns: list[str],
) -> dict[str, Any]:
    values = (
        list(signature)
        if isinstance(signature, (list, tuple))
        else [signature]
    )
    columns = edge_columns.copy()
    columns.extend(
        f"field_{index}"
        for index in range(len(columns), len(values))
    )
    return {
        columns[index]: _json_safe(value)
        for index, value in enumerate(values)
    }


def _sorted_edge_records(
    signatures: list[Any],
    edge_columns: list[str],
) -> list[dict[str, Any]]:
    records = [
        _edge_record(signature, edge_columns)
        for signature in signatures
    ]
    return sorted(
        records,
        key=lambda record: tuple(
            json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=False,
            )
            for value in record.values()
        ),
    )


def compare_history_states(
    network: Any,
    *,
    session_id: str,
    state_a: int,
    state_b: int,
) -> NetworkStateComparison:
    """Compare exact state IDs and normalize NeKo's set-derived result."""
    available_ids = set(state_ids(network))
    missing_ids = [
        state_id
        for state_id in (state_a, state_b)
        if state_id not in available_ids
    ]
    if missing_ids:
        available = ", ".join(str(state_id) for state_id in sorted(available_ids))
        missing = ", ".join(str(state_id) for state_id in missing_ids)
        raise ValueError(
            f"Unknown history state ID(s): {missing}. "
            f"Available state IDs: {available or '(none)'}."
        )

    comparison = network.compare_states(state_a, state_b)
    edge_columns = [str(column) for column in network.edges.columns]
    return NetworkStateComparison(
        session_id=session_id,
        state_a=state_a,
        state_b=state_b,
        edge_columns=edge_columns,
        added_nodes=sorted(
            str(node) for node in comparison.get("added_nodes", [])
        ),
        removed_nodes=sorted(
            str(node) for node in comparison.get("removed_nodes", [])
        ),
        added_edges=_sorted_edge_records(
            comparison.get("added_edges", []),
            edge_columns,
        ),
        removed_edges=_sorted_edge_records(
            comparison.get("removed_edges", []),
            edge_columns,
        ),
    )
