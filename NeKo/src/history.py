"""Structured NeKo network-history responses and normalization helpers."""

from __future__ import annotations

import json
import math
from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import Field

from mcp_biomodelling_servers.structured_outputs import StructuredOutputModel

HistoryStateId = Annotated[int, Field(ge=0)]
HistoryAction = Literal["undo", "redo", "checkout"]


class NeKoHistoryResult(StructuredOutputModel):
    """Fields shared by NeKo network-history results."""

    server: Literal["NeKo"]


class HistoryStateSummary(StructuredOutputModel):
    """One state in NeKo's branching network history."""

    state_id: HistoryStateId = Field(
        description="Stable NeKo history state ID.",
    )
    parent_state_ids: list[HistoryStateId] = Field(
        description="Direct parent state IDs.",
    )
    child_state_ids: list[HistoryStateId] = Field(
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
    details_uri: str = ""
    explanation_status: str = "unavailable"
    ancestry_gaps: list[int] = Field(default_factory=list)
    relationships_truncated: bool = False


class NetworkHistorySummary(NeKoHistoryResult):
    """Creation-ordered summary of a session's branching history."""

    session_id: str = Field(min_length=1)
    current_state_id: HistoryStateId | None
    root_state_id: HistoryStateId | None
    max_states: int | None = Field(
        ge=2,
        description="Retention limit; null means unbounded history.",
    )
    state_count: int = Field(ge=0)
    states: list[HistoryStateSummary]
    network_id: str | None = None
    database_name: str | None = None
    offset: int = 0
    next_offset: int | None = None
    pruned_state_count: int | None = None
    working_state_matches_current: bool | None = None


class HistoryNavigationResult(NeKoHistoryResult):
    """Result of moving through NeKo network history."""

    session_id: str = Field(min_length=1)
    action: HistoryAction
    requested_state_id: HistoryStateId | None
    previous_state_id: HistoryStateId | None
    current_state_id: HistoryStateId | None
    moved: bool
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    message: str = Field(min_length=1)


class NetworkStateComparison(NeKoHistoryResult):
    """Deterministic topology difference between two NeKo states."""

    session_id: str = Field(min_length=1)
    state_a: HistoryStateId
    state_b: HistoryStateId
    edge_columns: list[str]
    added_nodes: list[str]
    removed_nodes: list[str]
    added_edges: list[dict[str, Any]]
    removed_edges: list[dict[str, Any]]
    changed_edges: list[dict[str, Any]] = Field(default_factory=list)
    truncated: bool = False
    network_id: str | None = None
    details_uris: list[str] = Field(default_factory=list)
    node_changes: dict[str, list[dict[str, Any]]] | None = Field(default=None, exclude_if=lambda v: v is None)


class HistoryRetentionResult(NeKoHistoryResult):
    """Result of changing one session's history-retention policy."""

    session_id: str = Field(min_length=1)
    max_states: int | None = Field(ge=2)
    applies_to_current_network: bool
    state_count_before: int = Field(ge=0)
    state_count_after: int = Field(ge=0)
    pruned_state_ids: list[HistoryStateId]
    retained_state_ids: list[HistoryStateId]
    message: str = Field(min_length=1)


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
    network_id: str | None = None,
    offset: int = 0,
    limit: int = 25,
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
            )[:100],
            child_state_ids=sorted(
                int(child)
                for child in graph.successors(state["id"])
            )[:100],
            is_current=state["id"] == current_state_id,
            is_root=state["id"] == root_state_id,
            metadata=compact_metadata(state.get("metadata", {})),
            details_uri=f"neko://session/{session_id}/history/{network_id}/{state['id']}",
            explanation_status=state.get("metadata", {}).get("transition", {}).get("explanation_status", "unavailable"),
            ancestry_gaps=state.get("metadata", {}).get("ancestry_gaps", [])[:100],
            relationships_truncated=(len(list(graph.predecessors(state["id"]))) > 100
                                     or len(list(graph.successors(state["id"]))) > 100
                                     or len(state.get("metadata", {}).get("ancestry_gaps", [])) > 100),
        )
        for state in listed_states[offset:offset + limit]
    ]
    full = network.history_dict() if callable(getattr(network, "history_dict", None)) else {}
    return NetworkHistorySummary(
        server="NeKo",
        session_id=session_id,
        current_state_id=current_state_id,
        root_state_id=root_state_id,
        max_states=max_states,
        state_count=len(listed_states),
        states=states,
        network_id=network_id,
        database_name=getattr(network, "database_name", None),
        offset=offset,
        next_offset=offset + limit if offset + limit < len(listed_states) else None,
        pruned_state_count=len(full["pruned_state_ids"]) if "pruned_state_ids" in full else None,
        working_state_matches_current=full.get("working_state_matches_current"),
    )


def compact_metadata(metadata: dict) -> dict:
    """Keep display metadata small; authoritative records stay in the state resource."""
    if "transition" not in metadata and len(json.dumps(_json_safe(metadata))) <= 1024:
        return _json_safe(metadata)
    transition = metadata.get("transition", {})
    return {
        "label": str(metadata.get("label", metadata.get("method", "")))[:160],
        "method": str(metadata.get("method", ""))[:80],
        "details_omitted": True,
        "limitations": [str(item)[:200] for item in transition.get("limitations", [])[:3]],
        "edge_changes": {key: len(transition.get("edges", {}).get(key, []))
                         for key in ("added", "removed", "changed")},
    }


def bounded_change(change: dict) -> dict:
    """Keep accumulated edit evidence out of concise comparison responses."""
    if len(json.dumps(_json_safe(change))) <= 4096:
        return _json_safe(change)
    result: dict[str, Any] = {}
    for side in ("before", "after"):
        row = change[side]
        result[side] = {key: row.get(key) for key in ("source", "target", "Effect")}
        provenance = row.get("Provenance")
        result[side]["origin"] = provenance.get("origin", "unknown") if isinstance(provenance, dict) else "unknown"
    result["details_omitted"] = True
    return result


def bounded_edge(row: dict) -> dict:
    if len(json.dumps(_json_safe(row))) <= 4096:
        return _json_safe(row)
    provenance = row.get("Provenance")
    return {"source": row.get("source"), "target": row.get("target"),
            "Effect": row.get("Effect"), "origin": provenance.get("origin", "unknown") if isinstance(provenance, dict) else "unknown",
            "details_omitted": True}


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
    network_id: str | None = None,
    limit: int = 25,
) -> NetworkStateComparison:
    """Compare exact state IDs and normalize NeKo's set-derived result."""
    available_ids = set(state_ids(network))
    missing_ids = [
        state_id
        for state_id in (state_a, state_b)
        if state_id not in available_ids
    ]
    if missing_ids:
        available = ", ".join(str(state_id) for state_id in sorted(available_ids)[:25])
        if len(available_ids) > 25:
            available += f" … ({len(available_ids)} states total)"
        missing = ", ".join(str(state_id) for state_id in missing_ids)
        raise ValueError(
            f"Unknown history state ID(s): {missing}. "
            f"Available state IDs: {available or '(none)'}."
        )

    comparison = network.compare_states(state_a, state_b)
    edge_columns = [str(column) for column in network.edges.columns]
    added = _sorted_edge_records(comparison.get("added_edges", []), edge_columns)
    removed = _sorted_edge_records(comparison.get("removed_edges", []), edge_columns)
    node_changes = None
    if "edge_changes" in comparison:
        # Prefer exact canonical records; legacy tuples use display labels and omit provenance.
        edge_columns = network.get_state(state_b)["edges"]["columns"]
        added = [bounded_edge(row) for row in comparison["edge_changes"]["added"]]
        removed = [bounded_edge(row) for row in comparison["edge_changes"]["removed"]]
        node_changes = {key: _json_safe(rows[:limit]) for key, rows in comparison.get("node_changes", {}).items()}
    return NetworkStateComparison(
        server="NeKo",
        session_id=session_id,
        state_a=state_a,
        state_b=state_b,
        edge_columns=edge_columns,
        added_nodes=sorted(
            str(node) for node in comparison.get("added_nodes", [])
        )[:limit],
        removed_nodes=sorted(
            str(node) for node in comparison.get("removed_nodes", [])
        )[:limit],
        added_edges=added[:limit],
        removed_edges=removed[:limit],
        changed_edges=[bounded_change(change) for change in comparison.get("edge_changes", {}).get("changed", [])[:limit]],
        truncated=any(len(comparison.get(key, [])) > limit for key in
                      ("added_nodes", "removed_nodes", "added_edges", "removed_edges"))
                  or len(comparison.get("edge_changes", {}).get("changed", [])) > limit
                  or len(added) > limit or len(removed) > limit
                  or any(len(rows) > limit for rows in comparison.get("node_changes", {}).values()),
        network_id=network_id,
        details_uris=[f"neko://session/{session_id}/history/{network_id}/{sid}" for sid in (state_a, state_b)],
        node_changes=node_changes,
    )
