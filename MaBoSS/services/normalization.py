"""Pure normalization helpers for MaBoSS inputs and scientific outputs."""

import math

import pandas as pd

from ..contracts import (
    InitialStateProbabilitySpecification,
    MaBoSSJointStateProbability,
)
from ..scientific_outputs import (
    MaBoSSInitialStateGroup,
    MaBoSSLogicalRuleRecord,
    MaBoSSMutationRecord,
    MaBoSSParameterRecord,
    MaBoSSScientificTable,
    MaBoSSStateProbabilityRecord,
)


def initial_state_probability(value, *, state: object) -> float:
    """Validate a runtime probability, including direct Python calls."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Probability for state {state!r} must be a finite number "
            "between 0 and 1."
        )
    probability = float(value)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(
            f"Probability for state {state!r} must be a finite number "
            "between 0 and 1."
        )
    return probability


def initial_state_key(
    state: object,
    *,
    node_count: int,
) -> int | tuple[int, ...]:
    """Normalize one legacy or JSON-native Boolean state."""
    if (
        node_count == 1
        and not isinstance(state, bool)
        and state in (0, 1, "0", "1")
    ):
        return int(state)

    if not isinstance(state, (list, tuple)):
        if isinstance(state, str):
            raise ValueError(
                "Multi-node initial states cannot use JSON object keys. "
                "Provide probDict as a list of records such as "
                "[{'state': [0, 0], 'probability': 1.0}]."
            )
        raise ValueError(
            "Each multi-node initial state must be a list or tuple of 0/1 "
            "values."
        )
    if len(state) != node_count:
        raise ValueError(
            f"Initial state {state!r} has {len(state)} values, but "
            f"{node_count} nodes were requested."
        )
    if any(
        isinstance(value, bool) or value not in (0, 1)
        for value in state
    ):
        raise ValueError(
            f"Initial state {state!r} must contain only integer 0/1 values."
        )

    normalized = tuple(int(value) for value in state)
    return normalized[0] if node_count == 1 else normalized


def normalize_initial_state_probabilities(
    nodes: str | list[str],
    probabilities: InitialStateProbabilitySpecification,
) -> tuple[str | list[str], list[float] | dict[int | tuple[int, ...], float]]:
    """Validate and convert initial-state inputs to pyMaBoSS's native form."""
    node_names = [nodes] if isinstance(nodes, str) else list(nodes)
    if not node_names:
        raise ValueError("At least one node must be provided.")
    if len(node_names) != len(set(node_names)):
        raise ValueError("Initial-state node names must be unique.")

    node_count = len(node_names)
    node_arg: str | list[str] = (
        node_names[0] if node_count == 1 else node_names
    )

    if isinstance(probabilities, list) and all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in probabilities
    ):
        if node_count != 1:
            raise ValueError(
                "A numeric [P(OFF), P(ON)] list can configure only one node. "
                "For multiple nodes, provide JSON state/probability records."
            )
        if len(probabilities) != 2:
            raise ValueError(
                "A single-node probability list must contain exactly "
                "[P(OFF), P(ON)]."
            )
        normalized_list = [
            initial_state_probability(value, state=state)
            for state, value in enumerate(probabilities)
        ]
        if not math.isclose(
            sum(normalized_list),
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("Initial-state probabilities must sum to 1.")
        return node_arg, normalized_list

    entries: list[tuple[object, object]]
    if isinstance(probabilities, list):
        if not probabilities:
            raise ValueError(
                "At least one initial-state probability record is required."
            )
        entries = []
        for entry in probabilities:
            if isinstance(entry, MaBoSSJointStateProbability):
                entries.append((entry.state, entry.probability))
            elif isinstance(entry, dict):
                if set(entry) != {"state", "probability"}:
                    raise ValueError(
                        "Each initial-state record must contain exactly "
                        "'state' and 'probability'."
                    )
                entries.append((entry["state"], entry["probability"]))
            else:
                raise ValueError(
                    "Joint initial-state probabilities must be records with "
                    "'state' and 'probability' fields."
                )
    elif isinstance(probabilities, dict):
        if not probabilities:
            raise ValueError(
                "At least one initial-state probability is required."
            )
        entries = list(probabilities.items())
    else:
        raise ValueError(
            "probDict must be [P(OFF), P(ON)], a legacy state mapping, or "
            "a JSON-native list of state/probability records."
        )

    normalized_mapping: dict[int | tuple[int, ...], float] = {}
    for state, value in entries:
        normalized_state = initial_state_key(
            state,
            node_count=node_count,
        )
        if normalized_state in normalized_mapping:
            raise ValueError(
                f"Initial state {state!r} is specified more than once."
            )
        normalized_mapping[normalized_state] = initial_state_probability(
            value,
            state=state,
        )

    if not math.isclose(
        sum(normalized_mapping.values()),
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("Initial-state probabilities must sum to 1.")

    return node_arg, normalized_mapping


def scientific_scalar(value):
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


def scientific_table(df: pd.DataFrame) -> MaBoSSScientificTable:
    """Preserve dataframe values and index without embedding them in Markdown."""
    return MaBoSSScientificTable(
        columns=[str(column) for column in df.columns],
        index_name=(
            str(df.index.name)
            if df.index.name is not None
            else None
        ),
        index=[scientific_scalar(value) for value in df.index.tolist()],
        row_count=len(df),
        column_count=len(df.columns),
        rows=[
            [scientific_scalar(value) for value in row]
            for row in df.itertuples(index=False, name=None)
        ],
    )


def initial_state_groups(initial_state) -> list[MaBoSSInitialStateGroup]:
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
            normalized_probability = scientific_scalar(probability)
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


def logical_rule_records(logical_rules) -> list[MaBoSSLogicalRuleRecord]:
    """Normalize pyMaBoSS's logical-rule mapping."""
    return [
        MaBoSSLogicalRuleRecord(node=str(node), rule=str(rule))
        for node, rule in logical_rules.items()
    ]


def mutation_records(mutations) -> list[MaBoSSMutationRecord]:
    """Normalize pyMaBoSS's mutation mapping."""
    return [
        MaBoSSMutationRecord(node=str(node), state=str(state))
        for node, state in mutations.items()
    ]


def parameter_records(parameters) -> list[MaBoSSParameterRecord]:
    """Normalize current MaBoSS parameters while preserving scalar types."""
    return [
        MaBoSSParameterRecord(
            name=str(name),
            value=scientific_scalar(value),
        )
        for name, value in parameters.items()
    ]
