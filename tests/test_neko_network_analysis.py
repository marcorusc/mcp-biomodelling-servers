"""Focused tests for framework-independent NeKo analysis services."""

import pandas as pd
import pytest

from NeKo.services.network_analysis import (
    edge_degrees,
    partition_gene_set_edges,
    reference_list,
)


def test_edge_degrees_counts_both_endpoints_and_ignores_empty_values() -> None:
    edges = pd.DataFrame(
        [
            {"source": "P53", "target": "MDM2"},
            {"source": "P53", "target": "AKT1"},
            {"source": None, "target": "AKT1"},
        ]
    )

    assert edge_degrees(edges) == {"P53": 2, "MDM2": 1, "AKT1": 2}


def test_edge_degrees_rejects_malformed_backend_data() -> None:
    with pytest.raises(ValueError, match="missing required columns: target"):
        edge_degrees(pd.DataFrame({"source": ["P53"]}))


def test_partition_gene_set_edges_handles_empty_backend_tables() -> None:
    incident, internal, boundary = partition_gene_set_edges(
        pd.DataFrame(),
        {"P53"},
    )

    assert list(incident.columns) == ["source", "target", "Effect"]
    assert internal.empty
    assert boundary.empty


def test_reference_list_normalizes_and_deduplicates_evidence() -> None:
    assert reference_list("PMID:1; PMID:2,PMID:1") == [
        "PMID:1",
        "PMID:2",
    ]
    assert reference_list("none") == []
