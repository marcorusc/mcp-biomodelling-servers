"""Preserve the curated interaction graph for evidence-backed ODE authoring."""

from __future__ import annotations

import hashlib
import json

from mcp_biomodelling_servers.ode_handoff import ODEEdge, ODENetwork, ODENode

from .network_analysis import node_records, optional_text, reference_list


def ode_network(network) -> ODENetwork:
    nodes = {}
    for node in node_records(network):
        node_id = node.uniprot or node.gene_symbol
        if node_id:
            nodes[node_id] = ODENode(node_id=node_id, **node.model_dump())
    edges = {}
    # Use the original edge table to retain database mechanism/context columns.
    for _, row in network.edges.iterrows():
        source, target = str(row["source"]), str(row["target"])
        for endpoint in (source, target):
            if endpoint not in nodes:
                match = next(
                    (n for n in nodes.values() if n.gene_symbol == endpoint), None
                )
                nodes[endpoint] = ODENode(
                    node_id=endpoint,
                    gene_symbol=match.gene_symbol if match else None,
                    uniprot=match.uniprot if match else None,
                )
        effect = optional_text(row.get("Effect", row.get("effect")))
        refs = reference_list(row.get("References", row.get("references")))
        metadata = {}
        for key, value in row.items():
            if str(key).lower() in ("source", "target", "effect", "references"):
                continue
            metadata[str(key)] = (
                sorted(map(str, value))
                if isinstance(value, (list, tuple, set))
                else optional_text(value)
            )
        identity = json.dumps([source, target, effect, metadata], sort_keys=True)
        edge_id = "edge_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
        if edge_id in edges:
            edges[edge_id].references = sorted(
                set(edges[edge_id].references) | set(refs)
            )
        else:
            edges[edge_id] = ODEEdge(
                edge_id=edge_id,
                source=source,
                target=target,
                effect=effect,
                references=sorted(refs),
                metadata=metadata,
            )
    return ODENetwork(
        nodes=sorted(nodes.values(), key=lambda n: n.node_id),
        edges=sorted(edges.values(), key=lambda e: e.edge_id),
    )
