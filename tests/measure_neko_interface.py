"""Measure advertised schemas and fixed protocol fixtures, without database I/O.

Run ``python -m tests.measure_neko_interface > report.json``. These fixtures
measure serialization overhead, not real-client tool-selection accuracy or
language-model token usage. Real library workflow tests are separate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from importlib.metadata import version
from pathlib import Path

from mcp import Client

from tests.test_neko_mcp_errors import (
    _clear_sessions,
    _create_session,
    _HistoryNetwork,
    mcp,
    session_manager,
)


def size(value: object) -> int:
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode())


async def measure(payloads: dict | None = None, *, evidence_stress: bool = False) -> dict:
    _clear_sessions()
    network = _HistoryNetwork()
    network.convert_edgelist_into_genesymbol = lambda: network.edges.copy()
    session_id = _create_session(network)
    try:
        async with Client(mcp) as client:
            listing = await client.list_tools()
            tools = listing.model_dump(mode="json", by_alias=True, exclude_none=True)
            if payloads is not None:
                payloads["tools"] = tools
                payloads["responses"] = {}
            schemas = {
                tool["name"]: {
                    "input_bytes": size(tool.get("inputSchema", {})),
                    "output_bytes": size(tool.get("outputSchema", {})),
                }
                for tool in tools["tools"]
            }
            responses = {}
            for name, arguments in (
                ("status", {}),
                ("list_genes_and_interactions", {"verbosity": "preview"}),
                ("list_network_history", {}),
                ("compare_network_states", {"state_a": 2, "state_b": 4}),
            ):
                result = await client.call_tool(name, {
                    "session_id": session_id, **arguments,
                })
                if result.is_error:
                    raise RuntimeError(f"Measurement fixture failed: {name}: {result}")
                responses[name] = size(result.model_dump(
                    mode="json", by_alias=True, exclude_none=True,
                ))
                if payloads is not None:
                    payloads["responses"][name] = result.model_dump(
                        mode="json", by_alias=True, exclude_none=True,
                    )
            network._metadata[4]["explanation"] = "evidence" * 1024
            result = await client.call_tool("list_network_history", {
                "session_id": session_id,
            })
            if result.is_error:
                raise RuntimeError("Large-metadata history fixture failed")
            responses["history_with_8192_byte_explanation"] = size(result.model_dump(
                mode="json", by_alias=True, exclude_none=True,
            ))
            if payloads is not None:
                payloads["responses"]["history_with_8192_byte_explanation"] = result.model_dump(
                    mode="json", by_alias=True, exclude_none=True,
                )
            if "edit_interactions" in schemas:
                network.edit_interactions = lambda edits, expected_state_id: {
                    "changed": True, "previous_state_id": 4, "state_id": 8,
                    "changes": [{"operation": "add", "source": "P04637", "target": "Q00987",
                                 "before": None, "after": {"Effect": "inhibition",
                                 "Provenance": {"origin": "manual"}}}],
                }
                result = await client.call_tool("edit_interactions", {
                    "session_id": session_id,
                    "expected_network_id": session_manager.get_session(session_id).network_id,
                    "expected_state_id": 4,
                    "edits": [{"operation": "add", "source": "TP53", "target": "MDM2",
                               "effect": "inhibition", "rationale": "Synthetic fixture",
                               "evidence": {"summary": "Test evidence", "citations": ["Test publication"]}}],
                })
                if result.is_error:
                    raise RuntimeError("Editor measurement failed")
                responses["edit_interactions_one_change"] = size(result.model_dump(
                    mode="json", by_alias=True, exclude_none=True,
                ))
                if payloads is not None:
                    payloads["responses"]["edit_interactions_one_change"] = result.model_dump(
                        mode="json", by_alias=True, exclude_none=True,
                    )
            if evidence_stress:
                explanation = "Synthetic experimental evidence requires careful interpretation of assay conditions. " * 12
                provenance = {
                    "origin": "manual",
                    "original": {"origin": "manual", "database_name": None,
                                 "effect": "inhibition", "references": "123456"},
                    "edits": [{"operation": "set_effect", "origin": "manual", "state_id": i,
                               "before_effect": "stimulation", "after_effect": "inhibition",
                               "rationale": explanation, "evidence": {
                                   "summary": explanation, "pmids": ["123456"],
                                   "citations": ["Synthetic publication fixture"],
                               }} for i in range(20)],
                }
                network.edges["Provenance"] = [provenance] + [None] * (len(network.edges) - 1)
                session_manager.get_session(session_id).invalidate_edges_cache()
                for verbosity in ("summary", "preview", "full"):
                    name = f"inventory_{verbosity}_with_20_evidence_edits"
                    result = await client.call_tool("list_genes_and_interactions", {
                        "session_id": session_id, "verbosity": verbosity,
                    })
                    if result.is_error:
                        raise RuntimeError(f"Evidence fixture failed: {result}")
                    raw = result.model_dump(mode="json", by_alias=True, exclude_none=True)
                    responses[name] = size(raw)
                    if payloads is not None:
                        payloads["responses"][name] = raw
            return {
                "measurement": "compact UTF-8 JSON bytes; excludes JSON-RPC envelope",
                "fixture": "four saved states, one branch, three nodes, two edges",
                "limitations": "stub library; no client-selection or token benchmark",
                "packages": {name: version(name) for name in ("mcp", "pydantic")},
                "tool_count": len(schemas),
                "list_tools_bytes": size(tools),
                "input_schema_bytes": sum(s["input_bytes"] for s in schemas.values()),
                "output_schema_bytes": sum(s["output_bytes"] for s in schemas.values()),
                "schemas": schemas,
                "response_bytes": responses,
            }
    finally:
        _clear_sessions()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payloads", type=Path, help="Save exact JSON payloads for independent token accounting.")
    parser.add_argument("--evidence-stress", action="store_true", help="Also measure one edge with 20 accumulated evidence records.")
    args = parser.parse_args()
    payloads = {} if args.payloads else None
    print(json.dumps(asyncio.run(measure(payloads, evidence_stress=args.evidence_stress)), indent=2, sort_keys=True))
    if args.payloads:
        args.payloads.write_text(json.dumps(payloads, ensure_ascii=False))
