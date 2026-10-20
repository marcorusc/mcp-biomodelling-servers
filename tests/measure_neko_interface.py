"""Measure advertised schemas and fixed protocol fixtures, without database I/O.

Run ``python -m tests.measure_neko_interface > report.json``. These fixtures
measure serialization overhead, not real-client tool-selection accuracy or
language-model token usage. Real library workflow tests are separate.
"""

from __future__ import annotations

import asyncio
import json
from importlib.metadata import version

from mcp import Client

from tests.test_neko_mcp_errors import (
    _clear_sessions,
    _create_session,
    _HistoryNetwork,
    mcp,
)


def size(value: object) -> int:
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode())


async def measure() -> dict:
    _clear_sessions()
    network = _HistoryNetwork()
    network.convert_edgelist_into_genesymbol = lambda: network.edges.copy()
    session_id = _create_session(network)
    try:
        async with Client(mcp) as client:
            listing = await client.list_tools()
            tools = listing.model_dump(mode="json", by_alias=True, exclude_none=True)
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
            network._metadata[4]["explanation"] = "evidence" * 1024
            result = await client.call_tool("list_network_history", {
                "session_id": session_id,
            })
            if result.is_error:
                raise RuntimeError("Large-metadata history fixture failed")
            responses["history_with_8192_byte_explanation"] = size(result.model_dump(
                mode="json", by_alias=True, exclude_none=True,
            ))
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
    print(json.dumps(asyncio.run(measure()), indent=2, sort_keys=True))
