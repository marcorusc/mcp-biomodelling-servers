"""Measure tool definitions and a targeted rate response; no scientific run.

Pass a source checkout to compare a revision in a fresh interpreter. Token counts
are optional (o200k_base), and do not claim a particular client's context usage.
"""
import asyncio
import importlib
import json
import os
import sys
from pathlib import Path

if len(sys.argv) > 1:
    sys.path.insert(0, str(Path(sys.argv[1]).resolve()))


if os.environ.get("WAVE2_TOKENIZER_PATH"):
    sys.path.append(os.environ["WAVE2_TOKENIZER_PATH"])


def size(value):
    encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    result = {"bytes": len(encoded.encode())}
    try:
        import tiktoken
        result["o200k_tokens"] = len(tiktoken.get_encoding("o200k_base").encode(encoded))
    except ImportError:
        pass
    return result


async def main():
    result = {}
    for server in ("NeKo", "MaBoSS", "PhysiCell"):
        # Import-only library stubs keep schema measurement offline. Tool registrations are real.
        importlib.import_module("tests.test_" + server.lower() + "_mcp_errors")
        module = importlib.import_module(server + ".server")
        tools = await module.mcp.list_tools()
        result[server] = {
            "tool_count": len(tools),
            "input_definitions": size([{"name": t.name, "description": t.description, "parameters": t.input_schema} for t in tools]),
            "complete_listing": size([t.model_dump(mode="json", by_alias=True, exclude_none=True) for t in tools]),
        }
    try:
        from MaBoSS.services.model import model_revision  # noqa: F401
    except ImportError:
        pass
    else:
        import io

        import maboss

        from tests.test_maboss_mcp_errors import _create_simulation_session

        nodes = [maboss.Node(f"N{i}", f"N{(i + 1) % 100}", "@logic ? $up : 0", "@logic ? 0 : $down") for i in range(100)]
        sim = maboss.Simulation(maboss.Network(nodes), parameters={"$up": 1, "$down": 1})
        files = io.StringIO()
        sim.print_bnd(out=files)
        sim.print_cfg(out=files)
        server = importlib.import_module("MaBoSS.server")
        response = server.get_maboss_nodes(
            session_id=_create_simulation_session(sim), selected_nodes=["N0"])
        result["rate_fixture"] = {"nodes": 100, "selected_nodes": 1,
            "raw_bnd_cfg": size(files.getvalue()),
            "targeted_structured_response": size(response.structured_content),
            "complete_tool_result": size(response.model_dump(mode="json", by_alias=True, exclude_none=True))}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
