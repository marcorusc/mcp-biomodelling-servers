"""Cross-server regression tests for MCP initialization metadata."""

import asyncio
from typing import Literal

import pytest
from mcp import Client


async def _initialization_metadata(
    mode: Literal["auto", "legacy"],
) -> dict[str, dict[str, str | None]]:
    """Read each server's metadata through the public MCP client."""
    from tests import test_maboss_mcp_errors as maboss_tests
    from tests import test_neko_mcp_errors as neko_tests
    from tests import test_physicell_mcp_errors as physicell_tests

    server_modules = {
        "MaBoSS": maboss_tests.maboss_server,
        "NeKo": neko_tests.neko_server,
        "PhysiCell": physicell_tests.physicell_server,
    }
    metadata = {}
    for expected_name, server_module in server_modules.items():
        async with Client(server_module.mcp, mode=mode) as client:
            assert client.server_info is not None
            metadata[expected_name] = {
                "name": client.server_info.name,
                "version": client.server_info.version,
                "instructions": client.instructions,
                "local_instructions": server_module.mcp.instructions,
            }
    return metadata


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_all_servers_publish_initialization_instructions(
    mode: Literal["auto", "legacy"],
) -> None:
    """Instructions must survive both v2 and legacy initialization paths."""
    from tests import test_maboss_mcp_errors as maboss_tests
    from tests import test_neko_mcp_errors as neko_tests
    from tests import test_physicell_mcp_errors as physicell_tests

    expected = {
        "MaBoSS": {
            "version": maboss_tests.maboss_server.__version__,
            "instructions": (
                maboss_tests.maboss_server.MABOSS_SERVER_INSTRUCTIONS
            ),
            "manual": "docs://maboss/agent_manual",
            "prompt": "maboss_workflow_prompt",
        },
        "NeKo": {
            "version": neko_tests.neko_server.__version__,
            "instructions": neko_tests.neko_server.NEKO_SERVER_INSTRUCTIONS,
            "manual": "docs://neko/agent_manual",
            "prompt": "neko_workflow_prompt",
        },
        "PhysiCell": {
            "version": physicell_tests.physicell_server.__version__,
            "instructions": (
                physicell_tests.physicell_server.PHYSICELL_SERVER_INSTRUCTIONS
            ),
            "manual": "docs://physicell/agent_manual",
            "prompt": "physicell_workflow_prompt",
        },
    }

    metadata = asyncio.run(_initialization_metadata(mode))

    assert set(metadata) == set(expected)
    for server_name, expected_metadata in expected.items():
        published = metadata[server_name]
        instructions = expected_metadata["instructions"]
        assert published["name"] == server_name
        assert published["version"] == expected_metadata["version"]
        assert published["instructions"] == instructions
        assert published["local_instructions"] == instructions
        assert expected_metadata["manual"] in instructions
        assert expected_metadata["prompt"] in instructions
