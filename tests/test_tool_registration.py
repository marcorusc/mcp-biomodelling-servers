"""Public-protocol regression coverage for expected-error translation."""

import asyncio

import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer

from mcp_biomodelling_servers.tool_registration import domain_tool


@pytest.mark.parametrize('mode', ['auto', 'legacy'])
@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('error_type', [ValueError, RuntimeError, OSError, KeyError, TypeError, AttributeError])
def test_domain_errors_are_actionable_and_programming_errors_remain_masked(mode, asynchronous, error_type):
    server = MCPServer('boundary-test')
    message = 'diagnostic-marker-should-only-be-visible-for-domain-errors'

    def fail(value: int) -> str:
        raise error_type(message)

    async def async_fail(value: int) -> str:
        raise error_type(message)

    handler = async_fail if asynchronous else fail
    assert domain_tool(server, name='failure')(handler) is handler

    async def exercise():
        async with Client(server, mode=mode) as client:
            listing = await client.list_tools()
            assert listing.tools[0].input_schema['properties']['value']['type'] == 'integer'
            result = await client.call_tool('failure', {'value': 1})
            assert result.is_error
            text = '\n'.join(item.text for item in result.content if hasattr(item, 'text'))
            assert (message in text) == (error_type not in (TypeError, AttributeError))

    asyncio.run(exercise())


@pytest.mark.parametrize('mode', ['auto', 'legacy'])
@pytest.mark.parametrize('asynchronous', [False, True])
def test_registration_preserves_successful_structured_output(mode, asynchronous):
    server = MCPServer('boundary-test')

    def increment(value: int) -> dict[str, int]:
        return {'value': value + 1}

    async def async_increment(value: int) -> dict[str, int]:
        return increment(value)

    handler = async_increment if asynchronous else increment
    domain_tool(server, name='increment', structured_output=True)(handler)

    async def exercise():
        async with Client(server, mode=mode) as client:
            result = await client.call_tool('increment', {'value': 7})
            assert not result.is_error
            assert result.structured_content == {'value': 8}

    asyncio.run(exercise())
