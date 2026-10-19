"""Public-protocol regression coverage for expected-error translation."""

import asyncio

import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.shared.exceptions import MCPError

from mcp_biomodelling_servers.tool_registration import (
    INTERNAL_ERROR_MESSAGE,
    domain_tool,
)


@pytest.mark.parametrize('mode', ['auto', 'legacy'])
@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('error_type', [ValueError, RuntimeError, OSError, KeyError, TypeError, AttributeError, AssertionError, ZeroDivisionError, Exception])
def test_domain_errors_are_actionable_and_programming_errors_remain_masked(mode, asynchronous, error_type, caplog):
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
            expected = error_type in (ValueError, RuntimeError, OSError, KeyError)
            assert (message in text) == expected
            if not expected:
                assert INTERNAL_ERROR_MESSAGE in text

    asyncio.run(exercise())
    unexpected_logs = [record for record in caplog.records
                       if record.name == 'mcp_biomodelling_servers.tool_registration']
    if error_type in (ValueError, RuntimeError, OSError, KeyError):
        assert not unexpected_logs
    else:
        assert len(unexpected_logs) == 1
        assert unexpected_logs[0].exc_info[0] is error_type
        assert str(unexpected_logs[0].exc_info[1]) == message


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


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('error', [
    asyncio.CancelledError(), KeyboardInterrupt(), SystemExit(),
    ToolError('intentional tool failure'),
    MCPError(-32602, 'intentional protocol error', {'reason': 'fixture'}),
])
def test_registration_preserves_control_flow_and_explicit_errors(asynchronous, error, caplog):
    # Capture the registered callable to inspect exceptions before SDK handling.
    class Registration:
        def tool(self, **metadata):
            def register(handler):
                self.handler = handler
            return register

    server = Registration()

    def fail():
        raise error

    async def async_fail():
        raise error

    handler = async_fail if asynchronous else fail
    assert domain_tool(server)(handler) is handler
    if asynchronous:
        async def exercise():
            # Check the wrapper directly: Python 3.10's task boundary may
            # recreate CancelledError when asyncio.run retrieves the result.
            with pytest.raises(type(error)) as caught:
                await server.handler()
            assert caught.value is error

        asyncio.run(exercise())
    else:
        with pytest.raises(type(error)) as caught:
            server.handler()
        assert caught.value is error
    assert not caplog.records
