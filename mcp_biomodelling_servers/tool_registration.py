"""Keep expected domain failures actionable across the public MCP boundary."""

from functools import wraps
from inspect import iscoroutinefunction

from mcp.server.mcpserver.exceptions import ToolError

# These servers use these exceptions for invalid input, absent sessions and
# failed backend operations. Programming errors remain subject to SDK masking.
DOMAIN_ERRORS = (ValueError, RuntimeError, OSError, KeyError)


def domain_tool(server, **metadata):
    """Register a guarded tool without changing the directly callable handler."""

    def register(handler):
        if iscoroutinefunction(handler):
            @wraps(handler)
            async def guarded(*args, **kwargs):
                try:
                    return await handler(*args, **kwargs)
                except DOMAIN_ERRORS as exc:
                    raise ToolError(str(exc)) from exc
        else:
            @wraps(handler)
            def guarded(*args, **kwargs):
                try:
                    return handler(*args, **kwargs)
                except DOMAIN_ERRORS as exc:
                    raise ToolError(str(exc)) from exc

        server.tool(**metadata)(guarded)
        return handler

    return register
