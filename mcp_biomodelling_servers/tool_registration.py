"""Keep expected domain failures actionable across the public MCP boundary."""

import logging
from functools import wraps
from inspect import iscoroutinefunction

from mcp.server.mcpserver.exceptions import ToolError
from mcp.shared.exceptions import MCPError

logger = logging.getLogger(__name__)
INTERNAL_ERROR_MESSAGE = "Internal tool error. See the server logs for diagnostic details."

# These servers use these exceptions for invalid input, absent sessions and
# failed backend operations. Unexpected failures are masked here because SDK
# versions do not consistently hide their exception messages.
DOMAIN_ERRORS = (ValueError, RuntimeError, OSError, KeyError)


def domain_tool(server, **metadata):
    """Register a guarded tool without changing the directly callable handler."""

    def register(handler):
        if iscoroutinefunction(handler):
            @wraps(handler)
            async def guarded(*args, **kwargs):
                try:
                    return await handler(*args, **kwargs)
                except (ToolError, MCPError):
                    raise
                except DOMAIN_ERRORS as exc:
                    raise ToolError(str(exc)) from exc
                except Exception:
                    logger.exception("Unexpected failure in MCP tool %s", handler.__name__)
                    raise ToolError(INTERNAL_ERROR_MESSAGE) from None
        else:
            @wraps(handler)
            def guarded(*args, **kwargs):
                try:
                    return handler(*args, **kwargs)
                except (ToolError, MCPError):
                    raise
                except DOMAIN_ERRORS as exc:
                    raise ToolError(str(exc)) from exc
                except Exception:
                    logger.exception("Unexpected failure in MCP tool %s", handler.__name__)
                    raise ToolError(INTERNAL_ERROR_MESSAGE) from None

        server.tool(**metadata)(guarded)
        return handler

    return register
