"""Preserve actionable domain errors at the MCP boundary."""

from functools import wraps

from mcp.server.mcpserver.exceptions import ResourceError, ToolError

from ..app import mcp


def tool(**metadata):
    """Register a guarded MCP callable while retaining the plain Python handler."""

    def register(handler):
        @wraps(handler)
        def guarded(*args, **kwargs):
            try:
                return handler(*args, **kwargs)
            except (ValueError, RuntimeError, OSError) as exc:
                raise ToolError(str(exc)) from exc

        mcp.tool(**metadata)(guarded)
        return handler

    return register


def resource(uri, **metadata):
    """Translate expected read failures without losing useful error details."""

    def register(handler):
        @wraps(handler)
        def guarded(*args, **kwargs):
            try:
                return handler(*args, **kwargs)
            except (ValueError, RuntimeError, OSError) as exc:
                raise ResourceError(str(exc)) from exc

        mcp.resource(uri, **metadata)(guarded)
        return handler

    return register
