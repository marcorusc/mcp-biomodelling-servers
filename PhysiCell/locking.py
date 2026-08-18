"""Session-locking decorators shared by PhysiCell registration modules."""

import inspect
from contextlib import ExitStack
from functools import wraps

from mcp.server.mcpserver.exceptions import ResourceNotFoundError

from .session_manager import session_manager


def session_locked(handler):
    """Run a synchronous handler under its session's exclusive lease."""
    signature = inspect.signature(handler)

    @wraps(handler)
    def locked_handler(*args, **kwargs):
        arguments = signature.bind_partial(*args, **kwargs).arguments
        with session_manager.session_scope(arguments.get("session_id")):
            return handler(*args, **kwargs)

    return locked_handler


def optional_session_locked(handler):
    """Lease an optional session without creating an absent default."""
    signature = inspect.signature(handler)

    @wraps(handler)
    def locked_handler(*args, **kwargs):
        arguments = signature.bind_partial(*args, **kwargs).arguments
        with session_manager.optional_session_scope(
            arguments.get("session_id")
        ):
            return handler(*args, **kwargs)

    return locked_handler


def resource_session_locked(handler):
    """Lease a resource session and preserve typed lookup failures."""
    signature = inspect.signature(handler)

    @wraps(handler)
    def locked_handler(*args, **kwargs):
        arguments = signature.bind_partial(*args, **kwargs).arguments
        session_id = arguments["session_id"]
        stack = ExitStack()
        try:
            stack.enter_context(session_manager.session_scope(session_id))
        except ValueError as exc:
            raise ResourceNotFoundError(
                f"PhysiCell session not found: {session_id}"
            ) from exc

        with stack:
            return handler(*args, **kwargs)

    return locked_handler
