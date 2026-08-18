"""Session-locking decorators shared by MaBoSS registration modules."""

import inspect
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


def resource_session_locked(handler):
    """Lease a resource session and expose unknown IDs as resource errors."""
    signature = inspect.signature(handler)

    @wraps(handler)
    def locked_handler(*args, **kwargs):
        arguments = signature.bind_partial(*args, **kwargs).arguments
        try:
            with session_manager.session_scope(arguments.get("session_id")):
                return handler(*args, **kwargs)
        except KeyError as exc:
            raise ResourceNotFoundError(exc.args[0]) from exc

    return locked_handler
