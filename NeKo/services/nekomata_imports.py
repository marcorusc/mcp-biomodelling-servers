"""Deferred nekomata imports, so that starting the server needs no network.

Importing any ``neko`` module also imports the ``omnipath`` client, which
contacts omnipathdb.org while it is imported, and ``pypath_common``, which
starts a session log. Binding the server's nekomata names to these stand-ins
keeps ``initialize`` and ``tools/list`` free of both; the first tool call that
uses a name imports its module.
"""
from __future__ import annotations

import importlib
from threading import Lock
from typing import Any

_LOCK = Lock()


class DeferredAttribute:
    """Stand-in for a nekomata attribute, resolved once on first use."""

    def __init__(self, module: str, name: str) -> None:
        self._module = module
        self._name = name
        self._target: Any = None

    def resolve(self) -> Any:
        """Import the module if needed and return the real attribute."""
        if self._target is None:
            with _LOCK:
                if self._target is None:
                    module = importlib.import_module(self._module)
                    self._target = getattr(module, self._name)
        return self._target

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.resolve()(*args, **kwargs)

    def __getattr__(self, attribute: str) -> Any:
        # Instance fields are found before this hook; guard copies made without __init__.
        if attribute in {"_module", "_name", "_target"}:
            raise AttributeError(attribute)
        return getattr(self.resolve(), attribute)

    def __repr__(self) -> str:
        return f"<deferred {self._module}.{self._name}>"
