"""Collision-safe loading of the external :mod:`maboss` distribution."""

import importlib
import os
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

_EXECUTABLE = re.compile(r"MaBoSS(?:_(?:\d+n|dn))?(?:\.exe)?")


def _resolved_search_path(entry: str) -> Path | None:
    """Resolve a Python search entry without failing on unusable paths."""
    try:
        return Path(entry or os.getcwd()).resolve()
    except OSError:
        return None


def load_pymaboss() -> ModuleType:
    """Import pyMaBoSS without resolving this server package by mistake.

    ``MaBoSS`` and the third-party ``maboss`` package collide on the default
    case-insensitive Windows and macOS filesystems. The installed distribution
    also contains an entry-point module named ``maboss.py``. Temporarily hiding
    only this package's parent directory preserves the established workaround
    without leaving global search-path mutations behind.
    """
    package_parent = Path(__file__).resolve().parent.parent
    original_search_path = sys.path.copy()
    sys.path[:] = [
        entry
        for entry in original_search_path
        if _resolved_search_path(entry) != package_parent
    ]
    try:
        module = importlib.import_module("maboss")
    finally:
        sys.path[:] = original_search_path

    module_file = getattr(module, "__file__", None)
    if module_file is not None:
        resolved_module = Path(module_file).resolve()
        if resolved_module == package_parent / "maboss.py" or (
            package_parent / "MaBoSS"
        ) in resolved_module.parents:
            raise ImportError(
                "The MaBoSS server package shadowed the external `maboss` "
                "distribution. Install pyMaBoSS and use the packaged entry point."
            )
    return module


pymaboss = load_pymaboss()


@contextmanager
def maboss_executable() -> Iterator[None]:
    """Name the missing MaBoSS executable instead of a bare FileNotFoundError.

    pyMaBoSS runs the external ``MaBoSS`` executable to check, load and run
    models. pip and uvx install only the Python packages, not the executable.
    """
    try:
        yield
    except FileNotFoundError as exc:
        name = Path(str(exc.filename or "")).name
        if not _EXECUTABLE.fullmatch(name):
            raise
        raise RuntimeError(
            f"The MaBoSS executable {name!r} is not on PATH. pyMaBoSS needs it to "
            "load and run models, and pip/uvx installs do not provide it. Install "
            "MaBoSS and put its executables on PATH; see the MaBoSS server README."
        ) from exc
