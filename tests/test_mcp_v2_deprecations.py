"""Regression checks for MCP APIs deprecated by the 2026-07-28 protocol."""

import ast
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parent.parent
SERVER_PATHS = (
    REPOSITORY_ROOT / "MaBoSS" / "server.py",
    REPOSITORY_ROOT / "NeKo" / "server.py",
    REPOSITORY_ROOT / "PhysiCell" / "server.py",
)
DEPRECATED_CONTEXT_LOG_METHODS = {
    "debug",
    "info",
    "warning",
    "error",
    "critical",
    "log",
}


@pytest.mark.parametrize("server_path", SERVER_PATHS, ids=lambda path: path.parent.name)
def test_servers_do_not_use_deprecated_context_logging(server_path: Path) -> None:
    """Keep protocol logging notifications out of MCP v2 server handlers."""
    tree = ast.parse(server_path.read_text(), filename=str(server_path))
    deprecated_calls: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ctx"
            and node.func.attr in DEPRECATED_CONTEXT_LOG_METHODS
        ):
            deprecated_calls.append((node.lineno, node.func.attr))

    assert not deprecated_calls, (
        f"{server_path.relative_to(REPOSITORY_ROOT)} uses deprecated MCP context "
        f"logging helpers: {deprecated_calls}"
    )
