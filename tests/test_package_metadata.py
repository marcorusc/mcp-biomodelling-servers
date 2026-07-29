"""Tests for package metadata shared by the MCP servers."""

import re
from pathlib import Path

from mcp_biomodelling_servers import __version__


def test_source_version_matches_project_metadata() -> None:
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    match = re.search(
        r'^version = "([^"]+)"$',
        pyproject.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )

    assert match is not None
    assert __version__ == match.group(1)
