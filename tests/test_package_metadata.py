"""Tests for package metadata shared by the MCP servers."""

import json
import re
from pathlib import Path

import pytest

from mcp_biomodelling_servers import __version__

PROJECT_ROOT = Path(__file__).parent.parent
REGISTRY_MANIFESTS = (
    PROJECT_ROOT / "NeKo" / "server.json",
    PROJECT_ROOT / "MaBoSS" / "server.json",
    PROJECT_ROOT / "PhysiCell" / "server.json",
)
PACKAGE_NAME = "mcp-biomodelling-servers"


def _project_version() -> str:
    pyproject = PROJECT_ROOT / "pyproject.toml"
    match = re.search(
        r'^version = "([^"]+)"$',
        pyproject.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )

    assert match is not None
    return match.group(1)


def test_source_version_matches_project_metadata() -> None:
    assert __version__ == _project_version()


@pytest.mark.parametrize("manifest_path", REGISTRY_MANIFESTS)
def test_registry_manifest_matches_project_metadata(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packages = manifest["packages"]

    assert manifest["version"] == _project_version()
    assert len(packages) == 1
    assert packages[0]["identifier"] == PACKAGE_NAME
    assert packages[0]["version"] == _project_version()
