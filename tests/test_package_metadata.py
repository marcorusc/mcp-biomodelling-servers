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


def _project_text() -> str:
    return (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")


def _project_version() -> str:
    match = re.search(
        r'^version = "([^"]+)"$',
        _project_text(),
        flags=re.MULTILINE,
    )

    assert match is not None
    return match.group(1)


def test_source_version_matches_project_metadata() -> None:
    assert __version__ == _project_version()


def test_project_requires_coordinated_neko_release() -> None:
    assert (
        re.search(
            r'^\s*"nekomata>=1\.9\.0,<2",$',
            _project_text(),
            flags=re.MULTILINE,
        )
        is not None
    )


@pytest.mark.parametrize("manifest_path", REGISTRY_MANIFESTS)
def test_registry_manifest_matches_project_metadata(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packages = manifest["packages"]

    assert manifest["version"] == _project_version()
    assert len(packages) == 1
    assert packages[0]["identifier"] == PACKAGE_NAME
    assert packages[0]["version"] == _project_version()
