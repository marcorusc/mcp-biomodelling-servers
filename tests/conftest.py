"""Keep artifact-producing tests inside their disposable test directories."""
import pytest


@pytest.fixture(autouse=True)
def isolated_artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_BIOMODELLING_ARTIFACT_ROOT", str(tmp_path / "artifact-cache"))
