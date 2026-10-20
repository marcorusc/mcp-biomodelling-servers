"""Run the combined workflow against installed NeKo or an explicit source checkout."""

import os
import subprocess
import sys

import pytest


def test_real_neko_wave2_workflow(tmp_path):
    environment = os.environ.copy()
    source = environment.get("NEKO_WAVE2_SOURCE")
    if source:
        environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    environment["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    completed = subprocess.run(
        [sys.executable, "-m", "tests.neko_wave2_workflow"], env=environment,
        capture_output=True, text=True, timeout=90,
    )
    if completed.returncode == 77 and not source and not environment.get("NEKO_WAVE2_REQUIRED"):
        pytest.skip("Installed NeKo predates Wave 2; set NEKO_WAVE2_SOURCE to the candidate checkout.")
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert "REAL_NEKO_WORKFLOW_OK" in completed.stdout
