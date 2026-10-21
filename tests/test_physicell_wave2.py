"""Isolate the real Settings package from legacy protocol-test import stubs."""
import os
import subprocess
import sys


def test_real_physicell_wave2_workflow():
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/physicell_wave2_checks.py", "-q", "--tb=short"],
        env={**os.environ, "MPLBACKEND": "Agg"}, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
