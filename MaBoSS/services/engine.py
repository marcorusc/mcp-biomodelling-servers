"""Validate and simulate through the bundled cmaboss engine, never the executable.

pyMaBoSS validates every new ``Simulation`` and runs simulations by launching
the external ``MaBoSS`` executable, which pip and uvx do not install. The server
instead uses the compiled ``cmaboss`` engine that pyMaBoSS depends on, in a
child Python process: cmaboss holds the GIL for a whole simulation and aborts
the process on some malformed input, so running it in-process could block or
kill the server. Results keep the file-backed interface of pyMaBoSS's own
``Result``, so run preservation copies the exact engine inputs and outputs.
"""

import importlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

WORKER = Path(__file__).with_name("engine_worker.py")
PREFIX = "res"
REJECTED = 2


def _engine_call(simulation, action: str, directory: Path) -> subprocess.CompletedProcess:
    (directory / "input.bnd").write_text(str(simulation.network), encoding="utf-8")
    (directory / "input.cfg").write_text(simulation.str_cfg(), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(WORKER), action, str(directory), simulation.get_cmaboss().__name__],
        capture_output=True, text=True, check=False,
    )


def _describe(completed: subprocess.CompletedProcess) -> str:
    detail = completed.stderr.strip().splitlines()
    return detail[-1] if detail else f"exit status {completed.returncode}"


def install(pymaboss) -> None:
    """Route pyMaBoSS validation and simulation through the cmaboss worker."""
    stored_result = importlib.import_module(pymaboss.__name__ + ".results.storedresult").StoredResult
    simulation_type = pymaboss.Simulation

    class EngineResult(stored_result):
        """Saved cmaboss outputs with the attributes of a file-backed Result."""

        def __init__(self, path: Path, completed: subprocess.CompletedProcess):
            super().__init__(str(path), PREFIX)
            self.prefix = PREFIX
            self._bnd = str(path / "input.bnd")
            self._cfg = str(path / "input.cfg")
            self._err = completed.returncode
            self.error_detail = _describe(completed) if completed.returncode else ""

    def check(self, command=None):
        """Raise ValueError if cmaboss rejects the model; return no messages."""
        directory = Path(tempfile.mkdtemp(prefix="maboss-check-"))
        try:
            completed = _engine_call(self, "check", directory)
        finally:
            shutil.rmtree(directory, ignore_errors=True)
        if completed.returncode == REJECTED:
            raise ValueError(f"MaBoSS rejected the model: {_describe(completed)}")
        if completed.returncode:
            raise RuntimeError(f"The cmaboss engine failed while checking the model: {_describe(completed)}")
        return []

    def run(self, **options):
        """Simulate full trajectories in a child process; see the module docstring."""
        if options:
            raise ValueError("The MaBoSS server runs only complete trajectory simulations: " + ", ".join(options))
        directory = Path(tempfile.mkdtemp(prefix="maboss-run-"))
        completed = _engine_call(self, "run", directory)
        if completed.stderr.strip():
            (directory / "engine_stderr.txt").write_text(completed.stderr, encoding="utf-8")
        return EngineResult(directory, completed)

    simulation_type.check = check
    simulation_type.run = run
