"""Revision-scoped worker execution, integrity inventories, and exports."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path

from mcp_biomodelling_servers.structured_outputs import artifact_file_summary

WORKER = Path(__file__).resolve().parents[1] / "worker.py"


def files(directory: Path) -> list[Path]:
    result = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlinks are not allowed in BioMASS artifact trees.")
        if (
            path.is_file()
            and "__pycache__" not in path.parts
            and ".cache" not in path.parts
        ):
            path.resolve().relative_to(directory.resolve())
            result.append(path)
    return result


def summaries(directory: Path, sid: str):
    return [artifact_file_summary(p, session_id=sid) for p in files(directory)]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(directory: Path) -> None:
    inventory = {
        str(p.relative_to(directory)): digest(p)
        for p in files(directory)
        if p.name != "integrity.json"
    }
    (directory / "integrity.json").write_text(
        json.dumps(inventory, indent=2), encoding="utf-8"
    )


def verify(directory: Path) -> None:
    inventory = json.loads((directory / "integrity.json").read_text())
    actual = {
        str(p.relative_to(directory)): digest(p)
        for p in files(directory)
        if p.name != "integrity.json"
    }
    if inventory != actual:
        raise ValueError(
            "Revision artifacts changed after generation; regenerate the model."
        )


def revision_path(directory: Path, revision: str, known: list[str]) -> Path:
    if revision not in known or not re.fullmatch(r"rev_[0-9a-f]{32}", revision):
        raise ValueError("Unknown model revision.")
    path = directory / revision
    path.resolve().relative_to(directory.resolve())
    if path.is_symlink():
        raise ValueError("Symlinked revisions are not supported.")
    verify(path)
    return path


def run_worker(directory: Path, request: dict, timeout_seconds: float = 60) -> dict:
    # A multi-worker tool may pass the fractional remainder of its public budget.
    if not 0 < timeout_seconds <= 300:
        raise ValueError("Worker timeout must be positive and at most 300 seconds.")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "request.json").write_text(
        json.dumps(request, allow_nan=False, indent=2), encoding="utf-8"
    )
    environment = os.environ.copy()
    environment.update(
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(directory / ".cache"),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONHASHSEED="0",
    )
    with (directory / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-B", str(WORKER)],
            cwd=directory,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=(os.name == "posix"),
        )
        try:
            process.wait(timeout=timeout_seconds)
        except BaseException:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
            raise
    if process.returncode:
        error_path = directory / "error.json"
        if error_path.exists():
            error = json.loads(error_path.read_text())
            raise RuntimeError(f"{error['type']}: {error['message']}")
        raise RuntimeError(
            f"BioMASS worker exited with code {process.returncode}. See worker log."
        )
    return json.loads((directory / "result.json").read_text())


def job(
    directory: Path, request: dict, timeout_seconds: float, snapshot: dict | None = None
) -> tuple[Path, dict]:
    prefix = "rev" if request["operation"] == "generate" else request["operation"]
    name = prefix + "_" + uuid.uuid4().hex
    pending = directory / (".pending_" + name)
    destination = directory / name
    try:
        result = run_worker(pending, request, timeout_seconds)
        if snapshot is not None:
            (pending / "snapshot.json").write_text(
                json.dumps(snapshot, indent=2), encoding="utf-8"
            )
        shutil.rmtree(pending / ".cache", ignore_errors=True)
        # Strip installed template bytecode copied by BioMASS itself.
        for cache in pending.rglob("__pycache__"):
            shutil.rmtree(cache)
        seal(pending)
        pending.rename(destination)
        return destination, result
    except BaseException as exc:
        # Preserve diagnostics only. Incomplete models are never published.
        if pending.exists():
            log = pending / "worker.log"
            if log.is_file():
                log.rename(directory / (name + ".failed.log"))
            shutil.rmtree(pending)
        if isinstance(exc, subprocess.TimeoutExpired):
            raise TimeoutError(
                f"BioMASS {request['operation']} exceeded {timeout_seconds} seconds; worker terminated."
            ) from exc
        raise


def export_bundle(directory: Path, source: Path, revision: str) -> Path:
    verify(source)
    destination = directory / ("export_" + uuid.uuid4().hex + ".zip")
    candidates = [(source, "model")]
    for child in sorted(directory.iterdir()):
        if child.is_dir() and child.name.startswith(("graph_", "simulate_")):
            request_file = child / "request.json"
            if (
                request_file.is_file()
                and json.loads(request_file.read_text()).get("revision") == revision
            ):
                verify(child)
                candidates.append((child, child.name))
    try:
        with zipfile.ZipFile(
            destination, "x", compression=zipfile.ZIP_DEFLATED
        ) as bundle:
            for path, prefix in candidates:
                for artifact in files(path):
                    bundle.write(artifact, f"{prefix}/{artifact.relative_to(path)}")
            bundle.writestr("requirements.txt", "biomass==0.14.0\n")
            bundle.writestr("run_simulation.py", REPRODUCE)
        return destination
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


REPRODUCE = '''"""Run an explicitly configured scenario from this exported bundle.
Usage: python run_simulation.py simulate_<run_id>
With no run directory, describe the available saved scenarios.
"""
import json
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
if len(sys.argv) != 2:
    print("Choose a saved scenario:", *[p.name for p in root.glob("simulate_*")])
    raise SystemExit(0)
run = (root / sys.argv[1]).resolve()
if run.parent != root or not run.name.startswith("simulate_"):
    raise ValueError("Select an exported simulation directory.")
import numpy as np
from biomass import create_model
from biomass.dynamics.solver import solve_ode
sys.path.insert(0, str(run))
os.chdir(run)
model = create_model("generated_model")
saved = json.loads((run / "simulation.json").read_text())
for condition in saved["conditions"]:
    p = [condition["parameters"][name] for name in model.parameters]
    y0 = [condition["initials"][name] for name in model.species]
    sol = solve_ode(model.problem.diffeq, y0, model.problem.t, tuple(p))
    if sol is None or not np.isfinite(sol.y).all():
        raise RuntimeError("Simulation failed.")
    np.savetxt(run / (condition["name"] + "_reproduced.csv"),
               np.column_stack([sol.t, sol.y.T]), delimiter=",",
               header=",".join(["time", *model.species]), comments="")
'''
