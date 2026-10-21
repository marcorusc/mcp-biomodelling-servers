"""Read immutable, integrity-checked run artifacts without a live session."""
import json
import re
import uuid
from pathlib import Path

import pandas as pd

from mcp_biomodelling_servers.artifact_manager import artifact_bases, safe_artifact_path
from mcp_biomodelling_servers.handoff import sha256_file

from .model import inspect_nodes


def effective_configuration(session, sim):
    return {"parameters": dict(sim.param), "initial_state": sim.network.str_istate(),
            "outputs": list(sim.network.get_output()), "readouts": session.readouts,
            "nodes": inspect_nodes(sim, list(sim.network), session.readouts),
            "upstream_neko_manifest": session.upstream_neko_manifest_path}


def load_run(root, run_id, session_id=None):
    if not re.fullmatch(r"run_[a-f0-9]{32}", run_id):
        raise ValueError("Invalid run ID")
    if session_id is not None and (Path(session_id).name != session_id or session_id in (".", "..")):
        raise ValueError("Invalid session ID")
    candidates = []
    for base in artifact_bases(root):
        if base.is_symlink():
            raise ValueError("Symlinked artifact root")
        directories = [base / session_id] if session_id else (base.iterdir() if base.is_dir() else [])
        for directory in directories:
            path = directory / (run_id + ".json")
            if path.is_file():
                if directory.is_symlink() or path.is_symlink():
                    raise ValueError("Symlinked run artifacts")
                candidates.append(path)
    if len(candidates) != 1:
        raise ValueError("Run not found for this session, or identity is ambiguous")
    path = candidates[0]
    manifest = json.loads(path.read_text())
    if manifest.get("run_id") != run_id or manifest.get("session_id") != path.parent.name:
        raise ValueError("Run identity mismatch")
    if manifest.get("status", "complete") != "complete":
        raise ValueError("Run is incomplete")
    for entry in manifest["files"]:
        name = entry["name"]
        if Path(name).name != name:
            raise ValueError("Invalid run artifact name")
        artifact = path.parent / name
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError("Missing or symlinked run artifact")
        if artifact.stat().st_size != entry["size_bytes"] or sha256_file(artifact) != entry["sha256"]:
            raise ValueError("Run artifact integrity mismatch: " + name)
    return manifest, path


def run_table(manifest, path, *, offset=0, limit=100, final=False):
    entries = [entry for entry in manifest["files"] if entry["role"] == "state_probabilities"]
    if len(entries) != 1:
        raise ValueError("Missing or ambiguous state table")
    file = path.parent / entries[0]["name"]
    # Chunked reading bounds resident memory even for historical large outputs.
    selected = None
    columns = pd.read_csv(file, nrows=0).columns[:101].tolist()
    for chunk in pd.read_csv(file, index_col=0, chunksize=1000, usecols=columns):
        if final:
            selected = chunk.tail(1)
        elif offset >= len(chunk):
            offset -= len(chunk)
        else:
            piece = chunk.iloc[offset:offset + limit]
            selected = piece if selected is None else pd.concat([selected, piece])
            limit -= len(piece)
            offset = 0
            if limit == 0:
                break
    return selected if selected is not None else pd.DataFrame()


def record_failure(artifact_dir, session_id, phase, error, revision, result=None):
    """Preserve evidence where possible without masking the original failure."""
    try:
        return _record_failure(artifact_dir, session_id, phase, error, revision, result)
    except OSError as exc:
        return f"unavailable (diagnostic storage also failed: {exc})"


def _record_failure(artifact_dir, session_id, phase, error, revision, result=None):
    """A durable incomplete invocation record, never a completed run inventory."""
    failure_id = "failed_" + uuid.uuid4().hex
    path = safe_artifact_path(artifact_dir, failure_id + ".json")
    evidence, limitations = [], []
    if result is not None:
        source = Path(getattr(result, "_path", "/nonexistent"))
        if source.is_dir() and not source.is_symlink():
            for file in source.iterdir():
                if not file.is_file() or file.is_symlink():
                    continue
                if file.stat().st_size > 16 * 1024 * 1024:
                    limitations.append(file.name + ": exceeds diagnostic copy bound")
                    continue
                try:
                    data = file.read_bytes()
                    target = safe_artifact_path(artifact_dir, failure_id + "." + file.name)
                    with target.open("xb") as out:
                        out.write(data)
                    evidence.append({"name": target.name, "sha256": sha256_file(target), "size_bytes": len(data)})
                except OSError as exc:
                    limitations.append(file.name + ": " + str(exc))
    with path.open("x") as out:
        json.dump({"status": "incomplete", "session_id": session_id, "phase": phase,
                   "model_revision": revision, "error": str(error)[:4000],
                   "diagnostic_files": evidence, "limitations": limitations}, out)
    return path
