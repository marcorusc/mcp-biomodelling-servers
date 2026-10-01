"""Transactional publication helpers for MaBoSS artifact sets."""

import hashlib
import json
import logging
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from mcp_biomodelling_servers.artifact_manager import safe_artifact_path

logger = logging.getLogger(__name__)


def require_unused_artifact_paths(paths: list[Path]) -> None:
    """Reject a handoff prefix when any destination already exists."""
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing MaBoSS handoff artifacts: "
            + ", ".join(str(path) for path in existing)
            + ". Choose a different artifact_prefix."
        )


def link_artifact_without_overwrite(
    source: Path,
    destination: Path,
) -> None:
    """Atomically publish one complete temporary artifact if absent."""
    if not source.is_file():
        raise FileNotFoundError(
            f"Expected temporary handoff artifact was not created: {source}"
        )
    try:
        os.link(source, destination)
    except FileExistsError as exc:
        raise FileExistsError(
            "Refusing to overwrite a MaBoSS handoff artifact created "
            f"concurrently: {destination}"
        ) from exc


def rollback_artifacts(paths: list[Path]) -> None:
    """Best-effort cleanup for an incomplete multi-file handoff."""
    for path in reversed(paths):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning(
                "Could not roll back incomplete handoff artifact %s",
                path,
                exc_info=True,
            )


def preserve_simulation_result(result, artifact_dir: Path, session_id: str) -> Path:
    """Preserve one file-backed PyMaBoSS run before its temporary files expire.

    All published names are flat and unique. The inventory is published last;
    the existing caller still owns the legacy final-row result.csv and response.
    This copies observed data without selecting a grid or validating science.
    """
    run_id = "run_" + uuid.uuid4().hex
    source_dir = Path(result._path)
    prefix = result.prefix
    if not isinstance(prefix, str) or not prefix or Path(prefix).name != prefix:
        raise ValueError("Invalid PyMaBoSS result prefix")
    if not source_dir.is_dir() or source_dir.is_symlink():
        raise ValueError("Invalid PyMaBoSS result directory")

    def source_file(value) -> Path:
        path = Path(value)
        if path.parent.resolve() != source_dir.resolve() or path.is_symlink():
            raise ValueError("PyMaBoSS result file is outside its run directory")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    # Copy the exact inputs passed to the engine, not a later reserialization.
    sources = [("input.bnd", source_file(result._bnd)),
               ("input.cfg", source_file(result._cfg))]
    raw_files = [source_file(path) for path in sorted(source_dir.iterdir())
                 if path.name.startswith(prefix)]
    probability_file = source_file(result.get_probtraj_file())
    if probability_file.resolve() not in {path.resolve() for path in raw_files}:
        raise ValueError("Probability trajectory is missing from engine outputs")
    sources.extend(("engine." + path.name, path) for path in raw_files)
    published = []
    with tempfile.TemporaryDirectory(prefix=".maboss-run-", dir=artifact_dir) as temporary:
        staging = Path(temporary)
        inventory = []

        def record(name: str, role: str, source_name: str | None = None) -> None:
            path = staging / name
            entry = {"name": name, "role": role, "size_bytes": path.stat().st_size,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            if source_name is not None:
                entry["source_name"] = source_name
            inventory.append(entry)

        for suffix, source in sources:
            name = run_id + "." + suffix
            shutil.copyfile(source, safe_artifact_path(staging, name))
            record(name, "engine_input" if suffix.startswith("input.") else "engine_output", source.name)

        states = result.get_states_probtraj()
        nodes = result.get_nodes_probtraj()
        for suffix, table in (("states.csv", states), ("nodes.csv", nodes)):
            name = run_id + "." + suffix
            table.to_csv(safe_artifact_path(staging, name), index=True, index_label="Time")
            record(name, "state_probabilities" if suffix == "states.csv" else "node_marginals")
        manifest_name = run_id + ".json"
        manifest = {
            "server": "MaBoSS", "session_id": session_id, "run_id": run_id,
            "preserved_at": datetime.now(timezone.utc).isoformat(),
            "engine_result_prefix": prefix,
            "state_timepoints": states.index.tolist(),
            "node_timepoints": nodes.index.tolist(),
            "files": inventory,
        }
        safe_artifact_path(staging, manifest_name).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        names = [entry["name"] for entry in inventory] + [manifest_name]
        destinations = [safe_artifact_path(artifact_dir, name) for name in names]
        require_unused_artifact_paths(destinations)
        try:
            for name, destination in zip(names, destinations, strict=True):
                link_artifact_without_overwrite(staging / name, destination)
                published.append(destination)
        except Exception:
            rollback_artifacts(published)
            raise
    return safe_artifact_path(artifact_dir, manifest_name)
