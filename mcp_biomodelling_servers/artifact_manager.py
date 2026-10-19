"""Shared cache-backed artifact storage, with read-only legacy discovery."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .timestamps import utc_now

METADATA_FILENAME = "session_meta.json"


def artifact_base(server_root: Path) -> Path:
    """Current write root: configured/cache root, server namespace."""
    configured = os.environ.get("MCP_BIOMODELLING_ARTIFACT_ROOT")
    if configured:
        root = Path(configured).expanduser()
        if not root.is_absolute():
            raise ValueError("MCP_BIOMODELLING_ARTIFACT_ROOT must be absolute.")
    else:
        cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))).expanduser()
        if not cache.is_absolute():
            cache = Path.home() / ".cache"
        root = cache / "mcp-biomodelling" / "artifacts"
    return root / Path(server_root).name


def _validate_session(session_id: str) -> None:
    if not session_id or "/" in session_id or "\\" in session_id or session_id in (".", ".."):
        raise ValueError(f"Invalid session_id for artifact directory: {session_id!r}")


def artifact_bases(server_root: Path) -> tuple[Path, ...]:
    """Current storage first, then the unmoved package-local legacy storage."""
    return tuple(dict.fromkeys((artifact_base(server_root), server_root / "artifacts")))


def get_artifact_dir(server_root: Path, session_id: str) -> Path:
    """Return and create the current write directory; never resume legacy writes."""
    _validate_session(session_id)
    artifact_dir = artifact_base(server_root) / session_id
    if artifact_dir.is_symlink() or artifact_dir.parent.is_symlink():
        raise ValueError("Symlinked artifact directories are not supported.")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def safe_artifact_path(artifact_dir: Path, filename: str) -> Path:
    """Construct a validated path for a file inside *artifact_dir*.

    Strips any directory components from *filename* and confirms the
    resulting path stays within the artifact sandbox (prevents traversal).

    Args:
        artifact_dir: The session artifact directory (from :func:`get_artifact_dir`).
        filename:     Desired filename.  Only the basename is kept.

    Returns:
        Resolved ``Path`` inside *artifact_dir*.

    Raises:
        ValueError: If *filename* is empty or would escape the sandbox.
    """
    safe_name = Path(filename).name
    if not safe_name:
        raise ValueError(f"Invalid filename: {filename!r}")
    target = (artifact_dir / safe_name).resolve()
    try:
        target.relative_to(artifact_dir.resolve())
    except ValueError as error:
        raise ValueError(
            f"Path traversal detected: {filename!r} resolves outside the artifact sandbox."
        ) from error
    return target


def list_artifacts(server_root: Path, session_id: str | None = None) -> list[Path]:
    """List artifact files for a session (or all sessions).

    Args:
        server_root: Server directory.
        session_id:  If given, list only that session's files.
                     If ``None``, list all files across every session.

    Returns:
        Sorted list of ``Path`` objects.
    """
    if session_id is not None:
        _validate_session(session_id)
    files: list[Path] = []
    for base in artifact_bases(server_root):
        if not base.is_dir():
            continue
        directories = [base / session_id] if session_id is not None else base.iterdir()
        for entry in directories:
            if entry.is_dir():
                files.extend(p for p in entry.iterdir() if p.is_file())
    return sorted(set(files))


def clean_artifacts(server_root: Path, session_id: str) -> int:
    """Remove all artifact files for *session_id*.

    Args:
        server_root: Server directory.
        session_id:  Target session.

    Returns:
        Number of files removed.
    """
    _validate_session(session_id)
    session_dir = artifact_base(server_root) / session_id
    if session_dir.is_symlink() or session_dir.parent.is_symlink():
        raise ValueError("Symlinked artifact directories are not supported.")
    if not session_dir.exists():
        return 0
    count = 0
    for item in list(session_dir.iterdir()):
        if item.is_file():
            item.unlink()
            count += 1
    # Remove directory if now empty
    try:
        session_dir.rmdir()
    except OSError:
        pass
    return count


# ---------------------------------------------------------------------------
# Session metadata helpers
# ---------------------------------------------------------------------------

def write_session_meta(
    server_root: Path,
    session_id: str,
    server_name: str,
    label: str | None = None,
) -> None:
    """Write a ``session_meta.json`` file into the session artifact directory.

    This is called once at session-creation time so that the artifact directory
    can be identified even after the server process has restarted.

    Args:
        server_root: Server directory (``Path(__file__).parent`` in server code).
        session_id:  UUID session identifier.
        server_name: Human-readable server name, e.g. ``"NeKo"``.
        label:       Optional free-text label supplied by the user/LLM.
    """
    art_dir = get_artifact_dir(server_root, session_id)
    meta: dict[str, Any] = {
        "session_id": session_id,
        "server": server_name,
        "created_at": utc_now().isoformat(),
        "label": label or "",
    }
    meta_path = art_dir / METADATA_FILENAME
    meta_path.write_text(json.dumps(meta, indent=2))


def read_session_meta(server_root: Path, session_id: str) -> dict[str, Any] | None:
    """Read the metadata dict for *session_id*, or ``None`` if not found.

    Args:
        server_root: Server directory.
        session_id:  Session whose metadata to read.

    Returns:
        Parsed JSON dict, or ``None`` if the file does not exist / is invalid.
    """
    _validate_session(session_id)
    for base in artifact_bases(server_root):
        directory = base / session_id
        if directory.exists():
            # A current orphan must not inherit metadata from a legacy snapshot.
            return _read_meta(directory / METADATA_FILENAME)
    return None


def _read_meta(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def list_artifact_sessions(
    server_root: Path,
    server_name: str | None = None,
) -> list[dict[str, Any]]:
    """Scan the artifacts directory and return metadata for every session found on disk.

    Sessions that have a ``session_meta.json`` are returned with full metadata;
    orphan directories (no metadata file) are included with minimal info so they
    remain discoverable.

    Args:
        server_root:  Server directory.
        server_name:  When set, only sessions whose metadata ``server`` field
                      matches this name are returned.  Pass ``None`` to return all.

    Returns:
        List of dicts, each containing at minimum:
        ``session_id``, ``server``, ``created_at``, ``label``, ``files``.
        Sorted newest-first by ``created_at``.
    """
    results: list[dict[str, Any]] = []
    for base in artifact_bases(server_root):
        if not base.is_dir():
            continue
        for entry in sorted(base.iterdir()):
            if not entry.is_dir():
                continue
            meta = _read_meta(entry / METADATA_FILENAME) or {
                "session_id": entry.name, "server": "unknown",
                "created_at": "", "label": "",
            }
            if server_name and meta.get("server", "unknown") not in (server_name, "unknown"):
                continue
            meta["files"] = sorted(f.name for f in entry.iterdir()
                                   if f.is_file() and f.name != METADATA_FILENAME)
            # Keep colliding IDs separate: snapshots in different locations are distinct.
            meta["artifact_directory"] = str(entry.resolve())
            meta["legacy"] = base != artifact_base(server_root)
            results.append(meta)
    results.sort(key=lambda m: m.get("created_at") or "", reverse=True)
    return results
