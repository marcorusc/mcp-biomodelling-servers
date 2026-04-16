"""Shared artifact-directory management for MCP Bio-Modelling servers.

Each server gets a per-session sandbox:
    <server_root>/artifacts/<session_id>/

All file I/O from MCP tools must go through these helpers so that:
  - Files are never written to an arbitrary CWD.
  - Different sessions never clobber each other's output.
  - Cross-server paths are stable and predictable.

Usage (inside any server.py):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from artifact_manager import get_artifact_dir, safe_artifact_path, list_artifacts, clean_artifacts

    _SERVER_ROOT = Path(__file__).parent

    # get (and create) the session artifact directory
    art_dir = get_artifact_dir(_SERVER_ROOT, session_id)

    # build a safe path for a file inside it
    out_path = safe_artifact_path(art_dir, "output.bnd")
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

METADATA_FILENAME = "session_meta.json"


def get_artifact_dir(server_root: Path, session_id: str) -> Path:
    """Return (and create) the per-session artifact directory for a server.

    Layout:  <server_root>/artifacts/<session_id>/

    Args:
        server_root: Absolute path to the server directory
                     (use ``Path(__file__).parent`` in server code).
        session_id:  Session identifier string (e.g. ``"session_1"``).

    Returns:
        Path to the session artifact directory (guaranteed to exist).
    """
    if not session_id or "/" in session_id or "\\" in session_id or session_id in (".", ".."):
        raise ValueError(f"Invalid session_id for artifact directory: {session_id!r}")
    artifact_dir = server_root / "artifacts" / session_id
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
    except ValueError:
        raise ValueError(
            f"Path traversal detected: {filename!r} resolves outside the artifact sandbox."
        )
    return target


def list_artifacts(server_root: Path, session_id: Optional[str] = None) -> List[Path]:
    """List artifact files for a session (or all sessions).

    Args:
        server_root: Server directory.
        session_id:  If given, list only that session's files.
                     If ``None``, list all files across every session.

    Returns:
        Sorted list of ``Path`` objects.
    """
    base = server_root / "artifacts"
    if not base.exists():
        return []
    if session_id is not None:
        session_dir = base / session_id
        if not session_dir.exists():
            return []
        return sorted(p for p in session_dir.iterdir() if p.is_file())
    # All sessions
    files: List[Path] = []
    for entry in base.iterdir():
        if entry.is_dir():
            files.extend(p for p in entry.iterdir() if p.is_file())
    return sorted(files)


def clean_artifacts(server_root: Path, session_id: str) -> int:
    """Remove all artifact files for *session_id*.

    Args:
        server_root: Server directory.
        session_id:  Target session.

    Returns:
        Number of files removed.
    """
    session_dir = server_root / "artifacts" / session_id
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
    label: Optional[str] = None,
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
    meta: Dict[str, Any] = {
        "session_id": session_id,
        "server": server_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": label or "",
    }
    meta_path = art_dir / METADATA_FILENAME
    meta_path.write_text(json.dumps(meta, indent=2))


def read_session_meta(server_root: Path, session_id: str) -> Optional[Dict[str, Any]]:
    """Read the metadata dict for *session_id*, or ``None`` if not found.

    Args:
        server_root: Server directory.
        session_id:  Session whose metadata to read.

    Returns:
        Parsed JSON dict, or ``None`` if the file does not exist / is invalid.
    """
    meta_path = server_root / "artifacts" / session_id / METADATA_FILENAME
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return None


def list_artifact_sessions(
    server_root: Path,
    server_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
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
    base = server_root / "artifacts"
    if not base.exists():
        return []

    results: List[Dict[str, Any]] = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        meta = read_session_meta(server_root, entry.name)
        if meta is None:
            meta = {
                "session_id": entry.name,
                "server": "unknown",
                "created_at": "",
                "label": "",
            }
        # Filter by server name when requested
        if server_name and meta.get("server", "unknown") not in (server_name, "unknown"):
            continue
        # Attach non-metadata file listing (skip the meta file itself)
        files = sorted(
            f.name for f in entry.iterdir()
            if f.is_file() and f.name != METADATA_FILENAME
        )
        meta["files"] = files
        results.append(meta)

    # Sort newest-first (empty created_at sorts to the end)
    results.sort(key=lambda m: m.get("created_at") or "", reverse=True)
    return results

