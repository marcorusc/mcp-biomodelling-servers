"""Caller-supplied model text for servers that cannot read the caller's files."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from .artifact_manager import safe_artifact_path
from .import_lineage import MAX_ANCESTRY_BYTES

MAX_INLINE_SOURCE_CHARS = MAX_ANCESTRY_BYTES


def resolve_source_path(
    path: str | None,
    text: str | None,
    *,
    artifact_dir: Path,
    suffix: str,
    path_name: str,
    text_name: str,
) -> str | None:
    """Return a readable source path from at most one of a path or inline text.

    Inline text is stored verbatim in artifact_dir under a content-addressed
    name, so repeated imports of identical text reuse one file.
    """
    if path is not None and text is not None:
        raise ValueError(f"Provide either {path_name} or {text_name}, not both.")
    if text is None:
        return path
    data = text.encode("utf-8")
    if len(data) > MAX_INLINE_SOURCE_CHARS:
        raise ValueError(f"{text_name} exceeds {MAX_INLINE_SOURCE_CHARS} bytes.")
    name = f"inline-{hashlib.sha256(data).hexdigest()[:16]}{suffix}"
    destination = safe_artifact_path(artifact_dir, name)
    if destination.is_symlink():
        raise ValueError("Inline source destinations cannot be symlinks.")
    if destination.is_file() and destination.read_bytes() == data:
        return str(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".inline-", suffix=".tmp", dir=artifact_dir
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return str(destination)
