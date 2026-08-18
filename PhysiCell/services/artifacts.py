"""Validation and transactional publication helpers for PhysiCell artifacts."""

import os
import shutil
from pathlib import Path

from mcp_biomodelling_servers.handoff import (
    sha256_file,
    verify_handoff_artifact,
)


def validate_export_filename(filename: str, expected_suffix: str) -> str:
    """Require a plain export basename with the expected extension."""
    if not filename or not filename.strip():
        raise ValueError("Export filename cannot be empty.")
    if filename != filename.strip():
        raise ValueError("Export filename cannot contain surrounding whitespace.")
    if "\x00" in filename:
        raise ValueError("Export filename cannot contain null bytes.")
    if filename in {".", ".."}:
        raise ValueError(f"Invalid export filename: {filename!r}.")
    if Path(filename).is_absolute() or "/" in filename or "\\" in filename:
        raise ValueError(
            "Export filename must be a basename without directory components: "
            f"{filename!r}."
        )
    if Path(filename).suffix.lower() != expected_suffix.lower():
        raise ValueError(
            f"Export filename must use the {expected_suffix} extension: "
            f"{filename!r}."
        )
    return filename


def require_unused_handoff_paths(paths: list[Path]) -> None:
    """Reject an import prefix when any destination already exists."""
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing PhysiCell handoff artifacts: "
            + ", ".join(str(path) for path in existing)
            + ". Choose a different artifact_prefix."
        )


def copy_verified_handoff_artifact(artifact, destination: Path) -> None:
    """Copy one verified source artifact and retain its recorded bytes."""
    source = verify_handoff_artifact(artifact)
    try:
        shutil.copyfile(source, destination)
    except OSError as exc:
        raise RuntimeError(
            f"Could not copy handoff artifact {source}: {exc}"
        ) from exc
    if destination.stat().st_size != artifact.size_bytes:
        raise RuntimeError(f"Copied handoff artifact size changed for {source}.")
    if sha256_file(destination) != artifact.sha256:
        raise RuntimeError(
            f"Copied handoff artifact digest changed for {source}."
        )


def link_handoff_artifact_without_overwrite(
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
            "Refusing to overwrite a PhysiCell handoff artifact created "
            f"concurrently: {destination}"
        ) from exc


def rollback_handoff_artifacts(paths: list[Path]) -> None:
    """Best-effort cleanup for an incomplete multi-file import."""
    for path in reversed(paths):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
