"""Transactional publication helpers for MaBoSS artifact sets."""

import logging
import os
from pathlib import Path

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
