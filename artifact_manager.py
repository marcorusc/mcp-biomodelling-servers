"""Compatibility import for the canonical shared artifact manager."""
from mcp_biomodelling_servers.artifact_manager import (  # noqa: F401
    METADATA_FILENAME, artifact_base, artifact_bases, clean_artifacts,
    get_artifact_dir, list_artifact_sessions, list_artifacts,
    read_session_meta, safe_artifact_path, write_session_meta,
)
