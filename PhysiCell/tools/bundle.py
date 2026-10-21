"""Portable export registration using the existing session lease."""
from pathlib import Path
from typing import Annotated

from pydantic import Field

from ..app import tool
from ..contracts import NON_IDEMPOTENT, NonEmptyString
from ..locking import session_locked
from ..services.bundle import export_bundle
from ..services.configuration import ConfigurationBundleResult
from ..session_manager import get_current_session


@tool(title="Export portable PhysiCell configuration", annotations=NON_IDEMPOTENT, structured_output=True)
@session_locked
def export_configuration_bundle(session_id: Annotated[NonEmptyString | None, Field(description="Session configuration to package.")] = None,
                                working_directory: Annotated[NonEmptyString | None, Field(description="Original executable working directory, required to resolve relative source inputs.")] = None) -> ConfigurationBundleResult:
    """Save a unique ZIP with XML, supported inputs, checksums and working-directory instructions.

    Relative source inputs require the original executable working_directory.
    Does not include an executable or grant scientific approval.
    """
    session = get_current_session(session_id)
    if session is None or session.config is None:
        raise ValueError("Create or load a configuration first")
    return ConfigurationBundleResult(**export_bundle(session, Path(__file__).resolve().parents[1], working_directory))
