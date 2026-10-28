"""Portable transfer of NeKo handoffs to servers without a shared filesystem."""

from typing import Annotated

from pydantic import Field

from mcp_biomodelling_servers.artifact_manager import artifact_bases
from mcp_biomodelling_servers.handoff_bundle import (
    NeKoHandoffBundle,
    build_handoff_bundle,
)

from ..app import tool
from ..contracts import READ_ONLY_CLOSED, NonEmptyString
from ..src.helpers import _SERVER_ROOT


@tool(title="Export NeKo handoff bundle", annotations=READ_ONLY_CLOSED, structured_output=True)
def export_handoff_bundle(
    manifest_path: Annotated[
        NonEmptyString,
        Field(description="Manifest path returned by export_neko_handoff or export_biomass_handoff."),
    ],
) -> NeKoHandoffBundle:
    """Pack a verified handoff and every file it references into one JSON value.

    Use when the consuming server cannot read NeKo's files (separate sandboxes):
    pass the returned object unchanged as handoff_bundle to the consumer's
    import_neko_handoff. Reads only files on disk, so it also works after the
    exporting session has ended. Integrity is checked again on import.
    """
    bundle = build_handoff_bundle(manifest_path, allowed_roots=artifact_bases(_SERVER_ROOT))
    if not isinstance(bundle, NeKoHandoffBundle):
        raise ValueError("This manifest was not produced by NeKo.")
    return bundle
