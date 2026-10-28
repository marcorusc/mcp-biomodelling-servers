"""Portable transfer of MaBoSS handoffs to servers without a shared filesystem."""

from pathlib import Path
from typing import Annotated

from pydantic import Field

from mcp_biomodelling_servers.artifact_manager import artifact_bases
from mcp_biomodelling_servers.handoff_bundle import (
    MaBoSSHandoffBundle,
    build_handoff_bundle,
)

from ..app import tool
from ..contracts import READ_ONLY, NonEmptyString

_SERVER_ROOT = Path(__file__).parents[1]


@tool(title="Export MaBoSS handoff bundle", annotations=READ_ONLY, structured_output=True)
def export_handoff_bundle(
    manifest_path: Annotated[
        NonEmptyString,
        Field(description="Manifest path returned by export_maboss_handoff."),
    ],
) -> MaBoSSHandoffBundle:
    """Pack a verified handoff, its NeKo parent and all referenced files into one JSON value.

    Use when PhysiCell cannot read MaBoSS's files (separate sandboxes): pass the
    returned object unchanged as handoff_bundle to import_maboss_handoff. Reads
    only files on disk. Integrity is checked again on import.
    """
    bundle = build_handoff_bundle(manifest_path, allowed_roots=artifact_bases(_SERVER_ROOT))
    if not isinstance(bundle, MaBoSSHandoffBundle):
        raise ValueError("This manifest was not produced by MaBoSS.")
    return bundle
