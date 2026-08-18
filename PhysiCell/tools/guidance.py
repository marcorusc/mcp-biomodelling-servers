"""Prompt and static documentation resource registrations for PhysiCell."""

from ..app import mcp
from ..physicell_guidance import PHYSICELL_AGENT_MANUAL


@mcp.prompt(
    name="physicell_workflow_prompt",
    title="Build or revise a PhysiCell configuration",
    description=(
        "Operating manual for creating, modifying, integrating, and "
        "exporting PhysiCell configurations."
    ),
)
def physicell_workflow_prompt() -> str:
    """Return the PhysiCell configuration-building workflow."""
    return PHYSICELL_AGENT_MANUAL


@mcp.resource(
    uri="docs://physicell/agent_manual",
    name="PhysiCell Agent Operations Manual",
    title="PhysiCell agent operations manual",
    description=(
        "Single source of truth for PhysiCell configuration workflows, "
        "repeatable operations, PhysiBoSS integration, and export."
    ),
    mime_type="text/markdown",
)
def physicell_agent_manual_resource() -> str:
    """Return the PhysiCell agent operations manual."""
    return PHYSICELL_AGENT_MANUAL
