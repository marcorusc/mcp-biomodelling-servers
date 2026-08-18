"""Prompt and static documentation resource registrations for MaBoSS."""

from ..app import mcp
from ..guidance import MABOSS_AGENT_MANUAL


@mcp.prompt(
    name="maboss_workflow_prompt",
    title="MaBoSS modelling workflow",
    description="System prompt and operating manual for the MaBoSS agent.",
)
def maboss_workflow_prompt() -> str:
    """Return the MaBoSS modelling workflow and operating rules."""
    return MABOSS_AGENT_MANUAL


@mcp.resource(
    uri="docs://maboss/agent_manual",
    name="MaBoSS Agent Operations Manual",
    title="MaBoSS agent operations manual",
    description=(
        "Single source of truth for MaBoSS workflows, resources, tool "
        "categories, and rules."
    ),
    mime_type="text/markdown",
)
def maboss_agent_manual_resource() -> str:
    """Return the MaBoSS agent operations manual."""
    return MABOSS_AGENT_MANUAL
