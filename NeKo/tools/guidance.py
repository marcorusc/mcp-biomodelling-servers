"""Static NeKo prompt and documentation-resource registrations."""

from ..app import mcp
from ..guidance import NEKO_AGENT_MANUAL


@mcp.prompt(
    name="neko_workflow_prompt",
    title="NeKo modelling workflow",
    description="System prompt and operating manual for the NeKo agent.",
)
def neko_workflow_prompt() -> str:
    """Guide a complete NeKo-to-MaBoSS modelling workflow."""
    return NEKO_AGENT_MANUAL


@mcp.resource(
    uri="docs://neko/agent_manual",
    name="NeKo Agent Operations Manual",
    title="NeKo agent operations manual",
    description=(
        "The single source of truth for NeKo workflows, tool categories, "
        "and rules."
    ),
    mime_type="text/markdown",
)
def neko_agent_manual_resource() -> str:
    """Return the authoritative NeKo workflow manual as Markdown."""
    return NEKO_AGENT_MANUAL
