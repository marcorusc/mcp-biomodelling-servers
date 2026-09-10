"""Static agent manual resource and workflow prompt."""

from ..app import mcp
from ..guidance import BIOMASS_AGENT_MANUAL


@mcp.resource(
    "docs://biomass/agent_manual",
    name="BioMASS Agent Operations Manual",
    mime_type="text/markdown",
)
def biomass_agent_manual_resource() -> str:
    return BIOMASS_AGENT_MANUAL


@mcp.prompt(name="biomass_workflow_prompt")
def biomass_workflow_prompt() -> str:
    """Guide evidence-backed ODE construction, visualization, and simulation."""
    return BIOMASS_AGENT_MANUAL
