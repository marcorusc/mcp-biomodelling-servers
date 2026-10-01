"""Shared MCP application instance for PhysiCell registration modules."""

from mcp.server.mcpserver import MCPServer

from mcp_biomodelling_servers import __version__
from mcp_biomodelling_servers.tool_registration import domain_tool

from .physicell_guidance import PHYSICELL_SERVER_INSTRUCTIONS

mcp = MCPServer(
    "PhysiCell",
    title="PhysiCell Configuration Builder",
    description=(
        "Create, inspect, and export PhysiCell simulation configuration files."
    ),
    instructions=PHYSICELL_SERVER_INSTRUCTIONS,
    version=__version__,
)


def tool(**metadata):
    """Register tools with actionable expected domain errors."""
    return domain_tool(mcp, **metadata)
