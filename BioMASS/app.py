"""Shared MCP application instance."""

from mcp.server.mcpserver import MCPServer

from mcp_biomodelling_servers import __version__

from .guidance import BIOMASS_SERVER_INSTRUCTIONS

mcp = MCPServer(
    "BioMASS",
    title="BioMASS ODE Model Builder",
    description="Construct, inspect, visualize, and simulate evidence-backed ODE models.",
    instructions=BIOMASS_SERVER_INSTRUCTIONS,
    version=__version__,
)
