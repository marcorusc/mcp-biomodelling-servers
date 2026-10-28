"""Shared MCP application instance."""

from mcp.server.mcpserver import MCPServer

from mcp_biomodelling_servers import __version__
from mcp_biomodelling_servers.handoff import enable_local_handoff_blobs

from .guidance import BIOMASS_SERVER_INSTRUCTIONS

mcp = MCPServer(
    "BioMASS",
    title="BioMASS ODE Model Builder",
    description="Construct, inspect, visualize, and simulate evidence-backed ODE models.",
    instructions=BIOMASS_SERVER_INSTRUCTIONS,
    version=__version__,
)

# Sandboxed hosts give each server a private filesystem: resolve handoff
# artifacts it cannot see from verified handoff_bundle copies.
enable_local_handoff_blobs()
