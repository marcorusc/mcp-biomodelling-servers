"""Shared MCP application instance for MaBoSS registration modules."""

from mcp.server.mcpserver import MCPServer

from mcp_biomodelling_servers import __version__

from .guidance import MABOSS_SERVER_INSTRUCTIONS

mcp = MCPServer(
    "MaBoSS",
    title="MaBoSS Boolean Model Simulator",
    description=(
        "Configure, simulate, analyze, and visualize Boolean models with MaBoSS."
    ),
    instructions=MABOSS_SERVER_INSTRUCTIONS,
    version=__version__,
)
