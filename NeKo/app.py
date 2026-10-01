"""NeKo MCP application object shared by tool-registration modules."""

from mcp.server.mcpserver import MCPServer

from mcp_biomodelling_servers import __version__
from mcp_biomodelling_servers.tool_registration import domain_tool

from .guidance import NEKO_SERVER_INSTRUCTIONS

mcp = MCPServer(
    "NeKo",
    title="NeKo Signalling Network Builder",
    description=(
        "Build and analyze signalling networks from biological interaction "
        "databases."
    ),
    instructions=NEKO_SERVER_INSTRUCTIONS,
    version=__version__,
)


def tool(**metadata):
    """Register tools with actionable expected domain errors."""
    return domain_tool(mcp, **metadata)
