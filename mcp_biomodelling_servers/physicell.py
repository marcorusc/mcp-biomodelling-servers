"""Entry point for the PhysiCell MCP server.

When installed via pip/uvx, run as:
    uvx --from mcp-biomodelling-servers mcp-physicell-server
"""
from __future__ import annotations


def main() -> None:
    from mcp_biomodelling_servers.PhysiCell.server import mcp  # noqa: PLC0415

    mcp.run()


if __name__ == "__main__":
    main()
