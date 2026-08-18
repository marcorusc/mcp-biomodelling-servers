"""Entry point for the MaBoSS MCP server.

When installed via pip/uvx, run as:
    uvx --from mcp-biomodelling-servers mcp-maboss-server
"""
from __future__ import annotations


def main() -> None:
    from mcp_biomodelling_servers.MaBoSS.server import mcp  # noqa: PLC0415

    mcp.run()


if __name__ == "__main__":
    main()
