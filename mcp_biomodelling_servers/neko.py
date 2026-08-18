"""Entry point for the NeKo MCP server.

When installed via pip/uvx, run as:
    uvx --from mcp-biomodelling-servers mcp-neko-server
"""
from __future__ import annotations


def main() -> None:
    from mcp_biomodelling_servers.NeKo.server import mcp  # noqa: PLC0415

    mcp.run()


if __name__ == "__main__":
    main()
