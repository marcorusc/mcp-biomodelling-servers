"""Entry point for the PhysiCell MCP server.

When installed via pip/uvx, run as:
    uvx --from mcp-biomodelling-servers mcp-physicell-server
"""
from __future__ import annotations

from collections.abc import Sequence

from mcp_biomodelling_servers.cli import print_info


def main(argv: Sequence[str] | None = None) -> None:
    if print_info(
        "mcp-physicell-server",
        "Build, edit and export PhysiCell and PhysiBoSS configuration files.",
        argv,
    ):
        return
    from mcp_biomodelling_servers.PhysiCell.server import mcp  # noqa: PLC0415

    mcp.run()


if __name__ == "__main__":
    main()
