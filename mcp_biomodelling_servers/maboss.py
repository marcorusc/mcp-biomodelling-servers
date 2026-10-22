"""Entry point for the MaBoSS MCP server.

When installed via pip/uvx, run as:
    uvx --from mcp-biomodelling-servers mcp-maboss-server
"""
from __future__ import annotations

from collections.abc import Sequence

from mcp_biomodelling_servers.cli import print_info


def main(argv: Sequence[str] | None = None) -> None:
    if print_info(
        "mcp-maboss-server",
        "Configure, simulate and analyse stochastic Boolean models with MaBoSS.",
        argv,
    ):
        return
    from mcp_biomodelling_servers.MaBoSS.server import mcp  # noqa: PLC0415

    mcp.run()


if __name__ == "__main__":
    main()
