"""Console entry point for the BioMASS MCP server."""

from __future__ import annotations

from collections.abc import Sequence

from mcp_biomodelling_servers.cli import print_info


def main(argv: Sequence[str] | None = None) -> None:
    if print_info(
        "mcp-biomass-server",
        "Construct, inspect and simulate ODE models with BioMASS.",
        argv,
    ):
        return
    from mcp_biomodelling_servers.BioMASS.server import mcp  # noqa: PLC0415

    mcp.run()


if __name__ == "__main__":
    main()
