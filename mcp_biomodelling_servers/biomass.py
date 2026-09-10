"""Console entry point for the BioMASS MCP server."""


def main() -> None:
    from mcp_biomodelling_servers.BioMASS.server import mcp

    mcp.run()


if __name__ == "__main__":
    main()
