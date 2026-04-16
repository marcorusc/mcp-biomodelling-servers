"""Entry point for the MaBoSS MCP server.

When installed via pip/uvx, run as:
    uvx --from mcp-biomodelling-servers mcp-maboss-server
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    # Resolve the installed mcp_biomodelling_servers/ directory.
    pkg_dir = Path(__file__).resolve().parent

    # MaBoSS/ is installed at mcp_biomodelling_servers/MaBoSS/ (see pyproject.toml sources).
    # Add it to sys.path so that server.py's bare import of session_manager resolves.
    # server.py itself inserts parent.parent (= pkg_dir) for artifact_manager.
    maboss_dir = str(pkg_dir / "MaBoSS")
    if maboss_dir not in sys.path:
        sys.path.insert(0, maboss_dir)

    os.chdir(maboss_dir)

    from mcp_biomodelling_servers.MaBoSS.server import mcp  # noqa: PLC0415
    mcp.run()


if __name__ == "__main__":
    main()
