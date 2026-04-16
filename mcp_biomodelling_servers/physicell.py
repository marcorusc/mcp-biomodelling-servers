"""Entry point for the PhysiCell MCP server.

When installed via pip/uvx, run as:
    uvx --from mcp-biomodelling-servers mcp-physicell-server
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    # Resolve the installed mcp_biomodelling_servers/ directory.
    pkg_dir = Path(__file__).resolve().parent

    # PhysiCell/ is installed at mcp_biomodelling_servers/PhysiCell/.
    # PhysiCell's server.py already inserts current_dir and current_dir.parent into
    # sys.path, so session_manager and artifact_manager are found automatically.
    # We still set cwd so any relative file ops in PhysiCell tools work.
    physicell_dir = str(pkg_dir / "PhysiCell")
    if physicell_dir not in sys.path:
        sys.path.insert(0, physicell_dir)

    os.chdir(physicell_dir)

    from mcp_biomodelling_servers.PhysiCell.server import mcp  # noqa: PLC0415
    mcp.run()


if __name__ == "__main__":
    main()
