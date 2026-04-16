"""Entry point for the NeKo MCP server.

When installed via pip/uvx, run as:
    uvx --from mcp-biomodelling-servers mcp-neko-server
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    # Resolve the installed mcp_biomodelling_servers/ directory.
    pkg_dir = Path(__file__).resolve().parent

    # NeKo/ is installed at mcp_biomodelling_servers/NeKo/ (see pyproject.toml sources).
    # Add it to sys.path so that server.py's bare imports (utils, session_manager,
    # src.*) resolve correctly.
    neko_dir = str(pkg_dir / "NeKo")
    if neko_dir not in sys.path:
        sys.path.insert(0, neko_dir)

    # server.py also inserts Path(__file__).parent.parent (= pkg_dir) for
    # artifact_manager, which lives at mcp_biomodelling_servers/artifact_manager.py.
    # We change the working directory so any relative file ops in NeKo tools work.
    os.chdir(neko_dir)

    from mcp_biomodelling_servers.NeKo.server import mcp  # noqa: PLC0415
    mcp.run()


if __name__ == "__main__":
    main()
