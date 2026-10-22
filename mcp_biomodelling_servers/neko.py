"""Entry point for the NeKo MCP server.

When installed via pip/uvx, run as:
    uvx --from mcp-biomodelling-servers mcp-neko-server
"""
from __future__ import annotations

import os
import time
from collections.abc import Sequence

from mcp_biomodelling_servers.artifact_manager import artifact_root
from mcp_biomodelling_servers.cli import print_info

_ENVIRONMENT = """\
  PYPATH_LOG  log file for the pypath libraries NeKo uses (default: a new
              file under <artifact root>/logs/pypath/ for each process)
"""


def default_pypath_log() -> None:
    """Keep pypath's session log out of the working directory.

    nekomata imports ``pypath_common``, which otherwise creates ``pypath_log/``
    wherever the server starts. It reads ``PYPATH_LOG`` once, when first
    imported, so set it before the server can import nekomata. An explicit
    value is kept.
    """
    if os.environ.get("PYPATH_LOG"):
        return
    try:
        logs = artifact_root() / "logs" / "pypath"
    except ValueError:
        return  # Artifact tools report the invalid root; logging must not block startup.
    stamp = time.strftime("%Y%m%dT%H%M%S")
    os.environ["PYPATH_LOG"] = str(logs / f"pypath-{stamp}-{os.getpid()}.log")


def main(argv: Sequence[str] | None = None) -> None:
    if print_info(
        "mcp-neko-server",
        "Build, curate and analyse signalling networks with NeKo.",
        argv,
        _ENVIRONMENT,
    ):
        return
    default_pypath_log()
    from mcp_biomodelling_servers.NeKo.server import mcp  # noqa: PLC0415

    mcp.run()


if __name__ == "__main__":
    main()
