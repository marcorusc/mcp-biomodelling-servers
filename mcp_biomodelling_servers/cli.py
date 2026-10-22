"""Informational flags shared by the four server entry points.

``--help`` and ``--version`` print and exit before any server module is
imported. Every other invocation starts the stdio server exactly as before,
ignoring its arguments.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence

from mcp_biomodelling_servers import __version__

_HELP = """\
usage: {command} [-h | --help] [--version]

{summary}

Started without these flags, {command} serves the Model Context Protocol
over stdio: it reads JSON-RPC messages from standard input and waits for an
MCP client. Configure it in an MCP client rather than running it in a terminal.

options:
  -h, --help  show this help message and exit
  --version   show the mcp-biomodelling-servers version and exit

environment:
  MCP_BIOMODELLING_ARTIFACT_ROOT
              absolute directory for session artifacts (default:
              ${{XDG_CACHE_HOME:-~/.cache}}/mcp-biomodelling/artifacts)
{environment}
documentation: https://github.com/marcorusc/mcp-biomodelling-servers
"""


def print_info(
    command: str,
    summary: str,
    argv: Sequence[str] | None = None,
    environment: str = "",
) -> bool:
    """Print help or the version if requested; return whether one was printed."""
    arguments = sys.argv[1:] if argv is None else list(argv)
    if "-h" in arguments or "--help" in arguments:
        print(_HELP.format(command=command, summary=summary, environment=environment), end="")
        return True
    if "--version" in arguments:
        print(f"{command} (mcp-biomodelling-servers) {__version__}")
        return True
    return False
