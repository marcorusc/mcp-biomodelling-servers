# MCP Bio-Modelling Servers

<!-- mcp-name: io.github.marcorusc/NeKo -->
<!-- mcp-name: io.github.marcorusc/MaBoSS -->
<!-- mcp-name: io.github.marcorusc/PhysiCell -->

[![PyPI](https://img.shields.io/pypi/v/mcp-biomodelling-servers)](https://pypi.org/project/mcp-biomodelling-servers/)
[![MCP Registry](https://img.shields.io/badge/MCP_Registry-active-brightgreen)](https://registry.modelcontextprotocol.io)

This package provides three stateful
[Model Context Protocol](https://modelcontextprotocol.io/) servers for
mechanistic and systems-biology modelling:

| Server | Modelling role | Upstream project | MCP Registry name |
|---|---|---|---|
| MaBoSS | Configure, simulate, and analyze stochastic Boolean models | [pyMaBoSS](https://github.com/colomoto/pyMaBoSS) | `io.github.marcorusc/MaBoSS` |
| NeKo | Build and analyze signalling networks from interaction databases | [NeKo](https://github.com/sysbio-curie/Neko) | `io.github.marcorusc/NeKo` |
| PhysiCell | Build, inspect, and export PhysiCell and PhysiBoSS configuration files | [PhysiCell-settings](https://github.com/marcorusc/PhysiCell_Settings) | `io.github.marcorusc/PhysiCell` |

All three servers use MCP over stdio and are distributed together as
`mcp-biomodelling-servers`.

## Publication

For more details, please check the related article:

> **"Intelligent tool orchestration for rapid mechanistic model prototyping: MCP servers as AI-biology interfaces"**<br>
> Marco Ruscone, Miguel Vazquez & Alfonso Valencia, *npj Systems Biology and Applications* (2026)<br>
> [https://doi.org/10.1038/s41540-026-00767-3](https://doi.org/10.1038/s41540-026-00767-3)

## Requirements

- Python 3.10–3.14.
- MCP Python SDK 2.x, installed automatically with this package.
- The modelling-package dependencies declared in `pyproject.toml`, installed
  automatically by `pip` or `uvx`.
- The Graphviz system runtime for NeKo history diagrams. The Python `graphviz`
  package is not a replacement for the external `dot` renderer.

Check whether Graphviz is available with:

```bash
dot -V
```

If this command is missing, install Graphviz using your operating system or
environment package manager. See the
[Graphviz installation guide](https://graphviz.org/download/) for
platform-specific instructions.

## Installation

### Install with pip

```bash
python -m pip install mcp-biomodelling-servers
```

The installation provides three console entry points:

```bash
mcp-neko-server
mcp-maboss-server
mcp-physicell-server
```

### Run in an isolated environment with uvx

```bash
uvx --from mcp-biomodelling-servers mcp-neko-server
uvx --from mcp-biomodelling-servers mcp-maboss-server
uvx --from mcp-biomodelling-servers mcp-physicell-server
```

Conda is optional. It remains useful when you want one explicitly managed
environment for local development or additional native scientific software,
but it is not required for the packaged entry points.

## Configure an MCP client

The following example uses `uvx` and works with clients that accept the common
`mcp.json` stdio configuration:

```jsonc
{
  "servers": {
    "neko": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "mcp-biomodelling-servers",
        "mcp-neko-server"
      ]
    },
    "maboss": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "mcp-biomodelling-servers",
        "mcp-maboss-server"
      ]
    },
    "physicell": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "mcp-biomodelling-servers",
        "mcp-physicell-server"
      ]
    }
  }
}
```

If the package is already installed in the client environment, each entry can
instead use its console script directly:

```jsonc
{
  "servers": {
    "neko": {
      "type": "stdio",
      "command": "mcp-neko-server"
    },
    "maboss": {
      "type": "stdio",
      "command": "mcp-maboss-server"
    },
    "physicell": {
      "type": "stdio",
      "command": "mcp-physicell-server"
    }
  }
}
```

Refer to your MCP client's documentation for its configuration-file location
and reload procedure. For Visual Studio Code, see
[Use MCP servers in VS Code](https://code.visualstudio.com/docs/copilot/chat/mcp-servers).

## Sessions, artifacts, and errors

Each server can maintain multiple isolated modelling sessions. Tools that
create or load a model return a session identifier; pass that identifier to
subsequent operations when more than one session is active.

Generated models, configuration files, plots, and other outputs are kept in
session-scoped artifact directories. Artifact-listing tools return the paths
needed to inspect or hand files to another modelling server.

Under MCP SDK 2.x, failures to execute a tool are returned as tool errors so
the client and model can distinguish them from successful scientific results.
Validation tools may still return a successful result describing an invalid
model or configuration when validity itself is the requested result.

## Run from source

Clone the repository and install it with its development dependencies:

```bash
git clone https://github.com/marcorusc/mcp-biomodelling-servers.git
cd mcp-biomodelling-servers
python -m pip install ".[dev]"
```

You can then run the same console entry points or invoke a server module
directly with the selected Python interpreter:

```bash
python MaBoSS/server.py
python NeKo/server.py
python PhysiCell/server.py
```

## Repository layout

```text
MaBoSS/                     MaBoSS server, manual, and Registry manifest
NeKo/                       NeKo server, manual, and Registry manifest
PhysiCell/                  PhysiCell server, manual, and Registry manifest
mcp_biomodelling_servers/   Installed package namespace and entry points
tests/                      Protocol, runtime, concurrency, and package tests
```

The server-specific READMEs describe the modelling workflows and exposed tool
families in more detail.

## MCP SDK and protocol compatibility

The package uses the stable MCP Python SDK 2.x API. The SDK negotiates the
appropriate MCP protocol revision with the connected client; the protocol
revision is independent of the MCP Registry schema used by each `server.json`.

## License

The package metadata declares the project under the MIT license. The wrapped
modelling packages retain their own licenses; consult their upstream projects
for details.
