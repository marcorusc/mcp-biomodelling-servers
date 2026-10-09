# MCP Bio-Modelling Servers

<!-- mcp-name: io.github.marcorusc/NeKo -->
<!-- mcp-name: io.github.marcorusc/MaBoSS -->
<!-- mcp-name: io.github.marcorusc/PhysiCell -->
<!-- mcp-name: io.github.marcorusc/BioMASS -->

[![PyPI](https://img.shields.io/pypi/v/mcp-biomodelling-servers?cacheSeconds=300)](https://pypi.org/project/mcp-biomodelling-servers/)
[![MCP Registry](https://img.shields.io/badge/MCP_Registry-active-brightgreen)](https://registry.modelcontextprotocol.io)

This package provides four stateful
[Model Context Protocol](https://modelcontextprotocol.io/) servers for
mechanistic and systems-biology modelling:

| Server | Modelling role | Upstream project | MCP Registry name |
|---|---|---|---|
| MaBoSS | Configure, simulate, and analyze stochastic Boolean models | [pyMaBoSS](https://github.com/colomoto/pyMaBoSS) | `io.github.marcorusc/MaBoSS` |
| NeKo | Build and analyze signalling networks from interaction databases | [NeKo](https://github.com/sysbio-curie/Neko) | `io.github.marcorusc/NeKo` |
| BioMASS | Construct, visualize, and simulate evidence-backed ODE models | [BioMASS](https://github.com/biomass-dev/biomass) | `io.github.marcorusc/BioMASS` |
| PhysiCell | Build, inspect, and export PhysiCell and PhysiBoSS configuration files | [PhysiCell-settings](https://github.com/marcorusc/PhysiCell_Settings) | `io.github.marcorusc/PhysiCell` |

All four servers use MCP over stdio and are distributed together as
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

NeKo (`nekomata`) 1.10.1 or newer (below 2.0) is required to preserve SIF
evidence references. This minimum is enforced by the source dependency metadata
and CI. Until a package release includes this metadata change, install the
already published pair explicitly:

```bash
python -m pip install "mcp-biomodelling-servers==2.3.0" "nekomata==1.10.1"
```

The installation provides four console entry points:

```bash
mcp-neko-server
mcp-maboss-server
mcp-physicell-server
mcp-biomass-server
```

### Run in an isolated environment with uvx

```bash
uvx --from mcp-biomodelling-servers mcp-neko-server
uvx --from mcp-biomodelling-servers mcp-maboss-server
uvx --from mcp-biomodelling-servers mcp-physicell-server
uvx --from mcp-biomodelling-servers mcp-biomass-server
```

Conda is optional. It remains useful when you want one explicitly managed
environment for local development or additional native scientific software,
but it is not required for the packaged entry points.

### Install as an agent plugin

Claude Code, Codex, and Agent Plugins 1.0 clients such as Visual Studio Code can
install each server as a plugin, with one install per server. A plugin starts its
server with `uvx` (so `uv` must be installed) and pins it to the matching
published `mcp-biomodelling-servers` release, so no client configuration file
needs to be written by hand. Install only the servers you need.

| Plugin | Server | Install string |
|---|---|---|
| `neko` | NeKo | `neko@marcorusc` |
| `maboss` | MaBoSS | `maboss@marcorusc` |
| `physicell` | PhysiCell | `physicell@marcorusc` |
| `biomass` | BioMASS | `biomass@marcorusc` |

In Claude Code, add the marketplace once, then install each plugin you want:

```bash
claude plugin marketplace add marcorusc/mcp-biomodelling-servers
claude plugin install neko@marcorusc
```

Add `--scope project` to the marketplace command to declare it in the
repository's own `.claude/settings.json` instead of your user settings.

In Codex:

```bash
codex plugin marketplace add marcorusc/mcp-biomodelling-servers
codex plugin add neko@marcorusc
```

Agent Plugins 1.0 clients (Cursor, GitHub Copilot, VS Code, Kiro) define the
plugin package, `plugin.json` and `mcp.json`, and leave installation to the
client. Each plugin here is the `plugins/<name>` directory of this repository.
In VS Code, open the Command Palette, run **Chat: Install Plugin from Source**,
choose a git repository, and enter `marcorusc/mcp-biomodelling-servers`.

The BioMASS plugin runs the server without the `biomass-graph` extra. For graph
rendering, use the manual configuration below.

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
    },
    "biomass": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "mcp-biomodelling-servers[biomass-graph]", "mcp-biomass-server"]
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
    },
    "biomass": {
      "type": "stdio",
      "command": "mcp-biomass-server"
    }
  }
}
```

Refer to your MCP client's documentation for its configuration-file location
and reload procedure. For Visual Studio Code, see
[Use MCP servers in VS Code](https://code.visualstudio.com/docs/copilot/chat/mcp-servers).

## ODE models with BioMASS

NeKo's `export_biomass_handoff` preserves the curated network's references and
available mechanism metadata for BioMASS. The calling agent reads the literature,
records evidence and assumptions, and authors Text2Model reactions. BioMASS
supports standalone text too, along with graph rendering and bounded exploratory
simulation. Calibration and sensitivity analysis are deferred.

For visualization, install `mcp-biomodelling-servers[biomass-graph]` and the Graphviz
system runtime. See the [BioMASS manual](BioMASS/README.md) for an MCP client
configuration, the 22 tools, graph interpretation limits, and a runnable example.

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
python -m BioMASS.server
```

## Repository layout

```text
MaBoSS/                     MaBoSS server, manual, and Registry manifest
NeKo/                       NeKo server, manual, and Registry manifest
PhysiCell/                  PhysiCell server, manual, and Registry manifest
BioMASS/                    ODE construction, visualization, and simulation server
mcp_biomodelling_servers/   Installed package namespace and entry points
tests/                      Protocol, runtime, concurrency, and package tests
```

The server-specific READMEs describe the modelling workflows and exposed tool
families in more detail.

## MCP SDK and protocol compatibility

The package uses the stable MCP Python SDK 2.x API. The SDK negotiates the
appropriate MCP protocol revision with the connected client; the protocol
revision is independent of the MCP Registry schema used by each `server.json`.

## Releasing

The `release.yml` workflow runs the full CI and compatibility suites, builds and
checks the wheel and source archive, publishes to PyPI, then publishes all four
server manifests to the official MCP Registry. Both publishing steps use GitHub
OIDC; no API token is needed.

Before the first automated release, configure a GitHub trusted publisher on
[the PyPI project's Publishing settings](https://pypi.org/manage/project/mcp-biomodelling-servers/settings/publishing/):

- Owner: `marcorusc`
- Repository: `mcp-biomodelling-servers`
- Workflow filename: `release.yml`
- Environment: `pypi`

Create the matching `pypi` environment in the repository's GitHub settings.
See the [PyPI trusted publishing guide](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
and [MCP Registry GitHub Actions guide](https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/github-actions.mdx).

For each release, synchronize the version in `pyproject.toml`, the source
fallback in `mcp_biomodelling_servers/__init__.py`, and all four `server.json`
files, including their pinned `--from` arguments. Build with `python -m build`,
check with `python -m twine check --strict dist/*`, and run
`python scripts/check_release.py --tag v<version>` using Python 3.12+.
Start with a clean `dist/` directory. Commit and push the reviewed changes before
creating and pushing the matching `v<version>` tag to trigger publication.

If PyPI succeeds but MCP registration fails, manually dispatch the release
workflow **on the same release tag**, with `publish_pypi` disabled. This retries
registration without trying to upload the existing PyPI version again.

## License

The package metadata declares the project under the MIT license. The wrapped
modelling packages retain their own licenses; consult their upstream projects
for details.

### Artifact storage and ordinary import provenance

All four servers write to
`$MCP_BIOMODELLING_ARTIFACT_ROOT/<server>/<session_id>/` when the environment
variable is set to an absolute directory. Otherwise they use
`${XDG_CACHE_HOME:-~/.cache}/mcp-biomodelling/artifacts/<server>/<session_id>/`.
Server namespaces are `NeKo`, `MaBoSS`, `PhysiCell`, and `BioMASS`. Set the variable
before starting a server and keep it stable for that process. The default never
writes into the package installation. The root-level `artifact_manager` import
remains available as a compatibility shim for the packaged shared helper.

Listings also discover existing `<server package>/artifacts/<session_id>/`
directories. Each session listing includes `artifact_directory` and `legacy`;
identical session IDs at different locations remain separate entries. No legacy
files are moved or deleted, and cleanup applies only to the current write root.
BioMASS restore reads snapshots from the current root; legacy snapshots remain
available for inspection but are not automatically restored or migrated.
Discovery scans the current installation's legacy directory, not other checkouts
or previously configured cache roots.

`NeKo.create_network(..., sif_file=..., parent_manifest=None, derived_from=None)`
and `MaBoSS.bnet_to_bnd_and_cfg(..., parent_manifest=None, derived_from=None)`
accept optional provenance arguments. Existing calls remain valid and now record
the imported source's SHA-256 digest. Successful imports retain an active source
record, a session import history, and a unique `source-import.<id>.json` record
on disk before reporting success. Import records preserve historical digests even
if a later export replaces a mutable sidecar. Replacing the model clears its active source
record; history retains the original import evidence. MaBoSS conversion provenance
becomes active only when its generated BND/CFG model is loaded.

* `parent_manifest` is a path to a validated source-lineage sidecar or, for BNET,
  a NeKo-to-MaBoSS handoff manifest. The referenced artifact's size and SHA-256
  must still match. Arbitrary parent labels are rejected.
* With only `parent_manifest`, the input must be byte-identical to a matching
  parent artifact. For edited copies, also supply `derived_from` as the exact
  path of the original artifact recorded by that parent. Keep the original
  bytes intact and import an edited copy at a separate path. Equal bytes remain
  classified as `byte-identical` even if `derived_from` is supplied.
* Ordinary NeKo SIF/BNET exports and MaBoSS BND/CFG exports write sibling
  `<filename>.lineage.json` files. NeKo BNET handoff exports also carry this
  sidecar; MaBoSS handoff exports with an active ordinary import write a sidecar
  alongside the handoff JSON. The independent
  `mcp-biomodelling-source-lineage` v1 schema records source/parent digests,
  `standalone`, `byte-identical`, or `edited-derivative` relationships, and UTC
  ISO8601 timestamps. Historical digests are retained even if old inputs later
  change; using an export as a new parent verifies its current exported bytes.

Source lineage is provenance, not approval or evidence of biological equivalence.
The required handoff v1 fields, accepted lineage rules, and approval checks
are unchanged; the shared artifact vocabulary adds `neko_sif` for source sidecars,
while existing Boolean handoff slots still require BNET. Source sidecars do not grant or replace acceptance. Mutable
`Network.*`/`output.*` files remain mutable, and retained handoff snapshots keep
existing overwrite protection. This extension does not validate SIF ancestry
through Boolean or ODE handoff contracts, does not add BNET import to NeKo, and
does not propagate ordinary source sidecars through BioMASS or PhysiCell imports.
Existing aware timestamps and numeric session clocks retain their formats.

Boolean NeKo/MaBoSS handoff exports now use v2 portable source ancestry while
retaining v1 reading and exact stage-parent validation. See
[Boolean lineage v2](docs/boolean-lineage-v2.md) for derivative semantics,
approval boundaries, offline verification and safe capture helpers.
