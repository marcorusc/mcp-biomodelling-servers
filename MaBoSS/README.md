# MaBoSS MCP Server

The MaBoSS server exposes stochastic Boolean-model construction, configuration,
simulation, analysis, and export through the Model Context Protocol (MCP). It
uses [pyMaBoSS](https://github.com/colomoto/pyMaBoSS) as its simulation engine
and can participate in a verified modelling pipeline from NeKo to MaBoSS to
PhysiCell.

The server is distributed in the
[`mcp-biomodelling-servers`](https://pypi.org/project/mcp-biomodelling-servers/)
Python package and communicates over stdio.

## Installation and startup

Install all three bio-modelling servers:

```bash
python -m pip install mcp-biomodelling-servers
```

Start the installed MaBoSS server:

```bash
mcp-maboss-server
```

Alternatively, run it in an isolated environment:

```bash
uvx --from mcp-biomodelling-servers mcp-maboss-server
```

See the [repository README](../README.md) for the supported Python versions,
complete MCP client configuration, source-checkout instructions, and shared
runtime behavior.

## Sessions and model state

Every model belongs to a MaBoSS session. Start by calling
`create_session()`, retain the returned `session_id`, and pass it explicitly
when more than one model is active. When `session_id` is omitted, tools use the
active default session.

- `create_session` creates an isolated in-memory session and may attach a
  human-readable label.
- `list_sessions` reports active sessions and whether each has a model or
  simulation result.
- `list_artifact_sessions` discovers session artifacts left by earlier server
  processes.
- `set_default_session` selects the session used by calls that omit
  `session_id`.
- `delete_session` removes in-memory state and, by default, its artifact files.

Generated BND, CFG, CSV, PNG, and handoff files are stored under the server's
session-scoped `artifacts/<session_id>/` directory. Use
`list_generated_files()` to obtain their resolved paths instead of guessing
them.

## Recommended standalone workflow

For a standalone BNET model, use the following order:

1. Call `create_session()`.
2. Convert the BNET with `bnet_to_bnd_and_cfg(bnet_path)`.
3. Load the generated files with `build_simulation()`.
4. Call `get_maboss_nodes()` and use only the returned node names.
5. Inspect the current configuration with `update_maboss_parameters()`,
   `get_maboss_initial_state()`, and `get_maboss_logical_rules()`.
6. Select a small, biologically meaningful set of outputs with
   `set_maboss_output_nodes(...)`.
7. Optionally update parameters, initial-state probabilities, or logical
   rules.
8. Call `run_simulation()`.
9. Read numerical results with `get_simulation_result()`.
10. Render the trajectory with `visualize_network_trajectories()`.
11. Export updated BND/CFG files or a typed PhysiCell handoff when needed.

`bnet_to_bnd_and_cfg()` writes `output.bnd` and `output.cfg` into the session
artifact directory. With no paths supplied, `build_simulation()` automatically
loads those files.

> A Boolean network with N visible outputs can produce up to 2^N states.
> Always call `set_maboss_output_nodes()` before `run_simulation()`. Selecting
> only the outputs needed for the scientific question reduces both runtime and
> result size.

## Inspecting and configuring a model

### Simulation parameters

Call `update_maboss_parameters()` without a `parameters` value to inspect all
parameters supported by the loaded model. Supply only the values that should
change:

```json
{
  "parameters": {
    "sample_count": 10000,
    "max_time": 100,
    "time_tick": 0.5,
    "thread_count": 4
  },
  "session_id": "SESSION_ID"
}
```

Common parameters include:

| Parameter | Meaning |
|---|---|
| `sample_count` | Number of stochastic trajectories |
| `max_time` | Simulation time horizon |
| `time_tick` | Interval between reported time points |
| `discrete_time` | Discrete-time toggle, `0` or `1` |
| `thread_count` | Number of simulation threads |

The server also accepts backend parameters exposed by the loaded MaBoSS
version. Unknown parameter names produce a recoverable tool error.

### Output nodes

First call `get_maboss_nodes()`, then provide a non-empty subset:

```json
{
  "output_nodes": ["Apoptosis", "Proliferation"],
  "session_id": "SESSION_ID"
}
```

### Initial-state probabilities

Inspect the existing distribution with `get_maboss_initial_state()` before
changing it.

For one node, provide `[P(OFF), P(ON)]`:

```json
{
  "nodes": "DNA_damage",
  "probDict": [0.2, 0.8],
  "session_id": "SESSION_ID"
}
```

For multiple nodes, use JSON-native state/probability records. State-vector
order must match the order in `nodes`, each state must contain only `0` or `1`,
and the probabilities must sum to 1:

```json
{
  "nodes": ["DNA_damage", "Growth_signal"],
  "probDict": [
    {"state": [0, 0], "probability": 0.4},
    {"state": [1, 0], "probability": 0.6}
  ],
  "session_id": "SESSION_ID"
}
```

The listed states do not have to cover every possible combination, but states
must be unique and each vector must have the same length as `nodes`.

### Logical rules and mutations

- `get_maboss_logical_rules` returns the current Boolean rules.
- `change_maboss_rule` replaces one node's rule and restores the previous rule
  if model validation rejects the change.
- `get_maboss_mutations` reports mutations currently stored in the model.
- `simulate_mutation` runs a mutant copy without changing the session's base
  simulation.

For example:

```json
{
  "nodes": ["TP53", "AKT"],
  "state": ["ON", "OFF"],
  "session_id": "SESSION_ID"
}
```

Mutation states are `ON`, `OFF`, or `WT`. A single state may be applied to all
requested nodes, or a state list may be supplied in node order.

## Running and reading a simulation

`run_simulation()` executes the loaded model in a worker thread, stores the
pyMaBoSS result in the session, reports MCP progress, and writes a non-empty
state-probability table to `result.csv`.

`get_simulation_result()` returns both:

- a Markdown state-probability table for the model to read; and
- JSON-native structured content containing the numerical trajectory table.

The same Markdown result is available through
`maboss://session/{session_id}/result`.

## Native trajectory images

After `run_simulation()`, call:

```json
{
  "session_id": "SESSION_ID",
  "until": 50
}
```

on `visualize_network_trajectories`.

The optional `until` parameter is the maximum simulation time displayed. It
must be a finite number greater than zero. Omit it to plot the full available
trajectory.

The tool:

- delegates trajectory drawing to pyMaBoSS `plot_trajectory(until=...)`;
- uses a tight bounding box and padding so legends outside the axes are not
  cropped;
- returns the PNG as an MCP `ImageContent` block with MIME type `image/png`;
- returns text plus structured metadata for the session, time window, and
  artifact; and
- saves the exact same PNG bytes as `network_trajectory.png` in the session
  artifact directory.

An MCP client must support rendering image content blocks to display the plot
inline. The saved artifact remains available when a client presents only the
textual or structured part of the result.

## Cross-server handoffs

### NeKo to MaBoSS

Prefer `import_neko_handoff(manifest_path)` when the Boolean model comes from
the NeKo MCP server. The tool:

- validates the `neko-to-maboss` manifest and artifact hashes;
- converts the referenced BNET into MaBoSS BND and CFG files;
- verifies the converted node set;
- applies output nodes declared by NeKo when present;
- loads the model only after the complete import succeeds; and
- retains the NeKo provenance for a later PhysiCell handoff.

If the manifest declares no output nodes, call `set_maboss_output_nodes()`
before running or exporting the model.

### MaBoSS to PhysiCell

Call `export_maboss_handoff(target_cell_type=...)` to produce an
integrity-protected `maboss-to-physicell` handoff. It snapshots:

- the current BND and CFG model;
- MaBoSS parameters and selected output nodes;
- an optional non-empty simulation-result CSV;
- the intended PhysiCell cell type and biological context; and
- the complete upstream NeKo lineage, when one was imported.

A standalone MaBoSS model must provide `biological_context`. A model imported
from NeKo may inherit that context. The export refuses to overwrite existing
handoff artifacts, so choose a new `artifact_prefix` for every retained
handoff.

Use `export_maboss_bnd_cfg()` when only standalone MaBoSS files are needed.
Use `export_maboss_handoff()` when PhysiCell needs typed scientific context,
integrity metadata, and provenance.

## Tool reference

| Category | Tools |
|---|---|
| Sessions | `create_session`, `list_sessions`, `list_artifact_sessions`, `set_default_session`, `delete_session` |
| Import and loading | `import_neko_handoff`, `bnet_to_bnd_and_cfg`, `build_simulation` |
| Inspection | `get_maboss_nodes`, `get_maboss_initial_state`, `get_maboss_logical_rules`, `get_maboss_mutations` |
| Configuration | `change_maboss_rule`, `update_maboss_parameters`, `set_maboss_output_nodes`, `set_maboss_initial_state` |
| Simulation and analysis | `run_simulation`, `simulate_mutation`, `get_simulation_result`, `visualize_network_trajectories` |
| Export | `export_maboss_bnd_cfg`, `export_maboss_handoff` |
| Artifacts | `list_generated_files`, `clean_generated_files` |

## Resources and prompt

The server exposes read-only resources for clients that support MCP resource
templates:

| URI | Content |
|---|---|
| `docs://maboss/agent_manual` | Complete agent operating manual |
| `maboss://session/{session_id}/nodes` | Network node names |
| `maboss://session/{session_id}/parameters` | Current parameters |
| `maboss://session/{session_id}/initial_state` | Initial-state distribution |
| `maboss://session/{session_id}/logical_rules` | Boolean rules |
| `maboss://session/{session_id}/mutations` | Current mutations |
| `maboss://session/{session_id}/result` | Latest state-probability result |
| `maboss://session/{session_id}/files` | Generated artifact files |

`maboss_workflow_prompt` exposes the same recommended operating rules as an
MCP prompt.

## Errors and artifact cleanup

Invalid parameters, unknown nodes, missing model state, missing simulation
results, and failed scientific operations are returned as MCP tool errors with
actionable messages. A caller can correct the arguments or perform the named
prerequisite and retry without restarting the server.

`clean_generated_files(session_id=...)` removes artifacts for one session but
keeps its in-memory model. `delete_session(clean_files=true)` removes both.
These operations are destructive; retain any BND, CFG, CSV, PNG, or handoff
files needed for reproducibility before calling them.

## Further reading

- [MaBoSS website and documentation](https://maboss.curie.fr/)
- [pyMaBoSS source](https://github.com/colomoto/pyMaBoSS)
- [MCP Python SDK v2 media results](https://py.sdk.modelcontextprotocol.io/servers/media/)
- [MCP Python SDK v2 structured output](https://py.sdk.modelcontextprotocol.io/servers/structured-output/)
