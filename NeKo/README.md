# NeKo MCP Server

The NeKo server builds, curates, analyzes, and exports signalling networks
through the Model Context Protocol (MCP). It uses
[NeKo](https://github.com/sysbio-curie/Neko) to construct networks from curated
interaction databases or existing SIF files, and it preserves topology changes
in NeKo's native branching history.

The server is distributed in the
[`mcp-biomodelling-servers`](https://pypi.org/project/mcp-biomodelling-servers/)
Python package and communicates over stdio. A typed, integrity-protected
handoff connects a finished NeKo network to the MaBoSS MCP server.

## Installation and startup

Install all three bio-modelling servers:

```bash
python -m pip install mcp-biomodelling-servers
```

Start the installed NeKo server:

```bash
mcp-neko-server
```

Alternatively, run it in an isolated environment:

```bash
uvx --from mcp-biomodelling-servers mcp-neko-server
```

The external Graphviz `dot` executable is required to render the branching
history resource. See the [repository README](../README.md) for Graphviz setup,
supported Python versions, complete MCP client configuration, and
source-checkout instructions.

## Sessions and network state

Every modelling hypothesis should have its own NeKo session. Begin with
`create_session()`, retain the returned `session_id`, and pass it explicitly to
subsequent calls. A newly created session is not automatically selected as the
default; call `set_default_session()` if later calls should be allowed to omit
the ID.

```json
{
  "label": "TP53-MYC resistance network"
}
```

- `list_sessions` reports active in-memory sessions, network dimensions, the
  current default, and history-retention settings.
- `list_artifact_sessions` discovers exported files from current or previous
  server processes.
- `status` returns the node and interaction counts for one session.
- `reset_network` discards the current network while keeping the session and
  its settings.
- `delete_session` permanently removes the in-memory session.

`set_default_params()` stores connection defaults per session. They are reused
by `add_nodes(autoconnect=true)`.

## Recommended workflow

Use this order for a database-backed network:

1. Call `create_session()` and retain its ID.
2. Optionally configure reusable connection settings with
   `set_default_params()`.
3. Build the network with `create_network(...)`.
4. Curate ambiguous edges with `remove_bimodal_interactions()` and, when
   appropriate, `remove_undefined_interactions()`.
5. Check connectivity with `analyze_connectivity()` (isolated nodes and the
   full component partition).
6. If repair is needed, call `preview_connection_impact()` before selecting a
   mutating connection strategy.
8. Inspect the resulting topology and evidence.
9. Use the history tools to compare the new state with earlier alternatives.
10. Export an integrity-protected MaBoSS handoff with
    `export_neko_handoff(...)`.

During iterative construction, prefer `verbosity="summary"`. Use `preview` or
`full` only when the actual nodes, interactions, paths, or evidence are needed.

## Building a network

`create_network()` supports two curated database backends:

- `omnipath`, the default;
- `signor`.

Create a network from seed genes:

```json
{
  "list_of_initial_genes": ["EGFR", "KRAS", "TP53", "AKT1"],
  "database": "omnipath",
  "max_len": 2,
  "algorithm": "bfs",
  "only_signed": true,
  "connect_with_bias": false,
  "consensus": true,
  "session_id": "SESSION_ID",
  "verbosity": "summary"
}
```

The construction parameters are:

| Parameter | Contract |
|---|---|
| `max_len` | Maximum path length, from 1 to 4 |
| `algorithm` | `bfs` or `dfs` |
| `only_signed` | Keep only signed interactions when true |
| `connect_with_bias` | Avoid some already-connected node pairs |
| `consensus` | Require support from multiple curated sources when true |
| `verbosity` | `summary`, `preview`, or `full` |

Construction may query an external biological database and can take several
minutes. The tool reports MCP progress while database resources and the
network are being prepared.

An existing SIF can be used instead:

```json
{
  "list_of_initial_genes": [],
  "sif_file": "/absolute/path/input.sif",
  "database": "omnipath",
  "session_id": "SESSION_ID",
  "verbosity": "summary"
}
```

Additional seed genes may be supplied together with `sif_file`. A call that
cannot complete successfully does not replace a valid network already stored
in the session.

## Editing and curating the topology

### Add or remove nodes and interactions

- `add_nodes` adds one or more genes in a single call and, by default,
  autoconnects each new gene to any direct neighbour already in the network
  (cheap; does not search multi-step paths). Set `autoconnect=false` to skip.
- `remove_gene` removes a node and all incident edges.
- `remove_interaction` removes only the requested directed edge.

Inspect the current network before destructive edits:

```json
{
  "session_id": "SESSION_ID",
  "verbosity": "preview",
  "max_rows": 50
}
```

on `list_genes_and_interactions`.

Gene symbols should match the identifiers returned by the inspection tools.
`remove_interaction(node_A, node_B)` removes `node_A → node_B`; it does not
remove the reverse edge.

### Curate interaction signs

- `remove_bimodal_interactions` removes edges reported as both activating and
  inhibiting.
- `remove_undefined_interactions` removes edges whose effect cannot be mapped
  to a Boolean activation or inhibition.

These operations simplify later BNET export but change the scientific
topology. Inspect the affected network and its history after applying them.

## Inspecting topology and evidence

The server offers both concise text and JSON-native structured results:

- `list_genes_and_interactions` returns node and directed-interaction records.
- `filter_interactions` filters by effect, source, or target without modifying
  the network.
- `find_paths` reports directed paths between two genes up to a bounded length.
- `get_references` returns literature evidence for interactions involving one
  or two genes.
- `status` reports the current dimensions.

For example, inspect inhibitory edges targeting apoptosis:

```json
{
  "effect": ["inhibition"],
  "target": "Apoptosis",
  "session_id": "SESSION_ID",
  "verbosity": "preview",
  "format": "markdown",
  "max_rows": 50
}
```

`filter_interactions(format="json")` changes the model-readable text format;
the tool's structured result remains available independently.

## Auditing and repairing connectivity

`analyze_connectivity()` reports isolated nodes—nodes with no incident
edges—together with the complete connected-component partition of the
network. Use it before a BNET or handoff export.

### Scout candidate changes

`preview_connection_impact()` evaluates possible repair directions without
changing the active network:

| Method | Result |
|---|---|
| `hubs` | Ranks high-degree genes that may help orient a repair |
| `relax_max_len` | Simulates increasing the stored maximum path length |
| `unsigned` | Simulates allowing unsigned interactions |

Example:

```json
{
  "method": "relax_max_len",
  "session_id": "SESSION_ID",
  "verbosity": "preview",
  "format": "markdown"
}
```

The simulation methods operate on a copy. Their predicted changes are not
committed to the session.

### Apply a selected strategy

`bridge_components()` connects two explicit groups:

```json
{
  "comp_a": ["EGFR", "KRAS"],
  "comp_b": ["TP53", "BAX"],
  "max_len": 2,
  "mode": "OUT",
  "only_signed": true,
  "consensus": true,
  "session_id": "SESSION_ID"
}
```

`comp_a`/`comp_b` must be Gene Symbols (e.g. from a component's `nodes` list
in `analyze_connectivity()`'s output), never the numeric `component_id` that
`analyze_connectivity()` reports for each component - that ID is a report
label, not a valid gene name.

`mode` is `OUT`, `IN`, or `ALL`.

`connect_targeted_nodes()` supports:

- `connect_to_upstream_nodes`;
- `connect_subgroup`.

`apply_global_connection()` supports:

- `complete_connection`, with `algorithm` and `minimal`;
- `connect_network_radially`, with `direction="OUT"` or `"IN"`;
- `connect_as_atopo`, with `strategy_mode="radial"` or `"complete"` and an
  optional `outputs` list of gene symbols to anchor the topology. This
  strategy loops until the network is connected to every declared output, so
  its cost is open-ended on large networks.

The database-backed connection tools can be expensive — `complete_connection`
is O(N^2) over every node pair, and `connect_as_atopo` cost is open-ended.
Reinspect components and compare history states after every selected
strategy.

## Branching network history

NeKo automatically records snapshots for its supported topology-changing
operations. The MCP server exposes that native history directly, so undo,
redo, checkout, and comparison preserve the exact NeKo network states.

### List states

`list_network_history()` returns:

- the exact current state ID;
- retained state IDs;
- parent and child relationships;
- node and edge counts for every state;
- the configured retention limit.

State IDs are identifiers, not list positions. Always obtain them from
`list_network_history()` before navigation or comparison.

### Navigate

Use the single `navigate_network_history()` tool:

```json
{
  "action": "checkout",
  "state_id": 3,
  "session_id": "SESSION_ID"
}
```

The supported actions are:

| Action | `state_id` behavior |
|---|---|
| `undo` | Moves to the parent; `state_id` must be omitted |
| `redo` | Moves to a child; the ID may be omitted when there is one unambiguous child |
| `checkout` | Moves to the exact saved state; `state_id` is required |

Navigation does not delete other saved states. If an older state is checked
out and then modified, NeKo retains the previous future and records the new
topology on a separate branch.

### Compare without navigating

`compare_network_states(state_a, state_b)` reports added and removed nodes and
interactions without changing the currently checked-out state:

```json
{
  "state_a": 2,
  "state_b": 5,
  "session_id": "SESSION_ID"
}
```

### Configure retention

`set_network_history_limit(max_states)` sets the per-session policy:

- `null` keeps an unbounded history;
- an integer must be at least 2;
- a policy set before construction applies when a network is created;
- lowering the limit may immediately and irreversibly prune older snapshots.

Call `list_network_history()` before reducing the limit.

### Render the history

Read:

```text
neko://session/{session_id}/history
```

to obtain read-only HTML containing NeKo's inline SVG history diagram. This
resource does not change the checked-out state. Rendering requires the
Graphviz `dot` executable; if it is missing, the resource returns an actionable
rendering error.

## Export and the MaBoSS handoff

### Standalone exports

`export_network(format="sif")` writes a standalone tab-separated interaction
network. `export_network(format="bnet")` writes a Boolean model for MaBoSS.

BNET export requires one connected network. It sanitizes names that are invalid
in Boolean rules and reports renamed nodes and duplicate rules removed after
identifier normalization. Use `list_bnet_files()` to inspect BNET artifacts in
the session.

### Verified NeKo-to-MaBoSS transfer

Prefer `export_neko_handoff()` when the next step is simulation in the MaBoSS
MCP server:

```json
{
  "biological_context": "EGFR-driven survival and apoptosis decision",
  "output_nodes": ["Apoptosis", "Proliferation"],
  "artifact_prefix": "egfr_decision_v1",
  "session_id": "SESSION_ID"
}
```

The handoff:

- requires a connected network and non-empty `biological_context`;
- writes a sanitized BNET and versioned JSON manifest;
- records exact Boolean nodes, declared outputs, node renames, removed
  duplicates, package versions, session provenance, and the current NeKo
  history state;
- protects both files with artifact integrity metadata;
- translates requested original NeKo output names to their exact sanitized
  BNET names; and
- refuses to overwrite an existing artifact pair.

Choose a new `artifact_prefix` for every retained handoff. `output_nodes` may
be omitted, but MaBoSS must then select a small output set before running a
simulation.

Pass the returned manifest path to MaBoSS `import_neko_handoff()`. A standalone
BNET path can instead be passed to MaBoSS `bnet_to_bnd_and_cfg()`, but that
route does not preserve the typed scientific context and provenance manifest.

## Tool reference

| Category | Tools |
|---|---|
| Sessions | `create_session`, `list_sessions`, `list_artifact_sessions`, `set_default_session`, `status`, `reset_network`, `delete_session` |
| Construction and editing | `create_network`, `set_default_params`, `add_nodes`, `remove_gene`, `remove_interaction` |
| Curation | `remove_bimodal_interactions`, `remove_undefined_interactions` |
| Inspection and evidence | `list_genes_and_interactions`, `filter_interactions`, `find_paths`, `get_references`, `analyze_connectivity` |
| Connection strategies | `preview_connection_impact`, `bridge_components`, `connect_targeted_nodes`, `apply_global_connection` |
| Branching history | `list_network_history`, `navigate_network_history`, `compare_network_states`, `set_network_history_limit` |
| Export and artifacts | `export_network`, `export_neko_handoff`, `list_bnet_files`, `clean_generated_files` |

## Resources and prompt

| Name or URI | Content |
|---|---|
| `docs://neko/agent_manual` | Recommended agent workflow and operating rules |
| `neko://session/{session_id}/history` | Branching history as inline SVG HTML |
| `neko_workflow_prompt` | The operating manual as an MCP prompt |

## Structured output and errors

Inspection, session, history, export, and handoff tools return structured
JSON-native results in addition to concise model-readable text. This allows an
MCP host to consume exact nodes, interactions, state IDs, dimensions, paths,
provenance, and artifact metadata without parsing Markdown.

Invalid identifiers, missing prerequisites, unsupported strategies, failed
database operations, disconnected exports, and other modelling failures are
returned as MCP tool errors with recovery guidance. Correct the arguments or
perform the named prerequisite and retry in the same session.

## Artifact cleanup

`clean_generated_files(session_id=...)` deletes exported artifacts for the
selected session without resetting its in-memory network. It is destructive.
Retain any SIF, BNET, or handoff files needed for reproducibility before
calling it.

If both network state and artifacts should be discarded, clean the generated
files before calling `delete_session()`.

## Further reading

- [NeKo documentation](https://sysbio-curie.github.io/Neko/)
- [Exploring Network History in NeKo](https://sysbio-curie.github.io/Neko/tutorials/11_network_history/)
- [NeKo source](https://github.com/sysbio-curie/Neko)
- [MCP Python SDK v2 structured output](https://py.sdk.modelcontextprotocol.io/servers/structured-output/)
- [MCP Python SDK v2 resources](https://py.sdk.modelcontextprotocol.io/servers/resources/)
