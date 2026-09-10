# BioMASS ODE Model Builder

A stateful MCP server for evidence-backed ODE construction using
[BioMASS 0.14](https://biomass-core.readthedocs.io/en/latest/) and Text2Model.
It supports model inspection, graph visualization, export, and bounded
exploratory simulation. The calling agent reads literature and proposes
mechanisms. Calibration and sensitivity analysis are outside v1.

## Installation

```bash
python -m pip install mcp-biomodelling-servers
mcp-biomass-server
```

For graph rendering:

```bash
python -m pip install 'mcp-biomodelling-servers[biomass-graph]'
dot -V
```

Install the Graphviz system runtime too. Building PyGraphviz from source may
require a C compiler, Python development headers, and Graphviz development
headers (for example, `graphviz libgraphviz-dev build-essential` on Debian/Ubuntu).
If graph dependencies are unavailable, construction and simulation still work.

Example MCP client configuration with visualization enabled:

```json
{
  "servers": {
    "biomass": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "--from", "mcp-biomodelling-servers[biomass-graph]",
        "mcp-biomass-server"
      ]
    }
  }
}
```

From a source checkout, install with `python -m pip install '.[dev,biomass-graph]'`
and run `python -m BioMASS.server`. The repository's wheel path remapping does
not support editable installs.

## NeKo workflow

1. Curate a network in NeKo, then call its `export_biomass_handoff` with the
   biological context. The export preserves nodes, stable edge IDs, references,
   and available mechanism/context columns. It does not require connectivity.
2. `create_session`, then `import_neko_handoff` with the returned manifest path.
   Import verifies artifact integrity and stores a durable copy of the network
   and provenance inside the authoring snapshot.
3. Read papers with the calling agent's literature tools. `set_evidence` stores
   source identifiers, supporting passages or summaries, locations, biological
   context, limitations, and supporting/contradicting/context-only stances.
4. `build_reactions` adds or edits reactions; `set_reactions` replaces the complete list. Each record has
   a stable ID, Text2Model statement, originating edges, evidence IDs, and a
   supported/assumed/unreviewed status. Scientific annotations are optional and
   supplied by the agent; new records default to unreviewed. These links
   may be many-to-many; uncovered edges remain in the coverage report.
5. `configure_model` sets observables, time span, conditions, numerical defaults,
   units, and quantity provenance. It replaces the complete configuration.
6. Inspect, validate, generate, visualize, optionally simulate, and export.

Reaction example (IDs must correspond to imported edges and stored evidence):

```json
{
  "reaction_id": "binding",
  "statement": "E + S <--> ES | kf=0.003, kr=0.001 | E=100, S=50, ES=0",
  "status": "supported",
  "evidence_ids": ["binding_paper"],
  "edge_ids": ["edge_returned_by_import"]
}
```

Use `share_parameters_with` to refer to an earlier reaction ID. The renderer
resolves that ID to the correct Text2Model line number. Numeric sharing in
reaction records is rejected. Complete replacement with `set_reactions` clears numerical overrides and
conditions because generated parameter names depend on line numbers; reapply
those settings after inspecting the revised model.

For conversational construction, call `create_session`, then `build_reactions`
with templates or raw statements. `inspect_reactions` returns stable IDs and
generated symbols. Preview a batch using `expected_version`, then apply with
`preview=false`. Incremental edits retain compatible numerical settings. Read
[model editing](docs/model_editing.md) or `docs://biomass/model_editing` for details.
The agent chooses kinetics and any scientific annotations; the server validates
syntax, references, dependencies, and execution.

## Standalone text workflow

`import_text` preserves a complete Text2Model document verbatim, including
comments, directives, and numeric line references. Replacing a document also
replaces its line-evidence mapping and clears the old configuration. Evidence
can be stored before import, then linked by line number. In document mode,
put observables, simulation conditions, and time span in the text itself;
`configure_model` can override numeric defaults and record units/provenance.

`import_text_file` reads an existing UTF-8 file into the session with its source
path and hash. `build_reactions` can expand or edit it while preserving untouched
lines and parameter references. The original file is never modified. Removals
leave comment lines so later line numbers remain stable; dependent observables
and conditions must be repaired explicitly in the same batch.

[examples/enzyme.txt](examples/enzyme.txt) is a small executable enzyme model.
It specifies all three parameters, all four initial values (including zeros),
and the time span, so it runs without opting into placeholder values.

```json
{"scenario": {"name": "enzyme"}, "session_id": "returned-session-id"}
```

Pass that object to `run_simulation` after `generate_model`. For explicitly
hypothetical runs, set `scenario.allow_placeholders` to `true`. A supplied or
assumed value is distinguished from a Text2Model placeholder. Literature-derived
quantities require evidence IDs. Unspecified units remain explicit null values.

Scenario overrides replace generated defaults before condition assignments.
Each condition starts from fresh defaults. The actual parameters and initial
values used for each solve, including steady-state preparation, are recorded.
Parameter-sharing constraints remain effective; override their source parameter.
Simulation produces complete species and observable CSV tables, a trajectory
preview (up to 20 species), and a numerical scenario report.

## Graph visualization

- `visualize_model(format="png")`: default static graph and MCP image content.
- `visualize_model(format="svg")`: scalable static graph.
- `visualize_model(format="html")`: interactive graph; no browser is opened.
- `export_model_graph`: DOT for external graph software.

Static layouts are `dot` (default), `neato`, `fdp`, `circo`, and `twopi`.
HTML optionally exposes physics/layout controls. Layout selection applies to
static formats. Interactive HTML references the vis-network JavaScript and stylesheet CDN used by
BioMASS’s supported PyVis version; opening it requires network access. Graph artifacts are linked to an immutable model
revision and are included in that revision's exported bundle.

BioMASS projects reactants and modifiers onto products, combining repeated
species connections. It does not distinguish reactants from modifiers, or
activating from inhibiting modifiers. Therefore this visualization is not a
complete reaction graph or a signed causal network. Unresolved NeKo edges
remain in coverage reports. An interactive graph is not a simulation animation.
See the [upstream graph tutorial](https://biomass-core.readthedocs.io/en/latest/tutorial/nfkb.html).

## Tools and resources

The server exposes 22 tools:

| Family | Tools |
|---|---|
| Sessions | `create_session`, `list_sessions`, `close_session`, `restore_session` |
| Authoring | `import_neko_handoff`, `import_text`, `import_text_file`, `set_evidence`, `set_reactions`, `build_reactions`, `configure_model` |
| Inspection | `inspect_model`, `inspect_reactions`, `validate_model` |
| Generation | `generate_model` |
| Visualization | `visualize_model`, `export_model_graph` |
| Simulation | `run_simulation` |
| Artifacts | `export_model_bundle`, `list_generated_files`, `list_artifact_sessions`, `clean_generated_files` |

Read `docs://biomass/agent_manual` or request `biomass_workflow_prompt` for agent
instructions. Session resources expose `/model`, `/evidence`, `/coverage`,
`/files`, and `/revision/{revision}` under `biomass://session/{session_id}`.

Before writing reactions, agents should read these offline MCP resources:

| Resource | Reference |
|---|---|
| `docs://biomass/reaction_syntax` | [Supported syntax and generated kinetics](docs/reaction_syntax.md) |
| `docs://biomass/authoring_examples` | [Tested tool argument examples](docs/authoring_examples.md) |
| `docs://biomass/network_to_reactions` | [Evidence-to-mechanism authoring guide](docs/network_to_reactions.md) |
| `docs://biomass/model_editing` | [Conversational construction, templates, and file editing](docs/model_editing.md) |

These references ship with the server and are linked from its initialization
instructions, agent manual, workflow prompt, and authoring tool descriptions.
They document the supported BioMASS 0.14 subset. A separate skill is not required.
The syntax examples are converted in tests; the complete example workflows are
also generated and simulated. Signed edges do not automatically determine
mechanisms, kinetic laws, or one reaction per edge.

## Revisions, validation, and limits

Each session has locked authoring state and a durable JSON snapshot. Workers
use separate directories and fresh interpreters. Generated revisions have
integrity inventories; authoring edits clear the current revision pointer but
retain previous revisions. Explicitly select an old revision to inspect its
original model and evidence. Failed jobs retain only diagnostic logs.

Syntax validity, generation success, evidence coverage, and numerical execution
are separate results. None establishes biological validity. Validation with
`check_generation=true` uses a disposable worker and publishes no revision.

Accepted expressions contain finite constants, arithmetic, and model-symbol
references (`p[name]`, `u[name]`, `init[name]` where applicable). Python calls,
attributes, imports, and arbitrary Python model packages are unsupported.
Identifiers start with a letter and use letters, digits, and single underscores.
Numerical reaction values must be finite and nonnegative.

Workers default to 60 seconds, configurable from 1 to 300 seconds. v1 limits:
1 MiB / 2000 text lines, 500 species / 1000 reactions, 20 conditions, and 10001
integer time samples. Timeouts terminate the worker and its child processes on
POSIX systems. These processes isolate runtime state; they are not an OS sandbox.

Exports include generated Python, original text, numerical configuration,
evidence, assumptions, coverage, version metadata, graphs, and saved runs.
Extract the ZIP and run `python run_simulation.py` to list saved scenarios, then
`python run_simulation.py simulate_<run_id>` to reproduce one. The script uses
the recorded actual numerical conditions. `requirements.txt` pins BioMASS 0.14.0.
Closing a session preserves artifacts; cleanup preserves authoring and lineage
while removing generated revisions and runs.
