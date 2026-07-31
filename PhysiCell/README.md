# PhysiCell MCP Server

The PhysiCell server creates, loads, inspects, modifies, validates, and exports
PhysiCell configuration files through the Model Context Protocol (MCP). It uses
the [`physicell-settings`](https://pypi.org/project/physicell-settings/)
package to build `PhysiCell_settings.xml`, Cell Rules CSV files, and
PhysiBoSS intracellular-model settings.

This server is a configuration builder, not a PhysiCell simulation runtime. It
does not launch a PhysiCell executable or analyze tissue-simulation output.
Exported files must be placed in an appropriate PhysiCell project and executed
with that project's compiled application.

The server is distributed in the
[`mcp-biomodelling-servers`](https://pypi.org/project/mcp-biomodelling-servers/)
Python package and communicates over stdio.

## Installation and startup

Install all three bio-modelling servers:

```bash
python -m pip install mcp-biomodelling-servers
```

Start the installed PhysiCell server:

```bash
mcp-physicell-server
```

Alternatively, run it in an isolated environment:

```bash
uvx --from mcp-biomodelling-servers mcp-physicell-server
```

See the [repository README](../README.md) for supported Python versions,
complete MCP client configuration, and source-checkout instructions.

## Sessions and configuration state

Use one PhysiCell session per independent model. `create_session()` selects the
new session as the default unless `set_as_default=false` is supplied:

```json
{
  "set_as_default": true,
  "session_name": "hypoxic_tumor_v1"
}
```

Pass the full returned `session_id` explicitly whenever multiple configurations
are active. `set_default_session()` also accepts an unambiguous eight-character
session prefix.

- `list_sessions` reports active sessions, configuration presence, component
  counts, loaded-XML provenance, PhysiBoSS counts, and workflow progress.
- `list_artifact_sessions` discovers files retained from current or previous
  server processes.
- `get_workflow_status` and `get_simulation_summary` return the same workflow
  state and recommended next steps.
- `delete_session` permanently removes the in-memory session state.

Workflow progress records which configuration-building steps have been
performed. It does not prove that parameter values are biologically justified,
that all required custom code exists, or that the exported model will execute
successfully in a particular PhysiCell project.

## Build a new configuration

Use the following order:

1. Call `create_session()`.
2. Store the modelling objective with `analyze_biological_scenario()`.
3. Define space, mesh, dimensionality, and duration with
   `create_simulation_domain()`.
4. Call `add_single_substrate()` once for every diffusible substrate.
5. Call `add_single_cell_type()` once for every cell population.
6. Configure cell parameters and every required cell/substrate interaction.
7. Discover exact signal and behavior names.
8. Add as many signal-behavior rules as the model requires.
9. Optionally attach and configure intracellular MaBoSS models through
   PhysiBoSS.
10. Inspect the summary and read-only resources.
11. When cell rules exist, export their CSV first and the XML second.

`analyze_biological_scenario()` stores the supplied text as session context. It
does not infer cell types, substrates, parameter values, or rules
automatically.

### Recreating the domain resets the model

`create_simulation_domain()` creates a fresh `PhysiCellConfig` and replaces the
configuration currently stored in the session. Calling it after substrates,
cell types, rules, or PhysiBoSS models have been added discards those
components. Use the parameter patch tools—not a second domain call—to revise an
existing configuration.

## Define the simulation domain

For a planar model:

```json
{
  "domain_x": 2000,
  "domain_y": 2000,
  "use_2d": true,
  "dx": 20,
  "max_time": 7200,
  "session_id": "SESSION_ID"
}
```

In 2D mode, the z extent is one mesh voxel (`dx`) and PhysiCell's planar mode
is enabled. For a full 3D model, set `use_2d=false` and provide a positive
`domain_z`.

The domain is centered on the origin. `max_time` is expressed in minutes and
`dx` in micrometers. Smaller mesh spacing increases spatial resolution and
typically increases the cost of the eventual PhysiCell simulation.

## Add substrates and cell types

### Substrates

Call `add_single_substrate()` separately for oxygen, nutrients, drugs, or
other diffusing fields:

```json
{
  "substrate_name": "oxygen",
  "diffusion_coefficient": 100000,
  "decay_rate": 0.01,
  "initial_condition": 38,
  "units": "mmHg",
  "dirichlet_enabled": true,
  "dirichlet_value": 38,
  "session_id": "SESSION_ID"
}
```

When Dirichlet boundaries are enabled, the tool applies the value to the x and
y boundaries and also to z boundaries in 3D. When `dirichlet_value` is
omitted, the initial condition is used.

### Cell types

Use `get_available_cycle_models()` to inspect valid cycle names, then call
`add_single_cell_type()` once per population:

```json
{
  "cell_type_name": "cancer_cell",
  "cycle_model": "Ki67_basic",
  "session_id": "SESSION_ID"
}
```

Substrates and cell types must exist before tools can configure a specific
cell/substrate pair or attach an intracellular model.

## Flat partial-update semantics

The public configuration tools currently use flat optional arguments. Supply
only the values that should change. The deferred nested-parameter-object
redesign is not part of this release.

Each patch is prepared on a separate configuration copy and published only
after it succeeds. A failed patch leaves the active configuration unchanged.

### Patch cell parameters

`configure_cell_parameters()` supports:

- `volume_total`;
- `volume_nuclear`;
- `fluid_fraction`;
- `motility_speed`;
- `persistence_time`;
- `motility_enabled`;
- `apoptosis_rate`;
- `necrosis_rate`.

At least one value is required. Every omitted value is preserved.

```json
{
  "cell_type": "cancer_cell",
  "motility_speed": 0.8,
  "motility_enabled": true,
  "session_id": "SESSION_ID"
}
```

The same cell type can be revised later without repeating its motility values:

```json
{
  "cell_type": "cancer_cell",
  "apoptosis_rate": 0.001,
  "session_id": "SESSION_ID"
}
```

That second call preserves the earlier motility speed and enabled state.
Repeat the tool independently for every cell type that needs configuration.

### Patch one cell/substrate interaction

`set_substrate_interaction()` targets one existing cell-type/substrate pair:

```json
{
  "cell_type": "cancer_cell",
  "substrate": "oxygen",
  "uptake_rate": 10,
  "session_id": "SESSION_ID"
}
```

At least one of `secretion_rate` or `uptake_rate` is required. An omitted rate
is preserved, as are the existing secretion target and net export rate. Repeat
the tool for every required pair or call it again later to revise one rate.

## Signals, behaviors, and Cell Rules

Use:

- `list_all_available_signals()` for environment, contact, intracellular, and
  other signals available in the current configuration;
- `list_all_available_behaviors()` for configurable cell behaviors;
- `get_available_cycle_models()` for cell-cycle identifiers.

The signal and behavior discovery tools return descriptions and requirements
as structured data. Use their exact names when adding a rule.

`add_single_cell_rule()` adds one Hill-function relationship:

```json
{
  "cell_type": "cancer_cell",
  "signal": "oxygen",
  "direction": "decreases",
  "behavior": "apoptosis",
  "saturation_value": 0,
  "half_max": 5,
  "hill_power": 4,
  "session_id": "SESSION_ID"
}
```

`direction` is `increases` or `decreases`. `half_max` and `hill_power` must be
positive. Call the tool repeatedly for every signal-behavior relationship.

## Modify an existing XML configuration

Use this workflow:

1. Call `validate_xml_file(filepath)`.
2. If the structured result reports `valid=true`, call
   `load_xml_configuration(filepath)`.
3. Inspect it with `analyze_loaded_configuration()`.
4. Use `list_loaded_components()` for substrate, cell-type, or PhysiBoSS
   details.
5. Apply targeted patches and additions.
6. Inspect `get_simulation_summary()`.
7. Export to a session-scoped filename.

`validate_xml_file()` is read-only. A well-formed validation request for an
invalid XML file returns a successful tool result with `valid=false` and an
error description. The validation operation succeeded; the file did not pass
validation.

`load_xml_configuration()` validates and parses a candidate configuration
before replacing the session state. A missing, invalid, or unloadable file
does not replace the current configuration.

`analyze_loaded_configuration()` returns source provenance, domain information,
component names, existing rules, PhysiBoSS models, and modification guidance.
`list_loaded_components(component_type=...)` accepts `substrates`,
`cell_types`, `physiboss`, or `all`.

## PhysiBoSS integration

PhysiBoSS support must be available in the installed `physicell-settings`
package. A target PhysiCell cell type must exist before an intracellular MaBoSS
model can be attached.

### Preferred verified handoff

Export a `maboss-to-physicell` handoff from the MaBoSS MCP server, then call:

```json
{
  "manifest_path": "/path/to/maboss_to_physicell.handoff.json",
  "artifact_prefix": "cancer_cell_maboss_v1",
  "replace_existing": false,
  "session_id": "SESSION_ID"
}
```

on `import_maboss_handoff`.

The tool:

- verifies the manifest, BND, CFG, optional result table, and artifact hashes;
- verifies the complete upstream NeKo lineage when present;
- checks the exact Boolean node and output-node contracts;
- checks that the declared target cell type exists;
- copies the verified artifacts into the PhysiCell session;
- attaches the candidate model only after the complete import succeeds; and
- stores a separate MaBoSS context for the target cell type.

Use a new `artifact_prefix` for every retained import. Existing import
artifacts are not overwritten.

Importing models for different target cell types preserves independent
contexts. If the target already has an intracellular model, import fails unless
`replace_existing=true` is supplied. Explicit replacement resets that target's
previous PhysiBoSS settings, input links, output links, and mutations.

### Configure settings and biological mappings

After import, patch the target's timing and inheritance:

```json
{
  "cell_type": "cancer_cell",
  "intracellular_dt": 6,
  "time_stochasticity": 0,
  "inheritance_global": true,
  "session_id": "SESSION_ID"
}
```

`configure_physiboss_settings()` also supports `scaling` and `start_time`. At
least one value is required and every omitted setting is preserved. The tool
can be called repeatedly for the same target or separately for several target
cell types.

Then call repeatedly as needed:

- `add_physiboss_input_link` to map a PhysiCell signal to a Boolean node;
- `add_physiboss_output_link` to map a Boolean node to a PhysiCell behavior;
- `apply_physiboss_mutation` to fix a Boolean node to `0` or `1`.

Input and output link actions are `activation` or `inhibition`. Discover exact
PhysiCell signal and behavior names with the discovery tools, and use exact
Boolean node names from the MaBoSS context.

MaBoSS output nodes, parameters, and simulation summaries are preserved as
context. They are not automatically translated into PhysiBoSS timing or
signal/node/behavior mappings; those require explicit biological decisions.

Inspect all stored contexts with `get_maboss_context()`, or supply `cell_type`
to select one target.

### Standalone BND and CFG fallback

When no typed handoff manifest exists:

1. call `set_maboss_context()` with the standalone file paths and known model
   metadata;
2. call `add_physiboss_model()` for the intended cell type;
3. configure settings, links, and mutations as above.

This route does not verify a cross-server provenance chain or copy an
integrity-protected artifact set. Prefer `import_maboss_handoff()` whenever a
MaBoSS manifest is available.

## Inspect with read-only resources

The server exposes one static manual and seven session resource templates:

| URI | Content |
|---|---|
| `docs://physicell/agent_manual` | Complete agent operations manual |
| `physicell://session/{session_id}/workflow` | Progress and recommended next actions |
| `physicell://session/{session_id}/domain` | Space, mesh, dimensionality, duration, and time steps |
| `physicell://session/{session_id}/substrates` | Substrate diffusion and initial-condition summary |
| `physicell://session/{session_id}/cell_types` | Principal phenotype values for each cell type |
| `physicell://session/{session_id}/cell_rules` | Signal-behavior rules and ruleset references |
| `physicell://session/{session_id}/physiboss` | Intracellular models, context, settings, links, and mutations |
| `physicell://session/{session_id}/files` | All generated session artifacts |

The workflow and files resources work for a valid session before a simulation
domain exists. Domain, substrates, cell types, cell rules, and PhysiBoSS
resources require a configuration. These resources are read-only and do not
create missing sessions or modify workflow progress.

`physicell_workflow_prompt` and `get_help()` expose the same operating manual.

## Export XML and Cell Rules

### Export order when rules exist

When at least one Cell Rule exists:

1. call `export_cell_rules_csv()` first;
2. that tool writes the CSV and registers its enabled ruleset reference in the
   in-memory configuration;
3. call `export_xml_configuration()` afterward so the generated XML contains
   that ruleset reference.

```json
{
  "filename": "tumor_rules.csv",
  "session_id": "SESSION_ID"
}
```

followed by:

```json
{
  "filename": "tumor_settings.xml",
  "session_id": "SESSION_ID"
}
```

Without cell rules, only the XML export is required.

XML and CSV filenames must be plain basenames without directory components and
must use the matching `.xml` or `.csv` suffix. Outputs are confined to:

```text
PhysiCell/artifacts/{session_id}/
```

Reusing an export filename updates that session artifact. Choose distinct
filenames when several configuration versions must be retained.

`list_generated_files()` reports XML and CSV artifacts for the active session;
pass `session_id="all"` to aggregate those file types across sessions.
`list_artifact_sessions()` and the files resource also expose other retained
handoff artifacts.

The XML can be supplied to the appropriate compiled PhysiCell project. The
server cannot verify project-specific C++ extensions or execute the exported
model.

## Tool reference

| Category | Tools |
|---|---|
| Sessions and workflow | `create_session`, `list_sessions`, `list_artifact_sessions`, `set_default_session`, `get_workflow_status`, `get_simulation_summary`, `delete_session` |
| MaBoSS context and handoff | `import_maboss_handoff`, `set_maboss_context`, `get_maboss_context` |
| Existing XML | `validate_xml_file`, `load_xml_configuration`, `analyze_loaded_configuration`, `list_loaded_components` |
| Domain and scenario | `analyze_biological_scenario`, `create_simulation_domain` |
| Substrates and cells | `add_single_substrate`, `add_single_cell_type`, `configure_cell_parameters`, `set_substrate_interaction` |
| Discovery and rules | `get_available_cycle_models`, `list_all_available_signals`, `list_all_available_behaviors`, `add_single_cell_rule` |
| PhysiBoSS | `add_physiboss_model`, `configure_physiboss_settings`, `add_physiboss_input_link`, `add_physiboss_output_link`, `apply_physiboss_mutation` |
| Export and artifacts | `export_xml_configuration`, `export_cell_rules_csv`, `list_generated_files`, `clean_generated_files` |
| Help | `get_help` |

## Structured output and validation results

Session discovery, workflow status, XML validation and analysis, component
inspection, signal/behavior discovery, context inspection, verified handoff
import, exports, and artifact tools return JSON-native structured results in
addition to concise model-readable text.

Invalid arguments, missing prerequisites, unsafe filenames, failed
configuration changes, and integrity failures are returned as MCP tool errors
with recovery guidance. A valid call to `validate_xml_file()` may instead
return `valid=false`, because file validity is the scientific result being
requested.

## Cleanup

`clean_generated_files(session_id=...)` removes all artifacts for the selected
session while preserving its in-memory configuration. Retain any XML, CSV,
BND, CFG, result, or handoff files required for reproducibility before calling
it.

If both configuration state and artifacts should be discarded, clean the
generated files before calling `delete_session()`.

## Further reading

- [physicell-settings on PyPI](https://pypi.org/project/physicell-settings/)
- [PhysiCell project](https://physicell.org/)
- [PhysiCell Studio documentation](https://physicell-studio.readthedocs.io/)
- [PhysiBoSS](https://github.com/PhysiBoSS/PhysiBoSS)
- [MCP Python SDK v2 structured output](https://py.sdk.modelcontextprotocol.io/servers/structured-output/)
- [MCP Python SDK v2 resources](https://py.sdk.modelcontextprotocol.io/servers/resources/)
