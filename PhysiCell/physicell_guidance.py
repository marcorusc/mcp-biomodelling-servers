"""Editable workflow guidance shared by the PhysiCell MCP surfaces."""

PHYSICELL_AGENT_MANUAL = """# PhysiCell Agent Operations Manual

## 1. Choose the workflow

Use one session per independent model and pass its full `session_id` whenever
multiple configurations are active.

### Build a new configuration

1. `create_session()` to establish isolated state.
2. `analyze_biological_scenario()` to record the modelling objective.
3. `create_simulation_domain()` to define space, mesh, dimensionality, and time.
4. Call `add_single_substrate()` once for each diffusible substrate.
5. Call `add_single_cell_type()` once for each cell population.
6. Inspect exact names with `list_all_available_signals()` and
   `list_all_available_behaviors()`.
7. Configure each cell type and its substrate interactions.
8. Add as many signal-behavior rules as the model requires.
9. Optionally integrate intracellular MaBoSS models through PhysiBoSS.
10. Check `get_simulation_summary()`, then export the XML and rules CSV.

### Modify an existing configuration

1. Check the file with `validate_xml_file()`.
2. Load it with `load_xml_configuration()`.
3. Call `analyze_loaded_configuration()` and `list_loaded_components()`.
4. Apply targeted changes using the same configuration tools.
5. Recheck `get_simulation_summary()` and export to a new session artifact.

## 2. Repeatable configuration operations

- `add_single_substrate()` is called once per substrate.
- `add_single_cell_type()` is called once per cell type.
- `configure_cell_parameters()` is called separately for each cell type.
- `set_substrate_interaction()` is called for each cell-type/substrate pair.
- `add_single_cell_rule()` is called once per signal-behavior relationship.
- PhysiBoSS input links, output links, and mutations can each be called
  repeatedly for the same intracellular model.

These configuration tools use patch semantics:

- `configure_cell_parameters()` preserves every omitted volume, motility, and
  death parameter. Motility is enabled or disabled only when
  `motility_enabled` is supplied explicitly.
- `set_substrate_interaction()` preserves omitted secretion and uptake rates,
  as well as the existing secretion target and net export rate.
- `configure_physiboss_settings()` preserves every omitted intracellular
  timing, stochasticity, scaling, start-time, and inheritance setting.

Provide only the values that should change. Each patch requires at least one
value, and the same tool can be called again after discussion or simulation
analysis to revise one cell type or cell-type/substrate pair.

## 3. PhysiBoSS integration

Preferred typed workflow:

1. Export a `maboss-to-physicell` handoff from MaBoSS for the intended cell
   type.
2. Import it with `import_maboss_handoff()`. The tool verifies and copies the
   complete NeKo/MaBoSS lineage, then atomically attaches the model.
3. Set timing and inheritance with `configure_physiboss_settings()`.
4. Connect PhysiCell signals to Boolean nodes with
   `add_physiboss_input_link()`.
5. Connect Boolean nodes to cell behaviors with
   `add_physiboss_output_link()`.
6. Apply optional fixed-node perturbations with `apply_physiboss_mutation()`.

For standalone files without a handoff manifest, use the lower-level
`set_maboss_context()` and `add_physiboss_model()` tools.

Importing several manifests for different target cell types preserves a
separate MaBoSS context for each target. Replacing an existing target model
requires `replace_existing=true` and resets that target's previous PhysiBoSS
settings, mappings, and mutations.

Use node names returned by the MaBoSS server and signal/behavior names returned
by the PhysiCell discovery tools. MaBoSS simulation parameters and output nodes
are retained as context, but are not automatically translated into PhysiBoSS
timing or biological mappings.

## 4. Read-only session resources

Applications can load concise snapshots without calling an inspection tool:

- `physicell://session/{session_id}/workflow` reports progress and next steps.
- `physicell://session/{session_id}/domain` reports space and time settings.
- `physicell://session/{session_id}/substrates` lists diffusion components.
- `physicell://session/{session_id}/cell_types` summarizes phenotypes.
- `physicell://session/{session_id}/cell_rules` lists behavior rules.
- `physicell://session/{session_id}/physiboss` summarizes intracellular models.
- `physicell://session/{session_id}/files` lists generated artifacts.

These resources are read-only snapshots. Use tools to change a configuration.
The workflow and files resources work before a simulation domain exists; the
other five require a configured simulation.

## 5. Inspection and export

- `get_workflow_status()` and `get_simulation_summary()` expose the same
  complete workflow state.
- `get_maboss_context()` lists every stored target-cell context or selects one
  with `cell_type`.
- `list_generated_files()` lists session-scoped XML and CSV artifacts.
- `export_xml_configuration()` writes the PhysiCell settings XML.
- `export_cell_rules_csv()` writes the CBHG cell-rules file when rules exist.

All generated files are confined to `PhysiCell/artifacts/<session_id>/`.
Use `list_artifact_sessions()` to rediscover files after a server restart.
"""
