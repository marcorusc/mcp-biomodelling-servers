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

When revising an existing configuration, inspect the current values first.
`configure_cell_parameters()` and `set_substrate_interaction()` currently use
defaults for omitted arguments, so explicitly provide every value that must be
preserved during an update.

## 3. PhysiBoSS integration

1. Store upstream Boolean-model context with `set_maboss_context()`.
2. Attach the BND and CFG files with `add_physiboss_model()`.
3. Set timing and inheritance with `configure_physiboss_settings()`.
4. Connect PhysiCell signals to Boolean nodes with
   `add_physiboss_input_link()`.
5. Connect Boolean nodes to cell behaviors with
   `add_physiboss_output_link()`.
6. Apply optional fixed-node perturbations with `apply_physiboss_mutation()`.

Use node names returned by the MaBoSS server and signal/behavior names returned
by the PhysiCell discovery tools.

## 4. Inspection and export

- `get_workflow_status()` and `get_simulation_summary()` expose the same
  complete workflow state.
- `get_maboss_context()` checks the stored cross-server context.
- `list_generated_files()` lists session-scoped XML and CSV artifacts.
- `export_xml_configuration()` writes the PhysiCell settings XML.
- `export_cell_rules_csv()` writes the CBHG cell-rules file when rules exist.

All generated files are confined to `PhysiCell/artifacts/<session_id>/`.
Use `list_artifact_sessions()` to rediscover files after a server restart.
"""
