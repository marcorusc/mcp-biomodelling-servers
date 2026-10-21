"""Model-facing operating guidance for the MaBoSS MCP server."""

MABOSS_SERVER_INSTRUCTIONS = (
    "Create a session before loading or simulating a Boolean model, and pass "
    "`session_id` explicitly when working with multiple models. Use "
    "`import_neko_handoff` for a typed NeKo transfer, then inspect node names "
    "and restrict output nodes to the smallest biologically meaningful set "
    "before `run_simulation()` to control the exponential state space. Use "
    "`export_maboss_handoff` for a provenance-preserving PhysiCell transfer. "
    "Read `docs://maboss/agent_manual` or use `maboss_workflow_prompt` for the "
    "complete workflow."
)

MABOSS_AGENT_MANUAL = """
# MaBoSS Agent Operations Manual

## 1. Recommended Workflow (in order)
1. **Session:** `create_session()` — returns a session_id
2. **Load a model:** Prefer `import_neko_handoff(manifest_path)` for a typed
   NeKo transfer. For a standalone BNET, call
   `bnet_to_bnd_and_cfg(bnet_path)` followed by `build_simulation()`.
3. **Inspect nodes (MANDATORY):** `get_maboss_nodes()` — list ALL valid node names; always do this before any configuration step to avoid referencing non-existent nodes
4. **Inspect parameters:** `update_maboss_parameters()` (no args) — review current defaults
5. **Tune:** `update_maboss_parameters({"sample_count": 1000, "thread_count": 4})`
6. **Reduce output nodes (IMPORTANT):** `set_maboss_output_nodes(["Apoptosis", "Proliferation"])` — restricts the result to only the nodes you care about. Without this, MaBoSS enumerates ALL 2^N Boolean states, which becomes exponentially expensive for large networks (>20 nodes). Always set output nodes to the smallest biologically meaningful subset before running.
7. **Configure (optional):** `get_maboss_initial_state()` to inspect current state, then `set_maboss_initial_state(...)` if non-default probabilities are needed. For one node, use `[P(OFF), P(ON)]`. For multiple nodes, use JSON-native records such as `[{"state": [0, 0], "probability": 0.4}, {"state": [1, 0], "probability": 0.6}]`. State-vector order must match `nodes`, and probabilities must sum to 1. Only use node names returned by `get_maboss_nodes()`.
8. **Run:** `run_simulation()` — preserves an immutable run manifest and exact inputs/outputs; `result.csv` is only a latest-result alias
9. **Analyse:** `get_simulation_result()` — returns the state probability table as a Markdown table
10. **Visualise:** `visualize_network_trajectories()` — saves a PNG artifact
11. **Mutate:** `simulate_mutation(nodes, state)` — preserves a one-off mutant run without changing the latest ordinary selection
12. **PhysiCell handoff:** `export_maboss_handoff(target_cell_type=...)`
    snapshots the current model, parameters, outputs, optional result, and
    complete NeKo lineage.

> **State space warning:** A network with N nodes produces up to 2^N possible Boolean states.
> Always call `set_maboss_output_nodes` to restrict outputs before `run_simulation`.
> For a 30-node network this reduces the result from >1 billion states to only the states
> of the selected output nodes (typically 2-5 nodes).

## 2. Tool Categories
* **Session management:** `create_session`, `list_sessions`, `set_default_session`, `delete_session`
* **Pipeline:** `import_neko_handoff`, `bnet_to_bnd_and_cfg`, `build_simulation`, `run_simulation`
* **Handoff:** `import_neko_handoff`, `export_maboss_handoff`
* **Inspection (read, no side effects):** `get_maboss_nodes`, `get_maboss_initial_state`, `get_maboss_logical_rules`, `get_maboss_mutations`, `update_maboss_parameters` (no args)
* **Configuration:** `update_maboss_parameters`, `set_maboss_output_nodes`, `set_maboss_initial_state`
* **Analysis:** `get_simulation_result`, `simulate_mutation`, `visualize_network_trajectories`
* **Housekeeping:** `list_generated_files`, `clean_generated_files`

## 4. Key Parameters for `update_maboss_parameters`
| Parameter      | Type  | Description                                  |
| -------------- | ----- | -------------------------------------------- |
| `sample_count` | int   | Trajectories (larger = more precise, slower) |
| `max_time`     | float | Simulation time horizon                      |
| `time_tick`    | float | Discretisation step                          |
| `discrete_time`| int   | 0/1 toggle for discrete time mode            |
| `thread_count` | int   | Parallel threads (environment-dependent)     |

## 5. Critical Rules
* Always call `create_session()` before any simulation tool.
* All file I/O is scoped to `<configured-or-user-cache-root>/MaBoSS/<session_id>/`.
* Pass `session_id` explicitly when running multiple simulations in parallel.
* Call `update_maboss_parameters` with no args to list all valid keys.
* Set `thread_count` early to speed up iteration.
* Keep an imported NeKo manifest and its BNET artifact until the MaBoSS
  handoff has been exported; integrity is rechecked before lineage is emitted.
* `export_maboss_bnd_cfg` is a standalone file export. Use
  `export_maboss_handoff` when PhysiCell needs typed provenance and context.
"""

# Additional maintained manual guidance; no automatic scientific authorization.
MABOSS_AGENT_MANUAL += """

Wave 2: inspect selected nodes with get_maboss_nodes(selected_nodes=[...]) for
rules, output visibility, rate expressions and model_revision. Use
update_maboss_parameters(rates=[...], expected_revision=...) only for explicit
coefficient decisions; custom expressions fail without guessing.
edit_maboss_readouts requires approved named rules, rates and initial probabilities.
Readouts are stochastic nodes, not instantaneous observables, and cannot feed back
into the core. Every changed model requires the workflow's scientific review again.
Run IDs returned by ordinary/mutant execution support get_simulation_result and
visualize_network_trajectories after restart. List saved files to discover manifests.
Never rerun a simulation implicitly after a persistence failure.
"""
