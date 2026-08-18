"""User-facing server instructions and workflow guidance for NeKo."""

NEKO_SERVER_INSTRUCTIONS = (
    "Create a session before building a signalling network, and pass "
    "`session_id` explicitly when working with multiple networks. Run "
    "`preview_connection_impact()` before applying expensive connection "
    "strategies, inspect network history after topology changes, and prefer "
    "`verbosity='summary'` during iterative work. Export with `format='bnet'` "
    "for a standalone Boolean file, or use `export_neko_handoff` to preserve "
    "typed MaBoSS provenance. Read `docs://neko/agent_manual` or use "
    "`neko_workflow_prompt` for the complete workflow."
)


NEKO_AGENT_MANUAL = """
# NeKo to MaBoSS Workflow Manual

## 1. Recommended Execution Order
1. **Initialize:** `create_session()` -> `set_default_params(max_len=2,
   path_policy='one_shortest', reuse_policy='discovered_paths',
   only_signed=True, consensus=True)`
2. **Build:** `create_network([...list_of_initial_genes...], database='omnipath')`
3. **Curate:** `remove_bimodal_interactions()` -> `remove_undefined_interactions()`
4. **Audit Connectivity:** `analyze_connectivity()` reports both isolated
   (0-edge) nodes and the full connected-component partition in one call.
   - *If disconnected:* `preview_connection_impact()` -> Apply a connection tool
     (see the cost guide below before choosing one).
   - *If validating a requested gene specification:* call
     `analyze_gene_set(genes=[...])` to distinguish the requested-gene induced
     subgraph from connectivity supplied by added intermediates.
5. **Inspect history:** `list_network_history()` after topology changes. Use
   `compare_network_states(state_a, state_b)` before deciding whether to
   `navigate_network_history(action='checkout', state_id=...)`.
6. **Inspect network:** `list_genes_and_interactions(verbosity='preview')`
7. **Export:** Use `export_neko_handoff(biological_context=...,
   output_nodes=[...])` for a typed MaBoSS transfer. Use
   `export_network(format='bnet')` only when a standalone BNET is sufficient.

## 2. Tool Categories
* **Sessions:** `create_session`, `list_sessions`, `set_default_session`, `delete_session`, `status`, `reset_network`
* **Construction:** `create_network`, `add_nodes` (batch add, optional cheap
  direct-neighbour autoconnect), `remove_gene`, `remove_interaction`
* **Connectivity diagnostics:** `analyze_connectivity` (isolated nodes + full
  component partition), `analyze_gene_set` (requested-gene resolution,
  internal/boundary edges, and induced components),
  `preview_connection_impact` (non-mutating scout: hub ranking or a simulated
  parameter-relaxation preview)
* **Connection strategies:** `connect_targeted_nodes` (integrate specific
  nodes), `bridge_components` (connect group A <-> group B),
  `apply_global_connection` (whole-network closure)
  - See https://github.com/sysbio-curie/Neko/blob/development/docs_mkdocs/strategies/index.md
    for the concise strategy and path/reuse policy semantics.
* **Inspection:** `list_genes_and_interactions`, `find_paths`, `get_references`, `filter_interactions`
* **History:** `list_network_history`, `navigate_network_history`, `compare_network_states`, `set_network_history_limit`
* **Handoff:** `export_neko_handoff` records exact sanitized Boolean nodes,
  declared outputs, package versions, history state, and artifact digests.

## 3. Connection Strategy Cost Guide
Choose the cheapest strategy that can plausibly close the gap; escalate only
if it fails. All costs assume seed/group sizes of a few dozen genes.

| Tool | Strategy | Cardinality | Relative cost | Notes |
|---|---|---|---|---|
| `add_nodes` | `autoconnect=True` | new node(s) -> existing network | Very low | Direct-neighbour edges only (equivalent to maxlen=1); no multi-step path search |
| `connect_targeted_nodes` | `connect_to_upstream_nodes` | specific node(s) | Low, bounded by `depth` | Cascades upstream from the given nodes only |
| `connect_targeted_nodes` | `connect_subgroup` | one node list | Moderate, ~O(pairs in group) | Pairwise path search within the group only |
| `bridge_components` | `connect_component` (A, B) | group A <-> group B | Moderate-high | Also silently runs `connect_subgroup` on every node NOT in A or B as a side effect |
| `apply_global_connection` | `connect_network_radially` | whole network | Moderate-high, bounded by `max_len` hops | Expands outward/inward from existing seed nodes only, not all pairs |
| `apply_global_connection` | `connect_as_atopo` | whole network, output-anchored | High, open-ended | Runs `connect_network_radially` or `complete_connection` first, then loops upstream search until the network is fully connected - the loop is not bounded by `max_len` alone |
| `apply_global_connection` | `complete_connection` | whole network | **Highest - O(N^2) over every node pair** | Searches paths between every pair of nodes in the network; this is almost certainly the cause of a network exploding in size (e.g. `max_len=2` with 20+ seed genes producing hundreds of edges) |

**Large-network warning:** before calling `complete_connection` or
`connect_as_atopo`, check the current node count (via `status()`). Above
roughly 50 nodes, prefer `connect_targeted_nodes` or `bridge_components` to
close specific gaps instead, or run `preview_connection_impact()` first to see
the predicted edge-count delta without committing.

## 4. Critical Operating Rules
* **Session First:** Always call `create_session` before `create_network`.
* **Scout Before You Shoot:** Always run `preview_connection_impact()` before
  heavy connection tools, and check the cost guide above.
* **Output Names:** Handoff output nodes may use original NeKo names; export
  translates renamed symbols to the exact names stored in the sanitized BNET.
  If outputs are omitted, MaBoSS must select a small output set before running.
* **Token Frugality:** In iterative loops, ALWAYS use `verbosity='summary'`.
"""
