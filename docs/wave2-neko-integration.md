# Wave 2 NeKo wrapper integration

Scope: MCP-owned W2-D003–W2-D007. Scientific authorization remains external to
the API. Evidence, history navigation and exports never approve a mutation or
handoff. No scientific modelling state was changed during implementation.

## Agreed library boundary

The NeKo agent confirmed the contract in its `MCP_CONTRACT_REQUEST.md` reply
and `docs_mkdocs/interaction-curation.md`. The wrapper uses public APIs only:

- `Network.edit_interactions(edits, expected_state_id=None)`: atomic validation
  and mutation, actual before/after records, resulting state IDs, and coded errors.
- `Network.get_state(state_id)`: an exact snapshot without checkout.
- `history_dict()`, `export_history(path)`, `Network.from_history(path)`: the
  existing state graph, structured explanations, provenance and restoration.
- Read-only `database_name` and the structured `Provenance` edge column.

The tested source is the NeKo agent's uncommitted candidate based on official
`673875d09e4c49c97d0d760d3ea85bd2603efc82`. Code and contract hashes are in
[neko-wave2-source.json](measurements/neko-wave2-source.json). This does not
claim the API is released on PyPI. Before publishing, align the dependency minimum
and CI source pin with the final NeKo release/commit. Older libraries retain
existing interfaces but report a missing API for the new editor/history export.

## Editing and inspection

`edit_interactions` accepts 1–20 explicit `add`/`set_effect` records with source,
target, effect, rationale and evidence (summary, PMIDs and/or publication citations).
Unknown fields and unsupported operations are rejected. Evidence requires a
citation even when PMIDs are absent; PMIDs are never synthesized. NeKo owns
endpoint resolution, conflict validation and provenance semantics. There are no
low-level edge writes, implicit nodes, connection completion or automatic exports.

Get `network_id` and `state_id` from `status`, then supply `expected_network_id`,
`expected_state_id`, and an explicit `session_id`. Replacing a network changes its
identity even if the new state has the same number. The per-session lock covers
checks and mutation. Results contain actual canonical endpoints/effects, original
and manual edit origin, unchanged count, resulting state and a full-details URI.
No-op effect changes append no evidence and leave the edge cache valid.

`add_nodes` honors library refusal and counts observed growth. Already-present
nodes are unchanged. Partial failures retain accepted nodes, return an error,
invalidate the cache and skip autoconnect. Inventory, filtering and reference
queries preserve structured provenance and exact network/state identity. Original
resource evidence stays distinct from manual correction evidence. Database names
come from construction, never mutable defaults. No database release, acquisition
time, dataset fingerprint or database snapshot tracking was added.

## Existing history interfaces

`list_network_history(offset=0, limit=25)` pages at most 100 states. Display
metadata, relationship lists and ancestry-gap previews are bounded and point to
`neko://session/{session_id}/history/{network_id}/{state_id}` for full snapshots
and explanations. An old URI cannot silently read a replacement network.
`compare_network_states(limit=25)` retains exact canonical changes from the
library and links to both complete states. Large changed-edge evidence is omitted
explicitly from the summary, not from the full state. The HTML/SVG resource remains.

`export_network(format='history')` writes all retained snapshots, working state,
structured explanations and branch relationships in NeKo's supported JSON format,
with MCP identity attached. `Network.from_history(path)` reloads it directly.
Summaries report explanation status, limitations, pruning counts, available
ancestry gaps and whether working state matches saved state. They do not promise
complete history or identical reconstruction against another database of the same
name. Retention behavior and its explicit mutation tool are unchanged.

## Exports and compatible consumers

Curated SIF/BNET exports and Boolean/ODE handoffs carry an immutable typed history
companion. Its `neko-export` v1 envelope contains the exact `neko-history` payload,
MCP identity and a binding to the published model. `Network.from_history(companion)`
restores the source network, not the Boolean projection. Provenance beyond 16 MiB
fails explicitly; the wrapper never prunes history to fit a handoff.

BNET sanitization retains the library projection, updates its filename/hash to
the final bytes and updates exported identifiers. The companion records the
sanitizer name map. ODE provenance stays nested JSON. Legacy ODE edge IDs are
unchanged when provenance is absent.

| Document | Provenance-bearing version | Older documents readable |
| --- | --- | --- |
| NeKo → MaBoSS | 2.1, required `curation_file` | 1.0, 2.0 |
| NeKo source-lineage sidecar | 2.1, required `curation_file` | 1.0, 2.0 |
| NeKo → BioMASS | 2.0, required `curation_file` | 1.0 |
| MaBoSS → PhysiCell | Existing 2.0; verifies NeKo parent recursively | 1.0 |

Use the coordinated server package for new manifests. Older strict consumers
reject the new version rather than discard provenance. Retain referenced files
with a live handoff. Historical source-lineage bundles embed provenance bytes and
validate them without reading historical paths. Owner, role, state, size, hash
and ancestry checks stay active. Mismatched/tampered provenance fails before
import. Older artifacts remain historical and do not approve the current network.
Unrecorded working changes cannot export a handoff claiming the old saved state.

## Reproducible measurements

```sh
python -m tests.measure_neko_interface > /tmp/neko-interface.json
MPLBACKEND=Agg NEKO_WAVE2_SOURCE=/absolute/path/to/neko-candidate \
  python -m pytest tests/test_neko_wave2_runtime.py -q
```

The [before](measurements/neko-wave2-before.json) and
[after](measurements/neko-wave2-after.json) reports use MCP 2.2.0 and Pydantic
2.13.5. Sizes are compact UTF-8 JSON bytes, aliases enabled, absent optional
fields excluded, without the JSON-RPC envelope. Deterministic library doubles
exercise real MCP serialization; real-library acceptance runs separately.

| Metric | Before | After |
| --- | ---: | ---: |
| Tools | 33 | 34 |
| Complete tool listing | 81,561 | 89,240 |
| Input schemas | 29,613 | 31,977 |
| Output schemas | 37,234 | 42,109 |
| Status | 503 | 589 |
| Inventory preview | 1,075 | 1,161 |
| Four-state history | 2,497 | 4,673 |
| History with 8,192-byte explanation | 18,930 | 4,942 |
| State comparison | 1,378 | 2,044 |
| One interaction edit | No native editor | 1,422 |

The listing grows 9.4%; the large-explanation response shrinks 73.9%. Short
history results grow because they include identity, attribution status and detail
links. This is bounded routine output, not universally lower context usage.
The first edit needs one status and one editor call; subsequent edits can use the
returned identity/state if nothing intervened. Full evidence adds a resource read.
The previous manual SIF workaround had no equivalent native single-call baseline.

Tests reject wrong operations and malformed evidence before library mutation.
No live client/model tool-selection error rate or tokenizer benchmark was run.
Specialist separation remains intact; task subsets/deferred discovery depend on
the client and are not assumed or imposed here.

## Validation

Protocol tests cover bounds, actual changes/no-ops, stale network/state checks,
refusals and bounded history with complete detail resources. Integrity tests cover
rehashed identity/model mismatches, tampering, offline ancestry and sanitizer
rebinding. Existing handoff/authorization behavior is regression-tested.

`tests/neko_wave2_workflow.py` runs real updated NeKo in a fresh interpreter:
offline construction and explanations, manual addition, resource-effect correction,
atomic failure, branches/checkout, evidence inspection, history export/reload,
state-graph rendering, both handoffs, pyMaBoSS BNET conversion, and portable
ancestry after originals are deleted. It performs no biological simulation and
does not assert scientific validity of its synthetic citations. The runtime test
skips on legacy installed NeKo unless `NEKO_WAVE2_SOURCE` is supplied; an explicitly
configured candidate missing the API fails rather than skipping.

Validation on 2026-10-20 used Python 3.13.13. The headless full suite exercised
777 tests: 776 passed and one exposed the environment's stale MCP 2.4.0 package
metadata. Installing the freshly built 2.5.0 wheel resolved it; all seven metadata
tests then passed. The final focused protocol/integrity/real-source/metadata run
passed 38 tests. Ruff, mypy (16 CI modules), wheel/sdist builds, and wheel module-presence
checks passed. The initial GUI plotting-backend abort was avoided by using
`MPLBACKEND=Agg`; no plotting implementation was changed for this feature.
