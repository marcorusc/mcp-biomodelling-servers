# Wave 2 MCP feature integration

This implements the accepted MCP work in W2-D008–010, W2-D014, W2-D016–020,
alongside the previously implemented W2-D003–007. Scientific authorization stays
with the governing workflow. An edit, saved run, or configuration export does
not grant approval or establish biological validity.

## Interfaces and ownership

| Decision | MCP implementation |
| --- | --- |
| D008: summary readouts | New `edit_maboss_readouts`, explicit `add`/`set_rule`, at most 20 edits, exact expected model revision. |
| D009: named rates | Existing `get_maboss_nodes(selected_nodes=...)` and `update_maboss_parameters(rates=..., expected_revision=...)`, at most 25 selected nodes/edits. |
| D010: durable runs | Existing ordinary/mutant run, result, plot and generated-file interfaces; immutable run IDs and shared artifact preservation. |
| D014: existing Settings capabilities | Existing cell/PhysiBoSS configuration tools expose cycle transitions, fractional time stochasticity and output steepness. |
| D016: user parameters/CSV | New `configure_simulation_inputs`: typed parameters, existing CSV capture, explicit enable/disable, stored readback. |
| D017: portable configuration | New `export_configuration_bundle`: XML, supported inputs, provenance, checksums and working-directory instructions. |
| D018: required substrate | Materialize the library default before publishing a new configuration; expose actual values/origin and readiness. Preserve imported entries. |
| D019: compact inspection | Selected MaBoSS rules/rates/outputs and actual PhysiCell stored settings; cell-specific available behavior names. |
| D020: NeKo diagnostics | Import/construction distinction, accurate canonical counts, resource-specific consensus text, bounded paths with identity, actionable export-label failures. |

Only three tools are added: one MaBoSS editor and two PhysiCell tools. No new
history family, general-purpose executor, runtime dependency or engine/package
modification is introduced.

## MaBoSS readouts, rates and revisions

Readouts are stochastic simulated nodes, not instantaneous observables. Creation
requires an explicit Boolean rule, explanation, up/down rates and initial
probability. Rules accept core-node names, `0`, `1`, `!`, `&`, `|` and parentheses.
Readouts cannot feed back into the biological core or depend on other readouts.
Name collisions, unresolved regulators and malformed rules fail before publication.
Adding a readout preserves the existing output selection; use
`set_maboss_output_nodes` explicitly to select it.

Inspect `get_maboss_nodes(selected_nodes=["A"])`, then pass its `model_revision`
to the editor. Revisions hash effective BND/CFG content and readout definitions.
They change with numerical settings, initial states, output selection and rules.
Edits validate and prepare their result before replacing the active model.
The governing workflow must obtain approval for the resulting exact revision.

Rate inspection returns expressions, directly referenced parameter values,
unresolved parameters and output status. It does not claim to evaluate the
instantaneous rate in every state. Explicit coefficient edits support standard
`@logic ? coefficient : 0` / `@logic ? 0 : coefficient` forms and existing
unconditional constants/parameters. Custom expressions are rejected when their
coefficient cannot be identified safely. Values are inlined per node so that a
shared global parameter does not silently change other nodes. Global parameter
and per-node rate updates cannot be mixed in one call.

Readout definitions and source revision are retained in a `MCP_READOUTS_V1` BND
comment and verified on reload. MaBoSS-to-PhysiCell handoffs now write version
**2.1**, with required model revision and explicit readout definitions (an empty
dictionary is valid). Consumers still accept 1.0/2.0 without that extension.
The verifier checks definitions against actual BND rules, permits only declared
readout additions beyond the NeKo core, forbids core feedback/reclassification,
and retains recursive ancestry, owner, size and hash checks. Earlier strict
consumers need the coordinated update to read 2.1.

## Persistent execution and bounded results

Ordinary and mutant runs share the existing preservation helper. Each complete
inventory records a unique run ID, full session ID, source revision, kind,
requested mutations, effective parameters/rates/initial states/output selection,
readout definitions, exact engine BND/CFG inputs, produced raw outputs and tables.
The inventory is published last. Mutation execution uses a copy and does not
replace the latest ordinary selection.

`get_simulation_result(run_id=...)` and
`visualize_network_trajectories(run_id=...)` work from saved artifacts after
session removal/restart without executing MaBoSS. An explicit owner, when
supplied, must match. Unknown, ambiguous or tampered IDs fail without fallback.
Omitting an ID selects the active session's latest preserved ordinary run and
returns the resolved ID. `list_generated_files` accepts a persisted full session
ID and exposes inventories after restart.

Result inspection defaults to the final snapshot. Set `final_only=false` for
pages (default 25, maximum 100 timepoints); tables expose at most 100 states.
Plots expose at most 1,000 timepoints and 100 states, with an optional `until`
bound. Every variant has a unique image and rendering metadata. Full tables
remain in artifacts; these caps do not alter engine output or create new analysis.
Mutation responses now return a compact run summary rather than a full trajectory.
The legacy `result.csv` remains a convenience alias, not the historical record.

Engine failures and saving failures are reported separately. Available engine
files are copied into incomplete diagnostic records (16 MiB per-file limit);
if diagnostic storage also fails, the original phase remains visible. Earlier
successful selection remains intact, and a saving failure never triggers a retry.
No stationary-distribution calculation is enabled automatically.

## PhysiCell configuration and bundles

Configuration tools read back stored values and preserve explicit zero/false.
Cycle edits target existing phase pairs; they do not silently convert a
duration-based cycle to a transition-rate policy. Fractional time stochasticity
is retained. Output steepness is a nonnegative **integer**, matching the target
PhysiBoSS engine reader, while omission preserves the library default.

For that engine, output mapping uses a Hill-transformed smoothed Boolean state:
activation is `base + (value-base)*H`, inhibition is
`base + (value-base)*(1-H)`. Finite steepness is not an exact ON/OFF switch.
Behavior discovery filters phase-exit names using the selected cell's actual
cycle and returns aliases only when the backend provides them.

Typed user parameters require explicit `int`, `double`, `bool` or `string`,
matching values and units. No-update calls inspect selected parameters, the
stored `number_of_cells` and CSV setup. Registering an existing CSV captures
its bytes/hash in session artifacts and requires explicit `csv_enabled`.
It does not change cell counts or generate placement.

Supported CSV input is bounded to 16 MiB and four columns `x,y,z,type`.
Headerless PhysiCell CSV v1 uses numeric cell-type IDs; a matching header
selects v2 cell-type names. BOMs and quoted fields are rejected because the
target engine's version detector/parser does not support them consistently.
This validates file shape and finite coordinates, not template compatibility.
The executable decides how user parameters, seeds, counts and CSV loading behave;
CSV placement can be additive to template-generated cells.

The existing Settings serializer inserts a required substrate if none exists.
Creation/loading materializes that default before exposing readiness, and
inspection distinguishes `library_default`, `imported` and
`explicit_configuration` origins. Names, units and numerical defaults are
unchanged. A default is not claimed to be biologically neutral. Adding another
substrate preserves it; duplicate-name addition fails explicitly because the
current Settings API cannot guarantee coherent replacement IDs/references.
Safe substrate replacement/removal remains a library-coordinated operation.

Configuration bundles preserve XML plus enabled initial-cell CSV, cell-rule CSVs
(including currently stored rules) and per-cell BND/CFG inputs. Collision-free
paths allow two cell types to reference different files with the same basename.
A copied configuration is serialized; source files, active settings and previous
exports remain unchanged. Publication is atomic and unique. Unknown external
input references and missing supported inputs fail before publication.

Relative source references require an explicit original `working_directory`;
the XML directory is not assumed. Extracted bundles use their root as the
executable working directory. The manifest includes configuration revision,
dependency hashes/sizes, substrate origins and verified upstream lineage.
An existing compatible executable/template is still required: compilation,
engine installation, compatibility validation and scientific approval are not
provided by a configuration export.

## NeKo diagnostics and coordinated dependency

SIF import reports imported topology separately from explicit subsequent
construction. Counts use the canonical node inventory, including label-only
nodes. SIGNOR consensus compares grouped sign-record totals; tied flags do not
establish a sign. Missing consensus fields resolve to undefined in the inspected
library. No universal multi-database/high-confidence or SIGNOR-no-op claim remains.

Path responses separate errors/warnings from actual paths and report the requested
length/direction scope. Available labels and UniProt IDs accompany paths;
ambiguous or unresolved identity stays explicit. Display is capped at 20 paths
by default (maximum 100). The library still computes/captures paths within the
requested length before this display bound; this is not a computation limit.
No path within a bound does not establish disconnected components. BNET export
fails with offending node IDs/rows for missing labels, without inventing names.

CI now pins NeKo **1.10.2** at the pushed commit
`7b845ae0d7cfc3578c7191b2147cb3f5ab5f33e3`. The lightweight public contract is
`history_status()`, `state_count` and
`list_states(offset=..., limit=..., detail="summary")`. Routine MCP history uses
those interfaces without building a full graph or materializing all metadata.
Unknown legacy change counts remain null. Full evidence is still accessible
through existing state/history resources. Older candidates retain a public-API
fallback. The package range stays `nekomata>=1.10.2,<2`; verify the published
release contents before replacing the Git pin.

PhysiCell Settings remains pinned to 0.6.3 candidate
`82f66c4504da2408d9e8037b78896ac29551d150`. Engine semantics were checked against
the local PhysiCell source at `69a23dbe7eaac4f41dc9548ad85072eff7921a03`.

## Governing-workflow adoption

The separate modelling/orchestrator repository was not edited. Its maintenance
work should permit the named readout/rate operations only under its existing
scientific authorization, bind review to the returned revision, accept verified
2.1 handoffs with declared isolated readouts, save run IDs for later inspection,
and consume actual stored-setting responses. Do not treat artifact preservation,
configuration completeness or a successful test as approval.

The deferred B-MB-2 import investigation, MaBoSS_test integration, new analysis
algorithms, expanded mapping validation and broader benchmark campaign remain
outside this implementation.

## Context measurements and verification

Measurements compare baseline commit `30bf4dd` with this feature implementation
using real registered MCP tool schemas and offline import fixtures. Values below
are `o200k_base` tokens; the JSON-RPC envelope is excluded. Complete listings
include output schemas. A client's actual prompt composition may differ.

| Server | Tools before → after | Input definitions before → after | Complete listing before → after |
| --- | ---: | ---: | ---: |
| NeKo | 34 → 34 | 9,568 → 9,642 | 21,015 → 21,222 |
| MaBoSS | 24 → 25 | 5,544 → 6,602 | 14,771 → 16,726 |
| PhysiCell | 34 → 36 | 8,653 → 9,627 | 18,640 → 20,555 |

A synthetic 100-node model's raw BND/CFG is 4,515 tokens; the complete selected
one-node rate tool result is 1,192 tokens (765 in its structured payload).
This includes the legacy inventory/output-name lists, revision and text content,
not just the smaller rate-detail fragment. Session IDs can slightly vary counts.
This demonstrates the targeted response's size, not equivalence to inspecting
every node. Raw byte/token data
and methodology are in [the measurement record](measurements/wave2-server-surface.json).
Specialist isolation remains appropriate: exposing all complete listings at once
would still be expensive. No whole-server runtime/memory performance claim is
made from these static measurements. Durable mutation storage and bundles add
intentional disk I/O; lightweight history avoids repeated full-history copying.

Reproduce schema measurements with `tests/measure_wave2_surface.py CHECKOUT`.
Token counts require an optional external `tiktoken` installation/cache, not a
production dependency. `tests/test_neko_wave2_runtime.py` exercises the combined
real NeKo → MaBoSS → PhysiCell path with readouts, lineage and a portable bundle.
Protocol and artifact tests use deterministic engine-result fixtures; these are
developer checks, not a stochastic-engine benchmark or scientific validation.

Validation on Python 3.13.13: all **807 tests passed** against installed NeKo
commit `7b845ae0d7cfc3578c7191b2147cb3f5ab5f33e3` with Wave 2 required.
After the namespace-import corrections, a focused 21-test run also passed,
including the combined real-library workflow and an isolated real Settings suite.
CI Ruff/syntax checks, mypy's 16 configured modules and `pip check` passed.
Wheel/sdist builds succeeded, and `tests/verify_wave2_wheel.py` imported all new
runtime modules plus MaBoSS/PhysiCell registrations from the extracted wheel
outside the source checkout. CI now repeats this packaging check. Python
3.10/3.14 compatibility lanes remain configured; they were not run locally here.
