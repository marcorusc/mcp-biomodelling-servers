# Wave 2 performance, context and dependency audit

Date: 2026-10-20. The added provenance has a measurable runtime and memory cost.
It is concentrated in mutation, retained history and exports. Routine status
queries remain small and fast. Context use is manageable with summaries, but
large evidence previews and repeated history listings remain a risk.

## Method and limits

Compare NeKo official `673875d` with candidate
`ba87f7631dd6420b5269852daceb303f773e8963` (identical Python source to the candidate
previously tested), and MCP `0e3f7fc` with `7afe965` before this audit's export
optimizations. Both use the same Python 3.13 environment and synthetic directed
networks, imported from SIF. Identifier mapping and provider discovery are offline;
no database download or scientific simulation is timed. Timings are local medians,
not production throughput guarantees. First-call MCP schema construction is
included in the raw samples but does not dominate the five-sample median.

`tests/benchmark_neko_runtime.py` measures imports, SIF construction, repeated
existing-node no-ops, label changes, history listing and exports. Each successful
label change saves one state. These common operations allow a valid before/after
comparison; the old library has no equivalent validated interaction editor.
Python heap is measured with `tracemalloc` in a separate untimed run. It excludes
imported libraries and is not total process RSS or a complete measure of native
allocations. Evidence accumulation can cost more than these label-only histories.

Raw measurements are in
[neko-wave2-performance.json](measurements/neko-wave2-performance.json).

## Runtime and memory

Small fixture: 100 nodes, 200 edges, 20 changes, 21 retained states.

| Operation / retained data | Before | Wave 2 |
| --- | ---: | ---: |
| SIF construction | 71.1 ms | 129.7 ms |
| Existing-node no-op | 2.0 ms | 8.3 ms |
| Label mutation with saved state | 3.6 ms | 18.9 ms |
| Library SIF export | 29.8 ms | 184.9 ms |
| MCP status, warm median | 0.71 ms | 0.56 ms |
| MCP history listing | 1.63 ms | 38.74 ms |
| Traced retained heap after history | 0.87 MB | 3.35 MB |
| Heap growth during the 20 changes | 0.37 MB | 2.27 MB |

Large fixture: 500 nodes, 1,000 edges, 50 changes, 51 retained states.

| Operation / retained data | Before | Wave 2 |
| --- | ---: | ---: |
| Label mutation with saved state | 9.63 ms | 90.50 ms |
| MCP history listing | 3.55 ms | 466.34 ms |
| Traced retained heap after history | 4.75 MB | 29.20 MB |

Do not interpret the sub-millisecond status difference as a speedup. The important
finding is that ordinary status remains cheap while history-related work scales
with the network and retained snapshots. No new runtime dependencies were added
by these Wave 2 changes.

Three alternating fresh-process import runs with a warm filesystem cache show
essentially unchanged startup: median library import 1,109 → 1,113 ms, additional
wrapper import 420 → 422 ms, and peak import-process RSS 144,808 → 145,676 KiB.
This excludes provider discovery and cannot predict a cold online launch.
The measured Python source directories grow from 267,832 to 296,577 bytes for
`NeKo/` plus shared MCP modules, and 261,123 to 305,038 bytes for `neko/`.
These are source-file sizes, not whole wheel or environment footprints.

The wrapper contributes avoidable work: with the **same new library**, the old
wrapper's history query takes 10.95 ms versus 38.74 ms for the new wrapper on the
small fixture. However, the old wrapper returns 373,900 bytes of explanations;
the new wrapper returns 26,223 bytes. Output bounding helps context but currently
does not bound backend work. `summarize_history()` calls `history_dict()` just to
obtain pruning and working-state flags, serializing every retained snapshot.

## Context

Counts use `tiktoken` 0.14.0, `o200k_base`, on compact MCP JSON. These are
reproducible serialized-payload counts, **not** the live client's internal prompt,
current conversation usage or remaining model context. UUIDs can move counts by
a few tokens. Clients differ in discovery, output-schema handling and text versus
structured-content rendering.

| Payload | Before tokens | Wave 2 tokens |
| --- | ---: | ---: |
| Names, descriptions and input schemas | 8,945 | 9,602 |
| Full MCP tool listing including output schemas | 19,138 | 21,078 |
| Status fixture | 146 | 184 |
| Small inventory preview | 306 | 342 |
| Four-state history | 760 | 1,548 |
| History with an 8,192-byte explanation | 4,869 | 1,617 |
| Single interaction edit | No equivalent tool | 494 |

The input-definition increase is 657 tokens, about 7.3%. The full listing grows
about 10.1%. Loading all tools from all modelling servers would add their schemas
too; this audit measures NeKo, not a live combined-client context.

An additional fixture puts 20 evidence records on one edge, each with approximately
1 KB of rationale and 1 KB of summary. Inventory **summary** is 721 bytes / 211
tokens; **preview** is 46,550 bytes / 6,224 tokens; **full** is essentially the
same for this small graph. Row limits do not bound accumulated evidence inside
each row. The editor's response stays compact, but inspection can still grow.

Prefer summaries, targeted interaction/reference queries and small history pages.
Read full state resources when their evidence is needed. The next context
optimization should explicitly make large evidence opt-in in previews while
preserving complete evidence in full inspection/resources. Do not silently drop
PMIDs or confuse a truncated preview with the authoritative record.

## Optimizations made in this audit

1. Reuse the history already serialized into the library BNET companion instead
   of asking the library to traverse it again. The reuse stays within the locked
   export operation. State/owner/model checks and rejection of unsaved working
   changes remain active.
2. Serialize MCP provenance companions and standalone history as compact JSON.
   No fields, snapshots or evidence are removed. The history payload alone drops
   from 2.63 MB indented to 1.25 MB compact for the small fixture, and from
   29.84 MB to 14.21 MB for the large fixture. The large BNET export now succeeds
   within the existing 16 MiB companion bound. Larger histories can still exceed
   it and must fail explicitly.

On three small-fixture BNET export samples, median MCP export time changes from
375 ms to 348 ms. Treat this modest difference as indicative, not a statistically
established throughput improvement. The storage reduction is deterministic.
The large optimized BNET export takes about 2.97 s.

## Remaining priorities for the NeKo PR

1. Add a public `history_status()` returning `pruned_state_count` and
   `working_state_matches_current` without serializing the retained graph. Keep
   the existing authoritative working-state check. MCP can then remove its
   full-history traversal from every summary without reimplementing semantics.
2. Consider paged/compact `list_states` access. Currently it deep-copies all
   transition explanations before the wrapper slices the requested page.
3. Profile repeated frame-to-JSON fingerprints, deep copies of nested provenance,
   repeated before/after evidence in transition metadata, and copying the whole
   network history when constructing an exporter. Any sharing must preserve
   independent old snapshots and atomic failure behavior.
4. For longer-term storage improvements, consider a versioned deduplicated
   history representation. This needs coordinated library/export compatibility;
   automatic pruning or weaker provenance checks are not substitutes.

The exact proposal and measurements were left for the NeKo agent in
`MCP_PERFORMANCE_REVIEW.md`. The audit does not modify NeKo source or claim those
remaining library optimizations are already implemented.

## Dependency alignment and PR readiness

MCP CI now pins:

```text
nekomata @ git+https://github.com/marcorusc/Neko.git@ba87f7631dd6420b5269852daceb303f773e8963
```

The existing package range `nekomata>=1.10.2,<2` matches the candidate's declared
1.10.2 version. The historical `requirements-ci-wave1.txt` filename is retained;
its comment identifies the Wave 2 NeKo / Wave 1 PhysiCell combination. CI checks
the new API and sets `NEKO_WAVE2_REQUIRED=1`, so an older library cannot turn the
combined workflow into a silent skip.

The resolver installed NeKo 1.10.2 from this exact SHA and PhysiCell Settings
0.6.3 from its existing pin; distribution `direct_url.json` records both commits,
and `pip check` passes. This closes the earlier gap between successful source-path
tests and an installable dependency combination.

Validation after installation: **780 tests passed**, 12 upstream deprecation
warnings, 215.87 seconds, with no source-path override and Wave 2 required.
The configured CI Ruff lint/syntax checks and mypy checks (16 modules) passed.
Workflow YAML and embedded Python were parsed locally. The Python 3.10/3.14
GitHub compatibility matrix has not been run from this session.

To reproduce installation and the required combined workflow:

```sh
python -m pip install -r requirements-ci-wave1.txt ".[dev,biomass-graph]"
python -m pip check
MPLBACKEND=Agg NEKO_WAVE2_REQUIRED=1 \
  python -m pytest tests/test_neko_wave2_runtime.py -q
```

The optional `NeKo candidate compatibility` workflow accepts a repository and
full immutable SHA for future candidates, resolves normal package dependencies,
checks the installed API and runs wrapper/integration tests. GitHub manual
dispatch becomes available once the workflow exists on the default branch.
No GitHub Actions run is claimed here.

The NeKo PR can reference MCP's pinned acceptance result and this performance
review. PyPI still reported 1.10.1 when checked; creating a PR does not require a
new PyPI release. Before releasing MCP, verify the actual published NeKo 1.10.2
wheel against the same required workflow, then replace/remove the temporary Git
pin. Do not assume a version number alone proves the required API is present.

## Reproduction

Use separate extracted checkouts for the old/new wrapper and library, the same
interpreter, and run timing experiments sequentially:

```sh
MPLBACKEND=Agg python tests/benchmark_neko_runtime.py \
  --source /absolute/neko-checkout --wrapper /absolute/mcp-checkout \
  --nodes 100 --steps 20 --output /tmp/runtime.json
python -m tests.measure_neko_interface \
  --evidence-stress --payloads /tmp/payloads.json > /tmp/interface.json
```

Use `--nodes 500 --steps 50` for the larger fixture, `--measure-export` for MCP
BNET export timing, and `--startup-only` for isolated fresh-process imports.
Encode `json.dumps(payload, separators=(",", ":"), ensure_ascii=False)` using
`tiktoken.get_encoding("o200k_base")` to reproduce token accounting. The optional
tokenizer was installed only under `/tmp`; it is not a new project dependency.
