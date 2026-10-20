# Wave 2 NeKo wrapper integration

Scope: MCP-owned W2-D003–W2-D007. Scientific mutation authorization remains
external to the API; providing evidence or exporting a model does not approve it.

## Coordination

The library implementation is developing in `/tmp/neko-wave2-curation`, branch
`codex/wave2-curation-history`, based on NeKo `673875d`. The MCP agent has left
an integration request and feedback in that worktree's temporary
`MCP_CONTRACT_REQUEST.md`. The public library contract must be confirmed before
integrating the editor and history persistence. This document does not declare
the developing source to be a released dependency or an agreed API.

The independent node-reporting correction uses the existing library contract:
`add_node` returning `False` is a refusal. Successful calls only count as added
when the node table grows; accepted existing nodes are reported as unchanged.
Partial failure preserves accepted nodes, returns a tool error identifying the
rejected input, invalidates the edge cache, and skips autoconnect.

## Reproducible interface measurements

Run from the repository root with its development dependencies installed:

```sh
python -m tests.measure_neko_interface > /tmp/neko-interface.json
```

The baseline in [measurements/neko-wave2-before.json](measurements/neko-wave2-before.json)
uses the interfaces from MCP `0e3f7fc`; node-reporting commit `591b1ca` leaves
those schemas unchanged. It uses MCP 2.2.0 and Pydantic 2.13.5, with compact UTF-8
JSON serialization, aliases enabled and absent optional fields excluded. Sizes
exclude the JSON-RPC envelope. The fixture exercises actual MCP protocol
serialization with a deterministic library double; it is not a combined-library
acceptance test.

| Baseline metric | Bytes/count |
| --- | ---: |
| Registered tools | 33 |
| Complete tool listing | 81,561 bytes |
| Input schemas | 29,613 bytes |
| Output schemas | 37,234 bytes |
| Status response | 503 bytes |
| Inventory preview response | 1,075 bytes |
| Four-state branching history response | 2,497 bytes |
| State-comparison response | 1,378 bytes |

The report also measures history with an 8,192-byte explanation to expose
growth from detailed provenance. Repeat these measurements after integration
and record the difference, including the new editor's schema and response.
Byte sizes do not establish token savings, tool-selection accuracy, or client
support for deferred discovery. Those require a separate client benchmark.

## Pending integration acceptance

- Explicit add/set_effect operations, existing endpoints only, evidence and
  rationale validated before library mutation; precise outcomes and stale-state
  rejection, including no-ops and atomic batch failure.
- Manual origin, available PMIDs and original evidence remain distinguishable
  in inspection and export; unbounded history stays in a separate artifact.
- Existing history tools expose state identity, branches, actual inclusion
  reasons, effective parameters and reported attribution/ancestry gaps.
- Database name comes from the constructed network, never mutable defaults.
- BNET/SIF and Boolean/ODE handoffs retain typed provenance with verified
  hashes and matching source/session/state identity. Existing import and
  authorization boundaries remain intact.
- Combined tests use the updated real NeKo source in a fresh interpreter,
  covering construction, manual edits, branching/checkout, history restore,
  export/handoff provenance and tamper rejection.

This list records outstanding work, not completed acceptance evidence.
