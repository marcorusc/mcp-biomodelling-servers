# Boolean handoff ancestry v2

NeKo and MaBoSS typed exports now write `mcp-biomodelling-handoff` version
`2.0`. Existing typed class names and default v1 constructors remain supported;
`load_handoff_manifest` reads both versions. No ODE contracts or simulation
settings change.

V2 requires an explicit `source_ancestry` list. An empty list means that the
current session has no recorded source import. Each import records:

```json
{
  "source": {"path": "/original/import.sif", "sha256": "...", "...": "artifact metadata"},
  "recorded_at": "2026-10-13T00:00:00Z",
  "relationship": "edited-derivative",
  "parent_manifest": {"path": "/original/parent.json", "sha256": "...", "...": "artifact metadata"},
  "derived_from": {"path": "/original/ancestor.sif", "sha256": "...", "...": "artifact metadata"},
  "evidence": {
    "schema_version": "1.0",
    "blobs": {"<sha256 of exact bytes>": "<base64 exact bytes>"}
  }
}
```

The abbreviated artifact objects above retain the existing full
`HandoffArtifact` fields in real records: server/session identity, role, name,
absolute path, suffix, media type, size, and SHA-256. Paths inside ancestry are
historical identity labels. They are never followed by portable validation.
The bundle includes exact imported bytes, original parent JSON snapshots,
ancestor bytes and complete transitive evidence. Original JSON is not rewritten,
so parent hashes and identities remain intact after relocation.

`standalone` prohibits parent references. `byte-identical` requires equal source
and ancestor digests. `edited-derivative` requires different digests, an explicit
`derived_from` artifact belonging to the verified parent, and matching format.
Neither relationship asserts biological equivalence or approval. An edited SIF
or BNET may introduce or remove nodes.

## Stage parents and source ancestry

The existing `parent_manifest` and `lineage` fields still describe the typed
NeKo → MaBoSS stage transition. Exact node/output checks remain for **both** v1
and v2. Ordinary BNET derivative conversion does not set these fields: its
parent belongs exclusively in `source_ancestry`. Likewise NeKo SIF source
ancestry does not populate typed upstream `lineage`.

MaBoSS propagates source ancestry from its typed NeKo stage parent; PhysiCell
checks that inherited records match and retains them in its source manifest and
local manifest snapshot. Current source records must belong to the exporting
session, or to the explicitly declared NeKo stage parent. Arbitrary session
history is not a model source.

Approval gates must continue to bind the **exact current model artifacts and
manifest**, not an ancestor's digest, relationship or prior approval. Embedded
hashes provide integrity, not signatures or independent authentication. An actor
who can replace an entire manifest and all its trusted digests can invent a new
history; the capture/gate trust boundary must retain the accepted current hash.

## Safe capture integration

```python
from mcp_biomodelling_servers.handoff import load_handoff_manifest
from mcp_biomodelling_servers.import_lineage import (
    verify_source_ancestry, copy_source_ancestry,
)

# First apply the capture application's trust checks to this current manifest.
manifest = load_handoff_manifest(current_manifest_path, verify_artifacts=False)
imports = manifest.source_ancestry or []  # None occurs only for v1.
verify_source_ancestry(imports)  # No filesystem reads.
paths = copy_source_ancestry(imports, capture_directory / "source-ancestry")
# paths maps digest -> local Path. Retain unchanged source_ancestry in capture JSON.
```

`verify_artifacts=False` skips reads of current-stage artifact paths but still
validates embedded ancestry. Existing capture logic must separately verify and
copy current-stage artifacts and stage-parent snapshots under its trusted root.
The new helpers never read original ancestry paths, and copy uses only digest
filenames. Existing destination bytes must match; symlinks are rejected.
A capture remains portable by retaining the embedded list unchanged; copied
blobs are optional independently accessible evidence. Copying ancestry alone
does not relocate current-stage paths.

`capture_import(..., trusted_root=...)` is the initial live import API. It reads
and verifies source/parent files and immediately snapshots evidence. The optional
root confines initial source and historical reference reads; outside paths and
symlink escapes are rejected. The default retains existing explicitly supplied
local-file import behavior. Capture applications should use the read-free APIs
above, not re-run a live import on arbitrary history. Legacy reference-only
`SourceImport` records can be upgraded with `snapshot_source_import` while their
files remain available. Missing files or corrupt evidence are errors, never a
reason to replace ancestry with a standalone record.

New source-export sidecars and import-event records use version `2.0`, with v1
reading retained. New typed handoffs require embedded evidence, including for
standalone imports. At initial import, stale source bytes fail verification;
after successful import, changing or deleting historical files does not alter
its durable snapshot. Export verifies the snapshot rather than treating the
historical filename as the current model.

Bounds: at most 128 imports/blobs and 128 traversal steps per bundle, depth 16,
16 MiB total unique decoded evidence, 24 MiB serialized v2 manifest/record.
V1 manifests retain the 1 MiB limit. Cycles, missing blobs, invalid versions,
owner/membership mismatches, unrelated bundled bytes and digest/size changes
fail validation. Large or deeply nested histories fail explicitly.
