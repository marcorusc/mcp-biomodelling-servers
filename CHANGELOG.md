# Changelog

## Unreleased

### Fixed

- Typed handoffs now work on hosts that give each MCP server a private
  filesystem, such as Claude Science (every server runs in its own sandbox).
  Before, every `import_*_handoff` call failed with "Handoff manifest does not
  exist", because the consumer could not read the producer's files.

### Added

- `export_handoff_bundle(manifest_path)` on NeKo and MaBoSS (read-only). It
  returns one JSON object holding a handoff manifest and every file it
  references, transitively, gzip-compressed and keyed by SHA-256. It reads
  only files on disk, so it works after the exporting session has ended.
- An optional `handoff_bundle` argument on MaBoSS and BioMASS
  `import_neko_handoff` and PhysiCell `import_maboss_handoff`, used instead of
  `manifest_path`. Pass exactly one of the two. The importer verifies every
  digest, stores the files under `<artifact root>/handoff-blobs/<sha256>/`, and
  runs the normal import. Manifests are not rewritten: digests, recorded paths,
  lineage and handoff versions are the same as with a path import.
- Missing-manifest and missing-artifact errors now name the bundle route.
- `bnet_to_bnd_and_cfg`, `build_simulation` and MaBoSS `import_neko_handoff`
  warn when the loaded settings make a heavy workload (nodes × `max_time` ×
  `sample_count` / `thread_count` above 10⁸), such as pyMaBoSS's BNET defaults
  (`max_time = 1000`, `sample_count = 10000`) on a mid-sized network. The
  warning names `update_maboss_parameters(parameters={...})` as the fix,
  because such runs can outlast a client's tool timeout.
- `sif_text` on NeKo `create_network` and `bnet_text` on MaBoSS
  `bnet_to_bnd_and_cfg`: complete file content, used instead of `sif_file` or
  `bnet_path` when the server cannot read the caller's files. The text is
  stored verbatim under a content-addressed name in the session artifact
  directory and recorded as the import source. Missing-path errors name these
  arguments.
- `export_biomass_handoff` text now includes the manifest path, so clients
  that only read text content can pass it on. Both NeKo handoff exports name
  `export_handoff_bundle` in their next-step hint.
- The NeKo "generated multiple BNET models" error now explains the cause: one
  model per sign assignment of bimodal interactions (2^n). It lists those
  interactions and names `edit_interactions` and `remove_bimodal_interactions`.
  The export still refuses to pick a sign arbitrarily.

### Changed

- `manifest_path` on the three import tools and `bnet_path` on MaBoSS
  `bnet_to_bnd_and_cfg` are no longer required in the input schema, since
  `handoff_bundle` or `bnet_text` can replace them. Path-based calls behave as
  before.
- NeKo guidance and the BNET export error now recommend resolving bimodal
  interactions with `edit_interactions` first. `remove_bimodal_interactions`
  is described as deleting edges and their references, which can disconnect
  the network.

## 2.5.1 (2026-10-22)

A patch release: documentation corrections, the agent plugins in a tagged
release, startup fixes, and fixes for two 2.5.0 bugs (BioMASS artifact
integrity under the default artifact root, and MaBoSS plotting on machines
with a display). Tools, schemas, handoff versions, scientific defaults and
engine behaviour are unchanged from 2.5.0.

### Added

- `--help` and `--version` on all four console entry points
  (`mcp-neko-server`, `mcp-maboss-server`, `mcp-physicell-server`,
  `mcp-biomass-server`). Both print and exit without starting the server or
  importing a modelling backend. Without them, a server starts exactly as
  before, ignoring any other arguments.
- The Claude Code, Codex and Agent Plugins 1.0 plugins under `plugins/` are
  part of a tagged release for the first time, pinned to
  `mcp-biomodelling-servers==2.5.1`.

### Fixed

- NeKo no longer contacts omnipathdb.org while starting. Importing `nekomata`
  imports the `omnipath` client, which downloads its query definitions at import
  time; offline, this logged a `urllib3` `TimeoutError` traceback. The server
  now imports `nekomata` on the first tool call that needs it, so `initialize`
  and `tools/list` are quiet and need no network. The first such call takes a
  few seconds longer.
- NeKo no longer creates `pypath_log/` in the working directory (for plugins,
  that directory is the plugin cache). The `mcp-neko-server` entry point now
  sends pypath's session log to `<artifact root>/logs/pypath/`, one file per
  process. Set `PYPATH_LOG` to a file path to choose another location.
- BioMASS lists, seals, exports and cleans its artifacts again under the
  default artifact root. 2.5.0 moved that root to
  `~/.cache/mcp-biomodelling/artifacts/`, and BioMASS skipped every file whose
  absolute path contained a `.cache` directory, a filter meant only for the
  matplotlib cache inside each artifact tree. Under the default root, tools
  therefore reported no files, revision integrity inventories were empty (so
  tampering went undetected), export bundles held no model artifacts and
  cleanup reported zero removals. Revisions generated by 2.5.0 under the
  default root carry an empty inventory and now fail verification with a
  request to regenerate the model.
- The MaBoSS server always renders plots with matplotlib's off-screen `Agg`
  backend. When a display was available, matplotlib could pick a GUI backend,
  and `visualize_network_trajectories`, which draws in a worker thread, could
  abort the whole server process (observed with TkAgg under WSLg).
- MaBoSS `run_simulation` now points clients to
  `get_simulation_result(run_id=...)`, using the `run_id` it returns, instead of
  the `maboss://session/{id}/result` resource, which clients without MCP
  resource support (such as background subagents) cannot read. The
  `load_simulation` and NeKo `compare_network_states` descriptions now mention
  tool alternatives too.

### Documentation

- README: rewrote the installation section for 2.5.x (it still recommended
  2.3.0 with `nekomata` 1.10.1), documented that `cmaboss` is only published
  as `1.0.0b*` pre-releases, explained that a plugin's server is shared by the
  whole client session, and listed the plugin manifests in the release steps.
- Corrected the handoff and source-lineage schema versions in the README and
  `docs/boolean-lineage-v2.md` to match what the code writes.
- Marked `docs/wave2-server-features.md` as a dated record and corrected its
  CI-pin and runtime-dependency statements.

This file starts at 2.5.1; for earlier releases, see the `v*` Git tags.
