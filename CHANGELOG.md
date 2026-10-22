# Changelog

## 2.5.1 (unreleased)

A patch release: documentation corrections, the agent plugins in a tagged
release, and startup fixes. Tools, schemas, handoff versions, scientific
defaults and engine behaviour are unchanged from 2.5.0.

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
