# Plugin install verification

Run 2026-10-09 on macOS with Claude Code 2.1.273 and uv 0.9.18, against the plugin trees merged at
`163c481`. Later merges did not touch `plugins/`, `.claude-plugin/` or `.agents/`
(`git diff --stat 163c481 HEAD -- plugins .claude-plugin .agents` is empty).

## Claude Code (verified)

| Check | Command | Outcome |
|---|---|---|
| Drift guards | `check_plugin.py --root . --spec __roadmap__/agent_plugins/plugin.spec.json` | `ok: plugin structure is consistent` |
| Manifests | `claude plugin validate .` and `claude plugin validate plugins/<x>/.claude-plugin/plugin.json` ×4 | all pass |
| Server launch | `claude --plugin-dir plugins/<x> mcp list` ×4 | `plugin:<x>:<x>: uvx --from mcp-biomodelling-servers==2.4.0 mcp-<x>-server - ✔ Connected` for neko, maboss, physicell, biomass |
| Marketplace install (local copies) | scratch marketplace, `claude plugin install <x>@scratch-biomod-local` ×4, `claude mcp list` | all four installed (version 2.4.0) and `✔ Connected` |
| Marketplace install (shipped `git-subdir` form) | scratch marketplace copying the committed sources with `url` = the fork and `ref` = `feat/agent-plugins`; installed `neko` and `biomass` | both installed from GitHub and `✔ Connected`, so `path: "./plugins/<x>"` is accepted |
| Release guard | clean `dist/` build, `scripts/check_release.py --tag v2.4.0` | passes; 45 hand-edit mutations each fail it (adversarial review, two rounds) |
| Real marketplace from the fork branch (after the relative-source change) | `claude plugin marketplace add LittleCoinCoin/mcp-biomodelling-servers#feat/agent-plugins`, `claude plugin install <x>@marcorusc` ×4, `claude mcp list` | all four installed and `✔ Connected`; removed afterwards |
| Cleanup | `claude plugin list`, `claude plugin marketplace list` | no scratch plugin or marketplace left |

Claude Code (and VS Code) installs resolve relative to the marketplace checkout. Codex's `git-subdir` sources
fetch upstream, so a remote Codex install only works once this lands on `main`.

## Codex and VS Code (manual, by Eliott)

Pending. A local Codex marketplace with `path` sources is prepared for the check.

## Deviation from the generator: relative Claude marketplace sources

The skill writes `git-subdir` sources with the upstream URL for sibling plugins in both marketplaces.
In the Claude marketplace that pins every install to upstream `main`, even when the marketplace was
added from a fork branch or a local path. Two things fail as a result:

- a Codex install from the fork branch (Codex reads its own `git-subdir` sources, which also fetch upstream);
- a VS Code install of a multi-plugin repo. The Agent Plugins 1.0 spec defines no marketplace format,
  and VS Code's docs defer to Claude Code's marketplace format without naming the file. Observed here: a
  folder with only `.agents/plugins/marketplace.json` is rejected ("no plugin or marketplace manifest"),
  while the worktree, which also has `.claude-plugin/marketplace.json`, installs all four plugins. So
  VS Code reads the Claude marketplace file and resolves its sources.

`.claude-plugin/marketplace.json` now uses relative sources (`"./plugins/<x>"`). These resolve inside
whichever checkout the marketplace came from. This is the shape of `sysbio-curie/MCP_Hackaton`, which
installs in Claude Code, Codex and VS Code. The Codex marketplace keeps `git-subdir`, which is what the
skill's checker requires for siblings and what MCP_Hackaton ships. `check_plugin.py` accepts both
shapes, and `check_release.py` asserts each file's own shape. A later `spawn` never overwrites existing
marketplace entries, so regenerating does not undo this.

## Codex and VS Code (manual, by Eliott)

| Host | Source | Outcome |
|---|---|---|
| Codex GUI | fork URL + `feat/agent-plugins` ref | marketplace added, four plugins listed; install failed for all four (the sources fetch upstream `main`, which has no `plugins/` yet) |
| Codex GUI | local marketplace with `path` sources | marketplace, discovery and install work; the agent found the NeKo tools and used them |
| VS Code | git URL | untestable before merge: no ref field, defaults to `main` |
| VS Code | local folder with only `.agents/plugins/marketplace.json` | "no plugin or marketplace manifest" (VS Code reads the Claude marketplace file) |
| VS Code | `chat.plugins.marketplaces` = `LittleCoinCoin/mcp-biomodelling-servers#feat/agent-plugins` (after the deviation) | "does not appear to be a valid plugin marketplace": the `#ref` looks ignored and the fork's `main` has no marketplace yet. `sysbio-curie/MCP_Hackaton`, which has the same layout, installs remotely from `main` |
| VS Code | local folder = this worktree (after the deviation) | all four plugins discovered and installable |

VS Code sends a `plugin.json` that carries the Agent Plugins `$schema` to a loader that does not
substitute `${PLUGIN_ROOT}` or set a working directory (microsoft/vscode #303219, #305310).
`sysbio-curie/MCP_Hackaton` drops `$schema` for that reason. These plugins keep it: their launch,
`uvx --from mcp-biomodelling-servers==<v> mcp-<x>-server`, uses no placeholder and does not depend on the
working directory.

