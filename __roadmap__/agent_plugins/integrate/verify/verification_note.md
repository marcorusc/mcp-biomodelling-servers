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
In the Claude marketplace that pins every Claude Code install to upstream `main`, even when the
marketplace was added from a fork branch or a local path. The Codex marketplace has the same property:
a Codex install from the fork branch failed because its `git-subdir` sources fetch upstream.

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
| VS Code | local folder with only `.agents/plugins/marketplace.json` and plugins at `<x>/` | "no plugin or marketplace manifest" |
| VS Code | `chat.plugins.marketplaces` = `LittleCoinCoin/mcp-biomodelling-servers#feat/agent-plugins` (after the deviation) | "does not appear to be a valid plugin marketplace" (cause not established, see below) |
| VS Code | local folder = this worktree (after the deviation) | all four plugins discovered and installable |

VS Code's marketplace format is Claude Code's. The VS Code docs
(code.visualstudio.com/docs/agent-customization/agent-plugins#_configure-plugin-marketplaces) refer to the
Claude Code plugin marketplace documentation for the marketplace schema, use `anthropics/claude-code` as
the example marketplace, and register extra marketplaces through `extraKnownMarketplaces` in
`.claude/settings.json`. So VS Code reads `.claude-plugin/marketplace.json`, which is consistent with the
rejected Codex-only folder above. The docs also say relative-path plugin entries are verified against the
repository, the resolved revision and the path, which is the shape the relative-source change ships.
The local worktree test does not discriminate on its own, because a local plugin location
(`chat.pluginLocations`) can find `plugins/<x>/plugin.json` without a marketplace file.

The same docs section says `#<ref>` selects a branch, tag or commit in `chat.plugins.marketplaces`, so the
`#feat/agent-plugins` failure is not explained by refs being unsupported. The slash in the branch name is
one unverified possibility.

VS Code sends a `plugin.json` that carries the Agent Plugins `$schema` to a loader that does not
substitute `${PLUGIN_ROOT}` or set a working directory (microsoft/vscode #303219, #305310).
`sysbio-curie/MCP_Hackaton` drops `$schema` for that reason. These plugins keep it: their launch,
`uvx --from mcp-biomodelling-servers==<v> mcp-<x>-server`, uses no placeholder and does not depend on the
working directory.


## Remote installs from the fork's `main`

`feat/agent-plugins` was fast-forwarded onto `LittleCoinCoin/mcp-biomodelling-servers` `main`, which is
the head of the PR, so every host could install over git with no ref, as users will after the merge.
For the Codex test only, commit `2009a39` pointed the Codex `git-subdir` URLs at the fork, and the next
commit reverts it. `check_release.py` rejects the test state, and passes again after the revert.

| Host | Source | Outcome |
|---|---|---|
| Claude Code (agent) | `claude plugin marketplace add LittleCoinCoin/mcp-biomodelling-servers`, install ×4 | all four installed and `✔ Connected`; removed afterwards |
| Codex (Eliott) | marketplace from the fork URL, `main` | installs and runs |
| VS Code (Eliott) | git URL of the fork | installs and runs, so the `$schema` loader issue above does not affect these plugins |
