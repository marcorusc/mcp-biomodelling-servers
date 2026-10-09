# Plugin manifests

**Goal**: Generate four sibling plugins (`neko`, `maboss`, `physicell`, `biomass`) and two `marcorusc` marketplaces from one spec with the spawning-agent-plugins skill.
**Pre-conditions**:
- [ ] Working tree on a leaf branch cut from `feat/agent-plugins`
- [ ] Skill present at `$SK = ~/.claude/plugins/cache/cracking-shells/spawning-agent-plugins/1.0.0/skills/spawning-agent-plugins`
**Success Gates**:
- ⬜ `python3 $SK/scripts/check_plugin.py --root . --spec __roadmap__/agent_plugins/plugin.spec.json` prints `ok`
- ⬜ `claude plugin validate .` passes and `claude plugin validate plugins/<x>/.claude-plugin/plugin.json` passes for all four
- ⬜ Every plugin's launch args equal the `value`s of its `<Server>/server.json` `runtimeArguments`, in both `mcp.json` files
- ⬜ No root plugin, no root `.mcp.json`, no `version` key on any marketplace entry
**References**: [skill SKILL.md](~/.claude/plugins/cache/cracking-shells/spawning-agent-plugins/1.0.0/skills/spawning-agent-plugins/SKILL.md) — workflow steps 2-4; `references/manifests.md#multi-plugin-repositories` — `plugins[]` mechanics

## Step 1: Write the spec and spawn the manifests
**Goal**: One spec records every decision; the generator writes every manifest from it.
**Implementation Logic**:
Write `__roadmap__/agent_plugins/plugin.spec.json` by hand (`init` does not support `plugins[]`).
Top level: `name` `mcp-biomodelling-servers`, `description`, `author {"name": "Marco Ruscone"}` (D7, never the git identity),
`homepage`/`repository` `https://github.com/marcorusc/mcp-biomodelling-servers`, `license` `MIT` (from pyproject; no LICENSE file is created),
`keywords`, `ecosystems` all three, `claude_marketplace.name` = `codex.marketplace_name` = `marcorusc` (D3).
`plugin_entries()` only inherits author/homepage/repository/license/ecosystems/keywords, so each of the four `plugins[]` entries carries its own
`dir` (`plugins/<x>`), `displayName`, `description` (from the README server table), `version_from` (`pyproject:pyproject.toml`),
`mcp {server: <x>, command: uvx, args: ["--from", "mcp-biomodelling-servers=={version}", "mcp-<x>-server"]}` (D8: equals server.json, no extras),
and a `codex` block (shortDescription, longDescription, category, capabilities, defaultPrompt of at most 50 chars).
Run `spawn --dry-run`, read it, then `spawn`. Do not hand-edit generated files; fix the spec and re-spawn with `--force` instead.
**Deliverables**: `__roadmap__/agent_plugins/plugin.spec.json`; `plugins/{neko,maboss,physicell,biomass}/{plugin.json,mcp.json,.claude-plugin/plugin.json,.claude-plugin/mcp.json}`; `.claude-plugin/marketplace.json`; `.agents/plugins/marketplace.json`
**Consistency Checks**: `python3 $SK/scripts/check_plugin.py --root . --spec __roadmap__/agent_plugins/plugin.spec.json && claude plugin validate . && for x in neko maboss physicell biomass; do claude plugin validate plugins/$x/.claude-plugin/plugin.json || exit 1; done` (expected: PASS)
**Commit**: `build(plugin): package the four servers as Claude Code, Codex and Agent Plugins 1.0 plugins`
