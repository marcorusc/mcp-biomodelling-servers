# Install docs

**Goal**: Tell installers how to install each plugin and tell maintainers where the plugin version sites are.
**Pre-conditions**:
- [ ] `plugin_manifests` merged: plugin names, dirs and marketplace names are fixed
**Success Gates**:
- ⬜ Every `<x>@marcorusc` string in `README.md` names an entry of `.claude-plugin/marketplace.json`, and every entry appears in the README
- ⬜ `git diff --check` is clean
- ⬜ No install syntax in the README that the skill's `install-snippet` output or the marketplace files do not support
**References**: `$SK/references/traps.md#namespaces` — install-string shape; `python3 $SK/scripts/spawn_plugin.py install-snippet --spec __roadmap__/agent_plugins/plugin.spec.json` — snippet to adapt

## Step 1: README section and agent-instructions note
**Goal**: One README subsection for all four plugins, one maintainer paragraph.
**Implementation Logic**:
README: add an "Install as an agent plugin" subsection under Installation. It says why (one install per tool, pinned to the published release,
no hand-written client config), lists the four plugins in a table (plugin, server, install string),
and gives Claude Code, Codex and Agent Plugins 1.0 / VS Code instructions adapted from `install-snippet` (which prints the top-level identity only;
adapt it per plugin, do not invent syntax). Keep the existing manual `mcp.json` configuration section as the alternative.
`.github/copilot-instructions.md`: one paragraph under "Dependencies and Packaging": `plugins/` holds generated plugin manifests;
a release moves their `version` and both `mcp.json` pins together with every `server.json`, and `scripts/check_release.py` enforces it.
Match the existing prose register of each file.
**Deliverables**: `README.md` (new subsection); `.github/copilot-instructions.md` (one paragraph)
**Consistency Checks**: `for n in $(jq -r '.plugins[].name' .claude-plugin/marketplace.json); do grep -q "$n@marcorusc" README.md || exit 1; done && git diff --check` (expected: PASS)
**Commit**: `docs(readme): document agent plugin installation`
