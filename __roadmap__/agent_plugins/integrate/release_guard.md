# Release guard

**Goal**: Make `scripts/check_release.py` fail a release whose plugin manifests, pins or marketplaces disagree with the package version and the `server.json` launch.
**Pre-conditions**:
- [ ] `plugin_manifests` merged: `plugins/<x>/` and both marketplaces exist on `feat/agent-plugins`
- [ ] Baseline `check_release.py --tag v2.4.0` exits 0 on a clean `dist/` build (measured in Phase 0)
**Success Gates**:
- ⬜ `rm -rf dist && uv run --no-project --with build python -m build && uv run --no-project --python 3.12 python scripts/check_release.py --tag v2.4.0` exits 0
- ⬜ Each of three mutations on a scratch copy of the tree makes `check_release.py` exit non-zero: one `plugin.json` version bumped, one `mcp.json` pin changed, a `version` key added to one marketplace entry
- ⬜ Neither built archive contains a `plugins/` member
- ⬜ ADVERSARIAL REVIEW: a Sonnet verifier that did not write the change attacks it and its findings are resolved before merge
**References**: [scripts/check_release.py](../../../scripts/check_release.py) — existing `for server in SERVERS` loop and assert idiom

## Step 1: Extend the per-server loop and add marketplace asserts
**Goal**: Plugin manifests become version sites that `check_release.py` enforces exactly like `server.json`.
**Implementation Logic**:
Inside the existing `for server in SERVERS` loop, with `x = server.lower()` and the loop's existing `command`:
both `plugins/<x>/plugin.json` and `plugins/<x>/.claude-plugin/plugin.json` have `name == x` and `version == version`;
both `plugins/<x>/mcp.json` and `plugins/<x>/.claude-plugin/mcp.json` have `mcpServers[x]` with `command == "uvx"` and
`args == ["--from", f"{project['name']}=={version}", command]` (the exact `value`s of the server.json runtimeArguments the loop already asserts);
the agent-plugins `mcp.json` server also has `type == "stdio"`.
After the loop: `.claude-plugin/marketplace.json` and `.agents/plugins/marketplace.json` each list exactly the four names and no entry carries `version`.
All of this runs before the archive section, so a missing `dist/` cannot hide it. Match the script's idiom: bare `assert ..., server`,
no new helpers unless they remove real repetition, and extend the final print line to mention the plugins.
**Deliverables**: `scripts/check_release.py` (modified `main()`)
**Consistency Checks**: `rm -rf dist && uv run --no-project --with build python -m build && uv run --no-project --python 3.12 python scripts/check_release.py --tag v2.4.0` (expected: PASS)
**Commit**: `build(release): enforce plugin manifest versions and pins in check_release`
