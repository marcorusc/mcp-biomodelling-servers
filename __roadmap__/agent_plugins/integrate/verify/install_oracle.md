# Install oracle

**Goal**: Prove with Claude Code that every plugin installs and its server connects, including through the shipped `git-subdir` source form.
**Pre-conditions**:
- [ ] `release_guard` and `docs` merged on `feat/agent-plugins`
- [ ] `feat/agent-plugins` pushed to the fork remote (`fork`), never to `origin`
**Success Gates**:
- ⬜ `claude --plugin-dir plugins/<x> mcp list` reports the `<x>` server `Connected` for all four
- ⬜ All four install from a scratch marketplace of local copies and appear in `claude plugin list`
- ⬜ At least one plugin installs from a scratch marketplace whose entry is the committed `git-subdir` source pointed at the fork with `ref` = `feat/agent-plugins`, and its server connects
- ⬜ After cleanup, `claude plugin list` and `claude plugin marketplace list` show no scratch marketplace or plugin
**References**: `$SK/SKILL.md` step 4 — scratch-marketplace install oracle

## Step 1: Local and git-subdir install checks
**Goal**: Exercise the installs a user will run, then leave Claude Code as found.
**Implementation Logic**:
Local: in the scratchpad, build a marketplace with a throwaway name whose plugin sources are copies of `plugins/<x>` (no `.git`), add it, install all four,
list, and run `claude --plugin-dir plugins/<x> mcp list` per plugin. git-subdir: a second scratch marketplace whose entries copy the committed
`source` objects with `url` swapped to the fork and `ref` added, so the exact `path: "./plugins/<x>"` form is exercised. If that form fails,
record it as a spec defect in the roadmap trail instead of hand-patching manifests. Prepare a scratch Codex marketplace with local `path` sources
and a checklist for Eliott's manual Codex and VS Code check. Remove every scratch marketplace and plugin afterwards.
Write the commands and outcomes to `__roadmap__/agent_plugins/integrate/verify/verification_note.md` for the PR body.
**Deliverables**: `__roadmap__/agent_plugins/integrate/verify/verification_note.md`
**Consistency Checks**: `test -s __roadmap__/agent_plugins/integrate/verify/verification_note.md && ! claude plugin marketplace list | grep -q scratch` (expected: PASS)
**Commit**: `docs(roadmap): record plugin install verification`
