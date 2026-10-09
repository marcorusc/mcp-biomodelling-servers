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
| Release guard | clean `dist/` build, `scripts/check_release.py --tag v2.4.0` | passes; 38 hand-edit mutations each fail it (adversarial review, two rounds) |
| Cleanup | `claude plugin list`, `claude plugin marketplace list` | no scratch plugin or marketplace left |

The real marketplace (`marcorusc/mcp-biomodelling-servers`) points its `git-subdir` sources at the upstream
default branch, so an install through it can only work once this lands on `main`.

## Codex and VS Code (manual, by Eliott)

Pending. A local Codex marketplace with `path` sources is prepared for the check.
