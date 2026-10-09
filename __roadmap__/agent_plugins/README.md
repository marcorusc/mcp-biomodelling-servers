# agent_plugins

## Context
Packages the four MCP servers of `marcorusc/mcp-biomodelling-servers` (NeKo, MaBoSS, PhysiCell, BioMASS) as four sibling agent plugins installable from GitHub by Claude Code, Codex and Agent Plugins 1.0 clients, generated with the `spawning-agent-plugins` skill. Ships as a draft PR from the fork `LittleCoinCoin/mcp-biomodelling-servers`; this `__roadmap__/` directory is review material and is removed before merge.

## Goal
Four plugins (`neko`, `maboss`, `physicell`, `biomass`) listed by a `marcorusc` marketplace in both ecosystems, pinned to the published PyPI release, with `scripts/check_release.py` enforcing their versions and pins.

## Pre-conditions
- [ ] Integration branch `feat/agent-plugins` starts at upstream `main` (`8551061`)
- [ ] Baseline `python -m build` + `scripts/check_release.py --tag v2.4.0` exits 0 on the untouched tree (measured)
- [ ] `mcp-biomodelling-servers==2.4.0` is published on PyPI

## Success Gates
- ✅ `python3 ~/.claude/plugins/cache/cracking-shells/spawning-agent-plugins/1.0.0/skills/spawning-agent-plugins/scripts/check_plugin.py --root . --spec __roadmap__/agent_plugins/plugin.spec.json` prints `ok`
- ✅ `claude plugin validate .` passes
- ✅ Clean `dist/` build passes `scripts/check_release.py --tag v2.4.0`
- ✅ All four plugins install from a scratch marketplace and their servers report `Connected`
- ✅ No change to `pyproject.toml`, `.github/workflows/`, or any `server.json`

## Status
```mermaid
graph TD
    plugin_manifests[Plugin manifests]:::done
    integrate[Release guard and docs]:::inprogress
    classDef done       fill:#166534,color:#bbf7d0
    classDef inprogress fill:#854d0e,color:#fef08a
    classDef planned    fill:#374151,color:#e5e7eb
    classDef amendment  fill:#1e3a5f,color:#bfdbfe
    classDef blocked    fill:#7f1d1d,color:#fecaca
```

## Nodes
| Node | Type | Status |
|:-----|:-----|:-------|
| `plugin_manifests.md` | 📄 Leaf Task | ✅ Done |
| `integrate/` | 📁 Directory | 🔄 In Progress |

## Amendment Log
| ID | Date | Source | Nodes Added | Rationale |
|:---|:-----|:-------|:------------|:----------|

## Progress
| Node | Branch | Commits | Notes |
|:-----|:-------|:--------|:------|
