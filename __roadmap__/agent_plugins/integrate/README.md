# Release guard and docs

## Context
Runs after `plugin_manifests` exists. Two parallel leaves on disjoint files: `release_guard` wires the new manifests into the repo's existing release check, `docs` tells installers and maintainers. Produces what `verify/` exercises.

## Goal
Plugin versions and pins are enforced by `check_release.py`, and the README and agent instructions document the plugins.

## Pre-conditions
- [ ] `plugins/<x>/` manifests and both marketplaces are merged on `feat/agent-plugins`

## Success Gates
- ✅ `release_guard` and `docs` merged with `--no-ff`; `release_guard` passed its adversarial verifier
- ✅ README install strings, marketplace entry names and plugin names agree (coordinator integration review)

## Status
```mermaid
graph TD
    release_guard[Release guard]:::planned
    docs[Install docs]:::planned
    verify[Install verification]:::planned
    classDef done       fill:#166534,color:#bbf7d0
    classDef inprogress fill:#854d0e,color:#fef08a
    classDef planned    fill:#374151,color:#e5e7eb
    classDef amendment  fill:#1e3a5f,color:#bfdbfe
    classDef blocked    fill:#7f1d1d,color:#fecaca
```

## Nodes
| Node | Type | Status |
|:-----|:-----|:-------|
| `release_guard.md` | 📄 Leaf Task | ⬜ Planned |
| `docs.md` | 📄 Leaf Task | ⬜ Planned |
| `verify/` | 📁 Directory | ⬜ Planned |

## Amendment Log
| ID | Date | Source | Nodes Added | Rationale |
|:---|:-----|:-------|:------------|:----------|

## Progress
| Node | Branch | Commits | Notes |
|:-----|:-------|:--------|:------|
