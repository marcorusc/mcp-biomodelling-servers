# Install verification

## Context
Runs last, on the merged result of every earlier leaf. Exercises real installs; produces the verification note quoted in the PR body.

## Goal
Claude Code installs every plugin and connects every server, including through the shipped `git-subdir` source form.

## Pre-conditions
- [ ] `release_guard` and `docs` merged on `feat/agent-plugins`

## Success Gates
- ✅ Verification note committed with commands and outcomes
- ✅ No scratch marketplace or plugin left installed afterwards

## Status
```mermaid
graph TD
    install_oracle[Install oracle]:::done
    classDef done       fill:#166534,color:#bbf7d0
    classDef inprogress fill:#854d0e,color:#fef08a
    classDef planned    fill:#374151,color:#e5e7eb
    classDef amendment  fill:#1e3a5f,color:#bfdbfe
    classDef blocked    fill:#7f1d1d,color:#fecaca
```

## Nodes
| Node | Type | Status |
|:-----|:-----|:-------|
| `install_oracle.md` | 📄 Leaf Task | ✅ Done |

## Amendment Log
| ID | Date | Source | Nodes Added | Rationale |
|:---|:-----|:-------|:------------|:----------|

## Progress
| Node | Branch | Commits | Notes |
|:-----|:-------|:--------|:------|
