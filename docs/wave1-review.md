# Wave 1 review candidate

Historical scope: the NeKo CI candidate has since advanced to Wave 2. See
[the Wave 2 integration report](wave2-neko-integration.md) for the current pin
and acceptance results; the review scope below describes the original Wave 1 PR.

This PR prepares MCP 2.5.0 with persistent artifact storage, timezone-aware
metadata, verified Boolean ancestry v2 with v1 compatibility, and explicit
unexpected-error masking in the shared NeKo/MaBoSS/PhysiCell wrapper.

NeKo 1.10.2 and PhysiCell Settings 0.6.3 are companion release candidates.
CI installs their exact Git commits from `requirements-ci-wave1.txt` so tests
cannot silently use older published libraries. Normal package dependencies
require these fixed versions or later compatible releases.

Review and release the libraries before publishing MCP 2.5.0. No release tags
or package publication are part of this PR. Before release, confirm the library
artifacts are available and run CI against the published distributions as well.
The independent plugin packaging PR also changes release metadata; coordinate
its version pins and release checks if it merges before this PR.

Source ancestry does not transfer scientific approval. Existing v1 handoffs
remain readable, and missing legacy evidence is not reconstructed. ODE ancestry,
new PhysiCell mapping/cycle MCP tools and all other Wave 2 features are deferred.
