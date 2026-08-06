# AI Coding Agent Instructions for MCP Bio-Modelling Servers

## Repository Purpose

This repository packages three stateful Model Context Protocol servers for
mechanistic and systems-biology modelling:

- **NeKo** builds, curates, analyzes, and exports signalling networks.
- **MaBoSS** configures and simulates stochastic Boolean models.
- **PhysiCell** builds, loads, validates, edits, and exports PhysiCell and
  PhysiBoSS configuration files. It does not run PhysiCell tissue simulations.

The servers run as independent stdio processes but share versioned contracts
for sessions, artifacts, structured results, and model handoffs. The supported
cross-server workflow is NeKo -> MaBoSS -> PhysiCell.

Current release metadata is defined in `pyproject.toml`. Do not hard-code a
different package or server version. The three `server.json` Registry manifests
and `mcp_biomodelling_servers/__init__.py` must remain synchronized with it.

## MCP v2 Architecture

All servers use the stable MCP Python SDK 2.x API:

```python
from mcp.server.mcpserver import Context, MCPServer

mcp = MCPServer(
    "ServerName",
    title="Client-facing title",
    description="Client-facing description",
    instructions="Concise workflow guidance",
    version=__version__,
)
```

Do not reintroduce `FastMCP`, deprecated MCP APIs, or an MCP 1.x dependency.
The project requires `mcp>=2,<3`, and MCP deprecation warnings fail the test
suite.

Each server publishes:

- constrained tools with explicit `ToolAnnotations`;
- read-only resources, including session-scoped URI templates;
- one workflow prompt and one complete agent-manual resource;
- concise initialization instructions;
- structured scientific results where downstream applications need typed data.

As of the 2.0.0 release checkpoint, the public surface contains 89 tools:
24 MaBoSS, 31 NeKo, and 34 PhysiCell. If the tool surface changes, update the
relevant server README/manual and the cross-server schema tests together.

## Session State and Concurrency

All three servers use session-based mutable state. Never add process-global
model state as a shortcut.

- `MaBoSS/session_manager.py` owns MaBoSS models, results, locks, leases, and
  lifecycle state.
- `NeKo/session_manager.py` owns networks, history settings, edge caches,
  locks, leases, and lifecycle state.
- `PhysiCell/session_manager.py` owns `PhysiCellConfig`, workflow metadata,
  MaBoSS handoff context, locks, leases, and lifecycle state.

MCP SDK v2 can execute synchronous handlers concurrently in worker threads.
Every model-backed tool, resource, and artifact operation must therefore use
the server's session lease/operation boundary. Required behavior:

1. Operations on the same session serialize.
2. Different sessions can continue concurrently.
3. First-use/default-session creation is atomic.
4. Active sessions are not evicted or cleaned up.
5. Deletion retires a session, rejects new work, waits for admitted work, and
   cleans artifacts only after the session drains.
6. Candidate models/configurations are built and validated locally, then
   published atomically so a failed update preserves the previous state.

Do not hold a blocking thread lock across an awaited progress update. For
async tools, move blocking scientific work through AnyIO's worker pool and
keep state publication inside the session boundary.

NeKo history, edge-cache invalidation, and process-wide stdout capture require
their existing synchronization. PhysiCell signal/behavior expansion also uses
an upstream process-global context and must retain its separate serialization.

## Tool Errors and Scientific Results

Use ordinary Python exceptions for caller-correctable high-level tool
failures. MCP SDK v2 converts them into tool results with `is_error=True`,
which clients and models can inspect. Do not replace these with successful
Markdown error strings, and do not use `MCPError` for normal tool failures.

Use `ResourceNotFoundError` for missing or invalid resource targets.

A scientifically valid empty or negative result is not necessarily an
execution failure. Examples that remain successful include:

- an empty path or query result;
- an empty simulation result;
- a validation result reporting `valid=false` for an existing XML file;
- an explicitly empty configured component collection.

Preserve this distinction when changing handlers or tests.

## Input Schemas and Safety Annotations

Public tool arguments must be representable in JSON and accurately constrained
by the generated MCP schema.

- Use `Literal[...]` or validated aliases for stable finite choices.
- Use Pydantic bounds and finite-number validation for biological and runtime
  parameters with stable constraints.
- Reject empty or whitespace-only identifiers and paths.
- Keep backend-discovered or scientifically open-ended values flexible.
- Keep handler-only values such as `ctx`, leased sessions, and model objects
  out of public schemas.
- Preserve compatibility deliberately; do not replace an established wire
  signature without a separately approved migration path.

Every tool must explicitly publish all four annotation hints: read-only,
destructive, idempotent, and open-world. Read-only tools must also be
non-destructive and idempotent. Base annotations on actual side effects, not
the tool's name.

MaBoSS joint initial-state probabilities must retain the JSON-native
state/probability record form while preserving the documented compatible
legacy inputs. PhysiCell cell, substrate, and PhysiBoSS update tools use flat
optional patch semantics: omission preserves the current value, and an
identifier-only call is invalid.

## Structured Outputs

Use the strict Pydantic contracts in:

- `mcp_biomodelling_servers/structured_outputs.py` for shared results;
- `MaBoSS/scientific_outputs.py` for Boolean simulation results;
- `NeKo/src/structured_outputs.py` and `NeKo/src/history.py` for network and
  history results;
- `PhysiCell/physicell_outputs.py` for configuration and workflow results;
- `mcp_biomodelling_servers/handoff.py` for cross-server handoffs.

For migrated tools, return validated structured content while retaining a
concise, useful human-readable MCP content block. Preserve native MCP media
content such as MaBoSS trajectory PNGs. Numeric scientific values belong in
typed fields rather than Markdown-only tables.

Avoid generic top-level module names that collide when all three source servers
are imported into one Python process. The dedicated cross-server schema tests
protect this shared-process boundary.

## Artifacts and File Safety

Generated files belong under a session sandbox:

```text
<server directory>/artifacts/<session_id>/
```

Use the shared artifact helpers and the session lease boundary for generation,
listing, and cleanup. Never write caller-selected outputs to the repository
root or an arbitrary current working directory.

`safe_artifact_path()` confines a basename by stripping directory components.
When invalid input must be visible to the caller, validate plain basenames,
suffixes, separators, traversal, absolute paths, whitespace, and null bytes
before calling it. PhysiCell XML/CSV exports already follow this stricter rule.

Artifact listings and results should expose complete session provenance and
typed file metadata. Do not silently overwrite retained handoff artifacts.

The following are generated local state, not source:

- `NeKo/artifacts/`, `NeKo/exports/`, and `NeKo/pypath_log/`;
- `MaBoSS/artifacts/`;
- `PhysiCell/artifacts/`;
- caches, logs, environment files, and `dist/` contents.

Do not stage generated artifacts or environment files.

## Typed Cross-Server Handoffs

Prefer the versioned handoff workflow over manually reconstructing context from
unrelated paths:

1. NeKo exports a history-backed Boolean model and NeKo-to-MaBoSS manifest.
2. MaBoSS verifies and imports that manifest, runs an explicitly configured
   simulation, and exports a MaBoSS-to-PhysiCell manifest.
3. PhysiCell verifies and copies the complete lineage before attaching the
   model to an existing target cell type.

Handoff manifests record package versions, sessions, biological context,
network/output nodes, settings, summaries, artifact roles, sizes, and SHA-256
digests. Preserve strict validation, create-if-absent publication, source
revalidation, rollback, and atomic session replacement.

Standalone MaBoSS models may enter PhysiCell without NeKo ancestry, but must
provide biological context. Do not infer PhysiBoSS timing, signal/node links,
behavior mappings, or mutations from a MaBoSS result; these remain explicit
scientific choices.

## Server-Specific Boundaries

### MaBoSS

- Inspect nodes before choosing initial states, mutations, or output nodes.
- Require a non-empty output-node selection before simulation or handoff
  export to avoid accidental full-state-space work.
- Preserve progress reporting for blocking simulation work.
- Return trajectory plots as complete native MCP image content with structured
  artifact metadata.

### NeKo

- Create networks from validated database choices or explicit source files.
- Retain branching history and use exact state IDs for checkout/comparison.
- Run connection strategies through history-aware `Network` methods.
- Preserve the unbounded history default unless a session limit is explicitly
  configured.
- Graphviz's external `dot` executable is required for HTML/SVG history
  rendering.
- Database-backed tools are open-world operations and may require network
  access.

### PhysiCell

- Treat the server as a configuration builder, not a PhysiCell execution or
  tissue-output analysis service.
- New-domain construction and existing-XML loading are distinct workflows.
- Configuration replacement is destructive and must reset stale derived
  workflow metadata while preserving valid session identity/context.
- Preserve flat atomic patch behavior for repeated cell, substrate, and
  PhysiBoSS revisions.
- Export Cell Rules CSV before XML when XML must reference the generated rules.

## Prompts, Manuals, and Documentation

The complete workflow manuals are exposed through:

- `docs://maboss/agent_manual` and `maboss_workflow_prompt`;
- `docs://neko/agent_manual` and `neko_workflow_prompt`;
- `docs://physicell/agent_manual` and `physicell_workflow_prompt`.

Keep initialization instructions concise. Put complete workflows in the shared
manual source used by prompts/resources/help tools, and keep the corresponding
server README accurate. Do not duplicate large scientific tool results in
resources.

## Dependencies and Packaging

The supported interpreter range is Python 3.10-3.14. Keep emitted syntax
compatible with Python 3.10 while retaining the Python 3.14/pandas 3 lane.

Important declared ranges include:

- `mcp>=2,<3`;
- `pydantic>=2.12`;
- `pandas>=2.2,<4`;
- `nekomata>=1.9.0,<2`;
- `maboss>=0.8.15,<0.9`;
- `physicell-settings>=0.6.2,<0.7`.

`requests` and `tabulate` are direct runtime dependencies. Do not reintroduce
the removed direct `paramiko`, `fastmcp`, or old PyPath dependency assumptions.
Conda is optional; pip and uvx installs must work from the published package.

Hatch maps the three source server directories under the installed
`mcp_biomodelling_servers` namespace. Ordinary wheel and source-distribution
builds are supported. Editable `uv run` installation is not currently
supported by this prefix-changing source layout; do not mix a package-layout
rewrite into an unrelated change.

The root and installed copies of `artifact_manager.py` must remain identical
until that packaging duplication is deliberately redesigned.

## Verification Workflow

Run checks in proportion to the change. For public tool, session, or scientific
behavior changes, use focused tests first and then the complete suite.

Reference commands from CI are:

```bash
python -m pytest tests/ -q

python -m ruff check \
  artifact_manager.py \
  mcp_biomodelling_servers/artifact_manager.py \
  mcp_biomodelling_servers/handoff.py \
  mcp_biomodelling_servers/structured_outputs.py \
  MaBoSS/scientific_outputs.py MaBoSS/session_manager.py \
  NeKo/src/history.py NeKo/src/structured_outputs.py NeKo/session_manager.py \
  PhysiCell/physicell_guidance.py PhysiCell/physicell_outputs.py \
  PhysiCell/session_manager.py tests/

python -m ruff check --isolated --select E9,F63,F7,F82 \
  MaBoSS/server.py NeKo/server.py PhysiCell/server.py

python -m mypy --ignore-missing-imports --strict-optional \
  artifact_manager.py \
  mcp_biomodelling_servers/handoff.py \
  mcp_biomodelling_servers/structured_outputs.py \
  MaBoSS/scientific_outputs.py MaBoSS/session_manager.py \
  NeKo/src/history.py NeKo/src/structured_outputs.py NeKo/session_manager.py \
  PhysiCell/physicell_guidance.py PhysiCell/physicell_outputs.py \
  PhysiCell/session_manager.py

python -m build
python -m pip check
git diff --check
```

The full repository does not yet pass the configured Ruff rule set. CI applies
the full policy to the modules above and fatal syntax checks to the large
server modules. Treat repository-wide lint cleanup as a separate checkpoint;
do not fold hundreds of mechanical findings into functional work.

Tests intentionally use protocol doubles for selected NeKo and PhysiCell
paths. Keep `tests/test_runtime_compatibility.py` as the real-package boundary,
and retain the Python 3.10/pandas 2, Python 3.12 reference, and Python
3.14/pandas 3 CI coverage.

## Change Discipline

- Inspect the current public schema before changing a tool signature.
- Preserve same-session locking, error semantics, annotations, structured
  output, documentation, and artifact behavior together.
- Add deterministic concurrency tests using events/barriers, not timing-only
  sleeps.
- Review distribution contents for artifacts, logs, environment files,
  secrets, and stale release archives.
- Do not create legal text or a `LICENSE` ownership line without repository
  owner confirmation.
- Do not stage or modify unrelated working-tree files.

`MCP_V2_UPGRADE_PLAN.md` is an owner-managed local ledger and must remain
ignored and untracked. `agents_context/` is also owner-managed local context
and must remain untracked. Preserve both unless the repository owner explicitly
changes that policy.
