import io
import copy
import sys
import os
import glob
import json
import logging
import re
import tempfile
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal, Optional, List
from pydantic import BeforeValidator, Field

import anyio
from neko.core.network import Network
from neko._outputs.exports import Exports
from neko.inputs import Universe, signor
from neko.core.tools import is_connected

from utils import *
from session_manager import session_manager, ensure_session, normalize_verbosity, DEFAULT_VERBOSITY

# Make the repo root importable so we can use the shared artifact_manager
sys.path.insert(0, str(Path(__file__).parent.parent))
from artifact_manager import safe_artifact_path, list_artifacts, clean_artifacts, write_session_meta, list_artifact_sessions as _list_artifact_sessions_on_disk
from mcp_biomodelling_servers import __version__
from mcp_biomodelling_servers.handoff import (
    HandoffNetwork,
    HandoffPackage,
    HandoffProvenance,
    NeKoHandoffExportResult,
    NeKoToMaBoSSHandoffManifest,
    bnet_node_names,
    handoff_artifact,
    write_handoff_manifest,
)

from src.helpers import (
    E_NO_NET, SUMMARY_HINT, _SERVER_ROOT,
    _short_table, _export_dir, _session_network, _invalidate,
    _compute_components, requires_network, session_locked,
    sanitize_bnet_file, _get_translators
)
from src.history import (
    HistoryNavigationResult,
    HistoryRetentionResult,
    NetworkHistorySummary,
    NetworkStateComparison,
    compare_history_states,
    state_ids,
    summarize_history,
)
from src.structured_outputs import (
    NeKoComponentRecord,
    NeKoConnectionPreviewResult,
    NeKoConnectivityResult,
    NeKoConnectorSimulation,
    NeKoHubCandidate,
    NeKoInteractionFilterResult,
    NeKoInteractionRecord,
    NeKoNetworkInventoryResult,
    NeKoNetworkStatusResult,
    NeKoNodeRecord,
    NeKoPathSearchResult,
    NeKoReferencedInteractionRecord,
    NeKoReferenceQueryResult,
)

import pandas as pd


from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError
from mcp.types import CallToolResult, ToolAnnotations
from mcp_biomodelling_servers.structured_outputs import (
    ArtifactSessionSummary,
    NeKoArtifactCleanupResult,
    NeKoArtifactFileListResult,
    NeKoArtifactSessionListResult,
    NeKoNetworkExportResult,
    NeKoSessionListResult,
    NeKoSessionSummary,
    artifact_file_summary,
    structured_report,
)

logger = logging.getLogger(__name__)
_stdout_capture_lock = Lock()

NEKO_SERVER_INSTRUCTIONS = (
    "Create a session before building a signalling network, and pass "
    "`session_id` explicitly when working with multiple networks. Run "
    "`preview_connection_impact()` before applying expensive connection "
    "strategies, inspect network history after topology changes, and prefer "
    "`verbosity='summary'` during iterative work. Export with `format='bnet'` "
    "for a standalone Boolean file, or use `export_neko_handoff` to preserve "
    "typed MaBoSS provenance. Read `docs://neko/agent_manual` or use "
    "`neko_workflow_prompt` for the complete workflow."
)

mcp = MCPServer(
    "NeKo",
    title="NeKo Signalling Network Builder",
    description=(
        "Build and analyze signalling networks from biological interaction databases."
    ),
    instructions=NEKO_SERVER_INSTRUCTIONS,
    version=__version__,
)

NonEmptyString = Annotated[
    str,
    Field(
        min_length=1,
        pattern=r".*\S.*",
        description="A non-empty string containing at least one non-whitespace character.",
    ),
]
NonEmptyStringList = Annotated[List[NonEmptyString], Field(min_length=1)]


def _lower_string(value):
    """Normalize values whose existing public contract is case-insensitive."""
    return value.lower() if isinstance(value, str) else value


Database = Literal["omnipath", "signor"]
SearchAlgorithm = Literal["bfs", "dfs"]
NormalizedVerbosity = Annotated[
    Literal["summary", "preview", "full"],
    BeforeValidator(_lower_string),
]
NormalizedExportFormat = Annotated[
    Literal["sif", "bnet"],
    BeforeValidator(_lower_string),
]
HandoffArtifactPrefix = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9_-])?$",
        description=(
            "Safe basename prefix for the paired .bnet and .handoff.json "
            "artifacts. Directory components and a trailing dot are forbidden."
        ),
    ),
]
OutputFormat = Literal["markdown", "json"]
NormalizedConnectorMethod = Annotated[
    Literal["hubs", "relax_max_len", "unsigned"],
    BeforeValidator(_lower_string),
]
BridgeMode = Literal["OUT", "IN", "ALL"]
TargetStrategy = Literal[
    "connect_to_upstream_nodes",
    "connect_subgroup",
]
AtopoStrategy = Literal["radial", "complete"]
GlobalStrategy = Literal[
    "complete_connection",
    "connect_network_radially",
    "connect_as_atopo",
]
RadialDirection = Literal["OUT", "IN"]

_READ_ONLY_CLOSED = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
_READ_ONLY_OPEN = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=True,
)
_IDEMPOTENT_CLOSED = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
_NON_IDEMPOTENT_CLOSED = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=False,
)
_NON_IDEMPOTENT_OPEN = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=True,
)
_DESTRUCTIVE_IDEMPOTENT_CLOSED = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=True,
    open_world_hint=False,
)
_DESTRUCTIVE_NON_IDEMPOTENT_CLOSED = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
    open_world_hint=False,
)
_DESTRUCTIVE_NON_IDEMPOTENT_OPEN = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
    open_world_hint=True,
)


def _optional_text(value) -> str | None:
    """Convert one dataframe scalar to a JSON-safe optional string."""
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _required_text(value, *, field_name: str) -> str:
    """Convert a required dataframe scalar or reject malformed network data."""
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"Network data contains an empty {field_name}.")
    return text


def _node_records(network) -> list[NeKoNodeRecord]:
    """Return all network nodes in their stored order."""
    nodes = getattr(network, "nodes", None)
    if nodes is None or nodes.empty:
        return []

    records = []
    for _, row in nodes.iterrows():
        records.append(
            NeKoNodeRecord(
                gene_symbol=_optional_text(row.get("Genesymbol")),
                uniprot=_optional_text(row.get("Uniprot")),
                node_type=_optional_text(row.get("Type")),
            )
        )
    return records


def _node_record_index(network) -> dict[str, NeKoNodeRecord]:
    """Index node records by both UniProt identifier and gene symbol."""
    index = {}
    for record in _node_records(network):
        if record.uniprot is not None:
            index[record.uniprot] = record
        if record.gene_symbol is not None:
            index.setdefault(record.gene_symbol, record)
    return index


def _node_record_for_identifier(
        identifier,
        *,
        node_index: dict[str, NeKoNodeRecord],
        uniprot_to_symbol: dict,
) -> NeKoNodeRecord:
    """Resolve an internal network identifier without losing its UniProt ID."""
    identifier_text = _required_text(identifier, field_name="node identifier")
    stored = node_index.get(identifier_text)
    if stored is not None:
        return stored
    return NeKoNodeRecord(
        gene_symbol=_optional_text(
            uniprot_to_symbol.get(identifier, uniprot_to_symbol.get(identifier_text))
        ),
        uniprot=identifier_text,
    )


def _interaction_records(df: pd.DataFrame) -> list[NeKoInteractionRecord]:
    """Convert an edge dataframe into strict JSON-safe interaction records."""
    records = []
    for _, row in df.iterrows():
        records.append(
            NeKoInteractionRecord(
                source=_required_text(row.get("source"), field_name="edge source"),
                target=_required_text(row.get("target"), field_name="edge target"),
                effect=_optional_text(row.get("Effect")),
            )
        )
    return records


def _reference_list(value) -> list[str]:
    """Normalize NeKo's comma/semicolon-delimited reference values."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_references = value
    else:
        text = _optional_text(value)
        if text is None or text.lower() == "none":
            return []
        raw_references = re.split(r"[;,]", text)

    references = []
    seen = set()
    for value in raw_references:
        reference = _optional_text(value)
        if reference is not None and reference not in seen:
            references.append(reference)
            seen.add(reference)
    return references


def _referenced_interaction_records(
        df: pd.DataFrame,
) -> list[NeKoReferencedInteractionRecord]:
    """Convert an edge dataframe while retaining its complete evidence list."""
    records = []
    for _, row in df.iterrows():
        references = _reference_list(row.get("References"))
        records.append(
            NeKoReferencedInteractionRecord(
                source=_required_text(row.get("source"), field_name="edge source"),
                target=_required_text(row.get("target"), field_name="edge target"),
                effect=_optional_text(row.get("Effect")),
                reference_count=len(references),
                references=references,
            )
        )
    return records


def _export_sanitized_bnet(
    network,
    destination: Path,
    *,
    overwrite: bool,
) -> dict:
    """Export and sanitize one BNET through a temporary session-local path."""
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing BNET artifact: {destination}"
        )

    try:
        with tempfile.TemporaryDirectory(
            dir=destination.parent,
            prefix=".neko-bnet-",
        ) as temporary_directory:
            temporary_prefix = Path(temporary_directory) / destination.stem
            try:
                Exports(network).export_bnet(str(temporary_prefix))
            except Exception as exc:
                raise RuntimeError(f"Error exporting BNET: {exc}") from exc
            generated_bnets = sorted(Path(temporary_directory).glob("*.bnet"))
            if not generated_bnets:
                raise FileNotFoundError(
                    "NeKo did not create a BNET file in the temporary export "
                    f"directory {temporary_directory}."
                )
            if len(generated_bnets) > 1:
                generated_names = ", ".join(
                    path.name for path in generated_bnets
                )
                raise RuntimeError(
                    "NeKo generated multiple BNET models "
                    f"({generated_names}); this export requires exactly one model."
                )
            temporary_bnet = generated_bnets[0]
            try:
                sanitizer_result = sanitize_bnet_file(str(temporary_bnet))
            except Exception as exc:
                raise RuntimeError(f"Error sanitizing BNET: {exc}") from exc

            if overwrite:
                temporary_bnet.replace(destination)
            else:
                try:
                    os.link(temporary_bnet, destination)
                except FileExistsError as exc:
                    raise FileExistsError(
                        "Refusing to overwrite BNET artifact created "
                        f"concurrently: {destination}"
                    ) from exc
            return sanitizer_result
    except OSError as exc:
        raise RuntimeError(
            f"Could not finalize BNET artifact {destination}: {exc}"
        ) from exc


def _neko_package_version() -> str:
    """Return the installed NeKo distribution version for provenance."""
    try:
        return package_version("nekomata")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Cannot export a handoff because the installed `nekomata` "
            "package version is unavailable."
        ) from exc


def _network_history_state_id(network) -> int | None:
    """Return a valid non-negative NeKo history state identifier when present."""
    value = getattr(network, "current_state_id", None)
    if value is None or isinstance(value, bool):
        return None
    try:
        state_id = int(value)
    except (TypeError, ValueError):
        return None
    return state_id if state_id >= 0 else None


# NOTE: Previous implementation used a single global `network` object.
# Now session-based management (see `session_manager.py`).
# Each tool can accept an optional `session_id` allowing multiple networks.
# If not provided, the default session is used (auto-created on first use).

# 1. Define the manual in one place so you only ever have to update it here.
NEKO_AGENT_MANUAL = """
# NeKo to MaBoSS Workflow Manual

## 1. Recommended Execution Order
1. **Initialize:** `create_session()` -> `set_default_params(max_len=2, only_signed=True, consensus=True)`
2. **Build:** `create_network([...list_of_initial_genes...], database='omnipath')`
3. **Curate:** `remove_bimodal_interactions()` -> `remove_undefined_interactions()`
4. **Audit Connectivity:** `analyze_connectivity()` reports both isolated
   (0-edge) nodes and the full connected-component partition in one call.
   - *If disconnected:* `preview_connection_impact()` -> Apply a connection tool
     (see the cost guide below before choosing one).
5. **Inspect history:** `list_network_history()` after topology changes. Use
   `compare_network_states(state_a, state_b)` before deciding whether to
   `navigate_network_history(action='checkout', state_id=...)`.
6. **Inspect network:** `list_genes_and_interactions(verbosity='preview')`
7. **Export:** Use `export_neko_handoff(biological_context=...,
   output_nodes=[...])` for a typed MaBoSS transfer. Use
   `export_network(format='bnet')` only when a standalone BNET is sufficient.

## 2. Tool Categories
* **Sessions:** `create_session`, `list_sessions`, `set_default_session`, `delete_session`, `status`, `reset_network`
* **Construction:** `create_network`, `add_nodes` (batch add, optional cheap
  direct-neighbour autoconnect), `remove_gene`, `remove_interaction`
* **Connectivity diagnostics:** `analyze_connectivity` (isolated nodes + full
  component partition), `preview_connection_impact` (non-mutating scout: hub
  ranking or a simulated parameter-relaxation preview)
* **Connection strategies:** `connect_targeted_nodes` (integrate specific
  nodes), `bridge_components` (connect group A <-> group B),
  `apply_global_connection` (whole-network closure)
* **Inspection:** `list_genes_and_interactions`, `find_paths`, `get_references`, `filter_interactions`
* **History:** `list_network_history`, `navigate_network_history`, `compare_network_states`, `set_network_history_limit`
* **Handoff:** `export_neko_handoff` records exact sanitized Boolean nodes,
  declared outputs, package versions, history state, and artifact digests.

## 3. Connection Strategy Cost Guide
Choose the cheapest strategy that can plausibly close the gap; escalate only
if it fails. All costs assume seed/group sizes of a few dozen genes.

| Tool | Strategy | Cardinality | Relative cost | Notes |
|---|---|---|---|---|
| `add_nodes` | `autoconnect=True` | new node(s) -> existing network | Very low | Direct-neighbour edges only (equivalent to maxlen=1); no multi-step path search |
| `connect_targeted_nodes` | `connect_to_upstream_nodes` | specific node(s) | Low, bounded by `depth` | Cascades upstream from the given nodes only |
| `connect_targeted_nodes` | `connect_subgroup` | one node list | Moderate, ~O(pairs in group) | Pairwise path search within the group only |
| `bridge_components` | `connect_component` (A, B) | group A <-> group B | Moderate-high | Also silently runs `connect_subgroup` on every node NOT in A or B as a side effect |
| `apply_global_connection` | `connect_network_radially` | whole network | Moderate-high, bounded by `max_len` hops | Expands outward/inward from existing seed nodes only, not all pairs |
| `apply_global_connection` | `connect_as_atopo` | whole network, output-anchored | High, open-ended | Runs `connect_network_radially` or `complete_connection` first, then loops upstream search until the network is fully connected - the loop is not bounded by `max_len` alone |
| `apply_global_connection` | `complete_connection` | whole network | **Highest - O(N^2) over every node pair** | Searches paths between every pair of nodes in the network; this is almost certainly the cause of a network exploding in size (e.g. `max_len=2` with 20+ seed genes producing hundreds of edges) |

**Large-network warning:** before calling `complete_connection` or
`connect_as_atopo`, check the current node count (via `status()`). Above
roughly 50 nodes, prefer `connect_targeted_nodes` or `bridge_components` to
close specific gaps instead, or run `preview_connection_impact()` first to see
the predicted edge-count delta without committing.

## 4. Critical Operating Rules
* **Session First:** Always call `create_session` before `create_network`.
* **Scout Before You Shoot:** Always run `preview_connection_impact()` before
  heavy connection tools, and check the cost guide above.
* **Output Names:** Handoff output nodes may use original NeKo names; export
  translates renamed symbols to the exact names stored in the sanitized BNET.
  If outputs are omitted, MaBoSS must select a small output set before running.
* **Token Frugality:** In iterative loops, ALWAYS use `verbosity='summary'`. 
"""


# 2. Expose it as an MCP Prompt (For smart clients that pull system prompts)
@mcp.prompt(name="neko_workflow_prompt", description="System prompt and operating manual for the NeKo agent.")
def neko_workflow_prompt() -> str:
    return NEKO_AGENT_MANUAL

# 3. Expose it as an MCP Resource (For clients that prefer to 'read' documentation)
@mcp.resource(
    uri="docs://neko/agent_manual",
    name="NeKo Agent Operations Manual",
    description="The single source of truth for NeKo workflows, tool categories, and rules.",
    mime_type="text/markdown"
)
def neko_agent_manual_resource() -> str:
    return NEKO_AGENT_MANUAL


@mcp.resource(
    uri="neko://session/{session_id}/history",
    name="Network History",
    description=(
        "Branching NeKo network history rendered as inline SVG HTML. "
        "Use list_network_history to identify the current state."
    ),
    mime_type="text/html",
)
def network_history_resource(session_id: str) -> str:
    """Render the history of an existing NeKo session without mutating it."""
    try:
        with session_manager.existing_session_scope(session_id) as sess:
            if sess.network is None:
                raise ResourceNotFoundError(
                    "No network in this session. Call create_network first."
                )
            try:
                return sess.network.history_html()
            except Exception as exc:
                raise RuntimeError(
                    "Unable to render NeKo history as HTML. "
                    f"Check the Graphviz installation: {exc}"
                ) from exc
    except KeyError as exc:
        raise ResourceNotFoundError(
            f"NeKo session not found: {session_id}"
        ) from exc

@mcp.tool(annotations=_DESTRUCTIVE_NON_IDEMPOTENT_OPEN)
async def create_network(
                   list_of_initial_genes: Annotated[List[NonEmptyString], Field(description="Gene symbols to seed the network (e.g. ['TP53', 'MYC', 'CASP3']). Can be empty if sif_file is provided.")],
                   ctx: Context,
                   database: Database = Field("omnipath", description="Knowledge-base to query. 'omnipath' (default) or 'signor'."),
                   sif_file: Optional[NonEmptyString] = Field(None, description="Absolute path to an existing SIF file to bootstrap the network from. Combined with list_of_initial_genes when both are given."),
                   max_len: int = Field(2, ge=1, le=4, description="Maximum path length used by complete_connection to bridge seed genes (1-4; larger = denser but slower)."),
                   algorithm: SearchAlgorithm = Field("bfs", description="Search algorithm for path completion: 'bfs' (breadth-first, default) or 'dfs' (depth-first)."),
                   only_signed: bool = Field(True, description="Restrict to signed (+/-) interactions only. Set False to allow unsigned interactions when network is sparse."),
                   connect_with_bias: bool = Field(False, description="Avoids looking for paths between pairs of nodes that are already connected by another path previously found."),
                   consensus: bool = Field(True, description="Require interactions supported by multiple curated sources (higher confidence)."),
                   session_id: Optional[NonEmptyString] = Field(None, description="Session ID to write the network into. Omit to use the active/default session."),
                   verbosity: NormalizedVerbosity = Field(DEFAULT_VERBOSITY, description="Output detail level: 'summary' (default, token-frugal), 'preview' (truncated tables), 'full'.")) -> str:
    """Build a NeKo gene regulatory network from seed genes and/or a SIF file.

    Calls complete_connection internally to bridge genes via the chosen database.
    After creation, run remove_bimodal_interactions() and analyze_connectivity()
    before exporting. Always call create_session() first.
    """
    verbosity = normalize_verbosity(verbosity)
    if database not in ["omnipath", "signor"]:
        raise ValueError("Unsupported database. Use `omnipath` or `signor`.")
    if sif_file is not None and not os.path.isfile(sif_file):
        raise FileNotFoundError(f"SIF file not found: {sif_file}")
    if sif_file is None and not list_of_initial_genes:
        raise ValueError(format_no_input_guidance())

    await ctx.report_progress(0, 4)

    def load_resources():
        if database == "signor":
            logger.info("Downloading SIGNOR database")
            signor_res = signor()
            signor_res.build()
            logger.info("SIGNOR database downloaded successfully")
            return signor_res.interactions
        return "omnipath"

    resources = await anyio.to_thread.run_sync(load_resources)
    await ctx.report_progress(1, 4)

    def build_network_locked() -> str:
        with session_manager.session_scope(session_id):
            sess = ensure_session(session_id)
            logger.info(
                "Creating NeKo network (session=%s) with genes=%s sif=%s",
                sess.session_id,
                list_of_initial_genes,
                sif_file,
            )

            # Build locally so a failed rebuild cannot replace a valid network.
            if sif_file is not None:
                try:
                    new_network = Network(
                        sif_file=sif_file,
                        resources=resources,
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Unable to create network from SIF file: {e}"
                    ) from e

                failed_genes = []
                for gene in list_of_initial_genes:
                    try:
                        new_network.add_node(gene)
                    except Exception as e:
                        failed_genes.append(f"{gene}: {e}")
                if failed_genes:
                    raise RuntimeError(
                        "Network loaded from the SIF file, but these genes "
                        f"could not be added: {'; '.join(failed_genes)}"
                    )
            else:
                new_network = Network(
                    list_of_initial_genes,
                    resources=resources,
                )
                logger.info(
                    "Running complete_connection (max_len=%s, only_signed=%s)",
                    max_len,
                    only_signed,
                )
                new_network.complete_connection(
                    maxlen=max_len,
                    algorithm=algorithm,
                    only_signed=only_signed,
                    connect_with_bias=connect_with_bias,
                    consensus=consensus,
                )

            history_max_states = sess.get_history_max_states()
            if history_max_states is not None:
                new_network.set_max_history(history_max_states)

            try:
                df_edges = new_network.convert_edgelist_into_genesymbol()
            except Exception as e:
                raise RuntimeError(
                    format_network_creation_error(
                        "build_failed",
                        list_of_initial_genes,
                        str(e),
                    )
                ) from e
            if df_edges is None:
                raise RuntimeError(
                    format_network_creation_error(
                        "build_failed",
                        list_of_initial_genes,
                        "NeKo could not convert the network edge table.",
                    )
                )

            # Publish only after construction and validation have succeeded.
            sess.set_network(new_network, edges_df=df_edges)

            if df_edges.empty:
                logger.warning(
                    "No interactions found in the network; "
                    "check the input parameters"
                )
                return format_empty_network_response(
                    list_of_initial_genes,
                    database,
                    max_len,
                    only_signed,
                )

            num_edges = len(df_edges)
            unique_nodes = pd.unique(
                df_edges[["source", "target"]].values.ravel()
            )
            num_nodes = len(unique_nodes)
            logger.info(
                "Network created successfully: %s nodes, %s edges",
                num_nodes,
                num_edges,
            )

            if verbosity == "summary":
                return (
                    f"Network created: session={sess.session_id} "
                    f"nodes={num_nodes} edges={num_edges}. "
                    "Check connectivity via "
                    f"analyze_connectivity(). {SUMMARY_HINT}"
                )

            preview_df = df_edges[
                [
                    column
                    for column in ["source", "target", "Effect"]
                    if column in df_edges.columns
                ]
            ].head(100)
            preview_md = clean_for_markdown(preview_df).to_markdown(
                index=False,
                tablefmt="plain",
            )
            lines = [
                f"Network created (session={sess.session_id})",
                f"Initial genes: {', '.join(list_of_initial_genes)}",
                f"Nodes: {num_nodes} | Edges: {num_edges}",
            ]
            if verbosity == "preview":
                lines.append("Preview (first 100):\n" + preview_md)
            elif verbosity == "full":
                lines.append(
                    "Full preview (first 100 interactions):\n" + preview_md
                )
                lines.append(
                    f"Parameters: database={database} max_len={max_len} "
                    f"algorithm={algorithm} only_signed={only_signed} "
                    f"consensus={consensus}"
                )
            return "\n".join(lines)

    response = await anyio.to_thread.run_sync(build_network_locked)
    await ctx.report_progress(4, 4)
    return response

@mcp.tool(annotations=_NON_IDEMPOTENT_OPEN)
@requires_network
def add_nodes(
        genes: Annotated[NonEmptyStringList, Field(description="Gene symbols to add (e.g. ['TP53'] or ['EGFR', 'AKT1']).")],
        session_id: Optional[NonEmptyString] = Field(None, description="Session ID; omit to use the active/default session."),
        autoconnect: bool = Field(True, description="After adding, connect each new gene to any direct (single-edge) neighbour already in the network. Cheap - does not search multi-step paths. Use connect_targeted_nodes() or apply_global_connection() afterwards for deeper integration."),
        only_signed: Optional[bool] = Field(None, description="Override the session default and keep only signed autoconnect edges."),
        consensus: Optional[bool] = Field(None, description="Override the session default and require consensus-supported autoconnect edges."),
        sess=None, network=None) -> str:
    """Add one or more gene nodes to the current network in a single call.

    autoconnect uses direct-neighbour lookup only (equivalent to maxlen=1),
    which is far cheaper than complete_connection. For multi-step bridging use
    connect_targeted_nodes() or apply_global_connection() after adding nodes.
    """
    added = 0
    failed_genes = []
    for gene in genes:
        try:
            network.add_node(gene)
            added += 1
        except Exception as e:
            failed_genes.append(f"{gene}: {e}")
    if failed_genes:
        _invalidate(sess)
        raise RuntimeError(
            f"Added {added}/{len(genes)} genes, but these additions failed: "
            f"{'; '.join(failed_genes)}"
        )

    if autoconnect:
        params = sess.get_completion_params()
        osgn = only_signed if only_signed is not None else params.get('only_signed', True)
        cons = consensus if consensus is not None else params.get('consensus', True)
        try:
            network.connect_nodes(only_signed=osgn, consensus_only=cons)
        except Exception as e:
            _invalidate(sess)
            raise RuntimeError(
                f"Added {added}/{len(genes)} genes, but autoconnect failed: {e}"
            ) from e

    _invalidate(sess)
    autoconnect_note = (
        "Autoconnected direct neighbours." if autoconnect else "No autoconnect."
    )
    return f"Added {added}/{len(genes)} genes. {autoconnect_note} {SUMMARY_HINT}"

@mcp.tool(annotations=_DESTRUCTIVE_NON_IDEMPOTENT_CLOSED)
@requires_network
def remove_gene(
        gene: Annotated[NonEmptyString, Field(description="Gene symbol to remove. Case-insensitive; closest match is suggested if not found.")],
        session_id: Optional[NonEmptyString] = Field(None, description="Session ID; omit to use the active/default session."),
        sess=None, network=None) -> str:
    """Remove a gene node (and all its edges) from the current network.

    Use list_genes_and_interactions() first to verify the exact symbol present in the network.
    """
    try:
        df_nodes = network.nodes if hasattr(network, 'nodes') else None
        if df_nodes is None:
            raise RuntimeError("Network node table unavailable.")

        # Collect possible identifiers (Genesymbol + Uniprot) for lookup (case-insensitive)
        symbols = set()
        symbol_map = {}  # lower -> original
        if 'Genesymbol' in df_nodes.columns:
            for val in df_nodes['Genesymbol'].dropna().astype(str):
                lv = val.upper()
                symbols.add(lv)
                symbol_map[lv] = val
        if 'Uniprot' in df_nodes.columns:
            for val in df_nodes['Uniprot'].dropna().astype(str):
                lv = val.upper()
                if lv not in symbol_map:
                    symbol_map[lv] = val  # prefer Genesymbol if duplicate
                symbols.add(lv)

        query = gene.upper()
        if query not in symbols:
            # Suggest similar (substring or Levenshtein-lite via length difference) - keep it lightweight
            candidates = list(symbol_map.values())
            partial = [c for c in candidates if query in c.upper() or c.upper() in query]
            # If no substring hits, fall back to first few for orientation
            suggestions = partial[:5] if partial else candidates[:5]
            msg = f"**Gene not found:** {gene} is not present in this session's network."
            if suggestions:
                msg += f"\n**Closest / sample nodes:** {', '.join(suggestions)}"
            msg += "\n**Tip:** Use list_genes_and_interactions(verbosity='preview') to inspect current nodes/interactions."
            raise ValueError(msg)

        # Use original casing for removal if stored
        original_name = symbol_map.get(query, gene)
        network.remove_node(original_name)
        _invalidate(sess)
        return f"Gene removed: {original_name}."
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error removing gene {gene}: {e}") from e

@mcp.tool(annotations=_DESTRUCTIVE_NON_IDEMPOTENT_CLOSED)
@requires_network
def remove_interaction(
        node_A: Annotated[NonEmptyString, Field(description="Source gene symbol (interaction goes A -> B).")],
        node_B: Annotated[NonEmptyString, Field(description="Target gene symbol (interaction goes A -> B).")],
        session_id: Optional[NonEmptyString] = Field(None, description="Session ID; omit to use the active/default session."),
        sess=None, network=None) -> str:
    """Remove a directed edge A -> B from the network (does not affect B -> A if it exists).

    Use filter_interactions() or list_genes_and_interactions() to locate the exact edge first.
    """
    try:
        df_edges = network.convert_edgelist_into_genesymbol()
    except Exception as e:
        raise RuntimeError(f"Unable to retrieve network edges: {e}") from e
    if df_edges.empty:
        raise ValueError(
            "No interactions exist in the network. "
            "Use list_genes_and_interactions() to inspect it first."
        )
    # Check if the interaction exists in the specified direction
    mask = (df_edges['source'] == node_A) & (df_edges['target'] == node_B)
    if not df_edges[mask].empty:
        try:
            network.remove_edge(node_A, node_B)
            _invalidate(sess)
            return f"Interaction removed: {node_A}->{node_B}."
        except Exception as e:
            raise RuntimeError(
                f"Error removing interaction {node_A} -> {node_B}: {e}"
            ) from e
    else:
        raise ValueError(
            f"Interaction not found: no interaction from {node_A} to {node_B} "
            "in the current network.\n"
            "**Tip:** Use list_genes_and_interactions() to inspect available "
            "interactions."
        )

# TO DO: Implement export of images with graphviz

# TO DO: implement GO enrichment

@mcp.tool(annotations=_IDEMPOTENT_CLOSED)
@session_locked
def export_network(
        format: NormalizedExportFormat = Field("sif", description="Export format: 'sif' (Simple Interaction Format, tab-separated) or 'bnet' (Boolean network for MaBoSS). BNET requires a fully connected network."),
        session_id: Optional[NonEmptyString] = Field(None, description="Session ID; omit to use the active/default session."),
        verbosity: NormalizedVerbosity = Field(DEFAULT_VERBOSITY, description="Output detail level: 'summary', 'preview', or 'full'.")) -> Annotated[CallToolResult, NeKoNetworkExportResult]:
    """Export the current network to SIF or BNET format.

    After BNET export, hand the file path to the MaBoSS server via bnet_to_bnd_and_cfg().
    BNET export fails if the network is not fully connected — run analyze_connectivity() first.
    """
    verbosity = normalize_verbosity(verbosity)
    export_format = format.lower()
    if export_format not in {"sif", "bnet"}:
        raise ValueError(format_unsupported_format_guidance(format))

    sess, network = _session_network(session_id)
    if network is None:
        raise RuntimeError(format_no_network_guidance())
    out_dir = _export_dir(sess.session_id)

    # ── SIF export ────────────────────────────────────────────────────────────
    if export_format == "sif":
        out_path = str(out_dir / "Network.sif")
        try:
            Exports(network).export_sif(out_path)
        except Exception as e:
            raise RuntimeError(f"Error exporting SIF: {e}") from e
        if verbosity == "summary":
            text = f"SIF exported: {out_path}. {SUMMARY_HINT}"
        else:
            try:
                df_prev = pd.read_csv(out_path, sep="\t", header=None,
                                      names=["source", "interaction", "target"],
                                      nrows=100, dtype=str).dropna(how="all")
                preview_md = _short_table(df_prev, max_rows=100)[0]
            except Exception:
                preview_md = "_Preview unavailable._"
            text = (
                f"SIF exported: `{out_path}`\n\n"
                f"Preview (first 100 rows):\n{preview_md}"
            )
        payload = NeKoNetworkExportResult(
            server="NeKo",
            session_id=sess.session_id,
            format="sif",
            file=artifact_file_summary(
                out_path,
                session_id=sess.session_id,
            ),
            renamed_nodes=[],
            duplicate_rules_removed=[],
        )
        return structured_report(text, payload)

    # ── BNET export ───────────────────────────────────────────────────────────
    else:
        if not is_connected(network):
            raise RuntimeError(format_connectivity_guidance())
        output_path = safe_artifact_path(out_dir, "Network.bnet")
        result = _export_sanitized_bnet(
            network,
            output_path,
            overwrite=True,
        )
        out_path = str(output_path)

        if verbosity == "summary":
            text = f"BNET exported: {out_path}. {SUMMARY_HINT}"
        else:
            try:
                df_prev = pd.read_csv(out_path, sep=",", header=None,
                                      names=["gene", "expression"],
                                      nrows=100, dtype=str).dropna(how="all")
                preview_md = _short_table(df_prev, max_rows=100)[0]
            except Exception:
                preview_md = "_Preview unavailable._"

            md_lines = [
                f"BNET exported: `{out_path}`",
                f"Next: call `bnet_to_bnd_and_cfg('{out_path}')` in the MaBoSS server.",
                "",
                "Preview (first 100 rows):",
                preview_md,
            ]
            if result["cleaned_names"]:
                md_lines.append(f"\n**Note:** Renamed to remove special characters: "
                                f"{', '.join(sorted(result['cleaned_names']))}")
            if result["duplicates_removed"]:
                md_lines.append(f"\n**Note:** Removed duplicate rules for (isoforms collapsed to first): "
                                f"{', '.join(sorted(set(result['duplicates_removed'])))}")
            text = "\n".join(md_lines)
        payload = NeKoNetworkExportResult(
            server="NeKo",
            session_id=sess.session_id,
            format="bnet",
            file=artifact_file_summary(
                out_path,
                session_id=sess.session_id,
            ),
            renamed_nodes=sorted(result["cleaned_names"]),
            duplicate_rules_removed=sorted(set(result["duplicates_removed"])),
        )
        return structured_report(text, payload)


@mcp.tool(annotations=_NON_IDEMPOTENT_CLOSED)
@session_locked
def export_neko_handoff(
        biological_context: NonEmptyString = Field(description="Biological question or modelling context that must remain attached to the network."),
        output_nodes: Optional[List[NonEmptyString]] = Field(None, description="Optional biologically meaningful MaBoSS output nodes. Original NeKo names are translated to sanitized BNET names."),
        artifact_prefix: HandoffArtifactPrefix = Field("neko_to_maboss", description="Safe prefix used for '<prefix>.bnet' and '<prefix>.handoff.json'. Choose a new prefix for every retained handoff."),
        session_id: Optional[NonEmptyString] = Field(None, description="Session ID to export; omit to use the active/default session.")) -> Annotated[CallToolResult, NeKoHandoffExportResult]:
    """Export a versioned, integrity-protected NeKo-to-MaBoSS handoff.

    The tool writes a sanitized BNET plus a JSON handoff manifest into the
    session artifact directory. Existing handoffs are never overwritten.
    Declared outputs are translated to the exact node names stored in the BNET.
    """
    sess, network = _session_network(session_id)
    if network is None:
        raise RuntimeError(format_no_network_guidance())
    if not is_connected(network):
        raise RuntimeError(format_connectivity_guidance())

    out_dir = _export_dir(sess.session_id)
    bnet_path = safe_artifact_path(
        out_dir,
        f"{artifact_prefix}.bnet",
    )
    manifest_path = safe_artifact_path(
        out_dir,
        f"{artifact_prefix}.handoff.json",
    )
    existing = [
        path
        for path in (bnet_path, manifest_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite an existing NeKo handoff artifact: "
            + ", ".join(str(path) for path in existing)
            + ". Choose a different artifact_prefix."
        )

    neko_version = _neko_package_version()
    created_paths: list[Path] = []
    try:
        sanitizer_result = _export_sanitized_bnet(
            network,
            bnet_path,
            overwrite=False,
        )
        created_paths.append(bnet_path)
        nodes = bnet_node_names(bnet_path)

        renamed_nodes = sorted(
            str(node)
            for node in sanitizer_result.get("cleaned_names", set())
        )
        node_renames = {
            str(original): str(renamed)
            for original, renamed in sorted(
                sanitizer_result.get("name_mapping", {}).items()
            )
        }
        duplicate_rules_removed = sorted({
            str(node)
            for node in sanitizer_result.get("duplicates_removed", [])
        })

        requested_outputs = list(output_nodes or [])
        if len(requested_outputs) != len(set(requested_outputs)):
            raise ValueError(
                "output_nodes must not contain duplicate names."
            )
        translated_outputs = [
            node_renames.get(node, node)
            for node in requested_outputs
        ]
        if len(translated_outputs) != len(set(translated_outputs)):
            raise ValueError(
                "Declared output nodes collapse onto the same sanitized "
                "MaBoSS node. Select only one original name for each output."
            )
        unknown_outputs = sorted(set(translated_outputs) - set(nodes))
        if unknown_outputs:
            available_preview = ", ".join(nodes[:25])
            raise ValueError(
                "Declared output nodes are absent from the sanitized BNET: "
                + ", ".join(unknown_outputs)
                + f". Available nodes include: {available_preview}."
            )

        bnet_reference = handoff_artifact(
            bnet_path,
            server="NeKo",
            session_id=sess.session_id,
            role="neko_bnet",
        )
        manifest = NeKoToMaBoSSHandoffManifest(
            source=HandoffProvenance(
                server="NeKo",
                session_id=sess.session_id,
                mcp_package=HandoffPackage(
                    name="mcp-biomodelling-servers",
                    version=__version__,
                ),
                modelling_package=HandoffPackage(
                    name="nekomata",
                    version=neko_version,
                ),
                operation="export_neko_handoff",
            ),
            biological_context=biological_context,
            network=HandoffNetwork(
                nodes=nodes,
                output_nodes=translated_outputs,
                renamed_nodes=renamed_nodes,
                node_renames=node_renames,
                duplicate_rules_removed=duplicate_rules_removed,
            ),
            history_state_id=_network_history_state_id(network),
            bnet_file=bnet_reference,
        )
        write_handoff_manifest(manifest_path, manifest)
        created_paths.append(manifest_path)
        manifest_reference = handoff_artifact(
            manifest_path,
            server="NeKo",
            session_id=sess.session_id,
            role="parent_manifest",
        )
        payload = NeKoHandoffExportResult(
            server="NeKo",
            session_id=sess.session_id,
            manifest_file=manifest_reference,
            manifest=manifest,
        )
    except Exception:
        for path in reversed(created_paths):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "Could not roll back incomplete handoff artifact %s",
                    path,
                    exc_info=True,
                )
        raise

    output_summary = (
        ", ".join(translated_outputs)
        if translated_outputs
        else (
            "none declared; select a small output set in MaBoSS before "
            "running a simulation"
        )
    )
    text = (
        "NeKo-to-MaBoSS handoff exported successfully.\n"
        f"  Manifest: {manifest_path}\n"
        f"  BNET: {bnet_path}\n"
        f"  Boolean nodes: {len(nodes)}\n"
        f"  Declared outputs: {output_summary}\n\n"
        "Next: pass the manifest path to the MaBoSS handoff import tool."
    )
    return structured_report(text, payload)


@mcp.tool(annotations=_READ_ONLY_CLOSED)
@session_locked
def list_genes_and_interactions(
        session_id: Optional[NonEmptyString] = Field(None, description="Session ID; omit to use the active/default session."),
        verbosity: NormalizedVerbosity = Field(DEFAULT_VERBOSITY, description="Output detail level: 'summary' (counts only), 'preview' (truncated table), 'full' (up to 100 rows)."),
        max_rows: int = Field(50, ge=1, description="Maximum rows to return in preview mode.")) -> Annotated[CallToolResult, NeKoNetworkInventoryResult]:
    """Return a Markdown table of all nodes and directed edges in the network.

    Equivalent to 'show the network'. Use filter_interactions() for targeted queries.
    """
    verbosity = normalize_verbosity(verbosity)
    sess, network = _session_network(session_id)
    if network is None:
        raise RuntimeError(E_NO_NET)

    try:
        df = sess.get_edges_df()
        if df is None:
            df = pd.DataFrame(columns=["source", "target", "Effect"])
        if "resources" in df.columns:
            df = df.drop(columns=["resources"])
        df = df[[c for c in ['source', 'target', 'Effect'] if c in df.columns]]

        nodes = _node_records(network)
        row_limit = max_rows if verbosity == "preview" else 100
        returned_nodes = [] if verbosity == "summary" else nodes[:row_limit]
        returned_df = (
            df.head(0)
            if verbosity == "summary"
            else df.head(row_limit)
        )
        interactions = _interaction_records(returned_df)
        truncated = (
            len(returned_nodes) < len(nodes)
            or len(interactions) < len(df)
        )
        payload = NeKoNetworkInventoryResult(
            server="NeKo",
            session_id=sess.session_id,
            verbosity=verbosity,
            total_node_count=len(nodes),
            total_interaction_count=len(df),
            returned_node_count=len(returned_nodes),
            returned_interaction_count=len(interactions),
            truncated=truncated,
            nodes=returned_nodes,
            interactions=interactions,
        )

        if verbosity == "summary":
            text = (
                f"Nodes: {len(nodes)}. Interactions: {len(df)}. "
                f"{SUMMARY_HINT}"
            )
            return structured_report(text, payload)

        node_rows = [
            {
                "Gene symbol": record.gene_symbol,
                "UniProt": record.uniprot,
                "Type": record.node_type,
            }
            for record in returned_nodes
        ]
        node_df = pd.DataFrame(
            node_rows,
            columns=["Gene symbol", "UniProt", "Type"],
        )
        node_table = (
            "_No nodes._"
            if node_df.empty
            else clean_for_markdown(node_df).to_markdown(
                index=False,
                tablefmt="plain",
            )
        )
        interaction_table = (
            "_Network loaded but contains no interactions._"
            if returned_df.empty
            else clean_for_markdown(returned_df).to_markdown(
                index=False,
                tablefmt="plain",
            )
        )
        note = "\n\n_Result truncated._" if truncated else ""
        text = (
            f"Nodes ({len(nodes)} total):\n{node_table}\n\n"
            f"Interactions ({len(df)} total):\n{interaction_table}{note}"
        )
        return structured_report(text, payload)
    except Exception as e:
        raise RuntimeError(f"Unable to retrieve network data: {e}") from e

@mcp.tool(annotations=_READ_ONLY_CLOSED)
@session_locked
def find_paths(
        source: Annotated[NonEmptyString, Field(description="Source gene symbol (path start).")],
        target: Annotated[NonEmptyString, Field(description="Target gene symbol (path end).")],
        maxlen: int = Field(3, ge=1, le=5, description="Maximum number of edges in a path (1-5; longer paths are slower to compute)."),
        session_id: Optional[NonEmptyString] = Field(None, description="Session ID; omit to use the active/default session."),
        verbosity: NormalizedVerbosity = Field(DEFAULT_VERBOSITY, description="Output detail level: 'summary' (count only), 'preview'/'full' (path listing).")) -> Annotated[CallToolResult, NeKoPathSearchResult]:
    """Find and display all directed paths between two genes up to maxlen edges.

    Useful for verifying biological signal flow before BNET export.
    Returns 'No paths found' if genes are in disconnected components.
    """
    verbosity = normalize_verbosity(verbosity)
    sess, network = _session_network(session_id)
    if network is None:
        raise RuntimeError(E_NO_NET)
    with _stdout_capture_lock:
        buffer = io.StringIO()
        old_stdout = sys.stdout
        try:
            # NeKo currently prints paths instead of returning them.
            sys.stdout = buffer
            network.print_my_paths(source, target, maxlen=maxlen)

            raw_output = buffer.getvalue().strip()
            lines = [
                line.strip()
                for line in raw_output.splitlines()
                if line.strip()
            ]
            payload = NeKoPathSearchResult(
                server="NeKo",
                session_id=sess.session_id,
                source=source,
                target=target,
                max_length=maxlen,
                has_output=bool(lines),
                output_line_count=len(lines),
                path_output_lines=lines,
            )

            if not lines:
                return structured_report("No paths found.", payload)
            if verbosity == "summary":
                text = f"Found {len(lines)} path lines. {SUMMARY_HINT}"
                return structured_report(text, payload)
            label = "Paths" if verbosity == "preview" else "Paths (full output)"
            text = f"{label}:\n```\n{raw_output}\n```"
            return structured_report(text, payload)

        except Exception as e:
            raise RuntimeError(f"Unable to find paths: {e}") from e
        finally:
            sys.stdout = old_stdout
            buffer.close()

@mcp.tool(annotations=_DESTRUCTIVE_IDEMPOTENT_CLOSED)
@session_locked
def reset_network(
        session_id: Optional[NonEmptyString] = Field(None, description="Session ID to reset; omit to use the active/default session.")) -> str:
    """Discard the current network in the session without deleting the session itself.

    Use delete_session() to remove the session entirely, or create_network() to rebuild.
    """
    sess = ensure_session(session_id)
    sess.set_network(None)
    return f"Session {sess.session_id} network reset."


@mcp.tool(annotations=_READ_ONLY_CLOSED)
@requires_network
def list_network_history(
        session_id: NonEmptyString | None = Field(
            None,
            description=(
                "Session ID; omit to inspect the active/default session."
            ),
        ),
        *,
        sess=None,
        network=None,
        ) -> NetworkHistorySummary:
    """List exact state IDs and branch relationships for the current network."""
    return summarize_history(
        network,
        session_id=sess.session_id,
        max_states=sess.get_history_max_states(),
    )


@mcp.tool(annotations=_NON_IDEMPOTENT_CLOSED)
@requires_network
def navigate_network_history(
        action: Annotated[
            Literal["undo", "redo", "checkout"],
            Field(
                description=(
                    "History operation: undo to a parent, redo to a child, "
                    "or checkout an exact state ID."
                )
            ),
        ],
        state_id: int | None = Field(
            None,
            ge=0,
            description=(
                "Exact target state ID. Required for checkout; optional for "
                "redo when the current state has one child; invalid for undo."
            ),
        ),
        session_id: NonEmptyString | None = Field(
            None,
            description=(
                "Session ID; omit to use the active/default session."
            ),
        ),
        *,
        sess=None,
        network=None,
        ) -> HistoryNavigationResult:
    """Move through NeKo's branching history while preserving saved states."""
    if action == "checkout" and state_id is None:
        raise ValueError("checkout requires an exact state_id.")
    if action == "undo" and state_id is not None:
        raise ValueError("undo does not accept state_id.")

    previous_state_id = network.current_state_id
    if action == "undo":
        network.undo()
    elif action == "redo":
        network.redo(state_id)
    else:
        network.checkout(state_id)

    current_state_id = network.current_state_id
    moved = previous_state_id != current_state_id
    if moved:
        _invalidate(sess)

    if moved:
        message = (
            f"Moved from state {previous_state_id} to state "
            f"{current_state_id}."
        )
    elif action == "undo":
        message = "Already at a root state; no parent state is available."
    elif action == "redo":
        message = "No child state is available from the current state."
    else:
        message = f"State {current_state_id} is already checked out."

    return HistoryNavigationResult(
        server="NeKo",
        session_id=sess.session_id,
        action=action,
        requested_state_id=state_id,
        previous_state_id=previous_state_id,
        current_state_id=current_state_id,
        moved=moved,
        node_count=len(network.nodes),
        edge_count=len(network.edges),
        message=message,
    )


@mcp.tool(annotations=_READ_ONLY_CLOSED)
@requires_network
def compare_network_states(
        state_a: Annotated[
            int,
            Field(ge=0, description="Exact ID of the baseline history state."),
        ],
        state_b: Annotated[
            int,
            Field(ge=0, description="Exact ID of the comparison history state."),
        ],
        session_id: NonEmptyString | None = Field(
            None,
            description=(
                "Session ID; omit to use the active/default session."
            ),
        ),
        *,
        sess=None,
        network=None,
        ) -> NetworkStateComparison:
    """Compare two saved topologies without changing the checked-out state."""
    return compare_history_states(
        network,
        session_id=sess.session_id,
        state_a=state_a,
        state_b=state_b,
    )


@mcp.tool(annotations=_DESTRUCTIVE_IDEMPOTENT_CLOSED)
@session_locked
def set_network_history_limit(
        max_states: Annotated[
            int | None,
            Field(
                ge=2,
                description=(
                    "Maximum retained states (minimum 2), or null for "
                    "unbounded history. Lowering this value may immediately "
                    "and irreversibly prune older snapshots."
                ),
            ),
        ],
        session_id: NonEmptyString | None = Field(
            None,
            description=(
                "Session ID; omit to configure the active/default session."
            ),
        ),
        ) -> HistoryRetentionResult:
    """Configure optional NeKo history pruning for this session."""
    sess = ensure_session(session_id)
    network = sess.network
    before_ids = state_ids(network) if network is not None else []

    if network is not None:
        network.set_max_history(max_states)
    sess.set_history_max_states(max_states)

    after_ids = state_ids(network) if network is not None else []
    pruned_ids = sorted(set(before_ids) - set(after_ids))
    if max_states is None:
        policy = "History retention is now unbounded."
    else:
        policy = f"History retention is limited to {max_states} states."
    if network is None:
        policy += " The policy will apply when a network is created."
    elif pruned_ids:
        policy += (
            " Pruned state IDs: "
            + ", ".join(str(state_id) for state_id in pruned_ids)
            + "."
        )

    return HistoryRetentionResult(
        server="NeKo",
        session_id=sess.session_id,
        max_states=max_states,
        applies_to_current_network=network is not None,
        state_count_before=len(before_ids),
        state_count_after=len(after_ids),
        pruned_state_ids=pruned_ids,
        retained_state_ids=after_ids,
        message=policy,
    )

@mcp.tool(annotations=_DESTRUCTIVE_IDEMPOTENT_CLOSED)
@session_locked
def clean_generated_files(
        session_id: Optional[NonEmptyString] = Field(None, description="Session ID whose artifact files (SIF, BNET, etc.) should be removed. Omit for the active/default session.")) -> Annotated[CallToolResult, NeKoArtifactCleanupResult]:
    """Delete all exported artifact files (SIF, BNET) for the given session."""
    sess = ensure_session(session_id)
    try:
        count = clean_artifacts(_SERVER_ROOT, sess.session_id)
        text = f"Cleaned {count} artifact file(s) from session {sess.session_id}."
        payload = NeKoArtifactCleanupResult(
            server="NeKo",
            session_id=sess.session_id,
            removed_count=count,
        )
        return structured_report(text, payload)
    except Exception as e:
        raise RuntimeError(f"Error during cleanup: {e}") from e

@mcp.tool(annotations=_DESTRUCTIVE_IDEMPOTENT_CLOSED)
@session_locked
def remove_bimodal_interactions(
        session_id: Optional[NonEmptyString] = Field(None, description="Session ID; omit to use the active/default session.")) -> str:
    """Remove all bimodal (simultaneously activating and inhibiting) edges from the network.

    Bimodal interactions are ambiguous and cause contradictory Boolean rules in BNET export.
    Run this as part of the standard curation step after create_network().
    """
    sess, network = _session_network(session_id)
    if network is None:
        raise RuntimeError(E_NO_NET)
    if "Effect" not in network.edges.columns:
        raise RuntimeError("No 'Effect' column found in network.edges.")
    before = len(network.edges)
    network.remove_bimodal_interactions()
    _invalidate(sess)
    after = len(network.edges)
    removed = before - after
    return f"Removed {removed} bimodal interactions from the network."

@mcp.tool(annotations=_DESTRUCTIVE_IDEMPOTENT_CLOSED)
@session_locked
def remove_undefined_interactions(
        session_id: Optional[NonEmptyString] = Field(None, description="Session ID; omit to use the active/default session.")) -> str:
    """Remove all edges whose Effect is 'undefined' (unknown sign) from the network.

    Undefined interactions cannot be mapped to Boolean activations or inhibitions.
    Run after remove_bimodal_interactions() in the standard curation sequence.
    """
    sess, network = _session_network(session_id)
    if network is None:
        raise RuntimeError(E_NO_NET)
    if "Effect" not in network.edges.columns:
        raise RuntimeError("No 'Effect' column found in network.edges.")
    before = len(network.edges)
    network.remove_undefined_interactions()
    _invalidate(sess)
    after = len(network.edges)
    removed = before - after
    return f"Removed {removed} undefined interactions from the network."


@mcp.tool(annotations=_READ_ONLY_CLOSED)
@session_locked
def list_bnet_files(
        session_id: Optional[NonEmptyString] = Field(None, description="Session ID to query; omit to use the active/default session.")) -> Annotated[CallToolResult, NeKoArtifactFileListResult]:
    """List names of all .bnet files in the session artifact directory (newline-separated)."""
    sess = ensure_session(session_id)
    files = [
        path
        for path in list_artifacts(_SERVER_ROOT, session_id=sess.session_id)
        if path.suffix == ".bnet"
    ]
    if not files:
        text = (
            f"No .bnet files found in session {sess.session_id} "
            "artifact directory."
        )
    else:
        text = "\n".join(path.name for path in files)
    payload = NeKoArtifactFileListResult(
        server="NeKo",
        scope="session",
        session_id=sess.session_id,
        count=len(files),
        files=[
            artifact_file_summary(path, session_id=sess.session_id)
            for path in files
        ],
    )
    return structured_report(text, payload)

@mcp.tool(annotations=_READ_ONLY_CLOSED)
@session_locked
def get_references(
        node1: Annotated[NonEmptyString, Field(description="Gene symbol. Returns all edges where this gene is source or target.")],
        node2: Optional[NonEmptyString] = Field(None, description="Second gene symbol. When provided, returns only edges between node1 and node2 (either direction)."),
        session_id: Optional[NonEmptyString] = Field(None, description="Session ID; omit to use the active/default session."),
        verbosity: NormalizedVerbosity = Field(DEFAULT_VERBOSITY, description="Output detail level: 'summary' (count only), 'preview'/'full' (Markdown table).")) -> Annotated[CallToolResult, NeKoReferenceQueryResult]:
    """Show literature references for interactions involving one or two genes.

    References are truncated to the first 5 per edge with a count of remaining.
    Useful for assessing interaction evidence before pruning.
    """
    verbosity = normalize_verbosity(verbosity)
    sess, network = _session_network(session_id)
    if network is None:
        raise RuntimeError(E_NO_NET)
    try:
        df = network.convert_edgelist_into_genesymbol()
    except Exception as e:
        raise RuntimeError(f"Unable to retrieve network edges: {e}") from e
    if df.empty:
        payload = NeKoReferenceQueryResult(
            server="NeKo",
            session_id=sess.session_id,
            node1=node1,
            node2=node2,
            interaction_count=0,
            interactions=[],
        )
        return structured_report(
            "_No interactions found in the network._",
            payload,
        )
    # Filter by node(s)
    if node2:
        mask = ((df['source'] == node1) & (df['target'] == node2)) | ((df['source'] == node2) & (df['target'] == node1))
        filtered = df[mask].copy()
    else:
        mask = (df['source'] == node1) | (df['target'] == node1)
        filtered = df[mask].copy()
    if filtered.empty:
        payload = NeKoReferenceQueryResult(
            server="NeKo",
            session_id=sess.session_id,
            node1=node1,
            node2=node2,
            interaction_count=0,
            interactions=[],
        )
        return structured_report("No matching interactions.", payload)
    # Only keep relevant columns
    cols = ['source', 'target', 'Effect', 'References']
    filtered = filtered[[c for c in cols if c in filtered.columns]]
    interactions = _referenced_interaction_records(filtered)
    payload = NeKoReferenceQueryResult(
        server="NeKo",
        session_id=sess.session_id,
        node1=node1,
        node2=node2,
        interaction_count=len(interactions),
        interactions=interactions,
    )
    # Truncate references for display
    def short_refs(refs):
        ref_list = _reference_list(refs)
        if len(ref_list) > 5:
            return '; '.join(ref_list[:5]) + f" (+{len(ref_list)-5} more)"
        return '; '.join(ref_list)
    display = filtered.copy()
    if "References" in display.columns:
        display['References'] = display['References'].apply(short_refs)
    # Clean for markdown
    if verbosity == "summary":
        text = f"References: {len(filtered)} interactions. {SUMMARY_HINT}"
        return structured_report(text, payload)
    md = clean_for_markdown(display).to_markdown(
        index=False,
        tablefmt="plain",
    )
    return structured_report(md, payload)

@mcp.tool(annotations=_IDEMPOTENT_CLOSED)
@session_locked
def set_default_params(
        max_len: Optional[int] = Field(None, ge=1, le=4, description="Default maximum path length for complete_connection calls (1-4)."),
        algorithm: Optional[SearchAlgorithm] = Field(None, description="Default path-search algorithm: 'bfs' or 'dfs'."),
        only_signed: Optional[bool] = Field(None, description="Default signed-only filter for complete_connection."),
        connect_with_bias: Optional[bool] = Field(None, description="Default activation-bias preference."),
        consensus: Optional[bool] = Field(None, description="Default multi-source consensus requirement."),
        session_id: Optional[NonEmptyString] = Field(None, description="Session ID; omit to use the active/default session.")) -> str:
    """Persist completion parameters in the session so add_nodes(autoconnect=True) reuses them."""
    sess = ensure_session(session_id)
    sess.update_default_params(max_len=max_len, algorithm=algorithm, only_signed=only_signed,
                               connect_with_bias=connect_with_bias, consensus=consensus)
    return "Defaults updated." 

@mcp.tool(annotations=_READ_ONLY_CLOSED)
@session_locked
def filter_interactions(
        effect: Optional[List[NonEmptyString]] = Field(None, description="Effect types to keep, e.g. ['stimulation', 'inhibition']. Omit to include all effects."),
        source: Optional[NonEmptyString] = Field(None, description="Keep only edges where the source matches this gene symbol."),
        target: Optional[NonEmptyString] = Field(None, description="Keep only edges where the target matches this gene symbol."),
        session_id: Optional[NonEmptyString] = Field(None, description="Session ID; omit to use the active/default session."),
        verbosity: NormalizedVerbosity = Field(DEFAULT_VERBOSITY, description="Output detail level: 'summary' (count), 'preview'/'full' (Markdown table)."),
        format: OutputFormat = Field("markdown", description="Output format: 'markdown' (default) or 'json'."),
        max_rows: int = Field(50, ge=1, description="Maximum rows returned in preview mode.")) -> Annotated[CallToolResult, NeKoInteractionFilterResult]:
    """Filter and display interactions by effect type, source gene, or target gene.

    Non-destructive - does not modify the network; use remove_interaction() to permanently delete edges.
    """
    verbosity = normalize_verbosity(verbosity)
    sess, network = _session_network(session_id)
    if network is None:
        raise RuntimeError(E_NO_NET)
    df = sess.get_edges_df()
    if df is None:
        df = pd.DataFrame(columns=["source", "target", "Effect"])
    if effect and 'Effect' in df.columns:
        df = df[df['Effect'].isin(effect)]
    if source:
        df = df[df['source'] == source]
    if target:
        df = df[df['target'] == target]

    total_match_count = len(df)
    if verbosity == "summary" and format != "json":
        returned_df = df.head(0)
    elif verbosity == "full":
        returned_df = df.head(500)
    else:
        returned_df = df.head(max_rows)
    returned_df = returned_df[
        [
            column
            for column in ["source", "target", "Effect"]
            if column in returned_df.columns
        ]
    ]
    interactions = _interaction_records(returned_df)
    truncated = len(interactions) < total_match_count
    payload = NeKoInteractionFilterResult(
        server="NeKo",
        session_id=sess.session_id,
        verbosity=verbosity,
        effect_filter=effect,
        source_filter=source,
        target_filter=target,
        total_match_count=total_match_count,
        returned_count=len(interactions),
        truncated=truncated,
        interactions=interactions,
    )

    if format == "json":
        text = json.dumps(
            [record.model_dump(mode="json") for record in interactions],
            separators=(",", ":"),
        )
        return structured_report(text, payload)
    if total_match_count == 0:
        text = "No interactions." if not any((effect, source, target)) else "No matches."
        return structured_report(text, payload)
    if verbosity == "summary":
        text = f"Filtered interactions: {total_match_count}. {SUMMARY_HINT}"
        return structured_report(text, payload)
    table = clean_for_markdown(returned_df).to_markdown(
        index=False,
        tablefmt="plain",
    )
    text = table + (" (truncated)" if truncated else "")
    return structured_report(text, payload)

@mcp.tool(annotations=_NON_IDEMPOTENT_CLOSED)
def create_session(
        label: Optional[str] = Field(None, description="Optional human-readable label for this session (e.g. 'TP53-MYC cancer'). Stored on disk so the session can be rediscovered after a server restart.")) -> str:
    """Create a new isolated modelling session (always call before create_network).

    Each session holds its own Network object and default completion parameters.
    Prevents accidental reuse of a previous network when starting a new hypothesis.
    A unique UUID is assigned — use it in all subsequent tool calls.
    """
    with session_manager.create_session_scope(set_as_default=False) as sess:
        sid = sess.session_id
        write_session_meta(
            _SERVER_ROOT,
            sid,
            server_name="NeKo",
            label=label,
        )
    label_info = f" ({label})" if label else ""
    return f"Created session: {sid}{label_info}"

@mcp.tool(annotations=_READ_ONLY_CLOSED)
def list_sessions() -> Annotated[CallToolResult, NeKoSessionListResult]:
    """List all active sessions with network presence and basic node/edge counts."""
    data = session_manager.list_sessions()
    default_session_id = session_manager.get_default_session_id()
    payload = NeKoSessionListResult(
        server="NeKo",
        count=len(data),
        sessions=[
            NeKoSessionSummary(
                session_id=sid,
                created_at=meta["created_at"],
                last_accessed=meta["last_accessed"],
                is_default=sid == default_session_id,
                has_network=meta["has_network"],
                node_count=meta["nodes"],
                edge_count=meta["edges"],
                history_max_states=meta["history_max_states"],
            )
            for sid, meta in data.items()
        ],
    )
    if not data:
        return structured_report("No sessions.", payload)
    lines = ["Sessions:"]
    for sid, meta in data.items():
        lines.append(f"- {sid}: has_network={meta['has_network']} nodes={meta['nodes']} edges={meta['edges']}")
    return structured_report("\n".join(lines), payload)

@mcp.tool(annotations=_READ_ONLY_CLOSED)
def list_artifact_sessions(
) -> Annotated[CallToolResult, NeKoArtifactSessionListResult]:
    """List all NeKo sessions that have artifact files on disk (including past server runs).

    Unlike list_sessions() which only shows in-memory sessions, this scans the
    artifacts/ directory and reads session_meta.json files, so previously created
    sessions are visible even after a server restart.

    Use the returned session_id and file paths to resume earlier work, e.g.:
      create_network(sif_file='/path/to/artifacts/<uuid>/Network.sif')
    """
    sessions = _list_artifact_sessions_on_disk(_SERVER_ROOT, server_name="NeKo")
    payload = NeKoArtifactSessionListResult(
        server="NeKo",
        count=len(sessions),
        sessions=[
            ArtifactSessionSummary(
                session_id=str(session["session_id"]),
                server=str(session.get("server") or "unknown"),
                label=str(session["label"]) if session.get("label") else None,
                created_at=(
                    str(session["created_at"])
                    if session.get("created_at")
                    else None
                ),
                files=[str(filename) for filename in session.get("files", [])],
            )
            for session in sessions
        ],
    )
    if not sessions:
        return structured_report("No artifact sessions found on disk.", payload)
    lines = ["## NeKo Artifact Sessions (on disk)\n"]
    for s in sessions:
        sid = s["session_id"]
        label = s.get("label") or ""
        created = s.get("created_at", "")[:19].replace("T", " ")  # trim to YYYY-MM-DD HH:MM:SS
        files = s.get("files", [])
        lines.append(f"- **{sid}**" + (f" ({label})" if label else ""))
        if created:
            lines.append(f"  Created: {created} UTC")
        if files:
            lines.append(f"  Files: {', '.join(files)}")
        else:
            lines.append("  Files: (none)")
    return structured_report("\n".join(lines), payload)

@mcp.tool(annotations=_IDEMPOTENT_CLOSED)
def set_default_session(
        session_id: Annotated[NonEmptyString, Field(description="Session ID to make the active default; used when session_id is omitted in subsequent tool calls.")]) -> str:
    """Set the default session used when session_id is omitted in other tool calls."""
    ok = session_manager.set_default(session_id)
    if not ok:
        raise ValueError(f"Session not found: {session_id}")
    return "Default set."

@mcp.tool(annotations=_DESTRUCTIVE_NON_IDEMPOTENT_CLOSED)
def delete_session(
        session_id: Annotated[NonEmptyString, Field(description="Session ID to permanently delete (irreversible).")]) -> str:
    """Permanently delete a session and its in-memory network."""
    ok = session_manager.delete_session(session_id)
    if not ok:
        raise ValueError(f"Session not found: {session_id}")
    return "Deleted."

@mcp.tool(annotations=_READ_ONLY_CLOSED)
@session_locked
def status(
        session_id: Optional[NonEmptyString] = Field(None, description="Session ID; omit to query the active/default session.")) -> Annotated[CallToolResult, NeKoNetworkStatusResult]:
    """Return a one-line session summary: session ID, node count, edge count."""
    sess, network = _session_network(session_id)
    if network is None:
        payload = NeKoNetworkStatusResult(
            server="NeKo",
            session_id=sess.session_id,
            has_network=False,
            node_count=0,
            interaction_count=0,
        )
        return structured_report(
            f"Session {sess.session_id}: no network.",
            payload,
        )
    df = sess.get_edges_df()
    edges = len(df) if df is not None else 0
    payload = NeKoNetworkStatusResult(
        server="NeKo",
        session_id=sess.session_id,
        has_network=True,
        node_count=len(network.nodes),
        interaction_count=edges,
    )
    text = (
        f"Session {sess.session_id}: "
        f"nodes={len(network.nodes)} edges={edges}."
    )
    return structured_report(text, payload)

# ===== Component & Strategy Tools =====
@mcp.tool(annotations=_READ_ONLY_CLOSED)
@session_locked
def analyze_connectivity(
        session_id: Optional[NonEmptyString] = Field(None, description="Session ID; omit to use the active/default session."),
        verbosity: NormalizedVerbosity = Field(DEFAULT_VERBOSITY, description="Output detail level: 'summary' (counts only), 'preview'/'full' (isolated node list and per-component stats)."),
        format: OutputFormat = Field("markdown", description="Output format: 'markdown' (default) or 'json'.")) -> Annotated[CallToolResult, NeKoConnectivityResult]:
    """Report isolated (0-edge) nodes AND the full connected-component partition.

    A network can have zero isolated nodes yet still be fragmented into
    several disconnected multi-node clusters (e.g. two unrelated 10-node
    islands). This tool reports both facets together, so "all_nodes_have_interactions"
    plus "component_count == 1" together confirm the network is fully connected.
    Use before choosing a connection strategy (see preview_connection_impact()).
    Each component's `component_id` is a report label only - to bridge two
    components with bridge_components(), pass the Gene Symbols listed in
    their `nodes`, not the `component_id` integers.
    """
    verbosity = normalize_verbosity(verbosity)
    sess, network = _session_network(session_id)
    if network is None:
        raise RuntimeError(E_NO_NET)

    u2s, _ = _get_translators(network)
    node_index = _node_record_index(network)

    all_nodes = {
        node
        for node in network.nodes["Uniprot"].tolist()
        if _optional_text(node) is not None
    }
    connected_nodes = (
        set(network.edges["source"].tolist())
        | set(network.edges["target"].tolist())
    )
    disconnected_uniprot = all_nodes - connected_nodes
    disconnected_nodes = [
        _node_record_for_identifier(
            node,
            node_index=node_index,
            uniprot_to_symbol=u2s,
        )
        for node in disconnected_uniprot
    ]
    disconnected_nodes.sort(
        key=lambda record: record.gene_symbol or record.uniprot or ""
    )

    comps = _compute_components(network)

    deg = {}
    try:
        sources = network.edges["source"].tolist()
        targets = network.edges["target"].tolist()
        for s, t in zip(sources, targets):
            if pd.notna(s) and s != "":
                s_str = str(s)
                deg[s_str] = deg.get(s_str, 0) + 1
            if pd.notna(t) and t != "":
                t_str = str(t)
                deg[t_str] = deg.get(t_str, 0) + 1
    except Exception:
        pass

    components = []
    for idx, comp in enumerate(comps):
        dvals = [deg.get(n, 0) for n in comp]
        avg_d = round(sum(dvals)/len(dvals), 2) if dvals else 0
        components.append(
            NeKoComponentRecord(
                component_id=idx,
                size=len(comp),
                average_degree=avg_d,
                nodes=[
                    _node_record_for_identifier(
                        node,
                        node_index=node_index,
                        uniprot_to_symbol=u2s,
                    )
                    for node in comp
                ],
            )
        )
    largest_component_size = max(
        (component.size for component in components),
        default=0,
    )
    payload = NeKoConnectivityResult(
        server="NeKo",
        session_id=sess.session_id,
        total_node_count=len(all_nodes),
        disconnected_count=len(disconnected_nodes),
        all_nodes_have_interactions=not disconnected_nodes,
        disconnected_nodes=disconnected_nodes,
        component_count=len(components),
        largest_component_size=largest_component_size,
        components=components,
    )

    if format == "json":
        text = json.dumps(payload.model_dump(mode="json"), separators=(",", ":"))
        return structured_report(text, payload)
    if verbosity == "summary":
        text = (
            f"Isolated nodes: {len(disconnected_nodes)}/{len(all_nodes)}. "
            f"Components={len(components)} largest={largest_component_size}. "
            f"{SUMMARY_HINT}"
        )
        return structured_report(text, payload)

    lines = []
    if disconnected_nodes:
        labels = [
            record.gene_symbol or record.uniprot or "(unknown)"
            for record in disconnected_nodes
        ]
        lines.append("Isolated nodes (Gene Symbols):\n" + "\n".join(labels))
    else:
        lines.append("No isolated nodes.")

    if not components:
        lines.append("No components (empty network).")
    else:
        component_lines = ["Components:"]
        for component in components:
            visible_nodes = (
                component.nodes[:5]
                if verbosity == "preview"
                else component.nodes
            )
            node_labels = [
                node.gene_symbol or node.uniprot or "(unknown)"
                for node in visible_nodes
            ]
            label = "sample" if verbosity == "preview" else "nodes"
            component_lines.append(
                f"- {component.component_id}: size={component.size} "
                f"avg_deg={component.average_degree} {label}={node_labels}"
            )
        lines.append("\n".join(component_lines))

    return structured_report("\n\n".join(lines), payload)

@mcp.tool(annotations=_READ_ONLY_OPEN)
@session_locked
def preview_connection_impact(
        method: NormalizedConnectorMethod = Field("hubs", description="Suggestion strategy: 'hubs' (rank high-degree nodes), 'relax_max_len' (simulate +1 max_len), 'unsigned' (simulate allowing unsigned interactions)."),
        top_k: int = Field(10, ge=1, description="Number of hub genes to report when method='hubs'."),
        session_id: Optional[NonEmptyString] = Field(None, description="Session ID; omit to use the active/default session."),
        format: OutputFormat = Field("markdown", description="Output format: 'markdown' (default) or 'json'."),
        verbosity: NormalizedVerbosity = Field(DEFAULT_VERBOSITY, description="Output detail level: 'summary', 'preview', or 'full'.")) -> Annotated[CallToolResult, NeKoConnectionPreviewResult]:
    """Preview possible repairs for a disconnected network without mutating it.

    Non-mutating scout: rank hub genes by degree, or simulate a parameter
    relaxation (relax_max_len/unsigned) on an in-memory copy to preview the
    predicted edge-count delta. Run before applying a connection strategy
    (connect_targeted_nodes, bridge_components, apply_global_connection) to
    estimate the benefit without committing to changes. Outputs Gene Symbols
    for readability.
    """
    verbosity = normalize_verbosity(verbosity)
    sess, network = _session_network(session_id)
    if network is None:
        raise RuntimeError(E_NO_NET)
        
    method = method.lower()
    
    u2s, _ = _get_translators(network)
    
    hub_candidates = []
    simulation = None
    rationale = ""
    
    # --- HUBS METHOD ---
    if method == 'hubs':
        deg = {}
        try:
            # Calculate degrees natively using Uniprot IDs from the edges dataframe
            sources = network.edges["source"].tolist()
            targets = network.edges["target"].tolist()
            for s, t in zip(sources, targets):
                if pd.notna(s) and s != "": deg[s] = deg.get(s, 0) + 1
                if pd.notna(t) and t != "": deg[t] = deg.get(t, 0) + 1
        except Exception:
            pass
            
        if not deg:
            payload = NeKoConnectionPreviewResult(
                server="NeKo",
                session_id=sess.session_id,
                method=method,
                rationale="No edge data are available to calculate hubs.",
                suggestion_count=0,
                hub_candidates=[],
            )
            if format == "json":
                text = json.dumps(
                    payload.model_dump(mode="json"),
                    separators=(",", ":"),
                )
                return structured_report(text, payload)
            return structured_report(
                "No edge data available to calculate hubs.",
                payload,
            )

        maxd = max(deg.values()) if deg else 1
        
        # Sort by degree and take top_k
        ranked_uniprot = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # TRANSLATE BACK TO GENE SYMBOLS for the LLM
        for uid, d in ranked_uniprot:
            uniprot = _required_text(uid, field_name="hub identifier")
            hub_candidates.append(
                NeKoHubCandidate(
                    gene_symbol=_optional_text(u2s.get(uid)),
                    uniprot=uniprot,
                    relative_score=round(d / maxd, 3),
                    degree=d,
                )
            )
            
        rationale = 'High-degree nodes (hubs) may naturally act as bridges between isolated components.'

    # --- SIMULATION METHODS ---
    elif method in ('relax_max_len', 'unsigned'):
        try:
            # IN-MEMORY COPY
            net_copy = copy.deepcopy(network)
            params = sess.get_completion_params()
            
            if method == 'relax_max_len':
                params['maxlen'] = params.get('maxlen', 2) + 1
                rationale = f"Simulating connection expansion by increasing max path length to {params['maxlen']}."
            if method == 'unsigned':
                params['only_signed'] = False
                rationale = "Simulating connection expansion by allowing unsigned (unverified direction) interactions."
                
            before_e = len(net_copy.edges)
            
            # Run the completion strategy on the dummy copy
            net_copy.complete_connection(**params)
            after_e = len(net_copy.edges)
            
            simulation = NeKoConnectorSimulation(
                predicted_new_edges=max(after_e - before_e, 0),
                simulated_max_length=params.get("maxlen"),
                simulated_only_signed=params.get("only_signed"),
            )
            
        except Exception as e:
            raise RuntimeError(f"Connector simulation failed: {e}") from e
    else:
        raise ValueError(
            "Unsupported connector method. "
            "Use 'hubs', 'relax_max_len', or 'unsigned'."
        )

    suggestion_count = len(hub_candidates) if method == "hubs" else int(
        simulation is not None
    )
    payload = NeKoConnectionPreviewResult(
        server="NeKo",
        session_id=sess.session_id,
        method=method,
        rationale=rationale,
        suggestion_count=suggestion_count,
        hub_candidates=hub_candidates,
        simulation=simulation,
    )

    # --- FORMAT OUTPUT ---
    if format == "json":
        text = json.dumps(payload.model_dump(mode="json"), separators=(",", ":"))
        return structured_report(text, payload)
    if verbosity == "summary":
        text = f"{method}: {suggestion_count} suggestions. {SUMMARY_HINT}"
        return structured_report(text, payload)

    lines = [f"Candidate connectors ({method}):"]
    for candidate in hub_candidates:
        label = candidate.gene_symbol or candidate.uniprot
        lines.append(
            f"- **{label}**: relative_score={candidate.relative_score} "
            f"(edges={candidate.degree})"
        )
    if simulation is not None:
        lines.append(
            f"- Predicted new edges: {simulation.predicted_new_edges}"
        )
        lines.append(
            "- Parameters simulated: "
            f"max_len={simulation.simulated_max_length}, "
            f"only_signed={simulation.simulated_only_signed}"
        )
            
    if rationale:
        lines.append(f"\n*Rationale: {rationale}*")
        
    return structured_report("\n".join(lines), payload)

@mcp.tool(annotations=_NON_IDEMPOTENT_OPEN)
@session_locked
def bridge_components(
        comp_a: NonEmptyStringList = Field(..., description="First group: a list of actual Gene Symbols already present in the network (e.g. ['TP53', 'MDM2']). This is NOT the integer 'component_id' reported by analyze_connectivity() - pass the gene names belonging to that component instead."),
        comp_b: NonEmptyStringList = Field(..., description="Second group: a list of actual Gene Symbols already present in the network (e.g. ['EGFR', 'AKT1']). This is NOT the integer 'component_id' reported by analyze_connectivity() - pass the gene names belonging to that component instead."),
        max_len: int = Field(2, ge=1, description="Maximum path length for connecting edges."),
        mode: BridgeMode = Field("OUT", description="Edge direction mode: 'OUT', 'IN', or 'ALL'."),
        only_signed: Optional[bool] = Field(None, description="Restrict to signed interactions."),
        consensus: Optional[bool] = Field(None, description="Require multi-source consensus."),
        session_id: Optional[NonEmptyString] = Field(None, description="Session ID.")) -> str:
    """Connect two named groups of genes (e.g. two disconnected components) together.

    comp_a/comp_b must be the Gene Symbols themselves (as found in
    list_genes_and_interactions() or analyze_connectivity()'s `nodes` lists),
    never the numeric `component_id` that analyze_connectivity() reports for
    each component - that ID is a label for humans, not a valid gene name.
    """
    sess, network = _session_network(session_id)
    if network is None:
        raise RuntimeError(E_NO_NET)
    
    params = sess.get_completion_params()
    only_signed = only_signed if only_signed is not None else params.get('only_signed', True)
    consensus = consensus if consensus is not None else params.get('consensus', True)

    # 1. TRANSLATION LAYER: Gene Symbols -> Uniprot
    _, s2u = _get_translators(network)
    uniprot_a = [s2u.get(gene, gene) for gene in comp_a] # Fallback to input if not found
    uniprot_b = [s2u.get(gene, gene) for gene in comp_b]

    # 2. BACKEND MATH: Run strictly in Uniprot IDs
    try:
        network.connect_component(
            uniprot_a,
            uniprot_b,
            maxlen=max_len,
            mode=mode,
            only_signed=only_signed,
            consensus=consensus,
        )
        _invalidate(sess)
        
        df = sess.get_edges_df()
        return f"Successfully bridged components. Network now has {len(df) if df is not None else 0} edges."
    except Exception as e:
        raise RuntimeError(f"Bridging failed: {e}") from e

@mcp.tool(annotations=_NON_IDEMPOTENT_OPEN)
@session_locked
def connect_targeted_nodes(
        strategy: Annotated[TargetStrategy, Field(description="Targeted strategy.")],
        nodes: NonEmptyStringList = Field(..., description="Non-empty target genes (Gene Symbols) to connect or expand."),
        max_len: int = Field(1, ge=1, description="Max path length or upstream depth."),
        only_signed: Optional[bool] = Field(None, description="Override the session default and keep only signed interactions."),
        consensus: Optional[bool] = Field(None, description="Override the session default and require consensus-supported interactions."),
        session_id: Optional[NonEmptyString] = Field(None, description="Session ID to update; omit to use the active/default session.")) -> str:
    """Integrate specific gene(s) into the existing network (upstream regulators or a dense subgroup).

    For whole-network closure strategies (including output-anchored topology
    mapping) use apply_global_connection() instead. See the cost guide in
    docs://neko/agent_manual before choosing a strategy on a large network.
    """
    sess, network = _session_network(session_id)
    if network is None:
        raise RuntimeError(E_NO_NET)
    
    params = sess.get_completion_params()
    osgn = only_signed if only_signed is not None else params.get('only_signed', True)
    cons = consensus if consensus is not None else params.get('consensus', True)

    # 1. TRANSLATION LAYER: Gene Symbols -> Uniprot
    _, s2u = _get_translators(network)
    uniprot_nodes = [s2u.get(n, n) for n in nodes]

    # 2. BACKEND MATH
    try:
        if strategy == "connect_to_upstream_nodes":
            network.connect_to_upstream_nodes(
                nodes_to_connect=uniprot_nodes,
                depth=max_len,
                only_signed=osgn,
                consensus=cons,
            )
        elif strategy == "connect_subgroup":
            network.connect_subgroup(
                group=uniprot_nodes,
                maxlen=max_len,
                only_signed=osgn,
                consensus=cons,
            )
        else:
            raise ValueError(
                "Unsupported targeted strategy. Use "
                "'connect_to_upstream_nodes' or 'connect_subgroup'."
            )

        _invalidate(sess)
        return f"Applied {strategy} to targeted nodes."
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Targeted strategy failed: {e}") from e

@mcp.tool(annotations=_NON_IDEMPOTENT_OPEN)
@session_locked
def apply_global_connection(
        strategy: Annotated[GlobalStrategy, Field(description="Global connection strategy.")],
        max_len: int = Field(2, ge=1, description="Maximum path length to search for connections."),
        algorithm: SearchAlgorithm = Field("bfs", description="[complete_connection] Search algorithm: 'bfs' or 'dfs'."),
        minimal: bool = Field(True, description="[complete_connection] Add only minimum required edges."),
        direction: RadialDirection = Field("OUT", description="[connect_network_radially] Growth direction ('OUT' or 'IN')."),
        strategy_mode: Optional[AtopoStrategy] = Field(None, description="[connect_as_atopo] Underlying closure strategy to run first: 'radial' or 'complete'."),
        outputs: Optional[NonEmptyStringList] = Field(None, description="[connect_as_atopo] Non-empty output gene symbols to anchor the topology; the network is grown until connected to every output."),
        only_signed: Optional[bool] = Field(None, description="Override the session default and keep only signed interactions."),
        consensus: Optional[bool] = Field(None, description="Override the session default and require consensus-supported interactions."),
        session_id: Optional[NonEmptyString] = Field(None, description="Session ID to update; omit to use the active/default session.")) -> str:
    """Apply a whole-network closure strategy to resolve missing edges.

    'complete_connection' is the most expensive strategy (O(N^2) over every
    node pair) and is the most likely cause of runaway network growth on large
    seed sets. 'connect_as_atopo' loops until the network is fully connected
    to the declared outputs, so its cost is open-ended. Prefer
    connect_targeted_nodes() or bridge_components() to close specific gaps on
    large networks. See the cost guide in docs://neko/agent_manual.
    """
    sess, network = _session_network(session_id)
    if network is None:
        raise RuntimeError(E_NO_NET)
    
    params = sess.get_completion_params()
    osgn = only_signed if only_signed is not None else params.get('only_signed', True)
    cons = consensus if consensus is not None else params.get('consensus', True)

    # TRANSLATION LAYER (only needed for output-anchored atopo)
    _, s2u = _get_translators(network)
    uniprot_outputs = [s2u.get(o, o) for o in outputs] if outputs else None

    try:
        if strategy == "complete_connection":
            network.complete_connection(
                maxlen=max_len,
                algorithm=algorithm,
                minimal=minimal,
                only_signed=osgn,
                consensus=cons,
            )
        elif strategy == "connect_network_radially":
            network.connect_network_radially(
                max_len=max_len,
                direction=direction,
                only_signed=osgn,
                consensus=cons,
            )
        elif strategy == "connect_as_atopo":
            network.connect_as_atopo(
                strategy=strategy_mode,
                max_len=max_len,
                outputs=uniprot_outputs,
                only_signed=osgn,
                consensus=cons,
            )
        else:
            raise ValueError(
                "Unsupported global strategy. Use 'complete_connection', "
                "'connect_network_radially', or 'connect_as_atopo'."
            )

        _invalidate(sess)
        df = sess.get_edges_df()
        return f"Successfully applied global {strategy}. Edges now = {len(df) if df is not None else 0}."
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Global strategy failed: {e}") from e

if __name__ == "__main__":
    mcp.run()

