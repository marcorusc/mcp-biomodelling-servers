import io
import copy
import sys
import os
import glob
import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Annotated, Optional, List
from pydantic import Field

from neko.core.network import Network
from neko._outputs.exports import Exports
from neko.inputs import Universe, signor
from neko.core.tools import is_connected

from utils import *
from session_manager import session_manager, ensure_session, normalize_verbosity, DEFAULT_VERBOSITY

# Make the repo root importable so we can use the shared artifact_manager
sys.path.insert(0, str(Path(__file__).parent.parent))
from artifact_manager import get_artifact_dir, safe_artifact_path, list_artifacts, clean_artifacts, write_session_meta, list_artifact_sessions as _list_artifact_sessions_on_disk
from mcp_biomodelling_servers import __version__

from src.helpers import (
    E_NO_NET, SUMMARY_HINT, _SERVER_ROOT,
    _short_table, _export_dir, _session_network, _invalidate,
    _compute_components, requires_network,
    sanitize_bnet_file, _get_translators
)
from neko.core.strategies import (
    connect_as_atopo,
    connect_component as strategy_connect_component,
    complete_connection as strategy_complete_connection,
    connect_network_radially,
    connect_to_upstream_nodes,
    connect_subgroup,
)

import pandas as pd


from mcp.server.mcpserver import Context, MCPServer

logger = logging.getLogger(__name__)

mcp = MCPServer(
    "NeKo",
    title="NeKo Signalling Network Builder",
    description=(
        "Build and analyze signalling networks from biological interaction databases."
    ),
    version=__version__,
)

# NOTE: Previous implementation used a single global `network` object.
# Now session-based management (see `session_manager.py`).
# Each tool can accept an optional `session_id` allowing multiple networks.
# If not provided, the default session is used (auto-created on first use).

# 1. Define the manual in one place so you only ever have to update it here.
NEKO_AGENT_MANUAL = """
# NeKo to MaBoSS Workflow Manual

## 1. Recommended Execution Order
1. **Initialize:** `create_session()` -> `set_default_params(max_len=2, only_signed=True, consensus=True)`
2. **Build:** `create_network([...seed genes...], database='omnipath')`
3. **Curate:** `remove_bimodal_interactions()` -> `remove_undefined_interactions()`
4. **Audit Connectivity:** `check_disconnected_nodes()`
   - *If disconnected:* `list_components()` -> `candidate_connectors()` -> Apply a connection tool.
5. **Inspect:** `list_genes_and_interactions(verbosity='preview')`
6. **Export:** `export_network(format='bnet')` -> Pass to MaBoSS.

## 2. Tool Categories
* **Sessions:** `create_session`, `list_sessions`, `set_default_session`, `delete_session`, `status`, `reset_network`
* **Connection Solvers:** `bridge_components`, `connect_targeted_nodes`, `apply_global_connection`
* **Inspection:** `list_genes_and_interactions`, `find_paths`, `get_references`, `filter_interactions`

## 3. Critical Operating Rules
* **Session First:** Always call `create_session` before `create_network`.
* **Scout Before You Shoot:** Always run `candidate_connectors()` before heavy connection tools.
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

@mcp.tool()
async def create_network(
                   list_of_initial_genes: Annotated[List[str], Field(description="Gene symbols to seed the network (e.g. ['TP53', 'MYC', 'CASP3']). Can be empty if sif_file is provided.")],
                   ctx: Context,
                   database: str = Field("omnipath", description="Knowledge-base to query. 'omnipath' (default) or 'signor'."),
                   sif_file: Optional[str] = Field(None, description="Absolute path to an existing SIF file to bootstrap the network from. Combined with list_of_initial_genes when both are given."),
                   max_len: int = Field(2, description="Maximum path length used by complete_connection to bridge seed genes (1-4; larger = denser but slower)."),
                   algorithm: str = Field("bfs", description="Search algorithm for path completion: 'bfs' (breadth-first, default) or 'dfs' (depth-first)."),
                   only_signed: bool = Field(True, description="Restrict to signed (+/-) interactions only. Set False to allow unsigned interactions when network is sparse."),
                   connect_with_bias: bool = Field(False, description="Avoids looking for paths between pairs of nodes that are already connected by another path previously found."),
                   consensus: bool = Field(True, description="Require interactions supported by multiple curated sources (higher confidence)."),
                   session_id: Optional[str] = Field(None, description="Session ID to write the network into. Omit to use the active/default session."),
                   verbosity: str = Field(DEFAULT_VERBOSITY, description="Output detail level: 'summary' (default, token-frugal), 'preview' (truncated tables), 'full'.")) -> str:
    """Build a NeKo gene regulatory network from seed genes and/or a SIF file.

    Calls complete_connection internally to bridge genes via the chosen database.
    After creation, run remove_bimodal_interactions() and check_disconnected_nodes()
    before exporting. Always call create_session() first.
    """
    verbosity = normalize_verbosity(verbosity)
    sess = ensure_session(session_id)
    await ctx.report_progress(0, 4)
    logger.info(
        "Creating NeKo network (session=%s) with genes=%s sif=%s",
        sess.session_id,
        list_of_initial_genes,
        sif_file,
    )
    # Validate database choice
    if database not in ["omnipath", "signor"]:
        return "_Unsupported database. Use `omnipath` or `signor`._"

    # If using SIGNOR, download and build the SIGNOR resource
    if database == "signor":
        logger.info("Downloading SIGNOR database")
        signor_res = signor()
        signor_res.build()
        logger.info("SIGNOR database downloaded successfully")
        resources = signor_res.interactions
    else:
        resources = "omnipath"

    await ctx.report_progress(1, 4)
    # Case 1: SIF file provided (with or without genes)
    if sif_file is not None and os.path.exists(sif_file):
        # Use NeKo's documented SIF loading
        try:
            sess.set_network(Network(sif_file=sif_file, resources=resources))
        except Exception as e:
            return f"**Error**: Unable to create network from SIF file. {str(e)}"
        # If genes are also provided, add them
        if list_of_initial_genes:
            for gene in list_of_initial_genes:
                try:
                    sess.network.add_node(gene)
                except Exception:
                    pass
    # Case 2: Only genes provided
    elif list_of_initial_genes:
        sess.set_network(Network(list_of_initial_genes, resources=resources))
        logger.info(
            "Running complete_connection (max_len=%s, only_signed=%s)",
            max_len,
            only_signed,
        )
        sess.network.complete_connection(
            maxlen=max_len,
            algorithm=algorithm,
            only_signed=only_signed,
            connect_with_bias=connect_with_bias,
            consensus=consensus
        )
    # Case 3: Neither provided - enhanced guidance
    else:
        return format_no_input_guidance()

    # If there are no edges, return enhanced guidance instead of dead-end error
    try:
        df_edges = sess.get_edges_df()
    except Exception as e:
        return format_network_creation_error("build_failed", list_of_initial_genes, str(e))

    if df_edges.empty:
        logger.warning(
            "No interactions found in the network; check the input parameters"
        )
        return format_empty_network_response(list_of_initial_genes, database, max_len, only_signed)

    # Compute basic statistics
    num_edges = len(df_edges)
    unique_nodes = pd.unique(df_edges[["source", "target"]].values.ravel())
    num_nodes = len(unique_nodes)

    await ctx.report_progress(4, 4)
    logger.info(
        "Network created successfully: %s nodes, %s edges",
        num_nodes,
        num_edges,
    )

    # Prepare a preview of the first 100 interactions
    if verbosity == "summary":
        return (f"Network created: session={sess.session_id} nodes={num_nodes} edges={num_edges}. "
                f"Disconnected components check via check_disconnected_nodes(). {SUMMARY_HINT}")
    # Build preview for preview/full
    preview_df = df_edges[[c for c in ['source', 'target', 'Effect'] if c in df_edges.columns]].head(100)
    preview_md = clean_for_markdown(preview_df).to_markdown(index=False, tablefmt="plain")
    lines = [f"Network created (session={sess.session_id})",
             f"Initial genes: {', '.join(list_of_initial_genes)}",
             f"Nodes: {num_nodes} | Edges: {num_edges}"]
    if verbosity == "preview":
        lines.append("Preview (first 100):\n" + preview_md)
    elif verbosity == "full":
        lines.append("Full preview (first 100 interactions):\n" + preview_md)
        lines.append(f"Parameters: database={database} max_len={max_len} algorithm={algorithm} only_signed={only_signed} consensus={consensus}")
    return "\n".join(lines)

@mcp.tool()
@requires_network
def add_gene(
        gene: Annotated[str, Field(description="Gene symbol to add (e.g. 'TP53'). Case-sensitive; use uppercase HGNC symbols.")],
        session_id: Optional[str] = Field(None, description="Session ID; omit to use the active/default session."),
        autoconnect: bool = Field(False, description="Re-run complete_connection after adding the gene to integrate it into the network topology."),
        sess=None, network=None) -> str:
    """Add a single gene node to the current network.

    To add multiple genes at once use extend_network().
    """
    try:
        network.add_node(gene)
        if autoconnect:
            try:
                # Attempt to complete connections again using session defaults
                params = sess.get_completion_params()
                network.complete_connection(**params)
            except Exception:
                pass
        _invalidate(sess)
        return f"Gene added: {gene}.{' Autoconnect attempted.' if autoconnect else ''} {SUMMARY_HINT}"
    except Exception as e:
        return f"**Error adding gene {gene}:** {str(e)}\n**Tip:** Ensure gene symbol is valid (e.g., 'TP53', not 'tp53')"

@mcp.tool()
@requires_network
def remove_gene(
        gene: Annotated[str, Field(description="Gene symbol to remove. Case-insensitive; closest match is suggested if not found.")],
        session_id: Optional[str] = Field(None, description="Session ID; omit to use the active/default session."),
        sess=None, network=None) -> str:
    """Remove a gene node (and all its edges) from the current network.

    Use list_genes_and_interactions() first to verify the exact symbol present in the network.
    """
    try:
        df_nodes = network.nodes if hasattr(network, 'nodes') else None
        if df_nodes is None:
            return "**Error:** Network node table unavailable."

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
            return msg

        # Use original casing for removal if stored
        original_name = symbol_map.get(query, gene)
        network.remove_node(original_name)
        _invalidate(sess)
        return f"Gene removed: {original_name}."
    except Exception as e:
        return f"**Error removing gene {gene}:** {str(e)}"

@mcp.tool()
@requires_network
def remove_interaction(
        node_A: Annotated[str, Field(description="Source gene symbol (interaction goes A -> B).")],
        node_B: Annotated[str, Field(description="Target gene symbol (interaction goes A -> B).")],
        session_id: Optional[str] = Field(None, description="Session ID; omit to use the active/default session."),
        sess=None, network=None) -> str:
    """Remove a directed edge A -> B from the network (does not affect B -> A if it exists).

    Use filter_interactions() or list_genes_and_interactions() to locate the exact edge first.
    """
    try:
        df_edges = network.convert_edgelist_into_genesymbol()
    except Exception as e:
        return f"**Error**: Unable to retrieve network edges. {str(e)}" 
    if df_edges.empty:
        return "_No interactions found in the network._"
    # Check if the interaction exists in the specified direction
    mask = (df_edges['source'] == node_A) & (df_edges['target'] == node_B)
    if not df_edges[mask].empty:
        try:
            network.remove_edge(node_A, node_B)
            _invalidate(sess)
            return f"Interaction removed: {node_A}->{node_B}."
        except Exception as e:
            return f"**Error removing interaction {node_A} -> {node_B}:** {str(e)}"
    else:
        return f"**Interaction not found:** No interaction from {node_A} to {node_B} in the current network.\n**Tip:** Use `get_network()` to see all available interactions"

# TO DO: Implement export of images with graphviz

# TO DO: implement GO enrichment

@mcp.tool()
def export_network(
        format: str = Field("sif", description="Export format: 'sif' (Simple Interaction Format, tab-separated) or 'bnet' (Boolean network for MaBoSS). BNET requires a fully connected network."),
        session_id: Optional[str] = Field(None, description="Session ID; omit to use the active/default session."),
        verbosity: str = Field(DEFAULT_VERBOSITY, description="Output detail level: 'summary', 'preview', or 'full'.")) -> str:
    """Export the current network to SIF or BNET format.

    After BNET export, hand the file path to the MaBoSS server via bnet_to_bnd_and_cfg().
    BNET export fails if the network is not fully connected — run check_disconnected_nodes() first.
    """
    verbosity = normalize_verbosity(verbosity)
    sess, network = _session_network(session_id)
    if network is None:
        return format_no_network_guidance()
    exporter = Exports(network)
    out_dir = _export_dir(sess.session_id)

    # ── SIF export ────────────────────────────────────────────────────────────
    if format.lower() == "sif":
        out_path = str(out_dir / "Network.sif")
        try:
            exporter.export_sif(out_path)
        except Exception as e:
            return f"**Error exporting SIF:** {e}"
        if verbosity == "summary":
            return f"SIF exported: {out_path}. {SUMMARY_HINT}"
        try:
            df_prev = pd.read_csv(out_path, sep="\t", header=None,
                                  names=["source", "interaction", "target"],
                                  nrows=100, dtype=str).dropna(how="all")
            preview_md = _short_table(df_prev, max_rows=100)[0]
        except Exception:
            preview_md = "_Preview unavailable._"
        return f"SIF exported: `{out_path}`\n\nPreview (first 100 rows):\n{preview_md}"

    # ── BNET export ───────────────────────────────────────────────────────────
    elif format.lower() == "bnet":
        if not is_connected(network):
            return format_connectivity_guidance()
        try:
            exporter.export_bnet(str(out_dir / "Network"))
        except Exception as e:
            return f"**Error exporting BNET:** {e}"

        bnet_files = sorted(out_dir.glob("*.bnet"))
        if not bnet_files:
            return "**Error:** No .bnet files were generated."
        out_path = str(bnet_files[0])

        try:
            result = sanitize_bnet_file(out_path)
        except Exception as e:
            return f"**Error sanitizing BNET:** {e}"

        if verbosity == "summary":
            return f"BNET exported: {out_path}. {SUMMARY_HINT}"

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
        return "\n".join(md_lines)

    # ── Unsupported format ────────────────────────────────────────────────────
    else:
        return format_unsupported_format_guidance(format)

@mcp.tool()
def list_genes_and_interactions(
        session_id: Optional[str] = Field(None, description="Session ID; omit to use the active/default session."),
        verbosity: str = Field(DEFAULT_VERBOSITY, description="Output detail level: 'summary' (counts only), 'preview' (truncated table), 'full' (up to 100 rows)."),
        max_rows: int = Field(50, description="Maximum rows to return in preview/full mode.")) -> str:
    """Return a Markdown table of all nodes and directed edges in the network.

    Equivalent to 'show the network'. Use filter_interactions() for targeted queries.
    """
    verbosity = normalize_verbosity(verbosity)
    sess, network = _session_network(session_id)
    if network is None:
        cols = ["source", "target", "Type", "Effect", "References"]
        empty_df = pd.DataFrame(columns=cols)
        return "_No network loaded._\n\n" + clean_for_markdown(empty_df).to_markdown(index=False, tablefmt="plain")

    try:
        df = sess.get_edges_df()
        if "resources" in df.columns:
            df = df.drop(columns=["resources"])
        if df.empty:
            return "_Network loaded but contains no interactions._\n\n" + clean_for_markdown(df).to_markdown(index=False, tablefmt="plain")
        df = df[[c for c in ['source', 'target', 'Effect'] if c in df.columns]]
        if verbosity == "summary":
            return f"Interactions: {len(df)}. {SUMMARY_HINT}"
        table, truncated = _short_table(df, max_rows=max_rows if verbosity == "preview" else 100)
        note = " (truncated)" if truncated else ""
        return table + note
    except Exception as e:
        # Try to get column headers from a failed conversion (fallback)
        try:
            cols = network.convert_edgelist_into_genesymbol().columns
            if "resources" in cols:
                cols = [c for c in cols if c != "resources"]
        except:
            cols = ["source", "target", "Type", "Effect", "References"]
        empty_df = pd.DataFrame(columns=cols)
        return f"**Error**: {str(e)}\n\n_Unable to retrieve data._\n\n" + clean_for_markdown(empty_df).to_markdown(index=False, tablefmt="plain")

@mcp.tool()
def find_paths(
        source: Annotated[str, Field(description="Source gene symbol (path start).")],
        target: Annotated[str, Field(description="Target gene symbol (path end).")],
        maxlen: int = Field(3, description="Maximum number of edges in a path (1-5; longer paths are slower to compute)."),
        session_id: Optional[str] = Field(None, description="Session ID; omit to use the active/default session."),
        verbosity: str = Field(DEFAULT_VERBOSITY, description="Output detail level: 'summary' (count only), 'preview'/'full' (path listing).")) -> str:
    """Find and display all directed paths between two genes up to maxlen edges.

    Useful for verifying biological signal flow before BNET export.
    Returns 'No paths found' if genes are in disconnected components.
    """
    verbosity = normalize_verbosity(verbosity)
    sess, network = _session_network(session_id)
    if network is None:
        return E_NO_NET
    buffer = io.StringIO()
    old_stdout = sys.stdout
    try:
        # Redirect stdout to our buffer
        sys.stdout = buffer
        network.print_my_paths(source, target, maxlen=maxlen)
        sys.stdout = old_stdout  # restore immediately after printing

        raw_output = buffer.getvalue().strip()
        buffer.close()

        if not raw_output:
            return "No paths found."
        if verbosity == "summary":
            # Count lines starting with PATH or similar
            lines = [l for l in raw_output.splitlines() if l.strip()]
            return f"Found {len(lines)} path lines. {SUMMARY_HINT}"
        label = "Paths" if verbosity == "preview" else "Paths (full output)"
        return f"{label}:\n```\n{raw_output}\n```"

    except Exception as e:
        # Ensure stdout is restored even on error
        sys.stdout = old_stdout
        return f"**Error:** {str(e)}"

@mcp.tool()
def reset_network(
        session_id: Optional[str] = Field(None, description="Session ID to reset; omit to use the active/default session.")) -> str:
    """Discard the current network in the session without deleting the session itself.

    Use delete_session() to remove the session entirely, or create_network() to rebuild.
    """
    sess = ensure_session(session_id)
    sess.set_network(None)
    return f"Session {sess.session_id} network reset."

@mcp.tool()
def clean_generated_files(
        session_id: Optional[str] = Field(None, description="Session ID whose artifact files (SIF, BNET, etc.) should be removed. Omit for the active/default session.")) -> str:
    """Delete all exported artifact files (SIF, BNET) for the given session."""
    sess = ensure_session(session_id)
    try:
        count = clean_artifacts(_SERVER_ROOT, sess.session_id)
        return f"Cleaned {count} artifact file(s) from session {sess.session_id}."
    except Exception as e:
        return f"Error during cleanup: {str(e)}"

@mcp.tool()
def remove_bimodal_interactions(
        session_id: Optional[str] = Field(None, description="Session ID; omit to use the active/default session.")) -> str:
    """Remove all bimodal (simultaneously activating and inhibiting) edges from the network.

    Bimodal interactions are ambiguous and cause contradictory Boolean rules in BNET export.
    Run this as part of the standard curation step after create_network().
    """
    sess, network = _session_network(session_id)
    if network is None:
        return E_NO_NET
    if "Effect" not in network.edges.columns:
        return "No 'Effect' column found in network.edges."
    before = len(network.edges)
    network.remove_bimodal_interactions()
    _invalidate(sess)
    after = len(network.edges)
    removed = before - after
    return f"Removed {removed} bimodal interactions from the network."

@mcp.tool()
def remove_undefined_interactions(
        session_id: Optional[str] = Field(None, description="Session ID; omit to use the active/default session.")) -> str:
    """Remove all edges whose Effect is 'undefined' (unknown sign) from the network.

    Undefined interactions cannot be mapped to Boolean activations or inhibitions.
    Run after remove_bimodal_interactions() in the standard curation sequence.
    """
    sess, network = _session_network(session_id)
    if network is None:
        return E_NO_NET
    if "Effect" not in network.edges.columns:
        return "No 'Effect' column found in network.edges."
    before = len(network.edges)
    network.remove_undefined_interactions()
    _invalidate(sess)
    after = len(network.edges)
    removed = before - after
    return f"Removed {removed} undefined interactions from the network."


@mcp.tool()
def list_bnet_files(
        session_id: Optional[str] = Field(None, description="Session ID to query; omit to use the active/default session.")) -> str:
    """List names of all .bnet files in the session artifact directory (newline-separated)."""
    sess = ensure_session(session_id)
    art_dir = get_artifact_dir(_SERVER_ROOT, sess.session_id)
    files = [f.name for f in art_dir.glob("*.bnet")]
    if not files:
        return f"No .bnet files found in session {sess.session_id} artifact directory."
    return "\n".join(files)

@mcp.tool()
def check_disconnected_nodes(
        session_id: Optional[str] = Field(None, description="Session ID; omit to use the active/default session.")) -> str:
    """List any nodes in the network that have no edges (isolated nodes)."""
    sess, network = _session_network(session_id)
    if network is None:
        return E_NO_NET
    
    u2s, s2u = _get_translators(network)
    
    all_nodes = set(network.nodes["Uniprot"].tolist())
    if is_connected(network):
        return "All nodes are connected."
        
    connected_nodes = set(network.edges["source"].tolist()) | set(network.edges["target"].tolist())
    disconnected_uniprot = all_nodes - connected_nodes
    
    disconnected_uniprot = [node for node in disconnected_uniprot if pd.notna(node) and node != ""]
    
    if not disconnected_uniprot:
        return "All nodes are connected."
    
    # If a Uniprot ID lacks a symbol, we fall back to showing the Uniprot ID
    disconnected_symbols = [u2s.get(uid, uid) for uid in disconnected_uniprot]
    
    disconnected_symbols.sort()
    
    return "Disconnected nodes (Gene Symbols):\n" + "\n".join(disconnected_symbols)

@mcp.tool()
def get_references(
        node1: Annotated[str, Field(description="Gene symbol. Returns all edges where this gene is source or target.")],
        node2: Optional[str] = Field(None, description="Second gene symbol. When provided, returns only edges between node1 and node2 (either direction)."),
        session_id: Optional[str] = Field(None, description="Session ID; omit to use the active/default session."),
        verbosity: str = Field(DEFAULT_VERBOSITY, description="Output detail level: 'summary' (count only), 'preview'/'full' (Markdown table).")) -> str:
    """Show literature references for interactions involving one or two genes.

    References are truncated to the first 5 per edge with a count of remaining.
    Useful for assessing interaction evidence before pruning.
    """
    verbosity = normalize_verbosity(verbosity)
    sess, network = _session_network(session_id)
    if network is None:
        return E_NO_NET
    try:
        df = network.convert_edgelist_into_genesymbol()
    except Exception as e:
        return f"**Error**: Unable to retrieve network edges. {str(e)}"
    if df.empty:
        return "_No interactions found in the network._"
    # Filter by node(s)
    if node2:
        mask = ((df['source'] == node1) & (df['target'] == node2)) | ((df['source'] == node2) & (df['target'] == node1))
        filtered = df[mask]
    else:
        mask = (df['source'] == node1) | (df['target'] == node1)
        filtered = df[mask]
    if filtered.empty:
        return "No matching interactions."
    # Only keep relevant columns
    cols = ['source', 'target', 'Effect', 'References']
    filtered = filtered[[c for c in cols if c in filtered.columns]]
    # Truncate references for display
    def short_refs(refs):
        if pd.isna(refs) or not refs or refs == 'None':
            return ''
        ref_list = [r.strip() for r in str(refs).replace(';', ',').split(',') if r.strip()]
        if len(ref_list) > 5:
            return '; '.join(ref_list[:5]) + f" (+{len(ref_list)-5} more)"
        return '; '.join(ref_list)
    filtered['References'] = filtered['References'].apply(short_refs)
    # Clean for markdown
    if verbosity == "summary":
        return f"References: {len(filtered)} interactions. {SUMMARY_HINT}"
    md = clean_for_markdown(filtered).to_markdown(index=False, tablefmt="plain")
    return md

@mcp.tool()
def extend_network(
        genes: Annotated[List[str], Field(description="Gene symbols to add (e.g. ['EGFR', 'AKT1']).")],
        session_id: Optional[str] = Field(None, description="Session ID; omit to use the active/default session."),
        verbosity: str = Field(DEFAULT_VERBOSITY, description="Output detail level: 'summary', 'preview', or 'full'."),
        autoconnect: bool = Field(True, description="Re-run complete_connection with session defaults after adding all genes.")) -> str:
    """Add multiple genes to the network in one call, optionally re-running connection completion.

    More efficient than calling add_gene() in a loop.
    """
    verbosity = normalize_verbosity(verbosity)
    sess, network = _session_network(session_id)
    if network is None:
        return E_NO_NET
    added = 0
    for g in genes:
        try:
            network.add_node(g)
            added += 1
        except Exception:
            pass
    if autoconnect:
        try:
            params = sess.get_completion_params()
            network.complete_connection(**params)
        except Exception:
            pass
    _invalidate(sess)
    if verbosity == 'summary':
        return f"Added {added}/{len(genes)} genes. {SUMMARY_HINT}"
    return f"Added {added}/{len(genes)} genes. Autoconnect={'yes' if autoconnect else 'no'}."

@mcp.tool()
def set_default_params(
        max_len: Optional[int] = Field(None, description="Default maximum path length for complete_connection calls (1-4)."),
        algorithm: Optional[str] = Field(None, description="Default path-search algorithm: 'bfs' or 'dfs'."),
        only_signed: Optional[bool] = Field(None, description="Default signed-only filter for complete_connection."),
        connect_with_bias: Optional[bool] = Field(None, description="Default activation-bias preference."),
        consensus: Optional[bool] = Field(None, description="Default multi-source consensus requirement."),
        session_id: Optional[str] = Field(None, description="Session ID; omit to use the active/default session.")) -> str:
    """Persist completion parameters in the session so extend_network() and add_gene(autoconnect=True) reuse them."""
    sess = ensure_session(session_id)
    sess.update_default_params(max_len=max_len, algorithm=algorithm, only_signed=only_signed,
                               connect_with_bias=connect_with_bias, consensus=consensus)
    return "Defaults updated." 

@mcp.tool()
def filter_interactions(
        effect: Optional[List[str]] = Field(None, description="Effect types to keep, e.g. ['stimulation', 'inhibition']. Omit to include all effects."),
        source: Optional[str] = Field(None, description="Keep only edges where the source matches this gene symbol."),
        target: Optional[str] = Field(None, description="Keep only edges where the target matches this gene symbol."),
        session_id: Optional[str] = Field(None, description="Session ID; omit to use the active/default session."),
        verbosity: str = Field(DEFAULT_VERBOSITY, description="Output detail level: 'summary' (count), 'preview'/'full' (Markdown table)."),
        format: str = Field("markdown", description="Output format: 'markdown' (default) or 'json'."),
        max_rows: int = Field(50, description="Maximum rows returned in preview mode.")) -> str:
    """Filter and display interactions by effect type, source gene, or target gene.

    Non-destructive - does not modify the network; use remove_interaction() to permanently delete edges.
    """
    verbosity = normalize_verbosity(verbosity)
    sess, network = _session_network(session_id)
    if network is None:
        return E_NO_NET
    df = sess.get_edges_df()
    if df is None or df.empty:
        return "No interactions." if format != 'json' else "{}"
    if effect and 'Effect' in df.columns:
        df = df[df['Effect'].isin(effect)]
    if source:
        df = df[df['source'] == source]
    if target:
        df = df[df['target'] == target]
    if df.empty:
        return "No matches." if format != 'json' else "{}"
    if format == 'json':
        # Return a compact JSON-like string (avoid importing json for token saving)
        subset = df.head(max_rows if verbosity != 'full' else 500)
        records = subset.to_dict(orient='records')
        return str(records)
    if verbosity == 'summary':
        return f"Filtered interactions: {len(df)}. {SUMMARY_HINT}"
    table, truncated = _short_table(df[[c for c in df.columns if c in ['source','target','Effect']]], max_rows if verbosity=='preview' else 100)
    return table + (" (truncated)" if truncated else "")

@mcp.tool()
def create_session(
        label: Optional[str] = Field(None, description="Optional human-readable label for this session (e.g. 'TP53-MYC cancer'). Stored on disk so the session can be rediscovered after a server restart.")) -> str:
    """Create a new isolated modelling session (always call before create_network).

    Each session holds its own Network object and default completion parameters.
    Prevents accidental reuse of a previous network when starting a new hypothesis.
    A unique UUID is assigned — use it in all subsequent tool calls.
    """
    sid = session_manager.create_session(set_as_default=False)
    write_session_meta(_SERVER_ROOT, sid, server_name="NeKo", label=label)
    label_info = f" ({label})" if label else ""
    return f"Created session: {sid}{label_info}"

@mcp.tool()
def list_sessions() -> str:
    """List all active sessions with network presence and basic node/edge counts."""
    data = session_manager.list_sessions()
    if not data:
        return "No sessions."
    lines = ["Sessions:"]
    for sid, meta in data.items():
        lines.append(f"- {sid}: has_network={meta['has_network']} nodes={meta['nodes']} edges={meta['edges']}")
    return "\n".join(lines)

@mcp.tool()
def list_artifact_sessions() -> str:
    """List all NeKo sessions that have artifact files on disk (including past server runs).

    Unlike list_sessions() which only shows in-memory sessions, this scans the
    artifacts/ directory and reads session_meta.json files, so previously created
    sessions are visible even after a server restart.

    Use the returned session_id and file paths to resume earlier work, e.g.:
      create_network(sif_file='/path/to/artifacts/<uuid>/Network.sif')
    """
    sessions = _list_artifact_sessions_on_disk(_SERVER_ROOT, server_name="NeKo")
    if not sessions:
        return "No artifact sessions found on disk."
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
    return "\n".join(lines)

@mcp.tool()
def set_default_session(
        session_id: Annotated[str, Field(description="Session ID to make the active default; used when session_id is omitted in subsequent tool calls.")]) -> str:
    """Set the default session used when session_id is omitted in other tool calls."""
    ok = session_manager.set_default(session_id)
    return "Default set." if ok else "Session not found."

@mcp.tool()
def delete_session(
        session_id: Annotated[str, Field(description="Session ID to permanently delete (irreversible).")]) -> str:
    """Permanently delete a session and its in-memory network."""
    ok = session_manager.delete_session(session_id)
    return "Deleted." if ok else "Session not found."

@mcp.tool()
def status(
        session_id: Optional[str] = Field(None, description="Session ID; omit to query the active/default session.")) -> str:
    """Return a one-line session summary: session ID, node count, edge count."""
    sess, network = _session_network(session_id)
    if network is None:
        return f"Session {sess.session_id}: no network."
    df = sess.get_edges_df()
    edges = len(df) if df is not None else 0
    return f"Session {sess.session_id}: nodes={len(network.nodes)} edges={edges}."

# ===== Component & Strategy Tools =====
@mcp.tool()
def list_components(
        session_id: Optional[str] = Field(None, description="Session ID; omit to use the active/default session."),
        verbosity: str = Field(DEFAULT_VERBOSITY, description="Output detail level: 'summary' (counts), 'preview'/'full' (per-component stats)."),
        format: str = Field("markdown", description="Output format: 'markdown' (default) or 'json'.")) -> str:
    """List connected components with size, average degree, and sample nodes.

    Use this after check_disconnected_nodes() to understand the component structure
    before choosing a strategy in apply_strategy().
    """
    verbosity = normalize_verbosity(verbosity)
    sess, network = _session_network(session_id)
    if network is None:
        return E_NO_NET
    
    comps = _compute_components(network)
    if not comps:
        return "No components (empty network)."
    
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

    data = []
    for idx, comp in enumerate(comps):
        dvals = [deg.get(n, 0) for n in comp]
        avg_d = round(sum(dvals)/len(dvals), 2) if dvals else 0
        data.append({"id": idx, "size": len(comp), "avg_degree": avg_d, "sample": comp[:5]})
        
    if format == 'json':
        return str(data)
        
    if verbosity == 'summary':
        return f"Components={len(comps)} largest={max((len(c) for c in comps), default=0)}. {SUMMARY_HINT}"
        
    lines = ["Components:"]
    for row in data:
        lines.append(f"- {row['id']}: size={row['size']} avg_deg={row['avg_degree']} sample={row['sample']}")
    return "\n".join(lines)

@mcp.tool()
def candidate_connectors(
        method: str = Field("hubs", description="Suggestion strategy: 'hubs' (rank high-degree nodes), 'relax_max_len' (simulate +1 max_len), 'unsigned' (simulate allowing unsigned interactions)."),
        top_k: int = Field(10, description="Number of hub genes to report when method='hubs'."),
        session_id: Optional[str] = Field(None, description="Session ID; omit to use the active/default session."),
        format: str = Field("markdown", description="Output format: 'markdown' (default) or 'json'."),
        verbosity: str = Field(DEFAULT_VERBOSITY, description="Output detail level: 'summary', 'preview', or 'full'.")) -> str:
    """Suggest nodes or parameter relaxations that could bridge disconnected components.

    Run before applying a connection strategy to estimate the benefit without committing to changes.
    Outputs Gene Symbols for readability.
    """
    verbosity = normalize_verbosity(verbosity)
    sess, network = _session_network(session_id)
    if network is None:
        return E_NO_NET
        
    method = method.lower()
    
    u2s, _ = _get_translators(network)
    
    suggestions = []
    rationale = ''
    
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
            return "No edge data available to calculate hubs."
            
        maxd = max(deg.values()) if deg else 1
        
        # Sort by degree and take top_k
        ranked_uniprot = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # TRANSLATE BACK TO GENE SYMBOLS for the LLM
        for uid, d in ranked_uniprot:
            symbol = u2s.get(uid, str(uid)) # Fallback to Uniprot ID if no symbol exists
            suggestions.append({"gene": symbol, "score": round(d/maxd, 3), "raw_degree": d})
            
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
            
            suggestions = [{
                "predicted_new_edges": after_e - before_e, 
                "simulated_maxlen": params.get('maxlen'), 
                "simulated_only_signed": params.get('only_signed')
            }]
            
        except Exception as e:
            return f"Simulation failed: {e}"
    else:
        return "Unsupported method. Please use 'hubs', 'relax_max_len', or 'unsigned'."

    # --- FORMAT OUTPUT ---
    if format == 'json':
        return str({"method": method, "suggestions": suggestions, "rationale": rationale})
        
    if verbosity == 'summary':
        return f"{method}: {len(suggestions)} suggestions. {SUMMARY_HINT}"
        
    lines = [f"Candidate connectors ({method}):"]
    for s in suggestions:
        if 'gene' in s:
            lines.append(f"- **{s['gene']}**: relative_score={s['score']} (edges={s['raw_degree']})")
        else:
            lines.append(f"- Predicted new edges: {s.get('predicted_new_edges', 0)}")
            lines.append(f"- Parameters simulated: max_len={s.get('simulated_maxlen')}, only_signed={s.get('simulated_only_signed')}")
            
    if rationale:
        lines.append(f"\n*Rationale: {rationale}*")
        
    return "\n".join(lines)

@mcp.tool()
def bridge_components(
        comp_a: List[str] = Field(..., description="First list of nodes (Gene Symbols) to bridge."),
        comp_b: List[str] = Field(..., description="Second list of nodes (Gene Symbols) to bridge."),
        max_len: int = Field(2, description="Maximum path length for connecting edges."),
        mode: str = Field("OUT", description="Edge direction mode: 'OUT' or 'IN'."),
        only_signed: Optional[bool] = Field(None, description="Restrict to signed interactions."),
        consensus: Optional[bool] = Field(None, description="Require multi-source consensus."),
        session_id: Optional[str] = Field(None, description="Session ID.")) -> str:
    """Connect two specific disconnected components or subgroups of genes together."""
    sess, network = _session_network(session_id)
    if network is None: return E_NO_NET
    
    params = sess.get_completion_params()
    only_signed = only_signed if only_signed is not None else params.get('only_signed', True)
    consensus = consensus if consensus is not None else params.get('consensus', True)

    # 1. TRANSLATION LAYER: Gene Symbols -> Uniprot
    _, s2u = _get_translators(network)
    uniprot_a = [s2u.get(gene, gene) for gene in comp_a] # Fallback to input if not found
    uniprot_b = [s2u.get(gene, gene) for gene in comp_b]

    # 2. BACKEND MATH: Run strictly in Uniprot IDs
    try:
        strategy_connect_component(network, uniprot_a, uniprot_b, 
                                   maxlen=max_len, mode=mode, 
                                   only_signed=only_signed, consensus=consensus)
        _invalidate(sess)
        
        df = sess.get_edges_df()
        return f"Successfully bridged components. Network now has {len(df) if df is not None else 0} edges."
    except Exception as e:
        return f"Bridging failed: {e}"

@mcp.tool()
def connect_targeted_nodes(
        strategy: Annotated[str, Field(description="Targeted strategy.", json_schema_extra={"enum": ["connect_to_upstream_nodes", "connect_subgroup", "connect_as_atopo"]})],
        nodes: List[str] = Field(..., description="Target genes (Gene Symbols) to connect or expand."),
        outputs: Optional[List[str]] = Field(None, description="[connect_as_atopo] Output gene symbols to anchor topology."),
        max_len: int = Field(1, description="Max path length or upstream depth."),
        strategy_mode: Optional[str] = Field(None, description="[connect_as_atopo] Sub-mode passed to atopo (e.g., 'hierarchy')."),
        only_signed: Optional[bool] = Field(None),
        consensus: Optional[bool] = Field(None),
        session_id: Optional[str] = None) -> str:
    """Apply strategies targeting specific genes (upstream regulators, dense subgroups, or topological mapping)."""
    sess, network = _session_network(session_id)
    if network is None: return E_NO_NET
    
    params = sess.get_completion_params()
    osgn = only_signed if only_signed is not None else params.get('only_signed', True)
    cons = consensus if consensus is not None else params.get('consensus', True)

    # 1. TRANSLATION LAYER: Gene Symbols -> Uniprot
    _, s2u = _get_translators(network)
    uniprot_nodes = [s2u.get(n, n) for n in nodes]
    uniprot_outputs = [s2u.get(o, o) for o in outputs] if outputs else None

    # 2. BACKEND MATH
    try:
        if strategy == "connect_to_upstream_nodes":
            connect_to_upstream_nodes(network, nodes_to_connect=uniprot_nodes, depth=max_len, only_signed=osgn, consensus=cons)
        elif strategy == "connect_subgroup":
            connect_subgroup(network, group=uniprot_nodes, maxlen=max_len, only_signed=osgn, consensus=cons)
        elif strategy == "connect_as_atopo":
            connect_as_atopo(network, strategy=strategy_mode, max_len=max_len, outputs=uniprot_outputs, only_signed=osgn, consensus=cons)
            
        _invalidate(sess)
        return f"Applied {strategy} to targeted nodes."
    except Exception as e:
        return f"Strategy failed: {e}"

@mcp.tool()
def apply_global_connection(
        strategy: Annotated[str, Field(description="Global connection strategy.", json_schema_extra={"enum": ["complete_connection", "connect_network_radially"]})],
        max_len: int = Field(2, description="Maximum path length to search for connections."),
        algorithm: str = Field("bfs", description="[complete_connection] Search algorithm: 'bfs' or 'dfs'."),
        minimal: bool = Field(True, description="[complete_connection] Add only minimum required edges."),
        direction: str = Field("OUT", description="[connect_network_radially] Growth direction ('OUT' or 'IN')."),
        only_signed: Optional[bool] = Field(None),
        consensus: Optional[bool] = Field(None),
        session_id: Optional[str] = None) -> str:
    """Apply a global connection strategy across the entire network to resolve missing edges."""
    sess, network = _session_network(session_id)
    if network is None: return E_NO_NET
    
    params = sess.get_completion_params()
    osgn = only_signed if only_signed is not None else params.get('only_signed', True)
    cons = consensus if consensus is not None else params.get('consensus', True)

    # BACKEND MATH (No translation needed for global functions)
    try:
        if strategy == "complete_connection":
            strategy_complete_connection(network, maxlen=max_len, algorithm=algorithm, minimal=minimal, only_signed=osgn, consensus=cons)
        elif strategy == "connect_network_radially":
            connect_network_radially(network, max_len=max_len, direction=direction, only_signed=osgn, consensus=cons)
            
        _invalidate(sess)
        df = sess.get_edges_df()
        return f"Successfully applied global {strategy}. Edges now = {len(df) if df is not None else 0}."
    except Exception as e:
        return f"Global strategy failed: {e}"

if __name__ == "__main__":
    mcp.run()
