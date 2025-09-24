"""NeKo MCP Server

Purpose: Provide session-scoped NeKo network construction and refinement tools for automated
multiscale modelling workflows (NeKo -> MaBoSS -> PhysiCell). Sessions isolate hypotheses and
store default completion parameters. Always prefer create_session() before create_network().

Recommended high-level workflow:
 1. create_session()
 2. set_default_params(max_len=2, only_signed=True, consensus=True)  # optional tuning
 3. create_network([...TNF pathway / fate genes...], database='omnipath')
 4. remove_bimodal_interactions(); remove_undefined_interactions()
 5. check_disconnected_nodes(); if disconnected -> list_components(); candidate_connectors(); apply_strategy('complete_connection')
 6. list_genes_and_interactions(verbosity='preview'); find_paths('TNF','CASP3')
 7. export_network(format='bnet') -> feed into MaBoSS (choose output nodes, run threads=10)
 8. Evaluate biological plausibility; refine via filter_interactions(), get_references(), apply_strategy(), pruning
 9. Iterate until stable; then integrate with PhysiCell (map Boolean outputs to behaviours)

Strategy tools (apply_strategy): complete_connection, connect_as_atopo, connect_component,
connect_network_radially, connect_to_upstream_nodes, connect_subgroup. Use candidate_connectors()
first to inspect possible bridging nodes or relaxation impacts.

Verbosity levels: 'summary' (token frugal), 'preview' (tables truncated), 'full'.

All tools that require a network return E_NO_NET with guidance if missing.
"""

import io
import sys
import os
import glob
import requests
import json
import logging
from pathlib import Path
from typing import Optional, List
from functools import wraps
#from hatch_mcp_server import HatchMCP

from neko.core.network import Network
from neko._outputs.exports import Exports
from neko.inputs import Universe, signor
from neko.core.tools import is_connected

from utils import *
from session_manager import session_manager, ensure_session, normalize_verbosity, DEFAULT_VERBOSITY
from neko.core.strategies import (
    connect_as_atopo,
    connect_component as strategy_connect_component,
    complete_connection as strategy_complete_connection,
    connect_network_radially,
    connect_to_upstream_nodes,
    connect_subgroup,
)

import pandas as pd


from mcp.server.fastmcp import Context, FastMCP

mcp = FastMCP("NeKo")

""" # Initialize MCP server with metadata
hatch_mcp = HatchMCP("NeKo",
                fast_mcp=mcp,
                origin_citation="NeKo: a tool for automatic network construction from prior knowledge, Marco Ruscone,  Eirini Tsirvouli,  Andrea Checcoli,  Denes Turei,  Emmanuel Barillot,  Julio Saez-Rodriguez,  Loredana Martignetti,  Åsmund Flobak,  Laurence Calzone, doi: https://doi.org/10.1101/2024.10.14.618311",
                mcp_citation="https://github.com/marcorusc/Hatch_Pkg_Dev/tree/main/NeKo") """

# NOTE: Previous implementation used a single global `network` object.
# Now session-based management (see `session_manager.py`).
# Each tool can accept an optional `session_id` allowing multiple networks.
# If not provided, the default session is used (auto-created on first use).

# Constants / shared small strings to reduce token footprints
E_NO_NET = "E_NO_NET: No network in session. Call create_session() then create_network()."
SUMMARY_HINT = "Set verbosity='preview' or 'full' for more details."

def _short_table(df, max_rows=25):
    """Return markdown table (plain) truncated with note if needed."""
    if df is None or df.empty:
        return "(no data)", True
    truncated = False
    if len(df) > max_rows:
        df = df.head(max_rows)
        truncated = True
    return clean_for_markdown(df).to_markdown(index=False, tablefmt="plain"), truncated

def _export_dir() -> Path:
    # Anchor exports to the directory of this server file to avoid writing to $HOME when CWD differs
    base = Path(__file__).parent
    d = base / "exports"
    d.mkdir(exist_ok=True)
    return d

def _session_network(session_id: Optional[str]):
    sess = ensure_session(session_id)
    return sess, sess.network

def _invalidate(sess):
    if sess:
        sess.invalidate_edges_cache()

# ===== Decorators & Helpers for strategies & network requirements =====
def requires_network(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        session_id = kwargs.get("session_id")
        sess, network = _session_network(session_id)
        if network is None:
            return E_NO_NET
        kwargs["sess"] = sess
        kwargs["network"] = network
        return fn(*args, **kwargs)
    return inner

def _compute_components(network) -> List[List[str]]:
    try:
        df = network.convert_edgelist_into_genesymbol()
    except Exception:
        return []
    if df is None or df.empty:
        return []
    adj: dict[str, set] = {}
    for _, r in df.iterrows():
        s = r['source']; t = r['target']
        if pd.isna(s) or pd.isna(t):
            continue
        adj.setdefault(s, set()).add(t)
        adj.setdefault(t, set()).add(s)
    visited = set()
    comps = []
    for n in adj.keys():
        if n in visited:
            continue
        stack = [n]
        cur = []
        visited.add(n)
        while stack:
            node = stack.pop()
            cur.append(node)
            for nb in adj.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        comps.append(cur)
    return comps

# Main function that creates the NeKo network from a list of initial genes.
# If the list of genes is empty but a SIF file is provided,
# it will create a network from the SIF file.
# If both are provided, it will create a network from the list of genes and the SIF file.
# If neither is provided, it will return an error message.
# If the database is not supported, it will return an error message.
# If the network is created successfully, it will return a Markdown formatted string with the network summary.
# If the network creation fails, it will return an error message.
# If the network is empty, it will return a Markdown formatted string with an empty table.
# If the network is not connected, it will return a Markdown formatted string with a warning message.
# If the network is connected, it will return a Markdown formatted string with the network summary.
# If the network is not reset, it will return an error message.
@mcp.tool()
def create_network(list_of_initial_genes: List[str],
                   ctx: Context,
                   database: str = "omnipath",
                   sif_file: Optional[str] = None,
                   max_len: int = 2,
                   algorithm: str = "bfs",
                   only_signed: bool = True,
                   connect_with_bias: bool = False,
                   consensus: bool = True,
                   session_id: Optional[str] = None,
                   verbosity: str = DEFAULT_VERBOSITY) -> str:
    """
Create a NeKo network from a list of genes and/or a SIF file.
If the list of genes is empty but a SIF file is provided, load the network from the SIF file.
If the list of genes is not empty and there is no SIF file, use just the list of genes.
If both are provided, load the network from the SIF file and then add all genes in the list.
As optional parameters, you can specify the maximum length of paths to complete (default is 2),
the algorithm to use for path completion (the user can choose between "bfs" and "dfs" which stands for breadth-first search and depth-first search, respectively),
whether to only include signed interactions, whether to connect with bias, and whether to use consensus.
If the database is not supported, it will return an error message.
If the network is created successfully, it will return a Markdown formatted string with the network summary
Args:
list_of_initial_genes (list[str]): List of gene symbols.
database (str): Database to use for network creation, either 'omnipath' or 'signor'.
sif_file (str): Path to a SIF file to load the network from.
max_len (int): Maximum length of paths to complete. Defaults to 2.
algorithm (str): Algorithm to use for path completion, either 'bfs' or 'dfs'. Defaults to 'bfs'.
only_signed (bool): Whether to only include signed interactions. Defaults to True.
connect_with_bias (bool): Whether to connect with bias. Defaults to False.
consensus (bool): Whether to use consensus. Defaults to True.
Returns:
str: Status message or Markdown formatted string with network summary.
    """
    verbosity = normalize_verbosity(verbosity)
    sess = ensure_session(session_id)
    ctx.info(f"Creating NeKo network (session={sess.session_id}) with genes={list_of_initial_genes} sif={sif_file}")
    logging.info(f"Creating NeKo network (session={sess.session_id}) with genes={list_of_initial_genes} sif={sif_file}") 
    # Validate database choice
    if database not in ["omnipath", "signor"]:
        return "_Unsupported database. Use `omnipath` or `signor`._"

    # If using SIGNOR, download and build the SIGNOR resource
    if database == "signor":
        ctx.info("Downloading SIGNOR database...")
        ctx.info("SIGNOR database downloaded successfully.")
        signor_res = signor()
        signor_res.build()
        resources = signor_res.interactions
    else:
        resources = "omnipath"

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
        ctx.warning("No interactions found in the network. Please check the input parameters.")
        return format_empty_network_response(list_of_initial_genes, database, max_len, only_signed)

    # Compute basic statistics
    num_edges = len(df_edges)
    unique_nodes = pd.unique(df_edges[["source", "target"]].values.ravel())
    num_nodes = len(unique_nodes)

    ctx.info("Network created successfully.")

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
def add_gene(gene: str, session_id: Optional[str] = None, autoconnect: bool = False, sess=None, network=None) -> str:
    """
    Add a gene to the current network. The gene must be a valid gene symbol.
    If no network exists, it prompts the user to create one first.
    Args:
        gene (str): Gene symbol to add.
    Returns:
        str: Status message.
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
def remove_gene(gene: str, session_id: Optional[str] = None, sess=None, network=None) -> str:
    """
    Remove a gene from the current network. The gene must be a valid gene symbol.
    If no network exists, it prompts the user to create one first.
    If the gene is not found in the network, it returns an error message.
    Args:
        gene (str): Gene symbol to remove.
    Returns:
        str: Status message.
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
            # Suggest similar (substring or Levenshtein-lite via length difference) – keep it lightweight
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
def remove_interaction(node_A: str, node_B: str, session_id: Optional[str] = None, sess=None, network=None) -> str:
    """
    Remove an interaction between two genes in the current network.
    If no network exists, it prompts the user to create one first.
    If the interaction is not found in the network, it returns an error message.
    It only deletes the interaction in the specified direction (A to B).
    Args:
        node_A (str): First gene symbol.
        node_B (str): Second gene symbol.
    Returns:
        str: Status message.
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

# TO DO: Implement multiple network enrichment strategies (connect upstream, connect as atopo, etc...)

# TO DO: Implement export of images with graphviz

# TO DO: implement better check of which nodes are disconnected and use MCM strategy to connect the disconnected component with the connected one

# TO DO: implement GO enrichment

@mcp.tool()
def export_network(format: str = "sif", session_id: Optional[str] = None, verbosity: str = DEFAULT_VERBOSITY) -> str:
    """
    When the user asks to export the current network,
    or asks to save the network in a specific format,
    or asks to export the network as SIF or BNET,
    this function exports the network in the specified format.
    If no network exists, it prompts the user to create one first.
    If the format is not supported, it returns an error message.
    The function returns a Markdown formatted string with the export status,
    including a preview of the exported file.
    If the exported file is in SIF format, it shows the first 100 lines as a Markdown table,
    containing the source, interaction, and target columns.
    If the exported file is in BNET format, it shows the first 100 lines as a Markdown table,
    containing the gene and Boolean expression columns.
    Args:
        format (str): Format to export the network, either 'sif' or 'bnet'.
    Returns:
        str: Markdown formatted string with export status and preview.
    """
    verbosity = normalize_verbosity(verbosity)
    sess, network = _session_network(session_id)
    if network is None:
        return format_no_network_guidance()
    exporter = Exports(network)

    # Helper to read & preview first 100 lines, returning Markdown
    def _preview_file(path: str, sep: str, cols: list[str]) -> str:
        """
        Try to read up to 100 rows of `path` (with pandas) using sep and columns list.
        If successful, return as Markdown table. Otherwise, return first 10
        raw lines in a fenced code block.
        """
        if not os.path.exists(path):
            return f"_File `{path}` not found._"

        # First, attempt to load with pandas and produce a Markdown table
        try:
            df_preview = pd.read_csv(path, sep=sep, header=None, names=cols, nrows=100, dtype=str)
            # Drop any fully-NaN rows (sometimes trailing newlines)
            df_preview.dropna(how="all", inplace=True)
            return clean_for_markdown(df_preview).to_markdown(index=False, tablefmt="plain")
        except Exception:
            # Fallback: just show the first 100 lines as-is
            lines = []
            with open(path, "r") as f:
                for _ in range(100):
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line.rstrip("\n"))
            if not lines:
                return "_File is empty or could not be read._"

            code_block = ["```"] + lines + ["```"]
            return "\n".join(code_block)

    # 1) Handle SIF export
    if format.lower() == "sif":
        out_dir = _export_dir()
        out_path = str(out_dir / "Network.sif")
        try:
            exporter.export_sif(out_path)
        except Exception as e:
            return f"**Error exporting SIF:** {str(e)}"

        # Now build a Markdown snippet showing the path + preview
        if verbosity == "summary":
            return f"SIF exported: {out_path}. {SUMMARY_HINT}"
        md_lines = [f"Exported to `{out_path}`", "Preview (first 100 lines):", ""]
        # SIF format is: source<TAB>interaction<TAB>target
        preview_md = _preview_file(out_path, sep="\t", cols=["source", "interaction", "target"])
        md_lines.append(preview_md)
        return "\n".join(md_lines)

    # 2) Handle BNET export
    elif format.lower() == "bnet":
        def clean_node_name(name: str) -> str:
            import re
            return re.sub(r"[^A-Za-z0-9_]", "_", name)

        # Check connectivity first with enhanced guidance
        if not is_connected(network):
            return format_connectivity_guidance()

        # Export
        try:
            out_dir = _export_dir()
            exporter.export_bnet(str(out_dir / "Network"))
            clean_bnet_headers(str(out_dir))
        except Exception as e:
            return f"**Error exporting BNET:** {str(e)}"

        # Find all .bnet files in the exports directory
        bnet_files = [os.path.basename(f) for f in glob.glob(str(out_dir / "*.bnet"))]
        if not bnet_files:
            return "**Error:** No .bnet files were generated."
        out_path = str(out_dir / bnet_files[0])

        # Clean node names in the BNET file (both columns)
        cleaned_names = set()
        try:
            # First pass: build mapping of original -> cleaned names
            with open(out_path, "r") as f:
                lines = f.readlines()
            name_map = {}
            for line in lines:
                if "," in line:
                    gene, _ = line.split(",", 1)
                    gene_clean = clean_node_name(gene.strip())
                    if gene_clean != gene.strip():
                        cleaned_names.add(gene.strip())
                    name_map[gene.strip()] = gene_clean
            # Second pass: rewrite lines with cleaned names in both columns
            new_lines = []
            import re
            for line in lines:
                if "," in line:
                    gene, expr = line.split(",", 1)
                    gene_clean = name_map.get(gene.strip(), gene.strip())
                    expr_clean = expr
                    for orig, clean in name_map.items():
                        if orig != clean:
                            expr_clean = re.sub(rf'(?<![A-Za-z0-9_]){re.escape(orig)}(?![A-Za-z0-9_])', clean, expr_clean)
                    new_lines.append(f"{gene_clean},{expr_clean}")
                else:
                    new_lines.append(line)
            with open(out_path, "w") as f:
                f.writelines(new_lines)
        except Exception as e:
            return f"**Error cleaning BNET gene names:** {str(e)}"

        # Check for special characters in gene/node names in the bnet file (should be clean now)
        special_char_issues = []
        try:
            with open(out_path, "r") as f:
                for i, line in enumerate(f):
                    if i >= 1000:
                        break
                    if "," in line:
                        gene = line.split(",", 1)[0].strip()
                        import re
                        if re.search(r"[^A-Za-z0-9_]", gene):
                            special_char_issues.append(gene)
        except Exception as e:
            return f"**Error reading BNET file for gene name check:** {str(e)}"

        if verbosity == "summary":
            return f"BNET exported: {out_path}. {SUMMARY_HINT}"
        md_lines = [f"Exported to `{out_path}`", "Preview (first 100 lines):", ""]
        preview_md = _preview_file(out_path, sep=",", cols=["gene", "expression"])
        md_lines.append(preview_md)
        if len(bnet_files) > 1:
            md_lines.append(f"\n_Warning: More than one .bnet file was found ({', '.join(bnet_files)}). Previewing only the first one._")
        if cleaned_names:
            md_lines.append(f"\n**Warning:** The following gene/node names were modified to remove special characters: {', '.join(sorted(cleaned_names))}")
        if special_char_issues:
            md_lines.append(f"\n**Warning:** The following gene/node names still contain special characters and may not be compatible: {', '.join(sorted(set(special_char_issues)))}")
        return "\n".join(md_lines)

    # 3) Unsupported format - provide guidance
    else:
        return format_unsupported_format_guidance(format)

@mcp.tool()
def network_dimension(session_id: Optional[str] = None) -> str:
    """
    When the user asks for the dimension of the current network,
    or asks how many nodes and edges are in the network,
    or simply asks for the network size,
    this function returns a summary string with the number of nodes and edges.
    If no network exists, it prompts the user to create one first.
    If the network is empty, it returns a message indicating that no nodes or edges are found.
    Returns:
        str: Summary string with number of nodes and edges.
    """
    sess, network = _session_network(session_id)
    if network is None:
        return E_NO_NET
    return f"Session {sess.session_id}: nodes={len(network.nodes)} edges={len(network.edges)}"

@mcp.tool()
def list_genes_and_interactions(session_id: Optional[str] = None, verbosity: str = DEFAULT_VERBOSITY, max_rows: int = 50) -> str:
    """
    When the user asks for a list of genes and interactions in the current network,
    or asks which interactions are present in the network,
    or simply asks for the interactions,
    or asks to show the network,
    this function returns a markdown table with the interactions, excluding the 'resources' column.
    If no network exists, it prompts the user to create one first.
    If the network is empty, it returns a message indicating that no interactions are found.
    If an error occurs during the conversion, it returns an error message with an empty table.
    
    Returns:
        str: Markdown table of interactions or an error message.
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
def find_paths(source: str, target: str, maxlen: int = 3, session_id: Optional[str] = None, verbosity: str = DEFAULT_VERBOSITY) -> str:
    """
    When the user asks for paths between two genes in the network,
    or asks to find paths from one gene to another,
    or asks for paths from a source gene to a target gene,
    this function finds all paths between the source and target genes up to a given length.
    It captures and returns the printed output of print_my_paths in a Markdown format.
    If no network exists, it prompts the user to create one first.
    If no paths are found, it returns a message indicating that no paths were found.
    Args:
        source (str): Source gene symbol.
        target (str): Target gene symbol.
        maxlen (int): Maximum length of paths to find. Defaults to 3.
    Returns:
        str: Markdown formatted string with paths or an error message.
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
def reset_network(session_id: Optional[str] = None) -> str:
    """
    When the user asks to reset the current network,
    or asks to clear the network,
    or simply asks to reset,
    this function resets the global network object to None.
    If no network exists, it returns a message indicating that no network was loaded.
    If the network is reset successfully, it returns a confirmation message.
    If the network is not reset, it returns an error message.
    Args:
        None
    Returns:
        str: Status message.
    """
    sess = ensure_session(session_id)
    sess.set_network(None)
    return f"Session {sess.session_id} network reset."

@mcp.tool()
def clean_generated_files(folder_path: str = "exports") -> str:
    """
    When the user asks to clean up generated files,
    or asks to remove .bnet files,
    or asks to delete all .bnet files,
    this function removes all .bnet files from the specified folder.
    If no .bnet files are found, it returns a message indicating that no files were found to clean.
    If the folder path is not specified, it defaults to the current directory.
    If the files are cleaned successfully, it returns a status message indicating how many files were cleaned.
    If an error occurs during file deletion, it returns an error message.
    Args:
        folder_path (str): Path to the folder to clean. Defaults to current directory.
    Returns:
        str: Status message indicating cleaned files.
    """
    bnet_files = glob.glob(os.path.join(folder_path, "*.bnet"))
    if not bnet_files:
        return "No .bnet files found to clean."

    for file_path in bnet_files:
        os.remove(file_path)

    return f"Cleaned {len(bnet_files)} .bnet files from {folder_path}."

@mcp.tool()
def get_help() -> str:
    """
    Get a description of available NeKo MCP tools and their usage.
    Returns:
        str: Help string.
    """
    return (
        "NeKo MCP Tools (workflow oriented):\n"
        "Start: create_session -> set_default_params -> create_network.\n"
        "Curation: remove_bimodal_interactions, remove_undefined_interactions.\n"
        "Connectivity: check_disconnected_nodes, list_components, candidate_connectors, apply_strategy.\n"
        "Inspection: list_genes_and_interactions, find_paths, get_references, filter_interactions, network_dimension.\n"
        "Export: export_network (sif|bnet), list_bnet_files, clean_generated_files.\n"
        "Sessions: list_sessions, set_default_session, delete_session, status, reset_network.\n"
        "Guidance: workflow_guide, get_help, get_help_json.\n"
        "Verbosity: summary|preview|full (use summary in loops). Always create_session() before create_network()."
    )

@mcp.tool()
def get_help_json() -> str:
    """Machine-readable help describing tools, categories, workflow and strategies.

    Returns a JSON string with keys:
      version: schema version (increment if structure changes)
      recommended_workflow: ordered list of key steps
      categories: mapping category -> list of tools
      strategies: supported strategy names with short descriptions
      verbosity: allowed levels and intent
      notes: assorted planner guidance
    """
    data = {
        "version": 1,
        "recommended_workflow": [
            "create_session",
            "set_default_params",
            "create_network",
            "remove_bimodal_interactions",
            "remove_undefined_interactions",
            "check_disconnected_nodes",
            "list_components (if disconnected)",
            "candidate_connectors",
            "apply_strategy",
            "list_genes_and_interactions",
            "find_paths",
            "export_network",
            "(MaBoSS simulation)",
            "refine (filter_interactions / get_references / apply_strategy)",
            "finalize for PhysiCell"
        ],
        "categories": {
            "sessions": ["create_session","list_sessions","set_default_session","delete_session","status","reset_network"],
            "build": ["create_network","extend_network","add_gene","remove_gene","remove_interaction","set_default_params"],
            "curation": ["remove_bimodal_interactions","remove_undefined_interactions"],
            "connectivity": ["check_disconnected_nodes","list_components","candidate_connectors","apply_strategy"],
            "inspection": ["network_dimension","list_genes_and_interactions","find_paths","get_references","filter_interactions"],
            "export": ["export_network","list_bnet_files","clean_generated_files","check_bnet_files_names"],
            "guidance": ["workflow_guide","get_help","get_help_json"]
        },
        "strategies": {
            "complete_connection": "Expand network to connect seed genes within maxlen constraints.",
            "connect_as_atopo": "Connect nodes using a chosen atopo strategy (e.g., radial/loop options).",
            "connect_component": "Bridge two disconnected components by exploring paths up to maxlen.",
            "connect_network_radially": "Grow network outward from central nodes radially.",
            "connect_to_upstream_nodes": "Attach upstream regulators for specified nodes (depth/rank limited).",
            "connect_subgroup": "Densely connect a specified subgroup of nodes."
        },
        "verbosity": {
            "summary": "Token-frugal status only.",
            "preview": "Truncated tables / partial listings.",
            "full": "Expanded context (first 100 rows for large tables)."
        },
        "notes": [
            "Always explicitly call create_session before create_network to avoid reusing stale state.",
            "Use candidate_connectors before heavy apply_strategy invocations to estimate benefit.",
            "For iterative refinement loops prefer verbosity='summary' until you need detail.",
            "Export only supports 'sif' and 'bnet' intentionally (keep scope focused).",
            "After export_network(format='bnet') hand the .bnet to MaBoSS for simulation.",
            "Edge list caching is session-scoped and invalidated automatically on mutations." 
        ]
    }
    return json.dumps(data, ensure_ascii=False)

@mcp.tool()
def remove_bimodal_interactions(session_id: Optional[str] = None) -> str:
    """
    Remove all 'bimodal' interactions from the current network object in memory.
    Returns:
        str: Status message.
    """
    sess, network = _session_network(session_id)
    if network is None:
        return E_NO_NET
    if "Effect" not in network.edges.columns:
        return "No 'Effect' column found in network.edges."
    before = len(network.edges)
    network.remove_bimodal_interactions()
    after = len(network.edges)
    removed = before - after
    return f"Removed {removed} bimodal interactions from the network."

@mcp.tool()
def remove_undefined_interactions(session_id: Optional[str] = None) -> str:
    """
    Remove all 'undefined' interactions from the current network object in memory.
    Returns:
        str: Status message.
    """
    sess, network = _session_network(session_id)
    if network is None:
        return E_NO_NET
    if "Effect" not in network.edges.columns:
        return "No 'Effect' column found in network.edges."
    before = len(network.edges)
    network.remove_undefined_interactions()
    after = len(network.edges)
    removed = before - after
    return f"Removed {removed} undefined interactions from the network."


@mcp.tool()
def list_bnet_files(folder_path: str = "exports") -> list:
    """
    List all .bnet files in the specified folder.
    Args:
        folder_path (str): Path to the folder to search for .bnet files. Defaults to current directory.
    Returns:
        list: List of .bnet file names found in the folder.
    """
    bnet_files = [os.path.basename(f) for f in glob.glob(os.path.join(folder_path, "*.bnet"))]
    return bnet_files

def download_signor_database():
    """
    Download the SIGNOR database from the specified URL and save it to the current directory.
    Returns:
        str: Status message indicating success or failure.
    """
    url = "https://signor.uniroma2.it/API/getHumanData.php"
    try:
        r = requests.get(url)
        r.raise_for_status()  # Raise an error for bad responses
        output_file = "SIGNOR_Human.tsv"
        with open(output_file, 'wb') as f:
            f.write(r.content)
        return "SIGNOR database downloaded successfully."
    except requests.RequestException as e:
        return f"Error downloading SIGNOR database: {str(e)}"

def clean_bnet_headers(folder_path: str = ".") -> str:
    """
    Remove the first two lines from any .bnet file in the specified folder if they are:
    '# model in BoolNet format' and 'targets, factors'.
    Args:
        folder_path (str): Path to the folder to clean .bnet files. Defaults to current directory.
    Returns:
        str: Status message listing cleaned files.
    """
    cleaned_files = []
    for file_path in glob.glob(os.path.join(folder_path, "*.bnet")):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        if len(lines) >= 2 and lines[0].strip() == "# model in BoolNet format" and lines[1].strip() == "targets, factors":
            with open(file_path, 'w') as file:
                file.writelines(lines[2:])
            cleaned_files.append(os.path.basename(file_path))
    if cleaned_files:
        return f"Cleaned headers from: {', '.join(cleaned_files)}"
    else:
        return "No .bnet files needed cleaning."
    
@mcp.tool()
def check_bnet_files_names(folder_path: str = "exports") -> str:
    """
    When the user asks to check for .bnet files,
    or asks for the names of .bnet files in a specific folder,
    or simply asks to list .bnet files,
    this function checks for .bnet files in the specified folder.
    If the folder path is not specified, it defaults to the current directory.
    If the user asks for .bnet files, it returns a list of their names.
    If the user asks for the existence of .bnet files, it checks if any .bnet files exist in the folder.
    If no .bnet files are found, it returns a message indicating that no files were found.
    If .bnet files are found, it returns their names.
    If an error occurs during the check, it returns an error message.
    Args:
        folder_path (str): Path to the folder to check for .bnet files. Defaults to current directory.
    Returns:
        str: Names of .bnet files or a message indicating no files found.
    """
    bnet_files = glob.glob(os.path.join(folder_path, "*.bnet"))
    
    if not bnet_files:
        return "No .bnet files found in the specified folder."

    file_list = [os.path.basename(f) for f in bnet_files]
    return "Found .bnet files:\n" + "\n".join(file_list)

@mcp.tool()
def check_disconnected_nodes(session_id: Optional[str] = None) -> str:
    """
    When the user asks to check for disconnected nodes in the current network,
    or asks for nodes that are not connected to any edges,
    or simply asks for disconnected nodes,
    this function checks for nodes in the network that do not have any edges connected to them.
    If no network exists, it prompts the user to create one first.
    If all nodes are connected, it returns a message indicating that.
    If there are disconnected nodes, it returns a list of those nodes.
    Returns:
        str: List of disconnected nodes or a message indicating all nodes are connected.    
    """
    sess, network = _session_network(session_id)
    if network is None:
        return E_NO_NET
    
    all_nodes = set(network.nodes["Uniprot"].tolist())
    connected_nodes = set(network.edges["source"].tolist()) | set(network.edges["target"].tolist())
    disconnected_nodes = all_nodes - connected_nodes
    disconnected_nodes = [node for node in disconnected_nodes if pd.notna(node) and node != ""]
    
    if not disconnected_nodes:
        return "All nodes are connected."
    
    return "Disconnected nodes:\n" + "\n".join(disconnected_nodes)

@mcp.tool()
def get_references(node1: str, node2: str = None, session_id: Optional[str] = None, verbosity: str = DEFAULT_VERBOSITY) -> str:
    """
    Retrieve references for interactions involving a node (or between two nodes).
    - If only node1 is provided: show all interactions (edges) where node1 is source or target, with their references.
    - If both node1 and node2 are provided: show only interactions between node1 and node2 (any direction), with references.
    Returns a Markdown table with columns: source, target, effect, references (truncated to first 5, with count if more).
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
def extend_network(genes: List[str], session_id: Optional[str] = None, verbosity: str = DEFAULT_VERBOSITY, autoconnect: bool = True) -> str:
    """Add multiple genes; optionally rerun completion once at end."""
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
def set_default_params(max_len: Optional[int] = None,
                       algorithm: Optional[str] = None,
                       only_signed: Optional[bool] = None,
                       connect_with_bias: Optional[bool] = None,
                       consensus: Optional[bool] = None,
                       session_id: Optional[str] = None) -> str:
    """Update session default completion parameters used for future autoconnect operations."""
    sess = ensure_session(session_id)
    sess.update_default_params(max_len=max_len, algorithm=algorithm, only_signed=only_signed,
                               connect_with_bias=connect_with_bias, consensus=consensus)
    return "Defaults updated." 

@mcp.tool()
def filter_interactions(effect: Optional[List[str]] = None,
                        source: Optional[str] = None,
                        target: Optional[str] = None,
                        session_id: Optional[str] = None,
                        verbosity: str = DEFAULT_VERBOSITY,
                        format: str = 'markdown',
                        max_rows: int = 50) -> str:
    """Filter interactions by effect type and/or source/target. Supports markdown or json output."""
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
def create_session() -> str:
    """Create a new modelling session (preferred first call).

    Explicit session creation prevents accidental reuse of an old default network when an
    autonomous planner starts a fresh hypothesis. Each session stores its own Network and
    default autoconnect parameters (see set_default_params()).
    """
    sid = session_manager.create_session(set_as_default=False)
    return f"Created session: {sid}"

@mcp.tool()
def list_sessions() -> str:
    """List active sessions with whether they have a network and basic size metrics."""
    data = session_manager.list_sessions()
    if not data:
        return "No sessions."
    lines = ["Sessions:"]
    for sid, meta in data.items():
        lines.append(f"- {sid}: has_network={meta['has_network']} nodes={meta['nodes']} edges={meta['edges']}")
    return "\n".join(lines)

@mcp.tool()
def set_default_session(session_id: str) -> str:
    """Set the default session used when session_id is omitted in other tool calls."""
    ok = session_manager.set_default(session_id)
    return "Default set." if ok else "Session not found."

@mcp.tool()
def delete_session(session_id: str) -> str:
    """Delete a session and its network (irreversible)."""
    ok = session_manager.delete_session(session_id)
    return "Deleted." if ok else "Session not found."

@mcp.tool()
def status(session_id: Optional[str] = None) -> str:
    """Compact session summary (nodes, edges) or note absence of network."""
    sess, network = _session_network(session_id)
    if network is None:
        return f"Session {sess.session_id}: no network."
    df = sess.get_edges_df()
    edges = len(df) if df is not None else 0
    return f"Session {sess.session_id}: nodes={len(network.nodes)} edges={edges}."

@mcp.tool()
def workflow_guide() -> str:
    """Return recommended end-to-end workflow (NeKo -> MaBoSS -> PhysiCell)."""
    return (
        "Workflow:\n"
        "1. create_session()\n"
        "2. set_default_params(max_len=2, only_signed=True, consensus=True)\n"
        "3. create_network([...TNF pathway genes...], database='omnipath')\n"
        "4. remove_bimodal_interactions(); remove_undefined_interactions()\n"
        "5. check_disconnected_nodes(); if disconnected -> list_components(); candidate_connectors(); apply_strategy('complete_connection')\n"
        "6. list_genes_and_interactions(verbosity='preview'); find_paths('TNF','CASP3')\n"
        "7. export_network(format='bnet') -> MaBoSS run (threads=10)\n"
        "8. Evaluate & refine: filter_interactions(), get_references(), apply_strategy(), pruning\n"
        "9. Finalize -> PhysiCell integration.\n"
        "Tips: create_session() first; use verbosity='summary' in loops; candidate_connectors() before heavy strategies." )

# ===== Component & Strategy Tools =====
@mcp.tool()
def list_components(session_id: Optional[str] = None, verbosity: str = DEFAULT_VERBOSITY, format: str = 'markdown') -> str:
    """List connected components with size, average degree and sample nodes.

    Args:
        session_id: Optional session identifier.
        verbosity: summary|preview|full controls output length.
        format: 'markdown' (default) or 'json'.
    Returns:
        Component statistics (string / JSON-like string).
    """
    verbosity = normalize_verbosity(verbosity)
    sess, network = _session_network(session_id)
    if network is None:
        return E_NO_NET
    comps = _compute_components(network)
    if not comps:
        return "No components (empty network or single component)."
    # Degree map
    try:
        df = network.convert_edgelist_into_genesymbol()
    except Exception:
        df = None
    deg = {}
    if df is not None and not df.empty:
        for _, r in df.iterrows():
            s = r['source']; t = r['target']
            deg[s] = deg.get(s, 0) + 1
            deg[t] = deg.get(t, 0) + 1
    data = []
    for idx, comp in enumerate(comps):
        dvals = [deg.get(n, 0) for n in comp]
        avg_d = round(sum(dvals)/len(dvals), 2) if dvals else 0
        data.append({"id": idx, "size": len(comp), "avg_degree": avg_d, "sample": comp[:5]})
    if format == 'json':
        return str(data)
    if verbosity == 'summary':
        return f"Components={len(comps)} largest={max(len(c) for c in comps)}. {SUMMARY_HINT}"
    lines = ["Components:"]
    for row in data:
        lines.append(f"- {row['id']}: size={row['size']} avg_deg={row['avg_degree']} sample={row['sample']}")
    return "\n".join(lines)

@mcp.tool()
def candidate_connectors(method: str = 'hubs', top_k: int = 10, session_id: Optional[str] = None, format: str = 'markdown', verbosity: str = DEFAULT_VERBOSITY) -> str:
    """Suggest nodes or parameter relaxations that could bridge components.

    Methods:
        hubs          : rank high-degree nodes.
        relax_max_len : simulate increasing maxlen by +1 (estimate new edges).
        unsigned      : simulate allowing unsigned interactions.
    Args:
        method: Strategy for suggestions.
        top_k: Number of hub genes to report when method='hubs'.
    Returns:
        Markdown or JSON-like string with suggestions and rationale.
    """
    verbosity = normalize_verbosity(verbosity)
    sess, network = _session_network(session_id)
    if network is None:
        return E_NO_NET
    method = method.lower()
    try:
        df = network.convert_edgelist_into_genesymbol()
    except Exception:
        df = None
    if df is None or df.empty:
        return "No data for suggestions."
    suggestions = []
    rationale = ''
    if method == 'hubs':
        deg = {}
        for _, r in df.iterrows():
            s = r['source']; t = r['target']
            deg[s] = deg.get(s, 0) + 1
            deg[t] = deg.get(t, 0) + 1
        maxd = max(deg.values()) if deg else 1
        ranked = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:top_k]
        suggestions = [{"gene": g, "score": round(d/maxd, 3)} for g, d in ranked]
        rationale = 'High-degree nodes may help bridge components.'
    elif method in ('relax_max_len','unsigned'):
        import tempfile
        try:
            tmpdir = tempfile.mkdtemp()
            sif_path = os.path.join(tmpdir, 'tmp.sif')
            Exports(network).export_sif(sif_path)
            net_copy = Network(sif_file=sif_path, resources='omnipath')
            params = sess.get_completion_params()
            if method == 'relax_max_len':
                params['maxlen'] = params.get('maxlen', 2) + 1
            if method == 'unsigned':
                params['only_signed'] = False
            before_e = len(net_copy.edges)
            net_copy.complete_connection(**params)
            after_e = len(net_copy.edges)
            suggestions = [{"predicted_new_edges": after_e - before_e, "new_maxlen": params.get('maxlen'), "only_signed": params.get('only_signed')}]
            rationale = 'Relaxation simulation.'
        except Exception as e:
            return f"Simulation failed: {e}"
    else:
        return 'Unsupported method. Use hubs|relax_max_len|unsigned'
    if format == 'json':
        return str({"method": method, "suggestions": suggestions, "rationale": rationale})
    if verbosity == 'summary':
        return f"{method}: {len(suggestions)} suggestions. {SUMMARY_HINT}"
    lines = [f"Candidate connectors ({method}):"]
    for s in suggestions:
        if 'gene' in s:
            lines.append(f"- {s['gene']} score={s['score']}")
        else:
            lines.append(f"- {s}")
    if rationale:
        lines.append(f"Rationale: {rationale}")
    return "\n".join(lines)

@mcp.tool()
def apply_strategy(strategy: str,
                   session_id: Optional[str] = None,
                   verbosity: str = DEFAULT_VERBOSITY,
                   max_len: Optional[int] = None,
                   maxlen: Optional[int] = None,
                   algorithm: str = 'bfs',
                   minimal: bool = True,
                   only_signed: Optional[bool] = None,
                   consensus: Optional[bool] = None,
                   connect_with_bias: bool = False,
                   strategy_mode: Optional[str] = None,
                   loops: bool = False,
                   direction: Optional[str] = None,
                   outputs: Optional[List[str]] = None,
                   nodes_to_connect: Optional[List[str]] = None,
                   depth: int = 1,
                   rank: int = 1,
                   comp_a: Optional[int] = None,
                   comp_b: Optional[int] = None,
                   subgroup: Optional[List[str]] = None,
                   mode: str = 'OUT') -> str:
    """Apply a NeKo connection/completion strategy to the current network.

    Supported strategies:
        complete_connection, connect_as_atopo, connect_component, connect_network_radially,
        connect_to_upstream_nodes, connect_subgroup.
    Use candidate_connectors first to gauge potential benefits.
    Returns a brief status (verbosity='summary') or edge count after application.
    """
    verbosity = normalize_verbosity(verbosity)
    sess, network = _session_network(session_id)
    if network is None:
        return E_NO_NET
    strat = strategy.lower()
    params_defaults = sess.get_completion_params()
    if only_signed is None:
        only_signed = params_defaults.get('only_signed', True)
    if consensus is None:
        consensus = params_defaults.get('consensus', True)
    if maxlen is None and max_len is not None:
        maxlen = max_len
    changed = False
    try:
        if strat == 'complete_connection':
            strategy_complete_connection(network,
                                         maxlen=maxlen if maxlen is not None else params_defaults.get('maxlen', 2),
                                         algorithm=algorithm,
                                         minimal=minimal,
                                         only_signed=only_signed,
                                         consensus=consensus,
                                         connect_with_bias=connect_with_bias)
            changed = True
        elif strat == 'connect_as_atopo':
            connect_as_atopo(network,
                             strategy=strategy_mode,
                             max_len=maxlen if maxlen is not None else 1,
                             loops=loops,
                             outputs=outputs,
                             only_signed=only_signed,
                             consensus=consensus)
            changed = True
        elif strat == 'connect_component':
            comps = _compute_components(network)
            if comp_a is None or comp_b is None:
                return 'comp_a and comp_b required'
            if comp_a >= len(comps) or comp_b >= len(comps):
                return 'Component index out of range'
            strategy_connect_component(network,
                                       comps[comp_a],
                                       comps[comp_b],
                                       maxlen=maxlen if maxlen is not None else 2,
                                       mode=mode,
                                       only_signed=only_signed,
                                       consensus=consensus)
            changed = True
        elif strat == 'connect_network_radially':
            connect_network_radially(network,
                                      max_len=maxlen if maxlen is not None else 1,
                                      direction=direction,
                                      loops=loops,
                                      consensus=consensus,
                                      only_signed=only_signed)
            changed = True
        elif strat == 'connect_to_upstream_nodes':
            connect_to_upstream_nodes(network,
                                       nodes_to_connect=nodes_to_connect,
                                       depth=depth,
                                       rank=rank,
                                       only_signed=only_signed,
                                       consensus=consensus)
            changed = True
        elif strat == 'connect_subgroup':
            if not subgroup:
                return 'subgroup required'
            connect_subgroup(network,
                              group=subgroup,
                              maxlen=maxlen if maxlen is not None else 1,
                              only_signed=only_signed,
                              consensus=consensus)
            changed = True
        else:
            return 'Unsupported strategy'
    except Exception as e:
        return f"Strategy failed: {e}"
    if changed:
        _invalidate(sess)
    if verbosity == 'summary':
        return f"Strategy {strat} applied. {SUMMARY_HINT}"
    df = sess.get_edges_df() or pd.DataFrame()
    return f"Strategy {strat} applied. Edges now={len(df)}."

if __name__ == "__main__":
    mcp.run()