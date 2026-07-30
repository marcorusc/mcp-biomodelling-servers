"""NeKo server-side helpers.

Contains constants, private helper functions, session decorators, and
standalone utility functions that support server.py tools but are not
themselves MCP-exposed tools or resources.
"""

import os
import sys
import glob
import inspect
import re
import tempfile
import requests
from pathlib import Path
from typing import Optional, List
from functools import wraps

import pandas as pd

# ── Path bootstrap ─────────────────────────────────────────────────────────────
# Allow imports from NeKo/ (utils, session_manager) and the repo root
# (artifact_manager) without requiring an installed package.
_NEKO_ROOT = Path(__file__).parent.parent   # .../NeKo/
_REPO_ROOT = _NEKO_ROOT.parent              # .../mcp-biomodelling-servers/

for _p in (_NEKO_ROOT, _REPO_ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from neko._outputs.exports import Exports   # noqa: E402 (post-path setup)
from utils import clean_for_markdown        # noqa: E402
from session_manager import ensure_session, session_manager  # noqa: E402
from artifact_manager import get_artifact_dir  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────

#: Canonical error message for tools that require a network but find none.
E_NO_NET = "E_NO_NET: No network in session. Call create_session() then create_network()."

#: One-line hint appended to summary-verbosity responses.
SUMMARY_HINT = "Set verbosity='preview' or 'full' for more details."

#: Absolute path to the NeKo server directory; used for per-session artifact dirs.
_SERVER_ROOT: Path = _NEKO_ROOT

# ── Table helpers ──────────────────────────────────────────────────────────────

def _short_table(df: "pd.DataFrame", max_rows: int = 25):
    """Return a plain Markdown table string truncated to *max_rows*, plus a boolean flag.

    Returns:
        tuple[str, bool]: (markdown_string, was_truncated)
    """
    if df is None or df.empty:
        return "(no data)", True
    truncated = False
    if len(df) > max_rows:
        df = df.head(max_rows)
        truncated = True
    return clean_for_markdown(df).to_markdown(index=False, tablefmt="plain"), truncated

# ── Session / network accessors ────────────────────────────────────────────────

def _export_dir(session_id: Optional[str] = None) -> Path:
    """Return (and create) the per-session artifact directory for NeKo exports.

    Falls back to a shared ``exports/`` directory when *session_id* is ``None``
    (legacy behaviour; should not occur in normal usage).
    """
    if session_id:
        return get_artifact_dir(_SERVER_ROOT, session_id)
    # Legacy fallback
    d = _SERVER_ROOT / "exports"
    d.mkdir(exist_ok=True)
    return d


def _session_network(session_id: Optional[str]):
    """Return ``(sess, sess.network)`` for the given (or default) session."""
    sess = ensure_session(session_id)
    return sess, sess.network


def _invalidate(sess) -> None:
    """Invalidate the edge cache on *sess* after a mutation."""
    if sess:
        sess.invalidate_edges_cache()

# ── Decorators ────────────────────────────────────────────────────────────────

def session_locked(fn):
    """Run a synchronous handler under its session's exclusive lease."""
    signature = inspect.signature(fn)

    @wraps(fn)
    def inner(*args, **kwargs):
        arguments = signature.bind_partial(*args, **kwargs).arguments
        with session_manager.session_scope(arguments.get("session_id")):
            return fn(*args, **kwargs)

    return inner

def requires_network(fn):
    """Decorator that guards tools requiring an active network.

    Injects ``sess`` and ``network`` keyword arguments into the decorated
    function. Raises a recoverable tool error if no network exists in the
    current session.
    """
    signature = inspect.signature(fn)
    public_signature = signature.replace(
        parameters=[
            parameter
            for name, parameter in signature.parameters.items()
            if name not in {"sess", "network"}
        ]
    )

    @wraps(fn)
    def inner(*args, **kwargs):
        arguments = public_signature.bind_partial(*args, **kwargs).arguments
        session_id = arguments.get("session_id")
        with session_manager.session_scope(session_id):
            sess, network = _session_network(session_id)
            if network is None:
                raise RuntimeError(E_NO_NET)
            kwargs["sess"] = sess
            kwargs["network"] = network
            return fn(*args, **kwargs)

    # These are internal injected values, not caller-controlled MCP inputs.
    inner.__signature__ = public_signature
    return inner

# ── Graph helpers ──────────────────────────────────────────────────────────────

def _compute_components(network) -> List[List[str]]:
    """Compute connected components (including isolated nodes) via iterative DFS.

    Returns a list of node-lists sorted from largest to smallest component.
    """
    # Seed all known nodes so that isolates appear as singleton components.
    try:
        all_nodes = set(network.nodes["Uniprot"].tolist())
    except Exception:
        all_nodes = set()

    adj: dict = {str(n): set() for n in all_nodes if pd.notna(n) and n != ""}

    # Build undirected adjacency from the edge list.
    try:
        sources = network.edges["source"].tolist()
        targets = network.edges["target"].tolist()
        for s, t in zip(sources, targets):
            if pd.notna(s) and pd.notna(t) and s != "" and t != "":
                s_str, t_str = str(s), str(t)
                adj.setdefault(s_str, set()).add(t_str)
                adj.setdefault(t_str, set()).add(s_str)
    except Exception:
        pass

    if not adj:
        return []

    visited: set = set()
    comps: List[List[str]] = []

    for n in adj:
        if n in visited:
            continue
        stack = [n]
        cur: List[str] = []
        visited.add(n)
        while stack:
            node = stack.pop()
            cur.append(node)
            for nb in adj.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        comps.append(cur)

    comps.sort(key=len, reverse=True)
    return comps

# ── File utilities ─────────────────────────────────────────────────────────────

def download_signor_database() -> str:
    """Download SIGNOR human interaction data (TSV) from the SIGNOR API and save locally."""
    url = "https://signor.uniroma2.it/API/getHumanData.php"
    try:
        r = requests.get(url)
        r.raise_for_status()
        output_file = "SIGNOR_Human.tsv"
        with open(output_file, "wb") as f:
            f.write(r.content)
        return "SIGNOR database downloaded successfully."
    except requests.RequestException as e:
        return f"Error downloading SIGNOR database: {str(e)}"


def sanitize_bnet_file(path: str) -> dict:
    """Strip the BoolNet header, clean node names, and deduplicate rows in-place.

    Performs three passes over a single ``.bnet`` file:

    1. **Header strip** — removes the ``# model in BoolNet format`` /
       ``targets, factors`` preamble written by NeKo's exporter.
    2. **Name cleaning** — replaces any character outside ``[A-Za-z0-9_]``
       with ``_`` in both the LHS gene column and every reference to that
       gene inside the RHS Boolean expression.
    3. **Deduplication** — when two Uniprot IDs map to the same gene symbol
       (e.g. isoforms), NeKo writes two rows with the same cleaned LHS name.
       Only the first rule is kept; subsequent duplicates are silently dropped
       because a valid BNET file must have unique targets.

    Args:
        path: Absolute path to the ``.bnet`` file to sanitize (modified in-place).

    Returns:
        dict with keys:
            ``cleaned_names`` (set[str]) — original names that were renamed.
            ``name_mapping`` (dict[str, str]) — renamed originals mapped to
            their sanitized MaBoSS node names.
            ``duplicates_removed`` (list[str]) — cleaned names whose extra rows were dropped.
    """
    _clean = lambda name: re.sub(r"[^A-Za-z0-9_]", "_", name)

    with open(path, "r") as fh:
        raw_lines = fh.readlines()

    # 1. Strip header
    lines = raw_lines
    if (
        len(lines) >= 2
        and lines[0].strip() == "# model in BoolNet format"
        and lines[1].strip() == "targets, factors"
    ):
        lines = lines[2:]

    # 2. Build name map (original -> cleaned) from LHS column
    name_map: dict = {}
    cleaned_names: set = set()
    for line in lines:
        if "," not in line:
            continue
        gene = line.split(",", 1)[0].strip()
        gene_clean = _clean(gene)
        name_map[gene] = gene_clean
        if gene_clean != gene:
            cleaned_names.add(gene)

    # 3. Rewrite with cleaned names and deduplicate LHS
    seen_lhs: set = set()
    duplicates_removed: list = []
    new_lines: list = []
    for line in lines:
        if "," not in line:
            new_lines.append(line)
            continue
        gene, expr = line.split(",", 1)
        gene_clean = name_map.get(gene.strip(), gene.strip())

        if gene_clean in seen_lhs:
            duplicates_removed.append(gene_clean)
            continue
        seen_lhs.add(gene_clean)

        # Rename all references inside the Boolean expression too
        expr_clean = expr
        for orig, clean in name_map.items():
            if orig != clean:
                expr_clean = re.sub(
                    rf"(?<![A-Za-z0-9_]){re.escape(orig)}(?![A-Za-z0-9_])",
                    clean,
                    expr_clean,
                )
        new_lines.append(f"{gene_clean},{expr_clean}")

    with open(path, "w") as fh:
        fh.writelines(new_lines)

    return {
        "cleaned_names": cleaned_names,
        "name_mapping": {
            original: cleaned
            for original, cleaned in name_map.items()
            if original != cleaned
        },
        "duplicates_removed": duplicates_removed,
    }


# Keep the old name as a thin alias so existing callers don't break immediately.
def clean_bnet_headers(folder_path: str = ".") -> str:  # pragma: no cover
    """Deprecated — use sanitize_bnet_file() instead."""
    import warnings
    warnings.warn("clean_bnet_headers is deprecated; use sanitize_bnet_file.", DeprecationWarning)
    results = []
    for fp in glob.glob(os.path.join(folder_path, "*.bnet")):
        r = sanitize_bnet_file(fp)
        results.append(os.path.basename(fp))
    return f"Sanitized: {', '.join(results)}" if results else "No .bnet files found."

def _get_translators(network):
    """Builds fast, two-way translation dictionaries for the network."""
    try:
        # Extract the node registry dataframe
        df = network.nodes
        
        # Ensure we only map valid rows
        clean_df = df.dropna(subset=['Uniprot', 'Genesymbol'])
        
        # Build the dictionaries
        uniprot_to_symbol = dict(zip(clean_df['Uniprot'], clean_df['Genesymbol']))
        symbol_to_uniprot = dict(zip(clean_df['Genesymbol'], clean_df['Uniprot']))
        
        return uniprot_to_symbol, symbol_to_uniprot
    except Exception as e:
        # Fallback to empty dicts if columns are missing
        return {}, {}
