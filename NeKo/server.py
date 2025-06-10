import io
import sys
import logging
import os
import glob
import requests
from hatchling import HatchMCP

from neko.core.network import Network
from neko._outputs.exports import Exports
from neko.inputs import Universe, signor
from neko.core.tools import is_connected

import pandas as pd

# Initialize MCP server with metadata
hatch_mcp = HatchMCP("NeKo",
                origin_citation="Origin citation for NeKo",
                mcp_citation="MCP citation for NeKo")

# Global network object for persistent state
network = None

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
@hatch_mcp.server.tool()
def create_network(list_of_initial_genes: list[str], database="omnipath", sif_file: str=None) -> str:
    """
    Create a NeKo network from a list of genes and/or a SIF file.
    If the list of genes is empty but a SIF file is provided, load the network from the SIF file.
    If the list of genes is not empty and there is no SIF file, use just the list of genes.
    If both are provided, load the network from the SIF file and then add all genes in the list.
    Args:
        list_of_initial_genes (list[str]): List of gene symbols.
        database (str): Database to use for network creation, either 'omnipath' or 'signor'.
        sif_file (str): Path to a SIF file to load the network from.
    Returns:
        str: Status message or Markdown formatted string with network summary.
    """
    global network
    hatch_mcp.logger.info(f"Creating NeKo network with initial genes: {list_of_initial_genes} and SIF file: {sif_file}")

    # Validate database choice
    if database not in ["omnipath", "signor"]:
        return "_Unsupported database. Use `omnipath` or `signor`._"

    # If using SIGNOR, download and build the SIGNOR resource
    if database == "signor":
        hatch_mcp.logger.info("Downloading SIGNOR database...")
        download_signor_database()
        hatch_mcp.logger.info("SIGNOR database downloaded successfully.")
        signor_res = signor("SIGNOR_Human.tsv")
        signor_res.build()
        resources = signor_res.interactions
    else:
        resources = "omnipath"

    # Case 1: SIF file provided (with or without genes)
    if sif_file is not None and os.path.exists(sif_file):
        # Use NeKo's documented SIF loading
        try:
            network = Network(sif_file=sif_file, resources=resources)
        except Exception as e:
            return f"**Error**: Unable to create network from SIF file. {str(e)}"
        # If genes are also provided, add them
        if list_of_initial_genes:
            for gene in list_of_initial_genes:
                network.add_node(gene)
    # Case 2: Only genes provided
    elif list_of_initial_genes:
        network = Network(list_of_initial_genes, resources=resources)
        network.complete_connection(
            maxlen=3,
            algorithm="bfs",
            only_signed=True,
            connect_with_bias=False,
            consensus=True
        )
    # Case 3: Neither provided
    else:
        return "_No initial genes or SIF file provided. Please provide at least one._"

    # If there are no edges, return a Markdown note
    try:
        df_edges = network.convert_edgelist_into_genesymbol()
    except Exception as e:
        return f"**Error**: Unable to build network. {str(e)}"

    if df_edges.empty:
        hatch_mcp.logger.warning("No interactions found in the network. Please check the input parameters.")
        # Build an empty table with the expected columns
        empty_df = pd.DataFrame(columns=["source", "target", "Type", "Effect", "References"])
        return "_No interactions found in the network._\n\n" + clean_for_markdown(empty_df).to_markdown(index=False, tablefmt="plain")

    # Compute basic statistics
    num_edges = len(df_edges)
    unique_nodes = pd.unique(df_edges[["source", "target"]].values.ravel())
    num_nodes = len(unique_nodes)

    hatch_mcp.logger.info("Network created successfully.")

    # Prepare a preview of the first 100 interactions
    preview_df = df_edges.head(100)
    preview_md = clean_for_markdown(preview_df).to_markdown(index=False, tablefmt="plain")

    # Build the Markdown summary
    md_lines = [
        "**Network created successfully!**",
        "",
        f"- **Initial genes:** {', '.join(list_of_initial_genes)}",
        f"- **Total nodes:** {num_nodes}",
        f"- **Total interactions:** {num_edges}",
        "",
        "**Preview of interactions (first 100 rows):**",
        "",
        preview_md
    ]

    return "\n".join(md_lines)

@hatch_mcp.server.tool()
def add_gene(gene: str) -> str:
    """
    Add a gene to the current network. The gene must be a valid gene symbol.
    If no network exists, it prompts the user to create one first.
    Args:
        gene (str): Gene symbol to add.
    Returns:
        str: Status message.
    """
    global network
    if network is None:
        return "No network exists. Please create a network first."
    network.add_node(gene)
    return f"Gene {gene} added to the network."

@hatch_mcp.server.tool()
def remove_gene(gene: str) -> str:
    """
    Remove a gene from the current network. The gene must be a valid gene symbol.
    If no network exists, it prompts the user to create one first.
    If the gene is not found in the network, it returns an error message.
    Args:
        gene (str): Gene symbol to remove.
    Returns:
        str: Status message.
    """
    global network
    if network is None:
        return "No network exists. Please create a network first."
    network.remove_node(gene)
    return f"Gene {gene} removed from the network."

@hatch_mcp.server.tool()
def export_network(format: str = "sif") -> str:
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
    global network

    if network is None:
        return "_No network exists. Please create a network first._"

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
        out_path = "Network.sif"
        try:
            exporter.export_sif(out_path)
        except Exception as e:
            return f"**Error exporting SIF:** {str(e)}"

        # Now build a Markdown snippet showing the path + preview
        md_lines = [
            f"**Exported to** `{out_path}`\n",
            "_Preview (first 100 lines) of the SIF file:_",
            ""
        ]
        # SIF format is: source<TAB>interaction<TAB>target
        preview_md = _preview_file(out_path, sep="\t", cols=["source", "interaction", "target"])
        md_lines.append(preview_md)
        return "\n".join(md_lines)

    # 2) Handle BNET export
    elif format.lower() == "bnet":
        # Check connectivity first
        if not is_connected(network):
            return "_Network is not fully connected. Please ensure the network is connected before exporting as bnet._"

        # Export
        try:
            exporter.export_bnet("./Network")
            clean_bnet_headers()
        except Exception as e:
            return f"**Error exporting BNET:** {str(e)}"

        # Find all .bnet files in the current directory
        bnet_files = [os.path.basename(f) for f in glob.glob("*.bnet")]
        if not bnet_files:
            return "**Error:** No .bnet files were generated."
        out_path = bnet_files[0]

        # Check for special characters in gene/node names in the bnet file
        special_char_issues = []
        try:
            with open(out_path, "r") as f:
                for i, line in enumerate(f):
                    if i >= 1000:
                        break  # Only check the first 1000 lines for performance
                    if "," in line:
                        gene = line.split(",", 1)[0].strip()
                        # Allow only alphanumerics and underscores
                        import re
                        if re.search(r"[^A-Za-z0-9_]", gene):
                            special_char_issues.append(gene)
        except Exception as e:
            return f"**Error reading BNET file for gene name check:** {str(e)}"

        md_lines = [
            f"**Exported to** `{out_path}`\n",
            "_Preview (first 100 lines) of the BNET file:_",
            ""
        ]
        # BNET format: typically "Gene, Boolean_expression"
        preview_md = _preview_file(out_path, sep=",", cols=["gene", "expression"])
        md_lines.append(preview_md)
        if len(bnet_files) > 1:
            md_lines.append(f"\n_Warning: More than one .bnet file was found ({', '.join(bnet_files)}). Previewing only the first one._")
        if special_char_issues:
            md_lines.append(f"\n**Warning:** The following gene/node names contain special characters and may not be compatible: {', '.join(sorted(set(special_char_issues)))}")
        return "\n".join(md_lines)

    # 3) Unsupported format
    else:
        return "_Unsupported format. Use `sif` or `bnet`._"

@hatch_mcp.server.tool()
def network_dimension() -> str:
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
    global network
    if network is None:
        return "No network exists. Please create a network first."
    return f"Nodes: {len(network.nodes)}, Edges: {len(network.edges)}"

@hatch_mcp.server.tool()
def list_genes_and_interactions() -> str:
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
    global network

    if network is None:
        # Use known column names to build an empty table
        cols = ["source", "target", "Type", "Effect", "References"]
        empty_df = pd.DataFrame(columns=cols)
        return "_No network loaded._\n\n" + clean_for_markdown(empty_df).to_markdown(index=False, tablefmt="plain")

    try:
        df = network.convert_edgelist_into_genesymbol()
        # Exclude the 'resources' column if present
        if "resources" in df.columns:
            df = df.drop(columns=["resources"])
        if df.empty:
            return "_Network loaded but contains no interactions._\n\n" + clean_for_markdown(df).to_markdown(index=False, tablefmt="plain")
        return clean_for_markdown(df).to_markdown(index=False, tablefmt="plain")

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

@hatch_mcp.server.tool()
def find_paths(source: str, target: str, maxlen: int = 3) -> str:
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
    global network

    if network is None:
        return "_No network exists. Please create a network first._"

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
            return "_No paths found._"

        # Wrap the raw output in a fenced code block for Markdown
        md = [
            f"**Paths from `{source}` to `{target}` (max length = {maxlen}):**",
            "",
            "```",
            raw_output,
            "```"
        ]
        return "\n".join(md)

    except Exception as e:
        # Ensure stdout is restored even on error
        sys.stdout = old_stdout
        return f"**Error:** {str(e)}"

@hatch_mcp.server.tool()
def reset_network() -> str:
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
    global network
    network = None
    return "Network reset."

@hatch_mcp.server.tool()
def clean_generated_files(folder_path: str = ".") -> str:
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

@hatch_mcp.server.tool()
def get_help() -> str:
    """
    Get a description of available NeKo MCP tools and their usage.
    Returns:
        str: Help string.
    """
    return (
        "Available tools:\n"
        "- create_network(list_of_initial_genes)\n"
        "- add_gene(gene)\n"
        "- remove_gene(gene)\n"
        "- export_network(format)\n"
        "- network_summary()\n"
        "- list_genes_and_interactions()\n"
        "- find_paths(source, target, maxlen)\n"
        "- reset_network()\n"
        "- get_help()\n"
    )

@hatch_mcp.server.tool()
def remove_bimodal_interactions() -> str:
    """
    Remove all 'bimodal' interactions from the current network object in memory.
    Returns:
        str: Status message.
    """
    global network
    if network is None:
        return "No network exists. Please create a network first."
    if "Effect" not in network.edges.columns:
        return "No 'Effect' column found in network.edges."
    before = len(network.edges)
    network.edges = network.edges[network.edges["Effect"].str.lower() != "bimodal"]
    after = len(network.edges)
    removed = before - after
    return f"Removed {removed} bimodal interactions from the network."

@hatch_mcp.server.tool()
def remove_undefined_interactions() -> str:
    """
    Remove all 'undefined' interactions from the current network object in memory.
    Returns:
        str: Status message.
    """
    global network
    if network is None:
        return "No network exists. Please create a network first."
    if "Effect" not in network.edges.columns:
        return "No 'Effect' column found in network.edges."
    before = len(network.edges)
    network.edges = network.edges[network.edges["Effect"].str.lower() != "undefined"]
    after = len(network.edges)
    removed = before - after
    return f"Removed {removed} undefined interactions from the network."


@hatch_mcp.server.tool()
def list_bnet_files(folder_path: str = ".") -> list:
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
    
@hatch_mcp.server.tool()
def check_bnet_files_names(folder_path: str = ".") -> str:
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

@hatch_mcp.server.tool()
def check_disconnected_nodes() -> str:
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
    global network
    if network is None:
        return "No network exists. Please create a network first."
    
    all_nodes = set(network.nodes["Uniprot"].tolist())
    connected_nodes = set(network.edges["source"].tolist()) | set(network.edges["target"].tolist())
    disconnected_nodes = all_nodes - connected_nodes
    disconnected_nodes = [node for node in disconnected_nodes if pd.notna(node) and node != ""]
    
    if not disconnected_nodes:
        return "All nodes are connected."
    
    return "Disconnected nodes:\n" + "\n".join(disconnected_nodes)

def clean_for_markdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Convert every cell to a string.
    2. Strip leading/trailing whitespace and collapse any internal whitespace to a single space.
    3. Replace 'nan' or entirely-blank cells with an empty string.
    4. Drop columns and rows that end up completely empty.
    """
    # 1) Make sure everything is a string so strip/regex-replace works
    df_str = df.astype(str)

    # 2) Strip leading/trailing whitespace, then collapse any run of whitespace/newlines to a single space
    df_str = df_str.applymap(lambda val: " ".join(val.split()))

    # 3) Replace the literal string 'nan' (that pandas sometimes shows for NaNs) with an actual empty string
    df_str = df_str.replace("nan", "", regex=False)

    # 4) Drop any columns that are now entirely empty
    df_str = df_str.dropna(axis=1, how="all")  # drop cols where every entry is NaN (after replacement, NaN still possible)
    df_str = df_str.loc[:, (df_str != "").any(axis=0)]  # also drop columns that are all empty strings

    # 5) Drop any rows that are now entirely empty
    df_str = df_str.dropna(axis=0, how="all")
    df_str = df_str.loc[(df_str != "").any(axis=1), :]

    return df_str

if __name__ == "__main__":
    hatch_mcp.logger.info("Starting MCP server")
    hatch_mcp.server.run()
