import pandas as pd

def format_unsupported_format_guidance(requested_format: str) -> str:
    """Provide guidance for unsupported export formats."""
    
    response = f"## Export Format Not Supported\n\n"
    response += f"**Requested format:** {requested_format}\n\n"
    response += f"### Available export formats:\n\n"
    response += f"**1. SIF format** (recommended for MaBoSS):\n"
    response += f"```\n"
    response += f"export_network('sif')\n"
    response += f"```\n"
    response += f"- Creates: Network.sif\n"
    response += f"- Format: source<TAB>interaction<TAB>target\n"
    response += f"- Use case: Import into MaBoSS for Boolean simulation\n\n"
    
    response += f"**2. BNET format** (for Boolean analysis):\n"
    response += f"```\n"
    response += f"export_network('bnet')\n"
    response += f"```\n"
    response += f"- Creates: Network.bnet\n"
    response += f"- Format: gene,Boolean_expression\n"
    response += f"- Use case: PyBoolNet, MaBoSS, other Boolean tools\n"
    response += f"- Requirement: Network must be connected\n\n"
    
    response += f"**Format comparison:**\n"
    response += f"- **SIF:** Works with any network, good for visualization/analysis\n"
    response += f"- **BNET:** Requires connected network, ready for Boolean simulation\n\n"
    
    response += f"**Next step:** Choose `export_network('sif')` or `export_network('bnet')`"
    
    return response

def format_no_network_for_modification(action: str, gene: str) -> str:
    """Provide guidance when trying to modify genes but no network exists."""
    
    response = f"## Gene {action.title()} Failed: No Network Available\n\n"
    response += f"**Target gene:** {gene}\n"
    response += f"**Action:** {action}\n\n"
    response += f"### Required steps:\n\n"
    response += f"**1. Create a network first:**\n"
    response += f"```\n"
    response += f"create_network(['{gene}', 'OTHER_GENES'])\n"
    response += f"```\n\n"
    response += f"**2. Then modify as needed:**\n"
    if action == "add":
        response += f"- Add more genes: `add_gene('GENE_NAME')`\n"
        response += f"- Build connections: Network will auto-complete pathways\n"
    else:
        response += f"- Remove genes: `remove_gene('GENE_NAME')`\n"
        response += f"- Network will maintain remaining connections\n"
    
    response += f"\n**Workflow suggestion:**\n"
    response += f"1. `create_network(['{gene}'])` → Start with your target gene\n"
    response += f"2. `add_gene('RELATED_GENE')` → Expand network\n"
    response += f"3. `export_network('sif')` → Save for downstream analysis"
    
    return response

def format_connectivity_guidance() -> str:
    """Provide guidance when network isn't connected for BNET export."""
    
    response = f"## BNET Export Failed: Network Not Connected\n\n"
    response += f"**Issue:** Boolean networks require fully connected components\n\n"
    response += f"### Solutions to improve connectivity:\n\n"
    response += f"**1. Expand pathway length:**\n"
    response += f"```\n"
    response += f"create_network(your_genes, max_len=3)  # or max_len=4\n"
    response += f"```\n\n"
    response += f"**2. Include unsigned interactions:**\n"
    response += f"```\n"
    response += f"create_network(your_genes, only_signed=False)\n"
    response += f"```\n\n"
    response += f"**3. Add bridge genes:**\n"
    response += f"- Add central hub genes (e.g., TP53, MYC, CTNNB1)\n"
    response += f"- Include pathway connectors\n"
    response += f"- Use `add_gene('HUB_GENE')` to manually connect components\n\n"
    response += f"**4. Try different algorithm:**\n"
    response += f"```\n"
    response += f"create_network(your_genes, algorithm='dfs')\n"
    response += f"```\n\n"
    response += f"**Alternative:** Export as SIF instead for partially connected networks:\n"
    response += f"```\n"
    response += f"export_network('sif')  # Works with disconnected components\n"
    response += f"```\n\n"
    response += f"**Workflow tip:** Use `get_network()` to check current connectivity before export"
    
    return response

def format_no_network_guidance() -> str:
    """Provide guidance when trying to export but no network exists."""
    
    response = f"## Export Failed: No Network Available\n\n"
    response += f"**Status:** No network exists to export\n\n"
    response += f"### Required steps:\n\n"
    response += f"**1. Create a network first:**\n"
    response += f"```\n"
    response += f"create_network(['TP53', 'MYC', 'RB1'])\n"
    response += f"```\n\n"
    response += f"**2. Then export in your desired format:**\n"
    response += f"- SIF format (for MaBoSS): `export_network('sif')`\n"
    response += f"- BNET format (for Boolean analysis): `export_network('bnet')`\n\n"
    response += f"**Available export formats:**\n"
    response += f"- **SIF:** Tab-separated network file (source → target)\n"
    response += f"- **BNET:** Boolean network format for MaBoSS/PyBoolNet\n\n"
    response += f"**Workflow suggestion:**\n"
    response += f"1. `create_network(['YOUR_GENES'])` → Build network\n"
    response += f"2. `export_network('sif')` → Export for MaBoSS\n"
    response += f"3. Use exported file in MaBoSS server for Boolean simulation"
    
    return response

def format_no_input_guidance() -> str:
    """Provide guidance when no input is provided to create_network."""
    
    response = f"## Network Creation: Input Required\n\n"
    response += f"**Error:** No initial genes or SIF file provided\n\n"
    response += f"### Choose one of these approaches:\n\n"
    response += f"**1. Start with gene list:**\n"
    response += f"```\n"
    response += f"create_network(['TP53', 'MYC', 'RB1'])\n"
    response += f"```\n"
    response += f"- Use 2-5 genes for focused networks\n"
    response += f"- Gene symbols should be human/mouse standard\n"
    response += f"- Common starting points: tumor suppressors, oncogenes, pathway genes\n\n"
    
    response += f"**2. Import existing network:**\n"
    response += f"```\n"
    response += f"create_network([], sif_file='path/to/network.sif')\n"
    response += f"```\n"
    response += f"- SIF format: source<TAB>interaction<TAB>target\n"
    response += f"- Can combine with genes: `create_network(['GENE1'], sif_file='path.sif')`\n\n"
    
    response += f"**3. Example workflows:**\n"
    response += f"- **Cancer pathway:** `create_network(['TP53', 'MDM2', 'CDKN1A'])`\n"
    response += f"- **Cell cycle:** `create_network(['CCND1', 'CDK4', 'RB1'])`\n"
    response += f"- **Apoptosis:** `create_network(['BCL2', 'BAX', 'CASP3'])`\n\n"
    
    response += f"**Available databases:** 'omnipath' (default), 'signor'\n"
    response += f"**Next step:** Choose your approach and try again!"
    
    return response

def format_network_creation_error(error_type: str, genes: list, error_msg: str) -> str:
    """Format enhanced error messages with actionable recovery guidance."""
    
    if error_type == "build_failed":
        response = f"## Network Creation Failed\n\n"
        response += f"**Error:** {error_msg}\n\n"
        response += f"**Input genes:** {', '.join(genes)}\n\n"
        response += f"### Possible Solutions:\n"
        response += f"1. **Check gene names:** Verify gene symbols are correct (human/mouse standard)\n"
        response += f"2. **Try different database:** Switch from 'omnipath' to 'signor' or vice versa\n"
        response += f"3. **Adjust parameters:**\n"
        response += f"   - Increase `max_len` from 2 to 3 for longer pathways\n"
        response += f"   - Set `only_signed=False` to include unsigned interactions\n"
        response += f"   - Try `algorithm='dfs'` instead of 'bfs'\n"
        response += f"4. **Start smaller:** Begin with 2-3 well-known genes (e.g., 'TP53', 'MYC')\n\n"
        response += f"**Next step:** Use `create_network()` again with modified parameters"
        return response
    
    return f"Network creation error: {error_msg}"

def format_empty_network_response(genes: list, database: str, max_len: int, only_signed: bool) -> str:
    """Format enhanced response for empty networks with actionable guidance."""
    
    # Build an empty table with the expected columns for consistency
    empty_df = pd.DataFrame(columns=["source", "target", "Effect"])
    table_preview = clean_for_markdown(empty_df).to_markdown(index=False, tablefmt="plain")
    
    response = f"## Empty Network Created\n\n"
    response += f"**Status:** Network created but no interactions found\n\n"
    response += f"**Input details:**\n"
    response += f"- Genes: {', '.join(genes)}\n"
    response += f"- Database: {database}\n"
    response += f"- Max path length: {max_len}\n"
    response += f"- Only signed: {only_signed}\n\n"
    
    response += f"### Actionable Solutions:\n"
    response += f"1. **Expand search scope:**\n"
    response += f"   - `create_network({genes}, max_len=3)` (longer pathways)\n"
    response += f"   - `create_network({genes}, only_signed=False)` (include unsigned)\n"
    response += f"2. **Try alternative database:**\n"
    response += f"   - `create_network({genes}, database='signor')` if using omnipath\n"
    response += f"   - `create_network({genes}, database='omnipath')` if using signor\n"
    response += f"3. **Add more seed genes:**\n"
    response += f"   - Include upstream regulators or downstream targets\n"
    response += f"   - Try pathway-specific genes (e.g., cell cycle, apoptosis)\n"
    response += f"4. **Manual network building:**\n"
    response += f"   - Use `add_gene('GENE_NAME')` to add nodes manually\n"
    response += f"   - Import existing SIF file with `create_network([], sif_file='path.sif')`\n\n"
    
    response += f"**Current network structure:**\n\n{table_preview}\n\n"
    response += f"**Note:** Empty network is still usable - you can add genes manually or export as starting point"
    
    return response

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