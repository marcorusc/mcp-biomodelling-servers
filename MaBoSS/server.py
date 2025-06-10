import logging
import os
import glob
from pathlib import Path
from hatchling import HatchMCP

import maboss
import pandas as pd

# Initialize MCP server with metadata
hatch_mcp = HatchMCP("MaBoSS",
                origin_citation="Origin citation for MaBoSS",
                mcp_citation="MCP citation for MaBoSS")

sim = None  # Global variable to hold the simulation state

result = None  # Global variable to hold the result of the last simulation

# tool for creating the bnd and the cfg files from a bnet file
@hatch_mcp.server.tool()
def bnet_to_bnd_and_cfg(bnet_path: str) -> str:
    """
    This function processes a bnet file and generates
    the corresponding BND and CFG files using MaBoSS.
    It calls the MaBoSS tool with the path to the bnet file,
    and returns the result of the simulation creation.
    Args:
        bnet_path (str): Path to the .bnet file.
    Returns:
        str: Processed result.
    """
    hatch_mcp.logger.info(f"Processing with MaBoSS tool. Input .bnet file: {bnet_path}")
    maboss.bnet_to_bnd_and_cfg(bnet_path, "output.bnd", "output.cfg")
    # check if the files were created
    # if the bnd and cfg files are created, return the paths
    # if not thrown an error
    try:
        with open("output.bnd", "r") as bnd_file:
            bnd_content = bnd_file.read()
            hatch_mcp.logger.info(f"Read BND content: {bnd_content}")
        with open("output.cfg", "r") as cfg_file:
            cfg_content = cfg_file.read()
            hatch_mcp.logger.info(f"Read CFG content: {cfg_content}")
        return "MaBoSS bnd and cfg files created successfully. Output files: output.bnd, output.cfg"
    except FileNotFoundError as e:
        hatch_mcp.logger.error(f"Error reading output files: {e}")
        return "Error in MaBoSS simulation creation. Output files not found."
    
@hatch_mcp.server.tool()
def run_simulation(bnd_path: str = "output.bnd", cfg_path: str = "output.cfg") -> str:
    """
    When the user requests to run a simulation,
    this function runs a MaBoSS simulation using the provided BND and CFG files.
    It loads the simulation and returns the result.
    This function is designed to be called after the BND and CFG files have been created.
    It initializes the simulation with the specified BND and CFG files,
    runs the simulation, and stores the result in a global variable.
    It also logs the process and handles any errors that may occur during the simulation run.
    Args:
        bnd_path (str): Path to the .bnd file.
        cfg_path (str): Path to the .cfg file.
    Returns:
        str: Result of the simulation.
    """
    global sim, result
    hatch_mcp.logger.info(f"Running MaBoSS simulation with BND: {bnd_path} and CFG: {cfg_path}")
    sim = maboss.load(bnd_path, cfg_path)
    if sim:
        result = sim.run()
        hatch_mcp.logger.info("MaBoSS simulation completed successfully.")
        return "MaBoSS simulation completed successfully."
    else:
        hatch_mcp.logger.error("Failed to create MaBoSS simulation.")
        return "Error in MaBoSS simulation creation."
    
@hatch_mcp.server.tool()
def get_simulation_result() -> str:
    """
    When the user requests the simulation result,
    or the last simulation result,
    this function retrieves the last simulation result and formats it as a Markdown table.

    Parse and interpret the Markdown-formatted MaBoSS state probability trajectory.

    Output Format
    -------------
    The Markdown string follows this structure:

    1. **Title and Subtitle**  
       - A bold title line:  
         **MaBoSS Simulation: State Probability Trajectory**  
       - An italicized subtitle immediately below:  
         _Below is the probability trajectory of each state over time:_  

    2. **Table Header Row (State Identifiers)**  
       - A single header row listing all unique Boolean states as column names.  
       - Each column header is a concatenation of active node names joined by `--`.  
         For example:  
         ```
         | NodeA -- NodeB -- NodeC | NodeD -- NodeE | NodeF -- NodeG -- NodeH | …
         ```  
       - Interpretation:  
         - Any node listed in a header string is ON (True) in that state.  
         - Any node not listed is OFF (False).  

    3. **Markdown Alignment Row**  
       - Directly under the header row is the alignment specifier, for example:  
         ```
         |:-----------------------:|:-----------------:|:-------------------------:| … |
         ```  
       - This row only controls visual alignment (left/center/right) when rendering Markdown.  
       - It carries no biological meaning.  

    4. **Single Probability Row (Final Timepoint)**  
       - Exactly one row of numeric probability values appears below the alignment row.  
       - Each value corresponds to the probability of the network being in the state named by its column.  
       - Example:  
         ```
         | 0.30 | 0.45 | 0.20 | 0.05 |
         ```  
       - Interpretation:  
         - Since this is a single (final) snapshot, these probabilities sum to (approximately) 1.  
         - Each probability indicates how likely the network is in that particular state at simulation end.

    How to Interpret
    ----------------
    - **Columns** = Distinct Boolean states (sets of ON nodes).  
    - **Row** = Final timepoint of the simulation (one snapshot).  
    - **Cell Values** = Probability of being in that state at the final time.  
    - Because there is only one row, this table conveys only the terminal distribution of states.  
      If state X has probability 0.45, it means 45% of simulated trajectories converge to X at the end.

    Edge Cases
    ----------
    - If the simulation failed or returned no states, the Markdown may instead be:
      ```
      _No paths found._  
      ```
      or  
      ```
      **Error:** <error message>
      ```
    - In such cases, there will be no header row or probability row.

    Parameters
    ----------
    markdown_text : str
        The raw Markdown string produced by the MaBoSS simulation tool. This docstring explains its structure 
        and how to map each section to biological meaning.  

    Returns
    -------
    None
        This function is purely documentation/specification. Downstream AI agents should follow this 
        schema to parse or validate the Markdown output.
    """
    global result

    if result is None:
        return "_No simulation has been run yet._"

    try:
        # Retrieve the probability trajectory DataFrame
        df_prob = result.get_last_states_probtraj()

        if df_prob.empty:
            return "_Simulation completed but returned no trajectory data._"

        # Convert the DataFrame into a Markdown table
        df_prob = clean_for_markdown(df_prob)  # Clean the DataFrame for Markdown compatibility
        md_table = df_prob.to_markdown(index=False, tablefmt="plain")

        md_lines = [
            "**MaBoSS Simulation: State Probability Trajectory**",
            "",
            "_Below is the probability trajectory of each state over time:_",
            "",
            md_table
        ]
        return "\n".join(md_lines)

    except Exception as e:
        return f"**Error retrieving simulation result:** {str(e)}"
    
@hatch_mcp.server.tool()
def check_bnd_and_cfg_name() -> str:
    """
    When the user requests to check the existence of BND (*.bnd) and CFG (*.cfg) files,
    or if it asks for the names of the BND and CFG files,
    this function checks if any files *.bnd and *.cfg exist in the current directory.
    It logs the check process and returns the names of the files if they exist,
    or an error message if they do not exist.
    Returns:
        str: Names of the BND and CFG files or an error message if they do not exist.
    """

    bnd_files = glob.glob("*.bnd")
    cfg_files = glob.glob("*.cfg")

    if not bnd_files and not cfg_files:
        return "No BND or CFG files found in the current directory."

    file_list = []
    if bnd_files:
        file_list.append(f"BND files: {', '.join(bnd_files)}")
    if cfg_files:
        file_list.append(f"CFG files: {', '.join(cfg_files)}")

    return "\n".join(file_list)
    
@hatch_mcp.server.tool()
def clean_bnd_and_cfg() -> str:
    """
    When the user requests to clean up the BND and CFG files,
    this function removes the output.bnd and output.cfg files created by MaBoSS.
    It logs the cleanup process and handles any errors that may occur during file removal.
    Returns:
        str: Confirmation message indicating successful cleanup.
    """
    import os

    try:
        if os.path.exists("output.bnd"):
            os.remove("output.bnd")
            hatch_mcp.logger.info("Removed output.bnd file.")
        if os.path.exists("output.cfg"):
            os.remove("output.cfg")
            hatch_mcp.logger.info("Removed output.cfg file.")
        return "BND and CFG files cleaned up successfully."
    except Exception as e:
        hatch_mcp.logger.error(f"Error during cleanup: {str(e)}")
        return f"Error during cleanup: {str(e)}"
    
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
