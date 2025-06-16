import logging
import os
import glob
from pathlib import Path
from hatchling import HatchMCP
import matplotlib.pyplot as plt
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
def build_simulation(bnd_path: str = "output.bnd", cfg_path: str = "output.cfg") -> str:
    """
    When the user requests to create a simulation,
    or if it ask to build a simulation,
    or if it ask to prepare a simulation,
    this function builds a MaBoSS simulation using the provided BND and CFG files.
    It loads the simulation and returns a series of MaBoSS parameters that summarize the status of the model.
    This function is designed to be called after the BND and CFG files have been created.
    It initializes the simulation with the specified BND and CFG files,
    It also logs the process and handles any errors that may occur while building the maboss simulation.
    Args:
        bnd_path (str): Path to the .bnd file.
        cfg_path (str): Path to the .cfg file.
    Returns:
        str: Parameters of the MaBoSS simulation or an error message if the simulation creation fails.
    """
    global sim
    hatch_mcp.logger.info(f"Creating MaBoSS simulation with BND: {bnd_path} and CFG: {cfg_path}")
    sim = maboss.load(bnd_path, cfg_path)
    if sim:
        hatch_mcp.logger.info("MaBoSS simulation completed successfully.")
        parameters = sim.param # Ordered Dict with the parameters of the simulation
        parameters_str = "\n".join([f"{key}: {value}" for key, value in parameters.items()])
        hatch_mcp.logger.info(f"Simulation parameters: {parameters_str}")
        return f"MaBoSS simulation created successfully with the following parameters:\n{parameters_str}"
    else:
        hatch_mcp.logger.error("Failed to create MaBoSS simulation.")
        return "Error in MaBoSS simulation creation. Please check the BND and CFG files." 

@hatch_mcp.server.tool()
def run_simulation() -> str:
    """
    When the user requests to run a simulation,
    or if it asks to execute a simulation,
    this function runs the MaBoSS simulation that has been built.
    It logs the process and handles any errors that may occur during the simulation run.
    Returns:
        str: Result of the MaBoSS simulation run or an error message if the run fails.
    """
    global sim, result
    if sim is None:
        return "No MaBoSS simulation has been built yet. Please build a simulation first."

    try:
        hatch_mcp.logger.info("Running MaBoSS simulation...")
        result = sim.run()
        hatch_mcp.logger.info("MaBoSS simulation completed successfully.")
        return "MaBoSS simulation run completed successfully."
    except Exception as e:
        hatch_mcp.logger.error(f"Error during MaBoSS simulation run: {str(e)}")
        return f"Error during MaBoSS simulation run: {str(e)}"

@hatch_mcp.server.tool()
def get_maboss_initial_state() -> str:
    """
    When the user requests the initial state of the MaBoSS simulation,
    this function retrieves the initial state of the simulation.
    It logs the process and returns the initial state as a string.
    Returns:
        str: Initial state of the MaBoSS simulation.
    """
    global sim
    if sim is None:
        return "No MaBoSS simulation has been built yet. Please build a simulation first."

    try:
        initial_state = sim.get_initial_state()
        hatch_mcp.logger.info(f"Initial state retrieved: {initial_state}")
        return f"Initial state of the MaBoSS simulation: {initial_state}"
    except Exception as e:
        hatch_mcp.logger.error(f"Error retrieving initial state: {str(e)}")
        return f"Error retrieving initial state: {str(e)}"
    

@hatch_mcp.server.tool()
def get_maboss_logical_rules() -> str:
    """
    When the user requests the logical rules of the MaBoSS simulation,
    this function retrieves the logical rules of the simulation.
    It logs the process and returns the logical rules as a string.
    Returns:
        str: Logical rules of the MaBoSS simulation.
    """
    global sim
    if sim is None:
        return "No MaBoSS simulation has been built yet. Please build a simulation first."

    try:
        logical_rules = sim.get_logical_rules()
        hatch_mcp.logger.info(f"Logical rules retrieved: {logical_rules}")
        return f"Logical rules of the MaBoSS simulation:\n{logical_rules}"
    except Exception as e:
        hatch_mcp.logger.error(f"Error retrieving logical rules: {str(e)}")
        return f"Error retrieving logical rules: {str(e)}"

@hatch_mcp.server.tool()
def get_maboss_mutations() -> str:
    """
    When the user requests the mutations of the MaBoSS simulation,
    this function retrieves the mutations of the simulation.
    It logs the process and returns the mutations as a string.
    Returns:
        str: Mutations of the MaBoSS simulation.
    """
    global sim
    if sim is None:
        return "No MaBoSS simulation has been built yet. Please build a simulation first."

    try:
        mutations = sim.get_mutations()
        hatch_mcp.logger.info(f"Mutations retrieved: {mutations}")
        return f"Mutations of the MaBoSS simulation:\n{mutations}"
    except Exception as e:
        hatch_mcp.logger.error(f"Error retrieving mutations: {str(e)}")
        return f"Error retrieving mutations: {str(e)}"

@hatch_mcp.server.tool()
def update_maboss_parameters(parameters: dict) -> str:
    """
    When the user requests to update the parameters of the MaBoSS simulation,
    such as the time tick (time_tick) or the max time (max_time), or the sample count (sample_count),
    or the discrete time (discrete_time), etc...
    this function updates the parameters of the simulation with the provided values.
    It logs the process and returns a confirmation message.
    Args:
        parameters (dict): Dictionary containing parameter names and their new values.
    Returns:
        str: Confirmation message indicating successful parameter update.
    """
    global sim
    if sim is None:
        return "No MaBoSS simulation has been built yet. Please build a simulation first."

    try:
        for key, value in parameters.items():
            sim.param[key] = value
        hatch_mcp.logger.info(f"MaBoSS parameters updated: {parameters}")
        return "MaBoSS parameters updated successfully."
    except Exception as e:
        hatch_mcp.logger.error(f"Error updating MaBoSS parameters: {str(e)}")
        return f"Error updating MaBoSS parameters: {str(e)}"
    
@hatch_mcp.server.tool()
def set_maboss_output_nodes(output_nodes: list) -> str:
    """
    When the user requests to set the output nodes of the MaBoSS simulation,
    this function sets the output nodes for the simulation.
    It logs the process and returns a confirmation message.
    Args:
        output_nodes (list): List of node names to be set as output nodes.
    Returns:
        str: Confirmation message indicating successful setting of output nodes.
    """
    global sim
    if sim is None:
        return "No MaBoSS simulation has been built yet. Please build a simulation first."

    try:
        hatch_mcp.logger.info(f"Former MaBoSS output nodes: {sim.network.get_output()}")
        sim.network.set_output(output_nodes)
        hatch_mcp.logger.info(f"MaBoSS output nodes set: {sim.network.get_output()}")
        return f"MaBoSS output nodes set successfully: {sim.network.get_output()}"
    except Exception as e:
        hatch_mcp.logger.error(f"Error setting MaBoSS output nodes: {str(e)}")
        return f"Error setting MaBoSS output nodes: {str(e)}"
    
@hatch_mcp.server.tool()
def set_maboss_initial_state(nodes, probDict) -> str:
    """
    When the user requests to set the initial state of the MaBoSS simulation,
    or if it asks to set the initial state probabilities,
    this function sets the initial state probabilities for one or more nodes in the MaBoSS simulation.

    Parameters
    ----------
    nodes : str or list/tuple of str
        The node name (as a string) or a list/tuple of node names whose initial state probabilities are to be set.
    probDict : list, dict, or nested dict
        The probability distribution for the initial states:
        - If `nodes` is a single node (str), `probDict` can be a list [P(0), P(1)] or a dict {0: P(0), 1: P(1)}.
        - If `nodes` is a list/tuple of multiple nodes, `probDict` must be a dict mapping tuples of 0/1 to probabilities, e.g. {(0,0): 0.4, (1,0): 0.6}.

    Returns
    -------
    str
        Success or error message.

    Example
    -------
    >>> set_maboss_initial_state('node1', [0.3, 0.7])
    >>> set_maboss_initial_state(['node1', 'node2'], {(0, 0): 0.4, (1, 0): 0.6, (0, 1): 0})
    """
    global sim
    if sim is None:
        return "No MaBoSS simulation has been built yet. Please build a simulation first."

    try:
        # Normalize nodes to correct type for MaBoSS API
        if isinstance(nodes, str):
            node_arg = nodes
        elif isinstance(nodes, (list, tuple)):
            node_arg = list(nodes)
        else:
            return "Invalid type for 'nodes'. Must be str or list/tuple of str."

        # Validate probDict type
        if isinstance(node_arg, str):
            if not (isinstance(probDict, list) or isinstance(probDict, dict)):
                return "For a single node, probDict must be a list or dict."
        elif isinstance(node_arg, list):
            if not isinstance(probDict, dict):
                return "For multiple nodes, probDict must be a dict mapping tuples to probabilities."

        hatch_mcp.logger.info(f"Former MaBoSS initial state: {sim.network.get_istate()}")

        sim.network.set_istate(node_arg, probDict)
        hatch_mcp.logger.info(f"Current MaBoSS initial state: {sim.network.get_istate()}")
        return f"Initial state set for MaBoSS simulation: {sim.network.get_istate()}"
    except Exception as e:
        hatch_mcp.logger.error(f"Error setting MaBoSS initial state: {str(e)}")
        return f"Error setting MaBoSS initial state: {str(e)}"
    
@hatch_mcp.server.tool()
def simulate_mutation(nodes, state="OFF") -> str:
    """
    When the user requests to run a simulation with mutation analysis, this tool simulates the effect of one or more node mutations on the MaBoSS network.

    This function creates a copy of the current MaBoSS simulation, applies the specified mutation(s), runs the simulation, and returns the final state probability distribution as a Markdown table (see Output Format below).

    Parameters
    ----------
    nodes : str or list/tuple of str
        The node name (as a string) or a list/tuple of node names to mutate. Each node will be mutated to the specified state.
    state : str or list/tuple of str, optional
        The mutation state(s) to apply. Must be one of 'ON', 'OFF', or 'WT'.
        - If a single string, all nodes are mutated to this state (default: 'OFF').
        - If a list/tuple, must match the length of `nodes`.

    Returns
    -------
    str
        Markdown-formatted table of the final state probability trajectory after mutation, or an error message.

    Output Format
    -------------
    The Markdown string follows the same structure as `get_simulation_result`, with a title, subtitle, and a table of state probabilities at the final timepoint. See that function's docstring for details.

    Example
    -------
    >>> simulate_mutation('FoxO3', 'OFF')
    >>> simulate_mutation(['FoxO3', 'AKT'], ['OFF', 'ON'])

    Notes
    -----
    - The mutation is applied using the MaBoSS API: `sim.mutate(node, state)`.
    - Valid states are: 'ON' (always up), 'OFF' (always down), 'WT' (wild-type, normal behavior).
    - The simulation is run after mutation, and the result is formatted as a Markdown table.
    - If the simulation is not built, an error message is returned.
    """
    global sim
    if sim is None:
        return "No MaBoSS simulation has been built yet. Please build a simulation first."

    try:
        hatch_mcp.logger.info("Running MaBoSS simulation with mutation analysis...")
        mutated_simulation = sim.copy()

        # Normalize nodes and state(s)
        if isinstance(nodes, str):
            node_list = [nodes]
        else:
            node_list = list(nodes)

        if isinstance(state, str):
            state_list = [state] * len(node_list)
        else:
            state_list = list(state)
            if len(state_list) != len(node_list):
                return "Length of 'state' must match length of 'nodes'."

        # Validate mutation states
        valid_states = {"ON", "OFF", "WT"}
        for s in state_list:
            if s not in valid_states:
                return f"Invalid mutation state: {s}. Must be one of {valid_states}."

        # Apply mutations
        for node, s in zip(node_list, state_list):
            mutated_simulation.mutate(node, s)
            hatch_mcp.logger.info(f"Applied mutation: {node} -> {s}")

        result = mutated_simulation.run()
        df_prob = result.get_last_states_probtraj()

        if df_prob.empty:
            return "_Simulation completed but returned no trajectory data._"

        df_prob = clean_for_markdown(df_prob)
        md_table = df_prob.to_markdown(index=False, tablefmt="plain")

        md_lines = [
            "**MaBoSS Simulation: State Probability Trajectory (with Mutation)**",
            "",
            f"_Below is the probability trajectory of each state over time after mutation: {dict(zip(node_list, state_list))}_",
            "",
            md_table
        ]
        return "\n".join(md_lines)
    except Exception as e:
        hatch_mcp.logger.error(f"Error running MaBoSS simulation with mutation: {str(e)}")
        return f"Error running MaBoSS simulation with mutation: {str(e)}"
    
@hatch_mcp.server.tool()
def visualize_network_trajectories() -> str:
    """
    When the user requests to visualize network trajectories,
    this function retrieves the last simulation result and plots the network trajectories.
    It logs the request, saves the plot to a file, and returns the file path for user access.
    Returns:
        str: The file path to the saved plot, or an error message.
    """
    hatch_mcp.logger.info("Request to visualize network trajectories received.")
    global result
    if result is None:
        return "No simulation has been run yet. Please run a simulation first."

    try:
        fig = result.plot_trajectory()
        # If plot_trajectory returns a figure, use it; otherwise, use plt.gcf()
        if fig is None:
            fig = plt.gcf()
        output_path = "network_trajectory.png"
        fig.savefig(output_path)
        plt.close(fig)
        hatch_mcp.logger.info(f"Network trajectory plot saved to {output_path}")
        return f"Network trajectory plot saved: {output_path}\nYou can open it with your image viewer."
    except Exception as e:
        hatch_mcp.logger.error(f"Error saving network trajectory plot: {str(e)}")
        return f"Error saving network trajectory plot: {str(e)}"
    
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
