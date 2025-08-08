"""
PhysiCell MCP Server with Session Management

This server provides tools for configuring PhysiCell biological simulations:
- Create simulation domains and add substrates
- Define cell types and their behaviors
- Create signal-behavior rules for realistic cell responses
- Export configurations for PhysiCell execution

Features lightweight session management and progress tracking.
"""

import sys
import os
import glob
import time
from pathlib import Path
from typing import Optional
#from hatch_mcp_server import HatchMCP

# Add the physicell_config package to Python path  
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from physicell_config import PhysiCellConfig
from physicell_config.config.embedded_signals_behaviors import (
    get_signals_behaviors,
    get_signal_by_name,
    get_behavior_by_name,
    update_signals_behaviors_context_from_config,
    get_expanded_signals,
    get_expanded_behaviors
)
from physicell_config.config.embedded_defaults import get_default_parameters

# Try to import PhysiBoSS module - only need to check availability
try:
    import physicell_config.modules.physiboss
    PHYSIBOSS_AVAILABLE = True
except ImportError:
    PHYSIBOSS_AVAILABLE = False

# Import session management
from session_manager import (
    session_manager, SessionState, WorkflowStep, MaBoSSContext,
    get_current_session, ensure_session
)

from mcp.server.fastmcp import Context, FastMCP

mcp = FastMCP()

# Initialize MCP server
""" hatch_mcp = HatchMCP("PhysiCell-Config-Builder",
                     fast_mcp=mcp,
                     origin_citation="PhysiCell: An Open Source Physics-Based Cell Simulator",
                     mcp_citation="https://github.com/marcorusc/Hatch_Pkg_Dev/tree/main/PhysiCell")
 """
# Legacy global variables for backward compatibility
# These are now managed through the session manager
config = None
scenario_context = ""


def _set_legacy_config(new_config):
    """Set config in current session for backward compatibility."""
    session = ensure_session()
    session.config = new_config
    global config
    config = new_config

def _set_legacy_scenario_context(context):
    """Set scenario context in current session."""
    session = ensure_session()
    session.scenario_context = context
    global scenario_context
    scenario_context = context

# ============================================================================
# SESSION MANAGEMENT TOOLS
# ============================================================================

@mcp.tool()
def create_session(set_as_default: bool = True, session_name: Optional[str] = None) -> str:
    """
    Create a new simulation session for managing PhysiCell configurations.
    Sessions allow you to work on multiple simulations independently.
    
    Args:
        set_as_default: Whether to set this as the default session for subsequent operations
        session_name: Optional human-readable name for cross-server linking (e.g., "gastric_cancer_v1")
    
    Returns:
        str: Session ID and instructions
    """
    session_id = session_manager.create_session(set_as_default, session_name)
    
    result = f"**Session created:** {session_id[:8]}..."
    if session_name:
        result += f" ({session_name})"
    result += "\n"
    result += f"**Next steps:**\n"
    result += f"1. `analyze_biological_scenario()` - Set your biological context\n"
    result += f"2. `create_simulation_domain()` - Define spatial framework\n"
    result += f"3. Use `get_workflow_status()` to track progress"
    
    return result

@mcp.tool()
def list_sessions() -> str:
    """
    List all active simulation sessions with their status and progress.
    
    Returns:
        str: Formatted list of sessions with progress information
    """
    sessions = session_manager.list_sessions()
    
    if not sessions:
        return "No active sessions. Use `create_session()` to start."
    
    result = f"## Active Sessions ({len(sessions)})\n\n"
    
    default_id = session_manager.get_default_session_id()
    
    for session in sessions:
        age_hours = (time.time() - session.created_at) / 3600
        progress = session.get_progress_percentage()
        
        # Mark default session
        default_marker = " (default)" if session.session_id == default_id else ""
        
        result += f"**{session.session_id[:8]}...{default_marker}**\n"
        result += f"- Age: {age_hours:.1f} hours\n"
        result += f"- Progress: {progress:.0f}%\n"
        result += f"- Components: {session.substrates_count} substrates, {session.cell_types_count} cell types, {session.rules_count} rules\n"
        
        if session.scenario_context:
            result += f"- Scenario: {session.scenario_context[:50]}{'...' if len(session.scenario_context) > 50 else ''}\n"
        
        result += "\n"
    
    result += "Use `switch_session(session_id)` to switch between sessions."
    
    return result

@mcp.tool()
def switch_session(session_id: str) -> str:
    """
    Switch to a different session as the default for subsequent operations.
    
    Args:
        session_id: The ID of the session to switch to (can be shortened to first 8 characters)
    
    Returns:
        str: Confirmation of session switch
    """
    # Allow partial session IDs
    if len(session_id) == 8:
        sessions = session_manager.list_sessions()
        matching_sessions = [s for s in sessions if s.session_id.startswith(session_id)]
        if len(matching_sessions) == 1:
            session_id = matching_sessions[0].session_id
        elif len(matching_sessions) > 1:
            return "Error: Ambiguous session ID. Multiple sessions match."
        else:
            return "Error: Session not found."
    
    success = session_manager.set_default_session(session_id)
    if success:
        session = session_manager.get_session(session_id)
        progress = session.get_progress_percentage()
        return f"**Switched to session:** {session_id[:8]}... (Progress: {progress:.0f}%)"
    else:
        return "Error: Session not found."

@mcp.tool()
def get_workflow_status() -> str:
    """
    Get the current workflow status and recommended next steps.
    
    Returns:
        str: Current progress and next recommended actions
    """
    session = get_current_session()
    if not session:
        return "No active session. Use `create_session()` to start."
    
    progress = session.get_progress_percentage()
    recommendations = session.get_next_recommended_steps()
    
    result = f"## Workflow Status\n\n"
    result += f"**Session:** {session.session_id[:8]}...\n"
    result += f"**Progress:** {progress:.0f}%\n\n"
    
    # Show completed steps
    completed_steps = [step.value.replace('_', ' ').title() for step in session.completed_steps]
    if completed_steps:
        result += f"**Completed Steps:**\n"
        for step in completed_steps:
            result += f"{step}\n"
        result += "\n"
    
    # Show next steps
    result += f"**Next Recommended Steps:**\n"
    for i, rec in enumerate(recommendations[:3], 1):
        result += f"{i}. {rec}\n"
    
    if session.scenario_context:
        result += f"\n**Current Scenario:** {session.scenario_context}"
    
    return result

@mcp.tool()
def delete_session(session_id: str) -> str:
    """
    Delete a simulation session permanently.
    
    Args:
        session_id: The ID of the session to delete
    
    Returns:
        str: Confirmation of deletion
    """
    success = session_manager.delete_session(session_id)
    if success:
        return f"**Session deleted:** {session_id[:8]}..."
    else:
        return "Error: Session not found"

@mcp.tool()
def set_maboss_context(model_name: str, bnd_file_path: str, cfg_file_path: str,
                      target_cell_type: str, available_nodes: str = "",
                      output_nodes: str = "", simulation_results: str = "",
                      biological_context: str = "") -> str:
    """
    Store MaBoSS model context for integration into PhysiCell simulation.
    This is typically called by the agent after analyzing a MaBoSS model.
    
    Args:
        model_name: Name of the MaBoSS model
        bnd_file_path: Path to the .bnd boolean network file
        cfg_file_path: Path to the .cfg configuration file  
        target_cell_type: Which cell type this model should be integrated into
        available_nodes: Comma-separated list of available boolean nodes
        output_nodes: Comma-separated list of output nodes
        simulation_results: Summary of MaBoSS simulation behavior
        biological_context: Original biological question/context
    
    Returns:
        str: Confirmation of context storage
    """
    session = get_current_session()
    if not session:
        return "Error: No active session. Use `create_session()` first."
    
    maboss_context = MaBoSSContext(
        model_name=model_name,
        bnd_file_path=bnd_file_path,
        cfg_file_path=cfg_file_path,
        available_nodes=[node.strip() for node in available_nodes.split(",") if node.strip()],
        output_nodes=[node.strip() for node in output_nodes.split(",") if node.strip()],
        simulation_results=simulation_results,
        target_cell_type=target_cell_type,
        biological_context=biological_context
    )
    
    session.maboss_context = maboss_context
    
    result = f"**MaBoSS context stored:**\n"
    result += f"- Model: {model_name}\n"
    result += f"- Target cell type: {target_cell_type}\n"
    result += f"- Available nodes: {len(maboss_context.available_nodes)}\n"
    result += f"- Output nodes: {len(maboss_context.output_nodes)}\n"
    if simulation_results:
        result += f"- Simulation results available\n"
    result += f"**Next step:** Continue with PhysiCell simulation setup."
    
    return result

@mcp.tool()
def get_maboss_context() -> str:
    """
    Get the stored MaBoSS context for the current session.
    Shows available boolean nodes and simulation results.
    
    Returns:
        str: MaBoSS context information
    """
    session = get_current_session()
    if not session:
        return "Error: No active session."
    
    if not session.maboss_context:
        return "No MaBoSS context available in current session."
    
    ctx = session.maboss_context
    result = f"## MaBoSS Context\n\n"
    result += f"**Model:** {ctx.model_name}\n"
    result += f"**Target Cell Type:** {ctx.target_cell_type}\n"
    result += f"**Files:**\n"
    result += f"- BND: {ctx.bnd_file_path}\n"
    result += f"- CFG: {ctx.cfg_file_path}\n\n"
    
    if ctx.available_nodes:
        result += f"**Available Nodes ({len(ctx.available_nodes)}):**\n"
        for node in ctx.available_nodes:
            result += f"- {node}\n"
        result += "\n"
    
    if ctx.output_nodes:
        result += f"**Output Nodes ({len(ctx.output_nodes)}):**\n"
        for node in ctx.output_nodes:
            result += f"- {node}\n"
        result += "\n"
    
    if ctx.simulation_results:
        result += f"**Simulation Results:**\n{ctx.simulation_results}\n\n"
    
    if ctx.biological_context:
        result += f"**Biological Context:**\n{ctx.biological_context}"
    
    return result

import time

# ============================================================================
# BIOLOGICAL SCENARIO ANALYSIS
# ============================================================================

@mcp.resource(
    uri="docs://tools/analyze_biological_scenario",
    name="Documentation for analyze_biological_scenario",
    description="Stores a description of the biological scenario for simulation context.",
    mime_type="text/markdown"
)
def docs_analyze_biological_scenario() -> str:
    return """
# Tool: analyze_biological_scenario

Stores the biological scenario/context for this simulation session.  
Call this first to set the context for parameter choices in later steps.

## Parameters
- `biological_scenario` (`str`, required): Description of the scenario, e.g. `"Breast cancer cells in hypoxic 3D tissue with immune infiltration."`

## Example

```python
analyze_biological_scenario(biological_scenario="Breast cancer cells in a 3D hypoxic tissue, infiltrated by CD8 T cells.")

Notes

This does not perform automatic analysis, but stores your description for context-aware setup.

Only one scenario is stored at a time; calling again will overwrite the previous context.
            """

@mcp.tool()
def analyze_biological_scenario(biological_scenario: str) -> str:
    """
    Store a biological scenario description to provide context for subsequent simulation setup.
    This context helps inform parameter choices for substrates, cell types, and rules.
    
    Args:
        biological_scenario: Description of the biological scenario or experimental setup
    
    Returns:
        str: Confirmation message with stored scenario context
    """
    if not biological_scenario or not biological_scenario.strip():
        return "Error: Biological scenario description cannot be empty"
    
    session = ensure_session()
    session.scenario_context = biological_scenario.strip()
    session.mark_step_complete(WorkflowStep.SCENARIO_ANALYSIS)
    
    # Update legacy global for backward compatibility
    _set_legacy_scenario_context(biological_scenario.strip())
    
    result = f"**Biological scenario stored:** {biological_scenario}\n"
    result += f"**Next step:** Use `create_simulation_domain()` to set up the spatial framework."
    
    return result

# ============================================================================
# SIMULATION SETUP
# ============================================================================

@mcp.resource(
    uri="docs://tools/create_simulation_domain",
    name="Documentation for create_simulation_domain",
    description="Creates a 3D simulation domain with specified size and time duration.",
    mime_type="text/markdown"
)
def docs_create_simulation_domain() -> str:
    return """
# Tool: create_simulation_domain

Sets up the spatial and temporal domain for the simulation.

## Parameters
- `domain_x` (`float`, required): Domain width in μm (e.g., 3000)
- `domain_y` (`float`, required): Domain height in μm (e.g., 3000)
- `domain_z` (`float`, required): Domain depth in μm (e.g., 500)
- `dx` (`float`, optional): Mesh spacing in μm (default: 20)
- `max_time` (`float`, optional): Maximum simulation time in minutes (default: 7200 = 5 days)

## Example

```python
create_simulation_domain(domain_x=3000, domain_y=3000, domain_z=500, dx=20, max_time=7200)

Notes

This should be the first configuration step for every new simulation.

After creating the domain, you can add substrates and cell types.
"""

@mcp.tool()
def create_simulation_domain(domain_x: float, domain_y: float, 
                           domain_z: float, dx: float = 20.0, 
                           max_time: float = 7200.0) -> str:
    """
    Create the spatial and temporal framework for a PhysiCell simulation.
    This sets up the 3D domain size, mesh resolution, and simulation duration.
    
    Args:
        domain_x: Domain width in micrometers
        domain_y: Domain height in micrometers  
        domain_z: Domain depth in micrometers
        dx: Mesh spacing in micrometers (default: 20)
        max_time: Maximum simulation time in minutes (default: 7200 = 5 days)
    
    Returns:
        str: Success message with domain specifications
    """
    # Basic validation
    if domain_x <= 0 or domain_y <= 0 or domain_z <= 0:
        return "Error: Domain dimensions must be positive"
    if dx <= 0:
        return "Error: Mesh spacing must be positive"
    if max_time <= 0:
        return "Error: Simulation time must be positive"
    
    session = ensure_session()
    
    # Create new PhysiCell configuration
    session.config = PhysiCellConfig()
    session.config.domain.set_bounds(
        -domain_x/2, domain_x/2,
        -domain_y/2, domain_y/2, 
        -domain_z/2, domain_z/2
    )
    session.config.domain.set_mesh(dx, dx, dx)
    session.config.options.set_max_time(max_time)
    session.config.options.set_time_steps(dt_diffusion=0.01, dt_mechanics=0.1, dt_phenotype=6.0)
    
    # Mark workflow step as complete
    session.mark_step_complete(WorkflowStep.DOMAIN_SETUP)
    
    # Update legacy global for backward compatibility
    _set_legacy_config(session.config)
    
    # Format result
    result = f"**Simulation domain created:**\n"
    result += f"- Domain: {domain_x}×{domain_y}×{domain_z} μm\n"
    result += f"- Mesh: {dx} μm\n"
    result += f"- Duration: {max_time/60:.1f} hours\n"
    result += f"- Progress: {session.get_progress_percentage():.0f}%\n"
    result += f"**Next step:** Use `add_single_substrate()` to add oxygen, nutrients, or drugs."
    
    return result

@mcp.resource(
    uri="docs://tools/add_single_substrate",
    name="Documentation for add_single_substrate",
    description="Adds a chemical substrate (e.g., oxygen) with its physical/chemical properties.",
    mime_type="text/markdown"
)
def docs_add_single_substrate() -> str:
    return """
# Tool: add_single_substrate

Adds a chemical substrate (oxygen, glucose, drug, etc.) to the simulation environment.

## Parameters
- `substrate_name` (`str`, required): Name (e.g., `"oxygen"`)
- `diffusion_coefficient` (`float`, required): Diffusion rate in μm²/min (e.g., 100000)
- `decay_rate` (`float`, required): Decay rate in 1/min (e.g., 0.01)
- `initial_condition` (`float`, required): Initial concentration (e.g., 38 for oxygen)
- `units` (`str`, optional): Concentration units (default: "dimensionless")
- `dirichlet_enabled` (`bool`, optional): Use boundary conditions (default: False)
- `dirichlet_value` (`float`, optional): Value at boundary (default: initial_condition)

## Example

```python
add_single_substrate(
    substrate_name="oxygen",
    diffusion_coefficient=100000,
    decay_rate=0.01,
    initial_condition=38,
    units="dimensionless",
    dirichlet_enabled=False
)

Notes

Call this once for each substrate you want to add.

For multiple substrates, repeat with different names and values.
    """

@mcp.tool()
def add_single_substrate(substrate_name: str, diffusion_coefficient: float, decay_rate: float,
                        initial_condition: float, units: str = "dimensionless",
                        dirichlet_enabled: bool = False, dirichlet_value: Optional[float] = None) -> str:
    """
    Add a chemical substrate (oxygen, glucose, drug, etc.) to the simulation environment.
    Substrates diffuse through the domain and can be consumed or secreted by cells.
    
    Args:
        substrate_name: Name of the substrate (e.g., 'oxygen', 'glucose', 'drug')
        diffusion_coefficient: Diffusion rate in μm²/min (typical: 100000 for oxygen)
        decay_rate: Decay rate in 1/min (typical: 0.01)
        initial_condition: Initial concentration (typical: 38 for oxygen)
        units: Concentration units (default: 'dimensionless')
        dirichlet_enabled: Whether to use boundary conditions (default: False)
        dirichlet_value: Boundary concentration (default: same as initial_condition)
    
    Returns:
        str: Success message with substrate parameters
    """
    session = get_current_session()
    if not session or not session.config:
        return "Error: Create simulation domain first using create_simulation_domain()"
    
    # Basic validation
    if not substrate_name or not substrate_name.strip():
        return "Error: Substrate name cannot be empty"
    if diffusion_coefficient < 0:
        return "Error: Diffusion coefficient must be non-negative"
    if decay_rate < 0:
        return "Error: Decay rate must be non-negative"
    
    if dirichlet_value is None:
        dirichlet_value = initial_condition
    
    # Add substrate to configuration
    session.config.substrates.add_substrate(
        substrate_name.strip(),
        diffusion_coefficient=diffusion_coefficient,
        decay_rate=decay_rate,
        initial_condition=initial_condition,
        dirichlet_enabled=dirichlet_enabled,
        dirichlet_value=dirichlet_value,
        units=units
    )
    
    # Update session counters
    session.substrates_count += 1
    session.mark_step_complete(WorkflowStep.SUBSTRATES_ADDED)
    
    # Update legacy global for backward compatibility
    _set_legacy_config(session.config)
    
    # Format result
    result = f"**Substrate added:** {substrate_name}\n"
    result += f"- Diffusion: {diffusion_coefficient:g} μm²/min\n"
    result += f"- Decay: {decay_rate:g} min⁻¹\n"
    result += f"- Initial: {initial_condition:g} {units}\n"
    if dirichlet_enabled:
        result += f"- Boundary: {dirichlet_value:g} {units}\n"
    result += f"- Progress: {session.get_progress_percentage():.0f}%\n"
    result += f"**Next step:** Use `add_single_cell_type()` to add cancer cells, immune cells, etc."
    
    return result

@mcp.resource(
    uri="docs://tools/add_single_cell_type",
    name="Documentation for add_single_cell_type",
    description="Adds a new cell type with a selected cell cycle model.",
    mime_type="text/markdown"
)
def docs_add_single_cell_type() -> str:
    return """
# Tool: add_single_cell_type

Adds a cell type (cancer, immune, fibroblast, etc.) with a basic cell cycle model.

## Parameters
- `cell_type_name` (`str`, required): e.g. "cancer_cell", "immune_cell"
- `cycle_model` (`str`, optional): Cell cycle model, e.g. "Ki67_basic" (default), "live_cell" (see `get_available_cycle_models` for options)

## Example

```python
add_single_cell_type(cell_type_name="cancer_cell", cycle_model="Ki67_basic")

Notes

Call this once for each cell type.

For custom cycle models, consult get_available_cycle_models.
                """

@mcp.tool()
def add_single_cell_type(cell_type_name: str, cycle_model: str = "Ki67_basic") -> str:
    """
    Add a cell type (cancer, immune, fibroblast, etc.) to the simulation.
    Cell types define the agents that will populate the simulation domain.
    
    Args:
        cell_type_name: Name of the cell type (e.g., 'cancer_cell', 'immune_cell', 'fibroblast')
        cycle_model: Cell cycle model (default: 'Ki67_basic')
    
    Returns:
        str: Success message with cell type details
    """
    session = get_current_session()
    if not session or not session.config:
        return "Error: Create simulation domain first using create_simulation_domain()"
    
    # Basic validation
    if not cell_type_name or not cell_type_name.strip():
        return "Error: Cell type name cannot be empty"
    
    cell_type_name = cell_type_name.strip()
    
    # Add cell type to configuration
    session.config.cell_types.add_cell_type(cell_type_name, template='default')
    session.config.cell_types.set_cycle_model(cell_type_name, cycle_model)
    
    # Update session counters
    session.cell_types_count += 1
    session.mark_step_complete(WorkflowStep.CELL_TYPES_ADDED)
    
    # Update legacy global for backward compatibility
    _set_legacy_config(session.config)
    
    # Format result
    result = f"**Cell type added:** {cell_type_name}\n"
    result += f"- Cycle model: {cycle_model}\n"
    result += f"- Progress: {session.get_progress_percentage():.0f}%\n"
    result += f"**Next step:** Use `add_single_cell_rule()` to create cell behavior rules.\n"
    result += f"First, use `list_all_available_signals()` and `list_all_available_behaviors()` to see options."
    
    return result



@mcp.tool()
def configure_cell_parameters(cell_type: str, volume_total: float = 2500.0, 
                             volume_nuclear: float = 500.0, fluid_fraction: float = 0.75,
                             motility_speed: float = 0.5, persistence_time: float = 5.0,
                             apoptosis_rate: float = 0.0001, necrosis_rate: float = 0.0001) -> str:
    """
When the user asks to configure cell properties, set cell size, or adjust cell behavior,
this function modifies the basic parameters for a cell type.
In particular, it sets the total and nuclear volume, fluid fraction,
motility speed, persistence time, and death rates.
The function modifies an existing cell type; it does not create a new one.
This allows for detailed configuration of cell properties based on the biological scenario.
To modify multiple cell types, use this function repeatedly.

Input parameters:
cell_type (str): Name of existing cell type to configure
volume_total (float): Total cell volume (μm³, default: 2500.0)
volume_nuclear (float): Nuclear volume (μm³, default: 500.0)
fluid_fraction (float): Cytoplasmic fluid fraction (0-1, default: 0.75)
motility_speed (float): Cell movement speed (μm/min, default: 0.5)
persistence_time (float): Directional persistence (min, default: 5.0)
apoptosis_rate (float): Apoptosis rate (1/min, default: 0.0001)
necrosis_rate (float): Necrosis rate (1/min, default: 0.0001)

Returns:
str: Success message with configured parameters
    """
    global config
    if config is None:
        return "Error: Create simulation domain first using create_simulation_domain()"
    
    try:
        # Set volume parameters
        config.cell_types.set_volume_parameters(cell_type, total=volume_total, 
                                              nuclear=volume_nuclear, fluid_fraction=fluid_fraction)
        
        # Set motility parameters
        config.cell_types.set_motility(cell_type, speed=motility_speed, 
                                     persistence_time=persistence_time, enabled=True)
        
        # Set death rates
        config.cell_types.set_death_rate(cell_type, 'apoptosis', apoptosis_rate)
        config.cell_types.set_death_rate(cell_type, 'necrosis', necrosis_rate)
        
        result = f"**Configured parameters for {cell_type}:**\n"
        result += f"- **Volume:** {volume_total:g} μm³ (nuclear: {volume_nuclear:g} μm³)\n"
        result += f"- **Motility:** {motility_speed:g} μm/min (persistence: {persistence_time:g} min)\n"
        result += f"- **Death rates:** apoptosis {apoptosis_rate:g}, necrosis {necrosis_rate:g} min⁻¹"
        
        return result
    except Exception as e:
        return f"Error configuring cell type '{cell_type}': {str(e)}"

@mcp.tool()
def set_substrate_interaction(cell_type: str, substrate: str, 
                             secretion_rate: float = 0.0, uptake_rate: float = 0.0) -> str:
    """
When the user asks to set oxygen consumption, drug uptake, or substrate secretion,
this function defines how a cell type interacts with a substrate.
In particular, it sets the secretion and uptake rates for the substrate.
The function modifies an existing cell type's interaction with a substrate;
it does not create a new interaction.
This allows for detailed configuration of how cells respond to their environment.
To modify multiple interactions, use this function repeatedly.

Input parameters:
cell_type (str): Name of existing cell type
substrate (str): Name of existing substrate
secretion_rate (float): Substrate secretion rate (1/min, default: 0.0)
uptake_rate (float): Substrate uptake rate (1/min, default: 0.0)

Returns:
str: Success message with interaction details
    """
    global config
    if config is None:
        return "Error: Create simulation domain first using create_simulation_domain()"
    
    try:
        config.cell_types.add_secretion(cell_type, substrate, 
                                      secretion_rate=secretion_rate,
                                      uptake_rate=uptake_rate)
        return f"**Substrate interaction set:** {cell_type} ↔ {substrate} (secretion: {secretion_rate:g}, uptake: {uptake_rate:g} min⁻¹)"
    except Exception as e:
        return f"Error setting substrate interaction: {str(e)}"

# ============================================================================
# PARAMETER DISCOVERY AND DEFAULTS
# ============================================================================

@mcp.tool()
def get_available_cycle_models() -> str:
    """
When the user asks about cell cycle models, what cycles are available, or needs to choose a cycle type,
this function returns the available cell cycle models in PhysiCell.

Returns:
str: Markdown-formatted list of available cell cycle models with descriptions
    """

    defaults = get_default_parameters()
    cycle_models = defaults.get("cell_cycle_models", {})
    
    result = "## Available Cell Cycle Models\n\n"
    for model_key, model_data in cycle_models.items():
        model_name = model_data.get("name", model_key)
        result += f"- **{model_key}**: {model_name}\n"
    
    result += "\n**Usage:** Use exact model names in add_single_cell_type() function.\n"
    result += "**Most common:** Ki67_basic, Ki67_advanced, live"
    
    return result

# ============================================================================
# SIGNAL AND BEHAVIOR DISCOVERY
# ============================================================================

@mcp.tool()
def list_all_available_signals() -> str:
    """
    Get the complete list of PhysiCell signals that can be used in cell rules.
    This function automatically updates the context from the current configuration
    and returns expanded signals including substrate-specific and cell-type-specific signals.
    
    Returns:
        str: Markdown-formatted list of all available signals with descriptions
    """
    session = get_current_session()
    
    # Update context from current config if available
    if session and session.config:
        update_signals_behaviors_context_from_config(session.config)
        # Use expanded signals which include context-specific signals
        try:
            signals_data = {signal['name']: signal for signal in get_expanded_signals()}
        except:
            # Fall back to basic signals if expanded version fails
            signals_data = get_signals_behaviors()["signals"]
    else:
        signals_data = get_signals_behaviors()["signals"]
    
    # Get current scenario context if available
    scenario_context = session.scenario_context if session else ""
    
    result = f"## PhysiCell Signals ({len(signals_data)} total)\n"
    if scenario_context:
        result += f"**Current scenario:** {scenario_context}\n\n"
    
    # Group signals by type for better organization
    signals_by_type = {}
    for signal_name, signal_info in signals_data.items():
        signal_type = signal_info.get("type", "other")
        if signal_type not in signals_by_type:
            signals_by_type[signal_type] = []
        signals_by_type[signal_type].append(signal_info)
    
    # Display signals grouped by type
    for signal_type, signals in signals_by_type.items():
        result += f"### {signal_type.upper()}\n"
        for signal in signals:
            signal_name = signal.get('name', 'Unknown')
            signal_desc = signal.get('description', 'No description')
            result += f"- **{signal_name}**: {signal_desc}\n"
            requires = signal.get('requires', [])
            if requires:
                result += f"  - *Requires: {', '.join(requires)}*\n"
        result += "\n"
    
    result += "**Note:** Use exact signal names in add_single_cell_rule() function.\n"
    result += "**Context:** Signals are automatically expanded based on current substrates and cell types."
    
    return result

@mcp.tool()
def list_all_available_behaviors() -> str:
    """
    Get the complete list of PhysiCell behaviors that can be controlled by rules.
    This function automatically updates the context from the current configuration
    and returns expanded behaviors including substrate-specific and cell-type-specific behaviors.
    
    Returns:
        str: Markdown-formatted list of all available behaviors with descriptions
    """
    session = get_current_session()
    
    # Update context from current config if available
    if session and session.config:
        update_signals_behaviors_context_from_config(session.config)
        # Use expanded behaviors which include context-specific behaviors
        try:
            behaviors_data = {behavior['name']: behavior for behavior in get_expanded_behaviors()}
        except:
            # Fall back to basic behaviors if expanded version fails
            behaviors_data = get_signals_behaviors()["behaviors"]
    else:
        behaviors_data = get_signals_behaviors()["behaviors"]
    
    # Get current scenario context if available
    scenario_context = session.scenario_context if session else ""
    
    result = f"## PhysiCell Behaviors ({len(behaviors_data)} total)\n"
    if scenario_context:
        result += f"**Current scenario:** {scenario_context}\n\n"
    
    # Group behaviors by type for better organization
    behaviors_by_type = {}
    for behavior_name, behavior_info in behaviors_data.items():
        behavior_type = behavior_info.get("type", "other")
        if behavior_type not in behaviors_by_type:
            behaviors_by_type[behavior_type] = []
        behaviors_by_type[behavior_type].append(behavior_info)
    
    # Display behaviors grouped by type
    for behavior_type, behaviors in behaviors_by_type.items():
        result += f"### {behavior_type.upper()}\n"
        for behavior in behaviors:
            behavior_name = behavior.get('name', 'Unknown')
            behavior_desc = behavior.get('description', 'No description')
            result += f"- **{behavior_name}**: {behavior_desc}\n"
            requires = behavior.get('requires', [])
            if requires:
                result += f"  - *Requires: {', '.join(requires)}*\n"
        result += "\n"
    
    result += "**Note:** Use exact behavior names in add_single_cell_rule() function.\n"
    result += "**Context:** Behaviors are automatically expanded based on current substrates and cell types."
    
    return result


# ============================================================================
# CELL RULES AND PHYSIBOSS
# ============================================================================

@mcp.tool()
def add_single_cell_rule(cell_type: str, signal: str, direction: str, behavior: str,
                        min_signal: float = 0, max_signal: float = 1, 
                        hill_power: float = 4.0, half_max: float = 0.5) -> str:
    """
    Add a signal-behavior rule that makes cells respond realistically to their environment.
    Rules define how cells change their behavior in response to environmental signals.
    
    Args:
        cell_type: Name of existing cell type
        signal: Signal name (use list_all_available_signals() to see options)
        direction: Signal direction ('increases' or 'decreases')
        behavior: Behavior name (use list_all_available_behaviors() to see options)
        min_signal: Minimum signal value (default: 0)
        max_signal: Maximum signal value (default: 1)
        hill_power: Hill coefficient (default: 4.0, typical: 1.0-8.0)
        half_max: Half-maximum signal level (default: 0.5)
    
    Returns:
        str: Success message with rule details
    """
    session = get_current_session()
    if not session or not session.config:
        return "Error: Create simulation domain first using create_simulation_domain()"
    
    # Basic validation
    if not cell_type or not cell_type.strip():
        return "Error: Cell type name cannot be empty"
    if not signal or not signal.strip():
        return "Error: Signal name cannot be empty"
    if direction not in ['increases', 'decreases']:
        return "Error: Direction must be 'increases' or 'decreases'"
    if not behavior or not behavior.strip():
        return "Error: Behavior name cannot be empty"
    if half_max <= 0:
        return "Error: Half-max value must be positive"
    if hill_power <= 0:
        return "Error: Hill power must be positive"
    
    # Update context from current config before adding rule
    update_signals_behaviors_context_from_config(session.config)
    
    # Add rule to configuration - check if we should use the new API or legacy
    try:
        # Try new API first (from test)
        from physicell_config.modules.cell_rules import CellRulesModule
        cell_rules = CellRulesModule(session.config)
        
        rule = {
            "cell_type": cell_type.strip(),
            "signal": signal.strip(),
            "direction": direction,
            "behavior": behavior.strip(),
            "min_signal": min_signal,
            "max_signal": max_signal,
            "hill_power": hill_power,
            "half_max": half_max
        }
        cell_rules.rules.append(rule)
        
    except (ImportError, AttributeError):
        # Fall back to legacy CSV API
        rules = session.config.cell_rules_csv
        rules.add_rule(
            cell_type=cell_type.strip(),
            signal=signal.strip(),
            direction=direction,
            behavior=behavior.strip(),
            base_value=min_signal,  # Map min_signal to base_value
            half_max=half_max,
            hill_power=hill_power,
            apply_to_dead=0
        )
    
    # Update session counters
    session.rules_count += 1
    session.mark_step_complete(WorkflowStep.RULES_CONFIGURED)
    
    # Update legacy global for backward compatibility
    _set_legacy_config(session.config)
    
    # Format result
    result = f"**Cell rule added:**\n"
    result += f"- Rule: {cell_type} | {signal} {direction} → {behavior}\n"
    result += f"- Signal range: {min_signal} to {max_signal}\n"
    result += f"- Half-max: {half_max}\n"
    result += f"- Hill power: {hill_power}\n"
    result += f"- Progress: {session.get_progress_percentage():.0f}%\n"
    
    # Check if ready for export based on core components (not arbitrary percentage)
    has_domain = WorkflowStep.DOMAIN_SETUP in session.completed_steps
    has_substrates = WorkflowStep.SUBSTRATES_ADDED in session.completed_steps
    has_cell_types = WorkflowStep.CELL_TYPES_ADDED in session.completed_steps
    ready_for_export = has_domain and has_substrates and has_cell_types
    
    if ready_for_export:
        session.mark_step_complete(WorkflowStep.READY_FOR_EXPORT)
        result += f"**Ready for export!** Use `export_xml_configuration()` to generate PhysiCell files."
    else:
        result += f"**Next step:** Add more rules or use `export_xml_configuration()` to finish."
    
    return result

@mcp.tool()
def add_physiboss_model(cell_type: str, bnd_file: str, cfg_file: str) -> str:
    """
When the user asks to add boolean networks, intracellular models, or MaBoSS models,
this function integrates a PhysiBoSS boolean network model into a cell type.
This function allows the user to specify the MaBoSS .bnd and .cfg files.
In particular, it sets the cell type name, the MaBoSS .bnd file path,
and the MaBoSS .cfg file path.
The function adds one PhysiBoSS model at a time, allowing for detailed configuration; for multiple models,
use this function repeatedly.
This function must be called after defining cell types.

Input parameters:
cell_type (str): Name of existing cell type
bnd_file (str): Path to MaBoSS .bnd boolean network file
cfg_file (str): Path to MaBoSS .cfg configuration file

Returns:
str: Success message with PhysiBoSS model details
    """
    session = get_current_session()
    if not session or not session.config:
        return "Error: Create simulation domain first using create_simulation_domain()"
    
    if not PHYSIBOSS_AVAILABLE:
        return "Error: PhysiBoSS module not available in this PhysiCell configuration package"
    
    try:
        # Use direct config.physiboss API (simpler and more reliable)
        session.config.physiboss.add_intracellular_model(
            cell_type_name=cell_type,
            model_type="maboss",
            bnd_filename=bnd_file,
            cfg_filename=cfg_file
        )
        
        # Update session tracking
        session.physiboss_models_count += 1
        session.mark_step_complete(WorkflowStep.PHYSIBOSS_MODELS_ADDED)
        
        # Auto-create MaBoSS context if not exists to enable PhysiBoSS progress tracking
        if not session.maboss_context:
            from session_manager import MaBoSSContext
            session.maboss_context = MaBoSSContext(
                model_name="auto_created",
                bnd_file_path=bnd_file,
                cfg_file_path=cfg_file,
                available_nodes=[],
                output_nodes=[],
                simulation_results="",
                target_cell_type=cell_type,
                biological_context=""
            )
        
        # Update legacy global for backward compatibility
        _set_legacy_config(session.config)
        
        result = f"**PhysiBoSS model added to {cell_type}:**\n"
        result += f"- Model file: {bnd_file}\n"
        result += f"- Config file: {cfg_file}\n"
        result += f"- Progress: {session.get_progress_percentage():.0f}%\n"
        result += f"**Next step:** Use `configure_physiboss_settings()` to set intracellular parameters."
        
        return result
    except Exception as e:
        return f"Error adding PhysiBoSS model: {str(e)}"

@mcp.tool()
def configure_physiboss_settings(cell_type: str, intracellular_dt: float = 6.0,
                                time_stochasticity: int = 0, scaling: float = 1.0,
                                start_time: float = 0.0, inheritance_global: bool = False) -> str:
    """
When the user asks to configure PhysiBoSS parameters, set intracellular settings, or adjust boolean network timing,
this function configures the intracellular settings for a PhysiBoSS model.
This function allows the user to specify timing, stochasticity, and inheritance parameters.
In particular, it sets the intracellular time step, time stochasticity, scaling factor,
start time, and global inheritance behavior.
The function configures one cell type at a time; for multiple cell types,
use this function repeatedly.
This function must be called after adding a PhysiBoSS model to the cell type.

Input parameters:
cell_type (str): Name of existing cell type with PhysiBoSS model
intracellular_dt (float): PhysiBoSS time step in minutes (default: 6.0)
time_stochasticity (int): Time stochasticity level (default: 0)
scaling (float): Scaling factor for intracellular dynamics (default: 1.0)
start_time (float): Start time for intracellular model in minutes (default: 0.0)
inheritance_global (bool): Whether to use global inheritance (default: False)

Returns:
str: Success message with PhysiBoSS settings details
    """
    session = get_current_session()
    if not session or not session.config:
        return "Error: Create simulation domain first using create_simulation_domain()"
    
    if not PHYSIBOSS_AVAILABLE:
        return "Error: PhysiBoSS module not available in this PhysiCell configuration package"
    
    try:
        # Use direct config.physiboss API (simpler and more reliable)
        session.config.physiboss.set_intracellular_settings(
            cell_type_name=cell_type,
            intracellular_dt=intracellular_dt,
            time_stochasticity=time_stochasticity,
            scaling=scaling,
            start_time=start_time,
            inheritance_global=inheritance_global
        )
        
        # Update session tracking
        session.physiboss_settings_count += 1
        session.mark_step_complete(WorkflowStep.PHYSIBOSS_SETTINGS_CONFIGURED)
        
        # Update legacy global for backward compatibility
        _set_legacy_config(session.config)
        
        result = f"**PhysiBoSS settings configured for {cell_type}:**\n"
        result += f"- Time step: {intracellular_dt} min\n"
        result += f"- Stochasticity: {time_stochasticity}\n"
        result += f"- Scaling: {scaling}\n"
        result += f"- Start time: {start_time} min\n"
        result += f"- Global inheritance: {inheritance_global}\n"
        result += f"- Progress: {session.get_progress_percentage():.0f}%\n"
        result += f"**Next step:** Use `add_physiboss_input_link()` to connect PhysiCell signals to boolean nodes."
        
        return result
    except Exception as e:
        return f"Error configuring PhysiBoSS settings: {str(e)}"

@mcp.tool()
def add_physiboss_input_link(cell_type: str, physicell_signal: str, 
                            boolean_node: str, action: str = "activation",
                            threshold: float = 1.0, smoothing: int = 0) -> str:
    """
When the user asks to link PhysiCell signals to boolean nodes or connect environment to networks,
this function creates an input connection from PhysiCell to the MaBoSS boolean network.
This function allows the user to specify the cell type, PhysiCell signal,
MaBoSS boolean node, action type, activation threshold, and smoothing.
In particular, it sets the cell type name, the PhysiCell signal name,
the MaBoSS boolean network node name, action type, activation threshold, and smoothing level.
The function adds one input link at a time, allowing for detailed configuration; for multiple links,
use this function repeatedly.
This function must be called after defining cell types and adding PhysiBoSS models.

Input parameters:
cell_type (str): Name of existing cell type with PhysiBoSS model
physicell_signal (str): PhysiCell signal name (use list_all_available_signals())
boolean_node (str): MaBoSS boolean network node name
action (str): Action type - "activation" or "inhibition" (default: "activation")
threshold (float): Activation threshold (default: 1.0)
smoothing (int): Smoothing level (default: 0)

Returns:
str: Success message with input link details
    """
    session = get_current_session()
    if not session or not session.config:
        return "Error: Create simulation domain first using create_simulation_domain()"
    
    if not PHYSIBOSS_AVAILABLE:
        return "Error: PhysiBoSS module not available in this PhysiCell configuration package"
    
    try:
        # Use direct config.physiboss API (simpler and more reliable)
        session.config.physiboss.add_intracellular_input(
            cell_type_name=cell_type,
            physicell_name=physicell_signal,
            intracellular_name=boolean_node,
            action=action,
            threshold=threshold,
            smoothing=smoothing
        )
        
        # Update session tracking
        session.physiboss_input_links_count += 1
        session.mark_step_complete(WorkflowStep.PHYSIBOSS_INPUTS_LINKED)
        
        # Update legacy global for backward compatibility
        _set_legacy_config(session.config)
        
        result = f"**PhysiBoSS input:** {physicell_signal} → {boolean_node}\n"
        result += f"- Action: {action}\n"
        result += f"- Threshold: {threshold}\n"
        result += f"- Smoothing: {smoothing}\n"
        result += f"- Progress: {session.get_progress_percentage():.0f}%\n"
        result += f"**Next step:** Use `add_physiboss_output_link()` to connect boolean nodes to cell behaviors."
        
        return result
    except Exception as e:
        return f"Error adding PhysiBoSS input link: {str(e)}"

@mcp.tool()
def add_physiboss_output_link(cell_type: str, boolean_node: str,
                             physicell_behavior: str, action: str = "activation",
                             value: float = 1000000, base_value: float = 0,
                             smoothing: int = 0) -> str:
    """
When the user asks to link boolean nodes to cell behaviors or connect networks to phenotypes,
this function creates an output connection from the MaBoSS boolean network to PhysiCell behaviors.
This function allows the user to specify the cell type, MaBoSS boolean node,
PhysiCell behavior, action type, value when active, base value, and smoothing.
In particular, it sets the cell type name, the MaBoSS boolean network node name,
the PhysiCell behavior name, action type, value when node is active, base value, and smoothing level.
The function adds one output link at a time, allowing for detailed configuration; for multiple links,
use this function repeatedly.
This function must be called after defining cell types and adding PhysiBoSS models.

Input parameters:
cell_type (str): Name of existing cell type with PhysiBoSS model
boolean_node (str): MaBoSS boolean network node name
physicell_behavior (str): PhysiCell behavior name (use list_all_available_behaviors())
action (str): Action type - "activation" or "inhibition" (default: "activation")
value (float): Behavior value when node is active (default: 1000000)
base_value (float): Base behavior value when node is inactive (default: 0)
smoothing (int): Smoothing level (default: 0)

Returns:
str: Success message with output link details
    """
    session = get_current_session()
    if not session or not session.config:
        return "Error: Create simulation domain first using create_simulation_domain()"
    
    if not PHYSIBOSS_AVAILABLE:
        return "Error: PhysiBoSS module not available in this PhysiCell configuration package"
    
    try:
        # Use direct config.physiboss API (simpler and more reliable)
        session.config.physiboss.add_intracellular_output(
            cell_type_name=cell_type,
            physicell_name=physicell_behavior,
            intracellular_name=boolean_node,
            action=action,
            value=value,
            base_value=base_value,
            smoothing=smoothing
        )
        
        # Update session tracking
        session.physiboss_output_links_count += 1
        session.mark_step_complete(WorkflowStep.PHYSIBOSS_OUTPUTS_LINKED)
        
        # Update legacy global for backward compatibility
        _set_legacy_config(session.config)
        
        result = f"**PhysiBoSS output:** {boolean_node} → {physicell_behavior}\n"
        result += f"- Action: {action}\n"
        result += f"- Active value: {value}\n"
        result += f"- Base value: {base_value}\n"
        result += f"- Smoothing: {smoothing}\n"
        result += f"- Progress: {session.get_progress_percentage():.0f}%\n"
        result += f"**Next step:** Use `apply_physiboss_mutation()` for genetic perturbations"
        
        return result
    except Exception as e:
        return f"Error adding PhysiBoSS output link: {str(e)}"

@mcp.tool()
def apply_physiboss_mutation(cell_type: str, node_name: str, fixed_value: int) -> str:
    """
When the user asks to simulate mutations, fix gene states, or apply genetic changes,
this function applies a mutation by fixing a boolean node to a specific value.
This function allows the user to specify the cell type, MaBoSS boolean node,
and the fixed value (0 or 1).
In particular, it sets the cell type name, the MaBoSS boolean network node name,
and the fixed value for the node.
The function applies one mutation at a time, allowing for detailed configuration; for multiple mutations,
use this function repeatedly.   

Input parameters:
cell_type (str): Name of existing cell type with PhysiBoSS model
node_name (str): MaBoSS boolean network node name to mutate
fixed_value (int): Fixed value for the node (0 or 1)

Returns:
str: Success message with mutation details
    """
    session = get_current_session()
    if not session or not session.config:
        return "Error: Create simulation domain first using create_simulation_domain()"
    
    if not PHYSIBOSS_AVAILABLE:
        return "Error: PhysiBoSS module not available in this PhysiCell configuration package"
    
    try:
        # Use direct config.physiboss API (simpler and more reliable)
        session.config.physiboss.add_intracellular_mutation(
            cell_type_name=cell_type,
            intracellular_name=node_name,
            value=fixed_value
        )
        
        # Update session tracking
        session.physiboss_mutations_count += 1
        session.mark_step_complete(WorkflowStep.PHYSIBOSS_MUTATIONS_APPLIED)
        
        # Update legacy global for backward compatibility
        _set_legacy_config(session.config)
        
        result = f"**Mutation applied:** {cell_type}.{node_name} = {fixed_value}\n"
        result += f"- Progress: {session.get_progress_percentage():.0f}%\n"
        result += f"**Next step:** Apply additional mutations or use `export_xml_configuration()` to finish."
        
        return result
    except Exception as e:
        return f"Error applying PhysiBoSS mutation: {str(e)}"

# ============================================================================
# UTILITY AND EXPORT TOOLS
# ============================================================================

@mcp.tool()
def get_simulation_summary() -> str:
    """
    Get a comprehensive summary of the current simulation configuration.
    Shows configured components, progress, and readiness for export.
    
    Returns:
        str: Markdown-formatted summary of current simulation state
    """
    session = get_current_session()
    if not session:
        return "No active session. Use `create_session()` to start."
    
    if not session.config:
        return "No simulation configured yet. Use `create_simulation_domain()` to start."
    
    # Get component counts using correct PhysiCell Settings API
    try:
        substrates = list(session.config.substrates.get_substrates().keys())
    except:
        substrates = []
    
    try:
        cell_types = list(session.config.cell_types.get_cell_types().keys())
    except:
        cell_types = []
    
    # Get rules count using cell_rules_csv API
    rules_count = 0
    try:
        rules_count = len(session.config.cell_rules_csv.get_rules())
    except:
        rules_count = 0
    
    # Calculate progress
    progress = session.get_progress_percentage()
    
    result = f"## Simulation Summary\n\n"
    result += f"**Session:** {session.session_id[:8]}...\n"
    result += f"**Progress:** {progress:.0f}%\n\n"
    
    if session.scenario_context:
        result += f"**Scenario:** {session.scenario_context}\n\n"
    
    # Component details
    result += f"**Components:**\n"
    result += f"- **Substrates ({len(substrates)}):** {', '.join(substrates[:3])}{'...' if len(substrates) > 3 else 'None' if not substrates else ''}\n"
    result += f"- **Cell Types ({len(cell_types)}):** {', '.join(cell_types[:3])}{'...' if len(cell_types) > 3 else 'None' if not cell_types else ''}\n"
    result += f"- **Rules:** {rules_count}\n"
    result += f"- **PhysiBoSS Models:** {session.physiboss_models_count}\n\n"
    
    # Workflow status
    completed_steps = [step.value.replace('_', ' ').title() for step in session.completed_steps]
    if completed_steps:
        result += f"**Completed Steps:**\n"
        for step in completed_steps:
            result += f"{step}\n"
        result += "\n"
    
    # Next recommendations
    recommendations = session.get_next_recommended_steps()
    if recommendations:
        result += f"**Next Steps:**\n"
        for rec in recommendations[:2]:
            result += f"• {rec}\n"
    
    # Export readiness based on core components, not arbitrary percentage
    has_domain = WorkflowStep.DOMAIN_SETUP in session.completed_steps
    has_substrates = WorkflowStep.SUBSTRATES_ADDED in session.completed_steps
    has_cell_types = WorkflowStep.CELL_TYPES_ADDED in session.completed_steps
    ready_for_export = has_domain and has_substrates and has_cell_types
    
    if ready_for_export:
        result += f"\n**Ready for export!** Use `export_xml_configuration()` to generate files."
    elif substrates and cell_types:
        result += f"\n**Basic setup complete.** Add rules or export now."
    else:
        result += f"\n**Setup incomplete.** Add substrates and cell types first."

    return result

@mcp.tool()
def export_xml_configuration(filename: str = "PhysiCell_settings.xml") -> str:
    """
When the user asks to export the simulation, save the configuration, or generate XML,
this function exports the complete PhysiCell configuration to an XML file.
This function generates the XML configuration file based on the current simulation setup,

Input parameters:
filename (str): Name for XML configuration file (default: 'PhysiCell_settings.xml')

Returns:
str: Markdown-formatted export status with file details
    """
    session = get_current_session()
    if not session or not session.config:
        return "**Error:** No simulation configured. Create domain and add components first."
    
    try:
        # Get simulation info for summary using correct API
        try:
            substrates = list(session.config.substrates.get_substrates().keys())
        except:
            substrates = []
        
        try:
            cell_types = list(session.config.cell_types.get_cell_types().keys())
        except:
            cell_types = []
        
        # Fallback to session counters if config access fails
        if not substrates and session.substrates_count > 0:
            substrates = [f"substrate_{i+1}" for i in range(session.substrates_count)]
        if not cell_types and session.cell_types_count > 0:
            cell_types = [f"cell_type_{i+1}" for i in range(session.cell_types_count)]
        
        # Export XML configuration
        xml_content = session.config.generate_xml()
        with open(filename, 'w') as f:
            f.write(xml_content)
        
        xml_size = len(xml_content) // 1024
        
        result = f"## XML Configuration Exported\n\n"
        result += f"**File:** {filename} ({xml_size}KB)\n"
        result += f"**Substrates:** {len(substrates)} ({', '.join(substrates[:3]) if substrates else 'None'}{'...' if len(substrates) > 3 else ''})\n"
        result += f"**Cell Types:** {len(cell_types)} ({', '.join(cell_types[:3]) if cell_types else 'None'}{'...' if len(cell_types) > 3 else ''})\n"
        result += f"**Progress:** {session.get_progress_percentage():.0f}%\n\n"
        result += f"**Next step:** Copy to PhysiCell project directory and run:\n"
        result += f"```bash\n./myproject {filename}\n```"
        
        # Update legacy global for backward compatibility
        _set_legacy_config(session.config)
        
        return result
        
    except Exception as e:
        return f"Error exporting XML configuration: {str(e)}"

@mcp.tool()
def export_cell_rules_csv(filename: str = "cell_rules.csv") -> str:
    """
When the user asks to export rules, save cell behaviors, or generate CSV,
this function exports the cell rules to a CSV file for PhysiCell.
This function generates the CSV file based on the current cell rules configuration,


Input parameters:
filename (str): Name for CSV rules file (default: 'cell_rules.csv')

Returns:
str: Markdown-formatted export status with file details
    """
    session = get_current_session()
    if not session or not session.config:
        return "**Error:** No simulation configured. Create domain and add components first."
    
    try:
        # Use the cell_rules_csv API directly
        rules = session.config.cell_rules_csv
        rule_count = len(rules.get_rules())
        
        if rule_count == 0:
            return "**No cell rules to export**\n\nUse add_single_cell_rule() to create signal-behavior relationships first."
        
        # Export cell rules CSV using the standard method
        rules.generate_csv(filename)
        
        result = f"## Cell Rules CSV Exported\n\n"
        result += f"**File:** {filename}\n"
        result += f"**Rules:** {rule_count}\n"
        result += f"**Progress:** {session.get_progress_percentage():.0f}%\n\n"
        result += f"**Next step:** Copy to PhysiCell project directory alongside XML configuration"
        
        # Update legacy global for backward compatibility
        _set_legacy_config(session.config)
        
        return result
        
    except Exception as e:
        return f"Error exporting cell rules CSV: {str(e)}"

# ============================================================================
# HELPER FUNCTIONS (inspired by NeKo)
# ============================================================================

def clean_for_markdown(text: str) -> str:
    """
    Clean text for markdown output by removing problematic characters.
    """
    if not isinstance(text, str):
        text = str(text)
    return text.replace("|", "\\|").replace("\n", " ").strip()

@mcp.tool()
def list_generated_files(folder_path: str = ".") -> str:
    """
When the user asks to see generated files or check what files have been created,
this function lists PhysiCell-related files in the specified folder.

Args:
folder_path (str): Path to the folder to search (default: current directory)

Returns:
str: List of PhysiCell-related files found
    """
    xml_files = glob.glob(os.path.join(folder_path, "*.xml"))
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    result = f"## Generated Files in {folder_path}\n\n"
    
    if xml_files:
        result += f"**XML files:**\n"
        for f in xml_files:
            result += f"- {os.path.basename(f)}\n"
        result += "\n"
    
    if csv_files:
        result += f"**CSV files:**\n"
        for f in csv_files:
            result += f"- {os.path.basename(f)}\n"
        result += "\n"
    
    if not xml_files and not csv_files:
        result += "No PhysiCell files found."
    
    return result

@mcp.tool()
def clean_generated_files(folder_path: str = ".") -> str:
    """
When the user asks to clean up generated files or remove old configurations,
this function removes PhysiCell XML and CSV files from the specified folder.

Args:
folder_path (str): Path to the folder to clean (default: current directory)

Returns:
str: Status message indicating cleaned files
    """
    xml_files = glob.glob(os.path.join(folder_path, "PhysiCell_*.xml"))
    csv_files = glob.glob(os.path.join(folder_path, "*_rules.csv"))
    
    all_files = xml_files + csv_files
    
    if not all_files:
        return f"No PhysiCell files found to clean in {folder_path}."
    
    for file_path in all_files:
        try:
            os.remove(file_path)
        except OSError as e:
            return f"Error removing {file_path}: {str(e)}"
    
    return f"**Cleaned {len(all_files)} PhysiCell files** from {folder_path}:\n" + "\n".join([f"- {os.path.basename(f)}" for f in all_files])

@mcp.resource(
    uri="docs://tools/index",
    name="Tool Documentation Index",
    description="Links to all tool documentation resources.",
    mime_type="text/markdown"
)
def docs_tools_index() -> str:
    return """
# Tool Documentation Index

- [analyze_biological_scenario](docs://tools/analyze_biological_scenario)
- [create_simulation_domain](docs://tools/create_simulation_domain)
- [add_single_substrate](docs://tools/add_single_substrate)
- [add_single_cell_type](docs://tools/add_single_cell_type)
... (add links to all tool docs)
"""

@mcp.tool()
def get_help() -> str:
    """
    When the user asks for help, available commands, or how to use the server,
    this function returns a guide to the available tools and their usage.
    
    Returns:
        str: Markdown-formatted help guide
    """
    return """# PhysiCell MCP Server Help

## Basic Workflow
1. **analyze_biological_scenario()** - Store your biological context
2. **create_simulation_domain()** - Set up spatial/temporal framework
3. **add_single_substrate()** - Add oxygen, nutrients, drugs, etc.
4. **add_single_cell_type()** - Add cancer cells, immune cells, etc.
5. **add_single_cell_rule()** - Create realistic cell responses
6. **export_xml_configuration()** - Generate PhysiCell XML
7. **export_cell_rules_csv()** - Generate rules CSV

## Key Functions
- **list_all_available_signals()** - See what signals cells can sense
- **list_all_available_behaviors()** - See what cells can do
- **get_simulation_summary()** - Check current setup
- **list_generated_files()** - See exported files
- **clean_generated_files()** - Remove old files

## Example Usage
```
analyze_biological_scenario("hypoxic tumor with immune infiltration")
create_simulation_domain(domain_x=2000, max_time=7200)
add_single_substrate("oxygen", 100000, 0.01, 38.0)
add_single_cell_type("cancer_cell")
add_single_cell_rule("cancer_cell", "oxygen", "decreases", "necrosis", 0.0001, 5.0)
export_xml_configuration("tumor_sim.xml")
```

Most parameters are optional with sensible defaults!"""

if __name__ == "__main__":
    mcp.run()
