"""
PhysiCell MCP Server with Session Management

This server provides tools for configuring PhysiCell biological simulations:
- Create simulation domains and add substrates
- Define cell types and their behaviors
- Create signal-behavior rules for realistic cell responses
- Export configurations for PhysiCell execution

Features lightweight session management and progress tracking.
"""

import inspect
import math
import sys
import time
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Annotated, Literal, Optional

from mcp.types import CallToolResult, ToolAnnotations
from pydantic import Field

# Add the physicell_config package to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Make repo root importable for shared artifact_manager
sys.path.insert(0, str(current_dir.parent))
from artifact_manager import (
    clean_artifacts,
    get_artifact_dir,
    list_artifacts,
    list_artifact_sessions as _list_artifact_sessions_on_disk,
    safe_artifact_path,
    write_session_meta,
)
from mcp_biomodelling_servers import __version__
from mcp_biomodelling_servers.structured_outputs import (
    ArtifactSessionSummary,
    PhysiCellArtifactCleanupResult,
    PhysiCellArtifactFileListResult,
    PhysiCellArtifactSessionListResult,
    PhysiCellRulesExportResult,
    PhysiCellSessionListResult,
    PhysiCellSessionSummary,
    PhysiCellXmlExportResult,
    artifact_file_summary,
    structured_report,
)
from physicell_outputs import (
    PhysiCellBehaviorListResult,
    PhysiCellBehaviorRecord,
    PhysiCellCellTypeRecord,
    PhysiCellCycleModelListResult,
    PhysiCellCycleModelRecord,
    PhysiCellDomainRecord,
    PhysiCellLoadedComponentsResult,
    PhysiCellLoadedConfigurationResult,
    PhysiCellMaBoSSContextRecord,
    PhysiCellMaBoSSContextResult,
    PhysiCellSignalListResult,
    PhysiCellSignalRecord,
    PhysiCellSubstrateRecord,
    PhysiCellWorkflowStatusResult,
    PhysiCellXmlValidationResult,
)
from physicell_guidance import PHYSICELL_AGENT_MANUAL

_SERVER_ROOT = current_dir

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
    get_current_session, analyze_and_update_session_from_config
)

from mcp.server.mcpserver import MCPServer

mcp = MCPServer(
    "PhysiCell",
    title="PhysiCell Configuration Builder",
    description=(
        "Create, inspect, and export PhysiCell simulation configuration files."
    ),
    version=__version__,
)


@mcp.prompt(
    name="physicell_workflow_prompt",
    title="Build or revise a PhysiCell configuration",
    description=(
        "Operating manual for creating, modifying, integrating, and "
        "exporting PhysiCell configurations."
    ),
)
def physicell_workflow_prompt() -> str:
    """Return the PhysiCell configuration-building workflow."""
    return PHYSICELL_AGENT_MANUAL


@mcp.resource(
    uri="docs://physicell/agent_manual",
    name="PhysiCell Agent Operations Manual",
    description=(
        "Single source of truth for PhysiCell configuration workflows, "
        "repeatable operations, PhysiBoSS integration, and export."
    ),
    mime_type="text/markdown",
)
def physicell_agent_manual_resource() -> str:
    """Return the PhysiCell agent operations manual."""
    return PHYSICELL_AGENT_MANUAL


_signals_behaviors_lock = RLock()

NonEmptyString = Annotated[
    str,
    Field(
        min_length=1,
        pattern=r".*\S.*",
        description="A non-empty string containing at least one non-whitespace character.",
    ),
]
ComponentType = Literal["substrates", "cell_types", "physiboss", "all"]
RuleDirection = Literal["increases", "decreases"]
PhysiBoSSAction = Literal["activation", "inhibition"]
MutationState = Literal[0, 1]

_READ_ONLY_TOOL = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
_IDEMPOTENT_TOOL = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
_NON_IDEMPOTENT_TOOL = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=False,
)
_DESTRUCTIVE_TOOL = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
    open_world_hint=False,
)
_IDEMPOTENT_DESTRUCTIVE_TOOL = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=True,
    open_world_hint=False,
)


def _session_locked(handler):
    """Run a synchronous handler under its session's exclusive lease."""
    signature = inspect.signature(handler)

    @wraps(handler)
    def locked_handler(*args, **kwargs):
        arguments = signature.bind_partial(*args, **kwargs).arguments
        with session_manager.session_scope(arguments.get("session_id")):
            return handler(*args, **kwargs)

    return locked_handler


def _optional_session_locked(handler):
    """Lease an optional session without creating an absent default."""
    signature = inspect.signature(handler)

    @wraps(handler)
    def locked_handler(*args, **kwargs):
        arguments = signature.bind_partial(*args, **kwargs).arguments
        with session_manager.optional_session_scope(
            arguments.get("session_id")
        ):
            return handler(*args, **kwargs)

    return locked_handler


def _require_session(session_id: Optional[str] = None) -> SessionState:
    """Return an existing session or raise an actionable tool error."""
    session = get_current_session(session_id)
    if session is not None:
        return session
    if session_id is not None:
        raise ValueError(f"Session not found: {session_id}")
    raise RuntimeError("No active session. Use `create_session()` first.")


def _require_configuration(session_id: Optional[str] = None) -> SessionState:
    """Return a session with a simulation configuration."""
    session = _require_session(session_id)
    if session.config is None:
        raise RuntimeError(
            "No simulation configured. Use `create_simulation_domain()` first."
        )
    return session


def _require_loaded_configuration(session_id: Optional[str] = None) -> SessionState:
    """Return a session containing a configuration loaded from XML."""
    session = _require_session(session_id)
    if session.config is None or not session.loaded_from_xml:
        raise RuntimeError(
            "No XML configuration loaded. Use `load_xml_configuration()` first."
        )
    return session


def _require_physiboss() -> None:
    """Raise when the installed PhysiCell package lacks PhysiBoSS support."""
    if not PHYSIBOSS_AVAILABLE:
        raise RuntimeError(
            "PhysiBoSS support is not available in the installed "
            "`physicell-settings` package."
        )


def _validate_export_filename(filename: str, expected_suffix: str) -> str:
    """Require a plain export basename with the expected file extension."""
    if not filename or not filename.strip():
        raise ValueError("Export filename cannot be empty.")
    if filename != filename.strip():
        raise ValueError("Export filename cannot contain surrounding whitespace.")
    if "\x00" in filename:
        raise ValueError("Export filename cannot contain null bytes.")
    if filename in {".", ".."}:
        raise ValueError(f"Invalid export filename: {filename!r}.")
    if Path(filename).is_absolute() or "/" in filename or "\\" in filename:
        raise ValueError(
            "Export filename must be a basename without directory components: "
            f"{filename!r}."
        )
    if Path(filename).suffix.lower() != expected_suffix.lower():
        raise ValueError(
            f"Export filename must use the {expected_suffix} extension: "
            f"{filename!r}."
        )
    return filename


def _optional_finite_float(value) -> float | None:
    """Convert an accessible backend value into a finite float."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _requirements(value) -> list[str]:
    """Normalize signal/behavior requirements into a string list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    try:
        return [str(item) for item in value]
    except TypeError:
        return [str(value)]


def _configuration_domain(config) -> PhysiCellDomainRecord | None:
    """Read spatial dimensions without making unavailable fields fatal."""
    try:
        return PhysiCellDomainRecord(
            x_size=float(config.domain.x_max - config.domain.x_min),
            y_size=float(config.domain.y_max - config.domain.y_min),
            z_size=float(config.domain.z_max - config.domain.z_min),
        )
    except (AttributeError, TypeError, ValueError):
        return None


def _loaded_substrate_records(
    config,
    names: list[str],
) -> list[PhysiCellSubstrateRecord]:
    """Read loaded substrate properties while retaining inaccessible entries."""
    records = []
    for name in names:
        try:
            substrate = config.substrates.get_substrate(name)
        except Exception:  # noqa: BLE001
            substrate = None
        records.append(
            PhysiCellSubstrateRecord(
                name=name,
                properties_accessible=substrate is not None,
                diffusion_coefficient=(
                    _optional_finite_float(substrate.diffusion_coefficient)
                    if substrate is not None
                    else None
                ),
                decay_rate=(
                    _optional_finite_float(substrate.decay_rate)
                    if substrate is not None
                    else None
                ),
                initial_condition=(
                    _optional_finite_float(substrate.initial_condition)
                    if substrate is not None
                    else None
                ),
            )
        )
    return records


def _loaded_cell_type_records(
    config,
    names: list[str],
) -> list[PhysiCellCellTypeRecord]:
    """Read loaded cell properties while retaining inaccessible entries."""
    records = []
    for name in names:
        try:
            cell_type = config.cell_types.get_cell_type(name)
            total_volume = _optional_finite_float(
                cell_type.phenotype.volume.total
            )
            motility_speed = _optional_finite_float(
                cell_type.phenotype.motility.speed
            )
            cycle_model = str(cell_type.cycle_model)
            physiboss_enabled = bool(
                getattr(cell_type.phenotype, "intracellular", None)
            )
            accessible = True
        except Exception:  # noqa: BLE001
            total_volume = None
            motility_speed = None
            cycle_model = None
            physiboss_enabled = False
            accessible = False
        records.append(
            PhysiCellCellTypeRecord(
                name=name,
                properties_accessible=accessible,
                total_volume=total_volume,
                motility_speed=motility_speed,
                cycle_model=cycle_model,
                physiboss_enabled=physiboss_enabled,
            )
        )
    return records


def _workflow_status_payload(
    session: Optional[SessionState],
) -> PhysiCellWorkflowStatusResult:
    """Build the shared structured status used by both summary aliases."""
    if session is None:
        return PhysiCellWorkflowStatusResult(
            server="PhysiCell",
            session_id=None,
            has_active_session=False,
            has_configuration=False,
            progress=0,
            scenario_context=None,
            substrates=[],
            cell_types=[],
            rules_count=0,
            physiboss_models_count=0,
            completed_steps=[],
            next_steps=[],
            ready_for_export=False,
            loaded_from_xml=False,
            original_xml_path=None,
            xml_modification_count=0,
        )

    substrates = []
    cell_types = []
    rules_count = 0
    if session.config is not None:
        try:
            substrates = list(
                session.config.substrates.get_substrates().keys()
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            cell_types = list(
                session.config.cell_types.get_cell_types().keys()
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            rules_count = len(session.config.cell_rules.get_rules())
        except Exception:  # noqa: BLE001
            pass

    required_steps = {
        WorkflowStep.DOMAIN_SETUP,
        WorkflowStep.SUBSTRATES_ADDED,
        WorkflowStep.CELL_TYPES_ADDED,
    }
    return PhysiCellWorkflowStatusResult(
        server="PhysiCell",
        session_id=session.session_id,
        has_active_session=True,
        has_configuration=session.config is not None,
        progress=session.get_progress_percentage(),
        scenario_context=session.scenario_context or None,
        substrates=[str(name) for name in substrates],
        cell_types=[str(name) for name in cell_types],
        rules_count=rules_count,
        physiboss_models_count=session.physiboss_models_count,
        completed_steps=sorted(
            step.value for step in session.completed_steps
        ),
        next_steps=session.get_next_recommended_steps(),
        ready_for_export=required_steps.issubset(session.completed_steps),
        loaded_from_xml=session.loaded_from_xml,
        original_xml_path=session.original_xml_path,
        xml_modification_count=session.xml_modification_count,
    )


# ============================================================================
# SESSION MANAGEMENT TOOLS
# ============================================================================

@mcp.tool(annotations=_NON_IDEMPOTENT_TOOL)
def create_session(
    set_as_default: bool = Field(default=True, description="Set this as the default session for subsequent operations."),
    session_name: Optional[NonEmptyString] = Field(default=None, description="Optional human-readable name for cross-server linking (e.g., 'gastric_cancer_v1')."),
) -> str:
    """Create a new PhysiCell simulation session.

    Sessions allow independent work on multiple simulation configurations.
    Create one before calling any setup tools.

    Returns:
        str: Session ID and instructions.
    """
    with session_manager.create_session_scope(
        set_as_default=set_as_default,
        session_name=session_name,
    ) as session:
        session_id = session.session_id
        write_session_meta(
            _SERVER_ROOT,
            session_id,
            server_name="PhysiCell",
            label=session_name,
        )

    result = f"**Session created:** {session_id[:8]}..."
    if session_name:
        result += f" ({session_name})"
    result += "\n"
    result += f"**Next steps:**\n"
    result += f"1. `analyze_biological_scenario()` - Set your biological context\n"
    result += f"2. `create_simulation_domain()` - Define spatial framework\n"
    result += f"3. Use `get_workflow_status()` to track progress"
    
    return result

@mcp.tool(annotations=_READ_ONLY_TOOL)
def list_sessions() -> Annotated[CallToolResult, PhysiCellSessionListResult]:
    """List all active simulation sessions with their status and progress.

    Returns:
        CallToolResult: Formatted text plus validated session metadata.
    """
    sessions = session_manager.list_session_snapshots()
    payload = PhysiCellSessionListResult(
        server="PhysiCell",
        count=len(sessions),
        sessions=[
            PhysiCellSessionSummary(
                session_id=session["session_id"],
                session_name=session["session_name"],
                created_at=session["created_at"],
                last_accessed=session["last_accessed"],
                is_default=session["is_default"],
                has_configuration=session["has_config"],
                progress=session["progress"],
                scenario_context=session["scenario_context"] or None,
                substrates_count=session["substrates_count"],
                cell_types_count=session["cell_types_count"],
                rules_count=session["rules_count"],
                physiboss_models_count=session["physiboss_models_count"],
                physiboss_settings_count=session["physiboss_settings_count"],
                physiboss_input_links_count=session["physiboss_input_links_count"],
                physiboss_output_links_count=session["physiboss_output_links_count"],
                physiboss_mutations_count=session["physiboss_mutations_count"],
                loaded_from_xml=session["loaded_from_xml"],
                xml_modification_count=session["xml_modification_count"],
            )
            for session in sessions
        ],
    )

    if not sessions:
        return structured_report(
            "No active sessions. Use `create_session()` to start.",
            payload,
        )

    result = f"## Active Sessions ({len(sessions)})\n\n"

    for session in sessions:
        age_hours = (time.time() - session["created_at"]) / 3600
        progress = session["progress"]

        # Mark default session
        default_marker = " (default)" if session["is_default"] else ""

        result += f"**{session['session_id'][:8]}...{default_marker}**\n"
        result += f"- Age: {age_hours:.1f} hours\n"
        result += f"- Progress: {progress:.0f}%\n"
        result += f"- Components: {session['substrates_count']} substrates, {session['cell_types_count']} cell types, {session['rules_count']} rules\n"

        scenario_context = session["scenario_context"]
        if scenario_context:
            result += f"- Scenario: {scenario_context[:50]}{'...' if len(scenario_context) > 50 else ''}\n"

        result += "\n"

    result += "Use `set_default_session(session_id)` to switch between sessions."

    return structured_report(result, payload)

@mcp.tool(annotations=_READ_ONLY_TOOL)
def list_artifact_sessions() -> Annotated[
    CallToolResult,
    PhysiCellArtifactSessionListResult,
]:
    """List all PhysiCell sessions that have artifact files on disk (including past server runs).

    Unlike list_sessions() which only shows in-memory sessions, this scans the
    artifacts/ directory and reads session_meta.json files, so previously created
    sessions are visible even after a server restart.

    Use the returned session_id and file paths to resume earlier work, e.g.:
      load_xml_configuration(xml_path='/path/to/artifacts/<uuid>/PhysiCell_settings.xml')
    """
    sessions = _list_artifact_sessions_on_disk(_SERVER_ROOT, server_name="PhysiCell")
    payload = PhysiCellArtifactSessionListResult(
        server="PhysiCell",
        count=len(sessions),
        sessions=[
            ArtifactSessionSummary(
                session_id=session["session_id"],
                server=str(session.get("server") or "PhysiCell"),
                label=session.get("label") or None,
                created_at=session.get("created_at") or None,
                files=[str(file_name) for file_name in session.get("files", [])],
            )
            for session in sessions
        ],
    )
    if not sessions:
        return structured_report("No artifact sessions found on disk.", payload)
    lines = ["## PhysiCell Artifact Sessions (on disk)\n"]
    for s in sessions:
        sid = s["session_id"]
        label = s.get("label") or ""
        created = s.get("created_at", "")[:19].replace("T", " ")
        files = s.get("files", [])
        lines.append(f"- **{sid[:8]}...**" + (f" ({label})" if label else ""))
        lines.append(f"  Full ID: `{sid}`")
        if created:
            lines.append(f"  Created: {created} UTC")
        if files:
            lines.append(f"  Files: {', '.join(files)}")
        else:
            lines.append("  Files: (none)")
    return structured_report("\n".join(lines), payload)

@mcp.tool(annotations=_IDEMPOTENT_TOOL)
@_session_locked
def set_default_session(
    session_id: Annotated[NonEmptyString, Field(description="ID of the session to activate. May be shortened to the first 8 characters.")],
) -> str:
    """Switch the default session used by subsequent tool calls.

    Returns:
        str: Confirmation of the session switch.
    """
    success = session_manager.set_default_session(session_id)
    if success:
        session = _require_session(session_id)
        progress = session.get_progress_percentage()
        return f"**Switched to session:** {session.session_id[:8]}... (Progress: {progress:.0f}%)"
    raise ValueError(f"Session not found: {session_id}")

@mcp.tool(annotations=_READ_ONLY_TOOL)
@_optional_session_locked
def get_workflow_status(
    session_id: Optional[NonEmptyString] = Field(
        default=None,
        description="Session to query. Omit to use the active session.",
    ),
) -> Annotated[CallToolResult, PhysiCellWorkflowStatusResult]:
    """Return the current workflow status and recommended next steps.

    Alias for `get_simulation_summary()` — provides identical information.
    Prefer `get_simulation_summary()` for new code.

    Returns:
        str: Progress summary with completed steps and next recommendations.
    """
    session = get_current_session(session_id)
    return structured_report(
        _format_simulation_summary(session),
        _workflow_status_payload(session),
    )

@mcp.tool(annotations=_DESTRUCTIVE_TOOL)
def delete_session(
    session_id: Annotated[NonEmptyString, Field(description="The ID of the session to delete permanently.")],
) -> str:
    """Delete a simulation session and all its state permanently.

    Returns:
        str: Confirmation of deletion.
    """
    success = session_manager.delete_session(session_id)
    if success:
        return f"**Session deleted:** {session_id[:8]}..."
    raise ValueError(f"Session not found: {session_id}")

@mcp.tool(annotations=_IDEMPOTENT_TOOL)
@_session_locked
def set_maboss_context(
    model_name: Annotated[NonEmptyString, Field(description="Name of the MaBoSS model.")],
    bnd_file_path: Annotated[NonEmptyString, Field(description="Absolute path to the .bnd boolean network file.")],
    cfg_file_path: Annotated[NonEmptyString, Field(description="Absolute path to the .cfg configuration file.")],
    target_cell_type: Annotated[NonEmptyString, Field(description="Cell type this boolean model will be integrated into.")],
    available_nodes: str = Field(default="", description="Comma-separated list of available boolean nodes."),
    output_nodes: str = Field(default="", description="Comma-separated list of output nodes."),
    simulation_results: str = Field(default="", description="Summary of MaBoSS simulation behaviour."),
    biological_context: str = Field(default="", description="Original biological question or context."),
    session_id: Optional[NonEmptyString] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Store MaBoSS model context for integration into a PhysiCell simulation.

    Typically called after analysing a MaBoSS model in the MaBoSS server.
    The stored context is used by `add_physiboss_model()` and `add_physiboss_input_link()`.

    Returns:
        str: Confirmation of context storage.
    """
    session = _require_session(session_id)
    
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

@mcp.tool(annotations=_READ_ONLY_TOOL)
@_session_locked
def get_maboss_context(
    session_id: Optional[NonEmptyString] = Field(default=None, description="Session to query. Omit to use the active session."),
) -> Annotated[CallToolResult, PhysiCellMaBoSSContextResult]:
    """Return the stored MaBoSS context for the current session.

    Shows the linked boolean model name, file paths, available nodes, and
    simulation results previously stored via `set_maboss_context()`.

    Returns:
        str: MaBoSS context information, or a message if none is stored.
    """
    session = _require_session(session_id)
    
    if not session.maboss_context:
        payload = PhysiCellMaBoSSContextResult(
            server="PhysiCell",
            session_id=session.session_id,
            has_context=False,
            context=None,
        )
        return structured_report(
            "No MaBoSS context available in current session.",
            payload,
        )
    
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
    
    payload = PhysiCellMaBoSSContextResult(
        server="PhysiCell",
        session_id=session.session_id,
        has_context=True,
        context=PhysiCellMaBoSSContextRecord(
            model_name=ctx.model_name,
            bnd_file_path=ctx.bnd_file_path,
            cfg_file_path=ctx.cfg_file_path,
            available_nodes=list(ctx.available_nodes),
            output_nodes=list(ctx.output_nodes),
            simulation_results=ctx.simulation_results or None,
            target_cell_type=ctx.target_cell_type,
            biological_context=ctx.biological_context or None,
        ),
    )
    return structured_report(result, payload)

# ============================================================================
# XML CONFIGURATION LOADING
# ============================================================================

@mcp.tool(annotations=_IDEMPOTENT_DESTRUCTIVE_TOOL)
def load_xml_configuration(
    filepath: Annotated[NonEmptyString, Field(description="Absolute path to the PhysiCell XML configuration file to load.")],
    session_name: Optional[NonEmptyString] = Field(default=None, description="Optional name for the session, useful for cross-server tracking."),
    session_id: Optional[NonEmptyString] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Load an existing PhysiCell XML configuration file into the current session.

    After loading, inspect with `analyze_loaded_configuration()` or modify directly
    using tools such as `configure_cell_parameters()` and `add_single_cell_rule()`.

    Returns:
        str: Summary of loaded components and next steps.
    """
    xml_path = Path(filepath)
    if not xml_path.exists():
        raise FileNotFoundError(f"PhysiCell XML file not found: {filepath}")
    if not xml_path.is_file():
        raise IsADirectoryError(f"PhysiCell XML path is not a file: {filepath}")

    config = PhysiCellConfig()
    try:
        is_valid, error_msg = config.validate_xml_file(str(xml_path))
    except Exception as exc:
        raise RuntimeError(
            f"Could not validate PhysiCell XML file '{filepath}': {exc}"
        ) from exc

    if not is_valid:
        raise ValueError(f"Invalid PhysiCell XML: {error_msg}")

    try:
        config.load_xml(str(xml_path))
    except Exception as exc:
        raise RuntimeError(
            f"Could not load PhysiCell XML file '{filepath}': {exc}"
        ) from exc

    with session_manager.ensure_session_scope(
        session_id,
        session_name=session_name,
    ) as session:
        session.replace_config(
            config,
            loaded_from_xml=True,
            original_xml_path=str(xml_path.absolute()),
        )
        analyze_and_update_session_from_config(session, config)

        # Concise summary
        parts = [f"{len(session.loaded_substrates)} substrates"]
        parts.append(f"{len(session.loaded_cell_types)} cell types")
        if session.loaded_physiboss_models:
            parts.append(f"{len(session.loaded_physiboss_models)} PhysiBoSS")
        if session.has_existing_rules:
            parts.append("rules")

    result = f"Loaded {xml_path.name}: {', '.join(parts)}"
    result += (
        "\nNext: analyze_loaded_configuration() or start modifying "
        "with existing tools"
    )
    return result

@mcp.tool(annotations=_READ_ONLY_TOOL)
def validate_xml_file(
    filepath: Annotated[NonEmptyString, Field(description="Absolute path to the PhysiCell XML file to validate.")],
) -> Annotated[CallToolResult, PhysiCellXmlValidationResult]:
    """Validate a PhysiCell XML configuration file without loading it.

    Returns:
        str: Validation result — valid confirmation or error description.
    """
    xml_path = Path(filepath)
    if not xml_path.exists():
        raise FileNotFoundError(f"PhysiCell XML file not found: {filepath}")
    if not xml_path.is_file():
        raise IsADirectoryError(f"PhysiCell XML path is not a file: {filepath}")

    config = PhysiCellConfig()
    try:
        is_valid, error_msg = config.validate_xml_file(str(xml_path))
    except Exception as exc:
        raise RuntimeError(
            f"Could not validate PhysiCell XML file '{filepath}': {exc}"
        ) from exc

    text = (
        f"Valid PhysiCell XML: {xml_path.name}"
        if is_valid
        else f"Invalid: {error_msg}"
    )
    payload = PhysiCellXmlValidationResult(
        server="PhysiCell",
        filepath=str(xml_path),
        filename=xml_path.name,
        valid=is_valid,
        error_message=None if is_valid else str(error_msg),
    )
    return structured_report(text, payload)

@mcp.tool(annotations=_IDEMPOTENT_TOOL)
@_session_locked
def analyze_loaded_configuration(
    session_id: Optional[NonEmptyString] = Field(default=None, description="Session to query. Omit to use the active session."),
) -> Annotated[CallToolResult, PhysiCellLoadedConfigurationResult]:
    """Show an overview of the loaded XML configuration with modification instructions.

    Returns:
        str: Configuration summary (domain, substrates, cell types) with tool hints.
    """
    session = _require_loaded_configuration(session_id)
    
    config = session.config
    lines = []
    
    # Source info
    if session.original_xml_path:
        lines.append(f"Source: {Path(session.original_xml_path).name}")
        if session.xml_modification_count > 0:
            lines.append(f"Modified: {session.xml_modification_count} times")
    
    # Domain
    try:
        domain_size = f"{config.domain.x_max-config.domain.x_min}x{config.domain.y_max-config.domain.y_min}x{config.domain.z_max-config.domain.z_min}"
        lines.append(f"Domain: {domain_size} μm")
    except:
        lines.append("Domain: configured")
    
    # Components with modification hints
    if session.loaded_substrates:
        lines.append(f"Substrates ({len(session.loaded_substrates)}): {', '.join(session.loaded_substrates)}")
        lines.append("  → Modify interactions: set_substrate_interaction(cell_type, substrate, ...)")
    
    if session.loaded_cell_types:
        lines.append(f"Cell types ({len(session.loaded_cell_types)}): {', '.join(session.loaded_cell_types)}")
        lines.append("  → Modify parameters: configure_cell_parameters(cell_type, ...)")
        lines.append("  → Add rules: add_single_cell_rule(cell_type, signal, ...)")
    
    if session.loaded_physiboss_models:
        lines.append(f"PhysiBoSS ({len(session.loaded_physiboss_models)}): {', '.join(session.loaded_physiboss_models)}")
        lines.append("  → Configure: configure_physiboss_settings(cell_type, ...)")
    
    lines.append("Use list_loaded_components() for detailed properties")
    
    session.mark_step_complete(WorkflowStep.XML_ANALYZED)
    payload = PhysiCellLoadedConfigurationResult(
        server="PhysiCell",
        session_id=session.session_id,
        source_path=session.original_xml_path,
        modification_count=session.xml_modification_count,
        domain=_configuration_domain(config),
        substrates=list(session.loaded_substrates),
        cell_types=list(session.loaded_cell_types),
        physiboss_models=list(session.loaded_physiboss_models),
        has_existing_rules=session.has_existing_rules,
        marked_analyzed=True,
    )
    return structured_report("\n".join(lines), payload)

@mcp.tool(annotations=_READ_ONLY_TOOL)
@_session_locked
def list_loaded_components(
    component_type: ComponentType = Field(default="all", description="Filter results: 'substrates', 'cell_types', 'physiboss', or 'all'."),
    session_id: Optional[NonEmptyString] = Field(default=None, description="Session to query. Omit to use the active session."),
) -> Annotated[CallToolResult, PhysiCellLoadedComponentsResult]:
    """List components from a loaded XML configuration with details and modification hints.

    Returns:
        str: Detailed component information with parameter values and tool hints.
    """
    session = _require_loaded_configuration(session_id)
    
    config = session.config
    lines = []
    substrate_records = (
        _loaded_substrate_records(config, session.loaded_substrates)
        if component_type in ["all", "substrates"]
        else []
    )
    cell_type_records = (
        _loaded_cell_type_records(config, session.loaded_cell_types)
        if component_type in ["all", "cell_types"]
        else []
    )
    physiboss_models = (
        list(session.loaded_physiboss_models)
        if component_type in ["all", "physiboss"]
        else []
    )
    
    if component_type in ["all", "substrates"] and session.loaded_substrates:
        lines.append("SUBSTRATES:")
        for name in session.loaded_substrates:
            try:
                substrate = config.substrates.get_substrate(name)
                if substrate:
                    lines.append(f"  {name}: D={substrate.diffusion_coefficient}, decay={substrate.decay_rate}, init={substrate.initial_condition}")
            except:
                lines.append(f"  {name}: properties not accessible")
        lines.append("  → Add interactions: set_substrate_interaction(cell_type, substrate, secretion_rate=X, uptake_rate=Y)")
        lines.append("")
    
    if component_type in ["all", "cell_types"] and session.loaded_cell_types:
        lines.append("CELL TYPES:")
        for name in session.loaded_cell_types:
            try:
                cell_type = config.cell_types.get_cell_type(name)
                if cell_type:
                    vol = cell_type.phenotype.volume.total
                    speed = cell_type.phenotype.motility.speed
                    cycle = cell_type.cycle_model
                    
                    physiboss = ""
                    if (hasattr(cell_type, 'phenotype') and hasattr(cell_type.phenotype, 'intracellular') and 
                        cell_type.phenotype.intracellular):
                        physiboss = ", PhysiBoSS enabled"
                    
                    lines.append(f"  {name}: vol={vol}, speed={speed}, cycle={cycle}{physiboss}")
            except:
                lines.append(f"  {name}: properties not accessible")
        
        lines.append("  → Modify parameters: configure_cell_parameters(cell_type, volume_total=X, motility_speed=Y, ...)")
        lines.append("  → Add rules: add_single_cell_rule(cell_type, signal, direction, behavior, ...)")
        lines.append("")
    
    if component_type in ["all", "physiboss"] and session.loaded_physiboss_models:
        lines.append("PHYSIBOSS MODELS:")
        for name in session.loaded_physiboss_models:
            lines.append(f"  {name}: Intracellular boolean network enabled")
        lines.append("  → Configure: configure_physiboss_settings(cell_type, intracellular_dt=X, ...)")
        lines.append("  → Add links: add_physiboss_input_link() / add_physiboss_output_link()")
        lines.append("")
    
    text = (
        "\n".join(lines).strip()
        if lines
        else f"No {component_type} components found in loaded configuration"
    )
    payload = PhysiCellLoadedComponentsResult(
        server="PhysiCell",
        session_id=session.session_id,
        component_type=component_type,
        substrate_count=len(substrate_records),
        cell_type_count=len(cell_type_records),
        physiboss_model_count=len(physiboss_models),
        substrates=substrate_records,
        cell_types=cell_type_records,
        physiboss_models=physiboss_models,
    )
    return structured_report(text, payload)

# ============================================================================
# BIOLOGICAL SCENARIO ANALYSIS
# ============================================================================

@mcp.tool(annotations=_IDEMPOTENT_TOOL)
def analyze_biological_scenario(
    biological_scenario: Annotated[NonEmptyString, Field(description="Description of the biological scenario or experimental setup (e.g., 'Breast cancer cells in hypoxic 3D tissue with immune infiltration').")],
    session_id: Optional[NonEmptyString] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Store a biological scenario description to provide context for subsequent simulation setup.

    This context is displayed alongside signal/behavior lists and helps inform parameter choices.
    Call this before `create_simulation_domain()` to anchor the session to a specific biology.

    Returns:
        str: Confirmation message.
    """
    if not biological_scenario or not biological_scenario.strip():
        raise ValueError("Biological scenario description cannot be empty.")
    
    with session_manager.ensure_session_scope(session_id) as session:
        session.scenario_context = biological_scenario.strip()
        session.mark_step_complete(WorkflowStep.SCENARIO_ANALYSIS)
    
    result = f"**Biological scenario stored:** {biological_scenario}\n"
    result += f"**Next step:** Use `create_simulation_domain()` to set up the spatial framework."
    
    return result

# ============================================================================
# SIMULATION SETUP
# ============================================================================

@mcp.tool(annotations=_IDEMPOTENT_DESTRUCTIVE_TOOL)
def create_simulation_domain(
    domain_x: Annotated[float, Field(gt=0, allow_inf_nan=False, description="Domain width in micrometers (e.g., 2000).")],
    domain_y: Annotated[float, Field(gt=0, allow_inf_nan=False, description="Domain height in micrometers (e.g., 2000).")],
    use_2d: bool = Field(default=True, description="If True (default), creates a quasi-2D simulation. The z thickness is automatically set to the mesh spacing (dx). Set to False for a full 3D simulation and provide domain_z."),
    domain_z: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False, description="Domain depth in micrometers. Required only for 3D simulations (use_2d=False). Ignored in 2D mode."),
    dx: float = Field(default=20.0, gt=0, allow_inf_nan=False, description="Mesh spacing in micrometers. Smaller values increase spatial resolution and computation time. In 2D, also sets the z thickness."),
    max_time: float = Field(default=7200.0, gt=0, allow_inf_nan=False, description="Maximum simulation time in minutes. 7200 = 5 days."),
    session_id: Optional[NonEmptyString] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Create the spatial and temporal framework for a PhysiCell simulation.

    Sets up the domain bounds, mesh resolution, and simulation duration.
    Must be called before adding substrates or cell types.
    The domain is centred at the origin: bounds run from -x/2 to +x/2 in each axis.

    In 2D mode (`use_2d=True`) the z extent is set to `dx` (one voxel thick)
    and `set_2D(True)` is applied so PhysiCell treats the simulation as planar.

    Returns:
        str: Confirmation with domain specs and next step.
    """
    # Basic validation
    if domain_x <= 0 or domain_y <= 0:
        raise ValueError("Domain dimensions must be positive.")
    if dx <= 0:
        raise ValueError("Mesh spacing must be positive.")
    if max_time <= 0:
        raise ValueError("Simulation time must be positive.")

    # Resolve z dimension
    if use_2d:
        domain_z = dx  # one voxel thick
    else:
        if domain_z is None:
            raise ValueError(
                "domain_z is required for 3D simulations. Set use_2d=True "
                "for a planar simulation."
            )
        if domain_z <= 0:
            raise ValueError("domain_z must be positive.")

    # Build locally so a failed rebuild cannot overwrite a valid session.
    config = PhysiCellConfig()
    config.domain.set_bounds(
        -domain_x/2, domain_x/2,
        -domain_y/2, domain_y/2,
        -domain_z/2, domain_z/2
    )
    config.domain.set_mesh(dx, dx, dx)

    if use_2d:
        config.domain.set_2D(True)

    config.options.set_max_time(max_time)
    config.options.set_time_steps(
        dt_diffusion=0.01,
        dt_mechanics=0.1,
        dt_phenotype=6.0,
    )

    with session_manager.ensure_session_scope(session_id) as session:
        session.replace_config(config)
        session.mark_step_complete(WorkflowStep.DOMAIN_SETUP)
        progress = session.get_progress_percentage()
    
    # Format result
    mode = "2D" if use_2d else "3D"
    result = f"**Simulation domain created ({mode}):**\n"
    result += f"- Domain: {domain_x}×{domain_y}" + (f" μm (z = {domain_z} μm, one voxel)\n" if use_2d else f"×{domain_z} μm\n")
    result += f"- Mesh: {dx} μm\n"
    result += f"- Duration: {max_time/60:.1f} hours\n"
    result += f"- Progress: {progress:.0f}%\n"
    result += f"**Next step:** Use `add_single_substrate()` to add oxygen, nutrients, or drugs."
    
    return result

@mcp.tool(annotations=_DESTRUCTIVE_TOOL)
@_session_locked
def add_single_substrate(
    substrate_name: Annotated[NonEmptyString, Field(description="Name of the substrate (e.g., 'oxygen', 'glucose', 'drug').")],
    diffusion_coefficient: Annotated[float, Field(ge=0, allow_inf_nan=False, description="Diffusion rate in μm²/min. Typical: 100000 for oxygen, 30000 for glucose.")],
    decay_rate: Annotated[float, Field(ge=0, allow_inf_nan=False, description="Decay/uptake rate in 1/min. Typical: 0.01.")],
    initial_condition: Annotated[float, Field(allow_inf_nan=False, description="Starting concentration everywhere in the domain. Typical: 38 for oxygen (mmHg).")],
    units: NonEmptyString = Field(default="dimensionless", description="Concentration units label (cosmetic only)."),
    dirichlet_enabled: bool = Field(default=False, description="If True, enforce a fixed boundary concentration (Dirichlet condition)."),
    dirichlet_value: Optional[float] = Field(default=None, allow_inf_nan=False, description="Fixed boundary concentration. Defaults to initial_condition if omitted."),
    session_id: Optional[NonEmptyString] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Add a chemical substrate (oxygen, glucose, drug, etc.) to the simulation environment.

    Substrates diffuse through the domain and can be consumed or secreted by cells.
    Call `create_simulation_domain()` first. Repeat for each substrate.

    Returns:
        str: Confirmation with substrate parameters and next step.
    """
    session = _require_configuration(session_id)
    
    # Basic validation
    if not substrate_name or not substrate_name.strip():
        raise ValueError("Substrate name cannot be empty.")
    if diffusion_coefficient < 0:
        raise ValueError("Diffusion coefficient must be non-negative.")
    if decay_rate < 0:
        raise ValueError("Decay rate must be non-negative.")
    
    if dirichlet_value is None:
        dirichlet_value = initial_condition
    
    name = substrate_name.strip()
    
    # Add substrate to configuration
    session.config.substrates.add_substrate(
        name,
        diffusion_coefficient=diffusion_coefficient,
        decay_rate=decay_rate,
        initial_condition=initial_condition,
        dirichlet_enabled=dirichlet_enabled,
        dirichlet_value=dirichlet_value,
        units=units
    )
    
    # Set per-boundary Dirichlet conditions when enabled
    dirichlet_boundaries = []
    if dirichlet_enabled:
        # Determine 2D vs 3D using the domain's use_2D flag
        try:
            is_3d = not session.config.domain.data.get("use_2D", False)
        except Exception:
            is_3d = False
        
        boundaries = ["xmin", "xmax", "ymin", "ymax"]
        if is_3d:
            boundaries += ["zmin", "zmax"]
        
        for boundary in boundaries:
            session.config.substrates.set_dirichlet_boundary(name, boundary, True, dirichlet_value)
        dirichlet_boundaries = boundaries
    
    # Update session counters
    session.substrates_count += 1
    session.mark_step_complete(WorkflowStep.SUBSTRATES_ADDED)
    
    # Format result
    result = f"**Substrate added:** {substrate_name}\n"
    result += f"- Diffusion: {diffusion_coefficient:g} μm²/min\n"
    result += f"- Decay: {decay_rate:g} min⁻¹\n"
    result += f"- Initial: {initial_condition:g} {units}\n"
    if dirichlet_enabled:
        result += f"- Dirichlet boundaries: {', '.join(dirichlet_boundaries)} = {dirichlet_value:g} {units}\n"
    result += f"- Progress: {session.get_progress_percentage():.0f}%\n"
    result += f"**Next step:** Use `add_single_cell_type()` to add cancer cells, immune cells, etc."
    
    return result

@mcp.tool(annotations=_DESTRUCTIVE_TOOL)
@_session_locked
def add_single_cell_type(
    cell_type_name: Annotated[NonEmptyString, Field(description="Name for this cell type (e.g., 'cancer_cell', 'immune_cell', 'fibroblast').")],
    cycle_model: NonEmptyString = Field(default="Ki67_basic", description="Cell cycle model. Use `get_available_cycle_models()` to list options. Common: 'Ki67_basic', 'Ki67_advanced', 'live'."),
    session_id: Optional[NonEmptyString] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Add a cell type (cancer, immune, fibroblast, etc.) to the simulation.

    Cell types define the agents that populate the simulation domain.
    Call `create_simulation_domain()` first. Repeat for each cell type.
    After adding, use `configure_cell_parameters()` to set size and motility,
    and `add_single_cell_rule()` to define signal-response behaviours.

    Returns:
        str: Confirmation with cell type details and next step.
    """
    session = _require_configuration(session_id)
    
    # Basic validation
    if not cell_type_name or not cell_type_name.strip():
        raise ValueError("Cell type name cannot be empty.")
    
    cell_type_name = cell_type_name.strip()
    
    # Add cell type to configuration
    session.config.cell_types.add_cell_type(cell_type_name, template='default')
    session.config.cell_types.set_cycle_model(cell_type_name, cycle_model)
    
    # Update session counters
    session.cell_types_count += 1
    session.mark_step_complete(WorkflowStep.CELL_TYPES_ADDED)
    
    # Format result
    result = f"**Cell type added:** {cell_type_name}\n"
    result += f"- Cycle model: {cycle_model}\n"
    result += f"- Progress: {session.get_progress_percentage():.0f}%\n"
    result += f"**Next step:** Use `add_single_cell_rule()` to create cell behavior rules.\n"
    result += f"First, use `list_all_available_signals()` and `list_all_available_behaviors()` to see options."
    
    return result



@mcp.tool(annotations=_IDEMPOTENT_TOOL)
@_session_locked
def configure_cell_parameters(
    cell_type: Annotated[NonEmptyString, Field(description="Name of an existing cell type to configure.")],
    volume_total: float = Field(default=2500.0, gt=0, allow_inf_nan=False, description="Total cell volume in μm³."),
    volume_nuclear: float = Field(default=500.0, gt=0, allow_inf_nan=False, description="Nuclear volume in μm³."),
    fluid_fraction: float = Field(default=0.75, ge=0, le=1, allow_inf_nan=False, description="Cytoplasmic fluid fraction (0–1)."),
    motility_speed: float = Field(default=0.5, ge=0, allow_inf_nan=False, description="Cell migration speed in μm/min."),
    persistence_time: float = Field(default=5.0, ge=0, allow_inf_nan=False, description="Directional persistence time in minutes."),
    apoptosis_rate: float = Field(default=0.0001, ge=0, allow_inf_nan=False, description="Spontaneous apoptosis rate in 1/min."),
    necrosis_rate: float = Field(default=0.0001, ge=0, allow_inf_nan=False, description="Spontaneous necrosis rate in 1/min."),
    session_id: Optional[NonEmptyString] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Modify volume, motility, and death-rate parameters for an existing cell type.

    The cell type must already exist (created by `add_single_cell_type()`).
    Call repeatedly to configure multiple cell types.

    Returns:
        str: Confirmation with the configured parameter values.
    """
    session = _require_configuration(session_id)
    
    try:
        # Set volume parameters
        session.config.cell_types.set_volume_parameters(cell_type, total=volume_total,
                                                        nuclear=volume_nuclear, fluid_fraction=fluid_fraction)

        # Set motility parameters
        session.config.cell_types.set_motility(cell_type, speed=motility_speed,
                                               persistence_time=persistence_time, enabled=True)

        # Set death rates
        session.config.cell_types.set_death_rate(cell_type, 'apoptosis', apoptosis_rate)
        session.config.cell_types.set_death_rate(cell_type, 'necrosis', necrosis_rate)

        # Track modification if loaded from XML
        if session.loaded_from_xml:
            session.mark_xml_modification()

        result = f"**Configured parameters for {cell_type}:**\n"
        result += f"- **Volume:** {volume_total:g} μm³ (nuclear: {volume_nuclear:g} μm³)\n"
        result += f"- **Motility:** {motility_speed:g} μm/min (persistence: {persistence_time:g} min)\n"
        result += f"- **Death rates:** apoptosis {apoptosis_rate:g}, necrosis {necrosis_rate:g} min⁻¹"
        
        return result
    except Exception as exc:
        raise RuntimeError(
            f"Could not configure cell type '{cell_type}': {exc}"
        ) from exc

@mcp.tool(annotations=_IDEMPOTENT_TOOL)
@_session_locked
def set_substrate_interaction(
    cell_type: Annotated[NonEmptyString, Field(description="Name of an existing cell type.")],
    substrate: Annotated[NonEmptyString, Field(description="Name of an existing substrate (must match a name passed to add_single_substrate()).")],
    secretion_rate: float = Field(default=0.0, ge=0, allow_inf_nan=False, description="Rate at which the cell secretes the substrate (1/min)."),
    uptake_rate: float = Field(default=0.0, ge=0, allow_inf_nan=False, description="Rate at which the cell consumes the substrate (1/min). Typical oxygen uptake: 10."),
    session_id: Optional[NonEmptyString] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Define how a cell type interacts with a substrate via secretion and uptake rates.

    Both cell type and substrate must already exist in the current session.
    Call repeatedly to configure multiple cell-substrate pairs.

    Returns:
        str: Confirmation with the configured rates.
    """
    session = _require_configuration(session_id)

    try:
        session.config.cell_types.add_secretion(cell_type, substrate,
                                                secretion_rate=secretion_rate,
                                                uptake_rate=uptake_rate)

        # Track modification if loaded from XML
        if session.loaded_from_xml:
            session.mark_xml_modification()
        
        return f"**Substrate interaction set:** {cell_type} ↔ {substrate} (secretion: {secretion_rate:g}, uptake: {uptake_rate:g} min⁻¹)"
    except Exception as exc:
        raise RuntimeError(
            f"Could not set the substrate interaction: {exc}"
        ) from exc

# ============================================================================
# PARAMETER DISCOVERY AND DEFAULTS
# ============================================================================

@mcp.tool(annotations=_READ_ONLY_TOOL)
def get_available_cycle_models(
) -> Annotated[CallToolResult, PhysiCellCycleModelListResult]:
    """List all available PhysiCell cell cycle models with their identifiers.

    Returns:
        str: Markdown list of model keys and names to use in `add_single_cell_type()`.
    """

    defaults = get_default_parameters()
    cycle_models = defaults.get("cell_cycle_models", {})
    
    result = "## Available Cell Cycle Models\n\n"
    for model_key, model_data in cycle_models.items():
        model_name = model_data.get("name", model_key)
        result += f"- **{model_key}**: {model_name}\n"
    
    result += "\n**Usage:** Use exact model names in add_single_cell_type() function.\n"
    result += "**Most common:** Ki67_basic, Ki67_advanced, live"
    
    models = [
        PhysiCellCycleModelRecord(
            key=str(model_key),
            name=str(model_data.get("name", model_key)),
        )
        for model_key, model_data in cycle_models.items()
    ]
    payload = PhysiCellCycleModelListResult(
        server="PhysiCell",
        model_count=len(models),
        models=models,
    )
    return structured_report(result, payload)

# ============================================================================
# SIGNAL AND BEHAVIOR DISCOVERY
# ============================================================================

@mcp.tool(annotations=_READ_ONLY_TOOL)
@_optional_session_locked
def list_all_available_signals(
    session_id: Optional[NonEmptyString] = Field(default=None, description="Session to query. Omit to use the active session."),
) -> Annotated[CallToolResult, PhysiCellSignalListResult]:
    """List all PhysiCell signals that can be used in cell rules.

    Automatically expands to include substrate-specific and cell-type-specific signals
    based on components already added to the current session.
    Call `list_all_available_behaviors()` to see the corresponding behavior targets.

    Returns:
        str: Markdown table of signals grouped by type.
    """
    session = get_current_session(session_id)
    
    # Update context from current config if available
    if session and session.config:
        with _signals_behaviors_lock:
            update_signals_behaviors_context_from_config(session.config)
            # Use expanded signals which include context-specific signals
            try:
                signals_data = {
                    signal["name"]: signal
                    for signal in get_expanded_signals()
                }
            except Exception:  # noqa: BLE001
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
    
    signals = [
        PhysiCellSignalRecord(
            name=str(signal_info.get("name", signal_name)),
            signal_type=str(signal_info.get("type", "other")),
            description=str(
                signal_info.get("description", "No description")
            ),
            requires=_requirements(signal_info.get("requires", [])),
        )
        for signal_name, signal_info in signals_data.items()
    ]
    payload = PhysiCellSignalListResult(
        server="PhysiCell",
        session_id=session.session_id if session is not None else None,
        scenario_context=scenario_context or None,
        signal_count=len(signals),
        signals=signals,
    )
    return structured_report(result, payload)

@mcp.tool(annotations=_READ_ONLY_TOOL)
@_optional_session_locked
def list_all_available_behaviors(
    session_id: Optional[NonEmptyString] = Field(default=None, description="Session to query. Omit to use the active session."),
) -> Annotated[CallToolResult, PhysiCellBehaviorListResult]:
    """List all PhysiCell behaviours that can be controlled by cell rules.

    Automatically expands to include substrate-specific and cell-type-specific behaviours
    based on components already added to the current session.
    Use exact behavior names in `add_single_cell_rule()`.

    Returns:
        str: Markdown table of behaviours grouped by type.
    """
    session = get_current_session(session_id)
    
    # Update context from current config if available
    if session and session.config:
        with _signals_behaviors_lock:
            update_signals_behaviors_context_from_config(session.config)
            # Use expanded behaviors which include context-specific behaviors
            try:
                behaviors_data = {
                    behavior["name"]: behavior
                    for behavior in get_expanded_behaviors()
                }
            except Exception:  # noqa: BLE001
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
    
    behaviors = [
        PhysiCellBehaviorRecord(
            name=str(behavior_info.get("name", behavior_name)),
            behavior_type=str(behavior_info.get("type", "other")),
            description=str(
                behavior_info.get("description", "No description")
            ),
            requires=_requirements(behavior_info.get("requires", [])),
        )
        for behavior_name, behavior_info in behaviors_data.items()
    ]
    payload = PhysiCellBehaviorListResult(
        server="PhysiCell",
        session_id=session.session_id if session is not None else None,
        scenario_context=scenario_context or None,
        behavior_count=len(behaviors),
        behaviors=behaviors,
    )
    return structured_report(result, payload)


# ============================================================================
# CELL RULES AND PHYSIBOSS
# ============================================================================

@mcp.tool(annotations=_NON_IDEMPOTENT_TOOL)
@_session_locked
def add_single_cell_rule(
    cell_type: Annotated[NonEmptyString, Field(description="Name of existing cell type.")],
    signal: Annotated[NonEmptyString, Field(description="Signal name. Use list_all_available_signals() to see options.")],
    direction: Annotated[RuleDirection, Field(description="'increases' or 'decreases' — whether the signal promotes or suppresses the behavior.")],
    behavior: Annotated[NonEmptyString, Field(description="Behavior name. Use list_all_available_behaviors() to see options.")],
    saturation_value: float = Field(default=1.0, allow_inf_nan=False, description="Value of the behavior when the signal is at saturation (maximum effect)."),
    half_max: float = Field(default=0.5, gt=0, allow_inf_nan=False, description="Signal level at which the behavior is halfway between its base value and the saturation value."),
    hill_power: float = Field(default=4.0, gt=0, allow_inf_nan=False, description="Hill coefficient controlling the sharpness of the dose-response curve (typical: 1–8)."),
    session_id: Optional[NonEmptyString] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Add a signal-behaviour rule that makes cells respond to environmental cues.

    Rules are Hill-function dose-response relationships: when `signal` rises,
    `behavior` is increased or decreased according to a Hill curve parameterised by
    `saturation_value`, `half_max`, and `hill_power`.
    Use `list_all_available_signals()` and `list_all_available_behaviors()` first.

    Returns:
        str: Confirmation with rule summary and export readiness indicator.
    """
    session = _require_configuration(session_id)
    
    # Basic validation
    if not cell_type or not cell_type.strip():
        raise ValueError("Cell type name cannot be empty.")
    if not signal or not signal.strip():
        raise ValueError("Signal name cannot be empty.")
    if direction not in ['increases', 'decreases']:
        raise ValueError("Direction must be 'increases' or 'decreases'.")
    if not behavior or not behavior.strip():
        raise ValueError("Behavior name cannot be empty.")
    if half_max <= 0:
        raise ValueError("Half-max value must be positive.")
    if hill_power <= 0:
        raise ValueError("Hill power must be positive.")
    
    with _signals_behaviors_lock:
        # The upstream package validates rules against process-global context.
        update_signals_behaviors_context_from_config(session.config)

        # Add rule using the CellRulesModule API (session.config.cell_rules)
        session.config.cell_rules.add_rule(
            cell_type=cell_type.strip(),
            signal=signal.strip(),
            direction=direction,
            behavior=behavior.strip(),
            saturation_value=saturation_value,
            half_max=half_max,
            hill_power=hill_power,
            apply_to_dead=0
        )
    
    # Update session counters
    session.rules_count += 1
    session.mark_step_complete(WorkflowStep.RULES_CONFIGURED)
    
    # Track modification if loaded from XML
    if session.loaded_from_xml:
        session.mark_xml_modification()
    
    # Format result
    result = f"**Cell rule added:**\n"
    result += f"- Rule: {cell_type} | {signal} {direction} → {behavior}\n"
    result += f"- Saturation value: {saturation_value}\n"
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

@mcp.tool(annotations=_DESTRUCTIVE_TOOL)
@_session_locked
def add_physiboss_model(
    cell_type: Annotated[NonEmptyString, Field(description="Name of an existing cell type to attach the boolean network to.")],
    bnd_file: Annotated[NonEmptyString, Field(description="Absolute path to the MaBoSS .bnd boolean network file.")],
    cfg_file: Annotated[NonEmptyString, Field(description="Absolute path to the MaBoSS .cfg configuration file.")],
    session_id: Optional[NonEmptyString] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Integrate a PhysiBoSS boolean network model into a cell type.

    The cell type must already exist. Requires PhysiBoSS support in the
    installed `physicell_config` package.
    After adding, call `configure_physiboss_settings()`, then
    `add_physiboss_input_link()` and `add_physiboss_output_link()`.

    Returns:
        str: Confirmation with model file paths and next step.
    """
    session = _require_configuration(session_id)
    _require_physiboss()
    
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
        
        result = f"**PhysiBoSS model added to {cell_type}:**\n"
        result += f"- Model file: {bnd_file}\n"
        result += f"- Config file: {cfg_file}\n"
        result += f"- Progress: {session.get_progress_percentage():.0f}%\n"
        result += f"**Next step:** Use `configure_physiboss_settings()` to set intracellular parameters."
        
        return result
    except Exception as exc:
        raise RuntimeError(f"Could not add the PhysiBoSS model: {exc}") from exc

@mcp.tool(annotations=_IDEMPOTENT_TOOL)
@_session_locked
def configure_physiboss_settings(
    cell_type: Annotated[NonEmptyString, Field(description="Name of an existing cell type with a PhysiBoSS model attached.")],
    intracellular_dt: float = Field(default=6.0, gt=0, allow_inf_nan=False, description="PhysiBoSS update interval in minutes. Should be a multiple of the phenotype time step."),
    time_stochasticity: int = Field(default=0, ge=0, description="Time stochasticity level (0 = deterministic)."),
    scaling: float = Field(default=1.0, gt=0, allow_inf_nan=False, description="Scaling factor applied to the intracellular dynamics."),
    start_time: float = Field(default=0.0, ge=0, allow_inf_nan=False, description="Simulation time in minutes at which the boolean network starts running."),
    inheritance_global: bool = Field(default=False, description="If True, daughter cells inherit the parent's boolean node states globally."),
    session_id: Optional[NonEmptyString] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Configure timing, stochasticity, and inheritance parameters for a PhysiBoSS model.
    Must be called after `add_physiboss_model()`. Repeat for each cell type.
    Checked with `list_all_available_signals()` and `list_all_available_behaviors()` the available signals and behaviors to use in `add_physiboss_input_link()` and `add_physiboss_output_link()`.
    Returns:
        str: Confirmation with the configured settings and next step.
    """
    session = _require_configuration(session_id)
    _require_physiboss()
    
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
        
        result = f"**PhysiBoSS settings configured for {cell_type}:**\n"
        result += f"- Time step: {intracellular_dt} min\n"
        result += f"- Stochasticity: {time_stochasticity}\n"
        result += f"- Scaling: {scaling}\n"
        result += f"- Start time: {start_time} min\n"
        result += f"- Global inheritance: {inheritance_global}\n"
        result += f"- Progress: {session.get_progress_percentage():.0f}%\n"
        result += f"**Next step:** Use `add_physiboss_input_link()` to connect PhysiCell signals to boolean nodes."
        
        return result
    except Exception as exc:
        raise RuntimeError(
            f"Could not configure PhysiBoSS settings: {exc}"
        ) from exc

@mcp.tool(annotations=_NON_IDEMPOTENT_TOOL)
@_session_locked
def add_physiboss_input_link(
    cell_type: Annotated[NonEmptyString, Field(description="Name of an existing cell type with a PhysiBoSS model.")],
    physicell_signal: Annotated[NonEmptyString, Field(description="PhysiCell signal name. Use list_all_available_signals() to see options.")],
    boolean_node: Annotated[NonEmptyString, Field(description="MaBoSS boolean node name to drive (from the .bnd file).")],
    action: PhysiBoSSAction = Field(default="activation", description="'activation' (signal turns node ON) or 'inhibition' (signal turns node OFF)."),
    threshold: float = Field(default=1.0, allow_inf_nan=False, description="Signal level at which the node is toggled."),
    smoothing: int = Field(default=0, ge=0, description="Smoothing level (0 = no smoothing)."),
    session_id: Optional[NonEmptyString] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Create an input link from a PhysiCell signal to a MaBoSS boolean node.
    Requires a PhysiBoSS model already attached to the cell type.
    Call `add_physiboss_output_link()` after all input links are defined.
    Returns:
        str: Confirmation with link details and next step.
    """
    session = _require_configuration(session_id)
    _require_physiboss()
    
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
        
        result = f"**PhysiBoSS input:** {physicell_signal} → {boolean_node}\n"
        result += f"- Action: {action}\n"
        result += f"- Threshold: {threshold}\n"
        result += f"- Smoothing: {smoothing}\n"
        result += f"- Progress: {session.get_progress_percentage():.0f}%\n"
        result += f"**Next step:** Use `add_physiboss_output_link()` to connect boolean nodes to cell behaviors."
        
        return result
    except Exception as exc:
        raise RuntimeError(
            f"Could not add the PhysiBoSS input link: {exc}"
        ) from exc

@mcp.tool(annotations=_NON_IDEMPOTENT_TOOL)
@_session_locked
def add_physiboss_output_link(
    cell_type: Annotated[NonEmptyString, Field(description="Name of an existing cell type with a PhysiBoSS model.")],
    boolean_node: Annotated[NonEmptyString, Field(description="MaBoSS boolean node name whose state drives the behavior.")],
    physicell_behavior: Annotated[NonEmptyString, Field(description="PhysiCell behavior name to control. Use list_all_available_behaviors() to see options.")],
    action: PhysiBoSSAction = Field(default="activation", description="'activation' (node ON increases behavior) or 'inhibition' (node ON decreases behavior)."),
    value: float = Field(default=1000000.0, allow_inf_nan=False, description="Behavior value applied when the node is active."),
    base_value: float = Field(default=0.0, allow_inf_nan=False, description="Baseline behavior value when the node is inactive."),
    smoothing: int = Field(default=0, ge=0, description="Smoothing level (0 = no smoothing)."),
    session_id: Optional[NonEmptyString] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Create an output link from a MaBoSS boolean node to a PhysiCell cell behaviour.
    Requires a PhysiBoSS model already attached to the cell type.
    Use `apply_physiboss_mutation()` afterward for genetic perturbations.
    Returns:
        str: Confirmation with link details and next step.
    """
    session = _require_configuration(session_id)
    _require_physiboss()
    
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
        
        result = f"**PhysiBoSS output:** {boolean_node} → {physicell_behavior}\n"
        result += f"- Action: {action}\n"
        result += f"- Active value: {value}\n"
        result += f"- Base value: {base_value}\n"
        result += f"- Smoothing: {smoothing}\n"
        result += f"- Progress: {session.get_progress_percentage():.0f}%\n"
        result += f"**Next step:** Use `apply_physiboss_mutation()` for genetic perturbations"
        
        return result
    except Exception as exc:
        raise RuntimeError(
            f"Could not add the PhysiBoSS output link: {exc}"
        ) from exc

@mcp.tool(annotations=_NON_IDEMPOTENT_TOOL)
@_session_locked
def apply_physiboss_mutation(
    cell_type: Annotated[NonEmptyString, Field(description="Name of an existing cell type with a PhysiBoSS model.")],
    node_name: Annotated[NonEmptyString, Field(description="MaBoSS boolean node name to fix (from the .bnd file).")],
    fixed_value: Annotated[MutationState, Field(description="Fixed node state: 0 (always OFF / loss-of-function) or 1 (always ON / gain-of-function).")],
    session_id: Optional[NonEmptyString] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Fix a MaBoSS boolean node to a constant value, simulating a genetic mutation.

    Requires a PhysiBoSS model already attached to the cell type.
    Call repeatedly to apply multiple mutations.

    Returns:
        str: Confirmation of the applied mutation.
    """
    session = _require_configuration(session_id)
    _require_physiboss()
    
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
        
        result = f"**Mutation applied:** {cell_type}.{node_name} = {fixed_value}\n"
        result += f"- Progress: {session.get_progress_percentage():.0f}%\n"
        result += f"**Next step:** Apply additional mutations or use `export_xml_configuration()` to finish."
        
        return result
    except Exception as exc:
        raise RuntimeError(
            f"Could not apply the PhysiBoSS mutation: {exc}"
        ) from exc

# ============================================================================
# UTILITY AND EXPORT TOOLS
# ============================================================================

@mcp.tool(annotations=_READ_ONLY_TOOL)
@_optional_session_locked
def get_simulation_summary(
    session_id: Optional[NonEmptyString] = Field(default=None, description="Session to query. Omit to use the active session."),
) -> Annotated[CallToolResult, PhysiCellWorkflowStatusResult]:
    """Return a comprehensive summary of the current simulation configuration.

    Shows session progress, configured components (substrates, cell types, rules,
    PhysiBoSS models), completed workflow steps, and export readiness.

    Returns:
        str: Markdown summary of current simulation state.
    """
    session = get_current_session(session_id)
    return structured_report(
        _format_simulation_summary(session),
        _workflow_status_payload(session),
    )


def _format_simulation_summary(session: Optional[SessionState]) -> str:
    """Format one already leased session without calling another MCP handler."""
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
    
    # Get rules count
    rules_count = 0
    try:
        rules_count = len(session.config.cell_rules.get_rules())
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

@mcp.tool(annotations=_IDEMPOTENT_TOOL)
@_session_locked
def export_xml_configuration(
    filename: NonEmptyString = Field(default="PhysiCell_settings.xml", description="Output filename for the XML configuration file."),
    session_id: Optional[NonEmptyString] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> Annotated[CallToolResult, PhysiCellXmlExportResult]:
    """Export the complete PhysiCell configuration to an XML file in the session artifact directory.

    The generated file can be passed directly to a PhysiCell executable.
    A simulation domain must have been created before calling this.

    Returns:
        CallToolResult: Export guidance plus validated XML artifact metadata.
    """
    session = _require_configuration(session_id)
    filename = _validate_export_filename(filename, ".xml")
    art_dir = get_artifact_dir(_SERVER_ROOT, session.session_id)
    out_path = str(safe_artifact_path(art_dir, filename))
    
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
        
        # Export XML configuration to session artifact directory
        xml_content = session.config.generate_xml()
        with open(out_path, 'w') as f:
            f.write(xml_content)

        xml_size = len(xml_content) // 1024

        result = f"## XML Configuration Exported\n\n"
        result += f"**File:** {out_path} ({xml_size}KB)\n"

        # Show XML modification info if loaded from XML
        if session.loaded_from_xml and session.original_xml_path:
            original_name = Path(session.original_xml_path).name
            source = "loaded"
            source_filename = original_name
            if session.xml_modification_count > 0:
                result += f"**Source:** Modified {session.xml_modification_count} times from {original_name}\n"
            else:
                result += f"**Source:** Exported from {original_name} (no modifications)\n"
        else:
            source = "created"
            source_filename = None
            result += f"**Source:** Created from scratch\n"

        result += f"**Substrates:** {len(substrates)} ({', '.join(substrates[:3]) if substrates else 'None'}{'...' if len(substrates) > 3 else ''})\n"
        result += f"**Cell Types:** {len(cell_types)} ({', '.join(cell_types[:3]) if cell_types else 'None'}{'...' if len(cell_types) > 3 else ''})\n"
        progress = session.get_progress_percentage()
        result += f"**Progress:** {progress:.0f}%\n\n"
        result += f"**Next step:** Copy to PhysiCell project directory and run:\n"
        result += f"```bash\n./myproject {filename}\n```"

        payload = PhysiCellXmlExportResult(
            server="PhysiCell",
            session_id=session.session_id,
            file=artifact_file_summary(
                out_path,
                session_id=session.session_id,
            ),
            source=source,
            source_filename=source_filename,
            modification_count=session.xml_modification_count,
            substrates=substrates,
            cell_types=cell_types,
            progress=progress,
        )
        return structured_report(result, payload)

    except Exception as exc:
        raise RuntimeError(
            f"Could not export the PhysiCell XML configuration: {exc}"
        ) from exc

@mcp.tool(annotations=_IDEMPOTENT_TOOL)
@_session_locked
def export_cell_rules_csv(
    filename: NonEmptyString = Field(default="cell_rules.csv", description="Output filename for the cell rules CSV file."),
    session_id: Optional[NonEmptyString] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> Annotated[CallToolResult, PhysiCellRulesExportResult]:
    """Export cell signal-behaviour rules to a CSV file in the session artifact directory.

    The generated file is used alongside `PhysiCell_settings.xml` by the PhysiCell executable.
    At least one rule must exist (added via `add_single_cell_rule()`).

    Returns:
        CallToolResult: Export guidance plus validated CSV artifact metadata.
    """
    session = _require_configuration(session_id)
    filename = _validate_export_filename(filename, ".csv")
    art_dir = get_artifact_dir(_SERVER_ROOT, session.session_id)
    out_path = str(safe_artifact_path(art_dir, filename))

    try:
        rule_count = len(session.config.cell_rules.get_rules())
    except Exception as exc:
        raise RuntimeError(
            f"Could not inspect the configured cell rules: {exc}"
        ) from exc

    if rule_count == 0:
        raise RuntimeError(
            "No cell rules are available to export. Use "
            "`add_single_cell_rule()` first."
        )

    try:
        # Export using the CellRulesModule API to the session artifact directory
        session.config.cell_rules.generate_csv(out_path)

        # Register the ruleset so the XML references the correct relative path.
        # PhysiCell standard layout: XML in project root, CSV in ./config/
        session.config.cell_rules.add_ruleset(
            name="default",
            folder=out_path,
            filename=filename,
            enabled=True
        )

        result = f"## Cell Rules CSV Exported\n\n"
        result += f"**File:** {out_path}\n"
        result += f"**XML path:** ./config/{filename} (enabled)\n"
        result += f"**Rules:** {rule_count}\n"
        progress = session.get_progress_percentage()
        result += f"**Progress:** {progress:.0f}%\n\n"
        result += f"**Next step:** Copy to PhysiCell project directory alongside XML configuration"

        payload = PhysiCellRulesExportResult(
            server="PhysiCell",
            session_id=session.session_id,
            file=artifact_file_summary(
                out_path,
                session_id=session.session_id,
            ),
            xml_reference=f"./config/{filename}",
            enabled=True,
            rule_count=rule_count,
            progress=progress,
        )
        return structured_report(result, payload)

    except Exception as exc:
        raise RuntimeError(
            f"Could not export the cell rules CSV: {exc}"
        ) from exc

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

@mcp.tool(annotations=_READ_ONLY_TOOL)
def list_generated_files(
    session_id: Optional[NonEmptyString] = Field(default=None, description="Session to query. Omit to use the active session. Pass 'all' to list files across every session."),
) -> Annotated[CallToolResult, PhysiCellArtifactFileListResult]:
    """List PhysiCell artifact files (XML/CSV) for the active session.

    Returns:
        CallToolResult: Grouped text plus validated artifact file metadata.
    """
    if session_id == "all":
        files = list_artifacts(_SERVER_ROOT, session_id=None)
        resolved_session_id = None
        scope = "all"
    else:
        with session_manager.session_scope(session_id) as session:
            files = list_artifacts(
                _SERVER_ROOT,
                session_id=session.session_id,
            )
            resolved_session_id = session.session_id
            scope = "session"

    xml_files = [f for f in files if str(f).endswith(".xml")]
    csv_files = [f for f in files if str(f).endswith(".csv")]

    result = "## Generated Artifact Files\n\n"

    if xml_files:
        result += "**XML files:**\n"
        for f in xml_files:
            result += f"- {f}\n"
        result += "\n"

    if csv_files:
        result += "**CSV files:**\n"
        for f in csv_files:
            result += f"- {f}\n"
        result += "\n"

    if not xml_files and not csv_files:
        result += "No PhysiCell artifact files found."

    listed_files = xml_files + csv_files
    payload = PhysiCellArtifactFileListResult(
        server="PhysiCell",
        scope=scope,
        session_id=resolved_session_id,
        count=len(listed_files),
        files=[
            artifact_file_summary(
                file_path,
                session_id=(
                    resolved_session_id
                    if resolved_session_id is not None
                    else file_path.parent.name
                ),
            )
            for file_path in listed_files
        ],
    )
    return structured_report(result, payload)


@mcp.tool(annotations=_IDEMPOTENT_DESTRUCTIVE_TOOL)
@_session_locked
def clean_generated_files(
    session_id: Optional[NonEmptyString] = Field(default=None, description="Session to clean. Omit to use the active session."),
) -> Annotated[CallToolResult, PhysiCellArtifactCleanupResult]:
    """Remove all artifact files (XML, CSV, etc.) for the active session.

    Returns:
        CallToolResult: Cleanup confirmation plus the validated removal count.
    """
    session = _require_session(session_id)

    try:
        count = clean_artifacts(_SERVER_ROOT, session.session_id)
        text = (
            f"**Cleaned {count} artifact file(s)** for session "
            f"{session.session_id[:8]}..."
        )
        payload = PhysiCellArtifactCleanupResult(
            server="PhysiCell",
            session_id=session.session_id,
            removed_count=count,
        )
        return structured_report(text, payload)
    except Exception as exc:
        raise RuntimeError(
            f"Could not clean generated PhysiCell files: {exc}"
        ) from exc

@mcp.tool(annotations=_READ_ONLY_TOOL)
def get_help() -> str:
    """Return the same workflow guide exposed as a prompt and resource.

    Returns:
        str: Markdown-formatted help guide.
    """
    return PHYSICELL_AGENT_MANUAL

if __name__ == "__main__":
    mcp.run()
