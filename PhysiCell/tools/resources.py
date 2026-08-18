"""Configuration resource registrations for PhysiCell sessions."""

from mcp.server.mcpserver.exceptions import ResourceNotFoundError

from ..app import mcp
from ..locking import resource_session_locked
from ..services.resource_views import (
    format_cell_rules_resource,
    format_cell_types_resource,
    format_domain_resource,
    format_physiboss_resource,
    format_substrates_resource,
)
from ..session_manager import SessionState, get_current_session


def _require_resource_configuration(session_id: str) -> SessionState:
    """Return configured state under the resource session lease."""
    session = get_current_session(session_id)
    if session is None:
        raise ResourceNotFoundError(
            f"PhysiCell session not found: {session_id}"
        )
    if session.config is None:
        raise ResourceNotFoundError(
            "No PhysiCell configuration in this session. "
            "Call create_simulation_domain first."
        )
    return session


@mcp.resource(
    uri="physicell://session/{session_id}/domain",
    name="PhysiCell Domain",
    title="PhysiCell simulation domain",
    description="Spatial domain, mesh, duration, and simulation time steps.",
    mime_type="text/markdown",
)
@resource_session_locked
def physicell_domain_resource(session_id: str) -> str:
    """Return the configured domain for an existing session."""
    return format_domain_resource(_require_resource_configuration(session_id))


@mcp.resource(
    uri="physicell://session/{session_id}/substrates",
    name="PhysiCell Substrates",
    title="PhysiCell substrates",
    description="Configured substrates and their principal diffusion values.",
    mime_type="text/markdown",
)
@resource_session_locked
def physicell_substrates_resource(session_id: str) -> str:
    """Return configured substrates for an existing session."""
    return format_substrates_resource(
        _require_resource_configuration(session_id)
    )


@mcp.resource(
    uri="physicell://session/{session_id}/cell_types",
    name="PhysiCell Cell Types",
    title="PhysiCell cell types",
    description="Configured cell types and their principal phenotype values.",
    mime_type="text/markdown",
)
@resource_session_locked
def physicell_cell_types_resource(session_id: str) -> str:
    """Return configured cell types for an existing session."""
    return format_cell_types_resource(
        _require_resource_configuration(session_id)
    )


@mcp.resource(
    uri="physicell://session/{session_id}/cell_rules",
    name="PhysiCell Cell Rules",
    title="PhysiCell cell rules",
    description="Configured signal-behavior rules and ruleset references.",
    mime_type="text/markdown",
)
@resource_session_locked
def physicell_cell_rules_resource(session_id: str) -> str:
    """Return configured cell rules for an existing session."""
    return format_cell_rules_resource(
        _require_resource_configuration(session_id)
    )


@mcp.resource(
    uri="physicell://session/{session_id}/physiboss",
    name="PhysiBoSS Integration",
    title="PhysiBoSS integration",
    description="MaBoSS context and intracellular model configuration.",
    mime_type="text/markdown",
)
@resource_session_locked
def physicell_physiboss_resource(session_id: str) -> str:
    """Return PhysiBoSS integration state for an existing session."""
    return format_physiboss_resource(
        _require_resource_configuration(session_id)
    )
