"""Offline workflow, syntax, and authoring references for MCP clients."""

from pathlib import Path

from ..app import mcp
from ..guidance import BIOMASS_AGENT_MANUAL

DOCS_DIRECTORY = Path(__file__).resolve().parents[1] / "docs"


@mcp.resource(
    "docs://biomass/agent_manual",
    name="BioMASS Agent Operations Manual",
    mime_type="text/markdown",
)
def biomass_agent_manual_resource() -> str:
    return BIOMASS_AGENT_MANUAL


@mcp.resource(
    "docs://biomass/reaction_syntax",
    name="BioMASS 0.14 Reaction Syntax and Kinetics",
    description="Read before authoring: canonical statements, generated kinetics, parameters, sharing, and the server's supported Text2Model subset.",
    mime_type="text/markdown",
)
def biomass_reaction_syntax_resource() -> str:
    return (DOCS_DIRECTORY / "reaction_syntax.md").read_text(encoding="utf-8")


@mcp.resource(
    "docs://biomass/authoring_examples",
    name="BioMASS Tested Authoring Examples",
    description="Complete tool argument examples for evidence, reaction records, configuration, standalone line provenance, and explicit simulation scenarios.",
    mime_type="text/markdown",
)
def biomass_authoring_examples_resource() -> str:
    return (DOCS_DIRECTORY / "authoring_examples.md").read_text(encoding="utf-8")


@mcp.resource(
    "docs://biomass/network_to_reactions",
    name="BioMASS Network-to-Reaction Evidence Guide",
    description="Map referenced edges to justified mechanisms, distinguish assumptions from evidence, retain unresolved edges, and avoid overwriting prior records.",
    mime_type="text/markdown",
)
def biomass_network_to_reactions_resource() -> str:
    return (DOCS_DIRECTORY / "network_to_reactions.md").read_text(encoding="utf-8")


@mcp.resource(
    "docs://biomass/model_editing",
    name="BioMASS Conversational Construction and Model Editing",
    description="Read before build_reactions: template participants, custom kinetics, file imports, versioned previews, dependency repairs, and optional metadata.",
    mime_type="text/markdown",
)
def biomass_model_editing_resource() -> str:
    return (DOCS_DIRECTORY / "model_editing.md").read_text(encoding="utf-8")


@mcp.prompt(name="biomass_workflow_prompt")
def biomass_workflow_prompt() -> str:
    """Guide evidence-backed ODE construction, visualization, and simulation."""
    return BIOMASS_AGENT_MANUAL
