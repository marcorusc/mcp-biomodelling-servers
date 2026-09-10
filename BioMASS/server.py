"""Typed MCP tools for evidence-backed ODE modelling."""

import base64
import json
import shutil
from pathlib import Path
from typing import Annotated, Literal

from mcp.types import CallToolResult, ImageContent
from pydantic import Field

from mcp_biomodelling_servers import __version__
from mcp_biomodelling_servers.artifact_manager import (
    list_artifact_sessions as disk_sessions,
)
from mcp_biomodelling_servers.ode_handoff import read_ode_handoff
from mcp_biomodelling_servers.structured_outputs import (
    ArtifactSessionSummary,
    artifact_file_summary,
    structured_report,
)

from .app import mcp
from .contracts import (
    DELETE,
    READ_ONLY,
    WRITE,
    EvidenceRecord,
    GraphOptions,
    LineEvidence,
    ModelConfiguration,
    NonEmpty,
    ReactionRecord,
    SimulationScenario,
    Timeout,
)
from .outputs import (
    BioMASSArtifactCleanupResult,
    BioMASSArtifactFileListResult,
    BioMASSArtifactSessionListResult,
    BioMASSJobResult,
    BioMASSSessionListResult,
    BioMASSSessionSummary,
    BioMASSStateResult,
    BioMASSValidationResult,
)
from .services import artifacts
from .services.authoring import validate_text
from .services.models import coverage, edited, render
from .session_manager import session_manager
from .tools import guidance  # noqa: F401
from .tools.registration import resource, tool

SessionID = Annotated[
    str | None, Field(description="BioMASS session ID; omit to use the active default.")
]
RevisionID = Annotated[
    str | None,
    Field(
        description="Generated revision ID; omit for the current revision. Edits require regeneration or explicit selection of an older revision."
    ),
]
JobTimeout = Annotated[
    Timeout, Field(description="Worker timeout in seconds, from 1 to 300; default 60.")
]


def state(sess) -> BioMASSStateResult:
    return BioMASSStateResult(
        session_id=sess.session_id,
        document=sess.document.model_copy(deep=True),
        coverage=coverage(sess.document),
    )


def selected(sess, revision):
    revision = revision or sess.document.current_revision
    if revision is None:
        raise ValueError(
            "Generate the edited model or explicitly select a retained revision first."
        )
    path = artifacts.revision_path(
        session_manager.directory(sess.session_id), revision, sess.document.revisions
    )
    snapshot = json.loads((path / "snapshot.json").read_text())
    return revision, path, snapshot


def outcome(sess, revision, operation, path, details, model_coverage):
    output_files = (
        artifacts.summaries(path, sess.session_id)
        if path.is_dir()
        else [artifact_file_summary(path, session_id=sess.session_id)]
    )
    return BioMASSJobResult(
        session_id=sess.session_id,
        revision=revision,
        operation=operation,
        files=output_files,
        details=details,
        coverage=model_coverage,
    )


@tool(annotations=WRITE, structured_output=True)
def create_session(
    label: Annotated[
        str | None, Field(description="Optional human-readable model label.")
    ] = None,
    set_as_default: Annotated[
        bool, Field(description="Make the created session the active default.")
    ] = True,
) -> BioMASSStateResult:
    """Create an isolated, durable BioMASS authoring session."""
    return state(session_manager.create(label, set_as_default))


@tool(annotations=READ_ONLY, structured_output=True)
def list_sessions() -> BioMASSSessionListResult:
    """List active sessions and current generated revisions."""
    with session_manager.lock:
        ids = list(session_manager.sessions)
    sessions = []
    for sid in ids:
        try:
            with session_manager.use(sid) as sess:
                sessions.append(
                    BioMASSSessionSummary(
                        session_id=sid,
                        created_at=sess.created_at,
                        last_accessed=sess.last_accessed,
                        is_default=sid == session_manager.default,
                        mode=sess.document.mode,
                        current_revision=sess.document.current_revision,
                    )
                )
        except ValueError:
            continue  # Session retired during this read-only snapshot.
    return BioMASSSessionListResult(count=len(sessions), sessions=sessions)


@tool(annotations=DELETE, structured_output=True)
def close_session(session_id: SessionID = None) -> str:
    """Close a session, retaining its authoring snapshot and artifacts on disk."""
    return f"Closed {session_manager.close(session_id)}; artifacts retained."


@tool(annotations=WRITE, structured_output=True)
def restore_session(
    session_id: Annotated[
        NonEmpty,
        Field(
            description="Complete UUID of a closed or previous-process artifact session."
        ),
    ],
) -> BioMASSStateResult:
    """Restore authoring state from a previously created BioMASS session."""
    return state(session_manager.restore(session_id))


@tool(annotations=WRITE, structured_output=True)
def import_neko_handoff(
    manifest_path: Annotated[
        NonEmpty, Field(description="Path to a NeKo-to-BioMASS handoff manifest.")
    ],
    session_id: SessionID = None,
) -> BioMASSStateResult:
    """Verify and import a NeKo network and its complete reference provenance."""
    with session_manager.use(session_id) as sess:
        if sess.document.mode != "empty":
            raise ValueError("Import a network into a fresh session.")
        manifest, network = read_ode_handoff(manifest_path)
        document = sess.document.model_copy(deep=True)
        document.mode, document.network, document.upstream = (
            "records",
            network,
            manifest,
        )
        document.biological_context = manifest.biological_context
        session_manager.save(sess, edited(document))
        return state(sess)


@tool(annotations=WRITE, structured_output=True)
def import_text(
    text: Annotated[
        NonEmpty, Field(description="Complete Text2Model document; preserved verbatim.")
    ],
    line_evidence: Annotated[
        list[LineEvidence] | None,
        Field(
            description="Complete evidence/assumption mapping for document lines; omitted clears old line links."
        ),
    ] = None,
    biological_context: Annotated[
        str | None,
        Field(description="Optional biological context for standalone modelling."),
    ] = None,
    session_id: SessionID = None,
) -> BioMASSStateResult:
    """Import or replace standalone text and its line provenance atomically.

    Read docs://biomass/reaction_syntax and docs://biomass/authoring_examples
    before writing text. Only the documented Text2Model subset is supported.
    """
    validate_text(text)
    with session_manager.use(session_id) as sess:
        if sess.document.mode == "records":
            raise ValueError(
                "Use reaction records for a NeKo model; standalone documents require another session."
            )
        document = sess.document.model_copy(deep=True)
        document.mode, document.text = "document", text
        document.line_evidence = line_evidence or []
        document.biological_context = biological_context
        # A new document can change generated numerical symbol identities.
        document.configuration = ModelConfiguration()
        session_manager.save(sess, edited(document))
        return state(sess)


@tool(annotations=WRITE, structured_output=True)
def set_evidence(
    records: Annotated[
        list[EvidenceRecord],
        Field(
            description="Evidence records to upsert by evidence_id; all other records are preserved."
        ),
    ],
    session_id: SessionID = None,
) -> BioMASSStateResult:
    """Store agent-extracted evidence, including contradictory or inaccessible sources."""
    with session_manager.use(session_id) as sess:
        document = sess.document.model_copy(deep=True)
        if len({r.evidence_id for r in records}) != len(records):
            raise ValueError("Evidence IDs must be unique in an update.")
        document.evidence.update({r.evidence_id: r for r in records})
        session_manager.save(sess, edited(document))
        return state(sess)


@tool(annotations=WRITE, structured_output=True)
def set_reactions(
    reactions: Annotated[
        list[ReactionRecord],
        Field(
            description="Complete ordered replacement of reaction records; use stable IDs and share_parameters_with."
        ),
    ],
    session_id: SessionID = None,
) -> BioMASSStateResult:
    """Replace the COMPLETE ordered reaction list, retaining prior records explicitly.

    Before authoring, read docs://biomass/reaction_syntax,
    docs://biomass/authoring_examples, and docs://biomass/network_to_reactions.
    Signed edges and PMID identifiers alone do not justify mechanisms or kinetics.
    """
    with session_manager.use(session_id) as sess:
        if sess.document.mode != "records":
            raise ValueError("Import a NeKo handoff before authoring reaction records.")
        document = sess.document.model_copy(deep=True)
        if document.reactions != reactions:
            # Reaction edits can change generated kfN and species identities.
            # Preserve observables/time settings but invalidate numerical overrides.
            document.configuration.parameters = {}
            document.configuration.initials = {}
            document.configuration.conditions = []
        document.reactions = reactions
        if reactions:
            render(document)
        session_manager.save(sess, edited(document))
        return state(sess)


@tool(annotations=WRITE, structured_output=True)
def configure_model(
    configuration: Annotated[
        ModelConfiguration,
        Field(
            description="Complete replacement configuration. In document mode, structural directives belong in the text."
        ),
    ],
    session_id: SessionID = None,
) -> BioMASSStateResult:
    """Configure observables, numerical defaults, conditions, time, and unit provenance."""
    with session_manager.use(session_id) as sess:
        document = sess.document.model_copy(deep=True)
        if document.mode == "empty":
            raise ValueError("Import a model first.")
        if document.mode == "document" and (
            configuration.observables
            or configuration.time_span
            or configuration.conditions
        ):
            raise ValueError(
                "Set observables, conditions and @sim tspan in the standalone text."
            )
        document.configuration = configuration
        render(document)
        session_manager.save(sess, edited(document))
        return state(sess)


@tool(annotations=READ_ONLY, structured_output=True)
def inspect_model(session_id: SessionID = None) -> BioMASSStateResult:
    """Inspect authoring state, evidence, retained revisions, and edge coverage."""
    with session_manager.use(session_id) as sess:
        return state(sess)


@tool(annotations=WRITE, structured_output=True)
def validate_model(
    check_generation: Annotated[
        bool,
        Field(
            description="Also compile and import in a disposable worker; no retained model revision."
        ),
    ] = False,
    timeout_seconds: JobTimeout = 60,
    session_id: SessionID = None,
) -> BioMASSValidationResult:
    """Report syntax and optional generation validity independently of scientific validity."""
    with session_manager.use(session_id) as sess:
        syntax, generated, issues = False, None, []
        try:
            text, mapping = render(sess.document)
            syntax = True
            if check_generation:
                request = {
                    "operation": "generate",
                    "text": text,
                    "line_mapping": mapping,
                    "configuration": sess.document.configuration.model_dump(
                        mode="json"
                    ),
                }
                try:
                    path, _ = artifacts.job(
                        session_manager.directory(sess.session_id),
                        request,
                        timeout_seconds,
                    )
                    shutil.rmtree(path)
                    generated = True
                except (RuntimeError, ValueError, TimeoutError) as exc:
                    generated = False
                    issues.append(str(exc))
        except (ValueError, SyntaxError) as exc:
            issues.append(str(exc))
        return BioMASSValidationResult(
            session_id=sess.session_id,
            syntax_valid=syntax,
            generation_valid=generated,
            issues=issues,
            coverage=coverage(sess.document),
        )


@tool(annotations=WRITE, structured_output=True)
def generate_model(
    timeout_seconds: JobTimeout = 60, session_id: SessionID = None
) -> BioMASSJobResult:
    """Generate a new immutable BioMASS revision from the current authoring state."""
    with session_manager.use(session_id) as sess:
        text, mapping = render(sess.document)
        model_coverage = coverage(sess.document)
        snapshot = {
            "document": sess.document.model_dump(mode="json"),
            "coverage": model_coverage.model_dump(),
            "mcp_package_version": __version__,
        }
        request = {
            "operation": "generate",
            "text": text,
            "line_mapping": mapping,
            "configuration": sess.document.configuration.model_dump(mode="json"),
        }
        path, details = artifacts.job(
            session_manager.directory(sess.session_id),
            request,
            timeout_seconds,
            snapshot,
        )
        document = sess.document.model_copy(deep=True)
        document.revisions.append(path.name)
        document.current_revision = path.name
        try:
            session_manager.save(sess, document)
        except (OSError, ValueError):
            shutil.rmtree(path)
            raise
        return outcome(sess, path.name, "generate", path, details, model_coverage)


def graph_job(sess, revision, fmt, options, timeout):
    revision, path, snapshot = selected(sess, revision)
    request = json.loads((path / "request.json").read_text())
    request.update(
        operation="graph",
        revision=revision,
        format=fmt,
        graph_options=options.model_dump(),
    )
    destination, details = artifacts.job(
        session_manager.directory(sess.session_id), request, timeout
    )
    return outcome(sess, revision, "graph", destination, details, snapshot["coverage"])


@tool(annotations=WRITE, structured_output=True)
def visualize_model(
    format: Annotated[
        Literal["png", "svg", "html"],
        Field(
            description="Graph output format; HTML is interactive and does not open a browser."
        ),
    ] = "png",
    options: Annotated[
        GraphOptions | None,
        Field(description="Bounded layout and interactive-control options."),
    ] = None,
    revision: RevisionID = None,
    timeout_seconds: JobTimeout = 60,
    session_id: SessionID = None,
) -> Annotated[CallToolResult, BioMASSJobResult]:
    """Visualize a generated species graph without requiring simulation or calibration."""
    with session_manager.use(session_id) as sess:
        result = graph_job(
            sess, revision, format, options or GraphOptions(), timeout_seconds
        )
        response = structured_report(
            f"Graph for {result.revision}. {result.details['interpretation_limits']}",
            result,
        )
        if format == "png":
            graph_file = next(f for f in result.files if f.name == "graph.png")
            response.content.append(
                ImageContent(
                    type="image",
                    mime_type="image/png",
                    data=base64.b64encode(Path(graph_file.path).read_bytes()).decode(),
                )
            )
        return response


@tool(annotations=WRITE, structured_output=True)
def export_model_graph(
    revision: RevisionID = None,
    timeout_seconds: JobTimeout = 60,
    session_id: SessionID = None,
) -> BioMASSJobResult:
    """Export BioMASS's directed species projection as DOT."""
    with session_manager.use(session_id) as sess:
        return graph_job(sess, revision, "dot", GraphOptions(), timeout_seconds)


@tool(annotations=WRITE, structured_output=True)
def run_simulation(
    scenario: Annotated[
        SimulationScenario,
        Field(
            description="Named numerical scenario; explicitly opt into placeholders if any remain."
        ),
    ],
    revision: RevisionID = None,
    timeout_seconds: JobTimeout = 60,
    session_id: SessionID = None,
) -> BioMASSJobResult:
    """Run bounded exploratory simulation, recording every condition's numerical values."""
    with session_manager.use(session_id) as sess:
        revision, path, snapshot = selected(sess, revision)
        evidence = snapshot["document"]["evidence"]
        if any(
            set(q.evidence_ids) - evidence.keys()
            for q in [*scenario.parameters.values(), *scenario.initials.values()]
        ):
            raise ValueError(
                "Scenario refers to evidence absent from the selected revision."
            )
        request = json.loads((path / "request.json").read_text())
        request.update(
            operation="simulate",
            revision=revision,
            scenario=scenario.model_dump(mode="json"),
        )
        request["configuration"]["parameters"].update(
            {k: v.model_dump(mode="json") for k, v in scenario.parameters.items()}
        )
        request["configuration"]["initials"].update(
            {k: v.model_dump(mode="json") for k, v in scenario.initials.items()}
        )
        destination, details = artifacts.job(
            session_manager.directory(sess.session_id), request, timeout_seconds
        )
        return outcome(
            sess, revision, "simulate", destination, details, snapshot["coverage"]
        )


@tool(annotations=WRITE, structured_output=True)
def export_model_bundle(
    revision: RevisionID = None, session_id: SessionID = None
) -> BioMASSJobResult:
    """Export a selected revision, evidence, graphs, runs, and reproduction script."""
    with session_manager.use(session_id) as sess:
        revision, path, snapshot = selected(sess, revision)
        archive = artifacts.export_bundle(
            session_manager.directory(sess.session_id), path, revision
        )
        return outcome(
            sess,
            revision,
            "export",
            archive,
            {"bundle": str(archive)},
            snapshot["coverage"],
        )


@tool(annotations=READ_ONLY, structured_output=True)
def list_generated_files(session_id: SessionID = None) -> BioMASSArtifactFileListResult:
    """List all nested session artifacts, including snapshots and diagnostics."""
    with session_manager.use(session_id) as sess:
        files = artifacts.summaries(
            session_manager.directory(sess.session_id), sess.session_id
        )
        return BioMASSArtifactFileListResult(
            session_id=sess.session_id, count=len(files), files=files
        )


@tool(annotations=READ_ONLY, structured_output=True)
def list_artifact_sessions() -> BioMASSArtifactSessionListResult:
    """Discover artifact sessions saved on disk across server restarts."""
    sessions = [
        ArtifactSessionSummary(**s) for s in disk_sessions(session_manager.root)
    ]
    return BioMASSArtifactSessionListResult(count=len(sessions), sessions=sessions)


@tool(annotations=DELETE, structured_output=True)
def clean_generated_files(session_id: SessionID = None) -> BioMASSArtifactCleanupResult:
    """Delete generated artifacts while preserving authoring state and source lineage."""
    with session_manager.use(session_id) as sess:
        directory = session_manager.directory(sess.session_id)
        artifacts.files(directory)  # Reject symlinked trees before any deletion.
        removed = 0
        for path in directory.iterdir():
            if path.name in ("session.json", "session_meta.json"):
                continue
            if path.is_dir():
                removed += len(artifacts.files(path))
                shutil.rmtree(path)
            else:
                path.unlink()
                removed += 1
        document = sess.document.model_copy(deep=True)
        document.current_revision, document.revisions = None, []
        session_manager.save(sess, document)
        return BioMASSArtifactCleanupResult(
            session_id=sess.session_id, removed_count=removed
        )


@resource("biomass://session/{session_id}/model", mime_type="application/json")
def model_resource(session_id: str) -> str:
    """Read-only authoring state and evidence coverage."""
    return inspect_model(session_id).model_dump_json()


@resource("biomass://session/{session_id}/files", mime_type="application/json")
def files_resource(session_id: str) -> str:
    """Read-only recursive artifact inventory."""
    return list_generated_files(session_id).model_dump_json()


@resource("biomass://session/{session_id}/evidence", mime_type="application/json")
def evidence_resource(session_id: str) -> str:
    """Read-only source evidence and line provenance."""
    with session_manager.use(session_id) as sess:
        return json.dumps(
            {
                "evidence": {
                    k: v.model_dump() for k, v in sess.document.evidence.items()
                },
                "line_evidence": [v.model_dump() for v in sess.document.line_evidence],
            }
        )


@resource("biomass://session/{session_id}/coverage", mime_type="application/json")
def coverage_resource(session_id: str) -> str:
    """Read-only supported, assumed, and unresolved edge coverage."""
    with session_manager.use(session_id) as sess:
        return coverage(sess.document).model_dump_json()


@resource(
    "biomass://session/{session_id}/revision/{revision}", mime_type="application/json"
)
def revision_resource(session_id: str, revision: str) -> str:
    """Read verified equations, species, numerical origins, and revision provenance."""
    with session_manager.use(session_id) as sess:
        _, path, snapshot = selected(sess, revision)
        return json.dumps(
            {
                "summary": json.loads((path / "model_summary.json").read_text()),
                "snapshot": snapshot,
            }
        )


if __name__ == "__main__":
    mcp.run()
