import logging
import os
import tempfile
from pathlib import Path
from typing import Annotated

import io
import json

import anyio
import matplotlib.pyplot as plt
import pandas as pd
from mcp.server.mcpserver import Context, Image
from mcp.types import CallToolResult, TextContent
from pydantic import Field

from mcp_biomodelling_servers.artifact_manager import (
    clean_artifacts,
    get_artifact_dir,
    list_artifacts,
    safe_artifact_path,
    write_session_meta,
)
from mcp_biomodelling_servers.artifact_manager import (
    list_artifact_sessions as _list_artifact_sessions_on_disk,
)
from mcp_biomodelling_servers import __version__
from mcp_biomodelling_servers.handoff import (
    HandoffNetwork,
    HandoffPackage,
    HandoffProvenance,
    MaBoSSHandoffExportResult,
    MaBoSSHandoffImportResult,
    MaBoSSSimulationHandoff,
    MaBoSSToPhysiCellHandoffManifest,
    NeKoToMaBoSSHandoffManifest,
    PhysiCellTarget,
    bnet_node_names,
    handoff_artifact,
    load_handoff_manifest,
    verify_handoff_artifact,
    verify_handoff_manifest,
    write_handoff_manifest,
)
from mcp_biomodelling_servers.import_lineage import (
    SourceExport,
    capture_import,
    write_source_export,
    write_source_import,
)
from mcp_biomodelling_servers.structured_outputs import (
    ArtifactSessionSummary,
    MaBoSSArtifactCleanupResult,
    MaBoSSArtifactFileListResult,
    MaBoSSArtifactSessionListResult,
    MaBoSSBnetConversionResult,
    MaBoSSModelExportResult,
    MaBoSSSessionListResult,
    MaBoSSSessionSummary,
    artifact_file_summary,
    structured_report,
)
from .app import mcp, tool
from .contracts import (
    DESTRUCTIVE as _DESTRUCTIVE_TOOL,
    IDEMPOTENT as _IDEMPOTENT_TOOL,
    IDEMPOTENT_DESTRUCTIVE as _IDEMPOTENT_DESTRUCTIVE_TOOL,
    NON_IDEMPOTENT as _NON_IDEMPOTENT_TOOL,
    READ_ONLY as _READ_ONLY_TOOL,
    HandoffArtifactPrefix,
    InitialStateProbabilitySpecification,
    MaBoSSParameterUpdates,
    MutationState,
    NonEmptyString,
)
from .guidance import MABOSS_AGENT_MANUAL, MABOSS_SERVER_INSTRUCTIONS
from .external import pymaboss as maboss
from .locking import (
    resource_session_locked as _resource_session_locked,
    session_locked as _session_locked,
)
from .services.artifacts import (
    link_artifact_without_overwrite as _link_artifact_without_overwrite,
    preserve_simulation_result as _preserve_simulation_result,
    publish_replacing_artifacts as _publish_replacing_artifacts,
    require_unused_artifact_paths as _require_unused_artifact_paths,
    rollback_artifacts as _rollback_artifacts,
)
from .services.handoff import (
    handoff_parameters as _handoff_parameters,
    maboss_package_version as _maboss_package_version,
)
from .services.formatting import clean_for_markdown
from .services.normalization import (
    initial_state_groups as _initial_state_groups,
    logical_rule_records as _logical_rule_records,
    mutation_records as _mutation_records,
    normalize_initial_state_probabilities as _normalize_initial_state_probabilities,
    parameter_records as _parameter_records,
    scientific_table as _scientific_table,
)
from .scientific_outputs import (
    MaBoSSInitialStateResult,
    MaBoSSLogicalRulesResult,
    MaBoSSMutationListResult,
    MaBoSSMutationRecord,
    MaBoSSMutationSimulationResult,
    MaBoSSNodeListResult,
    MaBoSSParameterResult,
    MaBoSSSimulationResult,
    MaBoSSSimulationRunResult,
    MaBoSSTrajectoryPlotResult,
)
from .session_manager import ensure_session, session_manager
from .tools.guidance import (
    maboss_agent_manual_resource,
    maboss_workflow_prompt,
)
from .tools.resources import (
    resource_initial_state,
    resource_logical_rules,
    resource_mutations,
    resource_network_nodes,
    resource_parameters,
    resource_simulation_result,
)

__all__ = [
    "MABOSS_AGENT_MANUAL",
    "MABOSS_SERVER_INSTRUCTIONS",
    "maboss_agent_manual_resource",
    "maboss_workflow_prompt",
    "mcp",
    "resource_initial_state",
    "resource_logical_rules",
    "resource_mutations",
    "resource_network_nodes",
    "resource_parameters",
    "resource_simulation_result",
]

from .services.runs import effective_configuration, load_run, run_table, record_failure
from .services.model import RateEdit, edit_rates, inspect_nodes, model_revision, print_bnd_with_readouts, read_readouts
from .tools.curation import edit_maboss_readouts  # noqa: F401

logger = logging.getLogger(__name__)

_SERVER_ROOT = Path(__file__).parent


# ---------------------------------------------------------------------------
# Read-only resources (URI templates — no side effects)
# ---------------------------------------------------------------------------


@mcp.resource(
    uri="maboss://session/{session_id}/files",
    name="Artifact Files",
    title="MaBoSS artifact files",
    description="List of artifact files (BND, CFG, PNG, …) generated for a session.",
    mime_type="text/markdown",
)
@_resource_session_locked
def resource_generated_files(session_id: str) -> str:
    """Return a Markdown list of artifact files for the given session."""
    sess = ensure_session(session_id)
    files = list_artifacts(_SERVER_ROOT, session_id=sess.session_id)
    if not files:
        return "No artifact files found for this session."
    return "## Artifact files\n\n" + "\n".join(f"- {f}" for f in files)


# ---------------------------------------------------------------------------
# Session management tools
# ---------------------------------------------------------------------------

@tool(
    title="Create MaBoSS session",
    annotations=_NON_IDEMPOTENT_TOOL,
    structured_output=True,
)
def create_session(
    set_as_default: bool = Field(
        default=True,
        description="When True (default), the new session becomes the active default for subsequent calls.",
    ),
    label: str | None = Field(
        default=None,
        description="Optional human-readable label (e.g. 'TP53-MYC Boolean run'). Stored on disk so the session can be rediscovered after a server restart.",
    ),
) -> str:
    """Create a new MaBoSS session.

    Returns the session ID (UUID) that must be passed to pipeline tools when running
    multiple independent simulations in parallel.
    """
    with session_manager.create_session_scope(
        set_as_default=set_as_default,
    ) as session:
        sid = session.session_id
        write_session_meta(_SERVER_ROOT, sid, server_name="MaBoSS", label=label)
    label_info = f" ({label})" if label else ""
    return f"Session created: {sid}{label_info}" + (" (set as default)" if set_as_default else "")


@tool(
    title="List MaBoSS sessions",
    annotations=_READ_ONLY_TOOL,
    structured_output=True,
)
def list_sessions() -> Annotated[CallToolResult, MaBoSSSessionListResult]:
    """List all active MaBoSS sessions with their simulation and result status."""
    sessions = session_manager.list_sessions()
    payload = MaBoSSSessionListResult(
        server="MaBoSS",
        count=len(sessions),
        sessions=[
            MaBoSSSessionSummary(
                session_id=sid,
                created_at=info["created_at"],
                last_accessed=info["last_accessed"],
                is_default=info["is_default"],
                has_simulation=info["has_simulation"],
                has_result=info["has_result"],
                bnd_path=info["bnd_path"],
                cfg_path=info["cfg_path"],
                upstream_neko_manifest_path=info[
                    "upstream_neko_manifest_path"
                ],
            )
            for sid, info in sessions.items()
        ],
    )
    if not sessions:
        return structured_report(
            "No active sessions. Call create_session() to start one.",
            payload,
        )
    lines = ["## MaBoSS Sessions\n"]
    for sid, info in sessions.items():
        default_marker = " **(default)**" if info["is_default"] else ""
        has_sim = "✓" if info["has_simulation"] else "✗"
        has_res = "✓" if info["has_result"] else "✗"
        lines.append(
            f"- **{sid}**{default_marker}: sim={has_sim}  result={has_res}  "
            f"bnd={info['bnd_path'] or '—'}  "
            "NeKo lineage="
            f"{info['upstream_neko_manifest_path'] or '—'}"
        )
    return structured_report("\n".join(lines), payload)


@tool(
    title="List MaBoSS artifact sessions",
    annotations=_READ_ONLY_TOOL,
    structured_output=True,
)
def list_artifact_sessions(
) -> Annotated[CallToolResult, MaBoSSArtifactSessionListResult]:
    """List all MaBoSS sessions that have artifact files on disk (including past server runs).

    Unlike list_sessions() which only shows in-memory sessions, this scans the
    artifacts/ directory and reads session_meta.json files, so previously created
    sessions are visible even after a server restart.

    Use the returned session_id and file paths to resume earlier work, e.g.:
      build_simulation(bnd_path='/path/to/artifacts/<uuid>/output.bnd',
                       cfg_path='/path/to/artifacts/<uuid>/output.cfg')
    """
    sessions = _list_artifact_sessions_on_disk(_SERVER_ROOT, server_name="MaBoSS")
    payload = MaBoSSArtifactSessionListResult(
        server="MaBoSS",
        count=len(sessions),
        sessions=[
            ArtifactSessionSummary(
                artifact_directory=session.get("artifact_directory"),
                legacy=bool(session.get("legacy", False)),
                session_id=str(session["session_id"]),
                server=str(session.get("server") or "unknown"),
                label=str(session["label"]) if session.get("label") else None,
                created_at=(
                    str(session["created_at"])
                    if session.get("created_at")
                    else None
                ),
                files=[str(filename) for filename in session.get("files", [])],
            )
            for session in sessions
        ],
    )
    if not sessions:
        return structured_report("No artifact sessions found on disk.", payload)
    lines = ["## MaBoSS Artifact Sessions (on disk)\n"]
    for s in sessions:
        sid = s["session_id"]
        label = s.get("label") or ""
        created = s.get("created_at", "")[:19].replace("T", " ")
        files = s.get("files", [])
        lines.append(f"- **{sid}**" + (f" ({label})" if label else ""))
        if created:
            lines.append(f"  Created: {created} UTC")
        if files:
            lines.append(f"  Files: {', '.join(files)}")
        else:
            lines.append("  Files: (none)")
    return structured_report("\n".join(lines), payload)


@tool(
    title="Set default MaBoSS session",
    annotations=_IDEMPOTENT_TOOL,
    structured_output=True,
)
def set_default_session(
    session_id: Annotated[
        NonEmptyString,
        Field(description="ID of the session to set as the active default."),
    ],
) -> str:
    """Set the default (active) MaBoSS session used when session_id is omitted in other tools."""
    if session_manager.set_default(session_id):
        return f"Default session set to: {session_id}"
    raise ValueError(f"Session not found: {session_id}")


@tool(
    title="Delete MaBoSS session",
    annotations=_DESTRUCTIVE_TOOL,
    structured_output=True,
)
def delete_session(
    session_id: Annotated[
        NonEmptyString,
        Field(description="ID of the session to delete."),
    ],
    clean_files: bool = Field(
        default=True,
        description="When True (default), also remove all artifact files for this session.",
    ),
) -> str:
    """Delete a MaBoSS session and optionally its artifact files."""
    session = session_manager.get_session(session_id)
    if session is None or not session_manager.delete_session(session_id):
        raise ValueError(f"Session not found: {session_id}")

    removed_files = (
        clean_artifacts(_SERVER_ROOT, session.session_id)
        if clean_files
        else 0
    )
    return f"Session {session.session_id} deleted." + (
        f" Removed {removed_files} artifact file(s)." if clean_files else ""
    )


# ---------------------------------------------------------------------------
# Pipeline tools
# ---------------------------------------------------------------------------

@tool(
    title="Import NeKo handoff",
    annotations=_NON_IDEMPOTENT_TOOL,
    structured_output=True,
)
@_session_locked
def import_neko_handoff(
    manifest_path: Annotated[
        NonEmptyString,
        Field(
            description=(
                "Path to a NeKo `neko-to-maboss` handoff manifest. Its BNET "
                "artifact and integrity metadata are verified before import."
            )
        ),
    ],
    artifact_prefix: HandoffArtifactPrefix = Field(
        default="neko_import",
        description=(
            "Safe prefix for the imported MaBoSS BND and CFG artifacts. "
            "Choose a new prefix for every retained import."
        ),
    ),
    session_id: NonEmptyString | None = Field(
        default=None,
        description=(
            "Session to replace only after a complete successful import. "
            "Omit to use the active default session."
        ),
    ),
) -> Annotated[CallToolResult, MaBoSSHandoffImportResult]:
    """Verify, convert, and atomically load a typed NeKo handoff."""
    sess = ensure_session(session_id)
    loaded_manifest = load_handoff_manifest(
        manifest_path,
        expected_handoff_type="neko-to-maboss",
        verify_artifacts=True,
    )
    if not isinstance(loaded_manifest, NeKoToMaBoSSHandoffManifest):
        raise ValueError(
            "The supplied handoff is not a NeKo-to-MaBoSS manifest."
        )

    source_manifest_path = Path(manifest_path).resolve()
    source_manifest_file = handoff_artifact(
        source_manifest_path,
        server="NeKo",
        session_id=loaded_manifest.source.session_id,
        role="parent_manifest",
    )
    bnet_path = Path(loaded_manifest.bnet_file.path)
    stored_nodes = bnet_node_names(bnet_path)
    if stored_nodes != loaded_manifest.network.nodes:
        raise ValueError(
            "The BNET target order does not match the NeKo handoff manifest."
        )

    art_dir = get_artifact_dir(_SERVER_ROOT, sess.session_id)
    bnd_path = safe_artifact_path(art_dir, f"{artifact_prefix}.bnd")
    cfg_path = safe_artifact_path(art_dir, f"{artifact_prefix}.cfg")
    _require_unused_artifact_paths([bnd_path, cfg_path])

    created_paths: list[Path] = []
    try:
        with tempfile.TemporaryDirectory(
            dir=art_dir,
            prefix=".neko-handoff-import-",
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            temporary_bnd = temporary_root / "model.bnd"
            temporary_cfg = temporary_root / "model.cfg"
            try:
                maboss.bnet_to_bnd_and_cfg(
                    str(bnet_path),
                    str(temporary_bnd),
                    str(temporary_cfg),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Error converting the verified NeKo BNET: {exc}"
                ) from exc

            for path, label in (
                (temporary_bnd, "BND"),
                (temporary_cfg, "CFG"),
            ):
                if not path.is_file():
                    raise FileNotFoundError(
                        f"MaBoSS conversion did not create the {label} file."
                    )

            try:
                candidate_simulation = maboss.load(
                    str(temporary_bnd),
                    str(temporary_cfg),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Error loading the converted MaBoSS simulation: {exc}"
                ) from exc
            if not candidate_simulation:
                raise RuntimeError(
                    "maboss.load returned no simulation for the verified "
                    "NeKo handoff."
                )

            loaded_nodes = [
                str(node)
                for node in candidate_simulation.network.keys()
            ]
            if (
                len(loaded_nodes) != len(set(loaded_nodes))
                or set(loaded_nodes) != set(stored_nodes)
            ):
                raise ValueError(
                    "Converted MaBoSS nodes do not match the verified NeKo "
                    "BNET nodes."
                )

            output_nodes = list(loaded_manifest.network.output_nodes)
            try:
                candidate_simulation.network.set_output(output_nodes)
            except Exception as exc:
                raise ValueError(
                    "Could not apply the NeKo-declared MaBoSS output nodes: "
                    f"{exc}"
                ) from exc
            applied_outputs = [
                str(node)
                for node in candidate_simulation.network.get_output()
            ]
            if (
                len(applied_outputs) != len(set(applied_outputs))
                or set(applied_outputs) != set(output_nodes)
            ):
                raise ValueError(
                    "MaBoSS did not retain the output-node selection declared "
                    "by the NeKo manifest."
                )

            verify_handoff_artifact(source_manifest_file)
            verify_handoff_artifact(loaded_manifest.bnet_file)
            _link_artifact_without_overwrite(temporary_bnd, bnd_path)
            created_paths.append(bnd_path)
            _link_artifact_without_overwrite(temporary_cfg, cfg_path)
            created_paths.append(cfg_path)

        bnd_file = handoff_artifact(
            bnd_path,
            server="MaBoSS",
            session_id=sess.session_id,
            role="maboss_bnd",
        )
        cfg_file = handoff_artifact(
            cfg_path,
            server="MaBoSS",
            session_id=sess.session_id,
            role="maboss_cfg",
        )
        payload = MaBoSSHandoffImportResult(
            server="MaBoSS",
            session_id=sess.session_id,
            source_manifest_file=source_manifest_file,
            source_manifest=loaded_manifest,
            bnd_file=bnd_file,
            cfg_file=cfg_file,
            nodes=stored_nodes,
            output_nodes=output_nodes,
            requires_output_selection=not output_nodes,
        )
    except Exception:
        _rollback_artifacts(created_paths)
        raise

    sess.set_simulation(
        candidate_simulation,
        str(bnd_path),
        str(cfg_path),
        upstream_neko_manifest_path=str(source_manifest_path),
    )
    output_guidance = (
        "Applied outputs: " + ", ".join(output_nodes)
        if output_nodes
        else (
            "No outputs were declared. All nodes were marked internal; call "
            "set_maboss_output_nodes() before run_simulation()."
        )
    )
    text = (
        "NeKo handoff imported into MaBoSS successfully.\n"
        f"  Session: {sess.session_id}\n"
        f"  Source manifest: {source_manifest_path}\n"
        f"  BND: {bnd_path}\n"
        f"  CFG: {cfg_path}\n"
        f"  Boolean nodes: {len(stored_nodes)}\n"
        f"  {output_guidance}"
    )
    return structured_report(text, payload)


@tool(
    title="Convert BNET to MaBoSS files",
    annotations=_NON_IDEMPOTENT_TOOL,
    structured_output=True,
)
@_session_locked
def bnet_to_bnd_and_cfg(
    bnet_path: Annotated[
        NonEmptyString,
        Field(description="Absolute or CWD-relative path to the .bnet file to convert."),
    ],
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to write the output files into. Omit to use the active default session.",
    ),
    parent_manifest: Annotated[NonEmptyString | None, Field(description="Path to a verified source-lineage sidecar or supported typed parent manifest; records provenance, not approval.")] = None,
    derived_from: Annotated[NonEmptyString | None, Field(description="Exact original artifact path recorded by parent_manifest, required for an edited derivative.")] = None,
) -> Annotated[CallToolResult, MaBoSSBnetConversionResult]:
    """Convert a BNET file to MaBoSS BND and CFG files.

    Output files are written to the session artifact directory
    (output.bnd and output.cfg under the configured/cache artifact root).
    After conversion, call build_simulation() to load the simulation.

    parent_manifest names a verified source-lineage sidecar or NeKo handoff.
    For edited copies, derived_from must name its original BNET artifact path.
    These arguments record provenance; they do not grant scientific approval.
    """
    sess = ensure_session(session_id)
    source_import = capture_import(bnet_path, server="MaBoSS", session_id=sess.session_id,
                                   parent_manifest=parent_manifest, derived_from=derived_from)
    art_dir = get_artifact_dir(_SERVER_ROOT, sess.session_id)
    bnd_out = str(safe_artifact_path(art_dir, "output.bnd"))
    cfg_out = str(safe_artifact_path(art_dir, "output.cfg"))

    logger.info("Converting %s -> %s, %s", bnet_path, bnd_out, cfg_out)
    with tempfile.TemporaryDirectory(prefix=".bnet-conversion-", dir=art_dir) as temporary:
        staging = Path(temporary)
        staged_outputs = [staging / Path(output).name for output in (bnd_out, cfg_out)]
        try:
            maboss.bnet_to_bnd_and_cfg(bnet_path, *map(str, staged_outputs))
        except Exception as e:
            logger.exception("bnet_to_bnd_and_cfg failed")
            raise RuntimeError(f"Error converting .bnet file: {e}") from e

        for path, label in zip(staged_outputs, ("BND", "CFG"), strict=True):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Expected {label} file was not created at {path}."
                )

        verify_handoff_artifact(source_import.source)
        staged_sidecars = [
            write_source_export(output, server="MaBoSS", session_id=sess.session_id,
                                imports=[source_import], published_path=art_dir / output.name)
            for output in staged_outputs
        ]
        staged_import = write_source_import(staging, source_import)
        _publish_replacing_artifacts(
            [*staged_outputs, *staged_sidecars], art_dir, import_record=staged_import,
        )
    sess.pending_source_import = source_import
    sess.import_history.append(source_import)
    logger.info("BND and CFG files created: %s, %s", bnd_out, cfg_out)
    text = (
        f"MaBoSS .bnd and .cfg files created successfully.\n"
        f"  BND: {bnd_out}\n"
        f"  CFG: {cfg_out}\n\n"
        f"Next: call build_simulation(session_id='{sess.session_id}') to load the simulation."
    )
    payload = MaBoSSBnetConversionResult(
        server="MaBoSS",
        session_id=sess.session_id,
        input_bnet_path=bnet_path,
        bnd_file=artifact_file_summary(bnd_out, session_id=sess.session_id),
        cfg_file=artifact_file_summary(cfg_out, session_id=sess.session_id),
    )
    return structured_report(text, payload)


@tool(
    title="Build MaBoSS simulation",
    annotations=_IDEMPOTENT_TOOL,
    structured_output=True,
)
@_session_locked
def build_simulation(
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to load the simulation into. Omit to use the active default session.",
    ),
    bnd_path: NonEmptyString | None = Field(
        default=None,
        description="Path to the .bnd file. Omit to use the file generated by bnet_to_bnd_and_cfg for this session.",
    ),
    cfg_path: NonEmptyString | None = Field(
        default=None,
        description="Path to the .cfg file. Omit to use the file generated by bnet_to_bnd_and_cfg for this session.",
    ),
) -> str:
    """Load a MaBoSS simulation from BND and CFG files into the session.

    When bnd_path/cfg_path are omitted, the files produced by the last
    bnet_to_bnd_and_cfg call for this session are used automatically.
    After loading, inspect parameters via the maboss://session/{id}/parameters
    resource, tune with update_maboss_parameters, then call run_simulation.
    """
    if not isinstance(bnd_path, str):
        bnd_path = None
    if not isinstance(cfg_path, str):
        cfg_path = None
    sess = ensure_session(session_id)
    art_dir = get_artifact_dir(_SERVER_ROOT, sess.session_id)

    if bnd_path is None:
        bnd_path = str(art_dir / "output.bnd")
    if cfg_path is None:
        cfg_path = str(art_dir / "output.cfg")

    imported_source = None
    expected = [(art_dir / name).resolve() for name in ("output.bnd", "output.cfg")]
    candidate = sess.pending_source_import or sess.source_import
    if candidate is not None and [Path(bnd_path).resolve(), Path(cfg_path).resolve()] == expected:
        for output in expected:
            record = SourceExport.model_validate_json(
                output.with_name(output.name + ".lineage.json").read_bytes()
            )
            if record.artifact.path != str(output) or record.imports != [candidate]:
                raise ValueError("Converted model provenance changed; convert the BNET again.")
            verify_handoff_artifact(record.artifact)
        imported_source = candidate

    logger.info("Loading MaBoSS simulation: BND=%s CFG=%s", bnd_path, cfg_path)

    try:
        loaded_sim = maboss.load(bnd_path, cfg_path)
    except Exception as e:
        logger.exception("Failed to load simulation")
        raise RuntimeError(f"Error loading MaBoSS simulation: {e}") from e

    if loaded_sim:
        readouts = read_readouts(bnd_path, loaded_sim)
        sess.set_simulation(loaded_sim, bnd_path, cfg_path)
        sess.readouts = readouts
        sess.source_import = imported_source
        logger.info("MaBoSS simulation loaded successfully")
        parameters_str = "\n".join(f"{k}: {v}" for k, v in loaded_sim.param.items())
        return (
            f"MaBoSS simulation loaded successfully.\n{parameters_str}\n\n"
            f"NEXT STEP: call get_maboss_nodes() to retrieve the list of valid node "
            f"names before calling set_maboss_output_nodes() or set_maboss_initial_state()."
        )
    else:
        logger.error("maboss.load returned None")
        raise RuntimeError(
            "maboss.load returned None. Check the BND and CFG files."
        )


@tool(
    title="Run MaBoSS simulation",
    annotations=_NON_IDEMPOTENT_TOOL,
    structured_output=True,
)
async def run_simulation(
    ctx: Context,
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to run. Omit to use the active default session.",
    ),
) -> Annotated[CallToolResult, MaBoSSSimulationRunResult]:
    """Execute the loaded MaBoSS simulation and store the result in the session.

    IMPORTANT: Call set_maboss_output_nodes() before this tool. Without it, MaBoSS becomes exponentially expensive. 
    Restricting to a small set of output nodes keeps the run time and result size manageable.

    Tune performance via update_maboss_parameters (sample_count, thread_count)
    before running large simulations. After completion, read the result from
    maboss://session/{id}/result.
    """
    await ctx.report_progress(0, 2)

    def run_locked():
        with session_manager.session_scope(session_id):
            sess = ensure_session(session_id)
            if sess.sim is None:
                raise RuntimeError(
                    "No MaBoSS simulation has been built yet. "
                    "Call bnet_to_bnd_and_cfg then build_simulation first."
                )
            simulation_network = getattr(sess.sim, "network", None)
            get_output = getattr(simulation_network, "get_output", None)
            if callable(get_output) and not list(get_output()):
                raise RuntimeError(
                    "No MaBoSS output nodes are selected. Call "
                    "set_maboss_output_nodes() with a small biologically "
                    "meaningful set before run_simulation()."
                )
            revision = model_revision(sess.sim, sess.readouts) if hasattr(sess.sim, "print_cfg") else None
            config = effective_configuration(sess, sess.sim) if revision else {}
            art_dir = get_artifact_dir(_SERVER_ROOT, sess.session_id)
            run_result = None
            try:
                run_result = sess.sim.run()
                if getattr(run_result, "_err", 0):
                    raise RuntimeError(f"Engine exit code {run_result._err}: {getattr(run_result, 'error_detail', '')}")
            except Exception as exc:
                diagnostic = record_failure(art_dir, sess.session_id, "engine", exc, revision, run_result)
                raise RuntimeError(f"MaBoSS engine failed; incomplete record: {diagnostic}: {exc}") from exc
            try:
                manifest_path = _preserve_simulation_result(
                    run_result, art_dir, sess.session_id, model_revision=revision,
                    configuration=config)
                manifest = json.loads(manifest_path.read_text())
            except Exception as exc:
                diagnostic = record_failure(art_dir, sess.session_id, "persistence", exc, revision, run_result)
                raise RuntimeError(f"Engine finished, but persistence failed; do not rerun automatically. Incomplete record: {diagnostic}: {exc}") from exc
            sess.set_result(run_result)
            sess.latest_run_id = manifest["run_id"]
            sess.result_revision = revision
            saved_csv_path = None
            df_result = pd.DataFrame()
            try:
                df_result = run_result.get_last_states_probtraj()
                if not df_result.empty:
                    saved_csv_path = safe_artifact_path(art_dir, "result.csv")
                    df_result.to_csv(saved_csv_path, index=False)
            except Exception:
                saved_csv_path = None
                logger.warning("Historical run preserved; latest result.csv alias could not be written")
            return (sess.session_id, len(df_result), len(df_result.columns), saved_csv_path,
                    manifest["run_id"], str(manifest_path), revision)

    resolved_session_id, row_count, column_count, csv_path, run_id, manifest_path, revision = (
        await anyio.to_thread.run_sync(run_locked)
    )
    await ctx.report_progress(2, 2)
    logger.info("MaBoSS simulation completed successfully")
    text = (
        "MaBoSS simulation completed successfully.\n"
        "Call `get_simulation_result()` to read the state probability table.\n"
        + (
            "The result is also saved to the session artifact directory as result.csv."
            if csv_path is not None
            else "No non-empty result table was written to result.csv."
        )
    )
    payload = MaBoSSSimulationRunResult(
        server="MaBoSS",
        session_id=resolved_session_id,
        run_id=run_id, run_manifest=manifest_path, model_revision=revision,
        result_available=True,
        trajectory_row_count=row_count,
        trajectory_column_count=column_count,
        result_file=(
            artifact_file_summary(
                csv_path,
                session_id=resolved_session_id,
            )
            if csv_path is not None
            else None
        ),
    )
    return structured_report(text, payload)

@tool(
    title="Export MaBoSS BND and CFG",
    annotations=_NON_IDEMPOTENT_TOOL,
    structured_output=True,
)
@_session_locked
def export_maboss_bnd_cfg(
    prefix: Annotated[NonEmptyString, Field(
        default="updated",
        description=(
            "Prefix for output filenames written to the session artifact directory. "
            "Produces '<prefix>.bnd' and '<prefix>.cfg'. Example: 'run2' -> run2.bnd/run2.cfg."
        ),
    )] = "updated",
    overwrite: Annotated[bool, Field(
        default=False,
        description="If True, overwrite existing files with the same names in the artifact directory.",
    )] = False,
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to export from. Omit to use the active default session.",
    ),
) -> Annotated[CallToolResult, MaBoSSModelExportResult]:
    """Export the current in-memory MaBoSS simulation to .bnd and .cfg files.

    Writes <prefix>.bnd and <prefix>.cfg into the configured/cache session artifact directory.
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise RuntimeError(
            "No MaBoSS simulation has been built yet. Call build_simulation first."
        )

    try:
        prefix = (prefix or "").strip()
        if not prefix:
            raise ValueError("Invalid prefix. Must be a non-empty string.")

        # Optionally normalize prefix a bit (avoid spaces)
        prefix = prefix.replace(" ", "_")

        art_dir = get_artifact_dir(_SERVER_ROOT, sess.session_id)
        bnd_name = f"{prefix}.bnd"
        cfg_name = f"{prefix}.cfg"

        bnd_out = str(safe_artifact_path(art_dir, bnd_name))
        cfg_out = str(safe_artifact_path(art_dir, cfg_name))

        if not overwrite:
            for p in (bnd_out, cfg_out):
                if os.path.exists(p):
                    raise FileExistsError(
                        f"Refusing to overwrite existing file: {p}\n"
                        f"Choose a different prefix or set overwrite=True."
                    )

        logger.info("Exporting MaBoSS model -> %s, %s", bnd_out, cfg_out)

        # Write .bnd
        with open(bnd_out, "w") as fbnd:
            print_bnd_with_readouts(sess, fbnd)

        # Write .cfg
        with open(cfg_out, "w") as fcfg:
            sess.sim.print_cfg(out=fcfg)

        # Sanity check
        for path, label in [(bnd_out, "BND"), (cfg_out, "CFG")]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Expected {label} file was not created at {path}."
                )

        for output in (bnd_out, cfg_out):
            write_source_export(output, server="MaBoSS", session_id=sess.session_id,
                                imports=[sess.source_import] if sess.source_import else [])
        logger.info("Export complete: %s, %s", bnd_out, cfg_out)
        text = (
            f"Exported current MaBoSS model successfully.\n"
            f"  BND: {bnd_out}\n"
            f"  CFG: {cfg_out}"
        )
        payload = MaBoSSModelExportResult(
            server="MaBoSS",
            session_id=sess.session_id,
            model_revision=model_revision(sess.sim, sess.readouts) if hasattr(sess.sim, "str_cfg") else None,
            output_nodes=list(sess.sim.network.get_output()) if hasattr(sess.sim, "network") else [],
            internal_nodes=[n for n in sess.sim.network if n not in sess.sim.network.get_output()] if hasattr(sess.sim, "network") else [],
            prefix=prefix,
            overwrite=overwrite,
            bnd_file=artifact_file_summary(
                bnd_out,
                session_id=sess.session_id,
            ),
            cfg_file=artifact_file_summary(
                cfg_out,
                session_id=sess.session_id,
            ),
        )
        return structured_report(text, payload)

    except (ValueError, FileExistsError, FileNotFoundError):
        raise
    except Exception as e:
        logger.exception("Error exporting MaBoSS model")
        raise RuntimeError(f"Error exporting MaBoSS model: {e}") from e


@tool(
    title="Export MaBoSS handoff",
    annotations=_NON_IDEMPOTENT_TOOL,
    structured_output=True,
)
@_session_locked
def export_maboss_handoff(
    target_cell_type: Annotated[
        NonEmptyString,
        Field(
            description=(
                "PhysiCell cell type intended to receive this Boolean model."
            )
        ),
    ],
    biological_context: NonEmptyString | None = Field(
        default=None,
        description=(
            "Biological context for PhysiCell integration. Omit to inherit "
            "the context of an imported NeKo handoff; required for standalone "
            "MaBoSS models."
        ),
    ),
    simulation_summary: NonEmptyString | None = Field(
        default=None,
        description=(
            "Optional scientific interpretation of the MaBoSS result. When "
            "omitted, a concise table-availability summary is generated."
        ),
    ),
    include_result: bool = Field(
        default=True,
        description=(
            "Snapshot the stored state-probability table as a CSV artifact "
            "when non-empty simulation results are available."
        ),
    ),
    artifact_prefix: HandoffArtifactPrefix = Field(
        default="maboss_to_physicell",
        description=(
            "Safe prefix for the BND, CFG, optional result CSV, and manifest. "
            "Choose a new prefix for every retained handoff."
        ),
    ),
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to export; omit to use the active default session.",
    ),
) -> Annotated[CallToolResult, MaBoSSHandoffExportResult]:
    """Export an integrity-protected MaBoSS-to-PhysiCell handoff."""
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise RuntimeError(
            "No MaBoSS simulation has been built yet. Import a NeKo handoff "
            "or call build_simulation first."
        )

    nodes = [str(node) for node in sess.sim.network.keys()]
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValueError(
            "The loaded MaBoSS simulation must contain unique Boolean nodes."
        )
    output_nodes = [
        str(node)
        for node in sess.sim.network.get_output()
    ]
    if not output_nodes:
        raise RuntimeError(
            "No MaBoSS output nodes are selected. Call "
            "set_maboss_output_nodes() with a small biologically meaningful "
            "set before exporting a PhysiCell handoff."
        )
    if len(output_nodes) != len(set(output_nodes)):
        raise ValueError("MaBoSS output nodes contain duplicate names.")
    unknown_outputs = sorted(set(output_nodes) - set(nodes))
    if unknown_outputs:
        raise ValueError(
            "MaBoSS output nodes are absent from the loaded network: "
            + ", ".join(unknown_outputs)
        )

    parent_manifest = None
    parent_manifest_file = None
    lineage = []
    inherited_context = None
    renamed_nodes: list[str] = []
    node_renames: dict[str, str] = {}
    duplicate_rules_removed: list[str] = []
    if sess.upstream_neko_manifest_path is not None:
        loaded_parent = load_handoff_manifest(
            sess.upstream_neko_manifest_path,
            expected_handoff_type="neko-to-maboss",
            verify_artifacts=True,
        )
        if not isinstance(loaded_parent, NeKoToMaBoSSHandoffManifest):
            raise ValueError(
                "The session's upstream handoff is not a NeKo manifest."
            )
        parent_manifest = loaded_parent
        parent_manifest_file = handoff_artifact(
            sess.upstream_neko_manifest_path,
            server="NeKo",
            session_id=loaded_parent.source.session_id,
            role="parent_manifest",
        )
        lineage = [loaded_parent.source]
        inherited_context = loaded_parent.biological_context
        renamed_nodes = list(loaded_parent.network.renamed_nodes)
        node_renames = dict(loaded_parent.network.node_renames)
        duplicate_rules_removed = list(
            loaded_parent.network.duplicate_rules_removed
        )

    context = (
        biological_context.strip()
        if biological_context is not None
        else inherited_context
    )
    if not context:
        raise ValueError(
            "biological_context is required for a standalone MaBoSS model "
            "because no NeKo context is available to inherit."
        )

    parameters = _handoff_parameters(sess.sim.param)
    if include_result and sess.result is not None and sess.result_revision is not None:
        if sess.result_revision != model_revision(sess.sim, sess.readouts):
            raise ValueError("Stored result belongs to an older model revision; export without it or explicitly run the reviewed current model")

    result_table = None
    result_row_count = 0
    result_column_count = 0
    if sess.result is not None:
        try:
            result_table = sess.result.get_last_states_probtraj()
        except Exception as exc:
            raise RuntimeError(
                f"Could not read the stored MaBoSS simulation result: {exc}"
            ) from exc
        if not isinstance(result_table, pd.DataFrame):
            raise ValueError(
                "The stored MaBoSS result did not return a pandas DataFrame."
            )
        result_row_count = len(result_table)
        result_column_count = len(result_table.columns)

    summary = (
        simulation_summary.strip()
        if simulation_summary is not None
        else (
            "Stored state-probability table contains "
            f"{result_row_count} row(s) and {result_column_count} column(s)."
            if result_table is not None
            else "No MaBoSS simulation result was stored at export time."
        )
    )

    art_dir = get_artifact_dir(_SERVER_ROOT, sess.session_id)
    bnd_path = safe_artifact_path(art_dir, f"{artifact_prefix}.bnd")
    cfg_path = safe_artifact_path(art_dir, f"{artifact_prefix}.cfg")
    result_path = safe_artifact_path(
        art_dir,
        f"{artifact_prefix}.result.csv",
    )
    manifest_path = safe_artifact_path(
        art_dir,
        f"{artifact_prefix}.handoff.json",
    )
    include_result_artifact = bool(
        include_result
        and result_table is not None
        and not result_table.empty
    )
    destinations = [bnd_path, cfg_path, manifest_path]
    if include_result_artifact:
        destinations.append(result_path)
    _require_unused_artifact_paths(destinations)

    created_paths: list[Path] = []
    try:
        with tempfile.TemporaryDirectory(
            dir=art_dir,
            prefix=".maboss-handoff-export-",
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            temporary_bnd = temporary_root / "model.bnd"
            temporary_cfg = temporary_root / "model.cfg"
            temporary_result = temporary_root / "result.csv"
            try:
                with temporary_bnd.open("w", encoding="utf-8") as bnd_file:
                    print_bnd_with_readouts(sess, bnd_file)
                with temporary_cfg.open("w", encoding="utf-8") as cfg_file:
                    sess.sim.print_cfg(out=cfg_file)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not snapshot the current MaBoSS model: {exc}"
                ) from exc

            if include_result_artifact:
                try:
                    result_table.to_csv(temporary_result, index=False)
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not snapshot the MaBoSS result table: {exc}"
                    ) from exc

            _link_artifact_without_overwrite(temporary_bnd, bnd_path)
            created_paths.append(bnd_path)
            _link_artifact_without_overwrite(temporary_cfg, cfg_path)
            created_paths.append(cfg_path)
            if include_result_artifact:
                _link_artifact_without_overwrite(
                    temporary_result,
                    result_path,
                )
                created_paths.append(result_path)

        bnd_file = handoff_artifact(
            bnd_path,
            server="MaBoSS",
            session_id=sess.session_id,
            role="maboss_bnd",
        )
        cfg_file = handoff_artifact(
            cfg_path,
            server="MaBoSS",
            session_id=sess.session_id,
            role="maboss_cfg",
        )
        result_file = (
            handoff_artifact(
                result_path,
                server="MaBoSS",
                session_id=sess.session_id,
                role="maboss_result",
            )
            if include_result_artifact
            else None
        )
        source_ancestry = []
        if sess.source_import is not None:
            source_ancestry = [sess.source_import]
        elif parent_manifest is not None:
            source_ancestry = parent_manifest.source_ancestry or []
        manifest = MaBoSSToPhysiCellHandoffManifest(
            schema_version="2.1",
            model_revision=model_revision(sess.sim, sess.readouts),
            readouts=sess.readouts,
            source_ancestry=source_ancestry,
            source=HandoffProvenance(
                server="MaBoSS",
                session_id=sess.session_id,
                mcp_package=HandoffPackage(
                    name="mcp-biomodelling-servers",
                    version=__version__,
                ),
                modelling_package=HandoffPackage(
                    name="maboss",
                    version=_maboss_package_version(),
                ),
                operation="export_maboss_handoff",
            ),
            lineage=lineage,
            biological_context=context,
            network=HandoffNetwork(
                nodes=nodes,
                output_nodes=output_nodes,
                renamed_nodes=renamed_nodes,
                node_renames=node_renames,
                duplicate_rules_removed=duplicate_rules_removed,
            ),
            bnd_file=bnd_file,
            cfg_file=cfg_file,
            parent_manifest=parent_manifest_file,
            simulation=MaBoSSSimulationHandoff(
                parameters=parameters,
                simulation_summary=summary,
                result_file=result_file,
            ),
            target=PhysiCellTarget(cell_type=target_cell_type),
        )
        if parent_manifest is not None:
            verify_handoff_artifact(parent_manifest_file)
            verify_handoff_manifest(parent_manifest)
        write_handoff_manifest(manifest_path, manifest)
        created_paths.append(manifest_path)
        if sess.source_import is not None:
            created_paths.append(write_source_export(
                manifest_path, server="MaBoSS", session_id=sess.session_id,
                imports=[sess.source_import], overwrite=False))
        manifest_file = handoff_artifact(
            manifest_path,
            server="MaBoSS",
            session_id=sess.session_id,
            role="parent_manifest",
        )
        payload = MaBoSSHandoffExportResult(
            server="MaBoSS",
            session_id=sess.session_id,
            manifest_file=manifest_file,
            manifest=manifest,
        )
    except Exception:
        _rollback_artifacts(created_paths)
        raise

    lineage_text = (
        f"NeKo session {parent_manifest.source.session_id}"
        if parent_manifest is not None
        else "standalone MaBoSS model"
    )
    result_text = (
        str(result_path)
        if include_result_artifact
        else "not included"
    )
    text = (
        "MaBoSS-to-PhysiCell handoff exported successfully.\n"
        f"  Manifest: {manifest_path}\n"
        f"  BND: {bnd_path}\n"
        f"  CFG: {cfg_path}\n"
        f"  Result CSV: {result_text}\n"
        f"  Boolean nodes: {len(nodes)}\n"
        f"  Output nodes: {', '.join(output_nodes)}\n"
        f"  Target cell type: {target_cell_type}\n"
        f"  Lineage: {lineage_text}\n\n"
        "Next: pass the manifest path to the PhysiCell handoff import tool."
    )
    return structured_report(text, payload)


# ---------------------------------------------------------------------------
# Inspection tools (read-only, no side effects)
# ---------------------------------------------------------------------------

@tool(
    title="Get MaBoSS nodes",
    annotations=_READ_ONLY_TOOL,
    structured_output=True,
)
@_session_locked
def get_maboss_nodes(
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to query. Omit to use the active default session.",
    ),
    selected_nodes: Annotated[list[NonEmptyString] | None, Field(max_length=25, description="Inspect rules, rate expressions and referenced parameters for these nodes.")] = None,
) -> Annotated[CallToolResult, MaBoSSNodeListResult]:
    """Return the list of node names in the loaded MaBoSS network.

    Always call this after build_simulation() and before set_maboss_output_nodes()
    or set_maboss_initial_state() to avoid referencing non-existent nodes.
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise RuntimeError(
            "No simulation loaded. Call bnet_to_bnd_and_cfg then build_simulation first."
        )
    nodes_list = list(sess.sim.network.keys())
    text = (
        "No nodes found in the MaBoSS network."
        if not nodes_list
        else "Network nodes:\n" + "\n".join(f"- {n}" for n in nodes_list)
    )
    payload = MaBoSSNodeListResult(
        server="MaBoSS",
        session_id=sess.session_id,
        node_count=len(nodes_list),
        nodes=[str(node) for node in nodes_list],
        model_revision=model_revision(sess.sim, sess.readouts) if hasattr(sess.sim, "print_cfg") else None,
        output_nodes=list(sess.sim.network.get_output()) if hasattr(sess.sim.network, "get_output") else [],
        details=inspect_nodes(sess.sim, selected_nodes, sess.readouts) if selected_nodes else [],
    )
    return structured_report(text, payload)


@tool(
    title="Get MaBoSS initial state",
    annotations=_READ_ONLY_TOOL,
    structured_output=True,
)
@_session_locked
def get_maboss_initial_state(
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to query. Omit to use the active default session.",
    ),
) -> Annotated[CallToolResult, MaBoSSInitialStateResult]:
    """Return the current initial state probability configuration of the MaBoSS simulation.

    Use this to inspect the state before calling set_maboss_initial_state().
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise RuntimeError(
            "No simulation loaded. Call bnet_to_bnd_and_cfg then build_simulation first."
        )
    try:
        initial_state = sess.sim.network.get_istate()
        groups = _initial_state_groups(initial_state)
        payload = MaBoSSInitialStateResult(
            server="MaBoSS",
            session_id=sess.session_id,
            group_count=len(groups),
            groups=groups,
        )
        return structured_report(f"Initial state:\n{initial_state}", payload)
    except Exception as e:
        raise RuntimeError(f"Error retrieving initial state: {e}") from e

@tool(
    title="Get MaBoSS logical rules",
    annotations=_READ_ONLY_TOOL,
    structured_output=True,
)
@_session_locked
def get_maboss_logical_rules(
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to query. Omit to use the active default session.",
    ),
    selected_nodes: Annotated[list[NonEmptyString] | None, Field(min_length=1, max_length=25, description="Inspect only these node rules; omitted returns up to 100 nodes.")] = None,
) -> Annotated[CallToolResult, MaBoSSLogicalRulesResult]:
    """Return the Boolean logical rules of the loaded MaBoSS network.

    rule_limitation flags nodes whose custom rate expressions mean the rule
    alone does not describe switching; inspect them with get_maboss_nodes.
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise RuntimeError(
            "No simulation loaded. Call bnet_to_bnd_and_cfg then build_simulation first."
        )
    try:
        total_node_count = truncated = None
        if hasattr(sess.sim, "print_cfg"):
            all_names = list(sess.sim.network)
            names = selected_nodes or all_names[:100]
            if selected_nodes is None:
                total_node_count, truncated = len(all_names), len(all_names) > len(names)
            records = inspect_nodes(sess.sim, names, sess.readouts)
            rules = [{"node": item["node"], "rule": item["rule"] or None,
                      "rule_limitation": item["rule_limitation"]} for item in records]
            lines = [item["node"] + " = " + (item["rule"] or "(no explicit rule)")
                     + (f"  [{item['rule_limitation']}]" if item["rule_limitation"] else "")
                     for item in rules]
            if truncated:
                lines.append(f"Showing {len(names)} of {total_node_count} nodes; use selected_nodes for others.")
            logical_rules = "\n".join(lines)
        else:
            logical_rules = sess.sim.get_logical_rules()
            rules = _logical_rule_records(logical_rules)
        payload = MaBoSSLogicalRulesResult(
            server="MaBoSS",
            session_id=sess.session_id,
            model_revision=model_revision(sess.sim, sess.readouts) if hasattr(sess.sim, "print_cfg") else None,
            rule_count=len(rules),
            rules=rules,
            total_node_count=total_node_count,
            truncated=truncated,
        )
        return structured_report(str(logical_rules), payload)
    except Exception as e:
        raise RuntimeError(f"Error retrieving logical rules: {e}") from e


@tool(
    title="Change MaBoSS logical rule",
    annotations=_IDEMPOTENT_TOOL,
    structured_output=True,
)
@_session_locked
def change_maboss_rule(
    node: Annotated[
        NonEmptyString,
        Field(description="Name of the node to change the rule for."),
    ],
    new_rule: Annotated[
        NonEmptyString,
        Field(description="New rule string to replace the existing rule."),
    ],
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to update. Omit to use the active default session.",
    ),
) -> str:
    """Change the Boolean rule for a specific node in the MaBoSS simulation network.

    The new rule is validated before being kept. If invalid, the previous rule is restored.
    Call get_maboss_logical_rules() first to inspect the current rules.
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise RuntimeError(
            "No MaBoSS simulation has been built yet. "
            "Call bnet_to_bnd_and_cfg then build_simulation first."
        )

    try:
        if not isinstance(node, str) or not node.strip():
            raise ValueError("Invalid node name. Must be a non-empty string.")

        if not isinstance(new_rule, str) or not new_rule.strip():
            raise ValueError("Invalid new_rule. Must be a non-empty string.")

        node = node.strip()
        new_rule = new_rule.strip()

        if node in sess.readouts:
            raise ValueError("Use edit_maboss_readouts to preserve readout provenance")
        if any(name in __import__("re").findall(r"[A-Za-z_][A-Za-z_0-9]*", new_rule) for name in sess.readouts):
            raise ValueError("Readouts cannot feed back into core rules")
        # Check node existence
        try:
            target_node = sess.sim.network[node]
        # pyMaBoSS does not expose a stable lookup exception contract.
        except Exception as node_error:  # noqa: BLE001
            try:
                available_nodes = list(sess.sim.network.keys())
            except Exception as list_error:  # noqa: BLE001
                raise ValueError(f"Unknown node '{node}'.") from list_error
            raise ValueError(
                f"Unknown node '{node}'. "
                f"Available nodes: {', '.join(available_nodes)}"
            ) from node_error

        # Save previous rule
        old_rule = target_node.logExp
        logger.info("Previous rule for %s: %s", node, old_rule)

        # Apply new rule
        target_node.logExp = new_rule

        # Validate the updated model
        try:
            check_result = sess.sim.check()
        except Exception as check_exc:
            target_node.logExp = old_rule
            logger.exception(
                "Validation failed unexpectedly after updating %s", node
            )
            raise RuntimeError(
                f"Rule change aborted for '{node}'. "
                f"Validation could not be completed: {check_exc}"
            ) from check_exc

        if check_result:
            # pyMaBoSS may return parser/check errors as a non-empty result
            target_node.logExp = old_rule
            logger.error(
                "Invalid rule for %s. Change reverted. Errors: %s",
                node,
                check_result,
            )
            raise ValueError(
                f"Rule change rejected for '{node}'. The previous rule has been restored.\n"
                f"Previous rule: {old_rule}\n"
                f"Proposed rule: {new_rule}\n"
                f"Validation errors: {check_result}"
            )

        logger.info("Updated rule for %s: %s", node, target_node.logExp)
        return (
            f"Rule changed successfully for '{node}'.\n"
            f"Previous rule: {old_rule}\n"
            f"New rule: {target_node.logExp}"
        )

    except (ValueError, RuntimeError):
        raise
    except Exception as e:
        logger.exception("Error changing MaBoSS rule")
        raise RuntimeError(f"Error changing MaBoSS rule: {e}") from e


@tool(
    title="Get MaBoSS mutations",
    annotations=_READ_ONLY_TOOL,
    structured_output=True,
)
@_session_locked
def get_maboss_mutations(
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to query. Omit to use the active default session.",
    ),
) -> Annotated[CallToolResult, MaBoSSMutationListResult]:
    """Return the mutation settings currently applied to the MaBoSS network."""
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise RuntimeError(
            "No simulation loaded. Call bnet_to_bnd_and_cfg then build_simulation first."
        )
    try:
        mutation_values = sess.sim.get_mutations()
        mutations = _mutation_records(mutation_values)
        payload = MaBoSSMutationListResult(
            server="MaBoSS",
            session_id=sess.session_id,
            mutation_count=len(mutations),
            mutations=mutations,
        )
        return structured_report(str(mutation_values), payload)
    except Exception as e:
        raise RuntimeError(f"Error retrieving mutations: {e}") from e


# ---------------------------------------------------------------------------
# Configuration tools
# ---------------------------------------------------------------------------

@tool(
    title="Update MaBoSS parameters",
    annotations=_IDEMPOTENT_TOOL,
    structured_output=True,
)
@_session_locked
def update_maboss_parameters(
    parameters: MaBoSSParameterUpdates | None = Field(  # noqa: B008
        default=None,
        description=(
            "Dict of {parameter_name: value} to update. "
            "Omit or pass null to list all current parameters and valid keys. "
            "Common keys: sample_count (int), max_time (float), time_tick (float), "
            "discrete_time (0|1), thread_count (int)."
        ),
    ),
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to update. Omit to use the active default session.",
    ),
    rates: Annotated[list[RateEdit] | None, Field(min_length=1, max_length=25, description="Explicit named-node rate coefficients; conditional logic is preserved.")] = None,
    expected_revision: Annotated[NonEmptyString | None, Field(description="Required current model revision for rate edits.")] = None,
) -> Annotated[CallToolResult, MaBoSSParameterResult]:
    """Update one or more MaBoSS simulation parameters, or list current values.

    Call with parameters=null (or omit it) to display all current parameter
    values and their valid keys before making changes.
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise RuntimeError(
            "No MaBoSS simulation has been built yet. "
            "Call bnet_to_bnd_and_cfg then build_simulation first."
        )
    rate_changes = None
    if rates:
        if parameters is not None:
            raise ValueError("Edit rates and global parameters in separate calls")
        rate_changes = edit_rates(sess, rates, expected_revision)
    try:
        parameter_updates = (
            None
            if parameters is None
            else parameters.model_dump(exclude_none=True)
        )
        if not parameter_updates:
            df = pd.DataFrame(
                [[k, v] for k, v in sess.sim.param.items()],
                columns=["parameter", "value"],
            )
            text = (
                "Current MaBoSS parameters "
                "(pass a parameters dict to update_maboss_parameters to modify):\n"
                + df.to_markdown(index=False, tablefmt="plain")
            )
            parameters_list = _parameter_records(sess.sim.param)
            payload = MaBoSSParameterResult(
                server="MaBoSS",
                session_id=sess.session_id,
                model_revision=model_revision(sess.sim, sess.readouts) if hasattr(sess.sim, "print_cfg") else None,
                rate_changes=rate_changes,
                stationary_distribution_enabled=bool(sess.sim.param.get("statdist_traj_count", 0)),
                mode="update" if rate_changes else "inspect",
                parameter_count=len(parameters_list),
                parameters=parameters_list,
                updated_parameters=[],
            )
            return structured_report(text, payload)
        allowed = set(sess.sim.param.keys())
        unknown = [k for k in parameter_updates if k not in allowed]
        if unknown:
            raise ValueError(
                "Unsupported parameter(s): " + ", ".join(unknown) +
                "\nCall update_maboss_parameters with no arguments to list valid keys."
            )
        for key, value in parameter_updates.items():
            sess.sim.param[key] = value
        logger.info("MaBoSS parameters updated: %s", parameter_updates)
        summary = ", ".join(f"{k}={v}" for k, v in parameter_updates.items())
        parameters_list = _parameter_records(sess.sim.param)
        payload = MaBoSSParameterResult(
            server="MaBoSS",
            session_id=sess.session_id,
            model_revision=model_revision(sess.sim, sess.readouts) if hasattr(sess.sim, "print_cfg") else None,
            stationary_distribution_enabled=bool(sess.sim.param.get("statdist_traj_count", 0)),
            mode="update",
            parameter_count=len(parameters_list),
            parameters=parameters_list,
            updated_parameters=list(parameter_updates),
        )
        return structured_report(f"Parameters updated: {summary}", payload)
    except ValueError:
        raise
    except Exception as e:
        logger.exception("Error updating MaBoSS parameters")
        raise RuntimeError(f"Error updating MaBoSS parameters: {e}") from e


@tool(
    title="Set MaBoSS output nodes",
    annotations=_IDEMPOTENT_TOOL,
    structured_output=True,
)
@_session_locked
def set_maboss_output_nodes(
    output_nodes: Annotated[
        list[NonEmptyString],
        Field(
            min_length=1,
            description="Non-empty list of node names to mark as output nodes (e.g. ['Apoptosis', 'Proliferation']).",
        ),
    ],
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to update. Omit to use the active default session.",
    ),
) -> str:
    """Set which nodes are treated as outputs in the MaBoSS simulation.

    Call get_maboss_nodes() first to obtain the exact valid node names.
    Limiting outputs reduces result size and speeds up large simulations.
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise RuntimeError(
            "No MaBoSS simulation has been built yet. "
            "Call bnet_to_bnd_and_cfg then build_simulation first."
        )
    try:
        logger.info("Previous output nodes: %s", sess.sim.network.get_output())
        sess.sim.network.set_output(output_nodes)
        logger.info("Updated output nodes: %s", sess.sim.network.get_output())
        return f"Output nodes set successfully: {sess.sim.network.get_output()}"
    except Exception as e:
        logger.exception("Error setting MaBoSS output nodes")
        raise RuntimeError(f"Error setting MaBoSS output nodes: {e}") from e


@tool(
    title="Set MaBoSS initial state",
    annotations=_IDEMPOTENT_TOOL,
    structured_output=True,
)
@_session_locked
def set_maboss_initial_state(
    nodes: Annotated[
        NonEmptyString | Annotated[list[NonEmptyString], Field(min_length=1)],
        Field(
            description=(
                "Node name (str) or non-empty list of node names to set initial "
                "state for. E.g. 'node1' or ['node1', 'node2']."
            )
        ),
    ],
    probDict: Annotated[InitialStateProbabilitySpecification, Field(
        description=(
            "Probability specification. "
            "Single node: list [P(OFF), P(ON)] or dict {0: P(OFF), 1: P(ON)}. "
            "Multiple nodes: JSON-native records with a Boolean state vector "
            "and probability, e.g. [{'state': [0, 0], 'probability': 0.4}, "
            "{'state': [1, 0], 'probability': 0.6}]. State-vector order must "
            "match nodes. Legacy tuple-key dictionaries remain available to "
            "direct Python callers. Probabilities must sum to 1."
        )
    )],
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to update. Omit to use the active default session.",
    ),
) -> str:
    """Set initial state probabilities for one or more nodes in the MaBoSS simulation.

    Call get_maboss_nodes() first to obtain the exact valid node names.
    Call get_maboss_initial_state() to inspect the current state before modifying it.

    Examples:
        set_maboss_initial_state('node1', [0.3, 0.7])
        set_maboss_initial_state(
            ['node1', 'node2'],
            [
                {'state': [0, 0], 'probability': 0.4},
                {'state': [1, 0], 'probability': 0.6},
            ],
        )
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        raise RuntimeError(
            "No MaBoSS simulation has been built yet. "
            "Call bnet_to_bnd_and_cfg then build_simulation first."
        )
    try:
        node_arg, normalized_probabilities = (
            _normalize_initial_state_probabilities(nodes, probDict)
        )
        available_nodes = {
            str(node)
            for node in sess.sim.network.keys()
        }
        requested_nodes = (
            [node_arg]
            if isinstance(node_arg, str)
            else node_arg
        )
        unknown_nodes = [
            node
            for node in requested_nodes
            if node not in available_nodes
        ]
        if unknown_nodes:
            raise ValueError(
                "Unknown MaBoSS initial-state node(s): "
                f"{', '.join(unknown_nodes)}. Call get_maboss_nodes() first."
            )

        logger.info("Previous initial state: %s", sess.sim.network.get_istate())
        sess.sim.network.set_istate(node_arg, normalized_probabilities)
        logger.info("Updated initial state: %s", sess.sim.network.get_istate())
        return f"Initial state set: {sess.sim.network.get_istate()}"
    except ValueError:
        raise
    except Exception as e:
        logger.exception("Error setting MaBoSS initial state")
        raise RuntimeError(f"Error setting MaBoSS initial state: {e}") from e


# ---------------------------------------------------------------------------
# Analysis tools
# ---------------------------------------------------------------------------

@tool(
    title="Simulate MaBoSS mutation",
    annotations=_NON_IDEMPOTENT_TOOL,
    structured_output=True,
)
async def simulate_mutation(
    ctx: Context,
    nodes: Annotated[
        NonEmptyString | Annotated[list[NonEmptyString], Field(min_length=1)],
        Field(
            description=(
                "Node name (str) or list of node names to mutate. "
                "E.g. 'FoxO3' or ['FoxO3', 'AKT']."
            )
        ),
    ],
    state: Annotated[
        MutationState
        | Annotated[list[MutationState], Field(min_length=1)],
        Field(
            default="OFF",
            description=(
                "Mutation state(s): 'ON', 'OFF', or 'WT'. "
                "A single string applies to all nodes. "
                "A list must match the length of nodes, e.g. ['OFF', 'ON']."
            ),
        ),
    ] = "OFF",
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session to use. Omit to use the active default session.",
    ),
) -> Annotated[CallToolResult, MaBoSSMutationSimulationResult]:
    """Run an explicitly requested mutant copy and preserve its exact inputs/results.

    Returns a compact run ID and artifact references. Retrieve a bounded table
    with get_simulation_result(run_id=...). The base model and latest ordinary
    run selection are preserved. Scientific execution requires authorization.
    """
    node_list = [nodes] if isinstance(nodes, str) else list(nodes)
    if isinstance(state, str):
        state_list = [state] * len(node_list)
    else:
        state_list = list(state)
        if len(state_list) != len(node_list):
            raise ValueError("Length of 'state' must match length of 'nodes'.")

    valid_states = {"ON", "OFF", "WT"}
    for mutation_state in state_list:
        if mutation_state not in valid_states:
            raise ValueError(
                f"Invalid mutation state '{mutation_state}'. "
                f"Must be one of {valid_states}."
            )

    await ctx.report_progress(0, 3)
    await ctx.report_progress(1, 3)

    def run_mutation_locked():
        with session_manager.session_scope(session_id):
            sess = ensure_session(session_id)
            if sess.sim is None:
                raise RuntimeError(
                    "No MaBoSS simulation has been built yet. "
                    "Call bnet_to_bnd_and_cfg then build_simulation first."
                )
            try:
                logger.info("Running mutant simulation")
                mutated_simulation = sess.sim.copy()
                for node, mutation_state in zip(
                    node_list,
                    state_list,
                    strict=True,
                ):
                    mutated_simulation.mutate(node, mutation_state)
                    logger.info(
                        "Applied mutation: %s -> %s",
                        node,
                        mutation_state,
                    )
                revision = model_revision(sess.sim, sess.readouts)
                config = effective_configuration(sess, mutated_simulation)
                art_dir = get_artifact_dir(_SERVER_ROOT, sess.session_id)
                mutation_result = None
                try:
                    mutation_result = mutated_simulation.run()
                    if getattr(mutation_result, "_err", 0):
                        raise RuntimeError(f"Engine exit code {mutation_result._err}: {getattr(mutation_result, 'error_detail', '')}")
                except Exception as exc:
                    diagnostic = record_failure(art_dir, sess.session_id, "engine", exc, revision, mutation_result)
                    raise RuntimeError(f"Mutant engine failed; incomplete record: {diagnostic}: {exc}") from exc
                try:
                    manifest_path = _preserve_simulation_result(
                        mutation_result, art_dir, sess.session_id, model_revision=revision,
                        kind="mutant", mutations=dict(zip(node_list, state_list, strict=True)),
                        configuration=config)
                except Exception as exc:
                    diagnostic = record_failure(art_dir, sess.session_id, "persistence", exc, revision, mutation_result)
                    raise RuntimeError(f"Mutant finished but persistence failed; do not rerun automatically. Incomplete record: {diagnostic}: {exc}") from exc
                manifest = json.loads(manifest_path.read_text())
                return sess.session_id, manifest, manifest_path
            except (ValueError, RuntimeError):
                raise
            except Exception as exc:
                raise RuntimeError(f"Error preparing mutant simulation: {exc}") from exc

    resolved_session_id, manifest, manifest_path = await anyio.to_thread.run_sync(run_mutation_locked)
    await ctx.report_progress(3, 3)
    payload = MaBoSSMutationSimulationResult(
        server="MaBoSS", session_id=resolved_session_id, run_id=manifest["run_id"],
        run_manifest=str(manifest_path), model_revision=manifest["model_revision"],
        mutations=[MaBoSSMutationRecord(node=n, state=v) for n, v in zip(node_list, state_list, strict=True)],
        has_trajectory_data=bool(manifest["state_timepoints"]), trajectory=_scientific_table(pd.DataFrame()))
    return structured_report(f"Mutant run preserved: {manifest['run_id']}. Retrieve bounded results with get_simulation_result(run_id=...).", payload)


@tool(
    title="Visualize MaBoSS trajectories",
    annotations=_NON_IDEMPOTENT_TOOL,
    structured_output=True,
)
def visualize_network_trajectories(
    session_id: Annotated[NonEmptyString | None, Field(description="Owner session; explicit run IDs also work after restart.")] = None,
    until: Annotated[float | None, Field(gt=0, allow_inf_nan=False, description="Maximum time displayed; omitted uses available data within plotting bounds.")] = None,
    run_id: Annotated[NonEmptyString | None, Field(description="Exact saved run ID; omit for latest ordinary run in the active session.")] = None,
) -> Annotated[CallToolResult, MaBoSSTrajectoryPlotResult]:
    """Plot a saved run without engine execution; retain every plot variant.

    At most 1000 timepoints and 100 state columns are rendered; full data remain saved.
    """
    if run_id is None:
        with session_manager.session_scope(session_id):
            sess = ensure_session(session_id)
            run_id, session_id = sess.latest_run_id, sess.session_id
        if run_id is None:
            raise RuntimeError("No simulation has been run yet with durable preservation; supply run_id or run a simulation first")
    manifest, path = load_run(_SERVER_ROOT, run_id, session_id)
    table = run_table(manifest, path, limit=1001).iloc[:, :100]
    truncated_time = len(table) > 1000
    table = table.head(1000)
    if until is not None:
        table = table.loc[table.index <= until]
    if table.empty:
        raise ValueError("No data in the requested time range")
    fig, axes = plt.subplots(figsize=(10, 6))
    try:
        table.plot(ax=axes)
        axes.set_xlabel("Time")
        axes.set_ylabel("State probability")
        fig.tight_layout()
        with io.BytesIO() as buffer:
            fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            png = buffer.getvalue()
        art_dir = get_artifact_dir(_SERVER_ROOT, manifest["session_id"])
        output_path = safe_artifact_path(art_dir, run_id + ".plot_" + __import__("uuid").uuid4().hex + ".png")
        metadata = {"run_id": run_id, "session_id": manifest["session_id"],
                    "model_revision": manifest.get("model_revision"), "until": until,
                    "max_timepoints": 1000, "max_states": 100,
                    "image_sha256": __import__("hashlib").sha256(png).hexdigest()}
        created = []
        try:
            with tempfile.TemporaryDirectory(prefix=".maboss-plot-", dir=art_dir) as staging:
                image_path = Path(staging) / output_path.name
                image_path.write_bytes(png)
                metadata_path = image_path.with_suffix(".json")
                metadata_path.write_text(json.dumps(metadata))
                for source, target in ((image_path, output_path), (metadata_path, output_path.with_suffix(".json"))):
                    _link_artifact_without_overwrite(source, target)
                    created.append(target)
        except Exception:
            _rollback_artifacts(created)
            raise
        payload = MaBoSSTrajectoryPlotResult(
            server="MaBoSS", session_id=manifest["session_id"], run_id=run_id,
            run_manifest=str(path), model_revision=manifest.get("model_revision"),
            until=until, time_window="full" if until is None and not truncated_time else "bounded",
            image_file=artifact_file_summary(output_path, session_id=manifest["session_id"]))
        return CallToolResult(content=[TextContent(type="text", text=f"Saved run {run_id}; plot capped at 1000 timepoints and 100 states: {output_path}"),
                                      Image(data=png, format="png").to_image_content()],
                              structured_content=payload.model_dump(mode="json"))
    finally:
        plt.close(fig)


@tool(
    title="Get MaBoSS simulation result",
    annotations=_READ_ONLY_TOOL,
    structured_output=True,
)
def get_simulation_result(
    session_id: Annotated[NonEmptyString | None, Field(description="Owner session; explicit run IDs also work after restart.")] = None,
    run_id: Annotated[NonEmptyString | None, Field(description="Exact saved run ID; omit for latest ordinary run in the active session.")] = None,
    offset: Annotated[int, Field(ge=0, description="Timepoint offset when final_only is false.")] = 0,
    limit: Annotated[int, Field(ge=1, le=100, description="Maximum returned timepoints.")] = 25,
    final_only: Annotated[bool, Field(description="Return just the final snapshot; false enables paged trajectories.")] = True,
) -> Annotated[CallToolResult, MaBoSSSimulationResult]:
    """Read a bounded saved run, including after restart. Omit run_id for latest ordinary run.

    Final snapshot is the default. Historical IDs never fall back to another run.
    Tables are capped at 100 state columns; full tables remain in artifacts.
    """
    if run_id is None:
        with session_manager.session_scope(session_id):
            sess = ensure_session(session_id)
            run_id, session_id = sess.latest_run_id, sess.session_id
        if run_id is None:
            raise RuntimeError("No preserved ordinary simulation; call run_simulation first or supply a run_id")
    manifest, path = load_run(_SERVER_ROOT, run_id, session_id)
    table = run_table(manifest, path, offset=offset, limit=limit, final=final_only).iloc[:, :100]
    payload = MaBoSSSimulationResult(
        server="MaBoSS", session_id=manifest["session_id"], run_id=run_id,
        run_manifest=str(path), model_revision=manifest.get("model_revision"),
        has_trajectory_data=not table.empty, trajectory=_scientific_table(table))
    return structured_report(f"Saved run {run_id}; up to 100 state columns. Full tables: {path}", payload)


# ---------------------------------------------------------------------------
# Housekeeping tools
# ---------------------------------------------------------------------------

@tool(
    title="List MaBoSS generated files",
    annotations=_READ_ONLY_TOOL,
    structured_output=True,
)
def list_generated_files(
    session_id: NonEmptyString | None = Field(
        default=None,
        description=(
            "Session whose artifact files to list. "
            "Omit for the active default session. Pass 'all' to list every session."
        ),
    ),
) -> Annotated[CallToolResult, MaBoSSArtifactFileListResult]:
    """List all artifact files (BND, CFG, PNG, …) for a session or across all sessions."""
    if session_id == "all":
        files = list_artifacts(_SERVER_ROOT, session_id=None)
        resolved_session_id = None
        scope = "all"
    else:
        if session_id is not None and session_manager.get_session(session_id) is None:
            resolved_session_id = session_id
            files = list_artifacts(_SERVER_ROOT, session_id=session_id)
        else:
            with session_manager.session_scope(session_id):
                sess = ensure_session(session_id)
                files = list_artifacts(_SERVER_ROOT, session_id=sess.session_id)
                resolved_session_id = sess.session_id
        scope = "session"

    if not files:
        text = "No artifact files found."
    else:
        text = "## Generated artifact files\n\n" + "\n".join(
            f"- {file_path}" for file_path in files
        )
    payload = MaBoSSArtifactFileListResult(
        server="MaBoSS",
        scope=scope,
        session_id=resolved_session_id,
        count=len(files),
        files=[
            artifact_file_summary(
                file_path,
                session_id=(
                    resolved_session_id
                    if resolved_session_id is not None
                    else file_path.parent.name
                ),
            )
            for file_path in files
        ],
    )
    return structured_report(text, payload)


@tool(
    title="Clean MaBoSS generated files",
    annotations=_IDEMPOTENT_DESTRUCTIVE_TOOL,
    structured_output=True,
)
@_session_locked
def clean_generated_files(
    session_id: NonEmptyString | None = Field(
        default=None,
        description="Session whose artifact files to remove. Omit to use the active default session.",
    ),
) -> Annotated[CallToolResult, MaBoSSArtifactCleanupResult]:
    """Remove all artifact files (BND, CFG, PNG, …) for the given session."""
    sess = ensure_session(session_id)
    try:
        count = clean_artifacts(_SERVER_ROOT, sess.session_id)
        logger.info(
            "Cleaned %s artifact file(s) for session %s",
            count,
            sess.session_id,
        )
        text = f"Removed {count} artifact file(s) for session {sess.session_id}."
        payload = MaBoSSArtifactCleanupResult(
            server="MaBoSS",
            session_id=sess.session_id,
            removed_count=count,
        )
        return structured_report(text, payload)
    except Exception as e:
        logger.exception("Error during cleanup")
        raise RuntimeError(f"Error during cleanup: {e}") from e


if __name__ == "__main__":
    mcp.run()
