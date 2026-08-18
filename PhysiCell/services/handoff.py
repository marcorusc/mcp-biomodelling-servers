"""Transactional MaBoSS-to-PhysiCell handoff import orchestration."""

import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from mcp_biomodelling_servers.artifact_manager import (
    get_artifact_dir,
    safe_artifact_path,
)
from mcp_biomodelling_servers.handoff import (
    MaBoSSToPhysiCellHandoffManifest,
    NeKoToMaBoSSHandoffManifest,
    PhysiCellHandoffImportResult,
    bnd_node_names,
    handoff_artifact,
    load_handoff_manifest,
    verify_handoff_artifact,
    verify_handoff_manifest,
)

from ..session_manager import MaBoSSContext, SessionState
from .resource_views import mapping_at


def import_maboss_handoff_transaction(
    *,
    session: SessionState,
    manifest_path: str,
    artifact_prefix: str,
    replace_existing: bool,
    server_root: Path,
    require_unused_paths: Callable[[list[Path]], None],
    copy_verified_artifact: Callable[[Any, Path], None],
    link_artifact_without_overwrite: Callable[[Path, Path], None],
    rollback_artifacts: Callable[[list[Path]], None],
    physiboss_tracking: Callable[
        [Any],
        tuple[list[str], int, int, int, int],
    ],
) -> tuple[str, PhysiCellHandoffImportResult]:
    """Verify, publish, and attach one complete MaBoSS handoff atomically."""
    loaded_manifest = load_handoff_manifest(
        manifest_path,
        expected_handoff_type="maboss-to-physicell",
        verify_artifacts=True,
    )
    if not isinstance(loaded_manifest, MaBoSSToPhysiCellHandoffManifest):
        raise ValueError(
            "The supplied handoff is not a MaBoSS-to-PhysiCell manifest."
        )

    source_manifest_path = Path(manifest_path).resolve()
    source_manifest_file = handoff_artifact(
        source_manifest_path,
        server="MaBoSS",
        session_id=loaded_manifest.source.session_id,
        role="parent_manifest",
    )
    stored_nodes = bnd_node_names(loaded_manifest.bnd_file.path)
    if stored_nodes != loaded_manifest.network.nodes:
        raise ValueError(
            "The MaBoSS BND node order does not match the handoff manifest."
        )

    neko_manifest = None
    if loaded_manifest.parent_manifest is not None:
        loaded_parent = load_handoff_manifest(
            loaded_manifest.parent_manifest.path,
            expected_handoff_type="neko-to-maboss",
            verify_artifacts=True,
        )
        if not isinstance(loaded_parent, NeKoToMaBoSSHandoffManifest):
            raise ValueError(
                "The MaBoSS handoff parent is not a NeKo manifest."
            )
        if (
            not loaded_manifest.lineage
            or loaded_parent.source != loaded_manifest.lineage[0]
        ):
            raise ValueError(
                "The NeKo parent provenance does not match MaBoSS lineage."
            )
        if set(loaded_parent.network.nodes) != set(
            loaded_manifest.network.nodes
        ):
            raise ValueError(
                "The NeKo and MaBoSS handoff node sets do not match."
            )
        if set(loaded_parent.network.output_nodes) != set(
            loaded_manifest.network.output_nodes
        ):
            raise ValueError(
                "The NeKo and MaBoSS output-node selections do not match."
            )
        neko_manifest = loaded_parent

    target_cell_type = loaded_manifest.target.cell_type
    try:
        cell_types = session.config.cell_types.get_cell_types()
    except Exception as exc:
        raise RuntimeError(
            f"Could not inspect PhysiCell cell types: {exc}"
        ) from exc
    if not isinstance(cell_types, Mapping):
        raise TypeError(
            "PhysiCell cell_types.get_cell_types() did not return a mapping."
        )
    if target_cell_type not in cell_types:
        available = ", ".join(str(name) for name in cell_types) or "none"
        raise ValueError(
            f"Target cell type {target_cell_type!r} is not configured. "
            f"Available cell types: {available}."
        )

    target_data = cell_types[target_cell_type]
    target_mapping = target_data if isinstance(target_data, Mapping) else {}
    existing_intracellular = mapping_at(
        target_mapping,
        "phenotype",
        "intracellular",
    )
    replaced_existing = bool(existing_intracellular)
    if replaced_existing and not replace_existing:
        raise ValueError(
            f"Cell type {target_cell_type!r} already has an intracellular "
            "model. Set replace_existing=true to replace it explicitly."
        )

    art_dir = get_artifact_dir(server_root, session.session_id)
    destinations: dict[str, Path] = {
        "manifest": safe_artifact_path(
            art_dir,
            f"{artifact_prefix}.handoff.json",
        ),
        "bnd": safe_artifact_path(art_dir, f"{artifact_prefix}.bnd"),
        "cfg": safe_artifact_path(art_dir, f"{artifact_prefix}.cfg"),
    }
    if loaded_manifest.simulation.result_file is not None:
        destinations["result"] = safe_artifact_path(
            art_dir,
            f"{artifact_prefix}.result.csv",
        )
    if neko_manifest is not None:
        destinations["neko_manifest"] = safe_artifact_path(
            art_dir,
            f"{artifact_prefix}.neko.handoff.json",
        )
        destinations["bnet"] = safe_artifact_path(
            art_dir,
            f"{artifact_prefix}.neko.bnet",
        )
    require_unused_paths(list(destinations.values()))

    try:
        candidate_config = session.config.copy()
    except Exception as exc:
        raise RuntimeError(
            f"Could not copy the PhysiCell configuration for import: {exc}"
        ) from exc
    try:
        candidate_config.physiboss.add_intracellular_model(
            cell_type_name=target_cell_type,
            model_type="maboss",
            bnd_filename=str(destinations["bnd"]),
            cfg_filename=str(destinations["cfg"]),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not attach the MaBoSS model to {target_cell_type!r}: {exc}"
        ) from exc

    tracking = physiboss_tracking(candidate_config)
    created_paths: list[Path] = []
    try:
        with tempfile.TemporaryDirectory(
            dir=art_dir,
            prefix=".maboss-handoff-import-",
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            temporary_paths = {
                key: temporary_root / destination.name
                for key, destination in destinations.items()
            }
            sources = {
                "manifest": source_manifest_file,
                "bnd": loaded_manifest.bnd_file,
                "cfg": loaded_manifest.cfg_file,
            }
            if loaded_manifest.simulation.result_file is not None:
                sources["result"] = loaded_manifest.simulation.result_file
            if neko_manifest is not None:
                assert loaded_manifest.parent_manifest is not None
                sources["neko_manifest"] = loaded_manifest.parent_manifest
                sources["bnet"] = neko_manifest.bnet_file

            for key, source in sources.items():
                copy_verified_artifact(
                    source,
                    temporary_paths[key],
                )

            reloaded_manifest = load_handoff_manifest(
                source_manifest_path,
                expected_handoff_type="maboss-to-physicell",
                verify_artifacts=True,
            )
            if reloaded_manifest != loaded_manifest:
                raise RuntimeError(
                    "The MaBoSS handoff changed while it was imported."
                )
            verify_handoff_artifact(source_manifest_file)
            verify_handoff_manifest(loaded_manifest)
            if neko_manifest is not None:
                assert loaded_manifest.parent_manifest is not None
                reloaded_parent = load_handoff_manifest(
                    loaded_manifest.parent_manifest.path,
                    expected_handoff_type="neko-to-maboss",
                    verify_artifacts=True,
                )
                if reloaded_parent != neko_manifest:
                    raise RuntimeError(
                        "The NeKo parent handoff changed while it was imported."
                    )

            for key, destination in destinations.items():
                link_artifact_without_overwrite(
                    temporary_paths[key],
                    destination,
                )
                created_paths.append(destination)

        manifest_snapshot_file = handoff_artifact(
            destinations["manifest"],
            server="PhysiCell",
            session_id=session.session_id,
            role="parent_manifest",
        )
        bnd_file = handoff_artifact(
            destinations["bnd"],
            server="PhysiCell",
            session_id=session.session_id,
            role="maboss_bnd",
        )
        cfg_file = handoff_artifact(
            destinations["cfg"],
            server="PhysiCell",
            session_id=session.session_id,
            role="maboss_cfg",
        )
        result_file = (
            handoff_artifact(
                destinations["result"],
                server="PhysiCell",
                session_id=session.session_id,
                role="maboss_result",
            )
            if "result" in destinations
            else None
        )
        neko_manifest_file = (
            handoff_artifact(
                destinations["neko_manifest"],
                server="PhysiCell",
                session_id=session.session_id,
                role="parent_manifest",
            )
            if "neko_manifest" in destinations
            else None
        )
        bnet_file = (
            handoff_artifact(
                destinations["bnet"],
                server="PhysiCell",
                session_id=session.session_id,
                role="neko_bnet",
            )
            if "bnet" in destinations
            else None
        )
        payload = PhysiCellHandoffImportResult(
            server="PhysiCell",
            session_id=session.session_id,
            source_manifest_file=source_manifest_file,
            source_manifest=loaded_manifest,
            manifest_snapshot_file=manifest_snapshot_file,
            bnd_file=bnd_file,
            cfg_file=cfg_file,
            result_file=result_file,
            neko_manifest=neko_manifest,
            neko_manifest_file=neko_manifest_file,
            bnet_file=bnet_file,
            target_cell_type=target_cell_type,
            nodes=list(loaded_manifest.network.nodes),
            output_nodes=list(loaded_manifest.network.output_nodes),
            replaced_existing=replaced_existing,
            context_count=(
                len(session.maboss_contexts)
                + (0 if target_cell_type in session.maboss_contexts else 1)
            ),
        )
    except Exception:
        rollback_artifacts(created_paths)
        raise

    context = MaBoSSContext(
        model_name=Path(loaded_manifest.bnd_file.path).stem,
        bnd_file_path=str(destinations["bnd"]),
        cfg_file_path=str(destinations["cfg"]),
        available_nodes=list(loaded_manifest.network.nodes),
        output_nodes=list(loaded_manifest.network.output_nodes),
        simulation_results=(
            loaded_manifest.simulation.simulation_summary or ""
        ),
        target_cell_type=target_cell_type,
        biological_context=loaded_manifest.biological_context or "",
        source_manifest_path=str(source_manifest_path),
        local_manifest_path=str(destinations["manifest"]),
        source_session_id=loaded_manifest.source.session_id,
        result_file_path=(
            str(destinations["result"])
            if "result" in destinations
            else ""
        ),
        simulation_parameters=dict(loaded_manifest.simulation.parameters),
        neko_session_id=(
            neko_manifest.source.session_id
            if neko_manifest is not None
            else ""
        ),
        neko_manifest_path=(
            loaded_manifest.parent_manifest.path
            if loaded_manifest.parent_manifest is not None
            else ""
        ),
        local_neko_manifest_path=(
            str(destinations["neko_manifest"])
            if "neko_manifest" in destinations
            else ""
        ),
        local_bnet_path=(
            str(destinations["bnet"])
            if "bnet" in destinations
            else ""
        ),
    )
    session.publish_physiboss_import(
        config=candidate_config,
        context=context,
        model_names=tracking[0],
        settings_count=tracking[1],
        input_links_count=tracking[2],
        output_links_count=tracking[3],
        mutations_count=tracking[4],
    )

    replacement_text = (
        "replaced the previous intracellular model"
        if replaced_existing
        else "attached a new intracellular model"
    )
    lineage_text = (
        f"NeKo session {neko_manifest.source.session_id}"
        if neko_manifest is not None
        else "standalone MaBoSS model"
    )
    text = (
        "MaBoSS handoff imported into PhysiCell successfully.\n"
        f"  Session: {session.session_id}\n"
        f"  Target cell type: {target_cell_type}\n"
        f"  Action: {replacement_text}\n"
        f"  BND: {destinations['bnd']}\n"
        f"  CFG: {destinations['cfg']}\n"
        f"  Boolean nodes: {len(loaded_manifest.network.nodes)}\n"
        f"  Output nodes: {', '.join(loaded_manifest.network.output_nodes)}\n"
        f"  Lineage: {lineage_text}\n\n"
        "Next: configure PhysiBoSS timing, then add biologically justified "
        "input and output links."
    )
    return text, payload
