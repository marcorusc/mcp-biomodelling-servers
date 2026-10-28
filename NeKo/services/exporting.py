"""Transactional file-export services for NeKo network artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _bimodal_edges(exporter: Any) -> list[str]:
    """Return consolidated bimodal source -> target pairs, if exposed."""
    interactions = getattr(exporter, "interactions", None)
    columns = getattr(interactions, "columns", ())
    if not {"source", "target", "Effect"}.issubset(columns):
        return []
    rows = interactions[interactions["Effect"] == "bimodal"]
    return [f"{source} -> {target}" for source, target in zip(rows["source"], rows["target"], strict=True)]


def _multiple_models_message(exporter: Any, generated_bnets: list[Path]) -> str:
    names = ", ".join(path.name for path in generated_bnets)
    bimodal = _bimodal_edges(exporter)
    count = f"{len(bimodal)} bimodal interaction(s)" if bimodal else "Bimodal interactions"
    unresolved = (
        "**Unresolved:** " + ", ".join(bimodal[:20])
        + (f", ... ({len(bimodal) - 20} more)" if len(bimodal) > 20 else "") + "\n\n"
        if bimodal else ""
    )
    return (
        "## BNET Export Failed: Bimodal Interactions Unresolved\n\n"
        "**Issue:** BNET export needs every interaction signed. "
        f"{count} remain, so NeKo generated multiple BNET models ({names}), "
        "one per sign assignment; this export requires exactly one model. A pair "
        "recorded with both signs counts as bimodal. Nothing was exported, and "
        "connectivity is not the cause.\n\n"
        + unresolved
        + "### Resolve them, then export again\n\n"
        "**1. Curate each edge with evidence (keeps the edge and its references):**\n"
        "`edit_interactions` with `operation='set_effect'` and "
        "`effect='stimulation'` or `'inhibition'`, a rationale and evidence.\n\n"
        "**2. Drop every bimodal edge:** `remove_bimodal_interactions()` deletes "
        "them all, can disconnect the network, and discards their references. "
        "Run `analyze_connectivity()` afterwards.\n"
    )


def export_sanitized_bnet(
    network: object,
    destination: Path,
    *,
    overwrite: bool,
    export_factory: Callable[[object], Any],
    sanitizer: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    """Export and sanitize one BNET through a temporary sibling directory."""
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing BNET artifact: {destination}"
        )

    nodes = getattr(network, "nodes", None)
    if nodes is not None and "Genesymbol" in nodes.columns:
        invalid = []
        for index, row in nodes.iterrows():
            label = row.get("Genesymbol")
            if not isinstance(label, str) or not label.strip():
                invalid.append({"row": str(index), "uniprot": str(row.get("Uniprot")), "label": repr(label)})
        if invalid:
            raise ValueError(
                f"BNET export has {len(invalid)} node(s) with missing or invalid labels: {invalid[:20]}. "
                "Inspect these node records and explicitly correct their labels in the source model, then retry. "
                "No automatic renaming was performed."
            )
    try:
        with tempfile.TemporaryDirectory(
            dir=destination.parent,
            prefix=".neko-bnet-",
        ) as temporary_directory:
            temporary_prefix = Path(temporary_directory) / destination.stem
            exporter = export_factory(network)
            try:
                exporter.export_bnet(str(temporary_prefix))
            except Exception as exc:
                raise RuntimeError(f"Error exporting BNET: {exc}") from exc
            generated_bnets = sorted(Path(temporary_directory).glob("*.bnet"))
            if not generated_bnets:
                raise FileNotFoundError(
                    "NeKo did not create a BNET file in the temporary export "
                    f"directory {temporary_directory}."
                )
            if len(generated_bnets) > 1:
                raise RuntimeError(
                    _multiple_models_message(exporter, generated_bnets)
                )
            temporary_bnet = generated_bnets[0]
            companion = temporary_bnet.with_name(temporary_bnet.name + ".neko.json")
            projection = None
            history = None
            if companion.exists():
                payload = json.loads(companion.read_bytes())
                projection = payload["export"]
                history = payload.get("history")
                if projection["sha256"] != hashlib.sha256(temporary_bnet.read_bytes()).hexdigest():
                    raise ValueError("NeKo companion does not match the generated BNET.")
            try:
                sanitizer_result = sanitizer(str(temporary_bnet))
            except Exception as exc:
                raise RuntimeError(f"Error sanitizing BNET: {exc}") from exc
            if projection is not None:
                projection["file"] = destination.name
                projection["sha256"] = hashlib.sha256(temporary_bnet.read_bytes()).hexdigest()
                for identifier in projection.get("identifiers", []):
                    old = identifier["exported_identifier"]
                    identifier["exported_identifier"] = sanitizer_result.get("name_mapping", {}).get(old, old)
                sanitizer_result["library_projection"] = projection
                # The library already serialized this exact source snapshot.
                # Keep it through finalization instead of traversing history again.
                sanitizer_result["library_history"] = history

            if overwrite:
                temporary_bnet.replace(destination)
            else:
                try:
                    os.link(temporary_bnet, destination)
                except FileExistsError as exc:
                    raise FileExistsError(
                        "Refusing to overwrite BNET artifact created "
                        f"concurrently: {destination}"
                    ) from exc
            return sanitizer_result
    except OSError as exc:
        raise RuntimeError(
            f"Could not finalize BNET artifact {destination}: {exc}"
        ) from exc
