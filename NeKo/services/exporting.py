"""Transactional file-export services for NeKo network artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any


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

    try:
        with tempfile.TemporaryDirectory(
            dir=destination.parent,
            prefix=".neko-bnet-",
        ) as temporary_directory:
            temporary_prefix = Path(temporary_directory) / destination.stem
            try:
                export_factory(network).export_bnet(str(temporary_prefix))
            except Exception as exc:
                raise RuntimeError(f"Error exporting BNET: {exc}") from exc
            generated_bnets = sorted(Path(temporary_directory).glob("*.bnet"))
            if not generated_bnets:
                raise FileNotFoundError(
                    "NeKo did not create a BNET file in the temporary export "
                    f"directory {temporary_directory}."
                )
            if len(generated_bnets) > 1:
                generated_names = ", ".join(
                    path.name for path in generated_bnets
                )
                raise RuntimeError(
                    "NeKo generated multiple BNET models "
                    f"({generated_names}); this export requires exactly one model."
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
