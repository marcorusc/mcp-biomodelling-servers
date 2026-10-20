"""Register NeKo's lossless public history without reimplementing its semantics."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from mcp_biomodelling_servers.handoff import handoff_artifact
from mcp_biomodelling_servers.neko_provenance import (
    MAX_NEKO_HISTORY_BYTES,
    NeKoExportBinding,
    NeKoExportEvidence,
    NeKoStateIdentity,
)


def history_dict(network):
    if not callable(getattr(network, "history_dict", None)):
        raise RuntimeError("History export requires the NeKo Wave 2 history API.")
    return network.history_dict()


def export_provenance(network, sess, artifact, *, node_renames=None, projection=None):
    """Write an immutable typed companion; return None for legacy networks."""
    if not callable(getattr(network, "history_dict", None)):
        if "Provenance" in getattr(getattr(network, "edges", None), "columns", []):
            raise RuntimeError("Cannot export curated provenance without the NeKo history API.")
        return None
    evidence = NeKoExportEvidence(
        mcp=NeKoStateIdentity(
            session_id=sess.session_id, network_id=sess.network_id,
            state_id=network.current_state_id, database_name=network.database_name,
        ),
        export=NeKoExportBinding(
            artifact=artifact, node_renames=node_renames or {}, library_projection=projection,
        ),
        history=history_dict(network),
    )
    raw = evidence.model_dump_json(indent=2).encode()
    if len(raw) > MAX_NEKO_HISTORY_BYTES:
        raise ValueError("NeKo provenance exceeds 16 MiB; no handoff was published.")
    path = Path(artifact.path).with_name(f"{Path(artifact.path).stem}.{uuid.uuid4().hex}.history.json")
    created = False
    try:
        with path.open("xb") as handle:
            created = True
            handle.write(raw)
        return handoff_artifact(path, server="NeKo", session_id=sess.session_id, role="neko_history")
    except Exception:
        if created:
            path.unlink(missing_ok=True)
        raise


def export_history(network, sess, directory):
    payload = history_dict(network)
    payload["mcp"] = {"session_id": sess.session_id, "network_id": sess.network_id}
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    path = directory / f"network.{uuid.uuid4().hex}.history.json"
    with path.open("x", encoding="utf-8") as handle:
        try:
            handle.write(raw)
        except Exception:
            path.unlink(missing_ok=True)
            raise
    return path
