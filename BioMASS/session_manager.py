"""Locked session state with durable snapshots and explicit retirement."""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

from mcp_biomodelling_servers.artifact_manager import (
    get_artifact_dir,
    write_session_meta,
)

from .outputs import ModelDocument


@dataclass
class BioMASSSession:
    session_id: str
    document: ModelDocument = field(default_factory=ModelDocument)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    lock: RLock = field(default_factory=RLock, repr=False)


class BioMASSSessionManager:
    def __init__(self, root: Path, max_sessions: int = 15):
        self.root = root
        self.max_sessions = max_sessions
        self.sessions: dict[str, BioMASSSession] = {}
        self.default: str | None = None
        self.lock = RLock()

    def directory(self, sid: str) -> Path:
        path = self.root / "artifacts" / sid
        if path.is_symlink() or (self.root / "artifacts").is_symlink():
            raise ValueError("Symlinked artifact directories are not supported.")
        path.resolve().relative_to(self.root.resolve())
        return get_artifact_dir(self.root, sid)

    def create(
        self, label: str | None = None, set_as_default: bool = True
    ) -> BioMASSSession:
        with self.lock:
            if len(self.sessions) >= self.max_sessions:
                raise RuntimeError(
                    "Session limit reached. Close an idle session before creating another."
                )
            sess = BioMASSSession(str(uuid.uuid4()))
            self.directory(sess.session_id)
            write_session_meta(
                self.root, sess.session_id, server_name="BioMASS", label=label
            )
            self.save(sess, sess.document)
            self.sessions[sess.session_id] = sess
            if set_as_default or self.default is None:
                self.default = sess.session_id
            return sess

    @contextmanager
    def use(self, sid: str | None):
        with self.lock:
            sid = sid or self.default
            if sid not in self.sessions:
                raise ValueError(
                    "Unknown or closed BioMASS session. Create or restore a session first."
                )
            sess = self.sessions[sid]
        with sess.lock:
            with self.lock:
                if self.sessions.get(sid) is not sess:
                    raise ValueError(
                        "Session was closed while waiting for an operation."
                    )
            sess.last_accessed = time.time()
            yield sess

    def save(self, sess: BioMASSSession, document: ModelDocument) -> None:
        directory = self.directory(sess.session_id)
        temporary = directory / "session.json.tmp"
        if temporary.is_symlink() or (directory / "session.json").is_symlink():
            raise ValueError("Symlinked session snapshots are not supported.")
        temporary.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(directory / "session.json")
        sess.document = document

    def close(self, sid: str | None) -> str:
        with self.use(sid) as sess:
            with self.lock:
                del self.sessions[sess.session_id]
                if self.default == sess.session_id:
                    self.default = next(iter(self.sessions), None)
            return sess.session_id

    def restore(self, sid: str) -> BioMASSSession:
        if str(uuid.UUID(sid)) != sid:
            raise ValueError("Restore requires a complete session UUID.")
        with self.lock:
            if sid in self.sessions:
                raise ValueError("Session is already active.")
            if len(self.sessions) >= self.max_sessions:
                raise RuntimeError("Session limit reached.")
            directory = self.root / "artifacts" / sid
            if directory.is_symlink() or (self.root / "artifacts").is_symlink():
                raise ValueError("Symlinked artifact directories are not supported.")
            if any(
                (directory / name).is_symlink()
                for name in ("session_meta.json", "session.json")
            ):
                raise ValueError("Symlinked session snapshots are not supported.")
            metadata = json.loads((directory / "session_meta.json").read_text())
            if metadata.get("server") != "BioMASS":
                raise ValueError("Not a BioMASS artifact session.")
            document = ModelDocument.model_validate_json(
                (directory / "session.json").read_bytes()
            )
            sess = BioMASSSession(sid, document)
            self.sessions[sid] = sess
            self.default = sid
            return sess


session_manager = BioMASSSessionManager(Path(__file__).parent)
