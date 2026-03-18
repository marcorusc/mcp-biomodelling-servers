"""Session management for MaBoSS MCP server.

Provides thread-safe, multi-session handling that mirrors the pattern
established in NeKo/session_manager.py and PhysiCell/session_manager.py.

Each session stores:
  - The loaded MaBoSS simulation object (``sim``)
  - The last simulation result (``result``)
  - Paths to the generated .bnd and .cfg files
  - Timestamps for LRU eviction
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------

@dataclass
class MaBoSSSession:
    """Per-session state for one MaBoSS workflow."""

    session_id: str
    sim: object | None = None       # maboss simulation object
    result: object | None = None    # result of the last sim.run()
    bnd_path: str | None = None     # absolute path to .bnd file used to load sim
    cfg_path: str | None = None     # absolute path to .cfg file used to load sim
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_accessed = time.time()

    def set_simulation(self, sim_obj: object, bnd_path: str, cfg_path: str) -> None:
        self.sim = sim_obj
        self.result = None          # reset result when simulation is rebuilt
        self.bnd_path = bnd_path
        self.cfg_path = cfg_path
        self.touch()

    def set_result(self, result_obj: object) -> None:
        self.result = result_obj
        self.touch()

    def clear(self) -> None:
        """Reset session state (keeps session alive, clears sim data)."""
        self.sim = None
        self.result = None
        self.bnd_path = None
        self.cfg_path = None
        self.touch()


# ---------------------------------------------------------------------------
# Session manager
# ---------------------------------------------------------------------------

class MaBoSSSessionManager:
    """Thread-safe manager for MaBoSS sessions."""

    def __init__(self, max_sessions: int = 15) -> None:
        self._sessions: Dict[str, MaBoSSSession] = {}
        self._default_session_id: Optional[str] = None
        self._lock = Lock()
        self._max_sessions = max_sessions

    def _resolve_session_id(self, session_id: str) -> Optional[str]:
        """Resolve an exact or unique prefix session ID to a full UUID."""
        if session_id in self._sessions:
            return session_id

        matches = [sid for sid in self._sessions if sid.startswith(session_id)]
        if len(matches) == 1:
            return matches[0]
        return None

    # -- CRUD ----------------------------------------------------------

    def create_session(self, set_as_default: bool = True) -> str:
        with self._lock:
            if len(self._sessions) >= self._max_sessions:
                # Evict least-recently-used session
                oldest = min(self._sessions.values(), key=lambda s: s.last_accessed)
                del self._sessions[oldest.session_id]
            sid = str(uuid.uuid4())
            self._sessions[sid] = MaBoSSSession(session_id=sid)
            if set_as_default or self._default_session_id is None:
                self._default_session_id = sid
            return sid

    def get_session(self, session_id: Optional[str] = None) -> Optional[MaBoSSSession]:
        with self._lock:
            sid = session_id if session_id is not None else self._default_session_id
            if sid is None:
                return None
            resolved_id = self._resolve_session_id(sid)
            if resolved_id is None:
                return None
            sess = self._sessions.get(resolved_id)
            if sess:
                sess.touch()
            return sess

    def list_sessions(self) -> Dict[str, dict]:
        with self._lock:
            return {
                sid: {
                    "has_simulation": s.sim is not None,
                    "has_result": s.result is not None,
                    "bnd_path": s.bnd_path,
                    "cfg_path": s.cfg_path,
                    "created_at": s.created_at,
                    "last_accessed": s.last_accessed,
                    "is_default": sid == self._default_session_id,
                }
                for sid, s in self._sessions.items()
            }

    def set_default(self, session_id: str) -> bool:
        with self._lock:
            resolved_id = self._resolve_session_id(session_id)
            if resolved_id is not None:
                self._default_session_id = resolved_id
                return True
            return False

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            resolved_id = self._resolve_session_id(session_id)
            if resolved_id in self._sessions:
                del self._sessions[resolved_id]
                if self._default_session_id == resolved_id:
                    self._default_session_id = next(iter(self._sessions), None)
                return True
            return False

    def get_default_session_id(self) -> Optional[str]:
        return self._default_session_id


# ---------------------------------------------------------------------------
# Module-level singleton + helpers
# ---------------------------------------------------------------------------

session_manager = MaBoSSSessionManager()


def ensure_session(session_id: Optional[str] = None) -> MaBoSSSession:
    """Return the requested session, auto-creating a default if none exists."""
    sess = session_manager.get_session(session_id)
    if sess is None:
        new_id = session_manager.create_session(set_as_default=True)
        result = session_manager.get_session(new_id)
        assert result is not None
        return result
    return sess

