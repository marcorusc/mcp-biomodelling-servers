"""Session management for NeKo MCP server.
Provides lightweight multi-network handling, caching and verbosity control.

Phase 1/2 implementation: supports a default session plus user-created sessions.
Future extensions: TTL cleanup, persistence, concurrent locks per session.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Optional, Literal
import time

# Verbosity levels
Verbosity = Literal["summary", "preview", "full"]
DEFAULT_VERBOSITY: Verbosity = "summary"
ALLOWED_VERBOSITY = {"summary", "preview", "full"}

@dataclass
class NeKoSession:
    session_id: str
    network: object | None = None  # Will hold a neko.core.network.Network instance
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    # Caching of converted edge list (genesymbol)
    _edges_cache: object | None = None  # pandas.DataFrame
    _edges_cache_dirty: bool = True
    # Default creation parameters (user can override later)
    default_params: dict = field(default_factory=lambda: {
        "max_len": 2,
        "algorithm": "bfs",
        "only_signed": True,
        "connect_with_bias": False,
        "consensus": True,
        "database": "omnipath"
    })

    def touch(self):
        self.last_accessed = time.time()

    def invalidate_edges_cache(self):
        self._edges_cache_dirty = True

    def set_network(self, network_obj):
        self.network = network_obj
        self.invalidate_edges_cache()
        # Reset created_at? Keep original; update last_accessed
        self.touch()

    def update_default_params(self, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                self.default_params[k] = v
        self.touch()

    def get_completion_params(self):
        keys = ["max_len", "algorithm", "only_signed", "connect_with_bias", "consensus"]
        # Map max_len to maxlen argument expected by Network.complete_connection
        params = self.default_params.copy()
        return dict(
            maxlen=params.get("max_len", 2),
            algorithm=params.get("algorithm", "bfs"),
            only_signed=params.get("only_signed", True),
            connect_with_bias=params.get("connect_with_bias", False),
            consensus=params.get("consensus", True)
        )

    def get_edges_df(self):
        """Return cached edges DataFrame (gene symbol converted)."""
        if self.network is None:
            return None
        if self._edges_cache is None or self._edges_cache_dirty:
            try:
                df = self.network.convert_edgelist_into_genesymbol()
            except Exception:
                df = None
            self._edges_cache = df
            self._edges_cache_dirty = False
        return self._edges_cache

class NeKoSessionManager:
    def __init__(self, max_sessions: int = 15):
        self._sessions: Dict[str, NeKoSession] = {}
        self._default_session_id: Optional[str] = None
        self._lock = Lock()
        self._max_sessions = max_sessions
        self._counter = 0  # simple incremental id for readability

    def create_session(self, set_as_default: bool = True) -> str:
        with self._lock:
            if len(self._sessions) >= self._max_sessions:
                # Remove oldest
                oldest = min(self._sessions.values(), key=lambda s: s.last_accessed)
                del self._sessions[oldest.session_id]
            self._counter += 1
            sid = f"session_{self._counter}"
            self._sessions[sid] = NeKoSession(session_id=sid)
            if set_as_default or self._default_session_id is None:
                self._default_session_id = sid
            return sid

    def get_session(self, session_id: Optional[str] = None) -> Optional[NeKoSession]:
        with self._lock:
            if session_id is None:
                session_id = self._default_session_id
            if session_id is None:
                return None
            sess = self._sessions.get(session_id)
            if sess:
                sess.touch()
            return sess

    def list_sessions(self) -> Dict[str, dict]:
        with self._lock:
            return {sid: {"has_network": s.network is not None,
                          "nodes": len(s.network.nodes) if s.network else 0,
                          "edges": len(s.network.edges) if s.network else 0,
                          "last_accessed": s.last_accessed,
                          "created_at": s.created_at} for sid, s in self._sessions.items()}

    def set_default(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                self._default_session_id = session_id
                return True
            return False

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                if self._default_session_id == session_id:
                    self._default_session_id = next(iter(self._sessions), None)
                return True
            return False

    def get_default_session_id(self) -> Optional[str]:
        return self._default_session_id

session_manager = NeKoSessionManager()

def ensure_session(session_id: Optional[str]) -> NeKoSession:
    sess = session_manager.get_session(session_id)
    if sess is None:
        # Auto-create a default if none exists yet
        new_id = session_manager.create_session(set_as_default=True)
        return session_manager.get_session(new_id)
    return sess

# Helper for verbosity validation

def normalize_verbosity(v: Optional[str]) -> Verbosity:
    if not v:
        return DEFAULT_VERBOSITY
    v_lower = v.lower()
    return v_lower if v_lower in ALLOWED_VERBOSITY else DEFAULT_VERBOSITY
