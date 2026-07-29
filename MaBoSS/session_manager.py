"""Session management for the MaBoSS MCP server.

The MCP SDK v2 may run synchronous handlers concurrently in worker threads.
Registry operations and accesses to each mutable pyMaBoSS model therefore use
separate locks: the manager lock protects session lifecycle metadata, while a
per-session lock serializes model, result, and artifact operations.

Each session stores:
  - The loaded MaBoSS simulation object (``sim``)
  - The last simulation result (``result``)
  - Paths to the generated .bnd and .cfg files
  - Timestamps for LRU eviction
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Condition, RLock

# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------


@dataclass
class MaBoSSSession:
    """Per-session state for one MaBoSS workflow."""

    session_id: str
    sim: object | None = None  # maboss simulation object
    result: object | None = None  # result of the last sim.run()
    bnd_path: str | None = None  # absolute path to .bnd file used to load sim
    cfg_path: str | None = None  # absolute path to .cfg file used to load sim
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    _operation_lock: RLock = field(
        default_factory=RLock,
        init=False,
        repr=False,
        compare=False,
    )
    # These lifecycle fields are protected by MaBoSSSessionManager._condition.
    _lease_count: int = field(default=0, init=False, repr=False, compare=False)
    _retired: bool = field(default=False, init=False, repr=False, compare=False)

    def touch(self) -> None:
        with self._operation_lock:
            self._touch_unlocked()

    def _touch_unlocked(self) -> None:
        self.last_accessed = time.time()

    def set_simulation(self, sim_obj: object, bnd_path: str, cfg_path: str) -> None:
        with self._operation_lock:
            self.sim = sim_obj
            self.result = None  # reset result when simulation is rebuilt
            self.bnd_path = bnd_path
            self.cfg_path = cfg_path
            self._touch_unlocked()

    def set_result(self, result_obj: object) -> None:
        with self._operation_lock:
            self.result = result_obj
            self._touch_unlocked()

    def clear(self) -> None:
        """Reset session state (keeps session alive, clears sim data)."""
        with self._operation_lock:
            self.sim = None
            self.result = None
            self.bnd_path = None
            self.cfg_path = None
            self._touch_unlocked()

    def snapshot(self, *, is_default: bool) -> dict:
        """Return a consistent, serialization-safe view of this session."""
        with self._operation_lock:
            return {
                "has_simulation": self.sim is not None,
                "has_result": self.result is not None,
                "bnd_path": self.bnd_path,
                "cfg_path": self.cfg_path,
                "created_at": self.created_at,
                "last_accessed": self.last_accessed,
                "is_default": is_default,
            }


_active_session = ContextVar[MaBoSSSession | None](
    "maboss_active_session",
    default=None,
)


# ---------------------------------------------------------------------------
# Session manager
# ---------------------------------------------------------------------------


class MaBoSSSessionManager:
    """Manage MaBoSS sessions and lease their mutable state to handlers."""

    def __init__(self, max_sessions: int = 15) -> None:
        self._sessions: dict[str, MaBoSSSession] = {}
        self._default_session_id: str | None = None
        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._max_sessions = max_sessions

    def _resolve_session_id(self, session_id: str) -> str | None:
        """Resolve an exact or unique prefix session ID to a full UUID."""
        if session_id in self._sessions:
            return session_id

        matches = [sid for sid in self._sessions if sid.startswith(session_id)]
        if len(matches) == 1:
            return matches[0]
        return None

    # -- CRUD ----------------------------------------------------------

    def _retire_session_unlocked(self, session: MaBoSSSession) -> None:
        """Remove a session from the registry and reject future leases."""
        self._sessions.pop(session.session_id, None)
        session._retired = True
        if self._default_session_id == session.session_id:
            self._default_session_id = next(iter(self._sessions), None)

    def _create_session_unlocked(self, *, set_as_default: bool) -> str:
        if len(self._sessions) >= self._max_sessions:
            idle_sessions = [
                session
                for session in self._sessions.values()
                if session._lease_count == 0
            ]
            if not idle_sessions:
                raise RuntimeError(
                    "MaBoSS session limit reached while every session is active. "
                    "Retry after an in-flight operation completes."
                )
            oldest = min(idle_sessions, key=lambda session: session.last_accessed)
            self._retire_session_unlocked(oldest)

        sid = str(uuid.uuid4())
        self._sessions[sid] = MaBoSSSession(session_id=sid)
        if set_as_default or self._default_session_id is None:
            self._default_session_id = sid
        return sid

    def create_session(self, set_as_default: bool = True) -> str:
        with self._condition:
            return self._create_session_unlocked(
                set_as_default=set_as_default,
            )

    @contextmanager
    def create_session_scope(
        self,
        *,
        set_as_default: bool = True,
    ) -> Iterator[MaBoSSSession]:
        """Create and lease a session before another handler can use it."""
        with self._condition:
            session_id = self._create_session_unlocked(
                set_as_default=set_as_default,
            )
            session = self._sessions[session_id]
            session._lease_count += 1
            self._condition.notify_all()

        try:
            with session._operation_lock:
                token = _active_session.set(session)
                try:
                    yield session
                finally:
                    _active_session.reset(token)
        finally:
            with self._condition:
                session._lease_count -= 1
                self._condition.notify_all()

    def ensure_session(self, session_id: str | None = None) -> MaBoSSSession:
        """Resolve a session or atomically create the initial/default session."""
        with self._condition:
            sid = session_id if session_id is not None else self._default_session_id
            resolved_id = self._resolve_session_id(sid) if sid is not None else None
            if resolved_id is not None:
                session = self._sessions[resolved_id]
                session.last_accessed = time.time()
                return session

            new_id = self._create_session_unlocked(set_as_default=True)
            return self._sessions[new_id]

    def get_session(self, session_id: str | None = None) -> MaBoSSSession | None:
        """Return a session for identity checks; use session_scope for model access."""
        with self._condition:
            sid = session_id if session_id is not None else self._default_session_id
            if sid is None:
                return None
            resolved_id = self._resolve_session_id(sid)
            if resolved_id is None:
                return None
            sess = self._sessions.get(resolved_id)
            if sess:
                sess.last_accessed = time.time()
            return sess

    def list_sessions(self) -> dict[str, dict]:
        with self._condition:
            sessions = list(self._sessions.values())
            default_session_id = self._default_session_id
            for session in sessions:
                session._lease_count += 1

        try:
            return {
                session.session_id: session.snapshot(
                    is_default=session.session_id == default_session_id,
                )
                for session in sessions
            }
        finally:
            with self._condition:
                for session in sessions:
                    session._lease_count -= 1
                self._condition.notify_all()

    def set_default(self, session_id: str) -> bool:
        with self._condition:
            resolved_id = self._resolve_session_id(session_id)
            if resolved_id is not None:
                self._default_session_id = resolved_id
                return True
            return False

    def delete_session(self, session_id: str) -> bool:
        """Retire a session and wait for every admitted operation to finish."""
        with self._condition:
            resolved_id = self._resolve_session_id(session_id)
            if resolved_id is None:
                return False

            session = self._sessions[resolved_id]
            self._retire_session_unlocked(session)
            self._condition.notify_all()
            while session._lease_count:
                self._condition.wait()
            return True

    def get_default_session_id(self) -> str | None:
        with self._condition:
            return self._default_session_id

    @contextmanager
    def session_scope(
        self,
        session_id: str | None = None,
    ) -> Iterator[MaBoSSSession]:
        """Lease and exclusively access one session for an atomic operation.

        Admission is protected by the manager condition, but waiting for the
        per-session lock never holds the registry lock. This prevents a queued
        operation on one session from blocking independent sessions.
        """
        with self._condition:
            session = self.ensure_session(session_id)
            session._lease_count += 1
            self._condition.notify_all()

        try:
            with session._operation_lock:
                session._touch_unlocked()
                token = _active_session.set(session)
                try:
                    yield session
                finally:
                    _active_session.reset(token)
        finally:
            with self._condition:
                session._lease_count -= 1
                self._condition.notify_all()


# ---------------------------------------------------------------------------
# Module-level singleton + helpers
# ---------------------------------------------------------------------------

session_manager = MaBoSSSessionManager()


def ensure_session(session_id: str | None = None) -> MaBoSSSession:
    """Return the requested session, auto-creating a default if none exists."""
    active_session = _active_session.get()
    if active_session is not None and (
        session_id is None
        or active_session.session_id == session_id
        or active_session.session_id.startswith(session_id)
    ):
        return active_session
    return session_manager.ensure_session(session_id)
