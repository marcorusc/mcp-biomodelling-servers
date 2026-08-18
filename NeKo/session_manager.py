"""Thread-safe session management for the NeKo MCP server.

The MCP SDK v2 may run synchronous handlers concurrently in worker threads.
Registry operations and accesses to each mutable NeKo network therefore use
separate locks: the manager lock protects session lifecycle metadata, while a
per-session lock serializes network, cache, history, parameter, and artifact
operations.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Condition, RLock
from typing import Any, Literal, cast

logger = logging.getLogger(__name__)

# Verbosity levels
Verbosity = Literal["summary", "preview", "full"]
DEFAULT_VERBOSITY: Verbosity = "summary"
ALLOWED_VERBOSITY = {"summary", "preview", "full"}


@dataclass
class NeKoSession:
    """Per-session state for one NeKo network workflow."""

    session_id: str
    network: object | None = None  # neko.core.network.Network
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    # Cached gene-symbol edge list.
    _edges_cache: object | None = None  # pandas.DataFrame
    _edges_cache_dirty: bool = True
    # None preserves NeKo's unbounded-history default.
    history_max_states: int | None = None
    # Default creation parameters (user can override later).
    default_params: dict = field(
        default_factory=lambda: {
            "max_len": 2,
            "path_policy": "one_shortest",
            "reuse_policy": "discovered_paths",
            "only_signed": True,
            "consensus": True,
            "database": "omnipath",
        }
    )
    _operation_lock: RLock = field(
        default_factory=RLock,
        init=False,
        repr=False,
        compare=False,
    )
    # These lifecycle fields are protected by NeKoSessionManager._condition.
    _lease_count: int = field(default=0, init=False, repr=False, compare=False)
    _retired: bool = field(default=False, init=False, repr=False, compare=False)

    def touch(self) -> None:
        with self._operation_lock:
            self._touch_unlocked()

    def _touch_unlocked(self) -> None:
        self.last_accessed = time.time()

    def invalidate_edges_cache(self) -> None:
        with self._operation_lock:
            self._edges_cache_dirty = True

    def set_network(
        self,
        network_obj: object | None,
        *,
        edges_df: object | None = None,
    ) -> None:
        """Replace the network and optionally seed its validated edge cache."""
        with self._operation_lock:
            self.network = network_obj
            self._edges_cache = edges_df
            self._edges_cache_dirty = network_obj is not None and edges_df is None
            self._touch_unlocked()

    def update_default_params(self, **kwargs: object) -> None:
        with self._operation_lock:
            for key, value in kwargs.items():
                if value is not None:
                    self.default_params[key] = value
            self._touch_unlocked()

    def get_completion_params(self) -> dict[str, object]:
        with self._operation_lock:
            params = self.default_params.copy()
            return {
                "maxlen": params.get("max_len", 2),
                "path_policy": params.get("path_policy", "one_shortest"),
                "reuse_policy": params.get(
                    "reuse_policy",
                    "discovered_paths",
                ),
                "only_signed": params.get("only_signed", True),
                "consensus": params.get("consensus", True),
            }

    def set_history_max_states(self, max_states: int | None) -> None:
        """Persist the history limit used by the current and future network."""
        with self._operation_lock:
            self.history_max_states = max_states
            self._touch_unlocked()

    def get_history_max_states(self) -> int | None:
        with self._operation_lock:
            return self.history_max_states

    def get_edges_df(self) -> object | None:
        """Return the cached gene-symbol edge table for the current network."""
        with self._operation_lock:
            if self.network is None:
                return None
            if self._edges_cache is None or self._edges_cache_dirty:
                try:
                    df = cast(Any, self.network).convert_edgelist_into_genesymbol()
                # NeKo/database adapters do not expose a stable exception contract.
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "Could not build the NeKo edge cache",
                        exc_info=True,
                    )
                    df = None
                self._edges_cache = df
                self._edges_cache_dirty = False
            return self._edges_cache

    def snapshot(self) -> dict[str, object]:
        """Return a consistent, serialization-safe view of this session."""
        with self._operation_lock:
            network = cast(Any, self.network)
            return {
                "has_network": network is not None,
                "nodes": len(network.nodes) if network is not None else 0,
                "edges": len(network.edges) if network is not None else 0,
                "history_max_states": self.history_max_states,
                "last_accessed": self.last_accessed,
                "created_at": self.created_at,
            }


_active_session = ContextVar[NeKoSession | None](
    "neko_active_session",
    default=None,
)


class NeKoSessionManager:
    """Manage NeKo sessions and lease their mutable state to handlers."""

    def __init__(self, max_sessions: int = 15) -> None:
        self._sessions: dict[str, NeKoSession] = {}
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

    def _retire_session_unlocked(self, session: NeKoSession) -> None:
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
                    "NeKo session limit reached while every session is active. "
                    "Retry after an in-flight operation completes."
                )
            oldest = min(
                idle_sessions,
                key=lambda session: session.last_accessed,
            )
            self._retire_session_unlocked(oldest)

        sid = str(uuid.uuid4())
        self._sessions[sid] = NeKoSession(session_id=sid)
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
    ) -> Iterator[NeKoSession]:
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

    def ensure_session(self, session_id: str | None = None) -> NeKoSession:
        """Resolve a session or atomically create the initial/default session."""
        with self._condition:
            sid = (
                session_id
                if session_id is not None
                else self._default_session_id
            )
            resolved_id = (
                self._resolve_session_id(sid)
                if sid is not None
                else None
            )
            if resolved_id is not None:
                session = self._sessions[resolved_id]
                session.last_accessed = time.time()
                return session

            if session_id is not None:
                raise KeyError(f"Unknown NeKo session: {session_id}")

            new_id = self._create_session_unlocked(set_as_default=True)
            return self._sessions[new_id]

    def get_session(
        self,
        session_id: str | None = None,
    ) -> NeKoSession | None:
        """Return a session for identity checks; use session_scope for model access."""
        with self._condition:
            sid = (
                session_id
                if session_id is not None
                else self._default_session_id
            )
            if sid is None:
                return None
            resolved_id = self._resolve_session_id(sid)
            if resolved_id is None:
                return None
            session = self._sessions.get(resolved_id)
            if session is not None:
                session.last_accessed = time.time()
            return session

    def list_sessions(self) -> dict[str, dict[str, object]]:
        with self._condition:
            sessions = list(self._sessions.values())
            for session in sessions:
                session._lease_count += 1

        try:
            return {
                session.session_id: session.snapshot()
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
    ) -> Iterator[NeKoSession]:
        """Lease and exclusively access one session for an atomic operation."""
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

    @contextmanager
    def existing_session_scope(
        self,
        session_id: str,
    ) -> Iterator[NeKoSession]:
        """Lease an existing session without creating a fallback session."""
        with self._condition:
            resolved_id = self._resolve_session_id(session_id)
            if resolved_id is None:
                raise KeyError(f"Unknown NeKo session: {session_id}")
            session = self._sessions[resolved_id]
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


session_manager = NeKoSessionManager()


def ensure_session(session_id: str | None) -> NeKoSession:
    """Return the requested session, auto-creating a default if none exists."""
    active_session = _active_session.get()
    if active_session is not None and (
        session_id is None
        or active_session.session_id == session_id
        or active_session.session_id.startswith(session_id)
    ):
        return active_session
    return session_manager.ensure_session(session_id)


def normalize_verbosity(v: str | None) -> Verbosity:
    if not v:
        return DEFAULT_VERBOSITY
    v_lower = v.lower()
    if v_lower in ALLOWED_VERBOSITY:
        return cast(Verbosity, v_lower)
    return DEFAULT_VERBOSITY
