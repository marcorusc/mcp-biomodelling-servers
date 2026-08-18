"""Unit tests for MaBoSS/session_manager.py."""

import importlib.util
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event

import pytest


def _load(name: str):
    """Load a session_manager module by server name with a unique module alias."""
    import sys

    module_name = f"{name}_session_manager"
    path = Path(__file__).parent.parent / name / "session_manager.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = (
        mod  # must register before exec so @dataclass resolves __module__
    )
    spec.loader.exec_module(mod)
    return mod


_sm = _load("MaBoSS")
MaBoSSSession = _sm.MaBoSSSession
MaBoSSSessionManager = _sm.MaBoSSSessionManager
ensure_session = _sm.ensure_session


# ---------------------------------------------------------------------------
# MaBoSSSession
# ---------------------------------------------------------------------------


class TestMaBoSSSession:
    def test_initial_state(self):
        s = MaBoSSSession(session_id="s1")
        assert s.session_id == "s1"
        assert s.sim is None
        assert s.result is None
        assert s.bnd_path is None
        assert s.cfg_path is None

    def test_set_simulation_clears_result(self):
        s = MaBoSSSession(session_id="s1")
        s.result = object()  # pretend we had a result
        s.set_simulation("sim_obj", "/a.bnd", "/a.cfg")
        assert s.sim == "sim_obj"
        assert s.result is None  # must be reset
        assert s.bnd_path == "/a.bnd"
        assert s.cfg_path == "/a.cfg"

    def test_set_result(self):
        s = MaBoSSSession(session_id="s1")
        s.set_result("res_obj")
        assert s.result == "res_obj"

    def test_touch_updates_last_accessed(self):
        s = MaBoSSSession(session_id="s1")
        before = s.last_accessed
        time.sleep(0.01)
        s.touch()
        assert s.last_accessed > before

    def test_clear_resets_state(self):
        s = MaBoSSSession(session_id="s1")
        s.set_simulation("sim", "/a.bnd", "/a.cfg")
        s.set_result("res")
        s.clear()
        assert s.sim is None
        assert s.result is None
        assert s.bnd_path is None
        assert s.cfg_path is None


# ---------------------------------------------------------------------------
# MaBoSSSessionManager
# ---------------------------------------------------------------------------


class TestMaBoSSSessionManager:
    def _fresh(self):
        return MaBoSSSessionManager(max_sessions=3)

    def test_create_returns_string_id(self):
        mgr = self._fresh()
        sid = mgr.create_session()
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_created_session_is_default(self):
        mgr = self._fresh()
        sid = mgr.create_session(set_as_default=True)
        assert mgr.get_default_session_id() == sid

    def test_get_default_when_none_returns_none(self):
        mgr = self._fresh()
        assert mgr.get_session() is None

    def test_get_session_by_id(self):
        mgr = self._fresh()
        sid = mgr.create_session()
        sess = mgr.get_session(sid)
        assert sess is not None
        assert sess.session_id == sid

    def test_get_session_by_short_prefix(self):
        mgr = self._fresh()
        sid = mgr.create_session()
        sess = mgr.get_session(sid[:8])
        assert sess is not None
        assert sess.session_id == sid

    def test_get_unknown_session_returns_none(self):
        mgr = self._fresh()
        assert mgr.get_session("nonexistent") is None

    def test_ensure_unknown_explicit_session_raises_without_creating(self):
        mgr = self._fresh()

        with pytest.raises(KeyError, match="Unknown MaBoSS session: nonexistent"):
            mgr.ensure_session("nonexistent")

        assert mgr.list_sessions() == {}

    def test_list_sessions_returns_all(self):
        mgr = self._fresh()
        mgr.create_session()
        mgr.create_session()
        sessions = mgr.list_sessions()
        assert len(sessions) == 2

    def test_set_default(self):
        mgr = self._fresh()
        sid1 = mgr.create_session()
        sid2 = mgr.create_session()
        assert mgr.get_default_session_id() == sid2
        result = mgr.set_default(sid1)
        assert result is True
        assert mgr.get_default_session_id() == sid1

    def test_set_default_by_short_prefix(self):
        mgr = self._fresh()
        sid1 = mgr.create_session()
        _sid2 = mgr.create_session()
        result = mgr.set_default(sid1[:8])
        assert result is True
        assert mgr.get_default_session_id() == sid1

    def test_set_default_unknown_returns_false(self):
        mgr = self._fresh()
        assert mgr.set_default("ghost") is False

    def test_delete_session(self):
        mgr = self._fresh()
        sid = mgr.create_session()
        result = mgr.delete_session(sid)
        assert result is True
        assert mgr.get_session(sid) is None

    def test_delete_session_by_short_prefix(self):
        mgr = self._fresh()
        sid = mgr.create_session()
        result = mgr.delete_session(sid[:8])
        assert result is True
        assert mgr.get_session(sid) is None

    def test_delete_default_promotes_next(self):
        mgr = self._fresh()
        sid1 = mgr.create_session()
        sid2 = mgr.create_session()
        mgr.delete_session(sid2)
        # After deleting the default, a remaining session should be default
        assert mgr.get_default_session_id() == sid1

    def test_lru_eviction_at_max_capacity(self):
        mgr = self._fresh()
        sids = [mgr.create_session() for _ in range(3)]
        # Touch the second one recently so it survives
        time.sleep(0.01)
        mgr.get_session(sids[1])
        # Adding a 4th triggers eviction
        mgr.create_session()
        remaining = set(mgr.list_sessions().keys())
        assert len(remaining) == 3
        assert sids[1] in remaining  # recently touched, should survive

    def test_delete_unknown_returns_false(self):
        mgr = self._fresh()
        assert mgr.delete_session("ghost") is False

    def test_first_use_creation_is_atomic(self):
        mgr = self._fresh()
        start = Barrier(8)

        def resolve_default() -> str:
            start.wait(timeout=2)
            return mgr.ensure_session().session_id

        with ThreadPoolExecutor(max_workers=8) as executor:
            session_ids = list(executor.map(lambda _: resolve_default(), range(8)))

        assert len(set(session_ids)) == 1
        assert list(mgr.list_sessions()) == [session_ids[0]]

    def test_create_scope_leases_new_session_atomically(self):
        mgr = MaBoSSSessionManager(max_sessions=1)

        with mgr.create_session_scope() as session:
            assert mgr.get_default_session_id() == session.session_id
            with pytest.raises(RuntimeError, match="every session is active"):
                mgr.create_session()

        assert list(mgr.list_sessions()) == [session.session_id]

    def test_same_session_operations_are_serialized(self):
        mgr = self._fresh()
        session_id = mgr.create_session()
        session = mgr.get_session(session_id)
        assert session is not None

        first_entered = Event()
        release_first = Event()
        second_started = Event()
        second_entered = Event()

        def first_operation() -> None:
            with mgr.session_scope(session_id):
                first_entered.set()
                assert release_first.wait(timeout=2)

        def second_operation() -> None:
            second_started.set()
            with mgr.session_scope(session_id):
                second_entered.set()

        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(first_operation)
            assert first_entered.wait(timeout=2)
            second_future = executor.submit(second_operation)
            assert second_started.wait(timeout=2)

            with mgr._condition:
                assert mgr._condition.wait_for(
                    lambda: session._lease_count == 2,
                    timeout=2,
                )
            assert not second_entered.is_set()

            release_first.set()
            first_future.result(timeout=2)
            second_future.result(timeout=2)

        assert second_entered.is_set()

    def test_different_sessions_operate_concurrently(self):
        mgr = self._fresh()
        first_id = mgr.create_session()
        second_id = mgr.create_session()
        both_entered = Barrier(2)

        def enter_session(session_id: str) -> None:
            with mgr.session_scope(session_id):
                both_entered.wait(timeout=2)

        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(enter_session, first_id)
            second_future = executor.submit(enter_session, second_id)
            first_future.result(timeout=2)
            second_future.result(timeout=2)

    def test_delete_waits_for_admitted_operation(self):
        mgr = self._fresh()
        session_id = mgr.create_session()
        session = mgr.get_session(session_id)
        assert session is not None

        operation_entered = Event()
        release_operation = Event()
        retained_session: list[bool] = []

        def active_operation() -> None:
            with mgr.session_scope(session_id):
                operation_entered.set()
                assert release_operation.wait(timeout=2)
                retained_session.append(
                    _sm.ensure_session(session_id) is session
                )

        with ThreadPoolExecutor(max_workers=2) as executor:
            operation_future = executor.submit(active_operation)
            assert operation_entered.wait(timeout=2)
            delete_future = executor.submit(mgr.delete_session, session_id)

            with mgr._condition:
                assert mgr._condition.wait_for(
                    lambda: session._retired,
                    timeout=2,
                )
            assert not delete_future.done()

            release_operation.set()
            operation_future.result(timeout=2)
            assert delete_future.result(timeout=2) is True

        assert mgr.get_session(session_id) is None
        assert retained_session == [True]

    def test_lru_evicts_idle_session_instead_of_active_session(self):
        mgr = MaBoSSSessionManager(max_sessions=2)
        active_id = mgr.create_session()
        idle_id = mgr.create_session()

        with mgr.session_scope(active_id):
            replacement_id = mgr.create_session()

        remaining = set(mgr.list_sessions())
        assert active_id in remaining
        assert replacement_id in remaining
        assert idle_id not in remaining

    def test_creation_fails_when_every_session_is_active(self):
        mgr = MaBoSSSessionManager(max_sessions=1)
        session_id = mgr.create_session()

        with mgr.session_scope(session_id):
            with pytest.raises(RuntimeError, match="every session is active"):
                mgr.create_session()

        assert list(mgr.list_sessions()) == [session_id]


# ---------------------------------------------------------------------------
# ensure_session (module-level helper)
# ---------------------------------------------------------------------------


class TestEnsureSession:
    def test_auto_creates_when_no_session_exists(self, monkeypatch):
        """ensure_session must create and return a session if none exist."""
        fresh_mgr = MaBoSSSessionManager()
        monkeypatch.setattr(_sm, "session_manager", fresh_mgr)
        sess = _sm.ensure_session()
        assert sess is not None
        assert sess.session_id is not None

    def test_returns_existing_default(self, monkeypatch):
        fresh_mgr = MaBoSSSessionManager()
        sid = fresh_mgr.create_session()
        monkeypatch.setattr(_sm, "session_manager", fresh_mgr)
        sess = _sm.ensure_session()
        assert sess.session_id == sid

    def test_returns_specified_session(self, monkeypatch):
        fresh_mgr = MaBoSSSessionManager()
        sid1 = fresh_mgr.create_session()
        fresh_mgr.create_session()
        monkeypatch.setattr(_sm, "session_manager", fresh_mgr)
        sess = _sm.ensure_session(sid1)
        assert sess.session_id == sid1
