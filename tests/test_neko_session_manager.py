"""Unit tests for NeKo/session_manager.py."""
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
    sys.modules[module_name] = mod  # must register before exec so @dataclass resolves __module__
    spec.loader.exec_module(mod)
    return mod


_sm = _load("NeKo")
NeKoSession = _sm.NeKoSession
NeKoSessionManager = _sm.NeKoSessionManager
ensure_session = _sm.ensure_session
normalize_verbosity = _sm.normalize_verbosity
DEFAULT_VERBOSITY = _sm.DEFAULT_VERBOSITY


# ---------------------------------------------------------------------------
# NeKoSession
# ---------------------------------------------------------------------------

class TestNeKoSession:
    def test_initial_state(self):
        s = NeKoSession(session_id="s1")
        assert s.session_id == "s1"
        assert s.network is None
        assert s._edges_cache_dirty is True

    def test_set_network_invalidates_cache(self):
        s = NeKoSession(session_id="s1")
        s._edges_cache_dirty = False  # pretend cache is warm
        s.set_network("fake_network")
        assert s.network == "fake_network"
        assert s._edges_cache_dirty is True

    def test_touch_updates_last_accessed(self):
        s = NeKoSession(session_id="s1")
        before = s.last_accessed
        time.sleep(0.01)
        s.touch()
        assert s.last_accessed > before

    def test_update_default_params(self):
        s = NeKoSession(session_id="s1")
        s.update_default_params(max_len=5, only_signed=False)
        assert s.default_params["max_len"] == 5
        assert s.default_params["only_signed"] is False

    def test_update_default_params_ignores_none(self):
        s = NeKoSession(session_id="s1")
        original = s.default_params["max_len"]
        s.update_default_params(max_len=None)
        assert s.default_params["max_len"] == original

    def test_get_completion_params_keys(self):
        s = NeKoSession(session_id="s1")
        params = s.get_completion_params()
        assert "maxlen" in params
        assert "algorithm" in params
        assert "only_signed" in params

    def test_get_edges_df_returns_none_without_network(self):
        s = NeKoSession(session_id="s1")
        assert s.get_edges_df() is None

    def test_invalidate_edges_cache(self):
        s = NeKoSession(session_id="s1")
        s._edges_cache_dirty = False
        s.invalidate_edges_cache()
        assert s._edges_cache_dirty is True

    def test_cache_invalidation_waits_for_conversion(self):
        conversion_started = Event()
        release_conversion = Event()

        class BlockingNetwork:
            def convert_edgelist_into_genesymbol(self):
                conversion_started.set()
                assert release_conversion.wait(timeout=2)
                return "converted"

        session = NeKoSession(session_id="s1")
        session.set_network(BlockingNetwork())

        with ThreadPoolExecutor(max_workers=2) as executor:
            conversion = executor.submit(session.get_edges_df)
            assert conversion_started.wait(timeout=2)
            invalidation = executor.submit(session.invalidate_edges_cache)
            assert invalidation.done() is False

            release_conversion.set()
            assert conversion.result(timeout=2) == "converted"
            invalidation.result(timeout=2)

        assert session._edges_cache_dirty is True


# ---------------------------------------------------------------------------
# NeKoSessionManager
# ---------------------------------------------------------------------------

class TestNeKoSessionManager:
    def _fresh(self):
        return NeKoSessionManager(max_sessions=3)

    def test_create_returns_string_id(self):
        mgr = self._fresh()
        assert isinstance(mgr.create_session(), str)

    def test_created_session_becomes_default(self):
        mgr = self._fresh()
        sid = mgr.create_session()
        assert mgr.get_default_session_id() == sid

    def test_get_session_by_id(self):
        mgr = self._fresh()
        sid = mgr.create_session()
        assert mgr.get_session(sid).session_id == sid

    def test_get_session_by_short_prefix(self):
        mgr = self._fresh()
        sid = mgr.create_session()
        sess = mgr.get_session(sid[:8])
        assert sess is not None
        assert sess.session_id == sid

    def test_get_unknown_session_returns_none(self):
        mgr = self._fresh()
        assert mgr.get_session("nope") is None

    def test_list_sessions_count(self):
        mgr = self._fresh()
        mgr.create_session()
        mgr.create_session()
        assert len(mgr.list_sessions()) == 2

    def test_set_default(self):
        mgr = self._fresh()
        sid1 = mgr.create_session()
        _sid2 = mgr.create_session()
        assert mgr.set_default(sid1) is True
        assert mgr.get_default_session_id() == sid1

    def test_set_default_by_short_prefix(self):
        mgr = self._fresh()
        sid1 = mgr.create_session()
        _sid2 = mgr.create_session()
        assert mgr.set_default(sid1[:8]) is True
        assert mgr.get_default_session_id() == sid1

    def test_set_default_unknown_returns_false(self):
        mgr = self._fresh()
        assert mgr.set_default("ghost") is False

    def test_delete_session(self):
        mgr = self._fresh()
        sid = mgr.create_session()
        assert mgr.delete_session(sid) is True
        assert mgr.get_session(sid) is None

    def test_delete_session_by_short_prefix(self):
        mgr = self._fresh()
        sid = mgr.create_session()
        assert mgr.delete_session(sid[:8]) is True
        assert mgr.get_session(sid) is None

    def test_delete_unknown_returns_false(self):
        mgr = self._fresh()
        assert mgr.delete_session("ghost") is False

    def test_lru_eviction(self):
        mgr = self._fresh()
        sids = [mgr.create_session() for _ in range(3)]
        time.sleep(0.01)
        mgr.get_session(sids[2])  # touch the last one
        mgr.create_session()  # triggers eviction
        remaining = set(mgr.list_sessions().keys())
        assert len(remaining) == 3
        assert sids[2] in remaining

    def test_first_use_creation_is_atomic(self):
        mgr = self._fresh()
        start = Barrier(8)

        def resolve_default() -> str:
            start.wait(timeout=2)
            return mgr.ensure_session().session_id

        with ThreadPoolExecutor(max_workers=8) as executor:
            session_ids = list(
                executor.map(lambda _: resolve_default(), range(8))
            )

        assert len(set(session_ids)) == 1
        assert list(mgr.list_sessions()) == [session_ids[0]]

    def test_create_scope_leases_new_session_atomically(self):
        mgr = NeKoSessionManager(max_sessions=1)

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
            assert second_entered.is_set() is False

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
            assert delete_future.done() is False

            release_operation.set()
            operation_future.result(timeout=2)
            assert delete_future.result(timeout=2) is True

        assert mgr.get_session(session_id) is None
        assert retained_session == [True]

    def test_lru_evicts_idle_session_instead_of_active_session(self):
        mgr = NeKoSessionManager(max_sessions=2)
        active_id = mgr.create_session()
        idle_id = mgr.create_session()

        with mgr.session_scope(active_id):
            replacement_id = mgr.create_session()

        remaining = set(mgr.list_sessions())
        assert active_id in remaining
        assert replacement_id in remaining
        assert idle_id not in remaining

    def test_creation_fails_when_every_session_is_active(self):
        mgr = NeKoSessionManager(max_sessions=1)
        session_id = mgr.create_session()

        with mgr.session_scope(session_id):
            with pytest.raises(RuntimeError, match="every session is active"):
                mgr.create_session()

        assert list(mgr.list_sessions()) == [session_id]


# ---------------------------------------------------------------------------
# ensure_session
# ---------------------------------------------------------------------------

class TestEnsureSession:
    def test_auto_creates_when_empty(self, monkeypatch):
        fresh_mgr = NeKoSessionManager()
        monkeypatch.setattr(_sm, "session_manager", fresh_mgr)
        sess = _sm.ensure_session(None)
        assert sess is not None

    def test_returns_specified_session(self, monkeypatch):
        fresh_mgr = NeKoSessionManager()
        sid = fresh_mgr.create_session()
        monkeypatch.setattr(_sm, "session_manager", fresh_mgr)
        sess = _sm.ensure_session(sid)
        assert sess.session_id == sid


# ---------------------------------------------------------------------------
# normalize_verbosity
# ---------------------------------------------------------------------------

class TestNormalizeVerbosity:
    @pytest.mark.parametrize("v", ["summary", "preview", "full"])
    def test_valid_values_pass_through(self, v):
        assert normalize_verbosity(v) == v

    @pytest.mark.parametrize("v", ["SUMMARY", "Preview", "FULL"])
    def test_case_insensitive(self, v):
        assert normalize_verbosity(v) == v.lower()

    @pytest.mark.parametrize("v", [None, "", "invalid", "verbose"])
    def test_invalid_falls_back_to_default(self, v):
        assert normalize_verbosity(v) == DEFAULT_VERBOSITY
