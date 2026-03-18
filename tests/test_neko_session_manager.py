"""Unit tests for NeKo/session_manager.py."""
import importlib.util
import time
from pathlib import Path

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

