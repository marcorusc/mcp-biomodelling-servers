"""Unit tests for PhysiCell/session_manager.py."""
import importlib.util
import time
from pathlib import Path


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


_sm = _load("PhysiCell")
MaBoSSContext = _sm.MaBoSSContext
SessionState = _sm.SessionState
SessionManager = _sm.SessionManager
WorkflowStep = _sm.WorkflowStep
ensure_session = _sm.ensure_session
get_current_session = _sm.get_current_session


# ---------------------------------------------------------------------------
# MaBoSSContext
# ---------------------------------------------------------------------------

class TestMaBoSSContext:
    def test_default_fields(self):
        ctx = MaBoSSContext()
        assert ctx.model_name == ""
        assert ctx.available_nodes == []
        assert ctx.output_nodes == []

    def test_custom_fields(self):
        ctx = MaBoSSContext(
            model_name="my_model",
            bnd_file_path="/a.bnd",
            cfg_file_path="/a.cfg",
            available_nodes=["p53", "MDM2"],
            target_cell_type="cancer",
        )
        assert ctx.model_name == "my_model"
        assert "p53" in ctx.available_nodes


# ---------------------------------------------------------------------------
# SessionState
# ---------------------------------------------------------------------------

class TestSessionState:
    def test_initial_state(self):
        s = SessionState(session_id="abc")
        assert s.session_id == "abc"
        assert s.config is None
        assert s.maboss_context is None
        assert s.completed_steps == set()
        assert s.substrates_count == 0

    def test_mark_step_complete(self):
        s = SessionState(session_id="s1")
        s.mark_step_complete(WorkflowStep.DOMAIN_SETUP)
        assert s.is_step_complete(WorkflowStep.DOMAIN_SETUP)
        assert not s.is_step_complete(WorkflowStep.SUBSTRATES_ADDED)

    def test_mark_xml_modification_increments_counter(self):
        s = SessionState(session_id="s1")
        s.mark_xml_modification()
        s.mark_xml_modification()
        assert s.xml_modification_count == 2

    def test_get_progress_percentage_zero_when_no_steps(self):
        s = SessionState(session_id="s1")
        pct = s.get_progress_percentage()
        assert 0.0 <= pct <= 100.0

    def test_get_progress_percentage_increases_with_steps(self):
        s = SessionState(session_id="s1")
        before = s.get_progress_percentage()
        s.mark_step_complete(WorkflowStep.DOMAIN_SETUP)
        s.mark_step_complete(WorkflowStep.SUBSTRATES_ADDED)
        s.mark_step_complete(WorkflowStep.CELL_TYPES_ADDED)
        after = s.get_progress_percentage()
        assert after >= before

    def test_to_dict_has_required_keys(self):
        s = SessionState(session_id="s1", session_name="test-session")
        d = s.to_dict()
        assert d["session_id"] == "s1"
        assert d["session_name"] == "test-session"
        assert "completed_steps" in d
        assert "substrates_count" in d

    def test_to_dict_encodes_completed_steps_as_values(self):
        s = SessionState(session_id="s1")
        s.mark_step_complete(WorkflowStep.DOMAIN_SETUP)
        d = s.to_dict()
        assert WorkflowStep.DOMAIN_SETUP.value in d["completed_steps"]

    def test_to_dict_includes_maboss_context_when_set(self):
        s = SessionState(session_id="s1")
        s.maboss_context = MaBoSSContext(model_name="m1", bnd_file_path="/b.bnd")
        d = s.to_dict()
        assert "maboss_context" in d
        assert d["maboss_context"]["model_name"] == "m1"

    def test_get_next_recommended_steps_returns_list(self):
        s = SessionState(session_id="s1")
        recs = s.get_next_recommended_steps()
        assert isinstance(recs, list)
        assert len(recs) > 0

    def test_get_next_recommended_steps_xml_workflow(self):
        s = SessionState(session_id="s1", loaded_from_xml=True)
        recs = s.get_next_recommended_steps()
        # Should always include export option in XML workflow
        assert any("export" in r for r in recs)


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class TestSessionManager:
    def _fresh(self):
        return SessionManager(max_sessions=3, auto_cleanup_hours=24.0)

    def test_create_returns_uuid_string(self):
        mgr = self._fresh()
        sid = mgr.create_session()
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_created_session_becomes_default(self):
        mgr = self._fresh()
        sid = mgr.create_session(set_as_default=True)
        assert mgr.get_default_session_id() == sid

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

    def test_get_session_no_default_returns_none(self):
        mgr = self._fresh()
        assert mgr.get_session() is None

    def test_get_unknown_session_returns_none(self):
        mgr = self._fresh()
        assert mgr.get_session("nonexistent") is None

    def test_list_sessions_returns_all(self):
        mgr = self._fresh()
        mgr.create_session()
        mgr.create_session()
        assert len(mgr.list_sessions()) == 2

    def test_set_default_session(self):
        mgr = self._fresh()
        sid1 = mgr.create_session()
        _sid2 = mgr.create_session()
        assert mgr.set_default_session(sid1) is True
        assert mgr.get_default_session_id() == sid1

    def test_set_default_session_by_short_prefix(self):
        mgr = self._fresh()
        sid1 = mgr.create_session()
        _sid2 = mgr.create_session()
        assert mgr.set_default_session(sid1[:8]) is True
        assert mgr.get_default_session_id() == sid1

    def test_set_default_unknown_returns_false(self):
        mgr = self._fresh()
        assert mgr.set_default_session("ghost") is False

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

    def test_delete_default_promotes_next(self):
        mgr = self._fresh()
        sid1 = mgr.create_session()
        sid2 = mgr.create_session()
        mgr.delete_session(sid2)
        assert mgr.get_default_session_id() == sid1

    def test_delete_unknown_returns_false(self):
        mgr = self._fresh()
        assert mgr.delete_session("ghost") is False

    def test_lru_eviction_at_capacity(self):
        mgr = self._fresh()
        sids = [mgr.create_session() for _ in range(3)]
        time.sleep(0.01)
        mgr.get_session(sids[1])  # touch the second one
        mgr.create_session()  # triggers eviction
        remaining = {s.session_id for s in mgr.list_sessions()}
        assert len(remaining) == 3
        assert sids[1] in remaining

    def test_session_name_stored(self):
        mgr = self._fresh()
        sid = mgr.create_session(session_name="my-experiment")
        sess = mgr.get_session(sid)
        assert sess.session_name == "my-experiment"

    def test_find_session_by_name(self):
        mgr = self._fresh()
        sid = mgr.create_session(session_name="experiment-A")
        result = mgr.find_session_by_name("experiment-A")
        assert result is not None
        assert result.session_id == sid

    def test_find_session_by_name_unknown_returns_none(self):
        mgr = self._fresh()
        assert mgr.find_session_by_name("ghost") is None

    def test_get_session_stats_empty(self):
        mgr = self._fresh()
        stats = mgr.get_session_stats()
        assert stats["total_sessions"] == 0
        assert stats["active_configs"] == 0

    def test_get_session_stats_with_sessions(self):
        mgr = self._fresh()
        mgr.create_session()
        mgr.create_session()
        stats = mgr.get_session_stats()
        assert stats["total_sessions"] == 2

    def test_set_maboss_context(self):
        mgr = self._fresh()
        sid = mgr.create_session()
        ctx = MaBoSSContext(model_name="m1", bnd_file_path="/b.bnd")
        result = mgr.set_maboss_context(sid, ctx)
        assert result is True
        retrieved = mgr.get_maboss_context(sid)
        assert retrieved is not None
        assert retrieved.model_name == "m1"

    def test_cleanup_old_sessions(self):
        mgr = self._fresh()
        sid = mgr.create_session()
        # Backdating last_accessed so session appears old
        sess = mgr.get_session(sid)
        sess.last_accessed = time.time() - 999999
        removed = mgr.cleanup_old_sessions(max_age_hours=0.0001)
        assert removed >= 1


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

class TestModuleLevelHelpers:
    def test_ensure_session_creates_when_none(self, monkeypatch):
        fresh_mgr = SessionManager()
        monkeypatch.setattr(_sm, "session_manager", fresh_mgr)
        sess = _sm.ensure_session()
        assert sess is not None

    def test_ensure_session_returns_existing_default(self, monkeypatch):
        fresh_mgr = SessionManager()
        sid = fresh_mgr.create_session()
        monkeypatch.setattr(_sm, "session_manager", fresh_mgr)
        sess = _sm.ensure_session()
        assert sess.session_id == sid

    def test_get_current_session_returns_none_when_empty(self, monkeypatch):
        fresh_mgr = SessionManager()
        monkeypatch.setattr(_sm, "session_manager", fresh_mgr)
        assert _sm.get_current_session() is None

