"""Deterministic concurrency tests for PhysiCell session lifecycle management."""

import importlib.util
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event

import pytest


def _load_session_manager():
    module_name = "physicell_concurrency_session_manager"
    path = Path(__file__).parent.parent / "PhysiCell" / "session_manager.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_sm = _load_session_manager()
SessionManager = _sm.SessionManager
WorkflowStep = _sm.WorkflowStep
MaBoSSContext = _sm.MaBoSSContext


def test_first_use_session_creation_is_atomic() -> None:
    manager = SessionManager()
    start = Barrier(8)

    def ensure_default() -> str:
        start.wait(timeout=2)
        return manager.ensure_session().session_id

    with ThreadPoolExecutor(max_workers=8) as executor:
        session_ids = list(executor.map(lambda _: ensure_default(), range(8)))

    assert len(set(session_ids)) == 1
    assert len(manager.list_sessions()) == 1


def test_create_scope_leases_new_session_atomically() -> None:
    manager = SessionManager(max_sessions=1)

    with manager.create_session_scope() as session:
        assert session._lease_count == 1
        with pytest.raises(RuntimeError, match="every session is active"):
            manager.create_session()


def test_same_session_operations_are_serialized() -> None:
    manager = SessionManager()
    session_id = manager.create_session()
    session = manager.get_session(session_id)
    assert session is not None

    first_entered = Event()
    release_first = Event()
    second_started = Event()
    second_entered = Event()

    def first_operation() -> None:
        with manager.session_scope(session_id):
            first_entered.set()
            assert release_first.wait(timeout=2)

    def second_operation() -> None:
        second_started.set()
        with manager.session_scope(session_id):
            second_entered.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(first_operation)
        assert first_entered.wait(timeout=2)
        second_future = executor.submit(second_operation)
        assert second_started.wait(timeout=2)

        with manager._condition:
            assert manager._condition.wait_for(
                lambda: session._lease_count == 2,
                timeout=2,
            )
        assert not second_entered.is_set()

        release_first.set()
        first_future.result(timeout=2)
        second_future.result(timeout=2)

    assert second_entered.is_set()


def test_different_sessions_remain_concurrent() -> None:
    manager = SessionManager()
    first_id = manager.create_session()
    second_id = manager.create_session()
    both_entered = Barrier(2)

    def enter_session(session_id: str) -> None:
        with manager.session_scope(session_id):
            both_entered.wait(timeout=2)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(enter_session, first_id)
        second_future = executor.submit(enter_session, second_id)
        first_future.result(timeout=2)
        second_future.result(timeout=2)


def test_delete_retires_then_waits_for_admitted_operation() -> None:
    manager = SessionManager()
    session_id = manager.create_session()
    session = manager.get_session(session_id)
    assert session is not None

    operation_entered = Event()
    release_operation = Event()

    def active_operation() -> None:
        with manager.session_scope(session_id):
            operation_entered.set()
            assert release_operation.wait(timeout=2)
            assert _sm.get_current_session(session_id) is session

    with ThreadPoolExecutor(max_workers=2) as executor:
        operation_future = executor.submit(active_operation)
        assert operation_entered.wait(timeout=2)
        delete_future = executor.submit(manager.delete_session, session_id)

        with manager._condition:
            assert manager._condition.wait_for(
                lambda: session._retired,
                timeout=2,
            )
        assert not delete_future.done()
        with pytest.raises(ValueError, match="Session not found"):
            with manager.session_scope(session_id):
                pass

        release_operation.set()
        operation_future.result(timeout=2)
        assert delete_future.result(timeout=2) is True

    assert manager.get_session(session_id) is None


def test_lru_evicts_idle_session_instead_of_active_session() -> None:
    manager = SessionManager(max_sessions=2)
    active_id = manager.create_session()
    idle_id = manager.create_session()

    with manager.session_scope(active_id):
        replacement_id = manager.create_session()

    remaining = {session.session_id for session in manager.list_sessions()}
    assert active_id in remaining
    assert replacement_id in remaining
    assert idle_id not in remaining


def test_ttl_cleanup_skips_active_session() -> None:
    manager = SessionManager()
    session_id = manager.create_session()
    session = manager.get_session(session_id)
    assert session is not None
    session.last_accessed = time.time() - 3600

    with manager.session_scope(session_id):
        session.last_accessed = time.time() - 3600
        assert manager.cleanup_old_sessions(max_age_hours=0) == 0

    assert manager.cleanup_old_sessions(max_age_hours=0) == 1


def test_configuration_replacement_resets_only_configuration_state() -> None:
    manager = SessionManager()
    session_id = manager.create_session(session_name="hypothesis")
    session = manager.get_session(session_id)
    assert session is not None
    old_config = object()
    new_config = object()

    with manager.session_scope(session_id):
        session.scenario_context = "tumour spheroid"
        session.maboss_context = MaBoSSContext(model_name="cell_fate")
        session.config = old_config
        session.completed_steps.update(
            {
                WorkflowStep.SCENARIO_ANALYSIS,
                WorkflowStep.DOMAIN_SETUP,
                WorkflowStep.SUBSTRATES_ADDED,
            }
        )
        session.substrates_count = 2
        session.cell_types_count = 3
        session.rules_count = 4
        session.loaded_from_xml = True
        session.original_xml_path = "/tmp/old.xml"
        session.xml_modification_count = 5
        session.replace_config(new_config)

        assert session.config is new_config
        assert session.completed_steps == {WorkflowStep.SCENARIO_ANALYSIS}
        assert session.substrates_count == 0
        assert session.cell_types_count == 0
        assert session.rules_count == 0
        assert session.loaded_from_xml is False
        assert session.original_xml_path is None
        assert session.xml_modification_count == 0
        assert session.scenario_context == "tumour spheroid"
        assert session.maboss_context is not None
        assert session.maboss_context.model_name == "cell_fate"
