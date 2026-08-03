"""Thread-safe session management for the PhysiCell MCP server.

The MCP SDK v2 may run synchronous handlers concurrently in worker threads.
Registry operations and accesses to each mutable PhysiCell configuration
therefore use separate locks: the manager condition protects session lifecycle
metadata, while a per-session lock serializes configuration, workflow, and
artifact operations.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Condition, RLock
from typing import Any

# Configure logging
logger = logging.getLogger(__name__)

class WorkflowStep(Enum):
    """Enumeration of workflow steps for progress tracking."""
    SCENARIO_ANALYSIS = "scenario_analysis"
    DOMAIN_SETUP = "domain_setup" 
    SUBSTRATES_ADDED = "substrates_added"
    CELL_TYPES_ADDED = "cell_types_added"
    CELL_PARAMETERS_CONFIGURED = "cell_parameters_configured"
    SUBSTRATE_INTERACTIONS_SET = "substrate_interactions_set"
    RULES_CONFIGURED = "rules_configured"
    PHYSIBOSS_MODELS_ADDED = "physiboss_models_added"
    PHYSIBOSS_SETTINGS_CONFIGURED = "physiboss_settings_configured"
    PHYSIBOSS_INPUTS_LINKED = "physiboss_inputs_linked"
    PHYSIBOSS_OUTPUTS_LINKED = "physiboss_outputs_linked"
    PHYSIBOSS_MUTATIONS_APPLIED = "physiboss_mutations_applied"
    READY_FOR_EXPORT = "ready_for_export"
    XML_LOADED = "xml_loaded"
    XML_ANALYZED = "xml_analyzed"

@dataclass
class MaBoSSContext:
    """Context information from MaBoSS model analysis."""
    model_name: str = ""
    bnd_file_path: str = ""
    cfg_file_path: str = ""
    available_nodes: list[str] = field(default_factory=list)
    output_nodes: list[str] = field(default_factory=list)
    simulation_results: str = ""  # Summary of MaBoSS simulation behavior
    target_cell_type: str = ""  # Which cell type this model targets
    biological_context: str = ""  # Original biological question/context
    source_manifest_path: str = ""
    local_manifest_path: str = ""
    source_session_id: str = ""
    result_file_path: str = ""
    simulation_parameters: dict[
        str,
        bool | int | float | str | None,
    ] = field(default_factory=dict)
    neko_session_id: str = ""
    neko_manifest_path: str = ""
    local_neko_manifest_path: str = ""
    local_bnet_path: str = ""

@dataclass
class SessionState:
    """Represents the state of a PhysiCell simulation session."""
    session_id: str
    session_name: str | None = None  # Human-readable name for cross-server linking
    config: object | None = None  # PhysiCellConfig instance
    scenario_context: str = ""
    maboss_contexts: dict[str, MaBoSSContext] = field(default_factory=dict)
    completed_steps: set[WorkflowStep] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    substrates_count: int = 0
    cell_types_count: int = 0
    rules_count: int = 0
    physiboss_models_count: int = 0
    physiboss_settings_count: int = 0
    physiboss_input_links_count: int = 0
    physiboss_output_links_count: int = 0
    physiboss_mutations_count: int = 0
    
    # XML-related fields
    loaded_from_xml: bool = False
    original_xml_path: str | None = None
    xml_modification_count: int = 0
    loaded_substrates: list[str] = field(default_factory=list)
    loaded_cell_types: list[str] = field(default_factory=list)
    loaded_physiboss_models: list[str] = field(default_factory=list)
    has_existing_rules: bool = False
    _operation_lock: RLock = field(
        default_factory=RLock,
        init=False,
        repr=False,
        compare=False,
    )
    # These lifecycle fields are protected by SessionManager._condition.
    _lease_count: int = field(default=0, init=False, repr=False, compare=False)
    _retired: bool = field(default=False, init=False, repr=False, compare=False)

    def touch(self) -> None:
        """Update the last-access timestamp under the session lock."""
        with self._operation_lock:
            self._touch_unlocked()

    def _touch_unlocked(self) -> None:
        self.last_accessed = time.time()

    @property
    def maboss_context(self) -> MaBoSSContext | None:
        """Return the most recently registered context for compatibility."""
        with self._operation_lock:
            return next(reversed(self.maboss_contexts.values()), None)

    @maboss_context.setter
    def maboss_context(self, context: MaBoSSContext | None) -> None:
        """Register one context through the legacy single-context attribute."""
        with self._operation_lock:
            if context is None:
                self.maboss_contexts.clear()
            else:
                self._store_maboss_context_unlocked(context)
            self._touch_unlocked()

    def _store_maboss_context_unlocked(self, context: MaBoSSContext) -> None:
        """Store a context last so compatibility reads return the newest one."""
        self.maboss_contexts.pop(context.target_cell_type, None)
        self.maboss_contexts[context.target_cell_type] = context

    def register_maboss_context(self, context: MaBoSSContext) -> None:
        """Store or replace the MaBoSS context for one target cell type."""
        if not context.target_cell_type:
            raise ValueError("MaBoSS context requires a target cell type.")
        with self._operation_lock:
            self._store_maboss_context_unlocked(context)
            self._touch_unlocked()

    def publish_physiboss_import(
        self,
        *,
        config: object,
        context: MaBoSSContext,
        model_names: list[str],
        settings_count: int,
        input_links_count: int,
        output_links_count: int,
        mutations_count: int,
    ) -> None:
        """Atomically publish a validated PhysiBoSS model import."""
        if not context.target_cell_type:
            raise ValueError("Imported MaBoSS context requires a target cell type.")
        with self._operation_lock:
            self.config = config
            self._store_maboss_context_unlocked(context)
            self.loaded_physiboss_models = list(model_names)
            self.physiboss_models_count = len(model_names)
            self.physiboss_settings_count = settings_count
            self.physiboss_input_links_count = input_links_count
            self.physiboss_output_links_count = output_links_count
            self.physiboss_mutations_count = mutations_count
            self.completed_steps.add(WorkflowStep.PHYSIBOSS_MODELS_ADDED)
            if self.loaded_from_xml:
                self.xml_modification_count += 1
            self._touch_unlocked()

    def publish_config_update(
        self,
        *,
        config: object,
        completed_step: WorkflowStep,
        configuration_changed: bool,
        physiboss_tracking: tuple[
            list[str],
            int,
            int,
            int,
            int,
        ] | None = None,
    ) -> None:
        """Atomically publish one validated configuration patch.

        A patch that explicitly confirms already-current values still
        completes its workflow step, but it does not replace the active
        configuration or inflate XML modification tracking.
        """
        with self._operation_lock:
            if configuration_changed:
                self.config = config
            self.completed_steps.add(completed_step)
            if physiboss_tracking is not None:
                (
                    model_names,
                    settings_count,
                    input_links_count,
                    output_links_count,
                    mutations_count,
                ) = physiboss_tracking
                self.loaded_physiboss_models = list(model_names)
                self.physiboss_models_count = len(model_names)
                self.physiboss_settings_count = settings_count
                self.physiboss_input_links_count = input_links_count
                self.physiboss_output_links_count = output_links_count
                self.physiboss_mutations_count = mutations_count
            if configuration_changed and self.loaded_from_xml:
                self.xml_modification_count += 1
            self._touch_unlocked()
    
    def mark_step_complete(self, step: WorkflowStep) -> None:
        """Mark a workflow step as completed."""
        with self._operation_lock:
            self.completed_steps.add(step)
            self._touch_unlocked()
    
    def mark_xml_modification(self) -> None:
        """Track modifications to XML-loaded configuration."""
        with self._operation_lock:
            self.xml_modification_count += 1
            self._touch_unlocked()
    
    def is_step_complete(self, step: WorkflowStep) -> bool:
        """Check if a workflow step is completed."""
        with self._operation_lock:
            return step in self.completed_steps

    def replace_config(
        self,
        config: object,
        *,
        loaded_from_xml: bool = False,
        original_xml_path: str | None = None,
    ) -> None:
        """Atomically replace configuration-derived state.

        Session identity and modelling context are intentionally preserved, but
        counters and workflow steps describing the previous configuration are
        cleared before the new configuration is published.
        """
        with self._operation_lock:
            self.config = config
            self.completed_steps = (
                {WorkflowStep.SCENARIO_ANALYSIS}
                if self.scenario_context
                else set()
            )
            self.substrates_count = 0
            self.cell_types_count = 0
            self.rules_count = 0
            self.physiboss_models_count = 0
            self.physiboss_settings_count = 0
            self.physiboss_input_links_count = 0
            self.physiboss_output_links_count = 0
            self.physiboss_mutations_count = 0
            self.loaded_from_xml = loaded_from_xml
            self.original_xml_path = original_xml_path
            self.xml_modification_count = 0
            self.loaded_substrates = []
            self.loaded_cell_types = []
            self.loaded_physiboss_models = []
            self.has_existing_rules = False
            if loaded_from_xml:
                self.completed_steps.add(WorkflowStep.XML_LOADED)
            self._touch_unlocked()
    
    def get_next_recommended_steps(self) -> list[str]:
        """Get recommended next steps based on current progress."""
        with self._operation_lock:
            return self._get_next_recommended_steps_unlocked()

    def _get_next_recommended_steps_unlocked(self) -> list[str]:
        recommendations = []
        
        # If loaded from XML, different workflow
        if self.loaded_from_xml:
            if WorkflowStep.XML_ANALYZED not in self.completed_steps:
                recommendations.append("analyze_loaded_configuration - Review loaded components")
            elif len(self.loaded_cell_types) > 0:
                recommendations.append("configure_cell_parameters - Adjust existing cell types")
                recommendations.append("add_single_cell_rule - Add behavior rules")
            
            if len(self.loaded_substrates) > 0 and len(self.loaded_cell_types) > 0:
                recommendations.append("set_substrate_interaction - Configure cell-substrate interactions")
            
            if len(self.loaded_physiboss_models) > 0:
                recommendations.append("configure_physiboss_settings - Adjust intracellular models")
            
            recommendations.append("export_xml_configuration - Save modified configuration")
            return recommendations
        
        # Original workflow for new configurations
        if WorkflowStep.SCENARIO_ANALYSIS not in self.completed_steps:
            recommendations.append("analyze_biological_scenario - Set the biological context")
        elif WorkflowStep.DOMAIN_SETUP not in self.completed_steps:
            recommendations.append("create_simulation_domain - Define spatial/temporal framework")
        elif WorkflowStep.SUBSTRATES_ADDED not in self.completed_steps:
            recommendations.append("add_single_substrate - Add oxygen, nutrients, drugs")
        elif WorkflowStep.CELL_TYPES_ADDED not in self.completed_steps:
            recommendations.append("add_single_cell_type - Add cancer cells, immune cells")
        elif (self.substrates_count > 0 and self.cell_types_count > 0 and 
              WorkflowStep.CELL_PARAMETERS_CONFIGURED not in self.completed_steps):
            recommendations.append("configure_cell_parameters - Set cell volumes, motility, death rates")
        elif (self.substrates_count > 0 and self.cell_types_count > 0 and 
              WorkflowStep.SUBSTRATE_INTERACTIONS_SET not in self.completed_steps):
            recommendations.append("set_substrate_interaction - Configure substrate uptake/secretion")
        elif (self.maboss_context and self.maboss_context.target_cell_type and 
              WorkflowStep.PHYSIBOSS_MODELS_ADDED not in self.completed_steps):
            recommendations.append(f"add_physiboss_model - Integrate MaBoSS model into {self.maboss_context.target_cell_type}")
        elif (self.physiboss_models_count > 0 and 
              WorkflowStep.PHYSIBOSS_SETTINGS_CONFIGURED not in self.completed_steps):
            recommendations.append("configure_physiboss_settings - Set intracellular parameters")
        elif (self.physiboss_models_count > 0 and 
              WorkflowStep.PHYSIBOSS_INPUTS_LINKED not in self.completed_steps):
            recommendations.append("add_physiboss_input_link - Connect PhysiCell signals to boolean nodes")
        elif (self.physiboss_models_count > 0 and 
              WorkflowStep.PHYSIBOSS_OUTPUTS_LINKED not in self.completed_steps):
            recommendations.append("add_physiboss_output_link - Connect boolean nodes to cell behaviors")
        elif WorkflowStep.RULES_CONFIGURED not in self.completed_steps:
            recommendations.append("add_single_cell_rule - Create additional cell behavior rules")
        elif WorkflowStep.READY_FOR_EXPORT not in self.completed_steps:
            recommendations.append("export_xml_configuration - Generate PhysiCell files")
        else:
            recommendations.append("All steps complete! Ready to run simulation.")
            
        return recommendations
    
    def get_progress_percentage(self) -> float:
        """Calculate progress as percentage of completed workflow steps."""
        with self._operation_lock:
            return self._get_progress_percentage_unlocked()

    def _get_progress_percentage_unlocked(self) -> float:
        # Core steps always required (removed READY_FOR_EXPORT to fix circular dependency)
        core_steps = {
            WorkflowStep.DOMAIN_SETUP,
            WorkflowStep.SUBSTRATES_ADDED,
            WorkflowStep.CELL_TYPES_ADDED
        }
        
        # Optional steps based on context
        optional_steps = set()
        
        # Scenario analysis is optional - useful but not required
        if self.scenario_context:
            optional_steps.add(WorkflowStep.SCENARIO_ANALYSIS)
        
        if self.substrates_count > 0 and self.cell_types_count > 0:
            optional_steps.add(WorkflowStep.CELL_PARAMETERS_CONFIGURED)
            optional_steps.add(WorkflowStep.SUBSTRATE_INTERACTIONS_SET)
        
        if self.maboss_context:
            optional_steps.update({
                WorkflowStep.PHYSIBOSS_MODELS_ADDED,
                WorkflowStep.PHYSIBOSS_SETTINGS_CONFIGURED,
                WorkflowStep.PHYSIBOSS_INPUTS_LINKED,
                WorkflowStep.PHYSIBOSS_OUTPUTS_LINKED
            })
        
        if self.rules_count > 0 or not self.maboss_context:
            optional_steps.add(WorkflowStep.RULES_CONFIGURED)
        
        relevant_steps = core_steps | optional_steps
        total_steps = len(relevant_steps)
        completed = len(self.completed_steps & relevant_steps)
        
        return (completed / total_steps) * 100 if total_steps > 0 else 0
    
    def to_dict(self) -> dict:
        """Convert session state to dictionary for serialization."""
        with self._operation_lock:
            return self._to_dict_unlocked()

    @staticmethod
    def _maboss_context_dict(context: MaBoSSContext) -> dict[str, Any]:
        """Return one serialization-safe MaBoSS context."""
        return {
            "model_name": context.model_name,
            "bnd_file_path": context.bnd_file_path,
            "cfg_file_path": context.cfg_file_path,
            "available_nodes": list(context.available_nodes),
            "output_nodes": list(context.output_nodes),
            "simulation_results": context.simulation_results,
            "target_cell_type": context.target_cell_type,
            "biological_context": context.biological_context,
            "source_manifest_path": context.source_manifest_path,
            "local_manifest_path": context.local_manifest_path,
            "source_session_id": context.source_session_id,
            "result_file_path": context.result_file_path,
            "simulation_parameters": dict(context.simulation_parameters),
            "neko_session_id": context.neko_session_id,
            "neko_manifest_path": context.neko_manifest_path,
            "local_neko_manifest_path": context.local_neko_manifest_path,
            "local_bnet_path": context.local_bnet_path,
        }

    def _to_dict_unlocked(self) -> dict:
        result: dict[str, Any] = {
            'session_id': self.session_id,
            'session_name': self.session_name,
            'scenario_context': self.scenario_context,
            'completed_steps': [step.value for step in self.completed_steps],
            'created_at': self.created_at,
            'last_accessed': self.last_accessed,
            'substrates_count': self.substrates_count,
            'cell_types_count': self.cell_types_count,
            'rules_count': self.rules_count,
            'physiboss_models_count': self.physiboss_models_count,
            'physiboss_settings_count': self.physiboss_settings_count,
            'physiboss_input_links_count': self.physiboss_input_links_count,
            'physiboss_output_links_count': self.physiboss_output_links_count,
            'physiboss_mutations_count': self.physiboss_mutations_count
        }
        
        contexts = list(self.maboss_contexts.values())
        if contexts:
            result["maboss_context"] = self._maboss_context_dict(contexts[-1])
            result["maboss_contexts"] = [
                self._maboss_context_dict(context) for context in contexts
            ]
        
        return result

    def snapshot(self, *, is_default: bool) -> dict[str, Any]:
        """Return a consistent, serialization-safe view of this session."""
        with self._operation_lock:
            result = self._to_dict_unlocked()
            result.update(
                {
                    "has_config": self.config is not None,
                    "progress": self._get_progress_percentage_unlocked(),
                    "is_default": is_default,
                    "loaded_from_xml": self.loaded_from_xml,
                    "xml_modification_count": self.xml_modification_count,
                }
            )
            return result


_active_session = ContextVar[SessionState | None](
    "physicell_active_session",
    default=None,
)

class SessionManager:
    """Manage PhysiCell sessions and lease mutable state to handlers."""
    
    def __init__(self, max_sessions: int = 10, auto_cleanup_hours: float = 24.0):
        self._sessions: dict[str, SessionState] = {}
        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._max_sessions = max_sessions
        self._auto_cleanup_hours = auto_cleanup_hours
        self._default_session_id: str | None = None

    def _resolve_session_id(self, session_id: str) -> str | None:
        """Resolve an exact or unique prefix session ID to a full UUID."""
        if session_id in self._sessions:
            return session_id

        matches = [sid for sid in self._sessions if sid.startswith(session_id)]
        if len(matches) == 1:
            return matches[0]
        return None

    def _retire_session_unlocked(self, session: SessionState) -> None:
        """Remove a session from the registry and reject future leases."""
        self._sessions.pop(session.session_id, None)
        session._retired = True
        if self._default_session_id == session.session_id:
            self._default_session_id = next(iter(self._sessions), None)

    def _cleanup_old_sessions_unlocked(
        self,
        max_age_seconds: float | None = None,
    ) -> int:
        """Retire expired idle sessions while the manager condition is held."""
        if max_age_seconds is None:
            max_age_seconds = self._auto_cleanup_hours * 3600

        current_time = time.time()
        expired_sessions = [
            session
            for session in self._sessions.values()
            if session._lease_count == 0
            and (current_time - session.last_accessed) > max_age_seconds
        ]
        for session in expired_sessions:
            self._retire_session_unlocked(session)

        if expired_sessions:
            logger.info("Cleaned up %d old sessions", len(expired_sessions))
        return len(expired_sessions)

    def _create_session_unlocked(
        self,
        *,
        set_as_default: bool,
        session_name: str | None,
    ) -> str:
        self._cleanup_old_sessions_unlocked()

        if len(self._sessions) >= self._max_sessions:
            idle_sessions = [
                session
                for session in self._sessions.values()
                if session._lease_count == 0
            ]
            if not idle_sessions:
                raise RuntimeError(
                    "PhysiCell session limit reached while every session is active. "
                    "Retry after an in-flight operation completes."
                )
            oldest_session = min(
                idle_sessions,
                key=lambda session: session.last_accessed,
            )
            self._retire_session_unlocked(oldest_session)
            logger.info(
                "Removed oldest session %s... due to limit",
                oldest_session.session_id[:8],
            )

        session_id = str(uuid.uuid4())
        self._sessions[session_id] = SessionState(
            session_id=session_id,
            session_name=session_name,
        )
        if set_as_default or self._default_session_id is None:
            self._default_session_id = session_id

        logger.info(
            "Created session %s...%s",
            session_id[:8],
            f" ({session_name})" if session_name else "",
        )
        return session_id

    def create_session(
        self,
        set_as_default: bool = True,
        session_name: str | None = None,
    ) -> str:
        """Create a new simulation session."""
        with self._condition:
            return self._create_session_unlocked(
                set_as_default=set_as_default,
                session_name=session_name,
            )

    @contextmanager
    def create_session_scope(
        self,
        *,
        set_as_default: bool = True,
        session_name: str | None = None,
    ) -> Iterator[SessionState]:
        """Create and lease a session before another handler can use it."""
        with self._condition:
            session_id = self._create_session_unlocked(
                set_as_default=set_as_default,
                session_name=session_name,
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

    def ensure_session(self, session_id: str | None = None) -> SessionState:
        """Resolve a session or atomically create the initial/default session.

        Explicit unknown IDs are never replaced by an unrelated new session.
        """
        with self._condition:
            if session_id is not None:
                resolved_id = self._resolve_session_id(session_id)
                if resolved_id is None:
                    raise ValueError(f"Session not found: {session_id}")
                session = self._sessions[resolved_id]
                session.last_accessed = time.time()
                return session

            if self._default_session_id is not None:
                session = self._sessions[self._default_session_id]
                session.last_accessed = time.time()
                return session

            new_id = self._create_session_unlocked(
                set_as_default=True,
                session_name=None,
            )
            return self._sessions[new_id]
    
    def get_session(self, session_id: str | None = None) -> SessionState | None:
        """Return a session for identity checks; use a scope for model access."""
        with self._condition:
            if session_id is None:
                session_id = self._default_session_id
                
            if session_id is None:
                return None

            resolved_id = self._resolve_session_id(session_id)
            if resolved_id is None:
                return None

            session = self._sessions.get(resolved_id)
            if session:
                session.last_accessed = time.time()
            return session
    
    def get_default_session_id(self) -> str | None:
        """Get the default session ID."""
        with self._condition:
            return self._default_session_id
    
    def set_default_session(self, session_id: str) -> bool:
        """Set the default session."""
        with self._condition:
            resolved_id = self._resolve_session_id(session_id)
            if resolved_id is not None:
                self._default_session_id = resolved_id
                return True
            return False
    
    def list_sessions(self) -> list[SessionState]:
        """Return live sessions for compatibility and identity-only callers."""
        with self._condition:
            return list(self._sessions.values())

    def list_session_snapshots(self) -> list[dict[str, Any]]:
        """Return consistent snapshots without exposing mutable session state."""
        with self._condition:
            sessions = list(self._sessions.values())
            default_session_id = self._default_session_id
            for session in sessions:
                session._lease_count += 1

        try:
            return [
                session.snapshot(
                    is_default=session.session_id == default_session_id,
                )
                for session in sessions
            ]
        finally:
            with self._condition:
                for session in sessions:
                    session._lease_count -= 1
                self._condition.notify_all()
    
    def delete_session(self, session_id: str) -> bool:
        """Retire a session and wait for admitted operations to finish."""
        with self._condition:
            resolved_id = self._resolve_session_id(session_id)
            if resolved_id is None:
                return False

            session = self._sessions[resolved_id]
            self._retire_session_unlocked(session)
            self._condition.notify_all()
            while session._lease_count:
                self._condition.wait()

            logger.info("Deleted session %s...", resolved_id[:8])
            return True
    
    def cleanup_old_sessions(self, max_age_hours: float | None = None) -> int:
        """Clean up sessions older than specified hours."""
        if max_age_hours is None:
            max_age_hours = self._auto_cleanup_hours

        with self._condition:
            return self._cleanup_old_sessions_unlocked(max_age_hours * 3600)

    @contextmanager
    def session_scope(
        self,
        session_id: str | None = None,
    ) -> Iterator[SessionState]:
        """Lease and exclusively access an existing session."""
        with self._condition:
            requested_id = (
                session_id
                if session_id is not None
                else self._default_session_id
            )
            if requested_id is None:
                raise RuntimeError(
                    "No active session. Use `create_session()` first."
                )
            resolved_id = self._resolve_session_id(requested_id)
            if resolved_id is None:
                raise ValueError(f"Session not found: {session_id}")
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

    @contextmanager
    def ensure_session_scope(
        self,
        session_id: str | None = None,
        *,
        session_name: str | None = None,
    ) -> Iterator[SessionState]:
        """Lease an explicit/default session, creating only an absent default."""
        with self._condition:
            if session_id is not None:
                resolved_id = self._resolve_session_id(session_id)
                if resolved_id is None:
                    raise ValueError(f"Session not found: {session_id}")
                session = self._sessions[resolved_id]
            elif self._default_session_id is not None:
                session = self._sessions[self._default_session_id]
            else:
                new_id = self._create_session_unlocked(
                    set_as_default=True,
                    session_name=session_name,
                )
                session = self._sessions[new_id]
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
    def optional_session_scope(
        self,
        session_id: str | None = None,
    ) -> Iterator[SessionState | None]:
        """Lease a session when available, or yield None for no default."""
        with self._condition:
            requested_id = (
                session_id
                if session_id is not None
                else self._default_session_id
            )
            if requested_id is None:
                session = None
            else:
                resolved_id = self._resolve_session_id(requested_id)
                if resolved_id is None:
                    raise ValueError(f"Session not found: {session_id}")
                session = self._sessions[resolved_id]
                session._lease_count += 1
                self._condition.notify_all()

        if session is None:
            yield None
            return

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
    
    def save_session(self, session_id: str, filepath: Path) -> bool:
        """Save session state to file."""
        try:
            with self.session_scope(session_id) as session:
                resolved_id = session.session_id
                session_data = session.to_dict()
                session_data["has_config"] = session.config is not None
                with open(filepath, "w") as file:
                    json.dump(session_data, file, indent=2)

            logger.info("Saved session %s... to %s", resolved_id[:8], filepath)
            return True
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            logger.error(
                "Failed to save session %s...: %s",
                session_id[:8],
                error,
                exc_info=True,
            )
            return False
    
    def get_session_stats(self) -> dict:
        """Get statistics about all sessions."""
        snapshots = self.list_session_snapshots()
        total_sessions = len(snapshots)
        total_configs = sum(1 for snapshot in snapshots if snapshot["has_config"])

        if total_sessions == 0:
            return {
                "total_sessions": 0,
                "active_configs": 0,
                "avg_progress": 0.0,
                "oldest_session_age_hours": 0.0,
            }

        avg_progress = (
            sum(float(snapshot["progress"]) for snapshot in snapshots)
            / total_sessions
        )
        current_time = time.time()
        oldest_age = (
            max(
                current_time - float(snapshot["created_at"])
                for snapshot in snapshots
            )
            / 3600
        )

        return {
            "total_sessions": total_sessions,
            "active_configs": total_configs,
            "avg_progress": avg_progress,
            "oldest_session_age_hours": oldest_age,
        }
    
    def set_maboss_context(self, session_id: str | None, maboss_context: MaBoSSContext) -> bool:
        """Set MaBoSS context for a session."""
        try:
            with self.session_scope(session_id) as session:
                session.maboss_context = maboss_context
                logger.info(f"Set MaBoSS context for session {session.session_id[:8]}...")
                return True
        except (RuntimeError, ValueError):
            return False
    
    def find_session_by_name(self, session_name: str) -> SessionState | None:
        """Find session by human-readable name."""
        with self._condition:
            for session in self._sessions.values():
                if session.session_name == session_name:
                    return session
            return None
    
    def get_maboss_context(self, session_id: str | None = None) -> MaBoSSContext | None:
        """Get MaBoSS context from session."""
        try:
            with self.session_scope(session_id) as session:
                return session.maboss_context
        except (RuntimeError, ValueError):
            return None

    def get_maboss_contexts(
        self,
        session_id: str | None = None,
    ) -> dict[str, MaBoSSContext]:
        """Get a snapshot of all target-cell MaBoSS contexts."""
        try:
            with self.session_scope(session_id) as session:
                return dict(session.maboss_contexts)
        except (RuntimeError, ValueError):
            return {}

# Global session manager instance
session_manager = SessionManager()

def get_current_session(session_id: str | None = None) -> SessionState | None:
    """Convenience function to get a session by ID, or the current default session."""
    active_session = _active_session.get()
    if active_session is not None and (
        session_id is None
        or active_session.session_id == session_id
        or active_session.session_id.startswith(session_id)
    ):
        return active_session
    return session_manager.get_session(session_id)

def ensure_session(session_id: str | None = None) -> SessionState:
    """Return the requested session, the default session, or create one if none exists."""
    active_session = _active_session.get()
    if active_session is not None and (
        session_id is None
        or active_session.session_id == session_id
        or active_session.session_id.startswith(session_id)
    ):
        return active_session
    return session_manager.ensure_session(session_id)

def analyze_and_update_session_from_config(session: SessionState, config):
    """Update session state from loaded PhysiCell configuration."""
    with session._operation_lock:
        _analyze_and_update_session_from_config_unlocked(session, config)


def _analyze_and_update_session_from_config_unlocked(
    session: SessionState,
    config: Any,
) -> None:
    """Populate XML-derived metadata while the session lock is held."""
    # Extract substrates
    session.loaded_substrates = []
    try:
        if hasattr(config, 'substrates'):
            if hasattr(config.substrates, 'substrate_list'):
                session.loaded_substrates = list(config.substrates.substrate_list.keys())
            elif hasattr(config.substrates, 'get_substrates'):
                session.loaded_substrates = list(config.substrates.get_substrates().keys())
    # PhysiCell-settings objects do not expose a stable exception contract.
    except Exception:  # noqa: BLE001
        logger.debug("Could not inspect PhysiCell substrates", exc_info=True)
    session.substrates_count = len(session.loaded_substrates)

    # Extract cell types
    session.loaded_cell_types = []
    try:
        if hasattr(config, 'cell_types'):
            if hasattr(config.cell_types, 'cell_type_list'):
                session.loaded_cell_types = list(config.cell_types.cell_type_list.keys())
            elif hasattr(config.cell_types, 'get_cell_types'):
                session.loaded_cell_types = list(config.cell_types.get_cell_types().keys())
    # PhysiCell-settings objects do not expose a stable exception contract.
    except Exception:  # noqa: BLE001
        logger.debug("Could not inspect PhysiCell cell types", exc_info=True)
    session.cell_types_count = len(session.loaded_cell_types)

    # Extract PhysiBoSS models
    session.loaded_physiboss_models = []
    for cell_type_name in session.loaded_cell_types:
        try:
            cell_type = config.cell_types.get_cell_type(cell_type_name)
            if (cell_type and hasattr(cell_type, 'phenotype') and
                hasattr(cell_type.phenotype, 'intracellular') and
                cell_type.phenotype.intracellular):
                session.loaded_physiboss_models.append(cell_type_name)
        # PhysiCell-settings objects do not expose a stable exception contract.
        except Exception:  # noqa: BLE001
            logger.debug(
                "Could not inspect PhysiBoSS model for cell type %s",
                cell_type_name,
                exc_info=True,
            )
    session.physiboss_models_count = len(session.loaded_physiboss_models)

    # Check for existing rules
    session.has_existing_rules = False
    session.rules_count = 0
    try:
        if hasattr(config, "cell_rules"):
            if hasattr(config.cell_rules, "get_rules"):
                session.rules_count = len(config.cell_rules.get_rules())
            if hasattr(config.cell_rules, "rulesets"):
                session.has_existing_rules = len(config.cell_rules.rulesets) > 0
            session.has_existing_rules = (
                session.has_existing_rules or session.rules_count > 0
            )
    # PhysiCell-settings objects do not expose a stable exception contract.
    except Exception:  # noqa: BLE001
        logger.debug("Could not inspect PhysiCell rules", exc_info=True)
    
    # Mark appropriate steps complete based on loaded content
    if session.substrates_count > 0 or session.cell_types_count > 0:
        session.mark_step_complete(WorkflowStep.DOMAIN_SETUP)
    
    if session.substrates_count > 0:
        session.mark_step_complete(WorkflowStep.SUBSTRATES_ADDED)
    
    if session.cell_types_count > 0:
        session.mark_step_complete(WorkflowStep.CELL_TYPES_ADDED)
    
    if session.physiboss_models_count > 0:
        session.mark_step_complete(WorkflowStep.PHYSIBOSS_MODELS_ADDED)
    
    if session.has_existing_rules:
        session.mark_step_complete(WorkflowStep.RULES_CONFIGURED)
