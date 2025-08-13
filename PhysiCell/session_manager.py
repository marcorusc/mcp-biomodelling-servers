"""
Session Management for PhysiCell MCP Server
Maintains compatibility with HatchMCP while adding robust state management.
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Set
import logging

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
    available_nodes: List[str] = field(default_factory=list)
    output_nodes: List[str] = field(default_factory=list)
    simulation_results: str = ""  # Summary of MaBoSS simulation behavior
    target_cell_type: str = ""  # Which cell type this model targets
    biological_context: str = ""  # Original biological question/context

@dataclass
class SessionState:
    """Represents the state of a PhysiCell simulation session."""
    session_id: str
    session_name: Optional[str] = None  # Human-readable name for cross-server linking
    config: Optional[object] = None  # PhysiCellConfig instance
    scenario_context: str = ""
    maboss_context: Optional[MaBoSSContext] = None  # Context from MaBoSS analysis
    completed_steps: Set[WorkflowStep] = field(default_factory=set)
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
    original_xml_path: Optional[str] = None
    xml_modification_count: int = 0
    loaded_substrates: List[str] = field(default_factory=list)
    loaded_cell_types: List[str] = field(default_factory=list)
    loaded_physiboss_models: List[str] = field(default_factory=list)
    has_existing_rules: bool = False
    
    def mark_step_complete(self, step: WorkflowStep):
        """Mark a workflow step as completed."""
        self.completed_steps.add(step)
        self.last_accessed = time.time()
    
    def mark_xml_modification(self):
        """Track modifications to XML-loaded configuration."""
        self.xml_modification_count += 1
        self.last_accessed = time.time()
    
    def is_step_complete(self, step: WorkflowStep) -> bool:
        """Check if a workflow step is completed."""
        return step in self.completed_steps
    
    def get_next_recommended_steps(self) -> List[str]:
        """Get recommended next steps based on current progress."""
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
        result = {
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
        
        if self.maboss_context:
            result['maboss_context'] = {
                'model_name': self.maboss_context.model_name,
                'bnd_file_path': self.maboss_context.bnd_file_path,
                'cfg_file_path': self.maboss_context.cfg_file_path,
                'available_nodes': self.maboss_context.available_nodes,
                'output_nodes': self.maboss_context.output_nodes,
                'simulation_results': self.maboss_context.simulation_results,
                'target_cell_type': self.maboss_context.target_cell_type,
                'biological_context': self.maboss_context.biological_context
            }
        
        return result

class SessionManager:
    """Thread-safe session manager for PhysiCell configurations."""
    
    def __init__(self, max_sessions: int = 10, auto_cleanup_hours: float = 24.0):
        self._sessions: Dict[str, SessionState] = {}
        self._lock = Lock()
        self._max_sessions = max_sessions
        self._auto_cleanup_hours = auto_cleanup_hours
        self._default_session_id = None
        
    def create_session(self, set_as_default: bool = True, session_name: Optional[str] = None) -> str:
        """Create a new simulation session."""
        with self._lock:
            # Cleanup old sessions if needed
            self._cleanup_old_sessions()
            
            # Check session limit
            if len(self._sessions) >= self._max_sessions:
                oldest_session = min(self._sessions.values(), key=lambda s: s.last_accessed)
                del self._sessions[oldest_session.session_id]
                logger.info(f"Removed oldest session {oldest_session.session_id[:8]}... due to limit")
            
            session_id = str(uuid.uuid4())
            self._sessions[session_id] = SessionState(
                session_id=session_id,
                session_name=session_name
            )
            
            if set_as_default or self._default_session_id is None:
                self._default_session_id = session_id
                
            logger.info(f"Created session {session_id[:8]}..." + 
                       (f" ({session_name})" if session_name else ""))
            return session_id
    
    def get_session(self, session_id: Optional[str] = None) -> Optional[SessionState]:
        """Get session by ID, or default session if ID is None."""
        with self._lock:
            if session_id is None:
                session_id = self._default_session_id
                
            if session_id is None:
                return None
                
            session = self._sessions.get(session_id)
            if session:
                session.last_accessed = time.time()
            return session
    
    def get_default_session_id(self) -> Optional[str]:
        """Get the default session ID."""
        return self._default_session_id
    
    def set_default_session(self, session_id: str) -> bool:
        """Set the default session."""
        with self._lock:
            if session_id in self._sessions:
                self._default_session_id = session_id
                return True
            return False
    
    def list_sessions(self) -> List[SessionState]:
        """List all active sessions."""
        with self._lock:
            return list(self._sessions.values())
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a specific session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                if self._default_session_id == session_id:
                    # Set new default if available
                    self._default_session_id = next(iter(self._sessions.keys()), None)
                logger.info(f"Deleted session {session_id[:8]}...")
                return True
            return False
    
    def cleanup_old_sessions(self, max_age_hours: float = None) -> int:
        """Clean up sessions older than specified hours."""
        if max_age_hours is None:
            max_age_hours = self._auto_cleanup_hours
            
        with self._lock:
            return self._cleanup_old_sessions(max_age_hours * 3600)
    
    def _cleanup_old_sessions(self, max_age_seconds: float = None) -> int:
        """Internal cleanup method (assumes lock is held)."""
        if max_age_seconds is None:
            max_age_seconds = self._auto_cleanup_hours * 3600
            
        current_time = time.time()
        old_sessions = [
            session_id for session_id, session in self._sessions.items()
            if (current_time - session.last_accessed) > max_age_seconds
        ]
        
        for session_id in old_sessions:
            del self._sessions[session_id]
            if self._default_session_id == session_id:
                self._default_session_id = next(iter(self._sessions.keys()), None)
        
        if old_sessions:
            logger.info(f"Cleaned up {len(old_sessions)} old sessions")
            
        return len(old_sessions)
    
    def save_session(self, session_id: str, filepath: Path) -> bool:
        """Save session state to file."""
        session = self.get_session(session_id)
        if not session:
            return False
            
        try:
            session_data = session.to_dict()
            # Don't save the actual config object, just metadata
            session_data['has_config'] = session.config is not None
            
            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            logger.info(f"Saved session {session_id[:8]}... to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save session {session_id[:8]}...: {e}")
            return False
    
    def get_session_stats(self) -> dict:
        """Get statistics about all sessions."""
        with self._lock:
            total_sessions = len(self._sessions)
            total_configs = sum(1 for s in self._sessions.values() if s.config is not None)
            
            if total_sessions == 0:
                return {
                    'total_sessions': 0,
                    'active_configs': 0,
                    'avg_progress': 0.0,
                    'oldest_session_age_hours': 0.0
                }
            
            avg_progress = sum(s.get_progress_percentage() for s in self._sessions.values()) / total_sessions
            current_time = time.time()
            oldest_age = max((current_time - s.created_at) for s in self._sessions.values()) / 3600
            
            return {
                'total_sessions': total_sessions,
                'active_configs': total_configs,
                'avg_progress': avg_progress,
                'oldest_session_age_hours': oldest_age
            }
    
    def set_maboss_context(self, session_id: Optional[str], maboss_context: MaBoSSContext) -> bool:
        """Set MaBoSS context for a session."""
        with self._lock:
            session = self.get_session(session_id)
            if session:
                session.maboss_context = maboss_context
                logger.info(f"Set MaBoSS context for session {session.session_id[:8]}...")
                return True
            return False
    
    def find_session_by_name(self, session_name: str) -> Optional[SessionState]:
        """Find session by human-readable name."""
        with self._lock:
            for session in self._sessions.values():
                if session.session_name == session_name:
                    return session
            return None
    
    def get_maboss_context(self, session_id: Optional[str] = None) -> Optional[MaBoSSContext]:
        """Get MaBoSS context from session."""
        session = self.get_session(session_id)
        return session.maboss_context if session else None

# Global session manager instance
session_manager = SessionManager()

def get_current_session() -> Optional[SessionState]:
    """Convenience function to get current session."""
    return session_manager.get_session()

def ensure_session() -> SessionState:
    """Ensure there's an active session, create one if needed."""
    session = session_manager.get_session()
    if session is None:
        session_id = session_manager.create_session()
        session = session_manager.get_session(session_id)
    return session

def analyze_and_update_session_from_config(session: SessionState, config):
    """Update session state from loaded PhysiCell configuration."""
    # Extract substrates
    session.loaded_substrates = []
    try:
        if hasattr(config, 'substrates'):
            if hasattr(config.substrates, 'substrate_list'):
                session.loaded_substrates = list(config.substrates.substrate_list.keys())
            elif hasattr(config.substrates, 'get_substrates'):
                session.loaded_substrates = list(config.substrates.get_substrates().keys())
    except:
        pass
    session.substrates_count = len(session.loaded_substrates)
    
    # Extract cell types
    session.loaded_cell_types = []
    try:
        if hasattr(config, 'cell_types'):
            if hasattr(config.cell_types, 'cell_type_list'):
                session.loaded_cell_types = list(config.cell_types.cell_type_list.keys())
            elif hasattr(config.cell_types, 'get_cell_types'):
                session.loaded_cell_types = list(config.cell_types.get_cell_types().keys())
    except:
        pass
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
        except:
            pass
    session.physiboss_models_count = len(session.loaded_physiboss_models)
    
    # Check for existing rules
    session.has_existing_rules = False
    try:
        if hasattr(config, 'cell_rules') and hasattr(config.cell_rules, 'rulesets'):
            session.has_existing_rules = len(config.cell_rules.rulesets) > 0
    except:
        pass
    
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
