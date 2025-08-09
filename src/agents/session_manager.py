"""Session Manager Agent - Session Lifecycle and State Management

The Session Manager is responsible for:
- Multi-session concurrent management
- Incremental state persistence
- Session recovery and migration  
- Intelligent historical data cleanup
"""

import asyncio
import json
import pickle
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import os

from loguru import logger

from ..core.event_system import EventSystem
from ..core.types import Context, RiskLevel
from ..learning.learning_data_manager import LearningDataManager, LearningData, DataPrivacyLevel
from ..learning.learning_events import LearningEventFactory, LearningEventType
from .base_agent import BaseAgent
from .agent_types import (
    AgentType, AgentRequest, AgentResponse, AgentEvent, AgentCapability
)


class SessionState(Enum):
    """Session lifecycle states"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    SUSPENDED = "suspended"
    RECOVERING = "recovering"
    TERMINATED = "terminated"


class PersistenceLevel(Enum):
    """Levels of session persistence"""
    MINIMAL = "minimal"          # Only critical state
    STANDARD = "standard"        # Standard session data
    COMPREHENSIVE = "comprehensive"  # Full state including history


@dataclass
class SessionData:
    """Complete session state data"""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    state: SessionState
    context: Context
    active_tasks: List[Dict[str, Any]] = field(default_factory=list)
    completed_tasks: List[Dict[str, Any]] = field(default_factory=list)
    failed_tasks: List[Dict[str, Any]] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    session_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Learning system integration
    learning_enabled: bool = True
    learning_data: List[Dict[str, Any]] = field(default_factory=list)
    user_patterns: Dict[str, Any] = field(default_factory=dict)
    adaptation_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionSnapshot:
    """Lightweight session snapshot for persistence"""
    session_id: str
    timestamp: datetime
    state: SessionState
    critical_data: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""


class SessionManager(BaseAgent):
    """
    Session lifecycle and state management agent.
    
    Manages multiple concurrent user sessions with intelligent
    state persistence and recovery capabilities.
    """
    
    def __init__(self, event_system: EventSystem, config: Optional[Dict[str, Any]] = None):
        super().__init__("session_manager", event_system, config)
        
        # Session management
        self._active_sessions: Dict[str, SessionData] = {}
        self._session_snapshots: Dict[str, List[SessionSnapshot]] = {}
        self._max_sessions = self.config.get('max_concurrent_sessions', 10)
        
        # Learning system integration
        self._learning_enabled = self.config.get('learning_enabled', True)
        self._learning_data_manager: Optional[LearningDataManager] = None
        
        # Persistence configuration
        self._persistence_level = PersistenceLevel.STANDARD
        self._persistence_interval = self.config.get('persistence_interval', 300)  # 5 minutes
        self._storage_path = self.config.get('storage_path', './sessions')
        
        # Cleanup configuration
        self._max_session_history = self.config.get('max_session_history', 1000)
        self._session_timeout = timedelta(hours=self.config.get('session_timeout_hours', 24))
        self._cleanup_interval = self.config.get('cleanup_interval', 3600)  # 1 hour
        
        # Statistics
        self._session_stats = {
            'total_sessions_created': 0,
            'active_sessions': 0,
            'sessions_recovered': 0,
            'persistence_operations': 0,
            'cleanup_operations': 0
        }
        
        # Initialize storage
        self._initialize_storage()
    
    @property
    def agent_type(self) -> AgentType:
        return AgentType.SESSION_MANAGER
    
    @property
    def name(self) -> str:
        return "Session Manager"
    
    @property
    def description(self) -> str:
        return "Session lifecycle and state management agent"
    
    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="create_session",
                description="Create new user session",
                input_types=["user_info", "session_config"],
                output_types=["session_data"],
                risk_level=RiskLevel.LOW,
                execution_time_ms=200
            ),
            AgentCapability(
                name="get_session",
                description="Get session information and state",
                input_types=["session_id"],
                output_types=["session_data"],
                risk_level=RiskLevel.LOW,
                execution_time_ms=50
            ),
            AgentCapability(
                name="update_session",
                description="Update session state and data",
                input_types=["session_update"],
                output_types=["update_result"],
                risk_level=RiskLevel.LOW,
                execution_time_ms=100
            ),
            AgentCapability(
                name="terminate_session",
                description="Terminate and cleanup session",
                input_types=["session_id"],
                output_types=["termination_result"],
                risk_level=RiskLevel.MEDIUM,
                execution_time_ms=300
            ),
            AgentCapability(
                name="recover_session",
                description="Recover session from persistent storage",
                input_types=["session_id", "recovery_options"],
                output_types=["recovery_result"],
                risk_level=RiskLevel.MEDIUM,
                execution_time_ms=500
            ),
            AgentCapability(
                name="list_sessions",
                description="List active or historical sessions",
                input_types=["query_params"],
                output_types=["session_list"],
                risk_level=RiskLevel.LOW,
                execution_time_ms=100
            ),
            AgentCapability(
                name="persist_session",
                description="Force persistence of session state",
                input_types=["session_id", "persistence_level"],
                output_types=["persistence_result"],
                risk_level=RiskLevel.LOW,
                execution_time_ms=200
            ),
            AgentCapability(
                name="migrate_session",
                description="Migrate session between environments",
                input_types=["migration_request"],
                output_types=["migration_result"],
                risk_level=RiskLevel.HIGH,
                execution_time_ms=1000
            )
        ]
    
    async def _initialize_agent(self) -> None:
        """Initialize session manager specific functionality."""
        self.logger.info("Initializing Session Manager agent")
        
        # Load configuration
        await self._load_session_config()
        
        # Initialize learning data manager if enabled
        if self._learning_enabled:
            learning_config = self.config.get('learning', {})
            self._learning_data_manager = LearningDataManager(
                event_system=self.event_system,
                config=learning_config
            )
            await self._learning_data_manager.initialize()
            self.logger.info("Learning Data Manager initialized")
        
        # Recover existing sessions
        await self._recover_existing_sessions()
        
        # Start background tasks
        persistence_task = asyncio.create_task(self._persistence_loop())
        self._background_tasks.add(persistence_task)
        
        cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._background_tasks.add(cleanup_task)
        
        monitor_task = asyncio.create_task(self._session_monitor_loop())
        self._background_tasks.add(monitor_task)
        
        # Start learning-related background task
        if self._learning_enabled:
            learning_task = asyncio.create_task(self._learning_analysis_loop())
            self._background_tasks.add(learning_task)
        
        self.logger.info("Session Manager initialization complete")
    
    async def _process_request(self, request: AgentRequest) -> AgentResponse:
        """Process incoming requests."""
        capability = request.target_capability
        start_time = datetime.now()
        
        try:
            if capability == "create_session":
                result = await self._create_session(request.parameters)
            elif capability == "get_session":
                result = await self._get_session(request.parameters)
            elif capability == "update_session":
                result = await self._update_session(request.parameters)
            elif capability == "terminate_session":
                result = await self._terminate_session(request.parameters)
            elif capability == "recover_session":
                result = await self._recover_session(request.parameters)
            elif capability == "list_sessions":
                result = await self._list_sessions(request.parameters)
            elif capability == "persist_session":
                result = await self._persist_session(request.parameters)
            elif capability == "migrate_session":
                result = await self._migrate_session(request.parameters)
            else:
                raise ValueError(f"Unknown capability: {capability}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                request_id=request.request_id,
                responding_agent=self.agent_id,
                success=True,
                data=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Error processing {capability}: {e}")
            return AgentResponse(
                request_id=request.request_id,
                responding_agent=self.agent_id,
                success=False,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def _handle_event(self, event: AgentEvent) -> None:
        """Handle agent events."""
        try:
            if event.event_type == "user.activity":
                await self._handle_user_activity(event)
            elif event.event_type == "session.timeout":
                await self._handle_session_timeout(event)
            elif event.event_type == "system.shutdown":
                await self._handle_system_shutdown(event)
            elif event.event_type == "task.completed":
                await self._handle_task_completion(event)
                
        except Exception as e:
            await self._handle_error(e, f"event_handling_{event.event_type}")
    
    async def _cleanup_agent(self) -> None:
        """Cleanup session manager resources."""
        # Persist all active sessions
        await self._persist_all_sessions()
        
        # Terminate active sessions gracefully
        for session_id in list(self._active_sessions.keys()):
            await self._terminate_session_internal(session_id, graceful=True)
        
        # Cleanup learning data manager
        if self._learning_data_manager:
            await self._learning_data_manager.shutdown()
        
        self.logger.info("Session Manager cleanup complete")
    
    # Private implementation methods
    
    async def _create_session(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create new user session."""
        user_id = parameters.get("user_id", "anonymous")
        context_data = parameters.get("context", {})
        preferences = parameters.get("preferences", {})
        
        # Check session limits
        if len(self._active_sessions) >= self._max_sessions:
            # Try to clean up idle sessions
            await self._cleanup_idle_sessions()
            
            if len(self._active_sessions) >= self._max_sessions:
                raise ValueError(f"Maximum session limit ({self._max_sessions}) reached")
        
        # Create session
        session_id = self._generate_session_id()
        current_time = datetime.now()
        
        # Create context
        if context_data:
            context = Context(**context_data)
        else:
            context = Context(
                user_id=user_id,
                session_id=session_id,
                timestamp=current_time
            )
        
        session_data = SessionData(
            session_id=session_id,
            user_id=user_id,
            created_at=current_time,
            last_activity=current_time,
            state=SessionState.INITIALIZING,
            context=context,
            user_preferences=preferences
        )
        
        # Store session
        self._active_sessions[session_id] = session_data
        
        # Update statistics
        self._session_stats['total_sessions_created'] += 1
        self._session_stats['active_sessions'] = len(self._active_sessions)
        
        # Set session to active
        session_data.state = SessionState.ACTIVE
        
        # Create initial snapshot
        await self._create_session_snapshot(session_data)
        
        # Emit session created event
        await self._emit_event("session.created", {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": current_time.isoformat()
        })
        
        return {
            "session_data": self._session_to_dict(session_data)
        }
    
    async def _get_session(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get session information."""
        session_id = parameters.get("session_id")
        if not session_id:
            raise ValueError("Session ID required")
        
        if session_id not in self._active_sessions:
            # Try to recover session
            recovered = await self._try_recover_session(session_id)
            if not recovered:
                raise ValueError(f"Session {session_id} not found")
        
        session_data = self._active_sessions[session_id]
        
        return {
            "session_data": self._session_to_dict(session_data)
        }
    
    async def _update_session(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Update session state and data."""
        session_id = parameters.get("session_id")
        updates = parameters.get("updates", {})
        
        if not session_id:
            raise ValueError("Session ID required")
        
        if session_id not in self._active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session_data = self._active_sessions[session_id]
        
        # Update session data
        if "context" in updates:
            context_updates = updates["context"]
            for key, value in context_updates.items():
                setattr(session_data.context, key, value)
        
        if "preferences" in updates:
            session_data.user_preferences.update(updates["preferences"])
        
        if "metadata" in updates:
            session_data.metadata.update(updates["metadata"])
        
        if "active_tasks" in updates:
            session_data.active_tasks = updates["active_tasks"]
        
        # Update activity timestamp
        session_data.last_activity = datetime.now()
        
        # Store learning data if enabled
        if self._learning_enabled and self._learning_data_manager:
            await self._capture_session_learning_data(session_data, updates)
        
        # Create snapshot for significant updates
        if any(key in updates for key in ["context", "active_tasks"]):
            await self._create_session_snapshot(session_data)
        
        return {
            "update_result": {
                "session_id": session_id,
                "updated_at": session_data.last_activity.isoformat(),
                "status": "success"
            }
        }
    
    async def _terminate_session(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Terminate and cleanup session."""
        session_id = parameters.get("session_id")
        graceful = parameters.get("graceful", True)
        
        if not session_id:
            raise ValueError("Session ID required")
        
        result = await self._terminate_session_internal(session_id, graceful)
        
        return {"termination_result": result}
    
    async def _terminate_session_internal(self, session_id: str, graceful: bool = True) -> Dict[str, Any]:
        """Internal session termination."""
        if session_id not in self._active_sessions:
            return {
                "session_id": session_id,
                "status": "not_found",
                "message": "Session not found"
            }
        
        session_data = self._active_sessions[session_id]
        
        # Update session state
        session_data.state = SessionState.TERMINATED
        session_data.last_activity = datetime.now()
        
        # Persist final state if graceful
        if graceful:
            await self._persist_session_internal(session_data, PersistenceLevel.COMPREHENSIVE)
        
        # Emit termination event
        await self._emit_event("session.terminated", {
            "session_id": session_id,
            "user_id": session_data.user_id,
            "terminated_at": session_data.last_activity.isoformat(),
            "graceful": graceful
        })
        
        # Remove from active sessions
        del self._active_sessions[session_id]
        
        # Update statistics
        self._session_stats['active_sessions'] = len(self._active_sessions)
        
        return {
            "session_id": session_id,
            "status": "terminated",
            "terminated_at": session_data.last_activity.isoformat()
        }
    
    async def _recover_session(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Recover session from persistent storage."""
        session_id = parameters.get("session_id")
        recovery_options = parameters.get("recovery_options", {})
        
        if not session_id:
            raise ValueError("Session ID required")
        
        # Check if session is already active
        if session_id in self._active_sessions:
            return {
                "recovery_result": {
                    "session_id": session_id,
                    "status": "already_active",
                    "message": "Session is already active"
                }
            }
        
        # Attempt recovery
        recovered_session = await self._load_session_from_storage(session_id)
        
        if not recovered_session:
            return {
                "recovery_result": {
                    "session_id": session_id,
                    "status": "not_found",
                    "message": "Session data not found in storage"
                }
            }
        
        # Restore session
        recovered_session.state = SessionState.RECOVERING
        recovered_session.last_activity = datetime.now()
        
        self._active_sessions[session_id] = recovered_session
        
        # Update statistics
        self._session_stats['sessions_recovered'] += 1
        self._session_stats['active_sessions'] = len(self._active_sessions)
        
        # Set session to active
        recovered_session.state = SessionState.ACTIVE
        
        # Emit recovery event
        await self._emit_event("session.recovered", {
            "session_id": session_id,
            "user_id": recovered_session.user_id,
            "recovered_at": datetime.now().isoformat()
        })
        
        return {
            "recovery_result": {
                "session_id": session_id,
                "status": "recovered",
                "session_data": self._session_to_dict(recovered_session)
            }
        }
    
    async def _list_sessions(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """List active or historical sessions."""
        user_id = parameters.get("user_id")
        state_filter = parameters.get("state")
        limit = parameters.get("limit", 50)
        include_terminated = parameters.get("include_terminated", False)
        
        # Get active sessions
        sessions = []
        for session_data in self._active_sessions.values():
            if user_id and session_data.user_id != user_id:
                continue
            
            if state_filter and session_data.state.value != state_filter:
                continue
            
            sessions.append(self._session_to_dict(session_data))
        
        # Include terminated sessions if requested
        if include_terminated:
            # Would load from persistent storage
            pass
        
        # Apply limit
        if limit:
            sessions = sessions[:limit]
        
        return {
            "session_list": {
                "sessions": sessions,
                "total_count": len(sessions),
                "active_count": len(self._active_sessions)
            }
        }
    
    async def _persist_session(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Force persistence of session state."""
        session_id = parameters.get("session_id")
        level = parameters.get("persistence_level", "standard")
        
        if not session_id:
            raise ValueError("Session ID required")
        
        if session_id not in self._active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session_data = self._active_sessions[session_id]
        persistence_level = PersistenceLevel(level)
        
        result = await self._persist_session_internal(session_data, persistence_level)
        
        return {
            "persistence_result": {
                "session_id": session_id,
                "persistence_level": level,
                "status": "success" if result else "failed",
                "timestamp": datetime.now().isoformat()
            }
        }
    
    async def _migrate_session(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate session between environments."""
        session_id = parameters.get("session_id")
        target_environment = parameters.get("target_environment")
        migration_options = parameters.get("migration_options", {})
        
        if not session_id or not target_environment:
            raise ValueError("Session ID and target environment required")
        
        if session_id not in self._active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        # For now, return a placeholder implementation
        return {
            "migration_result": {
                "session_id": session_id,
                "target_environment": target_environment,
                "status": "not_implemented",
                "message": "Session migration not yet implemented"
            }
        }
    
    def _initialize_storage(self) -> None:
        """Initialize session storage."""
        try:
            os.makedirs(self._storage_path, exist_ok=True)
            self.logger.info(f"Session storage initialized at: {self._storage_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize session storage: {e}")
    
    async def _load_session_config(self) -> None:
        """Load session configuration."""
        session_config = self.config.get("session", {})
        
        # Set persistence level
        level = session_config.get("persistence_level", "standard")
        try:
            self._persistence_level = PersistenceLevel(level)
        except ValueError:
            self.logger.warning(f"Invalid persistence level: {level}")
    
    async def _recover_existing_sessions(self) -> None:
        """Recover existing sessions from storage."""
        try:
            if not os.path.exists(self._storage_path):
                return
            
            session_files = [f for f in os.listdir(self._storage_path) if f.endswith('.session')]
            
            for session_file in session_files:
                session_id = session_file.replace('.session', '')
                
                try:
                    session_data = await self._load_session_from_storage(session_id)
                    if session_data:
                        # Only recover if session is not too old
                        age = datetime.now() - session_data.last_activity
                        if age < self._session_timeout:
                            session_data.state = SessionState.RECOVERING
                            self._active_sessions[session_id] = session_data
                            session_data.state = SessionState.ACTIVE
                            
                            self._session_stats['sessions_recovered'] += 1
                            
                            self.logger.info(f"Recovered session: {session_id}")
                
                except Exception as e:
                    self.logger.error(f"Failed to recover session {session_id}: {e}")
            
            self._session_stats['active_sessions'] = len(self._active_sessions)
            
        except Exception as e:
            self.logger.error(f"Error recovering existing sessions: {e}")
    
    async def _persistence_loop(self) -> None:
        """Background persistence loop."""
        while not self._shutdown_requested:
            try:
                await asyncio.sleep(self._persistence_interval)
                
                # Persist all active sessions
                await self._persist_all_sessions()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._handle_error(e, "persistence_loop")
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while not self._shutdown_requested:
            try:
                await asyncio.sleep(self._cleanup_interval)
                
                # Clean up idle sessions
                await self._cleanup_idle_sessions()
                
                # Clean up old snapshots
                await self._cleanup_old_snapshots()
                
                # Clean up old session files
                await self._cleanup_old_session_files()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._handle_error(e, "cleanup_loop")
    
    async def _session_monitor_loop(self) -> None:
        """Background session monitoring loop."""
        while not self._shutdown_requested:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                current_time = datetime.now()
                
                # Check for timed out sessions
                timed_out_sessions = []
                for session_id, session_data in self._active_sessions.items():
                    if current_time - session_data.last_activity > self._session_timeout:
                        timed_out_sessions.append(session_id)
                
                # Handle timed out sessions
                for session_id in timed_out_sessions:
                    await self._handle_session_timeout_internal(session_id)
                
                # Update activity statistics
                idle_sessions = [
                    s for s in self._active_sessions.values()
                    if s.state == SessionState.IDLE
                ]
                
                if len(idle_sessions) > 0:
                    self.logger.debug(f"{len(idle_sessions)} idle sessions detected")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._handle_error(e, "session_monitor")
    
    async def _persist_all_sessions(self) -> None:
        """Persist all active sessions."""
        for session_data in self._active_sessions.values():
            try:
                await self._persist_session_internal(session_data, self._persistence_level)
            except Exception as e:
                self.logger.error(f"Failed to persist session {session_data.session_id}: {e}")
    
    async def _persist_session_internal(self, session_data: SessionData, 
                                      level: PersistenceLevel) -> bool:
        """Internal session persistence."""
        try:
            session_file = os.path.join(self._storage_path, f"{session_data.session_id}.session")
            
            # Prepare data based on persistence level
            if level == PersistenceLevel.MINIMAL:
                persist_data = {
                    "session_id": session_data.session_id,
                    "user_id": session_data.user_id,
                    "created_at": session_data.created_at.isoformat(),
                    "last_activity": session_data.last_activity.isoformat(),
                    "state": session_data.state.value,
                    "context": asdict(session_data.context),
                    "active_tasks": session_data.active_tasks
                }
            elif level == PersistenceLevel.STANDARD:
                persist_data = self._session_to_dict(session_data)
                # Remove history for standard level
                persist_data.pop("session_history", None)
            else:  # COMPREHENSIVE
                persist_data = self._session_to_dict(session_data)
            
            # Save to file
            with open(session_file, 'w') as f:
                json.dump(persist_data, f, indent=2, default=str)
            
            # Create snapshot
            await self._create_session_snapshot(session_data)
            
            self._session_stats['persistence_operations'] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to persist session {session_data.session_id}: {e}")
            return False
    
    async def _load_session_from_storage(self, session_id: str) -> Optional[SessionData]:
        """Load session from storage."""
        try:
            session_file = os.path.join(self._storage_path, f"{session_id}.session")
            
            if not os.path.exists(session_file):
                return None
            
            with open(session_file, 'r') as f:
                session_dict = json.load(f)
            
            # Convert back to SessionData
            session_data = self._dict_to_session(session_dict)
            
            return session_data
            
        except Exception as e:
            self.logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    async def _create_session_snapshot(self, session_data: SessionData) -> None:
        """Create a session snapshot."""
        snapshot = SessionSnapshot(
            session_id=session_data.session_id,
            timestamp=datetime.now(),
            state=session_data.state,
            critical_data={
                "user_id": session_data.user_id,
                "active_tasks_count": len(session_data.active_tasks),
                "last_activity": session_data.last_activity.isoformat()
            }
        )
        
        # Generate checksum
        snapshot.checksum = self._generate_session_checksum(session_data)
        
        # Store snapshot
        if session_data.session_id not in self._session_snapshots:
            self._session_snapshots[session_data.session_id] = []
        
        self._session_snapshots[session_data.session_id].append(snapshot)
        
        # Maintain snapshot limit
        snapshots = self._session_snapshots[session_data.session_id]
        if len(snapshots) > 10:  # Keep last 10 snapshots
            self._session_snapshots[session_data.session_id] = snapshots[-10:]
    
    def _generate_session_checksum(self, session_data: SessionData) -> str:
        """Generate checksum for session data."""
        import hashlib
        
        # Create checksum from critical data
        checksum_data = f"{session_data.session_id}:{session_data.state.value}:{len(session_data.active_tasks)}"
        return hashlib.md5(checksum_data.encode()).hexdigest()
    
    async def _cleanup_idle_sessions(self) -> None:
        """Clean up idle sessions."""
        current_time = datetime.now()
        idle_threshold = timedelta(hours=2)  # Consider sessions idle after 2 hours
        
        idle_sessions = []
        for session_id, session_data in self._active_sessions.items():
            if (current_time - session_data.last_activity > idle_threshold and 
                session_data.state == SessionState.ACTIVE):
                session_data.state = SessionState.IDLE
                idle_sessions.append(session_id)
        
        # Suspend very old idle sessions
        suspend_threshold = timedelta(hours=6)
        for session_id, session_data in self._active_sessions.items():
            if (current_time - session_data.last_activity > suspend_threshold and
                session_data.state == SessionState.IDLE):
                session_data.state = SessionState.SUSPENDED
                # Persist before suspending
                await self._persist_session_internal(session_data, PersistenceLevel.STANDARD)
        
        if idle_sessions:
            self.logger.info(f"Marked {len(idle_sessions)} sessions as idle")
    
    async def _cleanup_old_snapshots(self) -> None:
        """Clean up old session snapshots."""
        cutoff_time = datetime.now() - timedelta(days=7)
        
        for session_id, snapshots in self._session_snapshots.items():
            original_count = len(snapshots)
            filtered_snapshots = [
                s for s in snapshots if s.timestamp > cutoff_time
            ]
            self._session_snapshots[session_id] = filtered_snapshots
            
            removed_count = original_count - len(filtered_snapshots)
            if removed_count > 0:
                self.logger.debug(f"Cleaned up {removed_count} old snapshots for session {session_id}")
    
    async def _cleanup_old_session_files(self) -> None:
        """Clean up old session files from storage."""
        try:
            if not os.path.exists(self._storage_path):
                return
            
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(days=30)  # Keep files for 30 days
            
            session_files = [f for f in os.listdir(self._storage_path) if f.endswith('.session')]
            removed_count = 0
            
            for session_file in session_files:
                file_path = os.path.join(self._storage_path, session_file)
                file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if file_modified < cutoff_time:
                    # Check if session is still active
                    session_id = session_file.replace('.session', '')
                    if session_id not in self._active_sessions:
                        os.remove(file_path)
                        removed_count += 1
            
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} old session files")
                self._session_stats['cleanup_operations'] += 1
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old session files: {e}")
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    
    def _session_to_dict(self, session_data: SessionData) -> Dict[str, Any]:
        """Convert session data to dictionary."""
        return {
            "session_id": session_data.session_id,
            "user_id": session_data.user_id,
            "created_at": session_data.created_at.isoformat(),
            "last_activity": session_data.last_activity.isoformat(),
            "state": session_data.state.value,
            "context": asdict(session_data.context),
            "active_tasks": session_data.active_tasks,
            "completed_tasks": session_data.completed_tasks,
            "failed_tasks": session_data.failed_tasks,
            "user_preferences": session_data.user_preferences,
            "session_history": session_data.session_history[-self._max_session_history:],
            "metadata": session_data.metadata
        }
    
    def _dict_to_session(self, session_dict: Dict[str, Any]) -> SessionData:
        """Convert dictionary to session data."""
        context_data = session_dict.get("context", {})
        context = Context(**context_data)
        
        return SessionData(
            session_id=session_dict["session_id"],
            user_id=session_dict["user_id"],
            created_at=datetime.fromisoformat(session_dict["created_at"]),
            last_activity=datetime.fromisoformat(session_dict["last_activity"]),
            state=SessionState(session_dict["state"]),
            context=context,
            active_tasks=session_dict.get("active_tasks", []),
            completed_tasks=session_dict.get("completed_tasks", []),
            failed_tasks=session_dict.get("failed_tasks", []),
            user_preferences=session_dict.get("user_preferences", {}),
            session_history=session_dict.get("session_history", []),
            metadata=session_dict.get("metadata", {})
        )
    
    async def _try_recover_session(self, session_id: str) -> bool:
        """Try to recover a session if not in active sessions."""
        session_data = await self._load_session_from_storage(session_id)
        
        if session_data:
            session_data.state = SessionState.ACTIVE
            self._active_sessions[session_id] = session_data
            self._session_stats['sessions_recovered'] += 1
            self._session_stats['active_sessions'] = len(self._active_sessions)
            return True
        
        return False
    
    async def _handle_user_activity(self, event: AgentEvent) -> None:
        """Handle user activity events."""
        session_id = event.data.get("session_id")
        
        if session_id and session_id in self._active_sessions:
            session_data = self._active_sessions[session_id]
            session_data.last_activity = datetime.now()
            
            # Update state if idle or suspended
            if session_data.state in [SessionState.IDLE, SessionState.SUSPENDED]:
                session_data.state = SessionState.ACTIVE
    
    async def _handle_session_timeout(self, event: AgentEvent) -> None:
        """Handle session timeout events."""
        session_id = event.data.get("session_id")
        if session_id:
            await self._handle_session_timeout_internal(session_id)
    
    async def _handle_session_timeout_internal(self, session_id: str) -> None:
        """Internal session timeout handling."""
        if session_id in self._active_sessions:
            self.logger.info(f"Session {session_id} timed out")
            await self._terminate_session_internal(session_id, graceful=True)
    
    async def _handle_system_shutdown(self, event: AgentEvent) -> None:
        """Handle system shutdown events."""
        # Persist all sessions before shutdown
        await self._persist_all_sessions()
    
    async def _handle_task_completion(self, event: AgentEvent) -> None:
        """Handle task completion events."""
        session_id = event.data.get("session_id")
        task_data = event.data.get("task", {})
        
        if session_id and session_id in self._active_sessions:
            session_data = self._active_sessions[session_id]
            
            # Move task from active to completed
            task_id = task_data.get("task_id")
            session_data.active_tasks = [
                t for t in session_data.active_tasks 
                if t.get("task_id") != task_id
            ]
            session_data.completed_tasks.append(task_data)
            
            # Update activity
            session_data.last_activity = datetime.now()
            
            # Add to history
            session_data.session_history.append({
                "timestamp": datetime.now().isoformat(),
                "event": "task_completed",
                "task_id": task_id
            })
            
            # Maintain history size
            if len(session_data.session_history) > self._max_session_history:
                session_data.session_history = session_data.session_history[-self._max_session_history:]
            
            # Store learning data if enabled
            if self._learning_enabled and self._learning_data_manager:
                await self._capture_task_completion_learning_data(session_data, task_data)
    
    # Learning system integration methods
    
    async def _capture_session_learning_data(self, session_data: SessionData, 
                                           updates: Dict[str, Any]) -> None:
        """Capture session interaction data for learning."""
        try:
            if not self._learning_data_manager:
                return
            
            learning_data = LearningData(
                user_id=session_data.user_id,
                agent_id=self.agent_id,
                session_id=session_data.session_id,
                data_type="session_update",
                data_content={
                    "session_state": session_data.state.value,
                    "updates": updates,
                    "active_tasks_count": len(session_data.active_tasks),
                    "completed_tasks_count": len(session_data.completed_tasks),
                    "user_preferences": session_data.user_preferences,
                    "interaction_timestamp": datetime.now().isoformat()
                },
                privacy_level=DataPrivacyLevel.PRIVATE,
                metadata={
                    "session_age_seconds": (datetime.now() - session_data.created_at).total_seconds(),
                    "last_activity": session_data.last_activity.isoformat()
                }
            )
            
            success = await self._learning_data_manager.store_learning_data(learning_data)
            if success:
                session_data.learning_data.append({
                    "data_id": learning_data.data_id,
                    "timestamp": learning_data.created_at.isoformat(),
                    "data_type": learning_data.data_type
                })
                
        except Exception as e:
            self.logger.error(f"Failed to capture session learning data: {e}")
    
    async def _capture_task_completion_learning_data(self, session_data: SessionData, 
                                                   task_data: Dict[str, Any]) -> None:
        """Capture task completion data for learning."""
        try:
            if not self._learning_data_manager:
                return
            
            learning_data = LearningData(
                user_id=session_data.user_id,
                agent_id=self.agent_id,
                session_id=session_data.session_id,
                data_type="task_completion",
                data_content={
                    "task_data": task_data,
                    "completion_time": datetime.now().isoformat(),
                    "session_context": {
                        "active_tasks": len(session_data.active_tasks),
                        "completed_tasks": len(session_data.completed_tasks),
                        "session_duration": (datetime.now() - session_data.created_at).total_seconds()
                    },
                    "user_preferences": session_data.user_preferences
                },
                privacy_level=DataPrivacyLevel.PRIVATE,
                metadata={
                    "task_success": task_data.get("success", True),
                    "execution_time": task_data.get("execution_time", 0)
                }
            )
            
            await self._learning_data_manager.store_learning_data(learning_data)
            
        except Exception as e:
            self.logger.error(f"Failed to capture task completion learning data: {e}")
    
    async def _analyze_user_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze user interaction patterns from learning data."""
        try:
            if not self._learning_data_manager:
                return {}
            
            # Retrieve user's learning data
            learning_data = await self._learning_data_manager.retrieve_learning_data(
                user_id=user_id,
                limit=500
            )
            
            if not learning_data:
                return {}
            
            patterns = {
                "interaction_frequency": {},
                "preferred_times": [],
                "common_task_types": {},
                "session_duration_patterns": [],
                "success_patterns": {}
            }
            
            for data in learning_data:
                # Analyze interaction frequency by day
                day = data.created_at.strftime("%A")
                patterns["interaction_frequency"][day] = patterns["interaction_frequency"].get(day, 0) + 1
                
                # Analyze preferred interaction times
                hour = data.created_at.hour
                if hour not in patterns["preferred_times"]:
                    patterns["preferred_times"].append(hour)
                
                # Analyze common task types
                if data.data_type == "task_completion":
                    task_type = data.data_content.get("task_data", {}).get("type", "unknown")
                    patterns["common_task_types"][task_type] = patterns["common_task_types"].get(task_type, 0) + 1
                
                # Analyze session durations
                if "session_duration" in data.data_content.get("session_context", {}):
                    duration = data.data_content["session_context"]["session_duration"]
                    patterns["session_duration_patterns"].append(duration)
            
            # Generate insights from patterns
            insights = self._generate_pattern_insights(patterns)
            
            # Emit learning event
            event = LearningEventFactory.user_pattern_detected(
                user_id=user_id,
                pattern_type="interaction_patterns",
                pattern_data=patterns,
                confidence=0.8
            )
            
            system_event = event.to_system_event()
            await self.event_system.emit(system_event)
            
            return {
                "patterns": patterns,
                "insights": insights,
                "data_points_analyzed": len(learning_data)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze user patterns: {e}")
            return {}
    
    def _generate_pattern_insights(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate insights from analyzed patterns."""
        insights = []
        
        # Interaction frequency insights
        freq = patterns.get("interaction_frequency", {})
        if freq:
            most_active_day = max(freq, key=freq.get)
            insights.append(f"Most active on {most_active_day} with {freq[most_active_day]} interactions")
        
        # Time preference insights
        preferred_times = patterns.get("preferred_times", [])
        if preferred_times:
            avg_hour = sum(preferred_times) / len(preferred_times)
            if 6 <= avg_hour <= 12:
                insights.append("Prefers morning interactions")
            elif 12 <= avg_hour <= 18:
                insights.append("Prefers afternoon interactions")
            else:
                insights.append("Prefers evening interactions")
        
        # Task type insights
        task_types = patterns.get("common_task_types", {})
        if task_types:
            most_common_task = max(task_types, key=task_types.get)
            insights.append(f"Most common task type: {most_common_task}")
        
        # Session duration insights
        durations = patterns.get("session_duration_patterns", [])
        if durations:
            avg_duration = sum(durations) / len(durations)
            if avg_duration < 300:  # 5 minutes
                insights.append("Prefers short, focused sessions")
            elif avg_duration > 1800:  # 30 minutes
                insights.append("Engages in extended sessions")
            else:
                insights.append("Has moderate session lengths")
        
        return insights
    
    async def _learning_analysis_loop(self) -> None:
        """Background learning analysis loop."""
        while not self._shutdown_requested:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Analyze patterns for active users
                active_users = set()
                for session_data in self._active_sessions.values():
                    if session_data.user_id and session_data.user_id != "anonymous":
                        active_users.add(session_data.user_id)
                
                # Perform pattern analysis for each user
                for user_id in active_users:
                    try:
                        patterns = await self._analyze_user_patterns(user_id)
                        if patterns:
                            self.logger.debug(f"Updated patterns for user {user_id}")
                    except Exception as e:
                        self.logger.error(f"Failed to analyze patterns for user {user_id}: {e}")
                
                # Update session data with learned patterns
                await self._update_session_adaptations()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in learning analysis loop: {e}")
    
    async def _update_session_adaptations(self) -> None:
        """Update session configurations based on learned patterns."""
        try:
            if not self._learning_data_manager:
                return
            
            for session_id, session_data in self._active_sessions.items():
                if not session_data.user_id or session_data.user_id == "anonymous":
                    continue
                
                # Get user profile
                profile = await self._learning_data_manager.get_user_profile(session_data.user_id)
                if not profile:
                    continue
                
                # Update session based on learned preferences
                adaptations = {}
                
                # Adapt session timeout based on user patterns
                if "session_duration_patterns" in profile.interaction_patterns:
                    durations = profile.interaction_patterns["session_duration_patterns"]
                    if isinstance(durations, list) and durations:
                        avg_duration = sum(durations) / len(durations)
                        # Extend timeout for users with longer sessions
                        if avg_duration > 1800:  # 30 minutes
                            adaptations["extended_timeout"] = True
                
                # Adapt persistence frequency based on user activity
                if profile.total_interactions > 100:
                    # More frequent persistence for active users
                    adaptations["frequent_persistence"] = True
                
                # Apply adaptations
                if adaptations:
                    session_data.adaptation_settings.update(adaptations)
                    self.logger.debug(f"Applied adaptations to session {session_id}: {list(adaptations.keys())}")
                
        except Exception as e:
            self.logger.error(f"Failed to update session adaptations: {e}")
    
    async def get_user_learning_insights(self, user_id: str) -> Dict[str, Any]:
        """Get learning insights for a specific user."""
        try:
            if not self._learning_data_manager:
                return {"error": "Learning system not enabled"}
            
            # Get user profile
            profile = await self._learning_data_manager.get_user_profile(user_id)
            if not profile:
                return {"error": "User profile not found"}
            
            # Analyze recent patterns
            patterns = await self._analyze_user_patterns(user_id)
            
            return {
                "user_profile": {
                    "total_interactions": profile.total_interactions,
                    "last_interaction": profile.last_interaction.isoformat() if profile.last_interaction else None,
                    "preferences": profile.preferences,
                    "interaction_patterns": profile.interaction_patterns
                },
                "recent_patterns": patterns,
                "learning_status": "active" if self._learning_enabled else "disabled"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get user learning insights: {e}")
            return {"error": str(e)}