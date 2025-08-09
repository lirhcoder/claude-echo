"""Agent Types and Enumerations

This module defines the core types, enums, and data structures used by
all agents in the Claude Voice Assistant system.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Union, Callable, Awaitable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import uuid
import asyncio

from ..core.types import RiskLevel, TaskStatus


class AgentType(Enum):
    """Types of agents in the system"""
    COORDINATOR = "coordinator"
    TASK_PLANNER = "task_planner"
    PRESENCE_MONITOR = "presence_monitor"
    AUTO_WORKER = "auto_worker"
    SECURITY_GUARDIAN = "security_guardian"
    HANDOVER_MANAGER = "handover_manager"
    SESSION_MANAGER = "session_manager"
    
    # Learning system agents
    LEARNING_AGENT = "learning_agent"
    USER_PROFILE_AGENT = "user_profile_agent"
    CORRECTION_AGENT = "correction_agent"


class AgentStatus(Enum):
    """Agent operational status"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    RECOVERING = "recovering"
    SHUTDOWN = "shutdown"


class MessageType(Enum):
    """Types of inter-agent messages"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    COMMAND = "command"
    EVENT = "event"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"


class AgentPriority(Enum):
    """Agent priority levels for task processing"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentCapability:
    """Agent capability definition"""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    risk_level: RiskLevel = RiskLevel.LOW
    execution_time_ms: Optional[int] = None
    requires_user_confirmation: bool = False


class AgentMessage(BaseModel):
    """Inter-agent communication message"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_agent: str
    target_agent: Optional[str] = None  # None for broadcasts
    message_type: MessageType
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    priority: AgentPriority = AgentPriority.NORMAL
    timeout: Optional[timedelta] = None
    
    class Config:
        arbitrary_types_allowed = True


class AgentRequest(BaseModel):
    """Agent request with context and parameters"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    requesting_agent: str
    target_capability: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    priority: AgentPriority = AgentPriority.NORMAL
    timeout: timedelta = Field(default_factory=lambda: timedelta(seconds=30))
    correlation_id: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


class AgentResponse(BaseModel):
    """Agent response to a request"""
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str
    responding_agent: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True


@dataclass
class AgentState:
    """Current state of an agent"""
    agent_id: str
    agent_type: AgentType
    status: AgentStatus
    current_task: Optional[str] = None
    last_activity: datetime = field(default_factory=datetime.now)
    error_count: int = 0
    processed_requests: int = 0
    active_connections: List[str] = field(default_factory=list)
    state_data: Dict[str, Any] = field(default_factory=dict)


class AgentEvent(BaseModel):
    """Agent lifecycle and operational events"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    source_agent: str
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    severity: str = "info"  # info, warning, error, critical
    correlation_id: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


@dataclass
class AgentMetrics:
    """Performance metrics for an agent"""
    agent_id: str
    requests_processed: int = 0
    requests_failed: int = 0
    average_response_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    uptime_seconds: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class CollaborationPattern(Enum):
    """Patterns for agent collaboration"""
    SEQUENTIAL = "sequential"  # One after another
    PARALLEL = "parallel"     # Concurrent execution
    PIPELINE = "pipeline"     # Data flows through agents
    BROADCAST = "broadcast"   # One to many
    CONSENSUS = "consensus"   # Agreement required
    DELEGATION = "delegation" # Hand off responsibility


@dataclass
class CollaborationPlan:
    """Plan for multi-agent collaboration"""
    initiator: str
    participants: List[str]
    pattern: CollaborationPattern
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    steps: List[Dict[str, Any]] = field(default_factory=list)
    timeout: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    created_at: datetime = field(default_factory=datetime.now)


# Type aliases for callbacks and handlers
AgentCallback = Callable[[AgentMessage], Awaitable[Optional[AgentResponse]]]
EventHandler = Callable[[AgentEvent], Awaitable[None]]
StateChangeHandler = Callable[[AgentState, AgentState], Awaitable[None]]


class AgentError(Exception):
    """Base exception for agent-related errors"""
    
    def __init__(self, message: str, agent_id: str = None, error_code: str = None):
        super().__init__(message)
        self.agent_id = agent_id
        self.error_code = error_code
        self.timestamp = datetime.now()


class AgentCommunicationError(AgentError):
    """Raised when agent communication fails"""
    pass


class AgentTimeoutError(AgentError):
    """Raised when agent operation times out"""
    pass


class AgentBusyError(AgentError):
    """Raised when agent is too busy to handle request"""
    pass


class AgentInitializationError(AgentError):
    """Raised when agent initialization fails"""
    pass


class AgentShutdownError(AgentError):
    """Raised during agent shutdown issues"""
    pass