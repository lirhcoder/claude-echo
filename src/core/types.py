"""Core Type Definitions

This module defines the fundamental data structures used throughout
the Claude Voice Assistant architecture.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import uuid


class RiskLevel(Enum):
    """Security risk levels for operations"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AdapterStatus(Enum):
    """Adapter availability status"""
    AVAILABLE = "available"
    BUSY = "busy"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class CommandResult:
    """Result of adapter command execution"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Intent(BaseModel):
    """Parsed user intent"""
    intent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_input: str
    intent_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True


class Task(BaseModel):
    """Individual task within an execution plan"""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    command: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    target_adapter: str
    status: TaskStatus = TaskStatus.PENDING
    risk_level: RiskLevel = RiskLevel.LOW
    dependencies: List[str] = Field(default_factory=list)
    timeout: Optional[timedelta] = None
    retry_count: int = 0
    max_retries: int = 3
    
    class Config:
        arbitrary_types_allowed = True


class ExecutionPlan(BaseModel):
    """Complete execution plan for user intent"""
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    intent: Intent
    tasks: List[Task]
    estimated_duration: Optional[timedelta] = None
    risk_level: RiskLevel = RiskLevel.LOW
    requires_user_confirmation: bool = False
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True


class Context(BaseModel):
    """System and user context information"""
    user_id: str
    session_id: str
    current_app: Optional[str] = None
    current_file: Optional[str] = None
    screen_content: Optional[str] = None
    recent_commands: List[str] = Field(default_factory=list)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    environment_vars: Dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True


@dataclass
class ExecutionResult:
    """Result of execution plan completion"""
    plan_id: str
    results: List[CommandResult]
    overall_success: bool
    total_execution_time: float
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class AdapterInfo(BaseModel):
    """Adapter registration information"""
    adapter_id: str
    name: str
    version: str
    description: str
    supported_commands: List[str]
    status: AdapterStatus = AdapterStatus.AVAILABLE
    priority: int = 100
    config: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class LayerMessage(BaseModel):
    """Inter-layer communication message"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_layer: str
    target_layer: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True