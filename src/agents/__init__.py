"""Agents Package - Intelligence Hub Agents

This package contains the 7 core AI agents that form the Intelligence Hub layer:

Core Agents:
1. Coordinator - Central orchestration and task distribution
2. TaskPlanner - Intelligent task decomposition and planning  
3. PresenceMonitor - User presence and context detection
4. AutoWorker - Autonomous task execution
5. SecurityGuardian - Security validation and risk management
6. HandoverManager - Human-AI transition handling
7. SessionManager - Session lifecycle and state management

Each agent operates asynchronously and communicates through the event system.
"""

# Import agent types and base classes
from .agent_types import (
    AgentType, AgentStatus, AgentMessage, AgentRequest, AgentResponse,
    AgentEvent, AgentCapability, AgentState, AgentMetrics,
    CollaborationPlan, CollaborationPattern, MessageType, AgentPriority
)

from .base_agent import BaseAgent

# Import core agents
from .coordinator import Coordinator
from .task_planner import TaskPlanner
from .presence_monitor import PresenceMonitor
from .auto_worker import AutoWorker
from .security_guardian import SecurityGuardian
from .handover_manager import HandoverManager
from .session_manager import SessionManager

# Import agent manager
from .agent_manager import AgentManager

__all__ = [
    # Type definitions
    "AgentType",
    "AgentStatus", 
    "AgentMessage",
    "AgentRequest",
    "AgentResponse",
    "AgentEvent",
    "AgentCapability",
    "AgentState",
    "AgentMetrics",
    "CollaborationPlan",
    "CollaborationPattern",
    "MessageType",
    "AgentPriority",
    
    # Base classes
    "BaseAgent",
    
    # Core agents
    "Coordinator",
    "TaskPlanner", 
    "PresenceMonitor",
    "AutoWorker",
    "SecurityGuardian",
    "HandoverManager",
    "SessionManager",
    
    # Management
    "AgentManager"
]

# Version info
__version__ = "1.0.0"