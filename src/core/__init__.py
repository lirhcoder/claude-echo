"""Core Framework Components

This module contains the foundational classes and interfaces for the 
Claude Voice Assistant architecture:

- BaseAdapter: Abstract base class for all adapters
- AdapterManager: Dynamic adapter loading and lifecycle management
- ConfigManager: YAML-based configuration system
- EventSystem: Async event-driven communication
- Layer interfaces: Standardized layer contracts
"""

from .base_adapter import BaseAdapter
from .adapter_manager import AdapterManager
from .config_manager import ConfigManager
from .event_system import EventSystem, Event
from .types import (
    CommandResult,
    Intent,
    ExecutionPlan, 
    Task,
    Context,
    ExecutionResult,
    RiskLevel
)
from .layers import (
    UserInterfaceLayer,
    IntelligenceHubLayer,
    AdapterLayer,
    ExecutionLayer
)

__all__ = [
    # Base classes
    "BaseAdapter",
    "AdapterManager", 
    "ConfigManager",
    "EventSystem",
    "Event",
    
    # Type definitions
    "CommandResult",
    "Intent",
    "ExecutionPlan",
    "Task", 
    "Context",
    "ExecutionResult",
    "RiskLevel",
    
    # Layer interfaces
    "UserInterfaceLayer",
    "IntelligenceHubLayer",
    "AdapterLayer",
    "ExecutionLayer"
]