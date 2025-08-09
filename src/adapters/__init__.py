"""Adapters Package - System Integration Adapters

This package contains all adapter implementations for the Claude Voice Assistant.
Adapters provide standardized interfaces to external systems and services.

Available Adapters:
- ClaudeCodeAdapter: Integration with Claude Code for development tasks
- SystemAdapter: Operating system integration (Windows/macOS/Linux)
- FileSystemAdapter: File system operations
- ApplicationAdapter: Application control and automation
- WebAdapter: Web browser and HTTP operations
"""

# Import core adapter infrastructure
from ..core.base_adapter import (
    BaseAdapter,
    AdapterError,
    CommandNotSupportedError,
    AdapterUnavailableError,
    CommandExecutionError
)

# Import adapter manager
from ..core.adapter_manager import AdapterManager

# Import concrete adapters
from .claude_code_adapter import (
    ClaudeCodeAdapter,
    MCPClient,
    ClaudeCodeError,
    MCPConnectionError,
    CodeGenerationError
)

__all__ = [
    # Base classes
    "BaseAdapter",
    "AdapterManager",
    
    # Concrete adapters
    "ClaudeCodeAdapter",
    "MCPClient",
    
    # Exceptions
    "AdapterError",
    "ClaudeCodeError", 
    "MCPConnectionError",
    "CodeGenerationError",
    "CommandNotSupportedError", 
    "AdapterUnavailableError",
    "CommandExecutionError"
]

# Version info
__version__ = "0.1.0"