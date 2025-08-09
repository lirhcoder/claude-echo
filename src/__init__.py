"""Claude Voice Assistant - Main Package

A layered architecture voice assistant system with AI agent collaboration.

This package implements:
- 4-layer architecture (UI -> Intelligence Hub -> Adapters -> Execution)
- 7 core agents for intelligent task processing
- Pluggable adapter system for extensibility
- Async/await patterns for high performance
"""

__version__ = "0.1.0"
__author__ = "Claude Voice Assistant Team"
__email__ = "noreply@anthropic.com"

# Package metadata
__all__ = [
    "core",
    "agents", 
    "adapters",
    "speech",
    "utils"
]

# Version info
VERSION = (0, 1, 0, "alpha")