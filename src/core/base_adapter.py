"""BaseAdapter - Abstract Base Class for All Adapters

This module defines the standard interface that all adapters must implement
to integrate with the Claude Voice Assistant system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncContextManager
import asyncio
from contextlib import asynccontextmanager

from .types import CommandResult, AdapterInfo, AdapterStatus, Context


class BaseAdapter(ABC):
    """
    Abstract base class for all adapters in the system.
    
    Adapters are the bridge between the intelligence layer and actual
    system operations. Each adapter handles a specific domain (e.g., file system,
    applications, web services).
    
    All adapters must implement the abstract methods defined here.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adapter with configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self._config = config or {}
        self._status = AdapterStatus.AVAILABLE
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._initialization_complete = False
    
    # Required Properties
    
    @property
    @abstractmethod
    def adapter_id(self) -> str:
        """Unique identifier for this adapter."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the adapter."""
        pass
    
    @property 
    @abstractmethod
    def version(self) -> str:
        """Version of the adapter."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this adapter does."""
        pass
    
    @property
    @abstractmethod
    def supported_commands(self) -> List[str]:
        """List of commands this adapter can execute."""
        pass
    
    # Required Methods
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the adapter and its dependencies.
        
        This method is called once when the adapter is loaded.
        It should perform any necessary setup operations.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """
        Clean up resources before adapter shutdown.
        
        This method is called when the adapter is being unloaded
        or the system is shutting down.
        """
        pass
    
    @abstractmethod
    async def execute_command(self, 
                            command: str, 
                            parameters: Dict[str, Any],
                            context: Optional[Context] = None) -> CommandResult:
        """
        Execute a command with the given parameters.
        
        Args:
            command: The command to execute
            parameters: Command parameters 
            context: Current system context
            
        Returns:
            CommandResult containing execution results
        """
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """
        Check if the adapter is currently available for use.
        
        Returns:
            True if adapter can handle requests, False otherwise
        """
        pass
    
    @abstractmethod 
    async def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current state of the adapter and its managed resources.
        
        Returns:
            Dictionary containing current state information
        """
        pass
    
    @abstractmethod
    async def get_command_suggestions(self, 
                                    context: Optional[Context] = None) -> List[str]:
        """
        Get command suggestions based on current context.
        
        Args:
            context: Current system context
            
        Returns:
            List of suggested commands
        """
        pass
    
    # Optional Override Methods
    
    async def validate_command(self, 
                             command: str, 
                             parameters: Dict[str, Any]) -> bool:
        """
        Validate a command and its parameters before execution.
        
        Args:
            command: Command to validate
            parameters: Command parameters
            
        Returns:
            True if command is valid, False otherwise
        """
        return command in self.supported_commands
    
    async def estimate_execution_time(self, 
                                    command: str,
                                    parameters: Dict[str, Any]) -> Optional[float]:
        """
        Estimate execution time for a command in seconds.
        
        Args:
            command: Command to estimate
            parameters: Command parameters
            
        Returns:
            Estimated execution time in seconds, or None if unknown
        """
        return None
    
    async def can_execute_concurrently(self, 
                                     command: str,
                                     parameters: Dict[str, Any]) -> bool:
        """
        Check if command can be executed concurrently with other commands.
        
        Args:
            command: Command to check
            parameters: Command parameters
            
        Returns:
            True if command supports concurrent execution
        """
        return True
    
    # Public Interface Methods
    
    @property
    def status(self) -> AdapterStatus:
        """Get current adapter status."""
        return self._status
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get adapter configuration."""
        return self._config.copy()
    
    def get_info(self) -> AdapterInfo:
        """
        Get complete adapter information.
        
        Returns:
            AdapterInfo object with all adapter metadata
        """
        return AdapterInfo(
            adapter_id=self.adapter_id,
            name=self.name,
            version=self.version,
            description=self.description,
            supported_commands=self.supported_commands,
            status=self._status,
            config=self._config
        )
    
    @asynccontextmanager
    async def execute_with_timeout(self, 
                                 timeout: float) -> AsyncContextManager[None]:
        """
        Context manager for executing operations with timeout.
        
        Args:
            timeout: Timeout in seconds
        """
        try:
            yield
        except asyncio.TimeoutError:
            self._status = AdapterStatus.ERROR
            raise
    
    async def health_check(self) -> bool:
        """
        Perform a health check on the adapter.
        
        Returns:
            True if adapter is healthy, False otherwise
        """
        try:
            return await self.is_available()
        except Exception:
            return False
    
    # Protected Methods
    
    def _update_status(self, status: AdapterStatus) -> None:
        """Update adapter status."""
        self._status = status
    
    def _add_active_task(self, task_id: str, task: asyncio.Task) -> None:
        """Add a task to active tasks tracking."""
        self._active_tasks[task_id] = task
    
    def _remove_active_task(self, task_id: str) -> None:
        """Remove a task from active tasks tracking."""
        if task_id in self._active_tasks:
            del self._active_tasks[task_id]
    
    async def _cancel_all_tasks(self) -> None:
        """Cancel all active tasks."""
        for task in self._active_tasks.values():
            if not task.done():
                task.cancel()
        
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks.values(), return_exceptions=True)
        
        self._active_tasks.clear()


class AdapterError(Exception):
    """Base exception for adapter-related errors."""
    
    def __init__(self, message: str, adapter_id: str = None):
        super().__init__(message)
        self.adapter_id = adapter_id


class CommandNotSupportedError(AdapterError):
    """Raised when an adapter doesn't support a command."""
    pass


class AdapterUnavailableError(AdapterError):
    """Raised when an adapter is not available for execution."""
    pass


class CommandExecutionError(AdapterError):
    """Raised when command execution fails."""
    
    def __init__(self, message: str, command: str, adapter_id: str = None):
        super().__init__(message, adapter_id)
        self.command = command