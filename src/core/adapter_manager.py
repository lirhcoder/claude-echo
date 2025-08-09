"""AdapterManager - Dynamic Adapter Loading and Management

This module provides the AdapterManager class which handles:
- Dynamic loading and unloading of adapters
- Adapter lifecycle management  
- Command routing to appropriate adapters
- Adapter health monitoring and recovery
"""

import asyncio
import importlib
import importlib.util
import inspect
from typing import Dict, List, Optional, Type, Any, Set
from pathlib import Path
from loguru import logger
import weakref

from .base_adapter import BaseAdapter, AdapterError, AdapterUnavailableError, CommandNotSupportedError
from .types import CommandResult, AdapterInfo, AdapterStatus, Context
from .event_system import EventSystem, Event


class AdapterManager:
    """
    Manages the lifecycle and execution of adapters in the system.
    
    The AdapterManager is responsible for:
    - Loading adapters from configured paths
    - Managing adapter lifecycles (initialize, cleanup)
    - Routing commands to appropriate adapters
    - Health monitoring and error recovery
    - Hot-reloading of adapters during development
    """
    
    def __init__(self, 
                 event_system: Optional[EventSystem] = None,
                 adapter_paths: Optional[List[str]] = None):
        """
        Initialize the AdapterManager.
        
        Args:
            event_system: Event system for inter-component communication
            adapter_paths: List of paths to search for adapters
        """
        self._adapters: Dict[str, BaseAdapter] = {}
        self._adapter_classes: Dict[str, Type[BaseAdapter]] = {}
        self._command_map: Dict[str, Set[str]] = {}  # command -> adapter_ids
        self._adapter_paths = adapter_paths or []
        self._event_system = event_system
        self._lock = asyncio.Lock()
        self._health_check_interval = 30.0  # seconds
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Weak references to prevent circular references
        self._adapter_refs: Dict[str, weakref.ReferenceType] = {}
    
    async def initialize(self) -> None:
        """Initialize the adapter manager and start health monitoring."""
        logger.info("Initializing AdapterManager")
        
        # Load all available adapters
        await self.discover_adapters()
        
        # Start health monitoring
        await self.start_health_monitoring()
        
        if self._event_system:
            await self._event_system.emit(Event(
                event_type="adapter_manager.initialized",
                data={"adapter_count": len(self._adapters)}
            ))
        
        logger.info(f"AdapterManager initialized with {len(self._adapters)} adapters")
    
    async def shutdown(self) -> None:
        """Shutdown the adapter manager and cleanup all resources."""
        logger.info("Shutting down AdapterManager")
        self._shutdown = True
        
        # Stop health monitoring
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup all adapters
        async with self._lock:
            cleanup_tasks = []
            for adapter_id, adapter in self._adapters.items():
                logger.debug(f"Cleaning up adapter: {adapter_id}")
                cleanup_tasks.append(self._cleanup_adapter(adapter))
            
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            self._adapters.clear()
            self._command_map.clear()
        
        if self._event_system:
            await self._event_system.emit(Event(
                event_type="adapter_manager.shutdown",
                data={}
            ))
        
        logger.info("AdapterManager shutdown complete")
    
    async def discover_adapters(self) -> int:
        """
        Discover and load all available adapters.
        
        Returns:
            Number of adapters loaded
        """
        logger.info("Discovering adapters...")
        loaded_count = 0
        
        for adapter_path in self._adapter_paths:
            try:
                count = await self._scan_adapter_directory(adapter_path)
                loaded_count += count
                logger.debug(f"Loaded {count} adapters from {adapter_path}")
            except Exception as e:
                logger.error(f"Error scanning adapter path {adapter_path}: {e}")
        
        return loaded_count
    
    async def register_adapter(self, 
                             adapter_class: Type[BaseAdapter], 
                             config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register a new adapter class and create an instance.
        
        Args:
            adapter_class: The adapter class to register
            config: Optional configuration for the adapter
            
        Returns:
            True if registration was successful
        """
        try:
            # Create adapter instance
            adapter_instance = adapter_class(config)
            adapter_id = adapter_instance.adapter_id
            
            # Check for conflicts
            if adapter_id in self._adapters:
                logger.warning(f"Adapter {adapter_id} already registered, skipping")
                return False
            
            async with self._lock:
                # Initialize the adapter
                if await adapter_instance.initialize():
                    self._adapters[adapter_id] = adapter_instance
                    self._adapter_classes[adapter_id] = adapter_class
                    
                    # Update command mapping
                    for command in adapter_instance.supported_commands:
                        if command not in self._command_map:
                            self._command_map[command] = set()
                        self._command_map[command].add(adapter_id)
                    
                    # Create weak reference
                    self._adapter_refs[adapter_id] = weakref.ref(adapter_instance)
                    
                    logger.info(f"Registered adapter: {adapter_id}")
                    
                    if self._event_system:
                        await self._event_system.emit(Event(
                            event_type="adapter.registered",
                            data={"adapter_id": adapter_id, "commands": adapter_instance.supported_commands}
                        ))
                    
                    return True
                else:
                    logger.error(f"Failed to initialize adapter: {adapter_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error registering adapter {adapter_class.__name__}: {e}")
            return False
    
    async def unregister_adapter(self, adapter_id: str) -> bool:
        """
        Unregister and cleanup an adapter.
        
        Args:
            adapter_id: ID of the adapter to unregister
            
        Returns:
            True if unregistration was successful
        """
        async with self._lock:
            if adapter_id not in self._adapters:
                logger.warning(f"Adapter {adapter_id} not found for unregistration")
                return False
            
            adapter = self._adapters[adapter_id]
            
            try:
                # Cleanup the adapter
                await self._cleanup_adapter(adapter)
                
                # Remove from mappings
                del self._adapters[adapter_id]
                if adapter_id in self._adapter_classes:
                    del self._adapter_classes[adapter_id]
                if adapter_id in self._adapter_refs:
                    del self._adapter_refs[adapter_id]
                
                # Update command mapping
                for command, adapter_set in self._command_map.items():
                    adapter_set.discard(adapter_id)
                
                # Remove empty command entries
                empty_commands = [cmd for cmd, adapters in self._command_map.items() if not adapters]
                for cmd in empty_commands:
                    del self._command_map[cmd]
                
                logger.info(f"Unregistered adapter: {adapter_id}")
                
                if self._event_system:
                    await self._event_system.emit(Event(
                        event_type="adapter.unregistered",
                        data={"adapter_id": adapter_id}
                    ))
                
                return True
                
            except Exception as e:
                logger.error(f"Error unregistering adapter {adapter_id}: {e}")
                return False
    
    async def execute_command(self,
                            command: str,
                            parameters: Dict[str, Any],
                            context: Optional[Context] = None,
                            preferred_adapter: Optional[str] = None) -> CommandResult:
        """
        Execute a command using the most appropriate adapter.
        
        Args:
            command: Command to execute
            parameters: Command parameters
            context: Current system context
            preferred_adapter: Preferred adapter ID (optional)
            
        Returns:
            CommandResult from the adapter execution
            
        Raises:
            CommandNotSupportedError: If no adapter supports the command
            AdapterUnavailableError: If no suitable adapter is available
        """
        # Check if command is supported
        if command not in self._command_map:
            raise CommandNotSupportedError(f"Command '{command}' not supported by any adapter")
        
        # Get candidate adapters
        candidate_adapters = self._command_map[command]
        
        # Try preferred adapter first if specified
        if preferred_adapter and preferred_adapter in candidate_adapters:
            adapter = self._adapters.get(preferred_adapter)
            if adapter and await adapter.is_available():
                try:
                    return await adapter.execute_command(command, parameters, context)
                except Exception as e:
                    logger.error(f"Error executing command with preferred adapter {preferred_adapter}: {e}")
                    # Fall through to try other adapters
        
        # Try other available adapters
        for adapter_id in candidate_adapters:
            adapter = self._adapters.get(adapter_id)
            if adapter and await adapter.is_available():
                try:
                    return await adapter.execute_command(command, parameters, context)
                except Exception as e:
                    logger.error(f"Error executing command with adapter {adapter_id}: {e}")
                    continue
        
        # No adapter could execute the command
        raise AdapterUnavailableError(f"No available adapter can execute command '{command}'")
    
    def get_adapter(self, adapter_id: str) -> Optional[BaseAdapter]:
        """
        Get an adapter by its ID.
        
        Args:
            adapter_id: ID of the adapter to retrieve
            
        Returns:
            The adapter instance, or None if not found
        """
        return self._adapters.get(adapter_id)
    
    def list_adapters(self) -> List[AdapterInfo]:
        """
        Get information about all registered adapters.
        
        Returns:
            List of AdapterInfo objects
        """
        return [adapter.get_info() for adapter in self._adapters.values()]
    
    def get_supported_commands(self) -> Dict[str, List[str]]:
        """
        Get all supported commands and their supporting adapters.
        
        Returns:
            Dictionary mapping commands to list of adapter IDs
        """
        return {cmd: list(adapters) for cmd, adapters in self._command_map.items()}
    
    def get_command_suggestions(self, context: Optional[Context] = None) -> List[str]:
        """
        Get command suggestions from all available adapters.
        
        Args:
            context: Current system context
            
        Returns:
            List of suggested commands
        """
        # This would typically be implemented with async gather,
        # but simplified for synchronous interface
        suggestions = set()
        for adapter in self._adapters.values():
            suggestions.update(adapter.supported_commands)
        return list(suggestions)
    
    async def start_health_monitoring(self) -> None:
        """Start periodic health monitoring of adapters."""
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_monitor_loop())
    
    async def _health_monitor_loop(self) -> None:
        """Health monitoring loop."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self._health_check_interval)
                
                if self._shutdown:
                    break
                
                # Check health of all adapters
                unhealthy_adapters = []
                for adapter_id, adapter in self._adapters.items():
                    try:
                        if not await adapter.health_check():
                            unhealthy_adapters.append(adapter_id)
                            logger.warning(f"Adapter {adapter_id} failed health check")
                    except Exception as e:
                        logger.error(f"Error during health check for adapter {adapter_id}: {e}")
                        unhealthy_adapters.append(adapter_id)
                
                # Emit health events
                if self._event_system and unhealthy_adapters:
                    await self._event_system.emit(Event(
                        event_type="adapter.health_check_failed",
                        data={"unhealthy_adapters": unhealthy_adapters}
                    ))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
    
    async def _scan_adapter_directory(self, directory_path: str) -> int:
        """
        Scan a directory for adapter modules and load them.
        
        Args:
            directory_path: Path to scan for adapters
            
        Returns:
            Number of adapters loaded
        """
        loaded_count = 0
        adapter_dir = Path(directory_path)
        
        if not adapter_dir.exists():
            logger.warning(f"Adapter directory does not exist: {directory_path}")
            return 0
        
        for py_file in adapter_dir.glob("**/*.py"):
            if py_file.name.startswith("_"):
                continue
                
            try:
                # Import the module
                module_name = f"adapters.{py_file.stem}"
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find adapter classes
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseAdapter) and 
                        obj != BaseAdapter and
                        not inspect.isabstract(obj)):
                        
                        if await self.register_adapter(obj):
                            loaded_count += 1
                            
            except Exception as e:
                logger.error(f"Error loading adapter from {py_file}: {e}")
        
        return loaded_count
    
    async def _cleanup_adapter(self, adapter: BaseAdapter) -> None:
        """Cleanup a single adapter."""
        try:
            await adapter.cleanup()
        except Exception as e:
            logger.error(f"Error during adapter cleanup: {e}")


class AdapterManagerError(Exception):
    """Base exception for AdapterManager errors."""
    pass


class AdapterDiscoveryError(AdapterManagerError):
    """Raised when adapter discovery fails."""
    pass


class AdapterRegistrationError(AdapterManagerError):
    """Raised when adapter registration fails."""
    pass