"""EventSystem - Async Event-Driven Communication System

This module provides a decoupled event-driven communication system for
inter-component and inter-layer communication.

Features:
- Async event emission and handling
- Event filtering and routing
- Event history and replay
- Error handling and recovery
- Performance monitoring
"""

import asyncio
from typing import Dict, List, Callable, Optional, Any, Union, Set, Awaitable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid
import weakref
from loguru import logger
import inspect
from collections import defaultdict

from .types import LayerMessage


class EventPriority(Enum):
    """Event priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Event:
    """Event data structure"""
    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventHandler:
    """Event handler registration"""
    handler_id: str
    callback: Callable
    event_patterns: List[str]
    is_async: bool
    priority: int = 100
    max_retries: int = 3
    timeout: Optional[float] = None


class EventFilter:
    """Base class for event filters"""
    
    def should_process(self, event: Event) -> bool:
        """
        Determine if an event should be processed.
        
        Args:
            event: Event to check
            
        Returns:
            True if event should be processed
        """
        return True


class PatternEventFilter(EventFilter):
    """Event filter based on event type patterns"""
    
    def __init__(self, patterns: List[str]):
        """
        Initialize with event type patterns.
        
        Args:
            patterns: List of event type patterns (supports wildcards)
        """
        self.patterns = patterns
    
    def should_process(self, event: Event) -> bool:
        """Check if event matches any pattern."""
        return any(self._match_pattern(pattern, event.event_type) for pattern in self.patterns)
    
    def _match_pattern(self, pattern: str, event_type: str) -> bool:
        """Match event type against pattern (supports * wildcards)."""
        if '*' not in pattern:
            return pattern == event_type
        
        # Simple wildcard matching
        import re
        pattern_regex = pattern.replace('*', '.*')
        return re.match(f'^{pattern_regex}$', event_type) is not None


class EventSystem:
    """
    Async event-driven communication system.
    
    The EventSystem provides:
    - Event emission and subscription
    - Pattern-based event routing
    - Event filtering and transformation
    - Error handling and retry logic
    - Event history and replay
    - Performance monitoring
    """
    
    def __init__(self, 
                 max_history_size: int = 1000,
                 default_timeout: float = 30.0):
        """
        Initialize the EventSystem.
        
        Args:
            max_history_size: Maximum number of events to keep in history
            default_timeout: Default timeout for event handlers
        """
        # Event handlers storage
        self._handlers: Dict[str, EventHandler] = {}
        self._pattern_handlers: Dict[str, List[str]] = defaultdict(list)  # pattern -> handler_ids
        self._type_handlers: Dict[str, Set[str]] = defaultdict(set)  # event_type -> handler_ids
        
        # Event processing
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Event history
        self._event_history: List[Event] = []
        self._max_history_size = max_history_size
        
        # Configuration
        self._default_timeout = default_timeout
        
        # Statistics
        self._stats = {
            'events_emitted': 0,
            'events_processed': 0,
            'handlers_executed': 0,
            'errors_count': 0
        }
        
        # Weak references to prevent memory leaks
        self._weak_refs: Set[weakref.ReferenceType] = set()
    
    async def initialize(self) -> None:
        """Initialize the event system and start processing."""
        logger.info("Initializing EventSystem")
        
        # Start event processing task
        self._processing_task = asyncio.create_task(self._process_events())
        
        # Emit initialization event
        await self.emit(Event(
            event_type="event_system.initialized",
            source="event_system"
        ))
        
        logger.info("EventSystem initialized successfully")
    
    async def shutdown(self) -> None:
        """Shutdown the event system."""
        logger.info("Shutting down EventSystem")
        self._shutdown = True
        
        # Emit shutdown event
        await self.emit(Event(
            event_type="event_system.shutdown",
            source="event_system"
        ))
        
        # Stop processing task
        if self._processing_task and not self._processing_task.done():
            # Process remaining events with timeout
            try:
                await asyncio.wait_for(
                    self._process_remaining_events(),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for remaining events to process")
            
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        # Clear handlers and history
        self._handlers.clear()
        self._pattern_handlers.clear()
        self._type_handlers.clear()
        self._event_history.clear()
        
        logger.info("EventSystem shutdown complete")
    
    async def emit(self, event: Event) -> None:
        """
        Emit an event to all interested handlers.
        
        Args:
            event: Event to emit
        """
        if self._shutdown:
            return
        
        # Add to history
        self._add_to_history(event)
        
        # Add to processing queue
        await self._event_queue.put(event)
        
        self._stats['events_emitted'] += 1
    
    def subscribe(self, 
                  event_types: Union[str, List[str]],
                  handler: Callable,
                  handler_id: Optional[str] = None,
                  priority: int = 100,
                  max_retries: int = 3,
                  timeout: Optional[float] = None) -> str:
        """
        Subscribe to events with a handler.
        
        Args:
            event_types: Event type(s) to subscribe to (supports wildcards)
            handler: Callback function (sync or async)
            handler_id: Optional unique handler ID
            priority: Handler priority (lower numbers = higher priority)
            max_retries: Maximum retry attempts on failure
            timeout: Handler timeout in seconds
            
        Returns:
            Handler ID for future reference
        """
        if isinstance(event_types, str):
            event_types = [event_types]
        
        # Generate handler ID if not provided
        if handler_id is None:
            handler_id = str(uuid.uuid4())
        
        # Check if handler is async
        is_async = inspect.iscoroutinefunction(handler)
        
        # Create handler registration
        event_handler = EventHandler(
            handler_id=handler_id,
            callback=handler,
            event_patterns=event_types,
            is_async=is_async,
            priority=priority,
            max_retries=max_retries,
            timeout=timeout or self._default_timeout
        )
        
        # Store handler
        self._handlers[handler_id] = event_handler
        
        # Index by patterns
        for pattern in event_types:
            if '*' in pattern:
                self._pattern_handlers[pattern].append(handler_id)
            else:
                self._type_handlers[pattern].add(handler_id)
        
        logger.debug(f"Subscribed handler {handler_id} to events: {event_types}")
        
        return handler_id
    
    def unsubscribe(self, handler_id: str) -> bool:
        """
        Unsubscribe a handler from events.
        
        Args:
            handler_id: ID of the handler to remove
            
        Returns:
            True if handler was found and removed
        """
        if handler_id not in self._handlers:
            return False
        
        handler = self._handlers[handler_id]
        
        # Remove from indices
        for pattern in handler.event_patterns:
            if '*' in pattern:
                if pattern in self._pattern_handlers:
                    self._pattern_handlers[pattern] = [
                        hid for hid in self._pattern_handlers[pattern] if hid != handler_id
                    ]
                    if not self._pattern_handlers[pattern]:
                        del self._pattern_handlers[pattern]
            else:
                self._type_handlers[pattern].discard(handler_id)
                if not self._type_handlers[pattern]:
                    del self._type_handlers[pattern]
        
        # Remove handler
        del self._handlers[handler_id]
        
        logger.debug(f"Unsubscribed handler: {handler_id}")
        return True
    
    def list_handlers(self) -> List[str]:
        """
        Get list of all registered handler IDs.
        
        Returns:
            List of handler IDs
        """
        return list(self._handlers.keys())
    
    def get_handler_info(self, handler_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific handler.
        
        Args:
            handler_id: Handler ID
            
        Returns:
            Handler information dictionary
        """
        if handler_id not in self._handlers:
            return None
        
        handler = self._handlers[handler_id]
        return {
            'handler_id': handler.handler_id,
            'event_patterns': handler.event_patterns,
            'is_async': handler.is_async,
            'priority': handler.priority,
            'max_retries': handler.max_retries,
            'timeout': handler.timeout
        }
    
    def get_event_history(self, 
                         event_type_filter: Optional[str] = None,
                         limit: Optional[int] = None) -> List[Event]:
        """
        Get event history.
        
        Args:
            event_type_filter: Optional event type filter
            limit: Optional limit on number of events
            
        Returns:
            List of historical events
        """
        events = self._event_history
        
        if event_type_filter:
            events = [e for e in events if e.event_type == event_type_filter]
        
        if limit:
            events = events[-limit:]
        
        return events
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get event system statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            **self._stats,
            'handlers_count': len(self._handlers),
            'history_size': len(self._event_history),
            'queue_size': self._event_queue.qsize()
        }
    
    async def replay_events(self, 
                           event_filter: Optional[EventFilter] = None,
                           handler_id: Optional[str] = None) -> int:
        """
        Replay historical events.
        
        Args:
            event_filter: Optional filter for events to replay
            handler_id: Optional specific handler to replay to
            
        Returns:
            Number of events replayed
        """
        events_to_replay = []
        
        for event in self._event_history:
            if event_filter is None or event_filter.should_process(event):
                events_to_replay.append(event)
        
        replayed_count = 0
        for event in events_to_replay:
            if handler_id:
                # Replay to specific handler
                await self._execute_handler(handler_id, event)
            else:
                # Replay to all matching handlers
                await self.emit(event)
            replayed_count += 1
        
        logger.info(f"Replayed {replayed_count} events")
        return replayed_count
    
    async def _process_events(self) -> None:
        """Main event processing loop."""
        logger.info("Started event processing loop")
        
        while not self._shutdown:
            try:
                # Get next event with timeout
                try:
                    event = await asyncio.wait_for(
                        self._event_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process the event
                await self._process_single_event(event)
                self._stats['events_processed'] += 1
                
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                self._stats['errors_count'] += 1
    
    async def _process_single_event(self, event: Event) -> None:
        """
        Process a single event.
        
        Args:
            event: Event to process
        """
        # Find matching handlers
        handler_ids = self._find_matching_handlers(event)
        
        if not handler_ids:
            return
        
        # Sort handlers by priority
        sorted_handlers = sorted(
            [(self._handlers[hid], hid) for hid in handler_ids],
            key=lambda x: x[0].priority
        )
        
        # Execute handlers
        tasks = []
        for handler, handler_id in sorted_handlers:
            task = asyncio.create_task(
                self._execute_handler_with_retry(handler_id, event)
            )
            tasks.append(task)
        
        # Wait for all handlers to complete
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log any exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    handler_id = sorted_handlers[i][1]
                    logger.error(f"Handler {handler_id} failed: {result}")
    
    def _find_matching_handlers(self, event: Event) -> Set[str]:
        """
        Find all handlers that match the event.
        
        Args:
            event: Event to match
            
        Returns:
            Set of matching handler IDs
        """
        matching_handlers = set()
        
        # Exact type matches
        matching_handlers.update(self._type_handlers.get(event.event_type, set()))
        
        # Pattern matches
        for pattern, handler_ids in self._pattern_handlers.items():
            if self._match_event_pattern(pattern, event.event_type):
                matching_handlers.update(handler_ids)
        
        return matching_handlers
    
    def _match_event_pattern(self, pattern: str, event_type: str) -> bool:
        """
        Match event type against pattern.
        
        Args:
            pattern: Pattern with potential wildcards
            event_type: Event type to match
            
        Returns:
            True if pattern matches
        """
        if '*' not in pattern:
            return pattern == event_type
        
        import re
        pattern_regex = pattern.replace('*', '.*')
        return re.match(f'^{pattern_regex}$', event_type) is not None
    
    async def _execute_handler_with_retry(self, handler_id: str, event: Event) -> None:
        """
        Execute a handler with retry logic.
        
        Args:
            handler_id: Handler to execute
            event: Event to process
        """
        if handler_id not in self._handlers:
            return
        
        handler = self._handlers[handler_id]
        
        for attempt in range(handler.max_retries + 1):
            try:
                await self._execute_handler(handler_id, event)
                self._stats['handlers_executed'] += 1
                return
                
            except Exception as e:
                if attempt == handler.max_retries:
                    logger.error(f"Handler {handler_id} failed after {handler.max_retries} retries: {e}")
                    self._stats['errors_count'] += 1
                    raise
                else:
                    logger.warning(f"Handler {handler_id} attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff
    
    async def _execute_handler(self, handler_id: str, event: Event) -> None:
        """
        Execute a single handler.
        
        Args:
            handler_id: Handler to execute
            event: Event to process
        """
        if handler_id not in self._handlers:
            return
        
        handler = self._handlers[handler_id]
        
        try:
            if handler.is_async:
                await asyncio.wait_for(
                    handler.callback(event),
                    timeout=handler.timeout
                )
            else:
                # Run sync handler in executor to avoid blocking
                await asyncio.get_event_loop().run_in_executor(
                    None, handler.callback, event
                )
                
        except asyncio.TimeoutError:
            raise Exception(f"Handler {handler_id} timed out after {handler.timeout}s")
        except Exception as e:
            raise Exception(f"Handler {handler_id} execution failed: {e}")
    
    def _add_to_history(self, event: Event) -> None:
        """
        Add event to history.
        
        Args:
            event: Event to add
        """
        self._event_history.append(event)
        
        # Maintain history size limit
        if len(self._event_history) > self._max_history_size:
            self._event_history = self._event_history[-self._max_history_size:]
    
    async def _process_remaining_events(self) -> None:
        """Process any remaining events in the queue."""
        while not self._event_queue.empty():
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=0.1
                )
                await self._process_single_event(event)
            except asyncio.TimeoutError:
                break
            except Exception as e:
                logger.error(f"Error processing remaining event: {e}")


class EventSystemError(Exception):
    """Base exception for EventSystem errors."""
    pass


class HandlerExecutionError(EventSystemError):
    """Raised when event handler execution fails."""
    pass


class EventTimeoutError(EventSystemError):
    """Raised when event processing times out."""
    pass