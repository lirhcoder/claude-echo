"""BaseAgent - Foundation class for all intelligent agents

This module provides the base functionality that all agents inherit,
including communication, state management, lifecycle, and error handling.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime, timedelta
import weakref
import json
from contextlib import asynccontextmanager

from loguru import logger

from ..core.event_system import EventSystem, Event, EventPriority
from ..core.types import Context
from .agent_types import (
    AgentType, AgentStatus, AgentMessage, AgentRequest, AgentResponse,
    AgentState, AgentEvent, AgentMetrics, AgentCapability, AgentCallback,
    EventHandler, StateChangeHandler, MessageType, AgentPriority,
    AgentError, AgentCommunicationError, AgentTimeoutError, AgentBusyError
)


class BaseAgent(ABC):
    """
    Base class for all intelligent agents in the system.
    
    This class provides:
    - Event-driven communication via EventSystem
    - Asynchronous state machine management
    - Lifecycle management (init, process, cleanup)
    - Error handling and recovery mechanisms
    - Performance monitoring and metrics
    - Inter-agent collaboration protocols
    """
    
    def __init__(self, 
                 agent_id: str,
                 event_system: EventSystem,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for this agent
            event_system: Event system for communication
            config: Optional configuration dictionary
        """
        self.agent_id = agent_id
        self.event_system = event_system
        self.config = config or {}
        
        # State management
        self._state = AgentState(
            agent_id=agent_id,
            agent_type=self.agent_type,
            status=AgentStatus.IDLE
        )
        self._previous_state: Optional[AgentState] = None
        self._state_lock = asyncio.Lock()
        
        # Communication
        self._message_handlers: Dict[str, AgentCallback] = {}
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._event_subscriptions: List[str] = []
        
        # Lifecycle management
        self._initialization_complete = False
        self._shutdown_requested = False
        self._background_tasks: Set[asyncio.Task] = set()
        
        # Error handling
        self._error_count = 0
        self._max_errors = self.config.get('max_errors', 10)
        self._recovery_in_progress = False
        
        # Performance monitoring
        self._metrics = AgentMetrics(agent_id=agent_id)
        self._start_time = datetime.now()
        
        # State change callbacks
        self._state_change_handlers: List[StateChangeHandler] = []
        
        # Setup logging
        self.logger = logger.bind(agent=agent_id)
    
    # Abstract properties that subclasses must implement
    
    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """Type of this agent."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this agent."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this agent does."""
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> List[AgentCapability]:
        """List of capabilities this agent provides."""
        pass
    
    # Abstract methods that subclasses must implement
    
    @abstractmethod
    async def _initialize_agent(self) -> None:
        """
        Initialize agent-specific functionality.
        Called during the initialization process.
        """
        pass
    
    @abstractmethod
    async def _process_request(self, request: AgentRequest) -> AgentResponse:
        """
        Process an incoming request.
        
        Args:
            request: The request to process
            
        Returns:
            Response to the request
        """
        pass
    
    @abstractmethod
    async def _handle_event(self, event: AgentEvent) -> None:
        """
        Handle an agent event.
        
        Args:
            event: The event to handle
        """
        pass
    
    @abstractmethod
    async def _cleanup_agent(self) -> None:
        """
        Cleanup agent-specific resources.
        Called during shutdown.
        """
        pass
    
    # Public API methods
    
    async def initialize(self) -> None:
        """Initialize the agent and start operation."""
        if self._initialization_complete:
            return
            
        try:
            await self._set_status(AgentStatus.INITIALIZING)
            self.logger.info(f"Initializing agent {self.name}")
            
            # Subscribe to agent events
            await self._setup_event_subscriptions()
            
            # Initialize agent-specific functionality
            await self._initialize_agent()
            
            # Start background tasks
            await self._start_background_tasks()
            
            await self._set_status(AgentStatus.IDLE)
            self._initialization_complete = True
            
            # Emit initialization complete event
            await self._emit_event("agent.initialized", {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type.value,
                "capabilities": [cap.name for cap in self.capabilities]
            })
            
            self.logger.info(f"Agent {self.name} initialized successfully")
            
        except Exception as e:
            await self._set_status(AgentStatus.ERROR)
            self.logger.error(f"Agent initialization failed: {e}")
            raise AgentError(f"Initialization failed: {e}", self.agent_id)
    
    async def shutdown(self) -> None:
        """Shutdown the agent gracefully."""
        if self._shutdown_requested:
            return
            
        self._shutdown_requested = True
        
        try:
            await self._set_status(AgentStatus.SHUTDOWN)
            self.logger.info(f"Shutting down agent {self.name}")
            
            # Cancel background tasks
            await self._stop_background_tasks()
            
            # Cancel pending requests
            await self._cancel_pending_requests()
            
            # Cleanup agent-specific resources
            await self._cleanup_agent()
            
            # Unsubscribe from events
            await self._cleanup_event_subscriptions()
            
            # Emit shutdown event
            await self._emit_event("agent.shutdown", {
                "agent_id": self.agent_id,
                "uptime_seconds": (datetime.now() - self._start_time).total_seconds()
            })
            
            self.logger.info(f"Agent {self.name} shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during agent shutdown: {e}")
    
    async def send_request(self, 
                          target_agent: str,
                          capability: str,
                          parameters: Dict[str, Any] = None,
                          timeout: timedelta = None,
                          priority: AgentPriority = AgentPriority.NORMAL) -> AgentResponse:
        """
        Send a request to another agent.
        
        Args:
            target_agent: ID of the target agent
            capability: Capability to invoke
            parameters: Request parameters
            timeout: Request timeout
            priority: Request priority
            
        Returns:
            Response from the target agent
        """
        request = AgentRequest(
            requesting_agent=self.agent_id,
            target_capability=capability,
            parameters=parameters or {},
            priority=priority,
            timeout=timeout or timedelta(seconds=30)
        )
        
        # Create message
        message = AgentMessage(
            source_agent=self.agent_id,
            target_agent=target_agent,
            message_type=MessageType.REQUEST,
            payload=request.dict(),
            priority=priority,
            timeout=timeout
        )
        
        # Send request and wait for response
        return await self._send_message_and_wait(message, timeout or timedelta(seconds=30))
    
    async def send_notification(self, 
                              target_agent: Optional[str],
                              event_type: str,
                              data: Dict[str, Any] = None) -> None:
        """
        Send a notification to another agent or broadcast.
        
        Args:
            target_agent: ID of target agent, None for broadcast
            event_type: Type of notification
            data: Notification data
        """
        message = AgentMessage(
            source_agent=self.agent_id,
            target_agent=target_agent,
            message_type=MessageType.NOTIFICATION,
            payload={
                "event_type": event_type,
                "data": data or {}
            }
        )
        
        await self._send_message(message)
    
    @property
    def status(self) -> AgentStatus:
        """Get current agent status."""
        return self._state.status
    
    @property
    def state(self) -> AgentState:
        """Get current agent state (copy)."""
        import copy
        return copy.deepcopy(self._state)
    
    @property
    def metrics(self) -> AgentMetrics:
        """Get current agent metrics."""
        # Update runtime metrics
        now = datetime.now()
        self._metrics.uptime_seconds = (now - self._start_time).total_seconds()
        self._metrics.last_updated = now
        return self._metrics
    
    def add_state_change_handler(self, handler: StateChangeHandler) -> None:
        """Add a state change callback handler."""
        self._state_change_handlers.append(handler)
    
    def remove_state_change_handler(self, handler: StateChangeHandler) -> None:
        """Remove a state change callback handler."""
        if handler in self._state_change_handlers:
            self._state_change_handlers.remove(handler)
    
    # Protected methods for subclasses
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any] = None) -> None:
        """Emit an agent event."""
        event = AgentEvent(
            event_type=event_type,
            source_agent=self.agent_id,
            data=data or {}
        )
        
        # Emit via event system
        system_event = Event(
            event_type=f"agent.{event_type}",
            data=event.dict(),
            source=self.agent_id,
            priority=EventPriority.NORMAL
        )
        
        await self.event_system.emit(system_event)
    
    async def _set_status(self, status: AgentStatus, 
                         task_info: Optional[str] = None) -> None:
        """Set agent status and notify handlers."""
        async with self._state_lock:
            if self._state.status == status:
                return
                
            previous_state = AgentState(
                agent_id=self._state.agent_id,
                agent_type=self._state.agent_type,
                status=self._state.status,
                current_task=self._state.current_task,
                last_activity=self._state.last_activity,
                error_count=self._state.error_count,
                processed_requests=self._state.processed_requests,
                active_connections=self._state.active_connections.copy(),
                state_data=self._state.state_data.copy()
            )
            
            # Update state
            self._state.status = status
            self._state.current_task = task_info
            self._state.last_activity = datetime.now()
            
            # Notify handlers
            for handler in self._state_change_handlers:
                try:
                    await handler(previous_state, self._state)
                except Exception as e:
                    self.logger.error(f"State change handler failed: {e}")
    
    async def _handle_error(self, error: Exception, 
                          context: str = "unknown") -> None:
        """Handle an error and potentially trigger recovery."""
        self._error_count += 1
        self._metrics.requests_failed += 1
        
        self.logger.error(f"Agent error in {context}: {error}")
        
        await self._emit_event("agent.error", {
            "error": str(error),
            "context": context,
            "error_count": self._error_count
        })
        
        # Check if recovery is needed
        if (self._error_count >= self._max_errors and 
            not self._recovery_in_progress):
            await self._trigger_recovery()
    
    # Private implementation methods
    
    async def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions."""
        # Subscribe to agent messages
        handler_id = self.event_system.subscribe(
            f"agent.message.{self.agent_id}",
            self._handle_agent_message
        )
        self._event_subscriptions.append(handler_id)
        
        # Subscribe to broadcast messages
        handler_id = self.event_system.subscribe(
            "agent.message.broadcast",
            self._handle_agent_message
        )
        self._event_subscriptions.append(handler_id)
        
        # Subscribe to system events
        handler_id = self.event_system.subscribe(
            "system.*",
            self._handle_system_event
        )
        self._event_subscriptions.append(handler_id)
    
    async def _cleanup_event_subscriptions(self) -> None:
        """Cleanup event subscriptions."""
        for handler_id in self._event_subscriptions:
            self.event_system.unsubscribe(handler_id)
        self._event_subscriptions.clear()
    
    async def _handle_agent_message(self, event: Event) -> None:
        """Handle incoming agent messages."""
        try:
            if event.data and 'message' in event.data:
                message_data = event.data['message']
                message = AgentMessage(**message_data)
                
                # Process different message types
                if message.message_type == MessageType.REQUEST:
                    await self._handle_request_message(message)
                elif message.message_type == MessageType.RESPONSE:
                    await self._handle_response_message(message)
                elif message.message_type == MessageType.NOTIFICATION:
                    await self._handle_notification_message(message)
                    
        except Exception as e:
            await self._handle_error(e, "message_handling")
    
    async def _handle_system_event(self, event: Event) -> None:
        """Handle system-wide events."""
        try:
            # Convert to agent event and handle
            agent_event = AgentEvent(
                event_type=event.event_type,
                source_agent="system",
                data=event.data
            )
            await self._handle_event(agent_event)
            
        except Exception as e:
            await self._handle_error(e, "system_event_handling")
    
    async def _handle_request_message(self, message: AgentMessage) -> None:
        """Handle an incoming request message."""
        try:
            await self._set_status(AgentStatus.PROCESSING, "handling_request")
            
            # Create request object
            request_data = message.payload
            request = AgentRequest(**request_data)
            
            # Process the request
            start_time = datetime.now()
            response = await self._process_request(request)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update metrics
            self._metrics.requests_processed += 1
            if self._metrics.requests_processed > 0:
                self._metrics.average_response_time = (
                    (self._metrics.average_response_time * (self._metrics.requests_processed - 1) + 
                     execution_time) / self._metrics.requests_processed
                )
            
            # Send response
            response_message = AgentMessage(
                source_agent=self.agent_id,
                target_agent=message.source_agent,
                message_type=MessageType.RESPONSE,
                payload=response.dict(),
                correlation_id=message.message_id
            )
            
            await self._send_message(response_message)
            await self._set_status(AgentStatus.IDLE)
            
        except Exception as e:
            await self._handle_error(e, "request_processing")
            
            # Send error response
            error_response = AgentResponse(
                request_id=message.payload.get('request_id', 'unknown'),
                responding_agent=self.agent_id,
                success=False,
                error=str(e)
            )
            
            response_message = AgentMessage(
                source_agent=self.agent_id,
                target_agent=message.source_agent,
                message_type=MessageType.RESPONSE,
                payload=error_response.dict(),
                correlation_id=message.message_id
            )
            
            await self._send_message(response_message)
            await self._set_status(AgentStatus.IDLE)
    
    async def _handle_response_message(self, message: AgentMessage) -> None:
        """Handle an incoming response message."""
        correlation_id = message.correlation_id
        if correlation_id and correlation_id in self._pending_requests:
            future = self._pending_requests.pop(correlation_id)
            if not future.done():
                response = AgentResponse(**message.payload)
                future.set_result(response)
    
    async def _handle_notification_message(self, message: AgentMessage) -> None:
        """Handle an incoming notification message."""
        try:
            event_type = message.payload.get('event_type')
            data = message.payload.get('data', {})
            
            agent_event = AgentEvent(
                event_type=event_type,
                source_agent=message.source_agent,
                data=data
            )
            
            await self._handle_event(agent_event)
            
        except Exception as e:
            await self._handle_error(e, "notification_handling")
    
    async def _send_message(self, message: AgentMessage) -> None:
        """Send a message via the event system."""
        event_type = f"agent.message.{message.target_agent}" if message.target_agent else "agent.message.broadcast"
        
        event = Event(
            event_type=event_type,
            data={"message": message.dict()},
            source=self.agent_id,
            priority=EventPriority.NORMAL if message.priority == AgentPriority.NORMAL else EventPriority.HIGH
        )
        
        await self.event_system.emit(event)
    
    async def _send_message_and_wait(self, message: AgentMessage, 
                                   timeout: timedelta) -> AgentResponse:
        """Send a message and wait for response."""
        # Create future for response
        future = asyncio.Future()
        self._pending_requests[message.message_id] = future
        
        try:
            # Send message
            await self._send_message(message)
            
            # Wait for response
            response = await asyncio.wait_for(future, timeout=timeout.total_seconds())
            return response
            
        except asyncio.TimeoutError:
            if message.message_id in self._pending_requests:
                del self._pending_requests[message.message_id]
            raise AgentTimeoutError(f"Request timed out after {timeout}")
        except Exception as e:
            if message.message_id in self._pending_requests:
                del self._pending_requests[message.message_id]
            raise AgentCommunicationError(f"Communication failed: {e}")
    
    async def _start_background_tasks(self) -> None:
        """Start background tasks."""
        # Health check task
        task = asyncio.create_task(self._health_check_loop())
        self._background_tasks.add(task)
        
        # Metrics update task
        task = asyncio.create_task(self._metrics_update_loop())
        self._background_tasks.add(task)
    
    async def _stop_background_tasks(self) -> None:
        """Stop all background tasks."""
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
    
    async def _cancel_pending_requests(self) -> None:
        """Cancel all pending requests."""
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while not self._shutdown_requested:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Perform basic health checks
                if self._state.status == AgentStatus.ERROR:
                    continue
                    
                # Check if stuck in processing state too long
                if (self._state.status == AgentStatus.PROCESSING and
                    (datetime.now() - self._state.last_activity).total_seconds() > 300):  # 5 minutes
                    self.logger.warning("Agent may be stuck in processing state")
                    await self._emit_event("agent.health.warning", {
                        "issue": "stuck_in_processing",
                        "duration": (datetime.now() - self._state.last_activity).total_seconds()
                    })
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._handle_error(e, "health_check")
    
    async def _metrics_update_loop(self) -> None:
        """Background metrics update loop."""
        while not self._shutdown_requested:
            try:
                await asyncio.sleep(60)  # Update every minute
                
                # Update memory and CPU usage if available
                # This would integrate with system monitoring
                
                # Emit metrics event
                await self._emit_event("agent.metrics", self._metrics.__dict__)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._handle_error(e, "metrics_update")
    
    async def _trigger_recovery(self) -> None:
        """Trigger agent recovery process."""
        if self._recovery_in_progress:
            return
            
        self._recovery_in_progress = True
        
        try:
            await self._set_status(AgentStatus.RECOVERING)
            self.logger.info("Starting agent recovery process")
            
            # Reset error count
            self._error_count = 0
            
            # Cancel current operations
            await self._cancel_pending_requests()
            
            # Re-initialize if needed
            if hasattr(self, '_recover_agent'):
                await self._recover_agent()
            
            await self._set_status(AgentStatus.IDLE)
            
            await self._emit_event("agent.recovered", {
                "recovery_time": datetime.now().isoformat()
            })
            
            self.logger.info("Agent recovery completed successfully")
            
        except Exception as e:
            self.logger.error(f"Agent recovery failed: {e}")
            await self._set_status(AgentStatus.ERROR)
        finally:
            self._recovery_in_progress = False