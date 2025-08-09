"""Agent Manager - Centralized Agent Lifecycle and Coordination

The Agent Manager is responsible for:
- Managing agent lifecycle (initialization, shutdown)
- Coordinating multi-agent collaborations
- Monitoring agent health and performance
- Load balancing and resource allocation
- Agent discovery and capability mapping
"""

import asyncio
from typing import Dict, Any, List, Optional, Set, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid

from loguru import logger

from ..core.event_system import EventSystem, Event, EventPriority
from ..core.config_manager import ConfigManager
from .base_agent import BaseAgent
from .agent_types import (
    AgentType, AgentStatus, AgentState, AgentMetrics,
    AgentMessage, AgentRequest, AgentResponse, AgentEvent,
    CollaborationPlan, CollaborationPattern
)

# Import all agent implementations
from .coordinator import Coordinator
from .task_planner import TaskPlanner


class AgentManagerStatus(Enum):
    """Status of the agent manager"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class AgentRegistration:
    """Agent registration information"""
    agent: BaseAgent
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    health_status: str = "healthy"
    load_factor: float = 0.0
    active_requests: int = 0


class AgentManager:
    """
    Centralized management system for all intelligent agents.
    
    Provides agent lifecycle management, health monitoring,
    and coordination services.
    """
    
    def __init__(self, 
                 event_system: EventSystem,
                 config_manager: ConfigManager):
        """
        Initialize the Agent Manager.
        
        Args:
            event_system: Event system for communication
            config_manager: Configuration manager
        """
        self.event_system = event_system
        self.config_manager = config_manager
        
        # Manager state
        self._status = AgentManagerStatus.INITIALIZING
        self._agents: Dict[str, AgentRegistration] = {}
        self._agent_types: Dict[AgentType, str] = {}  # type -> agent_id mapping
        
        # Collaboration management
        self._active_collaborations: Dict[str, CollaborationPlan] = {}
        self._collaboration_history: List[Dict[str, Any]] = []
        
        # Health monitoring
        self._health_check_interval = 30  # seconds
        self._health_check_task: Optional[asyncio.Task] = None
        
        # Performance monitoring
        self._performance_metrics: Dict[str, Any] = {
            'total_agents': 0,
            'active_agents': 0,
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'uptime_seconds': 0.0
        }
        self._start_time = datetime.now()
        
        # Event subscriptions
        self._event_subscriptions: List[str] = []
        
        # Agent class registry
        self._agent_classes: Dict[AgentType, Type[BaseAgent]] = {
            AgentType.COORDINATOR: Coordinator,
            AgentType.TASK_PLANNER: TaskPlanner,
            # Add other agents as they are implemented
        }
        
        self.logger = logger.bind(component="agent_manager")
    
    async def initialize(self) -> None:
        """Initialize the agent manager and core agents."""
        try:
            self.logger.info("Initializing Agent Manager")
            self._status = AgentManagerStatus.INITIALIZING
            
            # Subscribe to system events
            await self._setup_event_subscriptions()
            
            # Load configuration
            agent_config = self.config_manager.get_agent_config()
            
            # Initialize core agents
            await self._initialize_core_agents(agent_config)
            
            # Start health monitoring
            await self._start_health_monitoring()
            
            # Start performance monitoring
            await self._start_performance_monitoring()
            
            self._status = AgentManagerStatus.RUNNING
            
            # Emit system ready event
            await self._emit_system_event("agent_manager.initialized", {
                "total_agents": len(self._agents),
                "agent_types": list(self._agent_types.keys())
            })
            
            self.logger.info(f"Agent Manager initialized with {len(self._agents)} agents")
            
        except Exception as e:
            self._status = AgentManagerStatus.ERROR
            self.logger.error(f"Failed to initialize Agent Manager: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown all agents and the manager."""
        try:
            self.logger.info("Shutting down Agent Manager")
            self._status = AgentManagerStatus.SHUTTING_DOWN
            
            # Stop monitoring tasks
            if self._health_check_task and not self._health_check_task.done():
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown all agents
            await self._shutdown_all_agents()
            
            # Complete any active collaborations
            await self._cleanup_collaborations()
            
            # Cleanup event subscriptions
            await self._cleanup_event_subscriptions()
            
            self._status = AgentManagerStatus.STOPPED
            
            # Emit shutdown event
            await self._emit_system_event("agent_manager.shutdown", {
                "uptime_seconds": (datetime.now() - self._start_time).total_seconds()
            })
            
            self.logger.info("Agent Manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def register_agent(self, agent: BaseAgent) -> bool:
        """
        Register a new agent with the manager.
        
        Args:
            agent: Agent instance to register
            
        Returns:
            True if registration successful
        """
        try:
            if agent.agent_id in self._agents:
                self.logger.warning(f"Agent {agent.agent_id} already registered")
                return False
            
            # Initialize the agent
            await agent.initialize()
            
            # Register the agent
            registration = AgentRegistration(agent=agent)
            self._agents[agent.agent_id] = registration
            self._agent_types[agent.agent_type] = agent.agent_id
            
            # Update metrics
            self._performance_metrics['total_agents'] += 1
            if agent.status == AgentStatus.IDLE:
                self._performance_metrics['active_agents'] += 1
            
            # Emit registration event
            await self._emit_system_event("agent.registered", {
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type.value,
                "capabilities": [cap.name for cap in agent.capabilities]
            })
            
            self.logger.info(f"Registered agent: {agent.agent_id} ({agent.agent_type.value})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent.agent_id}: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the manager.
        
        Args:
            agent_id: ID of agent to unregister
            
        Returns:
            True if unregistration successful
        """
        try:
            if agent_id not in self._agents:
                self.logger.warning(f"Agent {agent_id} not found for unregistration")
                return False
            
            registration = self._agents[agent_id]
            agent = registration.agent
            
            # Shutdown the agent
            await agent.shutdown()
            
            # Remove from registrations
            del self._agents[agent_id]
            if agent.agent_type in self._agent_types:
                del self._agent_types[agent.agent_type]
            
            # Update metrics
            self._performance_metrics['total_agents'] -= 1
            if agent.status == AgentStatus.IDLE:
                self._performance_metrics['active_agents'] -= 1
            
            # Emit unregistration event
            await self._emit_system_event("agent.unregistered", {
                "agent_id": agent_id,
                "agent_type": agent.agent_type.value
            })
            
            self.logger.info(f"Unregistered agent: {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False
    
    async def send_request(self, 
                          target_agent: str,
                          capability: str,
                          parameters: Dict[str, Any] = None,
                          timeout: timedelta = None) -> AgentResponse:
        """
        Send a request to a specific agent.
        
        Args:
            target_agent: ID of target agent
            capability: Capability to invoke
            parameters: Request parameters
            timeout: Request timeout
            
        Returns:
            Agent response
        """
        if target_agent not in self._agents:
            raise ValueError(f"Agent {target_agent} not found")
        
        registration = self._agents[target_agent]
        agent = registration.agent
        
        # Update load tracking
        registration.active_requests += 1
        registration.load_factor = registration.active_requests / 10.0  # Simple load calculation
        
        try:
            # Send request to agent
            response = await agent.send_request(
                target_agent=target_agent,
                capability=capability,
                parameters=parameters or {},
                timeout=timeout or timedelta(seconds=30)
            )
            
            # Update metrics
            self._performance_metrics['total_requests'] += 1
            if response.success:
                self._performance_metrics['successful_requests'] += 1
            else:
                self._performance_metrics['failed_requests'] += 1
            
            return response
            
        finally:
            # Update load tracking
            registration.active_requests = max(0, registration.active_requests - 1)
            registration.load_factor = registration.active_requests / 10.0
    
    async def coordinate_agents(self, collaboration_plan: CollaborationPlan) -> Dict[str, Any]:
        """
        Coordinate a multi-agent collaboration.
        
        Args:
            collaboration_plan: Plan for agent collaboration
            
        Returns:
            Collaboration results
        """
        self._active_collaborations[collaboration_plan.plan_id] = collaboration_plan
        
        try:
            # Emit collaboration start event
            await self._emit_system_event("collaboration.started", {
                "plan_id": collaboration_plan.plan_id,
                "participants": collaboration_plan.participants,
                "pattern": collaboration_plan.pattern.value
            })
            
            # Execute based on collaboration pattern
            if collaboration_plan.pattern == CollaborationPattern.SEQUENTIAL:
                results = await self._execute_sequential_collaboration(collaboration_plan)
            elif collaboration_plan.pattern == CollaborationPattern.PARALLEL:
                results = await self._execute_parallel_collaboration(collaboration_plan)
            elif collaboration_plan.pattern == CollaborationPattern.PIPELINE:
                results = await self._execute_pipeline_collaboration(collaboration_plan)
            else:
                results = await self._execute_default_collaboration(collaboration_plan)
            
            # Store collaboration history
            self._collaboration_history.append({
                'plan_id': collaboration_plan.plan_id,
                'pattern': collaboration_plan.pattern.value,
                'participants': collaboration_plan.participants,
                'success': True,
                'execution_time': (datetime.now() - collaboration_plan.created_at).total_seconds(),
                'timestamp': datetime.now()
            })
            
            # Emit collaboration completion event
            await self._emit_system_event("collaboration.completed", {
                "plan_id": collaboration_plan.plan_id,
                "success": True,
                "results": results
            })
            
            return results
            
        except Exception as e:
            # Store failed collaboration
            self._collaboration_history.append({
                'plan_id': collaboration_plan.plan_id,
                'pattern': collaboration_plan.pattern.value,
                'participants': collaboration_plan.participants,
                'success': False,
                'error': str(e),
                'execution_time': (datetime.now() - collaboration_plan.created_at).total_seconds(),
                'timestamp': datetime.now()
            })
            
            # Emit collaboration failure event
            await self._emit_system_event("collaboration.failed", {
                "plan_id": collaboration_plan.plan_id,
                "error": str(e)
            })
            
            raise
            
        finally:
            # Clean up active collaboration
            if collaboration_plan.plan_id in self._active_collaborations:
                del self._active_collaborations[collaboration_plan.plan_id]
    
    def get_agent_by_type(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """Get agent by type."""
        agent_id = self._agent_types.get(agent_type)
        if agent_id and agent_id in self._agents:
            return self._agents[agent_id].agent
        return None
    
    def get_agent_by_id(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID."""
        if agent_id in self._agents:
            return self._agents[agent_id].agent
        return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        # Update uptime
        self._performance_metrics['uptime_seconds'] = (
            datetime.now() - self._start_time
        ).total_seconds()
        
        # Calculate active agents
        self._performance_metrics['active_agents'] = sum(
            1 for reg in self._agents.values()
            if reg.agent.status == AgentStatus.IDLE
        )
        
        # Get agent details
        agent_details = {}
        for agent_id, registration in self._agents.items():
            agent = registration.agent
            agent_details[agent_id] = {
                'type': agent.agent_type.value,
                'status': agent.status.value,
                'health': registration.health_status,
                'load_factor': registration.load_factor,
                'active_requests': registration.active_requests,
                'last_heartbeat': registration.last_heartbeat.isoformat()
            }
        
        return {
            'manager_status': self._status.value,
            'performance_metrics': self._performance_metrics,
            'agents': agent_details,
            'active_collaborations': len(self._active_collaborations),
            'collaboration_history_size': len(self._collaboration_history)
        }
    
    def get_available_capabilities(self) -> Dict[str, List[str]]:
        """Get all available capabilities across agents."""
        capabilities = {}
        
        for agent_id, registration in self._agents.items():
            agent = registration.agent
            if agent.status == AgentStatus.IDLE:
                agent_capabilities = [cap.name for cap in agent.capabilities]
                capabilities[agent_id] = agent_capabilities
        
        return capabilities
    
    # Private implementation methods
    
    async def _initialize_core_agents(self, config: Dict[str, Any]) -> None:
        """Initialize core system agents."""
        core_agents = config.get('core_agents', ['coordinator', 'task_planner'])
        
        for agent_name in core_agents:
            try:
                # Map agent name to type
                agent_type = self._map_name_to_type(agent_name)
                if agent_type not in self._agent_classes:
                    self.logger.warning(f"No implementation found for agent type: {agent_type}")
                    continue
                
                # Create agent instance
                agent_class = self._agent_classes[agent_type]
                agent_config = config.get(agent_name, {})
                
                # Create and register agent
                agent = agent_class(
                    event_system=self.event_system,
                    config=agent_config
                )
                
                await self.register_agent(agent)
                
            except Exception as e:
                self.logger.error(f"Failed to initialize agent {agent_name}: {e}")
    
    def _map_name_to_type(self, agent_name: str) -> AgentType:
        """Map agent name to AgentType enum."""
        name_mapping = {
            'coordinator': AgentType.COORDINATOR,
            'task_planner': AgentType.TASK_PLANNER,
            'presence_monitor': AgentType.PRESENCE_MONITOR,
            'auto_worker': AgentType.AUTO_WORKER,
            'security_guardian': AgentType.SECURITY_GUARDIAN,
            'handover_manager': AgentType.HANDOVER_MANAGER,
            'session_manager': AgentType.SESSION_MANAGER
        }
        return name_mapping.get(agent_name, AgentType.COORDINATOR)
    
    async def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions."""
        # Subscribe to agent events
        handler_id = self.event_system.subscribe(
            "agent.*",
            self._handle_agent_event
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
    
    async def _handle_agent_event(self, event: Event) -> None:
        """Handle agent-related events."""
        try:
            event_type = event.event_type
            source = event.source
            
            if event_type == "agent.error":
                await self._handle_agent_error(source, event.data)
            elif event_type == "agent.status_change":
                await self._handle_agent_status_change(source, event.data)
            elif event_type == "agent.metrics_update":
                await self._handle_agent_metrics_update(source, event.data)
                
        except Exception as e:
            self.logger.error(f"Error handling agent event {event.event_type}: {e}")
    
    async def _handle_system_event(self, event: Event) -> None:
        """Handle system-wide events."""
        try:
            event_type = event.event_type
            
            if event_type == "system.shutdown_requested":
                await self.shutdown()
            elif event_type == "system.health_check":
                await self._perform_system_health_check()
                
        except Exception as e:
            self.logger.error(f"Error handling system event {event.event_type}: {e}")
    
    async def _start_health_monitoring(self) -> None:
        """Start health monitoring background task."""
        self._health_check_task = asyncio.create_task(self._health_monitoring_loop())
    
    async def _start_performance_monitoring(self) -> None:
        """Start performance monitoring background task."""
        perf_task = asyncio.create_task(self._performance_monitoring_loop())
        # Don't store this task as it's fire-and-forget
    
    async def _health_monitoring_loop(self) -> None:
        """Background health monitoring loop."""
        while self._status == AgentManagerStatus.RUNNING:
            try:
                await asyncio.sleep(self._health_check_interval)
                
                # Check health of all agents
                for agent_id, registration in self._agents.items():
                    try:
                        agent = registration.agent
                        is_healthy = await agent.health_check()
                        
                        if is_healthy:
                            registration.health_status = "healthy"
                            registration.last_heartbeat = datetime.now()
                        else:
                            registration.health_status = "unhealthy"
                            self.logger.warning(f"Agent {agent_id} health check failed")
                            
                    except Exception as e:
                        registration.health_status = "error"
                        self.logger.error(f"Health check error for agent {agent_id}: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
    
    async def _performance_monitoring_loop(self) -> None:
        """Background performance monitoring loop."""
        while self._status == AgentManagerStatus.RUNNING:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                # Calculate average response time
                total_time = 0.0
                total_requests = 0
                
                for registration in self._agents.values():
                    agent_metrics = registration.agent.metrics
                    if agent_metrics.requests_processed > 0:
                        total_time += agent_metrics.average_response_time * agent_metrics.requests_processed
                        total_requests += agent_metrics.requests_processed
                
                if total_requests > 0:
                    self._performance_metrics['average_response_time'] = total_time / total_requests
                
                # Emit performance metrics event
                await self._emit_system_event("agent_manager.performance_metrics", 
                                            self._performance_metrics)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in performance monitoring loop: {e}")
    
    async def _shutdown_all_agents(self) -> None:
        """Shutdown all registered agents."""
        shutdown_tasks = []
        
        for agent_id, registration in self._agents.items():
            task = asyncio.create_task(self._shutdown_agent(agent_id, registration))
            shutdown_tasks.append(task)
        
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
    
    async def _shutdown_agent(self, agent_id: str, registration: AgentRegistration) -> None:
        """Shutdown a specific agent."""
        try:
            await registration.agent.shutdown()
            self.logger.info(f"Agent {agent_id} shutdown complete")
        except Exception as e:
            self.logger.error(f"Error shutting down agent {agent_id}: {e}")
    
    async def _cleanup_collaborations(self) -> None:
        """Cleanup any active collaborations."""
        for plan_id in list(self._active_collaborations.keys()):
            try:
                plan = self._active_collaborations[plan_id]
                self.logger.warning(f"Canceling active collaboration {plan_id}")
                
                # Emit cancellation event
                await self._emit_system_event("collaboration.cancelled", {
                    "plan_id": plan_id,
                    "reason": "system_shutdown"
                })
                
                del self._active_collaborations[plan_id]
                
            except Exception as e:
                self.logger.error(f"Error cleaning up collaboration {plan_id}: {e}")
    
    async def _execute_sequential_collaboration(self, plan: CollaborationPlan) -> Dict[str, Any]:
        """Execute sequential collaboration pattern."""
        results = {}
        
        for step in plan.steps:
            agent_id = step["agent_id"]
            capabilities = step["capabilities"]
            parameters = step["parameters"]
            
            for capability in capabilities:
                response = await self.send_request(agent_id, capability, parameters)
                results[f"{agent_id}_{capability}"] = response
                
                if not response.success:
                    raise Exception(f"Step failed: {agent_id}.{capability} - {response.error}")
        
        return results
    
    async def _execute_parallel_collaboration(self, plan: CollaborationPlan) -> Dict[str, Any]:
        """Execute parallel collaboration pattern."""
        tasks = []
        
        for step in plan.steps:
            agent_id = step["agent_id"]
            capabilities = step["capabilities"]
            parameters = step["parameters"]
            
            for capability in capabilities:
                task = asyncio.create_task(
                    self.send_request(agent_id, capability, parameters)
                )
                tasks.append((f"{agent_id}_{capability}", task))
        
        # Wait for all tasks
        results = {}
        for key, task in tasks:
            try:
                response = await task
                results[key] = response
            except Exception as e:
                results[key] = AgentResponse(
                    request_id="",
                    responding_agent=key.split("_")[0],
                    success=False,
                    error=str(e)
                )
        
        return results
    
    async def _execute_pipeline_collaboration(self, plan: CollaborationPlan) -> Dict[str, Any]:
        """Execute pipeline collaboration pattern."""
        results = {}
        pipeline_data = {}
        
        for step in plan.steps:
            agent_id = step["agent_id"]
            capabilities = step["capabilities"]
            parameters = {**step["parameters"], **pipeline_data}
            
            for capability in capabilities:
                response = await self.send_request(agent_id, capability, parameters)
                results[f"{agent_id}_{capability}"] = response
                
                if response.success and response.data:
                    pipeline_data.update(response.data)
                else:
                    raise Exception(f"Pipeline step failed: {agent_id}.{capability}")
        
        return results
    
    async def _execute_default_collaboration(self, plan: CollaborationPlan) -> Dict[str, Any]:
        """Execute default collaboration pattern (sequential)."""
        return await self._execute_sequential_collaboration(plan)
    
    async def _emit_system_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a system-wide event."""
        event = Event(
            event_type=event_type,
            data=data,
            source="agent_manager",
            priority=EventPriority.NORMAL
        )
        await self.event_system.emit(event)
    
    async def _handle_agent_error(self, agent_id: str, error_data: Dict[str, Any]) -> None:
        """Handle agent error events."""
        if agent_id in self._agents:
            registration = self._agents[agent_id]
            registration.health_status = "error"
            
            # Log error
            error_msg = error_data.get("error", "Unknown error")
            self.logger.error(f"Agent {agent_id} reported error: {error_msg}")
            
            # Consider restarting agent if critical
            error_count = error_data.get("error_count", 0)
            if error_count > 5:
                self.logger.warning(f"Agent {agent_id} has high error count, considering restart")
    
    async def _handle_agent_status_change(self, agent_id: str, status_data: Dict[str, Any]) -> None:
        """Handle agent status change events."""
        if agent_id in self._agents:
            old_status = status_data.get("old_status")
            new_status = status_data.get("new_status")
            
            self.logger.debug(f"Agent {agent_id} status changed: {old_status} -> {new_status}")
    
    async def _handle_agent_metrics_update(self, agent_id: str, metrics_data: Dict[str, Any]) -> None:
        """Handle agent metrics update events."""
        if agent_id in self._agents:
            registration = self._agents[agent_id]
            registration.last_heartbeat = datetime.now()
            
            # Update load factor based on metrics
            requests_processed = metrics_data.get("requests_processed", 0)
            if requests_processed > 0:
                registration.load_factor = min(requests_processed / 100.0, 1.0)
    
    async def _perform_system_health_check(self) -> None:
        """Perform comprehensive system health check."""
        try:
            healthy_agents = 0
            total_agents = len(self._agents)
            
            for agent_id, registration in self._agents.items():
                if registration.health_status == "healthy":
                    healthy_agents += 1
            
            health_ratio = healthy_agents / max(total_agents, 1)
            
            # Emit health status event
            await self._emit_system_event("system.health_status", {
                "healthy_agents": healthy_agents,
                "total_agents": total_agents,
                "health_ratio": health_ratio,
                "system_status": "healthy" if health_ratio > 0.8 else "degraded"
            })
            
        except Exception as e:
            self.logger.error(f"Error performing system health check: {e}")