"""Coordinator Agent - Central Orchestration Hub

The Coordinator is the central command center for the agent system,
responsible for:
- Receiving and parsing user requests
- Orchestrating multi-agent collaborations
- Managing system-wide state and context
- Routing requests to appropriate agents
- Aggregating and delivering final responses
"""

import asyncio
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid

from loguru import logger

from ..core.event_system import EventSystem
from ..core.types import Intent, Context, RiskLevel
from ..speech.types import VoiceCommand, VoiceResponse
from .base_agent import BaseAgent
from .agent_types import (
    AgentType, AgentStatus, AgentRequest, AgentResponse, AgentEvent,
    AgentCapability, CollaborationPlan, CollaborationPattern,
    MessageType, AgentPriority
)


class RequestStatus(Enum):
    """Status of user requests being processed"""
    RECEIVED = "received"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class UserRequest:
    """Complete user request context"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_input: str = ""
    intent: Optional[Intent] = None
    context: Optional[Context] = None
    status: RequestStatus = RequestStatus.RECEIVED
    collaboration_plan: Optional[CollaborationPlan] = None
    agent_responses: Dict[str, AgentResponse] = field(default_factory=dict)
    final_response: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class Coordinator(BaseAgent):
    """
    Central coordination agent that orchestrates all other agents.
    
    The Coordinator serves as the main entry point for user interactions
    and manages the overall flow of request processing across the system.
    """
    
    def __init__(self, event_system: EventSystem, config: Optional[Dict[str, Any]] = None):
        super().__init__("coordinator", event_system, config)
        
        # Request management
        self._active_requests: Dict[str, UserRequest] = {}
        self._request_queue: asyncio.Queue = asyncio.Queue()
        
        # Agent registry and status tracking
        self._registered_agents: Dict[str, AgentStatus] = {}
        self._agent_capabilities: Dict[str, List[AgentCapability]] = {}
        self._agent_load: Dict[str, int] = {}
        
        # Collaboration management
        self._active_collaborations: Dict[str, CollaborationPlan] = {}
        
        # Performance tracking
        self._request_count = 0
        self._successful_requests = 0
        self._average_response_time = 0.0
    
    @property
    def agent_type(self) -> AgentType:
        return AgentType.COORDINATOR
    
    @property
    def name(self) -> str:
        return "Coordinator"
    
    @property
    def description(self) -> str:
        return "Central orchestration hub managing user requests and agent coordination"
    
    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="process_user_request",
                description="Process and route user requests to appropriate agents",
                input_types=["voice_command", "text_input", "intent"],
                output_types=["voice_response", "text_response"],
                risk_level=RiskLevel.LOW
            ),
            AgentCapability(
                name="coordinate_agents",
                description="Coordinate multi-agent collaborations",
                input_types=["collaboration_plan"],
                output_types=["coordination_result"],
                risk_level=RiskLevel.MEDIUM
            ),
            AgentCapability(
                name="get_system_status",
                description="Get overall system status and agent health",
                input_types=["status_request"],
                output_types=["system_status"],
                risk_level=RiskLevel.LOW
            ),
            AgentCapability(
                name="manage_agent_registry",
                description="Register and manage agent lifecycle",
                input_types=["agent_info"],
                output_types=["registry_status"],
                risk_level=RiskLevel.HIGH
            )
        ]
    
    async def _initialize_agent(self) -> None:
        """Initialize coordinator-specific functionality."""
        self.logger.info("Initializing Coordinator agent")
        
        # Start request processing loop
        request_processor = asyncio.create_task(self._process_request_queue())
        self._background_tasks.add(request_processor)
        
        # Start agent monitoring loop
        agent_monitor = asyncio.create_task(self._monitor_agents())
        self._background_tasks.add(agent_monitor)
        
        # Subscribe to agent registration events
        self.event_system.subscribe(
            "agent.initialized",
            self._handle_agent_registration
        )
        
        self.event_system.subscribe(
            "agent.shutdown",
            self._handle_agent_deregistration
        )
        
        self.logger.info("Coordinator initialization complete")
    
    async def _process_request(self, request: AgentRequest) -> AgentResponse:
        """Process incoming agent requests."""
        capability = request.target_capability
        
        try:
            if capability == "process_user_request":
                return await self._handle_user_request(request)
            elif capability == "coordinate_agents":
                return await self._handle_coordination_request(request)
            elif capability == "get_system_status":
                return await self._handle_status_request(request)
            elif capability == "manage_agent_registry":
                return await self._handle_registry_request(request)
            else:
                raise ValueError(f"Unknown capability: {capability}")
                
        except Exception as e:
            self.logger.error(f"Error processing request {request.request_id}: {e}")
            return AgentResponse(
                request_id=request.request_id,
                responding_agent=self.agent_id,
                success=False,
                error=str(e)
            )
    
    async def _handle_event(self, event: AgentEvent) -> None:
        """Handle agent events."""
        try:
            if event.event_type == "user.voice_command":
                await self._handle_voice_command(event)
            elif event.event_type == "user.text_input":
                await self._handle_text_input(event)
            elif event.event_type == "agent.status_update":
                await self._handle_agent_status_update(event)
            elif event.event_type.startswith("collaboration."):
                await self._handle_collaboration_event(event)
                
        except Exception as e:
            await self._handle_error(e, f"event_handling_{event.event_type}")
    
    async def _cleanup_agent(self) -> None:
        """Cleanup coordinator resources."""
        # Complete any pending requests
        for request_id, user_request in self._active_requests.items():
            if user_request.status not in [RequestStatus.COMPLETED, RequestStatus.FAILED]:
                user_request.status = RequestStatus.FAILED
                user_request.error = "System shutdown"
        
        self.logger.info("Coordinator cleanup complete")
    
    # Public API methods
    
    async def process_voice_command(self, voice_command: VoiceCommand) -> VoiceResponse:
        """
        Process a voice command from the user.
        
        Args:
            voice_command: The voice command to process
            
        Returns:
            Voice response with results
        """
        user_request = UserRequest(
            user_input=voice_command.text,
            context=voice_command.context
        )
        
        self._active_requests[user_request.request_id] = user_request
        await self._request_queue.put(user_request)
        
        # Wait for processing to complete
        while user_request.status not in [RequestStatus.COMPLETED, RequestStatus.FAILED]:
            await asyncio.sleep(0.1)
        
        if user_request.status == RequestStatus.COMPLETED:
            return VoiceResponse(
                success=True,
                text=user_request.final_response or "Request completed successfully",
                audio_data=b"",  # TTS would generate this
                metadata={"request_id": user_request.request_id}
            )
        else:
            return VoiceResponse(
                success=False,
                text=f"Request failed: {user_request.error}",
                audio_data=b"",
                metadata={"request_id": user_request.request_id}
            )
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system status overview."""
        return {
            "coordinator_status": self.status.value,
            "registered_agents": len(self._registered_agents),
            "active_requests": len(self._active_requests),
            "active_collaborations": len(self._active_collaborations),
            "agent_statuses": self._registered_agents.copy(),
            "agent_load": self._agent_load.copy(),
            "performance_metrics": {
                "total_requests": self._request_count,
                "successful_requests": self._successful_requests,
                "success_rate": self._successful_requests / max(self._request_count, 1) * 100,
                "average_response_time": self._average_response_time
            }
        }
    
    async def coordinate_multi_agent_task(self, 
                                        task_description: str,
                                        required_capabilities: List[str],
                                        context: Optional[Context] = None) -> Dict[str, Any]:
        """
        Coordinate a complex task requiring multiple agents.
        
        Args:
            task_description: Description of the task
            required_capabilities: List of required capabilities
            context: Optional context information
            
        Returns:
            Coordination results
        """
        # Create collaboration plan
        collaboration_plan = await self._create_collaboration_plan(
            task_description, required_capabilities, context
        )
        
        # Execute the collaboration
        return await self._execute_collaboration(collaboration_plan)
    
    # Private implementation methods
    
    async def _process_request_queue(self) -> None:
        """Main request processing loop."""
        while not self._shutdown_requested:
            try:
                # Get next request with timeout
                try:
                    user_request = await asyncio.wait_for(
                        self._request_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process the request
                await self._process_user_request(user_request)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in request processing loop: {e}")
    
    async def _process_user_request(self, user_request: UserRequest) -> None:
        """Process a complete user request through the pipeline."""
        start_time = datetime.now()
        
        try:
            # Update request count
            self._request_count += 1
            
            # Step 1: Analyze the request
            user_request.status = RequestStatus.ANALYZING
            intent = await self._analyze_user_request(user_request)
            user_request.intent = intent
            
            # Step 2: Create execution plan
            user_request.status = RequestStatus.PLANNING
            collaboration_plan = await self._create_execution_plan(user_request)
            user_request.collaboration_plan = collaboration_plan
            
            # Step 3: Execute the plan
            user_request.status = RequestStatus.EXECUTING
            results = await self._execute_collaboration(collaboration_plan)
            
            # Step 4: Aggregate results
            user_request.status = RequestStatus.AGGREGATING
            final_response = await self._aggregate_results(user_request, results)
            user_request.final_response = final_response
            
            # Mark as completed
            user_request.status = RequestStatus.COMPLETED
            user_request.completed_at = datetime.now()
            self._successful_requests += 1
            
            # Update performance metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            self._average_response_time = (
                (self._average_response_time * (self._request_count - 1) + execution_time) 
                / self._request_count
            )
            
            self.logger.info(f"Request {user_request.request_id} completed in {execution_time:.2f}s")
            
        except Exception as e:
            user_request.status = RequestStatus.FAILED
            user_request.error = str(e)
            user_request.completed_at = datetime.now()
            
            self.logger.error(f"Request {user_request.request_id} failed: {e}")
    
    async def _analyze_user_request(self, user_request: UserRequest) -> Intent:
        """Analyze user input and extract intent."""
        # This would integrate with TaskPlanner for intent recognition
        # For now, create a simple intent
        
        # Try to get intent from TaskPlanner
        try:
            if "task_planner" in self._registered_agents:
                response = await self.send_request(
                    "task_planner",
                    "parse_intent",
                    {
                        "user_input": user_request.user_input,
                        "context": user_request.context.dict() if user_request.context else {}
                    },
                    timeout=timedelta(seconds=10)
                )
                
                if response.success and response.data:
                    return Intent(**response.data.get("intent", {}))
            
        except Exception as e:
            self.logger.warning(f"Could not get intent from TaskPlanner: {e}")
        
        # Fallback: create basic intent
        return Intent(
            user_input=user_request.user_input,
            intent_type="general_command",
            confidence=0.5,
            parameters={"input": user_request.user_input}
        )
    
    async def _create_execution_plan(self, user_request: UserRequest) -> CollaborationPlan:
        """Create an execution plan for the user request."""
        if not user_request.intent:
            raise ValueError("Cannot create execution plan without intent")
        
        # Determine required capabilities based on intent
        required_capabilities = await self._determine_required_capabilities(user_request.intent)
        
        # Find agents that can provide these capabilities
        agent_assignments = await self._assign_agents(required_capabilities)
        
        # Create collaboration plan
        collaboration_plan = CollaborationPlan(
            initiator=self.agent_id,
            participants=list(agent_assignments.keys()),
            pattern=self._determine_collaboration_pattern(agent_assignments),
            steps=await self._create_execution_steps(agent_assignments, user_request)
        )
        
        return collaboration_plan
    
    async def _determine_required_capabilities(self, intent: Intent) -> List[str]:
        """Determine what capabilities are needed for an intent."""
        # This is a simplified capability mapping
        # In a full implementation, this would be more sophisticated
        
        intent_type = intent.intent_type.lower()
        
        capability_mapping = {
            "file_operation": ["file_management", "security_check"],
            "system_command": ["system_execution", "security_check"],
            "information_query": ["information_retrieval"],
            "user_interaction": ["presence_monitoring", "session_management"],
            "complex_task": ["task_planning", "auto_execution", "security_check"]
        }
        
        # Default capabilities for unknown intents
        return capability_mapping.get(intent_type, ["task_planning", "auto_execution"])
    
    async def _assign_agents(self, required_capabilities: List[str]) -> Dict[str, List[str]]:
        """Assign agents to handle required capabilities."""
        agent_assignments = {}
        
        for capability in required_capabilities:
            # Find agents that have this capability
            suitable_agents = []
            
            for agent_id, capabilities in self._agent_capabilities.items():
                if any(cap.name == capability for cap in capabilities):
                    # Check agent status and load
                    if (self._registered_agents.get(agent_id) == AgentStatus.IDLE and
                        self._agent_load.get(agent_id, 0) < 5):  # Max 5 concurrent tasks
                        suitable_agents.append(agent_id)
            
            if suitable_agents:
                # Select agent with lowest load
                selected_agent = min(suitable_agents, key=lambda a: self._agent_load.get(a, 0))
                if selected_agent not in agent_assignments:
                    agent_assignments[selected_agent] = []
                agent_assignments[selected_agent].append(capability)
            else:
                self.logger.warning(f"No suitable agent found for capability: {capability}")
        
        return agent_assignments
    
    def _determine_collaboration_pattern(self, agent_assignments: Dict[str, List[str]]) -> CollaborationPattern:
        """Determine the appropriate collaboration pattern."""
        if len(agent_assignments) == 1:
            return CollaborationPattern.DELEGATION
        elif len(agent_assignments) <= 3:
            return CollaborationPattern.SEQUENTIAL
        else:
            return CollaborationPattern.PARALLEL
    
    async def _create_execution_steps(self, 
                                    agent_assignments: Dict[str, List[str]],
                                    user_request: UserRequest) -> List[Dict[str, Any]]:
        """Create detailed execution steps."""
        steps = []
        
        for agent_id, capabilities in agent_assignments.items():
            step = {
                "step_id": str(uuid.uuid4()),
                "agent_id": agent_id,
                "capabilities": capabilities,
                "parameters": {
                    "intent": user_request.intent.dict() if user_request.intent else {},
                    "context": user_request.context.dict() if user_request.context else {},
                    "user_input": user_request.user_input
                },
                "dependencies": []  # Would be calculated based on capability dependencies
            }
            steps.append(step)
        
        return steps
    
    async def _execute_collaboration(self, collaboration_plan: CollaborationPlan) -> Dict[str, Any]:
        """Execute a collaboration plan."""
        self._active_collaborations[collaboration_plan.plan_id] = collaboration_plan
        
        try:
            results = {}
            
            if collaboration_plan.pattern == CollaborationPattern.SEQUENTIAL:
                results = await self._execute_sequential(collaboration_plan)
            elif collaboration_plan.pattern == CollaborationPattern.PARALLEL:
                results = await self._execute_parallel(collaboration_plan)
            elif collaboration_plan.pattern == CollaborationPattern.DELEGATION:
                results = await self._execute_delegation(collaboration_plan)
            else:
                raise ValueError(f"Unsupported collaboration pattern: {collaboration_plan.pattern}")
            
            return results
            
        finally:
            if collaboration_plan.plan_id in self._active_collaborations:
                del self._active_collaborations[collaboration_plan.plan_id]
    
    async def _execute_sequential(self, collaboration_plan: CollaborationPlan) -> Dict[str, Any]:
        """Execute steps sequentially."""
        results = {}
        
        for step in collaboration_plan.steps:
            agent_id = step["agent_id"]
            capabilities = step["capabilities"]
            parameters = step["parameters"]
            
            # Execute each capability for this agent
            for capability in capabilities:
                try:
                    response = await self.send_request(
                        agent_id,
                        capability,
                        parameters,
                        timeout=timedelta(seconds=60)
                    )
                    
                    results[f"{agent_id}_{capability}"] = response
                    
                    # Update agent load
                    self._agent_load[agent_id] = self._agent_load.get(agent_id, 0) + 1
                    
                except Exception as e:
                    self.logger.error(f"Step execution failed for {agent_id}.{capability}: {e}")
                    results[f"{agent_id}_{capability}"] = AgentResponse(
                        request_id="",
                        responding_agent=agent_id,
                        success=False,
                        error=str(e)
                    )
        
        return results
    
    async def _execute_parallel(self, collaboration_plan: CollaborationPlan) -> Dict[str, Any]:
        """Execute steps in parallel."""
        tasks = []
        
        for step in collaboration_plan.steps:
            agent_id = step["agent_id"]
            capabilities = step["capabilities"]
            parameters = step["parameters"]
            
            for capability in capabilities:
                task = asyncio.create_task(
                    self._execute_agent_capability(agent_id, capability, parameters)
                )
                tasks.append((f"{agent_id}_{capability}", task))
        
        # Wait for all tasks to complete
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
    
    async def _execute_delegation(self, collaboration_plan: CollaborationPlan) -> Dict[str, Any]:
        """Execute by delegating to a single agent."""
        if not collaboration_plan.steps:
            raise ValueError("No steps in delegation plan")
        
        step = collaboration_plan.steps[0]
        agent_id = step["agent_id"]
        capabilities = step["capabilities"]
        parameters = step["parameters"]
        
        results = {}
        for capability in capabilities:
            response = await self._execute_agent_capability(agent_id, capability, parameters)
            results[f"{agent_id}_{capability}"] = response
        
        return results
    
    async def _execute_agent_capability(self, 
                                      agent_id: str, 
                                      capability: str, 
                                      parameters: Dict[str, Any]) -> AgentResponse:
        """Execute a specific capability on an agent."""
        try:
            response = await self.send_request(
                agent_id,
                capability,
                parameters,
                timeout=timedelta(seconds=60)
            )
            
            # Update agent load
            self._agent_load[agent_id] = self._agent_load.get(agent_id, 0) + 1
            
            return response
            
        except Exception as e:
            self.logger.error(f"Capability execution failed for {agent_id}.{capability}: {e}")
            raise
    
    async def _aggregate_results(self, 
                               user_request: UserRequest, 
                               results: Dict[str, Any]) -> str:
        """Aggregate results from multiple agents into a final response."""
        # This would integrate with HandoverManager for intelligent aggregation
        
        successful_results = []
        failed_results = []
        
        for key, result in results.items():
            if isinstance(result, AgentResponse):
                if result.success:
                    successful_results.append(result)
                else:
                    failed_results.append(result)
        
        if not successful_results and failed_results:
            return f"Request failed: {', '.join([r.error for r in failed_results if r.error])}"
        
        # Simple aggregation - in practice this would be more sophisticated
        if len(successful_results) == 1:
            result = successful_results[0]
            if result.data and 'message' in result.data:
                return result.data['message']
            else:
                return "Task completed successfully"
        else:
            return f"Task completed with {len(successful_results)} successful operations"
    
    async def _handle_user_request(self, request: AgentRequest) -> AgentResponse:
        """Handle direct user request processing."""
        try:
            user_input = request.parameters.get("user_input", "")
            context_data = request.parameters.get("context", {})
            
            # Create context object if provided
            context = None
            if context_data:
                context = Context(**context_data)
            
            # Create user request
            user_request = UserRequest(
                user_input=user_input,
                context=context
            )
            
            # Process the request
            self._active_requests[user_request.request_id] = user_request
            await self._process_user_request(user_request)
            
            # Return response
            if user_request.status == RequestStatus.COMPLETED:
                return AgentResponse(
                    request_id=request.request_id,
                    responding_agent=self.agent_id,
                    success=True,
                    data={
                        "response": user_request.final_response,
                        "request_id": user_request.request_id
                    }
                )
            else:
                return AgentResponse(
                    request_id=request.request_id,
                    responding_agent=self.agent_id,
                    success=False,
                    error=user_request.error or "Processing failed"
                )
                
        except Exception as e:
            return AgentResponse(
                request_id=request.request_id,
                responding_agent=self.agent_id,
                success=False,
                error=str(e)
            )
    
    async def _handle_coordination_request(self, request: AgentRequest) -> AgentResponse:
        """Handle agent coordination requests."""
        # Implementation for coordination capabilities
        return AgentResponse(
            request_id=request.request_id,
            responding_agent=self.agent_id,
            success=True,
            data={"message": "Coordination capability not yet implemented"}
        )
    
    async def _handle_status_request(self, request: AgentRequest) -> AgentResponse:
        """Handle system status requests."""
        try:
            overview = await self.get_system_overview()
            return AgentResponse(
                request_id=request.request_id,
                responding_agent=self.agent_id,
                success=True,
                data=overview
            )
        except Exception as e:
            return AgentResponse(
                request_id=request.request_id,
                responding_agent=self.agent_id,
                success=False,
                error=str(e)
            )
    
    async def _handle_registry_request(self, request: AgentRequest) -> AgentResponse:
        """Handle agent registry management requests."""
        # Implementation for registry management
        return AgentResponse(
            request_id=request.request_id,
            responding_agent=self.agent_id,
            success=True,
            data={"message": "Registry management capability not yet implemented"}
        )
    
    async def _handle_agent_registration(self, event) -> None:
        """Handle agent registration events."""
        try:
            agent_data = event.data
            agent_id = agent_data.get("agent_id")
            agent_type = agent_data.get("agent_type")
            capabilities = agent_data.get("capabilities", [])
            
            if agent_id and agent_id != self.agent_id:  # Don't register ourselves
                self._registered_agents[agent_id] = AgentStatus.IDLE
                self._agent_load[agent_id] = 0
                
                # Get capabilities from the agent
                try:
                    response = await self.send_request(
                        agent_id,
                        "get_capabilities",
                        {},
                        timeout=timedelta(seconds=5)
                    )
                    
                    if response.success and response.data:
                        caps = response.data.get("capabilities", [])
                        self._agent_capabilities[agent_id] = [
                            AgentCapability(**cap) for cap in caps
                        ]
                    
                except Exception as e:
                    self.logger.warning(f"Could not get capabilities from {agent_id}: {e}")
                    self._agent_capabilities[agent_id] = []
                
                self.logger.info(f"Registered agent: {agent_id} ({agent_type})")
        
        except Exception as e:
            self.logger.error(f"Error handling agent registration: {e}")
    
    async def _handle_agent_deregistration(self, event) -> None:
        """Handle agent deregistration events."""
        try:
            agent_data = event.data
            agent_id = agent_data.get("agent_id")
            
            if agent_id in self._registered_agents:
                del self._registered_agents[agent_id]
                if agent_id in self._agent_capabilities:
                    del self._agent_capabilities[agent_id]
                if agent_id in self._agent_load:
                    del self._agent_load[agent_id]
                
                self.logger.info(f"Deregistered agent: {agent_id}")
        
        except Exception as e:
            self.logger.error(f"Error handling agent deregistration: {e}")
    
    async def _monitor_agents(self) -> None:
        """Monitor agent health and status."""
        while not self._shutdown_requested:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check agent health
                for agent_id in list(self._registered_agents.keys()):
                    try:
                        # Ping agent
                        response = await self.send_request(
                            agent_id,
                            "health_check",
                            {},
                            timeout=timedelta(seconds=5)
                        )
                        
                        if response.success:
                            self._registered_agents[agent_id] = AgentStatus.IDLE
                        else:
                            self.logger.warning(f"Agent {agent_id} health check failed")
                            
                    except Exception as e:
                        self.logger.warning(f"Lost contact with agent {agent_id}: {e}")
                        # Mark agent as unavailable but don't remove yet
                        self._registered_agents[agent_id] = AgentStatus.ERROR
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._handle_error(e, "agent_monitoring")
    
    async def _handle_voice_command(self, event: AgentEvent) -> None:
        """Handle voice command events."""
        try:
            voice_command_data = event.data.get("voice_command", {})
            
            user_request = UserRequest(
                user_input=voice_command_data.get("text", ""),
                context=Context(**voice_command_data.get("context", {})) if voice_command_data.get("context") else None
            )
            
            self._active_requests[user_request.request_id] = user_request
            await self._request_queue.put(user_request)
            
        except Exception as e:
            await self._handle_error(e, "voice_command_handling")
    
    async def _handle_text_input(self, event: AgentEvent) -> None:
        """Handle text input events."""
        try:
            text_input = event.data.get("text", "")
            context_data = event.data.get("context", {})
            
            user_request = UserRequest(
                user_input=text_input,
                context=Context(**context_data) if context_data else None
            )
            
            self._active_requests[user_request.request_id] = user_request
            await self._request_queue.put(user_request)
            
        except Exception as e:
            await self._handle_error(e, "text_input_handling")
    
    async def _handle_agent_status_update(self, event: AgentEvent) -> None:
        """Handle agent status update events."""
        try:
            agent_id = event.source_agent
            status_data = event.data.get("status", {})
            
            if agent_id in self._registered_agents:
                new_status = AgentStatus(status_data.get("status", "idle"))
                self._registered_agents[agent_id] = new_status
                
                # Update load information
                load = status_data.get("load", 0)
                self._agent_load[agent_id] = load
            
        except Exception as e:
            await self._handle_error(e, "agent_status_update")
    
    async def _handle_collaboration_event(self, event: AgentEvent) -> None:
        """Handle collaboration-related events."""
        try:
            # Handle collaboration lifecycle events
            if event.event_type == "collaboration.started":
                plan_id = event.data.get("plan_id")
                self.logger.info(f"Collaboration {plan_id} started")
            elif event.event_type == "collaboration.completed":
                plan_id = event.data.get("plan_id")
                self.logger.info(f"Collaboration {plan_id} completed")
            elif event.event_type == "collaboration.failed":
                plan_id = event.data.get("plan_id")
                error = event.data.get("error", "Unknown error")
                self.logger.error(f"Collaboration {plan_id} failed: {error}")
            
        except Exception as e:
            await self._handle_error(e, "collaboration_event_handling")
    
    async def _create_collaboration_plan(self, 
                                       task_description: str,
                                       required_capabilities: List[str],
                                       context: Optional[Context] = None) -> CollaborationPlan:
        """Create a collaboration plan for a complex task."""
        # Find suitable agents
        agent_assignments = await self._assign_agents(required_capabilities)
        
        if not agent_assignments:
            raise ValueError("No agents available for required capabilities")
        
        # Create the plan
        collaboration_plan = CollaborationPlan(
            initiator=self.agent_id,
            participants=list(agent_assignments.keys()),
            pattern=self._determine_collaboration_pattern(agent_assignments),
            steps=[
                {
                    "step_id": str(uuid.uuid4()),
                    "agent_id": agent_id,
                    "capabilities": capabilities,
                    "parameters": {
                        "task_description": task_description,
                        "context": context.dict() if context else {}
                    },
                    "dependencies": []
                }
                for agent_id, capabilities in agent_assignments.items()
            ]
        )
        
        return collaboration_plan