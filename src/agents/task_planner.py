"""Task Planner Agent - Intelligent Task Decomposition and Planning

The Task Planner is responsible for:
- Parsing user intents from natural language
- Decomposing complex tasks into actionable steps
- Optimizing execution sequences and resource allocation
- Estimating execution times and resource requirements
- Creating detailed execution plans for other agents
"""

import asyncio
import re
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json

from loguru import logger

from ..core.event_system import EventSystem
from ..core.types import Intent, Task, ExecutionPlan, Context, RiskLevel, TaskStatus
from .base_agent import BaseAgent
from .agent_types import (
    AgentType, AgentRequest, AgentResponse, AgentEvent,
    AgentCapability, CollaborationPattern
)


class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class IntentCategory(Enum):
    """Categories of user intents"""
    FILE_OPERATION = "file_operation"
    SYSTEM_COMMAND = "system_command"
    APPLICATION_CONTROL = "application_control"
    INFORMATION_QUERY = "information_query"
    COMMUNICATION = "communication"
    SCHEDULING = "scheduling"
    AUTOMATION = "automation"
    UNKNOWN = "unknown"


@dataclass
class TaskTemplate:
    """Template for common task patterns"""
    name: str
    description: str
    intent_patterns: List[str]
    required_capabilities: List[str]
    estimated_duration: timedelta
    risk_level: RiskLevel
    complexity: TaskComplexity
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionStrategy:
    """Strategy for executing a set of tasks"""
    strategy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    pattern: CollaborationPattern
    estimated_time: timedelta
    resource_requirements: Dict[str, int] = field(default_factory=dict)
    success_probability: float = 1.0
    risk_score: float = 0.0


class TaskPlanner(BaseAgent):
    """
    Intelligent task planning and decomposition agent.
    
    Analyzes user input, creates execution plans, and optimizes
    task sequences for efficient multi-agent execution.
    """
    
    def __init__(self, event_system: EventSystem, config: Optional[Dict[str, Any]] = None):
        super().__init__("task_planner", event_system, config)
        
        # Intent recognition patterns
        self._intent_patterns = self._load_intent_patterns()
        
        # Task templates for common operations
        self._task_templates = self._load_task_templates()
        
        # Capability mappings
        self._capability_registry: Dict[str, List[str]] = {}
        
        # Planning history for learning
        self._planning_history: List[Dict[str, Any]] = []
        
        # Performance metrics
        self._plans_created = 0
        self._successful_plans = 0
        self._average_planning_time = 0.0
    
    @property
    def agent_type(self) -> AgentType:
        return AgentType.TASK_PLANNER
    
    @property
    def name(self) -> str:
        return "Task Planner"
    
    @property
    def description(self) -> str:
        return "Intelligent task decomposition and execution planning agent"
    
    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="parse_intent",
                description="Parse user input and extract structured intent",
                input_types=["text", "voice_command"],
                output_types=["intent"],
                risk_level=RiskLevel.LOW,
                execution_time_ms=100
            ),
            AgentCapability(
                name="create_execution_plan",
                description="Create detailed execution plan from intent",
                input_types=["intent", "context"],
                output_types=["execution_plan"],
                risk_level=RiskLevel.MEDIUM,
                execution_time_ms=500
            ),
            AgentCapability(
                name="optimize_plan",
                description="Optimize existing execution plan for efficiency",
                input_types=["execution_plan"],
                output_types=["execution_plan"],
                risk_level=RiskLevel.LOW,
                execution_time_ms=200
            ),
            AgentCapability(
                name="estimate_resources",
                description="Estimate resource requirements for tasks",
                input_types=["execution_plan", "task_list"],
                output_types=["resource_estimate"],
                risk_level=RiskLevel.LOW,
                execution_time_ms=150
            ),
            AgentCapability(
                name="decompose_task",
                description="Break down complex task into subtasks",
                input_types=["task", "context"],
                output_types=["task_list"],
                risk_level=RiskLevel.LOW,
                execution_time_ms=300
            )
        ]
    
    async def _initialize_agent(self) -> None:
        """Initialize task planner specific functionality."""
        self.logger.info("Initializing Task Planner agent")
        
        # Load configuration
        await self._load_configuration()
        
        # Initialize ML models (if available)
        await self._initialize_models()
        
        # Start learning process
        learning_task = asyncio.create_task(self._learning_loop())
        self._background_tasks.add(learning_task)
        
        self.logger.info("Task Planner initialization complete")
    
    async def _process_request(self, request: AgentRequest) -> AgentResponse:
        """Process incoming requests."""
        capability = request.target_capability
        start_time = datetime.now()
        
        try:
            if capability == "parse_intent":
                result = await self._parse_intent(request.parameters)
            elif capability == "create_execution_plan":
                result = await self._create_execution_plan(request.parameters)
            elif capability == "optimize_plan":
                result = await self._optimize_plan(request.parameters)
            elif capability == "estimate_resources":
                result = await self._estimate_resources(request.parameters)
            elif capability == "decompose_task":
                result = await self._decompose_task(request.parameters)
            else:
                raise ValueError(f"Unknown capability: {capability}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                request_id=request.request_id,
                responding_agent=self.agent_id,
                success=True,
                data=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Error processing {capability}: {e}")
            return AgentResponse(
                request_id=request.request_id,
                responding_agent=self.agent_id,
                success=False,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def _handle_event(self, event: AgentEvent) -> None:
        """Handle agent events."""
        try:
            if event.event_type == "plan.execution.completed":
                await self._handle_plan_completion(event)
            elif event.event_type == "plan.execution.failed":
                await self._handle_plan_failure(event)
            elif event.event_type == "capability.registered":
                await self._handle_capability_registration(event)
                
        except Exception as e:
            await self._handle_error(e, f"event_handling_{event.event_type}")
    
    async def _cleanup_agent(self) -> None:
        """Cleanup task planner resources."""
        # Save learning data
        await self._save_learning_data()
        self.logger.info("Task Planner cleanup complete")
    
    # Public API methods
    
    async def parse_user_input(self, user_input: str, context: Optional[Context] = None) -> Intent:
        """
        Parse user input into structured intent.
        
        Args:
            user_input: Natural language user input
            context: Optional context information
            
        Returns:
            Parsed intent
        """
        return await self._parse_natural_language(user_input, context)
    
    async def create_plan_from_intent(self, intent: Intent, context: Optional[Context] = None) -> ExecutionPlan:
        """
        Create execution plan from intent.
        
        Args:
            intent: Parsed user intent
            context: Optional context information
            
        Returns:
            Detailed execution plan
        """
        return await self._generate_execution_plan(intent, context)
    
    async def analyze_task_complexity(self, task_description: str) -> TaskComplexity:
        """
        Analyze the complexity of a task.
        
        Args:
            task_description: Description of the task
            
        Returns:
            Task complexity level
        """
        # Simple heuristic-based complexity analysis
        word_count = len(task_description.split())
        
        # Check for complexity indicators
        complex_keywords = ['multiple', 'several', 'batch', 'all', 'every', 'schedule', 'automate']
        moderate_keywords = ['find', 'search', 'compare', 'analyze', 'process']
        
        complexity_score = 0
        
        # Word count factor
        if word_count > 20:
            complexity_score += 2
        elif word_count > 10:
            complexity_score += 1
        
        # Keyword analysis
        task_lower = task_description.lower()
        for keyword in complex_keywords:
            if keyword in task_lower:
                complexity_score += 2
        
        for keyword in moderate_keywords:
            if keyword in task_lower:
                complexity_score += 1
        
        # Check for conditional statements
        if any(word in task_lower for word in ['if', 'when', 'unless', 'until']):
            complexity_score += 1
        
        # Map score to complexity
        if complexity_score >= 4:
            return TaskComplexity.VERY_COMPLEX
        elif complexity_score >= 3:
            return TaskComplexity.COMPLEX
        elif complexity_score >= 1:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE
    
    # Private implementation methods
    
    async def _parse_intent(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Parse intent from request parameters."""
        user_input = parameters.get("user_input", "")
        context_data = parameters.get("context", {})
        
        # Create context object
        context = None
        if context_data:
            try:
                context = Context(**context_data)
            except Exception as e:
                self.logger.warning(f"Could not create context: {e}")
        
        # Parse intent
        intent = await self._parse_natural_language(user_input, context)
        
        return {
            "intent": intent.dict()
        }
    
    async def _create_execution_plan(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create execution plan from parameters."""
        intent_data = parameters.get("intent")
        context_data = parameters.get("context", {})
        
        if not intent_data:
            raise ValueError("Intent data required for execution plan creation")
        
        # Create objects
        intent = Intent(**intent_data)
        context = Context(**context_data) if context_data else None
        
        # Generate plan
        plan = await self._generate_execution_plan(intent, context)
        
        self._plans_created += 1
        
        return {
            "execution_plan": plan.dict()
        }
    
    async def _optimize_plan(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize an existing execution plan."""
        plan_data = parameters.get("execution_plan")
        
        if not plan_data:
            raise ValueError("Execution plan data required")
        
        plan = ExecutionPlan(**plan_data)
        optimized_plan = await self._optimize_execution_plan(plan)
        
        return {
            "execution_plan": optimized_plan.dict()
        }
    
    async def _estimate_resources(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate resource requirements."""
        plan_data = parameters.get("execution_plan")
        task_list_data = parameters.get("task_list", [])
        
        if plan_data:
            plan = ExecutionPlan(**plan_data)
            tasks = plan.tasks
        else:
            tasks = [Task(**task_data) for task_data in task_list_data]
        
        if not tasks:
            raise ValueError("No tasks provided for resource estimation")
        
        estimate = await self._calculate_resource_requirements(tasks)
        
        return {
            "resource_estimate": estimate
        }
    
    async def _decompose_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose a complex task into subtasks."""
        task_data = parameters.get("task")
        context_data = parameters.get("context", {})
        
        if not task_data:
            raise ValueError("Task data required for decomposition")
        
        task = Task(**task_data)
        context = Context(**context_data) if context_data else None
        
        subtasks = await self._break_down_task(task, context)
        
        return {
            "task_list": [task.dict() for task in subtasks]
        }
    
    async def _parse_natural_language(self, user_input: str, context: Optional[Context] = None) -> Intent:
        """Parse natural language input into structured intent."""
        user_input = user_input.strip()
        
        if not user_input:
            return Intent(
                user_input="",
                intent_type="empty",
                confidence=0.0
            )
        
        # Determine intent category
        intent_category = self._classify_intent(user_input)
        
        # Extract parameters
        parameters = await self._extract_parameters(user_input, intent_category)
        
        # Calculate confidence
        confidence = self._calculate_confidence(user_input, intent_category, parameters)
        
        # Create intent
        intent = Intent(
            user_input=user_input,
            intent_type=intent_category.value,
            confidence=confidence,
            parameters=parameters,
            context=context.dict() if context else {}
        )
        
        self.logger.debug(f"Parsed intent: {intent_category.value} (confidence: {confidence:.2f})")
        
        return intent
    
    def _classify_intent(self, user_input: str) -> IntentCategory:
        """Classify user input into intent categories."""
        user_input_lower = user_input.lower()
        
        # File operation patterns
        if any(word in user_input_lower for word in ['file', 'folder', 'directory', 'copy', 'move', 'delete', 'rename']):
            return IntentCategory.FILE_OPERATION
        
        # System command patterns
        if any(word in user_input_lower for word in ['run', 'execute', 'launch', 'start', 'stop', 'restart', 'install']):
            return IntentCategory.SYSTEM_COMMAND
        
        # Application control patterns
        if any(word in user_input_lower for word in ['open', 'close', 'minimize', 'maximize', 'switch', 'tab']):
            return IntentCategory.APPLICATION_CONTROL
        
        # Information query patterns
        if any(word in user_input_lower for word in ['what', 'how', 'where', 'when', 'why', 'find', 'search', 'show']):
            return IntentCategory.INFORMATION_QUERY
        
        # Communication patterns
        if any(word in user_input_lower for word in ['send', 'email', 'message', 'call', 'notify', 'alert']):
            return IntentCategory.COMMUNICATION
        
        # Scheduling patterns
        if any(word in user_input_lower for word in ['schedule', 'remind', 'appointment', 'meeting', 'calendar']):
            return IntentCategory.SCHEDULING
        
        # Automation patterns
        if any(word in user_input_lower for word in ['automate', 'repeat', 'every', 'daily', 'weekly', 'monitor']):
            return IntentCategory.AUTOMATION
        
        return IntentCategory.UNKNOWN
    
    async def _extract_parameters(self, user_input: str, intent_category: IntentCategory) -> Dict[str, Any]:
        """Extract parameters from user input based on intent category."""
        parameters = {}
        
        # Common parameter extraction
        parameters.update(self._extract_common_parameters(user_input))
        
        # Category-specific parameter extraction
        if intent_category == IntentCategory.FILE_OPERATION:
            parameters.update(self._extract_file_parameters(user_input))
        elif intent_category == IntentCategory.SYSTEM_COMMAND:
            parameters.update(self._extract_system_parameters(user_input))
        elif intent_category == IntentCategory.APPLICATION_CONTROL:
            parameters.update(self._extract_application_parameters(user_input))
        elif intent_category == IntentCategory.INFORMATION_QUERY:
            parameters.update(self._extract_query_parameters(user_input))
        elif intent_category == IntentCategory.COMMUNICATION:
            parameters.update(self._extract_communication_parameters(user_input))
        elif intent_category == IntentCategory.SCHEDULING:
            parameters.update(self._extract_scheduling_parameters(user_input))
        elif intent_category == IntentCategory.AUTOMATION:
            parameters.update(self._extract_automation_parameters(user_input))
        
        return parameters
    
    def _extract_common_parameters(self, user_input: str) -> Dict[str, Any]:
        """Extract common parameters from any input."""
        parameters = {}
        
        # Extract quoted strings (likely file names or text)
        quoted_strings = re.findall(r'"([^"]*)"', user_input)
        if quoted_strings:
            parameters['quoted_text'] = quoted_strings
        
        # Extract file paths
        file_paths = re.findall(r'[/\\]?[\w\-_./\\]+\.\w+', user_input)
        if file_paths:
            parameters['file_paths'] = file_paths
        
        # Extract numbers
        numbers = re.findall(r'\b\d+\b', user_input)
        if numbers:
            parameters['numbers'] = [int(n) for n in numbers]
        
        # Extract URLs
        urls = re.findall(r'https?://[^\s]+', user_input)
        if urls:
            parameters['urls'] = urls
        
        return parameters
    
    def _extract_file_parameters(self, user_input: str) -> Dict[str, Any]:
        """Extract file operation specific parameters."""
        parameters = {}
        
        # File operations
        if 'copy' in user_input.lower():
            parameters['operation'] = 'copy'
        elif 'move' in user_input.lower():
            parameters['operation'] = 'move'
        elif 'delete' in user_input.lower():
            parameters['operation'] = 'delete'
        elif 'rename' in user_input.lower():
            parameters['operation'] = 'rename'
        elif 'create' in user_input.lower():
            parameters['operation'] = 'create'
        
        # Source and destination detection
        prepositions = ['to', 'from', 'into', 'in']
        for prep in prepositions:
            if prep in user_input.lower():
                parts = user_input.lower().split(prep)
                if len(parts) == 2:
                    parameters['source_context'] = parts[0].strip()
                    parameters['destination_context'] = parts[1].strip()
        
        return parameters
    
    def _extract_system_parameters(self, user_input: str) -> Dict[str, Any]:
        """Extract system command parameters."""
        parameters = {}
        
        # Common system commands
        commands = {
            'run': 'execute',
            'launch': 'start',
            'kill': 'terminate',
            'restart': 'restart'
        }
        
        for word, action in commands.items():
            if word in user_input.lower():
                parameters['action'] = action
                break
        
        # Extract program names
        common_programs = ['notepad', 'calculator', 'browser', 'terminal', 'cmd', 'powershell']
        for program in common_programs:
            if program in user_input.lower():
                parameters['program'] = program
        
        return parameters
    
    def _extract_application_parameters(self, user_input: str) -> Dict[str, Any]:
        """Extract application control parameters."""
        parameters = {}
        
        # Window operations
        if 'minimize' in user_input.lower():
            parameters['window_action'] = 'minimize'
        elif 'maximize' in user_input.lower():
            parameters['window_action'] = 'maximize'
        elif 'close' in user_input.lower():
            parameters['window_action'] = 'close'
        elif 'open' in user_input.lower():
            parameters['window_action'] = 'open'
        
        # Tab operations
        if 'new tab' in user_input.lower():
            parameters['tab_action'] = 'new'
        elif 'close tab' in user_input.lower():
            parameters['tab_action'] = 'close'
        elif 'switch tab' in user_input.lower():
            parameters['tab_action'] = 'switch'
        
        return parameters
    
    def _extract_query_parameters(self, user_input: str) -> Dict[str, Any]:
        """Extract information query parameters."""
        parameters = {}
        
        # Query types
        if user_input.lower().startswith('what'):
            parameters['query_type'] = 'what'
        elif user_input.lower().startswith('how'):
            parameters['query_type'] = 'how'
        elif user_input.lower().startswith('where'):
            parameters['query_type'] = 'where'
        elif user_input.lower().startswith('when'):
            parameters['query_type'] = 'when'
        elif user_input.lower().startswith('why'):
            parameters['query_type'] = 'why'
        
        # Search terms (everything after question words)
        question_words = ['what', 'how', 'where', 'when', 'why', 'find', 'search', 'show']
        for word in question_words:
            if word in user_input.lower():
                parts = user_input.lower().split(word, 1)
                if len(parts) == 2:
                    parameters['search_terms'] = parts[1].strip()
                break
        
        return parameters
    
    def _extract_communication_parameters(self, user_input: str) -> Dict[str, Any]:
        """Extract communication parameters."""
        parameters = {}
        
        # Communication types
        if 'email' in user_input.lower():
            parameters['communication_type'] = 'email'
        elif 'message' in user_input.lower():
            parameters['communication_type'] = 'message'
        elif 'call' in user_input.lower():
            parameters['communication_type'] = 'call'
        elif 'notify' in user_input.lower():
            parameters['communication_type'] = 'notification'
        
        # Extract email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', user_input)
        if emails:
            parameters['recipients'] = emails
        
        return parameters
    
    def _extract_scheduling_parameters(self, user_input: str) -> Dict[str, Any]:
        """Extract scheduling parameters."""
        parameters = {}
        
        # Time expressions
        time_patterns = [
            r'\b(\d{1,2}):(\d{2})\s*(am|pm)?\b',
            r'\b(tomorrow|today|yesterday)\b',
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\bin\s+(\d+)\s+(minutes?|hours?|days?|weeks?)\b'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, user_input.lower())
            if matches:
                parameters['time_expressions'] = matches
                break
        
        # Event types
        if 'meeting' in user_input.lower():
            parameters['event_type'] = 'meeting'
        elif 'appointment' in user_input.lower():
            parameters['event_type'] = 'appointment'
        elif 'reminder' in user_input.lower():
            parameters['event_type'] = 'reminder'
        
        return parameters
    
    def _extract_automation_parameters(self, user_input: str) -> Dict[str, Any]:
        """Extract automation parameters."""
        parameters = {}
        
        # Frequency expressions
        frequency_patterns = {
            'daily': 'daily',
            'weekly': 'weekly',
            'monthly': 'monthly',
            'hourly': 'hourly',
            'every day': 'daily',
            'every week': 'weekly',
            'every hour': 'hourly'
        }
        
        for pattern, frequency in frequency_patterns.items():
            if pattern in user_input.lower():
                parameters['frequency'] = frequency
                break
        
        # Trigger conditions
        if 'when' in user_input.lower():
            parts = user_input.lower().split('when')
            if len(parts) == 2:
                parameters['trigger_condition'] = parts[1].strip()
        
        return parameters
    
    def _calculate_confidence(self, user_input: str, intent_category: IntentCategory, parameters: Dict[str, Any]) -> float:
        """Calculate confidence score for intent classification."""
        confidence = 0.0
        
        # Base confidence based on category match
        if intent_category != IntentCategory.UNKNOWN:
            confidence += 0.6
        else:
            confidence += 0.3
        
        # Boost confidence based on extracted parameters
        if parameters:
            parameter_boost = min(len(parameters) * 0.1, 0.3)
            confidence += parameter_boost
        
        # Reduce confidence for very short or very long inputs
        word_count = len(user_input.split())
        if word_count < 3:
            confidence *= 0.8
        elif word_count > 20:
            confidence *= 0.9
        
        # Boost confidence for clear command structure
        if any(word in user_input.lower() for word in ['please', 'can you', 'i want', 'i need']):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    async def _generate_execution_plan(self, intent: Intent, context: Optional[Context] = None) -> ExecutionPlan:
        """Generate detailed execution plan from intent."""
        # Analyze task complexity
        complexity = await self.analyze_task_complexity(intent.user_input)
        
        # Find matching task template
        template = self._find_matching_template(intent)
        
        # Create tasks based on template or custom logic
        if template:
            tasks = await self._create_tasks_from_template(template, intent, context)
        else:
            tasks = await self._create_custom_tasks(intent, context)
        
        # Estimate execution duration
        estimated_duration = await self._estimate_total_duration(tasks)
        
        # Determine overall risk level
        risk_level = self._calculate_overall_risk(tasks)
        
        # Check if user confirmation is required
        requires_confirmation = self._requires_user_confirmation(tasks, risk_level)
        
        # Create execution plan
        plan = ExecutionPlan(
            intent=intent,
            tasks=tasks,
            estimated_duration=estimated_duration,
            risk_level=risk_level,
            requires_user_confirmation=requires_confirmation
        )
        
        # Store planning history
        self._planning_history.append({
            'plan_id': plan.plan_id,
            'intent_type': intent.intent_type,
            'complexity': complexity.value,
            'task_count': len(tasks),
            'estimated_duration': estimated_duration.total_seconds(),
            'risk_level': risk_level.value,
            'timestamp': datetime.now()
        })
        
        return plan
    
    def _find_matching_template(self, intent: Intent) -> Optional[TaskTemplate]:
        """Find a task template that matches the intent."""
        for template in self._task_templates:
            for pattern in template.intent_patterns:
                if self._pattern_matches(pattern, intent.user_input) or pattern == intent.intent_type:
                    return template
        return None
    
    def _pattern_matches(self, pattern: str, text: str) -> bool:
        """Check if a pattern matches the given text."""
        # Simple pattern matching - could be enhanced with regex or ML
        pattern_words = pattern.lower().split()
        text_words = text.lower().split()
        
        # Check if all pattern words are present
        return all(word in text_words for word in pattern_words)
    
    async def _create_tasks_from_template(self, template: TaskTemplate, intent: Intent, context: Optional[Context]) -> List[Task]:
        """Create tasks based on a template."""
        tasks = []
        
        for i, capability in enumerate(template.required_capabilities):
            task = Task(
                command=capability,
                parameters={
                    **template.parameters,
                    **intent.parameters,
                    'intent': intent.dict(),
                    'context': context.dict() if context else {}
                },
                target_adapter=self._map_capability_to_adapter(capability),
                risk_level=template.risk_level,
                dependencies=template.dependencies if i == 0 else [tasks[i-1].task_id],
                timeout=template.estimated_duration
            )
            tasks.append(task)
        
        return tasks
    
    async def _create_custom_tasks(self, intent: Intent, context: Optional[Context]) -> List[Task]:
        """Create custom tasks for intents without templates."""
        tasks = []
        
        # Determine required capabilities based on intent type
        capabilities = self._determine_capabilities_for_intent(intent.intent_type)
        
        for i, capability in enumerate(capabilities):
            task = Task(
                command=capability,
                parameters={
                    **intent.parameters,
                    'intent': intent.dict(),
                    'context': context.dict() if context else {}
                },
                target_adapter=self._map_capability_to_adapter(capability),
                risk_level=self._estimate_task_risk(capability, intent.parameters),
                dependencies=[tasks[i-1].task_id] if i > 0 else [],
                timeout=timedelta(seconds=30)  # Default timeout
            )
            tasks.append(task)
        
        return tasks
    
    def _determine_capabilities_for_intent(self, intent_type: str) -> List[str]:
        """Determine required capabilities for an intent type."""
        capability_mapping = {
            'file_operation': ['security_check', 'file_management'],
            'system_command': ['security_check', 'system_execution'],
            'application_control': ['application_management'],
            'information_query': ['information_retrieval'],
            'communication': ['security_check', 'communication'],
            'scheduling': ['calendar_management'],
            'automation': ['task_automation', 'monitoring'],
            'unknown': ['general_processing']
        }
        
        return capability_mapping.get(intent_type, ['general_processing'])
    
    def _map_capability_to_adapter(self, capability: str) -> str:
        """Map capability to appropriate adapter/agent."""
        adapter_mapping = {
            'security_check': 'security_guardian',
            'file_management': 'auto_worker',
            'system_execution': 'auto_worker',
            'application_management': 'auto_worker',
            'information_retrieval': 'auto_worker',
            'communication': 'auto_worker',
            'calendar_management': 'auto_worker',
            'task_automation': 'auto_worker',
            'monitoring': 'presence_monitor',
            'general_processing': 'auto_worker'
        }
        
        return adapter_mapping.get(capability, 'auto_worker')
    
    def _estimate_task_risk(self, capability: str, parameters: Dict[str, Any]) -> RiskLevel:
        """Estimate risk level for a task."""
        # High-risk capabilities
        if capability in ['system_execution', 'file_management']:
            return RiskLevel.HIGH
        
        # Medium-risk capabilities
        if capability in ['security_check', 'communication']:
            return RiskLevel.MEDIUM
        
        # Check parameters for risk indicators
        if any(param in str(parameters).lower() for param in ['delete', 'remove', 'format', 'install']):
            return RiskLevel.HIGH
        
        return RiskLevel.LOW
    
    async def _estimate_total_duration(self, tasks: List[Task]) -> timedelta:
        """Estimate total execution duration for all tasks."""
        total_seconds = 0
        
        for task in tasks:
            if task.timeout:
                total_seconds += task.timeout.total_seconds()
            else:
                # Default estimates based on capability
                capability_durations = {
                    'security_check': 2,
                    'file_management': 5,
                    'system_execution': 10,
                    'application_management': 3,
                    'information_retrieval': 8,
                    'communication': 5,
                    'calendar_management': 3,
                    'task_automation': 15,
                    'monitoring': 1,
                    'general_processing': 5
                }
                total_seconds += capability_durations.get(task.command, 5)
        
        # Add overhead for coordination
        total_seconds *= 1.2
        
        return timedelta(seconds=total_seconds)
    
    def _calculate_overall_risk(self, tasks: List[Task]) -> RiskLevel:
        """Calculate overall risk level for the execution plan."""
        max_risk = RiskLevel.LOW
        
        for task in tasks:
            if task.risk_level.value == RiskLevel.CRITICAL.value:
                return RiskLevel.CRITICAL
            elif task.risk_level.value == RiskLevel.HIGH.value:
                max_risk = RiskLevel.HIGH
            elif task.risk_level.value == RiskLevel.MEDIUM.value and max_risk == RiskLevel.LOW:
                max_risk = RiskLevel.MEDIUM
        
        return max_risk
    
    def _requires_user_confirmation(self, tasks: List[Task], risk_level: RiskLevel) -> bool:
        """Determine if user confirmation is required."""
        # Always require confirmation for high-risk operations
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            return True
        
        # Check for specific operations that always require confirmation
        dangerous_operations = ['delete', 'format', 'install', 'uninstall', 'modify_system']
        
        for task in tasks:
            if any(op in task.command.lower() for op in dangerous_operations):
                return True
            
            if any(op in str(task.parameters).lower() for op in dangerous_operations):
                return True
        
        return False
    
    async def _optimize_execution_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Optimize an existing execution plan."""
        # Create optimized tasks
        optimized_tasks = await self._optimize_task_sequence(plan.tasks)
        
        # Recalculate duration and risk
        estimated_duration = await self._estimate_total_duration(optimized_tasks)
        risk_level = self._calculate_overall_risk(optimized_tasks)
        
        # Create optimized plan
        optimized_plan = ExecutionPlan(
            intent=plan.intent,
            tasks=optimized_tasks,
            estimated_duration=estimated_duration,
            risk_level=risk_level,
            requires_user_confirmation=plan.requires_user_confirmation
        )
        
        return optimized_plan
    
    async def _optimize_task_sequence(self, tasks: List[Task]) -> List[Task]:
        """Optimize the sequence of tasks for better performance."""
        # Sort tasks by priority and dependencies
        optimized_tasks = []
        processed_tasks = set()
        
        # First, add all tasks with no dependencies
        for task in tasks:
            if not task.dependencies:
                optimized_tasks.append(task)
                processed_tasks.add(task.task_id)
        
        # Then add tasks whose dependencies have been processed
        remaining_tasks = [t for t in tasks if t.task_id not in processed_tasks]
        
        while remaining_tasks:
            ready_tasks = []
            
            for task in remaining_tasks:
                if all(dep_id in processed_tasks for dep_id in task.dependencies):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # Circular dependency or error - add remaining tasks as-is
                optimized_tasks.extend(remaining_tasks)
                break
            
            # Sort ready tasks by estimated execution time (shortest first)
            ready_tasks.sort(key=lambda t: t.timeout.total_seconds() if t.timeout else 5)
            
            optimized_tasks.extend(ready_tasks)
            for task in ready_tasks:
                processed_tasks.add(task.task_id)
                remaining_tasks.remove(task)
        
        return optimized_tasks
    
    async def _calculate_resource_requirements(self, tasks: List[Task]) -> Dict[str, Any]:
        """Calculate resource requirements for tasks."""
        requirements = {
            'cpu_cores': 0,
            'memory_mb': 0,
            'disk_space_mb': 0,
            'network_bandwidth': 0,
            'estimated_duration': timedelta(),
            'concurrent_tasks': 0
        }
        
        for task in tasks:
            # Estimate resources based on task type
            task_requirements = self._estimate_task_resources(task)
            
            for key, value in task_requirements.items():
                if key == 'estimated_duration':
                    requirements[key] += value
                else:
                    requirements[key] = max(requirements[key], value)
        
        # Calculate concurrent tasks
        requirements['concurrent_tasks'] = len([t for t in tasks if not t.dependencies])
        
        return requirements
    
    def _estimate_task_resources(self, task: Task) -> Dict[str, Any]:
        """Estimate resource requirements for a single task."""
        # Base requirements by task type
        resource_profiles = {
            'security_check': {'cpu_cores': 1, 'memory_mb': 50, 'disk_space_mb': 0, 'network_bandwidth': 10},
            'file_management': {'cpu_cores': 1, 'memory_mb': 100, 'disk_space_mb': 500, 'network_bandwidth': 0},
            'system_execution': {'cpu_cores': 2, 'memory_mb': 200, 'disk_space_mb': 100, 'network_bandwidth': 0},
            'application_management': {'cpu_cores': 1, 'memory_mb': 150, 'disk_space_mb': 50, 'network_bandwidth': 0},
            'information_retrieval': {'cpu_cores': 1, 'memory_mb': 100, 'disk_space_mb': 0, 'network_bandwidth': 100},
            'communication': {'cpu_cores': 1, 'memory_mb': 75, 'disk_space_mb': 10, 'network_bandwidth': 50},
        }
        
        profile = resource_profiles.get(task.command, {'cpu_cores': 1, 'memory_mb': 100, 'disk_space_mb': 50, 'network_bandwidth': 10})
        
        # Add estimated duration
        profile['estimated_duration'] = task.timeout or timedelta(seconds=5)
        
        return profile
    
    async def _break_down_task(self, task: Task, context: Optional[Context] = None) -> List[Task]:
        """Break down a complex task into simpler subtasks."""
        subtasks = []
        
        # Analyze the task complexity
        if task.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            # Add security check as first subtask
            security_task = Task(
                command="security_check",
                parameters={
                    "original_task": task.dict(),
                    "context": context.dict() if context else {}
                },
                target_adapter="security_guardian",
                risk_level=RiskLevel.MEDIUM
            )
            subtasks.append(security_task)
        
        # Break down based on command type
        if "batch" in task.command.lower() or "multiple" in str(task.parameters).lower():
            # Split batch operations
            items = task.parameters.get("items", [])
            if not items and "file_paths" in task.parameters:
                items = task.parameters["file_paths"]
            
            for item in items[:10]:  # Limit to 10 items for safety
                subtask = Task(
                    command=task.command.replace("batch_", ""),
                    parameters={
                        **task.parameters,
                        "target": item,
                        "batch_item": True
                    },
                    target_adapter=task.target_adapter,
                    risk_level=task.risk_level,
                    dependencies=[security_task.task_id] if subtasks else []
                )
                subtasks.append(subtask)
        else:
            # For non-batch operations, create validation and execution subtasks
            validation_task = Task(
                command=f"validate_{task.command}",
                parameters=task.parameters,
                target_adapter=task.target_adapter,
                risk_level=RiskLevel.LOW,
                dependencies=[subtasks[-1].task_id] if subtasks else []
            )
            subtasks.append(validation_task)
            
            execution_task = Task(
                command=task.command,
                parameters=task.parameters,
                target_adapter=task.target_adapter,
                risk_level=task.risk_level,
                dependencies=[validation_task.task_id]
            )
            subtasks.append(execution_task)
        
        return subtasks
    
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """Load intent recognition patterns."""
        return {
            "file_operation": [
                "copy file", "move file", "delete file", "rename file",
                "create folder", "backup files", "compress files"
            ],
            "system_command": [
                "run program", "start service", "stop process", "install software",
                "update system", "restart computer", "check status"
            ],
            "application_control": [
                "open application", "close window", "switch tab", "minimize window",
                "maximize window", "new document", "save file"
            ],
            "information_query": [
                "what is", "how to", "find information", "search for",
                "show me", "tell me about", "explain"
            ],
            "communication": [
                "send email", "make call", "send message", "notify",
                "schedule meeting", "create reminder"
            ],
            "scheduling": [
                "schedule task", "set reminder", "book appointment",
                "add to calendar", "plan meeting"
            ],
            "automation": [
                "automate task", "repeat action", "monitor changes",
                "daily backup", "weekly report"
            ]
        }
    
    def _load_task_templates(self) -> List[TaskTemplate]:
        """Load predefined task templates."""
        return [
            TaskTemplate(
                name="file_copy",
                description="Copy files from source to destination",
                intent_patterns=["copy", "duplicate", "backup"],
                required_capabilities=["security_check", "file_management"],
                estimated_duration=timedelta(seconds=10),
                risk_level=RiskLevel.LOW,
                complexity=TaskComplexity.SIMPLE
            ),
            TaskTemplate(
                name="system_install",
                description="Install software or system components",
                intent_patterns=["install", "setup", "add"],
                required_capabilities=["security_check", "system_execution"],
                estimated_duration=timedelta(minutes=5),
                risk_level=RiskLevel.HIGH,
                complexity=TaskComplexity.COMPLEX
            ),
            TaskTemplate(
                name="information_search",
                description="Search for information",
                intent_patterns=["find", "search", "look for"],
                required_capabilities=["information_retrieval"],
                estimated_duration=timedelta(seconds=30),
                risk_level=RiskLevel.LOW,
                complexity=TaskComplexity.MODERATE
            )
        ]
    
    async def _load_configuration(self) -> None:
        """Load task planner configuration."""
        # Load configuration from config manager or file
        pass
    
    async def _initialize_models(self) -> None:
        """Initialize ML models for intent recognition and planning."""
        # Initialize any ML models if available
        pass
    
    async def _learning_loop(self) -> None:
        """Background learning loop for improving planning accuracy."""
        while not self._shutdown_requested:
            try:
                await asyncio.sleep(300)  # Learn every 5 minutes
                
                # Analyze planning history and adjust strategies
                await self._analyze_planning_performance()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._handle_error(e, "learning_loop")
    
    async def _analyze_planning_performance(self) -> None:
        """Analyze planning performance and adjust strategies."""
        if len(self._planning_history) < 10:
            return
        
        # Calculate success rate by intent type
        intent_performance = {}
        
        for plan_data in self._planning_history[-100:]:  # Last 100 plans
            intent_type = plan_data['intent_type']
            if intent_type not in intent_performance:
                intent_performance[intent_type] = {'total': 0, 'successful': 0}
            
            intent_performance[intent_type]['total'] += 1
            # This would be updated when we receive execution feedback
        
        # Log performance insights
        self.logger.info(f"Planning performance analysis: {intent_performance}")
    
    async def _save_learning_data(self) -> None:
        """Save learning data for persistence."""
        # Save planning history and learned patterns
        try:
            # This would save to a persistent store
            self.logger.info(f"Saving {len(self._planning_history)} planning records")
        except Exception as e:
            self.logger.error(f"Failed to save learning data: {e}")
    
    async def _handle_plan_completion(self, event: AgentEvent) -> None:
        """Handle plan execution completion events."""
        try:
            plan_id = event.data.get("plan_id")
            success = event.data.get("success", False)
            
            # Update learning data
            for plan_data in self._planning_history:
                if plan_data['plan_id'] == plan_id:
                    plan_data['success'] = success
                    plan_data['completed_at'] = datetime.now()
                    break
            
            if success:
                self._successful_plans += 1
            
        except Exception as e:
            await self._handle_error(e, "plan_completion_handling")
    
    async def _handle_plan_failure(self, event: AgentEvent) -> None:
        """Handle plan execution failure events."""
        try:
            plan_id = event.data.get("plan_id")
            error = event.data.get("error", "Unknown error")
            
            # Update learning data
            for plan_data in self._planning_history:
                if plan_data['plan_id'] == plan_id:
                    plan_data['success'] = False
                    plan_data['error'] = error
                    plan_data['completed_at'] = datetime.now()
                    break
            
        except Exception as e:
            await self._handle_error(e, "plan_failure_handling")
    
    async def _handle_capability_registration(self, event: AgentEvent) -> None:
        """Handle new capability registration events."""
        try:
            agent_id = event.source_agent
            capabilities = event.data.get("capabilities", [])
            
            self._capability_registry[agent_id] = capabilities
            
            self.logger.info(f"Updated capabilities for agent {agent_id}: {len(capabilities)} capabilities")
            
        except Exception as e:
            await self._handle_error(e, "capability_registration")