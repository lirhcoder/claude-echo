"""Layer Interfaces - Standardized 4-Layer Architecture Contracts

This module defines the interfaces for the 4-layer architecture:
1. User Interface Layer - Handles user input/output
2. Intelligence Hub Layer - Core AI decision making 
3. Adapter Layer - System integration abstractions
4. Execution Layer - Direct system operations

Each layer has standardized interfaces for communication and data flow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
import asyncio
from datetime import datetime

from .types import (
    Context, Intent, ExecutionPlan, Task, ExecutionResult, 
    CommandResult, LayerMessage
)
from .event_system import EventSystem, Event


class BaseLayer(ABC):
    """Base class for all architecture layers"""
    
    def __init__(self, 
                 layer_name: str,
                 event_system: Optional[EventSystem] = None):
        """
        Initialize base layer.
        
        Args:
            layer_name: Name of this layer
            event_system: Event system for inter-layer communication
        """
        self.layer_name = layer_name
        self.event_system = event_system
        self._initialized = False
        self._context: Optional[Context] = None
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the layer and its components."""
        pass
    
    @abstractmethod 
    async def shutdown(self) -> None:
        """Shutdown the layer and cleanup resources."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the layer is healthy and operational."""
        pass
    
    async def set_context(self, context: Context) -> None:
        """Set the current context for this layer."""
        self._context = context
        
        if self.event_system:
            await self.event_system.emit(Event(
                event_type=f"{self.layer_name}.context_updated",
                data={"context": context.dict()},
                source=self.layer_name
            ))
    
    def get_context(self) -> Optional[Context]:
        """Get the current context."""
        return self._context
    
    async def send_message(self, 
                          target_layer: str, 
                          message_type: str,
                          payload: Dict[str, Any]) -> None:
        """
        Send a message to another layer.
        
        Args:
            target_layer: Target layer name
            message_type: Type of message
            payload: Message payload
        """
        if self.event_system:
            message = LayerMessage(
                source_layer=self.layer_name,
                target_layer=target_layer,
                message_type=message_type,
                payload=payload
            )
            
            await self.event_system.emit(Event(
                event_type=f"layer.message.{target_layer}",
                data=message.dict(),
                source=self.layer_name
            ))


class UserInterfaceLayer(BaseLayer):
    """
    User Interface Layer - Handles all user interactions
    
    Responsibilities:
    - Voice input/output processing
    - Command line interface
    - User presence detection
    - Input validation and sanitization
    """
    
    def __init__(self, event_system: Optional[EventSystem] = None):
        super().__init__("user_interface", event_system)
    
    @abstractmethod
    async def listen_for_input(self) -> AsyncGenerator[str, None]:
        """
        Listen for user input (voice or text).
        
        Yields:
            User input strings
        """
        pass
    
    @abstractmethod
    async def provide_output(self, 
                           message: str, 
                           output_type: str = "both") -> None:
        """
        Provide output to user.
        
        Args:
            message: Message to output
            output_type: "voice", "text", or "both"
        """
        pass
    
    @abstractmethod
    async def get_user_confirmation(self, 
                                  prompt: str, 
                                  timeout: float = 30.0) -> bool:
        """
        Get user confirmation for an action.
        
        Args:
            prompt: Confirmation prompt
            timeout: Timeout in seconds
            
        Returns:
            True if user confirmed, False otherwise
        """
        pass
    
    @abstractmethod
    async def is_user_present(self) -> bool:
        """
        Check if user is currently present.
        
        Returns:
            True if user is detected as present
        """
        pass
    
    @abstractmethod
    async def get_user_attention(self, urgency: str = "normal") -> bool:
        """
        Get user's attention.
        
        Args:
            urgency: "low", "normal", "high", "critical"
            
        Returns:
            True if user responded
        """
        pass
    
    async def process_input(self, raw_input: str) -> Intent:
        """
        Process raw input into structured intent.
        
        Args:
            raw_input: Raw user input
            
        Returns:
            Structured intent object
        """
        # Basic intent parsing - would be enhanced with NLP
        intent = Intent(
            user_input=raw_input,
            intent_type="command",  # Would be determined by NLP
            confidence=1.0,
            context=self._context.dict() if self._context else {}
        )
        
        if self.event_system:
            await self.event_system.emit(Event(
                event_type="ui.intent_parsed",
                data={"intent": intent.dict()},
                source=self.layer_name
            ))
        
        return intent
    
    async def present_results(self, results: ExecutionResult) -> None:
        """
        Present execution results to user.
        
        Args:
            results: Results to present
        """
        if results.overall_success:
            message = f"Task completed successfully in {results.total_execution_time:.2f}s"
        else:
            message = f"Task failed. Errors: {'; '.join(results.errors)}"
        
        await self.provide_output(message)


class IntelligenceHubLayer(BaseLayer):
    """
    Intelligence Hub Layer - Core AI decision making
    
    Responsibilities:
    - Intent understanding and planning
    - Task decomposition and orchestration
    - Security and risk assessment
    - Context management and learning
    """
    
    def __init__(self, event_system: Optional[EventSystem] = None):
        super().__init__("intelligence_hub", event_system)
    
    @abstractmethod
    async def understand_intent(self, intent: Intent) -> Intent:
        """
        Enhance intent understanding with AI analysis.
        
        Args:
            intent: Basic intent from UI layer
            
        Returns:
            Enhanced intent with AI analysis
        """
        pass
    
    @abstractmethod
    async def create_execution_plan(self, intent: Intent) -> ExecutionPlan:
        """
        Create detailed execution plan from intent.
        
        Args:
            intent: User intent to plan for
            
        Returns:
            Detailed execution plan
        """
        pass
    
    @abstractmethod
    async def assess_risk(self, plan: ExecutionPlan) -> ExecutionPlan:
        """
        Assess and update risk levels for execution plan.
        
        Args:
            plan: Execution plan to assess
            
        Returns:
            Plan with updated risk assessments
        """
        pass
    
    @abstractmethod
    async def coordinate_execution(self, plan: ExecutionPlan) -> ExecutionResult:
        """
        Coordinate the execution of a plan.
        
        Args:
            plan: Plan to execute
            
        Returns:
            Execution results
        """
        pass
    
    @abstractmethod
    async def handle_interruption(self, 
                                new_intent: Intent, 
                                current_plan: Optional[ExecutionPlan]) -> ExecutionPlan:
        """
        Handle user interruption during execution.
        
        Args:
            new_intent: New user intent
            current_plan: Currently executing plan (if any)
            
        Returns:
            Updated execution plan
        """
        pass
    
    @abstractmethod
    async def learn_from_execution(self, 
                                 plan: ExecutionPlan, 
                                 results: ExecutionResult) -> None:
        """
        Learn from execution results to improve future planning.
        
        Args:
            plan: Executed plan
            results: Execution results
        """
        pass
    
    async def should_require_confirmation(self, plan: ExecutionPlan) -> bool:
        """
        Determine if plan requires user confirmation.
        
        Args:
            plan: Execution plan to check
            
        Returns:
            True if confirmation is required
        """
        # Check for high-risk operations
        high_risk_tasks = [
            task for task in plan.tasks 
            if task.risk_level.value in ["high", "critical"]
        ]
        
        return len(high_risk_tasks) > 0 or plan.requires_user_confirmation


class AdapterLayer(BaseLayer):
    """
    Adapter Layer - System integration abstractions
    
    Responsibilities:
    - Adapter management and lifecycle
    - Command routing to appropriate adapters
    - Adapter health monitoring
    - Result aggregation and transformation
    """
    
    def __init__(self, event_system: Optional[EventSystem] = None):
        super().__init__("adapter", event_system)
    
    @abstractmethod
    async def discover_adapters(self) -> List[str]:
        """
        Discover available adapters.
        
        Returns:
            List of discovered adapter IDs
        """
        pass
    
    @abstractmethod
    async def execute_task(self, 
                          task: Task, 
                          context: Optional[Context] = None) -> CommandResult:
        """
        Execute a task using appropriate adapter.
        
        Args:
            task: Task to execute
            context: Execution context
            
        Returns:
            Command execution result
        """
        pass
    
    @abstractmethod
    async def get_adapter_capabilities(self, 
                                     adapter_id: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Get capabilities of adapters.
        
        Args:
            adapter_id: Specific adapter ID, or None for all adapters
            
        Returns:
            Dictionary mapping adapter IDs to their supported commands
        """
        pass
    
    @abstractmethod
    async def validate_task(self, task: Task) -> bool:
        """
        Validate that a task can be executed.
        
        Args:
            task: Task to validate
            
        Returns:
            True if task is valid and executable
        """
        pass
    
    @abstractmethod
    async def estimate_execution_time(self, task: Task) -> float:
        """
        Estimate execution time for a task.
        
        Args:
            task: Task to estimate
            
        Returns:
            Estimated execution time in seconds
        """
        pass
    
    async def batch_execute_tasks(self, 
                                tasks: List[Task],
                                context: Optional[Context] = None) -> List[CommandResult]:
        """
        Execute multiple tasks with optimal scheduling.
        
        Args:
            tasks: List of tasks to execute
            context: Execution context
            
        Returns:
            List of execution results
        """
        results = []
        
        # Simple sequential execution - could be enhanced with parallelization
        for task in tasks:
            try:
                result = await self.execute_task(task, context)
                results.append(result)
                
                if self.event_system:
                    await self.event_system.emit(Event(
                        event_type="adapter.task_completed",
                        data={"task_id": task.task_id, "success": result.success},
                        source=self.layer_name
                    ))
                    
            except Exception as e:
                error_result = CommandResult(
                    success=False,
                    error=str(e),
                    metadata={"task_id": task.task_id}
                )
                results.append(error_result)
        
        return results


class ExecutionLayer(BaseLayer):
    """
    Execution Layer - Direct system operations
    
    Responsibilities:
    - Direct system API calls
    - File system operations
    - Process management
    - Network operations
    - Hardware interaction
    """
    
    def __init__(self, event_system: Optional[EventSystem] = None):
        super().__init__("execution", event_system)
    
    @abstractmethod
    async def execute_system_command(self, 
                                   command: str, 
                                   parameters: Dict[str, Any]) -> CommandResult:
        """
        Execute a system-level command.
        
        Args:
            command: Command to execute
            parameters: Command parameters
            
        Returns:
            Execution result
        """
        pass
    
    @abstractmethod
    async def access_file_system(self, 
                               operation: str,
                               path: str,
                               **kwargs) -> CommandResult:
        """
        Perform file system operations.
        
        Args:
            operation: Operation type ("read", "write", "delete", etc.)
            path: File/directory path
            **kwargs: Additional operation parameters
            
        Returns:
            Operation result
        """
        pass
    
    @abstractmethod
    async def manage_process(self, 
                           operation: str,
                           process_info: Dict[str, Any]) -> CommandResult:
        """
        Manage system processes.
        
        Args:
            operation: Operation type ("start", "stop", "kill", etc.)
            process_info: Process information
            
        Returns:
            Operation result
        """
        pass
    
    @abstractmethod
    async def network_operation(self, 
                              operation: str,
                              target: str,
                              **kwargs) -> CommandResult:
        """
        Perform network operations.
        
        Args:
            operation: Operation type ("http_request", "ping", etc.)
            target: Target URL/host
            **kwargs: Additional parameters
            
        Returns:
            Operation result
        """
        pass
    
    @abstractmethod
    async def get_system_info(self, info_type: str) -> Dict[str, Any]:
        """
        Get system information.
        
        Args:
            info_type: Type of information to retrieve
            
        Returns:
            System information dictionary
        """
        pass
    
    async def execute_with_safety_check(self, 
                                      operation: str,
                                      parameters: Dict[str, Any]) -> CommandResult:
        """
        Execute operation with safety checks.
        
        Args:
            operation: Operation to execute
            parameters: Operation parameters
            
        Returns:
            Execution result
        """
        # Basic safety checks - would be enhanced with more sophisticated logic
        dangerous_operations = ["format", "delete_all", "shutdown_system"]
        
        if operation in dangerous_operations:
            return CommandResult(
                success=False,
                error=f"Operation '{operation}' requires explicit authorization"
            )
        
        # Execute the operation based on type
        if operation.startswith("file_"):
            return await self.access_file_system(
                operation.replace("file_", ""),
                parameters.get("path", ""),
                **{k: v for k, v in parameters.items() if k != "path"}
            )
        elif operation.startswith("process_"):
            return await self.manage_process(
                operation.replace("process_", ""),
                parameters
            )
        elif operation.startswith("network_"):
            return await self.network_operation(
                operation.replace("network_", ""),
                parameters.get("target", ""),
                **{k: v for k, v in parameters.items() if k != "target"}
            )
        else:
            return await self.execute_system_command(operation, parameters)


class LayerFactory:
    """Factory for creating layer instances"""
    
    @staticmethod
    def create_layer(layer_type: str, 
                    event_system: Optional[EventSystem] = None,
                    **kwargs) -> BaseLayer:
        """
        Create a layer instance.
        
        Args:
            layer_type: Type of layer to create
            event_system: Event system instance
            **kwargs: Additional layer-specific parameters
            
        Returns:
            Layer instance
            
        Raises:
            ValueError: If layer_type is not supported
        """
        layer_classes = {
            "user_interface": UserInterfaceLayer,
            "intelligence_hub": IntelligenceHubLayer,
            "adapter": AdapterLayer,
            "execution": ExecutionLayer
        }
        
        if layer_type not in layer_classes:
            raise ValueError(f"Unknown layer type: {layer_type}")
        
        return layer_classes[layer_type](event_system, **kwargs)


class LayerError(Exception):
    """Base exception for layer-related errors"""
    pass


class LayerInitializationError(LayerError):
    """Raised when layer initialization fails"""
    pass


class LayerCommunicationError(LayerError):
    """Raised when inter-layer communication fails"""
    pass


class LayerExecutionError(LayerError):
    """Raised when layer operation execution fails"""
    pass