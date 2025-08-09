"""Auto Worker Agent - Autonomous Task Execution

The Auto Worker is responsible for:
- Multi-task concurrent execution management
- Dynamic adapter selection and invocation
- Real-time execution monitoring and debugging
- Intelligent error handling and retry mechanisms
"""

import asyncio
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid

from loguru import logger

from ..core.event_system import EventSystem
from ..core.types import Task, ExecutionResult, RiskLevel, TaskStatus
from ..core.adapter_manager import AdapterManager
from .base_agent import BaseAgent
from .agent_types import (
    AgentType, AgentRequest, AgentResponse, AgentEvent, AgentCapability
)


class ExecutionStrategy(Enum):
    """Execution strategies for tasks"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PRIORITY_BASED = "priority_based"
    ADAPTIVE = "adaptive"


@dataclass
class ExecutionContext:
    """Context for task execution"""
    task: Task
    started_at: datetime
    adapter_id: Optional[str] = None
    retry_count: int = 0
    execution_future: Optional[asyncio.Future] = None
    error_history: List[str] = field(default_factory=list)


class AutoWorker(BaseAgent):
    """
    Autonomous task execution agent.
    
    Executes tasks using appropriate adapters with intelligent
    error handling and performance optimization.
    """
    
    def __init__(self, event_system: EventSystem, config: Optional[Dict[str, Any]] = None):
        super().__init__("auto_worker", event_system, config)
        
        # Execution management
        self._active_executions: Dict[str, ExecutionContext] = {}
        self._execution_queue: asyncio.Queue = asyncio.Queue()
        self._max_concurrent_tasks = self.config.get('max_concurrent_tasks', 5)
        
        # Adapter management
        self._adapter_manager: Optional[AdapterManager] = None
        self._available_adapters: Dict[str, Any] = {}
        
        # Performance tracking
        self._execution_stats = {
            'total_executed': 0,
            'successful': 0,
            'failed': 0,
            'retried': 0,
            'average_duration': 0.0
        }
        
        # Strategy configuration
        self._execution_strategy = ExecutionStrategy.ADAPTIVE
        self._retry_delays = [1, 2, 5, 10, 30]  # seconds
    
    @property
    def agent_type(self) -> AgentType:
        return AgentType.AUTO_WORKER
    
    @property
    def name(self) -> str:
        return "Auto Worker"
    
    @property
    def description(self) -> str:
        return "Autonomous task execution and adapter management agent"
    
    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="execute_task",
                description="Execute a single task using appropriate adapter",
                input_types=["task"],
                output_types=["execution_result"],
                risk_level=RiskLevel.MEDIUM,
                execution_time_ms=5000
            ),
            AgentCapability(
                name="execute_batch",
                description="Execute multiple tasks concurrently",
                input_types=["task_list"],
                output_types=["batch_execution_result"],
                risk_level=RiskLevel.HIGH,
                execution_time_ms=15000
            ),
            AgentCapability(
                name="get_execution_status",
                description="Get status of running executions",
                input_types=["execution_query"],
                output_types=["execution_status"],
                risk_level=RiskLevel.LOW,
                execution_time_ms=100
            ),
            AgentCapability(
                name="cancel_execution",
                description="Cancel a running task execution",
                input_types=["execution_id"],
                output_types=["cancellation_result"],
                risk_level=RiskLevel.MEDIUM,
                execution_time_ms=500
            ),
            AgentCapability(
                name="configure_execution",
                description="Configure execution parameters",
                input_types=["execution_config"],
                output_types=["config_status"],
                risk_level=RiskLevel.MEDIUM,
                execution_time_ms=200
            )
        ]
    
    async def _initialize_agent(self) -> None:
        """Initialize auto worker specific functionality."""
        self.logger.info("Initializing Auto Worker agent")
        
        # Initialize adapter manager (would be injected in practice)
        # self._adapter_manager = AdapterManager(self.event_system)
        # await self._adapter_manager.initialize()
        
        # Start execution processor
        processor_task = asyncio.create_task(self._execution_processor())
        self._background_tasks.add(processor_task)
        
        # Start monitoring task
        monitor_task = asyncio.create_task(self._execution_monitor())
        self._background_tasks.add(monitor_task)
        
        self.logger.info("Auto Worker initialization complete")
    
    async def _process_request(self, request: AgentRequest) -> AgentResponse:
        """Process incoming requests."""
        capability = request.target_capability
        start_time = datetime.now()
        
        try:
            if capability == "execute_task":
                result = await self._execute_task(request.parameters)
            elif capability == "execute_batch":
                result = await self._execute_batch(request.parameters)
            elif capability == "get_execution_status":
                result = await self._get_execution_status(request.parameters)
            elif capability == "cancel_execution":
                result = await self._cancel_execution(request.parameters)
            elif capability == "configure_execution":
                result = await self._configure_execution(request.parameters)
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
            if event.event_type == "adapter.available":
                await self._handle_adapter_available(event)
            elif event.event_type == "adapter.unavailable":
                await self._handle_adapter_unavailable(event)
            elif event.event_type == "task.priority_changed":
                await self._handle_priority_change(event)
                
        except Exception as e:
            await self._handle_error(e, f"event_handling_{event.event_type}")
    
    async def _cleanup_agent(self) -> None:
        """Cleanup auto worker resources."""
        # Cancel all active executions
        for execution_id, context in self._active_executions.items():
            if context.execution_future and not context.execution_future.done():
                context.execution_future.cancel()
        
        self.logger.info("Auto Worker cleanup complete")
    
    # Private implementation methods
    
    async def _execute_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single task."""
        task_data = parameters.get("task")
        if not task_data:
            raise ValueError("Task data required")
        
        task = Task(**task_data)
        result = await self._perform_task_execution(task)
        
        return {
            "execution_result": result.dict() if hasattr(result, 'dict') else result
        }
    
    async def _execute_batch(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multiple tasks."""
        task_list_data = parameters.get("task_list", [])
        if not task_list_data:
            raise ValueError("Task list required")
        
        tasks = [Task(**task_data) for task_data in task_list_data]
        
        # Limit concurrent tasks
        if len(tasks) > self._max_concurrent_tasks:
            self.logger.warning(f"Batch size {len(tasks)} exceeds limit {self._max_concurrent_tasks}")
            tasks = tasks[:self._max_concurrent_tasks]
        
        # Execute based on strategy
        if self._execution_strategy == ExecutionStrategy.PARALLEL:
            results = await self._execute_parallel(tasks)
        elif self._execution_strategy == ExecutionStrategy.PRIORITY_BASED:
            results = await self._execute_priority_based(tasks)
        else:
            results = await self._execute_sequential(tasks)
        
        return {
            "batch_execution_result": {
                "total_tasks": len(tasks),
                "results": results,
                "summary": self._create_batch_summary(results)
            }
        }
    
    async def _get_execution_status(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get execution status."""
        execution_id = parameters.get("execution_id")
        
        if execution_id:
            # Get specific execution status
            if execution_id in self._active_executions:
                context = self._active_executions[execution_id]
                status = {
                    "execution_id": execution_id,
                    "task_id": context.task.task_id,
                    "status": context.task.status.value,
                    "started_at": context.started_at.isoformat(),
                    "retry_count": context.retry_count,
                    "adapter_id": context.adapter_id
                }
            else:
                status = {"execution_id": execution_id, "status": "not_found"}
        else:
            # Get all execution statuses
            status = {
                "active_executions": len(self._active_executions),
                "execution_stats": self._execution_stats,
                "executions": [
                    {
                        "execution_id": eid,
                        "task_id": ctx.task.task_id,
                        "status": ctx.task.status.value,
                        "started_at": ctx.started_at.isoformat()
                    }
                    for eid, ctx in self._active_executions.items()
                ]
            }
        
        return {"execution_status": status}
    
    async def _cancel_execution(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Cancel a running execution."""
        execution_id = parameters.get("execution_id")
        if not execution_id:
            raise ValueError("Execution ID required")
        
        if execution_id not in self._active_executions:
            return {
                "cancellation_result": {
                    "success": False,
                    "reason": "Execution not found"
                }
            }
        
        context = self._active_executions[execution_id]
        
        # Cancel the execution
        if context.execution_future and not context.execution_future.done():
            context.execution_future.cancel()
        
        context.task.status = TaskStatus.CANCELLED
        
        # Clean up
        del self._active_executions[execution_id]
        
        return {
            "cancellation_result": {
                "success": True,
                "execution_id": execution_id,
                "cancelled_at": datetime.now().isoformat()
            }
        }
    
    async def _configure_execution(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Configure execution parameters."""
        config = parameters.get("config", {})
        
        if "max_concurrent_tasks" in config:
            self._max_concurrent_tasks = max(1, config["max_concurrent_tasks"])
        
        if "execution_strategy" in config:
            try:
                self._execution_strategy = ExecutionStrategy(config["execution_strategy"])
            except ValueError:
                self.logger.warning(f"Invalid execution strategy: {config['execution_strategy']}")
        
        if "retry_delays" in config:
            self._retry_delays = config["retry_delays"]
        
        return {
            "config_status": "updated",
            "current_config": {
                "max_concurrent_tasks": self._max_concurrent_tasks,
                "execution_strategy": self._execution_strategy.value,
                "retry_delays": self._retry_delays
            }
        }
    
    async def _perform_task_execution(self, task: Task) -> ExecutionResult:
        """Perform actual task execution."""
        execution_id = str(uuid.uuid4())
        context = ExecutionContext(
            task=task,
            started_at=datetime.now()
        )
        
        self._active_executions[execution_id] = context
        
        try:
            # Update task status
            task.status = TaskStatus.RUNNING
            
            # Select appropriate adapter
            adapter_id = await self._select_adapter(task)
            context.adapter_id = adapter_id
            
            # Execute with retry logic
            result = await self._execute_with_retry(task, context)
            
            # Update statistics
            self._execution_stats['total_executed'] += 1
            if result.overall_success:
                self._execution_stats['successful'] += 1
            else:
                self._execution_stats['failed'] += 1
            
            if context.retry_count > 0:
                self._execution_stats['retried'] += 1
            
            return result
            
        finally:
            # Clean up
            if execution_id in self._active_executions:
                del self._active_executions[execution_id]
    
    async def _select_adapter(self, task: Task) -> str:
        """Select appropriate adapter for task execution."""
        # Simple adapter selection logic (would be more sophisticated)
        target_adapter = task.target_adapter
        
        # Check if target adapter is available
        if target_adapter in self._available_adapters:
            return target_adapter
        
        # Fallback to default adapter
        return "default_adapter"
    
    async def _execute_with_retry(self, task: Task, context: ExecutionContext) -> ExecutionResult:
        """Execute task with retry logic."""
        max_retries = task.max_retries
        
        for attempt in range(max_retries + 1):
            try:
                # Simulate task execution (would use actual adapter)
                await asyncio.sleep(0.1)  # Simulate execution time
                
                # For demo purposes, simulate occasional failures
                import random
                if random.random() < 0.1:  # 10% failure rate
                    raise Exception("Simulated execution failure")
                
                # Create successful result
                result = ExecutionResult(
                    plan_id=task.task_id,
                    results=[],
                    overall_success=True,
                    total_execution_time=0.1
                )
                
                task.status = TaskStatus.COMPLETED
                return result
                
            except Exception as e:
                error_msg = str(e)
                context.error_history.append(error_msg)
                
                if attempt < max_retries:
                    # Wait before retry
                    retry_delay = self._retry_delays[min(attempt, len(self._retry_delays) - 1)]
                    await asyncio.sleep(retry_delay)
                    context.retry_count += 1
                    
                    self.logger.warning(f"Task {task.task_id} attempt {attempt + 1} failed, retrying: {error_msg}")
                else:
                    # Final failure
                    task.status = TaskStatus.FAILED
                    result = ExecutionResult(
                        plan_id=task.task_id,
                        results=[],
                        overall_success=False,
                        total_execution_time=0.1,
                        errors=[error_msg]
                    )
                    
                    self.logger.error(f"Task {task.task_id} failed after {max_retries} retries: {error_msg}")
                    return result
        
        # Should not reach here
        raise Exception("Unexpected execution path")
    
    async def _execute_sequential(self, tasks: List[Task]) -> List[ExecutionResult]:
        """Execute tasks sequentially."""
        results = []
        
        for task in tasks:
            result = await self._perform_task_execution(task)
            results.append(result)
        
        return results
    
    async def _execute_parallel(self, tasks: List[Task]) -> List[ExecutionResult]:
        """Execute tasks in parallel."""
        execution_tasks = []
        
        for task in tasks:
            exec_task = asyncio.create_task(self._perform_task_execution(task))
            execution_tasks.append(exec_task)
        
        results = await asyncio.gather(*execution_tasks, return_exceptions=True)
        
        # Convert exceptions to failed ExecutionResults
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_result = ExecutionResult(
                    plan_id=tasks[i].task_id,
                    results=[],
                    overall_success=False,
                    total_execution_time=0.0,
                    errors=[str(result)]
                )
                processed_results.append(failed_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_priority_based(self, tasks: List[Task]) -> List[ExecutionResult]:
        """Execute tasks based on priority and dependencies."""
        # Sort by priority (assuming higher risk = higher priority for demo)
        sorted_tasks = sorted(tasks, key=lambda t: t.risk_level.value, reverse=True)
        
        # For now, execute sequentially (could be enhanced with dependency resolution)
        return await self._execute_sequential(sorted_tasks)
    
    def _create_batch_summary(self, results: List[ExecutionResult]) -> Dict[str, Any]:
        """Create summary of batch execution results."""
        total = len(results)
        successful = sum(1 for r in results if r.overall_success)
        failed = total - successful
        
        total_time = sum(r.total_execution_time for r in results)
        
        return {
            "total_tasks": total,
            "successful_tasks": successful,
            "failed_tasks": failed,
            "success_rate": successful / total * 100 if total > 0 else 0,
            "total_execution_time": total_time,
            "average_execution_time": total_time / total if total > 0 else 0
        }
    
    async def _execution_processor(self) -> None:
        """Background execution processor."""
        while not self._shutdown_requested:
            try:
                # Process execution queue (if implemented)
                await asyncio.sleep(1)
                
                # Monitor execution capacity
                if len(self._active_executions) >= self._max_concurrent_tasks:
                    continue
                
                # Could pull tasks from queue here
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._handle_error(e, "execution_processor")
    
    async def _execution_monitor(self) -> None:
        """Background execution monitor."""
        while not self._shutdown_requested:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # Check for stuck executions
                current_time = datetime.now()
                
                for execution_id, context in list(self._active_executions.items()):
                    execution_time = current_time - context.started_at
                    
                    # Check for timeout
                    if context.task.timeout and execution_time > context.task.timeout:
                        self.logger.warning(f"Execution {execution_id} timed out")
                        
                        # Cancel execution
                        if context.execution_future and not context.execution_future.done():
                            context.execution_future.cancel()
                        
                        context.task.status = TaskStatus.FAILED
                        del self._active_executions[execution_id]
                
                # Update average duration
                if self._execution_stats['total_executed'] > 0:
                    # This would be calculated from actual execution times
                    pass
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._handle_error(e, "execution_monitor")
    
    async def _handle_adapter_available(self, event: AgentEvent) -> None:
        """Handle adapter availability events."""
        adapter_id = event.data.get("adapter_id")
        adapter_info = event.data.get("adapter_info", {})
        
        if adapter_id:
            self._available_adapters[adapter_id] = adapter_info
            self.logger.info(f"Adapter {adapter_id} is now available")
    
    async def _handle_adapter_unavailable(self, event: AgentEvent) -> None:
        """Handle adapter unavailability events."""
        adapter_id = event.data.get("adapter_id")
        
        if adapter_id and adapter_id in self._available_adapters:
            del self._available_adapters[adapter_id]
            self.logger.warning(f"Adapter {adapter_id} is no longer available")
    
    async def _handle_priority_change(self, event: AgentEvent) -> None:
        """Handle task priority change events."""
        task_id = event.data.get("task_id")
        new_priority = event.data.get("priority")
        
        # Find and update task priority in active executions
        for context in self._active_executions.values():
            if context.task.task_id == task_id:
                # Update task priority (would need to be implemented in Task class)
                self.logger.info(f"Updated priority for task {task_id} to {new_priority}")
                break