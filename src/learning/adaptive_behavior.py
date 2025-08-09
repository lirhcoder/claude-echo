"""Adaptive Behavior Manager - Intelligent System Adaptation

Manages adaptive behaviors based on learning insights, enabling the system
to automatically adjust and optimize its operations based on user patterns
and performance metrics.
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json

from loguru import logger

from ..core.event_system import EventSystem
from .base_learner import BaseLearner, LearningContext, LearningResult, LearningMode
from .learning_data_manager import LearningDataManager
from .learning_events import LearningEventFactory, LearningEventType


class AdaptationType(Enum):
    """Types of adaptive behaviors"""
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    USER_EXPERIENCE = "user_experience"
    RESOURCE_ALLOCATION = "resource_allocation"
    RESPONSE_PERSONALIZATION = "response_personalization"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"
    ERROR_PREVENTION = "error_prevention"


class AdaptationPriority(Enum):
    """Priority levels for adaptations"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class BehaviorPattern:
    """Pattern detected in system or user behavior"""
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pattern_type: str = "unknown"
    description: str = ""
    confidence_score: float = 0.0
    frequency: int = 0
    last_observed: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    impact_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "confidence_score": self.confidence_score,
            "frequency": self.frequency,
            "last_observed": self.last_observed.isoformat(),
            "context": self.context,
            "impact_metrics": self.impact_metrics
        }


@dataclass
class AdaptationStrategy:
    """Strategy for implementing an adaptive behavior"""
    strategy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    adaptation_type: AdaptationType = AdaptationType.PERFORMANCE_OPTIMIZATION
    priority: AdaptationPriority = AdaptationPriority.MEDIUM
    target_components: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    expected_impact: Dict[str, float] = field(default_factory=dict)
    rollback_strategy: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_applied: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0
    
    def get_success_rate(self) -> float:
        """Calculate success rate of this strategy."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "adaptation_type": self.adaptation_type.value,
            "priority": self.priority.value,
            "target_components": self.target_components,
            "conditions": self.conditions,
            "actions": self.actions,
            "expected_impact": self.expected_impact,
            "rollback_strategy": self.rollback_strategy,
            "created_at": self.created_at.isoformat(),
            "last_applied": self.last_applied.isoformat() if self.last_applied else None,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.get_success_rate()
        }


class AdaptiveBehaviorLearner(BaseLearner):
    """Learning algorithm for adaptive behavior patterns"""
    
    @property
    def learner_type(self) -> str:
        return "adaptive_behavior"
    
    @property
    def supported_learning_modes(self) -> List[LearningMode]:
        return [LearningMode.ONLINE, LearningMode.REINFORCEMENT]
    
    @property
    def input_data_types(self) -> List[str]:
        return ["system_metrics", "user_interactions", "performance_data"]
    
    @property
    def output_types(self) -> List[str]:
        return ["behavior_patterns", "adaptation_strategies", "recommendations"]
    
    async def _initialize_learner(self) -> None:
        """Initialize adaptive behavior learner."""
        self.adaptation_threshold = self.config.get('adaptation_threshold', 0.7)
        self.pattern_memory = {}
        self.strategy_performance = {}
        self.logger.info("Adaptive Behavior Learner initialized")
    
    async def _learn_from_data(self, data: List[Dict[str, Any]], 
                             context: LearningContext) -> LearningResult:
        """Learn adaptive patterns from system and user data."""
        try:
            patterns_detected = []
            strategies_generated = []
            
            # Analyze system performance patterns
            performance_patterns = await self._analyze_performance_patterns(data)
            patterns_detected.extend(performance_patterns)
            
            # Analyze user behavior patterns
            user_patterns = await self._analyze_user_behavior_patterns(data)
            patterns_detected.extend(user_patterns)
            
            # Generate adaptation strategies based on patterns
            for pattern in patterns_detected:
                if pattern.confidence_score >= self.adaptation_threshold:
                    strategies = await self._generate_adaptation_strategies(pattern)
                    strategies_generated.extend(strategies)
            
            # Update pattern memory
            for pattern in patterns_detected:
                self.pattern_memory[pattern.pattern_id] = pattern
            
            improvements = []
            metrics = {
                "patterns_detected": len(patterns_detected),
                "strategies_generated": len(strategies_generated),
                "confidence_average": sum(p.confidence_score for p in patterns_detected) / max(len(patterns_detected), 1)
            }
            
            if patterns_detected:
                improvements.append(f"Detected {len(patterns_detected)} behavioral patterns")
            
            if strategies_generated:
                improvements.append(f"Generated {len(strategies_generated)} adaptation strategies")
            
            return LearningResult(
                success=True,
                improvements=improvements,
                metrics=metrics,
                confidence_score=metrics["confidence_average"],
                data_points_processed=len(data)
            )
            
        except Exception as e:
            return LearningResult(
                success=False,
                error_message=str(e),
                data_points_processed=len(data)
            )
    
    async def _analyze_performance_patterns(self, data: List[Dict[str, Any]]) -> List[BehaviorPattern]:
        """Analyze system performance patterns."""
        patterns = []
        
        # Group data by time windows
        response_times = []
        error_rates = []
        
        for item in data:
            if item.get("data_type") == "system_metrics":
                content = item.get("data_content", {})
                if "response_time" in content:
                    response_times.append(content["response_time"])
                if "error_rate" in content:
                    error_rates.append(content["error_rate"])
        
        # Detect performance degradation pattern
        if len(response_times) > 10:
            recent_avg = sum(response_times[-5:]) / 5
            historical_avg = sum(response_times[:-5]) / (len(response_times) - 5)
            
            if recent_avg > historical_avg * 1.2:  # 20% increase
                pattern = BehaviorPattern(
                    pattern_type="performance_degradation",
                    description="Response time increase detected",
                    confidence_score=min((recent_avg / historical_avg - 1) * 2, 1.0),
                    frequency=len(response_times),
                    context={
                        "recent_avg": recent_avg,
                        "historical_avg": historical_avg,
                        "degradation_ratio": recent_avg / historical_avg
                    },
                    impact_metrics={"response_time_increase": recent_avg - historical_avg}
                )
                patterns.append(pattern)
        
        # Detect high error rate pattern
        if len(error_rates) > 5:
            recent_error_rate = sum(error_rates[-3:]) / 3
            if recent_error_rate > 0.05:  # 5% error rate
                pattern = BehaviorPattern(
                    pattern_type="high_error_rate",
                    description="Elevated error rate detected",
                    confidence_score=min(recent_error_rate * 10, 1.0),
                    frequency=len(error_rates),
                    context={"error_rate": recent_error_rate},
                    impact_metrics={"error_impact": recent_error_rate}
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _analyze_user_behavior_patterns(self, data: List[Dict[str, Any]]) -> List[BehaviorPattern]:
        """Analyze user behavior patterns."""
        patterns = []
        
        # Analyze interaction timing patterns
        interaction_times = []
        task_completions = []
        
        for item in data:
            if item.get("data_type") == "user_interaction":
                content = item.get("data_content", {})
                if "timestamp" in content:
                    try:
                        time = datetime.fromisoformat(content["timestamp"])
                        interaction_times.append(time.hour)
                    except:
                        pass
            
            elif item.get("data_type") == "task_completion":
                task_completions.append(item.get("data_content", {}))
        
        # Detect peak usage times
        if len(interaction_times) > 20:
            hour_counts = {}
            for hour in interaction_times:
                hour_counts[hour] = hour_counts.get(hour, 0) + 1
            
            max_count = max(hour_counts.values())
            peak_hours = [h for h, c in hour_counts.items() if c == max_count]
            
            if max_count > len(interaction_times) * 0.2:  # 20% of interactions in peak hour
                pattern = BehaviorPattern(
                    pattern_type="peak_usage_time",
                    description=f"Peak usage detected at hour(s): {peak_hours}",
                    confidence_score=max_count / len(interaction_times),
                    frequency=max_count,
                    context={
                        "peak_hours": peak_hours,
                        "peak_percentage": max_count / len(interaction_times)
                    }
                )
                patterns.append(pattern)
        
        # Detect common failure patterns
        failed_tasks = [t for t in task_completions if not t.get("success", True)]
        if len(failed_tasks) > 3:
            failure_types = {}
            for task in failed_tasks:
                task_type = task.get("task_type", "unknown")
                failure_types[task_type] = failure_types.get(task_type, 0) + 1
            
            for task_type, count in failure_types.items():
                if count >= 2:  # At least 2 failures of same type
                    pattern = BehaviorPattern(
                        pattern_type="task_failure_pattern",
                        description=f"Recurring failures in {task_type} tasks",
                        confidence_score=min(count / 5, 1.0),
                        frequency=count,
                        context={
                            "task_type": task_type,
                            "failure_count": count,
                            "total_failures": len(failed_tasks)
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _generate_adaptation_strategies(self, pattern: BehaviorPattern) -> List[AdaptationStrategy]:
        """Generate adaptation strategies for detected patterns."""
        strategies = []
        
        if pattern.pattern_type == "performance_degradation":
            # Strategy 1: Increase cache size
            strategy = AdaptationStrategy(
                name="Increase Cache Size",
                adaptation_type=AdaptationType.PERFORMANCE_OPTIMIZATION,
                priority=AdaptationPriority.HIGH,
                target_components=["cache_manager", "data_layer"],
                conditions={"response_time_threshold": pattern.context.get("recent_avg", 0)},
                actions=[
                    {"type": "increase_cache_size", "factor": 1.5},
                    {"type": "optimize_cache_policy", "policy": "lru_enhanced"}
                ],
                expected_impact={"response_time_improvement": 0.2}
            )
            strategies.append(strategy)
            
            # Strategy 2: Load balancing adjustment
            strategy = AdaptationStrategy(
                name="Adjust Load Balancing",
                adaptation_type=AdaptationType.RESOURCE_ALLOCATION,
                priority=AdaptationPriority.MEDIUM,
                target_components=["load_balancer"],
                conditions={"degradation_ratio": pattern.context.get("degradation_ratio", 1.0)},
                actions=[
                    {"type": "redistribute_load", "algorithm": "least_response_time"},
                    {"type": "increase_worker_count", "factor": 1.2}
                ],
                expected_impact={"load_distribution_improvement": 0.3}
            )
            strategies.append(strategy)
        
        elif pattern.pattern_type == "high_error_rate":
            # Strategy: Implement circuit breaker
            strategy = AdaptationStrategy(
                name="Implement Circuit Breaker",
                adaptation_type=AdaptationType.ERROR_PREVENTION,
                priority=AdaptationPriority.CRITICAL,
                target_components=["error_handler", "request_processor"],
                conditions={"error_rate_threshold": pattern.context.get("error_rate", 0)},
                actions=[
                    {"type": "enable_circuit_breaker", "failure_threshold": 5},
                    {"type": "implement_retry_logic", "max_retries": 3}
                ],
                expected_impact={"error_rate_reduction": 0.5}
            )
            strategies.append(strategy)
        
        elif pattern.pattern_type == "peak_usage_time":
            # Strategy: Pre-emptive resource scaling
            strategy = AdaptationStrategy(
                name="Pre-emptive Resource Scaling",
                adaptation_type=AdaptationType.RESOURCE_ALLOCATION,
                priority=AdaptationPriority.MEDIUM,
                target_components=["resource_manager"],
                conditions={"peak_hours": pattern.context.get("peak_hours", [])},
                actions=[
                    {"type": "schedule_scaling", "peak_hours": pattern.context.get("peak_hours", [])},
                    {"type": "pre_warm_resources", "factor": 1.5}
                ],
                expected_impact={"resource_availability": 0.4}
            )
            strategies.append(strategy)
        
        elif pattern.pattern_type == "task_failure_pattern":
            # Strategy: Task-specific optimization
            task_type = pattern.context.get("task_type", "unknown")
            strategy = AdaptationStrategy(
                name=f"Optimize {task_type} Task Processing",
                adaptation_type=AdaptationType.WORKFLOW_OPTIMIZATION,
                priority=AdaptationPriority.HIGH,
                target_components=["task_processor", "workflow_engine"],
                conditions={"task_type": task_type},
                actions=[
                    {"type": "adjust_task_parameters", "task_type": task_type},
                    {"type": "implement_task_validation", "validation_level": "enhanced"},
                    {"type": "add_fallback_mechanism", "fallback_type": "graceful_degradation"}
                ],
                expected_impact={"task_success_rate": 0.3}
            )
            strategies.append(strategy)
        
        return strategies
    
    async def _save_model(self) -> bool:
        """Save the current pattern memory and strategies."""
        try:
            model_data = {
                "pattern_memory": {pid: pattern.to_dict() for pid, pattern in self.pattern_memory.items()},
                "strategy_performance": self.strategy_performance,
                "model_version": self._model_version,
                "last_updated": datetime.now().isoformat()
            }
            
            model_file = self._model_storage_path / "adaptive_behavior_model.json"
            with open(model_file, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save adaptive behavior model: {e}")
            return False
    
    async def _load_model(self) -> bool:
        """Load existing pattern memory and strategies."""
        try:
            model_file = self._model_storage_path / "adaptive_behavior_model.json"
            if not model_file.exists():
                return True  # No existing model, start fresh
            
            with open(model_file, 'r') as f:
                model_data = json.load(f)
            
            # Restore pattern memory
            self.pattern_memory = {}
            for pid, pattern_data in model_data.get("pattern_memory", {}).items():
                # Recreate BehaviorPattern objects
                pattern = BehaviorPattern(
                    pattern_id=pattern_data["pattern_id"],
                    pattern_type=pattern_data["pattern_type"],
                    description=pattern_data["description"],
                    confidence_score=pattern_data["confidence_score"],
                    frequency=pattern_data["frequency"],
                    last_observed=datetime.fromisoformat(pattern_data["last_observed"]),
                    context=pattern_data["context"],
                    impact_metrics=pattern_data["impact_metrics"]
                )
                self.pattern_memory[pid] = pattern
            
            self.strategy_performance = model_data.get("strategy_performance", {})
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load adaptive behavior model: {e}")
            return False
    
    async def _cleanup_learner(self) -> None:
        """Cleanup adaptive behavior learner resources."""
        await self._save_model()


class AdaptiveBehaviorManager:
    """
    Central manager for adaptive behaviors in the system.
    
    Coordinates pattern detection, strategy generation, and adaptation
    implementation across all system components.
    """
    
    def __init__(self, 
                 event_system: EventSystem,
                 learning_data_manager: LearningDataManager,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Adaptive Behavior Manager.
        
        Args:
            event_system: Event system for communication
            learning_data_manager: Learning data manager
            config: Optional configuration dictionary
        """
        self.event_system = event_system
        self.learning_data_manager = learning_data_manager
        self.config = config or {}
        
        # Adaptive behavior learner
        self._learner = AdaptiveBehaviorLearner(
            learner_id="adaptive_behavior_learner",
            event_system=event_system,
            config=self.config.get('learner', {})
        )
        
        # Strategy management
        self._active_strategies: Dict[str, AdaptationStrategy] = {}
        self._strategy_history: List[Dict[str, Any]] = []
        self._adaptation_callbacks: Dict[str, List[Callable]] = {}
        
        # Pattern tracking
        self._detected_patterns: Dict[str, BehaviorPattern] = {}
        self._pattern_thresholds = self.config.get('pattern_thresholds', {
            'confidence_threshold': 0.7,
            'frequency_threshold': 3,
            'impact_threshold': 0.1
        })
        
        # Performance monitoring
        self._adaptation_metrics = {
            'total_adaptations': 0,
            'successful_adaptations': 0,
            'failed_adaptations': 0,
            'patterns_detected': 0,
            'strategies_generated': 0,
            'average_impact': 0.0
        }
        
        # Background tasks
        self._analysis_task: Optional[asyncio.Task] = None
        self._adaptation_task: Optional[asyncio.Task] = None
        
        # Setup logging
        self.logger = logger.bind(component="adaptive_behavior_manager")
    
    async def initialize(self) -> None:
        """Initialize the adaptive behavior manager."""
        try:
            self.logger.info("Initializing Adaptive Behavior Manager")
            
            # Initialize the learner
            await self._learner.initialize()
            
            # Load existing strategies
            await self._load_strategies()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.logger.info("Adaptive Behavior Manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Adaptive Behavior Manager: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the adaptive behavior manager."""
        try:
            self.logger.info("Shutting down Adaptive Behavior Manager")
            
            # Stop background tasks
            if self._analysis_task and not self._analysis_task.done():
                self._analysis_task.cancel()
                try:
                    await self._analysis_task
                except asyncio.CancelledError:
                    pass
            
            if self._adaptation_task and not self._adaptation_task.done():
                self._adaptation_task.cancel()
                try:
                    await self._adaptation_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown learner
            await self._learner.shutdown()
            
            # Save strategies
            await self._save_strategies()
            
            self.logger.info("Adaptive Behavior Manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def analyze_and_adapt(self, force_analysis: bool = False) -> Dict[str, Any]:
        """
        Perform analysis and generate adaptations.
        
        Args:
            force_analysis: Force immediate analysis regardless of schedule
            
        Returns:
            Analysis and adaptation results
        """
        try:
            # Collect recent learning data
            learning_data = await self.learning_data_manager.retrieve_learning_data(
                data_type=None,  # Get all types
                limit=1000
            )
            
            if not learning_data and not force_analysis:
                return {"message": "No data available for analysis"}
            
            # Convert to format expected by learner
            learner_data = []
            for data in learning_data:
                learner_data.append({
                    "data_type": data.data_type,
                    "data_content": data.data_content,
                    "metadata": data.metadata,
                    "timestamp": data.created_at.isoformat()
                })
            
            # Perform learning
            context = LearningContext(
                interaction_type="system_analysis",
                environment=self.config.get('environment', 'development')
            )
            
            result = await self._learner.learn(learner_data, context)
            
            if result.success:
                # Extract patterns and strategies from learner
                patterns = getattr(self._learner, 'pattern_memory', {})
                self._detected_patterns.update(patterns)
                
                # Generate and apply high-priority adaptations
                applied_adaptations = []
                for pattern in patterns.values():
                    if pattern.confidence_score >= self._pattern_thresholds['confidence_threshold']:
                        strategies = await self._learner._generate_adaptation_strategies(pattern)
                        
                        for strategy in strategies:
                            if strategy.priority in [AdaptationPriority.HIGH, AdaptationPriority.CRITICAL]:
                                success = await self._apply_adaptation_strategy(strategy)
                                applied_adaptations.append({
                                    "strategy_id": strategy.strategy_id,
                                    "name": strategy.name,
                                    "success": success
                                })
                
                # Update metrics
                self._adaptation_metrics['patterns_detected'] += len(patterns)
                self._adaptation_metrics['successful_adaptations'] += sum(1 for a in applied_adaptations if a['success'])
                self._adaptation_metrics['failed_adaptations'] += sum(1 for a in applied_adaptations if not a['success'])
                
                return {
                    "analysis_result": result.to_dict(),
                    "patterns_detected": len(patterns),
                    "adaptations_applied": len(applied_adaptations),
                    "applied_adaptations": applied_adaptations
                }
            
            else:
                return {
                    "analysis_result": result.to_dict(),
                    "error": "Analysis failed"
                }
                
        except Exception as e:
            self.logger.error(f"Failed to analyze and adapt: {e}")
            return {"error": str(e)}
    
    async def register_adaptation_callback(self, component: str, 
                                         callback: Callable[[AdaptationStrategy], bool]) -> None:
        """
        Register a callback for adaptation notifications.
        
        Args:
            component: Component name
            callback: Callback function to handle adaptations
        """
        if component not in self._adaptation_callbacks:
            self._adaptation_callbacks[component] = []
        
        self._adaptation_callbacks[component].append(callback)
        self.logger.info(f"Registered adaptation callback for component: {component}")
    
    async def get_adaptation_insights(self) -> Dict[str, Any]:
        """Get insights about current adaptations and patterns."""
        return {
            "metrics": self._adaptation_metrics,
            "active_strategies": len(self._active_strategies),
            "detected_patterns": len(self._detected_patterns),
            "pattern_summary": [
                {
                    "type": pattern.pattern_type,
                    "confidence": pattern.confidence_score,
                    "frequency": pattern.frequency,
                    "last_observed": pattern.last_observed.isoformat()
                }
                for pattern in self._detected_patterns.values()
            ],
            "strategy_summary": [
                {
                    "name": strategy.name,
                    "type": strategy.adaptation_type.value,
                    "priority": strategy.priority.value,
                    "success_rate": strategy.get_success_rate()
                }
                for strategy in self._active_strategies.values()
            ]
        }
    
    # Private implementation methods
    
    async def _apply_adaptation_strategy(self, strategy: AdaptationStrategy) -> bool:
        """Apply an adaptation strategy."""
        try:
            # Notify registered callbacks
            success_count = 0
            total_callbacks = 0
            
            for component in strategy.target_components:
                if component in self._adaptation_callbacks:
                    for callback in self._adaptation_callbacks[component]:
                        try:
                            result = callback(strategy)
                            if result:
                                success_count += 1
                            total_callbacks += 1
                        except Exception as e:
                            self.logger.error(f"Adaptation callback failed for {component}: {e}")
                            total_callbacks += 1
            
            # Update strategy statistics
            strategy.last_applied = datetime.now()
            if success_count > 0:
                strategy.success_count += 1
                success = True
            else:
                strategy.failure_count += 1
                success = False
            
            # Store strategy
            self._active_strategies[strategy.strategy_id] = strategy
            
            # Add to history
            self._strategy_history.append({
                "strategy_id": strategy.strategy_id,
                "name": strategy.name,
                "applied_at": strategy.last_applied.isoformat(),
                "success": success,
                "callbacks_notified": total_callbacks,
                "successful_callbacks": success_count
            })
            
            # Emit adaptation event
            event = LearningEventFactory.adaptation_applied(
                agent_id="adaptive_behavior_manager",
                adaptation_type=strategy.name,
                before_metrics={},  # Would be populated with actual before metrics
                after_metrics={"success": success}
            )
            
            system_event = event.to_system_event()
            await self.event_system.emit(system_event)
            
            self.logger.info(f"Applied adaptation strategy: {strategy.name} (Success: {success})")
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to apply adaptation strategy {strategy.name}: {e}")
            strategy.failure_count += 1
            return False
    
    async def _start_background_tasks(self) -> None:
        """Start background analysis and adaptation tasks."""
        analysis_interval = self.config.get('analysis_interval', 1800)  # 30 minutes
        self._analysis_task = asyncio.create_task(self._analysis_loop(analysis_interval))
        
        adaptation_interval = self.config.get('adaptation_interval', 3600)  # 1 hour  
        self._adaptation_task = asyncio.create_task(self._adaptation_loop(adaptation_interval))
    
    async def _analysis_loop(self, interval: int) -> None:
        """Background analysis loop."""
        while True:
            try:
                await asyncio.sleep(interval)
                await self.analyze_and_adapt()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")
    
    async def _adaptation_loop(self, interval: int) -> None:
        """Background adaptation maintenance loop."""
        while True:
            try:
                await asyncio.sleep(interval)
                
                # Review strategy performance
                await self._review_strategy_performance()
                
                # Clean up old strategies
                await self._cleanup_old_strategies()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in adaptation loop: {e}")
    
    async def _review_strategy_performance(self) -> None:
        """Review and optimize strategy performance."""
        try:
            for strategy in self._active_strategies.values():
                success_rate = strategy.get_success_rate()
                
                # Disable poorly performing strategies
                if success_rate < 0.3 and strategy.success_count + strategy.failure_count > 5:
                    self.logger.warning(f"Disabling poorly performing strategy: {strategy.name}")
                    # Could implement strategy disabling logic here
                
                # Promote successful strategies
                elif success_rate > 0.8 and strategy.success_count > 10:
                    if strategy.priority != AdaptationPriority.CRITICAL:
                        strategy.priority = AdaptationPriority.HIGH
                        self.logger.info(f"Promoted successful strategy: {strategy.name}")
                        
        except Exception as e:
            self.logger.error(f"Failed to review strategy performance: {e}")
    
    async def _cleanup_old_strategies(self) -> None:
        """Clean up old and unused strategies."""
        try:
            current_time = datetime.now()
            strategies_to_remove = []
            
            for strategy_id, strategy in self._active_strategies.items():
                # Remove strategies not used in the last 7 days
                if strategy.last_applied:
                    age = current_time - strategy.last_applied
                    if age > timedelta(days=7) and strategy.get_success_rate() < 0.5:
                        strategies_to_remove.append(strategy_id)
                else:
                    # Remove strategies created more than 24 hours ago but never applied
                    age = current_time - strategy.created_at
                    if age > timedelta(hours=24):
                        strategies_to_remove.append(strategy_id)
            
            for strategy_id in strategies_to_remove:
                del self._active_strategies[strategy_id]
                self.logger.debug(f"Removed unused strategy: {strategy_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old strategies: {e}")
    
    async def _load_strategies(self) -> None:
        """Load existing strategies from storage."""
        try:
            # Implementation would load strategies from persistent storage
            self.logger.info("Loaded existing adaptation strategies")
        except Exception as e:
            self.logger.error(f"Failed to load strategies: {e}")
    
    async def _save_strategies(self) -> None:
        """Save current strategies to storage."""
        try:
            # Implementation would save strategies to persistent storage
            self.logger.info(f"Saved {len(self._active_strategies)} adaptation strategies")
        except Exception as e:
            self.logger.error(f"Failed to save strategies: {e}")