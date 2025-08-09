"""Base Learner - Abstract Foundation for Learning Algorithms

Provides the base architecture for all learning algorithms in the system,
ensuring consistent interfaces and integration with the agent ecosystem.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
from pathlib import Path

from loguru import logger

from ..core.event_system import EventSystem
from .learning_events import LearningEvent, LearningEventType, LearningEventData, LearningEventFactory


class LearningMode(Enum):
    """Learning operation modes"""
    ONLINE = "online"          # Real-time learning from interactions
    BATCH = "batch"            # Periodic batch learning from accumulated data
    REINFORCEMENT = "reinforcement"  # Learning from feedback and rewards
    SUPERVISED = "supervised"   # Learning from labeled examples
    UNSUPERVISED = "unsupervised"  # Pattern discovery from unlabeled data


class LearningStage(Enum):
    """Stages of the learning process"""
    INITIALIZATION = "initialization"
    DATA_COLLECTION = "data_collection"
    TRAINING = "training"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    OPTIMIZATION = "optimization"


@dataclass
class LearningContext:
    """Context information for learning operations"""
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    interaction_type: Optional[str] = None
    environment: str = "development"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "interaction_type": self.interaction_type,
            "environment": self.environment,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class LearningResult:
    """Result of a learning operation"""
    success: bool
    learning_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    improvements: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    execution_time: float = 0.0
    data_points_processed: int = 0
    model_version: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "success": self.success,
            "learning_id": self.learning_id,
            "improvements": self.improvements,
            "metrics": self.metrics,
            "recommendations": self.recommendations,
            "confidence_score": self.confidence_score,
            "execution_time": self.execution_time,
            "data_points_processed": self.data_points_processed,
            "model_version": self.model_version,
            "error_message": self.error_message
        }


class BaseLearner(ABC):
    """
    Abstract base class for all learning algorithms.
    
    Provides standard lifecycle, event integration, and state management
    for learning components that can be plugged into the agent system.
    """
    
    def __init__(self, 
                 learner_id: str,
                 event_system: EventSystem,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base learner.
        
        Args:
            learner_id: Unique identifier for this learner
            event_system: Event system for communication
            config: Optional configuration dictionary
        """
        self.learner_id = learner_id
        self.event_system = event_system
        self.config = config or {}
        
        # Learning configuration
        self.learning_mode = LearningMode(self.config.get('learning_mode', 'online'))
        self.batch_size = self.config.get('batch_size', 100)
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.model_save_interval = self.config.get('model_save_interval', 300)  # seconds
        
        # State management
        self._stage = LearningStage.INITIALIZATION
        self._is_active = False
        self._model_version = "1.0.0"
        self._last_training = datetime.now()
        self._total_training_time = 0.0
        
        # Performance tracking
        self._learning_history: List[LearningResult] = []
        self._performance_metrics = {
            'total_learnings': 0,
            'successful_learnings': 0,
            'average_confidence': 0.0,
            'data_points_processed': 0,
            'model_updates': 0
        }
        
        # Event subscriptions
        self._event_subscriptions: List[str] = []
        
        # Storage paths
        self._model_storage_path = Path(self.config.get('model_path', f'./models/{learner_id}'))
        self._model_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logger.bind(learner=learner_id)
        
    # Abstract properties that subclasses must implement
    
    @property
    @abstractmethod
    def learner_type(self) -> str:
        """Type identifier for this learner."""
        pass
    
    @property
    @abstractmethod
    def supported_learning_modes(self) -> List[LearningMode]:
        """List of learning modes this learner supports."""
        pass
    
    @property
    @abstractmethod
    def input_data_types(self) -> List[str]:
        """Types of input data this learner can process."""
        pass
    
    @property
    @abstractmethod
    def output_types(self) -> List[str]:
        """Types of outputs this learner produces."""
        pass
    
    # Abstract methods that subclasses must implement
    
    @abstractmethod
    async def _initialize_learner(self) -> None:
        """Initialize learner-specific functionality."""
        pass
    
    @abstractmethod
    async def _learn_from_data(self, data: List[Dict[str, Any]], 
                             context: LearningContext) -> LearningResult:
        """
        Learn from provided data.
        
        Args:
            data: List of data points to learn from
            context: Learning context information
            
        Returns:
            Learning result with improvements and metrics
        """
        pass
    
    @abstractmethod
    async def _save_model(self) -> bool:
        """Save the current model to persistent storage."""
        pass
    
    @abstractmethod
    async def _load_model(self) -> bool:
        """Load model from persistent storage."""
        pass
    
    @abstractmethod
    async def _cleanup_learner(self) -> None:
        """Cleanup learner-specific resources."""
        pass
    
    # Public API methods
    
    async def initialize(self) -> None:
        """Initialize the learner and start operation."""
        try:
            self._stage = LearningStage.INITIALIZATION
            self.logger.info(f"Initializing learner {self.learner_type}")
            
            # Subscribe to relevant events
            await self._setup_event_subscriptions()
            
            # Load existing model if available
            await self._load_model()
            
            # Initialize learner-specific functionality
            await self._initialize_learner()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self._stage = LearningStage.MONITORING
            self._is_active = True
            
            # Emit initialization event
            await self._emit_learning_event(
                LearningEventType.MODEL_UPDATED,
                LearningEventData(
                    learning_type=self.learner_type,
                    metrics={"model_version": float(self._model_version.replace(".", ""))},
                    insights=[f"Learner {self.learner_type} initialized successfully"]
                )
            )
            
            self.logger.info(f"Learner {self.learner_type} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Learner initialization failed: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the learner gracefully."""
        try:
            self.logger.info(f"Shutting down learner {self.learner_type}")
            self._is_active = False
            
            # Save current model
            await self._save_model()
            
            # Cleanup learner-specific resources
            await self._cleanup_learner()
            
            # Cleanup event subscriptions
            await self._cleanup_event_subscriptions()
            
            self.logger.info(f"Learner {self.learner_type} shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during learner shutdown: {e}")
    
    async def learn(self, data: List[Dict[str, Any]], 
                   context: Optional[LearningContext] = None) -> LearningResult:
        """
        Main learning interface.
        
        Args:
            data: Data to learn from
            context: Optional learning context
            
        Returns:
            Learning result with improvements and metrics
        """
        if not self._is_active:
            return LearningResult(
                success=False,
                error_message="Learner is not active"
            )
        
        # Use default context if none provided
        if context is None:
            context = LearningContext()
        
        start_time = datetime.now()
        
        try:
            self._stage = LearningStage.TRAINING
            
            # Emit learning start event
            await self._emit_learning_event(
                LearningEventType.MODEL_TRAINING_STARTED,
                LearningEventData(
                    learning_type=self.learner_type,
                    data_points={"data_count": len(data)},
                    timestamp=start_time
                )
            )
            
            # Perform the actual learning
            result = await self._learn_from_data(data, context)
            
            # Calculate execution time
            result.execution_time = (datetime.now() - start_time).total_seconds()
            result.data_points_processed = len(data)
            result.model_version = self._model_version
            
            # Update performance metrics
            self._update_performance_metrics(result)
            
            # Store in history
            self._learning_history.append(result)
            
            # Emit learning completion event
            if result.success:
                await self._emit_learning_event(
                    LearningEventType.MODEL_TRAINING_COMPLETED,
                    LearningEventData(
                        learning_type=self.learner_type,
                        metrics=result.metrics,
                        insights=result.improvements,
                        confidence_score=result.confidence_score
                    )
                )
            else:
                await self._emit_learning_event(
                    LearningEventType.MODEL_TRAINING_FAILED,
                    LearningEventData(
                        learning_type=self.learner_type,
                        data_points={"error": result.error_message}
                    )
                )
            
            self._stage = LearningStage.MONITORING
            return result
            
        except Exception as e:
            self.logger.error(f"Learning failed: {e}")
            
            result = LearningResult(
                success=False,
                error_message=str(e),
                execution_time=(datetime.now() - start_time).total_seconds(),
                data_points_processed=len(data)
            )
            
            # Emit failure event
            await self._emit_learning_event(
                LearningEventType.MODEL_TRAINING_FAILED,
                LearningEventData(
                    learning_type=self.learner_type,
                    data_points={"error": str(e)}
                )
            )
            
            return result
    
    async def get_insights(self, context: Optional[LearningContext] = None) -> List[str]:
        """
        Get current insights from the learner.
        
        Args:
            context: Optional context for insights
            
        Returns:
            List of insights
        """
        insights = [
            f"Total learnings: {self._performance_metrics['total_learnings']}",
            f"Success rate: {self._get_success_rate():.1%}",
            f"Average confidence: {self._performance_metrics['average_confidence']:.2f}",
            f"Model version: {self._model_version}"
        ]
        
        # Add recent learning insights
        if self._learning_history:
            recent_learning = self._learning_history[-1]
            insights.extend(recent_learning.improvements)
        
        return insights
    
    async def get_recommendations(self, context: Optional[LearningContext] = None) -> List[str]:
        """
        Get recommendations for system optimization.
        
        Args:
            context: Optional context for recommendations
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Performance-based recommendations
        success_rate = self._get_success_rate()
        if success_rate < 0.8:
            recommendations.append("Consider adjusting learning parameters for better performance")
        
        # Data quality recommendations
        if self._performance_metrics['data_points_processed'] < 100:
            recommendations.append("More training data may improve learning accuracy")
        
        # Model freshness recommendations
        time_since_last_training = (datetime.now() - self._last_training).total_seconds()
        if time_since_last_training > 86400:  # 24 hours
            recommendations.append("Model may benefit from recent data updates")
        
        return recommendations
    
    @property
    def current_stage(self) -> LearningStage:
        """Get current learning stage."""
        return self._stage
    
    @property
    def is_active(self) -> bool:
        """Check if learner is active."""
        return self._is_active
    
    @property
    def performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            **self._performance_metrics,
            'success_rate': self._get_success_rate(),
            'model_version': self._model_version,
            'uptime_seconds': (datetime.now() - self._last_training).total_seconds(),
            'learning_history_size': len(self._learning_history)
        }
    
    # Protected helper methods
    
    async def _emit_learning_event(self, event_type: LearningEventType, 
                                 data: LearningEventData) -> None:
        """Emit a learning event."""
        event = LearningEvent(
            event_type=event_type,
            data=data,
            source_component=f"learner_{self.learner_id}"
        )
        
        system_event = event.to_system_event()
        await self.event_system.emit(system_event)
    
    def _update_performance_metrics(self, result: LearningResult) -> None:
        """Update performance metrics based on learning result."""
        self._performance_metrics['total_learnings'] += 1
        
        if result.success:
            self._performance_metrics['successful_learnings'] += 1
            self._performance_metrics['model_updates'] += 1
            self._last_training = datetime.now()
        
        self._performance_metrics['data_points_processed'] += result.data_points_processed
        
        # Update average confidence
        total_confidence = (self._performance_metrics['average_confidence'] * 
                          (self._performance_metrics['total_learnings'] - 1) + 
                          result.confidence_score)
        self._performance_metrics['average_confidence'] = (
            total_confidence / self._performance_metrics['total_learnings']
        )
        
        self._total_training_time += result.execution_time
    
    def _get_success_rate(self) -> float:
        """Calculate current success rate."""
        if self._performance_metrics['total_learnings'] == 0:
            return 1.0
        return (self._performance_metrics['successful_learnings'] / 
                self._performance_metrics['total_learnings'])
    
    async def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions for learning triggers."""
        # Subclasses can override to subscribe to specific events
        pass
    
    async def _cleanup_event_subscriptions(self) -> None:
        """Cleanup event subscriptions."""
        for handler_id in self._event_subscriptions:
            self.event_system.unsubscribe(handler_id)
        self._event_subscriptions.clear()
    
    async def _start_background_tasks(self) -> None:
        """Start background tasks for the learner."""
        # Model saving task
        asyncio.create_task(self._model_save_loop())
        
        # Performance monitoring task  
        asyncio.create_task(self._performance_monitoring_loop())
    
    async def _model_save_loop(self) -> None:
        """Background task for periodic model saving."""
        while self._is_active:
            try:
                await asyncio.sleep(self.model_save_interval)
                
                if self._stage == LearningStage.MONITORING:
                    await self._save_model()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in model save loop: {e}")
    
    async def _performance_monitoring_loop(self) -> None:
        """Background task for performance monitoring."""
        while self._is_active:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                # Emit performance metrics
                await self._emit_learning_event(
                    LearningEventType.AGENT_PERFORMANCE_ANALYZED,
                    LearningEventData(
                        agent_id=self.learner_id,
                        learning_type=self.learner_type,
                        metrics=self.performance_metrics,
                        confidence_score=self._performance_metrics['average_confidence']
                    )
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")


# Learning algorithm registry for dynamic loading
class LearnerRegistry:
    """Registry for learning algorithm implementations"""
    
    _learners: Dict[str, Type[BaseLearner]] = {}
    
    @classmethod
    def register(cls, learner_type: str, learner_class: Type[BaseLearner]):
        """Register a learner implementation."""
        cls._learners[learner_type] = learner_class
        logger.info(f"Registered learner type: {learner_type}")
    
    @classmethod
    def get_learner_class(cls, learner_type: str) -> Optional[Type[BaseLearner]]:
        """Get a learner class by type."""
        return cls._learners.get(learner_type)
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of available learner types."""
        return list(cls._learners.keys())
    
    @classmethod
    def create_learner(cls, learner_type: str, learner_id: str,
                      event_system: EventSystem,
                      config: Optional[Dict[str, Any]] = None) -> Optional[BaseLearner]:
        """Create a learner instance by type."""
        learner_class = cls.get_learner_class(learner_type)
        if learner_class:
            return learner_class(learner_id, event_system, config)
        return None