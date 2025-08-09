"""Learning Events - Event Types for Learning System

Defines events emitted by the learning system for integration
with the existing EventSystem architecture.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from pydantic import BaseModel

from ..core.event_system import Event


class LearningEventType(Enum):
    """Types of learning system events"""
    # User interaction learning
    USER_PATTERN_DETECTED = "learning.user_pattern_detected"
    USER_PREFERENCE_UPDATED = "learning.user_preference_updated"
    USER_BEHAVIOR_ANALYZED = "learning.user_behavior_analyzed"
    
    # Model training and updates
    MODEL_TRAINING_STARTED = "learning.model_training_started"
    MODEL_TRAINING_COMPLETED = "learning.model_training_completed"
    MODEL_TRAINING_FAILED = "learning.model_training_failed"
    MODEL_UPDATED = "learning.model_updated"
    
    # Performance and optimization
    PERFORMANCE_IMPROVEMENT_DETECTED = "learning.performance_improvement"
    ADAPTATION_APPLIED = "learning.adaptation_applied"
    OPTIMIZATION_SUGGESTION = "learning.optimization_suggestion"
    
    # Knowledge base
    KNOWLEDGE_LEARNED = "learning.knowledge_learned"
    KNOWLEDGE_UPDATED = "learning.knowledge_updated"
    INSIGHT_DISCOVERED = "learning.insight_discovered"
    
    # System learning
    AGENT_PERFORMANCE_ANALYZED = "learning.agent_performance_analyzed"
    COLLABORATION_PATTERN_LEARNED = "learning.collaboration_pattern_learned"
    SYSTEM_ADAPTATION_COMPLETED = "learning.system_adaptation_completed"


@dataclass
class LearningEventData:
    """Data structure for learning events"""
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    learning_type: Optional[str] = None
    data_points: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class LearningEvent(BaseModel):
    """Enhanced learning event with structured data"""
    event_type: LearningEventType
    data: LearningEventData
    source_component: str
    target_agents: List[str] = []
    priority: str = "normal"
    correlation_id: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def to_system_event(self) -> Event:
        """Convert to system Event for EventSystem integration"""
        return Event(
            event_type=self.event_type.value,
            data={
                "user_id": self.data.user_id,
                "agent_id": self.data.agent_id,
                "learning_type": self.data.learning_type,
                "data_points": self.data.data_points,
                "metrics": self.data.metrics,
                "insights": self.data.insights,
                "recommendations": self.data.recommendations,
                "confidence_score": self.data.confidence_score,
                "timestamp": self.data.timestamp.isoformat(),
                "source_component": self.source_component,
                "target_agents": self.target_agents,
                "priority": self.priority
            },
            source=self.source_component,
            correlation_id=self.correlation_id
        )


# Pre-defined learning event factories for common scenarios
class LearningEventFactory:
    """Factory for creating common learning events"""
    
    @staticmethod
    def user_pattern_detected(user_id: str, pattern_type: str, 
                            pattern_data: Dict[str, Any],
                            confidence: float) -> LearningEvent:
        """Create user pattern detection event"""
        return LearningEvent(
            event_type=LearningEventType.USER_PATTERN_DETECTED,
            data=LearningEventData(
                user_id=user_id,
                learning_type=pattern_type,
                data_points=pattern_data,
                confidence_score=confidence,
                insights=[f"Detected {pattern_type} pattern for user {user_id}"]
            ),
            source_component="learning_system"
        )
    
    @staticmethod
    def model_training_completed(model_name: str, performance_metrics: Dict[str, float],
                               improvements: List[str]) -> LearningEvent:
        """Create model training completion event"""
        return LearningEvent(
            event_type=LearningEventType.MODEL_TRAINING_COMPLETED,
            data=LearningEventData(
                learning_type=model_name,
                metrics=performance_metrics,
                insights=improvements,
                confidence_score=performance_metrics.get("accuracy", 0.0)
            ),
            source_component="learning_system"
        )
    
    @staticmethod
    def adaptation_applied(agent_id: str, adaptation_type: str,
                         before_metrics: Dict[str, float],
                         after_metrics: Dict[str, float]) -> LearningEvent:
        """Create adaptation application event"""
        improvements = []
        for metric, after_value in after_metrics.items():
            before_value = before_metrics.get(metric, 0.0)
            if after_value > before_value:
                improvement = ((after_value - before_value) / before_value) * 100
                improvements.append(f"{metric} improved by {improvement:.1f}%")
        
        return LearningEvent(
            event_type=LearningEventType.ADAPTATION_APPLIED,
            data=LearningEventData(
                agent_id=agent_id,
                learning_type=adaptation_type,
                data_points={"before": before_metrics, "after": after_metrics},
                metrics=after_metrics,
                insights=improvements
            ),
            source_component="learning_system",
            target_agents=[agent_id]
        )
    
    @staticmethod
    def knowledge_learned(knowledge_type: str, knowledge_data: Dict[str, Any],
                        source_interactions: int) -> LearningEvent:
        """Create knowledge learning event"""
        return LearningEvent(
            event_type=LearningEventType.KNOWLEDGE_LEARNED,
            data=LearningEventData(
                learning_type=knowledge_type,
                data_points=knowledge_data,
                metrics={"source_interactions": float(source_interactions)},
                insights=[f"Learned new {knowledge_type} knowledge from {source_interactions} interactions"]
            ),
            source_component="learning_system"
        )
    
    @staticmethod
    def performance_improvement_detected(component: str, improvement_type: str,
                                       improvement_percentage: float,
                                       recommendations: List[str]) -> LearningEvent:
        """Create performance improvement detection event"""
        return LearningEvent(
            event_type=LearningEventType.PERFORMANCE_IMPROVEMENT_DETECTED,
            data=LearningEventData(
                agent_id=component,
                learning_type=improvement_type,
                metrics={"improvement_percentage": improvement_percentage},
                recommendations=recommendations,
                confidence_score=min(improvement_percentage / 100.0, 1.0)
            ),
            source_component="learning_system"
        )