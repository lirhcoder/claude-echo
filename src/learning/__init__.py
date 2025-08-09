"""Learning System - Intelligent Learning and Adaptation Module

The Learning System provides:
- Multi-user learning data isolation
- Pluggable learning algorithms
- Performance analytics and insights
- Adaptive behavior optimization
- Knowledge base management
"""

from .base_learner import BaseLearner, LearningContext, LearningResult
from .learning_data_manager import LearningDataManager, LearningData, UserLearningProfile
from .learning_events import LearningEventType, LearningEvent, LearningEventData
from .adaptive_behavior import AdaptiveBehaviorManager, BehaviorPattern, AdaptationStrategy

__version__ = "1.0.0"
__all__ = [
    "BaseLearner",
    "LearningContext", 
    "LearningResult",
    "LearningDataManager",
    "LearningData",
    "UserLearningProfile", 
    "LearningEventType",
    "LearningEvent",
    "LearningEventData",
    "AdaptiveBehaviorManager",
    "BehaviorPattern",
    "AdaptationStrategy"
]