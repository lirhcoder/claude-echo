"""Speech Learning Types - Data Structures for Personalized Speech Learning

Defines the core data structures and types used throughout the speech learning system,
including voice profiles, pronunciation patterns, and adaptation parameters.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np

from ..learning.base_learner import LearningContext


class VoiceFeatureType(Enum):
    """Types of voice features that can be learned"""
    PITCH_PATTERN = "pitch_pattern"
    SPEECH_RATE = "speech_rate"
    VOLUME_LEVEL = "volume_level"
    ACCENT_MARKERS = "accent_markers"
    PRONUNCIATION_STYLE = "pronunciation_style"
    INTONATION_PATTERN = "intonation_pattern"


class AdaptationLevel(Enum):
    """Levels of adaptation intensity"""
    MINIMAL = "minimal"        # Basic adaptations only
    MODERATE = "moderate"      # Standard personalization
    AGGRESSIVE = "aggressive"  # Maximum personalization
    CUSTOM = "custom"         # User-defined parameters


@dataclass
class VoiceCharacteristics:
    """Core voice characteristics for a user"""
    user_id: str
    
    # Acoustic features
    average_pitch: float = 0.0
    pitch_variance: float = 0.0
    average_speech_rate: float = 0.0  # words per minute
    speech_rate_variance: float = 0.0
    average_volume: float = 0.0
    volume_variance: float = 0.0
    
    # Speaking patterns
    pause_patterns: List[float] = field(default_factory=list)
    stress_patterns: Dict[str, float] = field(default_factory=dict)
    intonation_peaks: List[Tuple[float, float]] = field(default_factory=list)  # (time, frequency)
    
    # Language characteristics
    accent_markers: Dict[str, float] = field(default_factory=dict)  # phoneme -> deviation score
    code_switching_frequency: float = 0.0  # Chinese-English switching rate
    
    # Quality metrics
    confidence_score: float = 0.0
    sample_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "user_id": self.user_id,
            "average_pitch": self.average_pitch,
            "pitch_variance": self.pitch_variance,
            "average_speech_rate": self.average_speech_rate,
            "speech_rate_variance": self.speech_rate_variance,
            "average_volume": self.average_volume,
            "volume_variance": self.volume_variance,
            "pause_patterns": self.pause_patterns,
            "stress_patterns": self.stress_patterns,
            "intonation_peaks": [(t, f) for t, f in self.intonation_peaks],
            "accent_markers": self.accent_markers,
            "code_switching_frequency": self.code_switching_frequency,
            "confidence_score": self.confidence_score,
            "sample_count": self.sample_count,
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class PronunciationPattern:
    """Pronunciation pattern data structure"""
    pattern_id: str
    user_id: str
    
    # Pattern details
    target_word: str
    actual_pronunciation: str
    expected_pronunciation: str
    phonetic_deviation: List[Tuple[str, str]] = field(default_factory=list)  # (expected, actual)
    
    # Context information
    context_words: List[str] = field(default_factory=list)
    language_context: str = "mixed"  # "english", "chinese", "mixed"
    
    # Statistics
    occurrence_count: int = 1
    correction_success_rate: float = 0.0
    first_observed: datetime = field(default_factory=datetime.now)
    last_observed: datetime = field(default_factory=datetime.now)
    
    # Learning metrics
    adaptation_priority: float = 1.0  # Higher = more important to adapt
    user_feedback_score: Optional[float] = None  # User satisfaction with corrections
    
    def update_occurrence(self) -> None:
        """Update occurrence statistics"""
        self.occurrence_count += 1
        self.last_observed = datetime.now()
    
    def calculate_priority(self) -> float:
        """Calculate adaptation priority based on various factors"""
        # Factors: frequency, recency, correction success rate, user feedback
        frequency_score = min(self.occurrence_count / 10.0, 1.0)
        
        days_since_last = (datetime.now() - self.last_observed).days
        recency_score = max(0.0, 1.0 - (days_since_last / 30.0))
        
        correction_score = 1.0 - self.correction_success_rate
        
        feedback_score = self.user_feedback_score or 0.5
        
        self.adaptation_priority = (
            frequency_score * 0.3 + 
            recency_score * 0.2 + 
            correction_score * 0.3 + 
            feedback_score * 0.2
        )
        
        return self.adaptation_priority


@dataclass
class SpeechContextPattern:
    """Speech context and intent patterns"""
    pattern_id: str
    user_id: str
    
    # Context information
    context_type: str  # "coding", "file_ops", "system", "query"
    common_phrases: List[str] = field(default_factory=list)
    vocabulary_preferences: Dict[str, str] = field(default_factory=dict)  # preferred_term -> alternatives
    
    # Usage patterns
    frequency_by_time: Dict[str, int] = field(default_factory=dict)  # hour -> count
    success_rate: float = 0.0
    common_corrections: Dict[str, str] = field(default_factory=dict)  # original -> corrected
    
    # Adaptation parameters
    confidence_threshold: float = 0.7
    custom_grammar_rules: List[str] = field(default_factory=list)
    intent_disambiguation_hints: Dict[str, List[str]] = field(default_factory=dict)
    
    # Statistics
    total_uses: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class AdaptationParameters:
    """Parameters controlling speech recognition adaptation"""
    user_id: str
    adaptation_level: AdaptationLevel = AdaptationLevel.MODERATE
    
    # Recognition parameters
    confidence_adjustment: float = 0.0  # -1.0 to 1.0
    language_weight_adjustment: Dict[str, float] = field(default_factory=dict)  # lang -> weight
    vocabulary_boost: Dict[str, float] = field(default_factory=dict)  # word -> boost_factor
    
    # Whisper model parameters
    temperature_adjustment: float = 0.0  # -1.0 to 1.0
    beam_size_adjustment: int = 0  # -5 to 5
    length_penalty_adjustment: float = 0.0  # -1.0 to 1.0
    
    # Context-specific parameters
    programming_context_boost: float = 1.0
    accent_adaptation_strength: float = 1.0
    pronunciation_correction_aggressiveness: float = 1.0
    
    # Real-time adaptation
    online_learning_rate: float = 0.01
    adaptation_decay_rate: float = 0.95  # How quickly old adaptations fade
    minimum_confidence_for_adaptation: float = 0.6
    
    # User preferences
    enable_real_time_adaptation: bool = True
    enable_pronunciation_correction: bool = True
    enable_vocabulary_learning: bool = True
    enable_context_learning: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "user_id": self.user_id,
            "adaptation_level": self.adaptation_level.value,
            "confidence_adjustment": self.confidence_adjustment,
            "language_weight_adjustment": self.language_weight_adjustment,
            "vocabulary_boost": self.vocabulary_boost,
            "temperature_adjustment": self.temperature_adjustment,
            "beam_size_adjustment": self.beam_size_adjustment,
            "length_penalty_adjustment": self.length_penalty_adjustment,
            "programming_context_boost": self.programming_context_boost,
            "accent_adaptation_strength": self.accent_adaptation_strength,
            "pronunciation_correction_aggressiveness": self.pronunciation_correction_aggressiveness,
            "online_learning_rate": self.online_learning_rate,
            "adaptation_decay_rate": self.adaptation_decay_rate,
            "minimum_confidence_for_adaptation": self.minimum_confidence_for_adaptation,
            "enable_real_time_adaptation": self.enable_real_time_adaptation,
            "enable_pronunciation_correction": self.enable_pronunciation_correction,
            "enable_vocabulary_learning": self.enable_vocabulary_learning,
            "enable_context_learning": self.enable_context_learning
        }


@dataclass
class SpeechLearningData:
    """Core data structure for speech learning events"""
    data_id: str
    user_id: str
    session_id: Optional[str] = None
    
    # Audio data
    audio_features: Optional[np.ndarray] = None  # Acoustic features extracted from audio
    audio_duration: float = 0.0
    
    # Recognition results
    original_text: str = ""
    corrected_text: Optional[str] = None
    confidence_scores: List[float] = field(default_factory=list)
    alternative_transcriptions: List[str] = field(default_factory=list)
    
    # Context information
    context_type: Optional[str] = None
    preceding_context: List[str] = field(default_factory=list)
    intent_classification: Optional[str] = None
    
    # User feedback
    user_accepted_result: Optional[bool] = None
    user_correction: Optional[str] = None
    user_satisfaction_rating: Optional[int] = None  # 1-5 scale
    
    # Learning targets
    pronunciation_errors: List[PronunciationPattern] = field(default_factory=list)
    vocabulary_gaps: List[str] = field(default_factory=list)
    context_mismatches: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0
    model_version: str = "unknown"
    
    def to_learning_context(self) -> LearningContext:
        """Convert to learning context for base learner"""
        return LearningContext(
            user_id=self.user_id,
            session_id=self.session_id,
            interaction_type="speech_recognition",
            metadata={
                "audio_duration": self.audio_duration,
                "confidence_scores": self.confidence_scores,
                "context_type": self.context_type,
                "processing_time": self.processing_time,
                "model_version": self.model_version
            },
            timestamp=self.timestamp
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "data_id": self.data_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "audio_duration": self.audio_duration,
            "original_text": self.original_text,
            "corrected_text": self.corrected_text,
            "confidence_scores": self.confidence_scores,
            "alternative_transcriptions": self.alternative_transcriptions,
            "context_type": self.context_type,
            "preceding_context": self.preceding_context,
            "intent_classification": self.intent_classification,
            "user_accepted_result": self.user_accepted_result,
            "user_correction": self.user_correction,
            "user_satisfaction_rating": self.user_satisfaction_rating,
            "pronunciation_errors": [p.__dict__ for p in self.pronunciation_errors],
            "vocabulary_gaps": self.vocabulary_gaps,
            "context_mismatches": self.context_mismatches,
            "timestamp": self.timestamp.isoformat(),
            "processing_time": self.processing_time,
            "model_version": self.model_version
        }


@dataclass
class PersonalizedVoiceProfile:
    """Complete personalized voice profile for a user"""
    user_id: str
    profile_version: str = "1.0.0"
    
    # Core characteristics
    voice_characteristics: VoiceCharacteristics = None
    adaptation_parameters: AdaptationParameters = None
    
    # Learned patterns
    pronunciation_patterns: Dict[str, PronunciationPattern] = field(default_factory=dict)
    context_patterns: Dict[str, SpeechContextPattern] = field(default_factory=dict)
    
    # Performance tracking
    recognition_accuracy: float = 0.0
    user_satisfaction: float = 0.0
    adaptation_effectiveness: float = 0.0
    
    # Configuration
    learning_enabled: bool = True
    auto_adaptation: bool = True
    privacy_mode: bool = False  # If true, limit data collection
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    total_sessions: int = 0
    total_utterances: int = 0
    
    def __post_init__(self):
        """Initialize components if not provided"""
        if self.voice_characteristics is None:
            self.voice_characteristics = VoiceCharacteristics(self.user_id)
        if self.adaptation_parameters is None:
            self.adaptation_parameters = AdaptationParameters(self.user_id)
    
    def update_performance_metrics(self, accuracy: float, satisfaction: float):
        """Update performance metrics with new measurements"""
        # Exponential moving average
        alpha = 0.1  # Learning rate
        self.recognition_accuracy = (
            alpha * accuracy + (1 - alpha) * self.recognition_accuracy
        )
        self.user_satisfaction = (
            alpha * satisfaction + (1 - alpha) * self.user_satisfaction
        )
        
        # Calculate adaptation effectiveness
        self.adaptation_effectiveness = (
            self.recognition_accuracy * 0.6 + self.user_satisfaction * 0.4
        )
        
        self.last_updated = datetime.now()
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the profile"""
        return {
            "user_id": self.user_id,
            "profile_version": self.profile_version,
            "recognition_accuracy": round(self.recognition_accuracy, 3),
            "user_satisfaction": round(self.user_satisfaction, 3),
            "adaptation_effectiveness": round(self.adaptation_effectiveness, 3),
            "total_sessions": self.total_sessions,
            "total_utterances": self.total_utterances,
            "pronunciation_patterns_count": len(self.pronunciation_patterns),
            "context_patterns_count": len(self.context_patterns),
            "days_since_creation": (datetime.now() - self.created_at).days,
            "days_since_update": (datetime.now() - self.last_updated).days,
            "learning_enabled": self.learning_enabled,
            "auto_adaptation": self.auto_adaptation
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "user_id": self.user_id,
            "profile_version": self.profile_version,
            "voice_characteristics": self.voice_characteristics.to_dict(),
            "adaptation_parameters": self.adaptation_parameters.to_dict(),
            "pronunciation_patterns": {k: v.__dict__ for k, v in self.pronunciation_patterns.items()},
            "context_patterns": {k: v.__dict__ for k, v in self.context_patterns.items()},
            "recognition_accuracy": self.recognition_accuracy,
            "user_satisfaction": self.user_satisfaction,
            "adaptation_effectiveness": self.adaptation_effectiveness,
            "learning_enabled": self.learning_enabled,
            "auto_adaptation": self.auto_adaptation,
            "privacy_mode": self.privacy_mode,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "total_sessions": self.total_sessions,
            "total_utterances": self.total_utterances
        }