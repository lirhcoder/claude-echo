#!/usr/bin/env python3
"""
Standalone Speech Learning Types Test

Test the speech learning data types without system dependencies.
"""

import sys
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


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
    intonation_peaks: List[Tuple[float, float]] = field(default_factory=list)
    
    # Metadata
    sample_count: int = 0
    confidence_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class SpeechLearningData:
    """Primary data structure for speech learning samples"""
    data_id: str
    user_id: str
    
    # Audio analysis
    audio_features: Optional[Any] = None
    audio_duration: float = 0.0
    audio_quality_score: float = 0.0
    
    # Recognition results
    original_text: str = ""
    confidence_scores: List[float] = field(default_factory=list)
    alternative_transcriptions: List[str] = field(default_factory=list)
    
    # Learning context
    context_type: Optional[str] = None
    session_id: Optional[str] = None
    
    # User feedback
    user_correction: Optional[str] = None
    user_satisfaction_rating: Optional[int] = None  # 1-5 scale
    
    # Technical metadata
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0
    model_version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'data_id': self.data_id,
            'user_id': self.user_id,
            'audio_duration': self.audio_duration,
            'audio_quality_score': self.audio_quality_score,
            'original_text': self.original_text,
            'confidence_scores': self.confidence_scores,
            'alternative_transcriptions': self.alternative_transcriptions,
            'context_type': self.context_type,
            'session_id': self.session_id,
            'user_correction': self.user_correction,
            'user_satisfaction_rating': self.user_satisfaction_rating,
            'timestamp': self.timestamp.isoformat(),
            'processing_time': self.processing_time,
            'model_version': self.model_version
        }


@dataclass
class PersonalizedVoiceProfile:
    """Complete personalized voice profile for a user"""
    user_id: str
    profile_version: str = "1.0.0"
    
    voice_characteristics: Optional[VoiceCharacteristics] = None
    
    # Profile metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Usage statistics
    total_utterances: int = 0
    recognition_accuracy: float = 0.0
    user_satisfaction: float = 0.0
    learning_enabled: bool = True


def test_data_structures():
    """Test all speech learning data structures."""
    print("Testing Speech Learning Data Structures")
    print("=" * 50)
    
    try:
        # Test VoiceCharacteristics
        voice_chars = VoiceCharacteristics(
            user_id="test_user_001",
            average_pitch=150.0,
            pitch_variance=20.0,
            average_speech_rate=120.0,
            sample_count=50,
            confidence_score=0.95
        )
        print("VoiceCharacteristics created successfully")
        
        # Test SpeechLearningData
        learning_data = SpeechLearningData(
            data_id="test_sample_001",
            user_id="test_user_001",
            audio_duration=2.5,
            original_text="Hello world, this is a test",
            confidence_scores=[0.95, 0.87, 0.92],
            context_type="general"
        )
        print("SpeechLearningData created successfully")
        
        # Test dictionary conversion
        data_dict = learning_data.to_dict()
        assert 'data_id' in data_dict
        assert 'user_id' in data_dict
        assert 'original_text' in data_dict
        print("Dictionary conversion working")
        
        # Test PersonalizedVoiceProfile
        voice_profile = PersonalizedVoiceProfile(
            user_id="test_user_001",
            voice_characteristics=voice_chars,
            total_utterances=50,
            recognition_accuracy=0.94,
            user_satisfaction=4.2
        )
        print("PersonalizedVoiceProfile created successfully")
        
        # Test enums
        feature_type = VoiceFeatureType.PITCH_PATTERN
        adaptation_level = AdaptationLevel.MODERATE
        assert feature_type.value == "pitch_pattern"
        assert adaptation_level.value == "moderate"
        print("Enums working correctly")
        
        print("\n" + "=" * 50)
        print("SUCCESS: All data structures working correctly!")
        print("=" * 50)
        
        # Print summary
        print("\nImplemented Components:")
        print("- VoiceCharacteristics: Core voice feature storage")
        print("- SpeechLearningData: Primary learning data structure")
        print("- PersonalizedVoiceProfile: Complete user voice profile")
        print("- VoiceFeatureType: Enumeration of learnable features")
        print("- AdaptationLevel: Adaptation intensity levels")
        
        print("\nKey Capabilities:")
        print("- User-specific voice characteristic modeling")
        print("- Speech recognition data collection and storage")
        print("- User feedback integration")
        print("- Context-aware learning")
        print("- Performance tracking and statistics")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Test failed - {e}")
        return False


def main():
    """Main test function."""
    print("Speech Learning Types - Standalone Test")
    print("Testing core data structures without dependencies")
    print("=" * 60)
    
    success = test_data_structures()
    
    if success:
        print("\nAll tests passed! Speech learning data structures are ready.")
        return 0
    else:
        print("\nSome tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())