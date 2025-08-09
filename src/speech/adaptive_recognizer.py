"""Adaptive Recognizer - Dynamic Speech Recognition with Learning Integration

Implements an adaptive speech recognition system that integrates learned patterns
from voice profiles, pronunciation patterns, accent characteristics, and context
to provide personalized recognition with improved accuracy.
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import uuid

from loguru import logger

from ..core.event_system import EventSystem
from .learning_types import (
    PersonalizedVoiceProfile, AdaptationParameters, SpeechLearningData,
    VoiceCharacteristics, PronunciationPattern
)
from .voice_profile_learner import VoiceProfileLearner
from .pronunciation_pattern_learner import PronunciationPatternLearner
from .accent_adaptation_learner import AccentAdaptationLearner
from .speech_context_learner import SpeechContextLearner
from .recognizer import SpeechRecognizer, RecognitionResult, RecognitionConfig


class AdaptiveRecognizer:
    """
    Adaptive speech recognition system that integrates learning components.
    
    Combines:
    - Base speech recognition (Whisper)
    - Voice profile adaptations
    - Pronunciation pattern corrections
    - Accent-specific adjustments
    - Context-aware recognition
    """
    
    def __init__(self, 
                 event_system: EventSystem,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize the adaptive recognizer."""
        self.event_system = event_system
        self.config = config or {}
        
        # Base recognizer
        self.base_recognizer: Optional[SpeechRecognizer] = None
        
        # Learning components
        self.voice_profile_learner: Optional[VoiceProfileLearner] = None
        self.pronunciation_learner: Optional[PronunciationPatternLearner] = None
        self.accent_learner: Optional[AccentAdaptationLearner] = None
        self.context_learner: Optional[SpeechContextLearner] = None
        
        # Adaptation settings
        self.adaptation_enabled = self.config.get('adaptation_enabled', True)
        self.real_time_learning = self.config.get('real_time_learning', True)
        self.confidence_boost_threshold = self.config.get('confidence_boost_threshold', 0.7)
        self.adaptation_strength = self.config.get('adaptation_strength', 0.3)
        
        # Recognition state
        self.user_profiles: Dict[str, PersonalizedVoiceProfile] = {}
        self.adaptation_cache: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Performance tracking
        self.adaptive_stats = {
            'recognitions_with_adaptation': 0,
            'accuracy_improvements': 0,
            'confidence_boosts': 0,
            'pronunciation_corrections': 0,
            'context_predictions': 0
        }
        
        logger.info("AdaptiveRecognizer initialized")
    
    async def initialize(self) -> bool:
        """Initialize the adaptive recognizer and learning components."""
        try:
            # Initialize base recognizer
            recognition_config = RecognitionConfig(
                model=self.config.get('model', 'base'),
                programming_keywords=True,
                abbreviation_expansion=True,
                punctuation_inference=True
            )
            
            self.base_recognizer = SpeechRecognizer(
                config=recognition_config,
                event_system=self.event_system
            )
            
            if not await self.base_recognizer.initialize():
                return False
            
            # Initialize learning components
            await self._initialize_learning_components()
            
            # Load existing user profiles
            await self._load_user_profiles()
            
            logger.info("Adaptive recognizer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize adaptive recognizer: {e}")
            return False
    
    async def _initialize_learning_components(self) -> None:
        """Initialize learning components."""
        try:
            # Voice profile learner
            self.voice_profile_learner = VoiceProfileLearner(
                learner_id="adaptive_voice_profile",
                event_system=self.event_system,
                config=self.config.get('voice_profile_config', {})
            )
            await self.voice_profile_learner.initialize()
            
            # Pronunciation pattern learner
            self.pronunciation_learner = PronunciationPatternLearner(
                learner_id="adaptive_pronunciation",
                event_system=self.event_system,
                config=self.config.get('pronunciation_config', {})
            )
            await self.pronunciation_learner.initialize()
            
            # Accent adaptation learner
            self.accent_learner = AccentAdaptationLearner(
                learner_id="adaptive_accent",
                event_system=self.event_system,
                config=self.config.get('accent_config', {})
            )
            await self.accent_learner.initialize()
            
            # Context learner
            self.context_learner = SpeechContextLearner(
                learner_id="adaptive_context",
                event_system=self.event_system,
                config=self.config.get('context_config', {})
            )
            await self.context_learner.initialize()
            
            logger.info("Learning components initialized")
            
        except Exception as e:
            logger.error(f"Learning component initialization failed: {e}")
            raise
    
    async def recognize_adaptive(self, 
                               user_id: str,
                               audio_data: Optional[np.ndarray] = None,
                               duration: Optional[float] = None,
                               context: Optional[str] = None) -> Optional[RecognitionResult]:
        """
        Perform adaptive speech recognition.
        
        Args:
            user_id: User identifier for personalization
            audio_data: Audio data to recognize (if None, uses buffer)
            duration: Duration to process from buffer
            context: Optional context hint
            
        Returns:
            Enhanced recognition result with adaptations
        """
        try:
            if not self.base_recognizer:
                logger.error("Base recognizer not initialized")
                return None
            
            # Get user profile
            user_profile = await self._get_or_create_user_profile(user_id)
            
            # Prepare adapted recognition parameters
            adapted_config = await self._prepare_adapted_config(user_profile, context)
            
            # Perform base recognition
            base_result = None
            if audio_data is not None:
                # Use provided audio data
                # Note: This would require extending the base recognizer to accept audio directly
                # For now, we'll use the buffer-based approach
                base_result = await self.base_recognizer.recognize_from_buffer(duration)
            else:
                base_result = await self.base_recognizer.recognize_from_buffer(duration)
            
            if not base_result:
                return None
            
            # Apply adaptations to the result
            adapted_result = await self._apply_adaptations(base_result, user_profile, context)
            
            # Learn from this recognition
            if self.real_time_learning:
                await self._learn_from_recognition(adapted_result, user_profile, context, audio_data)
            
            # Update statistics
            self.adaptive_stats['recognitions_with_adaptation'] += 1
            
            return adapted_result
            
        except Exception as e:
            logger.error(f"Adaptive recognition failed: {e}")
            return None
    
    async def _prepare_adapted_config(self, 
                                    user_profile: PersonalizedVoiceProfile,
                                    context: Optional[str]) -> RecognitionConfig:
        """Prepare recognition configuration with user adaptations."""
        try:
            base_config = self.base_recognizer.config
            adaptation_params = user_profile.adaptation_parameters
            
            # Create adapted configuration
            adapted_config = RecognitionConfig(
                model=base_config.model,
                language=base_config.language,
                timeout=base_config.timeout,
                use_gpu=base_config.use_gpu,
                programming_keywords=base_config.programming_keywords,
                abbreviation_expansion=base_config.abbreviation_expansion,
                punctuation_inference=base_config.punctuation_inference,
                
                # Apply adaptations
                beam_size=max(1, base_config.beam_size + adaptation_params.beam_size_adjustment),
                temperature=max(0.0, min(1.0, base_config.temperature + adaptation_params.temperature_adjustment)),
                length_penalty=base_config.length_penalty + adaptation_params.length_penalty_adjustment
            )
            
            return adapted_config
            
        except Exception as e:
            logger.error(f"Config adaptation failed: {e}")
            return self.base_recognizer.config
    
    async def _apply_adaptations(self, 
                               base_result: RecognitionResult,
                               user_profile: PersonalizedVoiceProfile,
                               context: Optional[str]) -> RecognitionResult:
        """Apply learned adaptations to recognition result."""
        try:
            adapted_result = RecognitionResult(
                text=base_result.text,
                confidence=base_result.confidence,
                language=base_result.language,
                processing_time=base_result.processing_time,
                audio_duration=base_result.audio_duration,
                model_used=base_result.model_used,
                alternative_texts=base_result.alternative_texts.copy(),
                word_timestamps=base_result.word_timestamps.copy()
            )
            
            improvements = []
            
            # Apply pronunciation corrections
            if self.pronunciation_learner:
                corrected_text = await self.pronunciation_learner.apply_pronunciation_corrections(
                    user_profile.user_id, adapted_result.text
                )
                if corrected_text != adapted_result.text:
                    adapted_result.text = corrected_text
                    improvements.append("pronunciation_correction")
                    self.adaptive_stats['pronunciation_corrections'] += 1
            
            # Apply confidence adjustments based on voice profile
            confidence_adjustment = await self._calculate_confidence_adjustment(
                base_result, user_profile, context
            )
            if confidence_adjustment != 0:
                adapted_result.confidence = max(0.0, min(1.0, 
                    adapted_result.confidence + confidence_adjustment))
                improvements.append(f"confidence_adjusted_{confidence_adjustment:+.2f}")
                self.adaptive_stats['confidence_boosts'] += 1
            
            # Apply context-based post-processing
            if self.context_learner and context:
                context_result = await self._apply_context_adaptations(
                    adapted_result, user_profile.user_id, context
                )
                if context_result:
                    adapted_result = context_result
                    improvements.append("context_adaptation")
                    self.adaptive_stats['context_predictions'] += 1
            
            # Apply accent-based adjustments
            if self.accent_learner:
                accent_result = await self._apply_accent_adaptations(
                    adapted_result, user_profile
                )
                if accent_result:
                    adapted_result = accent_result
                    improvements.append("accent_adaptation")
            
            # Add adaptation metadata
            if improvements:
                adapted_result.metadata = getattr(adapted_result, 'metadata', {})
                adapted_result.metadata['adaptations_applied'] = improvements
                self.adaptive_stats['accuracy_improvements'] += 1
            
            return adapted_result
            
        except Exception as e:
            logger.error(f"Adaptation application failed: {e}")
            return base_result
    
    async def _calculate_confidence_adjustment(self, 
                                             result: RecognitionResult,
                                             user_profile: PersonalizedVoiceProfile,
                                             context: Optional[str]) -> float:
        """Calculate confidence score adjustment based on user profile."""
        try:
            adjustment = 0.0
            adaptation_params = user_profile.adaptation_parameters
            
            # Base adjustment from user parameters
            adjustment += adaptation_params.confidence_adjustment * self.adaptation_strength
            
            # Adjustment based on voice characteristics confidence
            voice_confidence = user_profile.voice_characteristics.confidence_score
            if voice_confidence > 0.8:
                adjustment += 0.05  # Boost confidence for well-known voices
            elif voice_confidence < 0.4:
                adjustment -= 0.05  # Lower confidence for uncertain profiles
            
            # Context-based adjustment
            if context and context in adaptation_params.vocabulary_boost:
                boost = adaptation_params.vocabulary_boost[context]
                adjustment += boost * 0.1  # Convert vocabulary boost to confidence boost
            
            # Pronunciation pattern adjustment
            if self.pronunciation_learner:
                # Check if text contains words with known pronunciation patterns
                patterns = await self.pronunciation_learner.get_pronunciation_patterns(user_profile.user_id)
                for pattern in patterns.values():
                    if pattern.target_word.lower() in result.text.lower():
                        if pattern.occurrence_count >= 3:
                            adjustment += 0.03  # Small boost for known patterns
            
            # Clamp adjustment to reasonable range
            adjustment = max(-0.2, min(0.2, adjustment))
            
            return adjustment
            
        except Exception as e:
            logger.error(f"Confidence adjustment calculation failed: {e}")
            return 0.0
    
    async def _apply_context_adaptations(self, 
                                       result: RecognitionResult,
                                       user_id: str,
                                       context: str) -> Optional[RecognitionResult]:
        """Apply context-based adaptations."""
        try:
            if not self.context_learner:
                return None
            
            # Predict context and compare with provided context
            predicted_context = await self.context_learner.predict_context(user_id, result.text)
            
            if predicted_context and predicted_context != context:
                # Context mismatch - might need adjustment
                vocab_suggestions = await self.context_learner.get_vocabulary_suggestions(
                    user_id, context
                )
                
                # Simple word replacement based on context
                modified_text = result.text
                for suggestion in vocab_suggestions[:3]:  # Top 3 suggestions
                    if suggestion.lower() in result.text.lower():
                        # Boost confidence slightly for context-appropriate terms
                        result.confidence = min(1.0, result.confidence + 0.02)
                        break
            
            return result
            
        except Exception as e:
            logger.error(f"Context adaptation failed: {e}")
            return None
    
    async def _apply_accent_adaptations(self, 
                                      result: RecognitionResult,
                                      user_profile: PersonalizedVoiceProfile) -> Optional[RecognitionResult]:
        """Apply accent-based adaptations."""
        try:
            if not self.accent_learner:
                return None
            
            # Get accent adaptations for the user
            accent_adaptations = self.accent_learner.get_accent_adaptations(user_profile.user_id)
            
            if not accent_adaptations:
                return None
            
            # Apply accent-specific confidence adjustments
            for adaptation in accent_adaptations:
                if adaptation['type'] == 'prosodic_adjustment':
                    # Adjust confidence based on prosodic match
                    result.confidence = min(1.0, result.confidence + 0.02)
                elif adaptation['type'] == 'language_model':
                    # Handle bilingual boost
                    if adaptation['parameter'] == 'bilingual_boost':
                        # Detect language mixing in text
                        chinese_chars = sum(1 for char in result.text if '\u4e00' <= char <= '\u9fff')
                        if chinese_chars > 0:  # Contains Chinese characters
                            result.confidence = min(1.0, result.confidence + adaptation['value'])
            
            return result
            
        except Exception as e:
            logger.error(f"Accent adaptation failed: {e}")
            return None
    
    async def _learn_from_recognition(self, 
                                    result: RecognitionResult,
                                    user_profile: PersonalizedVoiceProfile,
                                    context: Optional[str],
                                    audio_data: Optional[np.ndarray]) -> None:
        """Learn from recognition result for continuous improvement."""
        try:
            # Create learning data
            learning_data = SpeechLearningData(
                data_id=str(uuid.uuid4()),
                user_id=user_profile.user_id,
                audio_features=audio_data,
                audio_duration=result.audio_duration,
                original_text=result.text,
                confidence_scores=[result.confidence],
                alternative_transcriptions=result.alternative_texts,
                context_type=context,
                timestamp=datetime.now(),
                processing_time=result.processing_time,
                model_version=result.model_used
            )
            
            # Learn with voice profile learner
            if self.voice_profile_learner and audio_data is not None:
                voice_learning_data = [learning_data.to_dict()]
                await self.voice_profile_learner.learn(
                    voice_learning_data, 
                    learning_data.to_learning_context()
                )
            
            # Learn with context learner
            if self.context_learner and context:
                context_learning_data = [learning_data.to_dict()]
                await self.context_learner.learn(
                    context_learning_data,
                    learning_data.to_learning_context()
                )
            
            # Update user profile in memory
            self.user_profiles[user_profile.user_id] = user_profile
            
        except Exception as e:
            logger.error(f"Recognition learning failed: {e}")
    
    async def _get_or_create_user_profile(self, user_id: str) -> PersonalizedVoiceProfile:
        """Get existing user profile or create new one."""
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]
        
        # Try to load from voice profile learner
        if self.voice_profile_learner:
            existing_profile = await self.voice_profile_learner.get_voice_profile(user_id)
            if existing_profile:
                self.user_profiles[user_id] = existing_profile
                return existing_profile
        
        # Create new profile
        new_profile = PersonalizedVoiceProfile(user_id=user_id)
        self.user_profiles[user_id] = new_profile
        
        return new_profile
    
    async def _load_user_profiles(self) -> None:
        """Load existing user profiles."""
        try:
            # In a full implementation, this would load from persistent storage
            logger.info("User profiles loaded")
        except Exception as e:
            logger.error(f"Failed to load user profiles: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self.base_recognizer:
                await self.base_recognizer.cleanup()
            
            if self.voice_profile_learner:
                await self.voice_profile_learner.shutdown()
            
            if self.pronunciation_learner:
                await self.pronunciation_learner.shutdown()
            
            if self.accent_learner:
                await self.accent_learner.shutdown()
            
            if self.context_learner:
                await self.context_learner.shutdown()
            
            logger.info("Adaptive recognizer cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    # Public API methods
    
    async def provide_user_feedback(self, 
                                  user_id: str,
                                  original_result: str,
                                  corrected_text: Optional[str],
                                  satisfaction_rating: Optional[int]) -> None:
        """Provide user feedback for learning improvement."""
        try:
            if corrected_text and corrected_text != original_result:
                # Create correction data for pronunciation learner
                if self.pronunciation_learner:
                    correction_data = [{
                        'user_id': user_id,
                        'original_text': original_result,
                        'corrected_text': corrected_text,
                        'user_correction': corrected_text,
                        'timestamp': datetime.now().isoformat()
                    }]
                    
                    from ..learning.base_learner import LearningContext
                    context = LearningContext(user_id=user_id)
                    await self.pronunciation_learner.learn(correction_data, context)
            
            # Update voice profile with satisfaction
            if self.voice_profile_learner and satisfaction_rating is not None:
                await self.voice_profile_learner.update_user_feedback(
                    user_id, original_result, corrected_text, satisfaction_rating
                )
            
            logger.info(f"User feedback processed for {user_id}")
            
        except Exception as e:
            logger.error(f"User feedback processing failed: {e}")
    
    async def get_user_profile_summary(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of user's voice profile."""
        try:
            profile = self.user_profiles.get(user_id)
            if not profile:
                return None
            
            return profile.get_summary_stats()
            
        except Exception as e:
            logger.error(f"Profile summary failed: {e}")
            return None
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get adaptation performance statistics."""
        total_recognitions = max(1, self.adaptive_stats['recognitions_with_adaptation'])
        
        return {
            **self.adaptive_stats,
            'adaptation_rate': (self.adaptive_stats['accuracy_improvements'] / total_recognitions) * 100,
            'confidence_boost_rate': (self.adaptive_stats['confidence_boosts'] / total_recognitions) * 100,
            'pronunciation_correction_rate': (self.adaptive_stats['pronunciation_corrections'] / total_recognitions) * 100,
            'active_user_profiles': len(self.user_profiles)
        }
    
    async def set_adaptation_strength(self, strength: float) -> None:
        """Set adaptation strength (0.0 to 1.0)."""
        self.adaptation_strength = max(0.0, min(1.0, strength))
        logger.info(f"Adaptation strength set to {self.adaptation_strength}")
    
    async def enable_real_time_learning(self, enabled: bool) -> None:
        """Enable or disable real-time learning."""
        self.real_time_learning = enabled
        logger.info(f"Real-time learning {'enabled' if enabled else 'disabled'}")