"""Speech Learning Manager - Central Management for Speech Learning System

Coordinates all speech learning components and provides a unified interface
for the personalized speech recognition and learning system.
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import uuid

from loguru import logger

from ..core.event_system import EventSystem
from ..learning.learning_data_manager import LearningDataManager
from ..learning.base_learner import LearningContext
from .adaptive_recognizer import AdaptiveRecognizer
from .voice_profile_learner import VoiceProfileLearner
from .pronunciation_pattern_learner import PronunciationPatternLearner  
from .accent_adaptation_learner import AccentAdaptationLearner
from .speech_context_learner import SpeechContextLearner
from .learning_types import SpeechLearningData, PersonalizedVoiceProfile


class SpeechLearningManager:
    """
    Central manager for the speech learning system.
    
    Coordinates:
    - Adaptive speech recognition
    - Voice profile learning
    - Pronunciation pattern learning
    - Accent adaptation
    - Context learning
    - Data management
    - Performance monitoring
    """
    
    def __init__(self, 
                 event_system: EventSystem,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize the speech learning manager."""
        self.event_system = event_system
        self.config = config or {}
        
        # Core components
        self.adaptive_recognizer: Optional[AdaptiveRecognizer] = None
        self.learning_data_manager: Optional[LearningDataManager] = None
        
        # Learning components
        self.voice_profile_learner: Optional[VoiceProfileLearner] = None
        self.pronunciation_learner: Optional[PronunciationPatternLearner] = None
        self.accent_learner: Optional[AccentAdaptationLearner] = None
        self.context_learner: Optional[SpeechContextLearner] = None
        
        # System state
        self.is_initialized = False
        self.learning_enabled = self.config.get('learning_enabled', True)
        self.auto_adaptation = self.config.get('auto_adaptation', True)
        
        # Performance monitoring
        self.system_stats = {
            'total_recognitions': 0,
            'learning_sessions': 0,
            'user_feedback_events': 0,
            'adaptations_applied': 0,
            'system_improvements': 0
        }
        
        logger.info("SpeechLearningManager initialized")
    
    async def initialize(self) -> bool:
        """Initialize all speech learning components."""
        try:
            logger.info("Initializing speech learning system...")
            
            # Initialize data manager
            self.learning_data_manager = LearningDataManager(
                event_system=self.event_system,
                config=self.config.get('data_manager_config', {})
            )
            await self.learning_data_manager.initialize()
            
            # Initialize learning components
            await self._initialize_learning_components()
            
            # Initialize adaptive recognizer
            self.adaptive_recognizer = AdaptiveRecognizer(
                event_system=self.event_system,
                config=self.config.get('adaptive_recognizer_config', {})
            )
            
            # Set learning components in adaptive recognizer
            self.adaptive_recognizer.voice_profile_learner = self.voice_profile_learner
            self.adaptive_recognizer.pronunciation_learner = self.pronunciation_learner
            self.adaptive_recognizer.accent_learner = self.accent_learner
            self.adaptive_recognizer.context_learner = self.context_learner
            
            if not await self.adaptive_recognizer.initialize():
                raise Exception("Failed to initialize adaptive recognizer")
            
            # Setup event handlers
            await self._setup_event_handlers()
            
            self.is_initialized = True
            logger.info("Speech learning system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Speech learning system initialization failed: {e}")
            return False
    
    async def _initialize_learning_components(self) -> None:
        """Initialize individual learning components."""
        try:
            # Voice profile learner
            self.voice_profile_learner = VoiceProfileLearner(
                learner_id="speech_voice_profile",
                event_system=self.event_system,
                config=self.config.get('voice_profile_config', {})
            )
            await self.voice_profile_learner.initialize()
            
            # Pronunciation pattern learner
            self.pronunciation_learner = PronunciationPatternLearner(
                learner_id="speech_pronunciation",
                event_system=self.event_system,
                config=self.config.get('pronunciation_config', {})
            )
            await self.pronunciation_learner.initialize()
            
            # Accent adaptation learner
            self.accent_learner = AccentAdaptationLearner(
                learner_id="speech_accent",
                event_system=self.event_system,
                config=self.config.get('accent_config', {})
            )
            await self.accent_learner.initialize()
            
            # Context learner
            self.context_learner = SpeechContextLearner(
                learner_id="speech_context",
                event_system=self.event_system,
                config=self.config.get('context_config', {})
            )
            await self.context_learner.initialize()
            
            logger.info("All learning components initialized")
            
        except Exception as e:
            logger.error(f"Learning components initialization failed: {e}")
            raise
    
    async def _setup_event_handlers(self) -> None:
        """Setup event handlers for system integration."""
        try:
            # Listen for recognition events
            await self.event_system.subscribe(
                'speech_recognition_completed',
                self._handle_recognition_event
            )
            
            # Listen for user feedback events
            await self.event_system.subscribe(
                'user_feedback_received',
                self._handle_feedback_event
            )
            
            # Listen for adaptation events
            await self.event_system.subscribe(
                'adaptation_applied',
                self._handle_adaptation_event
            )
            
            logger.info("Event handlers setup completed")
            
        except Exception as e:
            logger.error(f"Event handler setup failed: {e}")
    
    async def recognize_speech(self, 
                             user_id: str,
                             audio_data: Optional[Any] = None,
                             duration: Optional[float] = None,
                             context: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Perform personalized speech recognition.
        
        Args:
            user_id: User identifier
            audio_data: Audio data (optional, uses buffer if None)
            duration: Duration to process
            context: Optional context hint
            
        Returns:
            Recognition result with adaptation information
        """
        try:
            if not self.is_initialized or not self.adaptive_recognizer:
                logger.error("Speech learning system not initialized")
                return None
            
            # Perform adaptive recognition
            result = await self.adaptive_recognizer.recognize_adaptive(
                user_id=user_id,
                audio_data=audio_data,
                duration=duration,
                context=context
            )
            
            if not result:
                return None
            
            # Update statistics
            self.system_stats['total_recognitions'] += 1
            
            # Create learning data for storage
            if self.learning_enabled:
                await self._store_recognition_data(user_id, result, context, audio_data)
            
            # Convert result to dictionary
            result_dict = {
                'text': result.text,
                'confidence': result.confidence,
                'language': result.language,
                'processing_time': result.processing_time,
                'audio_duration': result.audio_duration,
                'model_used': result.model_used,
                'alternative_texts': result.alternative_texts,
                'word_timestamps': result.word_timestamps,
                'adaptations_applied': getattr(result, 'metadata', {}).get('adaptations_applied', []),
                'user_id': user_id,
                'context': context,
                'timestamp': datetime.now().isoformat()
            }
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Speech recognition failed: {e}")
            return None
    
    async def _store_recognition_data(self, 
                                    user_id: str,
                                    result: Any,
                                    context: Optional[str],
                                    audio_data: Optional[Any]) -> None:
        """Store recognition data for learning."""
        try:
            learning_data = SpeechLearningData(
                data_id=str(uuid.uuid4()),
                user_id=user_id,
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
            
            # Store in learning data manager
            if self.learning_data_manager:
                from ..learning.learning_data_manager import LearningData, DataPrivacyLevel
                storage_data = LearningData(
                    user_id=user_id,
                    data_type="speech_recognition",
                    data_content=learning_data.to_dict(),
                    privacy_level=DataPrivacyLevel.PRIVATE
                )
                
                await self.learning_data_manager.store_learning_data(storage_data)
            
        except Exception as e:
            logger.error(f"Recognition data storage failed: {e}")
    
    async def provide_user_feedback(self, 
                                  user_id: str,
                                  original_text: str,
                                  corrected_text: Optional[str] = None,
                                  satisfaction_rating: Optional[int] = None,
                                  context: Optional[str] = None) -> bool:
        """
        Process user feedback for continuous improvement.
        
        Args:
            user_id: User identifier
            original_text: Original recognition result
            corrected_text: User's correction (if any)
            satisfaction_rating: User satisfaction (1-5)
            context: Context of the recognition
            
        Returns:
            True if feedback processed successfully
        """
        try:
            if not self.is_initialized:
                return False
            
            # Process feedback with adaptive recognizer
            if self.adaptive_recognizer:
                await self.adaptive_recognizer.provide_user_feedback(
                    user_id=user_id,
                    original_result=original_text,
                    corrected_text=corrected_text,
                    satisfaction_rating=satisfaction_rating
                )
            
            # Create feedback learning data
            feedback_data = {
                'data_id': str(uuid.uuid4()),
                'user_id': user_id,
                'original_text': original_text,
                'corrected_text': corrected_text,
                'user_correction': corrected_text,
                'user_satisfaction_rating': satisfaction_rating,
                'context_type': context,
                'timestamp': datetime.now().isoformat()
            }
            
            # Learn from feedback with individual components
            if self.learning_enabled:
                learning_context = LearningContext(user_id=user_id)
                
                # Learn from pronunciation corrections
                if corrected_text and corrected_text != original_text and self.pronunciation_learner:
                    await self.pronunciation_learner.learn([feedback_data], learning_context)
                
                # Learn from context feedback
                if context and self.context_learner:
                    await self.context_learner.learn([feedback_data], learning_context)
            
            # Update statistics
            self.system_stats['user_feedback_events'] += 1
            
            # Store feedback data
            if self.learning_data_manager:
                from ..learning.learning_data_manager import LearningData, DataPrivacyLevel
                storage_data = LearningData(
                    user_id=user_id,
                    data_type="user_feedback",
                    data_content=feedback_data,
                    privacy_level=DataPrivacyLevel.PRIVATE
                )
                
                await self.learning_data_manager.store_learning_data(storage_data)
            
            logger.info(f"User feedback processed for {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"User feedback processing failed: {e}")
            return False
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive user profile information."""
        try:
            if not self.adaptive_recognizer:
                return None
            
            profile_summary = await self.adaptive_recognizer.get_user_profile_summary(user_id)
            
            if not profile_summary:
                return None
            
            # Add additional profile information
            additional_info = {}
            
            # Voice characteristics
            if self.voice_profile_learner:
                voice_profile = await self.voice_profile_learner.get_voice_profile(user_id)
                if voice_profile:
                    additional_info['voice_characteristics'] = {
                        'recognition_accuracy': voice_profile.recognition_accuracy,
                        'user_satisfaction': voice_profile.user_satisfaction,
                        'total_utterances': voice_profile.total_utterances,
                        'learning_enabled': voice_profile.learning_enabled
                    }
            
            # Pronunciation patterns
            if self.pronunciation_learner:
                patterns = await self.pronunciation_learner.get_pronunciation_patterns(user_id)
                additional_info['pronunciation_patterns_count'] = len(patterns)
            
            # Accent information
            if self.accent_learner:
                primary_accent = await self.accent_learner.get_primary_accent(user_id)
                if primary_accent:
                    additional_info['primary_accent'] = {
                        'accent_id': primary_accent.accent_id,
                        'confidence': primary_accent.confidence_score,
                        'sample_count': primary_accent.sample_count
                    }
            
            # Context patterns
            if self.context_learner:
                context_stats = self.context_learner.get_context_statistics()
                additional_info['context_patterns'] = context_stats
            
            return {
                **profile_summary,
                **additional_info,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"User profile retrieval failed: {e}")
            return None
    
    async def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        try:
            stats = {
                **self.system_stats,
                'is_initialized': self.is_initialized,
                'learning_enabled': self.learning_enabled,
                'auto_adaptation': self.auto_adaptation
            }
            
            # Add component statistics
            if self.adaptive_recognizer:
                stats['adaptive_recognizer'] = self.adaptive_recognizer.get_adaptation_statistics()
            
            if self.voice_profile_learner:
                stats['voice_profile_learner'] = self.voice_profile_learner.get_performance_stats()
            
            if self.pronunciation_learner:
                stats['pronunciation_learner'] = self.pronunciation_learner.get_learning_statistics()
            
            if self.accent_learner:
                stats['accent_learner'] = self.accent_learner.get_accent_statistics()
            
            if self.context_learner:
                stats['context_learner'] = self.context_learner.get_context_statistics()
            
            if self.learning_data_manager:
                stats['data_manager'] = await self.learning_data_manager.get_statistics()
            
            return stats
            
        except Exception as e:
            logger.error(f"System statistics retrieval failed: {e}")
            return {'error': str(e)}
    
    async def set_learning_preferences(self, 
                                     user_id: str,
                                     preferences: Dict[str, Any]) -> bool:
        """Set user learning preferences."""
        try:
            if self.learning_data_manager:
                success = await self.learning_data_manager.update_user_preferences(user_id, preferences)
                if success:
                    logger.info(f"Learning preferences updated for {user_id}")
                return success
            return False
            
        except Exception as e:
            logger.error(f"Learning preferences update failed: {e}")
            return False
    
    async def trigger_learning_session(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Trigger a manual learning session."""
        try:
            if not self.learning_enabled:
                return {'success': False, 'message': 'Learning is disabled'}
            
            # Get recent data for learning
            learning_data = []
            if self.learning_data_manager:
                if user_id:
                    data = await self.learning_data_manager.retrieve_learning_data(
                        user_id=user_id,
                        limit=50
                    )
                    learning_data.extend(data)
                else:
                    # Learn from all users
                    data = await self.learning_data_manager.retrieve_learning_data(limit=100)
                    learning_data.extend(data)
            
            if not learning_data:
                return {'success': False, 'message': 'No learning data available'}
            
            # Process data with learning components
            learning_results = {}
            context = LearningContext(user_id=user_id)
            
            # Group data by user
            user_data = {}
            for data in learning_data:
                uid = data.user_id or 'unknown'
                if uid not in user_data:
                    user_data[uid] = []
                user_data[uid].append(data.data_content)
            
            # Learn with each component
            for uid, data_points in user_data.items():
                user_context = LearningContext(user_id=uid)
                
                if self.voice_profile_learner:
                    result = await self.voice_profile_learner.learn(data_points, user_context)
                    learning_results[f'voice_profile_{uid}'] = result.success
                
                if self.pronunciation_learner:
                    result = await self.pronunciation_learner.learn(data_points, user_context)
                    learning_results[f'pronunciation_{uid}'] = result.success
                
                if self.context_learner:
                    result = await self.context_learner.learn(data_points, user_context)
                    learning_results[f'context_{uid}'] = result.success
            
            # Update statistics
            self.system_stats['learning_sessions'] += 1
            
            return {
                'success': True,
                'message': f'Learning session completed for {len(user_data)} users',
                'results': learning_results,
                'data_processed': len(learning_data)
            }
            
        except Exception as e:
            logger.error(f"Learning session failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _handle_recognition_event(self, event: Any) -> None:
        """Handle recognition completion events."""
        try:
            # Process recognition event for continuous improvement
            self.system_stats['adaptations_applied'] += 1
        except Exception as e:
            logger.error(f"Recognition event handling failed: {e}")
    
    async def _handle_feedback_event(self, event: Any) -> None:
        """Handle user feedback events."""
        try:
            # Process feedback event
            self.system_stats['system_improvements'] += 1
        except Exception as e:
            logger.error(f"Feedback event handling failed: {e}")
    
    async def _handle_adaptation_event(self, event: Any) -> None:
        """Handle adaptation events."""
        try:
            # Process adaptation event
            pass
        except Exception as e:
            logger.error(f"Adaptation event handling failed: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup all components."""
        try:
            if self.adaptive_recognizer:
                await self.adaptive_recognizer.cleanup()
            
            if self.voice_profile_learner:
                await self.voice_profile_learner.shutdown()
            
            if self.pronunciation_learner:
                await self.pronunciation_learner.shutdown()
            
            if self.accent_learner:
                await self.accent_learner.shutdown()
            
            if self.context_learner:
                await self.context_learner.shutdown()
            
            if self.learning_data_manager:
                await self.learning_data_manager.shutdown()
            
            logger.info("Speech learning system cleanup completed")
            
        except Exception as e:
            logger.error(f"System cleanup failed: {e}")