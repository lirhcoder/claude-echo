"""Voice Interface Module

This module provides a unified voice interface that integrates speech recognition,
synthesis, and intent parsing into a complete voice interaction pipeline.
"""

import asyncio
import time
from typing import Optional, Dict, Any, List, Callable, Tuple
from datetime import datetime
from enum import Enum
import logging

from loguru import logger

from ..core.base_adapter import BaseAdapter, AdapterError
from ..core.types import CommandResult, Context
from ..core.event_system import EventSystem, Event
from .types import (
    SpeechEventType, RecognitionConfig, SynthesisConfig, AudioConfig,
    RecognitionResult, SynthesisResult, ParsedIntent, SpeechProcessingStats
)
from .recognizer import SpeechRecognizer
from .synthesizer import SpeechSynthesizer
from .intent_parser import IntentParser
from .speech_learning_manager import SpeechLearningManager


class VoiceState(Enum):
    """Voice interface states"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"


class VoiceInterfaceError(AdapterError):
    """Voice interface specific error"""
    pass


class VoiceInterface:
    """
    Unified voice interface providing complete speech interaction pipeline.
    
    Features:
    - Integrated speech recognition, synthesis, and intent parsing
    - State machine for voice interaction flow
    - Conversation management with context
    - Performance monitoring and statistics
    - Event-driven architecture with comprehensive notifications
    """
    
    def __init__(self, 
                 recognition_config: Optional[RecognitionConfig] = None,
                 synthesis_config: Optional[SynthesisConfig] = None,
                 audio_config: Optional[AudioConfig] = None,
                 event_system: Optional[EventSystem] = None):
        """
        Initialize the voice interface.
        
        Args:
            recognition_config: Speech recognition configuration
            synthesis_config: Speech synthesis configuration
            audio_config: Audio processing configuration
            event_system: Event system for notifications
        """
        self.recognition_config = recognition_config or RecognitionConfig()
        self.synthesis_config = synthesis_config or SynthesisConfig()
        self.audio_config = audio_config or AudioConfig()
        self.event_system = event_system
        
        # Core components
        self._recognizer: Optional[SpeechRecognizer] = None
        self._synthesizer: Optional[SpeechSynthesizer] = None
        self._intent_parser: Optional[IntentParser] = None
        self._learning_manager: Optional[SpeechLearningManager] = None
        
        # State management
        self._state = VoiceState.IDLE
        self._state_lock = asyncio.Lock()
        
        # Session management
        self._session_id = f"voice_session_{int(time.time())}"
        self._conversation_context: List[Dict[str, Any]] = []
        self._current_context: Optional[Context] = None
        
        # Processing pipeline
        self._processing_queue: asyncio.Queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None
        self._response_handlers: Dict[str, Callable] = {}
        
        # Statistics and monitoring
        self._stats = SpeechProcessingStats(session_id=self._session_id)
        self._performance_metrics: Dict[str, float] = {}
        
        # Configuration
        self._auto_listen = False
        self._continuous_mode = False
        self._silence_timeout = 3.0  # seconds
        
        # Learning configuration
        self._learning_enabled = self.config.get('learning_enabled', True) if hasattr(self, 'config') else True
        self._adaptive_recognition = self.config.get('adaptive_recognition', True) if hasattr(self, 'config') else True
        self._current_user_id = None  # Will be set when user context is available
        
        logger.info(f"VoiceInterface initialized with session: {self._session_id}")
    
    async def initialize(self) -> bool:
        """
        Initialize the voice interface and all components.
        
        Returns:
            True if initialization successful
        """
        try:
            # Initialize speech recognizer
            self._recognizer = SpeechRecognizer(
                config=self.recognition_config,
                audio_config=self.audio_config,
                event_system=self.event_system
            )
            
            if not await self._recognizer.initialize():
                raise VoiceInterfaceError("Failed to initialize speech recognizer")
            
            # Initialize speech synthesizer
            self._synthesizer = SpeechSynthesizer(
                config=self.synthesis_config,
                event_system=self.event_system
            )
            
            if not await self._synthesizer.initialize():
                raise VoiceInterfaceError("Failed to initialize speech synthesizer")
            
            # Initialize intent parser
            self._intent_parser = IntentParser(event_system=self.event_system)
            
            # Initialize learning manager if enabled
            if self._learning_enabled:
                self._learning_manager = SpeechLearningManager(
                    event_system=self.event_system,
                    config=getattr(self, 'config', {}).get('learning_config', {})
                )
                
                if not await self._learning_manager.initialize():
                    logger.warning("Failed to initialize learning manager - continuing without learning")
                    self._learning_manager = None
                    self._learning_enabled = False
                else:
                    logger.info("Speech learning system initialized")
            
            # Start processing pipeline
            self._processing_task = asyncio.create_task(self._process_voice_pipeline())
            
            # Subscribe to relevant events if event system is available
            if self.event_system:
                await self._subscribe_to_events()
            
            self._state = VoiceState.IDLE
            logger.info("Voice interface initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize voice interface: {e}")
            self._state = VoiceState.ERROR
            return False
    
    async def cleanup(self) -> None:
        """Clean up resources and stop all components."""
        try:
            # Stop processing pipeline
            if self._processing_task and not self._processing_task.done():
                self._processing_task.cancel()
                try:
                    await self._processing_task
                except asyncio.CancelledError:
                    pass
            
            # Clean up components
            if self._recognizer:
                await self._recognizer.cleanup()
            
            if self._synthesizer:
                await self._synthesizer.cleanup()
            
            if self._learning_manager:
                await self._learning_manager.cleanup()
            
            # Clear context and statistics
            self._conversation_context.clear()
            self._state = VoiceState.IDLE
            
            logger.info("Voice interface cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def _subscribe_to_events(self) -> None:
        """Subscribe to relevant speech events."""
        if not self.event_system:
            return
        
        # Subscribe to voice activity events
        self.event_system.subscribe(
            SpeechEventType.VOICE_ACTIVITY_DETECTED.value,
            self._handle_voice_activity,
            handler_id="voice_interface_vad"
        )
        
        # Subscribe to recognition events
        self.event_system.subscribe(
            [
                SpeechEventType.RECOGNITION_COMPLETED.value,
                SpeechEventType.RECOGNITION_FAILED.value
            ],
            self._handle_recognition_event,
            handler_id="voice_interface_recognition"
        )
        
        # Subscribe to synthesis events
        self.event_system.subscribe(
            [
                SpeechEventType.SYNTHESIS_COMPLETED.value,
                SpeechEventType.SYNTHESIS_FAILED.value
            ],
            self._handle_synthesis_event,
            handler_id="voice_interface_synthesis"
        )
    
    async def start_listening(self, 
                            continuous: bool = False,
                            timeout: Optional[float] = None) -> bool:
        """
        Start listening for voice input.
        
        Args:
            continuous: Enable continuous listening mode
            timeout: Optional timeout for listening session
            
        Returns:
            True if listening started successfully
        """
        async with self._state_lock:
            if self._state not in [VoiceState.IDLE, VoiceState.SPEAKING]:
                logger.warning(f"Cannot start listening in state: {self._state}")
                return False
            
            try:
                # Start speech recognition
                if not await self._recognizer.start_listening():
                    return False
                
                self._continuous_mode = continuous
                self._state = VoiceState.LISTENING
                
                logger.info(f"Started listening (continuous: {continuous})")
                return True
                
            except Exception as e:
                logger.error(f"Failed to start listening: {e}")
                self._state = VoiceState.ERROR
                return False
    
    async def stop_listening(self) -> bool:
        """
        Stop listening for voice input.
        
        Returns:
            True if listening stopped successfully
        """
        async with self._state_lock:
            if self._state != VoiceState.LISTENING:
                return True
            
            try:
                # Stop speech recognition
                if not await self._recognizer.stop_listening():
                    return False
                
                self._state = VoiceState.IDLE
                self._continuous_mode = False
                
                logger.info("Stopped listening")
                return True
                
            except Exception as e:
                logger.error(f"Failed to stop listening: {e}")
                return False
    
    async def process_voice_input(self, 
                                duration: Optional[float] = None) -> Optional[ParsedIntent]:
        """
        Process current voice input from buffer.
        
        Args:
            duration: Duration in seconds to process (None for all buffer)
            
        Returns:
            Parsed intent or None if processing failed
        """
        async with self._state_lock:
            if self._state != VoiceState.LISTENING:
                logger.warning(f"Cannot process input in state: {self._state}")
                return None
            
            self._state = VoiceState.PROCESSING
        
        try:
            start_time = time.time()
            
            # Use adaptive recognition if available and user is identified
            if self._learning_manager and self._current_user_id and self._adaptive_recognition:
                # Get context hint from conversation
                context_hint = self._infer_context_from_conversation()
                
                # Use adaptive recognition
                adaptive_result = await self._learning_manager.recognize_speech(
                    user_id=self._current_user_id,
                    duration=duration,
                    context=context_hint
                )
                
                if adaptive_result:
                    # Convert adaptive result back to RecognitionResult format
                    recognition_result = self._convert_adaptive_result(adaptive_result)
                else:
                    # Fall back to base recognition
                    recognition_result = await self._recognizer.recognize_from_buffer(duration)
            else:
                # Standard recognition
                recognition_result = await self._recognizer.recognize_from_buffer(duration)
            
            if not recognition_result:
                logger.warning("No speech recognized from buffer")
                return None
            
            # Parse intent
            parsed_intent = await self._intent_parser.parse_intent(recognition_result)
            if not parsed_intent:
                logger.warning("Failed to parse intent from speech")
                return None
            
            # Update conversation context
            self._add_to_context(recognition_result, parsed_intent)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_processing_stats(recognition_result, parsed_intent, processing_time)
            
            logger.info(f"Processed voice input: {parsed_intent.intent_type.value}")
            return parsed_intent
            
        except Exception as e:
            logger.error(f"Voice input processing failed: {e}")
            return None
            
        finally:
            async with self._state_lock:
                if self._continuous_mode:
                    self._state = VoiceState.LISTENING
                else:
                    self._state = VoiceState.IDLE
    
    async def speak_response(self, 
                           text: str,
                           voice_settings: Optional[Dict[str, Any]] = None,
                           interrupt_current: bool = False) -> bool:
        """
        Speak a text response.
        
        Args:
            text: Text to speak
            voice_settings: Optional voice settings override
            interrupt_current: Whether to interrupt current speech
            
        Returns:
            True if speech started successfully
        """
        async with self._state_lock:
            if self._state == VoiceState.SPEAKING and not interrupt_current:
                logger.warning("Already speaking, cannot start new speech")
                return False
            
            self._state = VoiceState.SPEAKING
        
        try:
            # Speak the text
            success = await self._synthesizer.speak_text(text, voice_settings)
            
            if success:
                # Add to conversation context
                self._add_response_to_context(text)
                
                # Update statistics
                self._stats.total_syntheses += 1
                if success:
                    self._stats.successful_syntheses += 1
                
                logger.info(f"Speaking response: {text[:50]}...")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to speak response: {e}")
            return False
            
        finally:
            async with self._state_lock:
                if self._continuous_mode:
                    self._state = VoiceState.LISTENING
                else:
                    self._state = VoiceState.IDLE
    
    async def process_voice_command(self, 
                                  timeout: Optional[float] = None) -> Optional[ParsedIntent]:
        """
        Complete voice command processing pipeline.
        
        Args:
            timeout: Timeout for the entire process
            
        Returns:
            Parsed intent or None if failed
        """
        try:
            # Start listening if not already listening
            if self._state == VoiceState.IDLE:
                if not await self.start_listening():
                    return None
            
            # Wait for voice input with timeout
            start_time = time.time()
            while self._state == VoiceState.LISTENING:
                await asyncio.sleep(0.1)
                
                # Check for timeout
                if timeout and (time.time() - start_time) > timeout:
                    await self.stop_listening()
                    return None
                
                # Check if we have enough audio to process
                if self._recognizer.get_buffer_duration() > 1.0:
                    break
            
            # Process the voice input
            return await self.process_voice_input()
            
        except Exception as e:
            logger.error(f"Voice command processing failed: {e}")
            return None
    
    async def _process_voice_pipeline(self) -> None:
        """Main voice processing pipeline."""
        logger.info("Started voice processing pipeline")
        
        while True:
            try:
                # Get next processing task
                try:
                    task = await asyncio.wait_for(
                        self._processing_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                if task is None:  # Shutdown signal
                    break
                
                # Process the task based on type
                task_type = task.get('type')
                if task_type == 'voice_input':
                    await self._handle_voice_input_task(task)
                elif task_type == 'synthesis':
                    await self._handle_synthesis_task(task)
                elif task_type == 'intent_processing':
                    await self._handle_intent_processing_task(task)
                
            except Exception as e:
                logger.error(f"Error in voice processing pipeline: {e}")
    
    async def _handle_voice_input_task(self, task: Dict[str, Any]) -> None:
        """Handle voice input processing task."""
        try:
            duration = task.get('duration')
            result = await self.process_voice_input(duration)
            
            # Notify completion
            callback = task.get('callback')
            if callback:
                await callback(result)
                
        except Exception as e:
            logger.error(f"Voice input task failed: {e}")
    
    async def _handle_synthesis_task(self, task: Dict[str, Any]) -> None:
        """Handle speech synthesis task."""
        try:
            text = task.get('text', '')
            voice_settings = task.get('voice_settings')
            result = await self.speak_response(text, voice_settings)
            
            # Notify completion
            callback = task.get('callback')
            if callback:
                await callback(result)
                
        except Exception as e:
            logger.error(f"Synthesis task failed: {e}")
    
    async def _handle_intent_processing_task(self, task: Dict[str, Any]) -> None:
        """Handle intent processing task."""
        try:
            parsed_intent = task.get('parsed_intent')
            
            # Execute intent-based actions
            await self._execute_intent_actions(parsed_intent)
            
        except Exception as e:
            logger.error(f"Intent processing task failed: {e}")
    
    async def _execute_intent_actions(self, parsed_intent: ParsedIntent) -> None:
        """Execute actions based on parsed intent."""
        try:
            # This would integrate with the adapter system
            # For now, just log the intent
            logger.info(f"Executing actions for intent: {parsed_intent.intent_type.value}")
            logger.info(f"Suggested commands: {parsed_intent.suggested_commands}")
            logger.info(f"Target adapters: {parsed_intent.target_adapters}")
            
            # Emit intent execution event
            if self.event_system:
                await self.event_system.emit(Event(
                    event_type="voice.intent.executed",
                    data={
                        'intent_type': parsed_intent.intent_type.value,
                        'commands': parsed_intent.suggested_commands,
                        'adapters': parsed_intent.target_adapters
                    },
                    source='voice_interface'
                ))
            
        except Exception as e:
            logger.error(f"Intent action execution failed: {e}")
    
    async def _handle_voice_activity(self, event: Event) -> None:
        """Handle voice activity detection events."""
        try:
            if self._state == VoiceState.LISTENING:
                logger.debug("Voice activity detected")
                # Could trigger processing or adjust sensitivity
                
        except Exception as e:
            logger.error(f"Voice activity handler failed: {e}")
    
    async def _handle_recognition_event(self, event: Event) -> None:
        """Handle speech recognition events."""
        try:
            if event.event_type == SpeechEventType.RECOGNITION_COMPLETED.value:
                logger.debug("Recognition completed")
                # Could trigger automatic intent parsing
                
            elif event.event_type == SpeechEventType.RECOGNITION_FAILED.value:
                logger.warning("Recognition failed")
                # Could provide user feedback
                
        except Exception as e:
            logger.error(f"Recognition event handler failed: {e}")
    
    async def _handle_synthesis_event(self, event: Event) -> None:
        """Handle speech synthesis events."""
        try:
            if event.event_type == SpeechEventType.SYNTHESIS_COMPLETED.value:
                logger.debug("Synthesis completed")
                # Could return to listening mode
                
            elif event.event_type == SpeechEventType.SYNTHESIS_FAILED.value:
                logger.warning("Synthesis failed")
                # Could retry or provide alternative feedback
                
        except Exception as e:
            logger.error(f"Synthesis event handler failed: {e}")
    
    def _add_to_context(self, 
                       recognition_result: RecognitionResult, 
                       parsed_intent: ParsedIntent) -> None:
        """Add interaction to conversation context."""
        context_entry = {
            'timestamp': datetime.now(),
            'user_input': recognition_result.text,
            'intent_type': parsed_intent.intent_type.value,
            'confidence': parsed_intent.confidence,
            'entities': parsed_intent.entities,
            'parameters': parsed_intent.parameters
        }
        
        self._conversation_context.append(context_entry)
        
        # Maintain context size limit
        max_context_size = 50
        if len(self._conversation_context) > max_context_size:
            self._conversation_context = self._conversation_context[-max_context_size:]
    
    def _add_response_to_context(self, response_text: str) -> None:
        """Add system response to conversation context."""
        if self._conversation_context:
            # Add to the last context entry
            self._conversation_context[-1]['system_response'] = response_text
    
    def _update_processing_stats(self, 
                               recognition_result: RecognitionResult,
                               parsed_intent: ParsedIntent,
                               processing_time: float) -> None:
        """Update processing statistics."""
        # Update recognition stats
        self._stats.total_recognitions += 1
        if recognition_result:
            self._stats.successful_recognitions += 1
            
            # Update average recognition time
            old_avg = self._stats.average_recognition_time
            count = self._stats.successful_recognitions
            self._stats.average_recognition_time = \
                (old_avg * (count - 1) + recognition_result.processing_time) / count
            
            # Update average confidence
            old_avg = self._stats.average_confidence
            self._stats.average_confidence = \
                (old_avg * (count - 1) + recognition_result.confidence) / count
        
        # Update intent parsing stats
        self._stats.total_intents_parsed += 1
        if parsed_intent and parsed_intent.intent_type.value != 'unknown':
            self._stats.successful_intent_parses += 1
            
            # Update intent distribution
            intent_type = parsed_intent.intent_type.value
            self._stats.intent_type_distribution[intent_type] = \
                self._stats.intent_type_distribution.get(intent_type, 0) + 1
        
        # Update processing time
        self._performance_metrics['last_processing_time'] = processing_time
        
        # Update last activity timestamp
        self._stats.last_update = datetime.now()
    
    def get_state(self) -> VoiceState:
        """Get current voice interface state."""
        return self._state
    
    def get_conversation_context(self) -> List[Dict[str, Any]]:
        """Get conversation context."""
        return self._conversation_context.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            'session_id': self._session_id,
            'state': self._state.value,
            'conversation_turns': len(self._conversation_context),
            'processing_stats': self._stats.__dict__,
            'performance_metrics': self._performance_metrics,
            'component_stats': {
                'recognizer': self._recognizer.get_statistics() if self._recognizer else {},
                'synthesizer': self._synthesizer.get_statistics() if self._synthesizer else {},
                'intent_parser': self._intent_parser.get_statistics() if self._intent_parser else {}
            }
        }
    
    def is_listening(self) -> bool:
        """Check if currently listening."""
        return self._state == VoiceState.LISTENING
    
    def is_processing(self) -> bool:
        """Check if currently processing."""
        return self._state == VoiceState.PROCESSING
    
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._state == VoiceState.SPEAKING
    
    async def set_context(self, context: Context) -> None:
        """Set current system context."""
        self._current_context = context
        
        # Extract user ID from context if available
        if hasattr(context, 'user_id') and context.user_id:
            self._current_user_id = context.user_id
            logger.info(f"User ID set to: {self._current_user_id}")
        
        logger.info(f"Context updated for session: {self._session_id}")
    
    async def set_user_id(self, user_id: str) -> None:
        """Set current user ID for personalization."""
        self._current_user_id = user_id
        logger.info(f"User ID set to: {user_id}")
    
    def _infer_context_from_conversation(self) -> Optional[str]:
        """Infer context from recent conversation history."""
        if not self._conversation_context:
            return None
        
        # Look at recent intents to infer context
        recent_intents = [entry.get('intent_type') for entry in self._conversation_context[-3:]]
        
        # Map intent types to contexts
        intent_context_map = {
            'coding_request': 'programming',
            'file_operation': 'file_ops',
            'system_control': 'system',
            'application_control': 'application',
            'query_request': 'query',
            'navigation_request': 'navigation'
        }
        
        # Find most common recent context
        contexts = [intent_context_map.get(intent) for intent in recent_intents if intent]
        if contexts:
            from collections import Counter
            most_common = Counter(contexts).most_common(1)
            return most_common[0][0] if most_common else None
        
        return None
    
    def _convert_adaptive_result(self, adaptive_result: Dict[str, Any]) -> RecognitionResult:
        """Convert adaptive recognition result to RecognitionResult format."""
        from .types import RecognitionResult
        
        return RecognitionResult(
            text=adaptive_result.get('text', ''),
            confidence=adaptive_result.get('confidence', 0.0),
            language=adaptive_result.get('language', 'unknown'),
            processing_time=adaptive_result.get('processing_time', 0.0),
            audio_duration=adaptive_result.get('audio_duration', 0.0),
            model_used=adaptive_result.get('model_used', 'unknown'),
            alternative_texts=adaptive_result.get('alternative_texts', []),
            word_timestamps=adaptive_result.get('word_timestamps', [])
        )
    
    async def provide_user_feedback(self, 
                                  original_text: str,
                                  corrected_text: Optional[str] = None,
                                  satisfaction_rating: Optional[int] = None) -> bool:
        """Provide user feedback to improve recognition."""
        if not self._learning_manager or not self._current_user_id:
            logger.warning("Cannot provide feedback: learning not available or no user ID")
            return False
        
        try:
            context = self._infer_context_from_conversation()
            
            success = await self._learning_manager.provide_user_feedback(
                user_id=self._current_user_id,
                original_text=original_text,
                corrected_text=corrected_text,
                satisfaction_rating=satisfaction_rating,
                context=context
            )
            
            if success:
                logger.info("User feedback provided for learning improvement")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to provide user feedback: {e}")
            return False
    
    async def get_user_profile(self) -> Optional[Dict[str, Any]]:
        """Get current user's voice profile."""
        if not self._learning_manager or not self._current_user_id:
            return None
        
        try:
            return await self._learning_manager.get_user_profile(self._current_user_id)
        except Exception as e:
            logger.error(f"Failed to get user profile: {e}")
            return None
    
    async def trigger_learning_session(self) -> Dict[str, Any]:
        """Trigger a manual learning session for the current user."""
        if not self._learning_manager:
            return {'success': False, 'message': 'Learning system not available'}
        
        try:
            return await self._learning_manager.trigger_learning_session(self._current_user_id)
        except Exception as e:
            logger.error(f"Learning session failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning system statistics."""
        if not self._learning_manager:
            return {'learning_enabled': False}
        
        try:
            import asyncio
            return asyncio.run(self._learning_manager.get_system_statistics())
        except Exception as e:
            logger.error(f"Failed to get learning statistics: {e}")
            return {'error': str(e)}


class VoiceInterfaceAdapter(BaseAdapter):
    """Voice Interface Adapter for integration with the core system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the voice interface adapter."""
        super().__init__(config)
        
        # Create configurations from adapter config
        recognition_config = RecognitionConfig(
            model=self._config.get('recognition', {}).get('model', 'base'),
            language=self._config.get('recognition', {}).get('language', 'auto'),
            programming_keywords=self._config.get('recognition', {}).get('programming_keywords', True)
        )
        
        synthesis_config = SynthesisConfig(
            voice=self._config.get('synthesis', {}).get('voice', 'zh'),
            rate=self._config.get('synthesis', {}).get('rate', 150),
            volume=self._config.get('synthesis', {}).get('volume', 0.8)
        )
        
        audio_config = AudioConfig(
            sample_rate=self._config.get('audio', {}).get('sample_rate', 16000),
            noise_reduction=self._config.get('audio', {}).get('noise_reduction', True),
            vad_enabled=self._config.get('audio', {}).get('vad_enabled', True)
        )
        
        self._voice_interface = VoiceInterface(
            recognition_config=recognition_config,
            synthesis_config=synthesis_config,
            audio_config=audio_config
        )
    
    @property
    def adapter_id(self) -> str:
        return "voice_interface"
    
    @property
    def name(self) -> str:
        return "Voice Interface Adapter"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Unified voice interface with speech recognition, synthesis, and intent parsing"
    
    @property
    def supported_commands(self) -> List[str]:
        return [
            "start_listening",
            "stop_listening",
            "process_voice_input",
            "speak_response",
            "process_voice_command",
            "get_state",
            "get_context",
            "get_statistics",
            "set_context",
            "set_user_id",
            "provide_user_feedback",
            "get_user_profile",
            "trigger_learning_session",
            "get_learning_statistics"
        ]
    
    async def initialize(self) -> bool:
        """Initialize the voice interface."""
        try:
            success = await self._voice_interface.initialize()
            if success:
                self._update_status("available")
            return success
        except Exception as e:
            logger.error(f"Failed to initialize voice interface adapter: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Clean up the voice interface."""
        await self._voice_interface.cleanup()
    
    async def execute_command(self, 
                            command: str, 
                            parameters: Dict[str, Any],
                            context: Optional[Context] = None) -> CommandResult:
        """Execute a voice interface command."""
        try:
            if command == "start_listening":
                continuous = parameters.get('continuous', False)
                timeout = parameters.get('timeout')
                success = await self._voice_interface.start_listening(continuous, timeout)
                return CommandResult(
                    success=success,
                    data={'listening': success, 'continuous': continuous}
                )
            
            elif command == "stop_listening":
                success = await self._voice_interface.stop_listening()
                return CommandResult(
                    success=success,
                    data={'listening': not success}
                )
            
            elif command == "process_voice_input":
                duration = parameters.get('duration')
                parsed_intent = await self._voice_interface.process_voice_input(duration)
                return CommandResult(
                    success=parsed_intent is not None,
                    data={'parsed_intent': parsed_intent.__dict__ if parsed_intent else None}
                )
            
            elif command == "speak_response":
                text = parameters.get('text', '')
                if not text:
                    return CommandResult(
                        success=False,
                        error="text parameter required"
                    )
                
                voice_settings = parameters.get('voice_settings')
                interrupt_current = parameters.get('interrupt_current', False)
                success = await self._voice_interface.speak_response(
                    text, voice_settings, interrupt_current
                )
                return CommandResult(
                    success=success,
                    data={'spoken': success}
                )
            
            elif command == "process_voice_command":
                timeout = parameters.get('timeout')
                parsed_intent = await self._voice_interface.process_voice_command(timeout)
                return CommandResult(
                    success=parsed_intent is not None,
                    data={'parsed_intent': parsed_intent.__dict__ if parsed_intent else None}
                )
            
            elif command == "get_state":
                return CommandResult(
                    success=True,
                    data={
                        'state': self._voice_interface.get_state().value,
                        'listening': self._voice_interface.is_listening(),
                        'processing': self._voice_interface.is_processing(),
                        'speaking': self._voice_interface.is_speaking()
                    }
                )
            
            elif command == "get_context":
                return CommandResult(
                    success=True,
                    data={'context': self._voice_interface.get_conversation_context()}
                )
            
            elif command == "get_statistics":
                return CommandResult(
                    success=True,
                    data=self._voice_interface.get_statistics()
                )
            
            elif command == "set_context":
                if context:
                    await self._voice_interface.set_context(context)
                    return CommandResult(
                        success=True,
                        data={'context_set': True}
                    )
                else:
                    return CommandResult(
                        success=False,
                        error="context parameter required"
                    )
            
            elif command == "set_user_id":
                user_id = parameters.get('user_id')
                if not user_id:
                    return CommandResult(
                        success=False,
                        error="user_id parameter required"
                    )
                
                await self._voice_interface.set_user_id(user_id)
                return CommandResult(
                    success=True,
                    data={'user_id': user_id}
                )
            
            elif command == "provide_user_feedback":
                original_text = parameters.get('original_text')
                if not original_text:
                    return CommandResult(
                        success=False,
                        error="original_text parameter required"
                    )
                
                corrected_text = parameters.get('corrected_text')
                satisfaction_rating = parameters.get('satisfaction_rating')
                
                success = await self._voice_interface.provide_user_feedback(
                    original_text=original_text,
                    corrected_text=corrected_text,
                    satisfaction_rating=satisfaction_rating
                )
                
                return CommandResult(
                    success=success,
                    data={'feedback_processed': success}
                )
            
            elif command == "get_user_profile":
                profile = await self._voice_interface.get_user_profile()
                return CommandResult(
                    success=profile is not None,
                    data={'user_profile': profile}
                )
            
            elif command == "trigger_learning_session":
                result = await self._voice_interface.trigger_learning_session()
                return CommandResult(
                    success=result.get('success', False),
                    data=result
                )
            
            elif command == "get_learning_statistics":
                stats = self._voice_interface.get_learning_statistics()
                return CommandResult(
                    success=True,
                    data=stats
                )
            
            else:
                return CommandResult(
                    success=False,
                    error=f"Unknown command: {command}"
                )
                
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return CommandResult(
                success=False,
                error=str(e)
            )
    
    async def is_available(self) -> bool:
        """Check if the adapter is available."""
        return self._status == "available"
    
    async def get_current_state(self) -> Dict[str, Any]:
        """Get current adapter state."""
        return {
            'status': self._status,
            'voice_state': self._voice_interface.get_state().value,
            'statistics': self._voice_interface.get_statistics()
        }
    
    async def get_command_suggestions(self, context: Optional[Context] = None) -> List[str]:
        """Get command suggestions based on current state."""
        suggestions = []
        
        state = self._voice_interface.get_state()
        
        if state == VoiceState.IDLE:
            suggestions.extend(["start_listening", "process_voice_command"])
        elif state == VoiceState.LISTENING:
            suggestions.extend(["stop_listening", "process_voice_input"])
        elif state == VoiceState.PROCESSING:
            suggestions.append("get_state")
        elif state == VoiceState.SPEAKING:
            suggestions.append("get_state")
        
        suggestions.extend(["get_statistics", "get_context"])
        
        return suggestions