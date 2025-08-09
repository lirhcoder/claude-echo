"""Speech Recognition Module

This module provides speech-to-text functionality using OpenAI Whisper
with optimizations for programming-related content and Chinese-English support.
"""

import asyncio
import io
import os
import tempfile
import time
from typing import Optional, Dict, Any, List, Tuple, Union
from datetime import datetime
import logging

import numpy as np
import whisper
import pyaudio
import wave
import webrtcvad
from loguru import logger

from ..core.base_adapter import BaseAdapter, AdapterError
from ..core.types import CommandResult
from ..core.event_system import EventSystem, Event
from .types import (
    RecognitionConfig, RecognitionResult, AudioConfig, AudioBuffer,
    SpeechEventType, LanguageCode, RecognitionEngine, AudioFormat,
    PROGRAMMING_KEYWORDS, ABBREVIATION_EXPANSIONS, PUNCTUATION_PATTERNS
)


class SpeechRecognitionError(AdapterError):
    """Speech recognition specific error"""
    pass


class AudioInputError(SpeechRecognitionError):
    """Audio input related error"""
    pass


class WhisperModelError(SpeechRecognitionError):
    """Whisper model loading/execution error"""
    pass


class SpeechRecognizer:
    """
    Advanced speech recognition system with programming context optimization.
    
    Features:
    - OpenAI Whisper integration with multiple model sizes
    - Real-time audio streaming and processing
    - Voice Activity Detection (VAD)
    - Programming keyword recognition optimization
    - Chinese-English bilingual support
    - Audio preprocessing and enhancement
    - Async processing with event emission
    """
    
    def __init__(self, 
                 config: Optional[RecognitionConfig] = None,
                 audio_config: Optional[AudioConfig] = None,
                 event_system: Optional[EventSystem] = None):
        """
        Initialize the speech recognizer.
        
        Args:
            config: Recognition configuration
            audio_config: Audio processing configuration
            event_system: Event system for notifications
        """
        self.config = config or RecognitionConfig()
        self.audio_config = audio_config or AudioConfig()
        self.event_system = event_system
        
        # Whisper model
        self._whisper_model: Optional[whisper.Whisper] = None
        self._model_loading_lock = asyncio.Lock()
        
        # Audio components
        self._pyaudio: Optional[pyaudio.PyAudio] = None
        self._audio_stream: Optional[pyaudio.Stream] = None
        self._vad: Optional[webrtcvad.Vad] = None
        
        # Processing state
        self._is_listening = False
        self._is_processing = False
        self._audio_buffer = AudioBuffer(
            max_duration=self.config.max_audio_length,
            sample_rate=self.audio_config.sample_rate
        )
        
        # Statistics
        self._stats = {
            'total_recognitions': 0,
            'successful_recognitions': 0,
            'average_processing_time': 0.0,
            'average_confidence': 0.0,
            'errors': 0
        }
        
        logger.info(f"SpeechRecognizer initialized with model: {self.config.model}")
    
    async def initialize(self) -> bool:
        """
        Initialize the speech recognizer.
        
        Returns:
            True if initialization successful
        """
        try:
            # Load Whisper model
            await self._load_whisper_model()
            
            # Initialize audio components
            await self._initialize_audio()
            
            # Initialize VAD if enabled
            if self.audio_config.vad_enabled:
                await self._initialize_vad()
            
            # Emit initialization event
            if self.event_system:
                await self.event_system.emit(Event(
                    event_type=SpeechEventType.RECOGNITION_STARTED.value,
                    data={'model': self.config.model, 'language': self.config.language.value},
                    source='speech_recognizer'
                ))
            
            logger.info("Speech recognizer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize speech recognizer: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Stop listening if active
            if self._is_listening:
                await self.stop_listening()
            
            # Clean up audio resources
            if self._audio_stream:
                self._audio_stream.stop_stream()
                self._audio_stream.close()
                self._audio_stream = None
            
            if self._pyaudio:
                self._pyaudio.terminate()
                self._pyaudio = None
            
            # Clear buffers
            self._audio_buffer.clear()
            
            logger.info("Speech recognizer cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def _load_whisper_model(self) -> None:
        """Load the Whisper model."""
        async with self._model_loading_lock:
            if self._whisper_model is not None:
                return
            
            try:
                logger.info(f"Loading Whisper model: {self.config.model}")
                
                # Load model in executor to avoid blocking
                loop = asyncio.get_event_loop()
                self._whisper_model = await loop.run_in_executor(
                    None, 
                    lambda: whisper.load_model(
                        self.config.model,
                        device="cuda" if self.config.use_gpu else "cpu"
                    )
                )
                
                logger.info(f"Whisper model loaded successfully")
                
            except Exception as e:
                raise WhisperModelError(f"Failed to load Whisper model: {e}")
    
    async def _initialize_audio(self) -> None:
        """Initialize audio input components."""
        try:
            self._pyaudio = pyaudio.PyAudio()
            
            # Find appropriate audio device
            device_info = self._find_audio_device()
            
            logger.info(f"Using audio device: {device_info.get('name', 'Default')}")
            
        except Exception as e:
            raise AudioInputError(f"Failed to initialize audio: {e}")
    
    async def _initialize_vad(self) -> None:
        """Initialize Voice Activity Detection."""
        try:
            # WebRTC VAD supports only specific sample rates
            vad_sample_rate = 16000  # WebRTC VAD requirement
            self._vad = webrtcvad.Vad(2)  # Aggressiveness level 0-3
            
            logger.info("Voice Activity Detection initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize VAD: {e}")
            self._vad = None
    
    def _find_audio_device(self) -> Dict[str, Any]:
        """Find the best available audio input device."""
        if not self._pyaudio:
            raise AudioInputError("PyAudio not initialized")
        
        device_count = self._pyaudio.get_device_count()
        best_device = None
        
        for i in range(device_count):
            device_info = self._pyaudio.get_device_info_by_index(i)
            
            # Look for input devices
            if device_info['maxInputChannels'] > 0:
                # Prefer devices with "mic" in the name
                if 'mic' in device_info['name'].lower():
                    best_device = device_info
                    break
                elif best_device is None:
                    best_device = device_info
        
        return best_device or self._pyaudio.get_default_input_device_info()
    
    async def start_listening(self) -> bool:
        """
        Start continuous audio listening.
        
        Returns:
            True if listening started successfully
        """
        if self._is_listening:
            logger.warning("Already listening")
            return True
        
        try:
            # Create audio stream
            self._audio_stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.audio_config.channels,
                rate=self.audio_config.sample_rate,
                input=True,
                frames_per_buffer=self.audio_config.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self._audio_stream.start_stream()
            self._is_listening = True
            
            # Emit listening started event
            if self.event_system:
                await self.event_system.emit(Event(
                    event_type=SpeechEventType.AUDIO_INPUT_STARTED.value,
                    source='speech_recognizer'
                ))
            
            logger.info("Started listening for audio input")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start listening: {e}")
            return False
    
    async def stop_listening(self) -> bool:
        """
        Stop continuous audio listening.
        
        Returns:
            True if listening stopped successfully
        """
        if not self._is_listening:
            return True
        
        try:
            if self._audio_stream:
                self._audio_stream.stop_stream()
                self._audio_stream.close()
                self._audio_stream = None
            
            self._is_listening = False
            
            # Emit listening stopped event
            if self.event_system:
                await self.event_system.emit(Event(
                    event_type=SpeechEventType.AUDIO_INPUT_STOPPED.value,
                    source='speech_recognizer'
                ))
            
            logger.info("Stopped listening for audio input")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop listening: {e}")
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for incoming audio data."""
        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
            audio_data = audio_data / 32768.0  # Normalize to [-1, 1]
            
            # Apply audio preprocessing
            if self.audio_config.noise_reduction:
                audio_data = self._apply_noise_reduction(audio_data)
            
            # Add to buffer
            self._audio_buffer.append(audio_data)
            
            # Voice Activity Detection
            if self._vad and self.audio_config.vad_enabled:
                self._process_vad(in_data)
            
            return (None, pyaudio.paContinue)
            
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
            return (None, pyaudio.paAbort)
    
    def _apply_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply basic noise reduction to audio data.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Processed audio data
        """
        # Simple noise gate - remove very quiet signals
        noise_floor = 0.01
        audio_data = np.where(np.abs(audio_data) < noise_floor, 0, audio_data)
        
        # Apply gentle high-pass filter to remove low-frequency noise
        # This is a simple implementation - in production, use scipy.signal
        return audio_data
    
    def _process_vad(self, audio_bytes: bytes) -> None:
        """Process Voice Activity Detection."""
        try:
            # WebRTC VAD requires specific frame sizes (10, 20, or 30 ms)
            frame_duration = 20  # ms
            sample_rate = 16000
            frame_size = int(sample_rate * frame_duration / 1000)
            
            # Ensure we have enough data
            if len(audio_bytes) >= frame_size * 2:  # 2 bytes per sample (16-bit)
                # Take first frame
                frame = audio_bytes[:frame_size * 2]
                
                # Check voice activity
                is_speech = self._vad.is_speech(frame, sample_rate)
                
                if is_speech and self.event_system:
                    # Emit VAD event asynchronously
                    asyncio.create_task(
                        self.event_system.emit(Event(
                            event_type=SpeechEventType.VOICE_ACTIVITY_DETECTED.value,
                            data={'confidence': 0.8},  # WebRTC doesn't provide confidence
                            source='speech_recognizer'
                        ))
                    )
                    
        except Exception as e:
            logger.debug(f"VAD processing error: {e}")
    
    async def recognize_from_buffer(self, 
                                  duration: Optional[float] = None) -> Optional[RecognitionResult]:
        """
        Recognize speech from current audio buffer.
        
        Args:
            duration: Duration in seconds to process (None for all)
            
        Returns:
            Recognition result or None if failed
        """
        if self._is_processing:
            logger.warning("Recognition already in progress")
            return None
        
        if self._audio_buffer.is_empty():
            logger.warning("Audio buffer is empty")
            return None
        
        try:
            self._is_processing = True
            start_time = time.time()
            
            # Get audio data from buffer
            audio_data = self._audio_buffer.get_audio_data(duration)
            
            if len(audio_data) == 0:
                return None
            
            # Emit recognition started event
            if self.event_system:
                await self.event_system.emit(Event(
                    event_type=SpeechEventType.RECOGNITION_STARTED.value,
                    data={'audio_duration': len(audio_data) / self.audio_config.sample_rate},
                    source='speech_recognizer'
                ))
            
            # Perform recognition
            result = await self._recognize_audio(audio_data)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(result, processing_time)
            
            # Emit recognition completed event
            if self.event_system:
                await self.event_system.emit(Event(
                    event_type=SpeechEventType.RECOGNITION_COMPLETED.value,
                    data={
                        'text': result.text if result else '',
                        'confidence': result.confidence if result else 0.0,
                        'processing_time': processing_time
                    },
                    source='speech_recognizer'
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Recognition failed: {e}")
            
            if self.event_system:
                await self.event_system.emit(Event(
                    event_type=SpeechEventType.RECOGNITION_FAILED.value,
                    data={'error': str(e)},
                    source='speech_recognizer'
                ))
            
            return None
            
        finally:
            self._is_processing = False
    
    async def recognize_from_file(self, file_path: str) -> Optional[RecognitionResult]:
        """
        Recognize speech from audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Recognition result or None if failed
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            start_time = time.time()
            
            # Load audio file
            audio_data = await self._load_audio_file(file_path)
            
            # Perform recognition
            result = await self._recognize_audio(audio_data)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(result, processing_time)
            
            return result
            
        except Exception as e:
            logger.error(f"File recognition failed: {e}")
            return None
    
    async def _recognize_audio(self, audio_data: np.ndarray) -> Optional[RecognitionResult]:
        """
        Perform speech recognition on audio data.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Recognition result
        """
        if self._whisper_model is None:
            await self._load_whisper_model()
        
        try:
            # Prepare audio for Whisper
            audio_duration = len(audio_data) / self.audio_config.sample_rate
            
            # Ensure audio is the right format for Whisper
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Run Whisper inference in executor
            loop = asyncio.get_event_loop()
            whisper_result = await loop.run_in_executor(
                None,
                lambda: self._whisper_model.transcribe(
                    audio_data,
                    language=self.config.language.value if self.config.language != LanguageCode.AUTO else None,
                    beam_size=self.config.beam_size,
                    word_timestamps=True
                )
            )
            
            # Extract result
            text = whisper_result.get('text', '').strip()
            language = whisper_result.get('language', 'unknown')
            
            # Calculate confidence (Whisper doesn't provide direct confidence)
            # Use segment-level probabilities if available
            confidence = self._calculate_confidence(whisper_result)
            
            # Apply programming-specific optimizations
            if self.config.programming_keywords:
                text = self._optimize_programming_text(text)
            
            if self.config.abbreviation_expansion:
                text = self._expand_abbreviations(text)
            
            if self.config.punctuation_inference:
                text = self._infer_punctuation(text)
            
            # Create result object
            result = RecognitionResult(
                text=text,
                confidence=confidence,
                language=language,
                processing_time=time.time() - time.time(),  # Will be updated by caller
                audio_duration=audio_duration,
                model_used=self.config.model,
                alternative_texts=self._extract_alternatives(whisper_result),
                word_timestamps=whisper_result.get('segments', [])
            )
            
            return result
            
        except Exception as e:
            raise SpeechRecognitionError(f"Whisper recognition failed: {e}")
    
    def _calculate_confidence(self, whisper_result: Dict[str, Any]) -> float:
        """Calculate confidence score from Whisper result."""
        segments = whisper_result.get('segments', [])
        if not segments:
            return 0.8  # Default confidence
        
        # Average the probabilities from all segments
        total_prob = 0.0
        count = 0
        
        for segment in segments:
            if 'avg_logprob' in segment:
                # Convert log probability to probability
                prob = np.exp(segment['avg_logprob'])
                total_prob += prob
                count += 1
        
        return total_prob / count if count > 0 else 0.8
    
    def _extract_alternatives(self, whisper_result: Dict[str, Any]) -> List[str]:
        """Extract alternative transcriptions if available."""
        # Whisper doesn't provide alternatives by default
        # This could be enhanced with beam search results
        return []
    
    def _optimize_programming_text(self, text: str) -> str:
        """Optimize text for programming context."""
        words = text.split()
        optimized_words = []
        
        for word in words:
            word_lower = word.lower()
            
            # Check if word matches programming keywords
            for lang, keywords in PROGRAMMING_KEYWORDS.items():
                if word_lower in keywords:
                    optimized_words.append(word_lower)
                    break
            else:
                optimized_words.append(word)
        
        return ' '.join(optimized_words)
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common programming abbreviations."""
        words = text.split()
        expanded_words = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in ABBREVIATION_EXPANSIONS:
                expanded_words.append(ABBREVIATION_EXPANSIONS[word_lower])
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def _infer_punctuation(self, text: str) -> str:
        """Infer punctuation from speech patterns."""
        # Simple punctuation inference
        text_lower = text.lower()
        
        # Replace punctuation keywords with actual punctuation
        for punct, patterns in PUNCTUATION_PATTERNS.items():
            for pattern in patterns:
                if pattern in text_lower:
                    if punct == 'period':
                        text = text.replace(pattern, '.')
                    elif punct == 'comma':
                        text = text.replace(pattern, ',')
                    elif punct == 'colon':
                        text = text.replace(pattern, ':')
                    elif punct == 'semicolon':
                        text = text.replace(pattern, ';')
        
        return text
    
    async def _load_audio_file(self, file_path: str) -> np.ndarray:
        """Load audio file and convert to numpy array."""
        try:
            # Use Whisper's audio loading functionality
            loop = asyncio.get_event_loop()
            audio_data = await loop.run_in_executor(
                None,
                lambda: whisper.load_audio(file_path)
            )
            return audio_data
            
        except Exception as e:
            raise AudioInputError(f"Failed to load audio file: {e}")
    
    def _update_stats(self, result: Optional[RecognitionResult], processing_time: float) -> None:
        """Update recognition statistics."""
        self._stats['total_recognitions'] += 1
        
        if result:
            self._stats['successful_recognitions'] += 1
            
            # Update average confidence
            old_avg = self._stats['average_confidence']
            count = self._stats['successful_recognitions']
            self._stats['average_confidence'] = (old_avg * (count - 1) + result.confidence) / count
        else:
            self._stats['errors'] += 1
        
        # Update average processing time
        old_avg = self._stats['average_processing_time']
        count = self._stats['total_recognitions']
        self._stats['average_processing_time'] = (old_avg * (count - 1) + processing_time) / count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get recognition statistics."""
        return {
            **self._stats,
            'is_listening': self._is_listening,
            'is_processing': self._is_processing,
            'buffer_duration': self._audio_buffer.duration(),
            'model': self.config.model,
            'language': self.config.language.value
        }
    
    def is_listening(self) -> bool:
        """Check if currently listening for audio."""
        return self._is_listening
    
    def is_processing(self) -> bool:
        """Check if currently processing audio."""
        return self._is_processing
    
    def get_buffer_duration(self) -> float:
        """Get current audio buffer duration in seconds."""
        return self._audio_buffer.duration()


class SpeechRecognizerAdapter(BaseAdapter):
    """Speech Recognition Adapter for integration with the core system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the speech recognizer adapter."""
        super().__init__(config)
        
        # Create configurations from adapter config
        recognition_config = RecognitionConfig(
            model=self._config.get('model', 'base'),
            language=LanguageCode(self._config.get('language', 'auto')),
            timeout=self._config.get('timeout', 30.0),
            programming_keywords=self._config.get('programming_keywords', True)
        )
        
        audio_config = AudioConfig(
            sample_rate=self._config.get('sample_rate', 16000),
            noise_reduction=self._config.get('noise_reduction', True),
            vad_enabled=self._config.get('vad_enabled', True)
        )
        
        self._recognizer = SpeechRecognizer(
            config=recognition_config,
            audio_config=audio_config
        )
    
    @property
    def adapter_id(self) -> str:
        return "speech_recognizer"
    
    @property
    def name(self) -> str:
        return "Speech Recognition Adapter"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Converts speech to text using OpenAI Whisper with programming optimizations"
    
    @property
    def supported_commands(self) -> List[str]:
        return [
            "start_listening",
            "stop_listening", 
            "recognize_audio",
            "recognize_file",
            "get_status",
            "get_statistics"
        ]
    
    async def initialize(self) -> bool:
        """Initialize the speech recognizer."""
        try:
            success = await self._recognizer.initialize()
            if success:
                self._update_status("available")
            return success
        except Exception as e:
            logger.error(f"Failed to initialize speech recognizer adapter: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Clean up the speech recognizer."""
        await self._recognizer.cleanup()
    
    async def execute_command(self, 
                            command: str, 
                            parameters: Dict[str, Any],
                            context: Optional[Any] = None) -> CommandResult:
        """Execute a recognition command."""
        try:
            if command == "start_listening":
                success = await self._recognizer.start_listening()
                return CommandResult(
                    success=success,
                    data={'listening': success}
                )
            
            elif command == "stop_listening":
                success = await self._recognizer.stop_listening()
                return CommandResult(
                    success=success,
                    data={'listening': not success}
                )
            
            elif command == "recognize_audio":
                duration = parameters.get('duration')
                result = await self._recognizer.recognize_from_buffer(duration)
                return CommandResult(
                    success=result is not None,
                    data={'result': result.__dict__ if result else None}
                )
            
            elif command == "recognize_file":
                file_path = parameters.get('file_path')
                if not file_path:
                    return CommandResult(
                        success=False,
                        error="file_path parameter required"
                    )
                
                result = await self._recognizer.recognize_from_file(file_path)
                return CommandResult(
                    success=result is not None,
                    data={'result': result.__dict__ if result else None}
                )
            
            elif command == "get_status":
                return CommandResult(
                    success=True,
                    data={
                        'listening': self._recognizer.is_listening(),
                        'processing': self._recognizer.is_processing(),
                        'buffer_duration': self._recognizer.get_buffer_duration()
                    }
                )
            
            elif command == "get_statistics":
                return CommandResult(
                    success=True,
                    data=self._recognizer.get_statistics()
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
            'listening': self._recognizer.is_listening(),
            'processing': self._recognizer.is_processing(),
            'statistics': self._recognizer.get_statistics()
        }
    
    async def get_command_suggestions(self, context: Optional[Any] = None) -> List[str]:
        """Get command suggestions based on context."""
        suggestions = []
        
        if not self._recognizer.is_listening():
            suggestions.append("start_listening")
        else:
            suggestions.append("stop_listening")
            
        if self._recognizer.get_buffer_duration() > 0:
            suggestions.append("recognize_audio")
            
        suggestions.extend(["get_status", "get_statistics"])
        
        return suggestions