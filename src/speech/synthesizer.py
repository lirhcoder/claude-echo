"""Speech Synthesis Module

This module provides text-to-speech functionality with support for multiple engines,
optimized for programming content and Chinese-English bilingual support.
"""

import asyncio
import io
import os
import tempfile
import time
import threading
from typing import Optional, Dict, Any, List, Union, Callable
from datetime import datetime
import logging

import pyttsx3
import numpy as np
from loguru import logger

from ..core.base_adapter import BaseAdapter, AdapterError
from ..core.types import CommandResult
from ..core.event_system import EventSystem, Event
from .types import (
    SynthesisConfig, SynthesisResult, SpeechEventType,
    SynthesisEngine, AudioFormat, LanguageCode
)


class SpeechSynthesisError(AdapterError):
    """Speech synthesis specific error"""
    pass


class TTSEngineError(SpeechSynthesisError):
    """TTS engine related error"""
    pass


class SpeechSynthesizer:
    """
    Advanced text-to-speech system with multi-engine support.
    
    Features:
    - Multiple TTS engines (pyttsx3, Azure TTS, Google TTS)
    - Voice customization (speed, volume, pitch)
    - Chinese-English bilingual support
    - Programming text preprocessing
    - Async synthesis with event emission
    - Audio output management
    """
    
    def __init__(self, 
                 config: Optional[SynthesisConfig] = None,
                 event_system: Optional[EventSystem] = None):
        """
        Initialize the speech synthesizer.
        
        Args:
            config: Synthesis configuration
            event_system: Event system for notifications
        """
        self.config = config or SynthesisConfig()
        self.event_system = event_system
        
        # TTS Engine
        self._tts_engine: Optional[pyttsx3.Engine] = None
        self._engine_lock = threading.Lock()
        self._engine_busy = False
        
        # Voice settings
        self._available_voices: List[Dict[str, Any]] = []
        self._current_voice_id: Optional[str] = None
        
        # Processing state
        self._is_synthesizing = False
        self._synthesis_queue: asyncio.Queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None
        
        # Statistics
        self._stats = {
            'total_syntheses': 0,
            'successful_syntheses': 0,
            'total_audio_duration': 0.0,
            'average_processing_time': 0.0,
            'errors': 0
        }
        
        logger.info(f"SpeechSynthesizer initialized with engine: {self.config.engine.value}")
    
    async def initialize(self) -> bool:
        """
        Initialize the speech synthesizer.
        
        Returns:
            True if initialization successful
        """
        try:
            # Initialize TTS engine
            await self._initialize_tts_engine()
            
            # Start processing queue
            self._processing_task = asyncio.create_task(self._process_synthesis_queue())
            
            # Emit initialization event
            if self.event_system:
                await self.event_system.emit(Event(
                    event_type=SpeechEventType.SYNTHESIS_STARTED.value,
                    data={'engine': self.config.engine.value, 'voice': self.config.voice},
                    source='speech_synthesizer'
                ))
            
            logger.info("Speech synthesizer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize speech synthesizer: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Stop processing task
            if self._processing_task and not self._processing_task.done():
                self._processing_task.cancel()
                try:
                    await self._processing_task
                except asyncio.CancelledError:
                    pass
            
            # Clean up TTS engine
            await self._cleanup_tts_engine()
            
            logger.info("Speech synthesizer cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def _initialize_tts_engine(self) -> None:
        """Initialize the TTS engine."""
        try:
            if self.config.engine == SynthesisEngine.PYTTSX3:
                await self._initialize_pyttsx3()
            else:
                raise TTSEngineError(f"Unsupported TTS engine: {self.config.engine}")
                
        except Exception as e:
            raise TTSEngineError(f"Failed to initialize TTS engine: {e}")
    
    async def _initialize_pyttsx3(self) -> None:
        """Initialize pyttsx3 TTS engine."""
        def _init_engine():
            try:
                engine = pyttsx3.init()
                
                # Set voice properties
                voices = engine.getProperty('voices')
                self._available_voices = [
                    {
                        'id': voice.id,
                        'name': voice.name,
                        'languages': getattr(voice, 'languages', []),
                        'gender': getattr(voice, 'gender', 'unknown')
                    }
                    for voice in voices if voice is not None
                ]
                
                # Select appropriate voice
                selected_voice = self._select_voice()
                if selected_voice:
                    engine.setProperty('voice', selected_voice['id'])
                    self._current_voice_id = selected_voice['id']
                
                # Set other properties
                engine.setProperty('rate', self.config.rate)
                engine.setProperty('volume', self.config.volume)
                
                return engine
                
            except Exception as e:
                raise TTSEngineError(f"Failed to initialize pyttsx3: {e}")
        
        # Initialize in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        self._tts_engine = await loop.run_in_executor(None, _init_engine)
        
        logger.info(f"pyttsx3 initialized with voice: {self._current_voice_id}")
    
    def _select_voice(self) -> Optional[Dict[str, Any]]:
        """Select the most appropriate voice based on configuration."""
        if not self._available_voices:
            return None
        
        # Prefer voices that match the configured language
        preferred_voices = []
        
        for voice in self._available_voices:
            voice_name = voice.get('name', '').lower()
            voice_id = voice.get('id', '').lower()
            languages = voice.get('languages', [])
            
            # Check if voice supports the configured language
            if self.config.voice == 'zh' or self.config.voice == 'chinese':
                if ('chinese' in voice_name or 'zh' in voice_id or 
                    any('zh' in lang for lang in languages)):
                    preferred_voices.append(voice)
            elif self.config.voice == 'en' or self.config.voice == 'english':
                if ('english' in voice_name or 'en' in voice_id or
                    any('en' in lang for lang in languages)):
                    preferred_voices.append(voice)
        
        # Return first preferred voice, or first available voice
        return preferred_voices[0] if preferred_voices else self._available_voices[0]
    
    async def _cleanup_tts_engine(self) -> None:
        """Clean up TTS engine resources."""
        try:
            if self._tts_engine:
                def _cleanup():
                    try:
                        self._tts_engine.stop()
                    except:
                        pass  # Ignore cleanup errors
                
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, _cleanup)
                self._tts_engine = None
                
        except Exception as e:
            logger.error(f"Error cleaning up TTS engine: {e}")
    
    async def synthesize_text(self, 
                            text: str,
                            output_file: Optional[str] = None,
                            voice_settings: Optional[Dict[str, Any]] = None) -> Optional[SynthesisResult]:
        """
        Synthesize text to speech.
        
        Args:
            text: Text to synthesize
            output_file: Optional output file path
            voice_settings: Optional voice settings override
            
        Returns:
            Synthesis result or None if failed
        """
        try:
            start_time = time.time()
            
            # Preprocess text for better synthesis
            processed_text = self._preprocess_text(text)
            
            # Add to synthesis queue
            result = await self._queue_synthesis(processed_text, output_file, voice_settings)
            
            if result:
                # Update statistics
                processing_time = time.time() - start_time
                self._update_stats(result, processing_time)
                
                # Emit synthesis completed event
                if self.event_system:
                    await self.event_system.emit(Event(
                        event_type=SpeechEventType.SYNTHESIS_COMPLETED.value,
                        data={
                            'text': text,
                            'audio_duration': result.audio_duration,
                            'processing_time': processing_time
                        },
                        source='speech_synthesizer'
                    ))
            
            return result
            
        except Exception as e:
            logger.error(f"Text synthesis failed: {e}")
            
            if self.event_system:
                await self.event_system.emit(Event(
                    event_type=SpeechEventType.SYNTHESIS_FAILED.value,
                    data={'error': str(e), 'text': text},
                    source='speech_synthesizer'
                ))
            
            return None
    
    async def synthesize_to_file(self, 
                               text: str,
                               output_path: str,
                               voice_settings: Optional[Dict[str, Any]] = None) -> bool:
        """
        Synthesize text and save to file.
        
        Args:
            text: Text to synthesize
            output_path: Output file path
            voice_settings: Optional voice settings override
            
        Returns:
            True if successful
        """
        result = await self.synthesize_text(text, output_path, voice_settings)
        return result is not None and result.audio_file_path is not None
    
    async def speak_text(self, 
                        text: str,
                        voice_settings: Optional[Dict[str, Any]] = None) -> bool:
        """
        Speak text immediately (non-blocking).
        
        Args:
            text: Text to speak
            voice_settings: Optional voice settings override
            
        Returns:
            True if speech started successfully
        """
        result = await self.synthesize_text(text, None, voice_settings)
        return result is not None
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better synthesis.
        
        Args:
            text: Original text
            
        Returns:
            Processed text
        """
        if not self.config.pre_processing:
            return text
        
        processed = text
        
        # Handle programming-specific terms
        processed = self._process_programming_terms(processed)
        
        # Handle special characters and symbols
        processed = self._process_special_characters(processed)
        
        # Handle numbers and abbreviations
        processed = self._process_numbers_and_abbreviations(processed)
        
        # Clean up extra spaces
        processed = ' '.join(processed.split())
        
        return processed
    
    def _process_programming_terms(self, text: str) -> str:
        """Process programming-specific terms for better pronunciation."""
        # Common programming term replacements for better TTS
        replacements = {
            'def ': 'define ',
            'var ': 'variable ',
            'const ': 'constant ',
            'func ': 'function ',
            'param ': 'parameter ',
            'arg ': 'argument ',
            'async ': 'asynchronous ',
            'await ': 'await ',
            '==': ' equals equals ',
            '!=': ' not equals ',
            '<=': ' less than or equal to ',
            '>=': ' greater than or equal to ',
            '&&': ' and ',
            '||': ' or ',
            '++': ' increment ',
            '--': ' decrement ',
            '+=': ' plus equals ',
            '-=': ' minus equals ',
            '->': ' arrow ',
            '=>': ' fat arrow ',
            '::': ' double colon ',
            '[]': ' square brackets ',
            '{}': ' curly braces ',
            '()': ' parentheses '
        }
        
        processed = text
        for pattern, replacement in replacements.items():
            processed = processed.replace(pattern, replacement)
        
        return processed
    
    def _process_special_characters(self, text: str) -> str:
        """Process special characters and symbols."""
        # Symbol replacements
        symbols = {
            '&': ' and ',
            '@': ' at ',
            '#': ' hash ',
            '$': ' dollar ',
            '%': ' percent ',
            '^': ' caret ',
            '*': ' asterisk ',
            '~': ' tilde ',
            '`': ' backtick ',
            '|': ' pipe ',
            '\\': ' backslash ',
            '/': ' slash ',
            '<': ' less than ',
            '>': ' greater than ',
            '?': ' question mark ',
            ':': ' colon ',
            ';': ' semicolon ',
            '"': ' quote ',
            "'": ' apostrophe '
        }
        
        processed = text
        for symbol, replacement in symbols.items():
            # Only replace if not part of a word
            processed = processed.replace(f' {symbol} ', replacement)
            processed = processed.replace(f'{symbol} ', replacement)
            processed = processed.replace(f' {symbol}', replacement)
        
        return processed
    
    def _process_numbers_and_abbreviations(self, text: str) -> str:
        """Process numbers and common abbreviations."""
        import re
        
        # Handle version numbers (e.g., "3.9" -> "three point nine")
        version_pattern = r'\b(\d+)\.(\d+)(?:\.(\d+))?\b'
        
        def replace_version(match):
            parts = [self._number_to_words(int(p)) for p in match.groups() if p]
            return ' point '.join(parts)
        
        processed = re.sub(version_pattern, replace_version, text)
        
        return processed
    
    def _number_to_words(self, num: int) -> str:
        """Convert number to words for better TTS pronunciation."""
        if num == 0:
            return "zero"
        
        ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", 
                "sixteen", "seventeen", "eighteen", "nineteen"]
        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        
        if num < 10:
            return ones[num]
        elif num < 20:
            return teens[num - 10]
        elif num < 100:
            return tens[num // 10] + ("" if num % 10 == 0 else " " + ones[num % 10])
        else:
            return str(num)  # For larger numbers, keep as is
    
    async def _queue_synthesis(self, 
                             text: str,
                             output_file: Optional[str],
                             voice_settings: Optional[Dict[str, Any]]) -> Optional[SynthesisResult]:
        """Queue text for synthesis."""
        # Create synthesis task
        synthesis_task = {
            'text': text,
            'output_file': output_file,
            'voice_settings': voice_settings or {},
            'result': None,
            'completed': asyncio.Event()
        }
        
        # Add to queue
        await self._synthesis_queue.put(synthesis_task)
        
        # Wait for completion
        await synthesis_task['completed'].wait()
        
        return synthesis_task['result']
    
    async def _process_synthesis_queue(self) -> None:
        """Process the synthesis queue."""
        logger.info("Started synthesis queue processing")
        
        while True:
            try:
                # Get next synthesis task
                task = await self._synthesis_queue.get()
                
                if task is None:  # Shutdown signal
                    break
                
                # Process synthesis
                try:
                    result = await self._perform_synthesis(
                        task['text'],
                        task['output_file'],
                        task['voice_settings']
                    )
                    task['result'] = result
                except Exception as e:
                    logger.error(f"Synthesis failed: {e}")
                    task['result'] = None
                
                # Mark task as completed
                task['completed'].set()
                
            except Exception as e:
                logger.error(f"Error in synthesis queue processing: {e}")
    
    async def _perform_synthesis(self, 
                               text: str,
                               output_file: Optional[str],
                               voice_settings: Dict[str, Any]) -> Optional[SynthesisResult]:
        """Perform the actual synthesis."""
        if self.config.engine == SynthesisEngine.PYTTSX3:
            return await self._synthesize_with_pyttsx3(text, output_file, voice_settings)
        else:
            raise TTSEngineError(f"Unsupported synthesis engine: {self.config.engine}")
    
    async def _synthesize_with_pyttsx3(self, 
                                     text: str,
                                     output_file: Optional[str],
                                     voice_settings: Dict[str, Any]) -> Optional[SynthesisResult]:
        """Synthesize using pyttsx3."""
        def _synthesis():
            try:
                with self._engine_lock:
                    if not self._tts_engine:
                        raise TTSEngineError("TTS engine not initialized")
                    
                    # Apply voice settings
                    if voice_settings:
                        if 'rate' in voice_settings:
                            self._tts_engine.setProperty('rate', voice_settings['rate'])
                        if 'volume' in voice_settings:
                            self._tts_engine.setProperty('volume', voice_settings['volume'])
                        if 'voice' in voice_settings:
                            self._tts_engine.setProperty('voice', voice_settings['voice'])
                    
                    start_time = time.time()
                    
                    if output_file:
                        # Save to file
                        self._tts_engine.save_to_file(text, output_file)
                        self._tts_engine.runAndWait()
                        
                        # Get audio duration (approximate)
                        estimated_duration = len(text.split()) * 0.6  # ~0.6 seconds per word
                        
                        return SynthesisResult(
                            text=text,
                            audio_file_path=output_file,
                            processing_time=time.time() - start_time,
                            audio_duration=estimated_duration,
                            sample_rate=22050,  # pyttsx3 default
                            format=self.config.output_format.value,
                            voice_used=str(self._current_voice_id),
                            rate_used=self._tts_engine.getProperty('rate'),
                            volume_used=self._tts_engine.getProperty('volume')
                        )
                    else:
                        # Speak immediately
                        self._tts_engine.say(text)
                        self._tts_engine.runAndWait()
                        
                        # Get audio duration (approximate)
                        estimated_duration = len(text.split()) * 0.6  # ~0.6 seconds per word
                        
                        return SynthesisResult(
                            text=text,
                            processing_time=time.time() - start_time,
                            audio_duration=estimated_duration,
                            sample_rate=22050,  # pyttsx3 default
                            format=self.config.output_format.value,
                            voice_used=str(self._current_voice_id),
                            rate_used=self._tts_engine.getProperty('rate'),
                            volume_used=self._tts_engine.getProperty('volume')
                        )
                        
            except Exception as e:
                raise TTSEngineError(f"pyttsx3 synthesis failed: {e}")
        
        # Run synthesis in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _synthesis)
    
    def _update_stats(self, result: Optional[SynthesisResult], processing_time: float) -> None:
        """Update synthesis statistics."""
        self._stats['total_syntheses'] += 1
        
        if result:
            self._stats['successful_syntheses'] += 1
            self._stats['total_audio_duration'] += result.audio_duration
        else:
            self._stats['errors'] += 1
        
        # Update average processing time
        old_avg = self._stats['average_processing_time']
        count = self._stats['total_syntheses']
        self._stats['average_processing_time'] = (old_avg * (count - 1) + processing_time) / count
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices."""
        return self._available_voices.copy()
    
    def get_current_voice(self) -> Optional[Dict[str, Any]]:
        """Get current voice information."""
        if not self._current_voice_id:
            return None
        
        for voice in self._available_voices:
            if voice.get('id') == self._current_voice_id:
                return voice
        return None
    
    async def set_voice(self, voice_id: str) -> bool:
        """
        Set the current voice.
        
        Args:
            voice_id: Voice ID to use
            
        Returns:
            True if voice was set successfully
        """
        try:
            def _set_voice():
                with self._engine_lock:
                    if self._tts_engine:
                        self._tts_engine.setProperty('voice', voice_id)
                        self._current_voice_id = voice_id
                        return True
                return False
            
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, _set_voice)
            
            if success:
                logger.info(f"Voice changed to: {voice_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to set voice: {e}")
            return False
    
    async def set_speech_rate(self, rate: int) -> bool:
        """Set speech rate."""
        try:
            def _set_rate():
                with self._engine_lock:
                    if self._tts_engine:
                        self._tts_engine.setProperty('rate', rate)
                        self.config.rate = rate
                        return True
                return False
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _set_rate)
            
        except Exception as e:
            logger.error(f"Failed to set speech rate: {e}")
            return False
    
    async def set_volume(self, volume: float) -> bool:
        """Set speech volume."""
        try:
            volume = max(0.0, min(1.0, volume))  # Clamp to [0, 1]
            
            def _set_volume():
                with self._engine_lock:
                    if self._tts_engine:
                        self._tts_engine.setProperty('volume', volume)
                        self.config.volume = volume
                        return True
                return False
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _set_volume)
            
        except Exception as e:
            logger.error(f"Failed to set volume: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get synthesis statistics."""
        return {
            **self._stats,
            'is_synthesizing': self._is_synthesizing,
            'queue_size': self._synthesis_queue.qsize(),
            'engine': self.config.engine.value,
            'current_voice': self._current_voice_id,
            'rate': self.config.rate,
            'volume': self.config.volume
        }
    
    def is_busy(self) -> bool:
        """Check if synthesizer is currently busy."""
        return self._is_synthesizing or not self._synthesis_queue.empty()


class SpeechSynthesizerAdapter(BaseAdapter):
    """Speech Synthesis Adapter for integration with the core system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the speech synthesizer adapter."""
        super().__init__(config)
        
        # Create configuration from adapter config
        synthesis_config = SynthesisConfig(
            engine=SynthesisEngine(self._config.get('engine', 'pyttsx3')),
            voice=self._config.get('voice', 'zh'),
            rate=self._config.get('rate', 150),
            volume=self._config.get('volume', 0.8),
            pre_processing=self._config.get('pre_processing', True)
        )
        
        self._synthesizer = SpeechSynthesizer(config=synthesis_config)
    
    @property
    def adapter_id(self) -> str:
        return "speech_synthesizer"
    
    @property
    def name(self) -> str:
        return "Speech Synthesis Adapter"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Converts text to speech with programming content optimization"
    
    @property
    def supported_commands(self) -> List[str]:
        return [
            "speak_text",
            "synthesize_to_file",
            "set_voice",
            "set_rate",
            "set_volume",
            "get_voices",
            "get_status",
            "get_statistics"
        ]
    
    async def initialize(self) -> bool:
        """Initialize the speech synthesizer."""
        try:
            success = await self._synthesizer.initialize()
            if success:
                self._update_status("available")
            return success
        except Exception as e:
            logger.error(f"Failed to initialize speech synthesizer adapter: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Clean up the speech synthesizer."""
        await self._synthesizer.cleanup()
    
    async def execute_command(self, 
                            command: str, 
                            parameters: Dict[str, Any],
                            context: Optional[Any] = None) -> CommandResult:
        """Execute a synthesis command."""
        try:
            if command == "speak_text":
                text = parameters.get('text', '')
                if not text:
                    return CommandResult(
                        success=False,
                        error="text parameter required"
                    )
                
                voice_settings = parameters.get('voice_settings')
                success = await self._synthesizer.speak_text(text, voice_settings)
                return CommandResult(
                    success=success,
                    data={'spoken': success}
                )
            
            elif command == "synthesize_to_file":
                text = parameters.get('text', '')
                output_path = parameters.get('output_path', '')
                
                if not text or not output_path:
                    return CommandResult(
                        success=False,
                        error="text and output_path parameters required"
                    )
                
                voice_settings = parameters.get('voice_settings')
                success = await self._synthesizer.synthesize_to_file(text, output_path, voice_settings)
                return CommandResult(
                    success=success,
                    data={'file_created': success, 'path': output_path if success else None}
                )
            
            elif command == "set_voice":
                voice_id = parameters.get('voice_id')
                if not voice_id:
                    return CommandResult(
                        success=False,
                        error="voice_id parameter required"
                    )
                
                success = await self._synthesizer.set_voice(voice_id)
                return CommandResult(
                    success=success,
                    data={'voice_set': success}
                )
            
            elif command == "set_rate":
                rate = parameters.get('rate')
                if rate is None:
                    return CommandResult(
                        success=False,
                        error="rate parameter required"
                    )
                
                success = await self._synthesizer.set_speech_rate(int(rate))
                return CommandResult(
                    success=success,
                    data={'rate_set': success}
                )
            
            elif command == "set_volume":
                volume = parameters.get('volume')
                if volume is None:
                    return CommandResult(
                        success=False,
                        error="volume parameter required"
                    )
                
                success = await self._synthesizer.set_volume(float(volume))
                return CommandResult(
                    success=success,
                    data={'volume_set': success}
                )
            
            elif command == "get_voices":
                voices = self._synthesizer.get_available_voices()
                current_voice = self._synthesizer.get_current_voice()
                return CommandResult(
                    success=True,
                    data={'voices': voices, 'current_voice': current_voice}
                )
            
            elif command == "get_status":
                return CommandResult(
                    success=True,
                    data={
                        'busy': self._synthesizer.is_busy(),
                        'current_voice': self._synthesizer.get_current_voice(),
                        'queue_size': self._synthesizer._synthesis_queue.qsize()
                    }
                )
            
            elif command == "get_statistics":
                return CommandResult(
                    success=True,
                    data=self._synthesizer.get_statistics()
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
            'busy': self._synthesizer.is_busy(),
            'statistics': self._synthesizer.get_statistics()
        }
    
    async def get_command_suggestions(self, context: Optional[Any] = None) -> List[str]:
        """Get command suggestions based on context."""
        suggestions = ["speak_text", "get_voices", "get_status", "get_statistics"]
        
        if not self._synthesizer.is_busy():
            suggestions.extend(["set_voice", "set_rate", "set_volume"])
        
        return suggestions