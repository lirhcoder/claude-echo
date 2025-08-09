"""Speech Module Type Definitions

This module defines data structures specific to speech processing functionality.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import uuid
import numpy as np


class SpeechEventType(Enum):
    """Speech system event types"""
    AUDIO_INPUT_STARTED = "speech.audio.input_started"
    AUDIO_INPUT_STOPPED = "speech.audio.input_stopped"
    RECOGNITION_STARTED = "speech.recognition.started"
    RECOGNITION_COMPLETED = "speech.recognition.completed"
    RECOGNITION_FAILED = "speech.recognition.failed"
    SYNTHESIS_STARTED = "speech.synthesis.started"
    SYNTHESIS_COMPLETED = "speech.synthesis.completed"
    SYNTHESIS_FAILED = "speech.synthesis.failed"
    INTENT_PARSED = "speech.intent.parsed"
    INTENT_PARSE_FAILED = "speech.intent.parse_failed"
    VOICE_ACTIVITY_DETECTED = "speech.vad.detected"
    VOICE_ACTIVITY_ENDED = "speech.vad.ended"
    PROCESSING_ERROR = "speech.processing.error"


class AudioFormat(Enum):
    """Supported audio formats"""
    WAV = "wav"
    MP3 = "mp3" 
    FLAC = "flac"
    OGG = "ogg"
    M4A = "m4a"
    RAW = "raw"


class RecognitionEngine(Enum):
    """Speech recognition engines"""
    WHISPER = "whisper"
    AZURE_SPEECH = "azure_speech"
    GOOGLE_SPEECH = "google_speech"
    WINDOWS_SPEECH = "windows_speech"


class SynthesisEngine(Enum):
    """Text-to-speech engines"""
    PYTTSX3 = "pyttsx3"
    AZURE_TTS = "azure_tts"
    GOOGLE_TTS = "google_tts"
    WINDOWS_TTS = "windows_tts"


class LanguageCode(Enum):
    """Supported language codes"""
    CHINESE = "zh"
    CHINESE_CN = "zh-CN"
    ENGLISH = "en"
    ENGLISH_US = "en-US"
    AUTO = "auto"


class IntentType(Enum):
    """Intent classification types"""
    CODING_REQUEST = "coding_request"
    FILE_OPERATION = "file_operation"
    SYSTEM_CONTROL = "system_control"
    APPLICATION_CONTROL = "application_control"
    QUERY_REQUEST = "query_request"
    NAVIGATION_REQUEST = "navigation_request"
    UNKNOWN = "unknown"


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 16000
    channels: int = 1
    bit_depth: int = 16
    chunk_size: int = 1024
    format: AudioFormat = AudioFormat.WAV
    
    # Audio processing parameters
    noise_reduction: bool = True
    echo_cancellation: bool = True
    auto_gain_control: bool = True
    
    # Voice Activity Detection
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    vad_padding_ms: int = 300


@dataclass
class RecognitionConfig:
    """Speech recognition configuration"""
    engine: RecognitionEngine = RecognitionEngine.WHISPER
    model: str = "base"
    language: LanguageCode = LanguageCode.AUTO
    timeout: float = 30.0
    
    # Whisper-specific
    whisper_model_path: Optional[str] = None
    use_gpu: bool = False
    beam_size: int = 5
    
    # Optimization settings
    programming_keywords: bool = True
    abbreviation_expansion: bool = True
    punctuation_inference: bool = True
    
    # Performance settings
    max_audio_length: float = 30.0  # seconds
    chunk_processing: bool = True


@dataclass
class SynthesisConfig:
    """Text-to-speech configuration"""
    engine: SynthesisEngine = SynthesisEngine.PYTTSX3
    voice: str = "zh"
    rate: int = 150
    volume: float = 0.8
    
    # Voice characteristics
    pitch: Optional[float] = None
    voice_id: Optional[str] = None
    
    # Output settings
    output_format: AudioFormat = AudioFormat.WAV
    quality: str = "medium"  # low, medium, high
    
    # Processing settings
    pre_processing: bool = True
    post_processing: bool = True
    silence_padding: float = 0.2  # seconds


class RecognitionResult(BaseModel):
    """Speech recognition result"""
    result_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    language: str
    processing_time: float
    
    # Additional metadata
    audio_duration: Optional[float] = None
    model_used: Optional[str] = None
    alternative_texts: List[str] = Field(default_factory=list)
    word_timestamps: Optional[List[Dict[str, Any]]] = None
    
    # Quality metrics
    signal_quality: Optional[float] = None
    noise_level: Optional[float] = None
    
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True


class SynthesisResult(BaseModel):
    """Text-to-speech synthesis result"""
    result_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    audio_data: Optional[bytes] = None
    audio_file_path: Optional[str] = None
    
    # Generation metadata
    processing_time: float
    audio_duration: float
    sample_rate: int
    format: str
    
    # Settings used
    voice_used: str
    rate_used: int
    volume_used: float
    
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True


class ParsedIntent(BaseModel):
    """Parsed user intent from speech"""
    intent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Original input
    original_text: str
    processed_text: str
    
    # Intent classification
    intent_type: IntentType
    confidence: float = Field(ge=0.0, le=1.0)
    
    # Extracted entities
    entities: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Context information
    context_keywords: List[str] = Field(default_factory=list)
    programming_context: bool = False
    
    # Suggested actions
    suggested_commands: List[str] = Field(default_factory=list)
    target_adapters: List[str] = Field(default_factory=list)
    
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True


@dataclass
class VoiceActivityEvent:
    """Voice activity detection event"""
    activity_detected: bool
    confidence: float
    audio_level: float
    start_time: datetime
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    duration: Optional[timedelta] = None
    
    # Audio characteristics
    frequency_content: Optional[Dict[str, float]] = None
    signal_quality: Optional[float] = None


@dataclass
class SpeechProcessingStats:
    """Speech processing performance statistics"""
    session_id: str
    
    # Recognition stats
    total_recognitions: int = 0
    successful_recognitions: int = 0
    average_recognition_time: float = 0.0
    average_confidence: float = 0.0
    
    # Synthesis stats
    total_syntheses: int = 0
    successful_syntheses: int = 0
    average_synthesis_time: float = 0.0
    total_audio_generated: float = 0.0  # seconds
    
    # Intent parsing stats
    total_intents_parsed: int = 0
    successful_intent_parses: int = 0
    intent_type_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Audio processing stats
    total_audio_processed: float = 0.0  # seconds
    average_audio_quality: float = 0.0
    vad_events: int = 0
    
    # Error stats
    recognition_errors: int = 0
    synthesis_errors: int = 0
    processing_errors: int = 0
    
    # Performance metrics
    cpu_usage_avg: float = 0.0
    memory_usage_avg: float = 0.0
    
    session_start: datetime = Field(default_factory=datetime.now)
    last_update: datetime = Field(default_factory=datetime.now)


class AudioBuffer:
    """Thread-safe audio buffer for streaming audio data"""
    
    def __init__(self, max_duration: float = 30.0, sample_rate: int = 16000):
        """
        Initialize audio buffer.
        
        Args:
            max_duration: Maximum buffer duration in seconds
            sample_rate: Audio sample rate
        """
        self.max_samples = int(max_duration * sample_rate)
        self.sample_rate = sample_rate
        self.buffer: List[np.ndarray] = []
        self.total_samples = 0
        self._lock = None  # Will be set by asyncio
    
    def append(self, audio_chunk: np.ndarray) -> None:
        """
        Append audio chunk to buffer.
        
        Args:
            audio_chunk: Audio data to append
        """
        self.buffer.append(audio_chunk)
        self.total_samples += len(audio_chunk)
        
        # Remove old chunks if buffer is too large
        while self.total_samples > self.max_samples:
            if self.buffer:
                removed_chunk = self.buffer.pop(0)
                self.total_samples -= len(removed_chunk)
    
    def get_audio_data(self, last_n_seconds: Optional[float] = None) -> np.ndarray:
        """
        Get audio data from buffer.
        
        Args:
            last_n_seconds: Get only last N seconds of audio
            
        Returns:
            Combined audio data as numpy array
        """
        if not self.buffer:
            return np.array([])
        
        combined = np.concatenate(self.buffer)
        
        if last_n_seconds:
            samples_needed = int(last_n_seconds * self.sample_rate)
            if len(combined) > samples_needed:
                combined = combined[-samples_needed:]
        
        return combined
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.total_samples = 0
    
    def duration(self) -> float:
        """Get current buffer duration in seconds."""
        return self.total_samples / self.sample_rate
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.buffer) == 0


# Programming language keywords and patterns for speech optimization
PROGRAMMING_KEYWORDS = {
    'python': [
        'def', 'class', 'import', 'from', 'if', 'else', 'elif', 'for', 'while',
        'try', 'except', 'finally', 'with', 'as', 'lambda', 'return', 'yield',
        'async', 'await', 'and', 'or', 'not', 'in', 'is'
    ],
    'javascript': [
        'function', 'var', 'let', 'const', 'if', 'else', 'for', 'while', 'do',
        'switch', 'case', 'default', 'try', 'catch', 'finally', 'async', 'await',
        'class', 'extends', 'import', 'export', 'from'
    ],
    'general': [
        'variable', 'function', 'method', 'parameter', 'argument', 'return',
        'loop', 'condition', 'array', 'object', 'string', 'number', 'boolean',
        'null', 'undefined', 'true', 'false'
    ]
}

# Common abbreviation expansions for programming context
ABBREVIATION_EXPANSIONS = {
    'var': 'variable',
    'func': 'function',
    'param': 'parameter',
    'arg': 'argument',
    'obj': 'object',
    'str': 'string',
    'num': 'number',
    'bool': 'boolean',
    'int': 'integer',
    'float': 'floating point',
    'arr': 'array',
    'dict': 'dictionary',
    'len': 'length',
    'max': 'maximum',
    'min': 'minimum',
    'avg': 'average',
    'temp': 'temporary',
    'config': 'configuration',
    'init': 'initialize',
    'exec': 'execute',
    'impl': 'implement',
    'inst': 'instance'
}

# Punctuation inference patterns
PUNCTUATION_PATTERNS = {
    'period': [
        'end of line', 'end line', 'period', 'dot',
        'finish statement', 'end statement'
    ],
    'comma': [
        'comma', 'pause', 'separate', 'and'
    ],
    'colon': [
        'colon', 'define', 'start block', 'begin block'
    ],
    'semicolon': [
        'semicolon', 'end line semicolon', 'statement end'
    ],
    'parentheses': [
        'open paren', 'close paren', 'parentheses',
        'function call', 'parameters'
    ],
    'brackets': [
        'open bracket', 'close bracket', 'square brackets',
        'array access', 'index'
    ],
    'braces': [
        'open brace', 'close brace', 'curly braces',
        'code block', 'dictionary', 'object'
    ]
}