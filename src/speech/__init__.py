"""Speech Processing Module

This module provides comprehensive speech processing capabilities including:
- Speech recognition with OpenAI Whisper
- Text-to-speech synthesis with pyttsx3
- Intent parsing with programming context awareness
- Unified voice interface for complete interaction pipeline
"""

from .types import (
    # Configuration types
    RecognitionConfig,
    SynthesisConfig, 
    AudioConfig,
    
    # Result types
    RecognitionResult,
    SynthesisResult,
    ParsedIntent,
    SpeechProcessingStats,
    VoiceActivityEvent,
    
    # Enum types
    SpeechEventType,
    AudioFormat,
    RecognitionEngine,
    SynthesisEngine,
    LanguageCode,
    IntentType,
    
    # Utility types
    AudioBuffer,
    
    # Constants
    PROGRAMMING_KEYWORDS,
    ABBREVIATION_EXPANSIONS,
    PUNCTUATION_PATTERNS
)

from .recognizer import (
    SpeechRecognizer,
    SpeechRecognizerAdapter,
    SpeechRecognitionError,
    AudioInputError,
    WhisperModelError
)

from .synthesizer import (
    SpeechSynthesizer,
    SpeechSynthesizerAdapter,
    SpeechSynthesisError,
    TTSEngineError
)

from .intent_parser import (
    IntentParser,
    IntentParserAdapter,
    IntentParsingError
)

from .voice_interface import (
    VoiceInterface,
    VoiceInterfaceAdapter,
    VoiceState,
    VoiceInterfaceError
)

# Adapter registry for easy access
SPEECH_ADAPTERS = {
    'speech_recognizer': SpeechRecognizerAdapter,
    'speech_synthesizer': SpeechSynthesizerAdapter,
    'intent_parser': IntentParserAdapter,
    'voice_interface': VoiceInterfaceAdapter
}

# Version info
__version__ = "1.0.0"
__author__ = "Claude Voice Assistant Team"

# Module-level exports
__all__ = [
    # Core classes
    "SpeechRecognizer",
    "SpeechSynthesizer", 
    "IntentParser",
    "VoiceInterface",
    
    # Adapter classes
    "SpeechRecognizerAdapter",
    "SpeechSynthesizerAdapter",
    "IntentParserAdapter",
    "VoiceInterfaceAdapter",
    
    # Configuration types
    "RecognitionConfig",
    "SynthesisConfig",
    "AudioConfig",
    
    # Result types
    "RecognitionResult",
    "SynthesisResult", 
    "ParsedIntent",
    "SpeechProcessingStats",
    "VoiceActivityEvent",
    
    # Enum types
    "SpeechEventType",
    "AudioFormat",
    "RecognitionEngine",
    "SynthesisEngine",
    "LanguageCode",
    "IntentType",
    "VoiceState",
    
    # Utility classes
    "AudioBuffer",
    
    # Exception types
    "SpeechRecognitionError",
    "AudioInputError", 
    "WhisperModelError",
    "SpeechSynthesisError",
    "TTSEngineError",
    "IntentParsingError",
    "VoiceInterfaceError",
    
    # Constants
    "PROGRAMMING_KEYWORDS",
    "ABBREVIATION_EXPANSIONS", 
    "PUNCTUATION_PATTERNS",
    
    # Registry
    "SPEECH_ADAPTERS"
]


def get_speech_adapter(adapter_name: str, config: dict = None):
    """
    Get a speech adapter instance by name.
    
    Args:
        adapter_name: Name of the adapter
        config: Optional configuration dictionary
        
    Returns:
        Adapter instance or None if not found
    """
    adapter_class = SPEECH_ADAPTERS.get(adapter_name)
    if adapter_class:
        return adapter_class(config)
    return None


def list_speech_adapters():
    """
    Get list of available speech adapters.
    
    Returns:
        List of adapter names
    """
    return list(SPEECH_ADAPTERS.keys())


def create_voice_pipeline(recognition_config=None, synthesis_config=None, audio_config=None):
    """
    Create a complete voice processing pipeline.
    
    Args:
        recognition_config: Optional recognition configuration
        synthesis_config: Optional synthesis configuration  
        audio_config: Optional audio configuration
        
    Returns:
        VoiceInterface instance
    """
    return VoiceInterface(
        recognition_config=recognition_config,
        synthesis_config=synthesis_config,
        audio_config=audio_config
    )


# Module initialization
def _initialize_module():
    """Initialize the speech module."""
    import logging
    from loguru import logger
    
    logger.info("Speech processing module initialized")
    logger.info(f"Available adapters: {list(SPEECH_ADAPTERS.keys())}")
    logger.info(f"Module version: {__version__}")


# Auto-initialize when module is imported
_initialize_module()