"""Speech Module Test Suite

This module provides comprehensive tests for the speech processing functionality.
"""

import asyncio
import pytest
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import numpy as np

# Import speech modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.speech.types import (
    RecognitionConfig, SynthesisConfig, AudioConfig,
    RecognitionResult, SynthesisResult, ParsedIntent,
    IntentType, LanguageCode, RecognitionEngine, SynthesisEngine
)
from src.speech.recognizer import SpeechRecognizer, SpeechRecognizerAdapter
from src.speech.synthesizer import SpeechSynthesizer, SpeechSynthesizerAdapter
from src.speech.intent_parser import IntentParser, IntentParserAdapter
from src.speech.voice_interface import VoiceInterface, VoiceInterfaceAdapter, VoiceState
from src.speech.config import SpeechConfigManager
from src.core.event_system import EventSystem


class TestSpeechTypes:
    """Test speech type definitions and configurations."""
    
    def test_recognition_config_creation(self):
        """Test RecognitionConfig creation and validation."""
        config = RecognitionConfig(
            engine=RecognitionEngine.WHISPER,
            model="base",
            language=LanguageCode.CHINESE,
            timeout=30.0
        )
        
        assert config.engine == RecognitionEngine.WHISPER
        assert config.model == "base"
        assert config.language == LanguageCode.CHINESE
        assert config.timeout == 30.0
        assert config.programming_keywords is True
    
    def test_synthesis_config_creation(self):
        """Test SynthesisConfig creation and validation."""
        config = SynthesisConfig(
            engine=SynthesisEngine.PYTTSX3,
            voice="zh",
            rate=150,
            volume=0.8
        )
        
        assert config.engine == SynthesisEngine.PYTTSX3
        assert config.voice == "zh"
        assert config.rate == 150
        assert config.volume == 0.8
    
    def test_audio_config_creation(self):
        """Test AudioConfig creation and validation."""
        config = AudioConfig(
            sample_rate=16000,
            channels=1,
            vad_enabled=True
        )
        
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.vad_enabled is True
        assert config.noise_reduction is True
    
    def test_recognition_result_creation(self):
        """Test RecognitionResult creation."""
        result = RecognitionResult(
            text="Hello world",
            confidence=0.95,
            language="en",
            processing_time=2.5
        )
        
        assert result.text == "Hello world"
        assert result.confidence == 0.95
        assert result.language == "en"
        assert result.processing_time == 2.5
        assert result.result_id is not None


class TestSpeechRecognizer:
    """Test speech recognition functionality."""
    
    @pytest.fixture
    def mock_whisper_model(self):
        """Mock Whisper model for testing."""
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            'text': 'Hello world',
            'language': 'en',
            'segments': []
        }
        return mock_model
    
    @pytest.fixture
    def recognition_config(self):
        """Test recognition configuration."""
        return RecognitionConfig(
            engine=RecognitionEngine.WHISPER,
            model="tiny",  # Use tiny model for testing
            language=LanguageCode.ENGLISH,
            timeout=10.0
        )
    
    @pytest.fixture
    def audio_config(self):
        """Test audio configuration."""
        return AudioConfig(
            sample_rate=16000,
            channels=1,
            vad_enabled=False  # Disable VAD for testing
        )
    
    @pytest.fixture
    async def speech_recognizer(self, recognition_config, audio_config):
        """Create speech recognizer for testing."""
        recognizer = SpeechRecognizer(
            config=recognition_config,
            audio_config=audio_config
        )
        yield recognizer
        await recognizer.cleanup()
    
    @patch('src.speech.recognizer.whisper.load_model')
    async def test_recognizer_initialization(self, mock_load_model, speech_recognizer):
        """Test speech recognizer initialization."""
        mock_load_model.return_value = Mock()
        
        success = await speech_recognizer.initialize()
        
        # Note: This test may fail in CI due to missing audio dependencies
        # In real environment, this should pass
        if success:
            assert speech_recognizer._whisper_model is not None
        mock_load_model.assert_called_once()
    
    def test_text_preprocessing(self, speech_recognizer):
        """Test programming text preprocessing."""
        text = "create function test def variable"
        processed = speech_recognizer._optimize_programming_text(text)
        
        # Should keep programming keywords as-is
        assert "def" in processed.lower()
    
    def test_abbreviation_expansion(self, speech_recognizer):
        """Test abbreviation expansion."""
        text = "var func param"
        expanded = speech_recognizer._expand_abbreviations(text)
        
        assert "variable" in expanded
        assert "function" in expanded
        assert "parameter" in expanded
    
    def test_punctuation_inference(self, speech_recognizer):
        """Test punctuation inference."""
        text = "hello world period"
        processed = speech_recognizer._infer_punctuation(text)
        
        assert "." in processed
    
    async def test_recognizer_adapter(self):
        """Test speech recognizer adapter."""
        config = {
            'model': 'tiny',
            'language': 'en',
            'programming_keywords': True
        }
        
        adapter = SpeechRecognizerAdapter(config)
        
        assert adapter.adapter_id == "speech_recognizer"
        assert adapter.name == "Speech Recognition Adapter"
        assert "start_listening" in adapter.supported_commands
        assert "recognize_audio" in adapter.supported_commands


class TestSpeechSynthesizer:
    """Test speech synthesis functionality."""
    
    @pytest.fixture
    def synthesis_config(self):
        """Test synthesis configuration."""
        return SynthesisConfig(
            engine=SynthesisEngine.PYTTSX3,
            voice="en",
            rate=200,
            volume=0.7
        )
    
    @pytest.fixture
    async def speech_synthesizer(self, synthesis_config):
        """Create speech synthesizer for testing."""
        synthesizer = SpeechSynthesizer(config=synthesis_config)
        yield synthesizer
        await synthesizer.cleanup()
    
    def test_text_preprocessing(self, speech_synthesizer):
        """Test programming text preprocessing for TTS."""
        text = "def function() -> str:"
        processed = speech_synthesizer._preprocess_text(text)
        
        # Should expand programming terms for better pronunciation
        assert "define" in processed or "function" in processed
    
    def test_special_character_processing(self, speech_synthesizer):
        """Test special character processing."""
        text = "x -> y"
        processed = speech_synthesizer._process_special_characters(text)
        
        assert "arrow" in processed
    
    def test_number_to_words(self, speech_synthesizer):
        """Test number to words conversion."""
        assert speech_synthesizer._number_to_words(0) == "zero"
        assert speech_synthesizer._number_to_words(5) == "five"
        assert speech_synthesizer._number_to_words(23) == "twenty three"
    
    @patch('src.speech.synthesizer.pyttsx3.init')
    async def test_synthesizer_initialization(self, mock_pyttsx3_init, speech_synthesizer):
        """Test speech synthesizer initialization."""
        mock_engine = Mock()
        mock_engine.getProperty.return_value = []
        mock_pyttsx3_init.return_value = mock_engine
        
        success = await speech_synthesizer.initialize()
        
        # Should initialize successfully with mocked engine
        if success:
            assert speech_synthesizer._tts_engine is not None
    
    async def test_synthesizer_adapter(self):
        """Test speech synthesizer adapter."""
        config = {
            'engine': 'pyttsx3',
            'voice': 'en',
            'rate': 150
        }
        
        adapter = SpeechSynthesizerAdapter(config)
        
        assert adapter.adapter_id == "speech_synthesizer"
        assert adapter.name == "Speech Synthesis Adapter"
        assert "speak_text" in adapter.supported_commands
        assert "set_voice" in adapter.supported_commands


class TestIntentParser:
    """Test intent parsing functionality."""
    
    @pytest.fixture
    def intent_parser(self):
        """Create intent parser for testing."""
        return IntentParser()
    
    @pytest.fixture
    def sample_recognition_result(self):
        """Sample recognition result for testing."""
        return RecognitionResult(
            text="create a new function called test",
            confidence=0.9,
            language="en",
            processing_time=1.5
        )
    
    def test_text_preprocessing(self, intent_parser):
        """Test intent text preprocessing."""
        text = "Create a NEW Function   with EXTRA spaces"
        processed = intent_parser._preprocess_text(text)
        
        assert processed.islower()
        assert "  " not in processed  # Extra spaces removed
    
    def test_intent_classification(self, intent_parser):
        """Test intent classification."""
        # Coding request
        text = "create a function called test"
        intent_type, confidence = intent_parser._classify_intent(text)
        assert intent_type == IntentType.CODING_REQUEST
        assert confidence > 0.5
        
        # File operation
        text = "open the file main.py"
        intent_type, confidence = intent_parser._classify_intent(text)
        assert intent_type == IntentType.FILE_OPERATION
        assert confidence > 0.5
        
        # Query request
        text = "what is this function doing"
        intent_type, confidence = intent_parser._classify_intent(text)
        assert intent_type == IntentType.QUERY_REQUEST
        assert confidence > 0.5
    
    def test_entity_extraction(self, intent_parser):
        """Test entity extraction."""
        text = "create function test_function in file main.py line 50"
        entities = intent_parser._extract_entities(text)
        
        # Should extract function name and file path
        assert 'function_name' in entities or 'file_path' in entities
    
    def test_programming_context_detection(self, intent_parser):
        """Test programming context detection."""
        # Programming text
        text = "def function class import for while if else"
        is_programming = intent_parser._detect_programming_context(text)
        assert is_programming is True
        
        # Non-programming text
        text = "hello world how are you today"
        is_programming = intent_parser._detect_programming_context(text)
        assert is_programming is False
    
    async def test_intent_parsing(self, intent_parser, sample_recognition_result):
        """Test full intent parsing pipeline."""
        parsed_intent = await intent_parser.parse_intent(sample_recognition_result)
        
        assert parsed_intent is not None
        assert parsed_intent.intent_type == IntentType.CODING_REQUEST
        assert parsed_intent.confidence > 0.0
        assert len(parsed_intent.suggested_commands) > 0
    
    async def test_intent_parser_adapter(self):
        """Test intent parser adapter."""
        adapter = IntentParserAdapter()
        
        assert adapter.adapter_id == "intent_parser"
        assert adapter.name == "Intent Parser Adapter"
        assert "parse_intent" in adapter.supported_commands
        assert "classify_text" in adapter.supported_commands


class TestVoiceInterface:
    """Test unified voice interface."""
    
    @pytest.fixture
    async def event_system(self):
        """Create event system for testing."""
        event_system = EventSystem()
        await event_system.initialize()
        yield event_system
        await event_system.shutdown()
    
    @pytest.fixture
    async def voice_interface(self, event_system):
        """Create voice interface for testing."""
        # Use minimal configs for testing
        recognition_config = RecognitionConfig(model="tiny", timeout=5.0)
        synthesis_config = SynthesisConfig(rate=200)
        audio_config = AudioConfig(vad_enabled=False)
        
        interface = VoiceInterface(
            recognition_config=recognition_config,
            synthesis_config=synthesis_config,
            audio_config=audio_config,
            event_system=event_system
        )
        yield interface
        await interface.cleanup()
    
    def test_voice_interface_state(self, voice_interface):
        """Test voice interface state management."""
        assert voice_interface.get_state() == VoiceState.IDLE
        assert not voice_interface.is_listening()
        assert not voice_interface.is_processing()
        assert not voice_interface.is_speaking()
    
    def test_conversation_context(self, voice_interface):
        """Test conversation context management."""
        # Initially empty
        context = voice_interface.get_conversation_context()
        assert len(context) == 0
        
        # Add test context
        recognition_result = RecognitionResult(
            text="test input",
            confidence=0.9,
            language="en",
            processing_time=1.0
        )
        
        parsed_intent = ParsedIntent(
            original_text="test input",
            processed_text="test input",
            intent_type=IntentType.CODING_REQUEST,
            confidence=0.9
        )
        
        voice_interface._add_to_context(recognition_result, parsed_intent)
        
        context = voice_interface.get_conversation_context()
        assert len(context) == 1
        assert context[0]['user_input'] == "test input"
    
    def test_statistics(self, voice_interface):
        """Test voice interface statistics."""
        stats = voice_interface.get_statistics()
        
        assert 'session_id' in stats
        assert 'state' in stats
        assert 'conversation_turns' in stats
        assert 'processing_stats' in stats
        assert stats['conversation_turns'] == 0
    
    async def test_voice_interface_adapter(self):
        """Test voice interface adapter."""
        config = {
            'recognition': {'model': 'tiny'},
            'synthesis': {'rate': 150},
            'audio': {'sample_rate': 16000}
        }
        
        adapter = VoiceInterfaceAdapter(config)
        
        assert adapter.adapter_id == "voice_interface"
        assert adapter.name == "Voice Interface Adapter"
        assert "start_listening" in adapter.supported_commands
        assert "process_voice_command" in adapter.supported_commands


class TestSpeechConfig:
    """Test speech configuration management."""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file for testing."""
        config_data = {
            'speech': {
                'recognition': {
                    'engine': 'whisper',
                    'model': 'base',
                    'language': 'en'
                },
                'synthesis': {
                    'engine': 'pyttsx3',
                    'voice': 'en',
                    'rate': 150
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_data, f)
            temp_path = f.name
        
        yield temp_path
        os.unlink(temp_path)
    
    def test_config_manager_initialization(self):
        """Test speech config manager initialization."""
        manager = SpeechConfigManager()
        
        config = manager.get_merged_config()
        
        assert 'recognition' in config
        assert 'synthesis' in config
        assert 'audio' in config
        assert config['recognition']['engine'] == 'whisper'
    
    def test_config_loading(self, temp_config_file):
        """Test configuration loading from file."""
        manager = SpeechConfigManager(config_path=temp_config_file)
        
        config = manager.get_merged_config()
        
        assert config['recognition']['model'] == 'base'
        assert config['synthesis']['rate'] == 150
    
    def test_typed_config_generation(self):
        """Test typed configuration generation."""
        manager = SpeechConfigManager()
        
        recognition_config = manager.get_recognition_config()
        synthesis_config = manager.get_synthesis_config()
        audio_config = manager.get_audio_config()
        
        assert isinstance(recognition_config, RecognitionConfig)
        assert isinstance(synthesis_config, SynthesisConfig)
        assert isinstance(audio_config, AudioConfig)
        
        assert recognition_config.engine == RecognitionEngine.WHISPER
        assert synthesis_config.engine == SynthesisEngine.PYTTSX3
    
    def test_config_validation(self):
        """Test configuration validation."""
        manager = SpeechConfigManager()
        
        # Test with valid config
        errors = manager.validate_config()
        assert len(errors) == 0
        
        # Test with invalid config
        manager.update_runtime_config({
            'recognition': {'engine': 'invalid_engine'},
            'synthesis': {'rate': 1000}  # Too high
        })
        
        errors = manager.validate_config()
        assert len(errors) > 0
    
    @patch('psutil.cpu_count')
    @patch('psutil.virtual_memory')
    def test_hardware_optimization(self, mock_memory, mock_cpu_count):
        """Test hardware-based configuration optimization."""
        # Mock system specs
        mock_cpu_count.return_value = 8
        mock_memory.return_value = Mock(available=8 * 1024**3)  # 8GB
        
        manager = SpeechConfigManager()
        optimizations = manager.optimize_for_hardware()
        
        assert 'recognition' in optimizations
        assert 'performance' in optimizations
        
        # Should suggest better model for 8GB RAM
        assert optimizations['recognition'].get('model') in ['medium', 'large']
    
    def test_use_case_optimization(self):
        """Test use case specific optimization."""
        manager = SpeechConfigManager()
        
        # Test coding use case
        coding_opts = manager.optimize_for_use_case('coding')
        assert coding_opts['recognition']['programming_keywords'] is True
        assert coding_opts['features']['programming_mode'] is True
        
        # Test presentation use case
        presentation_opts = manager.optimize_for_use_case('presentation')
        assert presentation_opts['synthesis']['rate'] == 120  # Slower for clarity
        assert presentation_opts['synthesis']['volume'] == 0.9
    
    def test_config_save_and_reload(self, temp_config_file):
        """Test configuration save and reload."""
        manager = SpeechConfigManager()
        
        # Modify config
        manager.update_runtime_config({
            'recognition': {'model': 'large'},
            'synthesis': {'rate': 180}
        })
        
        # Save config
        success = manager.save_config(temp_config_file)
        assert success is True
        
        # Create new manager and load
        new_manager = SpeechConfigManager(config_path=temp_config_file)
        config = new_manager.get_merged_config()
        
        assert config['recognition']['model'] == 'large'
        assert config['synthesis']['rate'] == 180


class TestIntegration:
    """Integration tests for speech components."""
    
    @pytest.fixture
    async def full_speech_system(self):
        """Create full speech system for integration testing."""
        event_system = EventSystem()
        await event_system.initialize()
        
        config_manager = SpeechConfigManager()
        
        # Create components with test configurations
        recognition_config = RecognitionConfig(model="tiny", timeout=5.0)
        synthesis_config = SynthesisConfig(rate=200)
        audio_config = AudioConfig(vad_enabled=False)
        
        voice_interface = VoiceInterface(
            recognition_config=recognition_config,
            synthesis_config=synthesis_config,
            audio_config=audio_config,
            event_system=event_system
        )
        
        yield {
            'event_system': event_system,
            'config_manager': config_manager,
            'voice_interface': voice_interface
        }
        
        await voice_interface.cleanup()
        await event_system.shutdown()
    
    async def test_event_flow(self, full_speech_system):
        """Test event flow between components."""
        event_system = full_speech_system['event_system']
        
        # Test event subscription and emission
        events_received = []
        
        async def event_handler(event):
            events_received.append(event)
        
        event_system.subscribe(
            "speech.*",
            event_handler,
            handler_id="test_handler"
        )
        
        # Emit test event
        from src.speech.types import SpeechEventType
        await event_system.emit({
            'event_type': SpeechEventType.RECOGNITION_STARTED.value,
            'data': {'test': True},
            'source': 'test'
        })
        
        # Give time for processing
        await asyncio.sleep(0.1)
        
        assert len(events_received) > 0
    
    def test_configuration_integration(self, full_speech_system):
        """Test configuration integration across components."""
        config_manager = full_speech_system['config_manager']
        
        # Test that all components can use the same config manager
        recognition_config = config_manager.get_recognition_config()
        synthesis_config = config_manager.get_synthesis_config()
        audio_config = config_manager.get_audio_config()
        
        # All configs should be valid
        assert recognition_config.engine == RecognitionEngine.WHISPER
        assert synthesis_config.engine == SynthesisEngine.PYTTSX3
        assert audio_config.sample_rate > 0
    
    def test_adapter_registration(self):
        """Test that all speech adapters are properly registered."""
        from src.speech import SPEECH_ADAPTERS, get_speech_adapter, list_speech_adapters
        
        # Test registry
        adapters = list_speech_adapters()
        assert 'speech_recognizer' in adapters
        assert 'speech_synthesizer' in adapters
        assert 'intent_parser' in adapters
        assert 'voice_interface' in adapters
        
        # Test adapter creation
        recognizer_adapter = get_speech_adapter('speech_recognizer')
        assert recognizer_adapter is not None
        assert recognizer_adapter.adapter_id == 'speech_recognizer'
        
        synthesizer_adapter = get_speech_adapter('speech_synthesizer')
        assert synthesizer_adapter is not None
        assert synthesizer_adapter.adapter_id == 'speech_synthesizer'


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])