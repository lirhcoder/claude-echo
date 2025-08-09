#!/usr/bin/env python3
"""Speech Processing Demo

This script demonstrates the speech processing capabilities of the Claude Voice Assistant.
"""

import asyncio
import sys
import os
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.speech import (
    VoiceInterface, VoiceState,
    RecognitionConfig, SynthesisConfig, AudioConfig,
    LanguageCode, RecognitionEngine, SynthesisEngine,
    create_voice_pipeline, get_speech_adapter, list_speech_adapters
)
from src.speech.config import SpeechConfigManager
from src.core.event_system import EventSystem


async def demo_basic_components():
    """Demonstrate basic speech components."""
    logger.info("=== Basic Components Demo ===")
    
    # 1. Configuration Management
    logger.info("1. Configuration Management")
    config_manager = SpeechConfigManager()
    
    # Show available adapters
    adapters = list_speech_adapters()
    logger.info(f"Available speech adapters: {adapters}")
    
    # Get typed configurations
    recognition_config = config_manager.get_recognition_config()
    synthesis_config = config_manager.get_synthesis_config()
    audio_config = config_manager.get_audio_config()
    
    logger.info(f"Recognition engine: {recognition_config.engine.value}")
    logger.info(f"Recognition model: {recognition_config.model}")
    logger.info(f"Synthesis engine: {synthesis_config.engine.value}")
    logger.info(f"Audio sample rate: {audio_config.sample_rate}")
    
    # 2. Hardware optimization
    logger.info("\n2. Hardware Optimization")
    try:
        hardware_opts = config_manager.optimize_for_hardware()
        logger.info(f"Hardware optimizations: {hardware_opts}")
    except Exception as e:
        logger.warning(f"Hardware optimization failed: {e}")
    
    # 3. Use case optimization
    logger.info("\n3. Use Case Optimization")
    coding_opts = config_manager.optimize_for_use_case('coding')
    logger.info(f"Coding optimizations: {coding_opts}")


async def demo_individual_adapters():
    """Demonstrate individual speech adapters."""
    logger.info("\n=== Individual Adapters Demo ===")
    
    # 1. Speech Recognizer Adapter
    logger.info("1. Speech Recognizer Adapter")
    recognizer_adapter = get_speech_adapter('speech_recognizer', {
        'model': 'tiny',  # Use tiny model for demo
        'language': 'en',
        'programming_keywords': True
    })
    
    if recognizer_adapter:
        logger.info(f"Recognizer adapter ID: {recognizer_adapter.adapter_id}")
        logger.info(f"Supported commands: {recognizer_adapter.supported_commands}")
        
        # Try to initialize (may fail without proper dependencies)
        try:
            success = await recognizer_adapter.initialize()
            logger.info(f"Recognizer initialization: {success}")
            if success:
                await recognizer_adapter.cleanup()
        except Exception as e:
            logger.warning(f"Recognizer initialization failed: {e}")
    
    # 2. Speech Synthesizer Adapter  
    logger.info("\n2. Speech Synthesizer Adapter")
    synthesizer_adapter = get_speech_adapter('speech_synthesizer', {
        'engine': 'pyttsx3',
        'voice': 'en',
        'rate': 150
    })
    
    if synthesizer_adapter:
        logger.info(f"Synthesizer adapter ID: {synthesizer_adapter.adapter_id}")
        logger.info(f"Supported commands: {synthesizer_adapter.supported_commands}")
        
        try:
            success = await synthesizer_adapter.initialize()
            logger.info(f"Synthesizer initialization: {success}")
            if success:
                await synthesizer_adapter.cleanup()
        except Exception as e:
            logger.warning(f"Synthesizer initialization failed: {e}")
    
    # 3. Intent Parser Adapter
    logger.info("\n3. Intent Parser Adapter")
    parser_adapter = get_speech_adapter('intent_parser')
    
    if parser_adapter:
        logger.info(f"Parser adapter ID: {parser_adapter.adapter_id}")
        logger.info(f"Supported commands: {parser_adapter.supported_commands}")
        
        success = await parser_adapter.initialize()
        logger.info(f"Parser initialization: {success}")
        
        # Test intent classification
        from src.speech.types import RecognitionResult
        test_result = RecognitionResult(
            text="create a new function called test",
            confidence=0.9,
            language="en",
            processing_time=1.0
        )
        
        try:
            command_result = await parser_adapter.execute_command(
                "parse_intent",
                {"recognition_result": test_result}
            )
            
            if command_result.success:
                parsed_intent = command_result.data['parsed_intent']
                logger.info(f"Parsed intent type: {parsed_intent['intent_type']}")
                logger.info(f"Intent confidence: {parsed_intent['confidence']}")
                logger.info(f"Suggested commands: {parsed_intent['suggested_commands']}")
        except Exception as e:
            logger.warning(f"Intent parsing test failed: {e}")
        
        await parser_adapter.cleanup()


async def demo_voice_interface():
    """Demonstrate the unified voice interface."""
    logger.info("\n=== Voice Interface Demo ===")
    
    # Create event system
    event_system = EventSystem()
    await event_system.initialize()
    
    # Create voice interface with test configurations
    recognition_config = RecognitionConfig(
        model="tiny",  # Use smallest model for demo
        language=LanguageCode.ENGLISH,
        timeout=5.0
    )
    
    synthesis_config = SynthesisConfig(
        voice="en",
        rate=180,
        volume=0.8
    )
    
    audio_config = AudioConfig(
        sample_rate=16000,
        vad_enabled=False  # Disable VAD for demo
    )
    
    voice_interface = VoiceInterface(
        recognition_config=recognition_config,
        synthesis_config=synthesis_config,
        audio_config=audio_config,
        event_system=event_system
    )
    
    # Subscribe to voice events for demo
    events_received = []
    
    async def event_handler(event):
        events_received.append(event)
        logger.info(f"Voice event: {event.event_type} from {event.source}")
    
    event_system.subscribe(
        "speech.*",
        event_handler,
        handler_id="demo_handler"
    )
    
    try:
        # Initialize voice interface
        logger.info("Initializing voice interface...")
        success = await voice_interface.initialize()
        logger.info(f"Voice interface initialization: {success}")
        
        if success:
            # Show initial state
            logger.info(f"Initial state: {voice_interface.get_state().value}")
            logger.info(f"Is listening: {voice_interface.is_listening()}")
            logger.info(f"Is processing: {voice_interface.is_processing()}")
            
            # Get statistics
            stats = voice_interface.get_statistics()
            logger.info(f"Session ID: {stats['session_id']}")
            logger.info(f"Conversation turns: {stats['conversation_turns']}")
            
            # Show component statistics
            component_stats = stats['component_stats']
            logger.info(f"Component stats available: {list(component_stats.keys())}")
    
    except Exception as e:
        logger.error(f"Voice interface demo failed: {e}")
    
    finally:
        # Cleanup
        await voice_interface.cleanup()
        await event_system.shutdown()
    
    logger.info(f"Total events received: {len(events_received)}")


async def demo_voice_interface_adapter():
    """Demonstrate voice interface adapter."""
    logger.info("\n=== Voice Interface Adapter Demo ===")
    
    config = {
        'recognition': {
            'model': 'tiny',
            'language': 'en',
            'programming_keywords': True
        },
        'synthesis': {
            'engine': 'pyttsx3',
            'voice': 'en',
            'rate': 150
        },
        'audio': {
            'sample_rate': 16000,
            'vad_enabled': False
        }
    }
    
    adapter = get_speech_adapter('voice_interface', config)
    
    if adapter:
        logger.info(f"Voice interface adapter ID: {adapter.adapter_id}")
        logger.info(f"Supported commands: {adapter.supported_commands}")
        
        try:
            # Initialize adapter
            success = await adapter.initialize()
            logger.info(f"Adapter initialization: {success}")
            
            if success:
                # Test get_state command
                result = await adapter.execute_command("get_state", {})
                if result.success:
                    state_data = result.data
                    logger.info(f"Current state: {state_data['state']}")
                    logger.info(f"Listening: {state_data['listening']}")
                
                # Test get_statistics command
                result = await adapter.execute_command("get_statistics", {})
                if result.success:
                    stats = result.data
                    logger.info(f"Session ID: {stats.get('session_id', 'N/A')}")
                
                # Test command suggestions
                suggestions = await adapter.get_command_suggestions()
                logger.info(f"Command suggestions: {suggestions}")
            
        except Exception as e:
            logger.error(f"Voice interface adapter demo failed: {e}")
        
        finally:
            await adapter.cleanup()


async def demo_configuration_validation():
    """Demonstrate configuration validation."""
    logger.info("\n=== Configuration Validation Demo ===")
    
    config_manager = SpeechConfigManager()
    
    # Test valid configuration
    logger.info("1. Validating default configuration...")
    errors = config_manager.validate_config()
    if errors:
        logger.warning(f"Validation errors: {errors}")
    else:
        logger.info("Default configuration is valid!")
    
    # Test invalid configuration
    logger.info("\n2. Testing invalid configuration...")
    config_manager.update_runtime_config({
        'recognition': {
            'engine': 'invalid_engine',
            'model': 'invalid_model'
        },
        'synthesis': {
            'rate': 1000,  # Too high
            'volume': 2.0  # Too high
        },
        'audio': {
            'sample_rate': 123,  # Invalid
            'vad_threshold': 5.0  # Too high
        }
    })
    
    errors = config_manager.validate_config()
    logger.info(f"Validation errors found: {len(errors)}")
    for error in errors:
        logger.warning(f"  - {error}")
    
    # Show configuration summary
    logger.info("\n3. Configuration summary:")
    summary = config_manager.get_config_summary()
    logger.info(f"Recognition engine: {summary['recognition']['engine']}")
    logger.info(f"Synthesis engine: {summary['synthesis']['engine']}")
    logger.info(f"Features enabled: {summary['features']}")


async def main():
    """Main demo function."""
    logger.info("🎤 Claude Voice Assistant - Speech Processing Demo")
    logger.info("=" * 60)
    
    try:
        # Run all demos
        await demo_basic_components()
        await demo_individual_adapters()
        await demo_voice_interface()
        await demo_voice_interface_adapter()
        await demo_configuration_validation()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Speech processing demo completed successfully!")
        logger.info("\nNote: Some features may not work without proper audio hardware")
        logger.info("and dependencies (Whisper, pyttsx3, pyaudio, webrtcvad)")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Demo interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        logger.exception("Full traceback:")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    
    # Run demo
    asyncio.run(main())