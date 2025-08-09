"""Speech Configuration Management Module

This module provides configuration management and parameter optimization
for speech processing components.
"""

import os
import yaml
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging

from loguru import logger

from ..core.config_manager import ConfigManager
from .types import (
    RecognitionConfig, SynthesisConfig, AudioConfig,
    RecognitionEngine, SynthesisEngine, LanguageCode, AudioFormat
)


class SpeechConfigManager:
    """
    Speech-specific configuration manager.
    
    Features:
    - Hierarchical configuration loading
    - Environment-specific overrides
    - Dynamic parameter optimization
    - Configuration validation
    - Hot reloading support
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 base_config_manager: Optional[ConfigManager] = None):
        """
        Initialize the speech configuration manager.
        
        Args:
            config_path: Optional path to speech-specific config file
            base_config_manager: Base configuration manager for system-wide settings
        """
        self.config_path = config_path
        self.base_config_manager = base_config_manager
        
        # Configuration storage
        self._speech_config: Dict[str, Any] = {}
        self._default_config: Dict[str, Any] = {}
        self._user_config: Dict[str, Any] = {}
        self._runtime_config: Dict[str, Any] = {}
        
        # Load configurations
        self._load_default_config()
        if config_path:
            self._load_speech_config(config_path)
        self._load_environment_overrides()
        
        logger.info("Speech configuration manager initialized")
    
    def _load_default_config(self) -> None:
        """Load default speech configuration."""
        self._default_config = {
            'recognition': {
                'engine': 'whisper',
                'model': 'base',
                'language': 'auto',
                'timeout': 30.0,
                'use_gpu': False,
                'beam_size': 5,
                'programming_keywords': True,
                'abbreviation_expansion': True,
                'punctuation_inference': True,
                'max_audio_length': 30.0,
                'chunk_processing': True
            },
            'synthesis': {
                'engine': 'pyttsx3',
                'voice': 'zh',
                'rate': 150,
                'volume': 0.8,
                'pitch': None,
                'voice_id': None,
                'output_format': 'wav',
                'quality': 'medium',
                'pre_processing': True,
                'post_processing': True,
                'silence_padding': 0.2
            },
            'audio': {
                'sample_rate': 16000,
                'channels': 1,
                'bit_depth': 16,
                'chunk_size': 1024,
                'format': 'wav',
                'noise_reduction': True,
                'echo_cancellation': True,
                'auto_gain_control': True,
                'vad_enabled': True,
                'vad_threshold': 0.5,
                'vad_padding_ms': 300
            },
            'performance': {
                'max_concurrent_recognitions': 2,
                'max_concurrent_syntheses': 3,
                'buffer_size_mb': 50,
                'cache_enabled': True,
                'cache_size_mb': 100,
                'optimization_level': 'balanced'  # fast, balanced, quality
            },
            'features': {
                'continuous_listening': False,
                'voice_activity_detection': True,
                'noise_suppression': True,
                'echo_cancellation': True,
                'automatic_gain_control': True,
                'programming_mode': True,
                'multilingual_support': True,
                'context_awareness': True
            },
            'quality': {
                'recognition_min_confidence': 0.7,
                'synthesis_quality_preset': 'balanced',
                'audio_enhancement': True,
                'real_time_processing': True,
                'latency_optimization': True
            }
        }
    
    def _load_speech_config(self, config_path: str) -> None:
        """Load speech-specific configuration file."""
        try:
            path = Path(config_path)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                    
                if 'speech' in config_data:
                    self._speech_config = config_data['speech']
                else:
                    self._speech_config = config_data
                    
                logger.info(f"Loaded speech configuration from: {config_path}")
            else:
                logger.warning(f"Speech config file not found: {config_path}")
                
        except Exception as e:
            logger.error(f"Failed to load speech config: {e}")
    
    def _load_environment_overrides(self) -> None:
        """Load configuration overrides from environment variables."""
        env_overrides = {}
        
        # Recognition settings
        if os.getenv('SPEECH_RECOGNITION_ENGINE'):
            env_overrides.setdefault('recognition', {})['engine'] = os.getenv('SPEECH_RECOGNITION_ENGINE')
        
        if os.getenv('SPEECH_RECOGNITION_MODEL'):
            env_overrides.setdefault('recognition', {})['model'] = os.getenv('SPEECH_RECOGNITION_MODEL')
        
        if os.getenv('SPEECH_LANGUAGE'):
            env_overrides.setdefault('recognition', {})['language'] = os.getenv('SPEECH_LANGUAGE')
        
        # Synthesis settings
        if os.getenv('SPEECH_SYNTHESIS_ENGINE'):
            env_overrides.setdefault('synthesis', {})['engine'] = os.getenv('SPEECH_SYNTHESIS_ENGINE')
        
        if os.getenv('SPEECH_VOICE'):
            env_overrides.setdefault('synthesis', {})['voice'] = os.getenv('SPEECH_VOICE')
        
        if os.getenv('SPEECH_RATE'):
            try:
                env_overrides.setdefault('synthesis', {})['rate'] = int(os.getenv('SPEECH_RATE'))
            except ValueError:
                pass
        
        if os.getenv('SPEECH_VOLUME'):
            try:
                env_overrides.setdefault('synthesis', {})['volume'] = float(os.getenv('SPEECH_VOLUME'))
            except ValueError:
                pass
        
        # Audio settings
        if os.getenv('SPEECH_SAMPLE_RATE'):
            try:
                env_overrides.setdefault('audio', {})['sample_rate'] = int(os.getenv('SPEECH_SAMPLE_RATE'))
            except ValueError:
                pass
        
        # Feature toggles
        if os.getenv('SPEECH_PROGRAMMING_MODE'):
            env_overrides.setdefault('features', {})['programming_mode'] = \
                os.getenv('SPEECH_PROGRAMMING_MODE').lower() == 'true'
        
        if os.getenv('SPEECH_VAD_ENABLED'):
            env_overrides.setdefault('audio', {})['vad_enabled'] = \
                os.getenv('SPEECH_VAD_ENABLED').lower() == 'true'
        
        if env_overrides:
            self._runtime_config.update(env_overrides)
            logger.info(f"Applied {len(env_overrides)} environment overrides")
    
    def get_merged_config(self) -> Dict[str, Any]:
        """Get merged configuration from all sources."""
        merged = {}
        
        # Merge in order: default -> base_config -> speech_config -> runtime -> user
        self._deep_merge(merged, self._default_config)
        
        if self.base_config_manager:
            base_speech_config = self.base_config_manager.get('speech', {})
            self._deep_merge(merged, base_speech_config)
        
        self._deep_merge(merged, self._speech_config)
        self._deep_merge(merged, self._runtime_config)
        self._deep_merge(merged, self._user_config)
        
        return merged
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep merge source dictionary into target."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def get_recognition_config(self) -> RecognitionConfig:
        """Get recognition configuration."""
        config = self.get_merged_config().get('recognition', {})
        
        return RecognitionConfig(
            engine=RecognitionEngine(config.get('engine', 'whisper')),
            model=config.get('model', 'base'),
            language=LanguageCode(config.get('language', 'auto')),
            timeout=config.get('timeout', 30.0),
            whisper_model_path=config.get('whisper_model_path'),
            use_gpu=config.get('use_gpu', False),
            beam_size=config.get('beam_size', 5),
            programming_keywords=config.get('programming_keywords', True),
            abbreviation_expansion=config.get('abbreviation_expansion', True),
            punctuation_inference=config.get('punctuation_inference', True),
            max_audio_length=config.get('max_audio_length', 30.0),
            chunk_processing=config.get('chunk_processing', True)
        )
    
    def get_synthesis_config(self) -> SynthesisConfig:
        """Get synthesis configuration."""
        config = self.get_merged_config().get('synthesis', {})
        
        return SynthesisConfig(
            engine=SynthesisEngine(config.get('engine', 'pyttsx3')),
            voice=config.get('voice', 'zh'),
            rate=config.get('rate', 150),
            volume=config.get('volume', 0.8),
            pitch=config.get('pitch'),
            voice_id=config.get('voice_id'),
            output_format=AudioFormat(config.get('output_format', 'wav')),
            quality=config.get('quality', 'medium'),
            pre_processing=config.get('pre_processing', True),
            post_processing=config.get('post_processing', True),
            silence_padding=config.get('silence_padding', 0.2)
        )
    
    def get_audio_config(self) -> AudioConfig:
        """Get audio configuration."""
        config = self.get_merged_config().get('audio', {})
        
        return AudioConfig(
            sample_rate=config.get('sample_rate', 16000),
            channels=config.get('channels', 1),
            bit_depth=config.get('bit_depth', 16),
            chunk_size=config.get('chunk_size', 1024),
            format=AudioFormat(config.get('format', 'wav')),
            noise_reduction=config.get('noise_reduction', True),
            echo_cancellation=config.get('echo_cancellation', True),
            auto_gain_control=config.get('auto_gain_control', True),
            vad_enabled=config.get('vad_enabled', True),
            vad_threshold=config.get('vad_threshold', 0.5),
            vad_padding_ms=config.get('vad_padding_ms', 300)
        )
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return self.get_merged_config().get('performance', {})
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature configuration."""
        return self.get_merged_config().get('features', {})
    
    def get_quality_config(self) -> Dict[str, Any]:
        """Get quality configuration."""
        return self.get_merged_config().get('quality', {})
    
    def set_user_config(self, config: Dict[str, Any]) -> None:
        """Set user-specific configuration overrides."""
        self._user_config = config
        logger.info("User configuration updated")
    
    def update_runtime_config(self, config: Dict[str, Any]) -> None:
        """Update runtime configuration."""
        self._deep_merge(self._runtime_config, config)
        logger.info("Runtime configuration updated")
    
    def validate_config(self) -> List[str]:
        """
        Validate current configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        config = self.get_merged_config()
        
        # Validate recognition config
        recognition = config.get('recognition', {})
        
        # Engine validation
        engine = recognition.get('engine')
        if engine not in [e.value for e in RecognitionEngine]:
            errors.append(f"Invalid recognition engine: {engine}")
        
        # Model validation for Whisper
        if engine == 'whisper':
            model = recognition.get('model', 'base')
            valid_models = ['tiny', 'base', 'small', 'medium', 'large']
            if model not in valid_models:
                errors.append(f"Invalid Whisper model: {model}")
        
        # Language validation
        language = recognition.get('language')
        if language and language not in [l.value for l in LanguageCode]:
            errors.append(f"Invalid language code: {language}")
        
        # Validate synthesis config
        synthesis = config.get('synthesis', {})
        
        # Engine validation
        engine = synthesis.get('engine')
        if engine not in [e.value for e in SynthesisEngine]:
            errors.append(f"Invalid synthesis engine: {engine}")
        
        # Rate validation
        rate = synthesis.get('rate', 150)
        if not 50 <= rate <= 500:
            errors.append(f"Speech rate out of range (50-500): {rate}")
        
        # Volume validation
        volume = synthesis.get('volume', 0.8)
        if not 0.0 <= volume <= 1.0:
            errors.append(f"Volume out of range (0.0-1.0): {volume}")
        
        # Validate audio config
        audio = config.get('audio', {})
        
        # Sample rate validation
        sample_rate = audio.get('sample_rate', 16000)
        valid_rates = [8000, 16000, 22050, 44100, 48000]
        if sample_rate not in valid_rates:
            errors.append(f"Unsupported sample rate: {sample_rate}")
        
        # Channels validation
        channels = audio.get('channels', 1)
        if channels not in [1, 2]:
            errors.append(f"Unsupported channel count: {channels}")
        
        # VAD threshold validation
        vad_threshold = audio.get('vad_threshold', 0.5)
        if not 0.0 <= vad_threshold <= 1.0:
            errors.append(f"VAD threshold out of range (0.0-1.0): {vad_threshold}")
        
        return errors
    
    def optimize_for_hardware(self) -> Dict[str, Any]:
        """
        Optimize configuration based on hardware capabilities.
        
        Returns:
            Optimized configuration parameters
        """
        import psutil
        import torch
        
        optimizations = {}
        
        # CPU optimization
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory optimization
        memory = psutil.virtual_memory()
        available_memory_gb = memory.available / (1024**3)
        
        # GPU detection
        has_cuda = torch.cuda.is_available() if 'torch' in globals() else False
        
        # Recognition optimizations
        recognition_opts = {}
        
        if available_memory_gb < 2:
            recognition_opts['model'] = 'tiny'
            recognition_opts['chunk_processing'] = True
        elif available_memory_gb < 4:
            recognition_opts['model'] = 'base'
        elif available_memory_gb >= 8:
            recognition_opts['model'] = 'medium' if not has_cuda else 'large'
        
        if has_cuda:
            recognition_opts['use_gpu'] = True
            recognition_opts['beam_size'] = 10
        else:
            recognition_opts['use_gpu'] = False
            recognition_opts['beam_size'] = 5
        
        optimizations['recognition'] = recognition_opts
        
        # Performance optimizations
        performance_opts = {
            'max_concurrent_recognitions': min(cpu_count // 2, 4),
            'max_concurrent_syntheses': min(cpu_count, 6),
            'buffer_size_mb': min(int(available_memory_gb * 10), 100)
        }
        
        if cpu_percent > 80:
            performance_opts['optimization_level'] = 'fast'
        elif available_memory_gb > 8:
            performance_opts['optimization_level'] = 'quality'
        else:
            performance_opts['optimization_level'] = 'balanced'
        
        optimizations['performance'] = performance_opts
        
        # Audio optimizations
        audio_opts = {}
        
        if cpu_percent > 70:
            audio_opts['noise_reduction'] = False
            audio_opts['echo_cancellation'] = False
        
        optimizations['audio'] = audio_opts
        
        logger.info(f"Generated hardware optimizations: {optimizations}")
        return optimizations
    
    def optimize_for_use_case(self, use_case: str) -> Dict[str, Any]:
        """
        Optimize configuration for specific use cases.
        
        Args:
            use_case: Use case name ('coding', 'general', 'presentation', 'dictation')
            
        Returns:
            Use case optimized configuration
        """
        optimizations = {}
        
        if use_case == 'coding':
            optimizations.update({
                'recognition': {
                    'programming_keywords': True,
                    'abbreviation_expansion': True,
                    'punctuation_inference': True,
                    'language': 'en'  # Programming terms are often in English
                },
                'features': {
                    'programming_mode': True,
                    'context_awareness': True
                },
                'quality': {
                    'recognition_min_confidence': 0.8
                }
            })
        
        elif use_case == 'general':
            optimizations.update({
                'recognition': {
                    'language': 'auto'
                },
                'features': {
                    'multilingual_support': True
                },
                'quality': {
                    'recognition_min_confidence': 0.7
                }
            })
        
        elif use_case == 'presentation':
            optimizations.update({
                'synthesis': {
                    'rate': 120,  # Slower for clarity
                    'volume': 0.9,
                    'quality': 'high'
                },
                'audio': {
                    'noise_reduction': True,
                    'echo_cancellation': True
                }
            })
        
        elif use_case == 'dictation':
            optimizations.update({
                'recognition': {
                    'punctuation_inference': True,
                    'max_audio_length': 60.0  # Longer segments
                },
                'quality': {
                    'recognition_min_confidence': 0.85
                },
                'features': {
                    'continuous_listening': True
                }
            })
        
        logger.info(f"Generated use case optimizations for '{use_case}': {optimizations}")
        return optimizations
    
    def save_config(self, file_path: str) -> bool:
        """
        Save current configuration to file.
        
        Args:
            file_path: Path to save configuration
            
        Returns:
            True if successful
        """
        try:
            config = self.get_merged_config()
            
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump({'speech': config}, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def reload_config(self) -> bool:
        """
        Reload configuration from file.
        
        Returns:
            True if successful
        """
        try:
            if self.config_path:
                self._load_speech_config(self.config_path)
            self._load_environment_overrides()
            
            logger.info("Configuration reloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for debugging."""
        config = self.get_merged_config()
        
        summary = {
            'recognition': {
                'engine': config.get('recognition', {}).get('engine'),
                'model': config.get('recognition', {}).get('model'),
                'language': config.get('recognition', {}).get('language')
            },
            'synthesis': {
                'engine': config.get('synthesis', {}).get('engine'),
                'voice': config.get('synthesis', {}).get('voice'),
                'rate': config.get('synthesis', {}).get('rate')
            },
            'audio': {
                'sample_rate': config.get('audio', {}).get('sample_rate'),
                'vad_enabled': config.get('audio', {}).get('vad_enabled')
            },
            'features': config.get('features', {}),
            'validation_errors': self.validate_config()
        }
        
        return summary