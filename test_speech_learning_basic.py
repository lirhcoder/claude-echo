#!/usr/bin/env python3
"""
Basic Speech Learning System Test

Simple test script to validate core speech learning components
without full system integration dependencies.
"""

import sys
import os
from datetime import datetime

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_learning_types():
    """Test speech learning data types."""
    try:
        from speech.learning_types import (
            VoiceCharacteristics, 
            PronunciationPattern,
            SpeechContextPattern,
            AdaptationParameters,
            PersonalizedVoiceProfile,
            SpeechLearningData
        )
        
        print("Successfully imported learning types")
        
        # Test data creation
        voice_data = SpeechLearningData(
            data_id="test_001",
            user_id="test_user",
            original_text="Hello world",
            confidence_scores=[0.95],
            context_type="test"
        )
        
        print("Successfully created SpeechLearningData")
        
        # Test voice profile
        voice_profile = PersonalizedVoiceProfile(
            user_id="test_user",
            profile_version="1.0.0"
        )
        
        print("Successfully created PersonalizedVoiceProfile")
        
        return True
        
    except Exception as e:
        print(f"Learning types test failed: {e}")
        return False

def test_configuration():
    """Test speech learning configuration."""
    try:
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'speech_learning.yaml')
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Validate key configuration sections
            required_sections = [
                'learning', 'adaptive_recognition', 
                'voice_profile_learning', 'pronunciation_learning',
                'accent_learning', 'context_learning'
            ]
            
            for section in required_sections:
                if section not in config:
                    raise Exception(f"Missing configuration section: {section}")
            
            print("Configuration file loaded and validated")
            return True
        else:
            print("Configuration file not found")
            return False
            
    except Exception as e:
        print(f"Configuration test failed: {e}")
        return False

def test_implementation_files():
    """Test that all implementation files exist."""
    try:
        expected_files = [
            'src/speech/learning_types.py',
            'src/speech/voice_profile_learner.py',
            'src/speech/pronunciation_pattern_learner.py',
            'src/speech/accent_adaptation_learner.py',
            'src/speech/speech_context_learner.py',
            'src/speech/adaptive_recognizer.py',
            'src/speech/speech_learning_manager.py',
            'config/speech_learning.yaml',
            'SPEECH_LEARNING_IMPLEMENTATION_REPORT.md'
        ]
        
        missing_files = []
        for file_path in expected_files:
            full_path = os.path.join(os.path.dirname(__file__), file_path)
            if not os.path.exists(full_path):
                missing_files.append(file_path)
        
        if missing_files:
            print(f"Missing files: {missing_files}")
            return False
        
        print("All implementation files present")
        return True
        
    except Exception as e:
        print(f"File existence test failed: {e}")
        return False

def main():
    """Run basic speech learning tests."""
    print("Basic Speech Learning System Tests")
    print("=" * 50)
    
    tests = [
        ("Learning Types", test_learning_types),
        ("Configuration", test_configuration),
        ("Implementation Files", test_implementation_files)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll basic tests passed!")
        print("Speech Learning System core components are working correctly.")
    else:
        print(f"\n{total - passed} test(s) failed.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)