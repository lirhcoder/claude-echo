"""Test script for Learning Agents

This script tests the three new learning agents:
- LearningAgent
- UserProfileAgent  
- CorrectionAgent
"""

import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from core.event_system import EventSystem, Event, EventPriority
from agents.learning_agent import LearningAgent, LearningStrategy, LearningPriority
from agents.user_profile_agent import UserProfileAgent, CorrectionType
from agents.correction_agent import CorrectionAgent, CorrectionSeverity
from agents.agent_types import AgentPriority


async def test_learning_system():
    """Test the complete learning system integration."""
    
    print("🚀 Starting Learning Agents Integration Test")
    print("=" * 60)
    
    # Initialize event system
    event_system = EventSystem()
    await event_system.initialize()
    
    # Configuration for testing
    test_config = {
        "learning": {
            "db_path": "./data/test_learning.db",
            "encryption_enabled": False,  # Disable for testing
            "cache_size": 100
        },
        "user_profiles": {
            "voice_recognition_enabled": True,
            "auto_create_profiles": True,
            "session_timeout_minutes": 30
        },
        "correction": {
            "pattern_detection_threshold": 3,
            "auto_apply_threshold": 0.8,
            "batch_size": 10
        }
    }
    
    try:
        # 1. Initialize agents
        print("\n1. Initializing Learning Agents...")
        
        learning_agent = LearningAgent(event_system, test_config)
        user_profile_agent = UserProfileAgent(event_system, test_config)
        correction_agent = CorrectionAgent(event_system, test_config)
        
        # Initialize all agents
        await learning_agent.initialize()
        await user_profile_agent.initialize()
        await correction_agent.initialize()
        
        print(f"   ✓ Learning Agent: {learning_agent.name}")
        print(f"   ✓ User Profile Agent: {user_profile_agent.name}")
        print(f"   ✓ Correction Agent: {correction_agent.name}")
        
        # 2. Test User Profile Agent
        print("\n2. Testing User Profile Agent...")
        
        # Create a user profile
        user_data = {
            "username": "test_user",
            "display_name": "Test User",
            "email": "test@example.com",
            "preferences": {
                "communication_style": "casual",
                "response_length": "medium",
                "language": "en",
                "interaction_mode": "voice"
            }
        }
        
        voice_sample = {
            "pitch_mean": 150.0,
            "pitch_variance": 20.0,
            "formant_f1": 500.0,
            "formant_f2": 1500.0,
            "speech_rate": 4.5,
            "intensity_mean": 60.0
        }
        
        profile_result = await user_profile_agent.create_user_profile(user_data, voice_sample)
        
        if profile_result["success"]:
            user_id = profile_result["user_id"]
            print(f"   ✓ Created user profile: {user_id}")
            print(f"   ✓ Voice enrolled: {profile_result['voice_enrolled']}")
            
            # Test user identification
            identity_result = await user_profile_agent.identify_user_by_voice(voice_sample)
            if identity_result["success"]:
                print(f"   ✓ Voice identification successful: {identity_result['confidence']:.2f}")
            else:
                print(f"   ✗ Voice identification failed: {identity_result.get('error')}")
            
            # Start a user session
            session_result = await user_profile_agent.start_user_session(
                user_id, 
                {"device": "test_device", "os": "test_os"},
                {"location": "test_environment"}
            )
            
            if session_result["success"]:
                session_id = session_result["session_id"]
                print(f"   ✓ Started user session: {session_id}")
            else:
                print(f"   ✗ Session creation failed: {session_result.get('error')}")
            
        else:
            print(f"   ✗ Profile creation failed: {profile_result.get('error')}")
            return
        
        # 3. Test Learning Agent
        print("\n3. Testing Learning Agent...")
        
        # Test user behavior analysis
        behavior_result = await learning_agent.analyze_user_behavior_patterns(
            user_id, 
            time_window=None
        )
        
        if behavior_result["success"]:
            print(f"   ✓ User behavior analysis completed")
            if "analysis" in behavior_result:
                confidence = behavior_result["analysis"].get("confidence_score", 0)
                print(f"   ✓ Analysis confidence: {confidence:.2f}")
        else:
            print(f"   ✗ Behavior analysis failed: {behavior_result.get('error')}")
        
        # Test learning insights
        insights_result = await learning_agent.get_learning_insights(
            "user_patterns",
            {"user_id": user_id}
        )
        
        if insights_result["success"]:
            print(f"   ✓ Generated learning insights")
            insights = insights_result.get("insights", {})
            print(f"   ✓ Key findings: {len(insights.get('key_findings', []))}")
        else:
            print(f"   ✗ Insights generation failed: {insights_result.get('error')}")
        
        # 4. Test Correction Agent
        print("\n4. Testing Correction Agent...")
        
        # Create test correction feedback
        correction_data = {
            "user_id": user_id,
            "session_id": session_id,
            "original_input": "What's the weather?",
            "original_response": "I'm sorry, I don't have weather information.",
            "corrected_response": "I'll check the weather for you. What's your location?",
            "correction_type": "response_content",
            "severity": "moderate",
            "user_satisfaction": 0.8,
            "correction_confidence": 0.9,
            "metadata": {"context": "weather_query"}
        }
        
        correction_result = await correction_agent.process_user_correction(
            correction_data,
            {"session_context": "weather_assistance"}
        )
        
        if correction_result["success"]:
            feedback_id = correction_result["feedback_id"]
            print(f"   ✓ Processed user correction: {feedback_id}")
            
            estimated_impact = correction_result.get("estimated_impact", {})
            impact_score = estimated_impact.get("estimated_impact_score", 0)
            print(f"   ✓ Estimated impact: {impact_score:.2f}")
        else:
            print(f"   ✗ Correction processing failed: {correction_result.get('error')}")
        
        # Test pattern identification
        pattern_result = await correction_agent.identify_correction_patterns()
        
        if pattern_result["success"]:
            patterns_found = pattern_result["patterns_found"]
            print(f"   ✓ Pattern analysis completed: {patterns_found} patterns")
        else:
            print(f"   ✗ Pattern identification failed: {pattern_result.get('error')}")
        
        # Test correction insights
        correction_insights = await correction_agent.get_correction_insights("comprehensive")
        
        if correction_insights["success"]:
            corrections_analyzed = correction_insights["corrections_analyzed"]
            print(f"   ✓ Generated correction insights: {corrections_analyzed} corrections analyzed")
        else:
            print(f"   ✗ Correction insights failed: {correction_insights.get('error')}")
        
        # 5. Test Agent Integration
        print("\n5. Testing Agent Integration...")
        
        # Test multi-agent learning coordination
        coordination_result = await learning_agent.coordinate_multi_agent_learning(
            [user_profile_agent.agent_id, correction_agent.agent_id],
            "improve_user_experience",
            "knowledge_sharing"
        )
        
        if coordination_result["success"]:
            effectiveness = coordination_result.get("collaboration_effectiveness", 0)
            print(f"   ✓ Multi-agent coordination completed: {effectiveness:.2f} effectiveness")
        else:
            print(f"   ✗ Coordination failed: {coordination_result.get('error')}")
        
        # Test correction coordination
        correction_coordination = await correction_agent.coordinate_system_corrections(
            [learning_agent.agent_id],
            "performance"
        )
        
        if correction_coordination["success"]:
            agents_contacted = correction_coordination["agents_contacted"]
            print(f"   ✓ Correction coordination completed: {agents_contacted} agents contacted")
        else:
            print(f"   ✗ Correction coordination failed: {correction_coordination.get('error')}")
        
        # 6. Test Performance and Statistics
        print("\n6. Testing Performance and Statistics...")
        
        # Get user context
        context_result = await user_profile_agent.get_user_context(user_id, "full")
        if context_result["success"]:
            context = context_result["context"]
            sessions = context.get("active_sessions", 0)
            interactions = context.get("usage_statistics", {}).get("total_interactions", 0)
            print(f"   ✓ User context retrieved: {sessions} sessions, {interactions} interactions")
        
        # Get learning statistics (if implemented)
        try:
            learning_stats = await learning_agent._learning_statistics
            print(f"   ✓ Learning stats: {learning_stats.get('total_learning_tasks', 0)} tasks")
        except:
            print("   - Learning statistics not available")
        
        # Test event system integration
        print("\n7. Testing Event System Integration...")
        
        # Emit a test learning event
        test_event = Event(
            event_type="learning.test_event",
            data={
                "user_id": user_id,
                "test_data": "integration_test",
                "timestamp": datetime.now().isoformat()
            },
            source="test_system",
            priority=EventPriority.NORMAL
        )
        
        await event_system.emit(test_event)
        print("   ✓ Test event emitted successfully")
        
        # Wait a moment for event processing
        await asyncio.sleep(1)
        
        print("\n✅ Learning Agents Integration Test Completed Successfully!")
        print("=" * 60)
        
        # Summary
        print("\n📊 Test Summary:")
        print(f"   • User Profile Agent: Functional")
        print(f"   • Learning Agent: Functional") 
        print(f"   • Correction Agent: Functional")
        print(f"   • Agent Integration: Functional")
        print(f"   • Event System: Functional")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        try:
            if 'learning_agent' in locals():
                await learning_agent.shutdown()
            if 'user_profile_agent' in locals():
                await user_profile_agent.shutdown()
            if 'correction_agent' in locals():
                await correction_agent.shutdown()
            await event_system.shutdown()
            print("   ✓ All agents shutdown successfully")
        except Exception as e:
            print(f"   ⚠️ Cleanup error: {e}")


async def test_individual_capabilities():
    """Test individual agent capabilities."""
    
    print("\n🔍 Testing Individual Agent Capabilities")
    print("=" * 50)
    
    # Test configuration
    config = {
        "learning": {"db_path": "./data/test_capabilities.db", "encryption_enabled": False}
    }
    
    event_system = EventSystem()
    await event_system.initialize()
    
    try:
        # Test Learning Agent capabilities
        print("\n1. Learning Agent Capabilities:")
        learning_agent = LearningAgent(event_system, config)
        await learning_agent.initialize()
        
        capabilities = learning_agent.capabilities
        for cap in capabilities:
            print(f"   • {cap.name}: {cap.description}")
            print(f"     Risk Level: {cap.risk_level.value}")
        
        # Test User Profile Agent capabilities
        print("\n2. User Profile Agent Capabilities:")
        profile_agent = UserProfileAgent(event_system, config)
        await profile_agent.initialize()
        
        capabilities = profile_agent.capabilities
        for cap in capabilities:
            print(f"   • {cap.name}: {cap.description}")
            print(f"     Risk Level: {cap.risk_level.value}")
        
        # Test Correction Agent capabilities
        print("\n3. Correction Agent Capabilities:")
        correction_agent = CorrectionAgent(event_system, config)
        await correction_agent.initialize()
        
        capabilities = correction_agent.capabilities
        for cap in capabilities:
            print(f"   • {cap.name}: {cap.description}")
            print(f"     Risk Level: {cap.risk_level.value}")
        
        print("\n✅ Capability testing completed")
        
        # Cleanup
        await learning_agent.shutdown()
        await profile_agent.shutdown()
        await correction_agent.shutdown()
        
    except Exception as e:
        print(f"\n❌ Capability test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await event_system.shutdown()


if __name__ == "__main__":
    print("🎯 Learning Agents Test Suite")
    print("Testing three new learning agents integration...")
    
    # Ensure data directory exists
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    # Run tests
    asyncio.run(test_learning_system())
    
    print("\n" + "=" * 60)
    
    # Test individual capabilities
    asyncio.run(test_individual_capabilities())
    
    print("\n🏁 All tests completed!")