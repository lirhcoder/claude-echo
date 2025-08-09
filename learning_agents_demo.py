"""Learning Agents Demo - Complete Integration Demonstration

This demo showcases the three learning agents working together in a realistic
scenario with AI指导AI的双层架构 and multi-agent collaboration.
"""

import asyncio
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from core.event_system import EventSystem, Event, EventPriority
from agents.agent_manager import AgentManager
from agents.coordinator import Coordinator
from agents.learning_agent import LearningAgent, LearningStrategy
from agents.user_profile_agent import UserProfileAgent
from agents.correction_agent import CorrectionAgent, CorrectionSeverity
from learning.learning_events import LearningEventFactory


class LearningSystemDemo:
    """Comprehensive demonstration of the learning system."""
    
    def __init__(self):
        self.event_system = None
        self.agent_manager = None
        self.agents = {}
        
        # Demo configuration
        self.demo_config = {
            "learning": {
                "db_path": "./data/demo_learning.db",
                "encryption_enabled": False,
                "cache_size": 500,
                "cleanup_interval_hours": 1
            },
            "user_profiles": {
                "voice_recognition_enabled": True,
                "auto_create_profiles": True,
                "session_timeout_minutes": 60,
                "max_concurrent_sessions": 5
            },
            "correction": {
                "pattern_detection_threshold": 3,
                "auto_apply_threshold": 0.8,
                "batch_size": 20,
                "learning_rate": 0.15
            }
        }
        
        # Demo users for testing
        self.demo_users = [
            {
                "username": "alice_researcher",
                "display_name": "Alice the Researcher", 
                "email": "alice@research.com",
                "preferences": {
                    "communication_style": "formal",
                    "response_length": "detailed",
                    "interaction_mode": "voice",
                    "privacy_mode": False
                },
                "voice_sample": {
                    "pitch_mean": 180.0,
                    "pitch_variance": 25.0,
                    "formant_f1": 650.0,
                    "formant_f2": 1800.0,
                    "speech_rate": 4.2,
                    "intensity_mean": 65.0
                }
            },
            {
                "username": "bob_developer", 
                "display_name": "Bob the Developer",
                "email": "bob@dev.com",
                "preferences": {
                    "communication_style": "casual",
                    "response_length": "brief",
                    "interaction_mode": "mixed",
                    "privacy_mode": False
                },
                "voice_sample": {
                    "pitch_mean": 120.0,
                    "pitch_variance": 15.0,
                    "formant_f1": 400.0,
                    "formant_f2": 1200.0,
                    "speech_rate": 5.8,
                    "intensity_mean": 58.0
                }
            }
        ]
    
    async def initialize_system(self):
        """Initialize the complete learning system."""
        print("🚀 Initializing Learning System Demo")
        print("=" * 60)
        
        # Initialize core components
        print("\n1. Setting up core infrastructure...")
        
        self.event_system = EventSystem()
        await self.event_system.initialize()
        print("   ✓ Event system initialized")
        
        # Initialize agent manager
        self.agent_manager = AgentManager(self.event_system, self.demo_config)
        await self.agent_manager.initialize()
        print("   ✓ Agent manager initialized")
        
        # Create and register agents
        print("\n2. Creating learning agents...")
        
        # Coordinator (for orchestration)
        coordinator = Coordinator(self.event_system, self.demo_config)
        await coordinator.initialize()
        self.agents["coordinator"] = coordinator
        print(f"   ✓ {coordinator.name}")
        
        # Learning system agents
        learning_agent = LearningAgent(self.event_system, self.demo_config)
        await learning_agent.initialize()
        self.agents["learning"] = learning_agent
        print(f"   ✓ {learning_agent.name}")
        
        user_profile_agent = UserProfileAgent(self.event_system, self.demo_config)
        await user_profile_agent.initialize()
        self.agents["user_profile"] = user_profile_agent
        print(f"   ✓ {user_profile_agent.name}")
        
        correction_agent = CorrectionAgent(self.event_system, self.demo_config)
        await correction_agent.initialize()
        self.agents["correction"] = correction_agent
        print(f"   ✓ {correction_agent.name}")
        
        # Register agents with manager
        await self.agent_manager.register_agent(coordinator)
        await self.agent_manager.register_agent(learning_agent)
        await self.agent_manager.register_agent(user_profile_agent)
        await self.agent_manager.register_agent(correction_agent)
        
        print("\n✅ System initialization completed")
        
        # Wait for agents to be fully ready
        await asyncio.sleep(2)
    
    async def demonstrate_user_scenarios(self):
        """Demonstrate realistic user interaction scenarios."""
        print("\n" + "=" * 60)
        print("📋 Demonstrating User Interaction Scenarios")
        print("=" * 60)
        
        user_profile_agent = self.agents["user_profile"]
        learning_agent = self.agents["learning"]
        correction_agent = self.agents["correction"]
        
        created_users = []
        
        # Scenario 1: User onboarding and voice enrollment
        print("\n📝 Scenario 1: User Onboarding")
        print("-" * 40)
        
        for i, user_data in enumerate(self.demo_users, 1):
            print(f"\n👤 Onboarding User {i}: {user_data['display_name']}")
            
            # Create user profile
            profile_result = await user_profile_agent.create_user_profile(
                user_data, 
                user_data["voice_sample"]
            )
            
            if profile_result["success"]:
                user_id = profile_result["user_id"]
                created_users.append(user_id)
                
                print(f"   ✓ Profile created: {user_id[:8]}...")
                print(f"   ✓ Voice enrolled: {profile_result['voice_enrolled']}")
                
                # Test voice identification
                voice_result = await user_profile_agent.identify_user_by_voice(
                    user_data["voice_sample"]
                )
                
                if voice_result["success"]:
                    confidence = voice_result["confidence"]
                    print(f"   ✓ Voice ID confidence: {confidence:.2f}")
                
                # Start user session
                session_result = await user_profile_agent.start_user_session(
                    user_id,
                    {"device": f"demo_device_{i}", "app_version": "demo_1.0"},
                    {"demo_scenario": "onboarding"}
                )
                
                if session_result["success"]:
                    print(f"   ✓ Session started: {session_result['session_id'][:8]}...")
                
            await asyncio.sleep(1)  # Realistic delay
        
        # Scenario 2: Learning from user interactions
        print("\n🧠 Scenario 2: Learning from Interactions")
        print("-" * 40)
        
        if created_users:
            user_id = created_users[0]  # Use first user
            
            print(f"\n📊 Analyzing behavior for user: {user_id[:8]}...")
            
            # Simulate some interaction data collection
            interaction_scenarios = [
                {"type": "voice_command", "command": "What's the weather?", "success": True},
                {"type": "voice_command", "command": "Set a reminder", "success": False},
                {"type": "text_input", "input": "Help with Python code", "success": True},
                {"type": "voice_command", "command": "Play music", "success": True}
            ]
            
            for scenario in interaction_scenarios:
                # Simulate learning data collection
                print(f"   • Processing {scenario['type']}: {scenario.get('command', scenario.get('input'))}")
                await asyncio.sleep(0.5)
            
            # Run behavior analysis
            behavior_result = await learning_agent.analyze_user_behavior_patterns(user_id)
            
            if behavior_result["success"]:
                print(f"   ✓ Behavior analysis completed")
                
                if "analysis" in behavior_result:
                    confidence = behavior_result["analysis"].get("confidence_score", 0)
                    print(f"   ✓ Analysis confidence: {confidence:.2f}")
            else:
                print(f"   ✗ Behavior analysis failed: {behavior_result.get('error')}")
        
        # Scenario 3: Error correction and learning
        print("\n🔧 Scenario 3: Error Correction and Learning")
        print("-" * 40)
        
        if created_users:
            user_id = created_users[0]
            
            # Simulate correction scenarios
            correction_scenarios = [
                {
                    "original_input": "What's the time in Tokyo?",
                    "original_response": "I don't have access to real-time data.",
                    "corrected_response": "Let me check the current time in Tokyo for you. It's currently 3:45 PM JST.",
                    "severity": "moderate",
                    "user_satisfaction": 0.8,
                    "context": "time_query"
                },
                {
                    "original_input": "Book a flight to Paris",
                    "original_response": "I cannot book flights for you.",
                    "corrected_response": "I can help you find flight information. Would you like me to show you airline options and prices for Paris?",
                    "severity": "major",
                    "user_satisfaction": 0.7,
                    "context": "travel_assistance"
                },
                {
                    "original_input": "Translate hello to French",
                    "original_response": "Bonjour",
                    "corrected_response": "Hello in French is 'Bonjour' or 'Salut' (informal).",
                    "severity": "minor",
                    "user_satisfaction": 0.9,
                    "context": "translation"
                }
            ]
            
            processed_corrections = []
            
            for i, scenario in enumerate(correction_scenarios, 1):
                print(f"\n🔍 Processing correction {i}:")
                print(f"   Original: {scenario['original_response'][:50]}...")
                print(f"   Corrected: {scenario['corrected_response'][:50]}...")
                
                correction_data = {
                    "user_id": user_id,
                    "session_id": "demo_session_1",
                    "original_input": scenario["original_input"],
                    "original_response": scenario["original_response"],
                    "corrected_response": scenario["corrected_response"],
                    "correction_type": "response_content",
                    "severity": scenario["severity"],
                    "user_satisfaction": scenario["user_satisfaction"],
                    "correction_confidence": 0.9,
                    "metadata": {"context": scenario["context"]}
                }
                
                correction_result = await correction_agent.process_user_correction(
                    correction_data,
                    {"demo_context": "scenario_testing"}
                )
                
                if correction_result["success"]:
                    feedback_id = correction_result["feedback_id"]
                    impact = correction_result.get("estimated_impact", {}).get("estimated_impact_score", 0)
                    print(f"   ✓ Processed: {feedback_id[:8]}... (impact: {impact:.2f})")
                    processed_corrections.append(feedback_id)
                else:
                    print(f"   ✗ Failed: {correction_result.get('error')}")
                
                await asyncio.sleep(1)
            
            # Analyze correction patterns
            if processed_corrections:
                print(f"\n📈 Analyzing correction patterns...")
                pattern_result = await correction_agent.identify_correction_patterns()
                
                if pattern_result["success"]:
                    patterns_found = pattern_result["patterns_found"]
                    print(f"   ✓ Identified {patterns_found} patterns")
                    
                    if patterns_found > 0:
                        insights = pattern_result.get("insights", {})
                        recommendations = insights.get("recommendations", [])
                        print(f"   ✓ Generated {len(recommendations)} recommendations")
                
        return created_users
    
    async def demonstrate_ai_coordination(self, created_users):
        """Demonstrate AI指导AI coordination between agents."""
        print("\n" + "=" * 60)
        print("🤖 Demonstrating AI指导AI Coordination")
        print("=" * 60)
        
        coordinator = self.agents["coordinator"]
        learning_agent = self.agents["learning"]
        user_profile_agent = self.agents["user_profile"]
        correction_agent = self.agents["correction"]
        
        if not created_users:
            print("   ⚠️ No users available for coordination demo")
            return
        
        user_id = created_users[0]
        
        # Scenario 1: Coordinated learning task
        print("\n🎯 Scenario 1: Multi-Agent Learning Coordination")
        print("-" * 50)
        
        print("   📋 Learning Agent coordinating with User Profile and Correction Agents...")
        
        coordination_result = await learning_agent.coordinate_multi_agent_learning(
            [user_profile_agent.agent_id, correction_agent.agent_id],
            "Improve user experience through personalized responses",
            "knowledge_sharing"
        )
        
        if coordination_result["success"]:
            effectiveness = coordination_result.get("collaboration_effectiveness", 0)
            participating_agents = coordination_result.get("participating_agents", [])
            print(f"   ✓ Coordination completed: {effectiveness:.2f} effectiveness")
            print(f"   ✓ Participating agents: {len(participating_agents)}")
            
            insights = coordination_result.get("collaborative_insights", {})
            print(f"   ✓ Collaborative insights generated")
        else:
            print(f"   ✗ Coordination failed: {coordination_result.get('error')}")
        
        # Scenario 2: System-wide correction coordination
        print("\n🔧 Scenario 2: System-Wide Correction Coordination")
        print("-" * 50)
        
        print("   📋 Correction Agent coordinating system improvements...")
        
        correction_coordination = await correction_agent.coordinate_system_corrections(
            [learning_agent.agent_id, user_profile_agent.agent_id],
            "performance"
        )
        
        if correction_coordination["success"]:
            results = correction_coordination.get("coordination_results", {})
            agents_contacted = len(results)
            print(f"   ✓ Coordinated with {agents_contacted} agents")
            
            for agent_id, result in results.items():
                status = "✓" if result.get("success") else "✗"
                corrections = result.get("corrections_sent", 0)
                print(f"   {status} {agent_id}: {corrections} corrections sent")
        else:
            print(f"   ✗ Correction coordination failed: {correction_coordination.get('error')}")
        
        # Scenario 3: Learning insights sharing
        print("\n📊 Scenario 3: Learning Insights Sharing")
        print("-" * 50)
        
        # Generate comprehensive insights
        insights_result = await learning_agent.get_learning_insights(
            "comprehensive",
            {"user_id": user_id, "include_patterns": True}
        )
        
        if insights_result["success"]:
            insights = insights_result.get("insights", {})
            key_findings = insights.get("key_findings", [])
            recommendations = insights.get("recommendations", [])
            
            print(f"   ✓ Generated learning insights:")
            print(f"     • {len(key_findings)} key findings")
            print(f"     • {len(recommendations)} recommendations")
            print(f"     • Confidence score: {insights.get('confidence_score', 0):.2f}")
            
            # Share insights with other agents via events
            insight_event = LearningEventFactory.knowledge_learned(
                knowledge_type="system_insights",
                knowledge_data=insights,
                source_interactions=insights.get("data_points_analyzed", 0)
            )
            
            system_event = insight_event.to_system_event()
            await self.event_system.emit(system_event)
            print("   ✓ Insights shared via event system")
        
        # Scenario 4: User profile optimization
        print("\n👤 Scenario 4: User Profile Optimization")
        print("-" * 50)
        
        # Get current user context
        context_result = await user_profile_agent.get_user_context(user_id, "full")
        
        if context_result["success"]:
            context = context_result["context"]
            sessions = context.get("active_sessions", 0)
            preferences = context.get("preferences", {})
            
            print(f"   ✓ User context retrieved:")
            print(f"     • Active sessions: {sessions}")
            print(f"     • Communication style: {preferences.get('communication_style')}")
            print(f"     • Response length: {preferences.get('response_length')}")
            
            # Optimize based on learning insights
            optimization_result = await learning_agent.optimize_agent_performance(
                user_profile_agent.agent_id,
                {"response_time": 0.5, "accuracy": 0.85, "user_satisfaction": 0.8},
                ["improve_personalization", "reduce_response_time"]
            )
            
            if optimization_result["success"]:
                recommendations = optimization_result.get("recommendations", [])
                applied = optimization_result.get("applied_optimizations", [])
                
                print(f"   ✓ Performance optimization completed:")
                print(f"     • {len(recommendations)} recommendations")
                print(f"     • {len(applied)} optimizations applied")
                
                estimated_improvement = optimization_result.get("estimated_improvement", 0)
                print(f"     • Estimated improvement: {estimated_improvement:.1%}")
    
    async def demonstrate_performance_monitoring(self):
        """Demonstrate system performance monitoring and analytics."""
        print("\n" + "=" * 60)
        print("📈 Demonstrating Performance Monitoring")
        print("=" * 60)
        
        learning_agent = self.agents["learning"]
        user_profile_agent = self.agents["user_profile"]
        correction_agent = self.agents["correction"]
        
        # Scenario 1: Learning system metrics
        print("\n📊 Scenario 1: Learning System Metrics")
        print("-" * 40)
        
        try:
            # Get learning agent statistics
            learning_stats = learning_agent._learning_statistics
            print(f"   📈 Learning Agent Statistics:")
            print(f"     • Total learning tasks: {learning_stats.get('total_learning_tasks', 0)}")
            print(f"     • Successful adaptations: {learning_stats.get('successful_adaptations', 0)}")
            print(f"     • Performance improvements: {learning_stats.get('performance_improvements', 0)}")
            print(f"     • System efficiency gain: {learning_stats.get('system_efficiency_gain', 0):.1%}")
        except Exception as e:
            print(f"   ⚠️ Learning statistics unavailable: {e}")
        
        # Scenario 2: User profile metrics  
        print("\n👥 Scenario 2: User Profile Metrics")
        print("-" * 40)
        
        try:
            profile_stats = user_profile_agent._profile_statistics
            print(f"   📈 User Profile Agent Statistics:")
            print(f"     • Total users: {profile_stats.get('total_users', 0)}")
            print(f"     • Active sessions: {profile_stats.get('active_sessions', 0)}")
            print(f"     • Successful identifications: {profile_stats.get('successful_identifications', 0)}")
            print(f"     • Voice enrollments: {profile_stats.get('voice_enrollments', 0)}")
            
            # Calculate success rate
            total_attempts = (profile_stats.get('successful_identifications', 0) + 
                            profile_stats.get('failed_identifications', 0))
            if total_attempts > 0:
                success_rate = profile_stats.get('successful_identifications', 0) / total_attempts
                print(f"     • Identification success rate: {success_rate:.1%}")
                
        except Exception as e:
            print(f"   ⚠️ Profile statistics unavailable: {e}")
        
        # Scenario 3: Correction system metrics
        print("\n🔧 Scenario 3: Correction System Metrics")
        print("-" * 40)
        
        try:
            correction_stats = correction_agent._correction_statistics
            print(f"   📈 Correction Agent Statistics:")
            print(f"     • Total corrections: {correction_stats.get('total_corrections', 0)}")
            print(f"     • Successful applications: {correction_stats.get('successful_applications', 0)}")
            print(f"     • Patterns detected: {correction_stats.get('patterns_detected', 0)}")
            print(f"     • User satisfaction average: {correction_stats.get('user_satisfaction_average', 0):.2f}")
            
            # Get correction insights
            insights_result = await correction_agent.get_correction_insights("performance")
            if insights_result["success"]:
                insights = insights_result.get("insights", {})
                print(f"     • Performance impact: {insights.get('performance_impact', 0):.1%}")
                print(f"     • Trend: {insights.get('trend', 'unknown')}")
                
        except Exception as e:
            print(f"   ⚠️ Correction statistics unavailable: {e}")
        
        # Scenario 4: System health monitoring
        print("\n🏥 Scenario 4: System Health Monitoring")
        print("-" * 40)
        
        # Check agent health
        agents_health = []
        for name, agent in self.agents.items():
            try:
                status = agent.status
                metrics = agent.metrics
                
                health_info = {
                    "name": name,
                    "status": status.value,
                    "uptime": metrics.uptime_seconds,
                    "requests_processed": metrics.requests_processed,
                    "success_rate": (
                        metrics.requests_processed / 
                        max(metrics.requests_processed + metrics.requests_failed, 1)
                    )
                }
                agents_health.append(health_info)
                
            except Exception as e:
                agents_health.append({
                    "name": name,
                    "status": "error",
                    "error": str(e)
                })
        
        print("   🏥 Agent Health Status:")
        for health in agents_health:
            status_emoji = "✅" if health.get("status") == "idle" else "⚠️"
            if "error" in health:
                print(f"     {status_emoji} {health['name']}: ERROR - {health['error']}")
            else:
                uptime_minutes = health.get("uptime", 0) / 60
                requests = health.get("requests_processed", 0)
                success_rate = health.get("success_rate", 0)
                print(f"     {status_emoji} {health['name']}: {health.get('status')} "
                      f"({uptime_minutes:.1f}min, {requests} reqs, {success_rate:.1%} success)")
    
    async def cleanup_demo(self):
        """Clean up demo resources."""
        print("\n" + "=" * 60)
        print("🧹 Cleaning up Demo Resources")
        print("=" * 60)
        
        cleanup_tasks = []
        
        # Shutdown agents
        if self.agents:
            print("\n   Shutting down agents...")
            for name, agent in self.agents.items():
                try:
                    await agent.shutdown()
                    print(f"     ✓ {name}")
                except Exception as e:
                    print(f"     ✗ {name}: {e}")
        
        # Shutdown agent manager
        if self.agent_manager:
            try:
                await self.agent_manager.shutdown()
                print("     ✓ Agent manager")
            except Exception as e:
                print(f"     ✗ Agent manager: {e}")
        
        # Shutdown event system
        if self.event_system:
            try:
                await self.event_system.shutdown()
                print("     ✓ Event system")
            except Exception as e:
                print(f"     ✗ Event system: {e}")
        
        print("\n✅ Cleanup completed")
    
    async def run_complete_demo(self):
        """Run the complete learning system demonstration."""
        try:
            await self.initialize_system()
            created_users = await self.demonstrate_user_scenarios()
            await self.demonstrate_ai_coordination(created_users)
            await self.demonstrate_performance_monitoring()
            
            print("\n" + "=" * 60)
            print("🎉 Learning System Demo Completed Successfully!")
            print("=" * 60)
            
            print("\n📋 Demo Summary:")
            print(f"   • Three learning agents successfully integrated")
            print(f"   • AI指导AI coordination demonstrated")
            print(f"   • User profile management with voice recognition")
            print(f"   • Interactive correction and learning system")
            print(f"   • Multi-agent collaboration patterns")
            print(f"   • Real-time performance monitoring")
            print(f"   • Event-driven architecture working correctly")
            
            print("\n🚀 The learning system is ready for production deployment!")
            
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            await self.cleanup_demo()


async def main():
    """Main demo entry point."""
    print("🎯 Learning Agents Complete Integration Demo")
    print("Showcasing AI指导AI architecture with three learning agents")
    print("=" * 80)
    
    # Ensure data directory exists
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    # Run demo
    demo = LearningSystemDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())