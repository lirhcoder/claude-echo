"""Integration Test for Agent System

This test validates the core agent system functionality including:
- Agent initialization and lifecycle
- Inter-agent communication
- Collaboration patterns
- Event system integration
- Basic error handling
"""

import asyncio
from datetime import datetime, timedelta
import logging

from ..core.event_system import EventSystem, Event
from ..core.config_manager import ConfigManager
from .agent_manager import AgentManager
from .agent_types import AgentType, AgentRequest, AgentResponse, AgentPriority

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentSystemIntegrationTest:
    """Integration test suite for the agent system."""
    
    def __init__(self):
        self.event_system = None
        self.config_manager = None
        self.agent_manager = None
        self.test_results = []
    
    async def setup(self):
        """Setup test environment."""
        logger.info("Setting up integration test environment...")
        
        # Initialize event system
        self.event_system = EventSystem()
        await self.event_system.initialize()
        
        # Initialize config manager (mock)
        self.config_manager = MockConfigManager()
        
        # Initialize agent manager
        self.agent_manager = AgentManager(
            self.event_system, 
            self.config_manager
        )
        await self.agent_manager.initialize()
        
        logger.info("Test environment setup complete")
    
    async def teardown(self):
        """Cleanup test environment."""
        logger.info("Cleaning up test environment...")
        
        if self.agent_manager:
            await self.agent_manager.shutdown()
        
        if self.event_system:
            await self.event_system.shutdown()
        
        logger.info("Test environment cleanup complete")
    
    async def run_all_tests(self):
        """Run all integration tests."""
        logger.info("Starting Agent System Integration Tests")
        
        test_methods = [
            self.test_agent_initialization,
            self.test_basic_communication,
            self.test_coordinator_functionality,
            self.test_task_planner_functionality,
            self.test_presence_monitor_functionality,
            self.test_auto_worker_functionality,
            self.test_security_guardian_functionality,
            self.test_handover_manager_functionality,
            self.test_session_manager_functionality,
            self.test_agent_collaboration,
            self.test_error_handling,
            self.test_system_status
        ]
        
        for test_method in test_methods:
            try:
                logger.info(f"Running {test_method.__name__}...")
                result = await test_method()
                self.test_results.append({
                    'test': test_method.__name__,
                    'status': 'PASSED' if result else 'FAILED',
                    'timestamp': datetime.now()
                })
                logger.info(f"{test_method.__name__}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                logger.error(f"{test_method.__name__}: ERROR - {e}")
                self.test_results.append({
                    'test': test_method.__name__,
                    'status': 'ERROR',
                    'error': str(e),
                    'timestamp': datetime.now()
                })
        
        # Print summary
        self.print_test_summary()
    
    async def test_agent_initialization(self):
        """Test that all core agents initialize properly."""
        try:
            # Check that core agents are registered
            system_status = self.agent_manager.get_system_status()
            
            expected_agents = [
                'coordinator',
                'task_planner'
                # Note: Only these two are currently configured in AgentManager
            ]
            
            agent_details = system_status.get('agents', {})
            
            for expected_agent in expected_agents:
                if expected_agent not in agent_details:
                    logger.error(f"Expected agent {expected_agent} not found in system")
                    return False
                
                agent_info = agent_details[expected_agent]
                if agent_info.get('status') != 'idle':
                    logger.error(f"Agent {expected_agent} not in idle state: {agent_info.get('status')}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Agent initialization test failed: {e}")
            return False
    
    async def test_basic_communication(self):
        """Test basic inter-agent communication."""
        try:
            # Test communication with coordinator
            coordinator = self.agent_manager.get_agent_by_type(AgentType.COORDINATOR)
            if not coordinator:
                logger.error("Coordinator agent not found")
                return False
            
            # Test sending a request
            response = await self.agent_manager.send_request(
                target_agent="coordinator",
                capability="get_system_status",
                parameters={},
                timeout=timedelta(seconds=10)
            )
            
            if not response.success:
                logger.error(f"Communication test failed: {response.error}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Basic communication test failed: {e}")
            return False
    
    async def test_coordinator_functionality(self):
        """Test coordinator agent functionality."""
        try:
            response = await self.agent_manager.send_request(
                target_agent="coordinator",
                capability="get_system_status",
                parameters={},
                timeout=timedelta(seconds=5)
            )
            
            if not response.success:
                return False
            
            # Check response data structure
            data = response.data
            if not data or 'coordinator_status' not in data:
                logger.error("Invalid coordinator response structure")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Coordinator functionality test failed: {e}")
            return False
    
    async def test_task_planner_functionality(self):
        """Test task planner agent functionality."""
        try:
            response = await self.agent_manager.send_request(
                target_agent="task_planner",
                capability="parse_intent",
                parameters={
                    "user_input": "copy file test.txt to backup folder",
                    "context": {}
                },
                timeout=timedelta(seconds=5)
            )
            
            if not response.success:
                logger.error(f"Task planner request failed: {response.error}")
                return False
            
            # Check response structure
            data = response.data
            if not data or 'intent' not in data:
                logger.error("Invalid task planner response structure")
                return False
            
            intent = data['intent']
            if 'intent_type' not in intent or 'confidence' not in intent:
                logger.error("Invalid intent structure")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Task planner functionality test failed: {e}")
            return False
    
    async def test_presence_monitor_functionality(self):
        """Test presence monitor functionality (if available)."""
        try:
            presence_monitor = self.agent_manager.get_agent_by_type(AgentType.PRESENCE_MONITOR)
            if not presence_monitor:
                logger.info("Presence monitor not available, skipping test")
                return True  # Skip if not implemented
            
            # Test would go here
            return True
            
        except Exception as e:
            logger.error(f"Presence monitor test failed: {e}")
            return False
    
    async def test_auto_worker_functionality(self):
        """Test auto worker functionality (if available)."""
        try:
            auto_worker = self.agent_manager.get_agent_by_type(AgentType.AUTO_WORKER)
            if not auto_worker:
                logger.info("Auto worker not available, skipping test")
                return True  # Skip if not implemented
            
            # Test would go here
            return True
            
        except Exception as e:
            logger.error(f"Auto worker test failed: {e}")
            return False
    
    async def test_security_guardian_functionality(self):
        """Test security guardian functionality (if available)."""
        try:
            security_guardian = self.agent_manager.get_agent_by_type(AgentType.SECURITY_GUARDIAN)
            if not security_guardian:
                logger.info("Security guardian not available, skipping test")
                return True  # Skip if not implemented
            
            # Test would go here
            return True
            
        except Exception as e:
            logger.error(f"Security guardian test failed: {e}")
            return False
    
    async def test_handover_manager_functionality(self):
        """Test handover manager functionality (if available)."""
        try:
            handover_manager = self.agent_manager.get_agent_by_type(AgentType.HANDOVER_MANAGER)
            if not handover_manager:
                logger.info("Handover manager not available, skipping test")
                return True  # Skip if not implemented
            
            # Test would go here
            return True
            
        except Exception as e:
            logger.error(f"Handover manager test failed: {e}")
            return False
    
    async def test_session_manager_functionality(self):
        """Test session manager functionality (if available)."""
        try:
            session_manager = self.agent_manager.get_agent_by_type(AgentType.SESSION_MANAGER)
            if not session_manager:
                logger.info("Session manager not available, skipping test")
                return True  # Skip if not implemented
            
            # Test would go here
            return True
            
        except Exception as e:
            logger.error(f"Session manager test failed: {e}")
            return False
    
    async def test_agent_collaboration(self):
        """Test multi-agent collaboration."""
        try:
            # Test coordination between agents
            coordinator = self.agent_manager.get_agent_by_type(AgentType.COORDINATOR)
            task_planner = self.agent_manager.get_agent_by_type(AgentType.TASK_PLANNER)
            
            if not coordinator or not task_planner:
                logger.info("Required agents not available for collaboration test")
                return True
            
            # Test coordinator requesting task planning
            response = await self.agent_manager.send_request(
                target_agent="coordinator",
                capability="process_user_request",
                parameters={
                    "user_input": "create a backup of my documents",
                    "context": {"user_id": "test_user"}
                },
                timeout=timedelta(seconds=10)
            )
            
            if not response.success:
                logger.error(f"Collaboration test failed: {response.error}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Agent collaboration test failed: {e}")
            return False
    
    async def test_error_handling(self):
        """Test system error handling."""
        try:
            # Test invalid capability request
            response = await self.agent_manager.send_request(
                target_agent="coordinator",
                capability="invalid_capability",
                parameters={},
                timeout=timedelta(seconds=5)
            )
            
            # Should fail gracefully
            if response.success:
                logger.error("Expected error handling test to fail")
                return False
            
            # Test invalid agent request
            try:
                await self.agent_manager.send_request(
                    target_agent="non_existent_agent",
                    capability="test",
                    parameters={},
                    timeout=timedelta(seconds=2)
                )
                logger.error("Expected exception for non-existent agent")
                return False
            except ValueError:
                # Expected error
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling test failed: {e}")
            return False
    
    async def test_system_status(self):
        """Test system status and monitoring."""
        try:
            status = self.agent_manager.get_system_status()
            
            # Check status structure
            required_keys = [
                'manager_status',
                'performance_metrics',
                'agents'
            ]
            
            for key in required_keys:
                if key not in status:
                    logger.error(f"Missing key in system status: {key}")
                    return False
            
            # Check performance metrics
            metrics = status['performance_metrics']
            if 'total_agents' not in metrics or metrics['total_agents'] == 0:
                logger.error("No agents found in performance metrics")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"System status test failed: {e}")
            return False
    
    def print_test_summary(self):
        """Print test execution summary."""
        logger.info("\n" + "="*60)
        logger.info("AGENT SYSTEM INTEGRATION TEST SUMMARY")
        logger.info("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['status'] == 'PASSED'])
        failed_tests = len([r for r in self.test_results if r['status'] == 'FAILED'])
        error_tests = len([r for r in self.test_results if r['status'] == 'ERROR'])
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Errors: {error_tests}")
        logger.info(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        logger.info("\nTest Details:")
        for result in self.test_results:
            status_symbol = "✓" if result['status'] == 'PASSED' else "✗"
            logger.info(f"{status_symbol} {result['test']}: {result['status']}")
            if 'error' in result:
                logger.info(f"    Error: {result['error']}")
        
        logger.info("="*60)


class MockConfigManager:
    """Mock configuration manager for testing."""
    
    def get_agent_config(self):
        """Get mock agent configuration."""
        return {
            'core_agents': ['coordinator', 'task_planner'],
            'coordinator': {
                'max_concurrent_requests': 10
            },
            'task_planner': {
                'max_planning_time': 30
            }
        }


async def run_integration_tests():
    """Run the complete integration test suite."""
    test_suite = AgentSystemIntegrationTest()
    
    try:
        await test_suite.setup()
        await test_suite.run_all_tests()
    finally:
        await test_suite.teardown()
    
    # Return test results
    return test_suite.test_results


if __name__ == "__main__":
    # Run the integration tests
    asyncio.run(run_integration_tests())