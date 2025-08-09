"""Voice Programming Demo - Claude Code Integration Demonstration

This module provides a comprehensive demonstration of the voice-driven programming
capabilities enabled by the Claude Code adapter integration.

Features:
- Complete voice-to-code workflow demonstrations
- Real-time code generation examples
- Voice command processing pipeline
- Integration testing scenarios
- Performance benchmarking
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from loguru import logger

from ..core.types import Context, CommandResult
from ..core.event_system import EventSystem, Event

# Import speech modules only when available  
try:
    from ..speech.types import RecognitionResult, IntentType
    from ..speech.voice_interface import VoiceInterface
except ImportError:
    from dataclasses import dataclass
    from enum import Enum
    
    class IntentType(Enum):
        CODING_REQUEST = "coding_request"
    
    @dataclass 
    class RecognitionResult:
        text: str
        confidence: float
        processing_time: float
        language: str = "en-US"
    
    VoiceInterface = None

from .claude_code_adapter import ClaudeCodeAdapter
from .enhanced_intent_parser import EnhancedIntentParser, CodeIntentType


class VoiceProgrammingDemo:
    """
    Comprehensive demonstration of voice-driven programming capabilities.
    
    This class orchestrates the complete voice programming workflow from
    speech recognition through intent parsing to code generation and execution.
    """
    
    def __init__(self):
        """Initialize the voice programming demonstration system"""
        self.event_system = EventSystem()
        self.claude_adapter = ClaudeCodeAdapter()
        self.enhanced_parser = EnhancedIntentParser(self.event_system)
        self.voice_interface: Optional[VoiceInterface] = None
        
        # Demo configuration
        self.demo_workspace = Path("./voice_programming_demo")
        self.demo_workspace.mkdir(exist_ok=True)
        
        # Performance metrics
        self.metrics = {
            'commands_processed': 0,
            'successful_generations': 0,
            'average_processing_time': 0.0,
            'total_lines_generated': 0,
            'error_count': 0
        }
        
        # Demo scenarios
        self.demo_scenarios = self._initialize_demo_scenarios()
        
        logger.info("Voice Programming Demo initialized")
    
    async def initialize(self) -> bool:
        """Initialize all components for the demonstration"""
        try:
            # Initialize event system
            await self.event_system.initialize()
            
            # Initialize Claude Code adapter
            if not await self.claude_adapter.initialize():
                logger.error("Failed to initialize Claude Code adapter")
                return False
            
            # Set demo workspace as working directory
            await self.claude_adapter.execute_command(
                'set_working_directory',
                {'directory': str(self.demo_workspace)}
            )
            
            logger.info("Voice Programming Demo ready for demonstration")
            return True
            
        except Exception as e:
            logger.error(f"Demo initialization failed: {e}")
            return False
    
    def _initialize_demo_scenarios(self) -> List[Dict[str, Any]]:
        """Initialize demonstration scenarios"""
        return [
            {
                'name': 'Basic Function Generation',
                'description': 'Generate a simple Python function from voice command',
                'voice_input': 'Create a function called calculate_area that takes width and height as parameters',
                'expected_intent': CodeIntentType.GENERATE_FUNCTION,
                'target_language': 'python',
                'complexity': 'simple'
            },
            {
                'name': 'React Component Generation',
                'description': 'Generate a React component with props',
                'voice_input': 'Generate a React component called UserCard with name and email properties',
                'expected_intent': CodeIntentType.GENERATE_COMPONENT,
                'target_language': 'javascript',
                'framework': 'react',
                'complexity': 'medium'
            },
            {
                'name': 'Class with Methods',
                'description': 'Generate a Python class with initialization and methods',
                'voice_input': 'Create a class called DatabaseConnection with connect and disconnect methods',
                'expected_intent': CodeIntentType.GENERATE_CLASS,
                'target_language': 'python',
                'complexity': 'medium'
            },
            {
                'name': 'Code Refactoring',
                'description': 'Refactor existing code to improve readability',
                'voice_input': 'Refactor this function to make it more readable and add error handling',
                'expected_intent': CodeIntentType.REFACTOR_CODE,
                'complexity': 'complex',
                'requires_context': True
            },
            {
                'name': 'Test Generation',
                'description': 'Generate unit tests for existing code',
                'voice_input': 'Generate unit tests for the calculate_area function',
                'expected_intent': CodeIntentType.GENERATE_TEST,
                'target_language': 'python',
                'complexity': 'medium'
            },
            {
                'name': 'Git Operations',
                'description': 'Perform version control operations',
                'voice_input': 'Commit all changes with message "Add new functions"',
                'expected_intent': CodeIntentType.GIT_COMMIT,
                'complexity': 'simple'
            },
            {
                'name': 'Project Analysis',
                'description': 'Analyze project structure and provide insights',
                'voice_input': 'Analyze this project and tell me about its structure',
                'expected_intent': IntentType.QUERY_REQUEST,
                'complexity': 'simple'
            }
        ]
    
    async def run_complete_demo(self) -> Dict[str, Any]:
        """Run the complete voice programming demonstration"""
        logger.info("Starting complete voice programming demonstration")
        
        demo_results = {
            'start_time': time.time(),
            'scenarios_run': 0,
            'scenarios_passed': 0,
            'scenarios_failed': 0,
            'detailed_results': [],
            'performance_metrics': {},
            'generated_files': []
        }
        
        try:
            # Run all demo scenarios
            for i, scenario in enumerate(self.demo_scenarios):
                logger.info(f"Running scenario {i+1}: {scenario['name']}")
                
                scenario_result = await self._run_scenario(scenario)
                demo_results['detailed_results'].append(scenario_result)
                
                demo_results['scenarios_run'] += 1
                if scenario_result['success']:
                    demo_results['scenarios_passed'] += 1
                else:
                    demo_results['scenarios_failed'] += 1
                
                # Small delay between scenarios
                await asyncio.sleep(1)
            
            # Collect final metrics
            demo_results['end_time'] = time.time()
            demo_results['total_duration'] = demo_results['end_time'] - demo_results['start_time']
            demo_results['performance_metrics'] = self.metrics
            demo_results['generated_files'] = list(self.demo_workspace.rglob("*"))
            
            # Generate summary report
            await self._generate_demo_report(demo_results)
            
            logger.info(f"Demo completed: {demo_results['scenarios_passed']}/{demo_results['scenarios_run']} scenarios passed")
            
            return demo_results
            
        except Exception as e:
            logger.error(f"Demo execution failed: {e}")
            demo_results['error'] = str(e)
            return demo_results
    
    async def _run_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single demonstration scenario"""
        start_time = time.time()
        
        result = {
            'scenario_name': scenario['name'],
            'description': scenario['description'],
            'voice_input': scenario['voice_input'],
            'success': False,
            'processing_time': 0.0,
            'intent_detected': None,
            'code_generated': None,
            'files_created': [],
            'error': None
        }
        
        try:
            # Step 1: Simulate speech recognition
            recognition_result = RecognitionResult(
                text=scenario['voice_input'],
                confidence=0.95,
                processing_time=0.1,
                language='en-US'
            )
            
            # Step 2: Parse intent
            enhanced_intent = await self.enhanced_parser.parse_code_intent(recognition_result)
            
            if not enhanced_intent:
                result['error'] = "Intent parsing failed"
                return result
            
            result['intent_detected'] = enhanced_intent.code_intent_type.value if enhanced_intent.code_intent_type else enhanced_intent.intent_type.value
            
            # Step 3: Create context if needed
            context = None
            if scenario.get('requires_context'):
                context = Context(
                    user_id="demo_user",
                    session_id="demo_session",
                    current_file=str(self.demo_workspace / "example.py"),
                    current_app="vscode"
                )
            
            # Step 4: Process through Claude Code adapter
            command_result = await self.claude_adapter.process_voice_intent(enhanced_intent, context)
            
            if command_result.success:
                result['success'] = True
                result['code_generated'] = command_result.data.get('generated_code', 'No code generated')
                
                # Check for created files
                if 'file_written' in command_result.data:
                    result['files_created'].append(command_result.data.get('file_path', 'unknown'))
                
                # Update metrics
                self.metrics['successful_generations'] += 1
                if enhanced_intent.estimated_lines:
                    self.metrics['total_lines_generated'] += enhanced_intent.estimated_lines
            else:
                result['error'] = command_result.error
                self.metrics['error_count'] += 1
            
            # Update processing metrics
            result['processing_time'] = time.time() - start_time
            self.metrics['commands_processed'] += 1
            
            # Update average processing time
            old_avg = self.metrics['average_processing_time']
            count = self.metrics['commands_processed']
            self.metrics['average_processing_time'] = (old_avg * (count - 1) + result['processing_time']) / count
            
            return result
            
        except Exception as e:
            result['error'] = str(e)
            result['processing_time'] = time.time() - start_time
            self.metrics['error_count'] += 1
            logger.error(f"Scenario '{scenario['name']}' failed: {e}")
            return result
    
    async def _generate_demo_report(self, demo_results: Dict[str, Any]) -> None:
        """Generate comprehensive demonstration report"""
        report = {
            'demo_summary': {
                'total_scenarios': demo_results['scenarios_run'],
                'passed_scenarios': demo_results['scenarios_passed'],
                'failed_scenarios': demo_results['scenarios_failed'],
                'success_rate': (demo_results['scenarios_passed'] / demo_results['scenarios_run']) * 100 if demo_results['scenarios_run'] > 0 else 0,
                'total_duration_seconds': demo_results['total_duration'],
                'average_processing_time': self.metrics['average_processing_time']
            },
            'performance_metrics': self.metrics,
            'scenario_details': demo_results['detailed_results'],
            'generated_files': [str(f) for f in demo_results['generated_files']],
            'recommendations': self._generate_recommendations(demo_results)
        }
        
        # Save report to file
        report_file = self.demo_workspace / "voice_programming_demo_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate markdown summary
        await self._generate_markdown_summary(report)
        
        logger.info(f"Demo report saved to: {report_file}")
    
    def _generate_recommendations(self, demo_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on demo results"""
        recommendations = []
        
        success_rate = (demo_results['scenarios_passed'] / demo_results['scenarios_run']) * 100 if demo_results['scenarios_run'] > 0 else 0
        
        if success_rate < 80:
            recommendations.append("Consider improving intent recognition accuracy")
            recommendations.append("Review error patterns to enhance error handling")
        
        if self.metrics['average_processing_time'] > 5.0:
            recommendations.append("Optimize processing pipeline for better performance")
            recommendations.append("Consider caching frequently used code templates")
        
        if self.metrics['error_count'] > 0:
            recommendations.append("Implement more robust error recovery mechanisms")
            recommendations.append("Add validation for voice command parameters")
        
        if success_rate >= 90:
            recommendations.append("Excellent performance! Consider adding more complex scenarios")
            recommendations.append("Ready for production deployment with proper monitoring")
        
        return recommendations
    
    async def _generate_markdown_summary(self, report: Dict[str, Any]) -> None:
        """Generate markdown summary report"""
        summary = f"""# Voice Programming Demo Report

## Overview
- **Total Scenarios**: {report['demo_summary']['total_scenarios']}
- **Success Rate**: {report['demo_summary']['success_rate']:.1f}%
- **Total Duration**: {report['demo_summary']['total_duration_seconds']:.2f} seconds
- **Average Processing Time**: {report['demo_summary']['average_processing_time']:.2f} seconds

## Performance Metrics
- Commands Processed: {report['performance_metrics']['commands_processed']}
- Successful Generations: {report['performance_metrics']['successful_generations']}
- Total Lines Generated: {report['performance_metrics']['total_lines_generated']}
- Error Count: {report['performance_metrics']['error_count']}

## Scenario Results
"""
        
        for scenario in report['scenario_details']:
            status = "✅ PASSED" if scenario['success'] else "❌ FAILED"
            summary += f"""
### {scenario['scenario_name']} {status}
- **Description**: {scenario['description']}
- **Voice Input**: "{scenario['voice_input']}"
- **Intent Detected**: {scenario['intent_detected']}
- **Processing Time**: {scenario['processing_time']:.3f}s
"""
            if scenario['error']:
                summary += f"- **Error**: {scenario['error']}\n"
            
            if scenario['files_created']:
                summary += f"- **Files Created**: {', '.join(scenario['files_created'])}\n"
        
        summary += f"""
## Recommendations
"""
        for rec in report['recommendations']:
            summary += f"- {rec}\n"
        
        # Save markdown report
        report_file = self.demo_workspace / "DEMO_REPORT.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(summary)
    
    async def run_interactive_demo(self) -> None:
        """Run interactive demonstration mode"""
        logger.info("Starting interactive voice programming demo")
        
        print("\n🎤 Voice Programming Demo - Interactive Mode")
        print("=" * 50)
        print("Available commands:")
        print("1. Type voice commands to simulate speech input")
        print("2. Type 'help' to see example commands")
        print("3. Type 'metrics' to see performance statistics")
        print("4. Type 'quit' to exit")
        print()
        
        while True:
            try:
                user_input = input("Voice Command: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'metrics':
                    self._show_metrics()
                    continue
                elif not user_input:
                    continue
                
                # Process the voice command
                start_time = time.time()
                
                recognition_result = RecognitionResult(
                    text=user_input,
                    confidence=0.95,
                    processing_time=0.1,
                    language='en-US'
                )
                
                enhanced_intent = await self.enhanced_parser.parse_code_intent(recognition_result)
                
                if enhanced_intent:
                    print(f"✅ Intent detected: {enhanced_intent.code_intent_type.value if enhanced_intent.code_intent_type else enhanced_intent.intent_type.value}")
                    print(f"📝 Target language: {enhanced_intent.target_language}")
                    if enhanced_intent.framework:
                        print(f"🔧 Framework: {enhanced_intent.framework}")
                    print(f"📊 Complexity: {enhanced_intent.code_complexity}")
                    
                    # Process through adapter
                    result = await self.claude_adapter.process_voice_intent(enhanced_intent)
                    
                    if result.success:
                        print("✅ Command executed successfully!")
                        if 'generated_code' in result.data:
                            print("\n📄 Generated Code:")
                            print("-" * 30)
                            print(result.data['generated_code'])
                            print("-" * 30)
                    else:
                        print(f"❌ Command failed: {result.error}")
                else:
                    print("❌ Could not understand the voice command")
                
                processing_time = time.time() - start_time
                print(f"⏱️ Processing time: {processing_time:.3f}s\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Interactive demo error: {e}")
                print(f"❌ Error: {e}\n")
        
        print("👋 Demo session ended")
    
    def _show_help(self) -> None:
        """Show help with example commands"""
        print("\n📚 Example Voice Commands:")
        print("-" * 30)
        print("• Create a function called hello_world")
        print("• Generate a Python class called Student with name and age properties")
        print("• Build a React component called Button with text prop")
        print("• Refactor this code to make it more efficient")
        print("• Add comments to the current function")
        print("• Run all tests")
        print("• Commit changes with message 'Add new features'")
        print("• Analyze the current project structure")
        print()
    
    def _show_metrics(self) -> None:
        """Show current performance metrics"""
        print("\n📊 Performance Metrics:")
        print("-" * 30)
        for key, value in self.metrics.items():
            if isinstance(value, float):
                print(f"• {key}: {value:.3f}")
            else:
                print(f"• {key}: {value}")
        print()
    
    async def cleanup(self) -> None:
        """Clean up demo resources"""
        try:
            await self.claude_adapter.cleanup()
            await self.event_system.shutdown()
            logger.info("Voice programming demo cleanup completed")
        except Exception as e:
            logger.error(f"Demo cleanup failed: {e}")


async def main():
    """Main demonstration runner"""
    demo = VoiceProgrammingDemo()
    
    try:
        # Initialize demo
        if not await demo.initialize():
            logger.error("Failed to initialize demo")
            return
        
        print("Voice Programming Demo")
        print("Choose demo mode:")
        print("1. Run complete automated demo")
        print("2. Run interactive demo")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            results = await demo.run_complete_demo()
            print(f"\nDemo completed with {results['scenarios_passed']}/{results['scenarios_run']} scenarios passed")
        elif choice == "2":
            await demo.run_interactive_demo()
        else:
            print("Invalid choice")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    asyncio.run(main())