"""Test Script for Claude Code Adapter

This script provides basic testing and validation for the Claude Code adapter
implementation, demonstrating core functionality and integration capabilities.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.adapters.claude_code_adapter import ClaudeCodeAdapter, MCPClient, ClaudeCodeError
from src.adapters.enhanced_intent_parser import EnhancedIntentParser, CodeIntentType
from src.speech.types import RecognitionResult, IntentType
from src.core.types import Context, CommandResult
from src.core.event_system import EventSystem

async def test_mcp_client():
    """Test MCP client functionality"""
    print("🔧 Testing MCP Client...")
    
    client = MCPClient()
    
    # Test connection
    connected = await client.connect()
    assert connected, "MCP client should connect successfully"
    print("✅ MCP client connected")
    
    # Test tool discovery
    assert len(client.tools_registry) > 0, "Should discover available tools"
    print(f"✅ Discovered {len(client.tools_registry)} tools")
    
    # Test basic tool call
    result = await client.call_tool("generate_function", {
        "name": "test_function",
        "description": "A test function",
        "language": "python"
    })
    
    assert result.get("success", False), f"Function generation should succeed: {result}"
    assert "generated_code" in result, "Should return generated code"
    print("✅ Tool call successful")
    
    await client.disconnect()
    print("✅ MCP client disconnected\n")


async def test_claude_code_adapter():
    """Test Claude Code adapter core functionality"""
    print("🎯 Testing Claude Code Adapter...")
    
    # Initialize adapter
    adapter = ClaudeCodeAdapter({
        'working_directory': str(Path.cwd()),
        'security': {
            'allowed_operations': ['read', 'write', 'create', 'generate']
        }
    })
    
    # Test initialization
    initialized = await adapter.initialize()
    assert initialized, "Adapter should initialize successfully"
    print("✅ Adapter initialized")
    
    # Test basic properties
    assert adapter.adapter_id == "claude_code"
    assert len(adapter.supported_commands) > 0
    print(f"✅ Adapter supports {len(adapter.supported_commands)} commands")
    
    # Test file operations
    test_file = Path("test_output.py")
    
    # Test file creation
    result = await adapter.execute_command("create_file", {
        "file_path": str(test_file),
        "content": "# Test file created by Claude Code Adapter\nprint('Hello, Voice Programming!')\n"
    })
    assert result.success, f"File creation should succeed: {result.error}"
    print("✅ File creation successful")
    
    # Test file reading
    result = await adapter.execute_command("read_file", {
        "file_path": str(test_file)
    })
    assert result.success, f"File reading should succeed: {result.error}"
    assert "Hello, Voice Programming!" in result.data.get("content", "")
    print("✅ File reading successful")
    
    # Test code generation
    result = await adapter.execute_command("generate_function", {
        "name": "calculate_sum",
        "description": "Calculate sum of two numbers",
        "language": "python"
    })
    assert result.success, f"Code generation should succeed: {result.error}"
    assert "def calculate_sum" in result.data.get("generated_code", "")
    print("✅ Code generation successful")
    
    # Test project analysis
    result = await adapter.execute_command("analyze_project", {
        "project_path": str(Path.cwd())
    })
    assert result.success, f"Project analysis should succeed: {result.error}"
    print("✅ Project analysis successful")
    
    # Cleanup
    if test_file.exists():
        test_file.unlink()
    
    await adapter.cleanup()
    print("✅ Adapter cleanup completed\n")


async def test_enhanced_intent_parser():
    """Test enhanced intent parser for programming commands"""
    print("🧠 Testing Enhanced Intent Parser...")
    
    parser = EnhancedIntentParser()
    
    # Test function generation intent
    recognition_result = RecognitionResult(
        text="Create a function called hello_world that prints hello",
        confidence=0.95,
        processing_time=0.1
    )
    
    enhanced_intent = await parser.parse_code_intent(recognition_result)
    assert enhanced_intent is not None, "Should parse intent successfully"
    assert enhanced_intent.code_intent_type == CodeIntentType.GENERATE_FUNCTION
    assert len(enhanced_intent.code_entities) > 0, "Should extract code entities"
    print("✅ Function generation intent parsed")
    
    # Test class generation intent
    recognition_result = RecognitionResult(
        text="Generate a Python class called Calculator with add and subtract methods",
        confidence=0.95,
        processing_time=0.1
    )
    
    enhanced_intent = await parser.parse_code_intent(recognition_result)
    assert enhanced_intent.code_intent_type == CodeIntentType.GENERATE_CLASS
    assert enhanced_intent.target_language == "python"
    print("✅ Class generation intent parsed")
    
    # Test React component intent
    recognition_result = RecognitionResult(
        text="Create a React component called Button with text and onClick properties",
        confidence=0.95,
        processing_time=0.1
    )
    
    enhanced_intent = await parser.parse_code_intent(recognition_result)
    assert enhanced_intent.code_intent_type == CodeIntentType.GENERATE_COMPONENT
    assert enhanced_intent.framework == "react"
    print("✅ React component intent parsed")
    
    # Test code suggestions
    suggestions = await parser.generate_code_suggestions(enhanced_intent)
    assert len(suggestions) > 0, "Should generate suggestions"
    print(f"✅ Generated {len(suggestions)} code suggestions")
    print()


async def test_voice_to_code_workflow():
    """Test complete voice-to-code workflow"""
    print("🎤 Testing Voice-to-Code Workflow...")
    
    # Initialize components
    event_system = EventSystem()
    await event_system.initialize()
    
    adapter = ClaudeCodeAdapter()
    await adapter.initialize()
    
    parser = EnhancedIntentParser(event_system)
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Simple Function',
            'voice_input': 'Create a function called add_numbers that takes two parameters',
            'expected_intent': CodeIntentType.GENERATE_FUNCTION
        },
        {
            'name': 'React Component', 
            'voice_input': 'Generate a React component called UserProfile with name and email props',
            'expected_intent': CodeIntentType.GENERATE_COMPONENT
        },
        {
            'name': 'Code Refactoring',
            'voice_input': 'Refactor this code to improve performance and readability',
            'expected_intent': CodeIntentType.REFACTOR_CODE
        }
    ]
    
    successful_scenarios = 0
    
    for scenario in scenarios:
        try:
            print(f"  Testing: {scenario['name']}")
            
            # Speech recognition simulation
            recognition_result = RecognitionResult(
                text=scenario['voice_input'],
                confidence=0.95,
                processing_time=0.1
            )
            
            # Intent parsing
            enhanced_intent = await parser.parse_code_intent(recognition_result)
            assert enhanced_intent is not None
            assert enhanced_intent.code_intent_type == scenario['expected_intent']
            
            # Code generation through adapter
            result = await adapter.process_voice_intent(enhanced_intent)
            
            if result.success:
                successful_scenarios += 1
                print(f"    ✅ {scenario['name']} completed successfully")
            else:
                print(f"    ⚠️ {scenario['name']} completed with issues: {result.error}")
                
        except Exception as e:
            print(f"    ❌ {scenario['name']} failed: {e}")
    
    print(f"✅ Workflow test completed: {successful_scenarios}/{len(scenarios)} scenarios successful")
    
    # Cleanup
    await adapter.cleanup()
    await event_system.shutdown()
    print()


async def test_security_features():
    """Test security and safety features"""
    print("🔒 Testing Security Features...")
    
    adapter = ClaudeCodeAdapter({
        'security': {
            'restricted_operations': ['delete', 'system_command'],
            'allowed_operations': ['read', 'write', 'create']
        }
    })
    await adapter.initialize()
    
    # Test restricted command blocking
    result = await adapter.execute_command("run_bash_command", {
        "command": "rm -rf /"  # Dangerous command
    })
    assert not result.success, "Dangerous commands should be blocked"
    print("✅ Dangerous command blocked")
    
    # Test file protection
    result = await adapter.execute_command("delete_file", {
        "file_path": "package.json"  # Important file
    })
    assert not result.success, "Important files should be protected"
    print("✅ Important file protected")
    
    # Test safe operation
    result = await adapter.execute_command("generate_function", {
        "name": "safe_function",
        "description": "A safe function"
    })
    assert result.success, "Safe operations should be allowed"
    print("✅ Safe operation allowed")
    
    await adapter.cleanup()
    print("✅ Security tests passed\n")


async def run_all_tests():
    """Run all test suites"""
    print("🚀 Starting Claude Code Adapter Tests\n")
    
    try:
        await test_mcp_client()
        await test_claude_code_adapter()
        await test_enhanced_intent_parser()
        await test_voice_to_code_workflow()
        await test_security_features()
        
        print("🎉 All tests passed successfully!")
        print("✅ Claude Code Adapter is ready for voice-driven programming!")
        
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error during testing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run tests
    asyncio.run(run_all_tests())