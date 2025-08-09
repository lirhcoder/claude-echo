"""Simple Test for Claude Code Adapter (No Speech Dependencies)

A basic test that validates the core functionality of the Claude Code adapter
without requiring speech recognition dependencies.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Create mock types to avoid speech dependencies
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional

class IntentType(Enum):
    """Mock intent types"""
    CODING_REQUEST = "coding_request"
    FILE_OPERATION = "file_operation"
    SYSTEM_CONTROL = "system_control"
    QUERY_REQUEST = "query_request"

@dataclass
class ParsedIntent:
    """Mock parsed intent"""
    original_text: str
    processed_text: str
    intent_type: IntentType
    confidence: float
    entities: Dict[str, Any]
    parameters: Dict[str, Any]

@dataclass 
class Context:
    """Mock context"""
    user_id: str
    session_id: str
    current_file: Optional[str] = None
    current_app: Optional[str] = None

@dataclass
class CommandResult:
    """Mock command result"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None

# Import the adapter after mocking dependencies
from src.adapters.claude_code_adapter import ClaudeCodeAdapter, MCPClient


async def test_mcp_client_basic():
    """Test basic MCP client functionality"""
    print("[MCP] Testing MCP Client Basic Features...")
    
    client = MCPClient()
    
    # Test connection
    connected = await client.connect()
    if connected:
        print("[OK] MCP client connected successfully")
    else:
        print("[FAIL] MCP client connection failed")
        return False
    
    # Test tool discovery
    tools_count = len(client.tools_registry)
    if tools_count > 0:
        print(f"[OK] Discovered {tools_count} tools")
    else:
        print("[FAIL] No tools discovered")
        return False
    
    # Test basic tool call
    try:
        result = await client.call_tool("generate_function", {
            "name": "test_function",
            "description": "A test function for validation",
            "language": "python"
        })
        
        if result.get("success", False) and "generated_code" in result:
            print("✅ Tool call executed successfully")
            print(f"📝 Generated code preview: {result['generated_code'][:100]}...")
        else:
            print(f"⚠️ Tool call completed but with issues: {result}")
            
    except Exception as e:
        print(f"❌ Tool call failed: {e}")
        return False
    
    await client.disconnect()
    print("✅ MCP client test completed\n")
    return True


async def test_adapter_initialization():
    """Test adapter initialization and basic properties"""
    print("🎯 Testing Adapter Initialization...")
    
    # Create adapter with basic config
    config = {
        'working_directory': str(Path.cwd()),
        'security': {
            'allowed_operations': ['read', 'write', 'create', 'generate', 'analyze'],
            'restricted_operations': ['delete_system', 'modify_permissions']
        },
        'code_generation': {
            'coding_style': 'standard',
            'include_comments': True,
            'include_type_hints': True
        }
    }
    
    adapter = ClaudeCodeAdapter(config)
    
    # Test basic properties
    assert adapter.adapter_id == "claude_code", f"Expected 'claude_code', got '{adapter.adapter_id}'"
    assert adapter.name == "Claude Code Adapter", f"Unexpected adapter name: {adapter.name}"
    assert len(adapter.supported_commands) > 0, "Adapter should support commands"
    
    print(f"✅ Adapter ID: {adapter.adapter_id}")
    print(f"✅ Adapter Name: {adapter.name}")
    print(f"✅ Supported Commands: {len(adapter.supported_commands)}")
    
    # Test initialization
    try:
        initialized = await adapter.initialize()
        if initialized:
            print("✅ Adapter initialized successfully")
        else:
            print("⚠️ Adapter initialization had issues")
            
        # Test status
        available = await adapter.is_available()
        print(f"✅ Adapter availability: {available}")
        
        # Test current state
        state = await adapter.get_current_state()
        print(f"✅ Adapter state: {state['status']}")
        
        await adapter.cleanup()
        print("✅ Adapter cleanup completed")
        
    except Exception as e:
        print(f"❌ Adapter initialization failed: {e}")
        return False
    
    print("✅ Adapter initialization test completed\n")
    return True


async def test_file_operations():
    """Test file operation commands"""
    print("📁 Testing File Operations...")
    
    adapter = ClaudeCodeAdapter()
    await adapter.initialize()
    
    # Test file creation
    test_file = Path("test_voice_programming.py")
    test_content = '''# Test file generated by Claude Voice Assistant
def hello_voice_programming():
    """Demonstrate voice-driven code generation"""
    print("Hello from voice programming!")
    return "Success"

if __name__ == "__main__":
    result = hello_voice_programming()
    print(f"Result: {result}")
'''
    
    try:
        # Create file
        result = await adapter.execute_command("create_file", {
            "file_path": str(test_file),
            "content": test_content
        })
        
        if result.success:
            print("✅ File creation successful")
        else:
            print(f"❌ File creation failed: {result.error}")
            return False
        
        # Read file back
        result = await adapter.execute_command("read_file", {
            "file_path": str(test_file)
        })
        
        if result.success and "hello_voice_programming" in result.data.get("content", ""):
            print("✅ File reading successful")
        else:
            print(f"❌ File reading failed: {result.error}")
            return False
        
        # List files in directory
        result = await adapter.execute_command("list_files", {
            "directory_path": str(Path.cwd())
        })
        
        if result.success:
            print("✅ Directory listing successful")
        else:
            print(f"⚠️ Directory listing issues: {result.error}")
        
        # Cleanup test file
        if test_file.exists():
            test_file.unlink()
            print("✅ Test file cleaned up")
            
    except Exception as e:
        print(f"❌ File operations test failed: {e}")
        return False
    finally:
        await adapter.cleanup()
    
    print("✅ File operations test completed\n")
    return True


async def test_code_generation():
    """Test code generation commands"""
    print("⚙️ Testing Code Generation...")
    
    adapter = ClaudeCodeAdapter()
    await adapter.initialize()
    
    try:
        # Test function generation
        result = await adapter.execute_command("generate_function", {
            "name": "calculate_area",
            "description": "Calculate the area of a rectangle",
            "language": "python",
            "parameters": ["width", "height"]
        })
        
        if result.success:
            generated_code = result.data.get("generated_code", "")
            if "def calculate_area" in generated_code:
                print("✅ Function generation successful")
                print(f"📝 Generated function preview:\n{generated_code[:200]}...")
            else:
                print("⚠️ Function generated but format unexpected")
        else:
            print(f"❌ Function generation failed: {result.error}")
            return False
        
        # Test class generation
        result = await adapter.execute_command("generate_class", {
            "name": "Calculator", 
            "description": "A simple calculator class",
            "language": "python"
        })
        
        if result.success:
            generated_code = result.data.get("generated_code", "")
            if "class Calculator" in generated_code:
                print("✅ Class generation successful")
            else:
                print("⚠️ Class generated but format unexpected")
        else:
            print(f"❌ Class generation failed: {result.error}")
        
        # Test component generation
        result = await adapter.execute_command("generate_component", {
            "type": "react",
            "name": "Button",
            "description": "A reusable button component"
        })
        
        if result.success:
            print("✅ Component generation successful")
        else:
            print(f"⚠️ Component generation issues: {result.error}")
            
    except Exception as e:
        print(f"❌ Code generation test failed: {e}")
        return False
    finally:
        await adapter.cleanup()
    
    print("✅ Code generation test completed\n")
    return True


async def test_project_analysis():
    """Test project analysis capabilities"""
    print("📊 Testing Project Analysis...")
    
    adapter = ClaudeCodeAdapter()
    await adapter.initialize()
    
    try:
        # Analyze current project
        result = await adapter.execute_command("analyze_project", {
            "project_path": str(Path.cwd())
        })
        
        if result.success:
            project_info = result.data.get("project_info", {})
            print("✅ Project analysis successful")
            print(f"📁 Project type: {project_info.get('type', 'unknown')}")
            print(f"📄 Total files: {project_info.get('total_files', 0)}")
            print(f"🐍 Python files: {project_info.get('python_files', 0)}")
            print(f"📜 Has Git: {project_info.get('has_git', False)}")
        else:
            print(f"❌ Project analysis failed: {result.error}")
            return False
        
        # Get project info
        result = await adapter.execute_command("get_project_info", {
            "project_path": str(Path.cwd())
        })
        
        if result.success:
            print("✅ Project info retrieval successful")
        else:
            print(f"⚠️ Project info retrieval issues: {result.error}")
            
    except Exception as e:
        print(f"❌ Project analysis test failed: {e}")
        return False
    finally:
        await adapter.cleanup()
    
    print("✅ Project analysis test completed\n") 
    return True


async def test_security_validation():
    """Test security and validation features"""
    print("🔒 Testing Security Features...")
    
    config = {
        'security': {
            'restricted_operations': ['delete', 'system_admin'],
            'allowed_operations': ['read', 'write', 'create', 'generate']
        }
    }
    
    adapter = ClaudeCodeAdapter(config)
    await adapter.initialize()
    
    try:
        # Test dangerous command blocking
        result = await adapter.execute_command("run_bash_command", {
            "command": "rm -rf /"  # This should be blocked
        })
        
        if not result.success:
            print("✅ Dangerous command properly blocked")
        else:
            print("❌ Security issue: Dangerous command was not blocked")
            return False
        
        # Test safe command allowed
        result = await adapter.execute_command("generate_function", {
            "name": "safe_function",
            "description": "This is a safe operation"
        })
        
        if result.success:
            print("✅ Safe operation allowed correctly")
        else:
            print(f"⚠️ Safe operation had issues: {result.error}")
        
        print("✅ Security validation successful")
        
    except Exception as e:
        print(f"❌ Security test failed: {e}")
        return False
    finally:
        await adapter.cleanup()
    
    print("✅ Security test completed\n")
    return True


async def main():
    """Run all basic tests"""
    print("🚀 Starting Claude Code Adapter Basic Tests\n")
    print("=" * 60)
    
    test_results = []
    
    # Run all test suites
    tests = [
        ("MCP Client", test_mcp_client_basic),
        ("Adapter Init", test_adapter_initialization), 
        ("File Operations", test_file_operations),
        ("Code Generation", test_code_generation),
        ("Project Analysis", test_project_analysis),
        ("Security", test_security_validation)
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"Running {test_name} test...")
            result = await test_func()
            test_results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} test ERROR: {e}")
            test_results.append((test_name, False))
        
        print()
    
    # Summary
    print("=" * 60)
    print("📋 TEST SUMMARY:")
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests PASSED! Claude Code Adapter is working correctly!")
        print("🚀 Ready for voice-driven programming!")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please check the issues above.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)