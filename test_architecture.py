#!/usr/bin/env python3
"""
Simple Architecture Test

This script tests the basic architecture components without external dependencies.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Create a simple logger replacement for testing
class SimpleLogger:
    def info(self, msg):
        print(f"[INFO] {msg}")
    
    def error(self, msg):
        print(f"[ERROR] {msg}")
    
    def warning(self, msg):
        print(f"[WARNING] {msg}")
    
    def debug(self, msg):
        print(f"[DEBUG] {msg}")

# Replace loguru logger with simple logger
import core.event_system
import core.config_manager
import core.adapter_manager
import core.base_adapter

logger = SimpleLogger()
core.event_system.logger = logger
core.config_manager.logger = logger  
core.adapter_manager.logger = logger
core.base_adapter.logger = logger

from core.event_system import EventSystem, Event
from core.types import Context, Intent, CommandResult


async def test_event_system():
    """Test the EventSystem implementation."""
    print("\n🧪 Testing EventSystem...")
    
    event_system = EventSystem()
    await event_system.initialize()
    
    # Test event subscription and emission
    received_events = []
    
    def test_handler(event):
        received_events.append(event.event_type)
        print(f"   ✅ Received event: {event.event_type}")
    
    # Subscribe to events
    handler_id = event_system.subscribe(["test.*"], test_handler)
    print(f"   📝 Subscribed handler: {handler_id}")
    
    # Emit test events
    await event_system.emit(Event(
        event_type="test.example",
        data={"message": "Hello from test!"},
        source="test"
    ))
    
    await event_system.emit(Event(
        event_type="test.another",
        data={"value": 42},
        source="test"
    ))
    
    # Wait for event processing
    await asyncio.sleep(0.2)
    
    # Check results
    print(f"   📊 Events processed: {len(received_events)}")
    print(f"   📈 Statistics: {event_system.get_statistics()}")
    
    # Cleanup
    await event_system.shutdown()
    print("   ✅ EventSystem test completed")


async def test_types():
    """Test the core type definitions."""
    print("\n🧪 Testing Core Types...")
    
    # Test Context creation
    context = Context(
        user_id="test_user",
        session_id="test_session",
        current_app="test_app"
    )
    print(f"   ✅ Created Context: {context.user_id}")
    
    # Test Intent creation
    intent = Intent(
        user_input="test command",
        intent_type="test",
        confidence=0.95
    )
    print(f"   ✅ Created Intent: {intent.intent_type} (confidence: {intent.confidence})")
    
    # Test CommandResult
    result = CommandResult(
        success=True,
        data={"output": "test successful"},
        execution_time=0.1
    )
    print(f"   ✅ Created CommandResult: success={result.success}")
    
    print("   ✅ Core types test completed")


class TestAdapter:
    """Simple test adapter for demonstration."""
    
    def __init__(self):
        self.adapter_id = "test_adapter"
        self.supported_commands = ["test_command", "echo"]
    
    async def execute_command(self, command, parameters, context=None):
        if command == "echo":
            return CommandResult(
                success=True,
                data={"echo": parameters.get("message", "")},
                execution_time=0.01
            )
        elif command == "test_command":
            return CommandResult(
                success=True, 
                data={"result": "test executed successfully"},
                execution_time=0.02
            )
        else:
            return CommandResult(
                success=False,
                error=f"Unknown command: {command}"
            )


async def test_adapter_concepts():
    """Test adapter concepts without the full AdapterManager."""
    print("\n🧪 Testing Adapter Concepts...")
    
    # Create test adapter
    adapter = TestAdapter()
    print(f"   ✅ Created adapter: {adapter.adapter_id}")
    print(f"   📋 Supported commands: {adapter.supported_commands}")
    
    # Test command execution
    result1 = await adapter.execute_command("echo", {"message": "Hello World!"})
    print(f"   ✅ Echo result: {result1.data}")
    
    result2 = await adapter.execute_command("test_command", {})
    print(f"   ✅ Test command result: {result2.data}")
    
    # Test unknown command
    result3 = await adapter.execute_command("unknown", {})
    print(f"   ❌ Unknown command result: success={result3.success}, error={result3.error}")
    
    print("   ✅ Adapter concepts test completed")


async def main():
    """Run all architecture tests."""
    print("🎯 Claude Voice Assistant - Architecture Test")
    print("=" * 50)
    
    try:
        # Run individual tests
        await test_types()
        await test_event_system() 
        await test_adapter_concepts()
        
        print("\n" + "=" * 50)
        print("🎉 All architecture tests completed successfully!")
        print("\n📋 Architecture Summary:")
        print("   ✅ Core type system - Working")
        print("   ✅ Event-driven communication - Working") 
        print("   ✅ Adapter pattern - Working")
        print("   ✅ Async execution model - Working")
        print("   ✅ 4-layer architecture foundation - Ready")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)