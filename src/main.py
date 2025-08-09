#!/usr/bin/env python3
"""
Claude Voice Assistant - Main Entry Point

This module provides the main entry point for the Claude Voice Assistant
architecture. It demonstrates the 4-layer architecture implementation
and core framework components.
"""

import asyncio
import sys
from pathlib import Path
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.architecture import ClaudeVoiceAssistant


async def main():
    """Main entry point for the Claude Voice Assistant."""
    # Configure logging
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | {message}",
        level="INFO"
    )
    
    # Add file logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.add(
        log_dir / "claude-echo.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
        level="DEBUG",
        rotation="1 day",
        retention="1 week"
    )
    
    logger.info("🎤 Starting Claude Voice Assistant")
    logger.info("Architecture: 4-Layer (UI -> Intelligence -> Adapters -> Execution)")
    
    # Create and initialize the assistant
    assistant = ClaudeVoiceAssistant(
        config_dir="config",
        environment="development"
    )
    
    try:
        # Start the assistant
        await assistant.start()
        
    except KeyboardInterrupt:
        logger.info("👋 Shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        return 1
    
    logger.info("✅ Claude Voice Assistant stopped")
    return 0


def run_demo():
    """Run a simple demo of the architecture components."""
    import asyncio
    from core.config_manager import ConfigManager
    from core.event_system import EventSystem, Event
    from core.adapter_manager import AdapterManager
    
    async def demo():
        logger.info("🚀 Running Architecture Demo")
        
        # Demo 1: Configuration Management
        logger.info("📋 Testing ConfigManager...")
        config_manager = ConfigManager(config_dir="config", environment="development")
        await config_manager.initialize()
        
        # Test configuration reading
        app_name = config_manager.get("platform.name", "Unknown")
        log_level = config_manager.get("debug.log_level", "INFO")
        
        logger.info(f"   App Name: {app_name}")
        logger.info(f"   Log Level: {log_level}")
        
        # Demo 2: Event System
        logger.info("📡 Testing EventSystem...")
        event_system = EventSystem()
        await event_system.initialize()
        
        # Test event subscription and emission
        received_events = []
        
        def event_handler(event):
            received_events.append(event.event_type)
            logger.info(f"   Received event: {event.event_type}")
        
        event_system.subscribe("demo.*", event_handler)
        
        await event_system.emit(Event(
            event_type="demo.test_event",
            data={"message": "Hello from demo!"},
            source="demo"
        ))
        
        # Wait a bit for event processing
        await asyncio.sleep(0.1)
        
        logger.info(f"   Events received: {len(received_events)}")
        
        # Demo 3: Adapter Manager
        logger.info("🔌 Testing AdapterManager...")
        adapter_manager = AdapterManager(
            event_system=event_system,
            adapter_paths=["src/adapters"]
        )
        await adapter_manager.initialize()
        
        adapter_count = len(adapter_manager._adapters)
        logger.info(f"   Loaded adapters: {adapter_count}")
        
        # Cleanup
        await adapter_manager.shutdown()
        await event_system.shutdown()
        await config_manager.shutdown()
        
        logger.info("✅ Architecture demo completed successfully!")
    
    asyncio.run(demo())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Claude Voice Assistant")
    parser.add_argument("--demo", action="store_true", help="Run architecture demo")
    parser.add_argument("--config-dir", default="config", help="Configuration directory")
    parser.add_argument("--environment", default="development", help="Environment name")
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    else:
        sys.exit(asyncio.run(main()))