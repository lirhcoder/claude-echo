#!/usr/bin/env python3
"""
Claude Voice Assistant - Alpha Interactive Testing Mode

This script provides an interactive testing environment for Alpha phase,
allowing text-based simulation of voice commands to test core functionality.
"""

import asyncio
import sys
from pathlib import Path
from loguru import logger
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.event_system import EventSystem, Event
from core.config_manager import ConfigManager


class AlphaTestingInterface:
    """Interactive testing interface for Alpha phase."""
    
    def __init__(self):
        self.event_system = None
        self.config_manager = None
        self.test_results = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    async def initialize(self):
        """Initialize testing environment."""
        logger.info("🔧 Initializing Alpha Testing Interface...")
        
        # Initialize core systems
        self.config_manager = ConfigManager(
            config_dir="config", 
            environment="alpha_testing"
        )
        await self.config_manager.initialize()
        
        self.event_system = EventSystem()
        await self.event_system.initialize()
        
        # Subscribe to events for testing
        self.event_system.subscribe("speech.*", self._handle_speech_event)
        self.event_system.subscribe("command.*", self._handle_command_event)
        self.event_system.subscribe("response.*", self._handle_response_event)
        
        logger.info("✅ Alpha Testing Interface initialized")
    
    def _handle_speech_event(self, event):
        """Handle simulated speech events."""
        logger.info(f"🎤 [MOCK] Speech Input: {event.data.get('text', 'N/A')}")
        self.test_results.append({
            "timestamp": datetime.now().isoformat(),
            "type": "speech_input",
            "data": event.data
        })
    
    def _handle_command_event(self, event):
        """Handle command processing events."""
        logger.info(f"⚡ [MOCK] Command Processed: {event.data.get('command', 'N/A')}")
        self.test_results.append({
            "timestamp": datetime.now().isoformat(),
            "type": "command_processed",
            "data": event.data
        })
    
    def _handle_response_event(self, event):
        """Handle response events."""
        logger.info(f"🗣️  [MOCK] Response Generated: {event.data.get('response', 'N/A')}")
        self.test_results.append({
            "timestamp": datetime.now().isoformat(),
            "type": "response_generated",
            "data": event.data
        })
    
    async def simulate_voice_input(self, text_input: str):
        """Simulate voice input with text."""
        logger.info(f"🎙️  Simulating voice input: '{text_input}'")
        
        # Emit speech recognition event
        await self.event_system.emit(Event(
            event_type="speech.recognized",
            data={
                "text": text_input,
                "confidence": 0.95,
                "language": "zh-CN",
                "simulated": True
            },
            source="alpha_test_interface"
        ))
        
        # Simulate intent parsing
        intent = await self._parse_intent(text_input)
        
        # Emit command event
        await self.event_system.emit(Event(
            event_type="command.parsed",
            data={
                "command": intent["command"],
                "parameters": intent["parameters"],
                "confidence": intent["confidence"],
                "simulated": True
            },
            source="alpha_test_interface"
        ))
        
        # Simulate response generation
        response = await self._generate_response(intent)
        
        # Emit response event
        await self.event_system.emit(Event(
            event_type="response.generated",
            data={
                "response": response,
                "intent": intent["command"],
                "simulated": True
            },
            source="alpha_test_interface"
        ))
        
        return response
    
    async def _parse_intent(self, text: str) -> dict:
        """Mock intent parsing for Alpha testing."""
        text_lower = text.lower()
        
        # Simple intent matching for testing
        if "你好" in text_lower or "hello" in text_lower:
            return {
                "command": "greeting",
                "parameters": {},
                "confidence": 0.9
            }
        elif "时间" in text_lower or "time" in text_lower:
            return {
                "command": "get_time",
                "parameters": {},
                "confidence": 0.85
            }
        elif "天气" in text_lower or "weather" in text_lower:
            return {
                "command": "get_weather",
                "parameters": {"location": "当前位置"},
                "confidence": 0.8
            }
        elif "claude" in text_lower and "code" in text_lower:
            return {
                "command": "claude_code_integration",
                "parameters": {"action": "test"},
                "confidence": 0.95
            }
        else:
            return {
                "command": "unknown",
                "parameters": {"original_text": text},
                "confidence": 0.3
            }
    
    async def _generate_response(self, intent: dict) -> str:
        """Mock response generation for Alpha testing."""
        command = intent["command"]
        
        responses = {
            "greeting": "你好！我是Claude语音助手，目前运行在Alpha测试模式。",
            "get_time": f"当前时间是 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "get_weather": "[模拟] 今天天气晴朗，温度25°C，适合外出。",
            "claude_code_integration": "[模拟] Claude Code适配器已准备就绪，等待具体指令。",
            "unknown": f"抱歉，我在Alpha测试模式下还无法理解 '{intent['parameters'].get('original_text', '')}'"
        }
        
        return responses.get(command, "系统正在处理您的请求...")
    
    async def run_test_checklist(self):
        """Run the Alpha testing checklist interactively."""
        logger.info("📋 Starting Alpha Testing Checklist...")
        
        test_cases = [
            {"id": "A1.1", "description": "系统启动测试", "input": "系统状态检查"},
            {"id": "A1.2", "description": "语音识别初始化", "input": "你好Claude"},
            {"id": "A1.3", "description": "意图解析测试", "input": "现在几点了"},
            {"id": "A1.4", "description": "事件系统测试", "input": "天气怎么样"},
            {"id": "A1.5", "description": "Claude Code集成测试", "input": "测试Claude Code功能"},
        ]
        
        for test_case in test_cases:
            logger.info(f"\n--- {test_case['id']}: {test_case['description']} ---")
            print(f"🧪 测试输入: {test_case['input']}")
            
            response = await self.simulate_voice_input(test_case['input'])
            print(f"🤖 系统响应: {response}")
            
            # Wait for user input
            user_result = input("✅ 通过 / ❌ 失败 / ⚠️ 问题 (输入结果): ").strip()
            
            self.test_results.append({
                "test_id": test_case['id'],
                "description": test_case['description'],
                "input": test_case['input'],
                "response": response,
                "result": user_result,
                "timestamp": datetime.now().isoformat()
            })
            
            print()
    
    async def interactive_mode(self):
        """Run in interactive testing mode."""
        logger.info("🎮 Starting Interactive Alpha Testing Mode")
        print("\n" + "="*50)
        print("  Claude Voice Assistant - Alpha Interactive Testing")
        print("="*50)
        print("输入文本模拟语音命令，输入 'quit' 或 'exit' 退出")
        print("输入 'checklist' 运行完整测试清单")
        print("-"*50)
        
        while True:
            try:
                user_input = input("\n🎙️  模拟语音输入: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    break
                elif user_input.lower() in ['checklist', '测试清单']:
                    await self.run_test_checklist()
                elif user_input:
                    response = await self.simulate_voice_input(user_input)
                    print(f"🤖 系统响应: {response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 测试会话结束")
                break
    
    async def save_test_results(self):
        """Save test results to file."""
        results_file = Path(f"testing/alpha_test_results_{self.session_id}.json")
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(self.test_results),
                "results": self.test_results
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📄 Test results saved to: {results_file}")
    
    async def shutdown(self):
        """Shutdown testing environment."""
        if self.event_system:
            await self.event_system.shutdown()
        if self.config_manager:
            await self.config_manager.shutdown()
        
        await self.save_test_results()
        logger.info("🏁 Alpha Testing Interface shutdown complete")


async def main():
    """Main entry point for Alpha testing."""
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | {message}",
        level="INFO"
    )
    
    # Create testing interface
    test_interface = AlphaTestingInterface()
    
    try:
        await test_interface.initialize()
        await test_interface.interactive_mode()
        
    except KeyboardInterrupt:
        logger.info("👋 Testing interrupted by user")
    except Exception as e:
        logger.error(f"❌ Testing error: {e}")
    finally:
        await test_interface.shutdown()


if __name__ == "__main__":
    asyncio.run(main())