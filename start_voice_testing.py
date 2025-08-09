#!/usr/bin/env python3
"""
Claude Voice Assistant - 真实语音测试启动器
Real Voice Testing Launcher

使用真实语音功能测试智能学习系统
"""

import asyncio
import sys
import os
import time
from pathlib import Path
from datetime import datetime

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger
import threading
import queue
import json

# 导入必要的组件
from core.config_manager import ConfigManager
from core.event_system import EventSystem
from core.architecture import ClaudeVoiceAssistant

class RealVoiceTesting:
    """真实语音测试管理器"""
    
    def __init__(self):
        self.config_manager = None
        self.voice_assistant = None
        self.event_system = None
        self.testing_active = False
        self.test_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.test_results = []
        self.user_feedback = queue.Queue()
        
    async def initialize(self):
        """初始化语音测试环境"""
        try:
            logger.info("🎤 启动 Claude Voice Assistant 真实语音测试...")
            
            # 1. 加载真实语音配置
            self.config_manager = ConfigManager(
                config_dir="config",
                config_file="real_voice_config.yaml"
            )
            await self.config_manager.initialize()
            logger.success("✅ 配置管理器初始化完成")
            
            # 2. 初始化事件系统
            self.event_system = EventSystem()
            await self.event_system.initialize()
            logger.success("✅ 事件系统初始化完成")
            
            # 3. 初始化语音助手
            self.voice_assistant = ClaudeVoiceAssistant()
            await self.voice_assistant.initialize(self.config_manager)
            logger.success("✅ 语音助手核心系统初始化完成")
            
            # 4. 订阅测试相关事件
            self._setup_event_handlers()
            logger.success("✅ 事件处理器设置完成")
            
            # 5. 检查语音功能状态
            await self._check_voice_system()
            
            logger.success("🎉 语音测试环境初始化完成！")
            return True
            
        except Exception as e:
            logger.error(f"❌ 初始化失败: {e}")
            return False
    
    def _setup_event_handlers(self):
        """设置事件处理器"""
        # 语音识别事件
        self.event_system.subscribe(
            ["speech.recognition.*"],
            self._handle_speech_event
        )
        
        # 学习相关事件
        self.event_system.subscribe(
            ["learning.*"],
            self._handle_learning_event
        )
        
        # 用户交互事件
        self.event_system.subscribe(
            ["user.*"],
            self._handle_user_event
        )
        
        # 系统响应事件
        self.event_system.subscribe(
            ["response.*"],
            self._handle_response_event
        )
    
    async def _check_voice_system(self):
        """检查语音系统状态"""
        try:
            logger.info("🔍 检查语音系统状态...")
            
            # 检查音频设备
            try:
                import pyaudio
                p = pyaudio.PyAudio()
                
                # 检查输入设备（麦克风）
                input_devices = []
                output_devices = []
                
                for i in range(p.get_device_count()):
                    info = p.get_device_info_by_index(i)
                    if info['maxInputChannels'] > 0:
                        input_devices.append((i, info['name']))
                    if info['maxOutputChannels'] > 0:
                        output_devices.append((i, info['name']))
                
                p.terminate()
                
                logger.info(f"📱 可用输入设备: {len(input_devices)}个")
                for idx, name in input_devices[:3]:  # 显示前3个
                    logger.info(f"  - {name}")
                    
                logger.info(f"🔊 可用输出设备: {len(output_devices)}个")
                for idx, name in output_devices[:3]:  # 显示前3个
                    logger.info(f"  - {name}")
                    
            except ImportError:
                logger.warning("⚠️  pyaudio未安装，将使用基础音频功能")
            except Exception as e:
                logger.warning(f"⚠️  音频设备检查失败: {e}")
            
            # 检查Whisper模型
            try:
                import whisper
                logger.info("🤖 Whisper模型检查...")
                # 这里可以添加模型加载测试
                logger.success("✅ Whisper可用")
            except ImportError:
                logger.error("❌ Whisper未安装，请运行: pip install openai-whisper")
                return False
            except Exception as e:
                logger.warning(f"⚠️  Whisper检查警告: {e}")
            
            # 检查TTS
            try:
                import pyttsx3
                engine = pyttsx3.init()
                voices = engine.getProperty('voices')
                logger.info(f"🗣️  可用TTS语音: {len(voices) if voices else 0}个")
                engine.stop()
                logger.success("✅ TTS可用")
            except ImportError:
                logger.error("❌ pyttsx3未安装，请运行: pip install pyttsx3")
                return False
            except Exception as e:
                logger.warning(f"⚠️  TTS检查警告: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 语音系统检查失败: {e}")
            return False
    
    async def start_voice_testing(self):
        """开始语音测试"""
        if not await self.initialize():
            logger.error("❌ 初始化失败，无法开始测试")
            return
        
        self.testing_active = True
        logger.info("🎤 开始语音测试会话...")
        
        # 显示测试说明
        self._show_testing_instructions()
        
        # 启动用户界面线程
        ui_thread = threading.Thread(target=self._run_user_interface, daemon=True)
        ui_thread.start()
        
        try:
            # 主测试循环
            await self._main_testing_loop()
        except KeyboardInterrupt:
            logger.info("🛑 用户中断测试")
        finally:
            await self._cleanup()
    
    def _show_testing_instructions(self):
        """显示测试说明"""
        instructions = """
╭──────────────────────────────────────────────────────────────╮
│                🎤 Claude Voice Assistant 语音测试              │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  🎯 测试功能：                                                │
│    • 实时语音识别和命令执行                                    │
│    • 个性化学习和适应                                         │
│    • 错误纠正和改进                                           │
│    • 多用户支持（如果配置）                                    │
│                                                               │
│  🎮 测试命令：                                                │
│    • 输入 'listen' 开始语音识别                                │
│    • 输入 'stop' 停止当前识别                                  │
│    • 输入 'test [文本]' 测试文本命令                           │
│    • 输入 'stats' 查看学习统计                                 │
│    • 输入 'correct' 进行错误纠正                               │
│    • 输入 'help' 查看详细帮助                                  │
│    • 输入 'quit' 退出测试                                      │
│                                                               │
│  💡 使用建议：                                                │
│    • 请在安静环境中测试，确保麦克风工作正常                     │
│    • 说话清晰，距离麦克风适中                                  │
│    • 尝试不同的编程相关命令                                    │
│    • 注意系统的学习和适应反馈                                  │
│                                                               │
╰──────────────────────────────────────────────────────────────╯
        """
        print(instructions)
    
    def _run_user_interface(self):
        """运行用户界面"""
        while self.testing_active:
            try:
                user_input = input("🎤 Voice Test > ").strip()
                if user_input:
                    self.user_feedback.put(user_input)
            except EOFError:
                break
            except KeyboardInterrupt:
                break
    
    async def _main_testing_loop(self):
        """主测试循环"""
        logger.info("🔄 进入主测试循环")
        
        while self.testing_active:
            try:
                # 检查用户输入
                if not self.user_feedback.empty():
                    command = self.user_feedback.get()
                    await self._process_test_command(command)
                
                # 短暂等待
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ 测试循环错误: {e}")
                await asyncio.sleep(1)
    
    async def _process_test_command(self, command: str):
        """处理测试命令"""
        command = command.lower().strip()
        
        try:
            if command == 'quit' or command == 'exit':
                self.testing_active = False
                logger.info("👋 结束测试会话")
                
            elif command == 'listen':
                await self._start_voice_listening()
                
            elif command == 'stop':
                await self._stop_voice_listening()
                
            elif command.startswith('test '):
                test_text = command[5:].strip()
                await self._test_text_command(test_text)
                
            elif command == 'stats':
                await self._show_learning_stats()
                
            elif command == 'correct':
                await self._start_correction_mode()
                
            elif command == 'help':
                self._show_detailed_help()
                
            else:
                logger.warning(f"⚠️  未知命令: {command}")
                logger.info("💡 输入 'help' 查看可用命令")
                
        except Exception as e:
            logger.error(f"❌ 命令处理错误: {e}")
    
    async def _start_voice_listening(self):
        """开始语音监听"""
        logger.info("🎤 开始语音监听...")
        try:
            # 这里调用语音助手的语音监听功能
            result = await self.voice_assistant.start_voice_session()
            logger.info(f"📝 语音识别结果: {result}")
        except Exception as e:
            logger.error(f"❌ 语音监听失败: {e}")
    
    async def _stop_voice_listening(self):
        """停止语音监听"""
        logger.info("🛑 停止语音监听")
        try:
            await self.voice_assistant.stop_voice_session()
        except Exception as e:
            logger.error(f"❌ 停止语音监听失败: {e}")
    
    async def _test_text_command(self, text: str):
        """测试文本命令"""
        logger.info(f"📝 测试文本命令: {text}")
        try:
            result = await self.voice_assistant.process_text_input(text)
            logger.info(f"📋 命令执行结果: {result}")
        except Exception as e:
            logger.error(f"❌ 文本命令执行失败: {e}")
    
    async def _show_learning_stats(self):
        """显示学习统计"""
        logger.info("📊 学习统计信息:")
        try:
            # 获取学习统计数据
            stats = await self.voice_assistant.get_learning_statistics()
            
            print("╭────────────────────────────────────────╮")
            print("│           📊 学习统计信息              │")
            print("├────────────────────────────────────────┤")
            print(f"│  识别准确率: {stats.get('accuracy', 'N/A')}%      │")
            print(f"│  学习样本数: {stats.get('samples', 'N/A')}个      │")
            print(f"│  纠正次数:   {stats.get('corrections', 'N/A')}次  │")
            print(f"│  会话时长:   {stats.get('duration', 'N/A')}分钟  │")
            print("╰────────────────────────────────────────╯")
            
        except Exception as e:
            logger.error(f"❌ 获取学习统计失败: {e}")
    
    async def _start_correction_mode(self):
        """开始纠错模式"""
        logger.info("🔧 进入纠错模式")
        print("请说出需要纠正的内容，或输入 'done' 完成纠错")
        # 实现纠错逻辑
    
    def _show_detailed_help(self):
        """显示详细帮助"""
        help_text = """
📚 详细命令说明：

🎤 语音命令：
  listen          - 开始语音识别，说完后自动处理
  stop            - 停止当前的语音识别

💬 文本命令：
  test <内容>     - 测试文本命令，例如: test create file hello.py
  
📊 统计和监控：
  stats           - 显示学习统计信息
  
🔧 学习和纠错：
  correct         - 进入纠错模式，改进识别准确率
  
🆘 其他：
  help            - 显示此帮助信息
  quit/exit       - 退出测试程序

🎯 测试建议：
  1. 从简单的语音命令开始，如"创建文件"
  2. 逐渐尝试复杂的编程命令
  3. 注意观察系统的学习反馈
  4. 主动纠正错误识别，帮助系统学习
        """
        print(help_text)
    
    async def _handle_speech_event(self, event):
        """处理语音事件"""
        logger.debug(f"🎤 语音事件: {event.event_type}")
    
    async def _handle_learning_event(self, event):
        """处理学习事件"""
        logger.debug(f"🧠 学习事件: {event.event_type}")
    
    async def _handle_user_event(self, event):
        """处理用户事件"""
        logger.debug(f"👤 用户事件: {event.event_type}")
    
    async def _handle_response_event(self, event):
        """处理响应事件"""
        logger.debug(f"📢 响应事件: {event.event_type}")
    
    async def _cleanup(self):
        """清理资源"""
        logger.info("🧹 清理测试资源...")
        self.testing_active = False
        
        if self.voice_assistant:
            await self.voice_assistant.cleanup()
        
        if self.event_system:
            await self.event_system.cleanup()
        
        logger.success("✅ 清理完成")

async def main():
    """主函数"""
    try:
        # 检查依赖
        required_packages = ['whisper', 'pyttsx3', 'pyaudio']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error("❌ 缺少必要的包，请安装:")
            for package in missing_packages:
                if package == 'whisper':
                    logger.error("  pip install openai-whisper")
                elif package == 'pyaudio':
                    logger.error("  pip install pyaudio")
                else:
                    logger.error(f"  pip install {package}")
            return
        
        # 启动语音测试
        tester = RealVoiceTesting()
        await tester.start_voice_testing()
        
    except KeyboardInterrupt:
        logger.info("👋 程序被用户中断")
    except Exception as e:
        logger.error(f"❌ 程序运行错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    
    # 运行程序
    asyncio.run(main())