#!/usr/bin/env python3
"""
Claude Echo 语音助手 - 终端UI界面
Terminal UI Interface for Voice Assistant
"""

import asyncio
import sys
import os
from pathlib import Path
import threading
import queue
import time
from datetime import datetime

# 设置路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

class VoiceUI:
    """语音助手终端UI界面"""
    
    def __init__(self):
        self.running = False
        self.voice_queue = queue.Queue()
        self.command_history = []
        self.session_stats = {
            'commands_processed': 0,
            'session_start': datetime.now(),
            'recognition_accuracy': []
        }

    def display_header(self):
        """显示界面头部"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=" * 70)
        print("🎤 CLAUDE ECHO - 智能语音编程助手".center(70))
        print("=" * 70)
        print()

    def display_menu(self):
        """显示主菜单"""
        print("📋 主菜单:")
        print("  1. 🎤 开始语音对话")
        print("  2. ⌨️  文本命令模式") 
        print("  3. 📊 查看会话统计")
        print("  4. 🔧 系统设置")
        print("  5. 📖 使用帮助")
        print("  6. 🚪 退出程序")
        print("-" * 70)

    def display_voice_mode(self):
        """显示语音模式界面"""
        self.display_header()
        print("🎤 语音对话模式 - 已激活".center(70))
        print("-" * 70)
        print()
        print("📝 操作说明:")
        print("  • 按 Enter 开始录音 (5秒)")
        print("  • 输入 'quit' 返回主菜单")
        print("  • 输入 'help' 显示帮助")
        print()
        print("🎯 建议命令:")
        print("  • '你好Claude' - 基础问候")
        print("  • '创建Python文件' - 文件操作")
        print("  • '分析代码结构' - 代码分析")
        print("  • '运行测试' - 项目操作")
        print()
        print("-" * 70)

    def display_stats(self):
        """显示会话统计"""
        self.display_header()
        print("📊 会话统计信息".center(70))
        print("-" * 70)
        print()
        
        session_duration = datetime.now() - self.session_stats['session_start']
        avg_accuracy = sum(self.session_stats['recognition_accuracy']) / len(self.session_stats['recognition_accuracy']) if self.session_stats['recognition_accuracy'] else 0
        
        print(f"🕐 会话时长: {session_duration}")
        print(f"📝 处理命令: {self.session_stats['commands_processed']} 个")
        print(f"🎯 平均识别率: {avg_accuracy:.1%}")
        print()
        
        if self.command_history:
            print("📋 最近命令:")
            for i, cmd in enumerate(self.command_history[-5:], 1):
                print(f"  {i}. {cmd['text']} (置信度: {cmd['confidence']:.2f})")
        
        print()
        print("按任意键返回...")
        input()

    def display_help(self):
        """显示帮助信息"""
        self.display_header()
        print("📖 使用帮助".center(70))
        print("-" * 70)
        print()
        print("🎤 语音命令示例:")
        print()
        print("📁 文件操作:")
        print("  • '创建文件 main.py'")
        print("  • '读取文件内容'")
        print("  • '删除临时文件'")
        print()
        print("💻 编程任务:")
        print("  • '写一个计算器函数'")
        print("  • '添加错误处理'")
        print("  • '优化这段代码'")
        print()
        print("🔍 项目分析:")
        print("  • '分析项目结构'")
        print("  • '检查代码质量'")
        print("  • '运行单元测试'")
        print()
        print("🐛 调试帮助:")
        print("  • '找到这个错误的原因'")
        print("  • '修复语法问题'")
        print("  • '解释这个警告'")
        print()
        print("按任意键返回...")
        input()

    async def voice_recognition_loop(self):
        """语音识别循环"""
        try:
            # 导入语音识别模块
            import whisper
            import pyaudio
            import wave
            import tempfile
            import struct
            
            print("🔄 初始化语音识别系统...")
            model = whisper.load_model("base")
            audio = pyaudio.PyAudio()
            
            # 音频参数
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 16000
            RECORD_SECONDS = 5
            
            print("✅ 语音系统就绪")
            
            while True:
                user_input = input("\n按 Enter 开始录音，输入 'quit' 退出，'help' 显示帮助: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'help':
                    self.display_help()
                    self.display_voice_mode()
                    continue
                
                # 开始录音
                await self.record_and_process(model, audio, CHUNK, FORMAT, CHANNELS, RATE, RECORD_SECONDS)
                            
        except ImportError as e:
            print(f"❌ 导入错误: {e}")
            print("请安装必要依赖: pip install openai-whisper pyttsx3 pyaudio")
        except Exception as e:
            print(f"❌ 系统错误: {e}")

    async def record_and_process(self, model, audio, chunk, format, channels, rate, record_seconds):
        """录音和处理"""
        print("\n🔴 录音中... (5秒)")
        print("请开始说话...")
        
        # 录音
        stream = audio.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
        frames = []
        max_volume = 0
        
        for i in range(0, int(rate / chunk * record_seconds)):
            data = stream.read(chunk)
            frames.append(data)
            
            # 音量检测
            import struct
            audio_data = struct.unpack(f"{chunk}h", data)
            volume = max(abs(x) for x in audio_data)
            max_volume = max(max_volume, volume)
            
            # 进度显示
            progress = (i + 1) * chunk / (rate * record_seconds)
            volume_bar = '|' * min(10, volume // 1000)
            print(f"\r🔴 录音进度: {'=' * int(progress * 20)} {progress * 100:.0f}% 音量:{volume_bar:<10}", end="")
        
        print(f"\n⏹️ 录音完成 (音量: {max_volume})")
        stream.stop_stream()
        stream.close()
        
        # 保存和识别
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            wf = wave.open(tmp_file.name, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(format))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            # 语音识别
            print("🧠 正在识别...")
            try:
                result = model.transcribe(
                    tmp_file.name, 
                    language="zh", 
                    fp16=False, 
                    verbose=False,
                    temperature=0.0
                )
                text = result["text"].strip()
                
                # 计算置信度
                if result.get("segments"):
                    confidences = [1.0 - seg.get("no_speech_prob", 1.0) for seg in result["segments"]]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                else:
                    avg_confidence = 0.0
                
                print(f"\n💬 识别结果: '{text}'")
                print(f"📊 置信度: {avg_confidence:.2f}")
                
                # 记录统计
                self.command_history.append({
                    'text': text,
                    'confidence': avg_confidence,
                    'timestamp': datetime.now()
                })
                self.session_stats['commands_processed'] += 1
                self.session_stats['recognition_accuracy'].append(avg_confidence)
                
                # 处理命令
                if text and len(text) > 1 and avg_confidence > 0.3:
                    response = self.process_command(text)
                    print(f"🤖 Claude: {response}")
                    
                    # 语音回复
                    try:
                        import pyttsx3
                        engine = pyttsx3.init()
                        engine.say(response)
                        engine.runAndWait()
                    except:
                        pass
                else:
                    print("⚠️ 识别质量较低，建议:")
                    print("  - 在安静环境中重试")
                    print("  - 距离麦克风近一些")
                    print("  - 说话清晰一些")
                    
            except Exception as e:
                print(f"❌ 识别失败: {e}")
            
            # 清理临时文件
            try:
                time.sleep(1.0)
                os.unlink(tmp_file.name)
            except:
                pass

    def process_command(self, text):
        """处理语音命令"""
        text_lower = text.lower()
        
        # 基础响应
        if "你好" in text or "hello" in text:
            return "你好！我是Claude语音编程助手，准备协助您进行编程工作。"
        elif "时间" in text or "time" in text:
            return f"现在是 {datetime.now().strftime('%H点%M分')}"
        elif "创建" in text and "文件" in text:
            return "好的，我可以帮您创建文件。请告诉我具体的文件名和内容。"
        elif "分析" in text and ("代码" in text or "项目" in text):
            return "我将分析项目代码结构。请稍等，正在扫描项目文件..."
        elif "测试" in text or "test" in text:
            return "准备运行测试。请确认要测试的具体模块。"
        elif "帮助" in text or "help" in text:
            return "我可以帮您进行文件操作、代码编写、项目分析、调试等任务。"
        elif "状态" in text or "status" in text:
            return f"系统运行正常。已处理 {self.session_stats['commands_processed']} 个命令。"
        else:
            return f"收到指令：{text}。正在处理您的请求..."

    async def text_mode(self):
        """文本命令模式"""
        self.display_header()
        print("⌨️ 文本命令模式".center(70))
        print("-" * 70)
        print("输入命令 (输入 'exit' 退出):")
        
        while True:
            try:
                user_input = input("\n💬 您: ").strip()
                if user_input.lower() in ['exit', 'quit', '退出']:
                    break
                
                if user_input:
                    response = self.process_command(user_input)
                    print(f"🤖 Claude: {response}")
                    
                    # 记录统计
                    self.command_history.append({
                        'text': user_input,
                        'confidence': 1.0,  # 文本输入置信度为100%
                        'timestamp': datetime.now()
                    })
                    self.session_stats['commands_processed'] += 1
                    self.session_stats['recognition_accuracy'].append(1.0)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ 处理错误: {e}")

    async def run(self):
        """运行主程序"""
        self.running = True
        
        while self.running:
            self.display_header()
            self.display_menu()
            
            try:
                choice = input("请选择操作 (1-6): ").strip()
                
                if choice == '1':
                    self.display_voice_mode()
                    await self.voice_recognition_loop()
                elif choice == '2':
                    await self.text_mode()
                elif choice == '3':
                    self.display_stats()
                elif choice == '4':
                    print("\n🔧 系统设置:")
                    print("  • 语音模型: Whisper Base")
                    print("  • 识别语言: 中文+英文")
                    print("  • 置信度阈值: 0.3")
                    print("  • 录音时长: 5秒")
                    print("\n按任意键返回...")
                    input()
                elif choice == '5':
                    self.display_help()
                elif choice == '6':
                    print("\n👋 感谢使用 Claude Echo 语音助手!")
                    self.running = False
                else:
                    print("❌ 无效选择，请重试")
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print("\n\n👋 程序被中断")
                self.running = False
            except Exception as e:
                print(f"❌ 程序错误: {e}")
                time.sleep(2)

async def main():
    """主函数"""
    print("🚀 启动 Claude Echo 语音助手 UI...")
    ui = VoiceUI()
    await ui.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序退出")
    except Exception as e:
        print(f"启动错误: {e}")