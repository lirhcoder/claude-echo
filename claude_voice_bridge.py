#!/usr/bin/env python3
"""
Claude Code 语音命令直接桥接器 (增强版)
Enhanced Voice-to-Claude Code Direct Bridge

将语音直接转换为Claude Code命令，支持：
- 实时语音监听
- 自动命令执行
- 会话状态保持
- 智能命令优化
"""

import asyncio
import sys
import os
import subprocess
import tempfile
import wave
import time
import threading
import queue
import json
import signal
from pathlib import Path
from datetime import datetime
import keyboard  # 需要安装: pip install keyboard

# 设置路径
sys.path.insert(0, str(Path(__file__).parent / "src"))


class ClaudeVoiceBridge:
    """Claude Code语音命令直接桥接器"""
    
    def __init__(self):
        self.is_listening = False
        self.is_recording = False
        self.claude_session = None
        self.model = None
        self.session_active = False
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.stats = {
            'commands_sent': 0,
            'session_start': datetime.now(),
            'last_command_time': None
        }
        
    def print_banner(self):
        """显示启动横幅"""
        print("=" * 80)
        print("🎤 CLAUDE CODE 语音命令直接桥接器 (增强版)".center(80))
        print("=" * 80)
        print()
        print("🚀 核心功能:")
        print("  ✨ 语音自动转换为Claude Code命令")
        print("  🔄 实时语音监听和处理")
        print("  💬 保持Claude Code会话状态")
        print("  🧠 智能命令优化和过滤")
        print()
        print("⌨️ 快捷键操作:")
        print("  F1   - 开始/停止语音监听")
        print("  F2   - 手动录音(5秒)")  
        print("  F3   - 查看会话状态")
        print("  ESC  - 退出程序")
        print()
        print("💡 使用技巧:")
        print("  - 说话前短暂停顿，确保识别准确")
        print("  - 使用清晰的中文或英文命令")
        print("  - 避免长时间连续说话")
        print("-" * 80)
        print()
        
    def setup_dependencies(self):
        """设置和检查依赖"""
        print("🔍 检查系统环境...")
        
        # 检查必要模块
        dependencies = [
            ('whisper', 'openai-whisper', 'Whisper语音识别'),
            ('pyaudio', 'pyaudio', 'PyAudio音频处理'),
            ('keyboard', 'keyboard', 'Keyboard全局热键')
        ]
        
        missing = []
        for module, package, desc in dependencies:
            try:
                __import__(module)
                print(f"   ✅ {desc} - 已安装")
            except ImportError:
                print(f"   ❌ {desc} - 未安装")
                missing.append(f"pip install {package}")
        
        if missing:
            print("\\n⚠️ 请先安装缺失依赖:")
            for cmd in missing:
                print(f"   {cmd}")
            return False
            
        # 检查Claude Code CLI
        try:
            result = subprocess.run(['claude', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("   ✅ Claude Code CLI - 已安装")
            else:
                print("   ❌ Claude Code CLI - 版本检查失败")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("   ❌ Claude Code CLI - 未找到")
            print("   请确保Claude Code已安装: https://claude.ai/code")
            return False
            
        return True
        
    def init_whisper(self):
        """初始化Whisper模型"""
        print("🧠 初始化语音识别系统...")
        try:
            import whisper
            # 使用small模型获得更好的准确率
            print("   正在加载Whisper模型 (small)...")
            self.model = whisper.load_model("small")
            print("✅ Whisper模型初始化完成")
            return True
        except Exception as e:
            print(f"❌ Whisper模型初始化失败: {e}")
            print("   尝试使用base模型...")
            try:
                self.model = whisper.load_model("base")
                print("✅ Whisper base模型初始化完成")
                return True
            except Exception as e2:
                print(f"❌ 所有模型初始化失败: {e2}")
                return False
                
    def setup_hotkeys(self):
        """设置全局热键"""
        print("⌨️ 设置全局热键...")
        try:
            keyboard.add_hotkey('f1', self.toggle_listening)
            keyboard.add_hotkey('f2', self.manual_record)
            keyboard.add_hotkey('f3', self.show_status)
            keyboard.add_hotkey('esc', self.shutdown)
            print("✅ 热键设置完成")
            return True
        except Exception as e:
            print(f"❌ 热键设置失败: {e}")
            return False
            
    def toggle_listening(self):
        """切换监听状态"""
        if not self.is_listening:
            self.is_listening = True
            print("\\n🎤 开始语音监听...")
            threading.Thread(target=self.continuous_listening, daemon=True).start()
        else:
            self.is_listening = False
            print("\\n⏹️ 停止语音监听")
            
    def manual_record(self):
        """手动录音"""
        if not self.is_recording:
            print("\\n🔴 手动录音模式 (5秒)...")
            threading.Thread(target=self.single_record, daemon=True).start()
        
    def show_status(self):
        """显示会话状态"""
        duration = datetime.now() - self.stats['session_start']
        print(f"\\n📊 会话状态:")
        print(f"   ⏱️ 运行时长: {duration}")
        print(f"   📤 发送命令: {self.stats['commands_sent']} 个")
        print(f"   🎤 监听状态: {'✅ 活跃' if self.is_listening else '❌ 停止'}")
        print(f"   💬 Claude会话: {'✅ 活跃' if self.session_active else '❌ 未连接'}")
        if self.stats['last_command_time']:
            last_cmd = datetime.now() - self.stats['last_command_time']
            print(f"   📝 上次命令: {last_cmd.seconds}秒前")
            
    def shutdown(self):
        """关闭程序"""
        print("\\n👋 正在关闭语音桥接器...")
        self.is_listening = False
        self.session_active = False
        if self.claude_session:
            try:
                self.claude_session.terminate()
            except:
                pass
        os._exit(0)
        
    def continuous_listening(self):
        """连续语音监听"""
        import pyaudio
        
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        SILENCE_THRESHOLD = 1000
        SILENCE_DURATION = 2  # 2秒静音后处理
        
        audio = pyaudio.PyAudio()
        
        try:
            stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                              input=True, frames_per_buffer=CHUNK)
            
            frames_buffer = []
            silence_frames = 0
            speaking = False
            
            print("👂 监听中... (说话时自动开始录音)")
            
            while self.is_listening:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    
                    # 检查音量
                    import struct
                    audio_data = struct.unpack(f"{CHUNK}h", data)
                    volume = max(abs(x) for x in audio_data)
                    
                    if volume > SILENCE_THRESHOLD:
                        # 检测到声音
                        if not speaking:
                            speaking = True
                            frames_buffer = []
                            print("\\n🔴 检测到语音，开始录音...")
                        frames_buffer.append(data)
                        silence_frames = 0
                    else:
                        # 静音
                        if speaking:
                            frames_buffer.append(data)
                            silence_frames += 1
                            
                            # 静音超过阈值，处理录音
                            if silence_frames > (RATE / CHUNK * SILENCE_DURATION):
                                speaking = False
                                silence_frames = 0
                                
                                if len(frames_buffer) > RATE / CHUNK:  # 至少1秒录音
                                    print("⏹️ 录音结束，正在识别...")
                                    self.process_audio_frames(frames_buffer, RATE, CHANNELS, FORMAT, audio.get_sample_size(FORMAT))
                                frames_buffer = []
                                
                except Exception as e:
                    if self.is_listening:  # 只在仍在监听时报告错误
                        print(f"⚠️ 监听出错: {e}")
                        
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"❌ 连续监听失败: {e}")
        finally:
            audio.terminate()
            
    def single_record(self):
        """单次录音"""
        if self.is_recording:
            return
            
        self.is_recording = True
        try:
            import pyaudio
            
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 16000
            RECORD_SECONDS = 5
            
            audio = pyaudio.PyAudio()
            
            print("请开始说话...")
            stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                              input=True, frames_per_buffer=CHUNK)
            
            frames = []
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
                
                # 显示进度
                progress = (i + 1) / (int(RATE / CHUNK * RECORD_SECONDS))
                print(f"\\r🔴 录音进度: {'=' * int(progress * 15)} {progress * 100:.0f}%", end="")
            
            print("\\n⏹️ 录音完成，正在识别...")
            
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
            self.process_audio_frames(frames, RATE, CHANNELS, FORMAT, audio.get_sample_size(FORMAT))
            
        except Exception as e:
            print(f"❌ 录音失败: {e}")
        finally:
            self.is_recording = False
            
    def process_audio_frames(self, frames, rate, channels, format, sample_width):
        """处理音频帧"""
        try:
            # 保存为临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                wf = wave.open(tmp_file.name, 'wb')
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(rate)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                # 语音识别
                result = self.model.transcribe(
                    tmp_file.name,
                    language="zh",
                    fp16=False,
                    verbose=False,
                    temperature=0.0,
                    no_speech_threshold=0.4,
                    condition_on_previous_text=False
                )
                
                text = result["text"].strip()
                
                # 计算置信度
                if result.get("segments"):
                    confidences = [1.0 - seg.get("no_speech_prob", 1.0) 
                                 for seg in result["segments"]]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                else:
                    avg_confidence = 0.0
                
                # 清理临时文件
                try:
                    time.sleep(0.1)
                    os.unlink(tmp_file.name)
                except:
                    pass
                
                # 处理识别结果
                if text and len(text) > 2 and avg_confidence > 0.3:
                    print(f"💬 识别: '{text}' (置信度: {avg_confidence:.2f})")
                    
                    # 优化命令
                    optimized_command = self.optimize_command(text)
                    if optimized_command != text:
                        print(f"🔧 优化为: '{optimized_command}'")
                    
                    # 发送到Claude Code
                    self.send_to_claude_direct(optimized_command)
                else:
                    print(f"⚠️ 识别质量较低: '{text}' (置信度: {avg_confidence:.2f})")
                    
        except Exception as e:
            print(f"❌ 音频处理失败: {e}")
            
    def optimize_command(self, text):
        """优化语音识别的命令文本"""
        # 常见的语音识别优化规则
        optimizations = {
            # 标点符号修正
            "创建一个文件": "创建文件",
            "帮我写": "写",
            "请你": "",
            "你能": "",
            "我想要": "",
            "我需要": "",
            
            # 命令规范化
            "新建文件": "创建文件",
            "新建": "创建",
            "建立": "创建",
            "制作": "创建",
            "生成": "创建",
            
            # 分析相关
            "看看": "查看",
            "检查一下": "检查",
            "分析一下": "分析",
            
            # 编程相关
            "写代码": "写一个",
            "编程": "编写代码",
            "代码实现": "实现",
        }
        
        optimized = text
        for old, new in optimizations.items():
            optimized = optimized.replace(old, new)
            
        # 移除多余的空格和标点
        optimized = optimized.strip(" ,，。！？")
        
        return optimized
        
    def send_to_claude_direct(self, command):
        """直接发送命令到Claude Code"""
        try:
            print(f"📤 发送到Claude Code: '{command}'")
            
            # 使用claude命令行直接执行
            process = subprocess.Popen(
                ['claude'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 发送命令并获取响应
            stdout, stderr = process.communicate(input=command, timeout=30)
            
            if stdout:
                print("📨 Claude响应:")
                print("-" * 40)
                # 只显示前几行避免输出过长
                lines = stdout.strip().split('\\n')
                for line in lines[:10]:  # 只显示前10行
                    print(line)
                if len(lines) > 10:
                    print(f"... (还有{len(lines)-10}行)")
                print("-" * 40)
            
            if stderr:
                print(f"⚠️ 警告: {stderr}")
            
            # 更新统计
            self.stats['commands_sent'] += 1
            self.stats['last_command_time'] = datetime.now()
            
        except subprocess.TimeoutExpired:
            print("⏰ Claude响应超时")
            try:
                process.kill()
            except:
                pass
        except Exception as e:
            print(f"❌ 发送命令失败: {e}")
            
    async def run(self):
        """运行主程序"""
        self.print_banner()
        
        # 检查依赖
        if not self.setup_dependencies():
            return
        
        # 初始化组件
        if not self.init_whisper():
            return
            
        if not self.setup_hotkeys():
            return
        
        print("✅ 系统初始化完成!")
        print("\\n🎯 按 F1 开始语音监听，F2 手动录音，F3 查看状态，ESC 退出")
        print("💡 或直接输入文本命令测试...")
        
        # 主循环
        try:
            while True:
                user_input = input("\\n> ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    break
                elif user_input.lower() == 'status':
                    self.show_status()
                elif user_input.lower() == 'help':
                    print("📖 可用命令:")
                    print("  F1/f1     - 开始/停止语音监听")
                    print("  F2/f2     - 手动录音")
                    print("  F3/status - 查看状态")
                    print("  help      - 显示帮助")
                    print("  quit      - 退出程序")
                elif user_input:
                    # 直接发送文本命令到Claude
                    self.send_to_claude_direct(user_input)
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()


async def main():
    """主函数"""
    bridge = ClaudeVoiceBridge()
    await bridge.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\n程序退出")
    except Exception as e:
        print(f"启动错误: {e}")
        import traceback
        traceback.print_exc()