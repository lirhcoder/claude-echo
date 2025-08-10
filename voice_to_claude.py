#!/usr/bin/env python3
"""
语音到Claude Code命令桥接器
Voice-to-Claude Code Command Bridge

将语音识别结果直接作为命令输入到Claude Code CLI
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
from pathlib import Path
from datetime import datetime
import json

# 设置路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

class VoiceToClaudeCommand:
    """语音到Claude Code命令的桥接器"""
    
    def __init__(self):
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.command_history = []
        self.claude_process = None
        
    def print_banner(self):
        """显示启动横幅"""
        print("=" * 70)
        print("🎤 CLAUDE CODE 语音命令桥接器".center(70))
        print("=" * 70)
        print()
        print("📝 功能说明:")
        print("  - 语音识别后直接发送到Claude Code CLI")
        print("  - 支持实时语音命令执行")
        print("  - 保持Claude Code会话状态")
        print()
        print("🎯 使用方法:")
        print("  1. 按 'r' + Enter 开始录音")
        print("  2. 清晰说出命令，如：'创建一个Python文件'")
        print("  3. 系统自动识别并发送到Claude Code")
        print("  4. 输入 'quit' 退出程序")
        print("-" * 70)
        print()
        
    def check_dependencies(self):
        """检查必要依赖"""
        print("🔍 检查系统依赖...")
        
        try:
            import whisper
            print("   ✅ Whisper - 已安装")
        except ImportError:
            print("   ❌ Whisper - 未安装")
            print("请运行: pip install openai-whisper")
            return False
            
        try:
            import pyaudio
            print("   ✅ PyAudio - 已安装")
        except ImportError:
            print("   ❌ PyAudio - 未安装")
            print("请运行: pip install pyaudio")
            return False
            
        # 检查Claude Code CLI是否可用
        try:
            result = subprocess.run(['claude', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("   ✅ Claude Code CLI - 已安装")
                return True
            else:
                print("   ❌ Claude Code CLI - 未找到")
                print("请确保Claude Code CLI已安装并在PATH中")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("   ❌ Claude Code CLI - 未找到")
            print("请确保Claude Code CLI已安装并在PATH中")
            return False
            
    def init_whisper(self):
        """初始化Whisper模型"""
        print("🧠 初始化语音识别模型...")
        try:
            import whisper
            self.model = whisper.load_model("base")
            print("✅ Whisper模型加载成功")
            return True
        except Exception as e:
            print(f"❌ Whisper模型加载失败: {e}")
            return False
            
    def record_audio(self, duration=5):
        """录音功能"""
        try:
            import pyaudio
            
            # 音频参数
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 16000
            
            audio = pyaudio.PyAudio()
            
            print(f"🔴 开始录音 ({duration}秒)...")
            print("请开始说话...")
            
            # 开始录音
            stream = audio.open(format=FORMAT,
                              channels=CHANNELS,
                              rate=RATE,
                              input=True,
                              frames_per_buffer=CHUNK)
            
            frames = []
            max_volume = 0
            
            for i in range(0, int(RATE / CHUNK * duration)):
                data = stream.read(CHUNK)
                frames.append(data)
                
                # 显示音量
                import struct
                audio_data = struct.unpack(f"{CHUNK}h", data)
                volume = max(abs(x) for x in audio_data)
                max_volume = max(max_volume, volume)
                
                # 显示进度
                progress = (i + 1) / (int(RATE / CHUNK * duration))
                volume_bar = '|' * min(10, volume // 1000)
                print(f"\r🔴 录音进度: {'=' * int(progress * 20)} {progress * 100:.0f}% 音量:{volume_bar:<10}", end="")
            
            print(f"\n⏹️ 录音完成 (最大音量: {max_volume})")
            
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
            return frames, RATE, CHANNELS, FORMAT, audio.get_sample_size(FORMAT)
            
        except Exception as e:
            print(f"❌ 录音失败: {e}")
            return None
            
    def transcribe_audio(self, frames, rate, channels, format, sample_width):
        """语音识别"""
        try:
            # 保存为临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                wf = wave.open(tmp_file.name, 'wb')
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(rate)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                print("🧠 正在识别语音...")
                
                # 使用Whisper识别
                result = self.model.transcribe(
                    tmp_file.name,
                    language="zh",
                    fp16=False,
                    verbose=False,
                    temperature=0.0,
                    no_speech_threshold=0.6
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
                    time.sleep(0.5)
                    os.unlink(tmp_file.name)
                except:
                    pass
                
                return text, avg_confidence
                
        except Exception as e:
            print(f"❌ 语音识别失败: {e}")
            return None, 0.0
            
    def send_to_claude_code(self, command):
        """将命令发送到Claude Code CLI"""
        try:
            print(f"📤 发送命令到Claude Code: '{command}'")
            
            # 启动Claude Code进程并发送命令
            process = subprocess.Popen(
                ['claude'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 发送命令
            process.stdin.write(command + '\n')
            process.stdin.flush()
            
            # 设置超时读取响应
            import threading
            import queue
            
            def read_output(pipe, q):
                try:
                    while True:
                        line = pipe.readline()
                        if not line:
                            break
                        q.put(('stdout', line.rstrip()))
                except:
                    pass
            
            def read_error(pipe, q):
                try:
                    while True:
                        line = pipe.readline()
                        if not line:
                            break
                        q.put(('stderr', line.rstrip()))
                except:
                    pass
            
            output_queue = queue.Queue()
            stdout_thread = threading.Thread(target=read_output, args=(process.stdout, output_queue))
            stderr_thread = threading.Thread(target=read_error, args=(process.stderr, output_queue))
            
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            
            # 收集输出
            print("📨 Claude Code响应:")
            print("-" * 50)
            
            response_lines = []
            timeout = 10  # 10秒超时
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    msg_type, line = output_queue.get(timeout=1)
                    print(line)
                    response_lines.append(line)
                    
                    # 如果看到典型的Claude Code结束标志，提前结束
                    if "🤖" in line or line.strip() == "" and len(response_lines) > 1:
                        break
                        
                except queue.Empty:
                    if not stdout_thread.is_alive() and not stderr_thread.is_alive():
                        break
                    continue
            
            print("-" * 50)
            
            # 记录命令历史
            self.command_history.append({
                'timestamp': datetime.now(),
                'command': command,
                'response_lines': len(response_lines)
            })
            
            # 清理进程
            try:
                process.stdin.close()
                process.terminate()
                process.wait(timeout=3)
            except:
                try:
                    process.kill()
                except:
                    pass
            
            return True
            
        except Exception as e:
            print(f"❌ 发送到Claude Code失败: {e}")
            return False
            
    def run_voice_command_loop(self):
        """运行语音命令循环"""
        print("✅ 语音命令桥接器已就绪")
        print("\n🎤 语音命令模式:")
        print("  输入 'r' 开始录音")
        print("  输入 'history' 查看命令历史")  
        print("  输入 'quit' 退出")
        
        while True:
            try:
                user_input = input("\n> ").strip().lower()
                
                if user_input == 'quit':
                    break
                    
                elif user_input == 'r':
                    # 录音并识别
                    audio_data = self.record_audio(5)
                    if audio_data:
                        frames, rate, channels, format, sample_width = audio_data
                        text, confidence = self.transcribe_audio(frames, rate, channels, format, sample_width)
                        
                        if text and len(text) > 1 and confidence > 0.3:
                            print(f"\n💬 识别结果: '{text}' (置信度: {confidence:.2f})")
                            
                            # 确认是否发送
                            confirm = input("是否发送到Claude Code? (y/n): ").strip().lower()
                            if confirm in ['y', 'yes', '是', '']:
                                self.send_to_claude_code(text)
                            else:
                                print("❌ 已取消发送")
                        else:
                            print("❌ 识别质量较低或为空，请重试")
                            print("建议: 在安静环境中，清晰地说出完整命令")
                    
                elif user_input == 'history':
                    # 显示命令历史
                    if self.command_history:
                        print("\n📋 命令历史:")
                        for i, cmd in enumerate(self.command_history[-5:], 1):
                            time_str = cmd['timestamp'].strftime('%H:%M:%S')
                            print(f"  {i}. [{time_str}] {cmd['command']} ({cmd['response_lines']}行响应)")
                    else:
                        print("📋 暂无命令历史")
                        
                elif user_input == 'help':
                    print("\n📖 可用命令:")
                    print("  r        - 开始录音并识别语音命令")
                    print("  history  - 查看最近的命令历史")
                    print("  help     - 显示此帮助信息")
                    print("  quit     - 退出程序")
                    
                elif user_input != '':
                    print("❓ 未知命令，输入 'help' 查看可用命令")
                    
            except KeyboardInterrupt:
                print("\n\n👋 程序被中断")
                break
            except Exception as e:
                print(f"❌ 程序错误: {e}")
        
    async def run(self):
        """主运行函数"""
        self.print_banner()
        
        # 检查依赖
        if not self.check_dependencies():
            print("\n❌ 依赖检查失败，请安装缺失的组件")
            return
            
        # 初始化Whisper
        if not self.init_whisper():
            print("\n❌ 语音识别初始化失败")
            return
            
        # 运行主循环
        self.run_voice_command_loop()
        
        print("\n👋 感谢使用语音到Claude Code命令桥接器!")


async def main():
    """主函数"""
    bridge = VoiceToClaudeCommand()
    await bridge.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序退出")
    except Exception as e:
        print(f"启动错误: {e}")
        import traceback
        traceback.print_exc()