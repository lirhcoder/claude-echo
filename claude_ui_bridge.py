#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claude Code UI桥接器
独立UI界面与外部Claude Code进程交互
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue
import subprocess
import tempfile
import os
import time
import json
from datetime import datetime
import sys
from pathlib import Path

# 添加语音识别支持
try:
    import whisper
    import pyaudio
    import wave
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

class ClaudeUIBridge:
    """Claude Code UI桥接器"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.setup_ui()
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.voice_model = None
        self.is_recording = False
        self.session_history = []
        
        # 初始化语音模块
        if VOICE_AVAILABLE:
            self.init_voice()
        
    def setup_ui(self):
        """设置UI界面"""
        self.root.title("Claude Code 语音UI桥接器")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # 标题
        title_label = tk.Label(main_frame, text="🎤 Claude Code 语音UI桥接器", 
                              font=("Arial", 16, "bold"), bg='#f0f0f0')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # 语音控制按钮
        voice_frame = ttk.LabelFrame(control_frame, text="语音控制", padding="5")
        voice_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.record_btn = ttk.Button(voice_frame, text="🎤 开始录音", 
                                   command=self.toggle_recording)
        self.record_btn.pack(fill=tk.X, pady=2)
        
        self.voice_status = tk.Label(voice_frame, text="准备就绪", 
                                   fg="green", bg='#f0f0f0')
        self.voice_status.pack(pady=2)
        
        # 快捷命令
        quick_frame = ttk.LabelFrame(control_frame, text="快捷命令", padding="5")
        quick_frame.pack(fill=tk.X, pady=(0, 10))
        
        quick_commands = [
            ("帮助", "help"),
            ("重置会话", "/reset"),
            ("当前目录", "pwd"),
            ("列出文件", "ls")
        ]
        
        for text, cmd in quick_commands:
            btn = ttk.Button(quick_frame, text=text, 
                           command=lambda c=cmd: self.send_quick_command(c))
            btn.pack(fill=tk.X, pady=1)
        
        # 会话管理
        session_frame = ttk.LabelFrame(control_frame, text="会话管理", padding="5")
        session_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(session_frame, text="保存会话", 
                  command=self.save_session).pack(fill=tk.X, pady=1)
        ttk.Button(session_frame, text="加载会话", 
                  command=self.load_session).pack(fill=tk.X, pady=1)
        ttk.Button(session_frame, text="清空历史", 
                  command=self.clear_history).pack(fill=tk.X, pady=1)
        
        # 设置
        settings_frame = ttk.LabelFrame(control_frame, text="设置", padding="5")
        settings_frame.pack(fill=tk.X)
        
        # Claude进程设置
        ttk.Label(settings_frame, text="Claude进程状态:").pack(anchor=tk.W)
        self.claude_status = tk.Label(settings_frame, text="未连接", 
                                    fg="red", bg='#f0f0f0')
        self.claude_status.pack(anchor=tk.W)
        
        ttk.Button(settings_frame, text="测试连接", 
                  command=self.test_claude_connection).pack(fill=tk.X, pady=2)
        
        # 主对话区域
        chat_frame = ttk.Frame(main_frame)
        chat_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        # 对话显示区
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, 
            wrap=tk.WORD, 
            height=30, 
            width=60,
            font=("Consolas", 10),
            bg='#ffffff',
            fg='#000000'
        )
        self.chat_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), 
                              pady=(0, 10))
        
        # 输入区域
        input_frame = ttk.Frame(chat_frame)
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        
        self.input_text = tk.Text(input_frame, height=3, wrap=tk.WORD,
                                font=("Consolas", 10))
        self.input_text.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        # 发送按钮
        send_btn = ttk.Button(input_frame, text="发送", command=self.send_message)
        send_btn.grid(row=0, column=1)
        
        # 绑定回车键
        self.input_text.bind('<Control-Return>', lambda e: self.send_message())
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), 
                       pady=(10, 0))
        
        # 添加初始消息
        self.add_message("系统", "Claude Code UI桥接器已启动", "system")
        self.add_message("系统", "您可以通过语音或文本与Claude Code交互", "system")
        
    def init_voice(self):
        """初始化语音识别"""
        try:
            self.add_message("系统", "正在加载语音识别模型...", "system")
            self.voice_model = whisper.load_model("base")
            self.add_message("系统", "语音识别模型加载成功", "system")
        except Exception as e:
            self.add_message("系统", f"语音识别初始化失败: {e}", "error")
            
    def add_message(self, sender, message, msg_type="user"):
        """添加消息到对话区"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # 设置颜色
        colors = {
            "user": "#0066cc",
            "assistant": "#009900", 
            "system": "#666666",
            "error": "#cc0000",
            "voice": "#9900cc"
        }
        
        self.chat_display.config(state=tk.NORMAL)
        
        # 添加时间戳和发送者
        self.chat_display.insert(tk.END, f"[{timestamp}] {sender}: ", "sender")
        self.chat_display.tag_config("sender", foreground=colors.get(msg_type, "#000000"), 
                                   font=("Consolas", 10, "bold"))
        
        # 添加消息内容
        self.chat_display.insert(tk.END, f"{message}\n\n", "message")
        self.chat_display.tag_config("message", foreground="#000000")
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
        
        # 添加到历史记录
        self.session_history.append({
            "timestamp": timestamp,
            "sender": sender,
            "message": message,
            "type": msg_type
        })
        
    def toggle_recording(self):
        """切换录音状态"""
        if not VOICE_AVAILABLE:
            messagebox.showerror("错误", "语音功能不可用，请安装: pip install openai-whisper pyaudio")
            return
            
        if not self.voice_model:
            messagebox.showerror("错误", "语音模型未加载")
            return
            
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        """开始录音"""
        self.is_recording = True
        self.record_btn.config(text="⏹️ 停止录音")
        self.voice_status.config(text="录音中...", fg="red")
        self.status_var.set("正在录音...")
        
        # 在新线程中录音
        threading.Thread(target=self.record_voice, daemon=True).start()
        
    def stop_recording(self):
        """停止录音"""
        self.is_recording = False
        self.record_btn.config(text="🎤 开始录音")
        self.voice_status.config(text="准备就绪", fg="green")
        self.status_var.set("就绪")
        
    def record_voice(self):
        """录音功能"""
        try:
            # 音频参数
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 16000
            RECORD_SECONDS = 5
            
            audio = pyaudio.PyAudio()
            
            # 开始录音
            stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                              input=True, frames_per_buffer=CHUNK)
            
            frames = []
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                if not self.is_recording:
                    break
                data = stream.read(CHUNK)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
            if frames:
                self.process_voice(frames, RATE, CHANNELS, FORMAT, audio.get_sample_size(FORMAT))
                
        except Exception as e:
            self.add_message("系统", f"录音失败: {e}", "error")
        finally:
            self.stop_recording()
            
    def process_voice(self, frames, rate, channels, format, sample_width):
        """处理语音识别"""
        try:
            self.status_var.set("正在识别语音...")
            
            # 保存为临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                wf = wave.open(tmp_file.name, 'wb')
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(rate)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                # 语音识别
                result = self.voice_model.transcribe(
                    tmp_file.name,
                    language="zh",
                    fp16=False,
                    verbose=False,
                    temperature=0.0,
                    initial_prompt="这是一个编程相关的语音命令"
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
                    os.unlink(tmp_file.name)
                except:
                    pass
                
                if text and len(text) > 1 and avg_confidence > 0.3:
                    self.add_message("语音输入", f"{text} (置信度: {avg_confidence:.2f})", "voice")
                    # 自动发送语音识别结果
                    self.send_to_claude(text)
                else:
                    self.add_message("系统", f"识别质量较低: '{text}' (置信度: {avg_confidence:.2f})", "error")
                    
        except Exception as e:
            self.add_message("系统", f"语音识别失败: {e}", "error")
        finally:
            self.status_var.set("就绪")
            
    def send_message(self):
        """发送文本消息"""
        message = self.input_text.get("1.0", tk.END).strip()
        if message:
            self.add_message("用户", message, "user")
            self.input_text.delete("1.0", tk.END)
            self.send_to_claude(message)
            
    def send_quick_command(self, command):
        """发送快捷命令"""
        self.add_message("快捷命令", command, "user")
        self.send_to_claude(command)
        
    def send_to_claude(self, message):
        """发送消息到Claude Code"""
        try:
            self.status_var.set("正在发送到Claude Code...")
            
            # 创建临时文件
            temp_file = f"claude_ui_cmd_{int(time.time())}.txt"
            
            try:
                # 写入命令到临时文件
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(message)
                
                # 使用文件重定向方式调用Claude
                cmd = f'claude < {temp_file}'
                
                result = subprocess.run(cmd, capture_output=True, text=True, 
                                      shell=True, timeout=30, encoding='utf-8', errors='ignore')
                
                if result.stdout:
                    response = result.stdout.strip()
                    self.add_message("Claude", response, "assistant")
                    self.claude_status.config(text="已连接", fg="green")
                else:
                    self.add_message("系统", "Claude无响应", "error")
                    self.claude_status.config(text="无响应", fg="orange")
                
                if result.stderr:
                    self.add_message("系统", f"警告: {result.stderr}", "error")
                    
            finally:
                # 清理临时文件
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except:
                    pass
                    
        except subprocess.TimeoutExpired:
            self.add_message("系统", "Claude响应超时", "error")
            self.claude_status.config(text="超时", fg="orange")
        except Exception as e:
            self.add_message("系统", f"发送失败: {e}", "error")
            self.claude_status.config(text="连接失败", fg="red")
        finally:
            self.status_var.set("就绪")
            
    def test_claude_connection(self):
        """测试Claude Code连接"""
        self.send_to_claude("help")
        
    def save_session(self):
        """保存会话"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"claude_session_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.session_history, f, ensure_ascii=False, indent=2)
            
            self.add_message("系统", f"会话已保存到: {filename}", "system")
            messagebox.showinfo("成功", f"会话已保存到: {filename}")
            
        except Exception as e:
            self.add_message("系统", f"保存会话失败: {e}", "error")
            
    def load_session(self):
        """加载会话"""
        from tkinter import filedialog
        
        try:
            filename = filedialog.askopenfilename(
                title="选择会话文件",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                
                # 清空当前显示
                self.chat_display.config(state=tk.NORMAL)
                self.chat_display.delete("1.0", tk.END)
                self.chat_display.config(state=tk.DISABLED)
                
                # 加载历史消息
                for item in history:
                    self.add_message(item["sender"], item["message"], item["type"])
                
                self.add_message("系统", f"会话已从 {os.path.basename(filename)} 加载", "system")
                
        except Exception as e:
            self.add_message("系统", f"加载会话失败: {e}", "error")
            
    def clear_history(self):
        """清空历史"""
        if messagebox.askyesno("确认", "是否清空所有历史记录？"):
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("1.0", tk.END)
            self.chat_display.config(state=tk.DISABLED)
            self.session_history.clear()
            self.add_message("系统", "历史记录已清空", "system")
            
    def run(self):
        """运行UI"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass

def main():
    """主函数"""
    print("启动 Claude Code UI桥接器...")
    
    # 检查依赖
    if not VOICE_AVAILABLE:
        print("警告: 语音功能不可用，请安装: pip install openai-whisper pyaudio")
    
    # 启动UI
    app = ClaudeUIBridge()
    app.run()

if __name__ == "__main__":
    main()