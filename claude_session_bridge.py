#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claude Code 会话桥接器
与正在运行的Claude Code进程实时交互
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import queue
import subprocess
import tempfile
import os
import time
import json
import signal
import psutil
from datetime import datetime
import sys
from pathlib import Path

# 语音支持
try:
    import whisper
    import pyaudio
    import wave
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

class ClaudeSessionBridge:
    """Claude Code 会话桥接器"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.setup_ui()
        
        # 会话管理
        self.claude_processes = []
        self.active_process = None
        self.session_file = None
        self.communication_dir = "claude_bridge_comm"
        
        # 语音
        self.voice_model = None
        self.is_recording = False
        
        # 通信队列
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # 初始化
        self.init_communication()
        if VOICE_AVAILABLE:
            self.init_voice()
        self.scan_claude_processes()
        
    def setup_ui(self):
        """设置UI界面"""
        self.root.title("Claude Code 会话桥接器")
        self.root.geometry("1000x800")
        self.root.configure(bg='#f0f0f0')
        
        # 样式
        style = ttk.Style()
        style.theme_use('clam')
        
        # 主容器
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧面板
        left_panel = ttk.Frame(main_container, width=300)
        main_container.add(left_panel, weight=1)
        
        # 右侧面板  
        right_panel = ttk.Frame(main_container)
        main_container.add(right_panel, weight=3)
        
        self.setup_left_panel(left_panel)
        self.setup_right_panel(right_panel)
        
    def setup_left_panel(self, parent):
        """设置左侧控制面板"""
        # 标题
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(title_frame, text="🎤 Claude 会话桥接", 
                font=("Arial", 14, "bold"), bg='#f0f0f0').pack()
        
        # Claude进程管理
        process_frame = ttk.LabelFrame(parent, text="Claude 进程管理", padding="10")
        process_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 进程列表
        ttk.Label(process_frame, text="检测到的Claude进程:").pack(anchor=tk.W)
        
        self.process_listbox = tk.Listbox(process_frame, height=4)
        self.process_listbox.pack(fill=tk.X, pady=5)
        self.process_listbox.bind('<<ListboxSelect>>', self.on_process_select)
        
        # 进程控制按钮
        process_btn_frame = ttk.Frame(process_frame)
        process_btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(process_btn_frame, text="刷新进程", 
                  command=self.scan_claude_processes).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(process_btn_frame, text="连接进程", 
                  command=self.connect_process).pack(side=tk.LEFT)
        
        # 连接状态
        self.connection_status = tk.Label(process_frame, text="未连接", 
                                        fg="red", bg='#f0f0f0')
        self.connection_status.pack(pady=5)
        
        # 语音控制
        voice_frame = ttk.LabelFrame(parent, text="语音控制", padding="10")
        voice_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.record_btn = ttk.Button(voice_frame, text="🎤 开始语音", 
                                   command=self.toggle_recording)
        self.record_btn.pack(fill=tk.X, pady=2)
        
        self.voice_status = tk.Label(voice_frame, text="语音准备就绪", 
                                   fg="green", bg='#f0f0f0')
        self.voice_status.pack(pady=2)
        
        # 快速命令
        quick_frame = ttk.LabelFrame(parent, text="快速命令", padding="10")
        quick_frame.pack(fill=tk.X, pady=(0, 10))
        
        quick_commands = [
            ("📋 帮助", "help"),
            ("🔄 重置", "/reset"),  
            ("📁 当前目录", "pwd"),
            ("📄 文件列表", "ls"),
            ("⏰ 当前时间", "date")
        ]
        
        for text, cmd in quick_commands:
            ttk.Button(quick_frame, text=text, 
                      command=lambda c=cmd: self.send_command(c)).pack(fill=tk.X, pady=1)
        
        # 会话管理
        session_frame = ttk.LabelFrame(parent, text="会话管理", padding="10")
        session_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(session_frame, text="💾 保存会话", 
                  command=self.save_session).pack(fill=tk.X, pady=1)
        ttk.Button(session_frame, text="📂 加载会话", 
                  command=self.load_session).pack(fill=tk.X, pady=1)
        ttk.Button(session_frame, text="🗑️ 清空对话", 
                  command=self.clear_conversation).pack(fill=tk.X, pady=1)
        
        # 高级设置
        advanced_frame = ttk.LabelFrame(parent, text="高级设置", padding="10")
        advanced_frame.pack(fill=tk.X)
        
        # 自动发送语音结果 - 改为默认关闭，让用户有机会编辑
        self.auto_send_voice = tk.BooleanVar(value=False)
        ttk.Checkbutton(advanced_frame, text="自动发送语音识别结果",
                       variable=self.auto_send_voice).pack(anchor=tk.W)
        
        # 显示详细日志
        self.show_verbose_log = tk.BooleanVar(value=False)
        ttk.Checkbutton(advanced_frame, text="显示详细日志",
                       variable=self.show_verbose_log).pack(anchor=tk.W)
        
    def setup_right_panel(self, parent):
        """设置右侧对话面板"""
        # 对话显示区
        chat_frame = ttk.Frame(parent)
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 对话标题
        chat_header = ttk.Frame(chat_frame)
        chat_header.pack(fill=tk.X, pady=(0, 5))
        
        tk.Label(chat_header, text="💬 与Claude Code的对话", 
                font=("Arial", 12, "bold"), bg='#f0f0f0').pack(side=tk.LEFT)
        
        # 清空按钮
        ttk.Button(chat_header, text="清空", 
                  command=self.clear_conversation).pack(side=tk.RIGHT)
        
        # 对话显示
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            width=70,
            height=35,
            font=("Consolas", 10),
            bg='#ffffff',
            fg='#000000'
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # 配置文本标签
        self.chat_display.tag_config("user", foreground="#0066cc", font=("Consolas", 10, "bold"))
        self.chat_display.tag_config("assistant", foreground="#009900", font=("Consolas", 10))
        self.chat_display.tag_config("system", foreground="#666666", font=("Consolas", 9, "italic"))
        self.chat_display.tag_config("error", foreground="#cc0000", font=("Consolas", 10, "bold"))
        self.chat_display.tag_config("voice", foreground="#9900cc", font=("Consolas", 10, "bold"))
        
        # 输入区
        input_frame = ttk.Frame(parent)
        input_frame.pack(fill=tk.X)
        
        # 输入框
        input_label_frame = ttk.LabelFrame(input_frame, text="输入命令", padding="5")
        input_label_frame.pack(fill=tk.X, pady=(0, 5))
        
        # 文本输入
        text_frame = ttk.Frame(input_label_frame)
        text_frame.pack(fill=tk.X)
        
        self.input_text = tk.Text(text_frame, height=3, wrap=tk.WORD,
                                font=("Consolas", 10), bg='#ffffff')
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # 发送按钮
        send_btn_frame = ttk.Frame(text_frame)
        send_btn_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        ttk.Button(send_btn_frame, text="📤\\n发送", 
                  command=self.send_text_input).pack(fill=tk.BOTH, expand=True)
        
        # 绑定快捷键
        self.input_text.bind('<Control-Return>', lambda e: self.send_text_input())
        self.input_text.bind('<Shift-Return>', lambda e: None)  # 允许Shift+Enter换行
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪 | 使用 Ctrl+Enter 发送消息")
        status_bar = ttk.Label(input_frame, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W, font=("Arial", 9))
        status_bar.pack(fill=tk.X, pady=5)
        
        # 初始欢迎消息
        self.add_chat_message("系统", "Claude Code 会话桥接器已启动", "system")
        self.add_chat_message("系统", "请先扫描并连接到正在运行的Claude Code进程", "system")
        
    def init_communication(self):
        """初始化通信目录"""
        try:
            os.makedirs(self.communication_dir, exist_ok=True)
            self.add_chat_message("系统", f"通信目录已创建: {self.communication_dir}", "system")
        except Exception as e:
            self.add_chat_message("系统", f"创建通信目录失败: {e}", "error")
            
    def init_voice(self):
        """初始化语音识别"""
        def load_voice_model():
            try:
                self.voice_status.config(text="正在加载语音模型...", fg="orange")
                self.voice_model = whisper.load_model("base")
                self.voice_status.config(text="语音模型已加载", fg="green")
                self.add_chat_message("系统", "语音识别模块初始化成功", "system")
            except Exception as e:
                self.voice_status.config(text="语音模型加载失败", fg="red")
                self.add_chat_message("系统", f"语音识别初始化失败: {e}", "error")
        
        # 在后台线程加载模型
        threading.Thread(target=load_voice_model, daemon=True).start()
        
    def scan_claude_processes(self):
        """扫描Claude Code进程"""
        self.process_listbox.delete(0, tk.END)
        self.claude_processes.clear()
        
        try:
            # 查找Claude相关进程
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    process_info = proc.info
                    if process_info['name'] and 'claude' in process_info['name'].lower():
                        # 排除当前脚本进程
                        if process_info['pid'] != os.getpid():
                            self.claude_processes.append(proc)
                            display_text = f"PID:{process_info['pid']} - {process_info['name']}"
                            self.process_listbox.insert(tk.END, display_text)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if not self.claude_processes:
                self.process_listbox.insert(tk.END, "未找到Claude进程")
                self.add_chat_message("系统", "未找到正在运行的Claude Code进程", "system")
                self.add_chat_message("系统", "请在CMD中启动Claude Code后重新扫描", "system")
            else:
                count = len(self.claude_processes)
                self.add_chat_message("系统", f"找到 {count} 个Claude进程", "system")
                
        except Exception as e:
            self.add_chat_message("系统", f"扫描进程失败: {e}", "error")
            
    def on_process_select(self, event):
        """进程选择事件"""
        selection = self.process_listbox.curselection()
        if selection and selection[0] < len(self.claude_processes):
            self.active_process = self.claude_processes[selection[0]]
            pid = self.active_process.pid
            name = self.active_process.name()
            self.add_chat_message("系统", f"已选择进程: PID {pid} - {name}", "system")
            
    def connect_process(self):
        """连接到选定的Claude进程"""
        if not self.active_process:
            messagebox.showwarning("警告", "请先选择一个Claude进程")
            return
            
        try:
            # 检查进程是否仍在运行
            if not self.active_process.is_running():
                messagebox.showerror("错误", "选定的进程已不在运行")
                self.scan_claude_processes()
                return
            
            pid = self.active_process.pid
            self.connection_status.config(text=f"已连接 (PID:{pid})", fg="green")
            self.add_chat_message("系统", f"已连接到Claude进程 PID:{pid}", "system")
            
            # 发送测试命令
            self.send_command("help")
            
        except Exception as e:
            self.connection_status.config(text="连接失败", fg="red")
            self.add_chat_message("系统", f"连接进程失败: {e}", "error")
            
    def toggle_recording(self):
        """切换录音状态"""
        if not VOICE_AVAILABLE:
            messagebox.showerror("错误", "语音功能不可用\\n请安装: pip install openai-whisper pyaudio")
            return
            
        if not self.voice_model:
            messagebox.showerror("错误", "语音模型未加载完成")
            return
            
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
            
    def start_recording(self):
        """开始录音"""
        self.is_recording = True
        self.record_btn.config(text="⏹️ 停止录音")
        self.voice_status.config(text="录音中... (5秒)", fg="red")
        self.status_var.set("正在录音...")
        
        # 录音线程
        threading.Thread(target=self.record_audio, daemon=True).start()
        
    def stop_recording(self):
        """停止录音"""
        self.is_recording = False
        self.record_btn.config(text="🎤 开始语音")
        self.voice_status.config(text="语音准备就绪", fg="green")
        self.status_var.set("就绪")
        
    def record_audio(self):
        """录音处理"""
        try:
            # 录音参数
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
            
            if frames and self.is_recording:
                self.process_voice_recognition(frames, RATE, CHANNELS, FORMAT, 
                                             audio.get_sample_size(FORMAT))
                
        except Exception as e:
            self.add_chat_message("系统", f"录音失败: {e}", "error")
        finally:
            self.stop_recording()
            
    def process_voice_recognition(self, frames, rate, channels, format, sample_width):
        """处理语音识别"""
        try:
            self.status_var.set("正在识别语音...")
            
            # 保存音频
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
                    initial_prompt="这是编程相关的语音命令"
                )
                
                text = result["text"].strip()
                
                # 计算置信度
                avg_confidence = 0.0
                if result.get("segments"):
                    confidences = [1.0 - seg.get("no_speech_prob", 1.0) 
                                 for seg in result["segments"]]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                
                # 清理临时文件
                try:
                    os.unlink(tmp_file.name)
                except:
                    pass
                
                if text and len(text) > 1 and avg_confidence > 0.3:
                    voice_msg = f"{text} (置信度: {avg_confidence:.2f})"
                    self.add_chat_message("语音输入", voice_msg, "voice")
                    
                    # 总是先将文本放入输入框，让用户有机会编辑
                    self.input_text.delete("1.0", tk.END)
                    self.input_text.insert("1.0", text)
                    
                    if self.auto_send_voice.get():
                        self.send_command(text)
                    else:
                        # 提示用户可以编辑后发送
                        self.status_var.set("语音识别完成，请检查后发送")
                else:
                    self.add_chat_message("系统", f"语音识别质量较低: '{text}' (置信度: {avg_confidence:.2f})", "error")
                    
        except Exception as e:
            self.add_chat_message("系统", f"语音识别失败: {e}", "error")
        finally:
            self.status_var.set("就绪")
            
    def send_text_input(self):
        """发送文本输入"""
        text = self.input_text.get("1.0", tk.END).strip()
        if text:
            self.add_chat_message("用户", text, "user")
            self.input_text.delete("1.0", tk.END)
            self.send_command(text)
            
    def send_command(self, command):
        """发送命令到Claude Code"""
        if not self.active_process:
            messagebox.showwarning("警告", "请先连接到Claude进程")
            return
            
        try:
            self.status_var.set("正在发送命令到Claude Code...")
            
            # 创建通信文件
            timestamp = int(time.time() * 1000)
            cmd_file = os.path.join(self.communication_dir, f"cmd_{timestamp}.txt")
            
            # 写入命令
            with open(cmd_file, 'w', encoding='utf-8') as f:
                f.write(command)
            
            # 使用Claude CLI处理
            result = subprocess.run(f'claude < "{cmd_file}"', 
                                  capture_output=True, text=True, shell=True, 
                                  timeout=30, encoding='utf-8', errors='ignore')
            
            # 处理响应
            if result.stdout:
                # 保持换行符，不要用strip()移除所有格式
                response = result.stdout.rstrip()
                self.add_chat_message("Claude", response, "assistant")
                self.connection_status.config(text="通信正常", fg="green")
            else:
                self.add_chat_message("系统", "Claude无响应", "error")
                
            if result.stderr and self.show_verbose_log.get():
                self.add_chat_message("系统", f"调试信息: {result.stderr}", "system")
                
            # 清理通信文件
            try:
                os.remove(cmd_file)
            except:
                pass
                
        except subprocess.TimeoutExpired:
            self.add_chat_message("系统", "Claude响应超时", "error")
            self.connection_status.config(text="响应超时", fg="orange")
        except Exception as e:
            self.add_chat_message("系统", f"发送命令失败: {e}", "error")
            self.connection_status.config(text="通信失败", fg="red")
        finally:
            self.status_var.set("就绪")
            
    def add_chat_message(self, sender, message, msg_type="user"):
        """添加消息到对话区"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        self.chat_display.config(state=tk.NORMAL)
        
        # 添加时间戳和发送者
        header = f"[{timestamp}] {sender}: "
        self.chat_display.insert(tk.END, header, msg_type)
        
        # 添加消息内容，保持原有换行格式
        self.chat_display.insert(tk.END, f"{message}\n\n")
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
        
    def save_session(self):
        """保存会话"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = filedialog.asksaveasfilename(
                title="保存会话",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                initialname=f"claude_session_{timestamp}.txt"
            )
            
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.chat_display.get("1.0", tk.END))
                
                self.add_chat_message("系统", f"会话已保存到: {os.path.basename(filename)}", "system")
                messagebox.showinfo("成功", f"会话已保存到:\\n{filename}")
                
        except Exception as e:
            self.add_chat_message("系统", f"保存会话失败: {e}", "error")
            
    def load_session(self):
        """加载会话"""
        try:
            filename = filedialog.askopenfilename(
                title="加载会话",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 显示在对话框中
                self.chat_display.config(state=tk.NORMAL)
                self.chat_display.insert(tk.END, "\\n--- 加载的会话内容 ---\\n")
                self.chat_display.insert(tk.END, content)
                self.chat_display.insert(tk.END, "--- 会话加载完成 ---\\n\\n")
                self.chat_display.config(state=tk.DISABLED)
                self.chat_display.see(tk.END)
                
                self.add_chat_message("系统", f"会话已从 {os.path.basename(filename)} 加载", "system")
                
        except Exception as e:
            self.add_chat_message("系统", f"加载会话失败: {e}", "error")
            
    def clear_conversation(self):
        """清空对话"""
        if messagebox.askyesno("确认", "是否清空当前对话？"):
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("1.0", tk.END)
            self.chat_display.config(state=tk.DISABLED)
            self.add_chat_message("系统", "对话已清空", "system")
            
    def run(self):
        """运行应用"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
        
    def on_closing(self):
        """关闭事件"""
        if messagebox.askokcancel("退出", "确定要退出吗？"):
            # 清理通信目录
            try:
                import shutil
                if os.path.exists(self.communication_dir):
                    shutil.rmtree(self.communication_dir)
            except:
                pass
            self.root.destroy()

def main():
    """主函数"""
    print("启动 Claude Code 会话桥接器...")
    
    if not VOICE_AVAILABLE:
        print("警告: 语音功能不可用，请安装依赖:")
        print("pip install openai-whisper pyaudio")
    
    try:
        app = ClaudeSessionBridge()
        app.run()
    except Exception as e:
        print(f"启动失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()