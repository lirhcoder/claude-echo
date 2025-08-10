#!/usr/bin/env python3
"""
简化版语音测试启动器
避免复杂配置问题，直接测试语音功能
"""

import asyncio
import sys
import os
import time
from pathlib import Path
from datetime import datetime

# 设置路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 50)
print("Claude Echo 简化语音测试")
print("=" * 50)

def test_voice_dependencies():
    """测试语音依赖是否安装正确"""
    print("\n[检查] 测试语音依赖...")
    
    try:
        import whisper
        print("[OK] Whisper 语音识别可用")
    except ImportError as e:
        print(f"[NO] Whisper 不可用: {e}")
        return False
    
    try:
        import pyttsx3
        print("[OK] pyttsx3 语音合成可用")
    except ImportError as e:
        print(f"[NO] pyttsx3 不可用: {e}")
        return False
        
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        print(f"[OK] 音频设备可用: {device_count} 个设备")
        p.terminate()
    except ImportError as e:
        print(f"[NO] pyaudio 不可用: {e}")
        return False
    except Exception as e:
        print(f"[NO] 音频设备检查失败: {e}")
        return False
    
    return True

def test_basic_speech():
    """测试基础语音功能"""
    print("\n[测试] 基础语音功能...")
    
    try:
        # 测试语音合成
        import pyttsx3
        engine = pyttsx3.init()
        print("[OK] 语音合成引擎初始化成功")
        
        # 测试说话
        print("[INFO] 系统将说话测试...")
        engine.say("语音测试成功，Claude Echo 已就绪")
        engine.runAndWait()
        
        print("[OK] 语音合成测试成功")
        
    except Exception as e:
        print(f"[NO] 语音合成测试失败: {e}")
        return False
    
    try:
        # 测试语音识别模型加载
        import whisper
        print("[INFO] 加载 Whisper 模型...")
        model = whisper.load_model("base")
        print("[OK] Whisper 模型加载成功")
        
    except Exception as e:
        print(f"[NO] Whisper 模型加载失败: {e}")
        return False
    
    return True

def interactive_voice_test():
    """交互式语音测试"""
    print("\n[开始] 交互式语音测试")
    print("=" * 50)
    
    try:
        import whisper
        import pyaudio
        import wave
        import tempfile
        import os
        
        # 加载模型
        print("加载语音识别模型...")
        model = whisper.load_model("base")
        
        # 音频参数
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        RECORD_SECONDS = 5
        
        audio = pyaudio.PyAudio()
        
        print("\n" + "=" * 50)
        print("[准备] 语音测试已就绪")
        print("=" * 50)
        print("说明:")
        print("1. 按 Enter 开始录音 (5秒)")
        print("2. 清晰地说出命令，如:")
        print("   - '你好 Claude'")
        print("   - 'Hello Claude'") 
        print("   - '帮我创建一个文件'")
        print("3. 输入 'quit' 退出测试")
        print("-" * 50)
        
        while True:
            user_input = input("\n按 Enter 开始录音，或输入 'quit' 退出: ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            print("\n[录音] 开始录音... (5秒)")
            
            # 开始录音
            stream = audio.open(format=FORMAT,
                              channels=CHANNELS,
                              rate=RATE,
                              input=True,
                              frames_per_buffer=CHUNK)
            
            frames = []
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
                
                # 显示录音进度
                progress = (i + 1) * CHUNK / (RATE * RECORD_SECONDS)
                print(f"\r录音进度: {'=' * int(progress * 20)} {progress * 100:.0f}%", end="")
            
            print("\n[完成] 录音结束")
            
            stream.stop_stream()
            stream.close()
            
            # 保存音频文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                wf = wave.open(tmp_file.name, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                # 语音识别
                print("[识别] 正在识别语音...")
                try:
                    result = model.transcribe(tmp_file.name, language="zh")
                    text = result["text"].strip()
                    confidence = result.get("segments", [{}])[0].get("confidence", 0.0) if result.get("segments") else 0.0
                    
                    print(f"\n识别结果: '{text}'")
                    print(f"置信度: {confidence:.2f}")
                    
                    if text:
                        # 简单的命令响应
                        response = process_voice_command(text)
                        print(f"系统响应: {response}")
                        
                        # 语音回复
                        try:
                            import pyttsx3
                            engine = pyttsx3.init()
                            engine.say(response)
                            engine.runAndWait()
                        except:
                            pass
                    else:
                        print("未识别到语音内容，请重试")
                        
                except Exception as e:
                    print(f"识别失败: {e}")
                
                # 清理临时文件
                os.unlink(tmp_file.name)
        
        audio.terminate()
        print("\n语音测试结束")
        
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n语音测试错误: {e}")

def process_voice_command(text):
    """处理语音命令"""
    text = text.lower()
    
    if "你好" in text or "hello" in text:
        return "你好！我是Claude语音助手，很高兴为您服务。"
    elif "时间" in text or "time" in text:
        return f"现在的时间是 {datetime.now().strftime('%H点%M分')}"
    elif "创建" in text and "文件" in text:
        return "好的，我可以帮您创建文件。请告诉我文件名和内容。"
    elif "状态" in text or "status" in text:
        return "系统运行正常，语音识别功能已激活。"
    elif "退出" in text or "quit" in text or "bye" in text:
        return "再见！感谢使用Claude语音助手。"
    else:
        return f"我听到您说：{text}。这是一个测试响应。"

async def main():
    """主函数"""
    print("正在初始化语音测试环境...")
    
    # 检查依赖
    if not test_voice_dependencies():
        print("\n[NO] 语音依赖检查失败")
        print("请运行: install_voice_deps.bat")
        return
    
    # 测试基础功能
    if not test_basic_speech():
        print("\n[NO] 基础语音功能测试失败")
        return
    
    print("\n[OK] 所有检查通过，可以开始语音测试")
    
    # 开始交互测试
    interactive_voice_test()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序异常: {e}")
        import traceback
        traceback.print_exc()