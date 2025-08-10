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
        
        # 音频参数 - 优化录音质量
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000  # Whisper推荐采样率
        RECORD_SECONDS = 5
        
        print("[提示] 录音质量优化提醒:")
        print("- 确保在相对安静的环境中")
        print("- 距离麦克风20-30cm") 
        print("- 说话声音清晰、语速适中")
        print("- 避免在录音中咳嗽或停顿过长")
        print()
        
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
            print("[提示] 请开始说话...")
            
            # 开始录音
            stream = audio.open(format=FORMAT,
                              channels=CHANNELS,
                              rate=RATE,
                              input=True,
                              frames_per_buffer=CHUNK)
            
            frames = []
            max_volume = 0
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
                
                # 检查音频音量
                import struct
                audio_data = struct.unpack(f"{CHUNK}h", data)
                volume = max(abs(x) for x in audio_data)
                max_volume = max(max_volume, volume)
                
                # 显示录音进度和音量
                progress = (i + 1) * CHUNK / (RATE * RECORD_SECONDS)
                volume_bar = '|' * min(10, volume // 1000)
                print(f"\r录音进度: {'=' * int(progress * 20)} {progress * 100:.0f}% 音量:{volume_bar:<10}", end="")
            
            print(f"\n[完成] 录音结束 (最大音量: {max_volume})")
            
            # 检查录音质量
            if max_volume < 1000:
                print("[警告] 录音音量较低，可能影响识别准确率")
                print("建议: 靠近麦克风或提高说话音量")
            
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
                    # 使用更好的参数进行识别
                    result = model.transcribe(
                        tmp_file.name, 
                        language="zh",  # 指定中文
                        fp16=False,     # 避免FP16警告
                        verbose=False,  # 减少输出
                        temperature=0.0,  # 更稳定的结果
                        no_speech_threshold=0.6,  # 调整静音阈值
                        logprob_threshold=-1.0,   # 改善识别质量
                        compression_ratio_threshold=2.4
                    )
                    text = result["text"].strip()
                    
                    # 计算平均置信度
                    if result.get("segments"):
                        confidences = [seg.get("no_speech_prob", 1.0) for seg in result["segments"]]
                        avg_confidence = 1.0 - sum(confidences) / len(confidences) if confidences else 0.0
                    else:
                        avg_confidence = 0.0
                    
                    print(f"\n识别结果: '{text}'")
                    print(f"置信度: {avg_confidence:.2f}")
                    
                    # 检查是否是有效的识别结果
                    if text and len(text) > 1 and avg_confidence > 0.3:
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
                        print("语音识别质量较低或为空，请在安静环境中重试")
                        print("建议: 1) 距离麦克风近一些 2) 说话清晰一些 3) 减少背景噪音")
                        
                except Exception as e:
                    print(f"识别失败: {e}")
                
                # 安全地清理临时文件
                try:
                    import time
                    time.sleep(0.5)  # 等待文件释放
                    os.unlink(tmp_file.name)
                except Exception as e:
                    print(f"[警告] 清理临时文件失败: {e}")
                    # 不影响继续测试
        
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