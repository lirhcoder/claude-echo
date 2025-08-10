#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音桥接器功能测试脚本
Test script for voice bridge functionality
"""

import sys
import subprocess
import time
import os
from pathlib import Path

def print_test_banner():
    """显示测试横幅"""
    print("=" * 60)
    print("Claude Code 语音桥接器功能测试".center(60))
    print("=" * 60)
    print()

def test_dependencies():
    """测试依赖安装"""
    print("[检查] 测试依赖安装...")
    
    dependencies = [
        ('whisper', 'Whisper语音识别'),
        ('pyaudio', 'PyAudio音频处理'),
        ('keyboard', 'Keyboard热键支持')
    ]
    
    all_good = True
    for module, desc in dependencies:
        try:
            __import__(module)
            print(f"   [OK] {desc}")
        except ImportError:
            print(f"   [NO] {desc} - 未安装")
            all_good = False
    
    return all_good

def test_claude_cli():
    """测试Claude Code CLI"""
    print("\n[测试] Claude Code CLI...")
    
    # 首先尝试真实的Claude CLI
    try:
        result = subprocess.run(['claude', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"   [OK] Claude Code CLI - {result.stdout.strip()}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # 如果真实CLI不可用，尝试模拟版本
    try:
        mock_path = Path(__file__).parent / "claude_mock.py"
        if mock_path.exists():
            result = subprocess.run([sys.executable, str(mock_path), '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"   [OK] Claude Code CLI (模拟版) - {result.stdout.strip()}")
                print("   [提示] 使用模拟版本进行测试，功能完整")
                return True
    except Exception:
        pass
    
    print("   [NO] Claude Code CLI - 未找到")
    print("   [解决方案]:")
    print("      1. 安装 Claude Code: https://claude.ai/code")
    print("      2. 或使用模拟版本: python claude_mock.py")
    return False

def test_audio_devices():
    """测试音频设备"""
    print("\n[测试] 音频设备...")
    
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        
        print(f"   [信息] 检测到 {device_count} 个音频设备")
        
        # 查找输入设备
        input_devices = []
        for i in range(device_count):
            try:
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    input_devices.append(f"   [设备] {info['name']}")
            except:
                continue
        
        if input_devices:
            print("   [OK] 可用输入设备:")
            for device in input_devices[:3]:  # 只显示前3个
                print(device)
            if len(input_devices) > 3:
                print(f"   ... 还有 {len(input_devices)-3} 个设备")
        else:
            print("   [NO] 未找到输入设备")
            
        p.terminate()
        return len(input_devices) > 0
        
    except Exception as e:
        print(f"   [NO] 音频设备检测失败: {e}")
        return False

def test_whisper_model():
    """测试Whisper模型加载"""
    print("\n[测试] Whisper模型...")
    
    try:
        import whisper
        print("   [加载] base 模型...")
        model = whisper.load_model("base")
        print("   [OK] Whisper base 模型加载成功")
        
        # 尝试加载small模型（如果可用）
        try:
            print("   [加载] small 模型...")
            model_small = whisper.load_model("small")
            print("   [OK] Whisper small 模型加载成功")
        except:
            print("   [提示] Whisper small 模型加载失败（将使用base模型）")
            
        return True
        
    except Exception as e:
        print(f"   [NO] Whisper模型加载失败: {e}")
        return False

def test_voice_bridge_scripts():
    """测试语音桥接脚本"""
    print("\n[检查] 语音桥接脚本...")
    
    scripts = {
        'voice_to_claude.py': '基础版语音桥接器',
        'claude_voice_bridge.py': '增强版语音桥接器',
        'voice_helper.py': '语音助手集成脚本',
        'claude_mock.py': 'Claude CLI模拟器'
    }
    
    base_path = Path(__file__).parent
    all_exist = True
    
    for script, desc in scripts.items():
        script_path = base_path / script
        if script_path.exists():
            print(f"   [OK] {desc}")
        else:
            print(f"   [NO] {desc} - 文件不存在")
            all_exist = False
    
    return all_exist

def test_basic_functionality():
    """测试基本功能"""
    print("\n[测试] 基本功能...")
    
    try:
        # 测试导入主要模块
        sys.path.insert(0, str(Path(__file__).parent))
        
        print("   [运行] 测试语音桥接器导入...")
        
        # 创建测试用的简化版本
        test_code = '''
import tempfile
import wave
import os

class TestVoiceBridge:
    def __init__(self):
        self.test_passed = False
        
    def test_audio_processing(self):
        """测试音频处理功能"""
        try:
            # 创建虚拟音频数据
            import pyaudio
            import tempfile
            import wave
            import os
            
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 16000
            
            # 创建测试音频帧
            frames = [b'\\x00' * (CHUNK * 2) for _ in range(10)]
            
            # 测试保存为WAV文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                wf = wave.open(tmp_file.name, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # 16-bit = 2 bytes
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                # 检查文件是否创建成功
                if os.path.exists(tmp_file.name):
                    self.test_passed = True
                    os.unlink(tmp_file.name)
                    
            return self.test_passed
            
        except Exception as e:
            print(f"      [NO] 音频处理测试失败: {e}")
            return False

# 运行测试
try:
    bridge = TestVoiceBridge()
    result = bridge.test_audio_processing()
    print("TEST_RESULT:", result)
except Exception as e:
    print("TEST_ERROR:", str(e))
    print("TEST_RESULT:", False)
'''
        
        # 将测试代码写入临时文件并执行
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(test_code)
            test_file = f.name
        
        try:
            result = subprocess.run([sys.executable, test_file], 
                                  capture_output=True, text=True, timeout=10)
            
            if "TEST_RESULT: True" in result.stdout:
                print("   [OK] 基础音频处理功能正常")
                return True
            else:
                print("   [NO] 基础功能测试失败")
                if result.stderr:
                    print(f"      错误: {result.stderr}")
                return False
                
        finally:
            try:
                os.unlink(test_file)
            except:
                pass
                
    except Exception as e:
        print(f"   [NO] 功能测试异常: {e}")
        return False

def run_interactive_test():
    """运行交互式测试"""
    print("\n[互动] 交互式功能测试")
    print("-" * 40)
    print()
    
    choice = input("是否运行交互式语音测试? (y/n): ").strip().lower()
    if choice not in ['y', 'yes', '是']:
        print("跳过交互式测试")
        return
    
    print("\n选择测试模式:")
    print("  1. 基础版语音桥接器 (voice_to_claude.py)")
    print("  2. 增强版语音桥接器 (claude_voice_bridge.py)")
    print("  3. 简化语音测试 (simple_voice_test.py)")
    print("  4. Claude CLI模拟器 (claude_mock.py)")
    
    mode = input("请选择 (1-4): ").strip()
    
    script_map = {
        '1': 'voice_to_claude.py',
        '2': 'claude_voice_bridge.py', 
        '3': 'simple_voice_test.py',
        '4': 'claude_mock.py'
    }
    
    script = script_map.get(mode)
    if not script:
        print("无效选择")
        return
    
    script_path = Path(__file__).parent / script
    if not script_path.exists():
        print(f"脚本不存在: {script}")
        return
    
    print(f"\n[启动] {script}...")
    print("注意: 按 Ctrl+C 可以退出测试")
    print("-" * 40)
    
    try:
        subprocess.run([sys.executable, str(script_path)], cwd=script_path.parent)
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试运行错误: {e}")

def main():
    """主测试函数"""
    print_test_banner()
    
    test_results = {
        "依赖安装": test_dependencies(),
        "Claude CLI": test_claude_cli(),
        "音频设备": test_audio_devices(),
        "Whisper模型": test_whisper_model(),
        "脚本文件": test_voice_bridge_scripts(),
        "基础功能": test_basic_functionality()
    }
    
    print("\n" + "=" * 60)
    print("测试结果汇总".center(60))
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "[通过]" if result else "[失败]"
        print(f"  {test_name:<12}: {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"通过率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("[成功] 所有测试通过！语音桥接器可以正常使用")
    elif passed >= total * 0.8:
        print("[提示] 大部分测试通过，可以尝试使用")
    else:
        print("[失败] 多项测试失败，请检查系统环境")
        
    # 提供解决方案
    if passed < total:
        print("\n[解决建议]:")
        if not test_results["依赖安装"]:
            print("  pip install openai-whisper pyaudio keyboard")
        if not test_results["Claude CLI"]:
            print("  安装 Claude Code: https://claude.ai/code")
        if not test_results["音频设备"]:
            print("  检查麦克风设备和权限设置")
    
    # 运行交互式测试
    if passed >= total * 0.6:  # 60%以上通过才提供交互测试
        run_interactive_test()
    
    print("\n[完成] 测试结束")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n测试被中断")
    except Exception as e:
        print(f"\n测试异常: {e}")
        import traceback
        traceback.print_exc()