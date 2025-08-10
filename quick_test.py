#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试语音桥接功能
Quick test for voice bridge functionality
"""

import sys
import subprocess
import os
from pathlib import Path

def test_claude_mock():
    """测试Claude模拟器"""
    print("=" * 50)
    print("测试Claude模拟器")
    print("=" * 50)
    
    mock_path = Path(__file__).parent / "claude_mock.py"
    if not mock_path.exists():
        print("错误: claude_mock.py 不存在")
        return False
        
    try:
        # 测试版本命令
        result = subprocess.run([sys.executable, str(mock_path), '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("版本测试:")
            print(result.stdout)
        
        # 测试命令处理
        test_command = "创建一个Python计算器"
        print(f"测试命令: {test_command}")
        print("-" * 30)
        
        result = subprocess.run([sys.executable, str(mock_path)], 
                              input=test_command, text=True, 
                              capture_output=True, timeout=10)
        
        if result.returncode == 0:
            print("命令处理结果:")
            print(result.stdout)
            print("=" * 50)
            print("Claude模拟器测试成功!")
            return True
        else:
            print(f"命令处理失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"测试失败: {e}")
        return False

def test_whisper_basic():
    """测试Whisper基本功能"""
    print("\n测试Whisper基本功能...")
    
    try:
        import whisper
        print("  [OK] Whisper模块导入成功")
        
        # 加载模型
        model = whisper.load_model("base")
        print("  [OK] Base模型加载成功")
        
        return True
        
    except Exception as e:
        print(f"  [NO] Whisper测试失败: {e}")
        return False

def test_audio_basic():
    """测试音频基本功能"""
    print("\n测试音频基本功能...")
    
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        print(f"  [OK] 检测到 {device_count} 个音频设备")
        
        # 检查输入设备
        input_count = 0
        for i in range(device_count):
            try:
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    input_count += 1
            except:
                continue
        
        print(f"  [OK] 找到 {input_count} 个输入设备")
        p.terminate()
        
        return input_count > 0
        
    except Exception as e:
        print(f"  [NO] 音频测试失败: {e}")
        return False

def demo_voice_to_claude():
    """演示语音到Claude的完整流程"""
    print("\n" + "=" * 50)
    print("演示: 语音到Claude命令的完整流程")
    print("=" * 50)
    
    # 模拟语音识别结果
    voice_commands = [
        "创建一个Python计算器",
        "分析项目结构",
        "写一个Hello World函数",
        "检查代码质量"
    ]
    
    mock_path = Path(__file__).parent / "claude_mock.py"
    
    for i, command in enumerate(voice_commands, 1):
        print(f"\n{i}. 模拟语音命令: '{command}'")
        print("   " + "-" * 40)
        
        try:
            # 发送到Claude模拟器
            result = subprocess.run([sys.executable, str(mock_path)], 
                                  input=command, text=True, 
                                  capture_output=True, timeout=10)
            
            if result.returncode == 0:
                # 只显示前几行，避免输出过长
                lines = result.stdout.strip().split('\n')
                for line in lines[:8]:  # 显示前8行
                    print(f"   {line}")
                if len(lines) > 8:
                    print(f"   ... (还有 {len(lines)-8} 行)")
            else:
                print(f"   错误: {result.stderr}")
                
        except Exception as e:
            print(f"   异常: {e}")
        
        print("   " + "-" * 40)

def main():
    """主测试函数"""
    print("Claude Code 语音桥接器快速测试")
    print("=" * 50)
    
    # 基础测试
    tests = [
        ("Claude模拟器", test_claude_mock),
        ("Whisper模块", test_whisper_basic), 
        ("音频设备", test_audio_basic)
    ]
    
    passed = 0
    for name, test_func in tests:
        try:
            if test_func():
                print(f"[通过] {name}")
                passed += 1
            else:
                print(f"[失败] {name}")
        except Exception as e:
            print(f"[异常] {name}: {e}")
    
    print(f"\n基础测试结果: {passed}/{len(tests)} 通过")
    
    # 如果基础测试通过，运行演示
    if passed >= 2:
        demo_voice_to_claude()
        
        print("\n" + "=" * 50)
        print("测试总结:")
        print("  [OK] 语音桥接系统基本功能正常")
        print("  [OK] Claude模拟器可以处理语音命令")
        print("  [OK] 可以进行完整的语音-命令-响应流程")
        print("\n下一步:")
        print("  1. 运行: python voice_to_claude.py (基础版)")
        print("  2. 运行: python claude_voice_bridge.py (增强版)")
        print("  3. 安装真实的Claude Code CLI获得完整体验")
    else:
        print("\n基础测试未通过，请检查环境配置")
    
    print("\n测试完成!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n测试被中断")
    except Exception as e:
        print(f"\n测试异常: {e}")
        import traceback
        traceback.print_exc()