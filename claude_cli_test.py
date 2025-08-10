#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claude Code CLI 基础连接测试 (无emoji版本)
"""

import subprocess
import sys
import os

def test_claude_basic():
    """基础Claude CLI测试"""
    print("=" * 50)
    print("Claude Code CLI 基础测试")
    print("=" * 50)
    
    # 测试1: 版本检查
    print("1. 检查Claude CLI版本...")
    try:
        result = subprocess.run('claude --version', capture_output=True, 
                              text=True, shell=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"   [成功] 版本: {version}")
        else:
            print(f"   [失败] 返回码: {result.returncode}")
            return False
    except Exception as e:
        print(f"   [异常] {e}")
        return False
    
    # 测试2: 简单交互 (使用临时文件)
    print("\n2. 测试命令交互...")
    
    # 创建临时命令文件
    temp_file = "temp_claude_cmd.txt"
    test_command = "help"
    
    try:
        # 写入命令到临时文件
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(test_command)
        
        # 使用文件重定向
        cmd = f'claude < {temp_file}'
        print(f"   执行: {cmd}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              shell=True, timeout=15, encoding='utf-8', errors='ignore')
        
        print(f"   返回码: {result.returncode}")
        
        if result.stdout:
            print("   Claude响应 (前10行):")
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines[:10], 1):
                if line.strip():  # 只显示非空行
                    print(f"     {i}. {line.strip()}")
            if len(lines) > 10:
                print(f"     ... (还有 {len(lines)-10} 行)")
        
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("   [超时] 命令执行超时")
        return False
    except Exception as e:
        print(f"   [异常] {e}")
        return False
    finally:
        # 确保清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)

def test_voice_style_command():
    """测试语音风格的命令"""
    print("\n3. 测试语音风格命令...")
    
    # 创建临时命令文件
    temp_file = "temp_voice_cmd.txt"
    voice_command = "创建一个简单的Python Hello World函数"
    
    try:
        # 写入命令到临时文件
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(voice_command)
        
        # 使用文件重定向
        cmd = f'claude < {temp_file}'
        print(f"   测试命令: {voice_command}")
        print(f"   执行: {cmd}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              shell=True, timeout=20, encoding='utf-8', errors='ignore')
        
        print(f"   返回码: {result.returncode}")
        
        if result.stdout:
            print("   Claude响应 (前15行):")
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines[:15], 1):
                if line.strip():  # 只显示非空行
                    print(f"     {i}. {line.strip()}")
            if len(lines) > 15:
                print(f"     ... (还有 {len(lines)-15} 行)")
        
        if result.stderr:
            print("   错误信息:")
            print(f"     {result.stderr}")
        
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("   [超时] 语音命令执行超时")
        return False
    except Exception as e:
        print(f"   [异常] {e}")
        return False
    finally:
        # 确保清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)

def main():
    """主函数"""
    print("Claude Code CLI 连接测试工具")
    print("用于验证语音桥接器的Claude CLI连接")
    
    # 基础测试
    basic_ok = test_claude_basic()
    
    # 语音命令测试
    voice_ok = False
    if basic_ok:
        voice_ok = test_voice_style_command()
    
    # 结果总结
    print("\n" + "=" * 50)
    print("测试结果总结:")
    print("=" * 50)
    
    if basic_ok:
        print("[通过] 基础Claude CLI功能正常")
    else:
        print("[失败] 基础Claude CLI功能异常")
    
    if voice_ok:
        print("[通过] 语音风格命令可以正常处理")
    else:
        print("[失败] 语音风格命令处理异常")
    
    # 建议
    print("\n建议:")
    if basic_ok and voice_ok:
        print("  [优秀] Claude CLI完全正常，可以使用真实CLI")
        print("  运行: python voice_to_claude.py")
    elif basic_ok:
        print("  [良好] Claude CLI基本正常，可以尝试使用")
        print("  如有问题可使用模拟器: python claude_mock.py")
    else:
        print("  [建议] 使用模拟器进行语音功能测试")
        print("  运行: python quick_test.py")
    
    print("\n语音桥接器已准备就绪!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n测试被中断")
    except Exception as e:
        print(f"\n\n测试异常: {e}")
        import traceback
        traceback.print_exc()