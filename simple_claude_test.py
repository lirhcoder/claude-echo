#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单Claude Code CLI测试
"""

import subprocess
import sys

def test_claude_with_echo():
    """使用echo命令测试Claude CLI"""
    print("=" * 50)
    print("简单Claude Code CLI连接测试")
    print("=" * 50)
    
    try:
        # 使用echo将命令传递给Claude
        test_command = "创建一个Hello World函数"
        
        print(f"测试命令: {test_command}")
        print("-" * 30)
        
        # 方法1: 使用echo + 管道
        cmd = f'echo "{test_command}" | claude'
        print(f"执行: {cmd}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              shell=True, timeout=20, encoding='utf-8', errors='ignore')
        
        print(f"返回码: {result.returncode}")
        
        if result.stdout:
            print("Claude响应:")
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines[:15], 1):
                print(f"  {i:2d}. {line}")
            if len(lines) > 15:
                print(f"  ... (还有 {len(lines)-15} 行)")
        
        if result.stderr:
            print("错误信息:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("命令执行超时")
        return False
    except Exception as e:
        print(f"测试异常: {e}")
        return False

def test_claude_version_only():
    """只测试Claude版本"""
    print("\n测试Claude版本信息:")
    try:
        result = subprocess.run('claude --version', capture_output=True, 
                              text=True, shell=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ Claude Code CLI版本: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ 版本检查失败: {result.returncode}")
            return False
    except Exception as e:
        print(f"❌ 版本检查异常: {e}")
        return False

def main():
    """主函数"""
    print("Claude Code CLI 连接测试")
    
    # 先测试版本
    if not test_claude_version_only():
        print("版本测试失败，无法继续")
        return
    
    # 测试基本命令
    success = test_claude_with_echo()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ 测试成功! Claude Code CLI可以正常工作")
        print("可以继续使用语音桥接器")
    else:
        print("⚠️  基本连接有问题，但版本检查正常")
        print("建议使用模拟器进行语音功能测试")
    
    print("下一步: python voice_to_claude.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n测试被中断")
    except Exception as e:
        print(f"\n测试异常: {e}")
        import traceback
        traceback.print_exc()