#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试真实Claude Code CLI连接
"""

import subprocess
import sys

def test_claude_direct():
    """直接测试Claude CLI"""
    print("=" * 50)
    print("测试真实Claude Code CLI")
    print("=" * 50)
    
    try:
        # 测试版本
        print("1. 测试版本命令:")
        result = subprocess.run('claude --version', 
                              capture_output=True, text=True, shell=True, timeout=5)
        if result.returncode == 0:
            print(f"   [成功] {result.stdout.strip()}")
        else:
            print(f"   [失败] 返回码: {result.returncode}")
            return False
            
        # 测试简单命令
        print("\n2. 测试简单命令:")
        test_command = "help"
        
        result = subprocess.run('claude', input=test_command, 
                              capture_output=True, text=True, shell=True, timeout=10)
        
        if result.returncode == 0:
            print("   [成功] Claude响应:")
            # 只显示前几行
            lines = result.stdout.strip().split('\n')
            for line in lines[:8]:
                print(f"   {line}")
            if len(lines) > 8:
                print(f"   ... (还有 {len(lines)-8} 行)")
        else:
            print(f"   [失败] 返回码: {result.returncode}")
            if result.stderr:
                print(f"   错误: {result.stderr}")
                
        return True
        
    except Exception as e:
        print(f"测试异常: {e}")
        return False

def test_voice_command():
    """测试语音命令"""
    print("\n" + "=" * 50)
    print("测试语音命令发送")
    print("=" * 50)
    
    test_commands = [
        "创建一个简单的Python函数",
        "显示当前工作目录",
        "检查Python版本"
    ]
    
    for i, command in enumerate(test_commands, 1):
        print(f"\n{i}. 测试命令: '{command}'")
        print("-" * 30)
        
        try:
            result = subprocess.run('claude', input=command, 
                                  capture_output=True, text=True, shell=True, timeout=15)
            
            if result.returncode == 0:
                print("   [成功] Claude响应:")
                lines = result.stdout.strip().split('\n')
                for line in lines[:10]:  # 显示前10行
                    print(f"   {line}")
                if len(lines) > 10:
                    print(f"   ... (还有 {len(lines)-10} 行)")
            else:
                print(f"   [失败] 返回码: {result.returncode}")
                if result.stderr:
                    print(f"   错误: {result.stderr}")
                    
        except subprocess.TimeoutExpired:
            print("   [超时] 命令执行超时")
        except Exception as e:
            print(f"   [异常] {e}")

def main():
    """主函数"""
    print("真实Claude Code CLI测试")
    
    # 基础测试
    if not test_claude_direct():
        print("\n[失败] 基础Claude CLI测试失败")
        print("请确认Claude Code已正确安装")
        return
    
    # 语音命令测试
    test_voice_command()
    
    print("\n" + "=" * 50)
    print("测试完成!")
    print("现在可以使用语音桥接器连接真实的Claude Code CLI")
    print("运行: python voice_to_claude.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n测试被中断")
    except Exception as e:
        print(f"\n测试异常: {e}")
        import traceback
        traceback.print_exc()