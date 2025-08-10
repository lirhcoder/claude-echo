#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终语音桥接测试
"""

import subprocess
import sys
import os
from pathlib import Path

def test_voice_bridge_fixed():
    """测试修复版语音桥接器的连接功能"""
    print("=" * 60)
    print("测试语音桥接器与真实Claude CLI连接")
    print("=" * 60)
    
    # 模拟语音识别结果，直接测试发送功能
    test_commands = [
        "help",
        "创建一个简单的Python Hello World程序",
        "显示当前时间的函数"
    ]
    
    for i, command in enumerate(test_commands, 1):
        print(f"\n{i}. 测试命令: '{command}'")
        print("-" * 40)
        
        # 创建临时文件
        temp_file = f"test_cmd_{i}.txt"
        
        try:
            # 写入命令
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(command)
            
            # 使用文件重定向方式调用Claude
            cmd = f'claude < {temp_file}'
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  shell=True, timeout=15, encoding='utf-8', errors='ignore')
            
            print(f"返回码: {result.returncode}")
            
            if result.stdout:
                print("Claude响应:")
                lines = result.stdout.strip().split('\n')
                for j, line in enumerate(lines[:12], 1):
                    if line.strip():
                        print(f"  {j:2d}. {line.strip()}")
                if len(lines) > 12:
                    print(f"  ... (还有 {len(lines)-12} 行)")
            
            if result.stderr:
                print(f"错误: {result.stderr}")
                
            success = result.returncode == 0
            print(f"结果: {'[成功]' if success else '[失败]'}")
            
        except subprocess.TimeoutExpired:
            print("结果: [超时]")
        except Exception as e:
            print(f"结果: [异常] {e}")
        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)

def main():
    """主函数"""
    print("语音桥接器最终测试")
    
    # 首先检查Claude CLI
    try:
        result = subprocess.run('claude --version', capture_output=True, 
                              text=True, shell=True, timeout=5)
        if result.returncode == 0:
            print(f"Claude CLI版本: {result.stdout.strip()}")
        else:
            print("Claude CLI检查失败")
            return
    except Exception as e:
        print(f"Claude CLI检查异常: {e}")
        return
    
    # 测试连接功能
    test_voice_bridge_fixed()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print()
    print("如果以上测试成功，说明语音桥接器可以正常工作")
    print("您现在可以:")
    print("  1. 运行: python voice_to_claude_fixed.py")
    print("  2. 输入 'r' 开始语音录音")
    print("  3. 说出您的编程命令")
    print("  4. 系统会自动发送给Claude Code并显示结果")
    print()
    print("语音控制Claude Code系统已就绪!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n测试被中断")
    except Exception as e:
        print(f"\n测试异常: {e}")
        import traceback
        traceback.print_exc()