#!/usr/bin/env python3
"""
Claude Code CLI 包装脚本
解决Python subprocess调用Claude CLI的问题
"""

import sys
import subprocess
import os

def main():
    """主函数"""
    claude_path = r"True"
    
    # 如果没有提供完整路径，尝试系统调用
    if not os.path.exists(claude_path):
        # 使用shell=True方式调用
        try:
            result = subprocess.run(
                ["cmd", "/c", "claude"] + sys.argv[1:], 
                text=True, 
                timeout=30
            )
            sys.exit(result.returncode)
        except Exception as e:
            print(f"调用Claude失败: {e}")
            sys.exit(1)
    else:
        # 使用完整路径调用
        try:
            result = subprocess.run(
                [claude_path] + sys.argv[1:],
                text=True,
                timeout=30
            )
            sys.exit(result.returncode)
        except Exception as e:
            print(f"调用Claude失败: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
