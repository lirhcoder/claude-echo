#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claude Code CLI 诊断脚本
Diagnostic script for Claude Code CLI detection
"""

import sys
import subprocess
import os
import shutil
from pathlib import Path

def print_separator(title):
    """打印分隔符"""
    print("\n" + "=" * 60)
    print(f"  {title}".center(60))
    print("=" * 60)

def check_claude_in_path():
    """检查Claude是否在PATH中"""
    print_separator("检查 Claude Code CLI 在系统PATH中")
    
    # 方法1: 使用shutil.which
    claude_path = shutil.which('claude')
    if claude_path:
        print(f"[找到] shutil.which 找到 Claude: {claude_path}")
        return True
    else:
        print("[未找到] shutil.which 未找到 Claude")
    
    # 方法2: 检查常见安装位置
    common_paths = [
        "C:\\Users\\%USERNAME%\\AppData\\Local\\Programs\\Claude\\claude.exe",
        "C:\\Program Files\\Claude\\claude.exe", 
        "C:\\Program Files (x86)\\Claude\\claude.exe",
        "%LOCALAPPDATA%\\Programs\\Claude\\claude.exe"
    ]
    
    print("\n[检查] 常见安装位置:")
    for path_template in common_paths:
        expanded_path = os.path.expandvars(path_template)
        if os.path.exists(expanded_path):
            print(f"[找到] {expanded_path}")
            return expanded_path
        else:
            print(f"[未找到] {expanded_path}")
    
    return None

def check_environment():
    """检查环境信息"""
    print_separator("系统环境信息")
    
    print(f"Python版本: {sys.version}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python可执行文件: {sys.executable}")
    
    # 检查PATH环境变量
    path_env = os.environ.get('PATH', '')
    print(f"\nPATH环境变量包含的路径数量: {len(path_env.split(os.pathsep))}")
    
    # 显示前几个PATH路径
    paths = path_env.split(os.pathsep)
    print("前10个PATH路径:")
    for i, path in enumerate(paths[:10], 1):
        print(f"  {i:2d}. {path}")
    
    # 检查是否在虚拟环境中
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    print(f"\n虚拟环境状态: {'是' if in_venv else '否'}")
    if in_venv:
        print(f"虚拟环境路径: {sys.prefix}")

def test_subprocess_calls():
    """测试不同的subprocess调用方法"""
    print_separator("测试不同的 subprocess 调用方法")
    
    methods = [
        ("subprocess.run(['claude', '--version'])", ['claude', '--version']),
        ("subprocess.run('claude --version', shell=True)", 'claude --version'),
        ("subprocess.run(['cmd', '/c', 'claude', '--version'])", ['cmd', '/c', 'claude', '--version']),
        ("subprocess.run('cmd /c claude --version', shell=True)", 'cmd /c claude --version'),
    ]
    
    for desc, cmd in methods:
        print(f"\n[测试] {desc}")
        try:
            if isinstance(cmd, list):
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            else:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, shell=True)
            
            if result.returncode == 0:
                print(f"[成功] 返回码: {result.returncode}")
                print(f"[输出] {result.stdout.strip()}")
                return cmd  # 返回成功的方法
            else:
                print(f"[失败] 返回码: {result.returncode}")
                if result.stderr:
                    print(f"[错误] {result.stderr.strip()}")
                    
        except subprocess.TimeoutExpired:
            print("[超时] 命令执行超时")
        except FileNotFoundError:
            print("[未找到] 文件未找到")
        except Exception as e:
            print(f"[异常] {e}")
    
    return None

def test_claude_manually():
    """手动测试Claude CLI"""
    print_separator("手动Claude CLI测试")
    
    # 尝试找到Claude的完整路径
    claude_path = check_claude_in_path()
    if not claude_path:
        print("未找到Claude CLI，无法进行手动测试")
        return False
    
    if isinstance(claude_path, str) and os.path.exists(claude_path):
        print(f"[测试] 使用完整路径: {claude_path}")
        try:
            result = subprocess.run([claude_path, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"[成功] {result.stdout.strip()}")
                return claude_path
            else:
                print(f"[失败] 返回码: {result.returncode}, 错误: {result.stderr}")
        except Exception as e:
            print(f"[异常] {e}")
    
    return False

def create_wrapper_script():
    """创建Claude CLI包装脚本"""
    print_separator("创建Claude CLI包装脚本")
    
    # 尝试找到Claude的路径
    claude_path = check_claude_in_path()
    if not claude_path:
        print("未找到Claude CLI，无法创建包装脚本")
        return False
    
    wrapper_path = Path(__file__).parent / "claude_wrapper.py"
    
    wrapper_content = f'''#!/usr/bin/env python3
"""
Claude Code CLI 包装脚本
解决Python subprocess调用Claude CLI的问题
"""

import sys
import subprocess
import os

def main():
    """主函数"""
    claude_path = r"{claude_path}"
    
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
            print(f"调用Claude失败: {{e}}")
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
            print(f"调用Claude失败: {{e}}")
            sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    try:
        with open(wrapper_path, 'w', encoding='utf-8') as f:
            f.write(wrapper_content)
        print(f"[创建] 包装脚本: {wrapper_path}")
        
        # 测试包装脚本
        result = subprocess.run([sys.executable, str(wrapper_path), '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"[测试成功] {result.stdout.strip()}")
            return str(wrapper_path)
        else:
            print(f"[测试失败] {result.stderr}")
            
    except Exception as e:
        print(f"[创建失败] {e}")
    
    return False

def provide_solutions():
    """提供解决方案"""
    print_separator("解决方案建议")
    
    print("基于诊断结果，建议以下解决方案：")
    print()
    print("1. [临时解决] 使用 shell=True 调用:")
    print("   subprocess.run('claude --version', shell=True)")
    print()
    print("2. [路径解决] 添加Claude到Python PATH:")
    print("   import os")
    print("   os.environ['PATH'] += os.pathsep + 'Claude安装路径'")
    print()
    print("3. [包装脚本] 使用生成的包装脚本:")
    wrapper_path = Path(__file__).parent / "claude_wrapper.py"
    if wrapper_path.exists():
        print(f"   python {wrapper_path} --version")
    print()
    print("4. [环境修复] 重启命令行或IDE后重试")
    print()
    print("5. [权限检查] 确保Claude CLI有执行权限")

def main():
    """主诊断函数"""
    print("Claude Code CLI 诊断工具")
    print("用于解决Python无法检测到Claude CLI的问题")
    
    # 步骤1: 检查环境
    check_environment()
    
    # 步骤2: 检查Claude是否在PATH中
    claude_found = check_claude_in_path()
    
    # 步骤3: 测试不同的调用方法
    working_method = test_subprocess_calls()
    
    # 步骤4: 手动测试
    manual_result = test_claude_manually()
    
    # 步骤5: 创建包装脚本
    wrapper_result = create_wrapper_script()
    
    # 步骤6: 提供解决方案
    provide_solutions()
    
    # 总结
    print_separator("诊断总结")
    
    if working_method:
        print("[成功] 找到可工作的调用方法")
        print(f"推荐使用: {working_method}")
    elif manual_result:
        print("[部分成功] 可以通过完整路径调用")
        print(f"Claude路径: {manual_result}")
    elif wrapper_result:
        print("[解决方案] 创建了包装脚本")
        print(f"使用: python {wrapper_result}")
    else:
        print("[需要手动解决] 未找到自动解决方案")
        print("请检查Claude Code CLI安装和PATH配置")
    
    print("\n建议: 重新运行语音桥接测试脚本查看效果")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n诊断被中断")
    except Exception as e:
        print(f"\n诊断异常: {e}")
        import traceback
        traceback.print_exc()