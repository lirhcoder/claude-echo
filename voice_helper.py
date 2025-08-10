#!/usr/bin/env python3
"""
Claude Code 语音助手集成脚本
Voice Assistant Integration Script for Claude Code
"""

import sys
import subprocess
import os
from pathlib import Path
from datetime import datetime


def print_banner():
    """显示启动横幅"""
    print("=" * 60)
    print("🎤 Claude Echo - 语音助手集成工具".center(60))
    print("=" * 60)
    print()


def check_dependencies():
    """检查必要的依赖是否已安装"""
    print("🔍 检查系统依赖...")
    
    dependencies = {
        'whisper': 'openai-whisper',
        'pyttsx3': 'pyttsx3', 
        'pyaudio': 'pyaudio'
    }
    
    missing = []
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"   ✅ {module} - 已安装")
        except ImportError:
            print(f"   ❌ {module} - 未安装")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️ 缺少依赖: {', '.join(missing)}")
        print("请运行: pip install " + " ".join(missing))
        return False
    
    print("✅ 所有依赖已准备就绪\n")
    return True


def get_script_path(mode):
    """根据模式获取脚本路径"""
    base_path = Path(__file__).parent
    
    scripts = {
        "ui": base_path / "claude_voice_ui.py",
        "test": base_path / "simple_voice_test.py", 
        "full": base_path / "src" / "main.py",
        "demo": base_path / "src" / "main.py"
    }
    
    return scripts.get(mode)


def start_voice_assistant(mode="ui"):
    """启动语音助手"""
    print_banner()
    
    # 检查依赖
    if not check_dependencies():
        print("请先安装缺失的依赖，然后重新运行")
        return False
    
    # 获取脚本路径
    script_path = get_script_path(mode)
    
    if not script_path:
        print(f"❌ 无效模式: '{mode}'")
        print("可用模式: ui, test, full, demo")
        return False
    
    if not script_path.exists():
        print(f"❌ 脚本不存在: {script_path}")
        return False
    
    print(f"🚀 启动模式: {mode}")
    print(f"📁 脚本路径: {script_path}")
    print(f"⏰ 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    try:
        # 启动脚本
        if mode == "demo":
            # 演示模式使用特殊参数
            result = subprocess.run([
                sys.executable, str(script_path), "--demo"
            ], cwd=script_path.parent)
        else:
            result = subprocess.run([
                sys.executable, str(script_path)
            ], cwd=script_path.parent)
        
        print("-" * 60)
        if result.returncode == 0:
            print("✅ 程序正常结束")
        else:
            print(f"⚠️ 程序结束，返回码: {result.returncode}")
            
        return result.returncode == 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断程序")
        return False
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False


def show_help():
    """显示帮助信息"""
    help_text = """
🎤 Claude Echo 语音助手集成工具

用法: python voice_helper.py [模式] [选项]

可用模式:
  ui        启动终端UI界面 (默认，推荐)
  test      启动简化语音测试
  full      启动完整架构系统  
  demo      运行架构演示
  help      显示此帮助信息

示例:
  python voice_helper.py          # 启动UI界面
  python voice_helper.py ui       # 启动UI界面
  python voice_helper.py test     # 语音功能测试
  python voice_helper.py full     # 完整系统
  python voice_helper.py demo     # 架构演示

Claude Code 中使用:
  在 Claude Code 命令行中直接运行:
  > python voice_helper.py ui

依赖要求:
  - Python 3.8+
  - openai-whisper
  - pyttsx3  
  - pyaudio

更多信息:
  - 查看 CLAUDE_CODE_INTEGRATION.md
  - 查看 VOICE_TEST_RESULTS.md
  - 运行 python simple_voice_test.py 进行诊断
"""
    print(help_text)


def main():
    """主函数"""
    # 解析命令行参数
    if len(sys.argv) < 2:
        mode = "ui"  # 默认使用UI模式
    else:
        mode = sys.argv[1].lower()
    
    # 处理帮助命令
    if mode in ["help", "-h", "--help", "?"]:
        show_help()
        return
    
    # 处理版本信息
    if mode in ["version", "-v", "--version"]:
        print("Claude Echo Voice Assistant v4.0")
        print("Phase 4: Intelligent Learning System") 
        return
    
    # 启动语音助手
    success = start_voice_assistant(mode)
    
    if success:
        print("\n👋 感谢使用 Claude Echo 语音助手!")
    else:
        print("\n⚠️ 程序未能正常完成")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 程序被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)