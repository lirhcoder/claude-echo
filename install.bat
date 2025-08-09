@echo off
REM Claude Voice Assistant - Quick Installation Script
REM 快速安装和环境配置脚本

echo.
echo ============================================
echo  Claude Voice Assistant Alpha 测试安装
echo ============================================
echo.

REM Check if Python is installed
echo [1/6] 检查 Python 环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python 未安装或未添加到PATH
    echo 请访问 https://www.python.org 下载并安装 Python 3.9 或更高版本
    pause
    exit /b 1
)

python -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python 版本过低，需要 3.9 或更高版本
    python --version
    pause
    exit /b 1
)

echo ✅ Python 版本检查通过
python --version

REM Create virtual environment
echo.
echo [2/6] 创建虚拟环境...
if exist "venv\" (
    echo 虚拟环境已存在，跳过创建
) else (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ❌ 虚拟环境创建失败
        pause
        exit /b 1
    )
    echo ✅ 虚拟环境创建成功
)

REM Activate virtual environment
echo.
echo [3/6] 激活虚拟环境...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ❌ 虚拟环境激活失败
    pause
    exit /b 1
)
echo ✅ 虚拟环境已激活

REM Upgrade pip
echo.
echo [4/6] 更新 pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo ⚠️ pip 更新失败，继续安装
)

REM Install dependencies
echo.
echo [5/6] 安装项目依赖...
echo 这可能需要几分钟时间，请耐心等待...

REM Install core dependencies first
pip install pydantic loguru pyyaml aiofiles asyncio-mqtt
if %errorlevel% neq 0 (
    echo ❌ 核心依赖安装失败
    pause
    exit /b 1
)

REM Install speech dependencies (optional)
echo 正在安装语音处理依赖...
pip install openai-whisper pyttsx3 pyaudio wave
if %errorlevel% neq 0 (
    echo ⚠️ 语音依赖安装失败，将在Mock模式下运行
    echo 如需完整功能，请手动安装：pip install openai-whisper pyttsx3 pyaudio
)

REM Install additional dependencies
pip install requests aiohttp numpy
if %errorlevel% neq 0 (
    echo ⚠️ 部分依赖安装失败，可能影响功能
)

echo ✅ 依赖安装完成

REM Create necessary directories
echo.
echo [6/6] 初始化项目结构...
if not exist "logs\" mkdir logs
if not exist "temp\" mkdir temp
if not exist "test_projects\" mkdir test_projects
if not exist ".voice-assistant-backups\" mkdir .voice-assistant-backups
if not exist ".voice-assistant-cache\" mkdir .voice-assistant-cache

echo ✅ 项目结构初始化完成

REM Run initial setup
echo.
echo 运行初始化配置...
python -c "
import yaml
import os
from pathlib import Path

# Create basic test configuration
config = {
    'system': {
        'log_level': 'INFO',
        'environment': 'testing',
        'enable_mock_mode': True
    },
    'speech': {
        'enabled': False,
        'mock_mode': True
    },
    'adapters': {
        'claude_code': {
            'enabled': True,
            'mock_mode': True
        }
    }
}

config_path = Path('config/test_config.yaml')
config_path.parent.mkdir(exist_ok=True)
with open(config_path, 'w', encoding='utf-8') as f:
    yaml.dump(config, f, allow_unicode=True)

print('✅ 测试配置文件已生成')
"

REM Test basic import
echo.
echo 验证安装...
python -c "
try:
    import sys
    sys.path.append('src')
    from core.types import CommandResult, AdapterStatus
    from core.config_manager import ConfigManager
    print('✅ 核心模块导入成功')
    
    # Test config loading
    config = ConfigManager('config/test_config.yaml')
    print('✅ 配置管理器测试通过')
    
except ImportError as e:
    print(f'❌ 模块导入失败: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ 配置测试失败: {e}')
    sys.exit(1)
"

if %errorlevel% neq 0 (
    echo ❌ 安装验证失败
    pause
    exit /b 1
)

REM Create desktop shortcut
echo.
echo 创建快速启动脚本...
echo @echo off > start_claude_voice.bat
echo call venv\Scripts\activate.bat >> start_claude_voice.bat
echo python src\main.py --config config\test_config.yaml >> start_claude_voice.bat
echo pause >> start_claude_voice.bat

echo.
echo ============================================
echo             🎉 安装完成！
echo ============================================
echo.
echo Alpha 测试环境已准备就绪！
echo.
echo 📋 下一步:
echo   1. 运行 start_claude_voice.bat 启动系统
echo   2. 阅读 testing\alpha_test_checklist.md 开始测试
echo   3. 遇到问题请查看 logs\claude_voice.log
echo.
echo 📞 技术支持:
echo   - GitHub Issues: https://github.com/claude-voice/issues
echo   - 测试指南: docs\testing_guide.md
echo.
echo ⚠️  注意: 当前在Mock模式下运行，语音功能已禁用
echo    如需完整功能，请安装完整语音依赖
echo.
pause