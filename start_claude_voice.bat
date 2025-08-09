@echo off
REM Claude Voice Assistant - Quick Start Script for Windows
REM Windows快速启动脚本

title Claude Voice Assistant Alpha

echo.
echo ========================================
echo  Claude Voice Assistant Alpha 测试
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo ❌ 虚拟环境不存在，请先运行 install.bat
    pause
    exit /b 1
)

REM Activate virtual environment
echo 激活虚拟环境...
call venv\Scripts\activate.bat

REM Check if core modules exist
if not exist "src\main.py" (
    echo ❌ 源代码文件不存在，请检查项目完整性
    pause
    exit /b 1
)

REM Create logs directory if not exists
if not exist "logs\" mkdir logs

REM Set environment variables
set PYTHONPATH=src;%PYTHONPATH%
set CLAUDE_VOICE_ENV=alpha_testing

echo.
echo 🚀 启动 Claude Voice Assistant...
echo ⚠️  Alpha测试版本 - Mock模式运行
echo 📋 测试指南: testing\alpha_test_checklist.md
echo.

REM Start the application
python src\main.py --config config\test_config.yaml

echo.
echo 📊 应用已退出
echo 📝 查看日志: logs\alpha_test.log
echo.
pause