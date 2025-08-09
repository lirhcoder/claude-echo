@echo off
chcp 65001 >nul
title Claude Voice Assistant - Alpha Interactive Testing

echo.
echo ============================================
echo  Claude Voice Assistant Alpha Testing
echo ============================================
echo.
echo [INFO] Interactive Testing Mode
echo [INFO] 通过文本输入模拟语音命令进行测试
echo [INFO] 所有模块运行在MOCK模式
echo.

if not exist "venv\" (
    echo [ERROR] Virtual environment not found
    echo Please run install_alpha_fixed.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Set environment variables
set PYTHONPATH=src;%PYTHONPATH%
set CLAUDE_VOICE_ENV=alpha_testing
set CLAUDE_VOICE_DEBUG=1

echo Starting Alpha Interactive Testing...
echo.
python alpha_interactive_test.py

echo.
echo Alpha testing session completed
echo Check testing\ directory for test results
pause