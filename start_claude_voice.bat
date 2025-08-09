@echo off
chcp 65001 >nul
REM Claude Voice Assistant - Quick Start Script for Windows
REM Alpha Testing Version

title Claude Voice Assistant Alpha

echo.
echo ========================================
echo  Claude Voice Assistant Alpha Testing
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo [ERROR] Virtual environment not found
    echo Please run install.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if core modules exist
if not exist "src\main.py" (
    echo [ERROR] Source code files not found
    echo Please check project integrity
    pause
    exit /b 1
)

REM Create logs directory if not exists
if not exist "logs\" mkdir logs

REM Set environment variables
set PYTHONPATH=src;%PYTHONPATH%
set CLAUDE_VOICE_ENV=alpha_testing

echo.
echo Starting Claude Voice Assistant...
echo [INFO] Alpha testing version - Mock mode enabled
echo [INFO] Testing guide: testing\alpha_test_checklist.md
echo.

REM Start the application
python src\main.py --config config\test_config.yaml

echo.
echo Application exited
echo Check logs: logs\alpha_test.log
echo.
pause