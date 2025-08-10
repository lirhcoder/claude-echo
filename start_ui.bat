@echo off
chcp 65001 > nul
echo ======================================
echo  Claude Code UI Bridge Launcher
echo ======================================
echo.
echo Select startup mode:
echo 1. Simple UI Bridge (claude_ui_bridge.py)
echo 2. Advanced Session Bridge (claude_session_bridge.py) - Recommended
echo 3. Command Line Voice Bridge (voice_to_claude_fixed.py)
echo.
set /p choice=Please enter your choice (1-3): 

if "%choice%"=="1" (
    echo Starting Simple UI Bridge...
    python claude_ui_bridge.py
) else if "%choice%"=="2" (
    echo Starting Advanced Session Bridge...
    python claude_session_bridge.py
) else if "%choice%"=="3" (
    echo Starting Command Line Voice Bridge...
    python voice_to_claude_fixed.py
) else (
    echo Invalid choice, starting default interface...
    python claude_session_bridge.py
)

echo.
echo Program exited. Press any key to close...
pause > nul