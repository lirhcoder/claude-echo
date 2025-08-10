@echo off
chcp 65001 > nul
echo ======================================
echo  Claude Code Advanced Session Bridge
echo ======================================
echo.
echo Starting the recommended UI interface...
echo.
python claude_session_bridge.py

if %errorlevel% neq 0 (
    echo.
    echo Error occurred. Checking dependencies...
    echo.
    python -c "import tkinter; print('Tkinter: OK')"
    python -c "import psutil; print('psutil: OK')"
    echo.
    echo If you see import errors above, please install:
    echo pip install psutil
    echo.
    echo For voice features, also install:
    echo pip install openai-whisper pyaudio
)

echo.
echo Program exited. Press any key to close...
pause > nul