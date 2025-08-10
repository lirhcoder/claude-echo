@echo off
echo ======================================
echo  Claude Code UI 桥接器启动器
echo ======================================
echo.
echo 选择启动模式:
echo 1. 简单UI桥接器 (claude_ui_bridge.py)
echo 2. 高级会话桥接器 (claude_session_bridge.py) - 推荐
echo 3. 命令行语音桥接 (voice_to_claude_fixed.py)
echo.
set /p choice=请输入选择 (1-3): 

if "%choice%"=="1" (
    echo 启动简单UI桥接器...
    python claude_ui_bridge.py
) else if "%choice%"=="2" (
    echo 启动高级会话桥接器...
    python claude_session_bridge.py
) else if "%choice%"=="3" (
    echo 启动命令行语音桥接...
    python voice_to_claude_fixed.py
) else (
    echo 无效选择，启动默认界面...
    python claude_session_bridge.py
)

echo.
echo 程序已退出，按任意键关闭...
pause > nul