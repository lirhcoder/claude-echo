@echo off
chcp 65001 >nul
title Claude Echo - 语音测试环境

echo ============================================
echo  Claude Echo 第四阶段语音测试启动器
echo ============================================
echo.

REM 检查虚拟环境
if not exist "venv\" (
    echo [错误] 虚拟环境不存在
    echo 请先运行 install_alpha_fixed.bat
    pause
    exit /b 1
)

echo [1/4] 激活虚拟环境...
call venv\Scripts\activate.bat

echo [2/4] 检查语音依赖...
python -c "import whisper, pyttsx3, pyaudio; print('[OK] 语音依赖可用')" 2>nul
if %errorlevel% neq 0 (
    echo [警告] 语音依赖不完整，正在安装...
    call install_voice_deps.bat
)

echo [3/4] 检查音频设备...
python -c "import pyaudio; p=pyaudio.PyAudio(); print(f'[OK] 发现 {p.get_device_count()} 个音频设备'); p.terminate()" 2>nul
if %errorlevel% neq 0 (
    echo [错误] 音频设备检查失败
    echo 请确认麦克风已连接并授权应用访问
    pause
    exit /b 1
)

echo [4/4] 启动语音测试环境...
echo.
echo ============================================
echo  语音测试环境已就绪
echo ============================================
echo.
echo 使用指南:
echo  1. 测试指南: VOICE_TESTING_GUIDE.md
echo  2. 说 "你好Claude" 或 "Hello Claude" 开始
echo  3. 按 Ctrl+C 退出测试
echo.
echo 正在启动...
python simple_voice_test.py

echo.
echo 语音测试会话结束
echo 查看测试日志：logs/voice_testing.log
pause