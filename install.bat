@echo off
chcp 65001 >nul
REM Claude Voice Assistant - Quick Installation Script
REM Alpha Testing Environment Setup

echo.
echo ============================================
echo  Claude Voice Assistant Alpha Installation
echo ============================================
echo.

REM Check if Python is installed
echo [1/6] Checking Python environment...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please download and install Python 3.9+ from https://www.python.org
    pause
    exit /b 1
)

python -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python version is too old, requires 3.9+
    python --version
    pause
    exit /b 1
)

echo [OK] Python version check passed
python --version

REM Create virtual environment
echo.
echo [2/6] Creating virtual environment...
if exist "venv\" (
    echo Virtual environment already exists, skipping creation
) else (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created successfully
)

REM Activate virtual environment
echo.
echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated

REM Upgrade pip
echo.
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo [WARNING] pip upgrade failed, continuing installation
)

REM Install dependencies
echo.
echo [5/6] Installing project dependencies...
echo This may take several minutes, please wait...

REM Install core dependencies first
pip install pydantic loguru pyyaml aiofiles
if %errorlevel% neq 0 (
    echo [ERROR] Core dependencies installation failed
    pause
    exit /b 1
)

REM Install speech dependencies (optional) - Fixed to avoid MySQL dependency
echo Installing speech processing dependencies...
echo [INFO] Installing minimal speech dependencies for Alpha testing...
pip install openai-whisper pyttsx3 pyaudio
if %errorlevel% neq 0 (
    echo [WARNING] Speech dependencies failed, will run in Mock mode
    echo For full functionality, manually install: pip install openai-whisper pyttsx3 pyaudio
)

REM Install additional dependencies
pip install requests aiohttp numpy
if %errorlevel% neq 0 (
    echo [WARNING] Some dependencies failed, may affect functionality
)

echo [OK] Dependencies installation completed

REM Create necessary directories
echo.
echo [6/6] Initializing project structure...
if not exist "logs\" mkdir logs
if not exist "temp\" mkdir temp
if not exist "test_projects\" mkdir test_projects
if not exist ".voice-assistant-backups\" mkdir .voice-assistant-backups
if not exist ".voice-assistant-cache\" mkdir .voice-assistant-cache

echo [OK] Project structure initialized

REM Run initial setup - Create Python script file to avoid command line issues
echo.
echo Running initial configuration...

REM Create temporary Python script
echo import yaml > temp_config.py
echo import os >> temp_config.py
echo from pathlib import Path >> temp_config.py
echo. >> temp_config.py
echo # Create basic test configuration >> temp_config.py
echo config = { >> temp_config.py
echo     'system': { >> temp_config.py
echo         'log_level': 'INFO', >> temp_config.py
echo         'environment': 'testing', >> temp_config.py
echo         'enable_mock_mode': True >> temp_config.py
echo     }, >> temp_config.py
echo     'speech': { >> temp_config.py
echo         'enabled': False, >> temp_config.py
echo         'mock_mode': True >> temp_config.py
echo     }, >> temp_config.py
echo     'adapters': { >> temp_config.py
echo         'claude_code': { >> temp_config.py
echo             'enabled': True, >> temp_config.py
echo             'mock_mode': True >> temp_config.py
echo         } >> temp_config.py
echo     } >> temp_config.py
echo } >> temp_config.py
echo. >> temp_config.py
echo config_path = Path('config/test_config.yaml') >> temp_config.py
echo config_path.parent.mkdir(exist_ok=True) >> temp_config.py
echo with open(config_path, 'w', encoding='utf-8') as f: >> temp_config.py
echo     yaml.dump(config, f, allow_unicode=True) >> temp_config.py
echo. >> temp_config.py
echo print('[OK] Test configuration file generated') >> temp_config.py

REM Run the Python script
python temp_config.py
if %errorlevel% neq 0 (
    echo [ERROR] Configuration setup failed
) else (
    echo [OK] Configuration completed
)

REM Clean up temporary file
del temp_config.py >nul 2>&1

REM Test basic import - Create temporary test script
echo.
echo Verifying installation...

echo try: > temp_test.py
echo     import sys >> temp_test.py
echo     sys.path.append('src') >> temp_test.py
echo     from core.types import CommandResult, AdapterStatus >> temp_test.py
echo     from core.config_manager import ConfigManager >> temp_test.py
echo     print('[OK] Core modules imported successfully') >> temp_test.py
echo     # Test config loading >> temp_test.py
echo     config = ConfigManager('config/test_config.yaml') >> temp_test.py
echo     print('[OK] Configuration manager test passed') >> temp_test.py
echo except ImportError as e: >> temp_test.py
echo     print(f'[ERROR] Module import failed: {e}') >> temp_test.py
echo     sys.exit(1) >> temp_test.py
echo except Exception as e: >> temp_test.py
echo     print(f'[ERROR] Configuration test failed: {e}') >> temp_test.py
echo     sys.exit(1) >> temp_test.py

REM Run test script
python temp_test.py
set test_result=%errorlevel%

REM Clean up test file
del temp_test.py >nul 2>&1

if %test_result% neq 0 (
    echo [ERROR] Installation verification failed
    pause
    exit /b 1
)

REM Create quick start script
echo.
echo Creating quick start script...
echo @echo off > start_claude_voice.bat
echo call venv\Scripts\activate.bat >> start_claude_voice.bat
echo python src\main.py --config config\test_config.yaml >> start_claude_voice.bat
echo pause >> start_claude_voice.bat

echo.
echo ============================================
echo            Installation Complete!
echo ============================================
echo.
echo Alpha testing environment is ready!
echo.
echo Next steps:
echo   1. Run start_claude_voice.bat to start the system
echo   2. Read testing\alpha_test_checklist.md for testing guide
echo   3. Check logs\claude_voice.log for troubleshooting
echo.
echo Technical support:
echo   - GitHub Issues: https://github.com/lirhcoder/claude-echo/issues
echo   - Testing guide: docs\testing_guide.md
echo.
echo NOTE: Currently running in Mock mode, speech features disabled
echo       Alpha version focuses on core architecture testing
echo.
pause