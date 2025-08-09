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

REM Install speech dependencies (optional)
echo Installing speech processing dependencies...
pip install openai-whisper pyttsx3 pyaudio wave
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

REM Run initial setup
echo.
echo Running initial configuration...
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

print('[OK] Test configuration file generated')
"

REM Test basic import
echo.
echo Verifying installation...
python -c "
try:
    import sys
    sys.path.append('src')
    from core.types import CommandResult, AdapterStatus
    from core.config_manager import ConfigManager
    print('[OK] Core modules imported successfully')
    
    # Test config loading
    config = ConfigManager('config/test_config.yaml')
    print('[OK] Configuration manager test passed')
    
except ImportError as e:
    print(f'[ERROR] Module import failed: {e}')
    sys.exit(1)
except Exception as e:
    print(f'[ERROR] Configuration test failed: {e}')
    sys.exit(1)
"

if %errorlevel% neq 0 (
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
echo       For full functionality, install complete speech dependencies
echo.
pause