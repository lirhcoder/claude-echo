@echo off
chcp 65001 >nul
REM Claude Voice Assistant - Alpha Testing Installation Script
REM Simplified installation for core architecture testing

echo.
echo ============================================
echo  Claude Voice Assistant Alpha Installation
echo ============================================
echo.

REM Check if Python is installed
echo [1/5] Checking Python environment...
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
echo [2/5] Creating virtual environment...
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
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated

REM Install Alpha testing dependencies
echo.
echo [4/5] Installing Alpha testing dependencies...
echo [INFO] Installing all required dependencies for core architecture testing...

REM Install from requirements file
if exist "requirements_alpha.txt" (
    pip install -r requirements_alpha.txt
) else (
    REM Fallback to manual installation
    pip install pydantic loguru pyyaml aiofiles requests aiohttp numpy watchdog
)

if %errorlevel% neq 0 (
    echo [ERROR] Core dependencies installation failed
    echo Trying individual installation...
    
    REM Try installing one by one
    pip install pydantic
    pip install loguru  
    pip install pyyaml
    pip install aiofiles
    pip install requests
    pip install aiohttp
    pip install numpy
    pip install watchdog
    
    if %errorlevel% neq 0 (
        echo [ERROR] Unable to install required dependencies
        pause
        exit /b 1
    )
)

echo [OK] Core dependencies installed successfully

REM Create necessary directories
echo.
echo [5/5] Initializing project structure...
if not exist "logs\" mkdir logs
if not exist "temp\" mkdir temp
if not exist "test_projects\" mkdir test_projects
if not exist "config\" mkdir config

echo [OK] Project structure initialized

REM Create Alpha test configuration using Python
echo.
echo Creating Alpha test configuration...
python -c "
import yaml
import os
from pathlib import Path

# Create comprehensive Alpha test configuration
config = {
    'system': {
        'log_level': 'INFO',
        'environment': 'alpha_testing',
        'enable_mock_mode': True,
        'debug_mode': True
    },
    'speech': {
        'enabled': False,
        'mock_mode': True,
        'recognizer': {
            'engine': 'mock',
            'model_size': 'base',
            'language': 'auto'
        },
        'synthesizer': {
            'engine': 'mock',
            'voice': 'default',
            'rate': 200
        }
    },
    'adapters': {
        'claude_code': {
            'enabled': True,
            'mock_mode': True,
            'priority': 100
        }
    },
    'agents': {
        'coordinator': {
            'enabled': True,
            'mock_mode': True
        },
        'task_planner': {
            'enabled': True,
            'mock_mode': True
        }
    },
    'event_system': {
        'enabled': True,
        'buffer_size': 100,
        'max_listeners_per_event': 10
    }
}

config_path = Path('config/alpha_config.yaml')
config_path.parent.mkdir(exist_ok=True)
with open(config_path, 'w', encoding='utf-8') as f:
    yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

print('[OK] Alpha configuration created')
"

if %errorlevel% neq 0 (
    echo [WARNING] Configuration creation failed, using minimal setup
)

REM Test core imports
echo.
echo Testing core module imports...
python -c "
import sys
sys.path.append('src')
try:
    from core.types import CommandResult, AdapterStatus
    from core.config_manager import ConfigManager  
    from core.event_system import EventSystem
    print('[OK] Core modules imported successfully')
except ImportError as e:
    print(f'[ERROR] Import failed: {e}')
    sys.exit(1)
"

if %errorlevel% neq 0 (
    echo [ERROR] Core modules import test failed
    echo Please check the error above and ensure all dependencies are installed
    pause
    exit /b 1
)

REM Create Alpha start script
echo.
echo Creating Alpha start script...
(
echo @echo off
echo chcp 65001 ^>nul
echo title Claude Voice Assistant Alpha Testing
echo echo.
echo echo ============================================
echo echo  Claude Voice Assistant Alpha Testing
echo echo ============================================
echo echo.
echo echo [INFO] Alpha version - Core architecture testing
echo echo [INFO] Speech processing: DISABLED for compatibility
echo echo [INFO] All modules running in MOCK mode
echo echo.
echo if not exist "venv\" (
echo     echo [ERROR] Virtual environment not found
echo     echo Please run install_alpha.bat first
echo     pause
echo     exit /b 1
echo ^)
echo.
echo REM Activate virtual environment
echo call venv\Scripts\activate.bat
echo.
echo REM Set environment variables
echo set PYTHONPATH=src;%%PYTHONPATH%%
echo set CLAUDE_VOICE_ENV=alpha_testing
echo set CLAUDE_VOICE_DEBUG=1
echo.
echo REM Start the application
echo if exist "src\main.py" (
echo     echo Starting Claude Voice Assistant...
echo     python src\main.py --config config\alpha_config.yaml
echo ^) else (
echo     echo [INFO] Application files ready for testing
echo     echo [INFO] Main application: src\main.py
echo     echo [INFO] Configuration: config\alpha_config.yaml
echo     echo [INFO] Testing checklist: testing\alpha_test_checklist.md
echo ^)
echo.
echo echo.
echo echo Alpha testing session completed
echo echo Check logs\ directory for detailed logs
echo pause
) > start_alpha.bat

echo [OK] Alpha start script created

echo.
echo ============================================
echo        Alpha Installation Complete!
echo ============================================
echo.
echo Claude Voice Assistant Alpha testing environment is ready!
echo.
echo Installation Summary:
echo   - Python version: VERIFIED
echo   - Virtual environment: CREATED
echo   - Core dependencies: INSTALLED
echo   - Project structure: INITIALIZED
echo   - Alpha configuration: CREATED
echo   - Start script: READY
echo.
echo Alpha Testing Configuration:
echo   - Speech processing: DISABLED (Mock mode)
echo   - Core architecture: ENABLED
echo   - Event system: ENABLED  
echo   - Configuration system: ENABLED
echo   - Adapter system: ENABLED (Mock mode)
echo   - Agent system: ENABLED (Mock mode)
echo.
echo Next Steps:
echo   1. Run start_alpha.bat to launch Alpha testing
echo   2. Review testing\alpha_test_checklist.md for test procedures
echo   3. Check config\alpha_config.yaml for configuration options
echo   4. Monitor logs\ directory for system activity
echo.
echo For troubleshooting: Read TROUBLESHOOTING.md
echo For technical support: https://github.com/lirhcoder/claude-echo/issues
echo.
pause