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

REM Install minimal dependencies for Alpha testing
echo.
echo [4/5] Installing Alpha testing dependencies...
echo [INFO] Installing core dependencies only (no speech processing)
pip install pydantic loguru pyyaml aiofiles requests aiohttp numpy
if %errorlevel% neq 0 (
    echo [ERROR] Core dependencies installation failed
    pause
    exit /b 1
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

REM Create test configuration using Python
echo.
echo Creating Alpha test configuration...
python -c "import yaml; import os; from pathlib import Path; config = {'system': {'log_level': 'INFO', 'environment': 'alpha_testing', 'enable_mock_mode': True}, 'speech': {'enabled': False, 'mock_mode': True}, 'adapters': {'claude_code': {'enabled': True, 'mock_mode': True}}}; config_path = Path('config/alpha_config.yaml'); config_path.parent.mkdir(exist_ok=True); yaml.dump(config, open(config_path, 'w', encoding='utf-8'), allow_unicode=True); print('[OK] Alpha configuration created')"

if %errorlevel% neq 0 (
    echo [WARNING] Configuration creation failed, using defaults
)

REM Create Alpha start script
echo.
echo Creating Alpha start script...
(
echo @echo off
echo chcp 65001 ^>nul
echo title Claude Voice Assistant Alpha
echo echo.
echo echo ============================================
echo echo  Claude Voice Assistant Alpha Testing
echo echo ============================================
echo echo.
echo echo [INFO] Alpha version - Core architecture testing only
echo echo [INFO] Speech processing disabled for compatibility
echo echo.
echo if not exist "venv\" (
echo     echo [ERROR] Virtual environment not found
echo     echo Please run install_alpha.bat first
echo     pause
echo     exit /b 1
echo ^)
echo call venv\Scripts\activate.bat
echo set PYTHONPATH=src;%%PYTHONPATH%%
echo set CLAUDE_VOICE_ENV=alpha_testing
echo if exist "src\main.py" (
echo     python src\main.py --config config\alpha_config.yaml
echo ^) else (
echo     echo [INFO] Main application not ready yet - Alpha environment configured
echo     echo [INFO] Project structure initialized for core testing
echo ^)
echo echo.
echo echo Alpha testing environment ready
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
echo Configuration:
echo   - Core dependencies: INSTALLED
echo   - Speech processing: DISABLED (Alpha focus on architecture)
echo   - Mock mode: ENABLED
echo   - Testing environment: READY
echo.
echo Next steps:
echo   1. Run start_alpha.bat to test the environment
echo   2. Read testing\alpha_test_checklist.md for testing procedures
echo   3. Check logs\ directory for any issues
echo.
echo Alpha testing focus:
echo   - Core architecture validation
echo   - Configuration system testing
echo   - Basic functionality verification
echo   - Error handling and logging
echo.
echo For technical support:
echo   - GitHub Issues: https://github.com/lirhcoder/claude-echo/issues
echo.
pause