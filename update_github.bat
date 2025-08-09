@echo off
chcp 65001 >nul
REM Simple Git update script

echo.
echo ======================================
echo  Updating GitHub Repository
echo ======================================
echo.

REM Add all files
git add .

REM Create commit
git commit -m "Update Claude Voice Assistant - fix encoding issues in batch files"

REM Push to repository
echo Pushing to GitHub...
echo Please enter your GitHub credentials when prompted
echo Username: lirhcoder
echo Password: Use your Personal Access Token

git push origin main

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Repository updated successfully
    echo Repository: https://github.com/lirhcoder/claude-echo
) else (
    echo.
    echo [ERROR] Push failed
    echo Check your credentials and network connection
)

echo.
pause