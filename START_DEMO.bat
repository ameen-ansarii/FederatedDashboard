@echo off
setlocal enabledelayedexpansion
title Federated Learning Education System
cd /d "%~dp0"

echo ========================================================
echo   Federated Learning-Based Education System (FL-EdSys)
echo ========================================================
echo.
echo Preparing system for presentation...
echo.

:: 1. Detect Python (test if executable actually runs, avoids broken copied venvs)
set "PY_BIN="

if exist "%~dp0.venv_win\Scripts\python.exe" (
    "%~dp0.venv_win\Scripts\python.exe" --version >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        set "PY_BIN=%~dp0.venv_win\Scripts\python.exe"
        echo Using local Windows environment: .venv_win
    )
)

if "!PY_BIN!"=="" if exist "%~dp0venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" --version >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        set "PY_BIN=%~dp0venv\Scripts\python.exe"
        echo Using local environment: venv
    )
)

if "!PY_BIN!"=="" if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" --version >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        set "PY_BIN=%~dp0.venv\Scripts\python.exe"
        echo Using local environment: .venv
    )
)

if "!PY_BIN!"=="" (
    python --version >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        set "PY_BIN=python"
    ) else (
        py --version >nul 2>&1
        if !ERRORLEVEL! equ 0 (
            set "PY_BIN=py"
        )
    )
)

:: If Python is completely missing from this laptop
if "!PY_BIN!"=="" (
    echo [ERROR] Python is not installed on this computer!
    echo.
    echo Opening https://www.python.org/downloads in your browser...
    start https://www.python.org/downloads/
    echo.
    echo Step 1: Download and install Python 3.10 or higher.
    echo Step 2: CRITICAL - Ensure you check "Add Python to PATH" on the installer!
    echo Step 3: Once installed, double-click START_DEMO.bat again.
    echo.
    pause
    exit /b 1
)

:: 2. Check and auto-install minimal Python packages if missing
echo Checking required dependencies...
"!PY_BIN!" -c "import fastapi, uvicorn, torch, pandas" >nul 2>&1
if !ERRORLEVEL! neq 0 (
    echo.
    echo [First-time setup detected]
    echo Installing required packages from requirements.txt...
    echo (Downloading PyTorch and dependencies may take 1-2 minutes, please wait...)
    echo.
    "!PY_BIN!" -m pip install -r requirements.txt
    if !ERRORLEVEL! neq 0 (
        echo.
        echo [ERROR] Package installation failed. Check your internet connection.
        echo.
        pause
        exit /b 1
    )
)

echo.
echo ========================================================
echo   DASHBOARD RUNNING AT: http://127.0.0.1:8000
echo   Both Frontend and PyTorch AI Engine are active!
echo   Browser will open automatically in 3 seconds...
echo   (Keep this window open while presenting)
echo ========================================================
echo.

:: Launch browser in background after 3 seconds
start "" /b cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8000"

:: Start the application
"!PY_BIN!" backend_lite.py
pause
