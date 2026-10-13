@echo off
title Federated Learning Education System - Launcher
echo ========================================================
echo   Federated Learning-Based Education System (FL-EdSys)
echo ========================================================
echo.

:: 1. Verify Python installation
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in your system PATH!
    echo Please install Python 3.10+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

:: 2. Verify Node.js installation
node --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Node.js is not installed or not in your system PATH!
    echo Please install Node.js (LTS version) from https://nodejs.org
    echo.
    pause
    exit /b 1
)

echo [1/3] Checking Backend Python dependencies...
python -c "import fastapi, uvicorn, torch, pandas" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Installing required backend packages from requirements.txt...
    pip install -r requirements.txt
) else (
    echo Backend dependencies OK.
)

echo.
echo [2/3] Checking Frontend Node.js dependencies...
if not exist "frontend\node_modules" (
    echo Installing frontend node_modules (npm install)...
    cd frontend
    call npm install
    cd ..
) else (
    echo Frontend node_modules OK.
)

echo.
echo [3/3] Starting Backend & Frontend servers...
echo.

:: Launch Backend in its own window
start "FL-EdSys Backend (Port 8000)" cmd /k "python backend_lite.py"

:: Launch Frontend in its own window
start "FL-EdSys Frontend (Port 5173)" cmd /k "cd frontend && npm run dev"

echo ========================================================
echo   System is now starting!
echo   - Backend:  http://127.0.0.1:8000
echo   - Frontend: http://localhost:5173
echo ========================================================
echo.
echo Both terminal windows are running. You can close this launcher window.
timeout /t 5 >nul
