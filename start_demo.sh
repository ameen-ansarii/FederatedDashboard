#!/usr/bin/env bash
# ========================================================
#   Federated Learning-Based Education System (FL-EdSys)
#   macOS & Linux One-Command Launcher
# ========================================================

echo "========================================================"
echo "  Federated Learning-Based Education System (FL-EdSys)"
echo "========================================================"
echo ""

# 1. Determine Python 3 executable
if command -v python3 &> /dev/null; then
    PY_CMD="python3"
elif command -v python &> /dev/null; then
    PY_CMD="python"
else
    echo "[ERROR] Python 3 is not found on this machine!"
    echo "Please install Python 3.10+ from https://python.org"
    exit 1
fi

echo "Using: $($PY_CMD --version)"
echo "Checking required dependencies..."

# 2. Check and auto-install minimal Python packages if missing
$PY_CMD -c "import fastapi, uvicorn, torch, pandas" &> /dev/null
if [ $? -ne 0 ]; then
    echo "First-time setup: Installing required packages from requirements.txt..."
    $PY_CMD -m pip install -r requirements.txt
fi

echo ""
echo "========================================================"
echo "  DASHBOARD RUNNING AT: http://127.0.0.1:8000"
echo "  Both Frontend and PyTorch AI Engine are active!"
echo "  Browser will open automatically in 3 seconds..."
echo "  (Keep this terminal window open while presenting)"
echo "========================================================"
echo ""

# 3. Open browser after brief delay
(
    sleep 3
    if command -v open &> /dev/null; then
        open "http://127.0.0.1:8000"
    elif command -v xdg-open &> /dev/null; then
        xdg-open "http://127.0.0.1:8000"
    fi
) &

# 4. Start the server
$PY_CMD backend_lite.py
