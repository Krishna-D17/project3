@echo off
setlocal enabledelayedexpansion

:: Determine project root directory
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

echo ===================================================
echo MediaProof - Starting Backend
echo ===================================================

if not exist "%PROJECT_DIR%.venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment .venv is not found!
    echo Please run setup.bat first to create the environment and install dependencies.
    exit /b 1
)

set "VENV_PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"

echo Starting FastAPI backend on http://127.0.0.1:8000 ...
"%VENV_PYTHON%" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
