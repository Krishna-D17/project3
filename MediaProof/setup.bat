@echo off
setlocal enabledelayedexpansion

:: Determine project root directory
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

echo ===================================================
echo MediaProof - Step 1 Setup and Verification
echo ===================================================

:: Check Python availability (prefer Python 3.11 executable if found)
set "BASE_PYTHON="

if exist "C:\Users\krish\AppData\Roaming\uv\python\cpython-3.11.14-windows-x86_64-none\python.exe" (
    set "BASE_PYTHON=C:\Users\krish\AppData\Roaming\uv\python\cpython-3.11.14-windows-x86_64-none\python.exe"
) else (
    py -3.11 --version >nul 2>&1
    if !errorlevel! equ 0 (
        set "BASE_PYTHON=py -3.11"
    ) else (
        py --version >nul 2>&1
        if !errorlevel! equ 0 (
            set "BASE_PYTHON=py"
        ) else (
            python --version >nul 2>&1
            if !errorlevel! equ 0 (
                set "BASE_PYTHON=python"
            )
        )
    )
)

if "%BASE_PYTHON%"=="" (
    echo [ERROR] Python is not found. Please install Python 3.11 before running setup.bat.
    exit /b 1
)

echo Base Python launcher: %BASE_PYTHON%

:: Create .venv if it does not exist
if not exist "%PROJECT_DIR%.venv" (
    echo Creating virtual environment in .venv ...
    %BASE_PYTHON% -m venv "%PROJECT_DIR%.venv"
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment .venv already exists.
)

set "VENV_PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment Python executable not found at %VENV_PYTHON%
    exit /b 1
)

echo Using venv Python: %VENV_PYTHON%

:: Upgrade pip
echo Upgrading pip inside .venv ...
"%VENV_PYTHON%" -m pip install --upgrade pip
if !errorlevel! neq 0 (
    echo [ERROR] Failed to upgrade pip inside .venv.
    exit /b 1
)

:: Install requirements.txt
echo Installing requirements.txt ...
"%VENV_PYTHON%" -m pip install -r "%PROJECT_DIR%requirements.txt"
if !errorlevel! neq 0 (
    echo [ERROR] Failed to install requirements.txt inside .venv.
    exit /b 1
)

:: Install requirements-dev.txt
echo Installing requirements-dev.txt ...
"%VENV_PYTHON%" -m pip install -r "%PROJECT_DIR%requirements-dev.txt"
if !errorlevel! neq 0 (
    echo [ERROR] Failed to install requirements-dev.txt inside .venv.
    exit /b 1
)

:: Run environment check
echo.
echo Running environment verification script ...
"%VENV_PYTHON%" "%PROJECT_DIR%scripts\check_environment.py"
if !errorlevel! neq 0 (
    echo [ERROR] Environment verification failed.
    exit /b 1
)

:: Run pytest
echo.
echo Running test suite ...
"%VENV_PYTHON%" -m pytest "%PROJECT_DIR%tests"
if !errorlevel! neq 0 (
    echo [ERROR] Pytest execution failed.
    exit /b 1
)

echo.
echo ===================================================
echo MediaProof Setup completed successfully!
echo ===================================================
exit /b 0
