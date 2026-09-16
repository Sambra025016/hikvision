@echo off
setlocal
cd /d "%~dp0"

rem Use a project-local virtual environment to keep dependencies consistent.
set "PROJECT_PYTHON=%CD%\.venv\Scripts\python.exe"

if not exist "%PROJECT_PYTHON%" (
    echo [Setup] Creating the .venv virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [Error] Could not create .venv. Make sure Python is available.
        exit /b 1
    )
)

rem Install the PTZ package only when its imports are unavailable.
"%PROJECT_PYTHON%" -c "import requests, hikvision" >nul 2>nul
if errorlevel 1 (
    echo [Setup] Installing project dependencies...
    "%PROJECT_PYTHON%" -m pip install -e .
    if errorlevel 1 (
        echo [Error] Dependency installation failed. Check pip and the network.
        exit /b 1
    )
)

"%PROJECT_PYTHON%" main.py %*
exit /b %errorlevel%
