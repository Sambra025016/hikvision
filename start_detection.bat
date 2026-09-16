@echo off
setlocal
cd /d "%~dp0"

set "PROJECT_PYTHON=%CD%\.venv\Scripts\python.exe"

if not exist "%PROJECT_PYTHON%" (
    echo [Setup] Creating the .venv virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [Error] Could not create .venv. Make sure Python is available.
        exit /b 1
    )
)

"%PROJECT_PYTHON%" -c "import requests, numpy, cv2, matplotlib, motion_app" >nul 2>nul
if errorlevel 1 (
    echo [Setup] Installing video detection dependencies...
    "%PROJECT_PYTHON%" -m pip install --upgrade "pip==24.0"
    if errorlevel 1 (
        echo [Error] Could not upgrade pip for Python 3.7.
        echo [Hint] Check PIP_NO_INDEX and HTTP_PROXY/HTTPS_PROXY settings.
        exit /b 1
    )
    "%PROJECT_PYTHON%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [Error] Dependency installation failed.
        echo [Hint] Check PIP_NO_INDEX and HTTP_PROXY/HTTPS_PROXY settings.
        exit /b 1
    )
)

"%PROJECT_PYTHON%" -m motion_app.app %*
exit /b %errorlevel%
