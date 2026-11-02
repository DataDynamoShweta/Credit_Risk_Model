@echo off
setlocal EnableExtensions
title Credit Risk Model - Setup
cd /d "%~dp0"
set "VENV=%CD%\.venv"
set "PY=%VENV%\Scripts\python.exe"
set "MODEL_ID=Qwen/Qwen2.5-3B-Instruct"
set "MODEL_DIR=%CD%\models\Qwen2.5-3B-Instruct"

echo [1/5] Checking Python 3.10+...
python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.10 or newer is required and must be on PATH.
    echo         Install it from https://www.python.org/downloads/
    goto :fail
)

echo [2/5] Preparing virtual environment...
if not exist "%PY%" (
    python -m venv "%VENV%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        goto :fail
    )
) else (
    echo        Existing .venv found - reusing.
)

echo [3/5] Installing dependencies (torch is large; this can take a while)...
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto :pipfail
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :pipfail
"%PY%" -m pip install "huggingface_hub[cli]"
if errorlevel 1 goto :pipfail

echo [4/5] Downloading %MODEL_ID% (about 6 GB, resumable, skipped if present)...
if not exist "%CD%\models" mkdir "%CD%\models"
"%VENV%\Scripts\huggingface-cli.exe" download %MODEL_ID% --local-dir "%MODEL_DIR%"
if errorlevel 1 (
    echo [ERROR] Model download failed. Check your connection and free disk space.
    echo         If the model requires authentication, run:
    echo         "%VENV%\Scripts\huggingface-cli.exe" login
    goto :fail
)

echo [5/5] Configuring environment file...
if exist ".env.example" (
    if not exist ".env" (
        copy ".env.example" ".env" >nul
        echo        Created .env from .env.example - please review it.
    ) else (
        echo        .env already exists - left unchanged.
    )
)

echo.
echo Setup complete. Run run.bat to start the API.
pause
endlocal
exit /b 0

:pipfail
echo [ERROR] Dependency installation failed. See the output above.
:fail
echo.
echo Setup FAILED.
pause
endlocal
exit /b 1
