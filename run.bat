@echo off
setlocal EnableExtensions
title Credit Risk Model - API
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Run setup.bat first.
    goto :fail
)
echo Activating virtual environment...
call ".venv\Scripts\activate.bat"

set "HF_HOME=%CD%\models\.hf_cache"
set "HF_MODEL_DIR=%CD%\models\Qwen2.5-3B-Instruct"
set "HF_HUB_OFFLINE=1"

echo Starting FastAPI at http://localhost:8000  (docs: /docs)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
if errorlevel 1 (
    echo [ERROR] The API exited with an error. See the log above.
    goto :fail
)
endlocal
exit /b 0

:fail
pause
endlocal
exit /b 1
