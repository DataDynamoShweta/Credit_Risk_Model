@echo off
setlocal EnableDelayedExpansion
title Credit Risk Model - Setup, Train, Validate & Serve
color 0B

REM ============================================================
REM  Credit Risk Model Validation Platform - Windows Launcher
REM  Author  : DataDynamoShweta
REM  Purpose : One-click setup, train, validate, serve & demo
REM ============================================================

REM ---------- 0. Paths ----------
cd /d "%~dp0"
set "ROOT=%CD%"
set "VENV=%ROOT%\.venv"
set "PY=%VENV%\Scripts\python.exe"
set "PIP=%VENV%\Scripts\pip.exe"
set "UVICORN=%VENV%\Scripts\uvicorn.exe"
set "ARTIFACTS=%ROOT%\artifacts"
set "REPORTS=%ROOT%\reports"
set "REPORT_FILE=%REPORTS%\validation_report.md"
set "MODEL_FILE=%ARTIFACTS%\champion.joblib"
set "API_HOST=127.0.0.1"
set "API_PORT=8000"
set "API_URL=http://%API_HOST%:%API_PORT%"

REM ---------- Banner ----------
echo.
echo  ============================================================
echo    CREDIT RISK MODEL VALIDATION PLATFORM - Launcher
echo  ============================================================
echo    Root      : %ROOT%
echo    Venv      : %VENV%
echo    Artifacts : %ARTIFACTS%
echo    Reports   : %REPORTS%
echo    API       : %API_URL%
echo  ============================================================
echo.

REM ---------- 1. Prerequisite check: Python ----------
echo [1/8] Checking Python installation...
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not on PATH.
    echo         Install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PYVER=%%v"
echo        Detected Python %PYVER%
echo.

REM ---------- 2. Create venv if missing ----------
echo [2/8] Preparing virtual environment...
if not exist "%PY%" (
    echo        Creating .venv...
    python -m venv "%VENV%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo        Existing .venv found - reusing.
)
echo.

REM ---------- 3. Upgrade pip ----------
echo [3/8] Upgrading pip...
"%PY%" -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [ERROR] pip upgrade failed.
    pause
    exit /b 1
)
echo        pip upgraded.
echo.

REM ---------- 4. Install project + dev deps ----------
echo [4/8] Installing project dependencies...
if exist "%ROOT%\pyproject.toml" (
    "%PY%" -m pip install -e ".[dev]" --quiet
) else if exist "%ROOT%\requirements.txt" (
    "%PY%" -m pip install -r "%ROOT%\requirements.txt" --quiet
) else (
    echo [ERROR] Neither pyproject.toml nor requirements.txt found.
    pause
    exit /b 1
)
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)
echo        Dependencies installed.
echo.

REM ---------- 5. Ensure folders ----------
echo [5/8] Ensuring required folders exist...
if not exist "%ARTIFACTS%"  mkdir "%ARTIFACTS%"
if not exist "%REPORTS%"    mkdir "%REPORTS%"
if not exist "%ROOT%\governance\audit_trail\decisions"  mkdir "%ROOT%\governance\audit_trail\decisions"
if not exist "%ROOT%\governance\audit_trail\monitoring" mkdir "%ROOT%\governance\audit_trail\monitoring"
echo        Folders ready.
echo.

REM ---------- 6. Train champion model ----------
echo [6/8] Training champion model...
if exist "%MODEL_FILE%" (
    echo        Existing model found - retraining to keep artefacts fresh...
)
if exist "%ROOT%\pipelines\train.py" (
    "%PY%" "%ROOT%\pipelines\train.py" --output-dir "%ARTIFACTS%"
    if errorlevel 1 (
        echo [ERROR] Training failed. Check the log above.
        pause
        exit /b 1
    )
) else (
    echo [WARN] pipelines\train.py not found - skipping training.
)
if not exist "%MODEL_FILE%" (
    echo [ERROR] Expected model artefact not produced: %MODEL_FILE%
    pause
    exit /b 1
)
echo        Champion model ready: %MODEL_FILE%
echo.

REM ---------- 7. Run validation suite ----------
echo [7/8] Running validation suite...
if exist "%ROOT%\pipelines\validate.py" (
    "%PY%" "%ROOT%\pipelines\validate.py" ^
        --model "%MODEL_FILE%" ^
        --report "%REPORT_FILE%"
    if errorlevel 1 (
        echo [WARN] Validation pipeline returned non-zero. Continuing anyway.
    )
) else (
    echo [WARN] pipelines\validate.py not found - skipping validation.
)

REM ---------- 8. Show model results summary ----------
echo.
echo  ============================================================
echo    MODEL RESULTS - Validation Report Summary
echo  ============================================================
if exist "%REPORT_FILE%" (
    echo    Report : %REPORT_FILE%
    echo    --------------------------------------------------------
    REM Print first 40 lines of the report (executive summary area)
    powershell -NoProfile -Command "Get-Content -LiteralPath '%REPORT_FILE%' -TotalCount 40"
    echo    --------------------------------------------------------
) else (
    echo    [WARN] No validation report found at %REPORT_FILE%.
)
echo.

REM ---------- 9. Launch API ----------
echo [8/8] Starting FastAPI service...
echo        URL    : %API_URL%
echo        Docs   : %API_URL%/docs
echo        Redoc  : %API_URL%/redoc
echo        Health : %API_URL%/health
echo.
echo        A new window will open. Keep it open while testing.
echo        This window will display live API responses.
echo.

REM Open Swagger UI in the default browser after a short delay
start "" /min cmd /c "timeout /t 4 >nul & start %API_URL%/docs"

REM Launch API in its own window
start "Credit Risk API" cmd /k ""%UVICORN%" api.main:app --host %API_HOST% --port %API_PORT% --reload"

REM Give the API time to boot
echo        Waiting for API to become ready...
timeout /t 8 /nobreak >nul

REM ---------- 10. Smoke tests ----------
echo.
echo  ============================================================
echo    SMOKE TESTS
echo  ============================================================
echo.
echo  [A] GET /health
powershell -NoProfile -Command "try { (Invoke-RestMethod -Uri '%API_URL%/health' -TimeoutSec 10) | ConvertTo-Json -Depth 4 } catch { Write-Host 'Health check failed:' $_.Exception.Message }"
echo.

echo  [B] GET /model-card
powershell -NoProfile -Command "try { (Invoke-RestMethod -Uri '%API_URL%/model-card' -TimeoutSec 10) | ConvertTo-Json -Depth 4 } catch { Write-Host 'Model card failed:' $_.Exception.Message }"
echo.

echo  [C] POST /predict  (sample application)
powershell -NoProfile -Command ^
  "try { $body = @{ features = @{ loan_amnt = 15000; int_rate = 11.5; annual_inc = 62000; dti = 18.2; emp_length = 5 }; request_id = 'demo-001' } | ConvertTo-Json -Depth 5; Invoke-RestMethod -Uri '%API_URL%/predict' -Method POST -Body $body -ContentType 'application/json' -TimeoutSec 10 | ConvertTo-Json -Depth 5 } catch { Write-Host 'Predict failed:' $_.Exception.Message }"
echo.

echo  ============================================================
echo    DONE
echo  ============================================================
echo.
echo  The API is running in the "Credit Risk API" window.
echo  Close that window when finished, or press any key to end.
echo.
pause
endlocal
exit /b 0