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
set "MODEL_ID=Qwen/Qwen2.5-3B-Instruct"
set "HF_MODEL_DIR=%ROOT%\models\Qwen2.5-3B-Instruct"
set "HF_HOME=%ROOT%\models\.hf_cache"

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

REM ---------- 1. Prerequisite check: Python (3.10-3.13 have prebuilt wheels) ----------
echo [1/8] Checking Python installation...
set "SELPY="
for %%V in (3.12 3.11 3.13 3.10) do (
    if not defined SELPY (
        py -%%V -c "import sys" >nul 2>&1
        if not errorlevel 1 set "SELPY=py -%%V"
    )
)
if not defined SELPY (
    python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>&1
    if not errorlevel 1 set "SELPY=python"
)
if not defined SELPY (
    echo [ERROR] No supported Python found. Install Python 3.11 or 3.12 from
    echo         https://www.python.org/downloads/  ^(3.14+ lacks prebuilt wheels for torch etc.^)
    pause
    exit /b 1
)
echo        Using: %SELPY%
echo.

REM ---------- 2. Create venv if missing (or unsupported version) ----------
echo [2/8] Preparing virtual environment...
if exist "%PY%" (
    "%PY%" -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>&1
    if errorlevel 1 (
        echo        Existing .venv uses an unsupported Python - recreating...
        rmdir /s /q "%VENV%"
    )
)
if not exist "%PY%" (
    echo        Creating .venv...
    %SELPY% -m venv "%VENV%"
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
    "%PY%" -m pip install --prefer-binary -e ".[dev]" --quiet
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

REM ---------- 4b. LLM dependencies + model download (Hugging Face) ----------
echo        Installing LLM dependencies from requirements.txt...
"%PY%" -m pip install --prefer-binary -r "%ROOT%\requirements.txt" --quiet
if errorlevel 1 (
    echo [ERROR] LLM dependency installation failed.
    pause
    exit /b 1
)
echo        Downloading %MODEL_ID% (about 6 GB; resumable, skipped if present)...
if not exist "%ROOT%\models" mkdir "%ROOT%\models"
"%PY%" -c "from huggingface_hub import snapshot_download; snapshot_download('%MODEL_ID%', local_dir=r'%HF_MODEL_DIR%')"
if errorlevel 1 (
    echo [ERROR] Model download failed. Check connection and free disk space.
    echo         If authentication is required set HF_TOKEN in .env or run: "%VENV%\Scripts\hf.exe" auth login
    pause
    exit /b 1
)
if exist "%ROOT%\.env.example" if not exist "%ROOT%\.env" (
    copy "%ROOT%\.env.example" "%ROOT%\.env" >nul
    echo        Created .env from .env.example - please review it.
)
echo.

REM ---------- 5. Ensure folders ----------
echo [5/8] Ensuring required folders exist...
if not exist "%ARTIFACTS%"  mkdir "%ARTIFACTS%"
if not exist "%REPORTS%"    mkdir "%REPORTS%"
if not exist "%ROOT%\governance\audit_trail\decisions"  mkdir "%ROOT%\governance\audit_trail\decisions"
if not exist "%ROOT%\governance\audit_trail\monitoring" mkdir "%ROOT%\governance\audit_trail\monitoring"
if not exist "%ROOT%\governance\audit_trail\changes"    mkdir "%ROOT%\governance\audit_trail\changes"
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
echo        Frontend: %API_URL%/planner
echo        Docs   : %API_URL%/docs
echo        Redoc  : %API_URL%/redoc
echo        Health : %API_URL%/health
echo.
echo        A new window will open. Keep it open while testing.
echo        This window will display live API responses.
echo.

REM Launch API in its own window (the frontend UI is served by the API at /planner)
start "Credit Risk API" cmd /k ""%UVICORN%" api.main:app --host %API_HOST% --port %API_PORT% --reload"

REM Wait until the API answers /health (up to ~90s), then open the frontend
echo        Waiting for API to become ready...
powershell -NoProfile -Command "$ok=$false; for($i=0;$i -lt 45;$i++){ try { Invoke-RestMethod -Uri '%API_URL%/health' -TimeoutSec 2 | Out-Null; $ok=$true; break } catch { Start-Sleep -Seconds 2 } }; if($ok){ exit 0 } else { exit 1 }"
if errorlevel 1 (
    echo [WARN] API did not respond within 90s - check the "Credit Risk API" window.
) else (
    echo        API is up. Opening frontend and API docs in browser...
    start "" "%ROOT%\frontend\credit_risk_agent_workflow_planner.html"
    start "" "%API_URL%/docs"
)

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
  "try { $body = @{ features = @{ loan_amnt = 15000.0; int_rate = 11.5; annual_inc = 62000.0; dti = 18.2; emp_length = 5; home_ownership = 'RENT' }; request_id = 'demo-001' } | ConvertTo-Json -Depth 5; Invoke-RestMethod -Uri '%API_URL%/predict' -Method POST -Body $body -ContentType 'application/json' -TimeoutSec 10 | ConvertTo-Json -Depth 5 } catch { Write-Host 'Predict failed:' $_.Exception.Message }"
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