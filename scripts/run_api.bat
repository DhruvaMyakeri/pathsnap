@echo off
REM ---------------------------------------------------------------------------
REM PathSnap API launcher - LAPTOP 1 (model + HMM + server-side scoring)
REM
REM Usage:  scripts\run_api.bat                      serve the student_test trips
REM         scripts\run_api.bat "D:\eval_trips"      serve the trips in that folder
REM
REM Optional environment variables (set them before running to override):
REM   PATHSNAP_EVAL_DIR   folder with the trip sub-folders to serve
REM   PATHSNAP_PORT       port to listen on            (default 8000)
REM   PATHSNAP_USE_HMM    0 disables HMM decoding       (default: on)
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0.."

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] .venv not found in %CD%
    echo         Create it first - see README.md, section Installation.
    exit /b 1
)
call ".venv\Scripts\activate.bat"

if not "%~1"=="" set "PATHSNAP_EVAL_DIR=%~1"
if "%PATHSNAP_PORT%"=="" set "PATHSNAP_PORT=8000"

echo.
echo ==== PathSnap API ====
if defined PATHSNAP_EVAL_DIR echo Eval dir : %PATHSNAP_EVAL_DIR%
if not defined PATHSNAP_EVAL_DIR echo Eval dir : default student_test partition
echo Listening: http://0.0.0.0:%PATHSNAP_PORT%   - Laptop 2 uses http://THIS-LAPTOP-IP:%PATHSNAP_PORT%
echo Find this laptop's IP with:  ipconfig   (IPv4 Address of the Wi-Fi adapter)
echo Press CTRL+C to stop.
echo.

python -m uvicorn api.main:app --host 0.0.0.0 --port %PATHSNAP_PORT%
endlocal
