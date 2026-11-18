@echo off
REM ---------------------------------------------------------------------------
REM PathSnap evaluator UI launcher - LAPTOP 2
REM
REM Usage:  scripts\run_ui.bat http://192.168.1.20:8000     (Laptop 1's API address)
REM         scripts\run_ui.bat                               (defaults to http://localhost:8000)
REM
REM Needs only ui\requirements.txt installed (streamlit, streamlit-folium, folium, requests),
REM either in .venv or in the Python on PATH.
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0.."

if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

if not "%~1"=="" set "PATHSNAP_API_URL=%~1"
if "%PATHSNAP_API_URL%"=="" set "PATHSNAP_API_URL=http://localhost:8000"
if "%PATHSNAP_UI_PORT%"=="" set "PATHSNAP_UI_PORT=8501"

echo.
echo ==== PathSnap UI ====
echo API      : %PATHSNAP_API_URL%
echo Open     : http://localhost:%PATHSNAP_UI_PORT%   in a browser on this laptop
echo Press CTRL+C to stop.
echo.

python -m streamlit run ui\app.py --server.headless true --server.port %PATHSNAP_UI_PORT% --browser.gatherUsageStats false
endlocal
