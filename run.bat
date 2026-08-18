@echo off
setlocal EnableExtensions
chcp 65001 >nul

rem Always run from the folder that contains this file.
cd /d "%~dp0"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "VENV_DIR=.venv"
if not defined MAX_UPLOAD_PDF_BYTES set "MAX_UPLOAD_PDF_BYTES=52428800"
if not defined MAX_RESOURCE_BYTES set "MAX_RESOURCE_BYTES=20971520"
if not defined MAX_UPLOAD_PDF_PAGES set "MAX_UPLOAD_PDF_PAGES=500"
if not defined RESOURCE_DISCOVERY_TIMEOUT set "RESOURCE_DISCOVERY_TIMEOUT=10"
if not defined FAST_RESOURCE_DISCOVERY_TIMEOUT set "FAST_RESOURCE_DISCOVERY_TIMEOUT=6"
if not defined FAST_RESOURCE_DISCOVERY_WORKERS set "FAST_RESOURCE_DISCOVERY_WORKERS=8"
if not defined FAST_RESOURCE_DISCOVERY_BYTES set "FAST_RESOURCE_DISCOVERY_BYTES=6291456"

rem Default local LM Studio endpoint. The UI can override it.
if not defined LMSTUDIO_BASE_URL set "LMSTUDIO_BASE_URL=http://127.0.0.1:1234"
set "LMSTUDIO_MAX_TOKENS="
set "LMSTUDIO_MAX_RETRY_TOKENS="
set "LMSTUDIO_DISABLE_THINKING="
rem Never send local LM Studio traffic through a system/corporate proxy.
set "NO_PROXY=127.0.0.1,localhost,::1"
set "no_proxy=127.0.0.1,localhost,::1"
set "PY_LAUNCHER="
set "PY_ARGS="

call :detect_python
if not errorlevel 1 goto python_ready

echo.
echo Python 3.10 or newer was not found.
echo Trying to install Python 3.12 automatically with winget...
echo.

where winget >nul 2>&1
if errorlevel 1 goto no_python

winget install --id Python.Python.3.12 -e --scope user --accept-package-agreements --accept-source-agreements
if errorlevel 1 goto python_install_failed

call :detect_python
if errorlevel 1 goto python_path_not_refreshed

:python_ready
echo.
echo [1/3] Python is ready.

if exist "%VENV_DIR%\Scripts\python.exe" goto venv_ready

echo [2/3] Creating the virtual environment...
"%PY_LAUNCHER%" %PY_ARGS% -m venv "%VENV_DIR%"
if errorlevel 1 goto venv_failed

:venv_ready
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

if not exist "%VENV_PY%" goto venv_failed
if not exist "requirements.txt" goto requirements_missing
if not exist "app_optimized.py" goto app_missing

echo [2/3] Installing or updating required packages...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto dependency_failed
"%VENV_PY%" -m pip install -r "requirements.txt"
if errorlevel 1 goto dependency_failed

echo [3/3] Starting the Adaptive Learning Game...
echo Close this window or press Ctrl+C to stop the server.
echo.
echo Local generation: start LM Studio Local Server in the Developer tab.
echo Default LM Studio URL: %LMSTUDIO_BASE_URL%
echo LM Studio output tokens: uncapped by the app ^(model/context settings still apply^)
echo LM Studio Thinking: controlled only by the loaded model settings
echo PDF drag-and-drop limit: 100 MB per file in Streamlit; the app reads up to 50 MB per PDF by default.
echo Resource discovery: Fast mode is default ^(parallel search, top 10 candidates only^).
echo.
"%VENV_PY%" -m streamlit run "app_optimized.py" --server.maxUploadSize 100 --server.address 0.0.0.0
set "APP_EXIT=%ERRORLEVEL%"

if "%APP_EXIT%"=="0" goto done
echo.
echo The game stopped with error code %APP_EXIT%.
pause
exit /b %APP_EXIT%

:detect_python
set "PY_LAUNCHER="
set "PY_ARGS="

where py >nul 2>&1
if not errorlevel 1 (
    py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PY_LAUNCHER=py"
        set "PY_ARGS=-3"
        exit /b 0
    )
)

where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PY_LAUNCHER=python"
        exit /b 0
    )
)

for %%V in (313 312 311 310) do (
    if exist "%LocalAppData%\Programs\Python\Python%%V\python.exe" (
        "%LocalAppData%\Programs\Python\Python%%V\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
        if not errorlevel 1 (
            set "PY_LAUNCHER=%LocalAppData%\Programs\Python\Python%%V\python.exe"
            exit /b 0
        )
    )
)
exit /b 1

:no_python
echo.
echo Python could not be installed automatically because winget is unavailable.
echo Install Python 3.10 or newer from python.org, enable "Add Python to PATH",
echo and then run this file again.
pause
exit /b 1

:python_install_failed
echo.
echo Automatic Python installation failed.
echo Install Python 3.10 or newer manually and run this file again.
pause
exit /b 1

:python_path_not_refreshed
echo.
echo Python was installed, but this terminal cannot see it yet.
echo Close this window and double-click run.bat again.
pause
exit /b 1

:venv_failed
echo.
echo Could not create or use the .venv virtual environment.
echo Delete the .venv folder and run this file again.
pause
exit /b 1

:requirements_missing
echo.
echo requirements.txt was not found next to run.bat.
pause
exit /b 1

:app_missing
echo.
echo app_optimized.py was not found next to run.bat.
pause
exit /b 1

:dependency_failed
echo.
echo Installing Python packages failed. Check your internet connection,
echo proxy/firewall settings, and available disk space, then run again.
pause
exit /b 1

:done
endlocal
exit /b 0
