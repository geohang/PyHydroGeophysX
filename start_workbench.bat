@echo off
setlocal EnableExtensions
title PyHydroGeophysX Desktop Workbench

rem Always resolve paths relative to this file, not the directory from which
rem Explorer or a terminal happened to start the launcher.
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_DIR=%%~fI"
set "VENV_DIR=%REPO_DIR%\.venv-workbench"
set "PYTHON_EXE="
set "PYTHON_ARGS="
set "FALLBACK_EXE="
set "FALLBACK_ARGS="

rem Let a source checkout run without being installed first.
if defined PYTHONPATH (
    set "PYTHONPATH=%REPO_DIR%;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%REPO_DIR%"
)

echo ========================================
echo PyHydroGeophysX Desktop Workbench
echo ========================================
echo.

if not exist "%REPO_DIR%\PyHydroGeophysX\qt_apps\launcher.py" (
    echo [ERROR] The workbench was not found under:
    echo         "%REPO_DIR%"
    echo.
    echo Keep start_workbench.bat in the examples folder of the downloaded
    echo PyHydroGeophysX source package.
    goto :failed
)

rem Prefer the reusable environment created by this launcher.
call :try_python "%VENV_DIR%\Scripts\python.exe"
if defined PYTHON_EXE goto :launch

rem An advanced user may explicitly select an interpreter.
if defined PYHYDROGEOPHYSX_PYTHON (
    call :try_python "%PYHYDROGEOPHYSX_PYTHON%"
    if defined PYTHON_EXE goto :launch
)

rem Prefer an activated conda environment, when the launcher inherits one.
if defined CONDA_PREFIX (
    call :try_python "%CONDA_PREFIX%\python.exe"
    if defined PYTHON_EXE goto :launch
)

rem Try the standard Windows Python launcher and commands.
where py.exe >nul 2>&1
if not errorlevel 1 (
    py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
    if not errorlevel 1 (
        if not defined FALLBACK_EXE (
            set "FALLBACK_EXE=py.exe"
            set "FALLBACK_ARGS=-3"
        )
        py -3 -c "import PySide6, pyqtgraph" >nul 2>&1
        if not errorlevel 1 (
            set "PYTHON_EXE=py.exe"
            set "PYTHON_ARGS=-3"
            goto :launch
        )
    )
)

for %%P in (python.exe python3.exe) do (
    where %%P >nul 2>&1
    if not errorlevel 1 (
        call :try_python "%%P"
        if defined PYTHON_EXE goto :launch
    )
)

rem Conda is often not on PATH when a .bat file is opened from Explorer.
rem Search its common per-user and system environment locations.
for %%R in (
    "%USERPROFILE%\.conda\envs"
    "%USERPROFILE%\miniconda3\envs"
    "%USERPROFILE%\anaconda3\envs"
    "%LOCALAPPDATA%\miniconda3\envs"
    "%LOCALAPPDATA%\anaconda3\envs"
    "%ProgramData%\miniconda3\envs"
    "%ProgramData%\anaconda3\envs"
) do (
    if exist "%%~R" (
        for /d %%D in ("%%~R\*") do (
            call :try_python "%%~fD\python.exe"
            if defined PYTHON_EXE goto :launch
        )
    )
)

rem PySide6 was not found, but a suitable Python may be available. Create an
rem isolated environment so the user's existing environments are not changed.
if not defined FALLBACK_EXE goto :no_python

echo The desktop dependencies (PySide6, pyqtgraph) are not installed in an
echo available Python environment. Creating a reusable workbench environment:
echo   "%VENV_DIR%"
echo.
"%FALLBACK_EXE%" %FALLBACK_ARGS% -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo.
    echo [ERROR] Python could not create the workbench environment.
    goto :failed
)

echo Installing PyHydroGeophysX desktop dependencies...
echo This one-time setup can take several minutes.
echo.
"%VENV_DIR%\Scripts\python.exe" -m pip install --disable-pip-version-check -e "%REPO_DIR%[desktop]"
if errorlevel 1 (
    echo.
    echo [ERROR] Dependency installation failed. Check the messages above and
    echo         your internet connection, then double-click this file again.
    goto :failed
)

echo.
echo NOTE: this environment has the interface only. Two more groups are optional:
echo   3D viewers (Mesh 3D, velocity volume):
echo     "%VENV_DIR%\Scripts\python.exe" -m pip install -e "%REPO_DIR%[desktop-3d]"
echo   Forward modeling and inversion engines:
echo     "%VENV_DIR%\Scripts\python.exe" -m pip install -e "%REPO_DIR%[geophysics]"
echo.

set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "PYTHON_ARGS="

:launch
echo Using Python:
"%PYTHON_EXE%" %PYTHON_ARGS% -c "import sys; print('  ' + sys.executable)"
echo.
echo Starting the workbench. The window opens in a moment.
echo Keep this window open while using the app: it carries the startup log and
echo any error message. Closing the workbench closes this window too.
echo ========================================
echo.

rem Used by automated checks; normal double-clicks never set this variable.
if /i "%PHGX_WORKBENCH_DRY_RUN%"=="1" exit /b 0

"%PYTHON_EXE%" %PYTHON_ARGS% -m PyHydroGeophysX.qt_apps.launcher %*
if errorlevel 1 (
    echo.
    echo [ERROR] The workbench stopped unexpectedly. See the message above.
    goto :failed
)
exit /b 0

:try_python
if not exist "%~1" (
    where "%~1" >nul 2>&1
    if errorlevel 1 exit /b 0
)
"%~1" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if errorlevel 1 exit /b 0
if not defined FALLBACK_EXE (
    set "FALLBACK_EXE=%~1"
    set "FALLBACK_ARGS="
)
"%~1" -c "import PySide6, pyqtgraph" >nul 2>&1
if errorlevel 1 exit /b 0
set "PYTHON_EXE=%~1"
set "PYTHON_ARGS="
exit /b 0

:no_python
echo [ERROR] Python 3.9 or newer was not found.
echo.
echo Install Python from https://www.python.org/downloads/windows/
echo Make sure "Add Python to PATH" is selected, then double-click this
echo file again.
goto :failed

:failed
echo.
pause
exit /b 1
