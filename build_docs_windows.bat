@echo off
echo ========================================
echo PyHydroGeophysX Documentation Builder
echo ========================================

REM Set environment variables to avoid frozen module warnings
set PYDEVD_DISABLE_FILE_VALIDATION=1
set PYTHONFROZENMODULES=off


python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

echo.
echo 1. Installing documentation requirements...
pip install -r docs\requirements.txt
if errorlevel 1 (
    echo Error: Failed to install documentation requirements
    exit /b 1
)

echo.
echo 2. Generating API documentation...
sphinx-apidoc -f -o docs\source\api PyHydroGeophysX PyHydroGeophysX\examples
if errorlevel 1 (
    echo Error: Failed to generate API documentation
    exit /b 1
)

echo.
echo 3. Building documentation...
cd docs
call make.bat clean
call make.bat html
if errorlevel 1 (
    echo Error: Failed to build HTML documentation
    cd ..
    exit /b 1
)
cd ..

echo.
echo ========================================
echo Documentation built successfully!
echo ========================================
echo.
echo Open: docs\build\html\index.html

set /p choice="Open documentation now? (y/n): "
if /i "%choice%"=="y" (
    start docs\build\html\index.html
)