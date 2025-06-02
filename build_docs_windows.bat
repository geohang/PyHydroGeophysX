@echo off
echo ========================================
echo PyHydroGeophysX Documentation Builder
echo ========================================

REM Set environment variables
set PYDEVD_DISABLE_FILE_VALIDATION=1
set PYTHONFROZENMODULES=off

python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

echo.
echo 1. Installing documentation requirements...
pip install sphinx==7.1.2 sphinx-rtd-theme==1.3.0 myst-parser==2.0.0 nbsphinx==0.9.1 sphinx-copybutton==0.5.2 sphinx-gallery==0.14.0
if errorlevel 1 (
    echo Error: Failed to install documentation requirements
    exit /b 1
)

echo.
echo 2. Generating API documentation...
sphinx-apidoc -f -o docs\source\api PyHydroGeophysX
if errorlevel 1 (
    echo Error: Failed to generate API documentation
    exit /b 1
)

echo.
echo 3. Building documentation locally...
cd docs
sphinx-build -b html source build\html
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
echo Local preview: docs\build\html\index.html
echo Online version: Will be available after GitHub Pages setup
echo.

set /p choice="Open local preview now? (y/n): "
if /i "%choice%"=="y" (
    start docs\build\html\index.html
)

echo.
echo NEXT STEPS TO PUBLISH ONLINE:
echo 1. git add .
echo 2. git commit -m "Setup GitHub Pages documentation"
echo 3. git push origin main
echo 4. Enable GitHub Pages in repository settings
echo 5. Your docs will be live at: https://yourusername.github.io/PyHydroGeophysX/