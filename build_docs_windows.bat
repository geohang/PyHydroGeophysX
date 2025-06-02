@echo off
echo ========================================
echo PyHydroGeophysX Documentation Builder
echo ========================================

REM Set environment variables
set PYDEVD_DISABLE_FILE_VALIDATION=1
set PYTHONFROZENMODULES=off
set SPHINX_GALLERY_PLOT=False

python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

echo.
echo 1. Installing documentation requirements...
pip install sphinx==7.1.2 sphinx-rtd-theme==1.3.0 myst-parser==2.0.0 nbsphinx==0.9.1 sphinx-copybutton==0.5.2 sphinx-gallery==0.14.0 numpy scipy matplotlib tqdm palettable
if errorlevel 1 (
    echo Error: Failed to install documentation requirements
    exit /b 1
)

echo.
echo 2. Creating required directories...
if not exist docs\source\_static mkdir docs\source\_static

echo.
echo 3. Generating API documentation...
sphinx-apidoc -f -o docs\source\api PyHydroGeophysX
if errorlevel 1 (
    echo Error: Failed to generate API documentation
    exit /b 1
)

echo.
echo 4. Building documentation locally...
cd docs
sphinx-build -b html source build\html --keep-going
if errorlevel 1 (
    echo Warning: Documentation build completed with some errors
    echo Check the output above for details
)
cd ..

echo.
echo ========================================
echo Documentation build completed!
echo ========================================
echo.
echo Local preview: docs\build\html\index.html
echo.

set /p choice="Open local preview now? (y/n): "
if /i "%choice%"=="y" (
    start docs\build\html\index.html
)

echo.
echo NEXT STEPS TO PUBLISH ONLINE:
echo 1. Apply all the code fixes provided in the documentation
echo 2. git add .
echo 3. git commit -m "Fix documentation build issues"
echo 4. git push origin main
echo 5. Your docs will be live at: https://geohang.github.io/PyHydroGeophysX/