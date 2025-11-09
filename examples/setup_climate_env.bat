@echo off
REM Setup script for climate data fetching environment
REM This creates a separate conda environment with PyDaymet 0.19+ and NumPy 2.x

echo ====================================================================
echo Setting up Climate Data Fetching Environment
echo ====================================================================
echo.
echo This environment uses:
echo   - Python 3.10+
echo   - NumPy 2.x
echo   - PyDaymet 0.19+
echo   - pandas 2.x
echo.
echo This is separate from the main PyHydroGeophysX environment
echo which uses NumPy 1.x for PyGIMLi compatibility.
echo.
echo ====================================================================

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: conda not found. Please install Anaconda or Miniconda.
    exit /b 1
)

REM Create new environment
echo.
echo [1/4] Creating conda environment 'climate_fetch'...
conda create -n climate_fetch python=3.10 -y
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create conda environment
    exit /b 1
)

REM Activate environment and install packages
echo.
echo [2/4] Installing NumPy 2.x and dependencies...
call conda activate climate_fetch
pip install "numpy>=2.0" "pandas>=2.0" "pydaymet>=0.19"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install packages
    exit /b 1
)

REM Verify installation
echo.
echo [3/4] Verifying installation...
python -c "import numpy; import pandas; import pydaymet; print(f'NumPy: {numpy.__version__}'); print(f'pandas: {pandas.__version__}'); print(f'PyDaymet: {pydaymet.__version__}')"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Package verification failed
    exit /b 1
)

echo.
echo [4/4] Creating example configuration file...
echo { > climate_config_example.json
echo   "coords": [-105.3, 40.0], >> climate_config_example.json
echo   "dates": ["2021-09-01", "2021-11-30"], >> climate_config_example.json
echo   "output": "data/climate/climate_data.csv", >> climate_config_example.json
echo   "crs": 4326, >> climate_config_example.json
echo   "variables": ["prcp", "tmin", "tmax", "srad", "vp", "dayl"], >> climate_config_example.json
echo   "pet_method": "penman_monteith", >> climate_config_example.json
echo   "time_scale": "daily", >> climate_config_example.json
echo   "region": "na", >> climate_config_example.json
echo   "antecedent_days": [1, 3, 7, 14], >> climate_config_example.json
echo   "pet_params": { >> climate_config_example.json
echo     "arid_correction": false >> climate_config_example.json
echo   } >> climate_config_example.json
echo } >> climate_config_example.json

echo.
echo ====================================================================
echo Setup completed successfully!
echo ====================================================================
echo.
echo To fetch climate data, run:
echo   conda activate climate_fetch
echo   python fetch_climate_data.py --config climate_config_example.json
echo.
echo Or use command-line arguments:
echo   python fetch_climate_data.py --coords -105.3 40.0 --dates 2021-09-01 2021-11-30 --output climate_data.csv
echo.
echo ====================================================================

call conda deactivate
