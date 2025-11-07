# Troubleshooting Guide

## Common Issues and Solutions

### NumPy Compatibility Error with pyGIMLi

**Error Message:**
```
Buffer dtype mismatch, expected 'long' but got 'long long'
```

OR

```
ImportError: DLL load failed while importing _multiarray_umath: The specified module could not be found.
ImportError: numpy._core.multiarray failed to import
```

**Cause:**
PyGIMLi 1.5.4 was compiled against NumPy 1.x and is not compatible with NumPy 2.x due to ABI changes. The second error occurs when you have mixed DLL files from different NumPy versions.

**Solution:**
1. Downgrade NumPy and pydaymet to compatible versions:

```bash
conda run -p <your_env_path> pip install "numpy==1.26.4" "pydaymet<0.19" --force-reinstall
```

2. For Windows PowerShell:
```powershell
$env:CONDA_ENV = "C:\Users\<your_username>\.conda\envs\<env_name>"
conda run -p $env:CONDA_ENV pip install "numpy==1.26.4" "pydaymet<0.19" --force-reinstall
```

**Note:** PyDaymet 0.19+ requires NumPy 2.x, but pyGIMLi requires NumPy 1.x. Use pydaymet 0.17-0.18 for compatibility.

**Verification:**
Check your NumPy version:
```python
import numpy as np
print(np.__version__)  # Should be 1.26.4
```

---

### PyProj DeprecationWarning

**Warning Message:**
```
DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated
```

**Cause:**
This is a known deprecation warning from pyproj when using newer NumPy versions. It doesn't affect functionality but will become an error in future NumPy releases.

**Solution:**
This warning is harmless and can be ignored. It will be fixed in future versions of pyproj.

---

### RESIPY Permission Error on Windows

**Error Message:**
```
PermissionError: [WinError 5] Access is denied: 'project_dir'
```

**Cause:**
RESIPY tries to remove/recreate the project directory, which can fail on Windows (especially with OneDrive or network drives).

**Solution:**
The code automatically falls back to a temporary directory with a warning. You can also:

1. Use a local directory (not OneDrive/network drive)
2. Close any applications that might have files open in the project directory
3. Run the command with administrator privileges

---

### Missing Dependencies

**Error Message:**
```
ImportError: pydaymet is required for climate data retrieval
```

**Solution:**
Install the required package:

```bash
pip install pydaymet
```

For all optional dependencies:
```bash
pip install PyHydroGeophysX[geophysics,climate]
```

---

### API Key Issues

**Error Message:**
```
Warning: API key for openai not found in environment variables
```

**Solution:**
Set the appropriate environment variable:

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY = "your-key-here"
```

**Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=your-key-here
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY='your-key-here'
```

Or set it in your Python code (not recommended for shared notebooks):
```python
import os
os.environ['OPENAI_API_KEY'] = 'your-key-here'
```

---

### Climate Data Integration Error

**Error Message:**
```
Usecols do not match columns, columns expected but not found: [0, 3]
```

**Cause:**
There is a three-way compatibility conflict:
- PyGIMLi 1.5.4 requires NumPy 1.x (incompatible with NumPy 2.x)
- PyDaymet 0.19+ requires NumPy 2.x (incompatible with pyGIMLi)
- PyDaymet 0.17.1 (last version supporting NumPy 1.x) has bugs with pandas 2.3+

**Temporary Solution:**
Disable climate data integration in your workflow:

```python
workflow_config = {
    'use_climate': False,  # Disable climate integration
    # ... other config
}
```

**Long-term Solution:**
Wait for PyGIMLi to release a NumPy 2.x compatible version, then:
```bash
pip install "pydaymet>=0.19" "numpy>=2.0"
```

**Alternative (for climate-only workflows):**
Create a separate environment without pyGIMLi:
```bash
conda create -n climate python=3.11
conda activate climate
pip install pydaymet>=0.19 numpy>=2.0
```

---

## Environment Setup

### Recommended Package Versions

For stable operation, we recommend:

- Python: 3.11.x
- NumPy: 1.26.4 (not 2.x)
- pyGIMLi: 1.5.4
- RESIPY: 3.6.3
- pydaymet: 0.19.4

### Complete Installation

```bash
# Create conda environment
conda create -n pyhydro python=3.11

# Activate environment
conda activate pyhydro

# Install pygimli first (requires specific NumPy version)
conda install -c gimli -c conda-forge pygimli

# Install PyHydroGeophysX with all features
pip install PyHydroGeophysX[geophysics,climate]

# Verify NumPy version (should be 1.26.x)
python -c "import numpy; print(numpy.__version__)"
```

---

## Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/geohang/PyHydroGeophysX/issues)
2. Open a new issue with:
   - Error message (full traceback)
   - Environment info (`conda list` or `pip freeze`)
   - Minimal reproducible example
3. Consult the [documentation](https://geohang.github.io/PyHydroGeophysX/)
