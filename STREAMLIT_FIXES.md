# Streamlit Cloud Deployment Fixes

## Problem Summary

The Streamlit Cloud deployment was failing with PyGIMLi critical errors:
```
CRITICAL - response for model with negative or zero resistivity is not defined.: 2.68309e-81 1.75386e+31
```

This occurred when PyGIMLi's ERT forward modeling received extreme or invalid resistivity values during inversion.

## Root Causes Identified

1. **Petrophysical conversion numerical instability** - Division by zero in conductivity-to-resistivity conversion
2. **Unconstrained line search** - Model bounds were not enforced during optimization iterations
3. **Initial model validation gaps** - Negative or zero initial values were not properly handled
4. **Missing input validation** - No checks before calling PyGIMLi forward modeling

## Fixes Implemented

### 1. Petrophysical Conversion Protection
**File**: `PyHydroGeophysX/petrophysics/resistivity_models.py`

**Changes**:
- Added minimum saturation threshold (0.001) to prevent division by zero
- Added minimum conductivity threshold (1e-6) before `1/sigma` conversion
- Added resistivity bounds clipping (0.1 to 1e6 ohm-m)

**Functions fixed**:
- `water_content_to_resistivity()` (lines 58-87)
- `WS_Model()` (lines 23-58)

### 2. Line Search Constraint Enforcement
**File**: `PyHydroGeophysX/inversion/ert_inversion.py`

**Changes**:
- **CRITICAL FIX**: Added `np.clip(mr1, min_mr, max_mr)` INSIDE the line search loop (after line 225)
- This ensures forward modeling never receives out-of-bounds log-resistivity values
- Added validation for non-finite values (NaN/Inf) with automatic correction

**Location**: Line 221-234 (line search while loop)

### 3. Initial Model Validation
**File**: `PyHydroGeophysX/inversion/ert_inversion.py`

**Changes**:
- Replaced unsafe `log(initial_model + 1e-6)` with `log(abs(initial_model) + 1.0)`
- Added immediate constraint enforcement after initial model creation
- Added warning message when non-positive values are detected

**Location**: Lines 145-165

### 4. Forward Modeling Input Validation
**File**: `PyHydroGeophysX/forward/ert_forward.py`

**Changes**:
- Added pre-call validation in `ertforward2()`:
  - Check for NaN/Inf values
  - Check for exp overflow (log-resistivity bounds: -20 to 20)
  - Validate all resistivity values are positive
  - Clip resistivity to safe range (0.001 to 1e6)

- Same validation added to `ertforandjac2()` and `ERTForwardModeling.forward()`

**Locations**: Lines 292-325, 337-375, 50-89

### 5. Chi-Squared Array-to-Scalar Conversion
**Files**: `PyHydroGeophysX/inversion/ert_inversion.py`, `PyHydroGeophysX/agents/ert_inversion_agent.py`

**Problem**:
Matrix multiplication `fdert = W.T @ W` produces (1,1) shaped array, not a scalar. When `chi2_ert = fdert / len(dr)` is stored in `iteration_chi2` list and later accessed with `float(iteration_chi2[-1])`, it raises `TypeError: only 0-dimensional arrays can be converted to Python scalars`.

**Changes**:
- In `ert_inversion.py` (line 191-195): Convert chi2_ert to scalar immediately after calculation
  ```python
  chi2_ert = fdert / len(dr)
  if isinstance(chi2_ert, np.ndarray):
      chi2_ert = float(chi2_ert.item())
  ```
- In `ert_inversion_agent.py` (lines 145, 151, 162): Use safe conversion for robustness
  ```python
  float(np.asarray(inversion_result.iteration_chi2[-1]).item())
  ```

**Locations**: `ert_inversion.py` lines 191-195, `ert_inversion_agent.py` lines 145, 151, 162

### 6. UnboundLocalError for Numpy Variable
**File**: `PyHydroGeophysX/agents/base_agent.py`

**Problem**:
`UnboundLocalError: cannot access local variable 'np' where it is not associated with a value` at line 1002. This occurs because Python found local `import numpy as np` statements inside the `run_unified_agent_workflow()` function (at lines 335, 384, 731). When Python sees a local assignment/import to a variable name anywhere in a function, it treats that name as local for the ENTIRE function scope, making it unbound before the import statement.

**Changes**:
- Removed all local `import numpy as np` statements inside `run_unified_agent_workflow()` (lines 335, 384, 731)
- The module-level import at line 12 is sufficient and already provides `np` to all functions

**Why this happened**:
The local imports were added conditionally (e.g., `if workflow_config.get('climate_data'): import numpy as np`), but this makes `np` a local variable throughout the function, causing errors when `np` is accessed before reaching those conditional blocks.

**Locations**: `base_agent.py` lines 335, 384, 731 (removed local imports)

### 7. TypeError for Dictionary Format Strings
**File**: `PyHydroGeophysX/agents/base_agent.py`

**Problem**:
`TypeError: unsupported format string passed to dict.__format__` at line 1051. Petrophysical parameters are stored as nested dictionaries like `{'mean': 1.3, 'std': 0.1}`, but the formatting code assumed they were simple floats. When Python tried to format `params['rho_sat']:.1f`, it attempted to apply a float format specifier to a dictionary, causing the error.

**Additional issues**:
- Line 1073: Formatting `chi2` with `.3f` when it might be the string `'N/A'`
- Lines 1260-1262: Similar issues in TDEM report template with chi2, conductivity_range, and resistivity_range

**Changes**:
- Added helper function `get_param_value()` to extract numeric values from nested parameter dictionaries (handles both `{'mean': X}` structure and plain floats)
- Applied safe formatting for chi2 values (check if numeric before formatting)
- Applied safe formatting for range values (check if list/array exists before formatting)

**Locations**: `base_agent.py` lines 1045-1068 (params extraction), 1070-1087 (chi2 formatting), 1247-1278 (TDEM report formatting)

### 8. Missing Reciprocal Filtering and Double K Computation (High Chi-Squared)
**File**: `PyHydroGeophysX/data_processing/ert_data_agent.py`

**Problem 1 - Missing Reciprocal Filtering**:
On Streamlit Cloud (without ResIPy), the chi-squared values were extremely high (~1,320,000 vs ~120 locally) because reciprocal error filtering was not being applied. When ResIPy is unavailable, the embedded DAS-1 parser doesn't compute reciprocal errors, so bad measurements with incorrect error estimates remained in the dataset.

**Problem 2 - Wrong Order: Reciprocal Filtering Before K Computation**:
The reciprocal filtering was being applied BEFORE computing geometric factors K:
1. Line 1553: `reciprocalProcessing()` called with k=1 (placeholder)
2. Line 1568: K computed AFTER reciprocal processing
3. Line 1586: `rhoa = R * K` computed

This is catastrophic because:
- `reciprocalProcessing()` compares `rhoa` between normal/reciprocal pairs
- At line 1553, all `rhoa` values are just raw resistance R (with k=1)
- Reciprocal pairs have **identical R** but **different K** values
- After K computation, forward and reciprocal rhoa differ by K_ratio = K_fwd / K_recip
- The function thinks reciprocal error is huge and filters out 465/945 measurements (49%!)

**Problem 3 - Geometric Factor Applied Twice** (now fixed by reordering):
Previously, `rhoa = R * K` was computed twice (before and after filtering), but this was solved by moving reciprocal processing to after K computation.

**Evidence from logs**:
- **Local (with ResIPy)**: 945 measurements → 930 after reciprocal filtering, chi² = 120 → 0.76
- **Cloud (embedded parser)**: 945 measurements → 945 (no filtering!), chi² = 4,534,542 (extremely high!)

**Root cause**:
1. Wrong function name: Used `ert_pg.filterReciprocal()` which doesn't exist in PyGIMLi
2. Correct function is `ert_pg.reciprocalProcessing(data, maxrec=0.05, maxerr=0.2)`
3. Double K application: `rhoa = R * K` computed twice in the workflow

**Changes**:
1. Fixed function name: Changed `filterReciprocal()` to `reciprocalProcessing()`

2. **CRITICAL**: Moved reciprocal filtering to AFTER K computation and rhoa calculation:
```python
# WRONG ORDER (before fix):
data = ert_pg.load(path)
data = reciprocalProcessing(data, ...)  # Called with k=1, wrong rhoa!
data['k'] = createGeometricFactors(data)
data['rhoa'] = data['r'] * data['k']

# CORRECT ORDER (after fix):
data = ert_pg.load(path)
data['k'] = createGeometricFactors(data)  # Compute K first
data['rhoa'] = data['r'] * data['k']      # Then compute correct rhoa
data = reciprocalProcessing(data, ...)    # THEN filter with correct rhoa
```

**Why order matters**:
- ResIPy computes K automatically during data loading, so reciprocal processing sees correct rhoa
- Embedded parser writes k=1 placeholder, so K must be computed before reciprocal processing
- If reciprocal processing runs with k=1, it compares R values but thinks they're rhoa values
- Reciprocal pairs with K_fwd=100, K_recip=50 and identical R=1.0 would show:
  - Before K: rhoa_fwd = 1.0, rhoa_recip = 1.0 (correct, no error)
  - After wrong-order K: rhoa_fwd = 100, rhoa_recip = 50 (50% reciprocal error!)
  - After correct-order K: rhoa_fwd = 100, rhoa_recip = 100 (correct, no error)

**Why this works**:
- PyGIMLi's `reciprocalProcessing()` averages normal/reciprocal pairs and sets proper error estimates
- Removes measurements with reciprocal error > 5% and total error > 20%
- Applying K only ONCE ensures chi² = sum((d_obs - d_model)²/error²) is computed correctly
- Previously chi² was inflated by K² because both d_obs and d_model had extra K factor

**Expected result**:
- Cloud deployment should now filter ~15 measurements (matching local behavior)
- Final dataset: ~930 measurements instead of 945 (NOT 480!)
- Chi² should be ~100-120 initially, converging to ~0.5-1.0 within 3-4 iterations

**Locations**: `ert_data_agent.py` lines 1547-1590 (reordered K computation before reciprocal processing)

## Why These Fixes Work

### Multiple Defense Layers
The fixes create a defensive programming approach with validation at multiple stages:

1. **Petrophysics layer** - Prevents invalid conversions from water content/saturation
2. **Inversion setup** - Ensures initial model is valid
3. **Iteration layer** - Enforces constraints at every line search step
4. **Forward modeling layer** - Final validation before PyGIMLi calls

### Critical Line Search Fix
The most important fix is enforcing constraints **during** the line search, not after:

**Before (BROKEN)**:
```python
while True:
    mr1 = mr + mu_LS * d_mr  # Can violate bounds!
    dr = ertforward2(...)     # PyGIMLi gets invalid values → ERROR
    ...
mr = np.clip(mr, min_mr, max_mr)  # Too late!
```

**After (FIXED)**:
```python
while True:
    mr1 = mr + mu_LS * d_mr
    mr1 = np.clip(mr1, min_mr, max_mr)  # ✓ Enforced BEFORE forward call
    dr = ertforward2(...)                # PyGIMLi gets valid values
    ...
```

## Testing Recommendations

### Local Testing
```bash
# 1. Run Streamlit locally
streamlit run examples/app_geophysics_workflow.py

# 2. Test with Standard ERT example
# Use the "Standard ERT" button and run the workflow

# 3. Monitor console for errors
# Should see NO PyGIMLi CRITICAL errors
```

### Streamlit Cloud Testing
1. Push these changes to GitHub
2. Deploy to Streamlit Cloud
3. Test workflows through the web interface
4. Check deployment logs for successful completion

## Expected Behavior After Fixes

✅ **No PyGIMLi CRITICAL errors** - All resistivity values will be valid and positive
✅ **Stable inversion** - Line search respects bounds at every iteration
✅ **Graceful handling of edge cases** - Warnings instead of crashes for problematic data
✅ **Numerical stability** - No more extreme values like 1e-81 or 1e+31

## Files Modified

| File | Lines Changed | Purpose |
|------|---------------|---------|
| `resistivity_models.py` | 23-58, 58-87 | Prevent division by zero in petrophysics |
| `ert_inversion.py` | 145-165, 191-199, 221-234 | Validate initial model, convert chi2 to scalar, enforce constraints in line search |
| `ert_forward.py` | 50-89, 292-325, 337-375 | Validate inputs before PyGIMLi calls |
| `ert_inversion_agent.py` | 145, 151, 162 | Handle chi2 array-to-scalar conversion safely |
| `base_agent.py` | 335, 384, 731, 1045-1068, 1070-1087, 1247-1278 | Fix numpy imports, dict formatting, safe value extraction |
| `ert_data_agent.py` | 1553, 1570-1572 | Fix reciprocal filtering function name, remove double K computation |

## Additional Notes

### Data Loading
The embedded DAS-1 parser in `ert_data_agent.py` is working correctly. The error was NOT from data loading but from numerical instability during inversion.

### Resipy Dependency
The app uses fallback parsers (embedded from ResIPy GPL-3.0) when the full Resipy package isn't available. This is already implemented and working.

### Default Constraints
The default model constraints are `(0.001, 1e4)` ohm-m (line 29 in base_agent.py). In log-space, this is approximately `(-6.9, 9.2)`. The fixes ensure these bounds are always respected.

## Contact

If issues persist after these fixes, check:
1. API key is correctly set in Streamlit Cloud secrets
2. All dependencies are installing correctly (check deployment logs)
3. Data files are being uploaded correctly

---
**Generated**: 2026-01-19
**Author**: Claude Sonnet 4.5 via PyHydroGeophysX debugging session
