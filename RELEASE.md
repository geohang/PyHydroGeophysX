# PyPI Release Guide for PyHydroGeophysX v0.3.0

## Unreleased

- Added `spd_cholesky` and `spd_cg` to `generalized_solver` for a square
  symmetric positive definite matrix, and changed the default `method` of
  `TimeLapseERTInversion`, `SRTInversion` and `TimeLapseSRTInversion` from
  `cgls` to `spd_cholesky`. Those three assemble the Gauss-Newton normal matrix
  `H = J^T W_d^T W_d J + lambda W_m^T W_m + ...` and pass it to the solver,
  but every method available until now was a least-squares method, so on such a
  matrix the solver worked on `H^T H d = H^T (-g)`: the condition number was
  squared, and each iteration cost two matrix-vector products where one would
  do. A fixed iteration budget then bought only the leading Krylov directions,
  which belong to the data term, so the result looked insensitive to the
  regularization weight. On a synthetic step with the data term outweighing the
  regularization, a hundred-fold change in lambda moved the CGLS update by 2.9%
  at 50 iterations and 33% at 300, against 94% for the exact solve;
  `scripts/lambda_sweep_solver_check.py` reproduces the table. **Pass
  `method='cgls'` to reproduce results from a run before this change.** The two
  inversions that pass a stacked least-squares system, `ERTInversion` and
  `JointERTSRTInversion`, keep their least-squares defaults, which were already
  correct. `generalized_solver` itself still defaults to `cgls`. The desktop
  studio's time-lapse pipeline (`DEFAULT_TL` in `inversion/_time_lapse_workflow`)
  passes `method` straight through, so its default moved too; the ADTLERT branch
  still forces `cgls`, where that string selects that backend's own GPU solver
  rather than anything in `solvers/linear_solvers`. Above 15000 model unknowns
  that pipeline auto-enables sparse mode, and the sparse `spd_cholesky` path is
  SuperLU, since SciPy has no sparse Cholesky.
- `generalized_solver` now warns once per process when a least-squares method is
  handed a square symmetric matrix. Detection is a shape check followed, only
  for a square matrix, by a two-pair random bilinear probe, so a stacked system
  pays one integer comparison.
- Added a keyword-only `overwrite_a` to `generalized_solver`, used by
  `spd_cholesky` to factor in the caller's own buffer instead of allocating a
  copy. The three normal-matrix inversions pass it, since nothing reads `H`
  after the solve. The tradeoff is that a partial factorization has already
  destroyed the matrix by the time a failure is detected, so that case raises
  rather than falling back; retry with `overwrite_a=False` or `spd_cg`.
- `TimeLapseSRTInversion` gained `target_chi_squared`, `convergence_tolerance`
  and `min_iterations` parameters. Its convergence test used to hard-code 1.5
  and 0.01 with no minimum-iteration guard, so a flat second iteration could end
  the inversion at iteration three. The first two defaults reproduce the old
  thresholds; `min_iterations` defaults to 5, matching `TimeLapseERTInversion`.
- Repaired the `cholesky` branch of `direct_solver` for sparse input. It called
  `scipy.sparse.linalg.cholesky`, which does not exist, so it raised
  `AttributeError` on every call, a bare `except` swallowed it, and it printed
  "Matrix not SPD" whether or not the matrix was. It now uses SuperLU. The dense
  branch's bare `except` was narrowed to `LinAlgError`.
- Added ADTLERT as an optional differentiable 2.5D ERT backend for the unified
  single-time and windowed time-lapse inversion pipelines, including shared GPU
  state, unified CuPy CUDA 12 installation and cuDSS acceleration on Windows
  and Linux, and fallback to the original PyHydro ERT engine when CUDA or
  cuDSS is unavailable. The slower SciPy forward solver is intentionally
  disabled; Linux is recommended for the best performance. Surveys with
  remote electrodes encoded as negative ABMN indices safely retain the original
  PyHydro engine because ADTLERT 0.1 cannot represent those electrodes.
- Consolidated optional-backend failures under the public
  `PyHydroGeophysX.BackendUnavailable` base class. Gravity/magnetics inversion
  failures now inherit from it, so one `except BackendUnavailable` handler can
  cover all numerical backends.
- Added agent UX safeguards: dry-run workflow preview, dict-compatible `AgentResult`, clearer file validation errors, and transparent quality-loop status.
- Updated the Streamlit app with default no-key demo mode, bundled cached ERT/joint-demo outputs, and a mandatory parsed-config confirmation step before execution.
- Reordered agent documentation toward user entry points and added troubleshooting guidance for common setup, data, and LLM failures.
- Added LLM token/cost accounting in agent ledgers and surfaced estimated cost in the Streamlit workflow UI.
- Added `examples/Ex_hello_agent.py` and notebook as a no-API-key local ERT hello-world path.

This guide walks you through publishing the updated PyHydroGeophysX package to PyPI.

## Prerequisites

1. **Install build tools:**
   ```powershell
   pip install --upgrade build twine
   ```

2. **PyPI account setup:**
   - Create account at https://pypi.org/account/register/
   - Create account at https://test.pypi.org/account/register/ (for testing)
   - Set up API tokens for secure uploads

3. **Configure PyPI credentials:**
   Create `~/.pypirc` file:
   ```ini
   [distutils]
   index-servers =
       pypi
       testpypi

   [pypi]
   username = __token__
   password = pypi-YOUR-API-TOKEN-HERE

   [testpypi]
   username = __token__
   password = pypi-YOUR-TEST-API-TOKEN-HERE
   ```

## Pre-Release Checklist

- [x] Version bumped to 0.3.0 in `setup.py` and `pyproject.toml`
- [x] CHANGELOG.md updated with new features
- [x] README.md updated with ERT data processing examples
- [x] Documentation updated (`docs/source/api/data_processing.rst`)
- [x] Dependencies updated (resipy>=3.4.0 added)
- [ ] All tests passing
- [ ] Documentation builds successfully
- [ ] Git repository clean (all changes committed)

## Step 1: Clean Previous Builds

Remove old build artifacts:
```powershell
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue
```

## Step 2: Build Distribution Packages

Build both source distribution and wheel:
```powershell
python -m build
```

This creates:
- `dist/PyHydroGeophysX-0.3.0.tar.gz` (source distribution)
- `dist/PyHydroGeophysX-0.3.0-py3-none-any.whl` (wheel)

## Step 3: Test on TestPyPI (Recommended)

Upload to TestPyPI first to verify everything works:
```powershell
python -m twine upload --repository testpypi dist/*
```

Install from TestPyPI to test:
```powershell
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ PyHydroGeophysX==0.3.0
```

Test the installation:
```powershell
python -c "from PyHydroGeophysX.data_processing.ert_data_agent import load_ert_resipy; print('Import successful!')"
```

## Step 4: Upload to PyPI

Once testing is successful, upload to the real PyPI:
```powershell
python -m twine upload dist/*
```

You'll see output like:
```
Uploading distributions to https://upload.pypi.org/legacy/
Uploading PyHydroGeophysX-0.3.0-py3-none-any.whl
Uploading PyHydroGeophysX-0.3.0.tar.gz
```

## Step 5: Verify Installation

Install from PyPI:
```powershell
pip install --upgrade PyHydroGeophysX
```

Verify version and new features:
```python
import PyHydroGeophysX
print(PyHydroGeophysX.__version__)  # Should print: 0.3.0

from PyHydroGeophysX.data_processing.ert_data_agent import load_ert_resipy
print("New ERT data processing module imported successfully!")
```

## Step 6: Create GitHub Release

1. **Commit and tag the release:**
   ```powershell
   git add .
   git commit -m "Release v0.3.0: Add ERT data processing module"
   git tag -a v0.3.0 -m "Version 0.3.0: ERT data processing with RESIPY integration"
   git push origin main
   git push origin v0.3.0
   ```

2. **Create GitHub Release:**
   - Go to https://github.com/geohang/PyHydroGeophysX/releases/new
   - Select tag: v0.3.0
   - Release title: "PyHydroGeophysX v0.3.0 - ERT Data Processing"
   - Description: Copy from CHANGELOG.md
   - Attach distribution files from `dist/` folder
   - Publish release

## Step 7: Update Documentation

Build and deploy documentation:
```powershell
cd docs
.\make.bat clean
.\make.bat html
```

If using GitHub Pages:
```powershell
# Copy built docs to gh-pages branch
git checkout gh-pages
Copy-Item -Recurse -Force docs\build\html\* .
git add .
git commit -m "Update docs for v0.3.0"
git push origin gh-pages
git checkout main
```

## Post-Release Tasks

1. **Announce the release:**
   - Update project homepage
   - Post on social media/mailing lists
   - Update any relevant forums or communities

2. **Monitor for issues:**
   - Watch GitHub issues for installation problems
   - Check PyPI download statistics
   - Monitor documentation feedback

3. **Update development version:**
   Consider bumping to 0.3.0-dev in main branch:
   ```python
   # In setup.py and pyproject.toml
   version = "0.3.0.dev0"
   ```

## Troubleshooting

### Import Error: Missing Dependencies
If users report missing dependencies, they may need to install extras:
```powershell
pip install PyHydroGeophysX[geophysics]  # For full features including RESIPY
```

### Documentation Not Building
Check Sphinx dependencies:
```powershell
pip install -r docs/requirements.txt
```

### Upload Fails with Authentication Error
- Verify API token is correct in `~/.pypirc`
- Check token hasn't expired
- Ensure token has upload permissions

### Package Not Found on PyPI
- Wait a few minutes for PyPI to index
- Clear pip cache: `pip cache purge`
- Check spelling and version number

## Version History

- **v0.3.0** (2025): Added ERT data processing, multi-agent workflows, EM forward/inversion, visualization, uncertainty modules
- **v0.2.0** (2025-11-06): Added ERT data processing module with RESIPY integration
- **v0.1.0** (2024): Initial release with core functionality

## Additional Resources

- PyPI Documentation: https://packaging.python.org/
- Twine Documentation: https://twine.readthedocs.io/
- Semantic Versioning: https://semver.org/
- Keep a Changelog: https://keepachangelog.com/
