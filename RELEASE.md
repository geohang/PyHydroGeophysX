# PyPI Release Guide for PyHydroGeophysX v0.3.0

## Unreleased

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
