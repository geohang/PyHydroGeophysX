# PyHydroGeophysX improvement roadmap

This roadmap comes from a full repository audit (2026-07-05, working tree at commit `aa02fee`) covering the core package, the multi-agent workflow, the Streamlit app, and the Qt desktop workbench. It is written from the point of view of an external user who finds the project on GitHub, installs it, and tries to run it. Items are ordered by user impact. File references use repository-relative paths.

## Fixed on branch `improve/oss-hygiene` (2026-07-05)

- Stopped tracking `docs/build/` (732 files) and `examples/results/` (1,381 files) in git; extended `.gitignore` so result artifacts no longer pollute `git status` on every app run.
- Removed the stale `examples/Ex_Time_lapse_measurement-Hang's MacBook Air.ipynb`.
- Bumped CI actions to `actions/checkout@v5` and `actions/setup-python@v6` (Node.js 24 runner requirement).
- Added a desktop entry point: `pyhydrogeophysx-workbench` launches the Qt workbench after install (`[project.gui-scripts]` in `pyproject.toml`).
- Replaced retired Anthropic model IDs: sidebar default `claude-3-5-sonnet-20241022` (retired 2025-10-28) and `agents/base_agent.py` default `claude-3-opus-20240229` (retired 2026-01-05) are now `claude-sonnet-5`; the Qt chat model list gained `claude-sonnet-5` as default.
- Updated `agents/_pricing.py` with current Claude and GPT-4.1 rates so cost previews stop falling back to a generic rate.
- Added a clear message when the Auto (LLM) run path is used without an initialized context agent, and a warning when a workflow runs with no API key set (keyless quick modes still work by design).
- README: removed the deleted `Ex_hello_agent` row, added a "Running the apps" section, noted the PyPI version lag.
- Added `CITATION.cff` so GitHub shows the "Cite this repository" button.

## P0: highest impact, do next

### 1. Publish v0.3.0 to PyPI
PyPI currently serves v0.1.0 with an MIT classifier, while the repo is at v0.3.0 under Apache-2.0. Anyone following the README against a pip install gets a package that does not match the documentation.

```bash
python -m pip install --upgrade build twine
python -m build            # from the repo root, produces dist/
python -m twine check dist/*
python -m twine upload dist/*
```

Before uploading: confirm `pyproject.toml` version, tag the release (`git tag v0.3.0`), and consider a GitHub Release that triggers the Zenodo DOI archive. After uploading, verify `pip install pyhydrogeophysx==0.3.0` in a fresh environment and check the classifiers on the PyPI page.

### 2. Create a real test suite and run it in CI
There is no `tests/` directory, and `.github/workflows/tests.yml` only checks that the package imports. Suggested start, no conda needed:

- `tests/test_petrophysics.py`: Archie / Waxman-Smits round trips, Hertz-Mindlin velocity values against known numbers.
- `tests/test_interpolation.py`: `core/interpolation.py` regular-grid fast path vs `griddata` reference on a small synthetic grid.
- `tests/test_io_utils.py`: `qt_apps/io_utils.py` load and write helpers (pure numpy/stdlib).
- `tests/test_ert_parsers.py`: the no-resipy unified ERT parser in `data_processing/ert_data_agent.py` against the small BERT/E4D samples in `examples/data/ERT/`.
- `tests/test_pricing.py`: `agents/_pricing.py` lookup and fallback behavior.

Then add `pytest -q` to the workflow. A second CI job using `mamba-org/setup-micromamba` can install pygimli for forward/inversion smoke tests on a tiny mesh.

### 3. Make the shipped examples runnable
- `examples/Ex_ERT_workflow.py` references `data/modflow/`, which does not exist under `examples/data/`. Either add a small sample dataset or point the example at `examples/data/parflow/` equivalents.
- Committed notebooks embed absolute Windows paths (`C:\Users\...`) in their outputs. Re-run them with repository-relative paths, or adopt `nbstripout` as a pre-commit hook so outputs stay out of git.
- Keep the README examples table in sync with the files that actually exist (one stale row was removed in this pass; a small CI check could grep the table against `examples/`).

## P1: robustness and maintainability

### 4. Qt workbench: global exception handler
`qt_apps/launcher.py` installs no `sys.excepthook`. An uncaught exception in a slot or worker callback kills the app with a console traceback that Windows users never see. Install an excepthook that logs the traceback and shows a QMessageBox with a "copy details" button.

### 5. Bridge hardening (Streamlit and Qt)
- Writes to `qt_bridge/*.json` are not atomic; Streamlit polls `full_workbench_result.json` every 3 s and can read a half-written file. Write to a temp file in the same directory, then `os.replace()`.
- A standalone Qt launch (no `--context`) defaults the bridge directory to `Path.cwd()/qt_bridge` (`qt_apps/state.py`), which scatters files wherever the app happened to start. Default to a stable location (for example `~/.pyhydrogeophysx/qt_bridge` or the repo `results/` dir when detectable) and log the chosen path at startup.

### 6. Qt workbench: persistence and error messages
- Save and restore window geometry, dock layout, and recent data paths with `QSettings` (`main_window.__init__` / `closeEvent`).
- When a module fails to import, the placeholder page should name the missing package and the matching extra, for example: "Seismic raw file support needs `obspy`: `pip install pyhydrogeophysx[seismic-raw]`". The lazy loader in `qt_apps/modules/__init__.py` already catches the ImportError; surface `exc.name` in the message.

### 7. Climate agent: stop resolving the fetch script from the working directory
`agents/climate_data_agent.py` builds `Path("fetch_climate_data.py").absolute()`, so the subprocess only works when the app happens to start in the right directory. Resolve relative to the package (`Path(__file__).parent / "fetch_climate_data.py"`) and ship the script inside the package.

### 8. Unify the two LLM stacks
`agents/base_agent.py` has its own `_query_openai/_query_gemini/_query_claude`, while `qt_apps/agent/providers.py` has the newer provider abstraction (OpenAI, Anthropic, OpenAI-compatible with base URL override). Consequences today: provider fixes land twice, and `examples/aquah_web.py` imports from `qt_apps` even though it has no Qt dependency.

Suggested shape: move `providers.py` to a Qt-free module such as `PyHydroGeophysX/llm/providers.py`; keep `qt_apps/agent/providers.py` as a thin re-export for compatibility; port `base_agent` query methods onto the shared providers. This also gives the one-click workflow the OpenAI-compatible (DeepSeek, local) option for free.

### 9. Exception hygiene in library code
About 32 bare `except:` blocks exist across `agents/` and `core/` (for example in `agents/ert_inversion_agent.py`, `agents/data_fusion_agent.py`, `core/mesh_3d.py`). Bare excepts swallow `KeyboardInterrupt` and hide real errors. Replace with `except Exception as exc:` plus a log line, and let the structured `AgentResult` error fields carry the message.

### 10. AQUAH chat: long-conversation limits
`examples/aquah_web.py` appends to the neutral message list without bound, so long chats grow slow and expensive. Add windowing (keep the system prompt plus the last N exchanges) or a summarize-and-truncate step. The Qt chat panel has the same growth pattern. Also consider a per-call timeout on `provider.complete()` (retries exist, timeouts do not) and an optional per-workflow cost cap surfaced in the UI.

## P2: infrastructure and polish

### 11. Shrink clone size with a history rewrite (deliberately deferred)
Old blobs from `docs/build/` and `examples/results/` remain in history, so fresh clones still download roughly 368 MiB of pack data. When ready, and after warning collaborators (all commit hashes change, forks must re-clone):

```bash
pip install git-filter-repo
git filter-repo --path docs/build --path examples/results --invert-paths
git push --force-with-lease origin main
```

### 12. Build docs in CI instead of committing them
`docs/build/` should never be tracked (fixed going forward). Replace the committed HTML with a GitHub Actions job that runs Sphinx and pushes to the `gh-pages` branch (or add `.readthedocs.yaml` and let Read the Docs build). This keeps the published docs in sync with `main` automatically.

### 13. Desktop packaging notes
`packaging/pyinstaller_workbench.spec` intentionally excludes pygimli, pyvista, and SimPEG, so prebuilt bundles cannot run forward modeling. State this in the release notes and in `docs/desktop_workbench.md`, and recommend a source install for full functionality.

### 14. Cross-platform checks
- `qt_apps/theme.py` uses Segoe UI / Consolas first; fallbacks exist, but a quick visual pass on macOS and Linux is worth one session.
- Audit for Windows-only calls (`os.startfile`, `ctypes.windll` in the Streamlit Qt-launch helper have Unix fallbacks already; keep it that way for new code).

### 15. Gemini support refresh
The Streamlit default `gemini-pro` and the pricing entry for it are dated, and the Qt chat shows a notice instead of a Gemini provider. Either finish Gemini support on the shared provider layer (see item 8) or mark it clearly as legacy in the UI. Verify current Google model IDs at the time of the change.

### 16. Repository conventions
- Commit messages: recent history is a run of "update"; short imperative summaries make `git log` useful to contributors.
- Consider pre-commit hooks (black and flake8 are already in the `dev` extra but there is no config), plus `nbstripout` for notebooks.
- `[tool.setuptools.package-data]` lists `data/*` and `examples/*` under the package, but `examples/` lives at the repo root, so wheels do not ship it; either move a minimal dataset into the package or drop the stale entry.

## Strengths worth keeping (and advertising)

- Petrophysics module with clean APIs (Archie, Waxman-Smits, Hertz-Mindlin, DEM) and NumPy-style docstrings.
- Time-lapse, windowed, and structure-constrained ERT inversion classes; few open packages offer these.
- ERT import that works with or without resipy (unified fallback parser with topography support).
- Provider-neutral chat agent with per-call approve/reject gating in the Qt workbench and an allow-run switch on the web side; no hardcoded API keys anywhere in the repo.
- One-click flow already includes a preview, edit, and confirm step before execution.
- Qt workbench patterns: lazy module loading with placeholder pages, worker threads joined on close, Qt-free `io_utils`, and a Streamlit bridge with a documented context schema (`docs/desktop_workbench.md`).
