# PyHydroGeophysX improvement roadmap

This roadmap comes from a full repository audit (2026-07-05, working tree at commit `aa02fee`) covering the core package, the multi-agent workflow, the Streamlit app, and the Qt desktop workbench. It is written from the point of view of an external user who finds the project on GitHub, installs it, and tries to run it. Two rounds of fixes landed on branch `improve/oss-hygiene` the same day; the sections below record what was done and what remains.

## Fixed on branch `improve/oss-hygiene` (round 1: hygiene and defaults)

- Stopped tracking `docs/build/` (732 files) and `examples/results/` (1,381 files) in git; extended `.gitignore` so result artifacts no longer pollute `git status` on every app run.
- Removed the stale `examples/Ex_Time_lapse_measurement-Hang's MacBook Air.ipynb`.
- Bumped CI actions to `actions/checkout@v5` and `actions/setup-python@v6` (Node.js 24 runner requirement).
- Added a desktop entry point: `pyhydrogeophysx-workbench` launches the Qt workbench after install (`[project.gui-scripts]` in `pyproject.toml`).
- Replaced retired Anthropic model IDs (sidebar default `claude-3-5-sonnet-20241022`, `BaseAgent` default `claude-3-opus-20240229`) with `claude-sonnet-5`; the Qt chat model list gained `claude-sonnet-5` as default; `agents/_pricing.py` gained current Claude and GPT-4.1 rates.
- Clear messages for missing API keys in the run path (keyless quick modes still work by design).
- README: removed the deleted `Ex_hello_agent` row, added a "Running the apps" section, noted the PyPI version lag. Added `CITATION.cff` (Zenodo DOI) so GitHub shows the cite button.

## Fixed on branch `improve/oss-hygiene` (round 2: tests, hardening, defects)

- **Test suite**: new `tests/` with 32 tests covering petrophysics round trips, profile interpolation (regular-grid fast path, nearest, NaN outside, irregular fallback), the Qt-free `io_utils`, the embedded no-resipy BERT parser on bundled sample data, the pricing table, message translation, and history windowing. CI now runs `pytest -q`; the pygimli-free tests run on plain `pip install -e .`.
- **Real bug found and fixed by the new tests**: `petrophysics.resistivity_models.resistivity_to_water_content` called `resistivity_to_saturation` with the wrong positional arguments and raised a `TypeError` on every call. It now maps `rhos` through the Archie identity (`rho_fluid=rhos, m=0`) so the documented Waxman-Smits inversion applies, and the water-content round trip is exact.
- **LLM provider layer moved to a Qt-free module**: `PyHydroGeophysX/llm/providers.py` is the canonical location; `qt_apps/agent/providers.py` is a module-alias shim so every old import keeps working; `examples/aquah_web.py` no longer imports Qt code. Client calls now carry a 120 s timeout (override with `PHGX_LLM_TIMEOUT_S`), and both chat surfaces send a safely windowed view of long conversations (`window_messages`, cuts only on user-turn boundaries so tool_use/tool_result pairs stay intact).
- **Qt hardening**: global excepthook with an error dialog (copyable traceback) in the launcher; atomic bridge JSON writes (temp file + `os.replace`) so Streamlit polling never reads a half-written result; a stable standalone bridge directory (repo `results/streamlit_workflow` when launched from a checkout, otherwise `~/.pyhydrogeophysx`) instead of scattering `qt_bridge/` around the current directory; window geometry and dock layout persist via `QSettings`; placeholder pages now name the missing optional package and the exact install command.
- **Climate agent**: `fetch_climate_data.py` did not exist anywhere in the repository, so the conda fetch path could never run. A standalone script now ships inside `agents/` (pydaymet `get_bycoords`, JSON config, CSV output; needs a live run to confirm against the Daymet service), and the agent resolves it relative to the package before falling back to the working directory.
- **Exception hygiene**: all 30 bare `except:` handlers across `agents/`, `core/`, `solvers/`, `inversion/`, `data_processing/`, `Geophy_modular/`, and `petrophysics/` were rewritten to `except Exception:` (AST-verified). Two additional stale backup files (`agents/__init__-HChen-W24.py`, `agents/data_fusion_agent-HChen-W24.py`) were deleted.
- **Defaults and packaging**: Gemini defaults updated from retired `gemini-pro` to `gemini-2.5-flash` (app sidebar, `BaseAgent`, doc text, pricing rows); `Ex_ERT_workflow.py` now checks for its demonstration dataset up front and explains what is missing instead of a bare `FileNotFoundError` (the dead `modflow_dir` variable is gone); package-data ships `qt_apps/modules/*.md` and drops the stale root-level `examples/*` entry; `docs/desktop_workbench.md` documents the PyInstaller bundle limits, `QSettings` persistence, and the error dialog.

Note: the QSettings and placeholder-hint changes live in `qt_apps/main_window.py` and `qt_apps/modules/__init__.py`, which also carry unrelated in-progress local work; they are edited in the working tree but deliberately left out of the branch commits until that work is ready.

## Functional QA of the desktop workbench (2026-07-06, env `pg`)

Every module was driven end to end through the agent action layer with the bundled datasets (about 80 scripted steps, all passing after the fix below):

- **Seismic**: SEG-Y load (`AP_411.sgy`, 13 records), record selection, regular shot geometry, topography import, STA/LTA auto-pick, review pause, manual pick edit, `pick_next_shot`, travel-time file load, SRT tomography (chi-square, rrms, VTK export).
- **ERT**: BERT field data (72 electrodes / 936 measurements), QC filter, single inversion with resistivity model + VTK; DAS-1 load, three-file time-lapse inversion (31 s) and full export.
- **EM**: SkyTEM TDEM CSV + line geometry, auto-calibrate, three-sounding line inversion (mean chi-square + section npz); FDEM method switch.
- **Gravity/Magnetics**: Bushveld gravity load, detrend + SimPEG 3D inversion; switch to magnetics + Britain aeromagnetic load.
- **Wizards**: Hydro → Geophysics ERT forward on the synthetic example (26 s, result ok); ERT → Water Content Monte Carlo (result ok); Seismic → Structure 3D build from three example lines (50-76 s, result ok) plus the structure handoff.
- **Mesh 3D**: sensor preview, PyGIMLi topography-prism generation (6,916 cells), 3D ERT forward (dipole-dipole), offscreen fallback, and an on-screen PyVistaQt render verified via a framebuffer screenshot (marker-colored mesh, axes, colorbar).
- **Bridge**: atomic result write to the stable per-user location.

**Fixed during QA (working tree, WIP files, uncommitted):** the three wizard modules stored their run worker in `self._worker` without `register_worker`, so `closeEvent` could not join a live run and closing the app mid-run destroyed a running QThread (a hard crash). All three now register the worker.

**Noted, not changed:**
- `send_structure_to_hydro` switches the active module, so a follow-up `get_status` in the same agent turn reports the hydro module rather than seismic3d. Consider returning the target module key in the result, or a system-prompt note for AQUAH.
- The example bundles for ERT → Water Content and Seismic → Structure live under `examples/results/synthetic_*`, which is no longer tracked, so a fresh clone lacks "use example data" for those two modules. Ship small bundles under `examples/data/` or generate them on demand (see item 3).
- resipy prints "pyvista not installed" at import even though `import pyvista` works in the same environment; the warning is resipy-internal and cosmetic here.

## P0: remaining items with the highest impact

### 1. Publish v0.3.0 to PyPI
PyPI still serves v0.1.0 with an MIT classifier, while the repo is at v0.3.0 under Apache-2.0.

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

Before uploading: confirm the version, tag the release (`git tag v0.3.0`), and consider a GitHub Release that triggers the Zenodo DOI archive. Afterwards verify `pip install pyhydrogeophysx==0.3.0` in a fresh environment.

### 2. Extend the test suite to the geophysics engines
The pygimli-free suite is in place. Next: a second CI job using `mamba-org/setup-micromamba` that installs pygimli and runs small forward/inversion smoke tests (tiny mesh, few electrodes), so the ERT/SRT code paths get coverage too. Also worth a live run: `agents/fetch_climate_data.py` against the Daymet service.

### 3. Clean the committed notebooks
The workflow demonstration dataset (`id.txt`, `top.txt`, `Porosity.npy`, `Watercontent.npy`) ships in `examples/data`, and the scripts run from the `examples` directory. The remaining problem is the committed notebooks: their outputs embed absolute Windows paths from the author machine. Re-run them with repository-relative paths or adopt `nbstripout`, and state in the README that examples are meant to run with `examples/` as the working directory.

## P1: remaining robustness work

### 4. Port `BaseAgent` LLM calls onto the shared provider layer
`agents/base_agent.py` still has its own `_query_openai/_query_gemini/_query_claude`. The shared layer (`PyHydroGeophysX/llm/providers.py`) now exists and is the natural home; porting needs a Gemini adapter and a live-key test of the one-click flow, so it was deferred rather than risked blind.

### 5. Cost controls in the UI
Cost previews exist, but there is no per-workflow budget. Add an optional cap surfaced in the sidebar (stop dispatching LLM calls once the estimate crosses it).

### 6. Small Qt polish items
Cap the log panel size (unbounded QTextEdit today), sync the Pan/Zoom toolbar buttons with the pyqtgraph view state, and add tooltips to module parameter fields.

## P2: infrastructure and polish

### 7. Shrink clone size with a history rewrite (deliberately deferred)
Old blobs remain in history, so fresh clones still download roughly 368 MiB. When ready, and after warning collaborators (hashes change, forks must re-clone):

```bash
pip install git-filter-repo
git filter-repo --path docs/build --path examples/results --invert-paths
git push --force-with-lease origin main
```

### 8. Build docs in CI instead of committing them
Replace the committed HTML workflow with a GitHub Actions job that runs Sphinx and pushes to the `gh-pages` branch (the site already serves from `gh-pages`), or add `.readthedocs.yaml`.

### 9. Repository conventions
Short imperative commit messages instead of "update"; consider pre-commit hooks (black and flake8 are already in the `dev` extra but unconfigured) plus `nbstripout` for notebooks; verify the Gemini model list against Google's current catalog when touching it next.

## Strengths worth keeping (and advertising)

- Petrophysics module with clean APIs (Archie, Waxman-Smits, Hertz-Mindlin, DEM), now with round-trip tests.
- Time-lapse, windowed, and structure-constrained ERT inversion classes; few open packages offer these.
- ERT import that works with or without resipy (unified fallback parser with topography support), now covered by a regression test.
- Provider-neutral chat agent with per-call approve/reject gating in the Qt workbench and an allow-run switch on the web side; no hardcoded API keys anywhere in the repo.
- One-click flow with a preview, edit, and confirm step before execution.
- Qt workbench patterns: lazy module loading with dependency-naming placeholders, worker threads joined on close, Qt-free `io_utils` with atomic bridge writes, persistent layout, and a documented Streamlit bridge (`docs/desktop_workbench.md`).
