# PyHydroGeophysX Professional Desktop Studio

> Published on the docs site: <https://geohang.github.io/PyHydroGeophysX/agents/desktop_studio.html>
> (Sphinx source: `docs/source/agents/desktop_studio.rst`). When editing this guide,
> update the Sphinx page too; the site builds only the `.rst` version.

PyHydroGeophysX ships two complementary front ends:

- **Streamlit web app** (`examples/app_geophysics_workflow.py`) is the agent, report,
  tutorial, and deployment portal. It runs in a browser and can be hosted remotely.
- **Qt desktop studio** (`PyHydroGeophysX/qt_apps/`) is a local desktop application
  for professional mouse interaction: data processing, hydro-to-geophysics profile
  selection, point picking, geometry editing, and forward modeling.

The two exchange small JSON files on disk (the "bridge"), so you can set up a run in
the browser and finish the interactive work on the desktop.

## 1. Install desktop dependencies

The desktop studio needs PySide6 and pyqtgraph in addition to numpy and pandas:

```bash
pip install -r requirements-desktop.txt
# or, as an extra:
pip install "PyHydroGeophysX[desktop]"
```

Optional packages add features:

- `pyvista`, `pyvistaqt`, `vtk` — the Mesh 3D viewer (without them the Mesh 3D page
  shows a clean install message).
- `scipy` — the gridded preview in the Gravity / Magnetics module.
- `pygimli` — the real hydro-to-geophysics forward modeling (ERT, SRT, TDEM, FDEM,
  gravity). Without it, the Hydro module still exports a survey configuration JSON.

## 2. Run the Qt studio locally

```bash
python -m PyHydroGeophysX.qt_apps.launcher
# open directly into a module:
python -m PyHydroGeophysX.qt_apps.launcher --module hydro_geophysics
# attach to a bridge context written by Streamlit:
python -m PyHydroGeophysX.qt_apps.launcher --context results/streamlit_workflow/qt_bridge/full_studio_context.json
```

Helper scripts are provided in `scripts/`:

- Windows: `scripts\start_qt_studio.bat`
- macOS / Linux: `scripts/start_qt_studio.sh`

Module keys for `--module`: `home`, `seismic`, `ert`, `mesh3d`, `em`, `gravmag`,
`hydro_geophysics`.

## 3. Run the Streamlit web app locally

```bash
streamlit run examples/app_geophysics_workflow.py
```

Open the **🖥️ Professional Studio** tab. On a local desktop it shows buttons to
launch the Qt studio (whole app, or directly into Geophysical Data Processing or
Hydro to Geophysics). The tab also shows the bridge file paths and a **Refresh result**
button.

## 4. How the Streamlit / Qt bridge works

The bridge directory is `<output_dir>/qt_bridge/` (default
`results/streamlit_workflow/qt_bridge/`).

1. When you click a launch button, Streamlit writes `full_studio_context.json`
   (project root, output directory, hydro data directory, current workflow config and
   result, the demo selection, and the Python executable to reuse).
2. Streamlit starts the Qt studio as a separate process and passes that context path.
3. The Qt app reads the context on startup, so it points at the same project and data.
4. When you save in the Qt app (File → Streamlit Bridge → Save Studio Result, or
   after a forward run), it writes `full_studio_result.json` with the per-module
   results.
5. Back in the browser, **Refresh result** reads that file and displays it.

That manifest is for the bridge, not for keeping or getting out your results.

**Keeping a result:** a finished run is not recorded in the Project until you say so.
**File → Save Runs to Project** (`Ctrl+S`) or the toolbar **Save** button adds this
session's runs; the status bar shows how many are waiting, and closing the studio
asks. The Model Viewer lists unsaved runs first and can save or discard them one at a
time.

**Getting a result out:** **File → Export Results…** (`Ctrl+E`) writes the open
module's products to a folder you choose. Every model export includes
`model_cells.csv`, one row per cell with its own coordinate, so a section can be
replotted without PyGIMLi. Modules also export their own supporting files (picks CSV,
electrode geometry JSON, processed EM curves, corrected gravity data, survey
configuration JSON, figures). Exporting and saving are independent.

## 5. Why a remote Streamlit server cannot open a Qt window

A Qt window is a native desktop window that opens on the machine where the Python
process runs. When Streamlit is hosted on a remote server, that server has no display
attached to your screen, so it cannot show a window in your browser. The Professional
Studio tab detects this case and switches to **download mode**.

Detection rules:

- `PHGX_FORCE_REMOTE_MODE=1` forces download mode.
- Streamlit Community Cloud is detected and uses download mode.
- On Linux with no `DISPLAY` or `WAYLAND_DISPLAY`, download mode is used.
- If PySide6 is not installed, download mode is used (with an install hint).
- `PHGX_ENABLE_LOCAL_QT=1` opts in to a local launch when PySide6 is present.

## 6. Download mode

In download mode the tab shows links to the desktop builds and the source. Override the
default links with environment variables:

- `PHGX_QT_DOWNLOAD_WINDOWS`
- `PHGX_QT_DOWNLOAD_MACOS`
- `PHGX_QT_DOWNLOAD_LINUX`
- `PHGX_QT_DOWNLOAD_SOURCE`

To run the desktop studio from source on the same project, install the desktop
dependencies and run the launcher as in section 2.

## 7. Modules

Geophysical Data Processing:

- **Seismic** — load a 2D gather, apply gain / clip / polarity / trace normalization,
  pick first arrivals, export picks CSV.
- **ERT** — load electrode coordinates, add / move / delete electrodes, set labels,
  export an electrode file and a survey-geometry JSON.
- **Mesh 3D** — view VTK/VTU/VTP/STL/PLY meshes, a demo grid, axes, and a clipping
  plane (requires PyVistaQt).
- **EM** — load time or frequency response curves, apply a tail-median background
  correction and moving-average smoothing, export the processed curve.
- **Gravity / Magnetics** — load `x, y, value` stations, remove the mean or a
  least-squares plane, preview a gridded map, export corrected data.

Hydro to Geophysics:

- Load hydrologic model outputs (`Watercontent.npy`, `Porosity.npy`, `top.txt`,
  `bot.npy`), pick a profile with two clicks, set survey geometry and petrophysical
  parameters, then export a survey configuration and run forward modeling for the
  selected methods.

## 8. Performance note

The hydro arrays are stored on a regular grid, so the studio samples a profile with
`scipy.ndimage.map_coordinates` (bilinear) rather than a Delaunay-based `griddata`. On
the test machine this reduced one profile extraction from minutes to under a second.
The mesh-side interpolation used by the forward models is unchanged.

## 9. Prebuilt bundles and persistence

- `packaging/pyinstaller_studio.spec` deliberately excludes the heavy engines
  (pygimli, SimPEG, pyvista/VTK) to keep the bundle small. A prebuilt executable can
  therefore load, view, QC, and export data, but forward modeling, inversion, and the
  3D mesh viewer are unavailable in it. Install from source (this repository) for the
  full feature set.
- Window size and dock layout persist between sessions via `QSettings`
  (organization "PyHydroGeophysX", application "Studio"). Delete that settings key
  to reset the layout to defaults.
- Uncaught errors show an error dialog with a copyable traceback instead of closing
  the app silently; the same text also goes to stderr and can be reported as a GitHub
  issue.
