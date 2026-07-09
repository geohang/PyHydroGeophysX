Desktop Workbench (Qt)
======================

PyHydroGeophysX ships two complementary front ends:

- The **Streamlit web app** (:doc:`webapp`) is the agent, report, tutorial, and
  deployment portal. It runs in a browser and can be hosted remotely.
- The **Qt desktop workbench** (``PyHydroGeophysX/qt_apps/``) is a local desktop
  application for hands-on mouse interaction: data processing, first-arrival picking,
  electrode geometry editing, hydro-to-geophysics profile selection, mesh building,
  and forward modeling and inversion.

The two exchange small JSON files on disk (the "bridge"), so you can set up a run in
the browser and finish the interactive work on the desktop.

.. contents:: On this page
   :local:
   :depth: 1

Download the Desktop App
------------------------

Prebuilt bundles for Windows and macOS are published on GitHub Releases. Each platform
has two variants, so you can pick what fits your machine:

.. button-link:: https://github.com/geohang/PyHydroGeophysX/releases/latest
   :color: primary
   :expand:

   Download the Desktop Workbench (Windows / macOS)

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Bundle
     - What it includes
   * - ``PyHydroGeophysX-Workbench-windows-light.zip``
     - Windows. Load, view, QC, pick, edit geometry, and export data in every module.
       Small download; starts fast.
   * - ``PyHydroGeophysX-Workbench-windows-full.zip``
     - Windows. Everything in light, plus the geophysics engines (PyGIMLi, SimPEG,
       PyVista/VTK): forward modeling, inversion, and the 3D mesh viewer work out of
       the box. Much larger download.
   * - ``PyHydroGeophysX-Workbench-macos-light.zip``
     - macOS. Same feature set as the Windows light build.
   * - ``PyHydroGeophysX-Workbench-macos-full.zip``
     - macOS. Same feature set as the Windows full build.

After unzipping, run ``PyHydroGeophysX-Workbench.exe`` inside the extracted folder
(Windows) or open ``PyHydroGeophysX-Workbench.app`` (macOS).

.. note::

   In the **light** bundles the heavy engines are left out on purpose, so forward
   modeling, inversion, and the 3D mesh viewer show an install message instead of
   running. Choose the **full** bundle, or :ref:`install from source
   <desktop-install-source>`, for the complete feature set.

.. note::

   The macOS bundles are not code-signed. If macOS blocks the first launch,
   right-click the app and choose **Open** once, or clear the quarantine flag with
   ``xattr -cr PyHydroGeophysX-Workbench.app``.

.. _desktop-install-source:

Install and Run from Source
---------------------------

The workbench needs PySide6 and pyqtgraph in addition to numpy and pandas:

.. code-block:: bash

   pip install -r requirements-desktop.txt
   # or, as an extra:
   pip install "pyhydrogeophysx[desktop]"

Optional packages add features:

- ``pygimli``: real forward modeling and inversion (ERT, SRT, TDEM, FDEM, gravity).
  Without it, the Hydro module still exports a survey configuration JSON.
- ``pyvista``, ``pyvistaqt``, ``vtk``: the 3D mesh viewer.
- ``simpeg``: gravity and magnetics 3D inversion.
- ``scipy``: gridding and interpolation in several modules.

Launch the workbench:

.. code-block:: bash

   python -m PyHydroGeophysX.qt_apps.launcher

   # open directly into a module:
   python -m PyHydroGeophysX.qt_apps.launcher --module hydro_geophysics

   # attach to a bridge context written by Streamlit:
   python -m PyHydroGeophysX.qt_apps.launcher --context results/streamlit_workflow/qt_bridge/full_workbench_context.json

If the package is installed (``pip install pyhydrogeophysx[desktop]``), the
``pyhydrogeophysx-workbench`` command starts the same application. Helper scripts are
provided in ``scripts/``: ``start_qt_workbench.bat`` (Windows) and
``start_qt_workbench.sh`` (macOS / Linux).

Module keys for ``--module``: ``home``, ``seismic``, ``ert``, ``mesh3d``, ``em``,
``gravmag``, ``hydro_geophysics``, ``geo_hydrology``, ``seismic3d``.

Modules
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - What it does
   * - Seismic Processing
     - Load 2D shot gathers (SEG-Y, Geometrics DAT), apply gain / AGC / normalization,
       pick first arrivals (assisted auto-picking plus manual and line picking),
       QC travel times, and run SRT travel-time tomography. Pre-picked travel-time
       files can be uploaded and inverted directly.
   * - ERT Processing
     - Load resistivity files by instrument format (BERT / unified, E4D, Syscal,
       and more), edit electrodes, QC the apparent-resistivity pseudosection, filter
       data, and run single or time-lapse inversion with per-step results.
   * - 3D Mesh Builder
     - Build ERT meshes (surface grid, borehole, crosshole arrays; flat, tilted,
       Gaussian-hill, file-based, or custom topography), view meshes in 3D with a
       clipping plane, and run 3D ERT forward modeling on the generated mesh.
   * - EM Processing
     - Load TDEM / FDEM soundings (single or multi-sounding line files), invert one
       sounding or a whole line into a stitched resistivity section, and view
       plan-view depth-slice maps on survey coordinates.
   * - Gravity / Magnetics
     - Load station data, remove regional trends, and run SimPEG 3D inversion with an
       interactive model viewer.
   * - Hydro -> Geophysics
     - Load hydrologic model outputs (water content, porosity, surfaces), pick a
       profile, set petrophysical parameters, and run forward modeling for the
       selected geophysical methods.
   * - ERT -> Water Content
     - Invert ERT results into water content estimates.
   * - Seismic -> Structure
     - Derive 3D structural surfaces from seismic lines.

The workbench also includes **AQUAH Chat**, an in-app assistant that can drive the
modules through natural language (OpenAI, Anthropic, or any OpenAI-compatible
provider; bring your own API key). Every proposed action shows an Approve / Reject
button before it runs.

How the Streamlit / Qt Bridge Works
-----------------------------------

The bridge directory is ``<output_dir>/qt_bridge/`` (default
``results/streamlit_workflow/qt_bridge/``).

1. In the web app's **Professional Workbench** tab, a launch button writes
   ``full_workbench_context.json`` (project root, output directory, hydro data
   directory, current workflow configuration and result, and the Python executable
   to reuse).
2. Streamlit starts the Qt workbench as a separate process and passes that context path.
3. The Qt app reads the context on startup, so it points at the same project and data.
4. When you save in the Qt app (File -> Save Workbench Result, or after a forward
   run), it writes ``full_workbench_result.json`` with the per-module results.
5. Back in the browser, the results panel reads that file and displays it.

Modules can also export their own files (picks CSV, electrode geometry JSON, processed
EM curves, corrected gravity data, survey configuration JSON, figures) into the output
directory.

Remote Servers and Download Mode
--------------------------------

A Qt window opens on the machine where the Python process runs. When Streamlit is
hosted on a remote server, that server has no display attached to your screen, so the
**Professional Workbench** tab switches to **download mode** and shows the download
links above instead of launch buttons. The default links point at the latest GitHub
Release and can be overridden with environment variables:

- ``PHGX_QT_DOWNLOAD_WINDOWS``
- ``PHGX_QT_DOWNLOAD_MACOS``
- ``PHGX_QT_DOWNLOAD_LINUX``
- ``PHGX_QT_DOWNLOAD_SOURCE``

``PHGX_FORCE_REMOTE_MODE=1`` forces download mode; ``PHGX_ENABLE_LOCAL_QT=1`` opts in
to a local launch when PySide6 is present.

Persistence and Troubleshooting
-------------------------------

- Window size and dock layout persist between sessions via ``QSettings``
  (organization "PyHydroGeophysX", application "Workbench"). Delete that settings key
  to reset the layout to defaults.
- Uncaught errors show a dialog with a copyable traceback instead of closing the app
  silently; the same text also goes to stderr and can be reported as a GitHub issue.
- If a module page shows a "could not be loaded" message, it names the missing
  optional package and the install command; the rest of the workbench is unaffected.

Building the Bundles Yourself
-----------------------------

The PyInstaller configuration lives at ``packaging/pyinstaller_workbench.spec``. The
``PHGX_BUILD_VARIANT`` environment variable selects ``light`` (default) or ``full``.
Helper scripts build and zip a bundle in one step:

.. code-block:: bash

   # Windows (PowerShell)
   scripts/build_workbench_exe.ps1 light

   # macOS / Linux
   bash scripts/build_workbench_exe.sh light

The GitHub Actions workflow ``.github/workflows/build-desktop.yml`` builds all four
bundles and attaches them to the Release for every version tag.
