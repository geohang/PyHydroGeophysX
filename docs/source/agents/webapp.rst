Agent Web App
=============

PyHydroGeophysX ships two interactive Streamlit applications:

1. **Geophysics Workflow App** – natural-language orchestration of ERT/SRT/TDEM/FDEM workflows.
2. **3D Mesh Builder** – graphical tool for building and exporting 3D ERT meshes (new in v0.3).

.. tip::

   Prefer a native desktop app? The :doc:`Qt Desktop Studio <desktop_studio>` is
   available as a downloadable Windows / macOS application (a small *light* build and a
   *full* build with the geophysics engines included).

   .. button-link:: https://github.com/geohang/PyHydroGeophysX/releases/latest
      :color: secondary
      :expand:

      Download the Desktop Studio (Windows / macOS)

.. contents:: On this page
   :local:
   :depth: 1

Geophysics Workflow App
-----------------------

Open the App
^^^^^^^^^^^^

- Live URL: `https://pyhydrogeophysx.streamlit.app/ <https://pyhydrogeophysx.streamlit.app/>`_
- Local script: ``examples/app_geophysics_workflow.py``
- Local launcher: ``pyhydrogeophysx-gui``

.. button-link:: https://pyhydrogeophysx.streamlit.app/
   :color: primary
   :expand:

   Launch PyHydroGeophysX Agent Web App

What It Does
^^^^^^^^^^^^

- Starts in demo mode with bundled cached ERT and joint ERT+SRT results — no API
  key required for a first click-through.
- Collects workflow goals and optional context in natural language.
- Shows the parsed configuration for review before running on user data.
- Shows an estimated LLM cost in the sidebar when the live LLM path is used.
- Routes tasks to agent pipelines for ERT, SRT, TDEM, FDEM, and conversion workflows.
- Provides a local raw SEG-Y processing panel for shot-gather preview, assisted
  first-break picking, travel-time export, and SRT inversion.
- Supports multi-method orchestration through ``DataFusionAgent``.
- Supports joint ERT+SRT workflows where compatible inputs are provided.
- Produces intermediate artifacts and workflow reports where available.

Required Inputs
^^^^^^^^^^^^^^^

- A clear task prompt (for example, "run a time-lapse ERT inversion").
- Paths or uploaded references to required survey/model files.
- For joint inversion: both ERT and SRT datasets for the same profile/domain.
- Optional provider credentials when you want LLM-backed interpretation.

Limitations and Notes
^^^^^^^^^^^^^^^^^^^^^

- Uploaded data quality and schema consistency still control result quality.
- Some workflows require optional dependencies (for example ``pygimli`` or SimPEG).
- Joint inversion expects physically consistent survey geometry across methods.
- LLM-generated recommendations should be reviewed before production decisions.
- For common failure modes and fixes, see :doc:`troubleshooting`.

3D Mesh Builder App
-------------------

The 3D Mesh Builder is a standalone Streamlit app for interactively configuring
and exporting tetrahedral meshes for ERT forward modeling and inversion.

Open the App
^^^^^^^^^^^^

.. code-block:: bash

   # Recommended: via the package launcher
   python -m PyHydroGeophysX.gui_mesh3d

   # Or directly with Streamlit
   streamlit run examples/app_mesh3d.py

What It Does
^^^^^^^^^^^^

The app is organized into three tabs:

**Electrode View**
  Visualizes electrode positions in an interactive 3D scatter plot (Plotly).
  Supports three array configurations:

  - *Surface Grid* — rectangular grid of surface electrodes.
  - *Borehole* — single vertical borehole electrode string.
  - *Crosshole* — two parallel boreholes for crosshole tomography.

  Four topography types are available: Flat, Linear Tilt, Gaussian Hill, and
  Custom Expression (any valid Python/NumPy expression using ``x`` and ``y``).

**Generate Mesh**
  Runs ``Mesh3DCreator`` from ``PyHydroGeophysX.core.mesh_3d`` with the configured
  parameters. Mesh statistics (cell count, node count, quality histogram) are
  displayed inline after generation.

**Export**
  Downloads the generated mesh as:

  - ``.bms`` — native PyGIMLi binary mesh format.
  - ``.vtk`` — VTK format for ParaView or other 3D visualizers.

Sidebar Controls
^^^^^^^^^^^^^^^^

+-------------------------------+---------------------------------------------+
| Control                       | Description                                 |
+===============================+=============================================+
| Mesh Type                     | surface or volume                           |
+-------------------------------+---------------------------------------------+
| Electrode Array Type          | Surface Grid / Borehole / Crosshole         |
+-------------------------------+---------------------------------------------+
| Grid Nx / Ny                  | Number of electrodes along each axis        |
+-------------------------------+---------------------------------------------+
| Electrode Spacing (m)         | Uniform spacing between adjacent electrodes |
+-------------------------------+---------------------------------------------+
| Topography Type               | Flat / Linear Tilt / Gaussian Hill / Custom |
+-------------------------------+---------------------------------------------+
| Max Mesh Cell Size            | Controls resolution (smaller = finer mesh)  |
+-------------------------------+---------------------------------------------+
| Mesh Depth (m)                | Vertical extent of the mesh below surface   |
+-------------------------------+---------------------------------------------+
| Output Filename               | Base name for exported files                |
+-------------------------------+---------------------------------------------+

Step-by-Step: Build a Surface ERT Mesh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Launch the app: ``python -m PyHydroGeophysX.gui_mesh3d``
2. In the sidebar, choose **Surface Grid** as the Electrode Array Type.
3. Set **Grid Nx = 10**, **Grid Ny = 5**, **Spacing = 5 m**.
4. Choose **Linear Tilt** topography to simulate sloping terrain.
5. Click the **Electrode View** tab to verify the electrode positions in 3D.
6. Click **Generate Mesh** tab, then press **Generate 3D Mesh**.
7. Inspect cell count and quality histogram shown below the mesh summary.
8. Click **Export** tab and download as ``.bms`` or ``.vtk``.

Step-by-Step: Build a Crosshole Mesh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Launch the app: ``python -m PyHydroGeophysX.gui_mesh3d``
2. Choose **Crosshole** array type; set borehole depth and electrode spacing.
3. Adjust **Max Mesh Cell Size** (start with 2.0 m for a fast first mesh).
4. Click **Electrode View** to confirm both borehole strings.
5. Generate and export as ``.bms`` for use with PyGIMLi crosshole inversion.

Requirements
^^^^^^^^^^^^

- ``PyHydroGeophysX`` package with the ``core.mesh_3d`` module.
- ``pygimli`` for mesh generation (``conda install -c gimli pygimli``).
- ``gmsh`` on PATH for tetrahedral meshing.
- ``plotly`` for 3D electrode visualization (``pip install plotly``).
- ``streamlit`` (``pip install streamlit``).

