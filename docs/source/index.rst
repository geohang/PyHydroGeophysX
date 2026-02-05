PyHydroGeophysX
===============

PyHydroGeophysX is an open-source Python toolkit for turning hydrologic model outputs into
geophysical responses and inversions (ERT, SRT, TDEM). It is built for watershed and
critical-zone researchers who need repeatable hydro-to-geophysics workflows.

Key Workflows
-------------

- Field ERT ingestion, QC, visualization, and export for inversion (RESIPY-compatible).
- Hydrologic model outputs (MODFLOW/ParFlow) -> petrophysics -> 2D/3D forward responses
  for ERT/SRT/EM, including time-lapse monitoring and survey sensitivity.
- Single-time, time-lapse, and windowed inversions with temporal regularization for
  monitoring datasets.
- Structure-constrained inversion and GM -> HM transfer: extract seismic interfaces and
  velocity structure to build meshes and parameterize hydrologic models.
- Multi-physics and uncertainty workflows: integrate ERT/SRT/EM and Monte Carlo to
  quantify parameter uncertainty.

Quickstart
----------

.. code-block:: python

   import numpy as np
   from PyHydroGeophysX.petrophysics import water_content_to_resistivity

   wc = np.linspace(0.1, 0.4, 6).reshape(2, 3)
   rho = water_content_to_resistivity(wc, rhos=100.0, n=2.0, porosity=0.3)
   print(rho)

Start Here
----------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc
      :class-card: sd-card-hover

      Install PyHydroGeophysX and optional dependencies.

   .. grid-item-card:: Documentation
      :link: documentation/index
      :link-type: doc
      :class-card: sd-card-hover

      User guide and task-oriented references.

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc
      :class-card: sd-card-hover

      End-to-end walkthroughs.

   .. grid-item-card:: Examples
      :link: auto_examples/index
      :link-type: doc
      :class-card: sd-card-hover

      Gallery of runnable scripts.

   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: doc
      :class-card: sd-card-hover

      Complete Python API docs.

   .. grid-item-card:: Agents
      :link: documentation/agents
      :link-type: doc
      :class-card: sd-card-hover

      Multi-agent system usage and limitations.

.. toctree::
   :maxdepth: 1
   :hidden:

   Home <self>
   Documentation <documentation/index>
   Installation <installation>
   Tutorials <tutorials/index>
   Examples <auto_examples/index>
   API Reference <api/index>
