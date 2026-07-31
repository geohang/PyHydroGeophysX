PyHydroGeophysX
===============

PyHydroGeophysX turns hydrology model outputs into geophysical responses and inversions.
Use this page as a start point for first-time setup, guided workflows, and agent tools.

Start Here
----------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc
      :class-card: sd-card-hover

      Install the package and optional dependencies.

   .. grid-item-card:: Quickstart
      :link: quickstart
      :link-type: doc
      :class-card: sd-card-hover

      Run a small, realistic first example.

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc
      :class-card: sd-card-hover

      Follow task-oriented workflow walkthroughs.

   .. grid-item-card:: Examples Gallery
      :link: auto_examples/index
      :link-type: doc
      :class-card: sd-card-hover

      Browse full scripts with figures and outputs.

   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: doc
      :class-card: sd-card-hover

      Explore modules, classes, and functions.

   .. grid-item-card:: Agent Web App
      :link: agents/webapp
      :link-type: doc
      :class-card: sd-card-hover

      Open the Streamlit app and agent usage guidance.

   .. grid-item-card:: Desktop Workbench
      :link: agents/desktop_workbench
      :link-type: doc
      :class-card: sd-card-hover

      Download the Qt desktop app for Windows and macOS.

   .. grid-item-card:: Environmental Geophysics Course
      :link: https://geohang.github.io/environmental-geophysics/
      :link-type: url
      :class-card: sd-card-hover

      Explore open lectures, interactive topic apps, field missions, and practice questions.

User Journeys
-------------

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: Hydrology -> ERT Workflow
      :link: tutorials/hydrology_to_ert
      :link-type: doc
      :class-card: sd-card-hover

      Load MODFLOW or ParFlow output, convert with petrophysics, and run ERT workflows.

   .. grid-item-card:: Hydrology -> TDEM Workflow
      :link: tutorials/hydrology_to_tdem
      :link-type: doc
      :class-card: sd-card-hover

      Build layered conductivity models and run TDEM forward and inversion steps.

   .. grid-item-card:: Agents and LLM Workflows
      :link: tutorials/agent_workflows
      :link-type: doc
      :class-card: sd-card-hover

      Use language-guided workflows and the hosted Streamlit app.

   .. grid-item-card:: Joint ERT + SRT Inversion
      :link: tutorials/joint_inversion
      :link-type: doc
      :class-card: sd-card-hover

      Couple ERT and SRT with structure constraints and geostatistics.

New In This Release
-------------------

- Dedicated ``SRTInversion`` and ``TimeLapseSRTInversion`` classes.
- New ``FDEMForwardModeling`` and ``FDEMInversion`` workflow support.
- Unified dispatcher: ``GeophysicalInversion`` for ``ert/srt/tdem/fdem/joint``.
- New ``JointERTSRTInversion`` with cross-gradient and geostatistical constraints.
- Cross-method utilities in ``StructuralConstraint`` and ``PetrophysicalCoupling``.

Quickstart Code
---------------

.. code-block:: python

   import numpy as np
   from PyHydroGeophysX.petrophysics import water_content_to_resistivity

   wc = np.array([[0.22, 0.28], [0.31, 0.35]])
   rho = water_content_to_resistivity(wc, rhos=100.0, n=2.0, porosity=0.3)
   print(rho)

Web App and Desktop App
-----------------------

Open the hosted app: `pyhydrogeophysx.streamlit.app <https://pyhydrogeophysx.streamlit.app/>`_

Download the desktop workbench (Windows / macOS): `GitHub Releases
<https://github.com/geohang/PyHydroGeophysX/releases/latest>`_; usage guide:
:doc:`agents/desktop_workbench`.

Citation
--------

If you use PyHydroGeophysX in your work, please cite:

- Chen, Hang and Niu, Qifei and Wu, Yuxin, *PyHydroGeophysX: An Extensible Open-Source Platform for Integrating Hydrological Models with Geophysical Measurements*. SSRN. https://ssrn.com/abstract=6238293 and https://doi.org/10.2139/ssrn.6238293
- Chen, H. (2026). *A Generalizable Automated Geophysical Agent Workflow for Accessible Subsurface Hydrology Analysis*. Big Data and Earth System, 100042.

.. toctree::
   :maxdepth: 1
   :hidden:

   Installation <installation>
   Quickstart <quickstart>
   Tutorials/Workflows <tutorials/index>
   Examples <auto_examples/index>
   API Reference <api/index>
   Agents + Web App <agents/index>
