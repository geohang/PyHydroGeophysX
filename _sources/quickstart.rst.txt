Quickstart
==========

This page gives a fast path to the most important workflows added in the current package architecture:
single-method inversion, joint inversion, EM workflows, agent usage, and the 3D Mesh Builder.

Install
-------

.. code-block:: bash

   pip install pyhydrogeophysx
   pip install "pyhydrogeophysx[geophysics]"   # PyGIMLi + SimPEG + RESIPY stack

3D Mesh Builder GUI
-------------------

Build and export 3D ERT meshes interactively — no API key required:

.. code-block:: bash

   # Recommended
   python -m PyHydroGeophysX.gui_mesh3d

   # Or directly
   streamlit run examples/app_mesh3d.py

The app opens three tabs: **Electrode View** (interactive 3D scatter), **Generate
Mesh** (runs ``Mesh3DCreator``), and **Export** (.bms / .vtk download).

For the full step-by-step guide see :ref:`agents/webapp:3D Mesh Builder App`.

Quick Petrophysics Check
------------------------

.. code-block:: python

   import numpy as np
   from PyHydroGeophysX.petrophysics import water_content_to_resistivity

   wc = np.array([0.20, 0.25, 0.30, 0.35])
   phi = np.array([0.35, 0.35, 0.35, 0.35])
   rho = water_content_to_resistivity(water_content=wc, rhos=100.0, n=2.0, porosity=phi)
   print(rho)

SRT Inversion (Single Time)
---------------------------

.. code-block:: python

   from PyHydroGeophysX.inversion import SRTInversion

   inv = SRTInversion(
       data_file="examples/data/srt/survey.sgt",
       lambda_val=50.0,
       max_iterations=20,
   )
   result = inv.run()
   print("Velocity model cells:", result.final_model.size)

FDEM Forward and Inversion
--------------------------

.. code-block:: python

   import numpy as np
   from PyHydroGeophysX.forward import FDEMForwardModeling, FDEMSurveyConfig
   from PyHydroGeophysX.inversion import FDEMInversion

   thicknesses = np.array([5.0, 10.0, 20.0])     # 4-layer model -> 3 thicknesses
   sigma_true = np.array([0.01, 0.02, 0.05, 0.08])
   cfg = FDEMSurveyConfig(frequencies=np.logspace(2, 4, 12))

   fwd = FDEMForwardModeling(thicknesses=thicknesses, survey_config=cfg)
   dobs = fwd.forward(sigma_true)
   uncert = 0.05 * np.maximum(np.abs(dobs), 1e-12)

   inv = FDEMInversion(
       frequencies=cfg.frequencies,
       dobs=dobs,
       uncertainties=uncert,
       thicknesses=thicknesses,
       receiver_component="secondary",
   )
   result = inv.run()
   print("FDEM chi2:", result.chi2)

Unified Method Dispatch
-----------------------

.. code-block:: python

   from PyHydroGeophysX.inversion import GeophysicalInversion

   srt = GeophysicalInversion("srt", data_file="examples/data/srt/survey.sgt")
   srt_result = srt.run()

   # Other options: "ert", "tdem", "fdem", "joint_ert_srt"

Joint ERT + SRT Inversion
-------------------------

.. code-block:: python

   from PyHydroGeophysX.inversion import JointERTSRTInversion

   joint = JointERTSRTInversion(
       ert_data="examples/data/ert/survey.dat",
       srt_data="examples/data/srt/survey.sgt",
       regularization_mode="geostat",      # or "smoothness"
       cross_gradient_mode="direct",       # or "spatial"
       lambda_cg_ert=120.0,
       lambda_cg_srt=80.0,
   )
   result = joint.run()
   print(result.chi2_ert, result.chi2_srt)

Agent Web App
-------------

- Open the hosted app: `https://pyhydrogeophysx.streamlit.app/ <https://pyhydrogeophysx.streamlit.app/>`_
- Web app usage and limitations: :doc:`agents/webapp`

Where To Go Next
----------------

- Tutorials: :doc:`tutorials/index`
- Examples gallery: :doc:`auto_examples/index`
- API reference: :doc:`api/index`
