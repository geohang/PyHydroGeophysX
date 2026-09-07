Hydrology to ERT
================

Workflow at a glance
--------------------

**Prepare:** Hydrological arrays, matching grid / mesh geometry and petrophysical parameters.

**Produce:** A resistivity model and simulated ERT measurements. The short snippet below only loads and converts water content; use the linked full workflow to generate measurements.

**Check / next step:** Compare predicted responses against the input model before trying time-lapse monitoring.

This workflow converts hydrologic model states into resistivity and forward ERT responses.

Steps
-----

1. Load MODFLOW/ParFlow outputs.
2. Convert water content to resistivity.
3. Run forward modeling or inversion.

.. code-block:: python

   import numpy as np

   from PyHydroGeophysX.model_output.water_content import MODFLOWWaterContent
   from PyHydroGeophysX.petrophysics import water_content_to_resistivity

   # idomain marks the active cells of the MODFLOW grid. The loader needs it to
   # unpack the WaterContent records into (nlay, nrow, ncol), so it cannot be None.
   idomain = np.loadtxt("examples/data/modflow/id.txt")

   wc = MODFLOWWaterContent("examples/data/modflow", idomain=idomain).load_timestep(0, nlay=3)
   rho = water_content_to_resistivity(wc, rhos=100.0, n=2.0, porosity=0.3)

Related Examples
----------------

- :doc:`/auto_examples/Ex_model_output`
- :doc:`/auto_examples/Ex_ERT_workflow`
- :doc:`/auto_examples/Ex_3D_ERT_forward`

