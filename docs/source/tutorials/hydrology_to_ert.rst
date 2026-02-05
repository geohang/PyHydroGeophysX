Hydrology to ERT
================

This workflow converts hydrologic model states into resistivity and forward ERT responses.

Steps
-----

1. Load MODFLOW/ParFlow outputs.
2. Convert water content to resistivity.
3. Run forward modeling or inversion.

.. code-block:: python

   from PyHydroGeophysX.model_output.modflow_output import MODFLOWWaterContent
   from PyHydroGeophysX.petrophysics import water_content_to_resistivity

   wc = MODFLOWWaterContent("examples/data/modflow", idomain=None).load_timestep(0)
   rho = water_content_to_resistivity(wc, rhos=100.0, n=2.0, porosity=0.3)

Related Examples
----------------

- :doc:`/auto_examples/Ex_model_output`
- :doc:`/auto_examples/Ex_ERT_workflow`
- :doc:`/auto_examples/Ex_3D_ERT_forward`

