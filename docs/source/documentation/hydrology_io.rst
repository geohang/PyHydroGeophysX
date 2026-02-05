Hydrology I/O
=============

PyHydroGeophysX reads hydrologic model outputs and prepares fields for
petrophysical conversion.

Supported Models
----------------

- MODFLOW (water content, porosity)
- ParFlow (saturation, porosity)

Common Entry Points
-------------------

.. code-block:: python

   from PyHydroGeophysX.model_output import MODFLOWWaterContent, ParflowSaturation

   wc = MODFLOWWaterContent("model_dir", idomain).load_timestep(0)
   sat = ParflowSaturation("model_dir", "run_name").load_timestep(100)

Tips
----

- Keep grid spacing and coordinate systems consistent with geophysical meshes.
- Use time ranges to align with survey dates.

Examples
--------

- :doc:`/auto_examples/Ex_model_output`
- :doc:`/auto_examples/Ex_ERT_workflow`
