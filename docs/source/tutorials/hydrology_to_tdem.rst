Hydrology to TDEM
=================

Use this workflow to map hydrologic outputs to conductivity profiles and TDEM responses.

Steps
-----

1. Extract a layered profile from hydrologic data.
2. Convert saturation/water content to conductivity.
3. Run TDEM forward and inversion routines.

.. code-block:: python

   from PyHydroGeophysX.forward.tdem_forward import TDEMForwardModeling, TDEMSurveyConfig

   cfg = TDEMSurveyConfig(source_type="VMD", source_location=[0, 0, 1], receiver_location=[0, 0, 1])
   fwd = TDEMForwardModeling(cfg)

Related Example
---------------

- :doc:`/auto_examples/Ex_TDEM_workflow`

