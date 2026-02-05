Hydrology to EM (TDEM and FDEM)
===============================

Use this workflow to map hydrologic outputs to conductivity profiles and EM responses.

Steps
-----

1. Extract a layered profile from hydrologic data.
2. Convert saturation/water content to conductivity.
3. Run TDEM and/or FDEM forward and inversion routines.

.. code-block:: python

   import numpy as np
   from PyHydroGeophysX.forward.tdem_forward import TDEMForwardModeling, TDEMSurveyConfig
   from PyHydroGeophysX.forward.fdem_forward import FDEMForwardModeling, FDEMSurveyConfig

   cfg = TDEMSurveyConfig(source_type="VMD", source_location=[0, 0, 1], receiver_location=[0, 0, 1])
   fwd = TDEMForwardModeling(cfg)

   fdem_cfg = FDEMSurveyConfig(frequencies=np.logspace(2, 4, 12))
   fdem_fwd = FDEMForwardModeling(thicknesses=np.array([5.0, 10.0, 20.0]), survey_config=fdem_cfg)

Related Example
---------------

- :doc:`/auto_examples/Ex_TDEM_workflow`
- :doc:`/auto_examples/Ex_FDEM_workflow`
