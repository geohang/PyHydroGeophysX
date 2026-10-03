Hydrology to EM (TDEM and FDEM)
===============================

Workflow at a glance
--------------------

**Prepare:** Layer thicknesses, water-content / saturation information, petrophysical parameters and EM survey geometry.

**Produce:** Synthetic EM responses and a recovered layered model in the full examples. The snippet below only constructs forward operators.

**Check / next step:** Start with a single sounding, then move to line data and lateral constraints in the EM examples.

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

   thicknesses = np.array([5.0, 10.0, 20.0])  # four layers

   # Central-loop TDEM: a 10 m circular loop with the receiver at its centre
   cfg = TDEMSurveyConfig(
       source_location=np.array([0.0, 0.0, 1.0]),
       source_radius=10.0,
       receiver_location=np.array([0.0, 0.0, 1.0]),
       times=np.logspace(-5, -2, 20),
   )
   fwd = TDEMForwardModeling(thicknesses=thicknesses, survey_config=cfg)

   fdem_cfg = FDEMSurveyConfig(frequencies=np.logspace(2, 4, 12))
   fdem_fwd = FDEMForwardModeling(thicknesses=thicknesses, survey_config=fdem_cfg)

Related Example
---------------

- :doc:`/auto_examples/Ex_TDEM_workflow`
- :doc:`/auto_examples/Ex_FDEM_workflow`
