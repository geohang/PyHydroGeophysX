TDEM Forward Modeling
=====================

This tutorial shows a minimal Time-Domain Electromagnetic (TDEM) forward
model using the SimPEG-backed workflow in PyHydroGeophysX. It is based on
`examples/Ex_TDEM_workflow.py` and the rendered example
:doc:`/auto_examples/Ex_TDEM_workflow`.

.. figure:: /auto_examples/images/thumb/sphx_glr_Ex_TDEM_workflow_thumb.png
   :alt: TDEM workflow example thumbnail
   :align: center
   :width: 70%

Prerequisites
-------------

- SimPEG installed (via `requirements.txt` or conda).

Steps
-----

1. Define a layered Earth model and survey configuration.
2. Run the 1D TDEM forward response.

.. code-block:: python

   import numpy as np
   from PyHydroGeophysX.forward.tdem_forward import (
       TDEMSurveyConfig, TDEMForwardModeling
   )

   thicknesses = np.array([5.0, 10.0])
   config = TDEMSurveyConfig(
       source_location=np.array([0.0, 0.0, 1.0]),
       receiver_location=np.array([0.0, 0.0, 1.0]),
       source_radius=5.0,
       times=np.logspace(-5, -3, 11)
   )

   fwd = TDEMForwardModeling(thicknesses=thicknesses, survey_config=config)
   conductivity = np.array([0.01, 0.05, 0.02])  # S/m for 3 layers
   response = fwd.forward(conductivity)

   print(response[:3])

Outputs
-------

- Forward response at the specified time channels.

Next
----

- Full example: :doc:`/auto_examples/Ex_TDEM_workflow`.
