Hydrology to ERT Forward Modeling
=================================

This tutorial connects hydrologic model outputs to ERT forward modeling. It
builds on `examples/Ex_ERT_workflow.py` and :doc:`/auto_examples/Ex_ERT_workflow`.

.. figure:: /auto_examples/images/thumb/sphx_glr_Ex_ERT_workflow_thumb.png
   :alt: Hydro to ERT workflow thumbnail
   :align: center
   :width: 70%

Steps
-----

1. Load hydrologic outputs.

.. code-block:: python

   from PyHydroGeophysX.model_output import MODFLOWWaterContent

   wc = MODFLOWWaterContent("model_dir", idomain).load_timestep(0)

2. Convert to resistivity using a petrophysical model.

.. code-block:: python

   from PyHydroGeophysX.petrophysics import water_content_to_resistivity

   rho = water_content_to_resistivity(wc, rhos=100.0, n=2.0, porosity=0.3)

3. Create a mesh and run forward modeling.

.. code-block:: python

   from PyHydroGeophysX.forward import ERTForwardModeling

   ert_fwd = ERTForwardModeling(mesh, data_scheme)
   synth = ert_fwd.forward(resistivity_model=rho)

Outputs
-------

- Synthetic ERT data
- Baseline model for survey planning or inversion tests

Next
----

- For time-lapse studies, see :doc:`/tutorials/tutorial_timelapse`.
- Review the full example: :doc:`/auto_examples/Ex_ERT_workflow`.
