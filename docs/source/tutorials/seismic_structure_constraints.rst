Seismic Structure Constraints
=============================

Workflow at a glance
--------------------

**Prepare:** A seismic-derived structural interpretation and ERT data for the same area.

**Produce:** An ERT model guided by existing structural information; this differs from jointly updating both methods.

**Check / next step:** Use joint inversion if both datasets should participate in the coupled solve.

Use seismic velocity structure to constrain ERT inversion meshes and improve geological realism.

Steps
-----

1. Build or load seismic travel-time inversion outputs.
2. Extract velocity interfaces or boundary weights for structural boundaries.
3. Apply interfaces in resistivity inversion meshes or joint inversion constraints.

.. code-block:: python

   import numpy as np
   import pygimli as pg
   from PyHydroGeophysX.inversion.cross_constraints import StructuralConstraint

   # Replace these paths with a matching mesh and per-cell velocity model.
   mesh = pg.load("velmesh.bms")
   velocity = np.load("Vinvmodel.npy")
   boundary_weights = StructuralConstraint.from_velocity_model(
       velocity_model=velocity,
       mesh=mesh,
       gradient_threshold=0.3,
   )

This snippet constructs boundary weights only. Use the structure-constrained
example below to apply the structural information in a complete ERT workflow.

Related Examples
----------------

- :doc:`/auto_examples/EX_SRT_forward`
- :doc:`/auto_examples/Ex_SRT_inv`
- :doc:`/auto_examples/Ex_Structure_resinv`
- :doc:`/auto_examples/Ex_joint_inversion`
