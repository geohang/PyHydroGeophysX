Seismic Structure Extraction and Constrained Meshes
===================================================

This tutorial follows the paper workflow: extract a velocity interface from
seismic tomography and use it to build a structure-constrained ERT mesh.
It aligns with `examples/Ex_SRT_inv.py`, `examples/Ex_Structure_resinv.py`,
and the rendered examples :doc:`/auto_examples/Ex_SRT_inv` and
:doc:`/auto_examples/Ex_Structure_resinv`.

.. figure:: /auto_examples/images/thumb/sphx_glr_Ex_Structure_resinv_thumb.png
   :alt: Structure-constrained inversion example thumbnail
   :align: center
   :width: 70%

Prerequisites
-------------

- PyGIMLi installed.

Steps
-----

1. Create or load a velocity model (e.g., from SRT inversion).
2. Extract the velocity interface.
3. Build a constrained ERT mesh using the interface.

.. code-block:: python

   import numpy as np
   import pygimli.meshtools as mt
   from pygimli.physics import ert
   from PyHydroGeophysX.Geophy_modular.seismic_processor import (
       extract_velocity_structure
   )
   from PyHydroGeophysX.Geophy_modular.structure_integration import (
       create_ert_mesh_with_structure
   )

   rect = mt.createRectangle(start=(0.0, 0.0), end=(20.0, -10.0), marker=1)
   mesh = mt.createMesh(rect, quality=34, area=0.6)

   centers = mesh.cellCenters()
   depth = -centers[:, 1]
   velocity = 800.0 + 80.0 * depth

   smooth_x, smooth_z, interface = extract_velocity_structure(
       mesh, velocity, threshold=1200, interval=2.0
   )

   sensor_x = np.linspace(0.0, 20.0, 12)
   sensors = np.c_[sensor_x, np.zeros_like(sensor_x)]
   ert_data = ert.createData(sensors, schemeName="wa")

   constrained_mesh, markers, regions = create_ert_mesh_with_structure(
       ert_data, (smooth_x, smooth_z), paraDepth=10.0, paraMaxCellSize=3.0
   )

   print(np.unique(markers))

Outputs
-------

- Smooth velocity interface for structural constraints.
- Constrained ERT mesh with layer markers.

Next
----

- SRT inversion: :doc:`/auto_examples/Ex_SRT_inv`.
- Structure-constrained inversion: :doc:`/auto_examples/Ex_Structure_resinv`.
