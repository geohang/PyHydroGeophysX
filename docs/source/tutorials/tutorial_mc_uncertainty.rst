Monte Carlo Uncertainty for ERT-to-Water Content
================================================

This tutorial demonstrates uncertainty quantification for water content
estimates derived from ERT resistivity models. It aligns with
`examples/Ex_MC_Hydro.py` and the rendered example
:doc:`/auto_examples/Ex_MC_Hydro`.

.. figure:: /auto_examples/images/thumb/sphx_glr_Ex_MC_Hydro_thumb.png
   :alt: Monte Carlo hydro example thumbnail
   :align: center
   :width: 70%

Prerequisites
-------------

- PyGIMLi installed.

Steps
-----

1. Build a small mesh and assign layer markers.
2. Create synthetic resistivity for multiple time steps.
3. Run Monte Carlo sampling to obtain water content distributions.

.. code-block:: python

   import numpy as np
   import pygimli.meshtools as mt
   from PyHydroGeophysX.Geophy_modular.ERT_to_WC import ERTtoWC

   rect = mt.createRectangle(start=(0.0, 0.0), end=(10.0, -5.0), marker=1)
   mesh = mt.createMesh(rect, quality=34, area=0.5)

   markers = np.ones(mesh.cellCount(), dtype=int)
   for i, cell in enumerate(mesh.cells()):
       markers[i] = 3 if cell.center().y() > -2.5 else 2

   res_t0 = np.where(markers == 3, 150.0, 500.0)
   res_t1 = np.where(markers == 3, 120.0, 450.0)
   res = np.vstack([res_t0, res_t1]).T

   converter = ERTtoWC(mesh, res, markers)
   converter.setup_layer_distributions({
       3: {
           "rhos": {"mean": 80.0, "std": 10.0},
           "n": {"mean": 2.0, "std": 0.1},
           "sigma_sur": {"mean": 0.005, "std": 0.001},
           "porosity": {"mean": 0.40, "std": 0.05},
       },
       2: {
           "rhos": {"mean": 300.0, "std": 40.0},
           "n": {"mean": 1.7, "std": 0.2},
           "sigma_sur": {"mean": 0.001, "std": 0.0005},
           "porosity": {"mean": 0.25, "std": 0.05},
       }
   })

   wc_all, sat_all, params = converter.run_monte_carlo(
       n_realizations=10, progress_bar=False
   )

   print(wc_all.shape)

Outputs
-------

- Water content ensembles for each time step.
- Summary statistics (mean, standard deviation, percentiles).

Next
----

- Full example: :doc:`/auto_examples/Ex_MC_Hydro`.
