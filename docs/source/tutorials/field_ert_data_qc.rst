Field ERT Data QC
=================

Use this workflow when starting from field resistivity files and preparing data for inversion.

Steps
-----

1. Load field data with instrument metadata.
2. Run built-in quality control plots.
3. Export to pyGIMLi/BERT format for inversion.

.. code-block:: python

   from PyHydroGeophysX.data_processing.ert_data_agent import (
       load_ert_resipy, qc_and_visualize, export_for_inversion, LocalRef
   )

   ert = load_ert_resipy(
       project_dir="data/ERT/E4D",
       data_file="data/ERT/E4D/2021-10-08_1400.ohm",
       instrument="E4D",
       crs="local",
       local_ref=LocalRef(origin_x=0.0, origin_y=0.0, azimuth_deg=90.0),
   )
   qc_and_visualize(ert, outdir="results/ert_data_process")
   export_for_inversion(ert, outdir="results/ert_data_process", fmt="pgimli")

