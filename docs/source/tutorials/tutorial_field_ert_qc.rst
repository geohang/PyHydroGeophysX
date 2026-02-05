Field ERT QC and Export
=======================

This tutorial walks through field ERT data loading, quality control, and export
for inversion. It is based on `examples/Ex_ERT_data_process.py` and the rendered
example :doc:`/auto_examples/Ex_ERT_data_process`.

.. figure:: /auto_examples/images/thumb/sphx_glr_Ex_ERT_data_process_thumb.png
   :alt: ERT QC example thumbnail
   :align: center
   :width: 70%

Prerequisites
-------------

- Install RESIPY for instrument parsers.
- Prepare a project directory with raw ERT files.

.. code-block:: bash

   pip install resipy

Steps
-----

1. Load field data and run QC.

.. code-block:: python

   from PyHydroGeophysX.data_processing.ert_data_agent import (
       load_ert_resipy, qc_and_visualize, export_for_inversion
   )

   ert = load_ert_resipy(
       project_dir="data/ERT/E4D",
       data_file="data/ERT/E4D/2021-10-08_1400.ohm",
       instrument="E4D"
   )

   qc_and_visualize(ert, outdir="results/ert_qc")

2. Export for inversion (pyGIMLi/BERT format).

.. code-block:: python

   export_for_inversion(ert, outdir="results/ert_qc", fmt="pgimli")

Outputs
-------

- QC plots (histogram, pseudosection)
- Summary statistics in JSON
- Exported inversion-ready data file

Next
----

- Try :doc:`/tutorials/tutorial_hydro_to_ert` for hydro-to-ERT modeling.
- Review the full example: :doc:`/auto_examples/Ex_ERT_data_process`.
