Geophysics: ERT
===============

ERT is central to the PyHydroGeophysX workflow. The package supports field data
processing, survey design, and forward modeling for synthetic studies.

Field Data QC and Export
------------------------

- Load data from commercial instruments via RESIPY
- Generate QC plots and summary statistics
- Export to pyGIMLi/BERT format

.. code-block:: python

   from PyHydroGeophysX.data_processing.ert_data_agent import (
       load_ert_resipy, qc_and_visualize, export_for_inversion
   )

   ert = load_ert_resipy("data/ERT/E4D", "data/ERT/E4D/2021-10-08_1400.ohm", "E4D")
   qc_and_visualize(ert, outdir="results/ert_qc")
   export_for_inversion(ert, outdir="results/ert_qc", fmt="pgimli")

Forward Modeling
----------------

- Build meshes (2D or 3D)
- Convert hydro outputs to resistivity
- Simulate ERT responses for survey planning

Examples
--------

- :doc:`/auto_examples/Ex_ERT_data_process`
- :doc:`/auto_examples/Ex_ERT_workflow`
- :doc:`/auto_examples/Ex_3D_ERT_forward`
