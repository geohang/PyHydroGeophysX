Getting Started
===============

.. figure:: /_static/concepts_diagram.svg
   :alt: Hydrology to geophysics workflow diagram
   :align: center
   :width: 90%

PyHydroGeophysX connects hydrologic model outputs to petrophysical models,
then into geophysical forward modeling and inversion. The goal is a repeatable
workflow you can apply to field or synthetic data.

Install
-------

Start with the full installation guide at :doc:`/installation`.

.. code-block:: bash

   pip install pyhydrogeophysx

First Run
---------

Verify the install and load a small output slice.

.. code-block:: python

   import PyHydroGeophysX as phg
   from PyHydroGeophysX.model_output import MODFLOWWaterContent

   print(phg.__version__)
   wc = MODFLOWWaterContent("model_dir", idomain).load_timestep(0)

First Workflow
--------------

A common first workflow is field ERT QC and export for inversion.

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
   export_for_inversion(ert, outdir="results/ert_qc", fmt="pgimli")

Where to Go Next
----------------

Tutorials

- :doc:`/tutorials/field_ert_data_qc`

Best examples

- :doc:`/auto_examples/Ex_model_output`
- :doc:`/auto_examples/Ex_TL_inversion`
