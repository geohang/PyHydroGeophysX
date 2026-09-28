Examples Gallery
================

For categories, outcomes and a recommended starting point, open
:doc:`Examples by task </examples/index>`.

For a compact model–data interaction with actual hydrological solver execution,
start with :doc:`/tutorials/modflow_feedback` and
:doc:`/auto_examples/Ex_MODFLOW_geophysics_feedback`.

The central theme is the interaction between geophysical data and hydrological
models: predicting observations from model states and interpreting data as
hydrological information. See :doc:`/tutorials/hydro_geophysical_interaction`.

Open the notebooks in JupyterLab or VS Code and run the cells in order: inspect
the data, edit parameters, run the model, then view the results. Every scientific
example has a companion ``.ipynb``; the ``.py`` versions use ``# %%`` cells for
editors and website generation.

This gallery contains Python scripts and notebook downloads. It spans
hydrological model access, petrophysical conversion, multi-method responses,
field inversion, time-lapse monitoring, joint inversion and uncertainty analysis.

Start with :doc:`/auto_examples/Ex_model_output`, then
:doc:`/auto_examples/Ex_ERT_workflow`. Use
:doc:`/auto_examples/Ex_hydro_to_multigeophys` to see how one hydrological profile
connects to several geophysical methods. Review each script’s input paths and
dependencies before running it; the site uses pre-generated figures.

Run order, run time and memory
------------------------------

Most examples read only the shipped files under ``examples/data``. Three read
what another example writes under ``examples/results`` and stop with
``FileNotFoundError`` when it is missing:

* ``Ex_SRT_inv`` reads ``results/SRT_forward/synthetic_seismic_data_long.dat``.
  Run ``EX_SRT_forward`` first; it takes about a minute.
* ``Ex_structure_TLresinv`` reads ``results/Structure_WC/mesh_with_interface.bms``.
  Run ``Ex_Structure_resinv`` first.
* ``Ex_MC_Hydro`` reads the models that ``Ex_structure_TLresinv`` writes to
  ``results/Structure_WC``. Run ``Ex_Structure_resinv`` and then
  ``Ex_structure_TLresinv`` first, which takes about 3 hours.

The heavier examples, as measured on the maintainer's machine:

=============================  ============  =============
Example                        Run time      Memory peak
=============================  ============  =============
``Ex_ERT_workflow``            about 12 min  about 13.5 GB
``Ex_TL_inversion``            about 42 min  about 15 GB
``Ex_structure_TLresinv``      about 3 h
``Ex_joint_inversion``         about 44 min
``Ex_hydro_to_multigeophys``                 about 8 GB
=============================  ============  =============

Self-contained numerical examples
---------------------------------

These examples need only the core NumPy/SciPy/Matplotlib dependencies, use
fixed random seeds, and save arrays and figures under ``examples/results/``.
Each has a companion notebook and can run without survey files or optional
geophysical engines. Their synthetic kernels are teaching surrogates.

* ``Ex_sensitivity_analysis.py``: cumulative sensitivity, model resolution,
  and reference-model dependence (DOI).
* ``Ex_ensemble_assimilation.py``: water-content forecasts updated with EnKF
  and ES-MDA through a hydro-geophysical observation operator.
* ``Ex_posterior_uncertainty.py``: correlated posterior resistivity covariance
  propagated into water-content intervals with fixed petrophysical parameters.

From the repository root, for example::

    python examples/Ex_ensemble_assimilation.py
    python -m pytest -q tests/test_core.py -k "posterior or sensitivity or resolution"
