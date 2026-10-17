:orphan:

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

Survey diagnostics and data assimilation on the MODFLOW model
-------------------------------------------------------------

These three examples start from the Treeline catchment MODFLOW model under
``examples/data`` and follow the hillslope transect of ``Ex_ERT_workflow``.
They run the package's own profile interpolation, petrophysics, ERT forward
model and inversion, so they need PyHydroGeophysX with pyGIMLi. Each takes
two to three minutes on the maintainer's machine, uses fixed random seeds,
writes its arrays and figure under ``examples/results/`` and has a companion
notebook.

* ``Ex_sensitivity_analysis.py``: predicts the ERT survey of a MODFLOW state
  with ``hydro_to_ert``, inverts it, and maps cumulative sensitivity, model
  resolution and the Oldenburg-Li depth of investigation over the MODFLOW
  units.
* ``Ex_posterior_uncertainty.py``: inverts the same survey, reads the inversion
  as a Gaussian posterior and propagates its covariance into water-content
  intervals, which it checks against the MODFLOW water content, at two
  strengths of the prior.
* ``Ex_ensemble_assimilation.py``: updates an ensemble of MODFLOW states with
  the day-210 survey in ``data/TL_measurements``, by EnKF and ES-MDA.

To use a ParFlow run instead, read saturation and porosity with
``ParflowSaturation`` and ``ParflowPorosity``; their product is the water
content. ParFlow numbers its layers from the bottom up, so reverse the layer
axis, and give the layer elevations of the ParFlow grid in place of ``top.txt``
and ``bot.npy``.

From the repository root, for example::

    python -m pip install -e ".[geophysics]"
    python examples/Ex_ensemble_assimilation.py
    python -m pytest -q tests/test_core.py -k "posterior or sensitivity or resolution or depth_of_investigation"


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example loads the bundled nine-station SQLite project through the same TEMcompany/TEM2Go reader used by the Qt Studio. It jointly fits the LM and HM gates, applies same-line L2 lateral constraints, and compares the recovered section with the known synthetic resistivity model.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_TEM_LMHM_LCI_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_TEM_LMHM_LCI`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Synthetic LM+HM Line Inversion with Lateral Constraints</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates a complete 1D FDEM workflow: 1. Build synthetic FDEM data from hydrological properties. 2. Invert the synthetic data with FDEMInversion.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_FDEM_workflow_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_FDEM_workflow`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. FDEM Forward + Inversion Workflow</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to load and process outputs from different  hydrological models using PyHydroGeophysX. We show examples for both  ParFlow and MODFLOW models.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_model_output_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_model_output`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Loading and Processing Hydrological Model Outputs</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to incorporate structural information from  seismic velocity models into ERT inversion for improved subsurface imaging.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_Structure_resinv_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_Structure_resinv`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Structure-Constrained Resistivity Inversion</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example uses the promoted, Qt-free potential-field API to inspect field datasets, separate regional and residual anomalies, evaluate analytic body responses, and run compact 3D SimPEG inversions.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_gravity_magnetics_inversion_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_gravity_magnetics_inversion`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Gravity and Magnetics Processing and Inversion</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example compares the sparse lower-RAM solver path (``save_memory=True``) with the standard time-lapse inversion path (``save_memory=False``). Both runs use the same measurements, mesh, and inversion parameters so runtime, process memory, and recovered resistivity distributions can be compared directly.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_TL_inversion_memory_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_TL_inversion_memory`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Memory-Optimized Time-Lapse ERT Inversion</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Update the MODFLOW water content of a hillslope transect with one ERT survey, using the ensemble Kalman filter (EnKF) and the ensemble smoother with multiple data assimilation (ES-MDA). The states come from the Treeline catchment MODFLOW model shipped under examples/data: TL_measurements holds the simulated water content of the final model year on the survey mesh, one row per day, and the Wenner survey that this water content predicts on day 210, in the wettest week of that year, with 5% noise. Ex_Time_lapse_measurement wrote both.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_ensemble_assimilation_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_ensemble_assimilation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Assimilating an ERT Survey into a MODFLOW Water-Content Forecast</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Check what an ERT survey can resolve before reading water content out of its inversion. The hydrological state is the water content of the Treeline catchment MODFLOW model shipped under examples/data, on the hillslope transect of Ex_ERT_workflow. hydro_to_ert turns it into the 72-electrode Wenner survey it predicts, ERTInversion inverts that survey, and three diagnostics of the inversion follow: cumulative sensitivity (how strongly each cell affects the data), model resolution (how much of each cell&#x27;s value the regularized inversion recovers) and the depth-of-investigation index of Oldenburg and Li (1999) (where the result follows the reference model rather than the data). The diagnostics are then summarized for each hydrostratigraphic unit of the MODFLOW model.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_sensitivity_analysis_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_sensitivity_analysis`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Where an ERT Survey Sees MODFLOW Water Content</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example loads a VTEM line and its survey geometry, inspects measured decays, calibrates the response to a documented reference resistivity, and creates a stitched resistivity section from independent 1D inversions.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_EM_line_section_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_EM_line_section`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Airborne EM Line Inversion</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Run a small MODFLOW 6 comparison using the topography and interpreted regolith / fractured-bedrock depths from Hang Chen&#x27;s Geophysics_informed_models repository. The bundled data are 13 KB and require no runtime data download. This example compares uniform layer thicknesses with spatially varying interpreted interfaces. Hydraulic properties and forcing are illustrative, not the paper&#x27;s calibration. It does not perform seismic inversion or infer conductivity from velocity.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_MODFLOW_geophysics_feedback_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_MODFLOW_geophysics_feedback`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Geophysical structure → MODFLOW → hydrological response</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Estimate how uncertain the water content read from an ERT inversion is, then check that estimate against the MODFLOW state that produced the survey. As in Ex_sensitivity_analysis, the water content of the Treeline catchment MODFLOW model on the Ex_ERT_workflow transect becomes a 72-electrode Wenner survey through hydro_to_ert, and ERTInversion inverts it. The inversion&#x27;s objective is then read as a Gaussian posterior of log resistivity: its data weights give the data covariance, and its regularization, smoothness with a weak pull toward the reference model, gives the prior covariance. linearized_posterior returns the posterior covariance at the recovered model, and propagate_petro_uncertainty samples it through the Waxman-Smits model of each unit into water-content intervals. The inversion runs at two strengths of that prior, to show how much the intervals depend on it. Unlike Ex_MC_Hydro, the petrophysical parameters stay at the values that produced the survey, so the intervals carry the survey&#x27;s uncertainty alone.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_posterior_uncertainty_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_posterior_uncertainty`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Posterior Uncertainty of Water Content Recovered from ERT</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows a minimal, robust workflow for one ERT survey:">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_ERT_single_inversion_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_ERT_single_inversion`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Single ERT File Inversion (No Time-Lapse)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates advanced time-lapse ERT inversion using structural constraints derived from seismic interpretation to monitor subsurface water content changes in layered geological media.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_structure_TLresinv_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_structure_TLresinv`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Structure-Constrained Time-Lapse Resistivity Inversion</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates different approaches for time-lapse electrical  resistivity tomography (ERT) inversion using PyHydroGeophysX.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_TL_inversion_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_TL_inversion`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Time-Lapse ERT Inversion Techniques</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to perform a 2D seismic refraction tomography (SRT) inversion and interpret the results to define subsurface structures.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_SRT_inv_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_SRT_inv`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Seismic Refraction Tomography (SRT) Inversion and Interface Delineation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example runs the package&#x27;s magnetotelluric (MT) chain on the MODFLOW model shipped in examples/data, and then on a field station.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_MT_workflow_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_MT_workflow`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. MT Workflow: From a Hydrological Model to Magnetotelluric Soundings and Back</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example compares direct cross-gradient and geostatistical cross-gradient coupling using the same ERT and SRT field data. Data loading, shared settings, the two inversion runs, result saving, and visualization are presented as separate steps.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_joint_inversion_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_joint_inversion`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Joint ERT-SRT Inversion: Cross-Gradient and Geostatistical Coupling</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the complete workflow for integrating hydrological  model outputs (MODFLOW water content) with Time-Domain Electromagnetic (TDEM)  forward modeling and inversion using SimPEG and PyHydroGeophysX.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_TDEM_workflow_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_TDEM_workflow`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. TDEM Workflow: From Hydrological Models to EM Responses and Inversion</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example extracts one two-dimensional profile from a hydrological-model snapshot, builds a common mesh, and simulates ERT, SRT, TDEM, FDEM, and gravity responses. Each processing and forward-modeling stage is kept separate so the intermediate hydrological profiles and mesh properties can be inspected.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_hydro_to_multigeophys_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_hydro_to_multigeophys`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Hydrology to Multi-Geophysics Responses</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates seismic refraction tomography forward modeling for watershed structure characterization using PyHydroGeophysX.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_EX_SRT_forward_thumb.png
    :alt:

  :doc:`/auto_examples/EX_SRT_forward`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Seismic Refraction Tomography (SRT) Forward Modeling</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to create synthetic time-lapse electrical  resistivity tomography (ERT) measurements for watershed monitoring applications.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_Time_lapse_measurement_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_Time_lapse_measurement`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Creating Synthetic Time-Lapse ERT Measurements</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the complete workflow for integrating hydrological  model outputs with ERT forward modeling and inversion using PyHydroGeophysX.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_ERT_workflow_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_ERT_workflow`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. ERT Workflow: From Hydrological Models to ERT responses and Inversion</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates Monte Carlo uncertainty quantification for converting ERT resistivity models to water content estimates.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_MC_Hydro_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_MC_Hydro`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Monte Carlo Uncertainty Quantification for Hydrologic Properties Estimation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the complete workflow for 3D ERT forward modeling using PyHydroGeophysX, integrating hydrological model outputs.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_3D_ERT_forward_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_3D_ERT_forward`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. 3D ERT Forward Modeling with MODFLOW Integration</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/Ex_TEM_LMHM_LCI
   /auto_examples/Ex_FDEM_workflow
   /auto_examples/Ex_model_output
   /auto_examples/Ex_Structure_resinv
   /auto_examples/Ex_gravity_magnetics_inversion
   /auto_examples/Ex_TL_inversion_memory
   /auto_examples/Ex_ensemble_assimilation
   /auto_examples/Ex_sensitivity_analysis
   /auto_examples/Ex_EM_line_section
   /auto_examples/Ex_MODFLOW_geophysics_feedback
   /auto_examples/Ex_posterior_uncertainty
   /auto_examples/Ex_ERT_single_inversion
   /auto_examples/Ex_structure_TLresinv
   /auto_examples/Ex_TL_inversion
   /auto_examples/Ex_SRT_inv
   /auto_examples/Ex_MT_workflow
   /auto_examples/Ex_joint_inversion
   /auto_examples/Ex_TDEM_workflow
   /auto_examples/Ex_hydro_to_multigeophys
   /auto_examples/EX_SRT_forward
   /auto_examples/Ex_Time_lapse_measurement
   /auto_examples/Ex_ERT_workflow
   /auto_examples/Ex_MC_Hydro
   /auto_examples/Ex_3D_ERT_forward


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_examples_python.zip </auto_examples/auto_examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_examples_jupyter.zip </auto_examples/auto_examples_jupyter.zip>`
