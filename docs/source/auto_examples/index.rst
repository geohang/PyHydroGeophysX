:orphan:

Examples Gallery
================

This gallery contains comprehensive examples demonstrating the capabilities of PyHydroGeophysX for integrating hydrological model outputs with geophysical forward modeling and inversion.

The examples are organized to show the complete workflow from loading hydrological model data to performing geophysical inversions:

**Basic Examples:**

* **Ex_model_output.py**: Loading and processing MODFLOW and ParFlow model outputs
* **Ex_ERT_workflow.py**: Workflow for integrating hydrological model outputs with ERT forward modeling and inversion

**Time-Lapse Analysis:**

* **Ex_Time_lapse_measurement.py**: Creating synthetic time-lapse ERT measurements
* **Ex_TL_inversion.py**: Time-lapse ERT inversion techniques
* **Ex_TL_inversion_memory.py**: Comparing memory-optimized and standard time-lapse ERT inversion
* **Ex_structure_TLresinv.py**: Structure-constrained time-lapse inversion

**Field Data Processing and Inversion:**

* **Ex_ERT_single_inversion.py**: Single-survey ERT inversion from field data
* **Ex_EM_line_section.py**: Airborne VTEM line calibration and stitched 1D inversion
* **Ex_TEM_LMHM_LCI.py**: Bundled nine-station LM+HM project and line LCI
  inversion tested against a known resistivity model
* **Ex_gravity_magnetics_inversion.py**: Gravity and magnetic QC, forward modeling, and compact 3D inversion

**Seismic Methods:**

* **EX_SRT_forward.py**: Seismic refraction tomography (SRT) forward modeling
* **Ex_SRT_inv.py**: Seismic refraction tomography (SRT) inversion and analysis

**Advanced Applications:**

* **Ex_Structure_resinv.py**: Structure-constrained resistivity inversion
* **Ex_joint_inversion.py**: Joint ERT-SRT inversion with cross-gradient coupling and geostatistical regularization
* **Ex_hydro_to_multigeophys.py**: Converting one hydrological profile to ERT, SRT, TDEM, FDEM, and gravity responses
* **Ex_MC_Hydro.py**: Monte Carlo uncertainty quantification for water content estimation

Each example includes detailed comments and demonstrates best practices for watershed geophysical monitoring applications.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates a complete 1D FDEM workflow: 1. Build synthetic FDEM data from hydrological properties. 2. Invert the synthetic data with FDEMInversion.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_FDEM_workflow_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_FDEM_workflow`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">FDEM Forward + Inversion Workflow</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="This example loads the bundled nine-station SQLite project through the same TEMcompany/TEM2Go reader used by the Qt Studio. It jointly fits the LM and HM gates, applies same-line L2 lateral constraints, and compares the recovered section with the known synthetic resistivity model.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_TEM_LMHM_LCI_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_TEM_LMHM_LCI`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Synthetic LM+HM line inversion with lateral constraints.</div>
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

      <div class="sphx-glr-thumbnail-title">Memory-Optimized Time-Lapse ERT Inversion</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows a minimal, robust workflow for one ERT survey:">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_ERT_single_inversion_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_ERT_single_inversion`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Single ERT File Inversion (No Time-Lapse)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates advanced time-lapse ERT inversion using structural constraints derived from seismic interpretation to monitor subsurface water  content changes in layered geological media.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to perform a 2D seismic refraction tomography (SRT)  inversion and interpret the results to define subsurface structures.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_SRT_inv_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_SRT_inv`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Seismic Refraction Tomography (SRT) Inversion and Interface Delineation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example compares direct cross-gradient and geostatistical cross-gradient coupling using the same ERT and SRT field data. Data loading, shared settings, the two inversion runs, result saving, and visualization are presented as separate steps.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_joint_inversion_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_joint_inversion`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Joint ERT-SRT Inversion: Cross-Gradient and Geostatistical Coupling</div>
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

      <div class="sphx-glr-thumbnail-title">Hydrology to Multi-Geophysics Responses</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates Monte Carlo uncertainty quantification for  converting ERT resistivity models to water content estimates.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_MC_Hydro_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_MC_Hydro`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Monte Carlo Uncertainty Quantification for Hydrologyic Properties Estimation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the complete workflow for 3D ERT forward modeling using PyHydroGeophysX, integrating hydrological model outputs.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_3D_ERT_forward_thumb.png
    :alt:

  :doc:`/auto_examples/Ex_3D_ERT_forward`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D ERT Forward Modeling with MODFLOW Integration</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/Ex_FDEM_workflow
   /auto_examples/Ex_model_output
   /auto_examples/Ex_Structure_resinv
   /auto_examples/Ex_TEM_LMHM_LCI
   /auto_examples/Ex_gravity_magnetics_inversion
   /auto_examples/Ex_TL_inversion_memory
   /auto_examples/Ex_EM_line_section
   /auto_examples/Ex_ERT_single_inversion
   /auto_examples/Ex_structure_TLresinv
   /auto_examples/Ex_TL_inversion
   /auto_examples/Ex_SRT_inv
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
