:orphan:

Examples Gallery
================

This gallery contains comprehensive examples demonstrating the capabilities of PyHydroGeophysX for integrating hydrological model outputs with geophysical forward modeling and inversion.

The examples are organized to show the complete workflow from loading hydrological model data to performing geophysical inversions:

**Data Processing:**

* **Ex_ERT_data_process.py**: Loading, quality control, and exporting field ERT data from commercial instruments (E4D, Syscal, ABEM, etc.) using RESIPY integration

**Basic Examples:**

* **Ex_model_output.py**: Loading and processing MODFLOW and ParFlow model outputs
* **Ex_ERT_workflow.py**: Workflow for integrating hydrological model outputs with ERT forward modeling and inversion

**Time-Lapse Analysis:**

* **Ex_Time_lapse_measurement.py**: Creating synthetic time-lapse ERT measurements
* **Ex_TL_inversion.py**: Time-lapse ERT inversion techniques
* **Ex_structure_TLresinv.py**: Structure-constrained time-lapse inversion

**Seismic Methods:**

* **EX_SRT_forward.py**: Seismic refraction tomography (SRT) forward modeling
* **EX_SRT_inv.py**: Seismic refraction tomography (SRT) inversion and analysis

**Advanced Applications:**

* **Ex_Structure_resinv.py**: Structure-constrained resistivity inversion
* **Ex_MC_Hydro.py**: Monte Carlo uncertainty quantification for water content estimation

**3D and Electromagnetic Methods (NEW):**

* **Ex_3D_ERT_forward.ipynb**: 3D ERT forward modeling with MODFLOW integration, mesh creation using Mesh3DCreator, and PyVista visualization
* **Ex_TDEM_workflow.ipynb**: Time-Domain Electromagnetic (TDEM) forward modeling and inversion from hydrological models using SimPEG

Each example includes detailed comments and demonstrates best practices for watershed geophysical monitoring applications.


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to load, quality control, and export field ERT data using PyHydroGeophysX&#x27;s data_processing module with RESIPY integration.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_ERT_data_process_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_Ex_ERT_data_process.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">ERT Field Data Processing with RESIPY</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to load and process outputs from different  hydrological models using PyHydroGeophysX. We show examples for both  ParFlow and MODFLOW models.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_model_output_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_Ex_model_output.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Loading and Processing Hydrological Model Outputs</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to incorporate structural information from  seismic velocity models into ERT inversion for improved subsurface imaging.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_Structure_resinv_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_Ex_Structure_resinv.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Structure-Constrained Resistivity Inversion</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to perform a 2D seismic refraction tomography (SRT)  inversion and interpret the results to define subsurface structures.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_SRT_inv_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_Ex_SRT_inv.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Seismic Refraction Tomography (SRT) Inversion and Interface Delineation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates advanced time-lapse ERT inversion using structural constraints derived from seismic interpretation to monitor subsurface water  content changes in layered geological media.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_structure_TLresinv_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_Ex_structure_TLresinv.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Structure-Constrained Time-Lapse Resistivity Inversion</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates different approaches for time-lapse electrical  resistivity tomography (ERT) inversion using PyHydroGeophysX.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_TL_inversion_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_Ex_TL_inversion.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Time-Lapse ERT Inversion Techniques</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates seismic refraction tomography forward modeling for watershed structure characterization using PyHydroGeophysX.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_EX_SRT_forward_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_EX_SRT_forward.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Seismic Refraction Tomography (SRT) Forward Modeling</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to create synthetic time-lapse electrical  resistivity tomography (ERT) measurements for watershed monitoring applications.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_Time_lapse_measurement_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_Ex_Time_lapse_measurement.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Creating Synthetic Time-Lapse ERT Measurements</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the complete workflow for integrating hydrological  model outputs with ERT forward modeling and inversion using PyHydroGeophysX.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_ERT_workflow_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_Ex_ERT_workflow.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. ERT Workflow: From Hydrological Models to ERT responses and Inversion</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates Monte Carlo uncertainty quantification for  converting ERT resistivity models to water content estimates.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_Ex_MC_Hydro_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_Ex_MC_Hydro.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ex. Monte Carlo Uncertainty Quantification for Hydrologyic Properties Estimation</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/Ex_ERT_data_process
   /auto_examples/Ex_model_output
   /auto_examples/Ex_Structure_resinv
   /auto_examples/Ex_SRT_inv
   /auto_examples/Ex_structure_TLresinv
   /auto_examples/Ex_TL_inversion
   /auto_examples/EX_SRT_forward
   /auto_examples/Ex_Time_lapse_measurement
   /auto_examples/Ex_ERT_workflow
   /auto_examples/Ex_MC_Hydro


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_examples_python.zip </auto_examples/auto_examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_examples_jupyter.zip </auto_examples/auto_examples_jupyter.zip>`
