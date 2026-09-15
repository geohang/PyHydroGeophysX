PyHydroGeophysX
===============

Connect **geophysical data and hydrological modeling**.
Predict survey responses from hydrological states, interpret observations through
petrophysics, and investigate how subsurface conditions change through time.
Start with :doc:`the model–data interaction <tutorials/hydro_geophysical_interaction>`,
then explore the workflows in Python or Desktop Studio.

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: Download Desktop Studio
      :link: agents/desktop_studio
      :link-type: doc

      Start with the interactive application. Find Windows and macOS downloads
      and installation instructions.

   .. grid-item-card:: Start with Python
      :link: quickstart
      :link-type: doc

      Install the package and run your first hydrogeophysical example.

   .. grid-item-card:: Explore examples
      :link: examples/index
      :link-type: doc

      Browse scripts, datasets and scientific figures for your workflow.

The loop this package closes
----------------------------

Information travels in both directions. A hydrological state predicts what a
survey should measure, and a survey constrains what the model should hold.

.. raw:: html

   <svg class="phgx-loop" viewBox="0 0 860 190" role="img"
        aria-label="Four stages: predict a response from a hydrological state,
        measure in the field, recover a model by inversion, interpret and write
        the result back to the hydrological model."
        xmlns="http://www.w3.org/2000/svg">
     <defs>
       <marker id="phgx-arrow" viewBox="0 0 10 10" refX="9" refY="5"
               markerWidth="6" markerHeight="6" orient="auto-start-reverse">
         <path d="M0,0 L10,5 L0,10 z" fill="currentColor"/>
       </marker>
     </defs>
     <g fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.85">
       <rect x="6"   y="20" width="185" height="58" rx="8"/>
       <rect x="228" y="20" width="185" height="58" rx="8"/>
       <rect x="450" y="20" width="185" height="58" rx="8"/>
       <rect x="672" y="20" width="182" height="58" rx="8"/>
       <path d="M195,49 L224,49" marker-end="url(#phgx-arrow)"/>
       <path d="M417,49 L446,49" marker-end="url(#phgx-arrow)"/>
       <path d="M639,49 L668,49" marker-end="url(#phgx-arrow)"/>
       <path d="M763,82 L763,132 Q763,146 749,146 L112,146 Q98,146 98,132 L98,82"
             marker-end="url(#phgx-arrow)"/>
     </g>
     <g fill="currentColor" font-size="14" text-anchor="middle">
       <text x="98"  y="45">1. Predict</text>
       <text x="320" y="45">2. Measure</text>
       <text x="542" y="45">3. Recover</text>
       <text x="763" y="45">4. Interpret</text>
     </g>
     <g fill="currentColor" font-size="11.5" text-anchor="middle" opacity="0.72">
       <text x="98"  y="65">state to response</text>
       <text x="320" y="65">field data and QC</text>
       <text x="542" y="65">inversion</text>
       <text x="763" y="65">water content, structure</text>
       <text x="430" y="166">write back to the hydrological model</text>
     </g>
   </svg>

Stage 1 uses ``Hydro_modular``; stage 4 ends in :doc:`model_input
<api/model_input>`, which writes MODFLOW 6 and ParFlow inputs without touching
the source model. :doc:`How the two connect
<tutorials/hydro_geophysical_interaction>` walks the whole loop.

Your surveys, from data to results
----------------------------------

.. figure:: /_static/studio_overview.png
   :alt: Desktop Studio interface for interactive geophysical processing
   :width: 100%

   Desktop Studio brings processing controls and scientific results into one workspace.
   Available features depend on the installed release and optional engines.

Work with **ERT, seismic, TDEM/FDEM, gravity and magnetics**, or connect hydrological
outputs through petrophysical relationships. See the :doc:`desktop guide
<agents/desktop_studio>` for supported inputs and project workflows.

Video walkthroughs: :doc:`Qt Desktop Studio <agents/desktop_studio>`,
:doc:`web agent <agents/webapp>`, :doc:`manual installation <installation>`,
and :ref:`ADERT speed comparison <adert-speed-comparison>`.

Install with an AI agent
------------------------

Prefer guided setup? Watch the installation video and copy a prompt that asks
your AI agent to check the environment, install the package and verify startup.

.. button-ref:: agent_install
   :color: primary

   Watch the video & copy the setup prompt

Choose a workflow
-----------------

One card per stage of the loop above. Each opens a tutorial that names the
inputs it expects and the result it produces.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 1. Predict a survey response
      :link: tutorials/hydrology_to_ert
      :link-type: doc

      Turn hydrological outputs into resistivity or conductivity through
      petrophysics, then simulate what a survey would record. The
      :doc:`electromagnetic route <tutorials/hydrology_to_tdem>` covers
      TDEM and FDEM.

   .. grid-item-card:: 2. Read and check field data
      :link: tutorials/field_ert_data_qc
      :link-type: doc

      Load a field dataset, inspect it, remove what should not be inverted,
      and export an inversion-ready file.
      :doc:`Supported formats <data_and_processing>`.

   .. grid-item-card:: 3. Recover a model
      :link: tutorials/time_lapse_monitoring
      :link-type: doc

      Invert one survey or a monitoring series. Structural constraints and
      :doc:`joint ERT and seismic inversion <tutorials/joint_inversion>`
      combine complementary measurements.

   .. grid-item-card:: 4. Return the result to the model
      :link: tutorials/hydrological_input_updates
      :link-type: doc

      Map an interpreted quantity onto the hydrological grid and write MODFLOW 6
      or ParFlow inputs for the next run.

Any stage can be driven by an assistant instead of by hand: see
:doc:`AI-assisted workflows <tutorials/agent_workflows>`.

More resources
--------------

:doc:`Installation <installation>` · :doc:`Data and processing <data_and_processing>` ·
:doc:`API reference <api/index>` ·
:doc:`Web app <agents/webapp>` · :doc:`Usage and downloads <usage>` ·
`Environmental Geophysics course <https://geohang.github.io/environmental-geophysics/>`_

Citation
--------

Please cite the software paper, and the workflow paper as well if you used the
AI-assisted workflows:

- Chen, H., Niu, Q., and Wu, Y. (2026). *PyHydroGeophysX: An Extensible
  Open-Source Platform for Integrating Hydrological Models with Geophysical
  Measurements*. SoftwareX, in press.
  https://doi.org/10.2139/ssrn.6238293
- Chen, H. (2026). *A Generalizable Automated Geophysical Agent Workflow for
  Accessible Subsurface Hydrology Analysis*. Big Data and Earth System, 100042.
- Chen, H., Niu, Q., Mendieta, A., Bradford, J., and McNamara, J. (2023).
  Geophysics-informed hydrologic modeling of a mountain headwater catchment for
  studying hydrological partitioning in the critical zone. *Water Resources
  Research*, 59(12), e2023WR035280. https://doi.org/10.1029/2023WR035280
  Cite this one for work that uses geophysical observations to inform a
  hydrological model, which is stage 4 of the loop above.

The engines underneath do the numerical work, so please also cite the ones your
analysis actually ran:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - If your analysis used
     - Please also cite
   * - ERT or seismic refraction forward modelling and inversion
     - Rücker, C., Günther, T., and Wagner, F. M. (2017). pyGIMLi.
       *Computers and Geosciences*, 109, 106-123.
   * - Differentiable time-lapse ERT (``engine="adtlert"``)
     - Yang, P., Fang, Z., Liu, Y., Su, X., Feng, D., and Chen, H. (2026).
       ADTLERT. arXiv:2608.14661.
   * - ERT data processing and quality control
     - Blanchy, G., Saneiyan, S., Boyd, J., McLachlan, P., and Binley, A.
       (2020). ResIPy. *Computers and Geosciences*, 137, 104423.
   * - TDEM or FDEM forward modelling and inversion
     - Cockett, R., Kang, S., Heagy, L. J., Pidlisecky, A., and Oldenburg,
       D. W. (2015). SimPEG. *Computers and Geosciences*, 85, 142-154.
   * - MODFLOW, read or written through FloPy
     - Bakker, M., Post, V., Langevin, C. D., Hughes, J. D., White, J. T.,
       Starn, J. J., and Fienen, M. N. (2016). *Groundwater*, 54(5), 733-739.
   * - ParFlow outputs or written ParFlow inputs
     - Kollet, S. J. and Maxwell, R. M. (2006). *Advances in Water Resources*,
       29(7), 945-958; Maxwell, R. M. (2013). *Advances in Water Resources*,
       53, 109-117.
   * - A petrophysical relationship
     - The reference for the model you used, listed with it in
       :doc:`the petrophysics API <api/petrophysics>`.
   * - Ensemble Kalman updating
     - Evensen, G. (2003). *Ocean Dynamics*, 53(4), 343-367.

BibTeX entries for the first group are in the
`project README <https://github.com/geohang/PyHydroGeophysX#citation>`_.

.. toctree::
   :maxdepth: 1
   :hidden:

   Desktop Studio <agents/desktop_studio>
   Getting started <getting_started>
   Tutorials <tutorials/index>
   Examples <examples/index>
   API Reference <api/index>
   AI + Web App <agents/index>
   Usage and Downloads <usage>
