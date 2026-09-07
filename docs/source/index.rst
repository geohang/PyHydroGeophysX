PyHydroGeophysX
===============

Connect hydrological models with geophysical measurements.
Process surveys, recover subsurface models, and explore results in Desktop Studio
or build reproducible workflows with Python.

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
      :link: auto_examples/index
      :link-type: doc

      Browse scripts, datasets and scientific figures for your workflow.

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

Install with an AI agent
------------------------

Prefer guided setup? Watch the installation video and copy a prompt that asks
your AI agent to check the environment, install the package and verify startup.

.. button-ref:: agent_install
   :color: primary

   Watch the video & copy the setup prompt

Choose a workflow
-----------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Hydrology to ERT
      :link: tutorials/hydrology_to_ert
      :link-type: doc

      Convert hydrological outputs to resistivity and simulate measurements.

   .. grid-item-card:: Hydrology to EM
      :link: tutorials/hydrology_to_tdem
      :link-type: doc

      Build layered models and run electromagnetic forward and inversion workflows.

   .. grid-item-card:: Joint ERT + seismic inversion
      :link: tutorials/joint_inversion
      :link-type: doc

      Combine complementary measurements with structural constraints.

   .. grid-item-card:: AI-assisted workflows
      :link: tutorials/agent_workflows
      :link-type: doc

      Configure an assistant and guide an analysis using natural language.

More resources
--------------

:doc:`Installation <installation>` · :doc:`API reference <api/index>` ·
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
   * - MODFLOW coupling through FloPy
     - Bakker, M., Post, V., Langevin, C. D., Hughes, J. D., White, J. T.,
       Starn, J. J., and Fienen, M. N. (2016). *Groundwater*, 54(5), 733-739.

BibTeX entries for every reference above are in the
`project README <https://github.com/geohang/PyHydroGeophysX#citation>`_.

A related study using this coupling: Chen, H., Niu, Q., Mendieta, A., Bradford,
J., and McNamara, J. (2023). Geophysics-informed hydrologic modeling of a
mountain headwater catchment for studying hydrological partitioning in the
critical zone. *Water Resources Research*, 59(12), e2023WR035280.
https://doi.org/10.1029/2023WR035280

.. toctree::
   :maxdepth: 1
   :hidden:

   Desktop Studio <agents/desktop_studio>
   Getting started <getting_started>
   Tutorials <tutorials/index>
   Examples <auto_examples/index>
   API Reference <api/index>
   AI + Web App <agents/index>
   Usage and Downloads <usage>
