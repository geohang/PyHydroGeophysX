Agents, Web App and Desktop
===========================

Choose the assistant that matches your research task, then explore its
workflows and setup below.

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: phgx-cards

   .. grid-item-card:: AQUAH for hydrogeophysics
      :link: aquah-assistant
      :link-type: ref

      Connect hydrological models with geophysical observations. Use
      natural-language guidance to configure, run and inspect hydrogeophysical
      workflows in the web app or Desktop Studio.

      +++
      Explore AQUAH workflows and setup

   .. grid-item-card:: GeoSAGE for geological modeling
      :link: geosage-assistant
      :link-type: ref

      Build pseudo-geological models from joint gravity–magnetic inversion
      results and generate geological interpretations. Available as an optional
      assistant in Desktop Studio.

      +++
      Explore GeoSAGE workflows and setup

.. _aquah-assistant:

AQUAH for hydrogeophysics
-----------------------------

AQUAH is PyHydroGeophysX's built-in assistant for natural-language
hydrogeophysical workflows. Start with the hosted web app or use Desktop Studio
for interactive work. Its agent workflows include ERT, seismic refraction,
FDEM and joint ERT+SRT inversion.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Open Agent Web App
      :link: webapp
      :link-type: doc
      :class-card: sd-card-hover

      Open AQUAH in your browser, with launch guidance and required inputs.

   .. grid-item-card:: Desktop Studio
      :link: desktop_studio
      :link-type: doc
      :class-card: sd-card-hover

      Use AQUAH alongside the scientific tools. Download Studio for Windows
      or macOS, or run it from source.

   .. grid-item-card:: Agent Quick Start
      :link: quick_start
      :link-type: doc
      :class-card: sd-card-hover

      Configure providers and run your first AQUAH workflow.

.. image:: /_static/agent.png
   :alt: AQUAH multi-agent hydrogeophysical workflow overview
   :align: center
   :width: 600px

.. _geosage-assistant:

GeoSAGE for geological modeling
-----------------------------------

`GeoSAGE <https://github.com/ZhengyangFang/GeoSAGE>`_ combines joint
gravity–magnetic inversion, pseudo-geological modeling and language-model-assisted
interpretation for the Hannah and Iowa case studies. Install it as an optional
assistant in PyHydroGeophysX Desktop Studio to run configured workflows or
explore existing models.

.. grid:: 1 1 3 3
   :gutter: 3
   :class-container: phgx-cards

   .. grid-item-card:: Joint gravity–magnetic inversion
      :link: https://github.com/ZhengyangFang/GeoSAGE#install

      Recover density and magnetic susceptibility models from gravity and
      magnetic observations using configured, reproducible workflows.

      +++
      Explore the source and setup

   .. grid-item-card:: Geological modeling and interpretation
      :link: https://github.com/ZhengyangFang/GeoSAGE/blob/main/docs/usage.md

      Build pseudo-geological models from recovered physical properties, then
      use specialized agents to generate and review interpretation reports.

      +++
      Read the GeoSAGE workflow guide

   .. grid-item-card:: GeoSAGE in Desktop Studio
      :link: geosage
      :link-type: doc

      Inspect model sections, 3D volumes and data-fit figures. Run numerical
      tasks offline, or configure an AI provider for interpretation and review.

      +++
      Install and use the optional assistant

**GeoSAGE reference**

Fang, Z., Yang, P., Liu, Y., Feng, D., & Chen, H. (2026). GeoSAGE: A Multi-Agent
Workflow for Geological Reasoning From Joint Gravity and Magnetic Inversion
Models. *Computers & Geosciences*, 106282.

:doc:`View the citation guide </citation>` for references to GeoSAGE,
PyHydroGeophysX and the numerical engines used in your analysis.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   webapp
   agent_workbench
   quick_start
   workflows
   agent_reference
   adding_an_assistant
   geosage
   troubleshooting
   overview
