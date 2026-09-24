:sd_hide_title:
:html_theme.sidebar_secondary.remove: true

PyHydroGeophysX
===============

.. raw:: html

   <div class="phgx-hero">
     <div class="phgx-hero__text">
       <p class="phgx-hero__eyebrow">Open-source hydrogeophysics</p>
       <h1 class="phgx-hero__title">PyHydroGeophysX</h1>
       <p class="phgx-hero__lead">Connect hydrological models with geophysical measurements.</p>
       <p class="phgx-hero__body">Predict survey responses from hydrological states, interpret field
       observations, and bring geophysical constraints back into hydrological models.</p>
       <p class="phgx-hero__actions">
         <a class="phgx-btn phgx-btn--primary" href="quickstart.html">Run the Python quickstart</a>
         <a class="phgx-btn" href="agents/desktop_studio.html">Download Desktop Studio</a>
       </p>
       <p class="phgx-hero__more"><a href="examples/index.html">View example workflows &rarr;</a></p>
     </div>
     <div class="phgx-hero__art">
       <svg class="phgx-flow" viewBox="0 0 470 356" role="img"
            aria-label="A loop with two directions. Down the left, a hydrological model feeds
            petrophysics, petrophysics feeds geophysical modelling, and that gives a predicted
            geophysical response. Up the right, measured geophysical data are inverted and fed
            to petrophysics, petrophysics gives hydrological properties, and those return to the
            hydrological model. The two directions meet at the bottom, where the predicted
            response and the measured data are compared against each other."
            xmlns="http://www.w3.org/2000/svg">
         <defs>
           <marker id="phgx-arrow" viewBox="0 0 10 10" refX="9" refY="5"
                   markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">
             <path d="M0,0 L10,5 L0,10 z" fill="currentColor"/>
           </marker>
           <!-- A marker resolves currentColor against itself, not against the
                path that references it, so the compare arrowheads need their
                own marker to pick up the accent colour. -->
           <marker id="phgx-arrow-compare" viewBox="0 0 10 10" refX="9" refY="5"
                   markerWidth="5" markerHeight="5" orient="auto-start-reverse">
             <path d="M0,0 L10,5 L0,10 z" class="phgx-flow__compare-head"/>
           </marker>
         </defs>

         <g class="phgx-flow__box">
           <rect x="135" y="8"   width="200" height="42" rx="9"/>
           <rect x="10"  y="98"  width="200" height="42" rx="9"/>
           <rect x="260" y="98"  width="200" height="42" rx="9"/>
           <rect x="10"  y="190" width="200" height="42" rx="9"/>
           <rect x="260" y="190" width="200" height="42" rx="9"/>
           <rect x="10"  y="282" width="200" height="52" rx="9"/>
           <rect x="260" y="282" width="200" height="52" rx="9"/>
         </g>

         <g class="phgx-flow__link">
           <path d="M135,29 H116 Q110,29 110,35 V94" marker-end="url(#phgx-arrow)"/>
           <path d="M110,140 V186" marker-end="url(#phgx-arrow)"/>
           <path d="M110,232 V278" marker-end="url(#phgx-arrow)"/>
           <path d="M360,278 V236" marker-end="url(#phgx-arrow)"/>
           <path d="M360,186 V144" marker-end="url(#phgx-arrow)"/>
           <path d="M360,94 V35 Q360,29 354,29 H341" marker-end="url(#phgx-arrow)"/>
         </g>

         <g class="phgx-flow__compare">
           <path d="M216,308 H254" marker-start="url(#phgx-arrow-compare)"
                 marker-end="url(#phgx-arrow-compare)" stroke="currentColor"
                 stroke-width="2" fill="none"/>
           <text x="235" y="272" text-anchor="middle" class="phgx-flow__note">compare</text>
         </g>

         <g class="phgx-flow__label" text-anchor="middle">
           <text x="235" y="34">Hydrological model</text>
           <text x="110" y="124">Petrophysics</text>
           <text x="360" y="124">Hydrological properties</text>
           <text x="110" y="216">Geophysical modelling</text>
           <text x="360" y="216">Petrophysics</text>
           <text x="110" y="303">Predicted</text>
           <text x="110" y="321">geophysical response</text>
           <text x="360" y="303">Measured</text>
           <text x="360" y="321">geophysical data</text>
         </g>

         <g class="phgx-flow__sub">
           <text x="370" y="262">inversion</text>
         </g>
       </svg>
     </div>
   </div>

Start with what you have
------------------------

Choose the path that matches your data and research question.

.. grid:: 1 1 3 3
   :gutter: 3
   :class-container: phgx-cards

   .. grid-item-card:: I have a hydrological model
      :link: tutorials/hydrology_to_ert
      :link-type: doc

      Start from MODFLOW, ParFlow, water content, saturation or porosity. Apply
      petrophysical relationships and predict ERT, seismic, EM or potential-field
      measurements.

      +++
      Hydrology to geophysics

   .. grid-item-card:: I have geophysical observations
      :link: tutorials/field_ert_data_qc
      :link-type: doc

      Quality-check field measurements, recover physical-property models, and
      translate the results into hydrologically meaningful quantities.

      +++
      Geophysics to hydrology

   .. grid-item-card:: I have repeated or multiple surveys
      :link: tutorials/time_lapse_monitoring
      :link-type: doc

      Analyse change through time, introduce structural information, combine
      complementary methods, and assess uncertainty.

      +++
      Monitoring and integration

Choose how you want to work
---------------------------

The same workflows are available from a script, from a desktop interface, or
through natural language.

.. grid:: 1 1 3 3
   :gutter: 3
   :class-container: phgx-cards

   .. grid-item-card:: Python
      :link: quickstart
      :link-type: doc

      Build reproducible workflows, automate repeated analyses, and integrate
      PyHydroGeophysX into research code.

      +++
      Python quickstart

   .. grid-item-card:: Desktop Studio
      :link: agents/desktop_studio
      :link-type: doc

      Process data, inspect surveys, build meshes, run inversions and visualise
      results through an interactive interface.

      +++
      Download Desktop Studio

   .. grid-item-card:: AI-assisted workflows
      :link: agents/index
      :link-type: doc

      Use natural-language guidance to configure, execute and inspect the
      supported hydrogeophysical workflows.

      +++
      Explore AQUAH and agents

See AQUAH in action
-------------------

AQUAH is the built-in assistant. It takes a goal in plain language, configures
the workflow, runs it and reports back. It runs in the desktop application and
in the browser, and both demos below show a complete workflow, not a mock-up.

.. raw:: html

   <div class="phgx-demos">
     <figure class="phgx-demo">
       <div class="setup-video">
         <iframe src="https://www.youtube-nocookie.com/embed/cSUEGBFGxrI"
           title="AQUAH running a workflow in the PyHydroGeophysX Desktop Studio"
           loading="lazy" allow="fullscreen; picture-in-picture" allowfullscreen></iframe>
       </div>
       <figcaption><strong>Desktop Studio.</strong> The assistant beside the
       scientific modules, with step-by-step approval or a run straight through
       to a report. <a href="agents/desktop_studio.html">Desktop guide</a></figcaption>
     </figure>
     <figure class="phgx-demo">
       <div class="setup-video">
         <iframe src="https://www.youtube-nocookie.com/embed/d4lgs_hQqDo"
           title="The PyHydroGeophysX web agent running a hydrogeophysical workflow"
           loading="lazy" allow="fullscreen; picture-in-picture" allowfullscreen></iframe>
       </div>
       <figcaption><strong>Web app.</strong> The same assistant in a browser, with
       nothing to install. <a href="agents/webapp.html">Web app guide</a></figcaption>
     </figure>
   </div>

   <p class="phgx-actions">
     <a class="phgx-btn phgx-btn--primary"
        href="https://github.com/geohang/PyHydroGeophysX/releases/latest">Download Desktop Studio</a>
     <a class="phgx-btn" href="https://pyhydrogeophysx.streamlit.app/">Open the web app</a>
     <a class="phgx-btn" href="agent_install.html">Install with an AI agent</a>
   </p>

Windows and macOS bundles are published on GitHub Releases. The
:doc:`AI-agent install prompt <agent_install>` is a text file you paste into a
coding assistant to have it check the environment, install the package and
verify startup; :doc:`installation` covers the manual route, with its own video.

Supported methods and models
----------------------------

Use each method on its own, or connect it to a hydrological, monitoring,
structural or joint workflow.

.. raw:: html

   <p class="phgx-tags">
     <a class="phgx-tag" href="methods/ert.html">ERT</a>
     <a class="phgx-tag" href="methods/seismic.html">Seismic refraction</a>
     <a class="phgx-tag" href="methods/em.html">TDEM</a>
     <a class="phgx-tag" href="methods/em.html">FDEM</a>
     <a class="phgx-tag" href="methods/potential_fields.html">Gravity</a>
     <a class="phgx-tag" href="methods/potential_fields.html">Magnetics</a>
     <a class="phgx-tag phgx-tag--hydro" href="methods/hydrological_models.html">MODFLOW</a>
     <a class="phgx-tag phgx-tag--hydro" href="methods/hydrological_models.html">ParFlow</a>
   </p>

:doc:`Compare the methods <methods/index>` to see what each one needs as input
and which workflows it can feed.

Featured workflows
------------------

.. grid:: 1 1 3 3
   :gutter: 3
   :class-container: phgx-cards phgx-cards--figure

   .. grid-item-card:: Hydrological model to ERT
      :link: tutorials/hydrology_to_ert
      :link-type: doc
      :img-top: /_static/Ex_ERT_workflow_fig_04.png
      :img-alt: A resistivity section derived from modelled water content, beside the apparent-resistivity pseudosection an ERT survey would record over it

      Convert a hydrological state into a resistivity model and predict the
      measurements an ERT survey would record.

      +++
      Open workflow

   .. grid-item-card:: Field observations to interpretation
      :link: auto_examples/Ex_MC_Hydro
      :link-type: doc
      :img-top: /_static/Ex_MC_Hydro_fig_02.png
      :img-alt: Water content estimated from a recovered resistivity section, with the spread that petrophysical parameter uncertainty produces

      Recover a model from field measurements, then estimate water content with
      the uncertainty the petrophysical parameters imply.

      +++
      Open workflow

   .. grid-item-card:: Time-lapse monitoring
      :link: tutorials/time_lapse_monitoring
      :link-type: doc
      :img-top: /_static/Ex_TL_inversion_fig_02.png
      :img-alt: Resistivity sections recovered from repeated surveys, showing how the subsurface changed between acquisitions

      Invert a monitoring series together so that the recovered change is
      constrained rather than differenced after the fact.

      +++
      Open workflow

Open-source hydrogeophysics
---------------------------

Python package · Desktop Studio · Reproducible examples · API reference

`GitHub <https://github.com/geohang/PyHydroGeophysX>`_ ·
:doc:`Installation <installation>` ·
:doc:`Data and processing <data_and_processing>` ·
:doc:`API reference <api/index>` ·
:doc:`Usage and downloads <usage>` ·
`Environmental Geophysics course <https://geohang.github.io/environmental-geophysics/>`_

Citing PyHydroGeophysX
----------------------

If PyHydroGeophysX contributes to your research, please cite the software paper
and any numerical engines your analysis actually ran.

.. button-ref:: citation
   :color: primary
   :ref-type: doc

   View citation guide

.. toctree::
   :maxdepth: 1
   :hidden:

   Getting started <getting_started>
   Workflows <tutorials/index>
   Methods <methods/index>
   Examples <examples/index>
   AI workflows <agents/index>
   API reference <api/index>
   Data and processing <data_and_processing>
   Usage and downloads <usage>
   Citation <citation>
