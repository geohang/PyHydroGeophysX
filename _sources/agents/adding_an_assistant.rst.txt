Adding an assistant
===================

PyHydroGeophysX can host several domain AI assistants side by side. AQUAH works
on hydrogeophysics; GeoSAGE, being ported now, works on geological modelling
from gravity and magnetic data. Each is a plug-in: a small package that
describes itself, and the studio builds its chat, its Workflow page and its
runs from that description. The user picks one at the top of the assistant
panel.

This page is for whoever adds one. It uses GeoSAGE as the running example; its
scaffold is in ``PyHydroGeophysX/agents/assistants/geosage/``.

What an assistant is made of
----------------------------

.. code-block:: text

   PyHydroGeophysX/agents/assistants/
       __init__.py          the registry, the Assistant description, the contract
       aquah/               AQUAH
           __init__.py      ASSISTANT = Assistant(...)
           workflow.py      run(payload, progress, ...)  and  tools()
       geosage/             GeoSAGE (scaffold)
           __init__.py      ASSISTANT = Assistant(..., status="in development")
           tools.py         TOOLS: one Tool per stage
           workflow.py      configure(payload), CONTROLLER_PROMPT, run(...)

Three things are the assistant's own:

1. **A description**, :class:`PyHydroGeophysX.agents.assistants.Assistant`:
   its name and domain, the chat persona, example requests, the kinds of input
   file the Workflow page offers, its glow colours, and where its workflow and
   tools are. Importing it must be cheap, so the workflow and tools are given
   as ``"module:attribute"`` strings and loaded when a run starts.
2. **Tools**: a dictionary of
   :class:`~PyHydroGeophysX.agents.runtime.tools.Tool`, one per capability.
3. **A workflow**: the function "Auto to report" runs, in a process of its own.

Everything else is shared: the controller loop that picks each step, the live
timeline, step-by-step approval, questions to the user, the chat, the report
view and the Light and Dark appearances.

The description
---------------

.. code-block:: python

   from PyHydroGeophysX.agents.assistants import Assistant

   ASSISTANT = Assistant(
       key="geosage",
       name="GeoSAGE",
       domain="Geological modeling",
       title="Geological Semantic Analysis for Geophysical Exploration",
       summary="Turns joint gravity–magnetic inversion models into ...",
       workflow="PyHydroGeophysX.agents.assistants.geosage.workflow:run",
       tools="PyHydroGeophysX.agents.assistants.geosage.tools:TOOLS",
       persona=PERSONA,                # chat rules that are GeoSAGE's own
       examples=("find serpentinite targets for natural hydrogen ...",),
       input_roles=(("Gravity data", "gravity_file"),
                    ("Magnetic data (TMI)", "magnetic_file"), ...),
       studio_modules=("one_click", "gravmag", "joint_inversion", "mesh3d"),
       colors=("#ff9500", "#ffcc00", "#34c759", "#30b0c7"),
       requires_packages=("simpeg",),
       status="in development",        # "ready" once the tools run
   )

``input_roles`` are what the Workflow page offers under *Or add files
yourself*; a run receives them as ``payload["inputs"][key]``. A key ending in
``_dir`` takes a folder, and a key listed in ``ordered_roles`` takes several
files in order. ``folder_classifier=True`` means the workflow can sort a chosen
folder's files into these roles (``mode='classify'``); AQUAH does, GeoSAGE does
not yet. ``studio_modules`` limits which studio modules the chat may drive.
While ``status`` is not ``"ready"``, or a package in ``requires_packages`` is
missing, the assistant is listed in the panel but greyed out with the reason.

Tools
-----

A tool is what the controller chooses between. It declares what it needs and
what it makes, and is offered only once its needs exist, so the order of a run
follows from the data:

.. code-block:: python

   Tool("run_joint_inversion",
        "Jointly invert gravity and magnetic data for density and magnetic "
        "susceptibility on a common mesh (deterministic, SimPEG, cross-gradient "
        "coupling), and report the data fit.",
        handler, requires=("field_data",), produces=("property_models",),
        agent="InversionAgent", label="Run joint gravity-magnetic inversion",
        module="joint_inversion")

The description is read by the model when it chooses, so write it for someone
who does not know the code. ``label`` is what the timeline shows, and
``module`` is the studio page that shows this kind of work.

A handler takes the run's :class:`~PyHydroGeophysX.agents.runtime.context.RunContext`
and returns ``(summary, outputs)``:

.. code-block:: python

   def run_joint_inversion(ctx):
       data = ctx.get("field_data")            # an earlier tool's product
       settings = ctx.config.get("inversion", {})
       # ... SimPEG joint inversion, files written under ctx.output_dir ...
       return (f"Converged in {n} iterations; gravity R² {r2g:.3f}, "
               f"magnetic R² {r2m:.3f}.",
               {"property_models": {"density": rho, "susceptibility": kappa,
                                    "mesh": mesh, "fit": fit}})

The summary is shown to the user the moment the step ends and is what the
controller reads before choosing again, so make it the step's conclusion with
its key numbers. A handler may raise: the failure is recorded, shown, and the
run carries on deciding. An LLM-based stage reaches the model through
``ctx.settings`` (``api_key``, ``model``, ``llm_provider``). A tool that needs
the user to choose calls ``ctx.ask(question, options)``: the question appears in
the Live tab, and with nobody to answer the first option is taken and noted.

The workflow
------------

The contract a workflow function honours is
:data:`PyHydroGeophysX.agents.assistants.WORKFLOW_CONTRACT`. In short: it
receives the run's payload (``request``, ``inputs``, model access,
``output_dir``, ``step_mode``) and returns a result with ``status``,
``interpretation``, ``warnings`` and ``report_files``. Build it on
:func:`PyHydroGeophysX.agents.runtime.entry.drive`, which runs the controller
over your tools and produces the step events the studio needs:

.. code-block:: python

   def run(payload, progress, *, approve=None, on_event=None, events=None, **_):
       config = configure(payload)                       # the ContextAgent
       ctx = RunContext(goal=payload["request"], config=config,
                        output_dir=payload["output_dir"], settings={...})
       on_step = step_by_step(approve) if payload.get("step_mode") and approve else None
       drive(ctx, tools=TOOLS, ask=make_ask(key, model, provider),
             progress_callback=progress, on_step=on_step, on_event=on_event,
             prompt=CONTROLLER_PROMPT, finish="report_files")
       return {"status": ..., "interpretation": ..., "report_files": ctx.get("report_files")}

``prompt`` frames the controller's decision for your domain (it must keep the
``{transcript}`` and ``{menu}`` fields), and ``finish`` names the product that
has to exist before the run may end. GeoSAGE's is the reviewed report, so it
cannot stop at the draft.

Registering it
--------------

Inside this repository, add the package beside ``aquah`` and ``geosage`` and
list it in ``BUILTIN`` in ``PyHydroGeophysX/agents/assistants/__init__.py``.

From a separate package, such as the GeoSAGE repository, declare an entry
point; nothing in PyHydroGeophysX needs editing:

.. code-block:: toml

   [project.entry-points."pyhydrogeophysx.assistants"]
   geosage = "geosage.pyhydrogeophysx:ASSISTANT"

A plug-in that fails to import is left out and reported by
:func:`PyHydroGeophysX.agents.assistants.load_errors`; it never takes the
others down.

What the studio does with it
----------------------------

- The assistant panel lists every registered assistant. Choosing one starts a
  new conversation with its persona and examples, and is remembered.
- The Workflow page offers its input roles and names it in everything it says;
  the Live tab shows its steps, each with the controller's reason, the module,
  the time taken, what it found and the figures it wrote.
- The glow round the central area and the orb take its colours.
- With *Approve each step before it runs*, every step waits for the user in the
  Live tab; a tool's own question appears there too.
- While the controller decides, its reasoning streams into the Live tab:
  :func:`~PyHydroGeophysX.agents.runtime.entry.make_ask` streams the reply, and
  ``drive`` sends the ``"why"`` field on as ``phase="thought"`` events. Keep the
  reasoning first in your controller prompt's JSON, as GeoSAGE's does.
- Pause and notes from the user reach your run through ``drive`` too
  (:mod:`PyHydroGeophysX.agents.runtime.steering`): a note is added to
  ``ctx.guidance`` and the transcript, and the next decision may change the
  settings in :data:`~PyHydroGeophysX.agents.runtime.recovery.ADJUSTABLE`.
- The route across the top of the Live tab comes from your tools'
  ``requires`` and ``produces``: :func:`~PyHydroGeophysX.agents.runtime.controller.route_ahead`
  works out, from what the run has, which tools still stand between it and
  ``finish``. A ``when`` gate that looks inside an artifact should check
  ``ctx.projected(key)`` first and answer from the configuration, since a
  projected artifact has nothing inside it yet.

Porting GeoSAGE
---------------

The paper's seven agents map onto the scaffold like this:

=================  ==============================================================
ContextAgent       ``workflow.configure``: request to configuration
DataAgent          ``prepare_data``: bounds, fields, gridding, uncertainties
PetrologyAgent     ``compile_priors``, and unit/group naming in ``build_quasi_geology``
InversionAgent     ``run_joint_inversion``: settings and the SimPEG backend
GeoAgent           ``build_quasi_geology``: prior-guided or GMM/BIC grouping
ReportAgent        ``write_report``: interpretation, target ranking, uncertainty
ReviewAgent        ``review_report``: checks, revision, the final report
=================  ==============================================================

A workable order is to fill in the deterministic stages first
(``prepare_data``, ``run_joint_inversion``, ``build_quasi_geology``) so a run
produces models and figures without a model key, then the LLM stages, and
then:

1. set ``status="ready"`` in ``geosage/__init__.py``;
2. replace the stand-in handlers in
   ``test_an_assistant_runs_its_own_tools_to_its_own_final_product``
   (``tests/test_agents.py``) with a small real case, or add one such test;
3. run it in the studio with *Approve each step before it runs* switched on,
   in Light and Dark.

To try the chain before the stages exist, replace the handlers with stand-ins
that return the declared products, as that test does; the studio's timeline
then shows the whole run.
