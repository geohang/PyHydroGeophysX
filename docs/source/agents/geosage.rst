GeoSAGE assistant
=================

GeoSAGE provides geological interpretation of joint gravity–magnetic inversion
models as an optional external assistant. Installing its Python package registers
``geosage.pyhydrogeophysx:ASSISTANT`` under the existing
``pyhydrogeophysx.assistants`` entry-point group, replacing the built-in GeoSAGE
scaffold. AQUAH's registry and workflows remain independent.

The source package is maintained at
`ZhengyangFang/GeoSAGE <https://github.com/ZhengyangFang/GeoSAGE>`_. During
integration review, install the matching local source branch; older releases
may not include this entry point. No private input data or results are included.

Install and start
-----------------

From the parent directory of the two source checkouts, in a separate Python
3.11–3.13 environment::

   python -m pip install -c GeoSAGE/constraints-tested.txt -e "./PyHydroGeophysX[desktop,desktop-3d]" -e ./GeoSAGE
   geosage-studio --module one_click

Alternatively launch Studio normally and choose **GeoSAGE** in the assistant
picker. Install the optional plugin into the same environment as Studio; a
frozen desktop build must include the package when rebuilt.

GeoSAGE uses the host's existing OpenAI/Anthropic API providers and the Codex CLI
and Claude Code CLI login bridge. Choose the provider in **Assistant → Settings**;
CLI providers use the same setup and sign-in controls as AQUAH and require no
copied API key. Both guided chat and automatic interpretation use that provider.
The CLI bridge currently carries text rather than image input. Local inspection
and numerical tasks stay offline even if a CLI is already signed in.

Using the workflow
------------------

Copy files in your file manager and paste them into the assistant's chat input,
or drag them into it. Describe your task; AI classification maps files to the
installed GeoSAGE plugin's input roles or identifies them as RAG reference
documents. Ambiguous purposes prompt a clarification. Geological literature
is searched locally and matching excerpts are included in the AI request.
This host chat retrieval is available independently of GeoSAGE's own reference handling.
See :doc:`desktop_studio` for supported reference formats and reading limits.

Choose a Project outside the source data folders. In **Data & reports**, add an
**Existing GeoSAGE inversion folder** to inspect an archive, or a **GeoSAGE
configuration (JSON)** for a configured inversion. The adapter requires the
GeoSAGE input schema and does not infer region, magnetic field or mesh settings
from a free-text goal. The independent geophysical module forms do not change
the GeoSAGE configuration; a notice on those pages makes the separation explicit.

Choose **View results**, **New inversion**, or **AI interpretation**.
Check the inputs and effective configuration
before starting. The two local tasks execute without provider credentials.
Its result is explicitly **Needs review**, not an accepted AI interpretation.
For interpretation and report review, configure a provider in the assistant
panel and select **Generate interpretation** in the workflow page. Enable per-step approval to inspect each of
the six stages before it runs. The existing process cancellation and between-step
pause mechanisms apply.

Completed task workflows open the result page with model, data-fit and report
actions, plus the local save state. Unavailable artifacts have disabled actions.
Saving through Ctrl+S or Saved Results updates that state; switching Project
disables actions for a run absent from the newly selected Project.

Inspect exported figures and the VTK volume in **Saved Results → Visualization**.
The compact view lists models and figures; **Details** reveals metadata and file
management. **Report** displays the saved Markdown report, including its figures.
An offline inspection contains a numerical evidence summary and is not presented
as an AI interpretation. Model sections and data-fit figures share GeoSAGE's
notebook plotting implementation, with scales derived from each dataset.
The property selector switches between density contrast, susceptibility and
categorical geological identifiers. The clipping plane exposes the interior.
The linked sections use physical coordinates and show values at the selected
cell. Clicking a section updates all crosshairs; property changes preserve the
selection, camera and clipping plane. Select two runs with Ctrl-click and choose
**Compare two models** for shared colour scales and a B − A map. Only identical
rectilinear coordinates are accepted; no resampling is performed.
Source mesh coordinates use metres and z elevation, positive up. They must not
be read as depth below ground. Use **Save to Project** to keep the run in history.

Without a compatible 3D renderer the exported sections and data-fit figures
remain viewable. Interactive coordinate-based sections also work without OpenGL
when PyVista can read the exported grid. Original source files and model arrays are preserved; new
products go only into the run's directory.

Extension points
----------------

Domain launchers can call ``qt_apps.launcher.main(argv, assistant="geosage")``
to choose their assistant explicitly after window construction. This works even
when saved preferences select another assistant or QSettings cannot be written.

``Assistant.offline_workflow`` defaults to ``False``. Assistants may opt in when
their workflow can provide useful results without model access; the dedicated
button sends no provider key, model or retrieval settings. It does not change
the behavior of existing assistants.

A VTK artifact can supply ``metadata.scalar_cmaps``, a mapping from scalar names
to Matplotlib colour-map names. The volume viewer uses it as per-field defaults
while respecting user-selected maps. Numeric fields ending in ``" ID"`` use
categorical colours with compact positions for non-contiguous labels. Dataset
arrays are not rewritten for display.

``needs_review`` is preserved in run history and filters. It denotes a completed
computation whose interpretation still requires review; ``incomplete`` denotes
an unfinished workflow.

See the GeoSAGE checkout's ``docs/studio.md`` for input schemas, installation
constraints, scientific display conventions and the optional regression suite.

Optional task setup and focused workspace
-----------------------------------------

``Assistant.workflow_setup`` is an optional lazy ``module:factory`` reference.
It is loaded only by the Qt workflow page, so discovery and headless execution
remain independent of Qt. The factory receives a parent widget and returns a
QWidget with:

* a ``changed`` signal and ``needs_ai``, ``action_label`` and ``primary_role`` properties;
* ``allowed_roles()`` and ``update_inputs(inputs)`` for task-specific inputs;
* ``request()`` for the task objective;
* ``prepare_payload(payload)`` returning a validated payload without writing files.
  It is called before allocating a run, then again with the actual destination;
* optional ``show_error(message)`` for an inline validation message;
* optional ``show_configuration(payload)`` to display the inputs and settings
  passed at launch. A prepared payload may provide ``run_label`` for history.

``Assistant.retrieval`` defaults to ``("rag", "mcp")``; unsupported controls are
hidden and cannot enter a payload. ``focused_workspace=True`` starts with the
assistant and log docks hidden and navigation focused on tasks and saved results.
Toolbar **Assistant** and **View → Show all processing tools** restore those controls. All three
capabilities are optional, preserving the existing AQUAH experience.

VTK artifacts may set ``metadata.linked_sections=True`` and
``metadata.field_metadata`` keyed by cell-field name. Fields may supply numeric
``limits`` and ``units``, or category ``colors`` / ``names`` keyed by string IDs.
Shared limits and category colours apply to 3D and linked orthogonal sections.
Run results may supply ``completion`` with ``numerical``, ``interpretation`` and
``review`` states; the workflow displays them separately from run status.

An optional setup method ``prepare_continuation(result)`` may return a new input
role dictionary and select the next task. A result with a nonempty
``continuation`` then exposes **Interpret these models…**. The host applies the
inputs and opens task setup; it never starts a worker or contacts an AI provider
from this action. Errors remain inline, and the action is disabled when the
original run is absent from the current Project. Assistants without the method
retain their existing behavior.

GeoSAGE writes a local ``studio_checkpoint.json`` after each completed stage and
a ``continue_config.json`` when completed numerical models are reusable. These
support continuing interpretation in a fresh run without repeating inversion;
they are not solver-iteration checkpoints.

Desktop regression checks
-------------------------

With desktop dependencies and the GeoSAGE plugin installed, run::

   python -m pytest tests/test_agents.py tests/test_geosage_plugin.py tests/test_assistant_desktop_extensions.py tests/test_scientific_sections.py tests/test_saved_result_previews.py tests/test_studio_diagnostics.py tests/test_cli_providers.py tests/test_cli_setup.py

The widget tests use Qt's offscreen platform and synthetic temporary inputs.
Windows native folder dialogs and GPU composition also require an interactive
check: open a model, switch between figures and reports, accept and cancel a
Project folder dialog, and return to the model. The model viewer retains its
OpenGL widget across these transitions. An offscreen render alone cannot detect
a black top-level window caused by native composition.
