GeoSAGE assistant
================

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

Using the workflow
------------------

Choose a Project outside the source data folders. In **Data & reports**, add an
**Existing GeoSAGE inversion folder** to inspect an archive, or a **GeoSAGE
configuration (JSON)** for a configured inversion. The adapter requires the
GeoSAGE input schema and does not infer region, magnetic field or mesh settings
from a free-text goal. The independent geophysical module forms do not change
the GeoSAGE configuration.

Choose **Inspect existing results · local**, **Run a new inversion · local**, or
**Interpret existing results · with AI**. Check the inputs and effective configuration
before starting. The two local tasks execute without provider credentials.
Its result is explicitly **Needs review**, not an accepted AI interpretation.
For interpretation and report review, configure a provider in the assistant
panel and select **Generate interpretation** in the workflow page. Enable per-step approval to inspect each of
the six stages before it runs. The existing process cancellation and between-step
pause mechanisms apply.

Inspect exported figures and the VTK volume in **Saved Results → Visualization**.
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
* optional ``show_error(message)`` for an inline validation message.

``Assistant.retrieval`` defaults to ``("rag", "mcp")``; unsupported controls are
hidden and cannot enter a payload. ``focused_workspace=True`` starts with the
assistant and log docks hidden and navigation limited to ``studio_modules``.
Toolbar **Assistant** and **View → Show all processing tools** restore those controls. All three
capabilities are optional, preserving the existing AQUAH experience.

VTK artifacts may set ``metadata.linked_sections=True`` and
``metadata.field_metadata`` keyed by cell-field name. Fields may supply numeric
``limits`` and ``units``, or category ``colors`` / ``names`` keyed by string IDs.
Shared limits and category colours apply to 3D and linked orthogonal sections.
Run results may supply ``completion`` with ``numerical``, ``interpretation`` and
``review`` states; the workflow displays them separately from run status.
