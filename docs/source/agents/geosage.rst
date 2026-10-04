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

**Run without AI** executes the numerical workflow without provider credentials.
Its result is explicitly **Needs review**, not an accepted AI interpretation.
For interpretation and report review, configure a provider in the assistant
panel and use **Auto to report**. Enable per-step approval to inspect each of
the six stages before it runs. The existing process cancellation and between-step
pause mechanisms apply.

Inspect exported figures and the VTK volume in **Saved Results → Visualization**.
The property selector switches between density contrast, susceptibility and
categorical geological identifiers. The clipping plane exposes the interior.
Source mesh coordinates use metres and z elevation, positive up. They must not
be read as depth below ground. Use **Save to Project** to keep the run in history.

Without a compatible 3D renderer the exported sections and data-fit figures
remain viewable. Original source files and model arrays are preserved; new
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
