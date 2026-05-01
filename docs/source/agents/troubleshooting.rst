Troubleshooting
===============

Use this page when an agent or the web app stops before completing a workflow.
Each item lists the symptom, likely cause, and the fastest fix.

.. _data-file-not-found:

Data File Not Found
-------------------

**Symptom:** A loader fails before processing and prints an attempted absolute
path.

**Probable cause:** The path was typed relative to a different working directory,
or the uploaded file was not mapped to the expected config field.

**Fix:** Use the absolute path shown in the error message, or re-upload the file
and confirm the parsed configuration before running.

.. _api-key-missing-or-invalid:

API Key Missing or Invalid
--------------------------

**Symptom:** The app says the LLM provider is not initialized, or an agent fails
while parsing a natural-language request.

**Probable cause:** The selected provider key is missing, expired, or belongs to
a different provider.

**Fix:** Use demo mode for a no-key walkthrough, or enter the correct
``OPENAI_API_KEY``, ``GEMINI_API_KEY``, or ``ANTHROPIC_API_KEY`` before using the
LLM-backed path.

.. _wrong-ert-instrument:

Wrong ERT Instrument Declared
-----------------------------

**Symptom:** ERT loading returns ``needs_review`` and reports different declared
and detected instruments.

**Probable cause:** The selected instrument does not match the file header or
export format.

**Fix:** Change the instrument field in the confirmation form to the detected
instrument, or re-export the file from the expected instrument software.

.. _pygimli-simpeg-import-fails:

pyGIMLi or SimPEG Import Fails
------------------------------

**Symptom:** The app or CLI reports that ``pygimli`` or SimPEG cannot be imported.

**Probable cause:** Optional inversion dependencies are not installed in the
current Python environment.

**Fix:** Install the missing package in the active environment. For pyGIMLi,
``conda install -c gimli pygimli`` is usually the most reliable path.

.. _mesh-generation-fails:

Mesh Generation Fails
---------------------

**Symptom:** The inversion stops during mesh creation or reports invalid electrode
geometry.

**Probable cause:** Electrode positions are missing, duplicated, non-monotonic, or
outside the expected survey geometry.

**Fix:** Inspect the electrode file, remove duplicates, and verify units and
coordinate columns before rerunning.

.. _inversion-diverges:

Inversion Diverges
------------------

**Symptom:** Chi-squared grows between attempts, or the model disagrees with the
data more after an update.

**Probable cause:** The starting model, damping value, data errors, or electrode
geometry are inconsistent with the observations.

**Fix:** Increase damping (lambda), check electrode positions, and inspect noisy
data points before launching another full inversion.

.. _streamlit-upload-size-exceeded:

Streamlit Upload Size Exceeded
------------------------------

**Symptom:** Upload fails in the hosted app, or a data-loading agent rejects a
large file.

**Probable cause:** The file is too large for hosted Streamlit execution.

**Fix:** Run the app locally for large datasets, or split/compress the data before
uploading. Files larger than 500 MB should be treated as local-workflow inputs.

.. _joint-ert-srt-geometry-mismatch:

Joint ERT+SRT Geometry Mismatch
-------------------------------

**Symptom:** A joint workflow stops after loading both ERT and SRT data.

**Probable cause:** The ERT electrodes and SRT receiver/source geometry do not
describe the same profile, coordinate system, or domain.

**Fix:** Confirm that both files use the same units, origin, profile direction,
and survey extent.

.. _llm-rate-limit-quota-error:

LLM Rate Limit or Quota Error
-----------------------------

**Symptom:** Natural-language parsing or interpretation fails with a provider
rate-limit, quota, or billing message.

**Probable cause:** The provider account is out of quota or temporarily throttled.

**Automatic retry (v0.3+):** All three providers now retry automatically up to
three times with exponential back-off (1 s, 2 s, 4 s) whenever the error text
contains ``rate limit``, ``resource exhausted``, ``quota``, ``too many requests``,
or ``429``.  Most transient throttle errors resolve within one retry and require
no user action.

If **all three attempts fail** (extended outage or hard quota cap):

* Wait a few minutes and re-run the workflow.
* Switch provider or model via the ``llm_provider`` / ``model`` parameters.
* Use ``dry_run=True`` or demo mode for a no-LLM walkthrough.
* For persistent quota issues, check your provider billing dashboard.

.. _resuming-failed-workflow:

Resuming a Failed or Interrupted Workflow
------------------------------------------

**Symptom:** A workflow crashes or is killed mid-run (power failure, out-of-memory,
unexpected error at step 3 of 5).  You do not want to repeat the expensive steps
that already finished.

**Fix (v0.3+):** Restart with ``resume=True``.

.. code-block:: python

    from PyHydroGeophysX.agents import AgentCoordinator

    coordinator = AgentCoordinator(api_key=api_key, output_dir='./results')
    # ... register agents ...

    # Skip steps that already have a checkpoint
    results = coordinator.execute_workflow(config, resume=True)

The coordinator looks for ``<output_dir>/checkpoints/<step>.pkl`` files.  Any
step that has a valid checkpoint is loaded from disk rather than re-executed.
Steps that failed or never ran are executed normally.

To force a full re-run despite existing checkpoints, remove the ``checkpoints/``
folder:

.. code-block:: bash

    rm -rf ./results/checkpoints/

.. _mesh3d-gmsh-not-found:

3D Mesh Generation Fails (gmsh not found)
------------------------------------------

**Symptom:** The 3D Mesh Builder app or ``Mesh3DCreator`` raises a
``gmsh not found`` or ``FileNotFoundError`` for the GMSH executable.

**Probable cause:** GMSH is not installed or not on ``PATH``.

**Fix:**

.. code-block:: bash

    # Conda (recommended)
    conda install -c conda-forge gmsh

    # Or download from https://gmsh.info and add the binary to PATH.

After installing, verify with:

.. code-block:: bash

    gmsh --version

Then re-launch the app: ``python -m PyHydroGeophysX.gui_mesh3d``

.. _mesh3d-plotly-missing:

3D Electrode View is Blank
---------------------------

**Symptom:** The **Electrode View** tab in the 3D Mesh Builder shows no chart or
a plain axis without points.

**Probable cause:** ``plotly`` is not installed in the active environment.

**Fix:**

.. code-block:: bash

    pip install plotly

.. _quality-loop-never-reaches-threshold:

Quality Loop Never Reaches Threshold
------------------------------------

**Symptom:** The quality loop exhausts all attempts and returns ``needs_review``.

**Probable cause:** Automatic parameter adjustment improved the result too little,
or the data geometry/noise prevents a reliable model.

**Fix:** Review the attempt log, adjust damping or data errors manually, and treat
the result as preliminary until the residuals and geometry are checked.

.. _ambiguous-natural-language-request:

Ambiguous Natural-Language Request
----------------------------------

**Symptom:** The context agent returns ``needs_review`` and lists missing fields.

**Probable cause:** The request does not specify enough information, such as data
type, file path, instrument, or inversion mode.

**Fix:** Add the missing fields in the request or edit them in the confirmation
form before running.

