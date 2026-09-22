Agent Workflow: folders, references and MCP
================================================================================

One place to ask, one place to inspect
--------------------------------------------------------------------------------

Use AQUAH on the right for the goal, provider, model, API key and execution mode.
Choose **Auto to report**. In the central Workflow page, choose a data folder or
add individual files. Folder selection is local: files are not moved or copied.
When you send the goal, the worker scans the folder and sends filenames and short
text previews to the selected AI provider for classification. Binary file content
is not decoded for classification. API usage may incur charges.

The editable classification table shows each file, suggested role, confidence
and evidence. Review these suggestions and send **continue** in AQUAH. Assign
unknown files a role or Ignore. Confidence is an AI estimate, not a calibrated
probability. Multiple surveys are not silently assumed to be time-lapse data.
Arrange time-lapse files chronologically with the up/down controls after roles
have been applied. Manual file selection remains available.

Scans are limited to 300 supported files, 2 KiB read per text preview, and 1200
characters sent per preview. Hidden, credential-named, linked files and common
generated-output directories are skipped. A truncated scan produces a warning:
select a smaller data folder to include the remaining files.

Coordinates and terrain
--------------------------------------------------------------------------------

The classifier distinguishes measurements, electrode coordinates, geophone
coordinates, independent terrain and reference documents. A file being
classified does not imply every workflow can consume it.

* ERT uses the existing ``electrode_file`` loader for electrode geometry,
  including elevations. Independent DEM/XYZ terrain must be prepared in the
  mesh workflow; it is not silently substituted for electrode coordinates.
* Raw SEG-Y accepts ``geophone_file`` with exactly three numeric columns:
  ``receiver_id x z`` (the optional header must use those names). IDs must match
  the SEG-Y receiver IDs; row order is not used to guess station identity.
* Raw SEG-Y also accepts ``topography_file`` with exactly two numeric columns:
  ``x z``. Elevations are interpolated at source/receiver x before travel-time
  export. Coordinates must share units and datum; missing IDs, duplicate profile
  x values, nonfinite values and out-of-profile points cause an error.
* Existing travel-time files must already contain their sensor geometry.
  GeoTIFF/3D XYZ terrain is recognized as a possible input but is not supported
  by this two-dimensional automatic profile adapter.

Model and reasoning
--------------------------------------------------------------------------------

The default OpenAI model is ``gpt-5.6-luna`` (an explicit ``OPENAI_MODEL`` environment
setting still overrides the default). The Reasoning selector defaults to
``medium`` and offers ``none``, ``low``, ``medium``, ``high``, ``xhigh`` and ``max``.
Reasoning options are sent only to supported OpenAI reasoning-model requests;
legacy models retain their usual parameters. Provider access is account-dependent.

See `OpenAI's GPT-5.6 Luna model documentation
<https://developers.openai.com/api/docs/models/gpt-5.6-luna>`_.

Progress displays actual backend stages and logs, along with elapsed time. It
does not fabricate activity or expose private model chain-of-thought. Stop
terminates the worker; inputs and partial outputs remain available for retry.

RAG: give the AI reference material
--------------------------------------------------------------------------------

**RAG · consult local references** retrieves text excerpts from repository
documentation (when available) and selected reference files. This initial
implementation uses bounded lexical retrieval, not a vector database or internet
search. Supported reference formats are Markdown, reStructuredText and plain
text. PDF/Word references need text extraction before retrieval. A wheel installed
without repository docs can still use explicitly selected reference text files.

Sources, line numbers and excerpts are saved with the report. Retrieved passages
are reference data, not executable instructions. The relevant excerpts are sent
to the selected AI provider. If no matching passages are found, the report says so.

MCP: expose existing project functions
--------------------------------------------------------------------------------

Install the optional SDK (Python 3.10 or newer):

.. code-block:: bash

   python -m pip install "pyhydrogeophysx[mcp]"
   python -m PyHydroGeophysX.mcp_server --root /path/to/project

The standard stdio MCP server exposes:

* ``list_workflows``: registered ERT, seismic, EM, hydrology and other workflows.
* ``scan_data_folder``: bounded local inventory under the configured root.
* ``search_docs``: reference excerpts with source locations.
* ``validate_recipe``: validation of an existing workflow JSON recipe.
* ``run_recipe``: enabled only when the server owner starts with ``--allow-run``;
  executes an existing trusted recipe through the isolated workflow CLI and saves
  results and logs into a unique ``mcp_runs`` directory.

The root scopes inventory and recipe-file selection. It is not an operating
system sandbox: execution-enabled clients and recipe contents must be trusted,
and referenced input files follow the workflow engine's normal access rules.
Do not expose this local server to unauthenticated remote clients.

An example MCP client configuration (use an absolute Python executable path):

.. code-block:: json

   {
     "mcpServers": {
       "pyhydrogeophysx": {
         "command": "/path/to/environment/bin/python",
         "args": ["-m", "PyHydroGeophysX.mcp_server", "--root", "/path/to/project"]
       }
     }
   }

On Windows, use the environment's ``python.exe`` and escaped backslashes or
forward slashes in JSON paths. Add ``--allow-run`` only when you want the client
to execute trusted project recipes. Existing recipes come from Studio workflow
exports; MCP does not provide arbitrary Python or shell execution tools.

The desktop **MCP · connect local tool catalog** option performs a real MCP stdio
connection and reads available workflow APIs into the agent's context. The unified
desktop pipeline still computes through its existing engine; this option does not
route every numerical function call through MCP or connect arbitrary remote servers.

See the `official MCP Python SDK <https://github.com/modelcontextprotocol/python-sdk>`_.

Reports and uncertainty
--------------------------------------------------------------------------------

OpenAI reasoning models use the Responses API for AQUAH function tools, retaining
reasoning state between tool results. This avoids the Luna Chat Completions
restriction on combining function tools with reasoning. OpenAI-compatible
third-party providers continue to use their Chat Completions endpoint.

Follow-up messages refine the current goal, including after folder classification.
Use **New chat** to start a new goal while retaining selected files. The
``max_attempts`` setting counts the initial evaluation; ``1`` disables retries.
``auto_adjust: false`` evaluates the supplied settings without optimizing them.
Solver ``max_iterations`` is a separate limit. Reports and the workspace display
**Needs review** if quality criteria are unmet even when computation completes.
Quality scores are heuristic diagnostics, not calibrated confidence levels.

Reciprocal ERT measurements are optional. When pairing is missing or incomplete,
the default export uses the full loaded dataset with source or estimated errors.
Missing reciprocal errors do not fail quality control; measured reciprocal errors
above the configured threshold can still be filtered.

Every successful automatic computation writes ``detailed_report.md`` and
``processing_audit.json``, along with the original engine reports and numerical
outputs. The detailed report includes input names/paths and SHA-256 hashes,
configuration, AI settings, recorded stages and elapsed times, observed Python
function calls, software versions/project references, RAG sources and warnings.
Reviewed file classifications are saved separately. Directory inputs are
identified as directories; individual consumed files require the backend log.

Runtime tracing covers Python calls in the worker, not native internals or child
processes, and is capped at 2000 distinct function names with a truncation flag.
Package references are obtained from installed distribution metadata;
missing metadata is reported rather than invented. Original HTML/PDF engine
reports remain unchanged; the full audit is in the detailed Markdown/JSON outputs.

The uncertainty section distinguishes measured fit from confidence. It identifies
data noise and missing observations, geometry and first-break errors, mesh and
regularization choices, nonuniqueness, petrophysical assumptions and AI limitations.
Confidence intervals or ensemble validation are not claimed unless the executed
workflow actually computed them.
