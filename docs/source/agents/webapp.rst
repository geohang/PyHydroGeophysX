Agent Web App
=============

The PyHydroGeophysX web app provides a guided, natural-language interface for running
agent-assisted geophysical workflows without local UI setup.

Open the App
------------

- Live URL: `https://pyhydrogeophysx.streamlit.app/ <https://pyhydrogeophysx.streamlit.app/>`_
- Local implementation script: ``examples/app_geophysics_workflow.py``

What It Does
------------

- Collects workflow goals and optional context in natural language.
- Routes tasks to agent pipelines for ERT, SRT, TDEM, and conversion workflows.
- Produces intermediate artifacts and workflow reports where available.

Required Inputs
---------------

- A clear task prompt (for example, "run a time-lapse ERT inversion").
- Paths or uploaded references to required survey/model files.
- Optional provider credentials when you want LLM-backed interpretation.

Limitations and Notes
---------------------

- Uploaded data quality and schema consistency still control result quality.
- Some workflows require optional dependencies (for example `pygimli` or SimPEG).
- LLM-generated recommendations should be reviewed before production decisions.
