Agents
======

PyHydroGeophysX includes an optional multi-agent system for automating
geophysical workflows using LLMs. It can coordinate data loading, QC,
forward modeling, inversion, and report generation.

What It Does
------------

- Interprets natural-language requests into workflow steps
- Orchestrates ERT, SRT, and TDEM processing chains
- Produces reports, plots, and intermediate artifacts

Reproducibility
---------------

- Store configuration files and prompts alongside results.
- Pin versions of PyHydroGeophysX and dependencies for each run.
- Archive agent outputs (logs, summaries, plots) in the results folder.

Limitations
-----------

- LLM responses can vary between runs.
- External API availability and rate limits can affect execution time.
- Always validate key outputs before publication or decision-making.

More Details
------------

- :doc:`/agents/index`
- :doc:`/agents/webapp`
- Hosted app: `https://pyhydrogeophysx.streamlit.app/ <https://pyhydrogeophysx.streamlit.app/>`_
- See `examples/Ex_multi_agent_workflow.ipynb` for a full workflow notebook.
