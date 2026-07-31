"""AQUAH in-app assistant for the PyHydroGeophysX Qt workbench.

This subpackage wires a natural-language chat panel (the right-hand dock) to the
workbench through a small, stable command layer:

- :class:`controller.WorkbenchController` exposes a handful of generic,
  JSON-friendly operations (list modules, navigate, describe the current module,
  apply an action, read state) over the live main window. All of these run on
  the Qt main thread.
- :mod:`tools` declares the provider-neutral tool specs.
- :mod:`providers` adapts those specs and the neutral conversation to each
  provider (OpenAI, Claude/Anthropic, or an OpenAI-compatible endpoint), makes
  one tool-enabled call, and normalises the reply. Provider and model are
  user-selectable at run time.
- :mod:`runtime` holds the ``QThread`` that performs one blocking model call off
  the UI thread.
- :class:`chat_panel.AquahChatPanel` is the dock widget: it renders the
  conversation, asks the user to approve every proposed tool call (per-step
  confirmation), and executes approved calls through the controller.

Importing this package pulls in PySide6, so it must only be imported inside the
desktop process (never from ``PyHydroGeophysX.qt_apps.__init__``). The provider
SDKs are imported lazily, the first time a model call is actually made.
"""

__all__ = ["controller", "tools", "providers", "runtime", "chat_panel"]
