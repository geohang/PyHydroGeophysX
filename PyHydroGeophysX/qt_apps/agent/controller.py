"""Main-thread command layer the AQUAH assistant uses to drive the workbench.

The chat panel never touches Qt widgets directly. Instead it calls
:meth:`WorkbenchController.dispatch`, which maps a small set of generic tool
names to operations on the live ``PyHydroGeophysXWorkbench``. Each call runs on
the Qt main thread (the panel invokes it from a button slot) and returns a plain,
JSON-serialisable ``dict`` so the result can be handed straight back to the
language model. ``dispatch`` never raises: any error becomes
``{"status": "failed", "error": ...}`` so the agent can read it and recover.

One result field breaks the plain-JSON rule on purpose: ``capture_view`` returns
the captured PNG under ``_image``. The leading underscore marks it as
panel-only, matching ``_anthropic_content`` in the provider layer; the chat panel
lifts it out into an image message and never lets it reach the text transcript.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject

from PyHydroGeophysX.qt_apps.agent import capture as capture_mod

#: One-line purpose per module so the agent can route a task to the right one.
MODULE_PURPOSES: Dict[str, str] = {
    "home": "Landing page / overview.",
    "seismic": "Process seismic shot gathers, pick first breaks, and run SRT travel-time "
               "tomography to get a velocity model from field data.",
    "ert": "Load FIELD ERT resistivity data and run ERT inversion (single-time or time-lapse). "
           "This INVERTS measured data; it does not forward-model synthetic data.",
    "mesh3d": "Build a 3D finite-element mesh with sensor/electrode geometry, then run 3D ERT "
              "FORWARD modeling on it to generate synthetic 3D ERT data. Use this for '3D ERT "
              "forward modeling': first 'generate' the mesh, then 'run_ert_forward'.",
    "em": "Forward-model and 1D-invert EM soundings (FDEM/TDEM). Has bundled TDEM and "
          "synthetic FDEM examples (use_example_data).",
    "joint_inversion": "Jointly invert ERT+SRT or FDEM+TDEM observations, or run the "
                       "sequential SRT-structure-constrained ERT workflow.",
    "gravmag": "Process gravity / magnetic station data: regional-residual separation, gridding, "
               "profiles, and simple body forward modeling. Has bundled examples (use_example_data).",
    "hydro_geophysics": "Generate SYNTHETIC geophysical data by 2D-profile FORWARD MODELING (ERT, "
                        "SRT, TDEM, FDEM, gravity) from a hydrologic model along a 2D line / "
                        "cross-section. Use this for forward modeling along a profile. Before loading "
                        "data, ask whether to use bundled example data or the user's hydro data. "
                        "(For 3D ERT forward, use mesh3d instead.)",
    "geo_hydrology": "Estimate water content / porosity (Monte Carlo) from an inverted ERT model. "
                     "Has built-in example data.",
    "seismic3d": "Build a 3D velocity model from several 2D seismic velocity lines. Has example data.",
}


class WorkbenchController(QObject):
    """A thin, JSON-friendly facade over the workbench main window."""

    #: Tool names handled directly by :meth:`dispatch`.
    GENERIC_TOOLS = (
        "list_modules",
        "navigate",
        "describe_current_module",
        "apply_action",
        "get_workbench_state",
        "capture_view",
    )

    def __init__(self, window: Any) -> None:
        super().__init__(window if isinstance(window, QObject) else None)
        self._window = window

    # -- public API ----------------------------------------------------------
    def dispatch(self, name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute one tool call by name and return a JSON-friendly result."""
        args = args or {}
        try:
            if name == "list_modules":
                return self._list_modules()
            if name == "navigate":
                return self._navigate(args)
            if name == "describe_current_module":
                return self._describe_current()
            if name == "apply_action":
                return self._apply_action(args)
            if name == "get_workbench_state":
                return self._get_state()
            if name == "capture_view":
                return self._capture_view(args)
            return {"status": "failed", "error": f"Unknown tool '{name}'."}
        except Exception as exc:  # noqa: BLE001 - tools must never crash the agent
            return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    def capabilities_summary(self) -> str:
        """A short human-readable list of modules + purposes for the system prompt."""
        lines = []
        for m in self._module_catalog():
            purpose = MODULE_PURPOSES.get(m["key"], "")
            lines.append(f"- {m['key']} ({m['title']}): {purpose}" if purpose
                         else f"- {m['key']} ({m['title']})")
        return "\n".join(lines)

    # -- helpers -------------------------------------------------------------
    def _module_catalog(self) -> List[Dict[str, str]]:
        from PyHydroGeophysX.qt_apps.modules import MODULE_ORDER, MODULE_SPECS

        catalog: List[Dict[str, str]] = []
        for key in MODULE_ORDER:
            if key == "home":
                catalog.append({"key": "home", "title": "Home"})
            elif key in MODULE_SPECS:
                catalog.append({"key": key, "title": MODULE_SPECS[key][2]})
        return catalog

    def _list_modules(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "modules": self._module_catalog(),
            "current": getattr(self._window.state, "selected_module", None),
        }

    def _navigate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from PyHydroGeophysX.qt_apps.modules import MODULE_ORDER

        key = str(args.get("module", "")).strip()
        if key not in set(MODULE_ORDER):
            return {
                "status": "failed",
                "error": f"Unknown module '{key}'.",
                "valid_modules": list(MODULE_ORDER),
            }
        self._window.show_module(key)
        return {"status": "ok", "navigated_to": key, "current_module": self._safe_describe()}

    def _describe_current(self) -> Dict[str, Any]:
        desc = self._safe_describe()
        if desc is None:
            return {"status": "failed", "error": "No active module."}
        return {"status": "ok", "describe": desc}

    def _safe_describe(self) -> Optional[Dict[str, Any]]:
        page = self._window.current_module()
        if page is None:
            return None
        try:
            desc = page.agent_describe()
        except Exception as exc:  # noqa: BLE001
            return {"module": getattr(page, "module_key", "?"),
                    "error": f"agent_describe failed: {exc}"}
        # Views are merged here rather than in each module's agent_describe:
        # every page overrides that method wholesale, so a base-class addition
        # would reach none of them.
        if isinstance(desc, dict):
            desc.setdefault("views", self._view_names(page))
        return desc

    @staticmethod
    def _view_names(page: Any) -> List[str]:
        try:
            return capture_mod.view_names(page)
        except Exception:  # noqa: BLE001 - describing a module must not fail on this
            return [capture_mod.PAGE_VIEW]

    def _capture_view(self, args: Dict[str, Any]) -> Dict[str, Any]:
        page = self._window.current_module()
        if page is None:
            return {"status": "failed", "error": "No active module."}
        view = str(args.get("view", "")).strip() or None
        try:
            name, png = capture_mod.capture(page, view)
        except LookupError:
            return {"status": "failed",
                    "error": f"Unknown view '{view}'.",
                    "views": self._view_names(page)}
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        result = {
            "status": "ok",
            "view": name,
            "module": getattr(page, "module_key", None),
            "_image": {"media_type": "image/png",
                       "data": base64.b64encode(png).decode("ascii")},
        }
        # Exact values for what the picture only shows approximately. Reading an
        # index off a crowded axis is the model's weakest step; the module knows
        # it precisely, so send both rather than making vision do arithmetic.
        context = self._view_context(page, name)
        if context:
            result["context"] = context
        return result

    @staticmethod
    def _view_context(page: Any, view: str) -> Optional[Dict[str, Any]]:
        getter = getattr(page, "agent_view_context", None)
        if not callable(getter):
            return None
        try:
            context = getter(view)
        except Exception:  # noqa: BLE001 - context is a bonus, never a failure
            return None
        return context if isinstance(context, dict) and context else None

    def _apply_action(self, args: Dict[str, Any]) -> Dict[str, Any]:
        page = self._window.current_module()
        if page is None:
            return {"status": "failed", "error": "No active module."}
        action = str(args.get("action", "")).strip()
        if not action:
            return {"status": "failed", "error": "Missing 'action'."}
        sub = args.get("args", {})
        if not isinstance(sub, dict):
            sub = {}
        result = page.agent_apply(action, sub)
        if not isinstance(result, dict):
            result = {"status": "ok", "result": result}
        return result

    def _get_state(self) -> Dict[str, Any]:
        st = self._window.state
        try:
            context = st.context_summary()
        except Exception:  # noqa: BLE001
            context = {}
        return {
            "status": "ok",
            "selected_module": getattr(st, "selected_module", None),
            "context": context,
            "modules_with_results": sorted(getattr(st, "module_results", {}).keys()),
        }
