"""Main-thread command layer the studio's assistants use to drive the studio.

The chat panel never touches Qt widgets directly. Instead it calls
:meth:`StudioController.dispatch`, which maps a small set of generic tool
names to operations on the live ``PyHydroGeophysXStudio``. Each call runs on
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
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject

from PyHydroGeophysX.qt_apps.agent import capture as capture_mod

#: One-line purpose per module so the agent can route a task to the right one.
MODULE_PURPOSES: Dict[str, str] = {
    "one_click": "Workflow workspace for data, progress and reports. To run end-to-end, the user selects Auto to report in the assistant panel and sends their goal there.",
    "home": "Landing page / overview.",
    "seismic": "Process seismic shot gathers, pick first breaks, and run SRT travel-time "
               "tomography to get a velocity model from field data; or stack the shots into "
               "a CMP reflection section and check whether any event is a reflection.",
    "boreholes": "Load wells, lithology logs, water levels and borehole geophysical logs "
                 "(LAS or tables); view them on a map, as logs side by side and as water "
                 "levels through time; add them to the Project Map to compare with surveys.",
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
    "mt": "Magnetotellurics: read instrument time series (Phoenix MTU-5C and legacy MTU, "
          "Metronix ATS, Zonge Z3D, LEMI-424) or EDI / EMTF XML sites, estimate impedance and "
          "tipper with robust remote-reference processing, then invert: Occam 1D per site "
          "(static shift, optional joint TEM, water content) and a 2D TE/TM profile. Has an "
          "example site (use_example_data).",
    "hydro_geophysics": "Generate SYNTHETIC geophysical data by 2D-profile FORWARD MODELING (ERT, "
                        "SRT, TDEM, FDEM, gravity) from a hydrologic model along a 2D line / "
                        "cross-section. Use this for forward modeling along a profile. Before loading "
                        "data, ask whether to use bundled example data or the user's hydro data. "
                        "(For 3D ERT forward, use mesh3d instead.)",
    "geo_hydrology": "Estimate water content / porosity (Monte Carlo) from an inverted ERT model. "
                     "Has built-in example data.",
    "seismic3d": "Build a 3D velocity model from several 2D seismic velocity lines. Has example data.",
}


class StudioController(QObject):
    """A thin, JSON-friendly facade over the studio main window."""

    #: Tool names handled directly by :meth:`dispatch`.
    GENERIC_TOOLS = (
        "list_modules",
        "navigate",
        "describe_current_module",
        "apply_action",
        "get_studio_state",
        "capture_view",
    )

    def __init__(self, window: Any) -> None:
        super().__init__(window if isinstance(window, QObject) else None)
        self._window = window
        self._chat_references = {}
        self._pending_chat_files = {}

    def queue_chat_files(self, paths):
        paths = list(dict.fromkeys(str(Path(p).resolve()) for p in paths))
        if not paths or any(not Path(p).exists() for p in paths):
            return {'status': 'failed', 'error': 'Paste existing local files or folders.'}
        pending = self._pending_chat_files.setdefault(self.attachment_scope(), [])
        known = {row['path'] for row in self.chat_attachments()}
        added = [path for path in paths if path not in known]
        if len(pending) + len(added) > 100:
            return {'status': 'failed', 'error': 'Paste at most 100 files at a time.'}
        pending.extend(added)
        return {'status': 'ok', 'added': added}

    def pending_chat_files(self):
        return list(self._pending_chat_files.get(self.attachment_scope(), []))

    def apply_chat_classification(self, rows):
        """Validate a whole classification before changing workflow inputs."""
        expected = self.pending_chat_files()
        if len(rows) != len(expected) or [row['path'] for row in rows] != expected:
            return {'status': 'failed', 'error': 'File selection changed; send the request again.'}
        allowed = {key for label, key in self.chat_attachment_roles()}
        unknown = [row for row in rows if row['role'] == 'unknown' or row['role'] not in allowed | {'ignore'}]
        if unknown:
            return {'status': 'needs_input', 'error': 'Tell me what these files contain or how you want to use them: ',
                    'files': unknown}
        groups = {}
        for row in rows:
            if row['role'] != 'ignore':
                role = row['role']
                path = Path(row['path'])
                value = str(path.parent) if role.endswith('_dir') and path.is_file() else str(path)
                groups.setdefault(role, []).append(value)
        from PyHydroGeophysX.agents.assistants import active
        groups = {role: list(dict.fromkeys(paths)) for role, paths in groups.items()}
        for role, paths in groups.items():
            if role != 'chat_reference' and role not in active().ordered_roles and len(paths) > 1:
                return {'status': 'needs_input', 'error': f'More than one file was identified as {role}; tell me which to use.'}
        if 'data_file' in groups and 'time_lapse_files' in groups:
            return {'status': 'needs_input', 'error': 'Tell me whether to run a single ERT survey or the time-lapse series.'}
        page = self._attachment_page(create=any(role != 'chat_reference' for role in groups))
        before = dict(getattr(page, '_inputs', {}))
        refs = list(self._chat_references.get(self.attachment_scope(), []))
        for role, paths in groups.items():
            result = self.add_chat_attachments(role, paths)
            if result.get('status') != 'ok':
                if page is not None:
                    page._inputs = before
                    page._refresh_inputs()
                self._chat_references[self.attachment_scope()] = refs
                return result
        self._pending_chat_files[self.attachment_scope()] = []
        return {'status': 'ok'}

    def attachment_scope(self):
        from PyHydroGeophysX.agents.assistants import active
        state = getattr(self._window, 'state', None)
        root = getattr(state, 'results_store_root', None) or getattr(state, 'output_dir', None)
        return (active().key, str(root or ''))

    def chat_attachment_roles(self):
        from PyHydroGeophysX.agents.assistants import active
        return [('Reference documents (RAG)', 'chat_reference')] + [
            (label, role) for label, role in active().input_roles if role != 'reference_file']

    def _attachment_page(self, create=False):
        page = getattr(self._window, '_pages', {}).get('one_click')
        if page is None and create and self._window is not None:
            self._window.show_module('one_click')
            page = self._window._pages.get('one_click')
        return page

    def chat_attachments(self):
        refs = self._chat_references.get(self.attachment_scope(), [])
        rows = []
        page = self._attachment_page()
        for role, value in getattr(page, '_inputs', {}).items():
            for path in value if isinstance(value, list) else [value]:
                rows.append({'role': role, 'path': str(path),
                             'rag': role == 'reference_file' or str(path) in refs})
        represented = {row['path'] for row in rows}
        rows.extend({'role': 'chat_reference', 'path': path, 'rag': True}
                    for path in refs if path not in represented)
        rows.extend({'role': 'pending', 'path': path, 'rag': False}
                    for path in self.pending_chat_files())
        return rows

    def add_chat_attachments(self, role, paths, use_rag=False):
        """Register local paths; never copy files or start a workflow."""
        from PyHydroGeophysX.agents.local_knowledge import REFERENCE_SUFFIXES, MAX_DOCUMENT_BYTES
        roles = dict((key, label) for label, key in self.chat_attachment_roles())
        if role not in roles:
            return {'status': 'failed', 'error': 'Choose a supported input role.'}
        paths = list(dict.fromkeys(str(Path(p).resolve()) for p in paths))
        if not paths:
            return {'status': 'failed', 'error': 'Choose at least one file.'}
        use_rag = use_rag or role == 'chat_reference'
        for filename in paths:
            path = Path(filename)
            if not path.exists() or (not role.endswith('_dir') and not path.is_file()):
                return {'status': 'failed', 'error': f'File not found: {filename}'}
            if use_rag and (path.suffix.lower() not in REFERENCE_SUFFIXES
                            or path.stat().st_size > MAX_DOCUMENT_BYTES):
                return {'status': 'failed', 'error':
                        'RAG accepts PDF, DOCX, TXT, Markdown, RST, CSV and JSON files up to 10 MB each.'}
        if role != 'chat_reference':
            page = self._attachment_page(create=True)
            if page is None:
                return {'status': 'failed', 'error': 'Workflow page is unavailable.'}
            result = page._agent_add_input(role, paths)
            if result.get('status') != 'ok':
                return result
        if use_rag:
            refs = self._chat_references.setdefault(self.attachment_scope(), [])
            refs.extend(path for path in paths if path not in refs)
        return {'status': 'ok', 'attachments': self.chat_attachments()}

    def remove_chat_attachment(self, role, path):
        if role == 'pending':
            pending = self._pending_chat_files.get(self.attachment_scope(), [])
            if path in pending:
                pending.remove(path)
            return {'status': 'ok'}
        page = self._attachment_page()
        if role != 'chat_reference' and page is not None:
            result = page._agent_remove_input(role, path)
            if result.get('status') != 'ok':
                return result
        refs = self._chat_references.get(self.attachment_scope(), [])
        if path in refs:
            refs.remove(path)
        return {'status': 'ok'}

    # -- public API ----------------------------------------------------------
    def reset_workflow_request(self):
        page = getattr(self._window, '_pages', {}).get('one_click')
        if page is not None and hasattr(page, 'reset_request'):
            page.reset_request()

    def run_to_report(self, request, settings, on_finished, on_step=None):
        self._window.show_module("one_click")
        page = self._window._pages["one_click"]
        if not hasattr(page, "submit_request"):
            return "Workflow page could not be loaded. See the Studio log."
        if not getattr(page, "_chat_connected", False):
            page.workflowFinished.connect(on_finished)
            # Each step as it starts and ends, so the chat can narrate the run.
            if on_step is not None and hasattr(page, "stepEvent"):
                page.stepEvent.connect(on_step)
            page._chat_connected = True
        return page.submit_request(request, settings)

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
            if name == "get_studio_state":
                return self._get_state()
            if name == "capture_view":
                return self._capture_view(args)
            return {"status": "failed", "error": f"Unknown tool '{name}'."}
        except Exception as exc:  # noqa: BLE001 - tools must never crash the agent
            return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    def set_assistant(self, key: str) -> None:
        """Make ``key`` the assistant the studio works with (chat, Workflow, glow)."""
        setter = getattr(self._window, "set_assistant", None)
        if callable(setter):
            setter(key)
        else:
            from PyHydroGeophysX.agents.assistants import set_active

            set_active(key)

    def capabilities_summary(self, modules=()) -> str:
        """A short human-readable list of modules + purposes for the system prompt.

        ``modules`` limits it to the keys an assistant works with; empty lists
        every module.
        """
        lines = []
        wanted = set(modules or ())
        for m in self._module_catalog():
            if wanted and m["key"] not in wanted:
                continue
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
