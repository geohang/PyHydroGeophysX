"""Gravity / magnetics agent: station data to QC maps and a 3D model.

The agents had no step for potential-field data, so a gravity or magnetic
survey given to a run was not processed at all. This agent does what the
Gravity / Magnetics page does with its default settings, without Qt: it reads
the station table, separates a polynomial regional trend from the residual
anomaly (:func:`~PyHydroGeophysX.data_processing.gravmag.qc_products`), maps
both, and inverts the residual for a 3D density-contrast or susceptibility
model with SimPEG
(:func:`~PyHydroGeophysX.inversion.gravmag.invert_gravmag`).

Every assumption the data do not settle - which field it is, the station
elevation, the inducing magnetic field - is recorded in ``assumptions`` so the
run can state it.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base_agent import AgentResult, BaseAgent

#: What a station file's header or name says about the field it holds.
_GRAVITY_WORDS = re.compile(r"mgal|grav|bouguer|free[ _-]?air|disturbance", re.IGNORECASE)
_MAGNETIC_WORDS = re.compile(r"(?<![a-z])nt(?![a-z])|magnet(?!otelluric)|tmi|aeromag",
                             re.IGNORECASE)

#: The inducing field the Gravity / Magnetics page starts from. Used only when
#: the run gives none, and then said to be an assumption.
DEFAULT_FIELD = {"inclination": 60.0, "declination": 0.0, "strength_nT": 50000.0}

#: The page's default inversion settings.
DEFAULT_SETTINGS = {
    "detrend": 1, "n_xy": 22, "n_z": 12, "max_iterations": 20, "max_stations": 600,
    "relative_error": 0.03, "solver": "linear", "auto_beta": True, "target_chi2": 1.0,
    "chi2_tolerance": 0.2, "max_beta_trials": 6, "sensitivity_power": 1.0,
}

#: Station elevation (m) when the table has no z column, as on the page.
DEFAULT_STATION_Z = 1.0


def field_kind(path: Any, header: Optional[List[str]] = None, text: str = "") -> Optional[str]:
    """``'gravity'``, ``'magnetics'`` or None, from a file's header, its name and ``text``.

    The header and the name are the data's own evidence; ``text`` - the
    request - only decides when they are silent.

    >>> field_kind('bushveld.csv', ['x_m', 'y_m', 'gravity_disturbance_mGal'])
    'gravity'
    >>> field_kind('survey.csv', ['x', 'y', 'total_field_anomaly_nT'])
    'magnetics'
    >>> field_kind('stations.csv', ['x', 'y', 'value'], 'invert the magnetic survey')
    'magnetics'
    >>> field_kind('stations.csv', ['x', 'y', 'value']) is None
    True
    """
    for evidence in (" ".join(header or []) + " " + Path(str(path)).stem, text):
        gravity = bool(_GRAVITY_WORDS.search(evidence))
        magnetic = bool(_MAGNETIC_WORDS.search(evidence)) or bool(
            re.search(r"磁法|磁测|航磁|磁异常|磁力", evidence))
        gravity = gravity or "重力" in evidence
        if gravity != magnetic:
            return "gravity" if gravity else "magnetics"
    return None


def resolve_kind(path: Any, header: Optional[List[str]], given: Any = None,
                 text: str = "") -> str:
    """The field a station file holds: the one named, else what :func:`field_kind` reads.

    >>> resolve_kind('s.csv', ['x', 'y', 'v'], 'Magnetic')
    'magnetics'
    >>> resolve_kind('bouguer.csv', ['x', 'y', 'v'])
    'gravity'

    Raises
    ------
    ValueError
        When neither the file nor the request says which field it is.
    """
    kind = str(given or "").strip().lower() or None
    if kind and kind.startswith("mag"):
        kind = "magnetics"
    elif kind and kind.startswith("grav"):
        kind = "gravity"
    kind = kind or field_kind(path, header, text)
    if kind is None:
        raise ValueError(f"{Path(str(path)).name} does not say whether it holds gravity (mGal) "
                         "or magnetic (nT) data, and neither does the request; say which.")
    return kind


class GravMagAgent(BaseAgent):
    """QC, regional-residual separation and 3D inversion of gravity or magnetic stations."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 llm_provider: str = "openai"):
        super().__init__("gravmag_agent", api_key, model, llm_provider)

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process one station file.

        Args:
            input_data: ``data_file`` (x, y, value[, z] table), ``output_dir``,
                and optionally ``kind`` ('gravity' or 'magnetics'), ``field``
                (inclination, declination, strength_nT), ``user_request``
                (read for the kind when the file does not say), ``invert``
                (default True), ``figure_style`` and any key of
                :data:`DEFAULT_SETTINGS`.

        Returns:
            The QC statistics, the inversion summary and the files written,
            with ``assumptions`` listing what the data did not settle.
        """
        self._log_execution("Starting gravity / magnetics processing")
        try:
            return self._run(input_data)
        except Exception as exc:  # noqa: BLE001 - the run reports the failure
            self._log_execution(f"Gravity / magnetics processing failed: {exc}", level='ERROR')
            return AgentResult(status="failed", summary="Gravity / magnetic data could not be "
                               "processed.", data={}, error=str(exc),
                               error_fix_hint="Give an x, y, value[, z] station table in "
                                              "projected metres, and say whether it is "
                                              "gravity (mGal) or magnetics (nT).")

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        from PyHydroGeophysX.data_processing import table_io
        from PyHydroGeophysX.data_processing.gravmag import qc_products, save_grid

        path = Path(str(input_data["data_file"]))
        if not path.is_file():
            raise FileNotFoundError(f"Gravity / magnetic station file not found: {path}")
        output_dir = Path(input_data.get("output_dir") or "results/gravmag")
        output_dir.mkdir(parents=True, exist_ok=True)
        table = table_io.load_xyz_table(path, min_cols=3)
        header = table_io.table_header(path) or []
        assumptions: List[str] = []

        kind = resolve_kind(path, header, input_data.get("kind"),
                            str(input_data.get("user_request") or ""))
        unit = "mGal" if kind == "gravity" else "nT"

        x, y, value = (np.asarray(table[:, i], dtype=float) for i in range(3))
        if table.shape[1] >= 4:
            z = np.asarray(table[:, 3], dtype=float)
            elevation = f"the table's fourth column{f' ({header[3]})' if len(header) > 3 else ''}"
        else:
            z = np.full(x.size, DEFAULT_STATION_Z)
            elevation = f"{DEFAULT_STATION_Z:g} m for every station (the table has no z column)"
            assumptions.append(f"The station file has no elevation column, so every station "
                               f"was placed {DEFAULT_STATION_Z:g} m above a flat datum; "
                               "terrain under the survey is not modelled.")
        settings = {**DEFAULT_SETTINGS,
                    **{key: input_data[key] for key in DEFAULT_SETTINGS if key in input_data}}
        settings.setdefault("noise_floor", input_data.get("noise_floor"))
        if settings.get("noise_floor") is None:
            settings["noise_floor"] = 0.5 if kind == "gravity" else 2.0

        field = None
        if kind == "magnetics":
            given = dict(input_data.get("field") or {})
            field = {**DEFAULT_FIELD, **{k: float(v) for k, v in given.items()
                                         if k in DEFAULT_FIELD and v is not None}}
            missing = [k for k in DEFAULT_FIELD if k not in given]
            if missing:
                assumptions.append(
                    "The inducing magnetic field was not given, so the inversion used "
                    + ", ".join(f"{k.replace('_nT', '')} {field[k]:g}"
                                + (" nT" if k == "strength_nT" else " deg") for k in missing)
                    + ". Give the site's IGRF inclination, declination and strength for the "
                      "survey date; the susceptibility model depends on them.")

        self._log_execution(f"{path.name}: {x.size} {kind} stations ({unit})")
        qc = qc_products(x, y, value, detrend=int(settings["detrend"]))
        data_paths: List[str] = []
        for name, grid in qc["grids"].items():
            data_paths += save_grid(grid, output_dir, name=name.lower(),
                                    log=lambda m: self._log_execution(str(m)))
        style = self._style(input_data)
        figures = [self._qc_figure(qc, kind, unit, output_dir, style)]

        inversion = None
        inversion_error = None
        if input_data.get("invert", True):
            try:
                inversion = self._invert(x, y, value, z, kind, field, settings, output_dir)
            except Exception as exc:  # noqa: BLE001 - the QC products stand on their own
                inversion_error = str(exc)
                self._log_execution(f"3D inversion did not run: {exc}", level='WARNING')
        if inversion is not None:
            data_paths += [p for p in (inversion.get("model_npz"), inversion.get("vtk")) if p]
            figures.append(self._model_figure(inversion, kind, output_dir, style))

        self.results = {
            "status": "success",
            "kind": kind,
            "unit": unit,
            "source_file": str(path),
            "columns": header[:4],
            "n_stations": int(qc["x"].size),
            "extent": {"x": [float(np.min(qc["x"])), float(np.max(qc["x"]))],
                       "y": [float(np.min(qc["y"])), float(np.max(qc["y"]))]},
            "elevation_source": elevation,
            "detrend": int(qc["detrend"]),
            "stats": qc["stats"],
            "field": field,
            "settings": settings,
            "inversion": inversion,
            "inversion_error": inversion_error,
            "assumptions": assumptions,
            "figures": [f for f in figures if f],
            "visualization_file": figures[-1],
            "data_paths": data_paths,
            "interpretation": None,
            "output_dir": str(output_dir),
        }
        return self.results

    @staticmethod
    def _style(input_data: Dict[str, Any]):
        from . import _figstyle as figstyle

        return figstyle.style_from_config(input_data)

    def _invert(self, x, y, value, z, kind, field, settings, output_dir) -> Dict[str, Any]:
        from PyHydroGeophysX.inversion.gravmag import invert_gravmag

        options = {key: settings[key] for key in (
            "detrend", "n_xy", "n_z", "max_iterations", "max_stations", "relative_error",
            "noise_floor", "solver", "auto_beta", "target_chi2", "chi2_tolerance",
            "max_beta_trials", "sensitivity_power")}
        result = invert_gravmag(x, y, value, kind, z=z, field=field, out_dir=str(output_dir),
                                random_seed=42, log=lambda m: self._log_execution(str(m)),
                                **options)
        ex, ey, ez = (np.asarray(edge, dtype=float) for edge in result["edges"])
        model = np.asarray(result["model3d"], dtype=float)
        top = float(ez[-1])
        centre_depth = top - 0.5 * (ez[:-1] + ez[1:])
        summary = {}
        for name, index in (("strongest_positive", np.nanargmax(model)),
                            ("strongest_negative", np.nanargmin(model))):
            i, j, k = np.unravel_index(index, model.shape)
            summary[name] = {"value": float(model[i, j, k]),
                             "x": float(0.5 * (ex[i] + ex[i + 1])),
                             "y": float(0.5 * (ey[j] + ey[j + 1])),
                             "depth": float(centre_depth[k])}
        bound = 1.0 if kind == "gravity" else 0.5
        out_dir = Path(result.get("output_dir") or output_dir)
        return {
            "chi2": float(result["chi2"]), "n_data": int(result["n_data"]),
            "n_input": int(result["n_input"]), "n_cells": int(result["n_cells"]),
            "shape": list(model.shape),
            "cell_size": [float(np.diff(ex)[0]), float(np.diff(ey)[0]), float(np.diff(ez)[0])],
            "depth_extent": float(ez[-1] - ez[0]),
            "label": result["label"], "model_range": result["model_range"],
            "at_bound": bool(np.nanmax(np.abs(model)) >= bound * (1 - 1e-6)),
            "bound": bound,
            "relative_error": result["relative_error"], "noise_floor": result["noise_floor"],
            "beta": {k: v for k, v in (result.get("beta") or {}).items()
                     if k in ("solver", "beta", "status", "reason", "sensitivity_weighted")},
            # The stations inverted, their residual anomaly and the model's
            # prediction, and the misfit at each beta tried, for the evaluation.
            "fit": dict(result.get("fit") or {}),
            "convergence": [float(v) for v in result.get("convergence") or []],
            "target_chi2": float(settings.get("target_chi2", 1.0)),
            "model_summary": summary,
            "edges": (ex, ey, ez), "model3d": model,
            "model_npz": str(out_dir / "model_grid.npz") if (out_dir / "model_grid.npz").exists()
            else None,
            "vtk": result.get("vtk"),
        }

    def _qc_figure(self, qc, kind, unit, output_dir, style) -> str:
        """Observed, regional and residual maps, with the stations."""
        from . import _figstyle as figstyle
        from PyHydroGeophysX.visualization.axis_units import set_length_axis

        names = ["Observed", "Regional", "Residual"] if qc["detrend"] else ["Observed"]
        fig = figstyle.detached_figure((min(4.6 * len(names), figstyle.MAX_FIGURE_WIDTH_IN), 4.4))
        fig.set_layout_engine("constrained")
        axes = np.atleast_1d(fig.subplots(1, len(names)))
        for ax, name in zip(axes, names):
            grid = qc["grids"][name]
            image = ax.pcolormesh(grid["xx"], grid["yy"], grid["zz"], shading="auto",
                                  cmap="RdBu_r" if name == "Residual" else "viridis")
            ax.plot(qc["x"], qc["y"], ".", color="k", markersize=1.0, alpha=0.4)
            ax.set_aspect("equal")
            ax.set_title(f"{name} {'gravity' if kind == 'gravity' else 'magnetic'} field",
                         fontsize=style.title_size)
            set_length_axis(ax, "x", "Easting", unit=style.length_unit, fontsize=style.label_size)
            set_length_axis(ax, "y", "Northing", unit=style.length_unit,
                            labelled=ax is axes[0], fontsize=style.label_size)
            fig.colorbar(image, ax=ax, label=unit, shrink=0.8)
        path = figstyle.save(fig, os.path.join(output_dir, "gravmag_qc.png"), style)
        self._log_execution(f"Saved visualization to {path}")
        return path

    def _model_figure(self, inversion, kind, output_dir, style) -> str:
        """Three depth slices of the model and a section through its strongest anomaly."""
        import matplotlib.colors as mcolors

        from . import _figstyle as figstyle
        from ._report_text import length
        from PyHydroGeophysX.visualization.axis_units import normalize_length_unit, set_length_axis

        unit = normalize_length_unit(style.length_unit)
        ex, ey, ez = inversion["edges"]
        model = inversion["model3d"]
        top = float(ez[-1])
        limit = float(np.nanmax(np.abs(model))) or 1.0
        norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        cmap = "RdBu_r" if kind == "gravity" else "PuOr_r"
        layers = model.shape[2]
        picks = sorted({layers - 1, int(round(layers * 0.6)), int(round(layers * 0.25))},
                       reverse=True)
        fig = figstyle.detached_figure((min(4.4 * (len(picks) + 1), figstyle.MAX_FIGURE_WIDTH_IN),
                                        4.4))
        fig.set_layout_engine("constrained")
        axes = np.atleast_1d(fig.subplots(1, len(picks) + 1))
        image = None
        for ax, k in zip(axes, picks):
            image = ax.pcolormesh(ex, ey, model[:, :, k].T, cmap=cmap, norm=norm)
            depth = top - 0.5 * (ez[k] + ez[k + 1])
            ax.set_aspect("equal")
            ax.set_title(f"Depth {length(depth, unit)}", fontsize=style.title_size)
            set_length_axis(ax, "x", "Easting", unit=style.length_unit, fontsize=style.label_size)
            set_length_axis(ax, "y", "Northing", unit=style.length_unit,
                            labelled=ax is axes[0], fontsize=style.label_size)
        strongest = inversion["model_summary"]["strongest_positive"]
        if abs(inversion["model_summary"]["strongest_negative"]["value"]) > abs(strongest["value"]):
            strongest = inversion["model_summary"]["strongest_negative"]
        j = int(np.clip(np.searchsorted(ey, strongest["y"]) - 1, 0, model.shape[1] - 1))
        ax = axes[-1]
        ax.pcolormesh(ex, ez, model[:, j, :].T, cmap=cmap, norm=norm)
        ax.set_title(f"Section at northing {length(strongest['y'], unit)}",
                     fontsize=style.title_size)
        set_length_axis(ax, "x", "Easting", unit=style.length_unit, fontsize=style.label_size)
        set_length_axis(ax, "y", "Depth", unit=style.length_unit, depth_reference=top,
                        fontsize=style.label_size)
        fig.colorbar(image, ax=axes.tolist(), label=inversion["label"], shrink=0.8)
        path = figstyle.save(fig, os.path.join(output_dir, "gravmag_model.png"), style)
        self._log_execution(f"Saved visualization to {path}")
        return path
