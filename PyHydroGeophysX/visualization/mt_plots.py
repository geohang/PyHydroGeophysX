"""Figures for magnetotelluric data: soundings, models, phase tensors and sections.

Every length - depth, distance along a profile - goes through
:mod:`.axis_units`, so the figures follow the package's metres/feet choice and
read Depth below a flat surface.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import EllipseCollection
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import NullFormatter

from .axis_units import set_length_axis, set_section_axes, to_display_length

_COLORS = {"xy": "tab:red", "yx": "tab:blue", "det": "tab:green"}
_INDEX = {"xy": (0, 1), "yx": (1, 0), "xx": (0, 0), "yy": (1, 1)}


def _components(tf: Any, components: Sequence[str]):
    rho = tf.apparent_resistivity()
    phase = tf.phase()
    rho_err = tf.apparent_resistivity_err() if tf.z_err is not None else None
    phase_err = tf.phase_err() if tf.z_err is not None else None
    for name in components:
        if name == "det":
            z = tf.determinant()
            r = np.abs(z) ** 2 / (tf.angular_frequency * 4e-7 * np.pi)
            yield name, r, np.degrees(np.angle(z)), None, None
            continue
        i, j = _INDEX[name]
        p = phase[:, i, j]
        if name == "yx":
            p = p + 180.0
        p = np.where(p > 180, p - 360, p)
        yield (name, rho[:, i, j], p, None if rho_err is None else rho_err[:, i, j],
               None if phase_err is None else phase_err[:, i, j])


def plot_mt_sounding(tf: Any, *, components: Sequence[str] = ("xy", "yx"), axes: Any = None,
                     predicted: Optional[Dict[str, Any]] = None, title: Optional[str] = None,
                     errors: bool = True) -> Any:
    """Apparent resistivity and phase against period, Zyx's phase folded by 180 degrees.

    ``predicted`` maps a component to ``(period, rho_a, phase)`` lines drawn
    over the data - an inversion's fit. Returns the figure.
    """
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(6.5, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2]})
    else:
        fig = axes[0].figure
    ax_rho, ax_phase = axes
    period = tf.period
    for name, rho, phase, rho_err, phase_err in _components(tf, components):
        color = _COLORS.get(name, None)
        label = {"xy": "Zxy", "yx": "Zyx", "det": "det(Z)"}.get(name, name)
        if errors and rho_err is not None:
            low = np.clip(rho - rho_err, rho * 0.1, None)
            ax_rho.errorbar(period, rho, yerr=[rho - low, rho_err], fmt="o", ms=4, color=color,
                            label=label, capsize=2, lw=0.8)
            ax_phase.errorbar(period, phase, yerr=phase_err, fmt="o", ms=4, color=color, capsize=2, lw=0.8)
        else:
            ax_rho.plot(period, rho, "o", ms=4, color=color, label=label)
            ax_phase.plot(period, phase, "o", ms=4, color=color)
    for name, (p, rho, phase) in (predicted or {}).items():
        color = _COLORS.get(name, "k")
        ax_rho.plot(p, rho, "-", color=color, lw=1.6)
        ax_phase.plot(p, phase, "-", color=color, lw=1.6)
    # Both, since the axes a caller passes need not share x.
    ax_rho.set_xscale("log")
    ax_phase.set_xscale("log")
    ax_rho.set_yscale("log")
    ax_rho.set_ylabel("Apparent resistivity (Ω·m)")
    ax_rho.grid(True, which="both", alpha=0.25)
    ax_rho.legend(loc="best", fontsize=9)
    ax_phase.set_ylim(0, 90)
    ax_phase.set_yticks([0, 15, 30, 45, 60, 75, 90])
    ax_phase.set_ylabel("Phase (°)")
    ax_phase.set_xlabel("Period (s)")
    ax_phase.grid(True, which="both", alpha=0.25)
    ax_rho.set_title(title if title is not None else (getattr(tf, "station", "") or "MT sounding"))
    fig.tight_layout()
    return fig


def plot_mt_model_1d(models: Dict[str, Any], *, ax: Any = None, length_unit: Optional[str] = None,
                     max_depth: Optional[float] = None) -> Any:
    """Layered resistivity models against depth, one step line each.

    ``models`` maps a label to an :class:`~PyHydroGeophysX.data_processing.mt.Occam1DResult`
    or a ``(thickness, resistivity)`` pair. Returns the figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 6.5))
    else:
        fig = ax.figure
    deepest = 0.0
    for label, model in models.items():
        if hasattr(model, "depth_profile"):
            depth, rho = model.depth_profile()
        else:
            thickness, resistivity = (np.asarray(v, dtype=float) for v in model)
            tops = np.r_[0.0, np.cumsum(thickness)]
            bottoms = np.r_[tops[1:], tops[-1] + max(thickness[-1] if thickness.size else 1.0, 1.0)]
            depth, rho = np.column_stack([tops, bottoms]).ravel(), np.repeat(resistivity, 2)
        ax.plot(rho, depth, lw=1.8, label=label)
        deepest = max(deepest, float(depth[-2]) if depth.size > 2 else float(depth[-1]))
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Resistivity (Ω·m)")
    ax.set_ylim(max_depth or deepest * 1.05, 0.0)
    set_length_axis(ax, "y", "Depth", unit=length_unit)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


def plot_phase_tensor_pseudosection(tfs: Sequence[Any], positions: Sequence[float], *,
                                    ax: Any = None, color_by: str = "phi_min",
                                    length_unit: Optional[str] = None, scale: float = 0.4) -> Any:
    """Phase-tensor ellipses along a profile against log period.

    Each ellipse's axes are ``phi_max`` and ``phi_min`` normalized by
    ``phi_max``, turned to ``alpha - beta`` from north (up); colour is
    ``color_by`` (``"phi_min"``, ``"beta"`` skew, ...).
    """
    from PyHydroGeophysX.data_processing.mt.analysis import phase_tensor

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure
    x_display = np.asarray(to_display_length(np.asarray(positions, dtype=float), length_unit))
    spacing = np.min(np.diff(np.sort(x_display))) if len(x_display) > 1 else 1.0
    xs, ys, ratios, angles, colors = [], [], [], [], []
    for x, tf in zip(x_display, tfs):
        pt = phase_tensor(tf)
        for k, period in enumerate(tf.period):
            if not np.isfinite(pt["phi_max"][k]) or pt["phi_max"][k] <= 0:
                continue
            xs.append(x)
            ys.append(np.log10(period))
            ratios.append(max(pt["phi_min"][k], 0.0) / pt["phi_max"][k])
            angles.append(90.0 - pt["strike"][k])
            colors.append(pt[color_by][k])
    if not xs:
        raise ValueError("no finite phase tensors to plot")
    y_extent = max(np.ptp(ys), 1.0)
    x_extent = max(np.ptp(xs), spacing)
    aspect = y_extent / x_extent
    # An ellipse no wider than the gap to its neighbours, across and down.
    steps = np.diff(np.unique(np.round(ys, 6)))
    period_gap = float(np.median(steps)) / aspect if steps.size else spacing
    size = 2.0 * scale * min(spacing, period_gap)
    widths = np.full(len(xs), size)
    heights = size * np.asarray(ratios)
    collection = EllipseCollection(widths, np.asarray(heights), angles, units="x",
                                   offsets=np.c_[xs, np.asarray(ys) / aspect], transOffset=ax.transData,
                                   cmap="viridis" if color_by != "beta" else "RdBu_r",
                                   norm=Normalize(0, 90) if color_by.startswith("phi") else Normalize(-6, 6))
    collection.set_array(np.asarray(colors))
    ax.add_collection(collection)
    ax.set_xlim(min(xs) - spacing, max(xs) + spacing)
    low, high = min(ys) / aspect, max(ys) / aspect
    ax.set_ylim(high + 0.5 / aspect, low - 0.5 / aspect)
    ticks = np.arange(np.floor(min(ys)), np.ceil(max(ys)) + 1)
    ax.set_yticks(ticks / aspect)
    ax.set_yticklabels([f"$10^{{{int(t)}}}$" for t in ticks])
    ax.set_ylabel("Period (s)")
    set_length_axis(ax, "x", "Distance", unit=length_unit)
    ax.set_aspect("equal")
    colorbar = fig.colorbar(collection, ax=ax, shrink=0.8)
    colorbar.set_label({"phi_min": "Φmin (°)", "phi_max": "Φmax (°)", "beta": "Skew β (°)"}.get(color_by, color_by))
    fig.tight_layout()
    return fig


def plot_mt_dimensionality(tf: Any, *, axes: Any = None, title: Optional[str] = None) -> Any:
    """A site's phase tensor against period: Φmax and Φmin, the skew β and the strike.

    ``|β|`` under 3 degrees (shaded) reads as a 1D or 2D response; the strike
    is ambiguous by 90 degrees and meaningful only where Φmax and Φmin part.
    ``axes`` are three axes sharing x. Returns the figure.
    """
    from PyHydroGeophysX.data_processing.mt.analysis import phase_tensor

    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(6.5, 7), sharex=True)
    else:
        fig = axes[0].figure
    ax_phi, ax_beta, ax_strike = axes
    pt = phase_tensor(tf)
    period = tf.period
    ax_phi.plot(period, pt["phi_max"], "o-", ms=3, lw=0.8, label="Φmax")
    ax_phi.plot(period, pt["phi_min"], "s-", ms=3, lw=0.8, label="Φmin")
    ax_phi.set_ylabel("Phase (°)")
    ax_phi.set_ylim(0, 90)
    ax_phi.legend(fontsize=8, loc="best")
    ax_beta.axhspan(-3, 3, color="#cccccc", alpha=0.5, lw=0)
    ax_beta.plot(period, pt["beta"], "o", ms=3, color="#8c2d04")
    ax_beta.set_ylabel("Skew β (°)")
    ax_strike.plot(period, np.mod(pt["strike"], 180.0), "o", ms=3, color="#1f4e79")
    ax_strike.set_ylim(0, 180)
    ax_strike.set_yticks([0, 45, 90, 135, 180])
    ax_strike.set_ylabel("Strike (°)")
    ax_strike.set_xlabel("Period (s)")
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.25)
    ax_phi.set_title(title if title is not None else (getattr(tf, "station", "") or "Phase tensor"))
    fig.tight_layout()
    return fig


def plot_mt_section(result: Any, *, ax: Any = None, length_unit: Optional[str] = None,
                    max_depth: Optional[float] = None, limits: Optional[Sequence[float]] = None,
                    cmap: str = "jet_r", stations: bool = True, title: Optional[str] = None) -> Any:
    """A 2D resistivity section from :func:`~PyHydroGeophysX.data_processing.mt.invert_profile`."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4.5))
    else:
        fig = ax.figure
    x_nodes, z_nodes, grid = result.section()
    x0, x1 = result.profile.station_x.min(), result.profile.station_x.max()
    margin = 0.15 * max(x1 - x0, 1.0)
    vmin, vmax = limits if limits is not None else np.nanpercentile(grid, [2, 98])
    mesh = ax.pcolormesh(x_nodes, z_nodes, grid, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax), shading="flat")
    if stations:
        ax.plot(result.profile.station_x, np.zeros_like(result.profile.station_x), "kv", ms=6, clip_on=False)
    ax.set_xlim(x0 - margin, x1 + margin)
    depth = max_depth or 0.5 * max(x1 - x0, 1.0)
    ax.set_ylim(-depth, 0.0)
    set_section_axes(ax, z=np.zeros(2), vertical="auto", unit=length_unit)
    colorbar = fig.colorbar(mesh, ax=ax, shrink=0.85)
    colorbar.set_label("Resistivity (Ω·m)")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_mt_pseudosection(tfs: Sequence[Any], positions: Sequence[float], *, component: str = "xy",
                          quantity: str = "rho", ax: Any = None,
                          length_unit: Optional[str] = None) -> Any:
    """Apparent resistivity or phase of one component along a profile, against period."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    else:
        fig = ax.figure
    i, j = _INDEX[component]
    x = np.asarray(positions, dtype=float)
    period = tfs[0].period
    if quantity == "rho":
        values = np.column_stack([tf.apparent_resistivity()[:, i, j] for tf in tfs])
        norm, label, cmap = LogNorm(*np.nanpercentile(values, [2, 98])), "Apparent resistivity (Ω·m)", "jet_r"
    else:
        values = np.column_stack([tf.phase()[:, i, j] + (180.0 if component == "yx" else 0.0) for tf in tfs])
        norm, label, cmap = Normalize(0, 90), "Phase (°)", "viridis"
    order = np.argsort(x)
    image = ax.pcolormesh(np.asarray(to_display_length(x[order], length_unit)), period, values[:, order],
                          shading="nearest", cmap=cmap, norm=norm)
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_ylabel("Period (s)")
    ax.set_xlabel("")
    set_length_axis(ax, "x", "Distance", unit=length_unit)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
    colorbar.set_label(f"{label}, Z{component}")
    fig.tight_layout()
    return fig
