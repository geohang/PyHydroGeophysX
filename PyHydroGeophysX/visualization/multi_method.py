"""Visualization helpers for multi-method geophysical workflows."""

from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _result_field(result, preferred):
    for key in preferred:
        if result is not None and hasattr(result, key):
            value = getattr(result, key)
            if value is not None:
                return np.asarray(value, dtype=float)
    return None


def _as_1d(values, name):
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    return arr


def _plot_on_mesh(ax, mesh, values, title, cmap="viridis"):
    try:
        import pygimli as pg

        arr = np.asarray(values, dtype=float).ravel()
        if mesh is not None and hasattr(mesh, "cellCount") and mesh.cellCount() == arr.size:
            pg.show(mesh, data=arr, ax=ax, cMap=cmap)
            ax.set_title(title)
            return True
    except Exception:
        pass
    return False


def plot_multi_method_panel(ert_result, srt_result, em_result, mesh=None):
    """Plot side-by-side ERT/SRT/EM model panels."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    panels = [
        ("ERT", _result_field(ert_result, ["final_model", "recovered_model"]), "plasma"),
        ("SRT", _result_field(srt_result, ["final_model"]), "viridis"),
        ("EM", _result_field(em_result, ["recovered_conductivity", "final_model"]), "cividis"),
    ]

    for ax, (name, values, cmap) in zip(axes, panels):
        if values is None:
            ax.text(0.5, 0.5, f"No {name} result", ha="center", va="center")
            ax.set_title(name)
            ax.axis("off")
            continue

        flat = np.asarray(values, dtype=float).ravel()
        if not _plot_on_mesh(ax, mesh, flat, f"{name} Model", cmap=cmap):
            ax.plot(flat, lw=1.8)
            ax.set_title(f"{name} Model")
            ax.set_xlabel("Cell Index")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, axes


def plot_hydro_vs_geophys(hydro_wc, inverted_wc, mesh=None):
    """Compare hydrological water content to geophysics-derived water content."""
    hydro_wc = _as_1d(hydro_wc, "hydro_wc")
    inverted_wc = _as_1d(inverted_wc, "inverted_wc")

    if hydro_wc.size != inverted_wc.size:
        x_src = np.linspace(0.0, 1.0, inverted_wc.size)
        x_tgt = np.linspace(0.0, 1.0, hydro_wc.size)
        inverted_wc = np.interp(x_tgt, x_src, inverted_wc)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if not _plot_on_mesh(axes[0], mesh, hydro_wc, "Hydrological Water Content", cmap="Blues"):
        axes[0].plot(hydro_wc, color="tab:blue", lw=1.8)
        axes[0].set_title("Hydrological Water Content")
        axes[0].set_xlabel("Cell Index")
        axes[0].set_ylabel("Water Content")
        axes[0].grid(True, alpha=0.3)

    if not _plot_on_mesh(axes[1], mesh, inverted_wc, "Geophysical-Derived Water Content", cmap="Oranges"):
        axes[1].plot(inverted_wc, color="tab:orange", lw=1.8)
        axes[1].set_title("Geophysical-Derived Water Content")
        axes[1].set_xlabel("Cell Index")
        axes[1].set_ylabel("Water Content")
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, axes


def plot_cross_section_with_wells(result, mesh, well_data: Optional[Dict[str, np.ndarray]] = None):
    """Plot a model cross-section and overlay optional well picks."""
    values = _result_field(result, ["final_model", "recovered_conductivity", "recovered_model"])
    if values is None:
        raise ValueError("result has no plottable model field.")

    flat = np.asarray(values, dtype=float).ravel()
    fig, ax = plt.subplots(figsize=(9, 5))

    plotted = _plot_on_mesh(ax, mesh, flat, "Cross-Section with Wells", cmap="viridis")
    if not plotted:
        ax.plot(flat, lw=1.8, label="Model")
        ax.set_xlabel("Cell Index")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

    if well_data is not None:
        x = np.asarray(well_data.get("x", []), dtype=float)
        z = np.asarray(well_data.get("z", []), dtype=float)
        labels = well_data.get("labels", None)

        if x.size and z.size and x.size == z.size:
            ax.scatter(x, z, c="k", s=30, marker="o", label="Wells")
            if labels is not None and len(labels) == x.size:
                for xi, zi, label in zip(x, z, labels):
                    ax.text(xi, zi, str(label), fontsize=8, ha="left", va="bottom")

    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")

    fig.tight_layout()
    return fig, ax


def plot_em_data_fit(
    times,
    observed,
    predicted,
    uncertainties=None,
    true_data=None,
    chi2: Optional[float] = None,
    time_scale: float = 1e3,
    time_label: str = "Time (ms)",
    data_label: str = "|Response|",
    ax=None,
):
    """Plot log-log EM data fit similar to TDEM workflow examples."""
    t = _as_1d(times, "times")
    dobs = _as_1d(observed, "observed")
    dpred = _as_1d(predicted, "predicted")
    if t.size != dobs.size or dobs.size != dpred.size:
        raise ValueError("times, observed, and predicted must have the same length.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        own_fig = True
    else:
        fig = ax.figure
        own_fig = False

    ax.loglog(t * time_scale, np.abs(dobs), "ko", markersize=6, label="Observed")
    ax.loglog(t * time_scale, np.abs(dpred), "r-", lw=2, label="Predicted")

    if true_data is not None:
        dtrue = _as_1d(true_data, "true_data")
        if dtrue.size != t.size:
            raise ValueError("true_data must have the same length as times.")
        ax.loglog(t * time_scale, np.abs(dtrue), "b--", lw=1.5, alpha=0.8, label="True")

    if uncertainties is not None:
        unc = _as_1d(uncertainties, "uncertainties")
        if unc.size != t.size:
            raise ValueError("uncertainties must have the same length as times.")
        lower = np.maximum(np.abs(dobs) - unc, np.finfo(float).tiny)
        upper = np.abs(dobs) + unc
        ax.fill_between(t * time_scale, lower, upper, alpha=0.2, color="gray", label="Uncertainty")

    title = "Data Fit"
    if chi2 is not None:
        title = f"Data Fit (chi2 = {chi2:.2f})"
    ax.set_title(title)
    ax.set_xlabel(time_label)
    ax.set_ylabel(data_label)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="best")

    if own_fig:
        fig.tight_layout()
    return fig, ax


def plot_em_residuals(
    times,
    observed,
    predicted,
    uncertainties,
    sigma_bound: float = 2.0,
    time_scale: float = 1e3,
    time_label: str = "Time (ms)",
    ax=None,
):
    """Plot normalized residuals with +/-sigma bounds."""
    t = _as_1d(times, "times")
    dobs = _as_1d(observed, "observed")
    dpred = _as_1d(predicted, "predicted")
    unc = _as_1d(uncertainties, "uncertainties")
    if not (t.size == dobs.size == dpred.size == unc.size):
        raise ValueError("times, observed, predicted, and uncertainties must have the same length.")

    if np.any(unc <= 0):
        raise ValueError("uncertainties must be positive.")

    residual = (dobs - dpred) / unc

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        own_fig = True
    else:
        fig = ax.figure
        own_fig = False

    ax.semilogx(t * time_scale, residual, "ro-", lw=1.4, markersize=4)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(sigma_bound, color="gray", ls="--", lw=0.8)
    ax.axhline(-sigma_bound, color="gray", ls="--", lw=0.8)
    ax.fill_between(t * time_scale, -sigma_bound, sigma_bound, alpha=0.12, color="green")
    ax.set_xlabel(time_label)
    ax.set_ylabel("Normalized Residual")
    ax.set_title("Data Residuals")
    ax.grid(True, alpha=0.4)

    if own_fig:
        fig.tight_layout()
    return fig, ax


def plot_em_fit_and_residuals(
    times,
    observed,
    predicted,
    uncertainties,
    true_data=None,
    chi2: Optional[float] = None,
    time_scale: float = 1e3,
):
    """Create side-by-side EM data fit and residual plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_em_data_fit(
        times=times,
        observed=observed,
        predicted=predicted,
        uncertainties=uncertainties,
        true_data=true_data,
        chi2=chi2,
        time_scale=time_scale,
        ax=axes[0],
    )
    plot_em_residuals(
        times=times,
        observed=observed,
        predicted=predicted,
        uncertainties=uncertainties,
        time_scale=time_scale,
        ax=axes[1],
    )
    fig.tight_layout()
    return fig, axes


def plot_time_lapse_panel(
    models: Sequence[Any],
    mesh=None,
    titles: Optional[Sequence[str]] = None,
    ncols: int = 4,
    cmap: str = "viridis",
):
    """Plot a grid of time-lapse model snapshots."""
    if not models:
        raise ValueError("models must contain at least one model.")
    if ncols <= 0:
        raise ValueError("ncols must be positive.")

    data_arrays = []
    for obj in models:
        if isinstance(obj, np.ndarray):
            arr = np.asarray(obj, dtype=float)
        else:
            arr = _result_field(obj, ["final_model", "recovered_model", "recovered_conductivity"])
            if arr is None:
                arr = np.asarray(obj, dtype=float)
        data_arrays.append(np.asarray(arr, dtype=float))

    n_models = len(data_arrays)
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    flattened = axes.ravel()
    for i, ax in enumerate(flattened):
        if i >= n_models:
            ax.axis("off")
            continue

        arr = data_arrays[i]
        title = titles[i] if titles is not None and i < len(titles) else f"Timestep {i + 1}"

        if arr.ndim == 2:
            im = ax.imshow(arr, cmap=cmap, aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlabel("X Index")
            ax.set_ylabel("Z Index")
            ax.set_title(title)
            continue

        flat = arr.ravel()
        if not _plot_on_mesh(ax, mesh, flat, title, cmap=cmap):
            ax.plot(flat, lw=1.5)
            ax.set_xlabel("Cell Index")
            ax.set_ylabel("Value")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, axes


def plot_petrophysical_scatter(
    x,
    y,
    color=None,
    xlabel: str = "Porosity (-)",
    ylabel: str = "Property",
    color_label: str = "Saturation (-)",
    cmap: str = "Blues",
    fit_line: bool = True,
    ax=None,
):
    """Plot petrophysical scatter diagnostics with optional trend line."""
    xv = _as_1d(x, "x")
    yv = _as_1d(y, "y")
    if xv.size != yv.size:
        raise ValueError("x and y must have the same length.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        own_fig = True
    else:
        fig = ax.figure
        own_fig = False

    if color is None:
        ax.scatter(xv, yv, s=24, c="tab:blue", alpha=0.8)
    else:
        cv = _as_1d(color, "color")
        if cv.size != xv.size:
            raise ValueError("color must have the same length as x and y.")
        scatter = ax.scatter(xv, yv, c=cv, s=24, cmap=cmap, alpha=0.9)
        fig.colorbar(scatter, ax=ax, label=color_label)

    if fit_line and xv.size >= 2:
        mask = np.isfinite(xv) & np.isfinite(yv)
        if np.count_nonzero(mask) >= 2:
            coeff = np.polyfit(xv[mask], yv[mask], 1)
            xline = np.linspace(np.nanmin(xv[mask]), np.nanmax(xv[mask]), 200)
            yline = coeff[0] * xline + coeff[1]
            ax.plot(xline, yline, "k--", lw=1.2, label="Linear fit")
            ax.legend(loc="best")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Petrophysical Relationship")
    ax.grid(True, alpha=0.3)

    if own_fig:
        fig.tight_layout()
    return fig, ax


def plot_layered_profiles(
    depth_edges,
    profiles: Dict[str, Sequence[float]],
    colors: Optional[Sequence[str]] = None,
    xscale: str = "linear",
):
    """Plot one or more layered profiles as step-like vertical columns."""
    z = _as_1d(depth_edges, "depth_edges")
    if z.size < 2:
        raise ValueError("depth_edges must contain at least two values.")

    labels = list(profiles.keys())
    if not labels:
        raise ValueError("profiles must contain at least one entry.")

    fig, axes = plt.subplots(1, len(labels), figsize=(4.0 * len(labels), 6), squeeze=False)
    axes = axes.ravel()

    for i, label in enumerate(labels):
        values = _as_1d(profiles[label], label)
        if values.size != z.size - 1:
            raise ValueError(f"profile '{label}' length must be len(depth_edges)-1.")
        ax = axes[i]
        color = colors[i] if colors is not None and i < len(colors) else None

        for j, v in enumerate(values):
            ax.fill_betweenx([z[j], z[j + 1]], np.finfo(float).tiny, v, alpha=0.35, color=color)
            ax.plot([v, v], [z[j], z[j + 1]], color=color if color else "k", lw=1.6)

        ax.set_xlabel(label)
        ax.set_ylabel("Depth (m)")
        ax.set_xscale(xscale)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    return fig, axes
