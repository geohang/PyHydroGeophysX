"""Visualization helpers for multi-method geophysical workflows."""

from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt


def _result_field(result, preferred):
    for key in preferred:
        if result is not None and hasattr(result, key):
            value = getattr(result, key)
            if value is not None:
                return np.asarray(value, dtype=float).ravel()
    return None


def _plot_on_mesh(ax, mesh, values, title, cmap="viridis"):
    try:
        import pygimli as pg

        if mesh is not None and hasattr(mesh, "cellCount") and mesh.cellCount() == values.size:
            pg.show(mesh, data=values, ax=ax, cMap=cmap)
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

        if not _plot_on_mesh(ax, mesh, values, f"{name} Model", cmap=cmap):
            ax.plot(values, lw=1.8)
            ax.set_title(f"{name} Model")
            ax.set_xlabel("Cell Index")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, axes


def plot_hydro_vs_geophys(hydro_wc, inverted_wc, mesh=None):
    """Compare hydrological water content to geophysics-derived water content."""
    hydro_wc = np.asarray(hydro_wc, dtype=float).ravel()
    inverted_wc = np.asarray(inverted_wc, dtype=float).ravel()

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

    fig, ax = plt.subplots(figsize=(9, 5))

    plotted = _plot_on_mesh(ax, mesh, values, "Cross-Section with Wells", cmap="viridis")
    if not plotted:
        ax.plot(values, lw=1.8, label="Model")
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
