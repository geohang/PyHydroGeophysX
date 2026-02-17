"""Time-lapse animation utilities for geophysical models."""

import io
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def create_timelapse_gif(
    mesh,
    models: Sequence[np.ndarray],
    filename: str,
    *,
    titles: Optional[Sequence[str]] = None,
    cmap=None,
    cmin: Optional[float] = None,
    cmax: Optional[float] = None,
    log_scale: bool = False,
    label: str = "",
    xlabel: str = "Distance (m)",
    ylabel: str = "Elevation (m)",
    coverage: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
    figsize: Tuple[float, float] = (8, 2.5),
    dpi: int = 150,
    duration: int = 100,
    first_frame_duration: int = 500,
    loop: int = 0,
) -> str:
    """Create a GIF animation of time-lapse model snapshots.

    Each frame renders the model on the given mesh using ``pg.show`` and
    captures it as a PIL image.  Requires ``Pillow``.

    Parameters
    ----------
    mesh : pygimli.Mesh
        Mesh shared by all models.
    models : sequence of array-like
        Model values for each frame.
    filename : str
        Output GIF path.
    titles : sequence of str, optional
        Title per frame.
    cmap : str or Colormap, optional
        Colormap.  Defaults to BlueDarkRed18_18_r if available.
    cmin, cmax : float, optional
        Fixed color limits.
    log_scale : bool
        Logarithmic color scale.
    label : str
        Colorbar label.
    coverage : array or list of arrays, optional
        Coverage mask(s).
    figsize : tuple
        Figure size per frame.
    dpi : int
        Resolution.
    duration : int
        Milliseconds per frame.
    first_frame_duration : int
        Duration of the first frame in ms (longer for visual pause).
    loop : int
        Number of loops (0 = infinite).

    Returns
    -------
    str
        Path to the saved GIF file.
    """
    import pygimli as pg
    from PIL import Image

    from .plotting import _get_cmap

    cmap = _get_cmap(cmap)

    # Prepare coverage
    if coverage is not None:
        cov_arr = np.asarray(coverage)
        single_cov = cov_arr.ndim == 1
    else:
        cov_arr = None
        single_cov = False

    frames = []
    for i, model in enumerate(models):
        fig, ax = plt.subplots(figsize=figsize)
        arr = np.asarray(model, dtype=float).ravel()

        kw = dict(
            cMap=cmap, logScale=log_scale, label=label,
            xlabel=xlabel, ylabel=ylabel,
            orientation="vertical", pad=0.3,
        )
        if cmin is not None:
            kw["cMin"] = cmin
        if cmax is not None:
            kw["cMax"] = cmax
        if cov_arr is not None:
            c = cov_arr if single_cov else cov_arr[i]
            kw["coverage"] = np.asarray(c).ravel() > -1

        pg.show(mesh, arr, ax=ax, **kw)
        if titles is not None and i < len(titles):
            ax.set_title(titles[i])

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())

    durations = [first_frame_duration] + [duration] * (len(frames) - 1)
    frames[0].save(
        filename,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=durations,
        loop=loop,
        optimize=True,
    )
    return filename


def create_timelapse_mp4(
    mesh,
    models: Sequence[np.ndarray],
    filename: str,
    *,
    titles: Optional[Sequence[str]] = None,
    cmap=None,
    cmin: Optional[float] = None,
    cmax: Optional[float] = None,
    log_scale: bool = False,
    label: str = "",
    xlabel: str = "Distance (m)",
    ylabel: str = "Elevation (m)",
    coverage: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
    figsize: Tuple[float, float] = (8, 2.5),
    dpi: int = 150,
    fps: int = 10,
) -> str:
    """Create an MP4 video of time-lapse model snapshots using matplotlib.

    Requires ``ffmpeg`` to be available on the system path.

    Parameters
    ----------
    mesh : pygimli.Mesh
    models : sequence of array-like
    filename : str
        Output ``.mp4`` path.
    fps : int
        Frames per second.
    (other parameters same as :func:`create_timelapse_gif`)

    Returns
    -------
    str
        Path to the saved MP4 file.
    """
    import pygimli as pg
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    from .plotting import _get_cmap

    cmap = _get_cmap(cmap)

    if coverage is not None:
        cov_arr = np.asarray(coverage)
        single_cov = cov_arr.ndim == 1
    else:
        cov_arr = None
        single_cov = False

    fig, ax = plt.subplots(figsize=figsize)

    def _draw(i):
        ax.clear()
        arr = np.asarray(models[i], dtype=float).ravel()
        kw = dict(
            cMap=cmap, logScale=log_scale, label=label,
            xlabel=xlabel, ylabel=ylabel,
            orientation="vertical", pad=0.3,
        )
        if cmin is not None:
            kw["cMin"] = cmin
        if cmax is not None:
            kw["cMax"] = cmax
        if cov_arr is not None:
            c = cov_arr if single_cov else cov_arr[i]
            kw["coverage"] = np.asarray(c).ravel() > -1
        pg.show(mesh, arr, ax=ax, **kw)
        if titles is not None and i < len(titles):
            ax.set_title(titles[i])

    anim = FuncAnimation(fig, _draw, frames=len(models), interval=1000 // fps)
    writer = FFMpegWriter(fps=fps)
    anim.save(filename, writer=writer, dpi=dpi)
    plt.close(fig)
    return filename


def create_difference_gif(
    mesh,
    models: Sequence[np.ndarray],
    reference: np.ndarray,
    filename: str,
    *,
    mode: str = "difference",
    titles: Optional[Sequence[str]] = None,
    cmap: str = "RdBu_r",
    symmetric: bool = True,
    label: str = "",
    figsize: Tuple[float, float] = (8, 2.5),
    dpi: int = 150,
    duration: int = 100,
    coverage: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
) -> str:
    """Create a GIF animation showing model changes relative to a reference.

    Parameters
    ----------
    mesh : pygimli.Mesh
    models : sequence of array-like
        Model snapshots for each frame.
    reference : array-like
        Baseline model to subtract / divide.
    mode : ``'difference'`` | ``'ratio'`` | ``'percent_change'``
    symmetric : bool
        Center colorbar at zero / one.
    (other parameters same as :func:`create_timelapse_gif`)

    Returns
    -------
    str
        Path to the saved GIF file.
    """
    ref = np.asarray(reference, dtype=float).ravel()

    diff_models = []
    for m in models:
        arr = np.asarray(m, dtype=float).ravel()
        if mode == "difference":
            diff_models.append(arr - ref)
        elif mode == "ratio":
            diff_models.append(np.where(np.abs(ref) > 1e-30, arr / ref, np.nan))
        elif mode == "percent_change":
            diff_models.append(np.where(np.abs(ref) > 1e-30, 100.0 * (arr - ref) / np.abs(ref), np.nan))
        else:
            raise ValueError(f"Unknown mode '{mode}'.")

    # Compute global limits
    if symmetric:
        all_vals = np.concatenate([d.ravel() for d in diff_models])
        vmax = np.nanmax(np.abs(all_vals))
        if mode == "ratio":
            cmin_val = 1.0 / max(vmax, 1.01)
            cmax_val = max(vmax, 1.01)
        else:
            cmin_val = -vmax
            cmax_val = vmax
    else:
        cmin_val = None
        cmax_val = None

    return create_timelapse_gif(
        mesh, diff_models, filename,
        titles=titles, cmap=cmap,
        cmin=cmin_val, cmax=cmax_val,
        label=label or mode.replace("_", " ").title(),
        figsize=figsize, dpi=dpi, duration=duration,
        coverage=coverage,
    )


def create_combined_timelapse_gif(
    mesh,
    filename: str,
    *,
    wc_models: Optional[Sequence[np.ndarray]] = None,
    res_models: Optional[Sequence[np.ndarray]] = None,
    app_res_data: Optional[Sequence] = None,
    precipitation: Optional[np.ndarray] = None,
    n_frames: Optional[int] = None,
    wc_cmap=None,
    res_cmap=None,
    app_res_cmap=None,
    wc_cmin: float = 0.0,
    wc_cmax: float = 0.32,
    res_cmin: float = 100,
    res_cmax: float = 2000,
    app_res_cmin: float = 100,
    app_res_cmax: float = 1000,
    app_res_plot_type: str = "scatter",
    app_res_scatter_size: float = 10,
    mesh_ylim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (14, 10),
    dpi: int = 100,
    duration: int = 100,
    first_frame_duration: int = 500,
    loop: int = 0,
    day_labels: Optional[Sequence[str]] = None,
) -> str:
    """Create a combined GIF with water content, resistivity, apparent
    resistivity pseudosection, and precipitation panels.

    The water content and resistivity model panels are placed side-by-side
    on the same row.

    Parameters
    ----------
    mesh : pygimli.Mesh
        Mesh shared by WC and resistivity models.
    filename : str
        Output GIF path.
    wc_models : sequence of array-like, optional
        Water content model per frame.
    res_models : sequence of array-like, optional
        Resistivity model per frame.
    app_res_data : sequence, optional
        PyGimli DataContainers (or file paths) of apparent resistivity per
        frame.  Converted to SimPEG internally for pseudosection plotting.
    precipitation : array-like, optional
        1-D precipitation array (length = ``n_frames``).
    n_frames : int, optional
        Number of frames.  Inferred from whichever data list is provided.
    wc_cmap, res_cmap, app_res_cmap :
        Colormaps for each panel.
    wc_cmin, wc_cmax : float
        Water content color limits.
    res_cmin, res_cmax : float
        Resistivity color limits.
    app_res_cmin, app_res_cmax : float
        Apparent resistivity color limits.
    app_res_plot_type : str
        ``'scatter'``, ``'pcolor'``, or ``'contourf'``.
    mesh_ylim : tuple of (ymin, ymax), optional
        Y-axis limits for the mesh model panels (WC and resistivity).
        E.g. ``(1600, 1720)`` to crop the vertical range.
    figsize : tuple
        Figure size per frame.
    dpi : int
        Resolution.
    duration : int
        Milliseconds per frame.
    first_frame_duration : int
        Duration of first frame in ms.
    loop : int
        0 = infinite loop.
    day_labels : sequence of str, optional
        Label for each frame (e.g. ``'Day 0'``, ``'Day 1'``, …).

    Returns
    -------
    str
        Path to the saved GIF file.
    """
    import pygimli as pg
    from PIL import Image

    from .plotting import _get_cmap

    # Determine frame count
    for seq in [wc_models, res_models, app_res_data]:
        if seq is not None:
            if n_frames is None:
                n_frames = len(seq)
            break
    if n_frames is None:
        raise ValueError("Must provide at least one of wc_models, res_models, or app_res_data.")

    # Colormaps
    if wc_cmap is None:
        try:
            from palettable.lightbartlein.diverging import BlueDarkRed18_18_r
            wc_cmap = BlueDarkRed18_18_r.mpl_colormap
        except ImportError:
            wc_cmap = plt.get_cmap("RdBu_r")
    elif isinstance(wc_cmap, str):
        wc_cmap = plt.get_cmap(wc_cmap)

    if res_cmap is None:
        try:
            from palettable.lightbartlein.diverging import BlueDarkRed18_18
            res_cmap = BlueDarkRed18_18.mpl_colormap
        except ImportError:
            res_cmap = plt.get_cmap("RdBu")
    elif isinstance(res_cmap, str):
        res_cmap = plt.get_cmap(res_cmap)

    if app_res_cmap is None:
        try:
            from palettable.lightbartlein.diverging import BlueDarkRed18_18
            app_res_cmap = BlueDarkRed18_18.mpl_colormap
        except ImportError:
            app_res_cmap = plt.get_cmap("RdBu")
    elif isinstance(app_res_cmap, str):
        app_res_cmap = plt.get_cmap(app_res_cmap)

    # Pre-convert SimPEG data if needed
    simpeg_data_list = None
    if app_res_data is not None:
        from .plotting import _convert_pygimli_to_simpeg
        from SimPEG.electromagnetics.static.utils.static_utils import (
            plot_pseudosection,
        )
        simpeg_data_list = []
        for d in app_res_data:
            try:
                from SimPEG import data as simpeg_data_mod
                if isinstance(d, simpeg_data_mod.Data):
                    simpeg_data_list.append(d)
                else:
                    dc_dat, _ = _convert_pygimli_to_simpeg(d)
                    simpeg_data_list.append(dc_dat)
            except Exception:
                dc_dat, _ = _convert_pygimli_to_simpeg(d)
                simpeg_data_list.append(dc_dat)

    # Determine layout rows:
    #  Row 0: precipitation (if present)
    #  Row 1: WC (left) + Resistivity (right) side-by-side
    #  Row 2: apparent resistivity pseudosection (if present)
    has_precip = precipitation is not None
    has_mesh_models = wc_models is not None or res_models is not None
    has_app_res = simpeg_data_list is not None

    # Build gridspec layout
    import matplotlib.gridspec as gridspec

    rows = []
    height_ratios = []
    if has_precip:
        rows.append("precip")
        height_ratios.append(1.0)
    if has_mesh_models:
        rows.append("mesh_models")
        height_ratios.append(2.5)
    if has_app_res:
        rows.append("app_res")
        height_ratios.append(2.0)

    if not rows:
        raise ValueError("No data panels provided.")

    frames = []
    for i in range(n_frames):
        if i % 10 == 0:
            print(f"Creating combined frame {i}/{n_frames}")

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(len(rows), 1, figure=fig, height_ratios=height_ratios,
                               hspace=0.35)

        row_idx = 0

        # --- Precipitation bar ---
        if has_precip:
            ax_p = fig.add_subplot(gs[row_idx])
            row_idx += 1
            precip_arr = np.asarray(precipitation)
            ax_p.bar(np.arange(len(precip_arr)), precip_arr, color="k", width=1.0)
            ax_p.axvline(x=i, color="red", linewidth=2, label="Current")
            ax_p.set_xlim([0, len(precip_arr) - 1])
            ax_p.set_ylabel("Precip (mm)")
            ax_p.legend(loc="upper right", fontsize=8)
            if day_labels is not None and i < len(day_labels):
                ax_p.set_title(day_labels[i], fontsize=12, fontweight="bold")
            else:
                ax_p.set_title(f"Day {i}", fontsize=12, fontweight="bold")

        # --- WC + Resistivity side-by-side ---
        if has_mesh_models:
            gs_inner = gridspec.GridSpecFromSubplotSpec(
                1, 2, subplot_spec=gs[row_idx], wspace=0.35)
            row_idx += 1

            # Water content (left)
            if wc_models is not None:
                ax_wc = fig.add_subplot(gs_inner[0, 0])
                arr = np.asarray(wc_models[i], dtype=float).ravel()
                pg.show(mesh, arr, ax=ax_wc, cMap=wc_cmap, logScale=False,
                        cMin=wc_cmin, cMax=wc_cmax, label="Water Content (-)",
                        xlabel="Distance (m)", ylabel="Elevation (m)",
                        orientation="vertical", pad=0.3)
                ax_wc.set_title("Water Content Model", fontsize=10)
                if mesh_ylim is not None:
                    ax_wc.set_ylim(mesh_ylim)

            # Resistivity (right)
            if res_models is not None:
                ax_res = fig.add_subplot(gs_inner[0, 1])
                arr = np.asarray(res_models[i], dtype=float).ravel()
                pg.show(mesh, arr, ax=ax_res, cMap=res_cmap, logScale=True,
                        cMin=res_cmin, cMax=res_cmax,
                        label="Resistivity (Ω·m)",
                        xlabel="Distance (m)", ylabel="Elevation (m)",
                        orientation="vertical", pad=0.3)
                ax_res.set_title("Resistivity Model", fontsize=10)
                if mesh_ylim is not None:
                    ax_res.set_ylim(mesh_ylim)

        # --- Apparent resistivity pseudosection ---
        if has_app_res:
            ax_app = fig.add_subplot(gs[row_idx])
            row_idx += 1

            scatter_opts = {"cmap": app_res_cmap, "marker": "s",
                            "s": app_res_scatter_size}
            pcolor_opts = {"cmap": app_res_cmap}
            contourf_opts = {"cmap": app_res_cmap}

            try:
                ax_app, _ = plot_pseudosection(
                    simpeg_data_list[i],
                    plot_type=app_res_plot_type,
                    ax=ax_app,
                    scale="linear",
                    cbar_label="App. Resistivity (Ω·m)",
                    mask_topography=True,
                    pcolor_opts=pcolor_opts,
                    contourf_opts=contourf_opts,
                    scatter_opts=scatter_opts,
                    data_locations=False,
                    cbar_opts={"location": "right", "shrink": 0.8, "aspect": 30},
                    clim=[app_res_cmin, app_res_cmax],
                )
                ax_app.set_title("Apparent Resistivity Pseudosection", fontsize=10)
            except Exception as e:
                ax_app.text(0.5, 0.5, f"Error: {e}", transform=ax_app.transAxes,
                           ha="center", va="center", fontsize=8)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())

    durations = [first_frame_duration] + [duration] * (len(frames) - 1)
    frames[0].save(
        filename,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=durations,
        loop=loop,
        optimize=True,
    )
    print(f"Combined GIF saved to {filename}")
    return filename


def _panel_height_ratios(panels: List[str]) -> List[float]:
    """Return GridSpec height ratios for the given panel list."""
    ratios = []
    for p in panels:
        if p == "precip":
            ratios.append(1.0)
        else:
            ratios.append(2.0)
    return ratios
