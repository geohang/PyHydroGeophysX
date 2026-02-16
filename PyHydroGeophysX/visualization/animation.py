"""Time-lapse animation utilities for geophysical models."""

import io
import os
from typing import Callable, Optional, Sequence, Tuple, Union

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
