"""Hydrological model output as the bundle the hydro-to-geophysics tools read.

The bundle is four arrays in one folder: water content, porosity, the top
surface and the layer bottoms. The desktop studio's hydro page reads the top
surface as ``top.npy`` (:data:`PyHydroGeophysX.Hydro_modular.hydro_to_geophysics.HYDRO_FILES`);
the Streamlit pages and :mod:`PyHydroGeophysX.data_access` read ``top.txt``.
The MODFLOW and ParFlow converters lived in the Streamlit app and wrote only
``top.txt``, so a converted model could not be opened in the studio. Both
names are written here.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np

PathLike = Union[str, Path]

#: The files a bundle holds. ``top.txt`` carries the same surface as ``top.npy``.
BUNDLE_FILES = ("Watercontent.npy", "Porosity.npy", "top.npy", "top.txt", "bot.npy")


def write_hydro_bundle(out_dir: PathLike, water_content: np.ndarray, porosity: np.ndarray,
                       top: np.ndarray, bot: np.ndarray) -> Path:
    """Write the four bundle arrays to ``out_dir``.

    Parameters
    ----------
    out_dir : str or Path
        Folder to write into; created when missing.
    water_content : ndarray
        ``(n_time, n_layers, ny, nx)`` volumetric water content.
    porosity : ndarray
        ``(n_layers, ny, nx)``.
    top : ndarray
        ``(ny, nx)`` top surface; written as ``top.npy`` and ``top.txt``.
    bot : ndarray
        ``(n_layers, ny, nx)`` layer bottom elevations.

    Returns
    -------
    Path
        ``out_dir``.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path / "Watercontent.npy"), water_content)
    np.save(str(out_path / "Porosity.npy"), porosity)
    np.save(str(out_path / "top.npy"), top)
    np.savetxt(str(out_path / "top.txt"), top)
    np.save(str(out_path / "bot.npy"), bot)
    return out_path


def _unit_layers(n_lay: int, n_rows: int, n_cols: int):
    """A surface at 0 and layers 1 m thick below it, for models without geometry."""
    top = np.zeros((n_rows, n_cols))
    bot = np.zeros((n_lay, n_rows, n_cols))
    for k in range(n_lay):
        bot[k, :, :] = -(k + 1)
    return top, bot


def modflow_to_hydro_bundle(modflow_dir: PathLike, idomain_file: str, model_name: str,
                            nlay: int = 3, out_dir: Optional[PathLike] = None) -> str:
    """Load MODFLOW outputs and write them as a hydro bundle.

    Water content comes from the model's ``WaterContent`` output, porosity from
    the model files through flopy (a uniform 0.3 when flopy is not installed).
    The model's own layer elevations are not read: the surface is set at 0 and
    each layer is 1 m thick.

    Parameters
    ----------
    modflow_dir : str or Path
        The MODFLOW model folder.
    idomain_file : str
        The active-cell array in that folder (``.npy`` or text).
    model_name : str
        The model name, for the porosity lookup.
    nlay : int
        Layers to read from the water-content output.
    out_dir : str or Path, optional
        Where to write the bundle; a new temporary folder by default.

    Returns
    -------
    str
        The folder holding the bundle.
    """
    from PyHydroGeophysX.model_output.water_content import MODFLOWWaterContent

    modflow_path = Path(modflow_dir)
    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="phgx_mf_")

    id_path = modflow_path / idomain_file
    if id_path.suffix == ".npy":
        idomain = np.load(str(id_path))
    else:
        idomain = np.loadtxt(str(id_path))

    # Water content, (nt, nlay, nrows, ncols).
    wc_proc = MODFLOWWaterContent(model_directory=str(modflow_path), idomain=idomain)
    water_content = wc_proc.load_time_range(start_idx=0, end_idx=None, nlay=nlay)

    # Porosity, (nlay, nrows, ncols). flopy is optional.
    porosity = None
    try:
        from PyHydroGeophysX.model_output.water_content import MODFLOWPorosity
        por_proc = MODFLOWPorosity(model_directory=str(modflow_path), model_name=model_name)
        porosity = por_proc.load_porosity()
    except ImportError:
        pass

    _, n_lay, n_rows, n_cols = water_content.shape
    if porosity is None:
        porosity = np.full((n_lay, n_rows, n_cols), 0.3)
    top, bot = _unit_layers(n_lay, n_rows, n_cols)
    return str(write_hydro_bundle(out_dir, water_content, porosity, top, bot))


def parflow_to_hydro_bundle(parflow_dir: PathLike, run_name: str,
                            out_dir: Optional[PathLike] = None) -> str:
    """Load ParFlow outputs and write them as a hydro bundle.

    Water content is saturation times porosity; cells outside the mask become
    NaN. As for MODFLOW, the surface is set at 0 and each layer is 1 m thick.

    Parameters
    ----------
    parflow_dir : str or Path
        The ParFlow run folder.
    run_name : str
        The run name the ``.pfb`` files start with.
    out_dir : str or Path, optional
        Where to write the bundle; a new temporary folder by default.

    Returns
    -------
    str
        The folder holding the bundle.
    """
    from PyHydroGeophysX.model_output.parflow_output import ParflowPorosity, ParflowSaturation

    pf_path = Path(parflow_dir)
    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="phgx_pf_")

    # Saturation, (nt, nz, ny, nx).
    sat_proc = ParflowSaturation(model_directory=str(pf_path), run_name=run_name)
    saturation = sat_proc.load_time_range(start_idx=0, end_idx=None)

    # Porosity, (nz, ny, nx).
    por_proc = ParflowPorosity(model_directory=str(pf_path), run_name=run_name)
    porosity = por_proc.load_porosity()

    try:
        mask = por_proc.load_mask()
        porosity[mask == 0] = np.nan
        for t in range(saturation.shape[0]):
            saturation[t][mask == 0] = np.nan
    except FileNotFoundError:
        pass

    water_content = saturation * porosity[np.newaxis, :, :, :]
    _, n_lay, n_rows, n_cols = water_content.shape
    top, bot = _unit_layers(n_lay, n_rows, n_cols)
    return str(write_hydro_bundle(out_dir, water_content, porosity, top, bot))


__all__ = ["BUNDLE_FILES", "modflow_to_hydro_bundle", "parflow_to_hydro_bundle",
           "write_hydro_bundle"]
