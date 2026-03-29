"""Sensitivity and resolution analysis utilities."""

from typing import Any, Dict, Optional, Tuple

import numpy as np


def _as_matrix(weights, size: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(weights)
    if arr.ndim == 1:
        return np.diag(arr)
    if arr.ndim == 2:
        return arr
    if size is None:
        raise ValueError("Cannot infer matrix size from scalar weights without `size`.")
    return np.eye(size) * float(arr)


def compute_resolution_matrix(
    J: Any,
    Wd: Any,
    Wm: Any,
    lam: Any,
) -> Any:
    """
    Compute model resolution matrix:

    R = (J^T Wd Wd J + lam Wm^T Wm)^(-1) J^T Wd Wd J
    """
    J = np.asarray(J, dtype=float)
    Wd_mat = _as_matrix(Wd, J.shape[0])
    Wm_mat = _as_matrix(Wm, J.shape[1])

    jt_wdwd_j = J.T @ Wd_mat @ Wd_mat @ J
    reg_term = float(lam) * (Wm_mat.T @ Wm_mat)

    lhs = jt_wdwd_j + reg_term
    lhs_inv = np.linalg.pinv(lhs)
    return lhs_inv @ jt_wdwd_j


def compute_depth_of_investigation(
    inv_class: Any,
    data: Any,
    mesh: Any,
    scale_low: float = 0.8,
    scale_high: float = 1.2,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Estimate DOI using two inversions with different reference/initial models.

    This follows the Oldenburg-Li style idea of quantifying model sensitivity
    to reference-model choice.
    """
    if hasattr(mesh, "cellCount"):
        n_cells = int(mesh.cellCount())
    else:
        raise ValueError("mesh must expose cellCount() for DOI estimation.")

    base = np.ones(n_cells, dtype=float)

    result_low = inv_class.run(initial_model=base * scale_low)
    result_high = inv_class.run(initial_model=base * scale_high)

    model_low = np.asarray(result_low.final_model, dtype=float).ravel()
    model_high = np.asarray(result_high.final_model, dtype=float).ravel()

    if model_low.size != model_high.size:
        raise ValueError("DOI models are incompatible in size.")

    doi = np.abs(model_high - model_low) / (np.abs(model_high) + np.abs(model_low) + 1e-12)
    doi = np.clip(doi, 0.0, 1.0)

    return doi, {"model_low": model_low, "model_high": model_high}


def compute_cumulative_sensitivity(
    J: Any,
) -> Any:
    """Compute cumulative sensitivity as absolute column sums of the Jacobian."""
    J = np.asarray(J, dtype=float)
    if J.ndim != 2:
        raise ValueError("J must be a 2D matrix.")
    return np.sum(np.abs(J), axis=0)


def plot_sensitivity_map(
    sensitivity: Any,
    mesh: Any,
    ax: Any = None,
) -> Any:
    """Plot a sensitivity map on a mesh (or as a simple line plot fallback)."""
    import matplotlib.pyplot as plt

    sens = np.asarray(sensitivity, dtype=float).ravel()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    # Try mesh-based plotting first when PyGIMLi mesh plotting is available.
    mesh_plot_ok = False
    try:
        import pygimli as pg

        if mesh is not None and hasattr(mesh, "cellCount") and mesh.cellCount() == sens.size:
            pg.show(mesh, data=sens, ax=ax, cMap="viridis", label="Sensitivity")
            ax.set_title("Sensitivity Map")
            mesh_plot_ok = True
    except Exception:
        mesh_plot_ok = False

    if not mesh_plot_ok:
        ax.plot(np.arange(sens.size), sens, "k-")
        ax.set_xlabel("Cell Index")
        ax.set_ylabel("Sensitivity")
        ax.set_title("Cumulative Sensitivity")
        ax.grid(True, alpha=0.3)

    return fig, ax
