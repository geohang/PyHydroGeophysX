"""
Cross-method constraint utilities for joint/cooperative inversion.

These utilities enable information sharing between geophysical methods and
provide hydrology-to-geophysics coupling helpers.
"""

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pygimli as pg
from scipy.sparse import diags, issparse

from PyHydroGeophysX.petrophysics.resistivity_models import (
    resistivity_to_water_content,
    water_content_to_resistivity,
)
from PyHydroGeophysX.petrophysics.velocity_models import (
    DEMModel,
    HertzMindlinModel,
    water_content_to_velocity,
)


class StructuralConstraint:
    """Build structural constraint terms from one method for another."""

    @staticmethod
    def _cell_centers_xy(mesh) -> np.ndarray:
        if mesh is None:
            return np.zeros((0, 2), dtype=float)

        centers_raw = np.asarray(mesh.cellCenters())
        if centers_raw.ndim == 2 and centers_raw.shape[1] >= 2:
            return centers_raw[:, :2].astype(float)

        centers = np.zeros((mesh.cellCount(), 2), dtype=float)
        for i, center in enumerate(mesh.cellCenters()):
            if hasattr(center, "x") and hasattr(center, "y"):
                centers[i, 0] = float(center.x())
                centers[i, 1] = float(center.y())
            else:
                centers[i, 0] = float(center[0])
                centers[i, 1] = float(center[1])
        return centers

    @staticmethod
    def _cell_neighbors(mesh) -> Iterable[Tuple[int, int]]:
        neighbors = []

        if mesh is not None and hasattr(mesh, "boundaries"):
            try:
                for boundary in mesh.boundaries():
                    left = boundary.leftCell()
                    right = boundary.rightCell()
                    if left is None or right is None:
                        continue
                    i = int(left.id())
                    j = int(right.id())
                    if i == j:
                        continue
                    pair = (min(i, j), max(i, j))
                    neighbors.append(pair)
            except Exception:
                neighbors = []

        if neighbors:
            return list(dict.fromkeys(neighbors))

        if mesh is not None and hasattr(mesh, "cellCount"):
            n_cells = int(mesh.cellCount())
        else:
            n_cells = 0

        return [(i, i + 1) for i in range(max(0, n_cells - 1))]

    @staticmethod
    def build_local_design_matrix(mesh) -> np.ndarray:
        """
        Build the local design matrix ``X=[x, z, 1]`` used by cross-gradient.

        This mirrors the legacy ``xyzpos[:, 2] = 1`` construction.
        """
        centers = StructuralConstraint._cell_centers_xy(mesh)
        if centers.size == 0:
            return np.zeros((0, 3), dtype=float)
        ones = np.ones((centers.shape[0], 1), dtype=float)
        return np.hstack((centers, ones))

    @staticmethod
    def build_neighborhood_matrix(
        mesh,
        Wm: Optional[Any] = None,
        source: str = "smoothness",
        correlation_lengths: Sequence[float] = (4.0, 4.0),
        threshold: float = 1e-2,
        binarize: bool = True,
    ) -> np.ndarray:
        """
        Build the neighborhood/correlation matrix used by cross-gradient.

        Parameters
        ----------
        mesh
            Mesh object.
        Wm
            Smoothness matrix. Required when ``source='smoothness'``.
        source
            One of ``'smoothness'`` (legacy direct mode) or ``'covariance'``
            (legacy spatial mode).
        correlation_lengths
            Correlation lengths passed to ``pg.utils.covarianceMatrix`` when
            ``source='covariance'``.
        threshold
            Entries with absolute value below this threshold are zeroed.
        binarize
            If ``True``, convert nonzero entries to 1.
        """
        src = str(source).lower().strip()

        if src == "smoothness":
            if Wm is None:
                raise ValueError("Wm is required for source='smoothness'.")
            Wm_arr = Wm.toarray() if issparse(Wm) else np.asarray(Wm, dtype=float)
            RCM = np.asarray(Wm_arr.T.dot(Wm_arr), dtype=float)
        elif src in ("covariance", "geostat"):
            corr = tuple(float(v) for v in correlation_lengths)
            cov = np.asarray(pg.utils.covarianceMatrix(mesh, I=list(corr)), dtype=float)
            RCM = cov.copy()
        else:
            raise ValueError("source must be 'smoothness', 'covariance', or 'geostat'.")

        if threshold > 0:
            RCM[np.abs(RCM) < float(threshold)] = 0.0
        if binarize:
            RCM = (RCM != 0).astype(float)
        return RCM

    @staticmethod
    def build_linearized_cross_gradient_blocks(
        RCM: np.ndarray,
        X: np.ndarray,
        model_a: np.ndarray,
        model_b: np.ndarray,
        mode: str = "direct",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build linearized cross-gradient blocks ``B1`` and ``B2``.

        ``B1`` multiplies model_a and ``B2`` multiplies model_b. The resulting
        penalty terms are ``||B1 m_a||^2`` and ``||B2 m_b||^2``.
        """
        R = np.asarray(RCM, dtype=float)
        if R.ndim != 2 or R.shape[0] != R.shape[1]:
            raise ValueError("RCM must be a square 2D matrix.")

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2 or X_arr.shape[0] != R.shape[0] or X_arr.shape[1] < 2:
            raise ValueError("X must have shape (n_cells, >=2) and align with RCM.")

        ma = np.asarray(model_a, dtype=float).reshape(-1, 1)
        mb = np.asarray(model_b, dtype=float).reshape(-1, 1)
        if ma.shape[0] != R.shape[0] or mb.shape[0] != R.shape[0]:
            raise ValueError("model_a and model_b sizes must match RCM size.")

        m = str(mode).lower().strip()
        if m == "direct":
            R_work = (R != 0).astype(float)
        elif m in {"spatial", "covariance"}:
            R_work = R.copy()
        else:
            raise ValueError("mode must be 'direct' or 'spatial'.")

        n_cells = R.shape[0]
        B1 = np.zeros((n_cells, n_cells), dtype=float)
        B2 = np.zeros((n_cells, n_cells), dtype=float)

        for i in range(n_cells):
            cc = np.asarray(R_work[i, :], dtype=float).ravel()
            if not np.any(np.abs(cc) > 0):
                continue

            W = np.diag(cc)
            XtWX = X_arr.T.dot(W.T).dot(X_arr)
            XtWW = X_arr.T.dot(W.T).dot(W)

            try:
                Xbar = np.linalg.solve(XtWX, XtWW)
            except np.linalg.LinAlgError:
                Xbar = np.linalg.pinv(XtWX).dot(XtWW)

            Xbar1 = Xbar[0:2, :]
            g1 = Xbar1.dot(ma)
            g2 = Xbar1.dot(mb)
            g1[np.abs(g1) < 1e-8] = 0.0
            g2[np.abs(g2) < 1e-8] = 0.0

            B1[i, :] = Xbar1[0, :] * float(g2[1, 0]) - Xbar1[1, :] * float(g2[0, 0])
            B2[i, :] = -(Xbar1[0, :] * float(g1[1, 0]) - Xbar1[1, :] * float(g1[0, 0]))

        return B1, B2

    @staticmethod
    def _boundary_weights_from_model(model: np.ndarray, mesh, gradient_threshold: float) -> Dict[Tuple[int, int], float]:
        model = np.asarray(model, dtype=float).ravel()
        weights: Dict[Tuple[int, int], float] = {}

        for i, j in StructuralConstraint._cell_neighbors(mesh):
            if i >= model.size or j >= model.size:
                continue
            denom = max(abs(model[i]), abs(model[j]), 1e-12)
            rel_grad = abs(model[j] - model[i]) / denom
            if rel_grad >= gradient_threshold:
                weights[(i, j)] = 0.1
            else:
                weights[(i, j)] = 1.0

        return weights

    @staticmethod
    def from_velocity_model(velocity_model, mesh, gradient_threshold: float = 0.3):
        """
        Extract structural boundaries from an SRT velocity model.

        Returns a pairwise-boundary weight map that can reduce smoothing
        across detected structural interfaces.
        """
        return StructuralConstraint._boundary_weights_from_model(
            np.asarray(velocity_model, dtype=float),
            mesh,
            gradient_threshold,
        )

    @staticmethod
    def from_conductivity_model(conductivity, mesh, gradient_threshold: float = 0.3):
        """Extract structural boundaries from an EM conductivity model."""
        return StructuralConstraint._boundary_weights_from_model(
            np.asarray(conductivity, dtype=float),
            mesh,
            gradient_threshold,
        )

    @staticmethod
    def build_cross_gradient_operator(mesh, model_a, model_b):
        """
        Build a cross-gradient operator surrogate.

        Returns a diagonal sparse matrix weighted by local cross-gradient
        magnitude, where small values indicate better structural agreement.
        """
        m_a = np.asarray(model_a, dtype=float).ravel()
        m_b = np.asarray(model_b, dtype=float).ravel()
        if m_a.size != m_b.size:
            raise ValueError("model_a and model_b must have the same size.")

        n_cells = m_a.size
        centers = StructuralConstraint._cell_centers_xy(mesh)
        if centers.shape[0] != n_cells:
            centers = np.column_stack((np.arange(n_cells, dtype=float), np.zeros(n_cells)))

        grad_ax = np.zeros(n_cells, dtype=float)
        grad_az = np.zeros(n_cells, dtype=float)
        grad_bx = np.zeros(n_cells, dtype=float)
        grad_bz = np.zeros(n_cells, dtype=float)

        for i, j in StructuralConstraint._cell_neighbors(mesh):
            if i >= n_cells or j >= n_cells:
                continue

            d = centers[j] - centers[i]
            dist = float(np.hypot(d[0], d[1]))
            if dist <= 0:
                continue

            da = (m_a[j] - m_a[i]) / dist
            db = (m_b[j] - m_b[i]) / dist

            wx = abs(d[0]) / dist
            wz = abs(d[1]) / dist

            grad_ax[i] += da * wx
            grad_ax[j] += da * wx
            grad_az[i] += da * wz
            grad_az[j] += da * wz

            grad_bx[i] += db * wx
            grad_bx[j] += db * wx
            grad_bz[i] += db * wz
            grad_bz[j] += db * wz

        cross = grad_ax * grad_bz - grad_az * grad_bx
        return diags(np.abs(cross) + 1e-12, format="csr")

    @staticmethod
    def apply_structural_weights_to_Wm(Wm, boundary_weights):
        """Modify an existing smoothness matrix Wm to respect boundaries."""
        is_sparse_input = issparse(Wm)
        Wm_coo = Wm.tocoo() if is_sparse_input else np.asarray(Wm)

        if isinstance(boundary_weights, dict):
            if not issparse(Wm_coo):
                from scipy.sparse import coo_matrix

                Wm_coo = coo_matrix(Wm_coo)

            row_to_cols = {}
            for row, col in zip(Wm_coo.row, Wm_coo.col):
                row_to_cols.setdefault(int(row), []).append(int(col))

            row_scale = np.ones(Wm_coo.shape[0], dtype=float)
            for row, cols in row_to_cols.items():
                unique_cols = list(dict.fromkeys(cols))
                if len(unique_cols) != 2:
                    continue
                pair = (min(unique_cols), max(unique_cols))
                if pair in boundary_weights:
                    row_scale[row] = float(boundary_weights[pair])

            weighted = diags(row_scale).dot(Wm_coo).tocsr()
            return weighted if is_sparse_input else weighted.toarray()

        if np.ndim(boundary_weights) == 1:
            weights = np.asarray(boundary_weights, dtype=float).ravel()
            if weights.size == Wm.shape[0]:
                scaled = diags(weights).dot(Wm)
                return scaled if is_sparse_input else np.asarray(scaled)
            if weights.size == Wm.shape[1]:
                scaled = Wm.dot(diags(weights))
                return scaled if is_sparse_input else np.asarray(scaled)
            raise ValueError("boundary_weights length must match Wm rows or columns.")

        bw = np.asarray(boundary_weights, dtype=float)
        if bw.shape[0] == Wm.shape[0]:
            scaled = bw.dot(Wm)
            return scaled
        if bw.shape[0] == Wm.shape[1]:
            scaled = Wm.dot(bw)
            return scaled

        raise ValueError("Unsupported boundary_weights format for Wm scaling.")


class PetrophysicalCoupling:
    """Coupling helpers from hydrological state to multi-method geophysics."""

    @staticmethod
    def _as_array(values, size=None) -> np.ndarray:
        arr = np.asarray(values, dtype=float).ravel()
        if size is None:
            return arr
        if arr.size == size:
            return arr
        if arr.size == 1:
            return np.full(size, float(arr[0]), dtype=float)
        idx = np.linspace(0.0, 1.0, arr.size)
        tgt = np.linspace(0.0, 1.0, size)
        return np.interp(tgt, idx, arr)

    @staticmethod
    def water_content_to_all_geophysics(water_content, porosity, **params):
        """
        Convert water content to resistivity, velocity, and conductivity.

        Uses existing petrophysical functions in the package.
        """
        wc = np.asarray(water_content, dtype=float).ravel()
        phi = PetrophysicalCoupling._as_array(porosity, wc.size)
        sat = np.clip(wc / np.clip(phi, 1e-6, None), 0.0, 1.0)

        rhos = float(params.get("rhos", 100.0))
        n_exp = float(params.get("n", 2.0))
        sigma_sur = float(params.get("sigma_sur", 0.0))

        resistivity = water_content_to_resistivity(
            water_content=wc,
            rhos=rhos,
            n=n_exp,
            porosity=phi,
            sigma_sur=sigma_sur,
        )
        conductivity = 1.0 / np.clip(resistivity, 1e-12, None)

        vel_mode = str(params.get("velocity_model", "hertz_mindlin")).lower()

        if vel_mode in {"dem", "differential_effective_medium"}:
            dem_model = DEMModel()
            _, _, velocity = dem_model.calculate_velocity(
                porosity=phi,
                saturation=sat,
                bulk_modulus=float(params.get("bulk_modulus", 36.0)),
                shear_modulus=float(params.get("shear_modulus", 45.0)),
                mineral_density=float(params.get("mineral_density", 2650.0)),
                aspect_ratio=float(params.get("aspect_ratio", 0.1)),
            )
        elif vel_mode in {"hertz_mindlin", "hm"}:
            hm_model = HertzMindlinModel(
                critical_porosity=float(params.get("critical_porosity", 0.4)),
                coordination_number=float(params.get("coordination_number", 4.0)),
            )
            v_high, v_low = hm_model.calculate_velocity(
                porosity=phi,
                saturation=sat,
                bulk_modulus=float(params.get("bulk_modulus", 36.0)),
                shear_modulus=float(params.get("shear_modulus", 45.0)),
                mineral_density=float(params.get("mineral_density", 2650.0)),
                depth=float(params.get("depth", 1.0)),
            )
            velocity = 0.5 * (v_high + v_low)
        else:
            velocity = water_content_to_velocity(
                water_content=wc,
                porosity=phi,
                v_dry=float(params.get("v_dry", 3500.0)),
                v_sat=float(params.get("v_sat", 4500.0)),
                model=str(params.get("empirical_model", "linear")),
            )

        return {
            "resistivity": np.asarray(resistivity, dtype=float),
            "velocity": np.asarray(velocity, dtype=float),
            "conductivity": np.asarray(conductivity, dtype=float),
        }

    @staticmethod
    def _align_to_reference(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
        ref = np.asarray(reference, dtype=float).ravel()
        val = np.asarray(values, dtype=float).ravel()
        if ref.size == val.size:
            return val

        if val.size == 1:
            return np.full(ref.size, float(val[0]), dtype=float)

        x_src = np.linspace(0.0, 1.0, val.size)
        x_tgt = np.linspace(0.0, 1.0, ref.size)
        return np.interp(x_tgt, x_src, val)

    @staticmethod
    def _stats(reference: np.ndarray, estimate: np.ndarray) -> Dict[str, float]:
        ref = np.asarray(reference, dtype=float).ravel()
        est = PetrophysicalCoupling._align_to_reference(ref, estimate)
        err = est - ref
        return {
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)),
        }

    @staticmethod
    def compare_inversions_to_hydro(
        hydro_wc,
        ert_result,
        srt_result=None,
        em_result=None,
        petro_params=None,
    ):
        """
        Compare inversion products back to hydrological water content.

        Returns misfit statistics for available methods.
        """
        params = petro_params or {}
        hydro_wc = np.asarray(hydro_wc, dtype=float).ravel()

        phi = params.get("porosity", 0.3)
        porosity = PetrophysicalCoupling._as_array(phi, hydro_wc.size)

        results = {}

        if ert_result is not None and hasattr(ert_result, "final_model") and ert_result.final_model is not None:
            resistivity = np.asarray(ert_result.final_model, dtype=float).ravel()
            rho_aligned = PetrophysicalCoupling._align_to_reference(hydro_wc, resistivity)
            wc_ert = resistivity_to_water_content(
                resistivity=rho_aligned,
                rhos=float(params.get("rhos", 100.0)),
                n=float(params.get("n", 2.0)),
                porosity=porosity,
                sigma_sur=float(params.get("sigma_sur", 0.0)),
            )
            results["ert"] = {
                "water_content": np.asarray(wc_ert, dtype=float),
                "stats": PetrophysicalCoupling._stats(hydro_wc, wc_ert),
            }

        if srt_result is not None and hasattr(srt_result, "final_model") and srt_result.final_model is not None:
            velocity = np.asarray(srt_result.final_model, dtype=float).ravel()
            v = PetrophysicalCoupling._align_to_reference(hydro_wc, velocity)
            v_dry = float(params.get("v_dry", 3500.0))
            v_sat = float(params.get("v_sat", 4500.0))
            saturation = np.clip((v - v_dry) / max(v_sat - v_dry, 1e-12), 0.0, 1.0)
            wc_srt = saturation * porosity
            results["srt"] = {
                "water_content": wc_srt,
                "stats": PetrophysicalCoupling._stats(hydro_wc, wc_srt),
            }

        if em_result is not None:
            conductivity = None
            if hasattr(em_result, "recovered_conductivity") and em_result.recovered_conductivity is not None:
                conductivity = np.asarray(em_result.recovered_conductivity, dtype=float).ravel()
            elif hasattr(em_result, "final_model") and em_result.final_model is not None:
                conductivity = np.asarray(em_result.final_model, dtype=float).ravel()

            if conductivity is not None:
                cond = PetrophysicalCoupling._align_to_reference(hydro_wc, conductivity)
                resistivity = 1.0 / np.clip(cond, 1e-12, None)
                wc_em = resistivity_to_water_content(
                    resistivity=resistivity,
                    rhos=float(params.get("rhos", 100.0)),
                    n=float(params.get("n", 2.0)),
                    porosity=porosity,
                    sigma_sur=float(params.get("sigma_sur", 0.0)),
                )
                results["em"] = {
                    "water_content": np.asarray(wc_em, dtype=float),
                    "stats": PetrophysicalCoupling._stats(hydro_wc, wc_em),
                }

        if results:
            rmse_values = [entry["stats"]["rmse"] for entry in results.values()]
            mae_values = [entry["stats"]["mae"] for entry in results.values()]
            results["summary"] = {
                "mean_rmse": float(np.mean(rmse_values)),
                "mean_mae": float(np.mean(mae_values)),
                "n_methods": len(rmse_values),
            }

        return results
