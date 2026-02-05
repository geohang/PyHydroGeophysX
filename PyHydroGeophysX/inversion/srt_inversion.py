"""
Seismic Refraction Tomography (SRT) inversion functionality.

Uses PyGIMLi's TravelTimeManager for forward modeling and Jacobian
computation. Provides custom Gauss-Newton inversion with the same
architecture as ERTInversion but for travel-time data.
"""

from typing import Any, Dict, Optional

import numpy as np
import pygimli as pg
from pygimli.physics import TravelTimeManager
from scipy.sparse import diags, issparse

from .base import InversionBase, InversionResult
from ..solvers.linear_solvers import generalized_solver


class SRTInversion(InversionBase):
    """
    Seismic Refraction Tomography inversion class.

    Inverts travel-time data for subsurface velocity via log-slowness,
    and uses a Gauss-Newton optimization loop.
    """

    def __init__(self, data_file: str, mesh: Optional[pg.Mesh] = None, **kwargs: Any):
        data = pg.load(data_file)
        if not isinstance(data, pg.DataContainer):
            raise TypeError("Loaded SRT data must be a PyGIMLi DataContainer.")

        super().__init__(data, mesh, **kwargs)

        defaults = {
            "lambda_val": 50.0,
            "method": "cgls",
            "zWeight": 0.2,
            "vTop": 500.0,
            "vBottom": 5000.0,
            "model_constraints": (100.0, 10000.0),
            "max_iterations": 20,
            "relativeError": 0.03,
            "absoluteError": 0.001,
            "paraMaxCellSize": 2.0,
            "paraDepth": 40.0,
            "quality": 32,
            "line_search_maxiter": 12,
            "line_search_c": 1e-4,
            "solver_maxiter": 200,
            "solver_tol": 1e-8,
            "lambda_rate": 1.0,
            "lambda_min": 1.0,
        }
        for key, value in defaults.items():
            if key not in self.parameters:
                self.parameters[key] = value

        self.mgr: Optional[TravelTimeManager] = None
        self.fop = None
        self.t_obs: Optional[np.ndarray] = None
        self.Wd = None
        self.Wd_sq = None
        self.Wd_diag: Optional[np.ndarray] = None
        self.Wm = None
        self._setup_complete = False

    @staticmethod
    def _to_col(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=float)
        return arr.reshape(-1, 1) if arr.ndim == 1 else arr

    @staticmethod
    def _cell_centers_xy(mesh: pg.Mesh) -> np.ndarray:
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

    def _estimate_data_errors(self, t_obs: np.ndarray) -> np.ndarray:
        if "err" in self.data.dataMap():
            err = np.asarray(self.data["err"].array(), dtype=float).ravel()
            valid = np.all(np.isfinite(err)) and np.any(err > 0)
            if valid and err.size == t_obs.size:
                return np.clip(err, 1e-12, None)

        rel = float(self.parameters["relativeError"])
        abs_err = float(self.parameters["absoluteError"])
        return np.sqrt((rel * np.abs(t_obs)) ** 2 + abs_err**2)

    def _apply_vertical_weighting(self, Wm):
        """Scale vertical smoothness rows by zWeight using cell-center geometry."""
        if self.mesh is None:
            return Wm

        z_weight = float(self.parameters["zWeight"])
        if z_weight <= 0:
            return Wm

        Wm_coo = Wm.tocoo()
        n_rows = Wm_coo.shape[0]
        row_to_cols: Dict[int, list] = {}
        for row, col in zip(Wm_coo.row, Wm_coo.col):
            row_to_cols.setdefault(int(row), []).append(int(col))

        centers = self._cell_centers_xy(self.mesh)
        n_cells = centers.shape[0]
        row_scale = np.ones(n_rows, dtype=float)

        for row, cols in row_to_cols.items():
            unique_cols = list(dict.fromkeys(cols))
            if len(unique_cols) != 2:
                continue
            c0, c1 = unique_cols
            if c0 >= n_cells or c1 >= n_cells:
                continue
            dx = abs(centers[c0, 0] - centers[c1, 0])
            dz = abs(centers[c0, 1] - centers[c1, 1])
            if dz > dx:
                row_scale[row] = z_weight

        return diags(row_scale).dot(Wm).tocsr()

    def _build_initial_velocity(self, n_model: int) -> np.ndarray:
        min_v, max_v = self.parameters["model_constraints"]
        min_v = max(float(min_v), 1e-6)
        max_v = max(float(max_v), min_v + 1e-6)

        if self.mesh is None:
            v = np.full(n_model, float(self.parameters["vTop"]), dtype=float)
            return np.clip(v, min_v, max_v)

        centers = self._cell_centers_xy(self.mesh)
        depth = centers[:, 1].max() - centers[:, 1]
        depth_span = np.ptp(depth)
        if depth_span <= 0:
            depth_norm = np.zeros_like(depth)
        else:
            depth_norm = depth / depth_span

        v_top = float(self.parameters["vTop"])
        v_bottom = float(self.parameters["vBottom"])
        velocity = v_top + depth_norm * (v_bottom - v_top)
        return np.clip(velocity, min_v, max_v)

    def setup(self) -> None:
        if self.mesh is None:
            temp_mgr = TravelTimeManager()
            self.mesh = temp_mgr.createMesh(
                self.data,
                paraMaxCellSize=float(self.parameters["paraMaxCellSize"]),
                quality=int(self.parameters["quality"]),
                paraDepth=float(self.parameters["paraDepth"]),
            )

        self.mgr = TravelTimeManager()
        self.mgr.setData(self.data)
        self.mgr.setMesh(self.mesh)
        self.fop = self.mgr.fop

        self.t_obs = self._to_col(np.asarray(self.data["t"], dtype=float))
        err = self._estimate_data_errors(self.t_obs.ravel())
        self.Wd_diag = 1.0 / np.clip(err, 1e-12, None)
        self.Wd = diags(self.Wd_diag)
        self.Wd_sq = diags(self.Wd_diag**2)

        rm = self.fop.regionManager()
        Ctmp = pg.matrix.RSparseMapMatrix()
        rm.setConstraintType(1)
        rm.fillConstraints(Ctmp)

        Wm = pg.utils.sparseMatrix2coo(Ctmp)
        cw = np.asarray(rm.constraintWeights().array(), dtype=float)
        Wm = diags(cw).dot(Wm).tocsr()
        self.Wm = self._apply_vertical_weighting(Wm)

        self._setup_complete = True

    def run(self, initial_model: Optional[np.ndarray] = None) -> InversionResult:
        if not self._setup_complete:
            self.setup()

        if self.fop is None or self.t_obs is None or self.Wm is None or self.Wd_diag is None:
            raise RuntimeError("SRT inversion setup is incomplete.")

        n_model = self.Wm.shape[1]

        if initial_model is None:
            v0 = self._build_initial_velocity(n_model)
        else:
            v0 = np.asarray(initial_model, dtype=float).ravel()
            if v0.size != n_model:
                raise ValueError(f"initial_model must have {n_model} parameters, got {v0.size}.")

        min_v, max_v = self.parameters["model_constraints"]
        min_v = max(float(min_v), 1e-6)
        max_v = max(float(max_v), min_v + 1e-6)

        v0 = np.clip(v0, min_v, max_v)
        m = np.log(1.0 / np.clip(v0, 1e-12, None))

        min_m = np.log(1.0 / max_v)
        max_m = np.log(1.0 / min_v)
        m = np.clip(m, min_m, max_m)
        m_ref = m.copy()

        lam = float(self.parameters["lambda_val"])
        lam_rate = float(self.parameters.get("lambda_rate", 1.0))
        lam_min = float(self.parameters.get("lambda_min", lam))

        result = InversionResult()
        prev_chi2: Optional[float] = None

        for iteration in range(int(self.parameters["max_iterations"])):
            slowness = np.exp(m)
            t_pred = self._to_col(np.asarray(self.fop.response(pg.Vector(slowness)), dtype=float))

            self.fop.createJacobian(pg.Vector(slowness))
            J = np.asarray(pg.utils.gmat2numpy(self.fop.jacobian()), dtype=float)
            if J.ndim == 1:
                J = J.reshape(-1, 1)

            J_log = J * slowness.reshape(1, -1)
            residual = self.t_obs - t_pred

            wd = self.Wd_diag.reshape(-1, 1)
            wd2 = wd**2
            phi_d = float((wd * residual).T.dot(wd * residual).item())

            delta_m = self._to_col(m - m_ref)
            reg_vec = self.Wm.dot(delta_m)
            phi_m = float(reg_vec.T.dot(reg_vec).item())

            chi2 = phi_d / max(self.t_obs.size, 1)
            d_phi = 1.0 if prev_chi2 is None else abs(chi2 - prev_chi2) / max(abs(prev_chi2), 1e-12)
            prev_chi2 = chi2

            result.iteration_models.append(np.exp(-m).copy())
            result.iteration_chi2.append(float(chi2))
            result.iteration_data_errors.append(residual.ravel().copy())

            if chi2 < 1.5 or d_phi < 0.01:
                break

            H_data = J_log.T.dot(wd2 * J_log)
            H_reg = lam * self.Wm.T.dot(self.Wm)
            H = H_data + (H_reg.toarray() if issparse(H_reg) else H_reg)

            g_data = -J_log.T.dot(wd2 * residual)
            g_reg = lam * self.Wm.T.dot(reg_vec)
            g = g_data + g_reg

            dm = generalized_solver(
                H,
                -g,
                method=str(self.parameters["method"]),
                maxiter=int(self.parameters.get("solver_maxiter", 200)),
                tol=float(self.parameters.get("solver_tol", 1e-8)),
            )
            dm = self._to_col(dm)

            current_obj = phi_d + lam * phi_m
            directional = float(dm.T.dot(g).item())
            step = 1.0
            accepted = False

            for _ in range(int(self.parameters.get("line_search_maxiter", 12))):
                m_trial = np.clip(self._to_col(m) + step * dm, min_m, max_m).ravel()
                s_trial = np.exp(m_trial)
                t_trial = self._to_col(np.asarray(self.fop.response(pg.Vector(s_trial)), dtype=float))
                res_trial = self.t_obs - t_trial

                phi_d_trial = float((wd * res_trial).T.dot(wd * res_trial).item())
                reg_trial = self.Wm.dot(self._to_col(m_trial - m_ref))
                phi_m_trial = float(reg_trial.T.dot(reg_trial).item())

                obj_trial = phi_d_trial + lam * phi_m_trial
                armijo = current_obj - float(self.parameters.get("line_search_c", 1e-4)) * step * directional

                if obj_trial < armijo:
                    m = m_trial
                    accepted = True
                    break
                step *= 0.5

            if not accepted:
                m = np.clip((self._to_col(m) + 0.1 * dm).ravel(), min_m, max_m)

            if lam_rate > 0:
                lam = max(lam_min, lam * lam_rate)

        final_velocity = np.exp(-m)
        final_slowness = np.exp(m)
        final_pred = np.asarray(self.fop.response(pg.Vector(final_slowness)), dtype=float).ravel()

        self.fop.createJacobian(pg.Vector(final_slowness))
        J_final = np.asarray(pg.utils.gmat2numpy(self.fop.jacobian()), dtype=float)
        if J_final.ndim == 1:
            J_final = J_final.reshape(-1, 1)
        coverage = np.sum(np.abs(J_final), axis=0)

        result.final_model = final_velocity
        result.predicted_data = final_pred
        result.coverage = coverage
        result.mesh = self.fop.paraDomain if hasattr(self.fop, "paraDomain") else self.mesh
        result.meta["inversion_parameters"] = dict(self.parameters)
        result.meta["final_lambda"] = lam

        return result
