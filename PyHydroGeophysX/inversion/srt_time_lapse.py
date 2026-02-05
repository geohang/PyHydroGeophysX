"""
Time-lapse Seismic Refraction Tomography (SRT) inversion functionality.

Jointly inverts multiple travel-time datasets with spatial and temporal
regularization using log-slowness parameterization.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygimli as pg
from pygimli.physics import TravelTimeManager
from scipy.sparse import block_diag as sparse_block_diag
from scipy.sparse import diags, issparse, lil_matrix

from .base import InversionBase, TimeLapseInversionResult
from ..solvers.linear_solvers import generalized_solver


class TimeLapseSRTInversion(InversionBase):
    """
    Time-lapse Seismic Refraction Tomography inversion.

    Architecture mirrors TimeLapseERTInversion while using
    TravelTimeManager and log-slowness model parameters.
    """

    def __init__(
        self,
        data_files: List[str],
        measurement_times: List[float],
        mesh: Optional[pg.Mesh] = None,
        **kwargs: Any,
    ):
        if len(data_files) != len(measurement_times):
            raise ValueError("Number of data_files must match measurement_times.")
        if len(data_files) < 2:
            raise ValueError("TimeLapseSRTInversion requires at least two datasets.")

        first_data = pg.load(data_files[0])
        if not isinstance(first_data, pg.DataContainer):
            raise TypeError("Loaded SRT data must be a PyGIMLi DataContainer.")

        super().__init__(first_data, mesh, **kwargs)

        defaults = {
            "lambda_val": 50.0,
            "alpha": 10.0,
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
            "solver_maxiter": 300,
            "solver_tol": 1e-8,
            "lambda_rate": 1.0,
            "lambda_min": 1.0,
        }
        for key, value in defaults.items():
            if key not in self.parameters:
                self.parameters[key] = value

        self.data_files = list(data_files)
        self.measurement_times = np.asarray(measurement_times, dtype=float)
        self.n_times = len(self.data_files)

        self.datasets: List[pg.DataContainer] = []
        self.managers: List[TravelTimeManager] = []
        self.fops = []

        self.t_obs: Optional[np.ndarray] = None
        self.Wd_diag: Optional[np.ndarray] = None
        self.Wd = None
        self.Wd_sq = None
        self.Wm = None
        self.Wt = None
        self.n_cells: Optional[int] = None
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

    def _estimate_errors(self, data: pg.DataContainer, t_obs: np.ndarray) -> np.ndarray:
        if "err" in data.dataMap():
            err = np.asarray(data["err"].array(), dtype=float).ravel()
            valid = np.all(np.isfinite(err)) and np.any(err > 0)
            if valid and err.size == t_obs.size:
                return np.clip(err, 1e-12, None)

        rel = float(self.parameters["relativeError"])
        abs_err = float(self.parameters["absoluteError"])
        return np.sqrt((rel * np.abs(t_obs)) ** 2 + abs_err**2)

    def _apply_vertical_weighting(self, Wm):
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

    def _build_initial_velocity(self) -> np.ndarray:
        if self.n_cells is None:
            raise RuntimeError("n_cells is undefined. Call setup() first.")

        min_v, max_v = self.parameters["model_constraints"]
        min_v = max(float(min_v), 1e-6)
        max_v = max(float(max_v), min_v + 1e-6)

        if self.mesh is None:
            v = np.full(self.n_cells, float(self.parameters["vTop"]), dtype=float)
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

    def _forward_and_jacobian(self, model_log_slowness: np.ndarray) -> Tuple[np.ndarray, Any]:
        if self.n_cells is None:
            raise RuntimeError("n_cells is undefined. Call setup() first.")

        m2d = np.reshape(model_log_slowness, (self.n_cells, self.n_times), order="F")

        pred_blocks = []
        jac_blocks = []

        for it, fop in enumerate(self.fops):
            s_it = np.exp(m2d[:, it])
            pred_it = np.asarray(fop.response(pg.Vector(s_it)), dtype=float).reshape(-1, 1)
            pred_blocks.append(pred_it)

            fop.createJacobian(pg.Vector(s_it))
            J_it = np.asarray(pg.utils.gmat2numpy(fop.jacobian()), dtype=float)
            if J_it.ndim == 1:
                J_it = J_it.reshape(-1, 1)

            J_log_it = J_it * s_it.reshape(1, -1)
            jac_blocks.append(J_log_it)

        pred = np.vstack(pred_blocks)
        J = sparse_block_diag(jac_blocks, format="csr")
        return pred, J

    def _objective_terms(self, model_log_slowness: np.ndarray) -> Tuple[float, float, float, float]:
        pred, _ = self._forward_and_jacobian(model_log_slowness)
        residual = self.t_obs - pred

        wd = self.Wd_diag.reshape(-1, 1)
        phi_d = float((wd * residual).T.dot(wd * residual).item())

        delta_m = self._to_col(model_log_slowness)
        reg_spatial = self.Wm.dot(delta_m)
        reg_temporal = self.Wt.dot(delta_m)

        phi_m = float(reg_spatial.T.dot(reg_spatial).item())
        phi_t = float(reg_temporal.T.dot(reg_temporal).item())
        return phi_d, phi_m, phi_t, float(phi_d / max(self.t_obs.size, 1))

    def setup(self) -> None:
        if self.mesh is None:
            temp_data = pg.load(self.data_files[0])
            temp_mgr = TravelTimeManager()
            self.mesh = temp_mgr.createMesh(
                temp_data,
                paraMaxCellSize=float(self.parameters["paraMaxCellSize"]),
                quality=int(self.parameters["quality"]),
                paraDepth=float(self.parameters["paraDepth"]),
            )

        self.datasets = []
        self.managers = []
        self.fops = []

        obs_blocks = []
        wd_blocks = []

        for data_file in self.data_files:
            data = pg.load(data_file)
            if not isinstance(data, pg.DataContainer):
                raise TypeError(f"Loaded SRT data is not a DataContainer: {data_file}")
            self.datasets.append(data)

            mgr = TravelTimeManager()
            mgr.setData(data)
            mgr.setMesh(self.mesh)
            self.managers.append(mgr)
            self.fops.append(mgr.fop)

            t_obs = np.asarray(data["t"], dtype=float).ravel()
            obs_blocks.append(self._to_col(t_obs))

            err = self._estimate_errors(data, t_obs)
            wd_blocks.append(1.0 / np.clip(err, 1e-12, None))

        self.t_obs = np.vstack(obs_blocks)
        self.Wd_diag = np.hstack(wd_blocks)
        self.Wd = diags(self.Wd_diag)
        self.Wd_sq = diags(self.Wd_diag**2)

        rm = self.fops[0].regionManager()
        Ctmp = pg.matrix.RSparseMapMatrix()
        rm.setConstraintType(1)
        rm.fillConstraints(Ctmp)

        Wm_single = pg.utils.sparseMatrix2coo(Ctmp)
        cw = np.asarray(rm.constraintWeights().array(), dtype=float)
        Wm_single = diags(cw).dot(Wm_single).tocsr()
        Wm_single = self._apply_vertical_weighting(Wm_single)

        self.n_cells = int(Wm_single.shape[1])
        self.Wm = sparse_block_diag([Wm_single] * self.n_times, format="csr")

        Wt = lil_matrix((self.n_cells * (self.n_times - 1), self.n_cells * self.n_times), dtype=float)
        for it in range(self.n_times - 1):
            row0 = it * self.n_cells
            row1 = (it + 1) * self.n_cells
            col_a0 = it * self.n_cells
            col_a1 = (it + 1) * self.n_cells
            col_b0 = (it + 1) * self.n_cells
            col_b1 = (it + 2) * self.n_cells

            Wt[row0:row1, col_a0:col_a1] = np.eye(self.n_cells)
            Wt[row0:row1, col_b0:col_b1] = -np.eye(self.n_cells)

        self.Wt = Wt.tocsr()
        self._setup_complete = True

    def run(self, initial_model: Optional[np.ndarray] = None) -> TimeLapseInversionResult:
        if not self._setup_complete:
            self.setup()

        if (
            self.t_obs is None
            or self.Wd_diag is None
            or self.Wm is None
            or self.Wt is None
            or self.n_cells is None
        ):
            raise RuntimeError("Time-lapse SRT inversion setup is incomplete.")

        min_v, max_v = self.parameters["model_constraints"]
        min_v = max(float(min_v), 1e-6)
        max_v = max(float(max_v), min_v + 1e-6)
        min_m = np.log(1.0 / max_v)
        max_m = np.log(1.0 / min_v)

        if initial_model is None:
            v0_single = self._build_initial_velocity()
            v0 = np.tile(v0_single, self.n_times)
        else:
            v0 = np.asarray(initial_model, dtype=float).ravel()
            if v0.size == self.n_cells:
                v0 = np.tile(v0, self.n_times)
            elif v0.size != self.n_cells * self.n_times:
                raise ValueError(
                    f"initial_model must have {self.n_cells} or {self.n_cells * self.n_times} values, "
                    f"got {v0.size}."
                )

        v0 = np.clip(v0, min_v, max_v)
        m = np.log(1.0 / np.clip(v0, 1e-12, None))
        m = np.clip(m, min_m, max_m)
        m_ref = m.copy()

        lam = float(self.parameters["lambda_val"])
        alpha = float(self.parameters["alpha"])
        lam_rate = float(self.parameters.get("lambda_rate", 1.0))
        lam_min = float(self.parameters.get("lambda_min", lam))

        chi2_history: List[float] = []

        for _ in range(int(self.parameters["max_iterations"])):
            pred, J = self._forward_and_jacobian(m)
            residual = self.t_obs - pred

            wd = self.Wd_diag.reshape(-1, 1)
            wd2 = wd**2

            phi_d = float((wd * residual).T.dot(wd * residual).item())
            delta_m = self._to_col(m - m_ref)
            reg_spatial = self.Wm.dot(delta_m)
            reg_temporal = self.Wt.dot(self._to_col(m))

            phi_m = float(reg_spatial.T.dot(reg_spatial).item())
            phi_t = float(reg_temporal.T.dot(reg_temporal).item())
            chi2 = phi_d / max(self.t_obs.size, 1)
            chi2_history.append(float(chi2))

            if len(chi2_history) > 1:
                d_phi = abs(chi2_history[-1] - chi2_history[-2]) / max(abs(chi2_history[-2]), 1e-12)
            else:
                d_phi = 1.0

            if chi2 < 1.5 or d_phi < 0.01:
                break

            H_data = J.T.dot(self.Wd_sq.dot(J))
            H_reg = lam * self.Wm.T.dot(self.Wm) + alpha * self.Wt.T.dot(self.Wt)
            H = H_data + H_reg

            g_data = -J.T.dot(wd2 * residual)
            g_reg = lam * self.Wm.T.dot(reg_spatial) + alpha * self.Wt.T.dot(reg_temporal)
            g = g_data + g_reg

            H_solve = H.toarray() if issparse(H) else np.asarray(H, dtype=float)
            dm = generalized_solver(
                H_solve,
                -g,
                method=str(self.parameters["method"]),
                maxiter=int(self.parameters.get("solver_maxiter", 300)),
                tol=float(self.parameters.get("solver_tol", 1e-8)),
            )
            dm = self._to_col(dm)

            current_obj = phi_d + lam * phi_m + alpha * phi_t
            directional = float(dm.T.dot(g).item())
            step = 1.0
            accepted = False

            for _ in range(int(self.parameters.get("line_search_maxiter", 12))):
                m_trial = np.clip(self._to_col(m) + step * dm, min_m, max_m).ravel()
                phi_d_trial, phi_m_trial, phi_t_trial, _ = self._objective_terms(m_trial)
                obj_trial = phi_d_trial + lam * phi_m_trial + alpha * phi_t_trial
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

        pred_final, _ = self._forward_and_jacobian(m)
        final_models = np.exp(-np.reshape(m, (self.n_cells, self.n_times), order="F"))

        all_coverage = []
        m2d = np.reshape(m, (self.n_cells, self.n_times), order="F")
        for it, fop in enumerate(self.fops):
            s_it = np.exp(m2d[:, it])
            fop.createJacobian(pg.Vector(s_it))
            J_it = np.asarray(pg.utils.gmat2numpy(fop.jacobian()), dtype=float)
            if J_it.ndim == 1:
                J_it = J_it.reshape(-1, 1)
            all_coverage.append(np.sum(np.abs(J_it), axis=0))

        result = TimeLapseInversionResult()
        result.timesteps = self.measurement_times
        result.final_models = final_models
        result.final_model = final_models[:, -1].copy()
        result.predicted_data = pred_final.ravel()
        result.mesh = self.fops[0].paraDomain if hasattr(self.fops[0], "paraDomain") else self.mesh
        result.all_coverage = all_coverage
        result.coverage = all_coverage[-1].copy() if all_coverage else None
        result.all_chi2 = chi2_history
        result.iteration_chi2 = chi2_history.copy()
        result.meta["inversion_parameters"] = dict(self.parameters)
        result.meta["final_lambda"] = lam

        return result
