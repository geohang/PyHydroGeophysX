"""
Seismic Refraction Tomography (SRT) inversion functionality.

Uses PyGIMLi's TravelTimeManager for forward modeling and Jacobian
computation. Provides custom Gauss-Newton inversion with the same
architecture as ERTInversion but for travel-time data.
"""

from typing import Any, Optional

import numpy as np
import pygimli as pg
from pygimli.physics import TravelTimeManager
import pygimli.physics.traveltime as tt
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
        """
        Initialize SRT inversion.

        Args:
            data_file: Path to travel-time data file (e.g. .dat/.sgt).
            mesh: Mesh for inversion (created if None).
            **kwargs: Additional inversion parameters, aligned with ERT naming where possible:
                - lambda_val: Regularization parameter.
                - method: Linear solver ('cgls', 'lsqr', etc.).
                - model_constraints: (min_velocity, max_velocity) bounds.
                - max_iterations: Maximum GN iterations.
                - target_chi_squared: Stop criterion for chi2 (default 1.0).
                - lambda_rate: Lambda reduction rate per iteration.
                - lambda_min: Minimum lambda value.
                - relativeError: Relative data error.
                - absoluteUError: Absolute data error (ERT-style name).
                - zWeight: Vertical smoothness weighting.
                - vTop: Starting-model velocity near surface.
                - vBottom: Starting-model velocity at depth.
                - paraMaxCellSize: Inversion mesh max cell size.
                - paraDepth: Inversion mesh depth.
                - quality: Mesh quality parameter.
                - line_search_maxiter: Max line-search halvings.
                - line_search_c: Armijo coefficient.
                - solver_maxiter: Linear solver max iterations.
                - solver_tol: Linear solver tolerance.
        """
        # Keep ERT-style name for user-facing consistency.
        if "absoluteUError" in kwargs and "absoluteError" not in kwargs:
            kwargs["absoluteError"] = kwargs["absoluteUError"]
        if "absoluteError" in kwargs and "absoluteUError" not in kwargs:
            kwargs["absoluteUError"] = kwargs["absoluteError"]

        data = tt.load(str(data_file))
        if not isinstance(data, pg.DataContainer):
            raise TypeError("Loaded SRT data must be a PyGIMLi DataContainer.")
        required_fields = ("s", "g", "t")
        missing = [field for field in required_fields if field not in data.dataMap()]
        if missing:
            raise ValueError(
                f"SRT data is missing required fields: {missing}. "
                "Expected travel-time data with 's', 'g', and 't'."
            )
        if data.size() == 0:
            raise ValueError(
                "Loaded SRT data has zero valid measurements. "
                "Check file format and sensor indexing."
            )

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
            "absoluteUError": 0.001,
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
    def _jacobian_to_numpy(jacobian: Any) -> np.ndarray:
        """Convert PyGIMLi Jacobian matrix to a dense NumPy array across versions."""
        try:
            return pg.utils.sparseMatrix2coo(jacobian).toarray().astype(float, copy=False)
        except Exception:
            pass
        try:
            return np.asarray(pg.utils.gmat2numpy(jacobian), dtype=float)
        except Exception:
            pass

        arr = np.asarray(jacobian, dtype=float)
        if arr.ndim == 0:
            raise TypeError(f"Unsupported Jacobian type: {type(jacobian)}")
        return arr

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
        abs_err = float(self.parameters.get("absoluteUError", self.parameters.get("absoluteError", 0.001)))
        return np.sqrt((rel * np.abs(t_obs)) ** 2 + abs_err**2)

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
        rm.setZWeight(float(self.parameters["zWeight"]))
        Ctmp = pg.matrix.RSparseMapMatrix()
        rm.setConstraintType(1)
        rm.fillConstraints(Ctmp)

        # `fillConstraints` already applies region-dependent weighting
        # (including zWeight). Do not multiply constraintWeights again.
        self.Wm = pg.utils.sparseMatrix2coo(Ctmp).tocsr()

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

        lam = float(self.parameters["lambda_val"])
        lam_rate = float(self.parameters.get("lambda_rate", 1.0))
        lam_min = float(self.parameters.get("lambda_min", lam))
        target_chi2 = float(self.parameters.get("target_chi_squared", 1.0))

        result = InversionResult()
        prev_chi2: Optional[float] = None

        print("SRTInversion note: reported chi2 below is evaluated at the start of each iteration (pre-update).")
        print("PyGIMLi inv.iter chi2 is reported after oneStep update.")
        for iteration in range(int(self.parameters["max_iterations"])):
            print(f"-------------------Iteration: {iteration} ---------------------------")
            slowness = np.exp(m)
            t_pred = self._to_col(np.asarray(self.fop.response(pg.Vector(slowness)), dtype=float))

            self.fop.createJacobian(pg.Vector(slowness))
            J = self._jacobian_to_numpy(self.fop.jacobian())
            if J.ndim == 1:
                J = J.reshape(-1, 1)

            J_log = J * slowness.reshape(1, -1)
            residual = self.t_obs - t_pred

            wd = self.Wd_diag.reshape(-1, 1)
            wd2 = wd**2
            phi_d = float((wd * residual).T.dot(wd * residual).item())

            m_col = self._to_col(m)
            reg_vec = self.Wm.dot(m_col)
            phi_m = float(reg_vec.T.dot(reg_vec).item())

            chi2 = phi_d / max(self.t_obs.size, 1)
            d_phi = 1.0 if prev_chi2 is None else abs(chi2 - prev_chi2) / max(abs(prev_chi2), 1e-12)
            prev_chi2 = chi2
            obj = phi_d + lam * phi_m

            result.iteration_models.append(np.exp(-m).copy())
            result.iteration_chi2.append(float(chi2))
            result.iteration_data_errors.append(residual.ravel().copy())

            print(f"chi2 (pre-update): {chi2:.6f}")
            print(f"dPhi (pre-update): {d_phi:.6f}")
            print(f"phi_d (pre-update): {phi_d:.6f}")
            print(f"phi_m (pre-update): {phi_m:.6f}")
            print(f"obj (pre-update): {obj:.6f}")
            print(f"lambda: {lam:.6f}")

            if chi2 <= target_chi2 or d_phi < 0.01:
                print("Converged: stopping criterion reached.")
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

            current_obj = obj
            directional = float(dm.T.dot(g).item())
            step = 1.0
            accepted = False
            accepted_step = step

            for _ in range(int(self.parameters.get("line_search_maxiter", 12))):
                m_trial = np.clip(self._to_col(m) + step * dm, min_m, max_m).ravel()
                s_trial = np.exp(m_trial)
                t_trial = self._to_col(np.asarray(self.fop.response(pg.Vector(s_trial)), dtype=float))
                res_trial = self.t_obs - t_trial

                phi_d_trial = float((wd * res_trial).T.dot(wd * res_trial).item())
                reg_trial = self.Wm.dot(self._to_col(m_trial))
                phi_m_trial = float(reg_trial.T.dot(reg_trial).item())

                obj_trial = phi_d_trial + lam * phi_m_trial
                armijo = current_obj - float(self.parameters.get("line_search_c", 1e-4)) * step * directional

                if obj_trial < armijo:
                    m = m_trial
                    accepted = True
                    accepted_step = step
                    break
                step *= 0.5

            if not accepted:
                accepted_step = 0.1
                print("Line search FAIL EXIT")
                m = np.clip((self._to_col(m) + accepted_step * dm).ravel(), min_m, max_m)

            print(f"accepted_step: {accepted_step:.6f}")
            print(f"update_norm: {float(np.linalg.norm(dm)):.6e}")

            if lam_rate > 0:
                lam = max(lam_min, lam * lam_rate)

        final_velocity = np.exp(-m)
        final_slowness = np.exp(m)
        final_pred = np.asarray(self.fop.response(pg.Vector(final_slowness)), dtype=float).ravel()

        self.fop.createJacobian(pg.Vector(final_slowness))
        # Use the same coverage definition as PyGIMLi TravelTimeManager:
        # standardizedCoverage = sign(|C^T * C * rayCoverage|).
        if self.mgr is not None:
            try:
                if hasattr(self.fop, "createConstraints"):
                    self.fop.createConstraints()
                coverage = np.asarray(self.mgr.standardizedCoverage(), dtype=float).ravel()
            except Exception:
                # Fallback with the same standardized-coverage formula.
                J = self.fop.jacobian()
                ray_coverage = np.asarray(J.transMult(np.ones(J.rows())), dtype=float).ravel()
                Ctmp = pg.matrix.RSparseMapMatrix()
                self.fop.regionManager().fillConstraints(Ctmp)
                C = pg.utils.sparseMatrix2coo(Ctmp).tocsr()
                coverage = np.sign(np.abs(C.T.dot(C.dot(ray_coverage))))
        else:
            coverage = None

        result.final_model = final_velocity
        result.predicted_data = final_pred
        result.coverage = coverage
        result.mesh = self.fop.paraDomain if hasattr(self.fop, "paraDomain") else self.mesh
        result.meta["inversion_parameters"] = dict(self.parameters)
        result.meta["final_lambda"] = lam

        print("End of inversion")
        return result
