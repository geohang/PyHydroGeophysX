"""
Joint ERT-SRT inversion with structural and geostatistical constraints.

This module integrates the legacy alternating inversion strategy from
``cg_joint_inversion_hang`` into the package architecture.
"""

from dataclasses import dataclass, field
from os import PathLike
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pygimli as pg
import pygimli.physics.traveltime as tt
from pygimli.physics import TravelTimeManager, ert
from scipy.sparse import csr_matrix, diags, vstack
from scipy.sparse.linalg import lsqr

from .base import InversionBase
from .cross_constraints import StructuralConstraint
from ..solvers.linear_solvers import generalized_solver


@dataclass
class JointERTSRTResult:
    """Container for joint ERT-SRT inversion outputs."""

    ert_log_resistivity: Optional[np.ndarray] = None
    srt_log_slowness: Optional[np.ndarray] = None
    ert_resistivity: Optional[np.ndarray] = None
    srt_velocity: Optional[np.ndarray] = None
    ert_predicted: Optional[np.ndarray] = None
    srt_predicted: Optional[np.ndarray] = None
    ert_coverage: Optional[np.ndarray] = None
    srt_coverage: Optional[np.ndarray] = None
    chi2_ert: Optional[float] = None
    chi2_srt: Optional[float] = None
    iteration_history: list = field(default_factory=list)
    mesh: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)


class JointERTSRTInversion(InversionBase):
    """
    Joint ERT-SRT inversion using alternating Gauss-Newton updates.

    - ERT model parameter: log-resistivity
    - SRT model parameter: log-slowness
    - Coupling: linearized cross-gradient terms (B1, B2)
    - Regularization: smoothness or geostatistical
    """

    def __init__(
        self,
        ert_data: Union[str, PathLike, pg.DataContainer],
        srt_data: Union[str, PathLike, pg.DataContainer],
        mesh: Optional[pg.Mesh] = None,
        **kwargs: Any,
    ):
        ert_dc = self._load_ert_data(ert_data)
        srt_dc = self._load_srt_data(srt_data)

        super().__init__(ert_dc, mesh, **kwargs)
        self.ert_data = ert_dc
        self.srt_data = srt_dc

        defaults = {
            "max_iterations": 50,
            "solver": "scipy_lsmr",
            "solver_maxiter": 400,
            "solver_tol": 1e-8,
            "line_search_maxiter": 20,
            "line_search_c": 1e-4,
            "target_chi2": 1.5,
            "convergence_tolerance": 0.01,
            "lambda_ert": 10.0,
            "lambda_srt": 10.0,
            "lambda_rate_ert": 0.8,
            "lambda_rate_srt": 0.8,
            "lambda_min_ert": 1.0,
            "lambda_min_srt": 1.0,
            "lambda_cg_ert": 120.0,
            "lambda_cg_srt": 80.0,
            "regularization_mode": "smoothness",
            "geostat_I": 5.0,
            "geostat_var": 0.01,
            "geostat_sparse_threshold": 0.01,
            "cross_gradient_mode": "direct",
            "cross_gradient_source": "smoothness",
            "cross_gradient_corr_lengths": (4.0, 4.0),
            "cross_gradient_threshold": 0.01,
            "ert_use_derived_rhoa": True,
            "ert_relative_error": 0.05,
            "ert_absolute_u_error": 0.0,
            "srt_relative_error": 0.05,
            "srt_absolute_error": 0.0,
            "ert_bounds": (10.0, 5000.0),
            "srt_velocity_bounds": (50.0, 5000.0),
            "vTop": 500.0,
            "vBottom": 5000.0,
            "mesh_quality": 30,
            "paraDX": 1.0,
            "paraMaxCellSize": 4.0,
            "boundaryMaxCellSize": 3000.0,
            "paraBoundary": 7.2,
            "smooth": (2, 2),
            "balanceDepth": True,
            "auto_disable_cross_gradient_first_iteration": False,
            "verbose": True,
        }
        for key, value in defaults.items():
            if key not in self.parameters:
                self.parameters[key] = value

        # Backward/forward-compatible aliases between old/new naming.
        if "ert_relative_error" not in kwargs:
            self.parameters["ert_relative_error"] = float(self.parameters.get("relativeError", 0.03))
        if "ert_absolute_u_error" not in kwargs:
            self.parameters["ert_absolute_u_error"] = float(self.parameters.get("absoluteUError", 0.0))
        if "srt_relative_error" not in kwargs:
            self.parameters["srt_relative_error"] = float(self.parameters.get("relativeError", 0.03))

        self.ert_manager = None
        self.srt_manager = None
        self.ert_fop = None
        self.srt_fop = None
        self.ert_grid: Optional[pg.Mesh] = None

        self.Wm_ert: Optional[csr_matrix] = None
        self.Wm_srt: Optional[csr_matrix] = None
        self.Wd_ert: Optional[csr_matrix] = None
        self.Wd_srt: Optional[csr_matrix] = None
        self.dobs_ert: Optional[np.ndarray] = None
        self.dobs_srt: Optional[np.ndarray] = None
        self.mr: Optional[np.ndarray] = None
        self.mv: Optional[np.ndarray] = None
        self.mr_ref: Optional[np.ndarray] = None
        self.mv_ref: Optional[np.ndarray] = None
        self.RCM: Optional[np.ndarray] = None
        self.X: Optional[np.ndarray] = None
        self._ert_error_source: str = "unknown"
        self._ert_observed_source: str = "unknown"

        self._setup_complete = False

    @staticmethod
    def _load_ert_data(data: Union[str, PathLike, pg.DataContainer]) -> pg.DataContainer:
        if isinstance(data, (str, PathLike)):
            loaded = ert.load(str(data))
        else:
            loaded = data

        if not isinstance(loaded, pg.DataContainer):
            raise TypeError("ert_data must be a path or a PyGIMLi DataContainer.")
        return loaded

    @staticmethod
    def _load_srt_data(data: Union[str, PathLike, pg.DataContainer]) -> pg.DataContainer:
        if isinstance(data, (str, PathLike)):
            loaded = tt.load(str(data))
        else:
            loaded = data

        if not isinstance(loaded, pg.DataContainer):
            raise TypeError("srt_data must be a path or a PyGIMLi DataContainer.")
        return loaded

    @staticmethod
    def _as_col(values: Any) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    @staticmethod
    def _needs_ert_k_rebuild(data: pg.DataContainer) -> bool:
        if "k" not in data.dataMap():
            return True

        try:
            k = np.asarray(data["k"], dtype=float).ravel()
        except Exception:
            return True

        if k.size == 0:
            return True
        if not np.all(np.isfinite(k)):
            return True
        if np.all(np.abs(k) < 1e-12):
            return True
        return False

    def _ensure_ert_geometric_factors(self) -> None:
        if not self._needs_ert_k_rebuild(self.ert_data):
            return
        try:
            self.ert_data["k"] = ert.createGeometricFactors(self.ert_data, numerical=True)
        except Exception:
            self.ert_data["k"] = ert.createGeometricFactors(self.ert_data)

    @staticmethod
    def _jacobian_to_numpy(jac: Any) -> np.ndarray:
        try:
            out = np.asarray(pg.utils.gmat2numpy(jac), dtype=float)
        except Exception:
            try:
                out = np.asarray(pg.utils.sparseMatrix2coo(jac).toarray(), dtype=float)
            except Exception:
                out = np.asarray(jac, dtype=float)
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        return out

    @staticmethod
    def _clip_log_model(model_log: np.ndarray, lower_lin: float, upper_lin: float, inverse: bool = False) -> np.ndarray:
        low = max(float(lower_lin), 1e-12)
        high = max(float(upper_lin), low + 1e-12)
        if inverse:
            min_log = np.log(1.0 / high)
            max_log = np.log(1.0 / low)
        else:
            min_log = np.log(low)
            max_log = np.log(high)
        return np.clip(np.asarray(model_log, dtype=float).ravel(), min_log, max_log)

    def _build_ert_observed_log_data(self) -> np.ndarray:
        use_derived = bool(self.parameters.get("ert_use_derived_rhoa", True))
        has_ui = ("u" in self.ert_data.dataMap()) and ("i" in self.ert_data.dataMap())

        if use_derived and has_ui:
            self._ensure_ert_geometric_factors()
            k = np.asarray(self.ert_data["k"], dtype=float).ravel()
            u = np.asarray(self.ert_data["u"], dtype=float).ravel()
            i = np.asarray(self.ert_data["i"], dtype=float).ravel()
            rhoa = k * u / np.clip(i, 1e-12, None)
            self._ert_observed_source = "derived_from_k_u_i"
        elif "rhoa" in self.ert_data.dataMap():
            rhoa = np.asarray(self.ert_data["rhoa"], dtype=float).ravel()
            self._ert_observed_source = "file_rhoa"
        elif has_ui:
            self._ensure_ert_geometric_factors()
            k = np.asarray(self.ert_data["k"], dtype=float).ravel()
            u = np.asarray(self.ert_data["u"], dtype=float).ravel()
            i = np.asarray(self.ert_data["i"], dtype=float).ravel()
            rhoa = k * u / np.clip(i, 1e-12, None)
            self._ert_observed_source = "derived_from_k_u_i_fallback"
        else:
            raise RuntimeError("ERT observed data unavailable: need either 'rhoa' or ('u','i','k').")

        rhoa = np.clip(rhoa, 1e-12, None)
        return self._as_col(np.log(rhoa))

    def _build_weights(self) -> None:
        t = self.dobs_srt.ravel()

        # --- ERT weighting ---
        # sigma = ert_relative_error + ert_absolute_u_error
        # Same simple pattern as SRT weighting.
        rel_ert = float(self.parameters["ert_relative_error"])
        abs_ert = float(self.parameters["ert_absolute_u_error"])
        sigma_ert = np.clip(rel_ert + abs_ert, 1e-12, None)
        w_ert = np.full(self.dobs_ert.size, 1.0 / sigma_ert, dtype=float)
        self._ert_error_source = f"rel={rel_ert}, abs={abs_ert}"

        # --- SRT weighting ---
        # sigma = srt_relative_error * |t| + srt_absolute_error
        rel_srt = float(self.parameters["srt_relative_error"])
        abs_srt = float(self.parameters["srt_absolute_error"])
        sigma_srt = np.clip(np.abs(rel_srt * t) + abs_srt, 1e-12, None)
        w_srt = 1.0 / sigma_srt

        self.Wd_ert = diags(w_ert, format="csr")
        self.Wd_srt = diags(w_srt, format="csr")

    def _build_smoothness_matrix(self) -> csr_matrix:
        rm = self.ert_fop.regionManager()
        Ctmp = pg.matrix.RSparseMapMatrix()
        rm.setConstraintType(1)
        rm.fillConstraints(Ctmp)
        return pg.utils.sparseMatrix2coo(Ctmp).tocsr()

    def _build_geostat_matrix(self) -> csr_matrix:
        if not hasattr(pg.matrix, "GeostatisticConstraintsMatrix"):
            raise RuntimeError("PyGIMLi GeostatisticConstraintsMatrix is unavailable.")

        corr = self.parameters["geostat_I"]
        if isinstance(corr, (tuple, list)):
            corr_arg = [float(v) for v in corr]
        else:
            corr_arg = float(corr)

        C = pg.matrix.GeostatisticConstraintsMatrix(
            mesh=self.mesh,
            I=corr_arg,
            var=float(self.parameters["geostat_var"]),
        )

        # PyGIMLi naming differs across versions:
        # older builds expose ``CM05.A`` while newer builds expose ``Cm05``.
        if hasattr(C, "CM05") and hasattr(C.CM05, "A"):
            Wm = np.asarray(C.CM05.A, dtype=float)
        elif hasattr(C, "Cm05"):
            cm05 = C.Cm05
            n_cols = int(cm05.cols())
            Wm = np.asarray(cm05.mult(np.eye(n_cols, dtype=float)), dtype=float)
        else:
            raise RuntimeError("Unable to extract geostatistical Cm05 matrix from PyGIMLi.")

        thr = float(self.parameters.get("geostat_sparse_threshold", 0.0))
        if thr > 0:
            Wm[np.abs(Wm) < thr] = 0.0
        return csr_matrix(Wm)

    def _build_regularization_matrices(self) -> None:
        mode = str(self.parameters["regularization_mode"]).lower().strip()
        if mode == "geostat":
            Wm = self._build_geostat_matrix()
        elif mode == "smoothness":
            Wm = self._build_smoothness_matrix()
        else:
            raise ValueError("regularization_mode must be 'smoothness' or 'geostat'.")

        self.Wm_ert = Wm.tocsr()
        self.Wm_srt = Wm.tocsr()

    def _init_models(self) -> None:
        n_cells = int(self.mesh.cellCount())

        rho0 = float(np.median(np.exp(self.dobs_ert.ravel())))
        mr0 = np.full(n_cells, np.log(max(rho0, 1e-12)), dtype=float)

        try:
            slowness0 = np.asarray(
                tt.createGradientModel2D(
                    self.srt_data,
                    self.mesh,
                    float(self.parameters["vTop"]),
                    float(self.parameters["vBottom"]),
                ),
                dtype=float,
            ).ravel()
        except Exception:
            centers = np.asarray(self.mesh.cellCenters(), dtype=float)
            if centers.ndim == 2 and centers.shape[1] >= 2:
                z = centers[:, 1]
            else:
                z = np.linspace(0.0, 1.0, n_cells)

            depth = z.max() - z
            span = np.ptp(depth)
            if span <= 0:
                depth_norm = np.zeros_like(depth)
            else:
                depth_norm = depth / span
            v_top = float(self.parameters["vTop"])
            v_bottom = float(self.parameters["vBottom"])
            vel = v_top + (v_bottom - v_top) * depth_norm
            slowness0 = 1.0 / np.clip(vel, 1e-12, None)

        mv0 = np.log(np.clip(slowness0, 1e-12, None))

        rho_bounds = self.parameters["ert_bounds"]
        vel_bounds = self.parameters["srt_velocity_bounds"]
        self.mr = self._clip_log_model(mr0, rho_bounds[0], rho_bounds[1], inverse=False)
        self.mv = self._clip_log_model(mv0, vel_bounds[0], vel_bounds[1], inverse=True)
        self.mr_ref = self.mr.copy()
        self.mv_ref = self.mv.copy()

    def setup(self) -> None:
        self._ensure_ert_geometric_factors()

        self.ert_manager = ert.ERTManager(self.ert_data)

        if self.mesh is None:
            self.ert_grid = self.ert_manager.createMesh(
                data=self.ert_data,
                quality=int(self.parameters["mesh_quality"]),
                paraDX=float(self.parameters["paraDX"]),
                paraMaxCellSize=float(self.parameters["paraMaxCellSize"]),
                boundaryMaxCellSize=float(self.parameters["boundaryMaxCellSize"]),
                smooth=self.parameters["smooth"],
                balanceDepth=bool(self.parameters["balanceDepth"]),
                paraBoundary=float(self.parameters["paraBoundary"]),
            )
            self.ert_manager.setMesh(self.ert_grid)
            self.mesh = self.ert_manager.fop.paraDomain
        else:
            self.ert_grid = self.mesh
            self.ert_manager.setMesh(self.mesh)

        try:
            self.mesh.setCellMarkers(np.ones((self.mesh.cellCount(),), dtype=float) * 2.0)
        except Exception:
            pass

        self.ert_fop = ert.ERTModelling()
        self.ert_fop.setData(self.ert_data)
        self.ert_fop.setMesh(self.ert_grid)

        self.srt_manager = TravelTimeManager()
        self.srt_manager.setData(self.srt_data)
        self.srt_manager.setMesh(self.mesh)
        self.srt_fop = self.srt_manager.createForwardOperator()
        self.srt_fop.setData(self.srt_data)
        self.srt_fop.setMesh(self.mesh)

        self.dobs_ert = self._build_ert_observed_log_data()
        self.dobs_srt = self._as_col(np.asarray(self.srt_data["t"], dtype=float))
        self._build_weights()
        if bool(self.parameters.get("verbose", True)):
            w_ert_diag = self.Wd_ert.diagonal()
            w_srt_diag = self.Wd_srt.diagonal()
            print(
                f"ERT observed source: {self._ert_observed_source} | "
                f"ERT error source: {self._ert_error_source}"
            )
            print(
                f"Wd_ert diag min/median/max: "
                f"{np.min(w_ert_diag):.4g}/{np.median(w_ert_diag):.4g}/{np.max(w_ert_diag):.4g}"
            )
            print(
                f"Wd_srt diag min/median/max: "
                f"{np.min(w_srt_diag):.4g}/{np.median(w_srt_diag):.4g}/{np.max(w_srt_diag):.4g}"
            )
        self._build_regularization_matrices()
        self._init_models()

        self.X = StructuralConstraint.build_local_design_matrix(self.mesh)
        self.RCM = StructuralConstraint.build_neighborhood_matrix(
            mesh=self.mesh,
            Wm=self.Wm_ert,
            source=str(self.parameters["cross_gradient_source"]),
            correlation_lengths=self.parameters["cross_gradient_corr_lengths"],
            threshold=float(self.parameters["cross_gradient_threshold"]),
            binarize=str(self.parameters["cross_gradient_mode"]).lower().strip() == "direct",
        )

        self._setup_complete = True

    def _ert_forward_and_jac(self, mr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rho = np.exp(np.asarray(mr, dtype=float).ravel())
        dr_lin = np.asarray(self.ert_fop.response(pg.Vector(rho)), dtype=float).ravel()
        dr_lin = np.clip(dr_lin, 1e-12, None)
        dr = self._as_col(np.log(dr_lin))

        self.ert_fop.createJacobian(pg.Vector(rho))
        J_lin = self._jacobian_to_numpy(self.ert_fop.jacobian())
        J = J_lin * rho.reshape(1, -1)
        J = J / dr_lin.reshape(-1, 1)
        return dr, J

    def _ert_forward(self, mr: np.ndarray) -> np.ndarray:
        rho = np.exp(np.asarray(mr, dtype=float).ravel())
        dr_lin = np.asarray(self.ert_fop.response(pg.Vector(rho)), dtype=float).ravel()
        dr_lin = np.clip(dr_lin, 1e-12, None)
        return self._as_col(np.log(dr_lin))

    def _srt_forward_and_jac(self, mv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        slowness = np.exp(np.asarray(mv, dtype=float).ravel())
        dt = self._as_col(np.asarray(self.srt_fop.response(pg.Vector(slowness)), dtype=float))

        self.srt_fop.createJacobian(pg.Vector(slowness))
        J_lin = self._jacobian_to_numpy(self.srt_fop.jacobian())
        J = J_lin * slowness.reshape(1, -1)
        return dt, J

    def _srt_forward(self, mv: np.ndarray) -> np.ndarray:
        slowness = np.exp(np.asarray(mv, dtype=float).ravel())
        return self._as_col(np.asarray(self.srt_fop.response(pg.Vector(slowness)), dtype=float))

    @staticmethod
    def _quad_norm(W, vec: np.ndarray) -> float:
        v = np.asarray(vec, dtype=float).reshape(-1, 1)
        wv = W.dot(v)
        return float(np.vdot(wv, wv).real)

    def _stack_system(
        self,
        J: np.ndarray,
        Wd: csr_matrix,
        Wm: csr_matrix,
        B: np.ndarray,
        model: np.ndarray,
        model_ref: np.ndarray,
        data_pred: np.ndarray,
        data_obs: np.ndarray,
        lambda_model: float,
        lambda_cg: float,
    ) -> Tuple[csr_matrix, np.ndarray, np.ndarray, float, float, float]:
        m = self._as_col(model)
        dm_ref = self._as_col(model - model_ref)
        dres = self._as_col(data_pred - data_obs)

        Bm = B.dot(m)
        WdJ = Wd.dot(J)
        # Use sqrt(lambda) so that the squared norm in the least-squares
        # system gives effective weight = lambda, matching ERTInversion.
        LmWm = np.sqrt(float(lambda_model)) * Wm
        LcB = np.sqrt(float(lambda_cg)) * csr_matrix(B)

        A = vstack((csr_matrix(WdJ), LmWm, LcB), format="csr")
        rhs = -np.vstack(
            (
                Wd.dot(dres),
                LmWm.dot(dm_ref),
                LcB.dot(m),
            )
        ).ravel()

        g = (
            J.T.dot(Wd.T.dot(Wd.dot(dres)))
            + LmWm.T.dot(LmWm.dot(dm_ref))
            + LcB.T.dot(LcB.dot(m))
        )

        phi_d = self._quad_norm(Wd, data_obs - data_pred)
        phi_m = self._quad_norm(LmWm, dm_ref)
        phi_cg = self._quad_norm(LcB, m)
        return A, rhs, np.asarray(g, dtype=float).reshape(-1, 1), phi_d, phi_m, phi_cg

    def _objective_value(
        self,
        data_pred: np.ndarray,
        data_obs: np.ndarray,
        Wd: csr_matrix,
        Wm: csr_matrix,
        B: np.ndarray,
        model: np.ndarray,
        model_ref: np.ndarray,
        lambda_model: float,
        lambda_cg: float,
    ) -> float:
        m = self._as_col(model)
        dm_ref = self._as_col(model - model_ref)
        LmWm = np.sqrt(float(lambda_model)) * Wm
        LcB = np.sqrt(float(lambda_cg)) * csr_matrix(B)
        phi_d = self._quad_norm(Wd, data_obs - data_pred)
        phi_m = self._quad_norm(LmWm, dm_ref)
        phi_cg = self._quad_norm(LcB, m)
        return phi_d + phi_m + phi_cg

    def _solve_linear_update(self, A: csr_matrix, rhs: np.ndarray) -> np.ndarray:
        method = str(self.parameters.get("solver", "scipy_lsmr")).lower().strip()
        maxiter = int(self.parameters.get("solver_maxiter", 400))
        tol = float(self.parameters.get("solver_tol", 1e-8))

        try:
            dm = generalized_solver(A, rhs, method=method, maxiter=maxiter, tol=tol)
            return self._as_col(dm)
        except Exception:
            # Fallback to SciPy LSMR if the chosen solver fails.
            dm = lsqr(A, rhs.ravel(), atol=tol, btol=tol, iter_lim=maxiter)[0]
            return np.asarray(dm, dtype=float).reshape(-1, 1)

    def _line_search(
        self,
        model: np.ndarray,
        dm: np.ndarray,
        grad: np.ndarray,
        objective_fn,
        clip_fn,
    ) -> np.ndarray:
        c = float(self.parameters.get("line_search_c", 1e-4))
        step = 1.0
        current_obj = float(objective_fn(model))
        directional = float(np.vdot(np.asarray(dm).ravel(), np.asarray(grad).ravel()))

        maxiter = int(self.parameters.get("line_search_maxiter", 20))
        for _ in range(maxiter):
            trial = clip_fn(np.asarray(model, dtype=float).ravel() + step * np.asarray(dm, dtype=float).ravel())
            obj_trial = float(objective_fn(trial))
            fgoal = current_obj - c * step * directional
            if obj_trial < fgoal:
                return trial
            step *= 0.5

        return clip_fn(np.asarray(model, dtype=float).ravel() + 0.1 * np.asarray(dm, dtype=float).ravel())

    def _compute_chi2(self, Wd: csr_matrix, data_obs: np.ndarray, data_pred: np.ndarray) -> float:
        phi_d = self._quad_norm(Wd, data_obs - data_pred)
        n_data = max(int(np.asarray(data_obs).size), 1)
        return float(phi_d / n_data)

    @staticmethod
    def _max_relative_residual(data_obs: np.ndarray, data_pred: np.ndarray) -> float:
        obs = np.asarray(data_obs, dtype=float).ravel()
        pred = np.asarray(data_pred, dtype=float).ravel()
        den = np.clip(np.abs(obs), 1e-12, None)
        return float(np.max(np.abs(obs - pred) / den))

    def run(self) -> JointERTSRTResult:
        if not self._setup_complete:
            self.setup()

        rho_bounds = self.parameters["ert_bounds"]
        vel_bounds = self.parameters["srt_velocity_bounds"]

        clip_r = lambda m: self._clip_log_model(m, rho_bounds[0], rho_bounds[1], inverse=False)
        clip_s = lambda m: self._clip_log_model(m, vel_bounds[0], vel_bounds[1], inverse=True)

        mr = self.mr.copy()
        mv = self.mv.copy()

        result = JointERTSRTResult(mesh=self.mesh)
        prev_chi2_ert = None
        prev_chi2_srt = None

        max_iter = int(self.parameters["max_iterations"])
        target_chi2 = float(self.parameters["target_chi2"])
        dphi_tol = float(self.parameters["convergence_tolerance"])
        verbose = bool(self.parameters.get("verbose", True))

        # Lambda cooling: start at initial values, reduce each iteration
        # (matching standalone ERTInversion / SRTInversion behaviour).
        lam_ert = float(self.parameters["lambda_ert"])
        lam_srt = float(self.parameters["lambda_srt"])
        lam_rate_ert = float(self.parameters.get("lambda_rate_ert", 0.8))
        lam_rate_srt = float(self.parameters.get("lambda_rate_srt", 0.8))
        lam_min_ert = float(self.parameters.get("lambda_min_ert", 1.0))
        lam_min_srt = float(self.parameters.get("lambda_min_srt", 1.0))

        for iteration in range(max_iter):
            lambda_cg_ert = float(self.parameters["lambda_cg_ert"])
            lambda_cg_srt = float(self.parameters["lambda_cg_srt"])
            if bool(self.parameters.get("auto_disable_cross_gradient_first_iteration", False)) and iteration == 0:
                lambda_cg_ert = 0.0
                lambda_cg_srt = 0.0

            B1, B2 = StructuralConstraint.build_linearized_cross_gradient_blocks(
                self.RCM,
                self.X,
                mr,
                mv,
                mode=str(self.parameters["cross_gradient_mode"]),
            )

            dt, Jt = self._srt_forward_and_jac(mv)
            A_s, rhs_s, g_s, phi_d_s, phi_m_s, phi_cg_s = self._stack_system(
                J=Jt,
                Wd=self.Wd_srt,
                Wm=self.Wm_srt,
                B=B2,
                model=mv,
                model_ref=self.mv_ref,
                data_pred=dt,
                data_obs=self.dobs_srt,
                lambda_model=lam_srt,
                lambda_cg=lambda_cg_srt,
            )
            d_mv = self._solve_linear_update(A_s, rhs_s)

            # Capture current lambda values for closures (avoid late-binding issues).
            _lam_srt = lam_srt
            _lam_cg_srt = lambda_cg_srt
            srt_obj = lambda trial_mv: self._objective_value(
                data_pred=self._srt_forward(trial_mv),
                data_obs=self.dobs_srt,
                Wd=self.Wd_srt,
                Wm=self.Wm_srt,
                B=B2,
                model=trial_mv,
                model_ref=self.mv_ref,
                lambda_model=_lam_srt,
                lambda_cg=_lam_cg_srt,
            )
            mv = self._line_search(mv, d_mv, g_s, srt_obj, clip_s)
            mv = clip_s(mv)

            B1, B2 = StructuralConstraint.build_linearized_cross_gradient_blocks(
                self.RCM,
                self.X,
                mr,
                mv,
                mode=str(self.parameters["cross_gradient_mode"]),
            )

            dr, Jr = self._ert_forward_and_jac(mr)
            A_r, rhs_r, g_r, phi_d_r, phi_m_r, phi_cg_r = self._stack_system(
                J=Jr,
                Wd=self.Wd_ert,
                Wm=self.Wm_ert,
                B=B1,
                model=mr,
                model_ref=self.mr_ref,
                data_pred=dr,
                data_obs=self.dobs_ert,
                lambda_model=lam_ert,
                lambda_cg=lambda_cg_ert,
            )
            d_mr = self._solve_linear_update(A_r, rhs_r)

            _lam_ert = lam_ert
            _lam_cg_ert = lambda_cg_ert
            ert_obj = lambda trial_mr: self._objective_value(
                data_pred=self._ert_forward(trial_mr),
                data_obs=self.dobs_ert,
                Wd=self.Wd_ert,
                Wm=self.Wm_ert,
                B=B1,
                model=trial_mr,
                model_ref=self.mr_ref,
                lambda_model=_lam_ert,
                lambda_cg=_lam_cg_ert,
            )
            mr = self._line_search(mr, d_mr, g_r, ert_obj, clip_r)
            mr = clip_r(mr)

            dt_eval, _ = self._srt_forward_and_jac(mv)
            dr_eval, _ = self._ert_forward_and_jac(mr)
            chi2_srt = self._compute_chi2(self.Wd_srt, self.dobs_srt, dt_eval)
            chi2_ert = self._compute_chi2(self.Wd_ert, self.dobs_ert, dr_eval)
            max_rel_srt = self._max_relative_residual(self.dobs_srt, dt_eval)
            max_rel_ert = self._max_relative_residual(self.dobs_ert, dr_eval)

            if prev_chi2_srt is None:
                dphi_srt = np.inf
            else:
                dphi_srt = abs(chi2_srt - prev_chi2_srt) / max(abs(prev_chi2_srt), 1e-12)
            if prev_chi2_ert is None:
                dphi_ert = np.inf
            else:
                dphi_ert = abs(chi2_ert - prev_chi2_ert) / max(abs(prev_chi2_ert), 1e-12)

            prev_chi2_srt = chi2_srt
            prev_chi2_ert = chi2_ert

            result.iteration_history.append(
                {
                    "iteration": iteration + 1,
                    "chi2_ert": float(chi2_ert),
                    "chi2_srt": float(chi2_srt),
                    "phi_d_ert": float(phi_d_r),
                    "phi_m_ert": float(phi_m_r),
                    "phi_cg_ert": float(phi_cg_r),
                    "phi_d_srt": float(phi_d_s),
                    "phi_m_srt": float(phi_m_s),
                    "phi_cg_srt": float(phi_cg_s),
                    "dphi_ert": float(dphi_ert),
                    "dphi_srt": float(dphi_srt),
                    "max_rel_res_ert": float(max_rel_ert),
                    "max_rel_res_srt": float(max_rel_srt),
                }
            )

            if verbose:
                if np.isfinite(dphi_srt):
                    dphi_srt_txt = f"{dphi_srt:.4f}"
                else:
                    dphi_srt_txt = "NA"
                if np.isfinite(dphi_ert):
                    dphi_ert_txt = f"{dphi_ert:.4f}"
                else:
                    dphi_ert_txt = "NA"
                print(f"-------------------Iteration: {iteration} ---------------------------")
                print(
                    f"SRT chi2: {chi2_srt:.4f} | dPhi: {dphi_srt_txt} | "
                    f"max rel. residual: {max_rel_srt:.4f}"
                )
                print(
                    f"SRT phi_d: {phi_d_s:.4e} | phi_m: {phi_m_s:.4e} | "
                    f"phi_cg: {phi_cg_s:.4e} (lambda={lam_srt:.3g}, lambda_cg={lambda_cg_srt:.3g})"
                )
                print(
                    f"ERT chi2: {chi2_ert:.4f} | dPhi: {dphi_ert_txt} | "
                    f"max rel. residual: {max_rel_ert:.4f}"
                )
                print(
                    f"ERT phi_d: {phi_d_r:.4e} | phi_m: {phi_m_r:.4e} | "
                    f"phi_cg: {phi_cg_r:.4e} (lambda={lam_ert:.3g}, lambda_cg={lambda_cg_ert:.3g})"
                )

            if chi2_ert < target_chi2 and chi2_srt < target_chi2:
                if verbose:
                    print("Converged: both chi2 below target.")
                break

            # Only allow dPhi-based stopping when chi2 is within 3x of
            # target; otherwise the inversion is stalling far from the
            # solution and should keep running up to max_iterations.
            chi2_close = (chi2_ert < 3.0 * target_chi2) and (chi2_srt < 3.0 * target_chi2)
            if chi2_close and dphi_ert < dphi_tol and dphi_srt < dphi_tol:
                if verbose:
                    print("Converged: dPhi tolerance reached near target chi2.")
                break

            # Lambda cooling — reduce regularization each iteration to let data drive the model.
            lam_ert = max(lam_min_ert, lam_ert * lam_rate_ert)
            lam_srt = max(lam_min_srt, lam_srt * lam_rate_srt)

        dr_final, _ = self._ert_forward_and_jac(mr)
        dt_final, _ = self._srt_forward_and_jac(mv)

        # --- ERT coverage (same as ERTInversion) ---
        try:
            ert_model_lin = np.exp(mr)
            dr_lin = np.asarray(self.ert_fop.response(pg.Vector(ert_model_lin)), dtype=float).ravel()
            dr_lin = np.clip(dr_lin, 1e-12, None)
            self.ert_fop.createJacobian(pg.Vector(ert_model_lin))
            covTrans = pg.core.coverageDCtrans(
                self.ert_fop.jacobian(),
                1.0 / pg.Vector(dr_lin),
                1.0 / pg.Vector(ert_model_lin),
            )
            paramSizes = np.zeros(len(ert_model_lin))
            ert_para = self.ert_fop.paraDomain
            for c in ert_para.cells():
                paramSizes[c.marker()] += c.size()
            ert_coverage = np.log10(np.asarray(covTrans, dtype=float) / paramSizes)
        except Exception:
            ert_coverage = None

        # --- SRT coverage (same as SRTInversion / TravelTimeManager) ---
        try:
            srt_slowness = np.exp(mv)
            self.srt_fop.createJacobian(pg.Vector(srt_slowness))
            if self.srt_manager is not None and hasattr(self.srt_manager, "standardizedCoverage"):
                if hasattr(self.srt_fop, "createConstraints"):
                    self.srt_fop.createConstraints()
                srt_coverage = np.asarray(self.srt_manager.standardizedCoverage(), dtype=float).ravel()
            else:
                J = self.srt_fop.jacobian()
                ray_cov = np.asarray(J.transMult(np.ones(J.rows())), dtype=float).ravel()
                Ctmp2 = pg.matrix.RSparseMapMatrix()
                self.srt_fop.regionManager().fillConstraints(Ctmp2)
                C2 = pg.utils.sparseMatrix2coo(Ctmp2).tocsr()
                srt_coverage = np.sign(np.abs(C2.T.dot(C2.dot(ray_cov))))
        except Exception:
            srt_coverage = None

        result.ert_log_resistivity = mr.copy()
        result.srt_log_slowness = mv.copy()
        result.ert_resistivity = np.exp(mr)
        result.srt_velocity = 1.0 / np.clip(np.exp(mv), 1e-12, None)
        result.ert_predicted = dr_final.ravel()
        result.srt_predicted = dt_final.ravel()
        result.ert_coverage = ert_coverage
        result.srt_coverage = srt_coverage
        result.chi2_ert = self._compute_chi2(self.Wd_ert, self.dobs_ert, dr_final)
        result.chi2_srt = self._compute_chi2(self.Wd_srt, self.dobs_srt, dt_final)
        result.meta["parameters"] = dict(self.parameters)
        result.meta["cross_gradient_mode"] = str(self.parameters["cross_gradient_mode"])
        result.meta["regularization_mode"] = str(self.parameters["regularization_mode"])
        result.meta["final_lambda_ert"] = lam_ert
        result.meta["final_lambda_srt"] = lam_srt

        self.mr = mr
        self.mv = mv
        return result
