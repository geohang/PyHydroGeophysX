"""
Time-lapse ERT inversion functionality.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pygimli as pg
import scipy.sparse as sp
from pygimli.physics import ert
from scipy.linalg import block_diag as dense_block_diag
from scipy.sparse import block_diag as sparse_block_diag
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import lsqr

from ..forward.ert_forward import ertforandjac2, ertforward2
from ..solvers.linear_solvers import (
    block_tridiagonal_cholesky_solve,
    generalized_solver,
)
from .base import InversionBase, TimeLapseInversionResult
from .temporal_weights import DEFAULT_LIMIT as DEFAULT_TEMPORAL_LIMIT
from .temporal_weights import temporal_weights

#: The solver 0.3.0 used for this inversion's model update (solvers.solver)
#: stopped these iterative least-squares methods at 2000 iterations, and
#: generalized_solver stops them at 200. 'cgls' - 0.3.0's default - is how an
#: old run is reproduced, so a method 0.3.0 offered keeps 0.3.0's cap.
_V030_ITERATIVE_METHODS = ("cgls", "lsqr", "rrlsqr", "rrls")
_V030_MAXITER = 2000


def _voltage_error(data, absolute_u_error, absolute_current: float = 0.1) -> np.ndarray:
    """Relative error of an absolute voltage error, in pyGIMLi's meaning.

    ``absoluteUError`` is a potential error in volts, as in
    ``pygimli.physics.ert.estimateError``: it adds ``absoluteUError / |U|``,
    where ``U`` is the file's measured potential, or else its resistance times
    the current - the file's own ``i``, or ``absolute_current`` amperes
    (pyGIMLi's ``absoluteCurrent``, 0.1 A) when it records none. Zero, the
    default, adds nothing.
    """
    size = int(data.size())
    value = abs(float(absolute_u_error or 0.0))
    if value == 0.0:
        return np.zeros(size)
    if data.haveData('u'):
        voltage = np.asarray(data['u'], dtype=float)
    else:
        current = (np.asarray(data['i'], dtype=float) if data.haveData('i')
                   else np.full(size, float(absolute_current)))
        if data.haveData('r'):
            resistance = np.asarray(data['r'], dtype=float)
        elif data.haveData('rhoa') and data.haveData('k'):
            resistance = np.asarray(data['rhoa'], dtype=float) / np.maximum(
                np.abs(np.asarray(data['k'], dtype=float)), 1e-12)
        else:
            raise ValueError("absoluteUError needs the potentials ('u'), or resistances "
                             "('r') or apparent resistivities with geometric factors.")
        voltage = resistance * current
    return value / np.maximum(np.abs(voltage), 1e-12)


def _sparse_temporal_difference_matrix(cell_count: int, size: int, dtype):
    """Build sparse first differences between adjacent model blocks."""
    time_difference = diags(
        (np.ones(size - 1, dtype=dtype), -np.ones(size - 1, dtype=dtype)),
        (0, 1),
        shape=(size - 1, size),
        format="csr",
        dtype=dtype,
    )
    return sp.kron(
        time_difference,
        sp.eye(cell_count, format="csr", dtype=dtype),
        format="csr",
    )


class _BlockDiagonal:
    """A block-diagonal matrix held as its blocks: the time-lapse Jacobian.

    Each survey's sensitivities fill one block and every other entry is zero.
    Assembled densely it stores N^2 blocks to hold N, and every product with it
    multiplies the zeros as well. This supports what the Gauss-Newton loop asks
    of the Jacobian - ``@`` with a vector, ``.transpose()`` - and gives the
    normal-matrix assembly the blocks themselves.
    """

    def __init__(self, blocks, transposed: bool = False):
        self.blocks = blocks
        self._transposed = transposed
        self._row_edges = np.concatenate([[0], np.cumsum([b.shape[0] for b in blocks])])
        self._col_edges = np.concatenate([[0], np.cumsum([b.shape[1] for b in blocks])])

    @property
    def shape(self):
        rows, cols = int(self._row_edges[-1]), int(self._col_edges[-1])
        return (cols, rows) if self._transposed else (rows, cols)

    def transpose(self):
        return _BlockDiagonal(self.blocks, not self._transposed)

    @property
    def T(self):
        return self.transpose()

    def __matmul__(self, vec):
        vec = np.asarray(vec)
        edges = self._row_edges if self._transposed else self._col_edges
        parts = [(block.T if self._transposed else block) @ vec[edges[k]:edges[k + 1]]
                 for k, block in enumerate(self.blocks)]
        return np.concatenate(parts, axis=0)

    dot = __matmul__


# ---------------------------------------------------------------------------
# calculate jacobian
# ---------------------------------------------------------------------------
def _calculate_jacobian(fwd_operators, model, mesh, size, as_sparse: bool = False,
                        dtype=np.float64, responses=None, with_responses: bool = False,
                        as_blocks: bool = False):
    """
    Calculate Jacobian matrix for multi-time model.

    Args:
        fwd_operators: List of forward operators
        model: Natural-log resistivity, reshaped to (cells, timesteps) in
            Fortran order so each timestep occupies a contiguous block.
        mesh: Mesh
        size: Number of timesteps
        as_sparse: Return a CSR block-diagonal Jacobian instead of a dense array.
        dtype: Floating-point dtype for responses and sensitivities.
        as_blocks: Return the dense blocks as a ``_BlockDiagonal`` instead of
            assembling them into one array.
        responses: Each operator's linear response to its block of ``model``,
            when every operator's last forward solve was exactly that block (as
            ``_calculate_forward(..., with_responses=True)`` just returned
            them). The solves are then not repeated; see ``ertforandjac2``.
        with_responses: Also return the operators' linear responses.

    Returns:
        obs: Predicted log apparent resistivity, stacked as a column vector.
        J: Jacobian matrix
        With ``with_responses``, the list of linear responses as a third value.
    """
    model_reshaped = np.reshape(model, (-1, size), order='F')
    obs = []
    linear = []

    jac_blocks = []

    for i in range(size):
        dr, Jr, response = ertforandjac2(
            fwd_operators[i], model_reshaped[:, i], mesh,
            response=None if responses is None else responses[i], with_response=True)
        linear.append(response)
        dr = dr.astype(dtype, copy=False)
        obs.append(dr)
        jac_blocks.append(csr_matrix(Jr, dtype=dtype) if as_sparse else Jr.astype(dtype, copy=False))
    
    # Stack observations
    obs_stacked = np.vstack([o.reshape(-1, 1) for o in obs]).astype(dtype, copy=False)
    
    if as_sparse:
        J = sparse_block_diag(jac_blocks, format="csr", dtype=dtype)
    elif as_blocks:
        J = _BlockDiagonal(jac_blocks)
    else:
        J = dense_block_diag(*jac_blocks).astype(dtype, copy=False)

    if with_responses:
        return obs_stacked, J, linear
    return obs_stacked, J


# ---------------------------------------------------------------------------
# calculate forward
# ---------------------------------------------------------------------------
def _calculate_forward(fwd_operators, model, mesh, size, with_responses: bool = False):
    """
    Calculate forward response for multi-time model.

    Args:
        fwd_operators: List of forward operators
        model: Model parameters (cells x timesteps)
        mesh: Mesh
        size: Number of timesteps
        with_responses: Also return each operator's linear response, which
            ``_calculate_jacobian(..., responses=)`` can build on.

    Returns:
        obs: Observed data for all timesteps, and with ``with_responses`` the
        list of linear responses as a second value.
    """
    model_reshaped = np.reshape(model, (-1, size), order='F')
    obs = []
    linear = []

    for i in range(size):
        if with_responses:
            dr, response = ertforward2(fwd_operators[i], model_reshaped[:, i], mesh,
                                       with_response=True)
            linear.append(response)
        else:
            dr = ertforward2(fwd_operators[i], model_reshaped[:, i], mesh)
        obs.append(dr)

    # Stack observations
    stacked = np.vstack([response.reshape(-1, 1) for response in obs])
    return (stacked, linear) if with_responses else stacked


# ---------------------------------------------------------------------------
# calculate forward separate
# ---------------------------------------------------------------------------
def _calculate_forward_separate(fwd_operators, model, mesh, size):
    """
    Calculate forward response for multi-time model without stacking.
    
    Args:
        fwd_operators: List of forward operators
        model: Model parameters (cells x timesteps)
        mesh: Mesh
        size: Number of timesteps
        
    Returns:
        obs: List of observed data for each timestep
    """
    model_reshaped = np.reshape(model, (-1, size), order='F')
    obs = []
    
    for i in range(size):
        dr = ertforward2(fwd_operators[i], model_reshaped[:, i], mesh)
        obs.append(dr)
    
    return obs


# ---------------------------------------------------------------------------
# Time Lapse ERTInversion
# ---------------------------------------------------------------------------
class TimeLapseERTInversion(InversionBase):
    """Time-lapse ERT inversion class."""
    
    def __init__(self, data_files: List[str], measurement_times: List[float],
                mesh: Optional[pg.Mesh] = None, **kwargs):
        """
        Initialize time-lapse ERT inversion.
        
        Args:
            data_files: List of paths to ERT data files
            measurement_times: List of measurement times
            mesh: Mesh for inversion (created if None)
            **kwargs: Additional parameters including:
                - lambda_val: Regularization parameter
                - alpha: Temporal regularization parameter
                - decay_rate: Temporal decay rate
                - method: Solver method. 'spd_cholesky' (the default) factors
                  the block-tridiagonal normal matrix block by block, an exact
                  Cholesky in O(N n^3) work and O(N n^2) memory; any other
                  method ('spd_cg', 'cgls', 'lsqr', ...) gets the whole matrix.
                - model_constraints: (min, max) model parameter bounds
                - max_iterations: Maximum iterations
                - absoluteError: Absolute resistance error floor [Ohm] (default 0.0001)
                - absoluteUError: Absolute voltage error [V], pyGIMLi's meaning:
                  adds absoluteUError / |U|, with U the file's potentials or
                  r * i, the current being the file's 'i' or absoluteCurrent
                  (default 0.1 A). Default 0, which adds nothing.
                - relativeError: Relative data error
                - lambda_rate: Lambda reduction rate
                - lambda_min: Minimum lambda value
                - save_memory: Keep the Jacobian in float32, and for a method
                  other than 'spd_cholesky' build it and the normal matrix
                  sparse rather than dense. The regularization operators are
                  sparse in either mode, and with 'spd_cholesky' the normal
                  matrix is factored block by block in float64 in either mode,
                  so with the default solver this halves the Jacobian's memory.
                - temporal_weighting: 'interval' (default) weights each adjacent
                  pair by the interval between the two surveys, so the temporal
                  constraint penalizes the rate of change; 'uniform' weights every
                  pair equally. What was applied is reported on the result, in
                  ``meta['temporal_weighting']``.
                - temporal_weight_limit: cap on how far an interval weight may
                  depart from the median interval, either way (default 10).
                - temporal_pair_weights: optional precomputed weights, one per
                  adjacent pair, including any decay. Overrides local temporal
                  weighting so windows can share a campaign-wide normalization.
                - zones: a-priori resistivity zones, a list of dicts with a
                  ``polygon`` in mesh coordinates, a ``resistivity`` and an
                  optional ``fixed`` (see ``ert_zones``). Every survey starts
                  from its a-priori model - the zones on that survey's median
                  apparent resistivity - the spatial constraint acts on the
                  departure from it, and fixed zones hold in every survey.
        """
        # Load ERT data
        self.data_files = data_files
        self.measurement_times = np.array(measurement_times)
        
        # Validate input
        if len(data_files) != len(measurement_times):
            raise ValueError("Number of data files must match number of measurement times")
        
        # Load first dataset to initialize base class
        data = ert.load(data_files[0])
        
        # Call parent initializer with first dataset
        super().__init__(data, mesh, **kwargs)
        
        # Set time-lapse specific default parameters
        tl_defaults = {
            'lambda_val': 100.0,
            'alpha': 10.0,
            'decay_rate': 0.0,
            # Weight each adjacent pair by the interval between the two surveys,
            # so the temporal constraint penalizes the rate of change rather than
            # the raw difference. Normalized by the median interval, so an evenly
            # sampled series is unaffected and 'alpha' keeps its meaning; pass
            # 'uniform' to reproduce a run from before this existed.
            'temporal_weighting': 'interval',
            'temporal_weight_limit': DEFAULT_TEMPORAL_LIMIT,
            # 'H' below is the Gauss-Newton normal matrix, which is square and
            # symmetric positive definite, so it wants a symmetric solver. The
            # old 'cgls' default is a least-squares method: on this matrix it
            # solves 'H^T H d = H^T (-g)' and so works with the square of the
            # condition number, which at a realistic 4D size leaves the update
            # far short of the Newton step and makes the result look insensitive
            # to lambda. Pass method='cgls' to reproduce a run from before this
            # became the default.
            'method': 'spd_cholesky',
            'absoluteError': 0.0001,
            'relativeError': 0.05,
            # Cooling is off by default. It used to be 0.8, which moved lambda on
            # every iteration and left the final chi2 attributable to no single
            # value; the caller relaxes lambda between converged runs instead.
            'lambda_rate': 1.0,
            'lambda_min': 1.0,
            'inversion_type': 'L2',  # 'L1', 'L2', or 'L1L2'
            'model_constraints':(0.0001,10000.0),  # min and max resistivity
            'save_memory': False,  # float32 Jacobian; see the docstring
            # Stopping. Both were hard-coded (chi2 < 1.5, dPhi < 0.01 after 5
            # iterations); a lambda sweep needs them configurable so a flattened
            # misfit means the lambda is spent, not that the budget ran out.
            'target_chi_squared': 1.0,
            'convergence_tolerance': 0.01,
            'min_iterations': 5,
            'verbose': True,
        }
        
        # Update parameters with time-lapse defaults
        for key, value in tl_defaults.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        self.use_sparse = bool(self.parameters.get('save_memory', False))
        self.dtype = np.float32 if self.use_sparse else np.float64
        
        # Number of timesteps
        self.size = len(data_files)
        
        # Initialize internal variables
        self.fwd_operators = []
        self.datasets = []
        self.rhos1 = None
        self.Wd = None
        self.Wm = None
        self.Wt = None
        self.temporal_weight_report: Dict[str, Any] = {}
        self._zone_prior = None  # the zones on the parameter mesh, once built

    def zone_prior(self):
        """The a-priori zones on this inversion's parameter mesh, or None.

        Every survey shares the mesh, so one prior serves them all. See
        ``ert_zones``.
        """
        zones = self.parameters.get('zones')
        if not zones:
            return None
        if self._zone_prior is None:
            if not self.fwd_operators:
                self.setup()
            from .ert_zones import zone_prior

            self._zone_prior = zone_prior(
                self.fwd_operators[0].paraDomain, zones,
                bounds=self.parameters['model_constraints'])
        return self._zone_prior

    def setup(self):
        """Set up time-lapse ERT inversion (load data, create operators, matrices, etc.)"""
        # Create mesh if not provided
        if self.mesh is None:
            ert_manager = ert.ERTManager(self.data)
            self.mesh = ert_manager.createMesh(data=self.data, quality=34)
        
        # Load all datasets and process
        rhos = []
        dataerr = []
        
        for i, fname in enumerate(self.data_files):
            # Load data
            dataert = ert.load(fname)
            self.datasets.append(dataert)
            
            # Handle geometric factors
            if np.all(dataert['k'] == 0.0):
                dataert['k'] = ert.createGeometricFactors(dataert, numerical=True)
            k = dataert['k'].array()
            
            # Get apparent resistivity
            if np.all(dataert['rhoa']) != 0.0:
                rhos.append(dataert['rhoa'].array())
            elif dataert.haveData('r'):
                rhos.append(dataert['r'].array() * k)
            else:
                # pyGIMLi lists 'r' even when the file has none; r * k would
                # replace this survey's rhoa with zeros, and log(0) with -inf.
                raise ValueError(
                    f"Dataset {fname}: some readings have zero apparent "
                    "resistivity and the file has no resistances to rebuild them "
                    "from. Filter those readings out before the time-lapse run.")
            
            # Get or estimate data errors
            if np.all(dataert['err']) != 0.0:
                dataerr.append(np.clip(dataert['err'].array(), 0.01, 0.50))
            else:
                # Seb's per-measurement formula: err_i = relativeError + absoluteError / |r_i|
                abs_e = float(self.parameters['absoluteError'])
                rel_e = float(self.parameters['relativeError'])
                # haveData, not "in dataMap": a zero-filled 'r' gave |r| = 0 and
                # put every reading of the survey at the 50 % error cap.
                if dataert.haveData('r'):
                    r_abs = np.abs(dataert['r'].array())
                elif dataert.haveData('k'):
                    r_abs = np.abs(dataert['rhoa'].array()) / np.maximum(
                        np.abs(dataert['k'].array()), 1e-10)
                else:
                    raise RuntimeError(
                        f"Dataset {fname}: cannot estimate error without 'r' or 'k'.")
                err_i = rel_e + abs_e / np.maximum(r_abs, 1e-10)
                # absoluteUError is volts, not ohms, and used to be ignored here
                # while the ADTLERT windowed path read it as an ohm floor.
                err_i = err_i + _voltage_error(
                    dataert, self.parameters.get('absoluteUError', 0.0),
                    float(self.parameters.get('absoluteCurrent', 0.1)))
                dataerr.append(np.clip(err_i, 0.01, 0.50))
            
            # Create forward operator
            fwd_operator = ert.ERTModelling()
            fwd_operator.setData(dataert)
            fwd_operator.setMesh(self.mesh)
            self.fwd_operators.append(fwd_operator)
        
        # Stack all data
        rhos_temp = np.concatenate(rhos)
        
        rhos_temp = rhos_temp.reshape((-1, 1)).astype(self.dtype, copy=False)
        self.rhos1 = np.log(rhos_temp).astype(self.dtype, copy=False)

        del rhos_temp  # Delete after use
        del rhos  # Delete after use

        # Data error and weighting matrix
        err_temp = np.concatenate(dataerr)
        data_weights = (1.0 / np.log(err_temp + 1)).astype(self.dtype, copy=False)
        # Wd is diagonal by construction, so keep it diagonal on both paths.
        # The dense branch used to build a D-by-D array to hold D numbers and
        # then square it with a full matmul. At D = 6000 that is 0.27 GB and
        # 1.7 s, and it leaves 'Jr.T @ Wd_sq @ Jr' costing O(D^2 P) every
        # Gauss-Newton iteration rather than O(D P): measured 0.512 s against
        # 0.115 s at D = 6000, P = 1200, for an identical result. At a realistic
        # 4D size, D = 20000, the array alone would be 3.0 GB. Every use of Wd
        # below is .dot, .T or @, all of which a scipy diagonal supports, so no
        # call site changes. Rd, Rs and Rt were already built with diags().
        self.Wd = diags(data_weights, dtype=self.dtype)
        self.Wd_sq = self.Wd.multiply(self.Wd).astype(self.dtype, copy=False)
        
        # Create model regularization matrix
        rm = self.fwd_operators[0].regionManager()
        Ctmp = pg.matrix.RSparseMapMatrix()
        rm.setConstraintType(1)
        rm.fillConstraints(Ctmp)
        Wm_r = pg.utils.sparseMatrix2coo(Ctmp)
        cw = rm.constraintWeights().array().astype(self.dtype, copy=False)
        Wm_r = diags(cw).dot(Wm_r)
        # One survey's spatial operator, kept sparse for the block-wise normal
        # matrix; its Gram matrix is built on first use.
        self._Wm_block = sp.csr_matrix(Wm_r, dtype=self.dtype)
        self._WmTWm_block = None

        # Sparse in either memory mode. Wm and Wt only ever multiply vectors -
        # the normal matrix is built from the blocks above - and the dense mode
        # used to store them dense: (N C) x (N n) and (N - 1) n x N n, almost
        # all zeros. At the 15000 unknowns below which the studio keeps the
        # dense mode that was 5.6 GB, and every product with them paid for it.
        self.Wm = sparse_block_diag([self._Wm_block] * self.size, format="csr", dtype=self.dtype)

        # Create temporal regularization matrix. One weight per adjacent pair,
        # repeated over the cells of that block row. Weighting by the interval
        # turns the penalty from one on the raw difference between surveys into
        # one on the rate of change, which is the only form that means the same
        # thing when the sampling is irregular.
        cell_count = self.fwd_operators[0].paraDomain.cellCount()
        supplied_weights = self.parameters.get('temporal_pair_weights')
        if supplied_weights is None:
            pair_weights, self.temporal_weight_report = temporal_weights(
                self.measurement_times,
                mode=str(self.parameters.get('temporal_weighting', 'interval')),
                limit=self.parameters.get('temporal_weight_limit', DEFAULT_TEMPORAL_LIMIT),
                decay_rate=float(self.parameters.get('decay_rate', 0.0)),
            )
        else:
            pair_weights = np.asarray(supplied_weights, dtype=float)
            if (pair_weights.shape != (self.size - 1,)
                    or not np.isfinite(pair_weights).all()
                    or np.any(pair_weights < 0)):
                raise ValueError("temporal_pair_weights must contain one finite, "
                                 "nonnegative weight per adjacent survey pair")
            self.temporal_weight_report = dict(self.parameters.get(
                '_temporal_weight_report', {
                    'mode': 'explicit', 'applied': True,
                    'note': 'Using precomputed temporal pair weights.',
                }))
        temporal_weights_full = np.repeat(pair_weights, cell_count).astype(
            self.dtype, copy=False)
        self._temporal_row_weights = temporal_weights_full
        Wt = _sparse_temporal_difference_matrix(cell_count, self.size, self.dtype)
        self.Wt = diags(temporal_weights_full, dtype=self.dtype).dot(Wt).tocsr()
    
    def _normal_blocks(self, jacobian, data_weights, Lambda, alpha,
                       model_weights=None, temporal_weights=None, shift=0.0):
        """The Gauss-Newton normal matrix as its blocks: ``(diagonal, couplings)``.

        ``H = J^T D J + Lambda Wm^T M Wm + alpha Wt^T T Wt + shift I``, with ``D``,
        ``M`` and ``T`` diagonal (their diagonals given; ``None`` is the
        identity). ``J`` is block-diagonal, one survey per block, and ``Wm`` is
        the same spatial operator on every block, so those two terms fill only
        the diagonal blocks. ``Wt`` differences adjacent surveys, so its term
        adds a diagonal to each diagonal block and couples each pair of
        neighbours through a diagonal matrix. ``H`` is therefore block
        tridiagonal: ``diagonal[k]`` is ``H[k, k]`` and ``couplings[k]`` the
        diagonal of ``H[k + 1, k]``. Blocks are float64 whatever ``self.dtype``,
        because the factorization of a normal matrix needs the precision.
        """
        blocks = jacobian.blocks
        n_cells = blocks[0].shape[1]
        spatial = self._Wm_block
        n_rows_m = spatial.shape[0]
        if model_weights is None and self._WmTWm_block is None:
            self._WmTWm_block = (spatial.T @ spatial).toarray().astype(np.float64, copy=False)
        # Wt = diag(w) D with D = [I -I] per adjacent pair, so
        # Wt^T T Wt = D^T diag(w^2 t) D.
        pair = np.asarray(self._temporal_row_weights, dtype=np.float64) ** 2
        if temporal_weights is not None:
            pair = pair * np.asarray(temporal_weights, dtype=np.float64)
        couplings = [alpha * pair[k * n_cells:(k + 1) * n_cells] for k in range(len(blocks) - 1)]
        diagonal, idx, rows = [], np.arange(n_cells), 0
        for k, block in enumerate(blocks):
            J = np.asarray(block, dtype=np.float64)
            weights = np.asarray(data_weights[rows:rows + J.shape[0]], dtype=np.float64)
            rows += J.shape[0]
            A = J.T @ (weights.reshape(-1, 1) * J)
            if model_weights is None:
                A += Lambda * self._WmTWm_block
            else:
                m_k = np.asarray(model_weights[k * n_rows_m:(k + 1) * n_rows_m], dtype=np.float64)
                A += Lambda * (spatial.T @ diags(m_k) @ spatial).toarray()
            if k:
                A[idx, idx] += couplings[k - 1]
            if k < len(blocks) - 1:
                A[idx, idx] += couplings[k]
            if shift:
                A[idx, idx] += shift
            diagonal.append(A)
        return diagonal, [-v for v in couplings]

    def _block_normal_matrix(self, jacobian, data_weights, Lambda, alpha,
                             model_weights=None, temporal_weights=None, shift=0.0):
        """The dense Gauss-Newton normal matrix, assembled from its blocks.

        For a solver other than ``spd_cholesky``, which needs the whole matrix.
        Multiplying the full block-diagonal arrays did this arithmetic plus
        N^2 - N blocks of zeros, and the constant spatial and temporal products
        again every iteration: a third of a four-survey run, growing as N^3.
        The matrix is the same one; only the order of the sums differs.
        """
        diagonal, couplings = self._normal_blocks(jacobian, data_weights, Lambda, alpha,
                                                  model_weights, temporal_weights, shift)
        n_cells = diagonal[0].shape[0]
        H = np.zeros((n_cells * len(diagonal),) * 2, dtype=self.dtype)
        idx = np.arange(n_cells)
        for k, A in enumerate(diagonal):
            cells = slice(k * n_cells, (k + 1) * n_cells)
            H[cells, cells] = A
        for k, e in enumerate(couplings):
            here, there = k * n_cells + idx, (k + 1) * n_cells + idx
            H[there, here] = e
            H[here, there] = e
        return H

    def run(self, initial_model: Optional[np.ndarray] = None) -> TimeLapseInversionResult:
        """
        Run time-lapse ERT inversion.
        
        Args:
            initial_model: Initial model parameters (if None, a homogeneous model is used)
            
        Returns:
            TimeLapseInversionResult with inversion results
        """
        # Make sure setup has been called
        if not self.fwd_operators:
            self.setup()
        
        use_sparse = self.use_sparse
        # The default solver factors the block-tridiagonal normal matrix block
        # by block, in either memory mode; any other method needs it whole. In
        # low-memory mode this replaced sparse products of the Jacobian's dense
        # blocks and a SuperLU solve, 70 % of a ten-survey run.
        block_cholesky = str(self.parameters.get('method', '')).lower().strip() == 'spd_cholesky'

        def _as_col(vec):
            arr = np.asarray(vec)
            if arr.dtype != self.dtype:
                arr = arr.astype(self.dtype, copy=False)
            return arr.reshape(-1, 1) if arr.ndim == 1 else arr

        def _matvec(mat, vec):
            res = mat.dot(_as_col(vec)) if sp.issparse(mat) else mat @ _as_col(vec)
            res_arr = np.asarray(res)
            return res_arr if res_arr.ndim > 1 else res_arr.reshape(-1, 1)

        def _ttm(mat, vec):
            return _matvec(mat.transpose(), _matvec(mat, vec))

        def _apply_data_weights(weight_mat, vec):
            weighted = _matvec(self.Wd, vec)
            weighted = _matvec(weight_mat, weighted)
            return _matvec(self.Wd.transpose(), weighted)

        def _quad(weighted_vec, vec):
            return (_as_col(vec).T @ _as_col(weighted_vec)).item()
        
        # Initialize result object
        result = TimeLapseInversionResult()
        result.timesteps = self.measurement_times
        
        # Set up initial model if not provided
        cell_count = self.fwd_operators[0].paraDomain.cellCount()

        # The median resistivity of each time step, taken from the apparent
        # resistivities the inversion fits (self.rhos1, stacked survey by
        # survey, rebuilt from r * k where needed): the homogeneous start model,
        # and the background of the a-priori model when there are zones. This
        # used to test hasattr(dataset, 'rhoa'), which is always False for a
        # DataContainerERT (its tokens are not attributes), so every survey
        # started from 100 ohm-m.
        sizes = [int(d.size()) for d in self.datasets]
        per_survey = np.split(np.asarray(self.rhos1, dtype=float).ravel(),
                              np.cumsum(sizes)[:-1])
        initial_rhos = []
        for log_rhoa in per_survey:
            rhoa_i = np.exp(log_rhoa[np.isfinite(log_rhoa)])
            if rhoa_i.size:
                initial_rhos.append(float(np.median(rhoa_i)))
            else:
                # Use default value if no apparent resistivity data
                initial_rhos.append(100.0)

        if initial_model is None:
            mr = np.log(np.repeat(initial_rhos, cell_count).reshape(-1, 1)).astype(self.dtype, copy=False)
        else:
            # Use provided initial model
            if initial_model.shape != (cell_count, self.size):
                raise ValueError(f"Initial model should have shape ({cell_count}, {self.size})")

            # Flatten in column-major order and log-transform
            mr = np.log(initial_model.flatten(order='F').reshape(-1, 1)).astype(self.dtype, copy=False)

        # A-priori zones: every survey starts from its a-priori model, the
        # spatial constraint acts on the departure from it - so the contrast at
        # a zone's edge costs nothing unless the data argue against it - and a
        # fixed zone keeps its value in every survey, whatever the run is
        # continued from. The model vector holds the surveys one after another,
        # so a per-cell mask repeats once per survey.
        prior = self.zone_prior()
        prior_mr = fixed_cells = fixed_rows = None
        if prior is not None:
            prior_mr = np.log(np.concatenate(
                [prior.with_background(rho) for rho in initial_rhos])).reshape(-1, 1).astype(
                    self.dtype, copy=False)
            if initial_model is None:
                mr = prior_mr.copy()
            else:
                held = np.tile(prior.fixed, self.size)
                mr[held] = prior_mr[held]
            if prior.any_fixed:
                fixed_cells = np.flatnonzero(prior.fixed)
                fixed_rows = np.flatnonzero(np.tile(prior.fixed, self.size))

        def _spatial(model):
            """What the spatial constraint acts on: the departure from the prior."""
            return model if prior_mr is None else model - prior_mr

        def _hold_fixed(system, rhs):
            """Decouple the fixed cells from a normal system in place.

            Their rows and columns become those of the identity and their
            right-hand side zero, so their update is exactly zero and the free
            cells solve the system with the fixed ones left out. The matrix stays
            symmetric positive definite, which the Cholesky paths need.
            """
            rhs[fixed_rows] = 0.0
            if isinstance(system, tuple):              # (diagonal blocks, couplings)
                for block in system[0]:
                    block[fixed_cells, :] = 0.0
                    block[:, fixed_cells] = 0.0
                    block[fixed_cells, fixed_cells] = 1.0
                for coupling in system[1]:
                    coupling[fixed_cells] = 0.0
                return system
            if sp.issparse(system):
                keep = np.ones(system.shape[0], dtype=system.dtype)
                keep[fixed_rows] = 0.0
                return (diags(keep) @ system @ diags(keep)
                        + diags(1.0 - keep)).tocsr()
            system[fixed_rows, :] = 0.0
            system[:, fixed_rows] = 0.0
            system[fixed_rows, fixed_rows] = 1.0
            return system

        # Regularization parameters
        Lambda = self.parameters['lambda_val']
        alpha = self.parameters['alpha']
        
        # Model constraints
        min_mr, max_mr = self.parameters['model_constraints']
        min_mr = np.log(min_mr)
        max_mr = np.log(max_mr)

        target_chi2 = float(self.parameters.get('target_chi_squared', 1.0))
        dphi_tol = float(self.parameters.get('convergence_tolerance', 0.01))
        min_iterations = int(self.parameters.get('min_iterations', 5))
        verbose = bool(self.parameters.get('verbose', True))
        stop_reason = 'iteration_cap'

        if verbose:
            print(min_mr, max_mr)

        # Track errors for each iteration
        Err_tot = []
        chi2_old = np.inf
        # True once mr has changed since Err_tot last scored it.
        model_moved = False
        # What the forward operators last solved: (log model, linear responses,
        # whether the Jacobians followed). The line search solves every survey
        # for the model it accepts, and each iteration used to solve them all
        # again before building the Jacobians. pyGIMLi's createJacobian reads
        # the potentials of an operator's last response() without checking the
        # model, so the solves are reused only while this record holds.
        solved = None

        # Choose inversion type
        inversion_type = self.parameters['inversion_type'].upper()
        if inversion_type not in ['L1', 'L2', 'L1L2']:
            if verbose:
                print(f"Invalid inversion type {inversion_type}, defaulting to L2")
            inversion_type = 'L2'
        
        # L1-specific parameters
        if inversion_type in ['L1', 'L1L2']:
            l1_epsilon = 1e-4
            irls_iter_max = 5 if inversion_type == 'L1' else 8
            irls_tol = 1e-3 if inversion_type == 'L1' else 1e-2
            threshold_c = 2.0  # For L1L2 hybrid
        
        # IRLS iterations for L1-norm
        for irls_iter in range(1 if inversion_type == 'L2' else irls_iter_max):
            if inversion_type in ['L1', 'L1L2'] and verbose:
                print(f'------------------- IRLS Iteration: {irls_iter + 1} ---------------------------')

            # Main inversion loop
            for nn in range(self.parameters['max_iterations']):
                if verbose:
                    print(f'-------------------ERT Iteration: {nn} ---------------------------')
                
                # Forward modeling and Jacobian computation; the forward solves
                # are skipped when the operators already hold this model's.
                held = solved is not None and np.array_equal(solved[0], mr)
                dr, Jr, linear = _calculate_jacobian(
                    self.fwd_operators, mr, self.mesh, self.size,
                    as_sparse=use_sparse and not block_cholesky, dtype=self.dtype,
                    responses=solved[1] if held else None, with_responses=True,
                    as_blocks=block_cholesky or not use_sparse
                )
                solved = (mr, linear, True)
                dr = dr.reshape(-1, 1)
                
                # Data misfit calculation
                dataerror_ert = _as_col(self.rhos1 - dr)
                
                # Handle different norms
                if inversion_type == 'L2':
                    # Standard L2 norm
                    data_weighted = _matvec(self.Wd_sq, dataerror_ert)
                    fdert = _quad(data_weighted, dataerror_ert)
                    
                    # Gradient computation with memory management
                    grad_data = -_matvec(Jr.transpose(), data_weighted)
                    
                    spatial = _spatial(mr)
                    model_term = _ttm(self.Wm, spatial)
                    fmert = Lambda * _quad(model_term, spatial)
                    grad_model = Lambda * model_term
                    
                    temp_term = _ttm(self.Wt, mr)
                    ftert = alpha * _quad(temp_term, mr)
                    grad_temporal = alpha * temp_term
                        


                    
                elif inversion_type == 'L1':
                    # L1 norm using IRLS
                    Rd = diags(1.0 / np.sqrt(dataerror_ert.flatten()**2 + l1_epsilon))

                    model_diff = _matvec(self.Wm, _spatial(mr))
                    Rs = diags(1.0 / np.sqrt(model_diff.flatten()**2 + l1_epsilon))
                    
                    temp_diff = _matvec(self.Wt, mr)
                    Rt = diags(1.0 / np.sqrt(temp_diff.flatten()**2 + l1_epsilon))
                    
                    # Objective functions with weighted L1 norms
                    data_weighted = _apply_data_weights(Rd, dataerror_ert)
                    fdert = _quad(data_weighted, dataerror_ert)
                    
                    model_weighted = _matvec(Rs, model_diff)
                    fmert = Lambda * _quad(model_weighted, model_diff)
                    
                    temp_weighted = _matvec(Rt, temp_diff)
                    ftert = alpha * _quad(temp_weighted, temp_diff)
                    
                    # Gradient computation
                    grad_data = -_matvec(Jr.transpose(), data_weighted)
                    grad_model = Lambda * _matvec(self.Wm.transpose(), model_weighted)
                    grad_temporal = alpha * _matvec(self.Wt.transpose(), temp_weighted)
                    
                else:  # L1L2 hybrid
                    # Compute hybrid L1-L2 weights for data misfit
                    effective_epsilon = l1_epsilon * (1 + 10*np.exp(-nn/5))
                    norm_values = (
                        np.abs(dataerror_ert.flatten())
                        / np.sqrt(effective_epsilon)
                    )
                    data_weights = np.ones_like(norm_values)
                    outlier_mask = norm_values > threshold_c
                    data_weights[outlier_mask] = (
                        threshold_c / norm_values[outlier_mask]
                    )
                    
                    Rd = diags(data_weights)
                    
                    # Model and temporal weights (pure L1)
                    model_diff = _matvec(self.Wm, _spatial(mr))
                    model_weights = 1.0 / np.sqrt(model_diff.flatten()**2 + l1_epsilon)
                    model_weights = np.maximum(model_weights, 1e-10)
                    Rs = diags(model_weights)
                    
                    temp_diff = _matvec(self.Wt, mr)
                    temp_weights = 1.0 / np.sqrt(temp_diff.flatten()**2 + l1_epsilon)
                    temp_weights = np.maximum(temp_weights, 1e-10)
                    Rt = diags(temp_weights)
                    
                    # Objective functions
                    data_weighted = _apply_data_weights(Rd, dataerror_ert)
                    fdert = _quad(data_weighted, dataerror_ert)
                    
                    model_weighted = _matvec(Rs, model_diff)
                    fmert = Lambda * _quad(model_weighted, model_diff)
                    
                    temp_weighted = _matvec(Rt, temp_diff)
                    ftert = alpha * _quad(temp_weighted, temp_diff)
                    
                    # Gradient computation
                    grad_data = -_matvec(Jr.transpose(), data_weighted)
                    grad_model = Lambda * _matvec(self.Wm.transpose(), model_weighted)
                    grad_temporal = alpha * _matvec(self.Wt.transpose(), temp_weighted)
                
                # Total gradient
                gc_r = grad_data + grad_model + grad_temporal
                
                # Total objective function
                ftot = fdert + fmert + ftert
                
                # Compute chi-squared and check convergence
                chi2_ert = _quad(_matvec(self.Wd_sq, dataerror_ert), dataerror_ert) / len(dr)
                dPhi = abs(chi2_ert - chi2_old) / chi2_old if nn > 0 else 1.0
                chi2_old = chi2_ert
                
                if verbose:
                    print(f'ERT chi2: {chi2_ert}')
                    print(f'dPhi: {dPhi}')
                    print(f'ERTphi_d: {fdert}, ERTphi_m: {fmert}, ERTphi_t: {ftert}')

                # Store iteration data
                Err_tot.append([chi2_ert, fmert, ftert])
                model_moved = False
                progress_callback = self.parameters.get('progress_callback')
                if callable(progress_callback):
                    progress_callback({
                        'event': 'timelapse_iteration_done',
                        'iteration': int(nn + 1),
                        'max_iterations': int(self.parameters['max_iterations']),
                        'irls_iteration': int(irls_iter + 1),
                        'irls_iterations': int(
                            1 if inversion_type == 'L2' else irls_iter_max
                        ),
                        'chi2': float(chi2_ert),
                        'dphi': float(dPhi),
                    })

                # Check for convergence
                if chi2_ert < target_chi2:
                    stop_reason = 'target'
                    if verbose:
                        print(f"Convergence reached at iteration {nn}")
                    break
                if dPhi < dphi_tol and nn > min_iterations:
                    stop_reason = 'plateau'
                    if verbose:
                        print(f"Convergence reached at iteration {nn}")
                    break
                
                if block_cholesky:
                    # spd_cholesky on the block-tridiagonal matrix as it is: the same
                    # exact factorization, block by block, without forming (N n)^2
                    # entries or multiplying the zeros around the Jacobian's blocks.
                    if inversion_type == 'L2':
                        weights = {'data_weights': self.Wd_sq.diagonal()}
                    else:
                        weights = {
                            'data_weights': self.Wd.diagonal() ** 2 * Rd.diagonal(),
                            'model_weights': Rs.diagonal(),
                            'temporal_weights': Rt.diagonal(),
                            'shift': l1_epsilon if inversion_type == 'L1L2' else 0.0,
                        }
                    diagonal_blocks, couplings = self._normal_blocks(
                        Jr, Lambda=Lambda, alpha=alpha, **weights)
                    del Jr
                    rhs = -gc_r
                    if fixed_rows is not None:
                        _hold_fixed((diagonal_blocks, couplings), rhs)
                    d_mr = block_tridiagonal_cholesky_solve(
                        diagonal_blocks, couplings, rhs, overwrite=True)
                    del diagonal_blocks
                else:
                    # Compute Hessian (or approximation)
                    if inversion_type == 'L2':
                        # Standard Gauss-Newton Hessian
                        if use_sparse:
                            H = (Jr.transpose().dot(self.Wd_sq.dot(Jr)) + 
                                 Lambda * self.Wm.transpose().dot(self.Wm) + 
                                 alpha * self.Wt.transpose().dot(self.Wt))
                        else:
                            H = self._block_normal_matrix(Jr, self.Wd_sq.diagonal(), Lambda, alpha)
                    elif inversion_type == 'L1':
                        # IRLS modified Hessian
                        if use_sparse:
                            weighted_J = Rd.dot(self.Wd.dot(Jr))
                            weighted_J = self.Wd.transpose().dot(weighted_J)
                            H = (Jr.transpose().dot(weighted_J) + 
                                 Lambda * self.Wm.transpose().dot(Rs.dot(self.Wm)) + 
                                 alpha * self.Wt.transpose().dot(Rt.dot(self.Wt)))
                        else:
                            H = self._block_normal_matrix(
                                Jr, self.Wd.diagonal() ** 2 * Rd.diagonal(), Lambda, alpha,
                                model_weights=Rs.diagonal(), temporal_weights=Rt.diagonal())
                    else:  # L1L2
                        # Hybrid Hessian with damping
                        if use_sparse:
                            weighted_J = Rd.dot(self.Wd.dot(Jr))
                            weighted_J = self.Wd.transpose().dot(weighted_J)
                            H = (Jr.transpose().dot(weighted_J) + 
                                 Lambda * self.Wm.transpose().dot(Rs.dot(self.Wm)) + 
                                 alpha * self.Wt.transpose().dot(Rt.dot(self.Wt)) + 
                                 l1_epsilon * sp.eye(Jr.shape[1], format='csr', dtype=self.dtype))
                        else:
                            H = self._block_normal_matrix(
                                Jr, self.Wd.diagonal() ** 2 * Rd.diagonal(), Lambda, alpha,
                                model_weights=Rs.diagonal(), temporal_weights=Rt.diagonal(),
                                shift=l1_epsilon)
                
                    # After using Jr for gradient computation
                    del Jr  # No longer needed

                    rhs = -gc_r
                    if fixed_rows is not None:
                        H = _hold_fixed(H, rhs)

                    # Solve for model update. overwrite_a lets 'spd_cholesky'
                    # factor in H's own buffer, which for a dense 4D normal matrix
                    # is the difference between one working copy and none; nothing
                    # reads H after this call. It is ignored by the other methods.
                    method = self.parameters['method']
                    if str(method).lower().strip() in _V030_ITERATIVE_METHODS:
                        solver_options = {'maxiter': _V030_MAXITER}
                    else:
                        solver_options = {}
                    d_mr = generalized_solver(
                        H, rhs,
                        method=method,
                        use_gpu=self.parameters.get('use_gpu', False),
                        parallel=self.parameters.get('parallel', False),
                        n_jobs=self.parameters.get('n_jobs', -1),
                        overwrite_a=True,
                        **solver_options,
                    )
                    d_mr = d_mr.reshape(-1, 1)
                    del H  # consumed by the solve, and rebuilt next iteration
                
                # Line search
                mu_LS = 1.0
                success = False
                best_mr = mr.copy()
                best_f = ftot
                
                # One line search for every norm. L1L2 used to take the full step
                # unchecked; it still does whenever that step lowers the objective,
                # which is the first trial. Its IRLS weights are those of this
                # iteration, so the L1 form of the objective below is its own.
                for iarm in range(20):
                    mr1 = mr + mu_LS * d_mr
                    mr1 = np.clip(mr1, min_mr, max_mr)
                    
                    try:
                        dr_new, linear = _calculate_forward(
                            self.fwd_operators, mr1, self.mesh, self.size, with_responses=True)
                        solved = (mr1, linear, False)
                        dr_new = dr_new.reshape(-1, 1)
                        dataerror_new = _as_col(self.rhos1 - dr_new)
                        
                        # Compute new objective function
                        if inversion_type == 'L2':
                            data_weighted_new = _matvec(self.Wd_sq, dataerror_new)
                            fdert_new = _quad(data_weighted_new, dataerror_new)
                            spatial_new = _spatial(mr1)
                            model_term_new = _ttm(self.Wm, spatial_new)
                            fmert_new = Lambda * _quad(model_term_new, spatial_new)
                            temp_term_new = _ttm(self.Wt, mr1)
                            ftert_new = alpha * _quad(temp_term_new, mr1)
                        else:  # L1 and L1L2, with this iteration's IRLS weights
                            data_weighted_new = _apply_data_weights(Rd, dataerror_new)
                            fdert_new = _quad(data_weighted_new, dataerror_new)
                            model_diff_new = _matvec(self.Wm, _spatial(mr1))
                            model_weighted_new = _matvec(Rs, model_diff_new)
                            fmert_new = Lambda * _quad(model_weighted_new, model_diff_new)
                            temp_diff_new = _matvec(self.Wt, mr1)
                            temp_weighted_new = _matvec(Rt, temp_diff_new)
                            ftert_new = alpha * _quad(temp_weighted_new, temp_diff_new)
                        
                        ftot_new = fdert_new + fmert_new + ftert_new
                        
                        if ftot_new < ftot:
                            best_f = ftot_new
                            best_mr = mr1.copy()
                            success = True
                            break
                            
                    except Exception as e:
                        if verbose:
                            print(f"Line search iteration {iarm} failed: {str(e)}")

                    mu_LS *= 0.5
                
                # Update model
                if success:
                    mr = best_mr
                    if Lambda > self.parameters['lambda_min']:
                        Lambda *= self.parameters['lambda_rate']
                model_moved = success
                if not success:
                    # No trial lowered the objective. This used to take a fixed
                    # 0.01 step down the gradient regardless, which could raise
                    # it; the model stays instead. With the model and weights
                    # unchanged, the next L2 or L1 iteration would repeat this one
                    # exactly - Jacobians and all - so the misfit has stopped
                    # changing and the loop ends. L1L2 re-weights each iteration,
                    # so it tries again.
                    if verbose:
                        print("Line search found no decrease; keeping the current model")
                    if inversion_type != 'L1L2':
                        stop_reason = 'plateau'
                        break
            
            # Check IRLS convergence
            if inversion_type in ['L1', 'L1L2'] and irls_iter > 0:
                irls_change = np.linalg.norm(mr - mr_previous) / np.linalg.norm(mr_previous)
                if verbose:
                    print(f"IRLS relative change: {irls_change}")
                if irls_change < irls_tol or chi2_ert < target_chi2:
                    if verbose:
                        print(f"IRLS converged after {irls_iter + 1} iterations")
                    break
            
            if inversion_type in ['L1', 'L1L2']:
                mr_previous = mr.copy()
        
        # Every row of Err_tot is scored before that iteration's update, so when
        # the loop ends on its iteration cap the last row, meta['chi2'] and
        # chi2_history described the model from before the final step, not the
        # one returned. Score the returned model the same way the loop does.
        iterations_run = len(Err_tot)
        if model_moved:
            if solved is not None and np.array_equal(solved[0], mr):
                # The line search has just solved the model it accepted.
                dr_final = np.vstack([np.log(r).reshape(-1, 1) for r in solved[1]])
            else:
                dr_final = _calculate_forward(self.fwd_operators, mr, self.mesh, self.size)
            err_final = _as_col(self.rhos1 - dr_final.reshape(-1, 1))
            chi2_final = _quad(_matvec(self.Wd_sq, err_final), err_final) / len(err_final)
            if inversion_type == 'L2':
                fm_final = Lambda * _quad(_ttm(self.Wm, _spatial(mr)), _spatial(mr))
                ft_final = alpha * _quad(_ttm(self.Wt, mr), mr)
            else:
                md_final = _matvec(self.Wm, _spatial(mr))
                fm_final = Lambda * _quad(_matvec(Rs, md_final), md_final)
                td_final = _matvec(self.Wt, mr)
                ft_final = alpha * _quad(_matvec(Rt, td_final), td_final)
            Err_tot.append([chi2_final, fm_final, ft_final])
        
        # Process final results
        # Reshape to (cells, timesteps)
        final_model = np.reshape(mr, (-1, self.size), order='F').astype(self.dtype, copy=False)
        final_model = np.exp(final_model).astype(self.dtype if self.use_sparse else np.float64, copy=False)
        
        # Compute coverage for middle time step. Its operator usually holds this
        # model's solve already, and its Jacobian when the loop stopped at the
        # start of an iteration; they are the same only if the model handed to
        # pyGIMLi here is the one the forward wrappers passed (no resistivity
        # clip, and not the float32 copy of low-memory mode).
        mid_idx = self.size // 2
        block = np.reshape(mr, (-1, self.size), order='F')[:, mid_idx]
        held = (solved is not None and np.array_equal(solved[0], mr)
                and np.array_equal(np.clip(np.exp(np.clip(block, -20, 20)), 0.001, 1e6),
                                   final_model[:, mid_idx]))
        if held:
            dr = pg.Vector(solved[1][mid_idx])
        else:
            dr = self.fwd_operators[mid_idx].response(pg.Vector(final_model[:, mid_idx]))
        if not (held and solved[2]):
            self.fwd_operators[mid_idx].createJacobian(pg.Vector(final_model[:, mid_idx]))
        
        covTrans = pg.core.coverageDCtrans(
            self.fwd_operators[mid_idx].jacobian(), 
            1.0 / dr,
            1.0 / pg.Vector(final_model[:, mid_idx])
        )
        
        paramSizes = np.zeros(len(final_model[:, mid_idx]))
        mesh2 = self.fwd_operators[mid_idx].paraDomain
        
        for c in mesh2.cells():
            paramSizes[c.marker()] += c.size()
            
        FinalJ = np.log10(covTrans / paramSizes)
        
        # Store results
        result.final_models = final_model
        result.all_coverage = [FinalJ.copy() for _ in range(self.size)]
        result.mesh = mesh2
        result.all_chi2 = Err_tot
        # Why the loop ended, so a caller driving lambda can tell "this lambda is
        # spent" apart from "this run ran out of iterations".
        result.meta['stop_reason'] = stop_reason
        result.meta['iterations'] = iterations_run
        result.meta['chi2'] = float(Err_tot[-1][0]) if Err_tot else float('nan')
        result.meta['lambda'] = float(self.parameters['lambda_val'])
        result.meta['final_lambda'] = float(Lambda)
        result.meta['chi2_history'] = [float(row[0]) for row in Err_tot]
        # How the temporal constraint was distributed over the sequence. Two runs
        # with the same alpha are not the same inversion if one weighted by the
        # interval and the other did not, so the result records which it was.
        result.meta['temporal_weighting'] = dict(
            getattr(self, 'temporal_weight_report', None) or {})
        if prior is not None:
            result.meta['zones'] = [dict(entry) for entry in prior.report]

        if verbose:
            print('End of inversion')
        return result


# Artifact/export orchestration promoted from qt_apps. This public module owns
# the API while the private sibling keeps those helpers separate from the
# numerical TimeLapseERTInversion class.
from ._time_lapse_workflow import (  # noqa: E402
    DEFAULT_TL,
    INVERSION_TYPES,
    BackendUnavailable,
    build_timelapse_config,
    default_times,
    run_timelapse_ert,
)
