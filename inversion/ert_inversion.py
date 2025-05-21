"""
Single-time ERT inversion functionality.
"""
import numpy as np
import pygimli as pg
from pygimli.physics import ert
from scipy.sparse import diags
import sys
from typing import Optional, Union, Dict, Any, Tuple

from .base import InversionBase, InversionResult
from ..forward.ert_forward import ertforward2, ertforandjac2
from ..solvers.linear_solvers import generalized_solver


class ERTInversion(InversionBase):
    """Single-time ERT inversion class."""
    
    def __init__(self, data_file: str, mesh: Optional[pg.Mesh] = None, **kwargs):
        """
        Initialize ERT inversion.
        
        Args:
            data_file: Path to ERT data file
            mesh: Mesh for inversion (created if None)
            **kwargs: Additional parameters including:
                - lambda_val: Regularization parameter
                - method: Solver method ('cgls', 'lsqr', etc.)
                - model_constraints: (min, max) model parameter bounds
                - max_iterations: Maximum iterations
                - absoluteUError: Absolute data error
                - relativeError: Relative data error
                - lambda_rate: Lambda reduction rate
                - lambda_min: Minimum lambda value
                - use_gpu: Whether to use GPU acceleration (requires CuPy)
                - parallel: Whether to use parallel CPU computation
                - n_jobs: Number of parallel jobs (-1 for all cores)
        """
        # Load ERT data
        data = ert.load(data_file)
        
        # Call parent initializer
        super().__init__(data, mesh, **kwargs)
        
        # Set ERT-specific default parameters

        ert_defaults = {
            'lambda_val': 10.0,
            'method': 'cgls',
            'absoluteUError': 0.0,
            'relativeError': 0.05,
            'lambda_rate': 1.0,
            'lambda_min': 1.0,
            'use_gpu': False,      # Add GPU acceleration option
            'parallel': False,     # Add parallel computation option
            'n_jobs': -1           # Number of parallel jobs (-1 means all available cores)
        }
        
        # Update parameters with ERT defaults
        for key, value in ert_defaults.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Initialize internal variables
        self.fwd_operator = None
        self.Wdert = None  # Data weighting matrix
        self.Wm_r = None   # Model weighting matrix
        self.rhos1 = None  # Log-transformed apparent resistivities
    
    def setup(self):
        """Set up ERT inversion (create operators, matrices, etc.)"""
        # Create mesh if not provided
        if self.mesh is None:
            ert_manager = ert.ERTManager(self.data)
            self.mesh = ert_manager.createMesh(data=self.data, quality=34)
        
        # Initialize forward operator
        self.fwd_operator = ert.ERTModelling()
        self.fwd_operator.setData(self.data)
        self.fwd_operator.setMesh(self.mesh)
        
        # Prepare data
        rhos = self.data['rhoa']
        self.rhos1 = np.log(rhos.array())
        self.rhos1 = self.rhos1.reshape(self.rhos1.shape[0], 1)
        
        # Data error matrix
        if np.all(self.data['err']) != 0.0:
            # If data has error values, use them
            Delta_rhoa_rhoa = self.data['err'].array()
        else:
            # Otherwise, estimate error
            ert_manager = ert.ERTManager(self.data)
            Delta_rhoa_rhoa = ert_manager.estimateError(
                self.data,
                absoluteUError=self.parameters['absoluteUError'],
                relativeError=self.parameters['relativeError']
            )
        
        # Create data weighting matrix
        self.Wdert = np.diag(1.0 / np.log(Delta_rhoa_rhoa + 1))
        
        # Create model regularization matrix
        rm = self.fwd_operator.regionManager()
        Ctmp = pg.matrix.RSparseMapMatrix()
        
        rm.setConstraintType(1)
        rm.fillConstraints(Ctmp)
        self.Wm_r = pg.utils.sparseMatrix2coo(Ctmp)
        cw = rm.constraintWeights().array()
        self.Wm_r = diags(cw).dot(self.Wm_r)
        self.Wm_r = self.Wm_r.todense()
    
    def run(self, initial_model: Optional[np.ndarray] = None) -> InversionResult:
        """
        Run ERT inversion.
        
        Args:
            initial_model: Initial model parameters (if None, a homogeneous model is used)
            
        Returns:
            InversionResult with inversion results
        """
        # Make sure setup has been called
        if self.fwd_operator is None:
            self.setup()
        
        # Initialize result object
        result = InversionResult()
        
        # Set up initial model if not provided
        if initial_model is None:
            rhomodel = np.median(np.exp(self.rhos1)) * np.ones((self.fwd_operator.paraDomain.cellCount(), 1))
            mr = np.log(rhomodel)
        else:
            if initial_model.ndim == 1:
                initial_model = initial_model.reshape(-1, 1)
            if np.min(initial_model) <= 0:
                # Assume linear model values, convert to log
                mr = np.log(initial_model + 1e-6)
            else:
                mr = np.log(initial_model)
        
        # Reference model is the initial model
        mr_R = mr.copy()
        
        # Regularization parameter
        L_mr = np.sqrt(self.parameters['lambda_val'])
        
        # Model constraints
        min_mr, max_mr = self.parameters['model_constraints']
        min_mr = np.log(min_mr)
        max_mr = np.log(max_mr)
        
        # Initial setup for the inversion
        delta_mr = (mr - mr_R)
        chi2_ert = 1
        
        # Main inversion loop
        for nn in range(self.parameters['max_iterations']):
            print(f'-------------------Iteration: {nn} ---------------------------')
            
            # Forward modeling and Jacobian computation
            dr, Jr = ertforandjac2(self.fwd_operator, mr, self.mesh)
            dr = dr.reshape(dr.shape[0], 1)
            
            # Data misfit calculation
            dataerror_ert = self.rhos1 - dr
            fdert = (np.dot(self.Wdert, dataerror_ert)).T.dot(np.dot(self.Wdert, dataerror_ert))
            
            # Model regularization term
            fmert = (L_mr * self.Wm_r * (mr - mr_R)).T.dot( self.Wm_r * (mr - mr_R))
            
            # Total objective function
            fc_r = fdert + fmert
            
            # Compute chi-squared and check convergence
            old_chi2 = chi2_ert
            chi2_ert = fdert / len(dr)
            dPhi = abs(chi2_ert - old_chi2) / old_chi2 if nn > 0 else 1.0
            
            print(f'chi2: {chi2_ert}')
            print(f'dPhi: {dPhi}')
            
            # Store iterations data
            result.iteration_models.append(np.exp(mr.ravel()))
            result.iteration_chi2.append(chi2_ert)
            result.iteration_data_errors.append(dataerror_ert.ravel())
            
            # Check for convergence
            if chi2_ert < 1.5 or dPhi < 0.01:
                break
            
            # System matrix and gradient
            gc_r = np.vstack((self.Wdert.dot(dr - self.rhos1), L_mr * self.Wm_r.dot(delta_mr)))
            N11_R = np.vstack((self.Wdert.dot(Jr), L_mr * self.Wm_r))
            
            gc_r = np.array(gc_r)
            gc_r = gc_r.reshape(-1, 1)
            
            # Alternative gradient formulation
            gc_r1 = Jr.T.dot(self.Wdert.T.dot(self.Wdert)).dot(dr - self.rhos1) + \
                   (L_mr * self.Wm_r).T.dot( self.Wm_r).dot(delta_mr)
            
            # Solve normal equations for update
            d_mr = generalized_solver(
                N11_R, -gc_r, 
                method=self.parameters['method'],
                use_gpu=self.parameters['use_gpu'],
                parallel=self.parameters.get('parallel', False),
                n_jobs=self.parameters.get('n_jobs', -1)
            )
            
            # Line search
            mu_LS = 1
            iarm = 1
            while True:
                mr1 = mr + mu_LS * d_mr
                dr = ertforward2(self.fwd_operator, mr1, self.mesh)
                dr = dr.reshape(dr.shape[0], 1)
                
                dataerror_ert = self.rhos1 - dr
                fdert = (np.dot(self.Wdert, dataerror_ert)).T.dot(np.dot(self.Wdert, dataerror_ert))
                fmert = (L_mr * self.Wm_r * (mr1 - mr_R)).T.dot( self.Wm_r * (mr1 - mr_R))
                
                ft_r = fdert + fmert
                
                fgoal = fc_r - 1e-4 * mu_LS * (d_mr.T.dot(gc_r1.reshape(gc_r1.shape[0], 1)))
                #print(f'ft_r: {ft_r}, fgoal: {fgoal}')
                
                if ft_r < fgoal:
                    break
                else:
                    iarm = iarm + 1
                    mu_LS = mu_LS / 2
                
                if iarm > 20:
                    print('Line search FAIL EXIT')
                    break
            
            # Update model
            mr = mr1
            
            # Apply model constraints
            mr = np.clip(mr, min_mr, max_mr)
            
            # Update lambda
            lambda_min = self.parameters['lambda_min']
            if L_mr > np.sqrt(lambda_min):
                L_mr = L_mr * self.parameters['lambda_rate']
        
        # Process final model
        final_model = np.exp(mr)
        
        # Compute final forward response
        dr = self.fwd_operator.response(pg.Vector(final_model.ravel()))
        
        # Compute coverage
        self.fwd_operator.createJacobian(pg.Vector(final_model.ravel()))
        covTrans = pg.core.coverageDCtrans(
            self.fwd_operator.jacobian(),
            1.0 / dr,
            1.0 / pg.Vector(final_model.ravel())
        )
        
        paramSizes = np.zeros(len(final_model))
        mesh2 = self.fwd_operator.paraDomain
        
        for c in mesh2.cells():
            paramSizes[c.marker()] += c.size()
            
        FinalJ = np.log10(covTrans / paramSizes)
        
        # Store results
        result.final_model = final_model.ravel()
        result.coverage = FinalJ
        result.predicted_data = dr.array()
        result.mesh = mesh2
        
        print('End of inversion')
        return result
