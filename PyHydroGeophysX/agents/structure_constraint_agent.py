"""
Structure Constraint Agent

Applies seismic velocity interfaces as structural constraints to ERT inversion.
Implements the workflow from Ex_Structure_resinv.py for creating structure-constrained
resistivity models.
"""

from typing import Dict, Any, Optional
import numpy as np
import os
from .base_agent import BaseAgent


class StructureConstraintAgent(BaseAgent):
    """
    Agent for applying structural constraints from seismic data to ERT inversion.
    
    This agent creates ERT meshes that honor geological boundaries derived from
    seismic velocity interfaces, leading to more accurate resistivity models
    that preserve sharp layer contrasts.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 llm_provider: str = "openai"):
        """Initialize Structure Constraint Agent."""
        super().__init__("structure_constraint", api_key, model, llm_provider)
        self.system_message = """You are an expert in structure-constrained geophysical inversion.
You understand how to incorporate a priori geological information from seismic data into
ERT inversions to improve layer boundary resolution and reduce artifacts."""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute structure-constrained ERT inversion.
        
        Args:
            input_data: Dictionary containing:
                - ert_data: ERT measurement data
                - seismic_data: (Optional) Seismic traveltime data for interface extraction
                - interface_coords: (Optional) Tuple of (x, z) coordinates from seismic
                - velocity_threshold: (Optional) Velocity threshold for interface extraction
                - seismic_params: (Optional) Parameters for seismic inversion
                - inversion_params: ERT inversion parameters
                - output_dir: Directory for saving results
                - mesh_quality: Mesh quality parameter (default: 31)
                
        Returns:
            Dictionary containing constrained resistivity model and mesh
        """
        self._log_execution("Starting structure-constrained ERT inversion")
        
        try:
            from pygimli.physics import ert
            from PyHydroGeophysX.core.mesh_utils import add_velocity_interface, extract_velocity_interface
            import pygimli as pg
            import pygimli.physics.traveltime as tt
            
            # Extract parameters
            ert_data = input_data.get('ert_data')
            seismic_data = input_data.get('seismic_data')  # Optional traveltime data
            interface_coords = input_data.get('interface_coords')  # Optional pre-computed
            velocity_threshold = input_data.get('velocity_threshold', 1000)
            seismic_params = input_data.get('seismic_params', {})
            inversion_params = input_data.get('inversion_params', {})
            output_dir = input_data.get('output_dir', 'results/structure_constrained')
            mesh_quality = input_data.get('mesh_quality', 31)
            mesh_params = input_data.get('mesh_params', {})
            
            os.makedirs(output_dir, exist_ok=True)
            
            if ert_data is None:
                raise ValueError("ert_data is required")
            
            # Check if we need to extract interface from seismic data
            if interface_coords is None:
                if seismic_data is None:
                    raise ValueError("Either interface_coords or seismic_data is required")
                
                self._log_execution("No interface_coords provided, will extract from seismic data")
                
                # Step 1: Create initial mesh for seismic inversion
                self._log_execution("Creating initial mesh for seismic inversion")
                ert1 = ert.ERTManager(ert_data)
                grid = ert1.createMesh(
                    data=ert_data,
                    quality=mesh_quality,
                    paraDX=mesh_params.get('paraDX', 0.5),
                    paraMaxCellSize=mesh_params.get('paraMaxCellSize', 2),
                    boundaryMaxCellSize=mesh_params.get('boundaryMaxCellSize', 3000),
                    smooth=[2, 2],
                    paraBoundary=0.1,
                    paraDepth=mesh_params.get('paraDepth', 30.0)
                )
                initial_mesh = ert1.fop.paraDomain
                initial_mesh.setCellMarkers(np.ones((initial_mesh.cellCount())) * 2)
                
                self._log_execution(f"Created initial mesh with {initial_mesh.cellCount()} cells (quality={mesh_quality})")
                
                # Step 2: Run seismic inversion
                self._log_execution("Running seismic travel time inversion...")
                TT = tt.TravelTimeManager()
                TT.setMesh(initial_mesh)
                TT.invert(
                    seismic_data,
                    lam=seismic_params.get('lam', 50),
                    zWeight=seismic_params.get('zWeight', 0.2),
                    vTop=seismic_params.get('vTop', 500),
                    vBottom=seismic_params.get('vBottom', 8000),
                    verbose=1,
                    limits=[500.,10000.]
                )
                
                velocity_model = TT.model.array()
                self._log_execution(f"Seismic inversion completed")
                self._log_execution(f"  Velocity range: {np.min(velocity_model):.0f} - {np.max(velocity_model):.0f} m/s")
                
                # Step 3: Extract interface at velocity threshold
                self._log_execution(f"Extracting velocity interface at {velocity_threshold} m/s threshold...")
                interface_x, interface_z = extract_velocity_interface(
                    initial_mesh,
                    velocity_model,
                    threshold=velocity_threshold,
                    interval=5
                )
                
                self._log_execution(f"Interface extracted with {len(interface_x)} points")
                self._log_execution(f"  Depth range: {np.min(interface_z):.1f} - {np.max(interface_z):.1f} m")
                
                # Store for output
                interface_coords = (interface_x, interface_z)
                
                # Store seismic results
                seismic_results = {
                    'velocity_model': velocity_model,
                    'mesh': initial_mesh,
                    'coverage': TT.standardizedCoverage()
                }
            else:
                interface_x, interface_z = interface_coords
                seismic_results = None
                self._log_execution(f"Using provided interface coordinates with {len(interface_x)} points")
            
            # Step 4: Create structure-constrained mesh
            
            # Step 4: Create structure-constrained mesh
            self._log_execution(f"Interface has {len(interface_x)} points")
            self._log_execution("Creating mesh with structural constraints")
            
            # Create mesh with velocity interface
            # Pass mesh parameters if provided
            mesh_kwargs = {}
            if 'paraDepth' in mesh_params:
                mesh_kwargs['paraDepth'] = mesh_params['paraDepth']
                self._log_execution(f"Using paraDepth: {mesh_params['paraDepth']}")
            if 'paraDX' in mesh_params:
                mesh_kwargs['paraDX'] = mesh_params['paraDX']
                self._log_execution(f"Using paraDX: {mesh_params['paraDX']}")
            if 'paraMaxCellSize' in mesh_params:
                mesh_kwargs['paraMaxCellSize'] = mesh_params['paraMaxCellSize']
                self._log_execution(f"Using paraMaxCellSize: {mesh_params['paraMaxCellSize']}")
            
            markers, mesh_with_interface = add_velocity_interface(
                ert_data,
                interface_x,
                interface_z,
                **mesh_kwargs
            )
            
            self._log_execution(f"Created constrained mesh with {mesh_with_interface.cellCount()} cells")
            
            # Step 5: Run structure-constrained ERT inversion
            # Get LLM recommendations for inversion parameters if needed
            if self.api_key and not inversion_params:
                self._log_execution("Requesting LLM recommendations for constrained inversion")
                inversion_params = self._get_recommended_params(ert_data, mesh_with_interface)
            
            # Set default parameters
            lam = inversion_params.get('lambda', 10.0)
            max_iterations = inversion_params.get('max_iterations', 20)
            limits = inversion_params.get('limits', [1.0, 10000.0])
            
            self._log_execution(f"Inversion parameters: lambda={lam}, max_iter={max_iterations}")
            
            # Calculate geometric factors if not present (required for field data)
            if not ert_data.allNonZero('k'):
                self._log_execution("Calculating geometric factors for field data")
                ert_data['k'] = ert.createGeometricFactors(ert_data)
                self._log_execution(f"Geometric factors calculated: min={min(ert_data['k']):.2f}, max={max(ert_data['k']):.2f}")
            else:
                self._log_execution("Geometric factors already present in data")
            
            # Set default error estimates if not present or invalid (required for field data)
            if not ert_data.allNonZero('err') or min(ert_data['err']) <= 0:
                self._log_execution("Setting default error estimates for field data (5% + 0.5 Ohm)")
                ert_data['err'] = ert.estimateError(ert_data, absoluteError=0.5, relativeError=0.05)
                self._log_execution(f"Error estimates set: min={min(ert_data['err']):.2f}, max={max(ert_data['err']):.2f}")
            else:
                self._log_execution("Error estimates already present in data")
            
            # Run constrained inversion
            self._log_execution("Running ERT inversion with structural constraints")
            mgr_constrained = ert.ERTManager()
            mgr_constrained.invert(
                data=ert_data,
                verbose=True,
                lam=lam,
                mesh=mesh_with_interface,
                limits=limits,
                maxIter=max_iterations
            )
            
            self._log_execution("Constrained inversion completed successfully")
            
            # Get results
            resistivity_model = mgr_constrained.model
            para_domain = mgr_constrained.paraDomain
            
            # Calculate coverage - coverage() returns numeric values for ERT
            coverage = mgr_constrained.coverage()
            coverage_array = np.array(coverage)
            self._log_execution(f"Coverage array shape: {coverage_array.shape}, dtype: {coverage_array.dtype}")
            self._log_execution(f"Coverage range: {np.min(coverage_array):.4f} - {np.max(coverage_array):.4f}")
            
            # Get coverage for para_domain cells only
            # Para domain markers should be >= 0 for valid cells
            markers_array = np.array(para_domain.cellMarkers())
            valid_mask = markers_array >= 0
            coverage_filtered = coverage_array[valid_mask]
            self._log_execution(f"Filtered coverage: {len(coverage_filtered)} cells from {len(coverage_array)} total")
            
            # Map layer markers from mesh to para_domain (will be done later in the code)
            # For now, save the original markers
            
            # Save results
            np.save(os.path.join(output_dir, 'resistivity_model.npy'), 
                   np.array(resistivity_model))
            np.save(os.path.join(output_dir, 'coverage.npy'), 
                   np.array(coverage_filtered))
            # Save original para_domain cell IDs for reference
            np.save(os.path.join(output_dir, 'para_cell_ids.npy'),
                   np.array(para_domain.cellMarkers()))
            # Save mesh markers for reference
            np.save(os.path.join(output_dir, 'mesh_cell_markers.npy'),
                   np.array(mesh_with_interface.cellMarkers()))
            mesh_with_interface.save(os.path.join(output_dir, 'constrained_mesh.bms'))
            
            self._log_execution("Results saved to disk")
            
            # Get LLM interpretation
            interpretation = None
            if self.api_key:
                self._log_execution("Generating interpretation of constrained inversion")
                interpretation = self._interpret_results(
                    resistivity_model,
                    markers,
                    interface_coords
                )
            
            # Calculate statistics
            res_mean = np.mean(resistivity_model)
            res_std = np.std(resistivity_model)
            res_range = [np.min(resistivity_model), np.max(resistivity_model)]
            
            # Map layer markers from mesh_with_interface to para_domain
            # para_domain has unique cell IDs, not layer markers
            # mesh_with_interface has layer markers (2=regolith, 3=fractured bedrock)
            
            mesh_markers = np.array(mesh_with_interface.cellMarkers())
            self._log_execution(f"Mesh with interface: {mesh_with_interface.cellCount()} cells, markers: {np.unique(mesh_markers)}")
            self._log_execution(f"Para domain: {para_domain.cellCount()} cells")
            
            # Map layer markers to para_domain cells based on cell center positions
            para_layer_markers = np.zeros(para_domain.cellCount(), dtype=int)
            
            # Get cell centers
            para_centers = para_domain.cellCenters()
            mesh_centers = mesh_with_interface.cellCenters()
            
            self._log_execution(f"Mapping layer markers from mesh to para_domain...")
            
            # For each para_domain cell, find closest mesh cell and copy its marker
            # Only copy markers 2 and 3 (layer markers), ignore 1 (background)
            from scipy.spatial import cKDTree
            tree = cKDTree(mesh_centers)
            
            for i in range(para_domain.cellCount()):
                para_center = para_centers[i]
                # Find closest mesh cell
                dist, idx = tree.query(para_center)
                mesh_marker = mesh_markers[idx]
                
                # Map mesh markers to para domain
                # mesh marker 1 (background) -> para marker 2 (will be simplified later)
                # mesh marker 2 (regolith) -> para marker 2
                # mesh marker 3 (fractured bedrock) -> para marker 3
                if mesh_marker == 3:
                    para_layer_markers[i] = 3
                else:
                    para_layer_markers[i] = 2
            
            unique_para_markers = np.unique(para_layer_markers)
            self._log_execution(f"Mapped para_domain layer markers: {unique_para_markers}")
            self._log_execution(f"  Marker 2 (regolith): {np.sum(para_layer_markers == 2)} cells")
            self._log_execution(f"  Marker 3 (bedrock): {np.sum(para_layer_markers == 3)} cells")
            
            # Save mapped layer markers
            np.save(os.path.join(output_dir, 'cell_markers.npy'), para_layer_markers)
            self._log_execution(f"Saved mapped layer markers to {output_dir}/cell_markers.npy")
            
            # Verify shapes match
            if len(resistivity_model) != len(para_layer_markers):
                self._log_execution(f"WARNING: Size mismatch! resistivity: {len(resistivity_model)}, markers: {len(para_layer_markers)}", level='WARN')
            
            self.results = {
                'status': 'success',
                'resistivity_model': np.array(resistivity_model),
                'mesh': para_domain,
                'constrained_mesh': mesh_with_interface,
                'coverage': coverage_filtered,
                'cell_markers': para_layer_markers,  # Mapped layer markers (2, 3) matching resistivity_model size
                'interface_coords': interface_coords,
                'seismic_results': seismic_results,  # Include seismic results if computed
                'velocity_threshold': velocity_threshold,  # Store threshold used
                'inversion_params': {
                    'lambda': lam,
                    'max_iterations': max_iterations,
                    'limits': limits
                },
                'statistics': {
                    'mean_resistivity': res_mean,
                    'std_resistivity': res_std,
                    'resistivity_range': res_range,
                    'num_cells': para_domain.cellCount(),  # Use para_domain count (matches resistivity)
                    'num_layers': len(np.unique(para_layer_markers))
                },
                'interpretation': interpretation,
                'output_dir': output_dir
            }
            
            self._log_execution(f"Resistivity range: {res_range[0]:.1f} - {res_range[1]:.1f} Ωm")
            self._log_execution(f"Number of layers: {len(np.unique(markers))}")
            
            return self.results
            
        except Exception as e:
            self._log_execution(f"Error during constrained inversion: {str(e)}", level='ERROR')
            self.results = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
    def _get_recommended_params(self, ert_data, mesh) -> Dict[str, Any]:
        """
        Get LLM recommendations for constrained inversion parameters.
        
        Args:
            ert_data: ERT measurement data
            mesh: Constrained mesh
            
        Returns:
            Recommended parameters dictionary
        """
        try:
            data_info = f"""
            ERT Data Characteristics:
            - Number of electrodes: {ert_data.sensorCount()}
            - Number of measurements: {ert_data.size()}
            - Mesh cells: {mesh.cellCount()}
            - With structural constraints from seismic
            """
            
            prompt = f"""Recommend inversion parameters for structure-constrained ERT inversion:

{data_info}

Structure-constrained inversions can typically use lower regularization than unconstrained
inversions because the structural boundaries provide additional information.

Provide recommendations for:
1. Lambda (regularization): Consider 5-20 for constrained inversions
2. Max iterations: typical 15-25
3. Resistivity limits: typical [1, 10000] Ωm

Return as: lambda=XX, max_iterations=XX, limits=[XX, XX]"""
            
            response = self.query_llm(prompt, self.system_message, 
                                     temperature=0.3, max_tokens=200)
            
            # Parse response with robust error handling
            params = {'lambda': 10.0, 'max_iterations': 20, 'limits': [1.0, 10000.0]}  # defaults
            
            try:
                import re
                # Parse lambda
                match = re.search(r'lambda[=:\s]+(\d+\.?\d*)', response, re.IGNORECASE)
                if match:
                    params['lambda'] = float(match.group(1))
                
                # Parse max_iterations
                match = re.search(r'max[_\s]*iterations[=:\s]+(\d+)', response, re.IGNORECASE)
                if match:
                    params['max_iterations'] = int(match.group(1))
                
                # Parse limits
                match = re.search(r'limits[=:\s]*\[(\d+\.?\d*)[,\s]+(\d+\.?\d*)\]', response, re.IGNORECASE)
                if match:
                    params['limits'] = [float(match.group(1)), float(match.group(2))]
            except (ValueError, AttributeError) as e:
                self._log_execution(f"Could not parse LLM response: {e}, using defaults")
            
            self._log_execution(f"LLM recommended parameters: {params}")
            
            return params
        except Exception as e:
            self._log_execution(f"Could not get LLM recommendations: {e}, using defaults")
            return {'lambda': 10.0, 'max_iterations': 20, 'limits': [1.0, 10000.0]}
    
    def _interpret_results(self, resistivity_model, markers, interface_coords) -> str:
        """
        Get LLM interpretation of constrained inversion results.
        
        Args:
            resistivity_model: Inverted resistivity values
            markers: Cell markers indicating layers
            interface_coords: Interface coordinates
            
        Returns:
            Interpretation string
        """
        try:
            res_array = np.array(resistivity_model)
            interface_x, interface_z = interface_coords
            
            results_summary = f"""
            Structure-Constrained ERT Inversion Results:
            - Resistivity range: {np.min(res_array):.1f} to {np.max(res_array):.1f} Ωm
            - Number of layers: {len(np.unique(markers))}
            - Interface depth range: {np.min(interface_z):.1f} to {np.max(interface_z):.1f} m
            - Interface lateral extent: {np.min(interface_x):.1f} to {np.max(interface_x):.1f} m
            """
            
            prompt = f"""Interpret these structure-constrained ERT inversion results:

{results_summary}

Provide a brief interpretation (2-3 sentences) about:
1. The advantages of using structural constraints in this case
2. What the resistivity values suggest about subsurface properties"""
            
            interpretation = self.query_llm(prompt, self.system_message,
                                           temperature=0.5, max_tokens=200)
            return interpretation
        except:
            return "Structure-constrained inversion completed with seismic-derived boundaries"
    
    def _log_execution(self, message: str, level: str = 'INFO'):
        """Log execution message."""
        print(f"[{self.name}] [{level}] {message}")
