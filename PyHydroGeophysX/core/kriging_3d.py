"""
3D Kriging utilities for seismic velocity interpolation.
"""
import numpy as np
import pyvista as pv
from scipy.interpolate import griddata
from typing import Tuple, Optional, Dict, List, Union
try:
    from gstools import Exponential, krige, vario_estimate_unstructured, vario_estimate
    import gstools as gs
    GSTOOLS_AVAILABLE = True
except ImportError:
    GSTOOLS_AVAILABLE = False
    print("Warning: gstools not installed. 3D kriging functionality will be limited.")


def create_3d_structured_grid(topography_data: np.ndarray, 
                            grid_resolution: int = 50,
                            z_cells: Optional[np.ndarray] = None) -> pv.StructuredGrid:
    """
    Create a 3D structured grid from topography data.
    
    Args:
        topography_data: Array of shape (n, 3) with columns [x, y, z]
        grid_resolution: Number of grid points in x and y directions
        z_cells: Array defining layer thicknesses. If None, uses default layers
        
    Returns:
        pv.StructuredGrid: 3D structured grid
    """
    # Extract coordinates
    x_data = topography_data[:, 0]
    y_data = topography_data[:, 1]
    z_data = topography_data[:, 2]
    
    # Create regular grid
    x = np.linspace(x_data.min(), x_data.max(), grid_resolution)
    y = np.linspace(y_data.min(), y_data.max(), grid_resolution)
    xx, yy = np.meshgrid(x, y)
    
    # Interpolate topography to regular grid
    zz = griddata(topography_data[:, :2], z_data, (xx, yy), method='nearest')
    
    # Default layer structure if not provided
    if z_cells is None:
        z_cells = np.array([0] + [1]*3 + [2]*6 + [3]*5 + [5]*4)
    
    # Create 3D grid
    xx1 = np.repeat(xx.reshape(xx.shape[0], xx.shape[1], 1), len(z_cells), axis=-1)
    yy1 = np.repeat(yy.reshape(yy.shape[0], yy.shape[1], 1), len(z_cells), axis=-1)
    zz1 = np.repeat(zz.reshape(zz.shape[0], zz.shape[1], 1), len(z_cells), axis=-1) - np.cumsum(z_cells).reshape((1, 1, -1))
    
    # Create structured grid
    mesh = pv.StructuredGrid(xx1, yy1, zz1)
    mesh["Elevation"] = zz1.ravel(order="F")
    
    return mesh


def estimate_directional_variograms(data_points: np.ndarray,
                                  values: np.ndarray,
                                  dip_angles: List[float] = [90, 85, 80],
                                  azimuth_angles: Optional[np.ndarray] = None,
                                  bins: Optional[np.ndarray] = None,
                                  angles_tol: float = 10.0,
                                  bandwidth: float = 15.0) -> Dict[str, np.ndarray]:
    """
    Estimate directional variograms for 3D data.
    
    Args:
        data_points: Array of shape (n, 3) with point coordinates
        values: Array of shape (n,) with values at each point
        dip_angles: List of dip angles in degrees
        azimuth_angles: Array of azimuth angles in degrees. If None, uses 0 to 180 in steps of 10
        bins: Bin edges for variogram. If None, uses default spacing
        angles_tol: Angular tolerance in degrees
        bandwidth: Bandwidth for directional variogram
        
    Returns:
        Dictionary with 'bins', 'variograms', and 'parameters' arrays
    """
    if not GSTOOLS_AVAILABLE:
        raise ImportError("gstools is required for variogram estimation")
    
    if azimuth_angles is None:
        azimuth_angles = np.arange(0, 190, 10)
    
    if bins is None:
        bins = np.linspace(0, 150, 31)
    
    bin_centers = []
    variograms = []
    parameters = []
    
    for dip in dip_angles:
        for az in azimuth_angles:
            angles = [np.deg2rad(az), np.deg2rad(dip)]
            
            bin_center, dir_vario, counts = gs.vario_estimate(
                data_points.T,
                values,
                bins,
                angles=angles,
                angles_tol=np.deg2rad(angles_tol),
                bandwidth=bandwidth,
                mesh_type="unstructured",
                return_counts=True,
            )
            
            parameters.append([dip, az])
            bin_centers.append(bin_center)
            variograms.append(dir_vario)
    
    return {
        'bins': np.array(bin_centers),
        'variograms': np.array(variograms),
        'parameters': np.array(parameters)
    }


def optimize_variogram_model(bins: np.ndarray,
                           variograms: np.ndarray,
                           n_samples: int = 10000,
                           n_best: int = 20,
                           parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> List[Dict]:
    """
    Find optimal variogram model parameters through random sampling.
    
    Args:
        bins: Bin centers from variogram estimation
        variograms: Variogram values for different directions
        n_samples: Number of random samples to test
        n_best: Number of best models to return
        parameter_ranges: Dictionary of parameter ranges for sampling
        
    Returns:
        List of dictionaries containing best model parameters
    """
    if not GSTOOLS_AVAILABLE:
        raise ImportError("gstools is required for variogram model optimization")
    
    # Default parameter ranges
    if parameter_ranges is None:
        parameter_ranges = {
            'yaw': (0, 180),
            'pitch': (0, 30),
            'roll': (0, 30),
            'len_x': (20, 200),
            'len_y': (20, 200),
            'len_z': (1, 20),
            'sill': (0.25, 0.5)
        }
    
    sample_results = []
    
    for i in range(n_samples):
        # Random sampling of parameters
        params = {
            'yaw': np.random.uniform(*parameter_ranges['yaw']),
            'pitch': np.random.uniform(*parameter_ranges['pitch']),
            'roll': np.random.uniform(*parameter_ranges['roll']),
            'len_x': np.random.uniform(*parameter_ranges['len_x']),
            'len_y': np.random.uniform(*parameter_ranges['len_y']),
            'len_z': np.random.uniform(*parameter_ranges['len_z']),
            'sill': np.random.uniform(*parameter_ranges['sill'])
        }
        
        # Create model
        fit_model = Exponential(
            dim=3,
            len_scale=[params['len_x'], params['len_y'], params['len_z']],
            angles=[np.deg2rad(params['yaw']), np.deg2rad(params['pitch']), np.deg2rad(params['roll'])],
            var=np.sqrt(params['sill']),
            nugget=0
        )
        
        # Calculate RMS error
        model_var = fit_model.variogram(bins[0, :])
        rms = np.sqrt(np.mean((model_var - variograms)**2))
        
        params['rms'] = rms
        params['model'] = fit_model
        sample_results.append(params)
    
    # Sort by RMS and return best models
    sample_results.sort(key=lambda x: x['rms'])
    return sample_results[:n_best]


def krige_seismic_velocity_3d(topography_file: Union[str, np.ndarray],
                             velocity_data: Union[str, np.ndarray],
                             grid_resolution: int = 50,
                             z_cells: Optional[np.ndarray] = None,
                             variogram_optimization: bool = True,
                             n_variogram_samples: int = 10000,
                             train_test_split: float = 0.8,
                             output_prefix: str = "seismic_3d",
                             save_results: bool = True) -> Tuple[pv.StructuredGrid, np.ndarray, Dict]:
    """
    Perform 3D kriging of seismic velocity data.
    
    This function creates a 3D structured grid from topography data and performs
    ordinary kriging to interpolate seismic velocities throughout the volume.
    
    Args:
        topography_file: Path to topography file or array of shape (n, 3) with [x, y, z]
        velocity_data: Path to velocity data file or array of shape (m, 4) with [x, y, z, velocity]
        grid_resolution: Number of grid points in x and y directions
        z_cells: Array defining layer thicknesses. If None, uses default layers
        variogram_optimization: Whether to optimize variogram parameters
        n_variogram_samples: Number of samples for variogram optimization
        train_test_split: Fraction of data to use for training (rest for validation)
        output_prefix: Prefix for output files
        save_results: Whether to save results to files
        
    Returns:
        Tuple of (mesh, kriged_field, kriging_variance)
        mesh: PyVista StructuredGrid with kriged velocity field
        kriged_field: Array of kriged velocity values
        kriging_info: Dictionary containing variogram info and validation results
    """
    if not GSTOOLS_AVAILABLE:
        raise ImportError("gstools is required for 3D kriging. Install with: pip install gstools")
    
    # Load data if file paths are provided
    if isinstance(topography_file, str):
        topography_data = np.loadtxt(topography_file)
    else:
        topography_data = topography_file
        
    if isinstance(velocity_data, str):
        velocity_data = np.loadtxt(velocity_data)
    
    # Create 3D grid
    print("Creating 3D structured grid...")
    mesh = create_3d_structured_grid(topography_data, grid_resolution, z_cells)
    
    # Split data for training and validation
    n_total = velocity_data.shape[0]
    n_train = int(n_total * train_test_split)
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_data = velocity_data[train_indices]
    test_data = velocity_data[test_indices] if len(test_indices) > 0 else None
    
    # Variogram analysis
    kriging_info = {}
    
    if variogram_optimization:
        print("Estimating directional variograms...")
        variogram_results = estimate_directional_variograms(
            train_data[:, :3],
            train_data[:, 3]
        )
        
        print(f"Optimizing variogram model with {n_variogram_samples} samples...")
        best_models = optimize_variogram_model(
            variogram_results['bins'],
            variogram_results['variograms'],
            n_samples=n_variogram_samples,
            n_best=1
        )
        
        best_model = best_models[0]['model']
        kriging_info['variogram_params'] = best_models[0]
        kriging_info['variogram_results'] = variogram_results
        
    else:
        # Use default model
        print("Using default variogram model...")
        best_model = Exponential(
            dim=3,
            len_scale=[100, 100, 10],
            angles=[0, 0, 0],
            var=0.5,
            nugget=0
        )
    
    # Perform kriging
    print("Performing 3D ordinary kriging...")
    krig = krige.Ordinary(
        best_model,
        train_data[:, :3].T,
        train_data[:, 3]
    )
    
    field, krige_var = krig.mesh(mesh, name="Velocity")
    
    # Validation if test data exists
    if test_data is not None:
        print("Validating kriging results...")
        predicted_values, _ = krig(test_data[:, :3].T)
        true_values = test_data[:, 3]
        
        rmse = np.sqrt(np.mean((predicted_values - true_values)**2))
        mae = np.mean(np.abs(predicted_values - true_values))
        r2 = 1 - np.sum((true_values - predicted_values)**2) / np.sum((true_values - np.mean(true_values))**2)
        
        kriging_info['validation'] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'n_test': len(test_data)
        }
        
        print(f"Validation RMSE: {rmse:.3f}")
        print(f"Validation MAE: {mae:.3f}")
        print(f"Validation R²: {r2:.3f}")
    
    # Add variance to mesh
    mesh["variance"] = krige_var.ravel(order="F")
    
    # Save results if requested
    if save_results:
        print(f"Saving results with prefix '{output_prefix}'...")
        mesh.save(f"{output_prefix}_mesh.vtk")
        np.save(f"{output_prefix}_field.npy", field)
        np.save(f"{output_prefix}_variance.npy", krige_var)
        
        if variogram_optimization:
            np.save(f"{output_prefix}_variogram_info.npy", kriging_info)
    
    return mesh, field, kriging_info


# Convenience functions for common use cases
def krige_from_2d_profiles(profile_velocities: Dict[str, np.ndarray],
                         topography_data: np.ndarray,
                         profile_locations: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]],
                         **kwargs) -> Tuple[pv.StructuredGrid, np.ndarray, Dict]:
    """
    Perform 3D kriging from 2D seismic profile data.
    
    Args:
        profile_velocities: Dictionary mapping profile names to velocity arrays
        topography_data: Topography data array
        profile_locations: Dictionary mapping profile names to (start_point, end_point) tuples
        **kwargs: Additional arguments passed to krige_seismic_velocity_3d
        
    Returns:
        Same as krige_seismic_velocity_3d
    """
    # Combine all profile data into 3D points
    all_points = []
    all_velocities = []
    
    for profile_name, velocities in profile_velocities.items():
        start_point, end_point = profile_locations[profile_name]
        
        # Create points along profile
        n_points = len(velocities)
        x_coords = np.linspace(start_point[0], end_point[0], n_points)
        y_coords = np.linspace(start_point[1], end_point[1], n_points)
        
        # Assume velocities have depth information or create synthetic depths
        # This would need to be adapted based on actual data format
        for i, vel_profile in enumerate(velocities):
            if isinstance(vel_profile, np.ndarray) and vel_profile.ndim > 0:
                depths = np.linspace(0, -30, len(vel_profile))  # Example depth range
                for j, (depth, vel) in enumerate(zip(depths, vel_profile)):
                    all_points.append([x_coords[i], y_coords[i], depth])
                    all_velocities.append(vel)
    
    # Convert to arrays
    velocity_data_3d = np.column_stack([np.array(all_points), np.array(all_velocities)])
    
    return krige_seismic_velocity_3d(topography_data, velocity_data_3d, **kwargs)