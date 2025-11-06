"""
Climate Data Agent for Meteorological Data Integration

This agent fetches daily climate variables (precipitation, temperature, ET)
from PyDaymet and computes potential evapotranspiration (PET) using multiple
methods for integration with ERT resistivity imaging and moisture analysis.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import pandas as pd
try:
    import xarray as xr
except ImportError:
    xr = None

from .base_agent import BaseAgent


class ClimateDataAgent(BaseAgent):
    """
    Agent for retrieving meteorological data and computing PET.
    
    Purpose: Given site geometry or ERT line coordinates and a time window,
    fetch daily climate variables and compute PET using supported methods,
    returning feature-ready time series for fusion with ERT inversions.
    
    Methods: Support PET via Penman-Monteith, Priestley-Taylor, and 
    Hargreaves-Samani with parameter hooks (e.g., arid_correction) to 
    improve estimates in arid regions.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 llm_provider: str = "openai"):
        """
        Initialize the Climate Data Agent.
        
        Args:
            api_key: LLM API key for AI-assisted analysis
            model: LLM model to use
            llm_provider: LLM provider ('openai', 'gemini', 'claude')
        """
        super().__init__("climate_data", api_key, model, llm_provider)
        
        # Check if pydaymet is available
        try:
            import pydaymet
            self.pydaymet = pydaymet
        except ImportError:
            raise ImportError(
                "pydaymet is required for climate data retrieval. "
                "Install with: pip install pydaymet"
            )
        
        # Default variables to retrieve
        self.default_variables = ['prcp', 'tmin', 'tmax', 'srad', 'vp', 'dayl']
        
        # Supported PET methods
        self.pet_methods = ['penman_monteith', 'priestley_taylor', 'hargreaves_samani']
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute climate data retrieval and PET computation.
        
        Args:
            input_data: Dictionary containing:
                - coords: List of (x, y) tuples or single tuple for point data
                - geometry: Polygon or bbox tuple for gridded data
                - dates: Tuple (start_date, end_date) or list of years
                - crs: Coordinate reference system (default: 4326)
                - variables: List of variables to retrieve
                - pet_method: PET calculation method or list of methods
                - pet_params: Parameters for PET calculation
                - time_scale: 'daily', 'monthly', or 'annual'
                - region: 'na', 'hi', or 'pr'
                - ert_timestamps: Optional timestamps for alignment
                - antecedent_days: List of days for antecedent totals (e.g., [1, 3, 7])
                
        Returns:
            Dictionary containing climate data and derived features
        """
        self._log("Starting climate data retrieval")
        
        # Extract parameters
        coords = input_data.get('coords')
        geometry = input_data.get('geometry')
        dates = input_data.get('dates')
        crs = input_data.get('crs', 4326)
        variables = input_data.get('variables', self.default_variables)
        pet_method = input_data.get('pet_method', 'penman_monteith')
        pet_params = input_data.get('pet_params', {})
        time_scale = input_data.get('time_scale', 'daily')
        region = input_data.get('region', 'na')
        
        # Validate inputs
        if dates is None:
            raise ValueError("dates parameter is required")
        
        if coords is None and geometry is None:
            raise ValueError("Either coords or geometry must be provided")
        
        # Retrieve climate data
        if coords is not None:
            climate_data = self._get_point_data(
                coords, dates, crs, variables, pet_method, pet_params,
                time_scale, region, input_data.get('to_xarray', False)
            )
        else:
            climate_data = self._get_gridded_data(
                geometry, dates, crs, variables, pet_method, pet_params,
                time_scale, region
            )
        
        # Compute derived features
        derived_features = self._compute_derived_features(
            climate_data,
            antecedent_days=input_data.get('antecedent_days', [1, 3, 7]),
            compute_p_minus_pet=True
        )
        
        # Align with ERT timestamps if provided
        ert_alignment = None
        if input_data.get('ert_timestamps') is not None:
            ert_alignment = self._align_with_ert(
                climate_data,
                derived_features,
                input_data['ert_timestamps']
            )
        
        # Compare PET methods if multiple requested
        pet_comparison = None
        if isinstance(pet_method, list) and len(pet_method) > 1:
            pet_comparison = self._compare_pet_methods(
                coords or geometry, dates, crs, variables, pet_method,
                pet_params, time_scale, region
            )
        
        # Store results
        self.results = {
            'climate_data': climate_data,
            'derived_features': derived_features,
            'ert_alignment': ert_alignment,
            'pet_comparison': pet_comparison,
            'metadata': {
                'dates': dates,
                'variables': variables,
                'pet_method': pet_method,
                'time_scale': time_scale,
                'region': region,
                'crs': crs
            }
        }
        
        self._log("Climate data retrieval completed")
        
        return self.results
    
    def _get_point_data(self, coords: Union[Tuple, List[Tuple]], dates: Tuple,
                       crs: Any, variables: List[str], pet_method: str,
                       pet_params: Dict, time_scale: str, region: str,
                       to_xarray: bool = False) -> Union[pd.DataFrame, Any]:
        """Retrieve climate data for point locations."""
        self._log(f"Retrieving point data for {len(coords) if isinstance(coords, list) else 1} location(s)")
        
        climate_data = self.pydaymet.get_bycoords(
            coords=coords,
            dates=dates,
            crs=crs,
            variables=variables,
            region=region,
            time_scale=time_scale,
            pet=pet_method if pet_method else None,
            pet_params=pet_params if pet_params else None,
            to_xarray=to_xarray
        )
        
        return climate_data
    
    def _get_gridded_data(self, geometry: Any, dates: Tuple, crs: Any,
                         variables: List[str], pet_method: str,
                         pet_params: Dict, time_scale: str, region: str) -> Any:
        """Retrieve gridded climate data for a region."""
        self._log("Retrieving gridded climate data")
        
        climate_data = self.pydaymet.get_bygeom(
            geometry=geometry,
            dates=dates,
            crs=crs,
            variables=variables,
            region=region,
            time_scale=time_scale,
            pet=pet_method if pet_method else None,
            pet_params=pet_params if pet_params else None
        )
        
        return climate_data
    
    def _compute_derived_features(self, climate_data: Union[pd.DataFrame, Any],
                                 antecedent_days: List[int] = [1, 3, 7],
                                 compute_p_minus_pet: bool = True) -> Dict[str, Any]:
        """
        Compute derived climate features for hydrologic analysis.
        
        Args:
            climate_data: Climate data from PyDaymet
            antecedent_days: Days for computing antecedent totals
            compute_p_minus_pet: Whether to compute P-PET
            
        Returns:
            Dictionary of derived features
        """
        self._log("Computing derived climate features")
        
        features = {}
        
        # Handle both DataFrame and xarray formats
        if isinstance(climate_data, pd.DataFrame):
            df = climate_data.copy()
            
            # Compute antecedent precipitation totals
            if 'prcp' in df.columns:
                for days in antecedent_days:
                    col_name = f'prcp_antecedent_{days}d'
                    df[col_name] = df['prcp'].rolling(window=days, min_periods=1).sum()
                    features[col_name] = df[col_name]
            
            # Compute P-PET if both available
            if compute_p_minus_pet and 'prcp' in df.columns:
                pet_cols = [col for col in df.columns if 'pet' in col.lower()]
                for pet_col in pet_cols:
                    p_minus_pet = df['prcp'] - df[pet_col]
                    col_name = f'p_minus_{pet_col}'
                    df[col_name] = p_minus_pet
                    features[col_name] = p_minus_pet
            
            # Store the enhanced dataframe
            features['enhanced_data'] = df
            
        elif xr is not None and isinstance(climate_data, xr.Dataset):
            ds = climate_data.copy()
            
            # Compute antecedent precipitation totals
            if 'prcp' in ds.data_vars:
                for days in antecedent_days:
                    var_name = f'prcp_antecedent_{days}d'
                    ds[var_name] = ds['prcp'].rolling(time=days, min_periods=1).sum()
            
            # Compute P-PET if both available
            if compute_p_minus_pet and 'prcp' in ds.data_vars:
                pet_vars = [var for var in ds.data_vars if 'pet' in var.lower()]
                for pet_var in pet_vars:
                    var_name = f'p_minus_{pet_var}'
                    ds[var_name] = ds['prcp'] - ds[pet_var]
            
            # Store the enhanced dataset
            features['enhanced_data'] = ds
        
        return features
    
    def _align_with_ert(self, climate_data: Union[pd.DataFrame, Any],
                       derived_features: Dict, ert_timestamps: List) -> Dict[str, Any]:
        """
        Align climate data with ERT acquisition timestamps.
        
        Args:
            climate_data: Climate data
            derived_features: Derived climate features
            ert_timestamps: List of ERT acquisition timestamps
            
        Returns:
            Dictionary with aligned data and concurrent features
        """
        self._log("Aligning climate data with ERT timestamps")
        
        aligned = {}
        
        if isinstance(climate_data, pd.DataFrame):
            # Ensure timestamps are datetime
            ert_times = pd.to_datetime(ert_timestamps)
            
            # Get climate data at or nearest to ERT timestamps
            if 'enhanced_data' in derived_features:
                df = derived_features['enhanced_data']
                
                # Align to nearest timestamp
                aligned_data = []
                for ert_time in ert_times:
                    # Find nearest climate data point
                    if hasattr(df.index, 'get_loc'):
                        try:
                            idx = df.index.get_indexer([ert_time], method='nearest')[0]
                            aligned_data.append(df.iloc[idx])
                        except (KeyError, IndexError, ValueError) as e:
                            # If time index doesn't exist, try matching dates
                            self._log(f"Warning: Could not align timestamp {ert_time} using index: {str(e)}", level='WARN')
                            try:
                                df_copy = df.copy()
                                if 'time' in df_copy.columns:
                                    df_copy['time'] = pd.to_datetime(df_copy['time'])
                                    df_copy = df_copy.set_index('time')
                                    idx = df_copy.index.get_indexer([ert_time], method='nearest')[0]
                                    aligned_data.append(df_copy.iloc[idx])
                                else:
                                    self._log(f"Warning: Cannot find 'time' column for alignment", level='WARN')
                            except (KeyError, IndexError, ValueError) as e2:
                                self._log(f"Warning: Failed to align timestamp {ert_time}: {str(e2)}", level='WARN')
                                continue
                
                if aligned_data:
                    aligned['ert_aligned_data'] = pd.DataFrame(aligned_data)
                    aligned['ert_timestamps'] = ert_times
        
        return aligned
    
    def _compare_pet_methods(self, coords_or_geom: Any, dates: Tuple, crs: Any,
                            variables: List[str], pet_methods: List[str],
                            pet_params: Dict, time_scale: str, region: str) -> Dict[str, Any]:
        """
        Compare different PET calculation methods.
        
        Args:
            coords_or_geom: Coordinates or geometry
            dates: Date range
            crs: Coordinate reference system
            variables: Variables to retrieve
            pet_methods: List of PET methods to compare
            pet_params: PET parameters
            time_scale: Time scale
            region: Region
            
        Returns:
            Dictionary with comparison results
        """
        self._log(f"Comparing {len(pet_methods)} PET methods")
        
        comparison = {'methods': pet_methods, 'data': {}}
        
        # Retrieve data with each method
        for method in pet_methods:
            try:
                if isinstance(coords_or_geom, (tuple, list)):
                    data = self._get_point_data(
                        coords_or_geom, dates, crs, variables, method,
                        pet_params, time_scale, region
                    )
                else:
                    data = self._get_gridded_data(
                        coords_or_geom, dates, crs, variables, method,
                        pet_params, time_scale, region
                    )
                
                comparison['data'][method] = data
                
                # Compute basic statistics
                if isinstance(data, pd.DataFrame):
                    pet_cols = [col for col in data.columns if 'pet' in col.lower()]
                    if pet_cols:
                        comparison[f'{method}_mean'] = data[pet_cols[0]].mean()
                        comparison[f'{method}_std'] = data[pet_cols[0]].std()
                
            except Exception as e:
                self._log(f"Warning: Failed to compute PET with {method}: {str(e)}", level='WARN')
                comparison['data'][method] = None
        
        # Compute sensitivity metrics
        pet_values = []
        for method in pet_methods:
            if comparison['data'][method] is not None:
                if isinstance(comparison['data'][method], pd.DataFrame):
                    pet_cols = [col for col in comparison['data'][method].columns 
                               if 'pet' in col.lower()]
                    if pet_cols:
                        pet_values.append(comparison['data'][method][pet_cols[0]].values)
        
        if len(pet_values) >= 2:
            # Compute coefficient of variation across methods
            pet_array = np.array(pet_values)
            comparison['method_mean'] = np.nanmean(pet_array, axis=0)
            comparison['method_std'] = np.nanstd(pet_array, axis=0)
            # Use appropriate epsilon for typical PET values (mm/day)
            # and avoid division by values close to zero
            mean_vals = comparison['method_mean']
            std_vals = comparison['method_std']
            # Only compute CV where mean is significantly above zero (> 0.1 mm/day)
            # This threshold can be adjusted based on regional conditions
            min_threshold = 0.1  # mm/day
            valid_mask = mean_vals > min_threshold
            if np.any(valid_mask):
                cv_values = np.where(valid_mask, std_vals / mean_vals, np.nan)
                comparison['coefficient_of_variation'] = np.nanmean(cv_values)
            else:
                comparison['coefficient_of_variation'] = np.nan
        
        return comparison
    
    def _log(self, message: str, level: str = 'INFO'):
        """Log a message."""
        print(f"[{level}] ClimateDataAgent: {message}")
    
    def get_climate_summary(self) -> str:
        """
        Generate a summary of retrieved climate data.
        
        Returns:
            Formatted string with climate data summary
        """
        if not self.results:
            return "No climate data retrieved yet."
        
        summary = []
        summary.append("=" * 60)
        summary.append("Climate Data Summary")
        summary.append("=" * 60)
        
        metadata = self.results.get('metadata', {})
        summary.append(f"\nDate Range: {metadata.get('dates')}")
        summary.append(f"Variables: {', '.join(metadata.get('variables', []))}")
        summary.append(f"PET Method: {metadata.get('pet_method')}")
        summary.append(f"Time Scale: {metadata.get('time_scale')}")
        summary.append(f"Region: {metadata.get('region')}")
        
        # Add derived features info
        if self.results.get('derived_features'):
            summary.append("\nDerived Features:")
            features = self.results['derived_features']
            for key in features:
                if key != 'enhanced_data':
                    summary.append(f"  - {key}")
        
        # Add ERT alignment info
        if self.results.get('ert_alignment'):
            alignment = self.results['ert_alignment']
            if 'ert_timestamps' in alignment:
                n_timestamps = len(alignment['ert_timestamps'])
                summary.append(f"\nERT Alignment: {n_timestamps} timestamps matched")
        
        # Add PET comparison info
        if self.results.get('pet_comparison'):
            comparison = self.results['pet_comparison']
            summary.append(f"\nPET Method Comparison:")
            summary.append(f"  Methods: {', '.join(comparison.get('methods', []))}")
            if 'coefficient_of_variation' in comparison:
                cv = comparison['coefficient_of_variation']
                summary.append(f"  Coefficient of Variation: {cv:.3f}")
        
        summary.append("=" * 60)
        
        return "\n".join(summary)
