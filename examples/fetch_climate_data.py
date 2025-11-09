"""
Standalone Climate Data Fetcher Script

This script runs in a separate Python environment with:
- PyDaymet 0.19+ (requires NumPy 2.x)
- pandas 2.x

It fetches climate data and saves it as CSV files that can be loaded
in the main PyHydroGeophysX environment (which uses NumPy 1.x for PyGIMLi).

Usage:
    python fetch_climate_data.py --config climate_config.json
    
Or run directly with command-line arguments:
    python fetch_climate_data.py --coords -105.3 40.0 --dates 2021-09-01 2021-11-30 --output climate_data.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union
import pandas as pd
import numpy as np

try:
    import pydaymet
except ImportError:
    print("ERROR: pydaymet not found. Install with: pip install pydaymet>=0.19")
    sys.exit(1)


def fetch_climate_data(
    coords: Union[Tuple[float, float], List[Tuple[float, float]]],
    dates: Tuple[str, str],
    crs: int = 4326,
    variables: List[str] = None,
    pet_method: str = 'penman_monteith',
    pet_params: Dict = None,
    time_scale: str = 'daily',
    region: str = 'na',
    antecedent_days: List[int] = None
) -> pd.DataFrame:
    """
    Fetch climate data using PyDaymet.
    
    Args:
        coords: (longitude, latitude) tuple or list of tuples
        dates: (start_date, end_date) tuple
        crs: Coordinate reference system (default: 4326 for WGS84)
        variables: List of variables to fetch
        pet_method: PET calculation method
        pet_params: Parameters for PET calculation
        time_scale: 'daily', 'monthly', or 'annual'
        region: 'na' (North America), 'hi' (Hawaii), or 'pr' (Puerto Rico)
        antecedent_days: List of days for computing antecedent totals
        
    Returns:
        DataFrame with climate data and derived features
    """
    if variables is None:
        variables = ['prcp', 'tmin', 'tmax', 'srad', 'vp', 'dayl']
    
    if pet_params is None:
        pet_params = {}
    
    if antecedent_days is None:
        antecedent_days = [1, 3, 7, 14]
    
    print(f"Fetching climate data for coordinates: {coords}")
    print(f"Date range: {dates[0]} to {dates[1]}")
    print(f"Variables: {', '.join(variables)}")
    print(f"PET method: {pet_method}")
    
    # Fetch data from PyDaymet
    try:
        climate_data = pydaymet.get_bycoords(
            coords=coords,
            dates=dates,
            crs=crs,
            variables=variables,
            region=region,
            time_scale=time_scale,
            pet=pet_method,
            pet_params=pet_params
        )
        
        print(f"✓ Successfully fetched {len(climate_data)} records")
        
    except Exception as e:
        print(f"ERROR: Failed to fetch climate data: {str(e)}")
        raise
    
    # Convert to DataFrame if not already
    if not isinstance(climate_data, pd.DataFrame):
        climate_data = climate_data.to_dataframe()
    
    # Compute derived features
    print("\nComputing derived features...")
    
    # Antecedent precipitation totals
    if 'prcp' in climate_data.columns:
        for days in antecedent_days:
            col_name = f'prcp_antecedent_{days}d'
            climate_data[col_name] = climate_data['prcp'].rolling(
                window=days, min_periods=1
            ).sum()
            print(f"  ✓ {col_name}")
    
    # P-PET (Precipitation minus Potential Evapotranspiration)
    pet_cols = [col for col in climate_data.columns if 'pet' in col.lower()]
    if pet_cols and 'prcp' in climate_data.columns:
        for pet_col in pet_cols:
            p_minus_pet_col = f'p_minus_{pet_col}'
            climate_data[p_minus_pet_col] = climate_data['prcp'] - climate_data[pet_col]
            print(f"  ✓ {p_minus_pet_col}")
    
    # Cumulative precipitation
    if 'prcp' in climate_data.columns:
        climate_data['prcp_cumulative'] = climate_data['prcp'].cumsum()
        print(f"  ✓ prcp_cumulative")
    
    # Temperature range
    if 'tmin' in climate_data.columns and 'tmax' in climate_data.columns:
        climate_data['temp_range'] = climate_data['tmax'] - climate_data['tmin']
        climate_data['temp_mean'] = (climate_data['tmax'] + climate_data['tmin']) / 2
        print(f"  ✓ temp_range, temp_mean")
    
    print(f"\n✓ Total columns in output: {len(climate_data.columns)}")
    
    return climate_data


def save_climate_data(
    climate_data: pd.DataFrame,
    output_path: str,
    metadata: Dict[str, Any] = None
):
    """
    Save climate data to CSV file with metadata.
    
    Args:
        climate_data: DataFrame with climate data
        output_path: Path to save CSV file
        metadata: Optional metadata to save alongside
    """
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save main data
    climate_data.to_csv(output_path, index=True)
    print(f"\n✓ Saved climate data to: {output_path}")
    print(f"  Shape: {climate_data.shape}")
    
    # Save metadata if provided
    if metadata:
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"✓ Saved metadata to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Fetch climate data using PyDaymet and save as CSV'
    )
    
    # Input options
    parser.add_argument('--config', type=str, help='Path to JSON config file')
    parser.add_argument('--coords', nargs=2, type=float, 
                       help='Coordinates as: longitude latitude')
    parser.add_argument('--dates', nargs=2, type=str,
                       help='Date range as: start_date end_date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='climate_data.csv',
                       help='Output CSV file path')
    
    # Optional parameters
    parser.add_argument('--crs', type=int, default=4326,
                       help='Coordinate reference system (default: 4326)')
    parser.add_argument('--variables', nargs='+', 
                       default=['prcp', 'tmin', 'tmax', 'srad', 'vp', 'dayl'],
                       help='Variables to fetch')
    parser.add_argument('--pet-method', type=str, default='penman_monteith',
                       choices=['penman_monteith', 'priestley_taylor', 'hargreaves_samani'],
                       help='PET calculation method')
    parser.add_argument('--time-scale', type=str, default='daily',
                       choices=['daily', 'monthly', 'annual'],
                       help='Time scale for data')
    parser.add_argument('--region', type=str, default='na',
                       choices=['na', 'hi', 'pr'],
                       help='Region (na=North America, hi=Hawaii, pr=Puerto Rico)')
    parser.add_argument('--antecedent-days', nargs='+', type=int,
                       default=[1, 3, 7, 14],
                       help='Days for antecedent totals')
    parser.add_argument('--arid-correction', action='store_true',
                       help='Apply arid correction for PET')
    
    args = parser.parse_args()
    
    # Load config from file if provided
    if args.config:
        print(f"Loading configuration from: {args.config}")
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        coords = tuple(config['coords'])
        dates = tuple(config['dates'])
        output_path = config.get('output', args.output)
        crs = config.get('crs', args.crs)
        variables = config.get('variables', args.variables)
        pet_method = config.get('pet_method', args.pet_method)
        pet_params = config.get('pet_params', {})
        time_scale = config.get('time_scale', args.time_scale)
        region = config.get('region', args.region)
        antecedent_days = config.get('antecedent_days', args.antecedent_days)
        
    else:
        # Use command-line arguments
        if not args.coords or not args.dates:
            parser.error("Either --config or both --coords and --dates are required")
        
        coords = tuple(args.coords)
        dates = tuple(args.dates)
        output_path = args.output
        crs = args.crs
        variables = args.variables
        pet_method = args.pet_method
        pet_params = {'arid_correction': args.arid_correction}
        time_scale = args.time_scale
        region = args.region
        antecedent_days = args.antecedent_days
    
    # Print configuration
    print("=" * 70)
    print("Climate Data Fetcher - Standalone Script")
    print("=" * 70)
    print(f"Environment: NumPy {np.__version__}, pandas {pd.__version__}")
    print(f"PyDaymet version: {pydaymet.__version__}")
    print("=" * 70)
    
    # Fetch climate data
    try:
        climate_data = fetch_climate_data(
            coords=coords,
            dates=dates,
            crs=crs,
            variables=variables,
            pet_method=pet_method,
            pet_params=pet_params,
            time_scale=time_scale,
            region=region,
            antecedent_days=antecedent_days
        )
        
        # Prepare metadata
        metadata = {
            'coords': coords,
            'dates': dates,
            'crs': crs,
            'variables': variables,
            'pet_method': pet_method,
            'pet_params': pet_params,
            'time_scale': time_scale,
            'region': region,
            'antecedent_days': antecedent_days,
            'shape': climate_data.shape,
            'columns': list(climate_data.columns),
            'date_range': [str(climate_data.index.min()), str(climate_data.index.max())],
            'fetched_at': pd.Timestamp.now().isoformat()
        }
        
        # Save to CSV
        save_climate_data(climate_data, output_path, metadata)
        
        # Print summary statistics
        print("\n" + "=" * 70)
        print("Summary Statistics:")
        print("=" * 70)
        print(climate_data.describe())
        
        print("\n✓ Climate data fetch completed successfully!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
