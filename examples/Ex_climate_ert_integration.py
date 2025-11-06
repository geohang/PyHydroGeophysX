"""
Example: Climate Data Integration with ERT Analysis
===================================================

This example demonstrates how to use the ClimateDataAgent to fetch
meteorological data (precipitation, temperature, ET) and integrate it
with ERT resistivity imaging for enhanced hydrologic analysis.

The workflow includes:
1. Fetching climate data using PyDaymet
2. Computing PET using multiple methods
3. Aligning climate data with ERT acquisition timestamps
4. Generating derived features (antecedent totals, P-PET)
5. Analyzing correlations between climate and resistivity changes

This integration enables better interpretation of resistivity changes
by relating them to rainfall events, drying periods, and water balance.
"""

import os
import sys
from datetime import datetime, timedelta

# Setup package path
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()

parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from PyHydroGeophysX.agents import ClimateDataAgent


def example_point_climate_data():
    """
    Example 1: Fetch climate data for ERT line coordinates (point data).
    
    This is useful when you have ERT line positions and want to get
    climate data at or near those locations.
    """
    print("=" * 80)
    print("Example 1: Point Climate Data for ERT Line")
    print("=" * 80)
    
    # Initialize the climate agent
    # Note: LLM API key is optional - only needed for AI-assisted analysis
    climate_agent = ClimateDataAgent()
    
    # Define ERT line coordinates (example: line at a site)
    # Coordinates should be in the specified CRS (default: EPSG:4326)
    # Note: Use (longitude, latitude) order for EPSG:4326
    ert_line_coords = [
        (-74.0060, 40.7128),  # Example: New York area (lon, lat)
        (-74.0058, 40.7130),
        (-74.0056, 40.7132),
    ]
    
    # Define date range for climate data
    # This should cover the ERT measurement campaign
    start_date = "2023-06-01"
    end_date = "2023-09-30"
    
    # Define ERT acquisition timestamps
    ert_timestamps = [
        "2023-06-15",
        "2023-07-15",
        "2023-08-15",
        "2023-09-15",
    ]
    
    print("\nFetching climate data for ERT line coordinates...")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Number of ERT timestamps: {len(ert_timestamps)}")
    
    # Configure climate data retrieval
    climate_config = {
        'coords': ert_line_coords[0],  # Use first point for simplicity
        'dates': (start_date, end_date),
        'crs': 4326,  # WGS84
        'variables': ['prcp', 'tmin', 'tmax', 'srad', 'vp', 'dayl'],
        'pet_method': 'penman_monteith',
        'pet_params': {
            'arid_correction': False,  # Set True for arid regions
            'soil_heat_flux': 0,
            'albedo': 0.23,
            'alpha': 1.26
        },
        'time_scale': 'daily',
        'region': 'na',
        'ert_timestamps': ert_timestamps,
        'antecedent_days': [1, 3, 7],  # Compute 1, 3, and 7-day antecedent totals
    }
    
    # Execute climate data retrieval
    results = climate_agent.execute(climate_config)
    
    # Display results
    print("\n" + climate_agent.get_climate_summary())
    
    # Access specific results
    if 'climate_data' in results:
        print("\nClimate data retrieved successfully!")
        climate_data = results['climate_data']
        print(f"Data shape: {climate_data.shape if hasattr(climate_data, 'shape') else 'N/A'}")
    
    if 'derived_features' in results:
        print("\nDerived features computed:")
        features = results['derived_features']
        for key in features:
            if key != 'enhanced_data':
                print(f"  - {key}")
    
    if 'ert_alignment' in results and results['ert_alignment']:
        print("\nClimate data aligned with ERT timestamps!")
        if 'ert_aligned_data' in results['ert_alignment']:
            aligned = results['ert_alignment']['ert_aligned_data']
            print(f"  Aligned data points: {len(aligned) if hasattr(aligned, '__len__') else 'N/A'}")
    
    return results


def example_gridded_climate_data():
    """
    Example 2: Fetch gridded climate data for a site polygon/bbox.
    
    This is useful when you have a site boundary and want gridded
    climate data covering the entire area.
    """
    print("\n" + "=" * 80)
    print("Example 2: Gridded Climate Data for Site Area")
    print("=" * 80)
    
    # Initialize the climate agent
    climate_agent = ClimateDataAgent()
    
    # Define site bounding box (minx, miny, maxx, maxy)
    # Example: Small area in Colorado
    site_bbox = (-105.3, 40.0, -105.2, 40.1)
    
    # Define date range
    start_date = "2023-07-01"
    end_date = "2023-07-31"
    
    print(f"\nFetching gridded climate data for site bbox: {site_bbox}")
    print(f"Date range: {start_date} to {end_date}")
    
    # Configure climate data retrieval for gridded data
    climate_config = {
        'geometry': site_bbox,
        'dates': (start_date, end_date),
        'crs': 4326,
        'variables': ['prcp', 'tmin', 'tmax', 'srad'],
        'pet_method': 'priestley_taylor',
        'pet_params': {
            'arid_correction': True,  # Colorado can be arid
            'soil_heat_flux': 0,
            'albedo': 0.23
        },
        'time_scale': 'daily',
        'region': 'na',
        'antecedent_days': [1, 3, 7],
    }
    
    # Execute climate data retrieval
    results = climate_agent.execute(climate_config)
    
    # Display results
    print("\n" + climate_agent.get_climate_summary())
    
    if 'climate_data' in results:
        climate_data = results['climate_data']
        print(f"\nGridded data type: {type(climate_data).__name__}")
        # If xarray dataset, show dimensions
        if hasattr(climate_data, 'dims'):
            print(f"Dimensions: {dict(climate_data.dims)}")
            print(f"Variables: {list(climate_data.data_vars)}")
    
    return results


def example_pet_method_comparison():
    """
    Example 3: Compare different PET calculation methods.
    
    This demonstrates the robustness of PET estimates by comparing
    Penman-Monteith, Priestley-Taylor, and Hargreaves-Samani methods.
    """
    print("\n" + "=" * 80)
    print("Example 3: PET Method Comparison")
    print("=" * 80)
    
    # Initialize the climate agent
    climate_agent = ClimateDataAgent()
    
    # Define location and date range
    coords = (-105.3, 40.0)  # Colorado example (lon, lat)
    start_date = "2023-06-01"
    end_date = "2023-08-31"
    
    print(f"\nComparing PET methods for location: {coords}")
    print(f"Date range: {start_date} to {end_date}")
    
    # List of PET methods to compare
    pet_methods = ['penman_monteith', 'priestley_taylor', 'hargreaves_samani']
    
    print(f"\nPET methods to compare: {', '.join(pet_methods)}")
    
    # Configure with multiple PET methods
    climate_config = {
        'coords': coords,
        'dates': (start_date, end_date),
        'crs': 4326,
        'variables': ['prcp', 'tmin', 'tmax', 'srad', 'vp', 'dayl'],
        'pet_method': pet_methods,  # List of methods triggers comparison
        'pet_params': {
            'arid_correction': True,
            'soil_heat_flux': 0,
            'albedo': 0.23,
            'alpha': 1.26
        },
        'time_scale': 'daily',
        'region': 'na',
    }
    
    # Execute with method comparison
    results = climate_agent.execute(climate_config)
    
    # Display results
    print("\n" + climate_agent.get_climate_summary())
    
    if 'pet_comparison' in results and results['pet_comparison']:
        comparison = results['pet_comparison']
        print("\nPET Method Comparison Results:")
        
        for method in pet_methods:
            if f'{method}_mean' in comparison:
                mean = comparison[f'{method}_mean']
                std = comparison[f'{method}_std']
                print(f"  {method}:")
                print(f"    Mean: {mean:.3f} mm/day")
                print(f"    Std Dev: {std:.3f} mm/day")
        
        if 'coefficient_of_variation' in comparison:
            cv = comparison['coefficient_of_variation']
            print(f"\n  Overall Coefficient of Variation: {cv:.3f}")
            print("  (Lower CV indicates more consistent PET estimates across methods)")
    
    return results


def example_climate_ert_event_analysis():
    """
    Example 4: Analyze rainfall events and resistivity response.
    
    This demonstrates event detection and analysis of resistivity
    changes following precipitation events.
    """
    print("\n" + "=" * 80)
    print("Example 4: Climate-ERT Event Analysis")
    print("=" * 80)
    
    # Initialize the climate agent
    climate_agent = ClimateDataAgent()
    
    # Define ERT site location
    site_coords = (-105.3, 40.0)  # (lon, lat)
    
    # Campaign dates
    start_date = "2023-05-01"
    end_date = "2023-09-30"
    
    # ERT measurement times (example: monthly surveys)
    ert_timestamps = [
        "2023-05-15 10:00:00",
        "2023-06-15 10:00:00",
        "2023-07-15 10:00:00",
        "2023-08-15 10:00:00",
        "2023-09-15 10:00:00",
    ]
    
    print(f"\nAnalyzing climate-ERT relationships for site: {site_coords}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Number of ERT surveys: {len(ert_timestamps)}")
    
    # Configure with event analysis features
    climate_config = {
        'coords': site_coords,
        'dates': (start_date, end_date),
        'crs': 4326,
        'variables': ['prcp', 'tmin', 'tmax', 'srad', 'vp', 'dayl'],
        'pet_method': 'penman_monteith',
        'pet_params': {'arid_correction': True},
        'time_scale': 'daily',
        'region': 'na',
        'ert_timestamps': ert_timestamps,
        'antecedent_days': [1, 3, 7, 14],  # Multiple time windows
    }
    
    # Execute
    results = climate_agent.execute(climate_config)
    
    # Display results
    print("\n" + climate_agent.get_climate_summary())
    
    print("\nEvent Analysis Features:")
    print("  - Antecedent precipitation totals (1, 3, 7, 14 days)")
    print("  - P-PET water balance proxy")
    print("  - Climate data aligned to ERT acquisition times")
    print("\nThese features can be used to:")
    print("  1. Detect infiltration events before ERT surveys")
    print("  2. Compute resistivity changes vs. precipitation")
    print("  3. Evaluate moisture response lag times")
    print("  4. Assess hysteresis effects across seasons")
    
    return results


def example_integrated_workflow():
    """
    Example 5: Complete integrated workflow with multi-agent system.
    
    This demonstrates how to use ClimateDataAgent with the full
    multi-agent workflow including ERT processing.
    """
    print("\n" + "=" * 80)
    print("Example 5: Integrated Climate + ERT Workflow")
    print("=" * 80)
    
    print("\nThis example shows how to integrate ClimateDataAgent")
    print("with the full multi-agent workflow:")
    print("\n  from PyHydroGeophysX.agents import (")
    print("      AgentCoordinator, ERTLoaderAgent, ERTInversionAgent,")
    print("      WaterContentAgent, ReportAgent, ClimateDataAgent")
    print("  )")
    print("\n  # Initialize coordinator")
    print("  coordinator = AgentCoordinator(output_dir='results/integrated')")
    print("\n  # Register all agents")
    print("  coordinator.register_agent('climate_data', ClimateDataAgent())")
    print("  coordinator.register_agent('ert_loader', ERTLoaderAgent())")
    print("  coordinator.register_agent('ert_inversion', ERTInversionAgent())")
    print("  coordinator.register_agent('water_content', WaterContentAgent())")
    print("  coordinator.register_agent('report', ReportAgent())")
    print("\n  # Configure workflow with climate data")
    print("  config = {")
    print("      'data_file': 'path/to/ert_data.dat',")
    print("      'use_climate': True,")
    print("      'climate_config': {")
    print("          'coords': (-105.3, 40.0),  # (lon, lat)")
    print("          'dates': ('2023-06-01', '2023-09-30'),")
    print("          'pet_method': 'penman_monteith',")
    print("      },")
    print("      'ert_timestamps': ['2023-06-15', '2023-07-15', ...],")
    print("      'inversion_params': {...},")
    print("  }")
    print("\n  # Execute complete workflow")
    print("  results = coordinator.execute_workflow(config)")
    print("\nThe report agent will include climate context for ERT analysis!")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Climate Data Agent Examples")
    print("Meteorological Data Integration with ERT Analysis")
    print("=" * 80)
    
    try:
        # Example 1: Point data
        example_point_climate_data()
        
        # Example 2: Gridded data
        example_gridded_climate_data()
        
        # Example 3: PET method comparison
        example_pet_method_comparison()
        
        # Example 4: Event analysis
        example_climate_ert_event_analysis()
        
        # Example 5: Integrated workflow
        example_integrated_workflow()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n⚠️  Error running examples: {str(e)}")
        print("\nNote: These examples require:")
        print("  1. PyDaymet installed: pip install pydaymet")
        print("  2. Internet connection for data retrieval")
        print("  3. Valid coordinates within Daymet coverage (North America)")


if __name__ == "__main__":
    main()
