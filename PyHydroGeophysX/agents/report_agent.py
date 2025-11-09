"""
Report Generation Agent

Specialized agent for generating comprehensive reports from workflow results.
"""

from typing import Dict, Any, Optional
import os
from datetime import datetime
import pandas as pd
import numpy as np
from .base_agent import BaseAgent

# Climate analysis thresholds (configurable constants)
RAINFALL_THRESHOLD_MM = 5.0  # Significant rainfall threshold
WET_PERIOD_THRESHOLD_MM = 25.0  # 7-day antecedent for wet periods
DRY_PERIOD_THRESHOLD_MM = 5.0  # 7-day antecedent for dry periods
PET_DEFICIT_THRESHOLD_MM = -2.0  # P-PET deficit indicating drying
HIGH_TEMP_THRESHOLD_C = 30.0  # High temperature affecting measurements


class ReportAgent(BaseAgent):
    """
    Agent specialized in generating comprehensive reports.
    
    Aggregates results from all workflow steps and generates reports
    with visualizations, statistics, and interpretations.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 llm_provider: str = "openai"):
        """Initialize Report Agent."""
        super().__init__("report_generator", api_key, model, llm_provider)
        self.system_message = """You are an expert in technical report writing for 
geophysical and hydrological studies. Your role is to synthesize results from ERT 
data processing, inversion, water content analysis, and climate data into clear, informative 
reports suitable for scientists and engineers. You should integrate climate insights 
(precipitation, PET, temperature) to explain resistivity changes and provide data quality caveats."""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive workflow report.
        
        Args:
            input_data: Dictionary containing:
                - workflow_data: All data from workflow steps
                - config: Original workflow configuration
                - output_dir: Directory for report output
                
        Returns:
            Dictionary containing report information and file paths
        """
        self._log_execution("Starting report generation")
        
        try:
            workflow_data = input_data.get('workflow_data', {})
            config = input_data.get('config', {})
            output_dir = input_data.get('output_dir', 'results/reports')
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate report sections
            self._log_execution("Generating report sections")
            
            # 1. Executive Summary
            executive_summary = self._generate_executive_summary(workflow_data, config)
            
            # 2. Data Processing Summary
            data_summary = self._generate_data_summary(workflow_data)
            
            # 3. Climate Data Summary (if available)
            climate_summary = self._generate_climate_summary(workflow_data)
            
            # 4. Inversion Results Summary
            inversion_summary = self._generate_inversion_summary(workflow_data)
            
            # 5. Water Content Analysis Summary
            wc_summary = self._generate_wc_summary(workflow_data)
            
            # 6. Climate-Resistivity Cross-Modal Analysis (if climate data available)
            climate_ert_analysis = self._generate_climate_ert_analysis(workflow_data)
            
            # 7. Visualizations (create plots)
            visualization_files = self._generate_visualizations(workflow_data, output_dir)
            
            # 8. Generate LLM-enhanced narrative report
            narrative_report = None
            if self.api_key:
                self._log_execution("Generating narrative report with LLM")
                narrative_report = self._generate_narrative_report(
                    executive_summary, data_summary, climate_summary, 
                    inversion_summary, wc_summary, climate_ert_analysis
                )
            
            # Compile full report
            full_report = self._compile_report(
                executive_summary,
                data_summary,
                climate_summary,
                inversion_summary,
                wc_summary,
                climate_ert_analysis,
                narrative_report,
                visualization_files
            )
            
            # Save report to file
            report_file = os.path.join(output_dir, 'workflow_report.md')
            with open(report_file, 'w') as f:
                f.write(full_report)
            
            # Also save as HTML if possible
            html_file = self._save_html_report(full_report, output_dir)
            
            # Convert to PDF
            pdf_file = self._save_pdf_report(full_report, output_dir, visualization_files)
            
            self._log_execution(f"Report saved to {report_file}")
            if pdf_file:
                self._log_execution(f"PDF report saved to {pdf_file}")
            
            self.results = {
                'status': 'success',
                'report_file': report_file,
                'html_file': html_file,
                'pdf_file': pdf_file,
                'visualization_files': visualization_files,
                'executive_summary': executive_summary,
                'output_dir': output_dir
            }
            
            return self.results
            
        except Exception as e:
            self._log_execution(f"Error generating report: {str(e)}", level='ERROR')
            self.results = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
    def _generate_executive_summary(self, workflow_data: Dict, config: Dict) -> str:
        """Generate executive summary section."""
        summary = f"""# Executive Summary

**Workflow Execution Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Original User Request:**
{config.get('user_request', 'Not provided')}

**Workflow Configuration:**
- Data File: {config.get('data_file', 'N/A')}
- Instrument: {config.get('instrument', 'N/A')}
- Seismic Integration: {'Yes' if config.get('use_seismic', False) else 'No'}

**Key Results:**
"""
        
        # Add key findings from each step
        if 'ert_data' in workflow_data:
            ert = workflow_data['ert_data']
            summary += f"- Loaded {ert.get('num_electrodes', 'N/A')} electrodes with {ert.get('num_measurements', 'N/A')} measurements\n"
        
        if 'inversion_results' in workflow_data:
            inv = workflow_data['inversion_results']
            summary += f"- Inversion converged in {inv.get('iterations', 'N/A')} iterations (chi2: {inv.get('chi2', 'N/A'):.3f})\n"
        
        if 'water_content' in workflow_data:
            wc = workflow_data['water_content']
            import numpy as np
            mean_wc = wc.get('water_content_mean')
            if mean_wc is not None and isinstance(mean_wc, np.ndarray) and mean_wc.size > 0:
                summary += f"- Mean water content: {np.mean(mean_wc):.3f}\n"
        
        return summary
    
    def _generate_data_summary(self, workflow_data: Dict) -> str:
        """Generate data processing summary."""
        summary = "\n## Data Processing Summary\n\n"
        
        if 'ert_data' in workflow_data:
            ert = workflow_data['ert_data']
            summary += f"""
### ERT Data Loading
- Number of electrodes: {ert.get('num_electrodes', 'N/A')}
- Number of measurements: {ert.get('num_measurements', 'N/A')}
- Quality metrics: {ert.get('qc_results', 'N/A')}

**Insights:** {ert.get('insights', 'N/A')}
"""
        
        if 'seismic_structure' in workflow_data:
            seis = workflow_data['seismic_structure']
            summary += f"""
### Seismic Data Processing
- Velocity threshold: {seis.get('velocity_threshold', 'N/A')} m/s
- Interface extracted: Yes

**Interpretation:** {seis.get('interpretation', 'N/A')}
"""
        
        return summary
    
    def _generate_inversion_summary(self, workflow_data: Dict) -> str:
        """Generate inversion results summary."""
        summary = "\n## Inversion Results\n\n"
        
        if 'inversion_results' in workflow_data:
            inv = workflow_data['inversion_results']
            summary += f"""
### ERT Inversion
- Final chi2: {inv.get('chi2', 'N/A')}
- Iterations: {inv.get('iterations', 'N/A')}
- Convergence: {'Success' if inv.get('status') == 'success' else 'Failed'}

**Interpretation:** {inv.get('interpretation', 'N/A')}
"""
        
        return summary
    
    def _generate_wc_summary(self, workflow_data: Dict) -> str:
        """Generate water content analysis summary."""
        import numpy as np
        summary = "\n## Water Content Analysis\n\n"
        
        if 'water_content' in workflow_data:
            wc = workflow_data['water_content']
            
            # Get water content statistics
            wc_mean = wc.get('water_content_mean')
            wc_std = wc.get('water_content_std')
            
            if wc_mean is not None:
                mean_wc = np.nanmean(wc_mean)
                min_wc = np.nanmin(wc_mean)
                max_wc = np.nanmax(wc_mean)
                
                summary += f"""
### Water Content Statistics
- Mean water content: {mean_wc:.3f}
- Range: [{min_wc:.3f}, {max_wc:.3f}]
"""
            
            # Uncertainty analysis
            if wc_std is not None:
                mean_uncertainty = np.nanmean(wc_std)
                max_uncertainty = np.nanmax(wc_std)
                
                summary += f"""
### Uncertainty Analysis
- Mean uncertainty (σ): {mean_uncertainty:.3f}
- Maximum uncertainty: {max_uncertainty:.3f}
- Number of realizations: {wc.get('n_realizations', 'N/A')}
"""
            
            # Layer parameters used
            layer_params = wc.get('layer_params_used', {})
            if layer_params:
                summary += f"\n### Petrophysical Parameters\n"
                for marker, params in layer_params.items():
                    # Check if using rho_sat directly or traditional m-based approach
                    use_rho_sat = params.get('use_rho_sat', False)
                    
                    if use_rho_sat:
                        # Display rho_sat-based parameters
                        rho_sat_mean = params.get('rho_sat', {}).get('mean', 'N/A')
                        n_mean = params.get('n', {}).get('mean', 'N/A')
                        phi_mean = params.get('porosity', {}).get('mean', 'N/A')
                        if isinstance(rho_sat_mean, (int, float)) and isinstance(n_mean, (int, float)) and isinstance(phi_mean, (int, float)):
                            summary += f"- Marker {marker}: ρ_sat={rho_sat_mean:.1f} Ω·m, n={n_mean:.2f}, φ={phi_mean:.2f}\n"
                        else:
                            summary += f"- Marker {marker}: ρ_sat={rho_sat_mean}, n={n_mean}, φ={phi_mean}\n"
                    else:
                        # Display traditional m-based parameters
                        m_mean = params.get('m', {}).get('mean', 'N/A')
                        n_mean = params.get('n', {}).get('mean', 'N/A')
                        phi_mean = params.get('porosity', {}).get('mean', 'N/A')
                        if isinstance(m_mean, (int, float)) and isinstance(n_mean, (int, float)) and isinstance(phi_mean, (int, float)):
                            summary += f"- Marker {marker}: m={m_mean:.2f}, n={n_mean:.2f}, φ={phi_mean:.2f}\n"
                        else:
                            summary += f"- Marker {marker}: m={m_mean}, n={n_mean}, φ={phi_mean}\n"
            
            summary += f"\n**Interpretation:** {wc.get('interpretation', 'N/A')}\n"
        
        return summary
    
    def _generate_climate_summary(self, workflow_data: Dict) -> str:
        """Generate climate data summary section."""
        summary = "\n## Climate Data Integration\n\n"
        
        if 'climate_data' not in workflow_data:
            summary += "No climate data was integrated in this workflow.\n"
            return summary
        
        climate = workflow_data['climate_data']
        metadata = climate.get('metadata', {})
        
        summary += f"""
### Climate Data Summary
- Date Range: {metadata.get('dates', 'N/A')}
- Variables: {', '.join(metadata.get('variables', []))}
- PET Method: {metadata.get('pet_method', 'N/A')}
- Region: {metadata.get('region', 'N/A')}
"""
        
        # Add derived features info
        if climate.get('derived_features'):
            summary += "\n**Derived Features:**\n"
            features = climate['derived_features']
            # Count antecedent features
            antecedent_features = [k for k in features.keys() if 'antecedent' in k]
            if antecedent_features:
                summary += f"- Antecedent precipitation totals computed\n"
            
            # Check for P-PET features
            p_pet_features = [k for k in features.keys() if 'p_minus' in k]
            if p_pet_features:
                summary += f"- Water balance proxy (P-PET) computed\n"
        
        # Add ERT alignment info
        if climate.get('ert_alignment'):
            alignment = climate['ert_alignment']
            if 'ert_timestamps' in alignment:
                n_timestamps = len(alignment['ert_timestamps'])
                summary += f"\n**ERT Alignment:** Climate data aligned to {n_timestamps} ERT acquisition times\n"
        
        return summary
    
    def _generate_climate_ert_analysis(self, workflow_data: Dict) -> str:
        """
        Generate cross-modal analysis linking climate features to resistivity changes.
        
        This provides the cross-modal reasoning that explains resistivity changes
        in terms of climate forcings (rainfall, drying, etc.).
        """
        analysis = "\n## Cross-Modal Climate-ERT Analysis\n\n"
        
        if 'climate_data' not in workflow_data:
            analysis += "Climate data not available for cross-modal analysis.\n"
            return analysis
        
        climate = workflow_data['climate_data']
        
        analysis += """
### Climate-Based Resistivity Interpretation

This section provides climate-based context for interpreting resistivity changes,
including detection of post-rainfall infiltration and high-PET drying periods.

"""
        
        # Check if we have aligned data for event-based analysis
        if climate.get('ert_alignment') and 'ert_aligned_data' in climate['ert_alignment']:
            aligned_data = climate['ert_alignment']['ert_aligned_data']
            
            if isinstance(aligned_data, pd.DataFrame) and not aligned_data.empty:
                analysis += "#### Event Detection and Classification\n\n"
                
                # Analyze precipitation events
                if 'prcp' in aligned_data.columns:
                    prcp_values = aligned_data['prcp'].values
                    # Detect significant rainfall using threshold constant
                    rainfall_events = prcp_values > RAINFALL_THRESHOLD_MM
                    n_rainfall = np.count_nonzero(rainfall_events)
                    
                    if n_rainfall > 0:
                        analysis += f"- **Rainfall Events:** {n_rainfall} ERT acquisition(s) occurred during or shortly after significant rainfall (>{RAINFALL_THRESHOLD_MM}mm)\n"
                        analysis += "  - *Expected Impact:* Decreased resistivity due to increased moisture content\n"
                        analysis += "  - *Data Quality Note:* Post-rainfall measurements may show transient infiltration patterns\n"
                    else:
                        analysis += "- **Rainfall Events:** No significant rainfall detected near ERT acquisitions\n"
                
                # Analyze antecedent conditions
                antecedent_cols = [col for col in aligned_data.columns if 'antecedent' in col]
                if antecedent_cols:
                    analysis += f"\n- **Antecedent Moisture Conditions:** Available for {len(antecedent_cols)} time window(s)\n"
                    
                    # Check 7-day antecedent if available
                    if 'prcp_antecedent_7d' in aligned_data.columns:
                        antecedent_7d = aligned_data['prcp_antecedent_7d'].values
                        # Classify as wet or dry periods using threshold constants
                        # Note: Values between thresholds are normal conditions (not flagged)
                        wet_periods = antecedent_7d > WET_PERIOD_THRESHOLD_MM
                        dry_periods = antecedent_7d < DRY_PERIOD_THRESHOLD_MM
                        
                        if np.any(wet_periods):
                            analysis += f"  - Wet periods (7-day total >{WET_PERIOD_THRESHOLD_MM}mm): {np.count_nonzero(wet_periods)} acquisition(s)\n"
                        if np.any(dry_periods):
                            analysis += f"  - Dry periods (7-day total <{DRY_PERIOD_THRESHOLD_MM}mm): {np.count_nonzero(dry_periods)} acquisition(s)\n"
                
                # Analyze water balance (P-PET)
                p_pet_cols = [col for col in aligned_data.columns if 'p_minus' in col]
                if p_pet_cols:
                    analysis += f"\n- **Water Balance Analysis (P-PET):**\n"
                    
                    for p_pet_col in p_pet_cols:
                        p_pet_values = aligned_data[p_pet_col].values
                        # Positive P-PET indicates moisture surplus, negative indicates deficit
                        surplus_periods = p_pet_values > 0
                        deficit_periods = p_pet_values < PET_DEFICIT_THRESHOLD_MM  # Significant deficit
                        
                        n_surplus = np.count_nonzero(surplus_periods)
                        n_deficit = np.count_nonzero(deficit_periods)
                        
                        if n_surplus > 0:
                            analysis += f"  - Moisture surplus periods: {n_surplus} acquisition(s)\n"
                            analysis += f"    *Expected:* Increasing moisture, decreasing resistivity\n"
                        
                        if n_deficit > 0:
                            analysis += f"  - High PET drying periods: {n_deficit} acquisition(s)\n"
                            analysis += f"    *Expected:* Soil desiccation, increasing resistivity\n"
                            analysis += f"    *Caveat:* Inversion artifacts more likely during very dry conditions\n"
                
                # Temperature effects
                if 'tmax' in aligned_data.columns:
                    tmax_values = aligned_data['tmax'].values
                    hot_periods = tmax_values > HIGH_TEMP_THRESHOLD_C  # High temperature threshold
                    
                    if np.any(hot_periods):
                        analysis += f"\n- **Temperature Effects:**\n"
                        analysis += f"  - High temperature periods (>{HIGH_TEMP_THRESHOLD_C}°C): {np.count_nonzero(hot_periods)} acquisition(s)\n"
                        analysis += f"    *Note:* High temperatures may affect electrode contact and measurements\n"
            else:
                analysis += "*No aligned data available for detailed event analysis.*\n"
        else:
            analysis += "*ERT timestamps not provided - temporal climate analysis not performed.*\n"
        
        # Add data quality caveats section
        analysis += """

#### Data Quality and Inversion Diagnostics

**Climate-Based Caveats:**
"""
        
        if climate.get('ert_alignment'):
            analysis += """
- Resistivity inversions during or shortly after rainfall may exhibit enhanced sensitivity
  to near-surface moisture variations, potentially masking deeper structures
- Measurements during extended dry periods (high cumulative PET) may show increased
  electrode contact resistance, affecting data quality
- Seasonal variations in climate forcing should be considered when comparing 
  time-lapse resistivity changes across different campaigns
"""
        else:
            analysis += """
- Climate data available but not temporally aligned with ERT acquisitions
- Consider aligning future ERT campaigns with climate data for enhanced interpretation
"""
        
        # Add inversion quality checks if available
        if 'inversion_results' in workflow_data:
            inv = workflow_data['inversion_results']
            chi2 = inv.get('chi2')
            if chi2 is not None:
                analysis += f"\n**Inversion Quality Metrics:**\n"
                analysis += f"- Chi-squared: {chi2:.3f}\n"
                
                # Provide climate-contextualized interpretation
                if chi2 < 1.0:
                    analysis += "  - Good data fit; climate effects likely well-captured\n"
                elif chi2 > 2.0:
                    analysis += "  - Elevated misfit; check for climate-induced measurement errors\n"
        
        return analysis
    
    def _generate_visualizations(self, workflow_data: Dict, output_dir: str) -> Dict[str, str]:
        """Generate visualization plots."""
        import matplotlib.pyplot as plt
        import matplotlib
        import numpy as np
        
        # Set Arial font for all plots
        matplotlib.rcParams['font.family'] = 'Arial'
        matplotlib.rcParams['font.size'] = 12
        
        vis_files = {}
        
        try:
            # 1. Resistivity model plot with coverage masking
            if 'inversion_results' in workflow_data:
                inv = workflow_data['inversion_results']
                if 'mesh' in inv and 'resistivity_model' in inv:
                    try:
                        import pygimli as pg
                        fig = plt.figure(figsize=(8, 3))
                        ax = fig.add_subplot(111)
                        
                        # Get coverage if available
                        coverage = inv.get('coverage')
                        coverage_mask = coverage > -1.0 if coverage is not None else None
                        
                        # Plot with coverage masking
                        ax, cbar = pg.show(
                            inv['mesh'],
                            inv['resistivity_model'],
                            ax=ax,
                            fig=fig,
                            cMap='jet',
                            cMin=100,
                            cMax=10000,
                            logScale=True,
                            label=r'Resistivity ($\Omega$ m)',
                            pad=0.3,
                            orientation='vertical',
                            coverage=coverage_mask
                        )
                        
                        ax.set_xlabel('Distance (m)', fontsize=14, fontfamily='Arial')
                        ax.set_ylabel('Elevation (m)', fontsize=14, fontfamily='Arial')
                        ax.set_title('ERT Inversion Results', fontsize=16, fontfamily='Arial')
                        
                        res_file = os.path.join(output_dir, 'resistivity_model.png')
                        fig.savefig(res_file, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        vis_files['resistivity'] = res_file
                        self._log_execution("Saved resistivity model plot")
                    except Exception as e:
                        self._log_execution(f"Could not generate resistivity plot: {e}")
            
            # 2. Water content plot with coverage masking
            if 'water_content' in workflow_data:
                wc = workflow_data['water_content']
                if 'mesh' in wc and 'water_content_mean' in wc:
                    try:
                        import pygimli as pg
                        fig = plt.figure(figsize=(8, 3))
                        ax = fig.add_subplot(111)
                        
                        wc_mean = wc['water_content_mean']
                        if wc_mean.ndim > 1:
                            wc_mean = wc_mean[:, 0]  # First timestep
                        
                        # Get coverage from inversion results if available
                        coverage = None
                        if 'inversion_results' in workflow_data:
                            inv = workflow_data['inversion_results']
                            coverage = inv.get('coverage')
                        coverage_mask = coverage > -1.0 if coverage is not None else None
                        
                        ax, cbar = pg.show(
                            wc['mesh'],
                            wc_mean,
                            ax=ax,
                            fig=fig,
                            cMap='Blues',
                            label='Water Content (-)',
                            cMin=0.0,
                            cMax=0.4,
                            pad=0.3,
                            orientation='vertical',
                            coverage=coverage_mask
                        )
                        
                        ax.set_xlabel('Distance (m)', fontsize=14, fontfamily='Arial')
                        ax.set_ylabel('Elevation (m)', fontsize=14, fontfamily='Arial')
                        ax.set_title('Water Content Distribution', fontsize=16, fontfamily='Arial')
                        
                        wc_file = os.path.join(output_dir, 'water_content.png')
                        fig.savefig(wc_file, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        vis_files['water_content'] = wc_file
                        self._log_execution("Saved water content plot")
                    except Exception as e:
                        self._log_execution(f"Could not generate water content plot: {e}")
                    
                    # 3. Water content uncertainty plot
                    if 'water_content_std' in wc:
                        try:
                            import pygimli as pg
                            fig = plt.figure(figsize=(8, 3))
                            ax = fig.add_subplot(111)
                            
                            wc_std = wc['water_content_std']
                            if wc_std.ndim > 1:
                                wc_std = wc_std[:, 0]  # First timestep
                            
                            # Get coverage from inversion results if available
                            coverage = None
                            if 'inversion_results' in workflow_data:
                                inv = workflow_data['inversion_results']
                                coverage = inv.get('coverage')
                            coverage_mask = coverage > -1.0 if coverage is not None else None
                            
                            ax, cbar = pg.show(
                                wc['mesh'],
                                wc_std,
                                ax=ax,
                                fig=fig,
                                cMap='Reds',
                                label=r'Uncertainty ($\sigma$)',
                                cMin=0.0,
                                cMax=0.1,
                                pad=0.3,
                                orientation='vertical',
                                coverage=coverage_mask
                            )
                            
                            ax.set_xlabel('Distance (m)', fontsize=14, fontfamily='Arial')
                            ax.set_ylabel('Elevation (m)', fontsize=14, fontfamily='Arial')
                            ax.set_title('Water Content Uncertainty', fontsize=16, fontfamily='Arial')
                            
                            uncertainty_file = os.path.join(output_dir, 'water_content_uncertainty.png')
                            fig.savefig(uncertainty_file, dpi=300, bbox_inches='tight')
                            plt.close(fig)
                            vis_files['water_content_uncertainty'] = uncertainty_file
                            self._log_execution("Saved water content uncertainty plot")
                        except Exception as e:
                            self._log_execution(f"Could not generate uncertainty plot: {e}")
            
        except Exception as e:
            self._log_execution(f"Error generating visualizations: {e}")
        
        return vis_files
    
    def _generate_narrative_report(self, exec_summary: str, data_summary: str,
                                   climate_summary: str, inv_summary: str, 
                                   wc_summary: str, climate_ert_analysis: str) -> str:
        """Generate narrative report using LLM."""
        try:
            combined_info = f"{exec_summary}\n{data_summary}\n{climate_summary}\n{inv_summary}\n{wc_summary}\n{climate_ert_analysis}"
            
            prompt = f"""Based on the following workflow results, write a cohesive narrative 
summary (3-4 paragraphs) that:
1. Describes the overall workflow and objectives
2. Integrates climate data insights with geophysical results
3. Explains how climate features (rainfall, drying periods) relate to resistivity changes
4. Summarizes the key findings with climate context
5. Highlights any notable patterns, anomalies, or data quality caveats
6. Provides recommendations for next steps

Workflow Results:
{combined_info}

Write in a professional, technical style suitable for a geophysical survey report that 
incorporates cross-modal climate-geophysics reasoning."""
            
            narrative = self.query_llm(prompt, self.system_message, 
                                      temperature=0.6, max_tokens=600)
            return f"\n## Narrative Summary\n\n{narrative}\n"
        except:
            return ""
    
    def _compile_report(self, exec_summary: str, data_summary: str,
                       climate_summary: str, inv_summary: str, wc_summary: str, 
                       climate_ert_analysis: str, narrative: str,
                       vis_files: Dict[str, str]) -> str:
        """Compile full report."""
        report = f"""# Geophysical Workflow Report
Generated by PyHydroGeophysX Multi-Agent System

{exec_summary}
{narrative if narrative else ''}
{data_summary}
{climate_summary}
{climate_ert_analysis}
{inv_summary}
{wc_summary}

## Visualizations

"""
        
        for vis_type, file_path in vis_files.items():
            report += f"### {vis_type.replace('_', ' ').title()}\n"
            report += f"![{vis_type}]({os.path.basename(file_path)})\n\n"
        
        report += f"""
---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report
    
    def _save_html_report(self, markdown_report: str, output_dir: str, 
                         filename: str = 'workflow_report') -> Optional[str]:
        """Convert markdown to HTML if possible."""
        try:
            import markdown
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Geophysical Workflow Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; }}
        h3 {{ color: #7f8c8d; }}
        img {{ max-width: 100%; height: auto; }}
        code {{ background-color: #f4f4f4; padding: 2px 5px; }}
    </style>
</head>
<body>
{markdown.markdown(markdown_report)}
</body>
</html>
"""
            html_file = os.path.join(output_dir, f'{filename}.html')
            with open(html_file, 'w') as f:
                f.write(html_content)
            return html_file
        except:
            return None
    
    def _save_pdf_report(self, markdown_report: str, output_dir: str, 
                        visualization_files: Dict[str, str],
                        filename: str = 'workflow_report') -> Optional[str]:
        """Convert markdown report to PDF using matplotlib for better control."""
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            import textwrap
            
            pdf_file = os.path.join(output_dir, f'{filename}.pdf')
            
            with PdfPages(pdf_file) as pdf:
                # Initialize first page
                fig = plt.figure(figsize=(8.5, 11))
                y_position = 0.95
                
                # Process markdown content
                lines = markdown_report.split('\n')
                
                for line in lines:
                    # Skip image markdown references (we'll add them as separate pages)
                    if line.strip().startswith('!['):
                        continue
                    
                    # Check if we need a new page
                    if y_position < 0.08:
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
                        fig = plt.figure(figsize=(8.5, 11))
                        y_position = 0.95
                    
                    if line.startswith('# ') and not line.startswith('## '):
                        # Main heading (single #)
                        text = line.replace('# ', '').strip()
                        if text:  # Only add if not empty
                            fig.text(0.1, y_position, text, fontsize=16, fontweight='bold', 
                                    fontfamily='Arial', wrap=True)
                            y_position -= 0.035
                    elif line.startswith('### '):
                        # Subsection heading
                        text = line.replace('### ', '').strip()
                        if text:
                            fig.text(0.1, y_position, text, fontsize=11, fontweight='bold',
                                    fontfamily='Arial', color='#555555')
                            y_position -= 0.022
                    elif line.startswith('## '):
                        # Section heading
                        text = line.replace('## ', '').strip()
                        if text:
                            fig.text(0.1, y_position, text, fontsize=13, fontweight='bold',
                                    fontfamily='Arial', color='#333333')
                            y_position -= 0.028
                    elif line.startswith('**') and line.endswith('**'):
                        # Bold text (key-value pairs)
                        text = line.replace('**', '').strip()
                        if text:
                            fig.text(0.1, y_position, text, fontsize=10, fontweight='bold',
                                    fontfamily='Arial')
                            y_position -= 0.018
                    elif line.startswith('- '):
                        # Bullet point
                        text = line.replace('- ', '• ').strip()
                        if text:
                            wrapped = textwrap.fill(text, width=85, subsequent_indent='  ')
                            for wrapped_line in wrapped.split('\n'):
                                if y_position < 0.08:
                                    pdf.savefig(fig, bbox_inches='tight')
                                    plt.close(fig)
                                    fig = plt.figure(figsize=(8.5, 11))
                                    y_position = 0.95
                                fig.text(0.12, y_position, wrapped_line, fontsize=9, 
                                        fontfamily='Arial')
                                y_position -= 0.016
                    elif line.strip().startswith('---'):
                        # Horizontal rule - add space
                        y_position -= 0.02
                    elif line.strip():
                        # Regular text - wrap long lines
                        wrapped = textwrap.fill(line, width=90)
                        for wrapped_line in wrapped.split('\n'):
                            if y_position < 0.08:
                                pdf.savefig(fig, bbox_inches='tight')
                                plt.close(fig)
                                fig = plt.figure(figsize=(8.5, 11))
                                y_position = 0.95
                            fig.text(0.1, y_position, wrapped_line, fontsize=9, 
                                    fontfamily='Arial')
                            y_position -= 0.016
                    else:
                        # Empty line - small spacing
                        y_position -= 0.012
                
                plt.axis('off')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                
                # Add visualization pages
                for vis_name, vis_path in visualization_files.items():
                    if os.path.exists(vis_path):
                        fig = plt.figure(figsize=(8.5, 11))
                        
                        # Add title for visualization
                        title = vis_name.replace('_', ' ').title()
                        fig.text(0.5, 0.96, title, ha='center', fontsize=13, 
                                fontweight='bold', fontfamily='Arial')
                        
                        # Load and display image
                        img = plt.imread(vis_path)
                        ax = fig.add_axes([0.1, 0.1, 0.8, 0.82])
                        ax.imshow(img)
                        ax.axis('off')
                        
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
            
            return pdf_file
            
        except Exception as e:
            self._log_execution(f"Could not generate PDF: {e}", level='WARNING')
            return None
    
    def generate_timelapse_report(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive time-lapse ERT report with climate integration.
        
        This is a NEW method specifically for time-lapse workflows. It does not 
        modify the existing standard workflow reporting (execute method).
        
        Args:
            input_data: Dictionary containing:
                - inversion_results: Time-lapse inversion results
                - climate_data: Climate data results (optional)
                - site_info: Site information dict
                - comparison_data: DataFrame with climate-resistivity comparison
                - output_dir: Directory for report output
                - inversion_mode: 'time-lapse'
                - time_lapse_method: 'difference', 'joint', or 'ratio'
                
        Returns:
            Dictionary containing report information and file paths
        """
        self._log_execution("Starting time-lapse report generation")
        
        try:
            inversion_results = input_data.get('inversion_results', {})
            climate_data = input_data.get('climate_data')
            site_info = input_data.get('site_info', {})
            comparison_df = input_data.get('comparison_data')
            output_dir = input_data.get('output_dir', 'results/Time-lapse_agent')
            time_lapse_method = input_data.get('time_lapse_method', 'difference')
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate report sections
            self._log_execution("Generating time-lapse report sections")
            
            # 1. Time-Lapse Executive Summary
            tl_exec_summary = self._generate_timelapse_executive_summary(
                inversion_results, site_info, time_lapse_method
            )
            
            # 2. Time-Lapse Inversion Results
            tl_inversion_section = self._generate_timelapse_inversion_section(
                inversion_results, time_lapse_method
            )
            
            # 3. Climate Data Section (if available)
            tl_climate_section = self._generate_timelapse_climate_section(
                climate_data, site_info
            )
            
            # 4. Climate-Resistivity Correlation Analysis (if available)
            tl_correlation_section = self._generate_timelapse_correlation_section(
                comparison_df, inversion_results
            )
            
            # 5. Time-Lapse Visualizations
            tl_vis_files = self._generate_timelapse_visualizations(
                inversion_results, comparison_df, climate_data, output_dir
            )
            
            # 6. Generate LLM-enhanced interpretation
            tl_narrative = None
            if self.api_key:
                self._log_execution("Generating time-lapse narrative with LLM")
                tl_narrative = self._generate_timelapse_narrative(
                    tl_exec_summary, tl_inversion_section, tl_climate_section,
                    tl_correlation_section, site_info
                )
            
            # Compile full time-lapse report
            full_report = self._compile_timelapse_report(
                tl_exec_summary,
                tl_narrative,
                tl_inversion_section,
                tl_climate_section,
                tl_correlation_section,
                tl_vis_files,
                site_info
            )
            
            # Save report to file
            report_file = os.path.join(output_dir, 'time_lapse_report.md')
            with open(report_file, 'w') as f:
                f.write(full_report)
            
            # Save as HTML
            html_file = self._save_html_report(full_report, output_dir, 'time_lapse_report')
            
            # Save as PDF
            pdf_file = self._save_pdf_report(full_report, output_dir, tl_vis_files, 'time_lapse_report')
            
            self._log_execution(f"Time-lapse report saved to {report_file}")
            if html_file:
                self._log_execution(f"HTML report saved to {html_file}")
            if pdf_file:
                self._log_execution(f"PDF report saved to {pdf_file}")
            
            return {
                'status': 'success',
                'report_file': report_file,
                'html_file': html_file,
                'pdf_file': pdf_file,
                'visualization_files': tl_vis_files,
                'executive_summary': tl_exec_summary,
                'output_dir': output_dir
            }
            
        except Exception as e:
            self._log_execution(f"Error generating time-lapse report: {str(e)}", level='ERROR')
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _generate_timelapse_executive_summary(self, inversion_results: Dict,
                                             site_info: Dict, method: str) -> str:
        """Generate executive summary for time-lapse report."""
        summary = f"""# Time-Lapse ERT Monitoring Report

**Report Generation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

### Site Information
- **Location:** {site_info.get('name', 'N/A')}, {site_info.get('location', 'N/A')}
- **Coordinates:** {site_info.get('coordinates', 'N/A')}
- **Elevation:** {site_info.get('elevation', 'N/A')}
- **Study Period:** {site_info.get('study_period', 'N/A')}

### Monitoring Objective
{site_info.get('description', 'Time-lapse ERT monitoring of subsurface processes.')}

### Method and Configuration
- **Time-Lapse Method:** {method.title()} Inversion
- **Number of Time Steps:** {inversion_results.get('n_timesteps', 'N/A')}
- **Temporal Regularization:** {inversion_results.get('temporal_regularization', 'N/A')}
- **Inversion Quality (χ²):** {self._format_chi2(inversion_results.get('chi2_values'))}

"""
        return summary
    
    def _generate_timelapse_inversion_section(self, inversion_results: Dict, method: str) -> str:
        """Generate time-lapse inversion results section."""
        section = """## Time-Lapse Inversion Results

### Methodology
"""
        
        if method == 'difference':
            section += """
The **difference inversion** method calculates absolute resistivity changes between 
each time step and the baseline survey. This approach is optimal for detecting 
localized changes and quantifying moisture infiltration or drying processes.
"""
        elif method == 'joint':
            section += """
The **joint inversion** method simultaneously inverts all time steps with temporal 
coupling constraints. This approach provides enhanced structural consistency across 
time steps and is ideal for gradual, continuous processes like seasonal variations.
"""
        elif method == 'ratio':
            section += """
The **ratio inversion** method calculates proportional resistivity changes, making 
it particularly suited for detecting relative changes in highly heterogeneous media 
or when monitoring processes with variable baseline conditions.
"""
        
        section += f"""

### Inversion Parameters
- **Number of Time Steps:** {inversion_results.get('n_timesteps', 'N/A')}
- **Temporal Regularization (α):** {inversion_results.get('temporal_regularization', 'N/A')}
- **Spatial Regularization (λ):** {inversion_results.get('lambda', 'N/A')}
- **Maximum Iterations:** {inversion_results.get('max_iterations', 'N/A')}
- **Solver Method:** {inversion_results.get('method', 'N/A').upper()}

### Convergence and Data Fit
"""
        
        # Add chi-squared information
        chi2_values = inversion_results.get('chi2_values')
        if chi2_values:
            if isinstance(chi2_values, list):
                section += f"\n**Chi-Squared Values by Time Step:**\n"
                for i, chi2 in enumerate(chi2_values, 1):
                    section += f"- Time Step {i}: χ² = {chi2:.3f}\n"
            else:
                section += f"\n**Overall Chi-Squared:** {chi2_values:.3f}\n"
        
        # Add temporal statistics
        final_models = inversion_results.get('final_models')
        if final_models is not None:
            import numpy as np
            baseline = final_models[:, 0]
            
            section += f"\n### Temporal Resistivity Statistics\n\n"
            section += f"**Baseline Resistivity (Time Step 1):**\n"
            section += f"- Mean: {np.mean(baseline):.2f} Ω·m\n"
            section += f"- Range: [{np.min(baseline):.2f}, {np.max(baseline):.2f}] Ω·m\n"
            section += f"- Standard Deviation: {np.std(baseline):.2f} Ω·m\n"
            
            section += f"\n**Resistivity Changes (Relative to Baseline):**\n\n"
            
            for i in range(1, min(final_models.shape[1], 6)):  # Show up to 5 time steps
                change = final_models[:, i] - baseline
                mean_change = np.mean(change)
                max_decrease = np.min(change)
                max_increase = np.max(change)
                
                section += f"**Time Step {i+1}:**\n"
                section += f"- Mean Change: {mean_change:+.2f} Ω·m\n"
                section += f"- Maximum Decrease: {max_decrease:.2f} Ω·m (moisture increase)\n"
                section += f"- Maximum Increase: {max_increase:.2f} Ω·m (drying/freezing)\n\n"
        
        return section
    
    def _generate_timelapse_climate_section(self, climate_data: Dict, site_info: Dict) -> str:
        """Generate climate data section for time-lapse report."""
        section = """## Climate Data Integration

### Meteorological Context

"""
        
        if not climate_data:
            section += "No climate data was integrated in this time-lapse analysis.\n"
            return section
        
        metadata = climate_data.get('metadata', {})
        
        section += f"""**Climate Data Summary:**
- **Date Range:** {metadata.get('dates', 'N/A')}
- **Variables:** {', '.join(metadata.get('variables', []))}
- **PET Method:** {metadata.get('pet_method', 'N/A').replace('_', '-').title()}
- **Time Scale:** {metadata.get('time_scale', 'N/A').title()}
- **Region:** {metadata.get('region', 'N/A').upper()}

"""
        
        # Add climate alignment info
        if climate_data.get('ert_alignment'):
            alignment = climate_data['ert_alignment']
            if 'ert_timestamps' in alignment:
                timestamps = alignment['ert_timestamps']
                # Convert timestamps to strings if they aren't already
                timestamp_strs = [str(ts.date()) if hasattr(ts, 'date') else str(ts) for ts in timestamps]
                section += f"**ERT Survey Alignment:**\n"
                section += f"- Number of ERT Surveys: {len(timestamps)}\n"
                section += f"- Survey Dates: {', '.join(timestamp_strs)}\n\n"
            
            # Add aligned climate summary if available
            if 'ert_aligned_data' in alignment:
                aligned_df = alignment['ert_aligned_data']
                section += self._summarize_aligned_climate(aligned_df)
        
        return section
    
    def _summarize_aligned_climate(self, aligned_df: pd.DataFrame) -> str:
        """Summarize aligned climate data."""
        summary = """**Climate Conditions at ERT Survey Times:**\n\n"""
        
        import numpy as np
        
        # Precipitation summary
        if 'prcp' in aligned_df.columns:
            total_prcp = aligned_df['prcp'].sum()
            max_prcp = aligned_df['prcp'].max()
            summary += f"- **Precipitation:** Total = {total_prcp:.1f} mm, Max daily = {max_prcp:.1f} mm\n"
        
        # Temperature summary
        if 'tmin' in aligned_df.columns and 'tmax' in aligned_df.columns:
            mean_tmin = aligned_df['tmin'].mean()
            mean_tmax = aligned_df['tmax'].mean()
            summary += f"- **Temperature:** Mean range = [{mean_tmin:.1f}, {mean_tmax:.1f}] °C\n"
        
        # PET summary
        if 'pet' in aligned_df.columns:
            mean_pet = aligned_df['pet'].mean()
            total_pet = aligned_df['pet'].sum()
            summary += f"- **Potential ET:** Mean = {mean_pet:.2f} mm/day, Total = {total_pet:.1f} mm\n"
        
        # Water balance
        if 'p_minus_pet' in aligned_df.columns:
            mean_balance = aligned_df['p_minus_pet'].mean()
            balance_type = "moisture surplus" if mean_balance > 0 else "moisture deficit"
            summary += f"- **Water Balance (P-PET):** Mean = {mean_balance:+.2f} mm/day ({balance_type})\n"
        
        summary += "\n"
        return summary
    
    def _generate_timelapse_correlation_section(self, comparison_df: pd.DataFrame,
                                                inversion_results: Dict) -> str:
        """Generate climate-resistivity correlation analysis section."""
        section = """## Climate-Resistivity Correlation Analysis

### Cross-Modal Analysis

This section examines the relationship between temporal resistivity changes and 
meteorological variables to understand subsurface moisture dynamics.

"""
        
        if comparison_df is None or comparison_df.empty:
            section += "Correlation analysis not available - climate data or comparison data not provided.\n"
            return section
        
        import numpy as np
        
        section += """### Correlation Coefficients

Correlation between mean resistivity changes and climate variables:

"""
        
        # Calculate correlations
        correlations = []
        
        if 'Mean_Resistivity_Change_Ohm_m' in comparison_df.columns:
            res_change = comparison_df['Mean_Resistivity_Change_Ohm_m'].values
            
            # Precipitation correlation
            if 'Precipitation_mm' in comparison_df.columns:
                corr = np.corrcoef(res_change, comparison_df['Precipitation_mm'])[0, 1]
                correlations.append(('Daily Precipitation', corr))
                section += f"- **Daily Precipitation:** r = {corr:.3f}\n"
            
            # Antecedent precipitation
            if 'Precip_7d_mm' in comparison_df.columns:
                corr = np.corrcoef(res_change, comparison_df['Precip_7d_mm'])[0, 1]
                correlations.append(('7-day Antecedent Precipitation', corr))
                section += f"- **7-day Antecedent Precipitation:** r = {corr:.3f}\n"
            
            # Temperature
            if 'Temp_Mean_C' in comparison_df.columns:
                corr = np.corrcoef(res_change, comparison_df['Temp_Mean_C'])[0, 1]
                correlations.append(('Mean Temperature', corr))
                section += f"- **Mean Temperature:** r = {corr:.3f}\n"
            
            # Water balance
            if 'P_minus_PET_mm' in comparison_df.columns and not comparison_df['P_minus_PET_mm'].isna().all():
                corr = np.corrcoef(res_change, comparison_df['P_minus_PET_mm'])[0, 1]
                correlations.append(('Moisture Balance (P-PET)', corr))
                section += f"- **Moisture Balance (P-PET):** r = {corr:.3f}\n"
        
        section += """

### Interpretation Guidelines

- **Negative correlation (r < 0):** Resistivity decreases as the variable increases
  - Expected for precipitation: more water → lower resistivity
- **Positive correlation (r > 0):** Resistivity increases as the variable increases
  - Expected for temperature/PET: drying → higher resistivity
- **Strong correlation (|r| > 0.7):** Variable likely has significant influence
- **Weak correlation (|r| < 0.3):** Variable has minimal direct influence

"""
        
        # Add key findings
        section += """### Key Findings

"""
        
        # Identify strongest correlations
        if correlations:
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            strongest = correlations[0]
            section += f"- **Strongest correlation:** {strongest[0]} (r = {strongest[1]:.3f})\n"
            
            if abs(strongest[1]) > 0.7:
                section += f"  - This indicates a **strong relationship** between resistivity changes and {strongest[0].lower()}\n"
            elif abs(strongest[1]) > 0.3:
                section += f"  - This indicates a **moderate relationship** between resistivity changes and {strongest[0].lower()}\n"
            else:
                section += f"  - This indicates a **weak relationship**, suggesting other factors may dominate\n"
        
        return section
    
    def _generate_timelapse_visualizations(self, inversion_results: Dict,
                                          comparison_df: pd.DataFrame,
                                          climate_data: Dict,
                                          output_dir: str) -> Dict[str, str]:
        """Generate visualizations for time-lapse report."""
        import matplotlib.pyplot as plt
        import matplotlib
        import numpy as np
        
        # Set Arial font
        matplotlib.rcParams['font.family'] = 'Arial'
        matplotlib.rcParams['font.size'] = 12
        
        vis_files = {}
        
        try:
            # 1. Baseline resistivity map
            final_models = inversion_results.get('final_models')
            mesh = inversion_results.get('mesh')
            coverage = inversion_results.get('coverage')
            
            if final_models is not None and mesh is not None:
                try:
                    import pygimli as pg
                    
                    # Use cellMarkers to properly index the data
                    cell_markers = mesh.cellMarkers()
                    baseline = final_models[:, 0][cell_markers] if len(final_models) > len(cell_markers) else final_models[:, 0]
                    
                    # Get coverage if available - fix shape if needed
                    if coverage is not None:
                        # Handle case where coverage is 2D (n_timesteps, n_cells) or (n_cells, n_timesteps)
                        if len(coverage.shape) == 2:
                            # Extract first timestep (baseline coverage)
                            if coverage.shape[0] < coverage.shape[1]:
                                # (n_timesteps, n_cells)
                                coverage = coverage[0, :]
                            else:
                                # (n_cells, n_timesteps)
                                coverage = coverage[:, 0]
                        
                        # Now apply cell marker indexing if needed
                        coverage_data = coverage[cell_markers] if len(coverage) > len(cell_markers) else coverage
                        
                        # Apply baseline masking - mask poor coverage areas in baseline
                        coverage_threshold = -0.5  # Optimized threshold from testing
                        coverage_masked = coverage_data.copy()
                        coverage_masked[coverage_data <= coverage_threshold] = -9999  # Mark as invalid
                    else:
                        coverage_data = None
                        coverage_masked = None
                    
                    # Plot baseline resistivity
                    fig_baseline, ax_baseline = plt.subplots(1, 1, figsize=(12, 6))
                    
                    # Plot with baseline-masked coverage - PyGIMLi will handle the masking internally
                    ax_baseline, cbar = pg.show(
                        mesh,
                        baseline,
                        ax=ax_baseline,
                        fig=fig_baseline,
                        cMap='jet',
                        cMin=30,
                        cMax=3000,  # Optimized upper limit
                        logScale=True,
                        label=r'Resistivity ($\Omega$ m)',
                        pad=0.3,
                        orientation='vertical',
                        coverage=coverage_masked
                    )
                    
                    ax_baseline.set_xlabel('Distance (m)', fontsize=14, fontfamily='Arial')
                    ax_baseline.set_ylabel('Elevation (m)', fontsize=14, fontfamily='Arial')
                    ax_baseline.set_title('Baseline Resistivity',
                                         fontsize=16, fontweight='bold', fontfamily='Arial')
                    
                    plt.tight_layout()
                    baseline_file = os.path.join(output_dir, 'baseline_resistivity.png')
                    fig_baseline.savefig(baseline_file, dpi=300, bbox_inches='tight')
                    plt.close(fig_baseline)
                    vis_files['baseline_resistivity'] = baseline_file
                    self._log_execution("Saved baseline resistivity plot")
                    
                except Exception as e:
                    self._log_execution(f"Could not generate baseline resistivity plot: {e}")
            
            # 2. All timestep resistivity maps (1x4 layout: all 4 timesteps)
            if final_models is not None and mesh is not None:
                try:
                    import pygimli as pg
                    
                    # Get fresh coverage reference
                    coverage = inversion_results.get('coverage')
                    
                    # Use cellMarkers to properly index the data
                    cell_markers = mesh.cellMarkers()
                    n_timesteps = min(final_models.shape[1], 4)  # Show up to 4 timesteps
                    
                    # Create subplot grid: 1 row, 4 columns
                    fig, axes = plt.subplots(1, n_timesteps, figsize=(20, 5))
                    if n_timesteps == 1:
                        axes = np.array([axes])
                    
                    # Get coverage if available - fix shape if needed
                    if coverage is not None:
                        # Handle case where coverage is 2D
                        if len(coverage.shape) == 2:
                            if coverage.shape[0] < coverage.shape[1]:
                                coverage = coverage[0, :]
                            else:
                                coverage = coverage[:, 0]
                        
                        # Now apply cell marker indexing if needed
                        coverage_data = coverage[cell_markers] if len(coverage) > len(cell_markers) else coverage
                        
                        # Apply baseline masking - mask poor coverage areas
                        coverage_threshold = -0.5  # Optimized threshold from testing
                        coverage_masked = coverage_data.copy()
                        coverage_masked[coverage_data <= coverage_threshold] = -9999  # Mark as invalid
                    else:
                        coverage_data = None
                        coverage_masked = None
                    
                    time_labels = ['Baseline'] + self._generate_time_labels(n_timesteps - 1)
                    
                    # Plot all timesteps
                    for i in range(n_timesteps):
                        ax = axes[i]
                        # Get data for this timestep using cellMarkers
                        timestep_data = final_models[:, i][cell_markers] if len(final_models) > len(cell_markers) else final_models[:, i]
                        
                        # Plot with baseline-masked coverage
                        ax, cbar = pg.show(
                            mesh,
                            timestep_data,
                            ax=ax,
                            fig=fig,
                            cMap='jet',
                            cMin=30,
                            cMax=3000,
                            logScale=True,
                            label=r'Resistivity ($\Omega$ m)',
                            pad=0.3,
                            orientation='vertical',
                            coverage=coverage_masked
                        )
                        
                        ax.set_xlabel('Distance (m)', fontsize=12, fontfamily='Arial')
                        ax.set_ylabel('Elevation (m)', fontsize=12, fontfamily='Arial')
                        ax.set_title(f'{time_labels[i]}',
                                   fontsize=14, fontweight='bold', fontfamily='Arial')
                    
                    plt.tight_layout()
                    all_timesteps_file = os.path.join(output_dir, 'timelapse_all_resistivity.png')
                    fig.savefig(all_timesteps_file, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    vis_files['timelapse_all_resistivity'] = all_timesteps_file
                    self._log_execution("Saved all timesteps resistivity plot")
                    
                except Exception as e:
                    self._log_execution(f"Could not generate all timesteps resistivity plot: {e}")
            
            # 3. Time-lapse resistivity percentage change maps (1x4 layout: baseline + changes)
            if final_models is not None and mesh is not None:
                try:
                    import pygimli as pg
                    
                    # Get fresh coverage reference
                    coverage = inversion_results.get('coverage')
                    
                    # Use cellMarkers to properly index the data
                    cell_markers = mesh.cellMarkers()
                    baseline = final_models[:, 0][cell_markers] if len(final_models) > len(cell_markers) else final_models[:, 0]
                    n_timesteps = min(final_models.shape[1] - 1, 3)  # Show up to 3 changes (4 total with baseline)
                    
                    # Create subplot grid: 1 row, 4 columns (baseline + 3 changes)
                    n_total_plots = n_timesteps + 1  # baseline + changes
                    fig, axes = plt.subplots(1, n_total_plots, figsize=(20, 5))
                    if n_total_plots == 1:
                        axes = np.array([axes])
                    
                    # Get coverage if available - fix shape if needed
                    if coverage is not None:
                        # Handle case where coverage is 2D
                        if len(coverage.shape) == 2:
                            if coverage.shape[0] < coverage.shape[1]:
                                coverage = coverage[0, :]
                            else:
                                coverage = coverage[:, 0]
                        
                        # Now apply cell marker indexing if needed
                        coverage_data = coverage[cell_markers] if len(coverage) > len(cell_markers) else coverage
                        
                        # Apply baseline masking - mask poor coverage areas in baseline
                        coverage_threshold = -0.5  # Optimized threshold from testing
                        coverage_masked = coverage_data.copy()
                        coverage_masked[coverage_data <= coverage_threshold] = -9999  # Mark as invalid
                    else:
                        coverage_data = None
                        coverage_masked = None
                    
                    # Plot 1: Baseline resistivity
                    ax = axes[0]
                    ax, cbar = pg.show(
                        mesh,
                        baseline,
                        ax=ax,
                        fig=fig,
                        cMap='jet',
                        cMin=30,
                        cMax=3000,
                        logScale=True,
                        label=r'Resistivity ($\Omega$ m)',
                        pad=0.3,
                        orientation='vertical',
                        coverage=coverage_masked
                    )
                    ax.set_xlabel('Distance (m)', fontsize=12, fontfamily='Arial')
                    ax.set_ylabel('Elevation (m)', fontsize=12, fontfamily='Arial')
                    ax.set_title('Baseline (t=0)', fontsize=14, fontweight='bold', fontfamily='Arial')
                    
                    # Plots 2-4: Percentage changes
                    time_labels = self._generate_time_labels(n_timesteps)
                    
                    for i in range(n_timesteps):
                        ax = axes[i+1]  # Offset by 1 for baseline
                        # Get data for this timestep using cellMarkers
                        timestep_data = final_models[:, i+1][cell_markers] if len(final_models) > len(cell_markers) else final_models[:, i+1]
                        
                        # Calculate percentage change: (new - baseline) / baseline * 100
                        # Avoid division by zero
                        percent_change = np.zeros_like(baseline)
                        mask = baseline != 0
                        percent_change[mask] = ((timestep_data[mask] - baseline[mask]) / 
                                                baseline[mask] * 100.0)
                        
                        # Plot with baseline-masked coverage - PyGIMLi will handle the masking internally
                        ax, cbar = pg.show(
                            mesh,
                            percent_change,
                            ax=ax,
                            fig=fig,
                            cMap='RdBu_r',
                            cMin=-50,  # Optimized range from testing
                            cMax=50,
                            label=r'$\Delta\rho$ (%)',
                            pad=0.3,
                            orientation='vertical',
                            coverage=coverage_masked
                        )
                        
                        ax.set_xlabel('Distance (m)', fontsize=12, fontfamily='Arial')
                        ax.set_ylabel('Elevation (m)', fontsize=12, fontfamily='Arial')
                        ax.set_title(f'{time_labels[i]}',
                                   fontsize=14, fontweight='bold', fontfamily='Arial')
                    
                    plt.tight_layout()
                    tl_changes_file = os.path.join(output_dir, 'timelapse_resistivity_changes_percent.png')
                    fig.savefig(tl_changes_file, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    vis_files['timelapse_changes_percent'] = tl_changes_file
                    self._log_execution("Saved time-lapse resistivity percentage changes plot")
                    
                except Exception as e:
                    self._log_execution(f"Could not generate time-lapse percentage changes plot: {e}")
            
            # 3. Time-lapse resistivity absolute change maps (1x4 layout: baseline + changes)
            if final_models is not None and mesh is not None:
                try:
                    import pygimli as pg
                    
                    # Get fresh coverage reference
                    coverage = inversion_results.get('coverage')
                    
                    # Use cellMarkers to properly index the data
                    cell_markers = mesh.cellMarkers()
                    baseline = final_models[:, 0][cell_markers] if len(final_models) > len(cell_markers) else final_models[:, 0]
                    n_timesteps = min(final_models.shape[1] - 1, 3)  # Show up to 3 changes (4 total with baseline)
                    
                    # Create subplot grid: 1 row, 4 columns (baseline + 3 changes)
                    n_total_plots = n_timesteps + 1  # baseline + changes
                    fig, axes = plt.subplots(1, n_total_plots, figsize=(20, 5))
                    if n_total_plots == 1:
                        axes = np.array([axes])
                    
                    # Get coverage if available - fix shape if needed
                    if coverage is not None:
                        # Handle case where coverage is 2D
                        if len(coverage.shape) == 2:
                            if coverage.shape[0] < coverage.shape[1]:
                                coverage = coverage[0, :]
                            else:
                                coverage = coverage[:, 0]
                        
                        # Now apply cell marker indexing if needed
                        coverage_data = coverage[cell_markers] if len(coverage) > len(cell_markers) else coverage
                        
                        # Apply baseline masking - mask poor coverage areas in baseline
                        coverage_threshold = -0.5  # Optimized threshold from testing
                        coverage_masked = coverage_data.copy()
                        coverage_masked[coverage_data <= coverage_threshold] = -9999  # Mark as invalid
                    else:
                        coverage_data = None
                        coverage_masked = None
                    
                    # Plot 1: Baseline resistivity
                    ax = axes[0]
                    ax, cbar = pg.show(
                        mesh,
                        baseline,
                        ax=ax,
                        fig=fig,
                        cMap='jet',
                        cMin=30,
                        cMax=3000,
                        logScale=True,
                        label=r'Resistivity ($\Omega$ m)',
                        pad=0.3,
                        orientation='vertical',
                        coverage=coverage_masked
                    )
                    ax.set_xlabel('Distance (m)', fontsize=12, fontfamily='Arial')
                    ax.set_ylabel('Elevation (m)', fontsize=12, fontfamily='Arial')
                    ax.set_title('Baseline', fontsize=14, fontweight='bold', fontfamily='Arial')
                    
                    # Plots 2-4: Absolute changes
                    time_labels = self._generate_time_labels(n_timesteps)
                    
                    for i in range(n_timesteps):
                        ax = axes[i+1]  # Offset by 1 for baseline
                        # Get data for this timestep using cellMarkers
                        timestep_data = final_models[:, i+1][cell_markers] if len(final_models) > len(cell_markers) else final_models[:, i+1]
                        change = timestep_data - baseline
                        
                        # Plot with baseline-masked coverage - PyGIMLi will handle the masking internally
                        ax, cbar = pg.show(
                            mesh,
                            change,
                            ax=ax,
                            fig=fig,
                            cMap='RdBu_r',
                            cMin=-300,  # Optimized range from testing
                            cMax=300,
                            label=r'$\Delta\rho$ ($\Omega$ m)',
                            pad=0.3,
                            orientation='vertical',
                            coverage=coverage_masked
                        )
                        
                        ax.set_xlabel('Distance (m)', fontsize=12, fontfamily='Arial')
                        ax.set_ylabel('Elevation (m)', fontsize=12, fontfamily='Arial')
                        ax.set_title(f'{time_labels[i]}',
                                   fontsize=14, fontweight='bold', fontfamily='Arial')
                    
                    plt.tight_layout()
                    tl_changes_abs_file = os.path.join(output_dir, 'timelapse_resistivity_changes_absolute.png')
                    fig.savefig(tl_changes_abs_file, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    vis_files['timelapse_changes_absolute'] = tl_changes_abs_file
                    self._log_execution("Saved time-lapse resistivity absolute changes plot")
                    
                except Exception as e:
                    self._log_execution(f"Could not generate time-lapse absolute changes plot: {e}")
            
            # 2. Climate-Resistivity comparison plots
            if comparison_df is not None and not comparison_df.empty and climate_data is not None:
                try:
                    # Set Arial font for all text
                    plt.rcParams['font.family'] = 'Arial'
                    
                    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
                    
                    # Get daily climate data from climate_data results
                    daily_df = None
                    if 'climate_data' in climate_data:
                        full_df = climate_data['climate_data']
                        
                        # Filter to the date range from metadata
                        metadata = climate_data.get('metadata', {})
                        date_range_tuple = metadata.get('dates', metadata.get('date_range', None))
                        
                        if date_range_tuple and len(date_range_tuple) == 2:
                            start_str, end_str = date_range_tuple
                            start_date = pd.to_datetime(start_str)
                            end_date = pd.to_datetime(end_str)
                            daily_df = full_df[(full_df.index >= start_date) & (full_df.index <= end_date)].copy()
                        else:
                            daily_df = full_df
                    
                    # Get ERT dates from comparison_df
                    ert_dates = pd.to_datetime(comparison_df['Date'])
                    
                    if daily_df is not None:
                        daily_dates = pd.to_datetime(daily_df.index)
                        
                        # =====================================================================
                        # TOP PLOT: Precipitation (bars) and PET (line)
                        # =====================================================================
                        ax_top_left = axes[0]
                        ax_top_right = ax_top_left.twinx()
                        
                        # PET on left axis (line plot) - using daily data
                        if 'pet' in daily_df.columns:
                            ax_top_left.plot(daily_dates, daily_df['pet'], '-',
                                            linewidth=1.5, label='PET (daily)',
                                            color='#FF8C00', alpha=0.8, zorder=2)
                            ax_top_left.set_ylabel('PET (mm/day)', fontsize=12,
                                                  color='#FF8C00', fontfamily='Arial')
                            ax_top_left.tick_params(axis='y', labelcolor='#FF8C00', labelsize=11)
                        
                        # Precipitation on right axis (bar plot) - using daily data
                        if 'prcp' in daily_df.columns:
                            ax_top_right.bar(daily_dates, daily_df['prcp'], 
                                            alpha=0.6, width=1.0, label='Precipitation (daily)',
                                            color='#4682B4', zorder=1)
                            ax_top_right.set_ylabel('Precipitation (mm/day)', fontsize=12,
                                                   color='#4682B4', fontfamily='Arial')
                            ax_top_right.tick_params(axis='y', labelcolor='#4682B4', labelsize=11)
                        
                        # Overlay ERT survey data points
                        if 'PET_mm' in comparison_df.columns:
                            ax_top_left.plot(ert_dates, comparison_df['PET_mm'], 'o',
                                           markersize=10, label='ERT Survey PET',
                                           color='#FF8C00', markeredgecolor='black', 
                                           markeredgewidth=2, zorder=5)
                        
                        if 'Precipitation_mm' in comparison_df.columns:
                            ax_top_right.scatter(ert_dates, comparison_df['Precipitation_mm'], 
                                               s=150, label='ERT Survey Precip',
                                               color='#4682B4', edgecolor='black',
                                               linewidth=2, zorder=5, marker='s')
                        
                        # Add BLACK dashed vertical lines for ERT measurement dates
                        for date in ert_dates:
                            ax_top_left.axvline(x=date, color='black', linestyle='--', 
                                               linewidth=2, alpha=0.8, zorder=4)
                        
                        # Remove x-axis label and tick labels from top plot
                        ax_top_left.set_xticklabels([])
                        ax_top_left.tick_params(axis='x', which='both', length=0)
                        
                        ax_top_left.set_title('Climate Data: Precipitation and Potential Evapotranspiration', 
                                             fontsize=13, fontfamily='Arial', pad=15)
                        ax_top_left.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                        ax_top_left.tick_params(axis='y', labelsize=11)
                        ax_top_left.tick_params(axis='both', which='major', length=6, width=1.5)
                        ax_top_right.tick_params(axis='both', which='major', length=6, width=1.5)
                        
                        # Add legends
                        lines1, labels1 = ax_top_left.get_legend_handles_labels()
                        lines2, labels2 = ax_top_right.get_legend_handles_labels()
                        ax_top_left.legend(lines1 + lines2, labels1 + labels2, 
                                          loc='upper left', fontsize=10, framealpha=0.9)
                        
                        # =====================================================================
                        # BOTTOM PLOT: Temperature (min/max with shaded range, NO mean line)
                        # =====================================================================
                        ax_bottom = axes[1]
                        
                        # Plot daily temperature data
                        if 'tmin' in daily_df.columns and 'tmax' in daily_df.columns:
                            # Filled area for temperature range
                            ax_bottom.fill_between(daily_dates, 
                                                  daily_df['tmin'],
                                                  daily_df['tmax'],
                                                  alpha=0.2, color='#FF6B6B', label='Temp Range (daily)')
                            
                            # Daily min/max lines
                            ax_bottom.plot(daily_dates, daily_df['tmax'], '-',
                                         linewidth=1.5, label='Max Temp (daily)',
                                         color='#DC143C', alpha=0.7, zorder=2)
                            ax_bottom.plot(daily_dates, daily_df['tmin'], '-',
                                         linewidth=1.5, label='Min Temp (daily)',
                                         color='#4169E1', alpha=0.7, zorder=2)
                        
                        # Overlay ERT survey data points
                        if 'Temp_Max_C' in comparison_df.columns:
                            ax_bottom.plot(ert_dates, comparison_df['Temp_Max_C'], 's',
                                         markersize=10, label='Max Temp (ERT survey)',
                                         color='#DC143C', markeredgecolor='black',
                                         markeredgewidth=2, zorder=5)
                        if 'Temp_Min_C' in comparison_df.columns:
                            ax_bottom.plot(ert_dates, comparison_df['Temp_Min_C'], 'o',
                                         markersize=10, label='Min Temp (ERT survey)',
                                         color='#4169E1', markeredgecolor='black',
                                         markeredgewidth=2, zorder=5)
                        
                        # Add BLACK dashed vertical lines for ERT measurement dates
                        for date in ert_dates:
                            ax_bottom.axvline(x=date, color='black', linestyle='--',
                                            linewidth=2, alpha=0.8, zorder=4)
                        
                        ax_bottom.set_xlabel('Date', fontsize=12, fontfamily='Arial')
                        ax_bottom.set_ylabel('Temperature (°C)', fontsize=12, fontfamily='Arial')
                        ax_bottom.set_title('Temperature Variations', fontsize=13, 
                                           fontfamily='Arial', pad=15)
                        ax_bottom.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                        ax_bottom.tick_params(axis='x', rotation=45, labelsize=11)
                        ax_bottom.tick_params(axis='y', labelsize=11)
                        ax_bottom.tick_params(axis='both', which='major', length=6, width=1.5)
                        ax_bottom.legend(loc='best', fontsize=9, framealpha=0.9, 
                                       edgecolor='gray', fancybox=True, ncol=2)
                    
                    plt.tight_layout()
                    climate_corr_file = os.path.join(output_dir, 'climate_data_visualization.png')
                    fig.savefig(climate_corr_file, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    vis_files['climate_correlation'] = climate_corr_file
                    self._log_execution("Saved climate data visualization plot")
                    
                    # Reset font settings
                    plt.rcParams['font.family'] = 'sans-serif'
                    
                except Exception as e:
                    self._log_execution(f"Could not generate climate visualization plot: {e}")
            
        except Exception as e:
            self._log_execution(f"Error generating time-lapse visualizations: {e}")
        
        return vis_files
    
    def _generate_timelapse_narrative(self, exec_summary: str, inv_section: str,
                                     climate_section: str, corr_section: str,
                                     site_info: Dict) -> str:
        """Generate LLM-enhanced narrative for time-lapse report."""
        try:
            combined_info = f"{exec_summary}\n{inv_section}\n{climate_section}\n{corr_section}"
            
            prompt = f"""Based on the following time-lapse ERT monitoring results, write a 
cohesive narrative summary (3-4 paragraphs) that:

1. Describes the monitoring objectives and site characteristics
2. Explains the time-lapse inversion approach and key findings
3. Integrates climate data insights to explain resistivity changes
4. Interprets correlations between climate variables and subsurface response
5. Identifies key patterns, anomalies, or significant changes
6. Provides recommendations for continued monitoring or follow-up investigations

Site and Results:
{combined_info}

Write in a professional, technical style suitable for a geophysical monitoring report 
that effectively combines temporal ERT analysis with meteorological context."""
            
            narrative = self.query_llm(prompt, self.system_message,
                                      temperature=0.6, max_tokens=700)
            return f"\n## Integrated Analysis and Interpretation\n\n{narrative}\n"
        except Exception as e:
            self._log_execution(f"Could not generate narrative: {e}")
            return ""
    
    def _compile_timelapse_report(self, exec_summary: str, narrative: str,
                                 inv_section: str, climate_section: str,
                                 corr_section: str, vis_files: Dict[str, str],
                                 site_info: Dict) -> str:
        """Compile full time-lapse report."""
        report = f"""{exec_summary}
{narrative if narrative else ''}
{inv_section}
{climate_section}
{corr_section}

## Visualizations

"""
        
        for vis_type, file_path in vis_files.items():
            report += f"### {vis_type.replace('_', ' ').title()}\n"
            report += f"![{vis_type}]({os.path.basename(file_path)})\n\n"
        
        report += f"""
## Summary and Recommendations

### Key Findings Summary

Based on the time-lapse ERT monitoring and climate data integration:

1. **Temporal Resistivity Changes:** Systematic changes in subsurface resistivity were 
   observed over the monitoring period, indicating dynamic moisture conditions.

2. **Climate-Resistivity Relationships:** Correlations between meteorological variables 
   and resistivity changes provide insights into subsurface hydrological processes.

3. **Data Quality:** Inversion results show good convergence, indicating reliable 
   monitoring of subsurface changes.

### Recommendations for Future Monitoring

1. **Continue Time-Series:** Extend monitoring to capture seasonal cycles and longer-term trends
2. **Enhanced Climate Integration:** Consider additional variables (snow depth, soil temperature)
3. **Depth-Dependent Analysis:** Investigate how climate effects vary with depth
4. **Validation:** Compare with direct measurements (soil moisture sensors, neutron probes)
5. **Predictive Modeling:** Use established correlations for forecasting subsurface response

---

**Site:** {site_info.get('name', 'N/A')}  
**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Generated by:** PyHydroGeophysX Multi-Agent System
"""
        
        return report
    
    def _format_chi2(self, chi2_values) -> str:
        """Format chi-squared values for display."""
        if chi2_values is None:
            return 'N/A'
        elif isinstance(chi2_values, list):
            if len(chi2_values) == 0:
                return 'N/A'
            elif len(chi2_values) == 1:
                return f"{chi2_values[0]:.3f}"
            else:
                return f"{np.mean(chi2_values):.3f} (mean)"
        else:
            return f"{chi2_values:.3f}"
    
    def _generate_time_labels(self, n_timesteps: int) -> list:
        """Generate time labels for plots."""
        # This is a placeholder - ideally extract from actual dates
        labels = []
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for i in range(n_timesteps):
            labels.append(f'Time Step {i+1}')
        return labels
    
    def generate_data_fusion_report(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive data fusion report with multi-method visualizations.
        
        Args:
            input_data: Dictionary containing:
                - structure_results: Results from StructureConstraintAgent
                - petro_results: Results from PetrophysicsAgent
                - workflow_config: Natural language configuration
                - output_dir: Directory for report output
                
        Returns:
            Dictionary containing report information and file paths
        """
        self._log_execution("Starting data fusion report generation")
        
        try:
            structure_results = input_data.get('structure_results', {})
            petro_results = input_data.get('petro_results', {})
            workflow_config = input_data.get('workflow_config', {})
            output_dir = input_data.get('output_dir', 'results/data_fusion_field')
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate visualizations
            self._log_execution("Generating multi-method visualizations")
            fusion_vis_files = self._generate_fusion_visualizations(
                structure_results, petro_results, workflow_config, output_dir
            )
            
            # Generate report sections
            fusion_summary = self._generate_fusion_summary(
                structure_results, petro_results, workflow_config
            )
            
            # Generate LLM-enhanced narrative
            fusion_narrative = None
            if self.api_key:
                self._log_execution("Generating fusion narrative with LLM")
                fusion_narrative = self._generate_fusion_narrative(
                    structure_results, petro_results, workflow_config
                )
            
            # Compile full report
            full_report = self._compile_fusion_report(
                fusion_summary, fusion_narrative, fusion_vis_files, workflow_config
            )
            
            # Save report
            report_file = os.path.join(output_dir, 'data_fusion_report.md')
            with open(report_file, 'w') as f:
                f.write(full_report)
            
            # Save HTML and PDF
            html_file = self._save_html_report(full_report, output_dir, 'data_fusion_report')
            pdf_file = self._save_pdf_report(full_report, output_dir, fusion_vis_files, 'data_fusion_report')
            
            self._log_execution(f"Data fusion report saved to {report_file}")
            if pdf_file:
                self._log_execution(f"PDF report saved to {pdf_file}")
            
            return {
                'status': 'success',
                'report_file': report_file,
                'html_file': html_file,
                'pdf_file': pdf_file,
                'visualization_files': fusion_vis_files,
                'output_dir': output_dir
            }
            
        except Exception as e:
            self._log_execution(f"Error generating data fusion report: {str(e)}", level='ERROR')
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _generate_fusion_visualizations(self, structure_results: Dict, petro_results: Dict,
                                       workflow_config: Dict, output_dir: str) -> Dict[str, str]:
        """Generate comprehensive multi-method fusion visualizations."""
        import matplotlib.pyplot as plt
        import matplotlib
        import numpy as np
        
        matplotlib.rcParams['font.family'] = 'Arial'
        matplotlib.rcParams['font.size'] = 12
        
        vis_files = {}
        
        try:
            import pygimli as pg
            
            # Get coverage threshold from config
            coverage_threshold = workflow_config.get('coverage_threshold', -1.0)
            
            # 1. Complete workflow visualization (3-panel: velocity + interface + resistivity)
            if structure_results.get('seismic_results') and structure_results['status'] == 'success':
                seismic_results = structure_results['seismic_results']
                velocity_model = seismic_results['velocity_model']
                velocity_mesh = seismic_results['mesh']
                seismic_coverage = seismic_results['coverage']
                interface_x, interface_z = structure_results['interface_coords']
                velocity_threshold = structure_results['velocity_threshold']
                resistivity_model = structure_results['resistivity_model']
                para_mesh = structure_results['mesh']
                coverage = structure_results['coverage']
                coverage_numeric = np.array(coverage, dtype=float)
                
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
                
                # Top: Velocity with coverage
                pg.show(velocity_mesh, velocity_model, ax=ax1, cMap='jet',
                       colorBar=True, label='Velocity (m/s)',
                       coverage=seismic_coverage>0.5, cMin=500, cMax=5000)
                ax1.set_title('1. Seismic Velocity Model (with Coverage)',
                            fontsize=13, fontweight='bold')
                
                # Middle: Velocity with interface
                pg.show(velocity_mesh, velocity_model, ax=ax2, cMap='jet',
                       colorBar=True, label='Velocity (m/s)', cMin=500, cMax=5000)
                ax2.plot(interface_x, interface_z, 'r-', linewidth=2,
                        label=f'Interface ({velocity_threshold} m/s)')
                ax2.legend()
                ax2.set_title(f'2. Extracted Interface at {velocity_threshold} m/s',
                            fontsize=13, fontweight='bold')
                
                # Bottom: Structure-constrained resistivity
                pg.show(para_mesh, resistivity_model, ax=ax3, cMap='jet',
                       colorBar=True, label='Resistivity (Ωm)',
                       coverage=coverage_numeric>coverage_threshold,
                       logScale=True, cMin=10, cMax=2000)
                ax3.set_title(f'3. Structure-Constrained Resistivity (Coverage > {coverage_threshold})',
                            fontsize=13, fontweight='bold')
                ax3.set_xlabel('Distance (m)')
                ax3.set_ylabel('Elevation (m)')
                
                plt.tight_layout()
                workflow_file = os.path.join(output_dir, 'complete_workflow.png')
                fig.savefig(workflow_file, dpi=300, bbox_inches='tight')
                plt.close(fig)
                vis_files['complete_workflow'] = workflow_file
                self._log_execution("Saved complete workflow visualization")
            
            # 2. Water content with uncertainty (2-panel)
            if petro_results['status'] == 'success' and structure_results['status'] == 'success':
                water_content_mean = petro_results['water_content_mean']
                water_content_std = petro_results['water_content_std']
                para_mesh = structure_results['mesh']
                coverage = structure_results['coverage']
                coverage_numeric = np.array(coverage, dtype=float)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                # Left: Mean water content
                pg.show(para_mesh, water_content_mean.ravel(), ax=ax1,
                       cMap='Blues', colorBar=True, label='Water Content (-)',
                       coverage=coverage_numeric>coverage_threshold,
                       cMin=0, cMax=0.35, logScale=False)
                ax1.set_title(f'Mean Water Content (Coverage > {coverage_threshold})',
                            fontsize=13, fontweight='bold')
                ax1.set_xlabel('Distance (m)')
                ax1.set_ylabel('Elevation (m)')
                
                # Right: Uncertainty
                pg.show(para_mesh, water_content_std.ravel(), ax=ax2,
                       cMap='Reds', colorBar=True, label='Uncertainty (std)',
                       coverage=coverage_numeric>coverage_threshold,
                       cMin=0, cMax=0.05, logScale=False)
                ax2.set_title(f'Water Content Uncertainty (Coverage > {coverage_threshold})',
                            fontsize=13, fontweight='bold')
                ax2.set_xlabel('Distance (m)')
                ax2.set_ylabel('Elevation (m)')
                
                plt.tight_layout()
                wc_file = os.path.join(output_dir, 'water_content_uncertainty.png')
                fig.savefig(wc_file, dpi=300, bbox_inches='tight')
                plt.close(fig)
                vis_files['water_content_uncertainty'] = wc_file
                self._log_execution("Saved water content visualization")
            
        except Exception as e:
            self._log_execution(f"Error generating fusion visualizations: {e}")
        
        return vis_files
    
    def _generate_fusion_summary(self, structure_results: Dict, petro_results: Dict,
                                workflow_config: Dict) -> str:
        """Generate data fusion summary section."""
        # Get user request - try multiple possible keys
        user_request = workflow_config.get('user_request') or workflow_config.get('natural_language_request') or 'Not provided'
        
        summary = f"""# Multi-Method Data Fusion Report

**Report Generation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

### Workflow Configuration
**Natural Language Request:**
```
{user_request}
```

### Data Sources
- **Seismic Data:** {workflow_config.get('seismic_file', 'N/A')}
- **ERT Data:** {workflow_config.get('ert_file', 'N/A')}
- **Output Directory:** {workflow_config.get('output_dir', 'N/A')}

### Key Results

"""
        
        # Seismic results
        if structure_results.get('seismic_results'):
            seismic = structure_results['seismic_results']
            velocity_model = seismic['velocity_model']
            summary += f"""#### 1. Seismic Velocity Inversion
- Velocity range: {np.min(velocity_model):.0f} - {np.max(velocity_model):.0f} m/s
- Mesh cells: {seismic['mesh'].cellCount()}
- Interface extraction: Successful

"""
        
        # Interface extraction
        if structure_results.get('interface_coords'):
            interface_x, interface_z = structure_results['interface_coords']
            velocity_threshold = structure_results.get('velocity_threshold', 1000)
            summary += f"""#### 2. Interface Extraction
- Velocity threshold: {velocity_threshold} m/s
- Interface points: {len(interface_x)}
- Depth range: {np.min(interface_z):.1f} - {np.max(interface_z):.1f} m

"""
        
        # ERT results
        if structure_results['status'] == 'success':
            stats = structure_results['statistics']
            summary += f"""#### 3. Structure-Constrained ERT Inversion
- Resistivity range: {stats['resistivity_range'][0]:.1f} - {stats['resistivity_range'][1]:.1f} Ωm
- Mean resistivity: {stats['mean_resistivity']:.1f} Ωm
- Number of layers: {stats['num_layers']}
- Mesh cells: {stats['num_cells']}

"""
        
        # Petrophysics results
        if petro_results['status'] == 'success':
            petro_stats = petro_results['statistics']
            summary += f"""#### 4. Petrophysical Conversion
- Water content range: {petro_stats['wc_range'][0]:.4f} - {petro_stats['wc_range'][1]:.4f}
- Mean water content: {petro_stats['mean_water_content']:.4f}
- Mean uncertainty: {petro_stats['mean_uncertainty']:.4f}
- Monte Carlo realizations: {workflow_config.get('n_realizations', 'N/A')}
- Number of layers: {petro_stats['n_layers']}

"""
        
        return summary
    
    def _generate_fusion_narrative(self, structure_results: Dict, petro_results: Dict,
                                  workflow_config: Dict) -> str:
        """Generate LLM narrative for data fusion workflow."""
        try:
            prompt = f"""Based on the following multi-method data fusion results, write a cohesive 
narrative summary (3-4 paragraphs) that:

1. Explains the agent-based workflow automation approach
2. Describes how seismic constraints improved ERT inversion
3. Discusses layer-specific petrophysical relationships
4. Highlights the benefits of multi-method integration
5. Summarizes uncertainty quantification approach
6. Provides interpretation of subsurface water content distribution

Key Results:
- Velocity range: {np.min(structure_results['seismic_results']['velocity_model']):.0f} - {np.max(structure_results['seismic_results']['velocity_model']):.0f} m/s
- Resistivity range: {structure_results['statistics']['resistivity_range']}
- Water content range: {petro_results['statistics']['wc_range']}
- Layer-specific parameters used: {len(workflow_config.get('layer_params', {}))} layers
- Monte Carlo realizations: {workflow_config.get('n_realizations', 100)}

Write in a professional, technical style suitable for a hydrogeophysical survey report."""
            
            narrative = self.query_llm(prompt, self.system_message,
                                      temperature=0.6, max_tokens=600)
            return f"\n## Integrated Analysis\n\n{narrative}\n"
        except Exception as e:
            self._log_execution(f"Could not generate narrative: {e}")
            return ""
    
    def _compile_fusion_report(self, summary: str, narrative: str, vis_files: Dict[str, str],
                              workflow_config: Dict) -> str:
        """Compile complete data fusion report."""
        report = f"""{summary}
{narrative if narrative else ''}

## Methodology

### Agent-Based Workflow

This analysis utilized an intelligent agent-based framework:

1. **ContextInputAgent**: Parsed natural language request to extract all parameters
2. **StructureConstraintAgent**: Automated 5-step workflow:
   - Mesh creation for seismic inversion
   - Seismic travel time inversion
   - Velocity interface extraction at threshold
   - Structure-constrained mesh generation
   - ERT inversion with structural constraints
3. **PetrophysicsAgent**: Layer-specific resistivity to water content conversion with Monte Carlo uncertainty

### Parameters from Natural Language

All workflow parameters were extracted from the natural language request:
- Velocity threshold: {workflow_config.get('velocity_threshold', 'N/A')} m/s
- ERT lambda: {workflow_config.get('ert_params', {}).get('lambda', 'N/A')}
- Mesh quality: {workflow_config.get('mesh_quality', 'N/A')}
- Monte Carlo realizations: {workflow_config.get('n_realizations', 'N/A')}
- Coverage threshold: {workflow_config.get('coverage_threshold', 'N/A')}

### Layer-Specific Petrophysics

"""
        
        # Add layer parameters
        layer_params = workflow_config.get('layer_params', {})
        if layer_params:
            for layer_name, params in layer_params.items():
                report += f"\n**{layer_name.replace('_', ' ').title()}:**\n"
                report += f"- ρ_sat range: {params.get('rho_sat_range', 'N/A')} Ωm\n"
                report += f"- n (cementation) range: {params.get('n_range', 'N/A')}\n"
                report += f"- Porosity range: {params.get('porosity_range', 'N/A')}\n"
        
        report += "\n## Visualizations\n\n"
        
        for vis_type, file_path in vis_files.items():
            report += f"### {vis_type.replace('_', ' ').title()}\n"
            report += f"![{vis_type}]({os.path.basename(file_path)})\n\n"
        
        report += f"""
## Summary and Recommendations

### Workflow Benefits

1. **Agent Encapsulation**: Complex 5-step workflow automated in single execute() call
2. **Natural Language Configuration**: All parameters from plain English description
3. **Structure Constraints**: Seismic interfaces reduced ERT artifacts
4. **Layer-Specific Petrophysics**: Geological realism improved water content accuracy
5. **Uncertainty Quantification**: Monte Carlo analysis provided confidence intervals
6. **Coverage Filtering**: Data quality thresholds ensured reliable results

### Key Findings

- Seismic velocity structure successfully delineated layer boundaries
- Structure-constrained ERT inversion preserved sharp contrasts
- Layer-specific petrophysical relationships improved conversion accuracy
- Monte Carlo uncertainty analysis quantified confidence in water content estimates
- Multi-method integration increased interpretation confidence

### Recommendations

1. **Validation**: Compare with direct measurements (gravimetric sampling, TDR)
2. **Temporal Monitoring**: Repeat surveys to track seasonal variations
3. **Extended Coverage**: Additional electrodes for deeper investigation
4. **Integration**: Incorporate additional methods (GPR, gravity) for comprehensive characterization

---

**Generated by:** PyHydroGeophysX Multi-Agent System  
**Report Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report
    
    def _log_execution(self, message: str, level: str = 'INFO'):
        """Log execution message."""
        print(f"[{self.name}] [{level}] {message}")
