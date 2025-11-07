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
            
            self._log_execution(f"Report saved to {report_file}")
            
            self.results = {
                'status': 'success',
                'report_file': report_file,
                'html_file': html_file,
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
        summary = "\n## Water Content Analysis\n\n"
        
        if 'water_content' in workflow_data:
            wc = workflow_data['water_content']
            summary += f"""
### Water Content Conversion
- Layer distributions: {len(wc.get('layer_distributions', {}))} geological layers
- Uncertainty analysis: {'Completed' if wc.get('params_used') else 'Not performed'}

**Interpretation:** {wc.get('interpretation', 'N/A')}
"""
        
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
        import numpy as np
        
        vis_files = {}
        
        try:
            # 1. Resistivity model plot
            if 'inversion_results' in workflow_data:
                inv = workflow_data['inversion_results']
                if 'mesh' in inv and 'resistivity_model' in inv:
                    try:
                        import pygimli as pg
                        fig, ax = plt.subplots(figsize=(12, 6))
                        pg.show(inv['mesh'], inv['resistivity_model'], 
                               ax=ax, cMap='jet', logScale=True,
                               label='Resistivity (Ohm-m)')
                        ax.set_title('ERT Inversion Results')
                        ax.set_xlabel('Distance (m)')
                        ax.set_ylabel('Elevation (m)')
                        
                        res_file = os.path.join(output_dir, 'resistivity_model.png')
                        plt.savefig(res_file, dpi=300, bbox_inches='tight')
                        plt.close()
                        vis_files['resistivity'] = res_file
                        self._log_execution("Saved resistivity model plot")
                    except Exception as e:
                        self._log_execution(f"Could not generate resistivity plot: {e}")
            
            # 2. Water content plot
            if 'water_content' in workflow_data:
                wc = workflow_data['water_content']
                if 'mesh' in wc and 'water_content_mean' in wc:
                    try:
                        import pygimli as pg
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        wc_mean = wc['water_content_mean']
                        if wc_mean.ndim > 1:
                            wc_mean = wc_mean[:, 0]  # First timestep
                        
                        pg.show(wc['mesh'], wc_mean,
                               ax=ax, cMap='Blues',
                               label='Water Content (-)',
                               cMin=0.0, cMax=0.5)
                        ax.set_title('Water Content Distribution')
                        ax.set_xlabel('Distance (m)')
                        ax.set_ylabel('Elevation (m)')
                        
                        wc_file = os.path.join(output_dir, 'water_content.png')
                        plt.savefig(wc_file, dpi=300, bbox_inches='tight')
                        plt.close()
                        vis_files['water_content'] = wc_file
                        self._log_execution("Saved water content plot")
                    except Exception as e:
                        self._log_execution(f"Could not generate water content plot: {e}")
            
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
    
    def _save_html_report(self, markdown_report: str, output_dir: str) -> Optional[str]:
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
            html_file = os.path.join(output_dir, 'workflow_report.html')
            with open(html_file, 'w') as f:
                f.write(html_content)
            return html_file
        except:
            return None
    
    def _log_execution(self, message: str, level: str = 'INFO'):
        """Log execution message."""
        print(f"[{self.name}] [{level}] {message}")
