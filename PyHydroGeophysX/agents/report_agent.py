"""
Report Generation Agent

Specialized agent for generating comprehensive reports from workflow results.
"""

import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

from ._chi2 import chi2_history
from ._document import bullets, control_block, facts, numbered, renumber, table
from . import _figstyle as figstyle
from ._figures import (FIGURE_CATALOG, llm_figure_topics,
                       missing_figure_warnings, plan_figures)
from ._method import SCHEME_DESCRIPTION, SCHEME_LABEL, resolve_scheme
from ._uncertainty import water_content_reliability

import numpy as np
import pandas as pd

from .base_agent import BaseAgent

# Climate analysis thresholds (configurable constants)
RAINFALL_THRESHOLD_MM = 5.0  # Significant rainfall threshold
WET_PERIOD_THRESHOLD_MM = 25.0  # 7-day antecedent for wet periods
DRY_PERIOD_THRESHOLD_MM = 5.0  # 7-day antecedent for dry periods
PET_DEFICIT_THRESHOLD_MM = -2.0  # P-PET deficit indicating drying
HIGH_TEMP_THRESHOLD_C = 30.0  # High temperature affecting measurements


# ---------------------------------------------------------------------------
# Report Agent
# ---------------------------------------------------------------------------
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
            visualization_files = self._generate_visualizations(workflow_data, output_dir, config)
            
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
            with open(report_file, 'w', encoding='utf-8') as f:
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
- Seismic Integration: {'Yes' if config.get('use_seismic', False) or workflow_data.get('seismic_structure') else 'No'}

**Key Results:**
"""
        
        # Add key findings from each step
        if 'ert_data' in workflow_data:
            ert = workflow_data['ert_data']
            num_elec = ert.get('num_electrodes') or ert.get('n_electrodes')
            num_meas = ert.get('num_measurements') or ert.get('n_measurements')
            summary += f"- Loaded {num_elec if num_elec is not None else 'N/A'} electrodes with {num_meas if num_meas is not None else 'N/A'} measurements\n"
        
        if 'inversion_results' in workflow_data:
            inv = workflow_data['inversion_results']
            summary += f"- Inversion completed {inv.get('iterations', 'N/A')} iterations (chi2: {inv.get('chi2', 'N/A')})\n"
            evaluation = workflow_data.get('evaluation_results') or {}
            summary += f"- Quality assessment: {evaluation.get('status', 'not evaluated')} — {evaluation.get('summary', 'Review convergence and data fit.')}\n"
            processing = inv.get('processing') or {}
            if processing:
                summary += (f"- Measurements supplied to inversion: {processing.get('inverted_measurements', 'N/A')} "
                            f"of {processing.get('input_measurements', 'N/A')} loaded; see processing log for filters.\n")
        
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
            num_elec = ert.get('num_electrodes') or ert.get('n_electrodes')
            num_meas = ert.get('num_measurements') or ert.get('n_measurements')
            summary += f"""
### ERT Data Loading
- Number of electrodes: {num_elec if num_elec is not None else 'N/A'}
- Number of measurements: {num_meas if num_meas is not None else 'N/A'}
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
            # Named when it is not the plain smoothness inversion, so the report
            # says which method produced the section it shows.
            method = f"- Method: {inv['inversion_method']}\n" if inv.get('inversion_method') else ""
            summary += f"""
### ERT Inversion
{method}- Final chi2: {inv.get('chi2', 'N/A')}
- Iterations: {inv.get('iterations', 'N/A')}
- Quality assessment: {(workflow_data.get('evaluation_results') or {}).get('status', 'not evaluated')}
- Assessment details: {(workflow_data.get('evaluation_results') or {}).get('summary', 'A completed solver run alone does not establish convergence or model validity.')}

**Interpretation:** {inv.get('interpretation', 'N/A')}
"""
        
        return summary
    
    def _generate_wc_summary(self, workflow_data: Dict) -> str:
        """Generate water content analysis summary."""
        import numpy as np
        summary = "\n## Water Content Analysis\n\n"
        
        # Check if petrophysics was skipped
        if workflow_data.get('skip_petrophysics', False):
            summary += "Water content conversion was not requested for this workflow.\n"
            summary += "Only resistivity inversion results are available.\n"
            return summary
        
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
            
            # Layer parameters used (means +/- std)
            # Collect petrophysical parameters from multiple sources
            layer_params = {}
            candidates = [
                wc.get('layer_params_used'),
                wc.get('layer_params'),
                workflow_data.get('petrophysics_results', {}).get('layer_params_used', {}),
                workflow_data.get('petrophysics_results', {}).get('layer_params', {}),
            ]
            # If only per-parameter scalars were provided, build a single-layer entry
            scalar_params = workflow_data.get('petrophysical_params', {}) or wc.get('petrophysical_params', {})

            for cand in candidates:
                if isinstance(cand, dict):
                    layer_params.update(cand)

            if not layer_params and scalar_params:
                layer_params = {
                    0: {
                        'use_rho_sat': 'rho_sat' in scalar_params,
                        'rho_sat': {'mean': scalar_params.get('rho_sat'), 'std': scalar_params.get('rho_sat_std', 'N/A')},
                        'm': {'mean': scalar_params.get('m'), 'std': scalar_params.get('m_std', 'N/A')},
                        'n': {'mean': scalar_params.get('n'), 'std': scalar_params.get('n_std', 'N/A')},
                        'porosity': {'mean': scalar_params.get('porosity'), 'std': scalar_params.get('porosity_std', 'N/A')},
                    }
                }
            if layer_params:
                # Check if user provided explicit parameters or defaults were used
                user_provided_params = workflow_data.get('petrophysical_params', {}) or wc.get('petrophysical_params', {})
                if not user_provided_params:
                    summary += f"\n### Petrophysical Parameters (Defaults Applied)\n"
                    summary += "**Note:** No explicit petrophysical parameters were provided. Default Archie parameters were used based on geological layer type.\n\n"
                else:
                    summary += f"\n### Petrophysical Parameters (User-Specified)\n"
                    summary += f"**User input parameters:** {user_provided_params}\n\n"
                
                summary += "**Parameters used per layer (means +/- std):**\n"
                for marker, params in layer_params.items():
                    if not isinstance(params, dict):
                        # If params is a scalar (e.g., rho_sat), skip to avoid attribute errors
                        continue
                    
                    # Determine layer name
                    layer_name = f"Layer {marker}"
                    if marker in [0, '0', 'regolith', 'Regolith']:
                        layer_name = "Regolith"
                    elif marker in [1, '1', 'bedrock', 'Bedrock']:
                        layer_name = "Bedrock"
                    elif marker in [2, '2']:
                        layer_name = "Layer 2 (Regolith)"
                    elif marker in [3, '3']:
                        layer_name = "Layer 3 (Bedrock)"
                    
                    use_rho_sat = params.get('use_rho_sat', False)
                    
                    def fmt(comp: str) -> str:
                        if isinstance(params.get(comp), dict):
                            mean_val = params[comp].get('mean', 'N/A')
                            std_val = params[comp].get('std', 'N/A')
                            if mean_val != 'N/A' and std_val != 'N/A':
                                return f"{mean_val:.3f} ± {std_val:.3f}"
                            return f"{mean_val}"
                        return str(params.get(comp, 'N/A'))
                    
                    if use_rho_sat:
                        summary += (
                            f"- **{layer_name}**: "
                            f"ρ_sat={fmt('rho_sat')} Ωm, "
                            f"n={fmt('n')}, "
                            f"φ={fmt('porosity')}\n"
                        )
                    else:
                        summary += (
                            f"- **{layer_name}**: "
                            f"m={fmt('m')}, "
                            f"n={fmt('n')}, "
                            f"φ={fmt('porosity')}, "
                            f"ρ_sat={fmt('rho_sat')} Ωm\n"
                        )
            else:
                summary += "\n### Petrophysical Parameters\n"
                summary += "Default Archie parameters were applied for the resistivity-to-water-content conversion.\n"
            
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
- Source: {metadata.get('source', metadata.get('region', 'N/A'))}
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
    
    def _generate_visualizations(self, workflow_data: Dict, output_dir: str,
                                 config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Generate visualization plots with dynamic colormap limits based on data."""
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np

        # Every figure below reads its colormap and colorbar orientation from
        # this; without it each one failed on a NameError that the per-figure
        # try/except turned into a log line, so the report had no figures.
        style = figstyle.style_from_config(config)

        # Set Arial font for all plots
        matplotlib.rcParams['font.family'] = 'Arial'
        matplotlib.rcParams['font.size'] = 12
        
        vis_files = {}
        
        def compute_colormap_limits(data, log_scale=False, percentile_range=(2, 98)):
            """
            Compute optimal colormap min/max based on data values.
            
            Args:
                data: Array of values
                log_scale: If True, compute limits for log-scale display
                percentile_range: Tuple of (min_percentile, max_percentile) to use
                
            Returns:
                Tuple of (cMin, cMax)
            """
            data_flat = np.array(data).ravel()
            # Remove NaN and Inf values
            data_clean = data_flat[np.isfinite(data_flat)]
            
            if len(data_clean) == 0:
                return (1, 1000) if log_scale else (0, 1)
            
            # For log scale, also remove non-positive values
            if log_scale:
                data_clean = data_clean[data_clean > 0]
                if len(data_clean) == 0:
                    return (1, 1000)
            
            # Use percentiles to exclude outliers
            cMin = np.percentile(data_clean, percentile_range[0])
            cMax = np.percentile(data_clean, percentile_range[1])
            
            # Ensure minimum spread for log scale
            if log_scale:
                cMin = max(cMin, 1e-3)  # Prevent too-small values
                if cMax <= cMin:  # Only pad a constant model; preserve real contrast.
                    cMin /= 1.05
                    cMax *= 1.05
            else:
                # For linear scale, round to nice values
                spread = cMax - cMin
                if spread < 0.01:
                    spread = 0.1
                cMin = max(0, cMin - spread * 0.05)
                cMax = cMax + spread * 0.05
            
            return (cMin, cMax)

        min_profile_height_ratio = float(
            workflow_data.get('min_profile_height_ratio', 1.0 / 3.0) or (1.0 / 3.0)
        )

        def _mesh_cell_centers(mesh):
            """Return mesh cell-center coordinates as finite numpy arrays."""
            if mesh is None:
                return None, None
            try:
                cells = list(mesh.cells())
                if len(cells) == 0:
                    return None, None
                xc = np.array([c.center().x() for c in cells], dtype=float)
                yc = np.array([c.center().y() for c in cells], dtype=float)
                finite = np.isfinite(xc) & np.isfinite(yc)
                if not np.any(finite):
                    return None, None
                return xc[finite], yc[finite]
            except Exception:
                return None, None

        def _infer_surface_side(yc, mask=None):
            """
            Infer whether the surface corresponds to max(y) or min(y).
            Returns (surface_is_max, y_surface).
            """
            if yc is None or len(yc) == 0:
                return True, 0.0
            y_min = float(np.nanmin(yc))
            y_max = float(np.nanmax(yc))
            if mask is not None and len(mask) == len(yc) and np.sum(mask) >= 3:
                y_ref = float(np.nanmedian(yc[np.asarray(mask, dtype=bool)]))
            else:
                y_ref = float(np.nanmedian(yc))
            surface_is_max = abs(y_max - y_ref) <= abs(y_ref - y_min)
            return surface_is_max, (y_max if surface_is_max else y_min)

        def _apply_vertical_limits(ax, mesh, coverage_mask=None):
            """
            Ensure the visible vertical extent is at least profile_length * ratio.
            Returns metadata tuple (applied, shown_span, min_required_span).
            """
            xc, yc = _mesh_cell_centers(mesh)
            if xc is None or yc is None or yc.size < 2:
                return False, None, None

            x_span = float(np.nanmax(xc) - np.nanmin(xc))
            min_required_span = max(5.0, min_profile_height_ratio * x_span)

            finite_cov = None
            if coverage_mask is not None and len(coverage_mask) == len(yc):
                finite_cov = np.asarray(coverage_mask, dtype=bool)
                if np.sum(finite_cov) < 3:
                    finite_cov = None

            y_min = float(np.nanmin(yc))
            y_max = float(np.nanmax(yc))
            surface_is_max, _ = _infer_surface_side(yc, finite_cov)
            if surface_is_max:
                y_surface = y_max
                y_cov_deep = float(np.nanmin(yc[finite_cov])) if finite_cov is not None else y_min
                shown_span = max(0.0, y_surface - y_cov_deep)
                target_span = max(shown_span, min_required_span)
                y_target_deep = max(y_surface - target_span, y_min)
                y_pad = max(2.0, 0.08 * target_span)
                y_lo = y_target_deep - y_pad
                y_hi = y_surface + y_pad
            else:
                y_surface = y_min
                y_cov_deep = float(np.nanmax(yc[finite_cov])) if finite_cov is not None else y_max
                shown_span = max(0.0, y_cov_deep - y_surface)
                target_span = max(shown_span, min_required_span)
                y_target_deep = min(y_surface + target_span, y_max)
                y_pad = max(2.0, 0.08 * target_span)
                y_lo = y_surface - y_pad
                y_hi = y_target_deep + y_pad

            if np.isfinite(y_lo) and np.isfinite(y_hi):
                ax.set_ylim(min(y_lo, y_hi), max(y_lo, y_hi))
                return True, target_span, min_required_span
            return False, None, min_required_span

        def compute_coverage_mask(mesh, coverage):
            """
            Choose coverage threshold so the deepest covered point reaches target depth.
            Returns (mask, threshold, keep_fraction, deepest_depth, min_required_depth).
            """
            if coverage is None:
                return None, None, None, None, None

            cov = np.array(coverage, dtype=float).reshape(-1)
            finite = np.isfinite(cov)
            if not np.any(finite):
                return None, None, None, None, None

            # Ensure mask length matches mesh cells when available
            if mesh is not None:
                try:
                    if len(cov) != int(mesh.cellCount()):
                        return None, None, None, None, None
                except Exception:
                    return None, None, None, None, None

            xc, yc = _mesh_cell_centers(mesh)
            if xc is not None and yc is not None and len(yc) == len(cov):
                profile_length = float(np.nanmax(xc) - np.nanmin(xc))
                min_required_depth = max(5.0, min_profile_height_ratio * profile_length)
            else:
                min_required_depth = 5.0

            surface_is_max, y_surface = _infer_surface_side(yc, None)

            # Start strict and relax using actual coverage values to get
            # the strictest threshold that still reaches target depth.
            threshold_candidates = np.unique(cov[finite])
            threshold_candidates = threshold_candidates[np.isfinite(threshold_candidates)]
            threshold_candidates = np.sort(threshold_candidates)[::-1]
            if threshold_candidates.size > 600:
                idx = np.linspace(0, threshold_candidates.size - 1, num=600, dtype=int)
                threshold_candidates = threshold_candidates[idx]

            best = None
            for threshold in threshold_candidates:
                threshold = float(threshold)
                mask = finite & (cov > threshold)
                keep_frac = float(np.mean(mask))
                n_keep = int(np.sum(mask))
                if n_keep < 3:
                    continue

                if yc is not None and len(yc) == len(mask):
                    ycov = yc[mask]
                    if surface_is_max:
                        deepest_depth = max(0.0, y_surface - float(np.nanmin(ycov)))
                    else:
                        deepest_depth = max(0.0, float(np.nanmax(ycov)) - y_surface)
                else:
                    deepest_depth = float("nan")

                candidate = (mask, threshold, keep_frac, deepest_depth)
                if best is None:
                    best = candidate
                else:
                    best_depth = best[3]
                    choose_candidate = False
                    if np.isnan(best_depth) and not np.isnan(deepest_depth):
                        choose_candidate = True
                    elif not np.isnan(deepest_depth) and deepest_depth > best_depth + 1e-9:
                        choose_candidate = True
                    elif (
                        not np.isnan(deepest_depth)
                        and not np.isnan(best_depth)
                        and abs(deepest_depth - best_depth) <= 1e-9
                        and keep_frac < best[2] - 1e-6
                    ):
                        # For equal depth, prefer smaller masked area.
                        choose_candidate = True
                    if choose_candidate:
                        best = candidate

                enough_depth = (not np.isnan(deepest_depth)) and (deepest_depth >= min_required_depth)
                if enough_depth:
                    return mask, threshold, keep_frac, deepest_depth, min_required_depth

            if best is not None:
                return best[0], best[1], best[2], best[3], min_required_depth

            fallback = finite.copy()
            keep_frac = float(np.mean(fallback))
            if yc is not None and len(yc) == len(fallback):
                ycov = yc[fallback]
                if surface_is_max:
                    deepest_depth = max(0.0, y_surface - float(np.nanmin(ycov)))
                else:
                    deepest_depth = max(0.0, float(np.nanmax(ycov)) - y_surface)
            else:
                deepest_depth = float("nan")
            return fallback, float(np.nanmin(cov[finite])), keep_frac, deepest_depth, min_required_depth
        
        try:
            # 1. Resistivity model plot with coverage masking
            if 'inversion_results' in workflow_data:
                inv = workflow_data['inversion_results']
                if 'mesh' in inv and 'resistivity_model' in inv:
                    try:
                        import pygimli as pg
                        fig = figstyle.detached_figure((8, 3))
                        ax = fig.add_subplot(111)
                        
                        # Get coverage if available
                        coverage = inv.get('coverage')
                        coverage_mask, coverage_threshold, keep_frac, deepest_depth, min_required_depth = compute_coverage_mask(
                            inv.get('mesh'), coverage
                        )
                        if coverage_mask is not None:
                            self._log_execution(
                                f"Coverage threshold: {coverage_threshold:.3f} "
                                f"(kept {keep_frac*100:.1f}% cells, "
                                f"deepest point {deepest_depth:.1f} m, "
                                f"min target {min_required_depth:.1f} m)"
                            )
                        
                        # Compute dynamic colormap limits for resistivity
                        res_model = np.array(inv['resistivity_model'])
                        cMin_res, cMax_res = compute_colormap_limits(res_model, log_scale=True)
                        self._log_execution(f"Resistivity colormap: {cMin_res:.1f} to {cMax_res:.1f} ohm-m")
                        
                        # Plot with coverage masking
                        ax, cbar = figstyle.pg_show(
                            inv['mesh'],
                            inv['resistivity_model'],
                            ax=ax,
                            fig=fig,
                            cMap=style.cmap_for('resistivity'),
                            cMin=cMin_res,
                            cMax=cMax_res,
                            logScale=True,
                            label=r'Resistivity ($\Omega$ m)',
                            pad=0.3,
                            orientation=style.colorbar_orientation,
                            coverage=coverage_mask
                        )
                        
                        ax.set_xlabel('Distance (m)', fontsize=14, fontfamily='Arial')
                        ax.set_ylabel('Elevation (m)', fontsize=14, fontfamily='Arial')
                        # pg.show may format negative elevations as positive
                        # depths. Keep signed coordinates for an elevation axis.
                        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
                        ax.set_title('ERT Inversion Results', fontsize=16, fontfamily='Arial')

                        applied, shown_span, min_span = _apply_vertical_limits(
                            ax, inv.get('mesh'), coverage_mask
                        )
                        if applied and shown_span is not None and min_span is not None:
                            self._log_execution(
                                f"Vertical display span set to {shown_span:.1f} m "
                                f"(min required {min_span:.1f} m)"
                            )
                        
                        res_file = os.path.join(output_dir, 'resistivity_model.png')
                        fig.savefig(res_file, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        vis_files['resistivity'] = res_file
                        self._log_execution("Saved resistivity model plot")
                    except Exception as e:
                        self._log_execution(f"Could not generate resistivity plot: {e}")
            
            # 2. Water content plot with coverage masking (only if petrophysics was run)
            if 'water_content' in workflow_data and not workflow_data.get('skip_petrophysics', False):
                wc = workflow_data['water_content']
                if 'mesh' in wc and 'water_content_mean' in wc:
                    try:
                        import pygimli as pg
                        fig = figstyle.detached_figure((8, 3))
                        ax = fig.add_subplot(111)
                        
                        wc_mean = wc['water_content_mean']
                        if wc_mean.ndim > 1:
                            wc_mean = wc_mean[:, 0]  # First timestep
                        
                        # Get coverage from inversion results if available
                        coverage = None
                        if 'inversion_results' in workflow_data:
                            inv = workflow_data['inversion_results']
                            coverage = inv.get('coverage')
                        coverage_mask, _, _, _, _ = compute_coverage_mask(wc.get('mesh'), coverage)
                        
                        # Compute dynamic colormap limits for water content
                        cMin_wc, cMax_wc = compute_colormap_limits(wc_mean, log_scale=False)
                        # Ensure water content is bounded [0, 1] 
                        cMin_wc = max(0.0, cMin_wc)
                        cMax_wc = min(1.0, cMax_wc) if cMax_wc > 0 else 0.5
                        self._log_execution(f"Water content colormap: {cMin_wc:.3f} to {cMax_wc:.3f}")
                        
                        ax, cbar = figstyle.pg_show(
                            wc['mesh'],
                            wc_mean,
                            ax=ax,
                            fig=fig,
                            cMap='Blues',
                            label='Water Content (-)',
                            cMin=cMin_wc,
                            cMax=cMax_wc,
                            pad=0.3,
                            orientation=style.colorbar_orientation,
                            coverage=coverage_mask
                        )
                        
                        ax.set_xlabel('Distance (m)', fontsize=14, fontfamily='Arial')
                        ax.set_ylabel('Elevation (m)', fontsize=14, fontfamily='Arial')
                        ax.set_title('Water Content Distribution', fontsize=16, fontfamily='Arial')

                        _apply_vertical_limits(ax, wc.get('mesh'), coverage_mask)
                        
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
                            fig = figstyle.detached_figure((8, 3))
                            ax = fig.add_subplot(111)
                            
                            wc_std = wc['water_content_std']
                            if wc_std.ndim > 1:
                                wc_std = wc_std[:, 0]  # First timestep
                            
                            # Get coverage from inversion results if available
                            coverage = None
                            if 'inversion_results' in workflow_data:
                                inv = workflow_data['inversion_results']
                                coverage = inv.get('coverage')
                            coverage_mask, _, _, _, _ = compute_coverage_mask(wc.get('mesh'), coverage)
                            
                            # Compute dynamic colormap limits for uncertainty
                            cMin_std, cMax_std = compute_colormap_limits(wc_std, log_scale=False)
                            cMin_std = max(0.0, cMin_std)
                            self._log_execution(f"Uncertainty colormap: {cMin_std:.4f} to {cMax_std:.4f}")
                            
                            ax, cbar = figstyle.pg_show(
                                wc['mesh'],
                                wc_std,
                                ax=ax,
                                fig=fig,
                                cMap='Reds',
                                label=r'Uncertainty ($\sigma$)',
                                cMin=cMin_std,
                                cMax=cMax_std,
                                pad=0.3,
                                orientation=style.colorbar_orientation,
                                coverage=coverage_mask
                            )
                            
                            ax.set_xlabel('Distance (m)', fontsize=14, fontfamily='Arial')
                            ax.set_ylabel('Elevation (m)', fontsize=14, fontfamily='Arial')
                            ax.set_title('Water Content Uncertainty', fontsize=16, fontfamily='Arial')

                            _apply_vertical_limits(ax, wc.get('mesh'), coverage_mask)
                            
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
2. Discusses only methods and data actually present in the supplied results
3. Integrates climate context only if climate observations were included
4. Summarizes findings without inventing geological causes or computed uncertainties
5. Highlights any notable patterns, anomalies, or data quality caveats
6. Provides recommendations for next steps

Workflow Results:
{combined_info}

Write in a professional, technical style suitable for a geophysical survey report that 
incorporates cross-modal climate-geophysics reasoning."""
            
            narrative = self.query_llm(prompt, self.system_message, 
                                      temperature=0.6, max_tokens=600)
            caveat = "**AI-generated interpretation - verify before citing.**"
            return f"\n## Narrative Summary\n\n{caveat}\n\n{narrative}\n"
        except Exception:
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
                - time_lapse_method: what was requested; the scheme reported is
                  the one that runs (see PyHydroGeophysX.agents._method)
                
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
            time_lapse_method = input_data.get('time_lapse_method')
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate report sections with error handling
            self._log_execution("Generating time-lapse report sections")
            
            workflow_config = input_data.get('workflow_config') or {}
            evaluation_results = (input_data.get('evaluation_results')
                                  or inversion_results.get('evaluation_results'))
            key_findings = self._timelapse_key_findings(
                inversion_results, climate_data, evaluation_results, workflow_config)

            # 1. Time-Lapse Executive Summary
            try:
                tl_exec_summary = self._generate_timelapse_executive_summary(
                    inversion_results, site_info, time_lapse_method,
                    workflow_config, key_findings, evaluation_results
                )
            except Exception as e:
                self._log_execution(f"Error generating executive summary: {e}", level='ERROR')
                raise Exception(f"Failed to generate executive summary: {str(e)}")

            # 2. Method, then the inversion results it produced
            try:
                tl_method_section = self._generate_timelapse_method_section(
                    inversion_results, workflow_config, site_info
                )
                tl_inversion_section = self._generate_timelapse_inversion_section(
                    inversion_results, time_lapse_method, site_info
                )
            except Exception as e:
                self._log_execution(f"Error generating inversion section: {e}", level='ERROR')
                raise Exception(f"Failed to generate inversion section: {str(e)}")

            # 3. Climate Data Section (if available)
            try:
                tl_climate_section = self._generate_timelapse_climate_section(
                    climate_data, site_info
                )
            except Exception as e:
                self._log_execution(f"Error generating climate section: {e}", level='ERROR')
                tl_climate_section = "\n## Climate Data Integration\n\n*Climate section unavailable*\n"
            
            # 4. Climate-Resistivity Correlation Analysis (if available)
            try:
                tl_correlation_section = self._generate_timelapse_correlation_section(
                    comparison_df, inversion_results
                )
            except Exception as e:
                self._log_execution(f"Error generating correlation section: {e}", level='ERROR')
                tl_correlation_section = "\n## Climate-Resistivity Correlation\n\n*Correlation analysis unavailable*\n"
            
            # 5. Time-Lapse Visualizations.
            #
            # The request is read for its figure preferences *before* anything
            # is drawn. Asking afterwards was useless: "make the figures bigger"
            # would have been recorded into a configuration the generators had
            # already finished reading.
            figure_request = self._read_figure_request(workflow_config)
            try:
                tl_vis_files = self._generate_timelapse_visualizations(
                    inversion_results, comparison_df, climate_data, output_dir,
                    site_info, workflow_config
                )
            except Exception as e:
                self._log_execution(f"Error generating visualizations: {e}", level='ERROR')
                tl_vis_files = {}

            # Which of them this request wanted to see, and what it wanted that
            # the run cannot show.
            figure_plan = self._figure_plan(tl_vis_files, workflow_config,
                                            figure_request)
            self._figure_warnings = missing_figure_warnings(figure_plan.get('missing') or [])
            for warning in self._figure_warnings:
                self._log_execution(warning, level='WARNING')
            
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
                self._timelapse_front_matter(
                    inversion_results, site_info, workflow_config),
                tl_exec_summary,
                tl_method_section,
                tl_inversion_section,
                self._generate_timelapse_water_content_section(
                    inversion_results, workflow_config, site_info),
                tl_climate_section,
                tl_correlation_section,
                self._figures_section(tl_vis_files, figure_plan),
                tl_narrative,
                self._timelapse_recommendations(
                    climate_data, inversion_results, workflow_config),
            )
            
            # Save report to file
            report_file = os.path.join(output_dir, 'time_lapse_report.md')
            with open(report_file, 'w', encoding='utf-8') as f:
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
                # A figure the request asked for that the run cannot produce is
                # reported, not left as an absence the reader has to notice.
                'figure_plan': figure_plan,
                'warnings': list(getattr(self, '_figure_warnings', []) or []),
                'output_dir': output_dir
            }
            
        except Exception as e:
            self._log_execution(f"Error generating time-lapse report: {str(e)}", level='ERROR')
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    #: Placed under the document-control block. A reader who opens this at the
    #: heading and stops reading should still know it is machine-written.
    NOTICE = (
        "This report was produced by an automated multi-agent workflow. The "
        "numerical results are computed; the interpretive text is written by a "
        "language model and has not been reviewed by a geophysicist. Both must "
        "be checked against field observations before they are relied upon. "
        "The limitations stated at the end of this document qualify every "
        "number in it."
    )

    def _timelapse_front_matter(self, inversion_results: Dict, site_info: Dict,
                                config: Optional[Dict] = None) -> str:
        """Title and document-control block.

        The equivalent of a deliverable's cover page: what this is, which site
        and surveys it covers, what produced it and when. It exists so the
        provenance of a file that will be forwarded, printed and filed travels
        with it, rather than living only in the run directory it came out of.
        """
        config = config or {}
        scheme_note = resolve_scheme(config.get('time_lapse_method'))[1]
        n_steps = inversion_results.get('n_timesteps')
        pairs = [
            ('Site', site_info.get('name')),
            ('Location', site_info.get('location')),
            ('Coordinates', site_info.get('coordinates')),
            ('Survey period', site_info.get('study_period')),
            ('Surveys inverted', n_steps),
            ('Inversion scheme', SCHEME_LABEL),
            ('Report date', datetime.now().strftime('%Y-%m-%d %H:%M')),
            ('Prepared by', 'PyHydroGeophysX multi-agent workflow'),
            ('Status', 'Automated output - technical review required'),
        ]
        block = control_block(
            'Time-Lapse Electrical Resistivity Tomography: Monitoring Report',
            pairs, notice=self.NOTICE)
        if scheme_note:
            block += f"\n{scheme_note}\n"
        return block

    def _generate_timelapse_executive_summary(self, inversion_results: Dict,
                                             site_info: Dict, method: str,
                                             config: Optional[Dict] = None,
                                             key_findings: str = '',
                                             evaluation_results: Optional[Dict] = None
                                             ) -> str:
        """The section a reader who reads nothing else will read.

        It answers three questions in order - what was asked for, what the run
        found, and how far the findings can be trusted - and every figure in it
        is repeated with its full working further down. The configuration
        details that used to occupy this space moved to the method section;
        knowing the temporal regularization was 10.0 is not a summary of
        anything.
        """
        config = config or {}
        request = str(config.get('user_request') or '').strip()
        summary = "## Executive Summary\n\n### Scope\n\n"
        if request:
            summary += (f"This report responds to the request: “{request}”\n\n")
        n_steps = inversion_results.get('n_timesteps')
        period = site_info.get('study_period')
        scope = (f"{n_steps} electrical resistivity surveys" if n_steps
                 else "Repeat electrical resistivity surveys")
        if period and period not in ('N/A', 'N/A to N/A'):
            scope += f" covering the period {period}"
        summary += (f"{scope} were inverted together and the resulting models "
                    f"compared to the first survey in the series.\n\n")

        if key_findings:
            summary += "### Principal Findings\n\n" + key_findings.rstrip() + "\n\n"

        summary += "### Confidence in These Findings\n\n"
        summary += self._timelapse_confidence_statement(
            inversion_results, evaluation_results, config,
            # The findings already carry the water-content caveat verbatim;
            # repeating it three paragraphs later reads as two separate
            # objections rather than one.
            include_water=not key_findings) + "\n\n"
        return summary

    def _timelapse_confidence_statement(self, inversion_results: Dict,
                                        evaluation_results: Optional[Dict],
                                        config: Optional[Dict],
                                        include_water: bool = True) -> str:
        """One paragraph on whether the findings can carry weight.

        Written from the run's own diagnostics rather than a stock sentence,
        because the useful case is the one where the answer is no: a report
        whose confidence statement never says "do not rely on this" is not
        reporting confidence.
        """
        evaluation = evaluation_results or {}
        status = evaluation.get('status')
        score = evaluation.get('quality_score')
        history = chi2_history(inversion_results.get('chi2_values'))
        parts = []
        if history:
            final = history[-1]
            verdict = ("within the range normally accepted for a converged inversion"
                       if 0.8 <= final <= 1.5 else
                       "outside the range normally accepted for a converged inversion")
            parts.append(f"The inversion reached a final chi-squared of {final:.3f}, "
                         f"{verdict}.")
        if score is not None:
            outcome = ('passes' if status == 'success' else 'does not pass')
            parts.append(f"The automated quality assessment scored the result "
                         f"{float(score):.1f} out of 100, which {outcome} the "
                         f"threshold configured for this workflow.")
        reliability = (water_content_reliability(inversion_results, config or {})
                       if include_water else None)
        if reliability:
            parts.append(reliability['sentence'])
        if not parts:
            parts.append("The run did not report the diagnostics needed to judge "
                         "how far these findings can be trusted.")
        parts.append("A good data fit constrains the model; it does not make it "
                     "unique, and it says nothing about whether the cause "
                     "attributed to a change is the right one.")
        return " ".join(parts)


    def _survey_dates(self, site_info: Optional[Dict], count: int) -> list:
        """Acquisition date per survey, or empty strings when unknown.

        Labelling rows "Time Step 4" makes a reader cross-reference the file
        list to find out when that was; the dates are already read off the
        file names for the study period, so they are carried through here.
        """
        dates = list((site_info or {}).get('survey_dates') or [])
        if len(dates) != count:
            return [''] * count
        return [str(d) for d in dates]

    @staticmethod
    def _temperature_correction_note(report: Optional[Dict]) -> str:
        """State what temperature the sections are reported at.

        Always stated, in every time-lapse report. Bulk resistivity falls about
        2 % per degC, so a seasonal temperature swing produces the same size of
        change as the moisture the survey is run to see - and a corrected section
        is indistinguishable from an uncorrected one on the page. A reader who is
        not told cannot know which they are looking at.
        """
        report = report or {}
        if report.get('applied'):
            reference = report.get('reference_temperature_C')
            law = 'Hayley et al. (2007)' if report.get('model') == 'hayley' else \
                'the linear (Campbell/Arps) law'
            text = (
                f"Every survey was corrected to a reference temperature of "
                f"{reference:g} °C using {law}, from "
                f"{report.get('temperature_source', 'the supplied temperatures')}. "
                f"The resistivities reported below are therefore at "
                f"{reference:g} °C, and the change between surveys is free of the "
                f"seasonal warming and cooling of the ground.")
            change = report.get('max_abs_change_percent')
            if isinstance(change, (int, float)):
                text += (f" The correction moved the models by up to "
                         f"{change:.1f} %; changes smaller than that would have "
                         f"been temperature rather than moisture.")
            return text + "\n"
        if report.get('requested'):
            return (
                "A temperature correction was requested but could not be applied "
                f"({report.get('error', 'reason not recorded')}). The sections "
                "below are raw inverted resistivity, so part of any change "
                "between surveys is the ground warming or cooling rather than a "
                "change in moisture.\n")
        return (
            "No temperature correction was applied. The sections below are "
            "inverted resistivity at the in-situ ground temperature, so part of "
            "any change between surveys - roughly 2 % per °C - is seasonal "
            "warming and cooling rather than a change in moisture. Over a few "
            "days this is negligible; across seasons it is not.\n")

    def _generate_timelapse_method_section(self, inversion_results: Dict,
                                           config: Optional[Dict] = None,
                                           site_info: Optional[Dict] = None) -> str:
        """What was done, in enough detail for someone to repeat it.

        The scheme is named from :mod:`PyHydroGeophysX.agents._method` rather
        than from ``time_lapse_method`` in the configuration. That field is
        written to the log and never reaches the solver, so reporting it
        described a difference inversion for every run - while the solver was
        inverting all surveys simultaneously under a temporal constraint.
        """
        config = config or {}
        section = "## Method\n\n### Inversion Approach\n\n"
        section += SCHEME_DESCRIPTION + "\n\n"
        note = resolve_scheme(config.get('time_lapse_method'))[1]
        if note:
            section += note + "\n\n"

        solver = inversion_results.get('method')
        solver_label = str(solver).upper() if solver else None
        section += "### Inversion Parameters\n\n"
        section += facts([
            ('Surveys inverted', inversion_results.get('n_timesteps')),
            ('Spatial constraint, lambda', inversion_results.get('lambda')),
            ('Temporal constraint, alpha',
             inversion_results.get('temporal_regularization')),
            ('Maximum iterations', inversion_results.get('max_iterations')),
            ('Linear solver', solver_label),
        ], label='Parameter', value='Value')

        section += "\n### Survey Inventory\n\n"
        count = int(inversion_results.get('n_timesteps') or 0)
        dates = self._survey_dates(site_info, count)
        # The gap since the previous survey, per row. A monitoring series reads
        # the same whether it was sampled hourly or monthly until something says
        # which, and that is the first thing a reader needs to judge the changes.
        timing = inversion_results.get('survey_timing') or {}
        gaps = list(timing.get('interval_text') or [])
        rows = [[i + 1, dates[i] if i < len(dates) and dates[i] else None,
                 gaps[i - 1] if 0 < i <= len(gaps) else None,
                 'Baseline' if i == 0 else 'Repeat']
                for i in range(count)]
        inventory = table(['Survey', 'Acquired', 'Since previous', 'Role'], rows,
                          align=['right', 'left', 'left', 'left'])
        section += inventory or "The run did not report how many surveys it inverted.\n"
        if timing.get('summary'):
            section += f"\n{timing['summary']}\n"
        elif count > 1:
            section += ("\nThe survey files carried no acquisition time, so the "
                        "steps are numbered rather than dated and the interval "
                        "between them is not known to the analysis.\n")

        section += "\n### Temperature Correction\n\n"
        section += self._temperature_correction_note(
            inversion_results.get('temperature_correction'))

        petro = config.get('petrophysical_params')
        if (inversion_results.get('time_lapse_water_content')
                or config.get('convert_to_water_content')):
            section += "\n### Petrophysical Conversion\n\n"
            section += (
                "Resistivity was converted to volumetric water content cell by "
                "cell through a petrophysical relationship, with the parameters "
                "of that relationship drawn repeatedly from their distributions "
                "so that the spread of the resulting water content reports the "
                "uncertainty of the conversion. ")
            section += (
                "Site-specific petrophysical parameters were supplied and used.\n"
                if petro else
                "No site-specific petrophysical parameters were supplied, so "
                "generated defaults were used. The conversion is therefore "
                "indicative rather than calibrated, and the water-content "
                "uncertainty reported below is dominated by that choice.\n")
        return section

    def _generate_timelapse_inversion_section(self, inversion_results: Dict,
                                              method: str,
                                              site_info: Optional[Dict] = None) -> str:
        """The recovered models and how well they fit the data."""
        section = "## Inversion Results\n\n### Convergence and Data Fit\n\n"

        # One row per *iteration*, not per time step, and a time-lapse row is
        # [chi2, phi_m, phi_t]. Printing the rows as "Time Step i" claimed nine
        # time steps for a five-survey series and put the regularization terms
        # on the chi-squared scale.
        history = chi2_history(inversion_results.get('chi2_values'))
        if history:
            rows = [[i, f"{value:.3f}"] for i, value in enumerate(history, 1)]
            section += table(['Iteration', 'Chi-squared'], rows,
                             align=['right', 'right'])
            start, final = history[0], history[-1]
            section += (
                f"\nThe data misfit fell from {start:.3f} at the first iteration "
                f"to {final:.3f} after {len(history)}. A chi-squared of 1 means "
                f"the model reproduces the data to within their assigned errors; "
                f"a value above 1 means it does not, and a value below 1 means "
                f"the errors were set too generously and the model is fitting "
                f"noise.\n")
        else:
            section += "The solver did not report a chi-squared history.\n"

        final_models = inversion_results.get('final_models')
        if final_models is None:
            return section
        try:
            baseline = final_models[:, 0]
            dates = self._survey_dates(site_info, final_models.shape[1])
            baseline_date = dates[0] if dates and dates[0] else None

            section += "\n### Baseline Resistivity Distribution\n\n"
            section += facts([
                ('Survey', f"1{f' ({baseline_date})' if baseline_date else ''}"),
                ('Mean', f"{np.mean(baseline):.1f} ohm-m"),
                ('Median', f"{np.median(baseline):.1f} ohm-m"),
                ('Minimum', f"{np.min(baseline):.1f} ohm-m"),
                ('Maximum', f"{np.max(baseline):.1f} ohm-m"),
                ('Standard deviation', f"{np.std(baseline):.1f} ohm-m"),
            ], label='Statistic', value='Value')

            section += "\n### Resistivity Change Relative to the Baseline\n\n"
            rows = []
            for i in range(1, final_models.shape[1]):
                change = final_models[:, i] - baseline
                # No cause attached. A resistivity decrease is consistent with
                # wetting, and also with warming, a lithological contrast or
                # inversion sensitivity; labelling every one "moisture
                # increase" asserted exactly what the report's own
                # interpretation is careful to say ERT cannot settle.
                rows.append([
                    i + 1,
                    dates[i] if i < len(dates) and dates[i] else None,
                    f"{np.mean(change):+.2f}",
                    f"{np.min(change):.2f}",
                    f"+{np.max(change):.2f}",
                ])
            section += table(
                ['Survey', 'Acquired', 'Mean change (ohm-m)',
                 'Largest decrease (ohm-m)', 'Largest increase (ohm-m)'],
                rows, align=['right', 'left', 'right', 'right', 'right'])
            section += (
                "\nThe mean change is taken over every cell of the model, so it "
                "is small whenever wetting in one part of the section offsets "
                "drying in another; the two extreme columns are what show "
                "whether anything localised happened.\n")
        except Exception as e:
            self._log_execution(f"Could not generate temporal statistics: {e}")
            section += ("\n### Resistivity Change Relative to the Baseline\n\n"
                        "Statistics unavailable - see the run log.\n")
        return section

    def _generate_timelapse_climate_section(self, climate_data: Dict, site_info: Dict) -> str:
        """Generate climate data section for time-lapse report."""
        section = """## Climate Data Integration

### Meteorological Context

"""
        
        if not climate_data:
            # Nothing to report and nothing was asked for: a heading whose
            # only content is "there is none" invites the reader to wonder
            # what went wrong. The Key Findings say plainly that none were
            # supplied, which is where a reader looks.
            return ""
        
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
            return ""
        
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
    
    def _generate_timelapse_water_content_section(self, inversion_results: Dict,
                                                  config: Optional[Dict] = None,
                                                  site_info: Optional[Dict] = None) -> str:
        """The water-content estimate, when the run produced one.

        A request that asked for water content used to get one sentence in the
        Key Findings and nothing else: no per-step numbers, no figure, while the
        arrays sat on disk under ``petrophysics/``. The product was computed and
        then left out of the document that reports the run.

        Returns an empty string when no conversion ran, so the heading only
        appears where there is something under it.
        """
        per_step = (inversion_results or {}).get('time_lapse_water_content') or []
        if not per_step:
            return ""

        section = "## Water Content\n\n"
        reliability = water_content_reliability(inversion_results, config or {})
        layering = (per_step[0] or {}).get('layering')
        if layering:
            section += f"{layering}\n\n"
        if reliability:
            section += f"{reliability['sentence']}\n\n"

        # Surveys are numbered as they are everywhere else in the report, and
        # dated where the dates are known. The old labels ran "Baseline, Time
        # Step 1, Time Step 2 ..." for surveys 1 to 5, so "Time Step 4" in this
        # table and "Time Step 4" in the resistivity table were different
        # surveys.
        dates = self._survey_dates(site_info, len(per_step))
        rows = []
        for index, step in enumerate(per_step):
            mean = np.asarray(step.get('water_content_mean'), dtype=float).ravel()
            std = np.asarray(step.get('water_content_std'), dtype=float).ravel()
            if mean.size == 0:
                continue
            low, high = np.nanpercentile(mean, [10, 90])
            typical = float(np.nanmean(std)) if std.size else float('nan')
            rows.append([
                index + 1,
                dates[index] if index < len(dates) and dates[index] else None,
                'Baseline' if index == 0 else 'Repeat',
                f"{np.nanmean(mean):.3f}",
                f"{low:.3f} - {high:.3f}",
                f"±{typical:.3f}",
            ])
        section += table(
            ['Survey', 'Acquired', 'Role', 'Mean', '10th-90th percentile',
             'Mean uncertainty'],
            rows, align=['right', 'left', 'left', 'right', 'right', 'right'])

        # Change relative to the baseline, which is what a monitoring survey is
        # for - but only where it exceeds the error bar it is measured against.
        baseline = np.asarray(per_step[0].get('water_content_mean'), dtype=float).ravel()
        if len(per_step) > 1 and baseline.size:
            section += "\n**Change from baseline (mean water content)**\n\n"
            typical = float(np.nanmean(np.asarray(
                per_step[0].get('water_content_std'), dtype=float).ravel()))
            change_rows = []
            for index, step in enumerate(per_step[1:], start=1):
                values = np.asarray(step.get('water_content_mean'), dtype=float).ravel()
                if values.size != baseline.size:
                    continue
                change = float(np.nanmean(values - baseline))
                verdict = ("not assessed" if not np.isfinite(typical) or typical <= 0
                           else "below the uncertainty on a single value"
                           if abs(change) < typical else
                           "exceeds the uncertainty on a single value")
                change_rows.append([
                    index + 1,
                    dates[index] if index < len(dates) and dates[index] else None,
                    f"{change:+.4f}", verdict])
            section += table(
                ['Survey', 'Acquired', 'Change from baseline', 'Significance'],
                change_rows, align=['right', 'left', 'right', 'left'])

        section += ("\nWater content is derived from the resistivity models through a "
                    "petrophysical relationship, so it carries the uncertainty of that "
                    "relationship in addition to the uncertainty of the inversion.\n")
        return section

    def _generate_timelapse_visualizations(self, inversion_results: Dict,
                                          comparison_df: pd.DataFrame,
                                          climate_data: Dict,
                                          output_dir: str,
                                          site_info: Optional[Dict] = None,
                                          config: Optional[Dict] = None
                                          ) -> Dict[str, str]:
        """Every figure the time-lapse report shows, in one visual style.

        Each generator below used to choose its own size, fonts, colormap and
        way of naming a survey, which produced figures that did not look like
        they belonged to the same document. They now ask
        :mod:`PyHydroGeophysX.agents._figstyle`, and the reader can change what
        it answers through ``config['figure_style']``.
        """
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np

        style = figstyle.style_from_config(config)
        matplotlib.rcParams['font.family'] = style.font_family
        matplotlib.rcParams['font.size'] = style.label_size
        survey_dates = self._survey_dates(
            site_info, int(inversion_results.get('n_timesteps') or 0))

        vis_files = {}
        
        try:
            # 1. Baseline resistivity map
            final_models = inversion_results.get('final_models')
            mesh = inversion_results.get('mesh')
            coverage = inversion_results.get('coverage')
            
            # Debug: Print what we received
            self._log_execution("Visualization generation", level='DEBUG')
            self._log_execution(f"-> final_models type: {type(final_models)}", level='DEBUG')
            self._log_execution(
                f"-> final_models shape: {final_models.shape if final_models is not None else 'None'}",
                level='DEBUG'
            )
            self._log_execution(f"-> mesh type: {type(mesh)}", level='DEBUG')
            self._log_execution(f"-> mesh exists: {mesh is not None}", level='DEBUG')
            self._log_execution(f"-> coverage type: {type(coverage)}", level='DEBUG')
            self._log_execution(
                f"-> coverage shape: {coverage.shape if coverage is not None and hasattr(coverage, 'shape') else 'None'}",
                level='DEBUG'
            )
            
            # Convert coverage to numpy array if it's a list
            if coverage is not None and isinstance(coverage, list):
                self._log_execution("-> Converting coverage from list to numpy array", level='DEBUG')
                coverage = np.array(coverage)
                self._log_execution(f"-> coverage shape after conversion: {coverage.shape}", level='DEBUG')
            
            if final_models is not None and mesh is not None:
                self._log_execution("-> Attempting to generate baseline resistivity plot", level='DEBUG')
                try:
                    import pygimli as pg
                    self._log_execution("-> PyGIMLi imported successfully", level='DEBUG')
                    
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
                        coverage_threshold = -0.5  # Legacy coverage cutoff; not a universal resolution criterion.
                        coverage_masked = coverage_data.copy()
                        coverage_masked[coverage_data <= coverage_threshold] = -9999  # Mark as invalid
                    else:
                        coverage_data = None
                        coverage_masked = None
                    
                    # Plot baseline resistivity
                    fig_baseline, _axes = figstyle.panels(1, style)
                    ax_baseline = _axes[0]
                    
                    # Plot with baseline-masked coverage - PyGIMLi will handle the masking internally
                    ax_baseline, cbar = figstyle.pg_show(
                        mesh,
                        baseline,
                        ax=ax_baseline,
                        fig=fig_baseline,
                        cMap=style.cmap_for('resistivity'),
                        cMin=30,
                        cMax=3000,  # Optimized upper limit
                        logScale=True,
                        label=r'Resistivity ($\Omega$ m)',
                        pad=0.3,
                        orientation=style.colorbar_orientation,
                        coverage=coverage_masked
                    )
                    figstyle.apply(ax_baseline, style,
                                   figstyle.survey_title(0, survey_dates))
                    fig_baseline.tight_layout()
                    baseline_file = os.path.join(output_dir, 'baseline_resistivity.png')
                    figstyle.save(fig_baseline, baseline_file, style)
                    vis_files['baseline_resistivity'] = baseline_file
                    self._log_execution("Saved baseline resistivity plot")
                    
                except Exception as e:
                    import traceback
                    self._log_execution(f"Could not generate baseline resistivity plot: {e}", level='ERROR')
                    self._log_execution(f"Traceback: {traceback.format_exc()}", level='ERROR')
                    self._log_execution(f"Baseline resistivity plot failed: {e}", level='ERROR')
            else:
                self._log_execution(
                    f"Skipping baseline plot: final_models={final_models is not None}, mesh={mesh is not None}",
                    level='WARNING'
                )
            
            # 2. All timestep resistivity maps (1x4 layout: all 4 timesteps)
            if final_models is not None and mesh is not None:
                try:
                    import pygimli as pg

                    # Get fresh coverage reference and convert to numpy array if needed
                    coverage = inversion_results.get('coverage')
                    if coverage is not None and isinstance(coverage, list):
                        coverage = np.array(coverage)
                    
                    # Use cellMarkers to properly index the data
                    cell_markers = mesh.cellMarkers()
                    n_timesteps = min(final_models.shape[1], 4)  # Show up to 4 timesteps
                    
                    # Create subplot grid: 1 row, 4 columns
                    fig, axes = figstyle.panels(n_timesteps, style)
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
                        coverage_threshold = -0.5  # Legacy coverage cutoff; not a universal resolution criterion.
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
                        ax, cbar = figstyle.pg_show(
                            mesh,
                            timestep_data,
                            ax=ax,
                            fig=fig,
                            cMap=style.cmap_for('resistivity'),
                            cMin=30,
                            cMax=3000,
                            logScale=True,
                            label=r'Resistivity ($\Omega$ m)',
                            pad=0.3,
                            orientation=style.colorbar_orientation,
                            coverage=coverage_masked
                        )
                        
                        figstyle.apply(ax, style,
                                       figstyle.survey_title(i, survey_dates),
                                       ylabel='Elevation (m)' if i == 0 else '')
                    
                    fig.tight_layout()
                    all_timesteps_file = os.path.join(output_dir, 'timelapse_all_resistivity.png')
                    figstyle.save(fig, all_timesteps_file, style)
                    vis_files['timelapse_all_resistivity'] = all_timesteps_file
                    self._log_execution("Saved all timesteps resistivity plot")
                    
                except Exception as e:
                    import traceback
                    self._log_execution(f"Could not generate all timesteps resistivity plot: {e}", level='ERROR')
                    self._log_execution(f"Traceback: {traceback.format_exc()}", level='ERROR')
                    self._log_execution(f"All timesteps plot failed: {e}", level='ERROR')
            
            # 3. Time-lapse resistivity percentage change maps (1x4 layout: baseline + changes)
            if final_models is not None and mesh is not None:
                try:
                    import pygimli as pg

                    # Get fresh coverage reference and convert to numpy array if needed
                    coverage = inversion_results.get('coverage')
                    if coverage is not None and isinstance(coverage, list):
                        coverage = np.array(coverage)
                    
                    # Use cellMarkers to properly index the data
                    cell_markers = mesh.cellMarkers()
                    baseline = final_models[:, 0][cell_markers] if len(final_models) > len(cell_markers) else final_models[:, 0]
                    n_timesteps = min(final_models.shape[1] - 1, 3)  # Show up to 3 changes (4 total with baseline)
                    
                    # Create subplot grid: 1 row, 4 columns (baseline + 3 changes)
                    n_total_plots = n_timesteps + 1  # baseline + changes
                    fig, axes = figstyle.panels(n_total_plots, style)
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
                        coverage_threshold = -0.5  # Legacy coverage cutoff; not a universal resolution criterion.
                        coverage_masked = coverage_data.copy()
                        coverage_masked[coverage_data <= coverage_threshold] = -9999  # Mark as invalid
                    else:
                        coverage_data = None
                        coverage_masked = None
                    
                    # Plot 1: Baseline resistivity
                    ax = axes[0]
                    ax, cbar = figstyle.pg_show(
                        mesh,
                        baseline,
                        ax=ax,
                        fig=fig,
                        cMap=style.cmap_for('resistivity'),
                        cMin=30,
                        cMax=3000,
                        logScale=True,
                        label=r'Resistivity ($\Omega$ m)',
                        pad=0.3,
                        orientation=style.colorbar_orientation,
                        coverage=coverage_masked
                    )
                    figstyle.apply(ax, style,
                                   figstyle.survey_title(0, survey_dates))
                    
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
                        ax, cbar = figstyle.pg_show(
                            mesh,
                            percent_change,
                            ax=ax,
                            fig=fig,
                            cMap=style.cmap_for('change'),
                            cMin=-50,  # Optimized range from testing
                            cMax=50,
                            label=r'$\Delta\rho$ (%)',
                            pad=0.3,
                            orientation=style.colorbar_orientation,
                            coverage=coverage_masked
                        )
                        
                        figstyle.apply(ax, style,
                                       figstyle.change_title(i, survey_dates),
                                       ylabel='')
                    
                    fig.tight_layout()
                    tl_changes_file = os.path.join(output_dir, 'timelapse_resistivity_changes_percent.png')
                    figstyle.save(fig, tl_changes_file, style)
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
                    # The three blocks above each do this; the one missing here
                    # is why only the absolute-changes figure died on a list.
                    if coverage is not None and isinstance(coverage, list):
                        coverage = np.array(coverage)
                    
                    # Use cellMarkers to properly index the data
                    cell_markers = mesh.cellMarkers()
                    baseline = final_models[:, 0][cell_markers] if len(final_models) > len(cell_markers) else final_models[:, 0]
                    n_timesteps = min(final_models.shape[1] - 1, 3)  # Show up to 3 changes (4 total with baseline)
                    
                    # Create subplot grid: 1 row, 4 columns (baseline + 3 changes)
                    n_total_plots = n_timesteps + 1  # baseline + changes
                    fig, axes = figstyle.panels(n_total_plots, style)
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
                        coverage_threshold = -0.5  # Legacy coverage cutoff; not a universal resolution criterion.
                        coverage_masked = coverage_data.copy()
                        coverage_masked[coverage_data <= coverage_threshold] = -9999  # Mark as invalid
                    else:
                        coverage_data = None
                        coverage_masked = None
                    
                    # Plot 1: Baseline resistivity
                    ax = axes[0]
                    ax, cbar = figstyle.pg_show(
                        mesh,
                        baseline,
                        ax=ax,
                        fig=fig,
                        cMap=style.cmap_for('resistivity'),
                        cMin=30,
                        cMax=3000,
                        logScale=True,
                        label=r'Resistivity ($\Omega$ m)',
                        pad=0.3,
                        orientation=style.colorbar_orientation,
                        coverage=coverage_masked
                    )
                    figstyle.apply(ax, style,
                                   figstyle.survey_title(0, survey_dates))
                    
                    # Plots 2-4: Absolute changes
                    time_labels = self._generate_time_labels(n_timesteps)
                    
                    for i in range(n_timesteps):
                        ax = axes[i+1]  # Offset by 1 for baseline
                        # Get data for this timestep using cellMarkers
                        timestep_data = final_models[:, i+1][cell_markers] if len(final_models) > len(cell_markers) else final_models[:, i+1]
                        change = timestep_data - baseline
                        
                        # Plot with baseline-masked coverage - PyGIMLi will handle the masking internally
                        ax, cbar = figstyle.pg_show(
                            mesh,
                            change,
                            ax=ax,
                            fig=fig,
                            cMap=style.cmap_for('change'),
                            cMin=-300,  # Optimized range from testing
                            cMax=300,
                            label=r'$\Delta\rho$ ($\Omega$ m)',
                            pad=0.3,
                            orientation=style.colorbar_orientation,
                            coverage=coverage_masked
                        )
                        
                        figstyle.apply(ax, style,
                                       figstyle.change_title(i, survey_dates),
                                       ylabel='')
                    
                    fig.tight_layout()
                    tl_changes_abs_file = os.path.join(output_dir, 'timelapse_resistivity_changes_absolute.png')
                    figstyle.save(fig, tl_changes_abs_file, style)
                    vis_files['timelapse_changes_absolute'] = tl_changes_abs_file
                    self._log_execution("Saved time-lapse resistivity absolute changes plot")
                    
                except Exception as e:
                    import traceback
                    self._log_execution(f"Could not generate time-lapse absolute changes plot: {e}", level='ERROR')
                    self._log_execution(f"Traceback: {traceback.format_exc()}", level='ERROR')
                    self._log_execution(f"Time-lapse changes plot failed: {e}", level='ERROR')
            
            # 2. Climate-Resistivity comparison plots
            if comparison_df is not None and not comparison_df.empty and climate_data is not None:
                try:
                    # Set Arial font for all text
                    plt.rcParams['font.family'] = 'Arial'
                    
                    fig, axes = figstyle.panels(1, style, rows=2)
                    
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
                            ax_top_left.set_ylabel('PET (mm/day)', fontsize=style.label_size,
                                                  color='#FF8C00')
                            ax_top_left.tick_params(axis='y', labelcolor='#FF8C00', labelsize=11)
                        
                        # Precipitation on right axis (bar plot) - using daily data
                        if 'prcp' in daily_df.columns:
                            ax_top_right.bar(daily_dates, daily_df['prcp'], 
                                            alpha=0.6, width=1.0, label='Precipitation (daily)',
                                            color='#4682B4', zorder=1)
                            ax_top_right.set_ylabel('Precipitation (mm/day)', fontsize=style.label_size,
                                                   color='#4682B4')
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
                                             fontsize=style.title_size, pad=15)
                        ax_top_left.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                        ax_top_left.tick_params(axis='y', labelsize=11)
                        ax_top_left.tick_params(axis='both', which='major', length=6, width=1.5)
                        ax_top_right.tick_params(axis='both', which='major', length=6, width=1.5)
                        
                        # Add legends
                        lines1, labels1 = ax_top_left.get_legend_handles_labels()
                        lines2, labels2 = ax_top_right.get_legend_handles_labels()
                        ax_top_left.legend(lines1 + lines2, labels1 + labels2, 
                                          loc='upper left', fontsize=style.tick_size, framealpha=0.9)
                        
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
                        
                        ax_bottom.set_xlabel('Date', fontsize=style.label_size)
                        ax_bottom.set_ylabel('Temperature (°C)', fontsize=style.label_size)
                        ax_bottom.set_title('Temperature Variations', fontsize=style.title_size, 
                                           pad=15)
                        ax_bottom.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                        ax_bottom.tick_params(axis='x', rotation=45, labelsize=11)
                        ax_bottom.tick_params(axis='y', labelsize=11)
                        ax_bottom.tick_params(axis='both', which='major', length=6, width=1.5)
                        ax_bottom.legend(loc='best', fontsize=style.tick_size, framealpha=0.9, 
                                       edgecolor='gray', fancybox=True, ncol=2)
                    
                    fig.tight_layout()
                    climate_corr_file = os.path.join(output_dir, 'climate_data_visualization.png')
                    figstyle.save(fig, climate_corr_file, style)
                    vis_files['climate_correlation'] = climate_corr_file
                    self._log_execution("Saved climate data visualization plot")
                    
                    # Reset font settings
                    plt.rcParams['font.family'] = 'sans-serif'
                    
                except Exception as e:
                    self._log_execution(f"Could not generate climate visualization plot: {e}")
            
        except Exception as e:
            self._log_execution(f"Error generating time-lapse visualizations: {e}")

        # Deliberately outside the block above. This figure was written inside
        # that except handler, so it was drawn only when the resistivity
        # figures failed - which is to say never on a successful run. The
        # product the request asked for reached the report as a table and no
        # picture, twice, while the arrays sat on disk.
        self._timelapse_water_content_figure(inversion_results, output_dir, vis_files,
                                             site_info, style)
        return vis_files

    def _timelapse_water_content_figure(self, inversion_results: Dict, output_dir: str,
                                        vis_files: Dict[str, str],
                                        site_info: Optional[Dict] = None,
                                        style: Optional[Any] = None) -> None:
        """Water content per survey, on one shared colour scale.

        Shared limits are the point of the figure: with per-panel scaling every
        survey looks the same and the change between them - which is what a
        monitoring survey is for - is exactly what cannot be seen.

        Adds to ``vis_files`` in place and never raises; a figure that cannot be
        drawn is logged and the rest of the report still goes out.
        """
        per_step = (inversion_results or {}).get('time_lapse_water_content') or []
        mesh = (inversion_results or {}).get('mesh')
        if not per_step:
            return
        if mesh is None:
            self._log_execution(
                "Water content was computed but no mesh was returned, so it "
                "could not be plotted.", level='WARNING')
            return
        try:
            import matplotlib.pyplot as plt
            import pygimli as pg

            style = style or figstyle.FigureStyle()
            cell_markers = mesh.cellMarkers()
            steps = [np.asarray(s.get('water_content_mean'), dtype=float).ravel()
                     for s in per_step
                     if s.get('water_content_mean') is not None]
            if not steps:
                return
            stacked = np.concatenate(steps)
            lo, hi = np.nanpercentile(stacked, [2, 98])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(np.nanmin(stacked)), float(np.nanmax(stacked))
            fig, axes = figstyle.panels(len(steps), style)
            dates = self._survey_dates(site_info, len(steps))
            for i, values in enumerate(steps):
                data = (values[cell_markers]
                        if len(values) > len(cell_markers) else values)
                figstyle.pg_show(mesh, data, ax=axes[i], fig=fig, cMap=style.cmap_for('water_content'),
                        cMin=lo, cMax=hi, logScale=False,
                        label='Water content (-)', pad=0.3,
                        orientation=style.colorbar_orientation)
                figstyle.apply(axes[i], style,
                               figstyle.survey_title(i, dates),
                               ylabel='Elevation (m)' if i == 0 else '')
            fig.tight_layout()
            wc_file = os.path.join(output_dir, 'timelapse_water_content.png')
            figstyle.save(fig, wc_file, style)
            vis_files['timelapse_water_content'] = wc_file
            self._log_execution("Saved time-lapse water content plot")
        except Exception as e:
            self._log_execution(f"Could not generate water content plot: {e}",
                                level='ERROR')

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
            # No heading of its own. The compiler already places this under
            # "## Interpretation"; a second heading here produced
            # "6. Interpretation" followed immediately by "7. Integrated
            # Analysis and Interpretation" with nothing in between.
            caveat = "**AI-generated interpretation - verify before citing.**"
            return f"{caveat}\n\n{narrative}\n"
        except Exception as e:
            self._log_execution(f"Could not generate narrative: {e}")
            return ""
    
    def _timelapse_key_findings(self, inversion_results, climate_data,
                                evaluation_results, config=None) -> str:
        """Key findings written from what the run produced, not from a template.

        The fixed text this replaced asserted three things every time: that
        systematic changes were observed, that climate-resistivity correlations
        gave insight, and that convergence was good. On the run that prompted
        this, no climate data had been supplied at all and the quality score was
        54.8/100 - so two of the three claims were false, printed directly under
        an AI narrative that had correctly hedged both.

        Each finding here is derived, and a finding with nothing behind it is
        omitted rather than asserted.
        """
        findings = []
        results = inversion_results or {}

        # 1. Resistivity change - reported only with numbers behind it.
        models = results.get('final_models')
        if models is not None and getattr(models, 'ndim', 0) == 2 and models.shape[1] > 1:
            baseline = models[:, 0]
            changes = [float(np.mean(models[:, i] - baseline))
                       for i in range(1, models.shape[1])]
            largest = max(changes, key=abs)
            direction = 'an increase' if largest > 0 else 'a decrease'
            findings.append(
                f"**Temporal Resistivity Changes:** Mean resistivity changed by "
                f"{changes[0]:+.2f} to {changes[-1]:+.2f} ohm-m relative to the baseline "
                f"across {len(changes)} repeat survey{'s' if len(changes) != 1 else ''}; "
                f"the largest mean shift was "
                f"{direction} of {abs(largest):.2f} ohm-m. Whether this reflects moisture, "
                f"temperature or inversion sensitivity is not established by these numbers "
                f"alone.")
        else:
            findings.append(
                "**Temporal Resistivity Changes:** Not summarised - the run did not return "
                "a model per time step.")

        # 2. Climate - say what was integrated, which is often nothing.
        if climate_data:
            findings.append(
                "**Climate-Resistivity Relationships:** Meteorological data were integrated; "
                "see the correlation section for the relationships actually computed.")
        else:
            findings.append(
                "**Climate-Resistivity Relationships:** No meteorological data were supplied, "
                "so no climate-resistivity relationship was computed and none should be "
                "inferred from this report.")

        # 3. Data quality - the measured score and misfit, whatever they say.
        evaluation = evaluation_results or {}
        score = evaluation.get('quality_score')
        history = chi2_history(results.get('chi2_values'))
        final_chi2 = history[-1] if history else None
        if score is not None:
            verdict = ('meets the configured threshold' if evaluation.get('status') == 'success'
                       else 'is below the configured threshold and needs review')
            quality_line = f"a heuristic quality score of {float(score):.1f}/100, which {verdict}"
        else:
            quality_line = "no quality score (the evaluation did not run or did not report one)"
        if final_chi2 is not None:
            fit = (f"a final chi-squared of {final_chi2:.3f}"
                   + (" (close to the target of 1)" if 0.8 <= final_chi2 <= 1.5 else
                      ", further from the target of 1 than a well-fitted inversion" ))
        else:
            fit = "no reported chi-squared"
        findings.append(f"**Data Quality:** The inversion returned {fit} and {quality_line}. "
                        f"A chi-squared near one does not by itself establish that the model "
                        f"is correct or unique.")

        # 4. Water content, when the run produced it.
        per_step = results.get('time_lapse_water_content') or []
        if per_step:
            reliability = water_content_reliability(results, config or {})
            verdict = f" {reliability['sentence']}" if reliability else ""
            # How the section was divided belongs next to the number, not in a
            # log line: one parameter set for the whole profile is part of why
            # the uncertainty is what it is.
            layering = (per_step[0] or {}).get('layering') if per_step else None
            structure = f" {layering}" if layering else ""
            findings.append(
                f"**Water Content:** Converted for {len(per_step)} time step(s) by Monte "
                f"Carlo petrophysics.{structure}{verdict}")

        return '\n\n'.join(f"{i}. {text}" for i, text in enumerate(findings, 1))

    # Captions live in FIGURE_CATALOG (PyHydroGeophysX.agents._figures) next to
    # the topic each figure belongs to, so the planner and the captions cannot
    # disagree about what a figure is.

    def _read_figure_request(self, config: Optional[Dict] = None
                             ) -> Optional[Dict[str, Any]]:
        """What the request asks to see, and how it asks it to look.

        Called before any figure is drawn, because a style preference is only
        useful to the generators that have not run yet. Writes the style it
        finds into ``config['figure_style']``, under anything already there -
        an explicit setting outranks a sentence.

        Returns the model's answer so the figure plan can reuse it instead of
        asking a second time.
        """
        config = config if config is not None else {}
        request = str(config.get('user_request') or '').strip()
        if not request or not getattr(self, 'api_key', None):
            return None

        def query(prompt):  # noqa: D401 - a one-shot question, no tools
            return self.query_llm(prompt, max_tokens=250)

        try:
            asked = llm_figure_topics(request, query)
        except Exception as e:  # noqa: BLE001 - a preference must not fail a run
            self._log_execution(f"Could not read figure preferences: {e}",
                                level='WARNING')
            return None
        if asked and asked.get('style'):
            config['figure_style'] = {**asked['style'],
                                      **(config.get('figure_style') or {})}
            self._log_execution(f"Figure style from the request: {asked['style']}")
        return asked

    def _figure_plan(self, vis_files: Dict[str, str],
                     config: Optional[Dict] = None,
                     asked: Optional[Dict[str, Any]] = None) -> Dict[str, list]:
        """Which figures this request wants shown, read by the model.

        The figure set used to be whatever the plotting code reached, so a
        request for water content could come back with four resistivity
        figures and none of what was asked for, and nothing compared the two.
        The request is now read for the subjects it wants to see, and the plan
        names what is drawn, what was deliberately left out, and what was asked
        for but has no data behind it.

        Falls back to drawing everything when there is no model to ask: an
        unanswered question must not be able to suppress a figure.
        """
        config = config if config is not None else {}
        request = str(config.get('user_request') or '').strip()
        try:
            return plan_figures(list(vis_files), request, topics=asked)
        except Exception as e:  # noqa: BLE001 - planning must never lose figures
            self._log_execution(f"Figure planning failed, showing all: {e}",
                                level='WARNING')
            return {'draw': list(vis_files), 'omitted': [], 'missing': []}

    def _figures_section(self, vis_files: Dict[str, str],
                         plan: Optional[Dict[str, list]] = None) -> str:
        """The planned figures, numbered and captioned.

        A figure left out on purpose is named rather than dropped silently -
        the reader is told the run has it and where to find it, which is the
        difference between an edited report and an incomplete one.
        """
        if not vis_files:
            return ""
        plan = plan or {'draw': list(vis_files), 'omitted': [], 'missing': []}
        drawn = [key for key in plan.get('draw') or [] if key in vis_files]
        if not drawn:
            return ""
        section = "## Figures\n\n"
        for number, vis_type in enumerate(drawn, 1):
            caption = (FIGURE_CATALOG.get(vis_type, {}).get('caption')
                       or vis_type.replace('_', ' ').capitalize() + '.')
            section += f"![Figure {number}]({os.path.basename(vis_files[vis_type])})\n\n"
            section += f"**Figure {number}.** {caption}\n\n"
        omitted = [key for key in plan.get('omitted') or [] if key in vis_files]
        if omitted:
            names = ', '.join(f"`{os.path.basename(vis_files[key])}`" for key in omitted)
            section += (f"The run also produced {names}, left out here because the "
                        f"request asked for a narrower set. They are in the output "
                        f"folder.\n\n")
        return section

    def _timelapse_recommendations(self, climate_data: Optional[Dict],
                                   inversion_results: Dict,
                                   config: Optional[Dict] = None) -> str:
        """Next steps that follow from this run, not a standing wish list.

        A recommendation to integrate climate data is useful to a run that had
        none and noise to a run that had it; the same is true of calibrating
        petrophysics. Each item here is conditional on what the run actually
        lacked.
        """
        config = config or {}
        items = []
        if not climate_data:
            items.append(
                "**Supply meteorological data.** Precipitation, temperature and "
                "potential evapotranspiration covering the survey dates would "
                "let the recovered changes be tested against the forcing that "
                "plausibly caused them. Until that is done, no hydrological "
                "cause for these changes is established.")
        if inversion_results.get('time_lapse_water_content') and not config.get(
                'petrophysical_params'):
            items.append(
                "**Calibrate the petrophysical relationship.** Samples or "
                "borehole logs from this site would replace the generated "
                "default parameters, which are the largest single contributor "
                "to the water-content uncertainty reported above.")
        items += [
            "**Extend the time series.** Five surveys over five days resolve "
            "an event, not a seasonal cycle; repeat surveys through a wetting "
            "and a drying season would separate the two."
            if (inversion_results.get('n_timesteps') or 0) <= 6 else
            "**Continue monitoring** on the established schedule so the "
            "seasonal envelope of the site can be established.",
            "**Examine the depth dependence of the recovered changes** against "
            "model sensitivity, so that changes reported at depth can be "
            "distinguished from the inversion's decreasing resolution there.",
            "**Validate against direct measurement.** Soil-moisture probes or "
            "borehole observations at one or two locations would anchor the "
            "converted water content to a measured value.",
        ]
        return numbered(items)

    def _compile_timelapse_report(self, front_matter: str, exec_summary: str,
                                  method_section: str, inv_section: str,
                                  water_content_section: str,
                                  climate_section: str, corr_section: str,
                                  figures_section: str, narrative: str,
                                  recommendations: str) -> str:
        """Assemble the deliverable in the order a reader needs it.

        Scope and findings first, then how the work was done, then what it
        produced, then what it is taken to mean, then what to do next. The
        interpretation sits after the results rather than before them so that
        its claims are read next to the numbers they rest on; it used to be
        printed immediately under the summary, where a model-written paragraph
        was the first technical content in the document.

        Section numbers are applied at the end by
        :func:`PyHydroGeophysX.agents._document.renumber` so that sections
        omitted for want of data - climate, correlation, water content - do not
        leave gaps in the numbering.
        """
        body = "\n\n".join(part.strip() for part in [
            exec_summary, method_section, inv_section, water_content_section,
            climate_section, corr_section, figures_section,
            f"## Interpretation\n\n{narrative.strip()}" if narrative else '',
            f"## Recommendations\n\n{recommendations.strip()}"
            if recommendations else '',
        ] if part and part.strip())
        return front_matter.rstrip() + "\n\n" + renumber(body) + "\n"

    def _format_chi2(self, chi2_values) -> str:
        """The headline chi-squared: what the inversion converged to.

        Not a mean. Averaging a time-lapse table folded the model- and
        temporal-regularization columns in with the misfit and reported
        1442.924 for a run whose chi-squared was 1.630; averaging a
        single-survey history mixes the starting misfit into the answer. The
        number a reader wants is the last iteration's.
        """
        if chi2_values is None:
            return 'N/A'
        if isinstance(chi2_values, (int, float)):
            return f"{chi2_values:.3f}"
        history = chi2_history(chi2_values)
        if not history:
            return 'N/A'
        if len(history) == 1:
            return f"{history[0]:.3f}"
        return f"{history[-1]:.3f} (final of {len(history)} iterations)"
    
    def _generate_time_labels(self, n_timesteps: int) -> list:
        """Generate time labels for plots."""
        # No acquisition dates are supplied here; label by one-based step index.
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
            with open(report_file, 'w', encoding='utf-8') as f:
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
        import matplotlib
        import matplotlib.pyplot as plt
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
                
                fig = figstyle.detached_figure((12, 12))
                ax1, ax2, ax3 = fig.subplots(3, 1)
                
                # Top: Velocity with coverage
                figstyle.pg_show(velocity_mesh, velocity_model, ax=ax1, cMap='jet',
                       colorBar=True, label='Velocity (m/s)',
                       coverage=seismic_coverage, cMin=500, cMax=3600)
                ax1.set_title('Seismic Velocity Model ',
                            fontsize=13, fontweight='bold')
                ax1.set_xlabel('Distance (m)')
                ax1.set_ylabel('Depth (m)' )                 
                # Middle: Velocity with interface
                figstyle.pg_show(velocity_mesh, velocity_model, ax=ax2, cMap='jet',
                       colorBar=True, label='Velocity (m/s)', cMin=500, cMax=3600,
                       coverage=seismic_coverage)
                ax2.plot(interface_x, interface_z, 'r-', linewidth=2,
                        label=f'Interface ({velocity_threshold} m/s)')
                ax2.legend()
                ax2.set_title(f'Extracted Interface at {velocity_threshold} m/s',
                            fontsize=13, fontweight='bold')
                ax2.set_xlabel('Distance (m)')
                ax2.set_ylabel('Depth (m)' )               
                # Bottom: Structure-constrained resistivity
                figstyle.pg_show(para_mesh, resistivity_model, ax=ax3, cMap='jet',
                       colorBar=True, label='Resistivity (Ωm)',
                       coverage=coverage_numeric>coverage_threshold,
                       logScale=True, cMin=10, cMax=2000)
                ax3.set_title(f'Structure-Constrained Resistivity ',
                            fontsize=13, fontweight='bold')
                ax3.set_xlabel('Distance (m)')
                ax3.set_ylabel('Depth (m)')
                
                fig.tight_layout()
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
                
                # Ensure data matches mesh size
                wc_mean_flat = water_content_mean.ravel()
                wc_std_flat = water_content_std.ravel()
                
                # Handle size mismatch - pad or truncate to match mesh
                mesh_cells = para_mesh.cellCount()
                if len(wc_mean_flat) != mesh_cells:
                    self._log_execution(f"Warning: Data size ({len(wc_mean_flat)}) != mesh size ({mesh_cells}), adjusting...")
                    if len(wc_mean_flat) < mesh_cells:
                        # Pad with NaN
                        wc_mean_padded = np.full(mesh_cells, np.nan)
                        wc_std_padded = np.full(mesh_cells, np.nan)
                        wc_mean_padded[:len(wc_mean_flat)] = wc_mean_flat
                        wc_std_padded[:len(wc_std_flat)] = wc_std_flat
                        wc_mean_flat = wc_mean_padded
                        wc_std_flat = wc_std_padded
                    else:
                        # Truncate
                        wc_mean_flat = wc_mean_flat[:mesh_cells]
                        wc_std_flat = wc_std_flat[:mesh_cells]
                
                fig = figstyle.detached_figure((14, 5))
                ax1, ax2 = fig.subplots(1, 2)
                
                # Left: Mean water content
                figstyle.pg_show(para_mesh, wc_mean_flat, ax=ax1,
                       cMap='Blues', colorBar=True, label='Water Content (-)',
                       coverage=coverage_numeric>coverage_threshold,
                       cMin=0, cMax=0.5, logScale=False)
                ax1.set_title(f'Mean Water Content ',
                            fontsize=13, fontweight='bold')
                ax1.set_xlabel('Distance (m)')
                ax1.set_ylabel('Depth (m)')
                
                # Right: Uncertainty
                figstyle.pg_show(para_mesh, wc_std_flat, ax=ax2,
                       cMap='Reds', colorBar=True, label='Uncertainty (std)',
                       coverage=coverage_numeric>coverage_threshold,
                       cMin=0, cMax=0.1, logScale=False)
                ax2.set_title(f'Water Content Uncertainty',
                            fontsize=13, fontweight='bold')
                ax2.set_xlabel('Distance (m)')
                ax2.set_ylabel('Depth (m)')
                
                fig.tight_layout()
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
    
    def _save_html_report(self, markdown_content: str, output_dir: str,
                          filename: str = 'workflow_report') -> None:
        """HTML output was removed; reports are Markdown only.

        The conversion depended on the optional ``markdown`` package and, when
        it was absent, every run logged two warnings and produced nothing. The
        Markdown report is the deliverable, and anything that needs HTML can
        convert it downstream. Kept as a no-op so the call sites and the
        ``html_file`` key in the results keep working.
        """
        return None

    def _save_pdf_report(self, markdown_content: str, output_dir: str, 
                         visualization_files: Optional[Dict[str, str]] = None,
                         filename: str = 'workflow_report') -> Optional[str]:
        """
        Convert markdown report to PDF.
        
        Tries multiple PDF generation methods:
        1. weasyprint (requires weasyprint package)
        2. pdfkit (requires pdfkit and wkhtmltopdf)
        3. markdown-pdf (requires md2pdf)
        
        Args:
            markdown_content: Report in markdown format
            output_dir: Output directory
            visualization_files: Dict of visualization files to embed
            filename: Output filename without extension
            
        Returns:
            Path to PDF file or None if conversion failed
        """
        pdf_file = os.path.join(output_dir, f'{filename}.pdf')
        
        # First, get HTML content
        html_content = None
        try:
            import markdown
            html_body = markdown.markdown(
                markdown_content, 
                extensions=['tables', 'fenced_code', 'toc']
            )
            
            # Create HTML with embedded images using absolute paths
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 800px; margin: auto; }}
        h1 {{ color: #0f4c75; border-bottom: 2px solid #0f4c75; padding-bottom: 10px; }}
        h2 {{ color: #1b4f72; margin-top: 30px; }}
        h3 {{ color: #2874a6; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #0f4c75; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        code {{ background-color: #e5e7eb; padding: 2px 6px; border-radius: 3px; font-family: monospace; }}
        img {{ max-width: 100%; height: auto; margin: 10px 0; }}
        hr {{ border: none; border-top: 1px solid #ccc; margin: 20px 0; }}
        @page {{ margin: 2cm; }}
    </style>
</head>
<body>
{html_body}
</body>
</html>"""
        except ImportError:
            # Names the deliverable now that HTML output is gone and this is the
            # only path left that wants the package.
            self._log_execution("markdown package not installed; skipping the PDF "
                                "(the Markdown report is still written)", level='WARNING')
            return None
        
        # Method 1: Try weasyprint
        try:
            from weasyprint import HTML
            HTML(string=html_content, base_url=output_dir).write_pdf(pdf_file)
            self._log_execution(f"PDF report saved (weasyprint): {pdf_file}")
            return pdf_file
        except ImportError:
            pass
        except Exception as e:
            self._log_execution(f"weasyprint failed: {e}", level='WARNING')
        
        # Method 2: Try pdfkit
        try:
            import pdfkit
            pdfkit.from_string(html_content, pdf_file, options={'encoding': 'UTF-8'})
            self._log_execution(f"PDF report saved (pdfkit): {pdf_file}")
            return pdf_file
        except ImportError:
            pass
        except Exception as e:
            self._log_execution(f"pdfkit failed: {e}", level='WARNING')
        
        # Method 3: Try md2pdf
        try:
            from md2pdf.core import md2pdf as convert_md2pdf
            convert_md2pdf(
                pdf_file,
                md_content=markdown_content,
                css_file_path=None,
                base_url=output_dir
            )
            self._log_execution(f"PDF report saved (md2pdf): {pdf_file}")
            return pdf_file
        except ImportError:
            pass
        except Exception as e:
            self._log_execution(f"md2pdf failed: {e}", level='WARNING')
        
        # Method 4: Try fpdf2 (basic text-based PDF)
        try:
            from fpdf import FPDF
            
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font('Helvetica', size=10)
            
            # Parse markdown to simple text
            lines = markdown_content.split('\n')
            for line in lines:
                # Handle headers
                if line.startswith('# '):
                    pdf.set_font('Helvetica', 'B', 16)
                    pdf.multi_cell(0, 10, line[2:])
                    pdf.set_font('Helvetica', size=10)
                elif line.startswith('## '):
                    pdf.set_font('Helvetica', 'B', 14)
                    pdf.multi_cell(0, 8, line[3:])
                    pdf.set_font('Helvetica', size=10)
                elif line.startswith('### '):
                    pdf.set_font('Helvetica', 'B', 12)
                    pdf.multi_cell(0, 7, line[4:])
                    pdf.set_font('Helvetica', size=10)
                elif line.startswith('**') and line.endswith('**'):
                    pdf.set_font('Helvetica', 'B', 10)
                    pdf.multi_cell(0, 6, line.strip('*'))
                    pdf.set_font('Helvetica', size=10)
                elif line.startswith('- '):
                    pdf.multi_cell(0, 5, '  • ' + line[2:])
                elif line.startswith('!['):
                    # Try to embed image
                    try:
                        import re
                        match = re.search(r'!\[.*?\]\((.*?)\)', line)
                        if match:
                            img_path = match.group(1)
                            if not os.path.isabs(img_path):
                                img_path = os.path.join(output_dir, img_path)
                            if os.path.exists(img_path):
                                pdf.image(img_path, w=180)
                    except Exception:
                        pass
                elif line.strip():
                    pdf.multi_cell(0, 5, line)
                else:
                    pdf.ln(3)
            
            pdf.output(pdf_file)
            self._log_execution(f"PDF report saved (fpdf2): {pdf_file}")
            return pdf_file
        except ImportError:
            pass
        except Exception as e:
            self._log_execution(f"fpdf2 failed: {e}", level='WARNING')
        
        self._log_execution("No PDF library available. Install weasyprint, pdfkit, md2pdf, or fpdf2", level='WARNING')
        return None
    
    def _log_execution(self, message: str, level: str = 'INFO'):
        """Log execution message."""
        prefix = f"[{self.name}] [{level}] "
        try:
            print(f"{prefix}{message}")
        except UnicodeEncodeError:
            # Keep logging robust on Windows terminals with non-UTF-8 code pages.
            encoding = getattr(sys.stdout, "encoding", None) or "ascii"
            safe_message = message.encode(encoding, errors="replace").decode(encoding, errors="replace")
            print(f"{prefix}{safe_message}")


