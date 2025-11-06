"""
Report Generation Agent

Specialized agent for generating comprehensive reports from workflow results.
"""

from typing import Dict, Any, Optional
import os
from datetime import datetime
from .base_agent import BaseAgent


class ReportAgent(BaseAgent):
    """
    Agent specialized in generating comprehensive reports.
    
    Aggregates results from all workflow steps and generates reports
    with visualizations, statistics, and interpretations.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Report Agent."""
        super().__init__("report_generator", api_key)
        self.system_message = """You are an expert in technical report writing for 
geophysical and hydrological studies. Your role is to synthesize results from ERT 
data processing, inversion, and water content analysis into clear, informative 
reports suitable for scientists and engineers."""
    
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
            
            # 3. Inversion Results Summary
            inversion_summary = self._generate_inversion_summary(workflow_data)
            
            # 4. Water Content Analysis Summary
            wc_summary = self._generate_wc_summary(workflow_data)
            
            # 5. Visualizations (create plots)
            visualization_files = self._generate_visualizations(workflow_data, output_dir)
            
            # 6. Generate LLM-enhanced narrative report
            narrative_report = None
            if self.api_key:
                self._log_execution("Generating narrative report with LLM")
                narrative_report = self._generate_narrative_report(
                    executive_summary, data_summary, inversion_summary, wc_summary
                )
            
            # Compile full report
            full_report = self._compile_report(
                executive_summary,
                data_summary,
                inversion_summary,
                wc_summary,
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
            if mean_wc is not None and hasattr(mean_wc, '__iter__'):
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
                                   inv_summary: str, wc_summary: str) -> str:
        """Generate narrative report using LLM."""
        try:
            combined_info = f"{exec_summary}\n{data_summary}\n{inv_summary}\n{wc_summary}"
            
            prompt = f"""Based on the following workflow results, write a cohesive narrative 
summary (3-4 paragraphs) that:
1. Describes the overall workflow and objectives
2. Summarizes the key findings
3. Highlights any notable patterns or anomalies
4. Provides recommendations for next steps

Workflow Results:
{combined_info}

Write in a professional, technical style suitable for a geophysical survey report."""
            
            narrative = self.query_llm(prompt, self.system_message, 
                                      temperature=0.6, max_tokens=500)
            return f"\n## Narrative Summary\n\n{narrative}\n"
        except:
            return ""
    
    def _compile_report(self, exec_summary: str, data_summary: str,
                       inv_summary: str, wc_summary: str, narrative: str,
                       vis_files: Dict[str, str]) -> str:
        """Compile full report."""
        report = f"""# Geophysical Workflow Report
Generated by PyHydroGeophysX Multi-Agent System

{exec_summary}
{narrative if narrative else ''}
{data_summary}
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
