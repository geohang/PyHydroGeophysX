"""
Inversion Evaluation Agent

Specialized agent for evaluating ERT inversion quality and automatically
adjusting regularization parameters to achieve optimal results.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Inversion Evaluation Agent
# ---------------------------------------------------------------------------
class InversionEvaluationAgent(BaseAgent):
    """
    Agent specialized in evaluating inversion quality and optimizing parameters.
    
    This agent:
    1. Evaluates inversion results using multiple quality metrics
    2. Determines if results are acceptable
    3. Automatically adjusts regularization parameters if needed
    4. Triggers re-inversion with improved parameters
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 llm_provider: str = "openai"):
        """Initialize Inversion Evaluation Agent."""
        super().__init__("inversion_evaluation", api_key, model, llm_provider)
        self.system_message = """You are an expert in geophysical inversion quality assessment.
Your role is to evaluate ERT inversion results based on data fit, model smoothness,
and physical plausibility. You understand chi-squared statistics, L-curves, and
optimal regularization parameter selection."""
        
        # Quality thresholds
        self.quality_thresholds = {
            'chi2_target': 1.0,  # Target chi-squared value
            'chi2_acceptable_range': (0.8, 1.5),  # Acceptable chi-squared range
            'chi2_poor': (0.0, 0.5),  # Under-fitted (too smooth)
            'chi2_overfit': (2.0, float('inf')),  # Over-fitted (too rough)
            'min_resistivity': 1.0,  # Minimum physically reasonable resistivity (Ωm)
            'max_resistivity': 10000.0,  # Maximum physically reasonable resistivity (Ωm)
            'max_gradient': 100.0,  # Maximum acceptable resistivity gradient
            'convergence_ratio': 0.9  # Ratio of chi2 improvement in last 3 iterations
        }
        
        # Parameter adjustment strategy
        self.adjustment_factors = {
            'underfit': 0.5,  # Reduce lambda by 50% if underfit
            'overfit': 2.0,   # Increase lambda by 100% if overfit
            'minor_adjust': 1.2  # Fine-tune by 20%
        }
        
        self.max_iterations = 5  # Maximum number of re-inversion attempts
        self.history = []  # Track evaluation history
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate inversion results and adjust parameters if needed.
        
        Args:
            input_data: Dictionary containing:
                - inversion_results: Results from ERTInversionAgent
                - ert_data: Original ERT data
                - inversion_params: Current inversion parameters
                - time_lapse_data: List of ERT datasets (for time-lapse)
                - inversion_mode: 'standard' or 'time-lapse'
                - auto_adjust: Whether to automatically adjust and re-run (default: True)
                - max_attempts: Maximum re-inversion attempts (default: 5)
                - custom_thresholds: Optional custom quality thresholds
                
        Returns:
            Dictionary containing:
                - status: 'success', 'needs_improvement', or 'error'
                - quality_score: Overall quality score (0-100)
                - quality_metrics: Detailed quality metrics
                - recommendations: List of improvement recommendations
                - adjusted_params: Adjusted parameters (if auto_adjust=True)
                - final_results: Best inversion results
                - evaluation_history: History of all attempts
        """
        self._log_execution("Starting inversion quality evaluation")
        
        try:
            # Extract input data
            inversion_results = input_data.get('inversion_results')
            original_params = input_data.get('inversion_params', {})
            auto_adjust = input_data.get('auto_adjust', True)
            max_attempts = input_data.get('max_attempts', self.max_iterations)
            custom_thresholds = input_data.get('custom_thresholds', {})
            
            # Update thresholds if custom ones provided
            if custom_thresholds:
                self.quality_thresholds.update(custom_thresholds)
            
            if not inversion_results or inversion_results.get('status') != 'success':
                return {
                    'status': 'error',
                    'error': 'Invalid or failed inversion results provided',
                    'quality_score': 0
                }
            
            # Initialize history
            self.history = []
            
            # Evaluate initial results
            evaluation = self._evaluate_quality(inversion_results, original_params)
            self.history.append(evaluation)
            
            self._log_execution(f"Initial quality score: {evaluation['quality_score']:.1f}/100")
            
            # If quality is acceptable or auto_adjust is disabled, return
            if evaluation['is_acceptable'] or not auto_adjust:
                return {
                    'status': 'success' if evaluation['is_acceptable'] else 'needs_improvement',
                    'quality_score': evaluation['quality_score'],
                    'quality_metrics': evaluation['metrics'],
                    'recommendations': evaluation['recommendations'],
                    'final_results': inversion_results,
                    'evaluation_history': self.history,
                    'attempts': 1
                }
            
            # Attempt to improve through parameter adjustment
            best_results = inversion_results
            best_score = evaluation['quality_score']
            current_params = original_params.copy()
            
            for attempt in range(1, max_attempts):
                self._log_execution(f"Attempt {attempt + 1}/{max_attempts}: Adjusting parameters")
                
                # Adjust parameters based on evaluation
                adjusted_params = self._adjust_parameters(
                    current_params, 
                    evaluation['metrics'],
                    evaluation['recommendations']
                )
                
                self._log_execution(
                    f"Adjusted lambda: {current_params.get('lambda', 20)} -> "
                    f"{adjusted_params.get('lambda', 20)}"
                )
                
                # Re-run inversion with adjusted parameters
                new_results = self._rerun_inversion(input_data, adjusted_params)
                
                if new_results.get('status') != 'success':
                    self._log_execution(f"Re-inversion failed: {new_results.get('error')}")
                    break
                
                # Evaluate new results
                new_evaluation = self._evaluate_quality(new_results, adjusted_params)
                self.history.append(new_evaluation)
                
                self._log_execution(f"New quality score: {new_evaluation['quality_score']:.1f}/100")
                
                # Update best results if improved
                if new_evaluation['quality_score'] > best_score:
                    best_results = new_results
                    best_score = new_evaluation['quality_score']
                    self._log_execution(f"[OK] Improvement found! Score: {best_score:.1f}/100")
                
                # Check if acceptable quality achieved
                if new_evaluation['is_acceptable']:
                    self._log_execution(f"[OK] Acceptable quality achieved after {attempt + 1} attempts")
                    break
                
                # Update current params for next iteration
                current_params = adjusted_params
                evaluation = new_evaluation
                
                # Check if we're making progress
                if len(self.history) >= 3:
                    recent_scores = [h['quality_score'] for h in self.history[-3:]]
                    if max(recent_scores) - min(recent_scores) < 2.0:
                        self._log_execution("Converged: No significant improvement in last 3 attempts")
                        break
            
            # Get LLM interpretation if available
            interpretation = None
            if self.api_key:
                interpretation = self._generate_interpretation(best_results, self.history)
            
            return {
                'status': 'success' if best_score >= 70 else 'needs_improvement',
                'quality_score': best_score,
                'quality_metrics': self.history[-1]['metrics'],
                'recommendations': self.history[-1]['recommendations'],
                'adjusted_params': current_params,
                'final_results': best_results,
                'evaluation_history': self.history,
                'attempts': len(self.history),
                'interpretation': interpretation
            }
            
        except Exception as e:
            self._log_execution(f"Error in evaluation: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'quality_score': 0
            }
    
    def _evaluate_quality(self, results: Dict[str, Any], 
                         params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive quality evaluation of inversion results.
        
        Returns:
            Dictionary containing quality metrics and overall assessment
        """
        metrics = {}
        scores = {}
        
        # 1. Data Fit Quality (Chi-squared)
        chi2_score, chi2_metrics = self._evaluate_data_fit(results)
        metrics['data_fit'] = chi2_metrics
        scores['data_fit'] = chi2_score
        
        # 2. Model Smoothness/Roughness
        smoothness_score, smoothness_metrics = self._evaluate_smoothness(results)
        metrics['smoothness'] = smoothness_metrics
        scores['smoothness'] = smoothness_score
        
        # 3. Physical Plausibility
        physics_score, physics_metrics = self._evaluate_physics(results)
        metrics['physical_plausibility'] = physics_metrics
        scores['physical_plausibility'] = physics_score
        
        # 4. Convergence Quality
        convergence_score, convergence_metrics = self._evaluate_convergence(results)
        metrics['convergence'] = convergence_metrics
        scores['convergence'] = convergence_score
        
        # Calculate overall quality score (weighted average)
        weights = {
            'data_fit': 0.40,
            'smoothness': 0.25,
            'physical_plausibility': 0.25,
            'convergence': 0.10
        }
        
        overall_score = sum(scores[key] * weights[key] for key in weights.keys())
        
        # Determine if results are acceptable
        is_acceptable = (
            overall_score >= 70 and
            scores['data_fit'] >= 60 and
            scores['physical_plausibility'] >= 70
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, scores)
        
        return {
            'quality_score': overall_score,
            'component_scores': scores,
            'metrics': metrics,
            'is_acceptable': is_acceptable,
            'recommendations': recommendations,
            'parameters': params
        }
    
    def _evaluate_data_fit(self, results: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Evaluate how well the model fits the observed data."""
        chi2_values = results.get('chi2_values', [])
        
        if not chi2_values:
            # Try to get from time_lapse_result
            tl_result = results.get('time_lapse_result')
            if tl_result and hasattr(tl_result, 'all_chi2'):
                chi2_values = tl_result.all_chi2
        
        if not chi2_values or len(chi2_values) == 0:
            return 50.0, {'status': 'unknown', 'chi2': None}
        
        # Get final chi2 value(s)
        if isinstance(chi2_values[0], list):
            # Multiple inversions (time-lapse)
            final_chi2_list = [chi2[-1] if chi2 else None for chi2 in chi2_values]
            final_chi2 = np.mean([c for c in final_chi2_list if c is not None])
        else:
            final_chi2 = chi2_values[-1] if len(chi2_values) > 0 else None
        
        if final_chi2 is None:
            return 50.0, {'status': 'unknown', 'chi2': None}
        
        # Score based on chi-squared value
        target_chi2 = self.quality_thresholds['chi2_target']
        acceptable_range = self.quality_thresholds['chi2_acceptable_range']
        
        if acceptable_range[0] <= final_chi2 <= acceptable_range[1]:
            # Within acceptable range
            distance = abs(final_chi2 - target_chi2)
            score = 100 - (distance * 20)  # Penalty for deviation from target
        elif final_chi2 < acceptable_range[0]:
            # Underfit (too smooth)
            score = 40 + (final_chi2 / acceptable_range[0]) * 20
        else:
            # Overfit (too rough)
            score = max(0, 60 - (final_chi2 - acceptable_range[1]) * 10)
        
        metrics = {
            'final_chi2': float(final_chi2),
            'target_chi2': target_chi2,
            'acceptable_range': acceptable_range,
            'status': 'good' if acceptable_range[0] <= final_chi2 <= acceptable_range[1] else 
                     ('underfit' if final_chi2 < acceptable_range[0] else 'overfit')
        }
        
        return float(np.clip(score, 0, 100)), metrics
    
    def _evaluate_smoothness(self, results: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Evaluate model smoothness and roughness."""
        # Get final model
        final_model = self._extract_final_model(results)
        
        if final_model is None or len(final_model) == 0:
            return 50.0, {'status': 'unknown'}
        
        # Calculate model gradient statistics
        gradients = np.abs(np.diff(final_model))
        mean_gradient = np.mean(gradients)
        max_gradient = np.max(gradients)
        std_gradient = np.std(gradients)
        
        # Score based on gradient statistics
        max_acceptable_gradient = self.quality_thresholds['max_gradient']
        
        if max_gradient > max_acceptable_gradient:
            score = max(0, 100 - (max_gradient - max_acceptable_gradient))
        else:
            # Good smoothness, but penalize if too smooth (no features)
            score = 100 - (std_gradient / mean_gradient if mean_gradient > 0 else 0) * 5
        
        metrics = {
            'mean_gradient': float(mean_gradient),
            'max_gradient': float(max_gradient),
            'std_gradient': float(std_gradient),
            'smoothness_index': float(std_gradient / mean_gradient if mean_gradient > 0 else 0),
            'status': 'good' if max_gradient <= max_acceptable_gradient else 'too_rough'
        }
        
        return float(np.clip(score, 0, 100)), metrics
    
    def _evaluate_physics(self, results: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Evaluate physical plausibility of resistivity values."""
        final_model = self._extract_final_model(results)
        
        if final_model is None or len(final_model) == 0:
            return 50.0, {'status': 'unknown'}
        
        min_res = float(np.min(final_model))
        max_res = float(np.max(final_model))
        mean_res = float(np.mean(final_model))
        
        # Check if values are within physically reasonable bounds
        min_acceptable = self.quality_thresholds['min_resistivity']
        max_acceptable = self.quality_thresholds['max_resistivity']
        
        violations = np.sum((final_model < min_acceptable) | (final_model > max_acceptable))
        violation_ratio = violations / len(final_model)
        
        # Score based on physical reasonableness
        if violation_ratio == 0:
            score = 100
        else:
            score = max(0, 100 - violation_ratio * 200)
        
        # Additional check for extreme ranges
        resistivity_range = max_res / min_res if min_res > 0 else float('inf')
        if resistivity_range > 1000:  # More than 3 orders of magnitude
            score *= 0.8  # Penalize extreme ranges
        
        metrics = {
            'min_resistivity': min_res,
            'max_resistivity': max_res,
            'mean_resistivity': mean_res,
            'resistivity_range': float(resistivity_range),
            'violations': int(violations),
            'violation_ratio': float(violation_ratio),
            'status': 'good' if violation_ratio < 0.01 else 'has_violations'
        }
        
        return float(np.clip(score, 0, 100)), metrics
    
    def _evaluate_convergence(self, results: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Evaluate convergence quality of the inversion."""
        chi2_values = results.get('chi2_values', [])
        
        if not chi2_values:
            tl_result = results.get('time_lapse_result')
            if tl_result and hasattr(tl_result, 'all_chi2'):
                chi2_values = tl_result.all_chi2
        
        if not chi2_values or len(chi2_values) == 0:
            return 50.0, {'status': 'unknown'}
        
        # For time-lapse, take first dataset's chi2 history
        if isinstance(chi2_values[0], list):
            chi2_history = chi2_values[0]
        else:
            chi2_history = chi2_values
        
        if len(chi2_history) < 3:
            return 60.0, {'status': 'insufficient_iterations'}
        
        # Check convergence in last iterations
        last_improvements = []
        for i in range(len(chi2_history) - 3, len(chi2_history) - 1):
            if chi2_history[i] > 0:
                improvement = (chi2_history[i] - chi2_history[i + 1]) / chi2_history[i]
                last_improvements.append(improvement)
        
        avg_improvement = np.mean(last_improvements) if last_improvements else 0
        
        # Score based on convergence
        target_ratio = self.quality_thresholds['convergence_ratio']
        
        if avg_improvement < 0.001:  # Converged well
            score = 100
        elif avg_improvement < 0.01:  # Good convergence
            score = 90
        elif avg_improvement < 0.05:  # Acceptable
            score = 70
        else:  # Still improving significantly
            score = 50
        
        metrics = {
            'total_iterations': len(chi2_history),
            'final_chi2': float(chi2_history[-1]),
            'initial_chi2': float(chi2_history[0]),
            'improvement_ratio': float((chi2_history[0] - chi2_history[-1]) / chi2_history[0]) if chi2_history[0] > 0 else 0,
            'last_iteration_improvement': float(avg_improvement),
            'status': 'converged' if avg_improvement < 0.01 else 'still_improving'
        }
        
        return float(score), metrics
    
    # Coverage evaluation removed - not needed for quality assessment
    # def _evaluate_coverage(self, results: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    #     """Evaluate model coverage (sensitivity)."""
    #     # Method commented out - coverage not included in quality metrics
    
    def _generate_recommendations(self, metrics: Dict[str, Any], 
                                 scores: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on evaluation."""
        recommendations = []
        
        # Data fit recommendations
        if scores['data_fit'] < 60:
            chi2_status = metrics['data_fit'].get('status', 'unknown')
            if chi2_status == 'underfit':
                recommendations.append(
                    "Model is underfit (chi² too low). Reduce regularization parameter (lambda) "
                    "to allow more model complexity."
                )
            elif chi2_status == 'overfit':
                recommendations.append(
                    "Model is overfit (chi² too high). Increase regularization parameter (lambda) "
                    "to smooth the model."
                )
        
        # Smoothness recommendations
        if scores['smoothness'] < 60:
            recommendations.append(
                "Model shows excessive roughness. Consider increasing spatial regularization "
                "or adjusting mesh quality."
            )
        
        # Physical plausibility recommendations
        if scores['physical_plausibility'] < 70:
            violations = metrics['physical_plausibility'].get('violation_ratio', 0)
            if violations > 0:
                recommendations.append(
                    f"Model contains {violations*100:.1f}% non-physical resistivity values. "
                    "Consider setting model constraints or adjusting regularization."
                )
        
        # Convergence recommendations
        if scores['convergence'] < 60:
            recommendations.append(
                "Inversion has not fully converged. Increase maximum iterations or "
                "adjust convergence criteria."
            )
        
        if not recommendations:
            recommendations.append("Results meet quality criteria. No adjustments needed.")
        
        return recommendations
    
    def _adjust_parameters(self, current_params: Dict[str, Any],
                          metrics: Dict[str, Any],
                          recommendations: List[str]) -> Dict[str, Any]:
        """
        Automatically adjust inversion parameters based on evaluation.
        
        Returns:
            Dictionary of adjusted parameters
        """
        adjusted = current_params.copy()
        
        # Get current lambda
        current_lambda = adjusted.get('lambda', 20.0)
        
        # Adjust based on data fit
        chi2_status = metrics['data_fit'].get('status', 'unknown')
        
        if chi2_status == 'underfit':
            # Model is too smooth - reduce lambda
            new_lambda = current_lambda * self.adjustment_factors['underfit']
            self._log_execution(f"Detected underfit: reducing lambda to {new_lambda:.2f}")
        elif chi2_status == 'overfit':
            # Model is too rough - increase lambda
            new_lambda = current_lambda * self.adjustment_factors['overfit']
            self._log_execution(f"Detected overfit: increasing lambda to {new_lambda:.2f}")
        else:
            # Fine-tune
            chi2_value = metrics['data_fit'].get('final_chi2', 1.0)
            target = self.quality_thresholds['chi2_target']
            
            if chi2_value < target:
                new_lambda = current_lambda * 0.8  # Slightly reduce
            else:
                new_lambda = current_lambda * 1.2  # Slightly increase
            
            self._log_execution(f"Fine-tuning: adjusting lambda to {new_lambda:.2f}")
        
        adjusted['lambda'] = new_lambda
        
        # Adjust iterations if convergence is poor
        if metrics['convergence'].get('status') == 'still_improving':
            current_iter = adjusted.get('max_iterations', 10)
            adjusted['max_iterations'] = min(current_iter + 5, 30)
            self._log_execution(f"Increasing max iterations to {adjusted['max_iterations']}")
        
        return adjusted
    
    def _rerun_inversion(self, original_input: Dict[str, Any],
                        adjusted_params: Dict[str, Any]) -> Dict[str, Any]:
        """Re-run inversion with adjusted parameters."""
        from .ert_inversion_agent import ERTInversionAgent

        # Create new inversion agent
        inversion_agent = ERTInversionAgent(
            api_key=self.api_key,
            model=self.model,
            llm_provider=self.llm_provider
        )
        
        # Prepare input with adjusted parameters
        reinversion_input = original_input.copy()
        reinversion_input['inversion_params'] = adjusted_params
        
        # Remove evaluation-specific keys
        for key in ['inversion_results', 'auto_adjust', 'max_attempts', 'custom_thresholds']:
            reinversion_input.pop(key, None)
        
        # Run inversion
        return inversion_agent.execute(reinversion_input)
    
    def _extract_final_model(self, results: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract final model from results."""
        # Try different result formats
        if 'baseline_model' in results:
            return np.array(results['baseline_model'])
        
        if 'final_model' in results:
            return np.array(results['final_model'])
        
        if 'final_models' in results:
            models = results['final_models']
            if isinstance(models, np.ndarray):
                return models[:, 0] if models.ndim > 1 else models
        
        # Try time_lapse_result
        tl_result = results.get('time_lapse_result')
        if tl_result:
            if hasattr(tl_result, 'final_models'):
                models = tl_result.final_models
                return models[:, 0] if models.ndim > 1 else models
            if hasattr(tl_result, 'final_model'):
                return tl_result.final_model
        
        return None
    
    def _generate_interpretation(self, results: Dict[str, Any],
                                history: List[Dict[str, Any]]) -> str:
        """Generate LLM-powered interpretation of evaluation results."""
        if not self.api_key:
            return None
        
        # Prepare summary
        summary = f"""
Inversion Quality Evaluation Summary:
- Total attempts: {len(history)}
- Final quality score: {history[-1]['quality_score']:.1f}/100
- Component scores:
  * Data fit: {history[-1]['component_scores']['data_fit']:.1f}/100
  * Smoothness: {history[-1]['component_scores']['smoothness']:.1f}/100
  * Physical plausibility: {history[-1]['component_scores']['physical_plausibility']:.1f}/100
  * Convergence: {history[-1]['component_scores']['convergence']:.1f}/100
  * Coverage: {history[-1]['component_scores']['coverage']:.1f}/100

Key metrics:
- Final chi²: {history[-1]['metrics']['data_fit'].get('final_chi2', 'N/A')}
- Resistivity range: {history[-1]['metrics']['physical_plausibility'].get('min_resistivity', 'N/A'):.1f} - {history[-1]['metrics']['physical_plausibility'].get('max_resistivity', 'N/A'):.1f} Ωm

Recommendations:
{chr(10).join('- ' + r for r in history[-1]['recommendations'])}
"""
        
        prompt = f"""Based on this ERT inversion quality evaluation, provide a brief 
interpretation (2-3 sentences) of the results and whether they are suitable for 
hydrogeophysical interpretation:

{summary}"""
        
        try:
            return self.query_llm(prompt, max_tokens=200)
        except Exception as e:
            self._log_execution(f"Failed to generate interpretation: {e}")
            return None
    
    def _log_execution(self, message: str):
        """Log execution messages."""
        prefix = f"[{self.name}] "
        try:
            print(f"{prefix}{message}")
        except UnicodeEncodeError:
            # Keep logging robust on Windows terminals with non-UTF-8 code pages.
            encoding = getattr(sys.stdout, "encoding", None) or "ascii"
            safe_message = message.encode(encoding, errors="replace").decode(encoding, errors="replace")
            print(f"{prefix}{safe_message}")
