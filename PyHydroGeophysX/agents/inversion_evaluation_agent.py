"""
Inversion Evaluation Agent

Specialized agent for evaluating ERT inversion quality and automatically
adjusting regularization parameters to achieve optimal results.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from ._chi2 import chi2_history

import numpy as np

from .base_agent import BaseAgent


def _or_default(value: Any, default: Any) -> Any:
    """``value`` unless it is absent, in which case ``default``.

    What ``dict.get(key, default)`` is usually assumed to do, and does not: a
    key that is present and ``None`` returns ``None``, not the default. A
    workflow configuration parsed from a plain-language request is full of those
    - the model writes ``"quality_threshold": null`` for a setting the user did
    not mention - and ``float(None)`` then failed the whole quality check on a
    run whose inversion was perfectly good.

    Examples
    --------
    >>> _or_default(None, 70)
    70
    >>> _or_default(0, 70)
    0
    >>> _or_default('', 70)
    ''
    """
    return default if value is None else value


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
            'chi2_poor': (2.0, float('inf')),  # Data not fitted within their errors
            'chi2_overfit': (0.0, 0.5),  # Possible overfit or overestimated errors
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
        
        self.max_iterations = 3  # Total evaluations, including the initial result.
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
                - max_attempts: Maximum total evaluations, including the initial
                  result (default: 3, allowing at most two re-inversions)
                - quality_threshold: Overall quality threshold (default: 70)
                - progress_callback: Optional callback for transparent loop logs
                - custom_thresholds: Optional custom quality thresholds
                
        Returns:
            Dictionary containing:

                - status: 'success', 'needs_review', or 'failed'
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
            original_params = input_data.get('inversion_params') or {}
            auto_adjust = input_data.get('auto_adjust', True)
            # `dict.get(key, default)` does not apply the default to a key that
            # is present and None, and a configuration parsed from a request
            # carries exactly that: an LLM asked for a workflow config writes
            # `"quality_threshold": null` for a setting the user did not mention.
            # `float(None)` then failed the whole quality check on a run whose
            # inversion was fine.
            max_attempts = int(_or_default(input_data.get('max_attempts'),
                                           self.max_iterations))
            if max_attempts < 1:
                raise ValueError('max_attempts must include at least the initial evaluation.')
            quality_threshold = float(_or_default(input_data.get('quality_threshold'), 70))
            progress_callback = input_data.get('progress_callback')
            custom_thresholds = input_data.get('custom_thresholds', {})
            transparent_log = []
            
            # Update thresholds if custom ones provided
            if custom_thresholds:
                self.quality_thresholds.update(custom_thresholds)
            
            if not inversion_results or inversion_results.get('status') != 'success':
                return {
                    'status': 'failed',
                    'error': 'Invalid or failed inversion results provided',
                    'quality_score': 0,
                    'error_fix_hint': (
                        'Run inversion successfully before quality evaluation. See: '
                        'https://geohang.github.io/PyHydroGeophysX/agents/troubleshooting.html#quality-loop-never-reaches-threshold'
                    )
                }
            
            # Initialize history
            self.history = []
            
            # Evaluate initial results
            evaluation = self._evaluate_quality(inversion_results, original_params, quality_threshold)
            self.history.append(evaluation)
            
            self._log_execution(f"Initial quality score: {evaluation['quality_score']:.1f}/100")
            first_line = self._format_attempt_log(
                attempt=1,
                max_attempts=max_attempts,
                evaluation=evaluation,
                current_params=original_params,
                adjusted_params=None,
            )
            transparent_log.append(first_line)
            self._emit_progress(progress_callback, first_line)
            
            # If quality is acceptable or auto_adjust is disabled, return
            if evaluation['is_acceptable'] or not auto_adjust:
                return {
                    'status': 'success' if evaluation['is_acceptable'] else 'needs_review',
                    'summary': (
                        'Inversion quality reached the configured threshold.'
                        if evaluation['is_acceptable']
                        else 'Inversion quality needs review before accepting results.'
                    ),
                    'quality_score': evaluation['quality_score'],
                    'quality_metrics': evaluation['metrics'],
                    'recommendations': evaluation['recommendations'],
                    'final_results': inversion_results,
                    'evaluation_history': self.history,
                    'attempts': 1,
                    'quality_threshold': quality_threshold,
                    'transparent_log': transparent_log,
                }
            
            # Attempt to improve through parameter adjustment
            best_results = inversion_results
            best_score = evaluation['quality_score']
            best_evaluation = evaluation
            best_params = original_params.copy()
            current_params = original_params.copy()
            
            for attempt in range(1, max_attempts):
                # Adjust parameters based on evaluation
                adjusted_params = self._adjust_parameters(
                    current_params, 
                    evaluation['metrics'],
                    evaluation['recommendations']
                )
                
                attempt_line = self._format_attempt_log(
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                    evaluation=evaluation,
                    current_params=current_params,
                    adjusted_params=adjusted_params,
                )
                transparent_log.append(attempt_line)
                self._log_execution(attempt_line)
                self._emit_progress(progress_callback, attempt_line)
                
                # Re-run inversion with adjusted parameters
                new_results = self._rerun_inversion(input_data, adjusted_params)
                
                if new_results.get('status') != 'success':
                    self._log_execution(f"Re-inversion failed: {new_results.get('error')}")
                    break
                
                # Evaluate new results
                new_evaluation = self._evaluate_quality(new_results, adjusted_params, quality_threshold)
                self.history.append(new_evaluation)
                
                self._log_execution(f"New quality score: {new_evaluation['quality_score']:.1f}/100")
                
                # Update best results if improved
                if new_evaluation['quality_score'] > best_score:
                    best_results = new_results
                    best_score = new_evaluation['quality_score']
                    best_evaluation = new_evaluation
                    best_params = adjusted_params.copy()
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
                interpretation = self._generate_interpretation(best_results, self.history, best_evaluation)
            
            return {
                'status': 'success' if best_evaluation['is_acceptable'] else 'needs_review',
                'summary': (
                    'Inversion quality reached the configured threshold.'
                    if best_evaluation['is_acceptable']
                    else (
                        f'Quality optimization stopped after {len(self.history)} attempts '
                        f'with score {best_score:.1f}; one or more quality criteria were not met.'
                    )
                ),
                'quality_score': best_score,
                'quality_metrics': best_evaluation['metrics'],
                'recommendations': best_evaluation['recommendations'],
                'adjusted_params': best_params,
                'final_results': best_results,
                'evaluation_history': self.history,
                'attempts': len(self.history),
                'interpretation': interpretation,
                'quality_threshold': quality_threshold,
                'transparent_log': transparent_log,
            }
            
        except Exception as e:
            self._log_execution(f"Error in evaluation: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'quality_score': 0,
                'error_fix_hint': (
                    'Check inversion result fields, data fit statistics, and inversion_params. See: '
                    'https://geohang.github.io/PyHydroGeophysX/agents/troubleshooting.html#quality-loop-never-reaches-threshold'
                )
            }

    def _format_attempt_log(
        self,
        attempt: int,
        max_attempts: int,
        evaluation: Dict[str, Any],
        current_params: Dict[str, Any],
        adjusted_params: Optional[Dict[str, Any]],
    ) -> str:
        """Format one transparent quality-loop status line.

        Parameters
        ----------
        attempt : int
            Current attempt number.
        max_attempts : int
            Maximum configured attempts.
        evaluation : dict
            Evaluation dictionary from ``_evaluate_quality``.
        current_params : dict
            Current inversion parameters.
        adjusted_params : dict, optional
            Adjusted parameters proposed for the next inversion.

        Returns
        -------
        str
            Plain-English status line for logs and UI progress.

        Raises
        ------
        None

        Examples
        --------
        >>> agent = InversionEvaluationAgent()
        >>> agent._format_attempt_log(1, 3, {"quality_score": 60, "metrics": {}}, {}, None).startswith("Attempt")
        True
        """
        metrics = evaluation.get("metrics", {})
        chi2 = metrics.get("data_fit", {}).get("final_chi2", "N/A")
        quality = evaluation.get("quality_score", 0.0)
        old_lambda = current_params.get("lambda", current_params.get("lam", 20))
        if adjusted_params:
            new_lambda = adjusted_params.get("lambda", adjusted_params.get("lam", old_lambda))
            change = f" Adjusting lambda {old_lambda} -> {new_lambda}."
        else:
            change = ""
        chi2_text = f"{chi2:.3f}" if isinstance(chi2, (int, float)) else str(chi2)
        return (
            f"Attempt {attempt}/{max_attempts}: chi2 = {chi2_text}, "
            f"quality = {quality:.1f}."
            f"{change}"
        )

    def _emit_progress(self, progress_callback: Any, message: str) -> None:
        """Send a progress message to a callback if one was provided."""
        if not progress_callback:
            return
        try:
            progress_callback("Evaluating inversion quality", 0.0, message)
        except TypeError:
            progress_callback(message)
    
    def _evaluate_quality(self, results: Dict[str, Any], 
                         params: Dict[str, Any],
                         quality_threshold: float = 70) -> Dict[str, Any]:
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
            overall_score >= quality_threshold and
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
        chi2_values = self._chi2_history(results)
        if results.get('chi2') is not None:
            chi2_values = [results['chi2']]
        
        if not chi2_values:
            # Try to get from time_lapse_result
            tl_result = results.get('time_lapse_result')
            if tl_result and hasattr(tl_result, 'all_chi2'):
                chi2_values = tl_result.all_chi2
        
        if not chi2_values or len(chi2_values) == 0:
            return 50.0, {'status': 'unknown', 'chi2': None}
        
        # The last iteration's chi-squared. A time-lapse row is
        # [chi2, phi_m, phi_t], so the misfit is column 0, not the last entry -
        # reading row[-1] scored the temporal regularization term instead.
        history = chi2_history(chi2_values)
        final_chi2 = history[-1] if history else None
        
        if final_chi2 is None or not np.isfinite(final_chi2) or final_chi2 < 0:
            return 50.0, {'status': 'unknown', 'chi2': None}
        
        # Score based on chi-squared value
        target_chi2 = self.quality_thresholds['chi2_target']
        acceptable_range = self.quality_thresholds['chi2_acceptable_range']
        
        if acceptable_range[0] <= final_chi2 <= acceptable_range[1]:
            # Within acceptable range
            distance = abs(final_chi2 - target_chi2)
            score = 100 - (distance * 20)  # Penalty for deviation from target
        elif final_chi2 < acceptable_range[0]:
            # Low misfit can indicate overfitting or overestimated errors.
            score = 40 + (final_chi2 / acceptable_range[0]) * 20
        else:
            # High misfit: data are not explained within their assigned errors.
            score = max(0, 60 - (final_chi2 - acceptable_range[1]) * 10)
        
        metrics = {
            'final_chi2': float(final_chi2),
            'target_chi2': target_chi2,
            'acceptable_range': acceptable_range,
            'status': 'good' if acceptable_range[0] <= final_chi2 <= acceptable_range[1] else 
                     ('overfit' if final_chi2 < acceptable_range[0] else 'underfit')
        }
        
        return float(np.clip(score, 0, 100)), metrics
    
    def _evaluate_smoothness(self, results: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Evaluate model smoothness and roughness."""
        # Get final model
        final_model = self._extract_final_model(results)
        
        if final_model is None or len(final_model) < 2:
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
        
        # Additional check for extreme contrasts. This is max/min, a ratio - it
        # was reported under the key 'resistivity_range', where 94.5 read as a
        # span of 94.5 ohm-m for a model spanning 105.8 to 10000.
        resistivity_ratio = max_res / min_res if min_res > 0 else float('inf')
        if resistivity_ratio > 1000:  # More than 3 orders of magnitude
            score *= 0.8  # Penalize extreme contrasts

        metrics = {
            'min_resistivity': min_res,
            'max_resistivity': max_res,
            'mean_resistivity': mean_res,
            'resistivity_span': float(max_res - min_res),
            'resistivity_ratio': float(resistivity_ratio),
            'violations': int(violations),
            'violation_ratio': float(violation_ratio),
            'status': 'good' if violation_ratio < 0.01 else 'has_violations'
        }
        
        return float(np.clip(score, 0, 100)), metrics
    
    def _evaluate_convergence(self, results: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Evaluate convergence quality of the inversion."""
        chi2_values = self._chi2_history(results)
        
        if not chi2_values:
            tl_result = results.get('time_lapse_result')
            if tl_result and hasattr(tl_result, 'all_chi2'):
                chi2_values = tl_result.all_chi2
        
        if not chi2_values or len(chi2_values) == 0:
            return 50.0, {'status': 'unknown'}
        
        # The chi-squared trajectory across iterations. The first entry of a
        # time-lapse result is one iteration's three objective terms, not a
        # history, so reading chi2_values[0] measured convergence against
        # [chi2, phi_m, phi_t] of iteration zero.
        history = chi2_history(chi2_values)
        
        if len(history) < 3:
            return 60.0, {'status': 'insufficient_iterations'}
        
        # Check convergence in last iterations
        last_improvements = []
        for i in range(len(history) - 3, len(history) - 1):
            if history[i] > 0:
                improvement = (history[i] - history[i + 1]) / history[i]
                last_improvements.append(improvement)
        
        avg_improvement = np.mean(last_improvements) if last_improvements else 0

        # The solver records why its loop ended. Preferring that to a slope
        # threshold stops the report saying "still_improving" about a run whose
        # own log says "Convergence reached at iteration 8" - two verdicts on
        # one question, only the weaker of which reached the reader.
        stop_reason = None
        tl_result = results.get('time_lapse_result')
        if tl_result is not None:
            stop_reason = (getattr(tl_result, 'meta', {}) or {}).get('stop_reason')
        stop_reason = stop_reason or results.get('stop_reason')
        
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
            'total_iterations': len(history),
            'final_chi2': float(history[-1]),
            'initial_chi2': float(history[0]),
            'improvement_ratio': float((history[0] - history[-1]) / history[0]) if history[0] > 0 else 0,
            'last_iteration_improvement': float(avg_improvement),
            'solver_stop_reason': stop_reason,
            # 'target' and 'plateau' are the two branches that print
            # "Convergence reached"; 'iteration_cap' is the loop running out.
            'status': ('converged'
                       if stop_reason in ('target', 'plateau')
                       else 'still_improving' if stop_reason == 'iteration_cap'
                       else 'converged' if avg_improvement < 0.01
                       else 'still_improving')
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
                    "Data misfit is high (chi² above the target). Check errors, geometry and convergence; "
                    "reducing regularization (lambda) may improve data fit."
                )
            elif chi2_status == 'overfit':
                recommendations.append(
                    "Data misfit is low (chi² below the target). Check for overestimated errors or "
                    "overfitting; increasing regularization (lambda) may be appropriate."
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
                new_lambda = current_lambda * 1.2
            else:
                new_lambda = current_lambda * 0.8
            
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
        if original_input.get('output_dir'):
            from pathlib import Path
            reinversion_input['output_dir'] = str(Path(original_input['output_dir']) / f'attempt_{len(self.history) + 1}')
        reinversion_input['inversion_params'] = adjusted_params
        
        # Remove evaluation-specific keys
        for key in ['inversion_results', 'auto_adjust', 'max_attempts', 'custom_thresholds']:
            reinversion_input.pop(key, None)
        
        # Run inversion
        return inversion_agent.execute(reinversion_input)
    
    def _extract_final_model(self, results: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract final model from results."""
        # Try different result formats
        if results.get('resistivity_model') is not None:
            return np.asarray(results['resistivity_model'])
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
    
    @staticmethod
    def _chi2_history(results):
        values = results.get('chi2_values')
        if values is None or len(values) == 0:
            inv = results.get('inversion_result')
            values = getattr(inv, 'iteration_chi2', None)
        if values is None:
            values = getattr(results.get('time_lapse_result'), 'all_chi2', None)
        if values is None:
            return []
        return [np.asarray(value).tolist() for value in values]

    def _generate_interpretation(self, results: Dict[str, Any],
                                history: List[Dict[str, Any]], best=None) -> str:
        """Generate LLM-powered interpretation of evaluation results."""
        if not self.api_key:
            return None
        
        # Interpret the retained result, which need not be the last attempt.
        retained = best or history[-1]
        summary = f"""
Inversion Quality Evaluation Summary:
- Total attempts: {len(history)}
- Final quality score: {retained['quality_score']:.1f}/100
- Component scores:
  * Data fit: {retained['component_scores']['data_fit']:.1f}/100
  * Smoothness: {retained['component_scores']['smoothness']:.1f}/100
  * Physical plausibility: {retained['component_scores']['physical_plausibility']:.1f}/100
  * Convergence: {retained['component_scores']['convergence']:.1f}/100

Key metrics:
- Final chi²: {retained['metrics']['data_fit'].get('final_chi2', 'N/A')}
- Resistivity range: {retained['metrics']['physical_plausibility'].get('min_resistivity', 'N/A')} - {retained['metrics']['physical_plausibility'].get('max_resistivity', 'N/A')} Ωm

Recommendations:
{chr(10).join('- ' + r for r in retained['recommendations'])}
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
