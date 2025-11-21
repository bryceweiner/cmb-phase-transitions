"""
Parameter Optimization Module
==============================

Auto-tune detection parameters for different CMB datasets.

Classes:
    ParameterOptimizer: Grid search optimization for detection parameters

Optimizes window_size, prominence_factor, and min_distance based on
dataset characteristics (resolution, noise, multipole range).
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from scipy.signal import find_peaks

from .utils import OutputManager
from .phase_detector import PhaseTransitionDetector


class ParameterOptimizer:
    """
    Optimize detection parameters for CMB datasets.
    
    Performs grid search over parameter space to find optimal detection
    settings for a given dataset. Uses metrics like SNR, detection stability,
    and consistency to select best parameters.
    
    Attributes:
        output (OutputManager): For logging
        
    Example:
        >>> optimizer = ParameterOptimizer()
        >>> params = optimizer.optimize_for_dataset(ell, C_ell, C_ell_err, beam_fwhm=7.3)
        >>> print(f"Optimal window_size: {params['window_size']}")
    """
    
    def __init__(self, output: OutputManager = None):
        """
        Initialize ParameterOptimizer.
        
        Parameters:
            output (OutputManager, optional): For logging
        """
        self.output = output if output is not None else OutputManager()
    
    def optimize_for_dataset(self,
                            ell: np.ndarray,
                            C_ell: np.ndarray,
                            C_ell_err: np.ndarray,
                            beam_fwhm_arcmin: float,
                            reference_beam_fwhm: float = 1.4) -> Dict[str, Any]:
        """
        Optimize detection parameters for a specific dataset.
        
        Performs grid search over parameter combinations and selects
        those maximizing detection quality metrics.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Power spectrum
            C_ell_err (ndarray): Uncertainties
            beam_fwhm_arcmin (float): Dataset beam FWHM in arcminutes
            reference_beam_fwhm (float): Reference beam (ACT default: 1.4)
            
        Returns:
            dict: Optimal parameters including:
                - window_size: Optimal Savitzky-Golay window
                - prominence_factor: Optimal peak prominence
                - min_distance: Minimum separation between transitions
                - score: Quality score of optimal parameters
                - all_scores: Scores for all tested combinations
        """
        self.output.log_message("\n" + "="*60)
        self.output.log_message("PARAMETER OPTIMIZATION")
        self.output.log_message("="*60)
        self.output.log_message(f"Dataset beam FWHM: {beam_fwhm_arcmin:.1f} arcmin")
        self.output.log_message(f"Reference beam: {reference_beam_fwhm:.1f} arcmin")
        
        resolution_ratio = beam_fwhm_arcmin / reference_beam_fwhm
        self.output.log_message(f"Resolution ratio: {resolution_ratio:.2f}×")
        
        # Define parameter grid based on resolution
        window_sizes = self._get_window_size_grid(resolution_ratio)
        prominence_factors = [0.3, 0.5, 0.7, 1.0]
        min_distances = [400, 500, 600]
        
        self.output.log_message(f"\nGrid search parameters:")
        self.output.log_message(f"  Window sizes: {window_sizes}")
        self.output.log_message(f"  Prominence factors: {prominence_factors}")
        self.output.log_message(f"  Min distances: {min_distances}")
        
        # Grid search
        best_score = -np.inf
        best_params = None
        all_results = []
        
        total_combinations = len(window_sizes) * len(prominence_factors) * len(min_distances)
        self.output.log_message(f"\nTesting {total_combinations} parameter combinations...")
        
        for window_size in window_sizes:
            for prom_factor in prominence_factors:
                for min_dist in min_distances:
                    params = {
                        'window_size': window_size,
                        'prominence_factor': prom_factor,
                        'min_distance': min_dist
                    }
                    
                    score, metrics = self._evaluate_parameters(
                        ell, C_ell, C_ell_err, params
                    )
                    
                    result = {
                        'params': params.copy(),
                        'score': score,
                        'metrics': metrics
                    }
                    all_results.append(result)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
        
        # Report results
        self.output.log_message(f"\nOptimization complete!")
        self.output.log_message(f"Best parameters:")
        self.output.log_message(f"  Window size: {best_params['window_size']}")
        self.output.log_message(f"  Prominence factor: {best_params['prominence_factor']}")
        self.output.log_message(f"  Min distance: {best_params['min_distance']}")
        self.output.log_message(f"  Quality score: {best_score:.3f}")
        
        # Sort all results by score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Show top 3
        self.output.log_message(f"\nTop 3 parameter sets:")
        for i, result in enumerate(all_results[:3]):
            p = result['params']
            s = result['score']
            self.output.log_message(
                f"  {i+1}. window={p['window_size']}, prom={p['prominence_factor']}, "
                f"dist={p['min_distance']}: score={s:.3f}"
            )
        
        return {
            **best_params,
            'score': best_score,
            'all_scores': all_results,
            'resolution_ratio': resolution_ratio
        }
    
    def _get_window_size_grid(self, resolution_ratio: float) -> List[int]:
        """
        Generate window size grid based on resolution ratio.
        
        Larger beams (lower resolution) benefit from larger windows
        to reduce sensitivity to high-frequency noise.
        
        Parameters:
            resolution_ratio (float): Beam FWHM / reference beam FWHM
            
        Returns:
            list: Window sizes to test
        """
        if resolution_ratio < 2.0:
            # High resolution (similar to ACT)
            return [20, 25, 30, 35]
        elif resolution_ratio < 4.0:
            # Medium resolution
            return [25, 30, 35, 40, 45]
        else:
            # Low resolution (e.g., Planck)
            return [30, 35, 40, 45, 50]
    
    def _evaluate_parameters(self,
                            ell: np.ndarray,
                            C_ell: np.ndarray,
                            C_ell_err: np.ndarray,
                            params: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate quality of a parameter set.
        
        Metrics:
        - Detection stability (across bootstrap samples)
        - Signal-to-noise ratio of detected features
        - Number of detections (prefer 2-5 transitions)
        - Separation quality (avoid clustered detections)
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Power spectrum
            C_ell_err (ndarray): Uncertainties
            params (dict): Parameters to evaluate
            
        Returns:
            tuple: (score, metrics_dict)
                score: Overall quality score (higher is better)
                metrics: Individual metric values
        """
        # Create detector with these parameters
        detector = PhaseTransitionDetector(
            output=self.output,
            window_size=params['window_size']
        )
        
        # Compute derivative
        dC_dell = detector.compute_derivative(ell, C_ell)
        
        # Detect transitions
        try:
            peaks, transitions, props = detector.detect_transitions(
                ell, dC_dell,
                min_distance=params['min_distance'],
                prominence_factor=params['prominence_factor']
            )
        except Exception:
            # If detection fails, return very low score
            return -1000.0, {'error': 'detection_failed'}
        
        n_detected = len(transitions)
        
        # Metric 1: Number of detections (prefer 2-5)
        if n_detected == 0:
            n_score = -100.0
        elif 2 <= n_detected <= 5:
            n_score = 10.0
        elif n_detected == 1 or n_detected == 6:
            n_score = 5.0
        else:
            n_score = 0.0
        
        # Metric 2: SNR of detected peaks
        if n_detected > 0 and len(peaks) > 0:
            peak_values = np.abs(dC_dell[peaks])
            noise_estimate = np.std(dC_dell)
            snr = np.mean(peak_values / noise_estimate) if noise_estimate > 0 else 0
            snr_score = min(snr, 10.0)  # Cap at 10
        else:
            snr_score = 0.0
        
        # Metric 3: Separation quality (avoid clustering)
        if n_detected > 1:
            separations = np.diff(sorted(transitions))
            min_sep = np.min(separations)
            mean_sep = np.mean(separations)
            std_sep = np.std(separations)
            
            # Prefer well-separated features
            if min_sep > params['min_distance'] * 0.8:
                sep_score = 5.0
            elif min_sep > params['min_distance'] * 0.5:
                sep_score = 2.0
            else:
                sep_score = 0.0
            
            # Bonus for consistent spacing
            if std_sep < mean_sep * 0.5:
                sep_score += 2.0
        else:
            sep_score = 0.0
        
        # Metric 4: Bootstrap stability (quick test with 10 samples)
        stability_score = self._quick_bootstrap_stability(
            ell, C_ell, C_ell_err, params, n_samples=10
        )
        
        # Combine scores with weights
        total_score = (
            n_score * 1.0 +
            snr_score * 2.0 +
            sep_score * 1.5 +
            stability_score * 3.0  # Stability is most important
        )
        
        metrics = {
            'n_detected': n_detected,
            'n_score': n_score,
            'snr': snr_score,
            'separation': sep_score,
            'stability': stability_score,
            'total': total_score
        }
        
        return total_score, metrics
    
    def _quick_bootstrap_stability(self,
                                   ell: np.ndarray,
                                   C_ell: np.ndarray,
                                   C_ell_err: np.ndarray,
                                   params: Dict[str, Any],
                                   n_samples: int = 10) -> float:
        """
        Quick bootstrap stability test.
        
        Resample data and check if similar number of transitions detected.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Power spectrum
            C_ell_err (ndarray): Uncertainties
            params (dict): Detection parameters
            n_samples (int): Number of bootstrap samples
            
        Returns:
            float: Stability score (0-10)
        """
        detector = PhaseTransitionDetector(
            output=self.output,
            window_size=params['window_size']
        )
        
        # Original detection count
        dC_dell = detector.compute_derivative(ell, C_ell)
        try:
            _, transitions_orig, _ = detector.detect_transitions(
                ell, dC_dell,
                min_distance=params['min_distance'],
                prominence_factor=params['prominence_factor']
            )
            n_orig = len(transitions_orig)
        except Exception:
            return 0.0
        
        if n_orig == 0:
            return 0.0
        
        # Bootstrap samples
        n_consistent = 0
        for _ in range(n_samples):
            # Resample with noise
            C_ell_boot = C_ell + np.random.normal(0, C_ell_err)
            dC_boot = detector.compute_derivative(ell, C_ell_boot)
            
            try:
                _, transitions_boot, _ = detector.detect_transitions(
                    ell, dC_boot,
                    min_distance=params['min_distance'],
                    prominence_factor=params['prominence_factor']
                )
                n_boot = len(transitions_boot)
                
                # Count as consistent if within ±1 of original
                if abs(n_boot - n_orig) <= 1:
                    n_consistent += 1
            except Exception:
                continue
        
        # Stability score: fraction consistent × 10
        stability = (n_consistent / n_samples) * 10.0
        return stability
    
    def optimize_for_planck(self,
                           ell: np.ndarray,
                           C_ell: np.ndarray,
                           C_ell_err: np.ndarray) -> Dict[str, Any]:
        """
        Convenience method to optimize for Planck 2018 EE data.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Power spectrum
            C_ell_err (ndarray): Uncertainties
            
        Returns:
            dict: Optimal parameters for Planck
        """
        return self.optimize_for_dataset(
            ell, C_ell, C_ell_err,
            beam_fwhm_arcmin=7.3,  # Planck 143 GHz
            reference_beam_fwhm=1.4  # ACT f150
        )

