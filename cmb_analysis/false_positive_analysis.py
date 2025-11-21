"""
False Positive Rate Analysis Module
====================================

Empirical false discovery rate calculation from null simulations.

Runs detection pipeline on Monte Carlo null realizations to establish
the rate at which "fake transitions" are detected in data without true features.

Classes:
    FalsePositiveAnalyzer: Compute empirical FDR from null ensemble

Paper reference: Statistical validation methodology
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy.stats import percentileofscore

from .utils import OutputManager
from .phase_detector import PhaseTransitionDetector
from .monte_carlo_simulator import MonteCarloSimulator


class FalsePositiveAnalyzer:
    """
    Analyze false positive rates from null hypothesis testing.
    
    Applies detection pipeline to Monte Carlo null realizations and compares
    to observed detections to establish statistical significance.
    
    Attributes:
        output (OutputManager): For logging
        detector (PhaseTransitionDetector): Detection algorithm
        
    Example:
        >>> analyzer = FalsePositiveAnalyzer()
        >>> fdr_results = analyzer.analyze_ensemble(ensemble, observed_transitions)
    """
    
    def __init__(self, output: OutputManager = None, detector: PhaseTransitionDetector = None):
        """
        Initialize FalsePositiveAnalyzer.
        
        Parameters:
            output (OutputManager, optional): For logging
            detector (PhaseTransitionDetector, optional): Detection method
        """
        self.output = output if output is not None else OutputManager()
        self.detector = detector if detector is not None else PhaseTransitionDetector(output=self.output)
    
    def detect_in_realization(self, ell: np.ndarray, C_ell_mock: np.ndarray,
                             min_distance: int = 500,
                             prominence_factor: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply detection pipeline to single mock realization.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell_mock (ndarray): Mock power spectrum
            min_distance (int): Minimum separation between peaks
            prominence_factor (float): Detection threshold factor
            
        Returns:
            tuple: (detected_multipoles, peak_heights)
        """
        # Compute derivative
        dC_dell = self.detector.compute_derivative(ell, C_ell_mock)
        
        # Detect transitions
        peaks, peak_ells, peak_heights = self.detector.detect_transitions(
            ell, dC_dell, min_distance=min_distance, prominence_factor=prominence_factor
        )
        
        return peak_ells, peak_heights
    
    def analyze_ensemble(self, ensemble: Dict[str, Any],
                        observed_transitions: np.ndarray,
                        n_observed: int = 3,
                        max_realizations: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze false positive rates across ensemble.
        
        For each null realization:
        1. Apply detection pipeline
        2. Count "fake" detections
        3. Compare to observed features
        
        Parameters:
            ensemble (dict): Output from MonteCarloSimulator.generate_ensemble()
            observed_transitions (ndarray): Observed transition multipoles
            n_observed (int): Number of observed transitions (default: 3)
            max_realizations (int, optional): Limit analysis (for speed)
            
        Returns:
            dict: False positive analysis results
        """
        self.output.log_section_header("FALSE POSITIVE RATE ANALYSIS")
        self.output.log_message(f"Observed transitions: {observed_transitions}")
        self.output.log_message(f"Number observed: {n_observed}")
        self.output.log_message("")
        
        realizations = ensemble['realizations']
        ell = ensemble['ell']
        n_total = len(realizations)
        
        if max_realizations is not None and max_realizations < n_total:
            n_total = max_realizations
            self.output.log_message(f"Limiting analysis to {n_total} realizations")
        
        # Storage for results
        n_detected_list = []  # Number of transitions per realization
        all_detected_multipoles = []  # All detected multipoles
        all_peak_heights = []  # All peak heights
        
        # Detection counts at different thresholds
        n_ge_1 = 0  # At least 1 detection
        n_ge_2 = 0  # At least 2 detections
        n_ge_3 = 0  # At least 3 detections
        n_ge_observed = 0  # At least n_observed detections
        
        # Multipole location matches
        matches_by_tolerance = {50: 0, 100: 0, 200: 0}
        
        # Progress reporting
        checkpoints = [int(n_total * f) for f in [0.1, 0.25, 0.5, 0.75, 1.0]]
        
        self.output.log_message("Running detection on null realizations...")
        
        for i in range(n_total):
            C_ell_mock = realizations[i]
            
            # Detect in this realization
            detected_ells, peak_heights = self.detect_in_realization(ell, C_ell_mock)
            
            n_detected = len(detected_ells)
            n_detected_list.append(n_detected)
            
            if n_detected > 0:
                all_detected_multipoles.extend(detected_ells)
                all_peak_heights.extend(peak_heights)
            
            # Count by threshold
            if n_detected >= 1:
                n_ge_1 += 1
            if n_detected >= 2:
                n_ge_2 += 1
            if n_detected >= 3:
                n_ge_3 += 1
            if n_detected >= n_observed:
                n_ge_observed += 1
            
            # Check if any detected multipoles match observed (within tolerance)
            if n_detected >= n_observed:
                for tolerance in matches_by_tolerance.keys():
                    match_count = 0
                    for obs_ell in observed_transitions:
                        distances = np.abs(detected_ells - obs_ell)
                        if np.min(distances) < tolerance:
                            match_count += 1
                    
                    if match_count >= n_observed:
                        matches_by_tolerance[tolerance] += 1
            
            # Progress
            if (i+1) in checkpoints:
                progress = (i+1) / n_total * 100
                self.output.log_message(f"  {progress:.0f}% complete ({i+1}/{n_total})")
        
        self.output.log_message("")
        
        # Compute false positive rates
        fpr_ge_1 = n_ge_1 / n_total
        fpr_ge_2 = n_ge_2 / n_total
        fpr_ge_3 = n_ge_3 / n_total
        fpr_ge_observed = n_ge_observed / n_total
        
        # Empirical p-value
        empirical_pvalue = fpr_ge_observed
        
        # Convert to sigma (one-tailed test)
        from scipy.special import erfcinv
        if empirical_pvalue > 0:
            empirical_sigma = np.sqrt(2) * erfcinv(2 * empirical_pvalue)
        else:
            empirical_sigma = np.inf
        
        self.output.log_section_header("FALSE POSITIVE RATE RESULTS")
        self.output.log_message(f"Total realizations analyzed: {n_total}")
        self.output.log_message("")
        
        self.output.log_message("Detection rates in null hypothesis:")
        self.output.log_message(f"  ≥1 detection: {n_ge_1}/{n_total} = {fpr_ge_1*100:.2f}%")
        self.output.log_message(f"  ≥2 detections: {n_ge_2}/{n_total} = {fpr_ge_2*100:.2f}%")
        self.output.log_message(f"  ≥3 detections: {n_ge_3}/{n_total} = {fpr_ge_3*100:.2f}%")
        self.output.log_message(f"  ≥{n_observed} detections: {n_ge_observed}/{n_total} = {fpr_ge_observed*100:.2f}%")
        self.output.log_message("")
        
        self.output.log_message("Location matches (at observed positions):")
        for tolerance, count in matches_by_tolerance.items():
            rate = count / n_total * 100
            self.output.log_message(f"  Within {tolerance} multipoles: {count}/{n_total} = {rate:.3f}%")
        self.output.log_message("")
        
        self.output.log_message(f"Empirical p-value: {empirical_pvalue:.4e}")
        if empirical_sigma < np.inf:
            self.output.log_message(f"Empirical significance: {empirical_sigma:.2f}σ")
        else:
            self.output.log_message(f"Empirical significance: >{np.sqrt(2)*erfcinv(2/n_total):.2f}σ (no detections in null)")
        self.output.log_message("")
        
        # Interpretation
        if fpr_ge_observed < 0.0001:
            interpretation = "Highly significant: observed features extremely rare in null"
        elif fpr_ge_observed < 0.001:
            interpretation = "Very significant: observed features very rare in null"
        elif fpr_ge_observed < 0.05:
            interpretation = "Significant: observed features rare in null"
        elif fpr_ge_observed < 0.1:
            interpretation = "Marginal: observed features uncommon in null"
        else:
            interpretation = "Not significant: observed features common in null (likely artifacts)"
        
        self.output.log_message(f"Interpretation: {interpretation}")
        self.output.log_message("")
        
        # Distribution statistics
        n_detected_array = np.array(n_detected_list)
        mean_detections = np.mean(n_detected_array)
        median_detections = np.median(n_detected_array)
        std_detections = np.std(n_detected_array)
        
        self.output.log_message("Detection count distribution:")
        self.output.log_message(f"  Mean: {mean_detections:.2f}")
        self.output.log_message(f"  Median: {median_detections:.0f}")
        self.output.log_message(f"  Std dev: {std_detections:.2f}")
        self.output.log_message(f"  Max: {np.max(n_detected_array)}")
        self.output.log_message("")
        
        # Compare observed to null distribution
        if len(all_detected_multipoles) > 0:
            all_detected_multipoles = np.array(all_detected_multipoles)
            all_peak_heights = np.array(all_peak_heights)
            
            # Multipole distribution
            multipole_percentiles = {}
            for obs_ell in observed_transitions:
                percentile = percentileofscore(all_detected_multipoles, obs_ell)
                multipole_percentiles[float(obs_ell)] = percentile
            
            self.output.log_message("Observed multipoles vs null distribution:")
            for obs_ell, percentile in multipole_percentiles.items():
                self.output.log_message(f"  ℓ={obs_ell:.0f}: {percentile:.1f}th percentile")
        
        return {
            'n_realizations': int(n_total),
            'n_detected_counts': {
                'ge_1': int(n_ge_1),
                'ge_2': int(n_ge_2),
                'ge_3': int(n_ge_3),
                'ge_observed': int(n_ge_observed)
            },
            'false_positive_rates': {
                'ge_1': float(fpr_ge_1),
                'ge_2': float(fpr_ge_2),
                'ge_3': float(fpr_ge_3),
                'ge_observed': float(fpr_ge_observed)
            },
            'location_matches': {
                str(tol): {'count': int(count), 'rate': float(count/n_total)}
                for tol, count in matches_by_tolerance.items()
            },
            'empirical_pvalue': float(empirical_pvalue),
            'empirical_sigma': float(empirical_sigma) if empirical_sigma < np.inf else None,
            'interpretation': interpretation,
            'detection_distribution': {
                'mean': float(mean_detections),
                'median': float(median_detections),
                'std': float(std_detections),
                'max': int(np.max(n_detected_array)),
                'counts': n_detected_list
            },
            'all_detected_multipoles': all_detected_multipoles.tolist() if len(all_detected_multipoles) > 0 else [],
            'all_peak_heights': all_peak_heights.tolist() if len(all_peak_heights) > 0 else []
        }
    
    def compare_peak_heights(self, observed_heights: np.ndarray,
                            null_heights: np.ndarray) -> Dict[str, Any]:
        """
        Compare observed peak heights to null distribution.
        
        Parameters:
            observed_heights (ndarray): Peak heights from observed data
            null_heights (ndarray): Peak heights from null ensemble
            
        Returns:
            dict: Comparison statistics
        """
        if len(null_heights) == 0:
            return {
                'comparison': 'No peaks in null distribution',
                'percentiles': {}
            }
        
        percentiles = {}
        for i, height in enumerate(observed_heights):
            percentile = percentileofscore(null_heights, height)
            percentiles[f'peak_{i+1}'] = {
                'height': float(height),
                'percentile': float(percentile),
                'interpretation': 'Extreme' if percentile > 99 else 'Typical'
            }
        
        return {
            'null_heights_mean': float(np.mean(null_heights)),
            'null_heights_std': float(np.std(null_heights)),
            'null_heights_max': float(np.max(null_heights)),
            'observed_percentiles': percentiles
        }
    
    def full_analysis(self, ell: np.ndarray, C_ell: np.ndarray, C_ell_err: np.ndarray,
                     observed_transitions: np.ndarray,
                     n_realizations: int = 10000) -> Dict[str, Any]:
        """
        Complete false positive analysis pipeline.
        
        1. Generate null ensemble
        2. Apply detection to each realization
        3. Compute false positive rates
        4. Compare to observations
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Observed power spectrum
            C_ell_err (ndarray): Uncertainties
            observed_transitions (ndarray): Observed transition locations
            n_realizations (int): Number of null realizations (default: 10000)
            
        Returns:
            dict: Complete analysis results
        """
        # Generate null ensemble
        simulator = MonteCarloSimulator(output=self.output)
        ensemble = simulator.generate_ensemble(ell, C_ell, C_ell_err, n_realizations=n_realizations)
        
        # Analyze false positives
        fpr_results = self.analyze_ensemble(ensemble, observed_transitions, n_observed=len(observed_transitions))
        
        # Add ensemble info
        fpr_results['ensemble'] = {
            'n_realizations': ensemble['n_realizations'],
            'f_sky': ensemble['f_sky'],
            'smooth_spectrum_used': True
        }
        
        return fpr_results
