"""
Detection Threshold Calibration Module
======================================

Calibrate detection thresholds to achieve target false positive rates.

Classes:
    ThresholdCalibrator: Determine optimal detection thresholds via null simulations

This module implements proper statistical calibration by:
1. Generating null hypothesis ensembles
2. Testing multiple threshold values
3. Measuring false positive rates
4. Identifying threshold for target FPR (e.g., 5%)

Paper reference: Statistical validation methodology
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy.interpolate import interp1d

from .utils import OutputManager
from .monte_carlo_simulator import MonteCarloSimulator
from .phase_detector import PhaseTransitionDetector


class ThresholdCalibrator:
    """
    Calibrate detection thresholds for acceptable false positive rates.
    
    Uses Monte Carlo null simulations to determine what prominence_factor
    threshold achieves target false positive rate (typically 5% or 0.27%).
    
    Attributes:
        output (OutputManager): For logging
        simulator (MonteCarloSimulator): For generating null realizations
        detector (PhaseTransitionDetector): Detection pipeline
        
    Example:
        >>> calibrator = ThresholdCalibrator()
        >>> result = calibrator.calibrate_threshold(ell, C_ell, C_ell_err, target_fpr=0.05)
        >>> print(f"Calibrated threshold: {result['calibrated_prominence_factor']}")
    """
    
    def __init__(self, output: OutputManager = None, f_sky: float = 0.4):
        """
        Initialize ThresholdCalibrator.
        
        Parameters:
            output (OutputManager, optional): For logging
            f_sky (float): Sky fraction for cosmic variance (default: 0.4 for ACT)
        """
        self.output = output if output is not None else OutputManager()
        self.simulator = MonteCarloSimulator(output=self.output, f_sky=f_sky, use_mps=True)
        self.f_sky = f_sky
    
    def test_threshold(self, ensemble: Dict[str, Any], 
                      prominence_factor: float,
                      n_target: int = 3,
                      window_size: int = 25) -> Dict[str, Any]:
        """
        Test single threshold value on null ensemble.
        
        Parameters:
            ensemble (dict): Null hypothesis ensemble from MonteCarloSimulator
            prominence_factor (float): Detection threshold to test
            n_target (int): Target number of transitions (default: 3)
            window_size (int): Savitzky-Golay window size (default: 25)
            
        Returns:
            dict: Detection statistics for this threshold
        """
        realizations = ensemble['realizations']
        ell = ensemble['ell']
        n_realizations = len(realizations)
        
        # Create detector with specified parameters
        detector = PhaseTransitionDetector(output=self.output, window_size=window_size)
        
        # Storage
        n_detected_list = []
        n_ge_target = 0
        
        # Test each realization
        for i, C_ell_mock in enumerate(realizations):
            # Compute derivative
            dC_dell = detector.compute_derivative(ell, C_ell_mock)
            
            # Detect with this threshold
            peaks, peak_ells, peak_heights = detector.detect_transitions(
                ell, dC_dell, 
                min_distance=500,
                prominence_factor=prominence_factor
            )
            
            n_detected = len(peak_ells)
            n_detected_list.append(n_detected)
            
            if n_detected >= n_target:
                n_ge_target += 1
        
        # Calculate statistics
        fpr = n_ge_target / n_realizations
        mean_detections = np.mean(n_detected_list)
        median_detections = np.median(n_detected_list)
        std_detections = np.std(n_detected_list)
        
        return {
            'prominence_factor': float(prominence_factor),
            'n_target': int(n_target),
            'n_realizations': int(n_realizations),
            'n_ge_target': int(n_ge_target),
            'false_positive_rate': float(fpr),
            'mean_detections': float(mean_detections),
            'median_detections': float(median_detections),
            'std_detections': float(std_detections),
            'detection_counts': n_detected_list
        }
    
    def calibrate_threshold(self,
                          ell: np.ndarray,
                          C_ell: np.ndarray,
                          C_ell_err: np.ndarray,
                          target_fpr: float = 0.05,
                          n_target: int = 3,
                          n_realizations: int = 10000,
                          test_thresholds: Optional[List[float]] = None,
                          window_size: int = 25) -> Dict[str, Any]:
        """
        Calibrate detection threshold to achieve target false positive rate.
        
        This is the STANDARD METHOD for empirical discovery:
        1. Generate null hypothesis ensemble
        2. Test range of thresholds
        3. Find threshold that gives target FPR (e.g., 5% or 0.27% for 3σ)
        4. Use calibrated threshold on real data
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Observed power spectrum
            C_ell_err (ndarray): Uncertainties
            target_fpr (float): Target false positive rate (default: 0.05 = 5%)
            n_target (int): Number of transitions to detect (default: 3)
            n_realizations (int): Monte Carlo null realizations (default: 10000)
            test_thresholds (list, optional): Prominence factors to test 
                                              (default: [0.5, 1, 2, 3, 5, 7, 10, 15, 20])
            window_size (int): Savitzky-Golay window size (default: 25)
            
        Returns:
            dict: Calibration results including calibrated threshold
        """
        self.output.log_section_header("DETECTION THRESHOLD CALIBRATION")
        self.output.log_message(f"Target false positive rate: {target_fpr*100:.2f}%")
        self.output.log_message(f"Target number of detections: {n_target}")
        self.output.log_message(f"Null realizations: {n_realizations}")
        self.output.log_message(f"Window size: {window_size}")
        self.output.log_message("")
        
        # Default threshold range
        if test_thresholds is None:
            test_thresholds = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
        
        # Generate null ensemble
        self.output.log_message("Generating null hypothesis ensemble...")
        ensemble = self.simulator.generate_ensemble(
            ell, C_ell, C_ell_err,
            n_realizations=n_realizations,
            use_smooth=True
        )
        
        # Test each threshold
        self.output.log_message("\nTesting threshold values...")
        self.output.log_message(f"{'Prominence':<12} {'FPR':<10} {'Mean Det':<10} {'Median':<10} {'Interpretation'}")
        self.output.log_message("-" * 65)
        
        results_by_threshold = []
        
        for prominence in test_thresholds:
            result = self.test_threshold(
                ensemble, prominence, 
                n_target=n_target,
                window_size=window_size
            )
            
            fpr = result['false_positive_rate']
            mean_det = result['mean_detections']
            median_det = result['median_detections']
            
            # Interpretation
            if fpr < 0.001:
                interp = "Very conservative"
            elif fpr < 0.01:
                interp = "Conservative (1%)"
            elif fpr < 0.05:
                interp = "Standard (5%)"
            elif fpr < 0.1:
                interp = "Moderate (10%)"
            elif fpr < 0.3:
                interp = "Liberal (30%)"
            else:
                interp = "Too permissive"
            
            self.output.log_message(
                f"{prominence:<12.1f} {fpr*100:<10.2f}% {mean_det:<10.2f} "
                f"{median_det:<10.0f} {interp}"
            )
            
            results_by_threshold.append(result)
        
        self.output.log_message("")
        
        # Find calibrated threshold via interpolation
        thresholds_array = np.array([r['prominence_factor'] for r in results_by_threshold])
        fprs_array = np.array([r['false_positive_rate'] for r in results_by_threshold])
        
        # Check if target FPR is achievable
        if fprs_array[-1] > target_fpr:
            self.output.log_message(f"⚠ WARNING: Target FPR {target_fpr*100:.2f}% not achievable")
            self.output.log_message(f"  Lowest FPR tested: {fprs_array[-1]*100:.2f}%")
            self.output.log_message(f"  Recommend testing higher thresholds or accepting higher FPR")
            calibrated_prominence = test_thresholds[-1]
            calibrated_fpr = fprs_array[-1]
        else:
            # Interpolate to find exact threshold for target FPR
            # Sort by FPR (descending)
            sort_idx = np.argsort(fprs_array)[::-1]
            fprs_sorted = fprs_array[sort_idx]
            thresh_sorted = thresholds_array[sort_idx]
            
            # Interpolate
            if target_fpr in fprs_sorted:
                # Exact match
                idx = np.where(fprs_sorted == target_fpr)[0][0]
                calibrated_prominence = thresh_sorted[idx]
                calibrated_fpr = target_fpr
            else:
                # Interpolate
                interp_func = interp1d(fprs_sorted, thresh_sorted, 
                                      kind='linear', fill_value='extrapolate')
                calibrated_prominence = float(interp_func(target_fpr))
                calibrated_fpr = target_fpr
        
        self.output.log_section_header("CALIBRATION RESULTS")
        self.output.log_message(f"Target FPR: {target_fpr*100:.2f}%")
        self.output.log_message(f"Calibrated prominence_factor: {calibrated_prominence:.2f}")
        self.output.log_message(f"Achieved FPR: {calibrated_fpr*100:.2f}%")
        self.output.log_message("")
        
        # Significance thresholds for reference
        self.output.log_message("Common significance thresholds:")
        self.output.log_message("  3σ (99.73% confidence): FPR = 0.27%")
        self.output.log_message("  5σ (discovery):         FPR = 0.000057% (5.7×10⁻⁵%)")
        self.output.log_message("  Standard (95% conf):    FPR = 5%")
        self.output.log_message("")
        
        # Find closest to standard significance levels
        standard_thresholds = {
            '3-sigma (0.27%)': 0.0027,
            '5-sigma (5.7e-5%)': 5.7e-7,
            '95% conf (5%)': 0.05
        }
        
        closest_matches = {}
        for name, target in standard_thresholds.items():
            if fprs_array.min() <= target <= fprs_array.max():
                interp_func = interp1d(fprs_sorted, thresh_sorted, 
                                      kind='linear', fill_value='extrapolate')
                thresh_for_target = float(interp_func(target))
                closest_matches[name] = thresh_for_target
        
        if closest_matches:
            self.output.log_message("Thresholds for standard significance levels:")
            for name, thresh in closest_matches.items():
                self.output.log_message(f"  {name:<25} prominence = {thresh:.2f}")
            self.output.log_message("")
        
        return {
            'target_fpr': float(target_fpr),
            'calibrated_prominence_factor': float(calibrated_prominence),
            'achieved_fpr': float(calibrated_fpr),
            'tested_thresholds': results_by_threshold,
            'standard_thresholds': closest_matches,
            'ensemble_info': {
                'n_realizations': ensemble['n_realizations'],
                'f_sky': ensemble['f_sky']
            },
            'parameters': {
                'n_target': int(n_target),
                'window_size': int(window_size)
            }
        }
    
    def detect_with_calibrated_threshold(self,
                                        ell: np.ndarray,
                                        C_ell: np.ndarray,
                                        C_ell_err: np.ndarray,
                                        calibrated_prominence: float,
                                        window_size: int = 25,
                                        n_transitions: int = 3) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Apply detection with calibrated threshold.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Power spectrum
            C_ell_err (ndarray): Uncertainties
            calibrated_prominence (float): Calibrated prominence_factor
            window_size (int): Savitzky-Golay window size (default: 25)
            n_transitions (int): Maximum transitions to report (default: 3)
            
        Returns:
            tuple: (transitions, errors, detection_info)
        """
        self.output.log_section_header("DETECTION WITH CALIBRATED THRESHOLD")
        self.output.log_message(f"Using calibrated prominence_factor: {calibrated_prominence:.2f}")
        self.output.log_message(f"Window size: {window_size}")
        self.output.log_message("")
        
        # Create detector
        detector = PhaseTransitionDetector(
            output=self.output,
            window_size=window_size
        )
        
        # Compute derivative
        dC_dell = detector.compute_derivative(ell, C_ell)
        
        # Detect with calibrated threshold
        peaks, peak_ells, peak_heights = detector.detect_transitions(
            ell, dC_dell,
            min_distance=500,
            prominence_factor=calibrated_prominence
        )
        
        n_detected = len(peak_ells)
        
        self.output.log_message(f"Detections with calibrated threshold: {n_detected}")
        
        if n_detected == 0:
            self.output.log_message("")
            self.output.log_message("✓ NO STATISTICALLY SIGNIFICANT DETECTIONS")
            self.output.log_message("  (No features exceed calibrated threshold)")
            self.output.log_message("")
            
            return np.array([]), np.array([]), {
                'n_detected': 0,
                'conclusion': 'No statistically significant detections above calibrated threshold',
                'calibrated_prominence': float(calibrated_prominence)
            }
        
        # Take top n_transitions or all if fewer
        transitions = peak_ells[:min(n_transitions, n_detected)]
        
        # Estimate uncertainties
        errors = detector.estimate_uncertainties(ell, dC_dell, peaks, min(n_transitions, n_detected))
        
        self.output.log_message("")
        self.output.log_message(f"Detected transitions (with calibrated threshold):")
        for i, (trans, err) in enumerate(zip(transitions, errors), 1):
            self.output.log_message(f"  ℓ_{i} = {trans:.0f} ± {err:.0f}")
        self.output.log_message("")
        
        return transitions, errors, {
            'n_detected': int(n_detected),
            'n_reported': len(transitions),
            'all_peaks': peak_ells.tolist(),
            'all_heights': peak_heights.tolist(),
            'calibrated_prominence': float(calibrated_prominence),
            'conclusion': f'{n_detected} detections above calibrated threshold'
        }
    
    def full_calibration_pipeline(self,
                                  ell: np.ndarray,
                                  C_ell: np.ndarray,
                                  C_ell_err: np.ndarray,
                                  target_fpr: float = 0.05,
                                  n_target: int = 3,
                                  n_realizations: int = 10000,
                                  window_size: int = 25) -> Dict[str, Any]:
        """
        Complete calibration pipeline.
        
        1. Generate null ensemble
        2. Test range of thresholds
        3. Find calibrated threshold for target FPR
        4. Apply to real data
        5. Report detections with proper statistical context
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Power spectrum
            C_ell_err (ndarray): Uncertainties
            target_fpr (float): Target false positive rate (default: 0.05 = 5%)
            n_target (int): Target number of transitions (default: 3)
            n_realizations (int): Null realizations (default: 10000)
            window_size (int): Savitzky-Golay window (default: 25)
            
        Returns:
            dict: Complete calibration and detection results
        """
        # Step 1: Calibrate threshold
        calibration = self.calibrate_threshold(
            ell, C_ell, C_ell_err,
            target_fpr=target_fpr,
            n_target=n_target,
            n_realizations=n_realizations,
            window_size=window_size
        )
        
        # Step 2: Detect with calibrated threshold
        transitions, errors, detection_info = self.detect_with_calibrated_threshold(
            ell, C_ell, C_ell_err,
            calibrated_prominence=calibration['calibrated_prominence_factor'],
            window_size=window_size,
            n_transitions=n_target
        )
        
        # Step 3: Compute empirical significance
        n_detected = detection_info['n_detected']
        
        if n_detected >= n_target:
            # Count how many null realizations had >= n_detected
            # Use the tested threshold closest to calibrated
            tested = calibration['tested_thresholds']
            calib_prom = calibration['calibrated_prominence_factor']
            
            # Find closest tested threshold
            closest_idx = np.argmin([abs(t['prominence_factor'] - calib_prom) for t in tested])
            closest_result = tested[closest_idx]
            
            empirical_fpr = closest_result['false_positive_rate']
            
            # Convert to sigma
            from scipy.special import erfcinv
            if empirical_fpr > 0:
                empirical_sigma = np.sqrt(2) * erfcinv(2 * empirical_fpr)
            else:
                empirical_sigma = np.inf
            
            significance = {
                'n_detected': int(n_detected),
                'empirical_fpr': float(empirical_fpr),
                'empirical_sigma': float(empirical_sigma) if empirical_sigma < np.inf else None,
                'interpretation': 'Significant detection' if empirical_fpr < target_fpr else 'Detection at target threshold'
            }
        else:
            significance = {
                'n_detected': int(n_detected),
                'empirical_fpr': None,
                'empirical_sigma': None,
                'interpretation': f'Only {n_detected} detections (below target of {n_target})'
            }
        
        # Compile complete results
        return {
            'calibration': calibration,
            'detection': {
                'transitions': transitions.tolist(),
                'errors': errors.tolist(),
                'info': detection_info
            },
            'significance': significance,
            'recommendation': self._generate_recommendation(
                n_detected, n_target, target_fpr, 
                calibration['calibrated_prominence_factor']
            )
        }
    
    def _generate_recommendation(self, n_detected: int, n_target: int,
                                target_fpr: float, calibrated_prominence: float) -> str:
        """Generate interpretation and recommendation."""
        
        if n_detected == 0:
            return (
                f"No statistically significant detections at {target_fpr*100:.2f}% FPR threshold. "
                f"With calibrated prominence_factor = {calibrated_prominence:.2f}, "
                f"no features exceed the noise floor. "
                f"Conclusion: No evidence for phase transitions in current data."
            )
        elif n_detected < n_target:
            return (
                f"Detected {n_detected} transitions (below target of {n_target}) "
                f"at {target_fpr*100:.2f}% FPR threshold (prominence = {calibrated_prominence:.2f}). "
                f"Evidence is weaker than anticipated. "
                f"Conclusion: Marginal evidence, requires independent confirmation."
            )
        else:
            return (
                f"Detected {n_detected} transitions at {target_fpr*100:.2f}% FPR threshold "
                f"(prominence = {calibrated_prominence:.2f}). "
                f"Conclusion: Statistically significant detections above calibrated noise floor."
            )

