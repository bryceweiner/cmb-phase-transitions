"""
Phase Transition Detector Module
=================================

Detect and characterize quantum phase transitions in CMB E-mode polarization data.

Classes:
    PhaseTransitionDetector: Find and analyze phase transitions from power spectrum

Methods use Savitzky-Golay filtering for derivative computation and peak detection
for identifying transition locations.

Paper reference: Methods section, line ~271
"""

import numpy as np
from scipy.signal import savgol_filter, find_peaks
from typing import Tuple, Dict, Any

from .utils import OutputManager
from .theoretical import TheoreticalCalculations
from .constants import H_RECOMB, Z_RECOMB, PAPER_REFERENCES


class PhaseTransitionDetector:
    """
    Detect and characterize phase transitions in CMB E-mode data.
    
    Uses derivative analysis to identify sharp discontinuities in the power spectrum
    that correspond to pre-recombination expansion events.
    
    Attributes:
        output (OutputManager): For logging
        theory (TheoreticalCalculations): For theoretical predictions
        window_size (int): Savitzky-Golay filter window
        poly_order (int): Polynomial order for filter
        
    Example:
        >>> detector = PhaseTransitionDetector()
        >>> transitions, errors = detector.detect_and_analyze(ell, C_ell)
        >>> print(f"Found {len(transitions)} transitions")
    """
    
    def __init__(self, output: OutputManager = None, 
                 window_size: int = 25, poly_order: int = 3):
        """
        Initialize PhaseTransitionDetector.
        
        Parameters:
            output (OutputManager, optional): For logging
            window_size (int): Window size for Savitzky-Golay filter (must be odd, default: 25)
            poly_order (int): Polynomial order for Savitzky-Golay filter (default: 3)
        """
        self.output = output if output is not None else OutputManager()
        self.theory = TheoreticalCalculations()
        self.window_size = window_size if window_size % 2 == 1 else window_size + 1
        self.poly_order = poly_order
    
    def compute_derivative(self, ell: np.ndarray, C_ell: np.ndarray) -> np.ndarray:
        """
        Compute dC_ℓ/dℓ using Savitzky-Golay filter.
        
        Paper reference: Methods section, line ~271
        
        The Savitzky-Golay filter computes the derivative while smoothing noise,
        making subtle discontinuities more apparent.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Power spectrum
            
        Returns:
            ndarray: Derivative dC_ℓ/dℓ
        """
        dC_dell = savgol_filter(C_ell, self.window_size, self.poly_order, 
                               deriv=1, delta=np.median(np.diff(ell)))
        return dC_dell
    
    def detect_transitions(self, ell: np.ndarray, dC_dell: np.ndarray,
                          min_distance: int = 500, 
                          prominence_factor: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect phase transitions as peaks in |dC_ℓ/dℓ|.
        
        Paper reference: Methods section, line ~271
        
        Phase transitions appear as sharp discontinuities in the power spectrum,
        which manifest as peaks in the absolute derivative.
        
        Parameters:
            ell (ndarray): Multipole values
            dC_dell (ndarray): Derivative dC_ℓ/dℓ
            min_distance (int): Minimum separation between peaks in multipole space
            prominence_factor (float): Threshold = median + factor × std
            
        Returns:
            tuple: (peak_indices, peak_multipoles, peak_heights)
                - peak_indices: Array indices of detected peaks
                - peak_multipoles: Multipole values of peaks
                - peak_heights: Heights of peaks in |dC_ℓ/dℓ|
        """
        abs_deriv = np.abs(dC_dell)
        
        # Calculate detection threshold
        background = np.median(abs_deriv)
        std_dev = np.std(abs_deriv)
        threshold = background + prominence_factor * std_dev
        
        self.output.log_message("Derivative statistics:")
        self.output.log_message(f"  Median: {background:.3e}")
        self.output.log_message(f"  Std dev: {std_dev:.3e}")
        self.output.log_message(f"  Threshold: {threshold:.3e}")
        self.output.log_message(f"  Max: {abs_deriv.max():.3e}")
        
        # Find peaks above background
        min_dist_points = int(min_distance / np.median(np.diff(ell)))
        peaks, properties = find_peaks(abs_deriv, 
                                      height=background,
                                      distance=min_dist_points)
        
        peak_ells = ell[peaks]
        peak_heights = abs_deriv[peaks]
        
        # Sort by height (strongest signals first)
        if len(peaks) > 0:
            sort_idx = np.argsort(peak_heights)[::-1]
            peak_ells = peak_ells[sort_idx]
            peak_heights = peak_heights[sort_idx]
            peaks = peaks[sort_idx]
        
        return peaks, peak_ells, peak_heights
    
    def estimate_uncertainties(self, ell: np.ndarray, dC_dell: np.ndarray,
                              peaks: np.ndarray, n_transitions: int = 3) -> np.ndarray:
        """
        Estimate uncertainties from peak widths.
        
        Uses half-width at half-maximum (HWHM) method to estimate the
        uncertainty on each transition location.
        
        Parameters:
            ell (ndarray): Multipole values
            dC_dell (ndarray): Derivative
            peaks (ndarray): Peak indices
            n_transitions (int): Number of transitions to analyze
            
        Returns:
            ndarray: Uncertainty estimates for each transition
        """
        trans_errors = []
        for peak_idx in peaks[:n_transitions]:
            # Extract local region around peak
            local_region = slice(max(0, peak_idx-10), min(len(ell), peak_idx+10))
            local_deriv = np.abs(dC_dell[local_region])
            local_ell = ell[local_region]
            
            # Find HWHM
            half_max = local_deriv.max() / 2
            width_points = np.sum(local_deriv > half_max)
            width_ell = width_points * np.median(np.diff(local_ell))
            trans_errors.append(width_ell / 2)
        
        return np.array(trans_errors)
    
    def analyze_transitions(self, transitions: np.ndarray, 
                           errors: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of detected transitions.
        
        Calculates:
        - Physical scales (paper line 76-78)
        - Theoretical baseline γ (paper line 107)
        - Observed scale-dependent γ(ℓ) (paper line 109)
        - Expansion factors (paper line 117)
        - Harmonic scaling ratios (paper line 99)
        - Vacuum energy variations (paper line 115)
        
        Parameters:
            transitions (ndarray): Detected multipole values
            errors (ndarray): Uncertainties on multipoles
            
        Returns:
            dict: Comprehensive analysis results
        """
        results = {
            'transitions': [],
            'gamma_baseline': None,
            'gamma_scale_dependent': [],
            'expansion_factors': [],
            'harmonic_ratios': [],
            'vacuum_energy': [],
            'power_law_fit': {}
        }
        
        # Calculate theoretical baseline
        gamma_baseline = self.theory.gamma_theoretical(H_RECOMB)
        results['gamma_baseline'] = float(gamma_baseline)
        
        self.output.log_section_header("COMPREHENSIVE TRANSITION ANALYSIS")
        self.output.log_message("")
        self.output.log_message(f"Theoretical baseline (paper line {PAPER_REFERENCES['gamma_theoretical']}):")
        self.output.log_message(f"  γ_theory = H/ln(πc²/ℏGH²) = {gamma_baseline:.6e} s⁻¹")
        self.output.log_message(f"  at z = {Z_RECOMB} (recombination)")
        self.output.log_message("")
        
        # Analyze each transition
        gamma_scale = []
        expansion_factors_list = []
        
        for i, (ell_val, err) in enumerate(zip(transitions, errors), 1):
            # Physical scale
            phys_scale = self.theory.physical_scale(ell_val)
            
            # Observed gamma from quantization
            gamma_obs = self.theory.gamma_from_quantization(ell_val, i, H_RECOMB)
            gamma_scale.append(gamma_obs)
            
            # Expansion factor
            exp_factor = self.theory.expansion_factor(gamma_obs, gamma_baseline)
            expansion_factors_list.append(exp_factor)
            
            self.output.log_message(f"Transition {i}: ℓ = {int(ell_val)} ± {int(err)}")
            self.output.log_message(f"  Physical scale: {phys_scale:.0f} pc")
            self.output.log_message(f"  γ(ℓ): {gamma_obs:.3e} s⁻¹ ({gamma_obs/gamma_baseline:.3f}× baseline)")
            self.output.log_message(f"  Expansion factor: {exp_factor:.2f}×")
            self.output.log_message("")
            
            results['transitions'].append({
                'n': i,
                'ell': float(ell_val),
                'ell_error': float(err),
                'physical_scale_pc': float(phys_scale),
                'gamma_obs': float(gamma_obs),
                'gamma_ratio': float(gamma_obs/gamma_baseline),
                'expansion_factor': float(exp_factor)
            })
            results['gamma_scale_dependent'].append(float(gamma_obs))
            results['expansion_factors'].append(float(exp_factor))
        
        # Power law fit: γ(ℓ) = A × ℓ^α
        log_ell = np.log(transitions)
        log_gamma = np.log(gamma_scale)
        coeffs = np.polyfit(log_ell, log_gamma, 1)
        alpha = coeffs[0]
        A = np.exp(coeffs[1])
        
        results['power_law_fit'] = {
            'alpha': float(alpha),
            'A': float(A),
            'formula': 'γ(ℓ) = A × ℓ^α'
        }
        
        self.output.log_message("Power law fit γ(ℓ) = A × ℓ^α:")
        self.output.log_message(f"  α = {alpha:.4f}")
        self.output.log_message(f"  A = {A:.3e} s⁻¹")
        self.output.log_message("")
        
        # Vacuum energy analysis
        self.output.log_subsection("SCALE-DEPENDENT VACUUM ENERGY")
        self.output.log_message("From ρ_Λ,eff(ℓ) = ρ_Λ,baseline × [γ_theory/γ_obs(ℓ)]²:")
        self.output.log_message("")
        
        # Table header
        self.output.log_message(f"{'Scale (pc)':<15} {'γ_obs':<18} {'γ_theory/γ_obs':<18} {'ρ_Λ,eff/ρ_Λ,base':<20} {'Expansion':<12}")
        self.output.log_message("-" * 85)
        
        vacuum_energy_results = []
        for i, (trans, gamma_obs, exp_fac) in enumerate(zip(transitions, gamma_scale, expansion_factors_list)):
            phys_scale = self.theory.physical_scale(trans)
            gamma_ratio = gamma_baseline / gamma_obs
            rho_ratio = gamma_ratio**2
            
            self.output.log_message(
                f"{phys_scale:<15.0f} {gamma_obs:<18.2e} {gamma_ratio:<18.2f} {rho_ratio:<20.2f} {exp_fac:<12.2f}×")
            
            vacuum_energy_results.append({
                'n': i+1,
                'scale_pc': float(phys_scale),
                'gamma_obs': float(gamma_obs),
                'gamma_ratio': float(gamma_ratio),
                'rho_Lambda_ratio': float(rho_ratio),
                'expansion_factor': float(exp_fac)
            })
        
        results['vacuum_energy'] = vacuum_energy_results
        self.output.log_message("")
        self.output.log_message("Expansion factors: larger regions (higher ρ_Λ,eff) expanded more")
        self.output.log_message("")
        
        # Harmonic scaling ratios
        self.output.log_message(f"Harmonic scaling (paper line {PAPER_REFERENCES['harmonic_ratio']}):")
        self.output.log_message("ℓ_(n+1)/ℓ_n = (n+1)/n × [γ(ℓ_n)/γ(ℓ_(n+1))]")
        self.output.log_message("")
        
        for i in range(len(transitions)-1):
            ratio_obs = transitions[i+1] / transitions[i]
            ratio_err = ratio_obs * np.sqrt((errors[i+1]/transitions[i+1])**2 + 
                                           (errors[i]/transitions[i])**2)
            
            ratio_predicted = self.theory.harmonic_ratio_corrected(i+1, gamma_scale[i], gamma_scale[i+1])
            deviation = abs(ratio_obs - ratio_predicted) / ratio_predicted * 100
            
            self.output.log_message(f"Ratio ℓ_{i+2}/ℓ_{i+1}:")
            self.output.log_message(f"  Observed: {ratio_obs:.4f} ± {ratio_err:.4f}")
            self.output.log_message(f"  Predicted: {ratio_predicted:.4f}")
            self.output.log_message(f"  Deviation: {deviation:.1f}%")
            self.output.log_message("")
            
            results['harmonic_ratios'].append({
                'ratio': f'ℓ_{i+2}/ℓ_{i+1}',
                'observed': float(ratio_obs),
                'error': float(ratio_err),
                'predicted': float(ratio_predicted),
                'deviation_percent': float(deviation)
            })
        
        return results
    
    def detect_and_analyze(self, ell: np.ndarray, C_ell: np.ndarray,
                          n_transitions: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete detection and analysis pipeline.
        
        Convenience method that runs the full pipeline:
        1. Compute derivative
        2. Detect peaks
        3. Estimate uncertainties
        4. Analyze transitions
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Power spectrum
            n_transitions (int): Number of transitions to detect
            
        Returns:
            tuple: (transitions, errors) - detected multipoles and uncertainties
        """
        # Compute derivative
        dC_dell = self.compute_derivative(ell, C_ell)
        
        # Detect transitions
        peaks, peak_ells, peak_heights = self.detect_transitions(ell, dC_dell)
        
        # Take top n_transitions
        transitions = peak_ells[:n_transitions]
        
        # Estimate uncertainties
        errors = self.estimate_uncertainties(ell, dC_dell, peaks, n_transitions)
        
        # Log detected transitions
        self.output.log_message("\nDetected transitions:")
        for i, (trans, err) in enumerate(zip(transitions, errors), 1):
            self.output.log_message(f"  ℓ_{i} = {trans:.0f} ± {err:.0f}")
        self.output.log_message("")
        
        return transitions, errors

