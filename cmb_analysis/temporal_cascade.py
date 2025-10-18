"""
Temporal Cascade Module
========================

Model the temporal cascade of pre-recombination expansion events.

Classes:
    TemporalCascade: Complete cascade model with predictions

Analyzes the temporal sequence of expansion events and predicts higher multipoles.

Paper reference: Supplementary Note 3
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, Any, List

from .utils import OutputManager
from .theoretical import TheoreticalCalculations
from .constants import H_RECOMB, C, G, HBAR


class TemporalCascade:
    """
    Temporal cascade model for pre-recombination expansion events.
    
    Implements:
    - Timeline reconstruction of expansion events
    - Information accumulation dynamics
    - Higher multipole predictions (ℓ₄, ℓ₅, ℓ₆)
    - Instantiation analysis (ℓ₆ as cosmic origin)
    - Comparison with standard inflation
    
    Attributes:
        output (OutputManager): For logging
        theory (TheoreticalCalculations): Theoretical calculations
        
    Example:
        >>> cascade = TemporalCascade()
        >>> results = cascade.calculate_model(transitions, errors, gamma_values)
        >>> print(f"Predicted ℓ₄ = {results['higher_multipoles']['predictions'][0]['ell']:.0f}")
    """
    
    def __init__(self, output: OutputManager = None):
        """Initialize TemporalCascade."""
        self.output = output if output is not None else OutputManager()
        self.theory = TheoreticalCalculations()
    
    def calculate_timeline(self, transitions: np.ndarray, errors: np.ndarray,
                          gamma_values: np.ndarray, H: float = H_RECOMB) -> List[Dict[str, Any]]:
        """
        Calculate timeline of expansion events.
        
        Parameters:
            transitions (ndarray): Detected multipoles
            errors (ndarray): Uncertainties
            gamma_values (ndarray): Observed γ(ℓ) values
            H (float): Hubble parameter
            
        Returns:
            list: Timeline events with physical scales and timescales
        """
        gamma_theory = self.theory.gamma_theoretical(H)
        l_P = np.sqrt(HBAR * G / C**3)  # Planck length
        
        timeline_events = []
        
        for i, (ell, err, gamma_obs) in enumerate(zip(transitions, errors, gamma_values), 1):
            # Physical scale
            scale_pc = self.theory.physical_scale(ell)
            scale_m = scale_pc * 3.086e16  # Convert pc to meters
            
            # Expansion factor
            f_exp = gamma_theory / gamma_obs
            
            # Light crossing time (characteristic timescale)
            t_crossing = scale_m / C
            
            # Information accumulation time
            t_accumulate = ell / (gamma_obs * H)
            
            # Horizon entropy bound
            S_horizon = np.pi * scale_m**2 / l_P**2
            
            self.output.log_message(f"Event {i} (ℓ = {int(ell)}±{int(err)}, {scale_pc:.0f} pc):")
            self.output.log_message(f"  Physical scale: {scale_m:.3e} m")
            self.output.log_message(f"  Light crossing time: {t_crossing:.3e} s ({t_crossing/3.156e7:.2e} years)")
            self.output.log_message(f"  Horizon entropy: {S_horizon:.3e} bits")
            self.output.log_message(f"  Expansion factor: {f_exp:.2f}×")
            self.output.log_message("")
            
            timeline_events.append({
                'n': i,
                'ell': float(ell),
                'ell_error': float(err),
                'scale_pc': float(scale_pc),
                'scale_m': float(scale_m),
                't_crossing': float(t_crossing),
                't_accumulate': float(t_accumulate),
                'S_horizon': float(S_horizon),
                'expansion_factor': float(f_exp),
                'gamma_obs': float(gamma_obs)
            })
        
        return timeline_events
    
    def predict_higher_multipoles(self, timeline_events: List[Dict[str, Any]],
                                  errors: np.ndarray, H: float = H_RECOMB) -> Dict[str, Any]:
        """
        Predict higher multipoles (ℓ₄, ℓ₅, ℓ₆) from exponential decay model.
        
        Parameters:
            timeline_events (list): Timeline from calculate_timeline
            errors (ndarray): Uncertainties from detected transitions
            H (float): Hubble parameter
            
        Returns:
            dict: Predictions including ℓ₄, ℓ₅, ℓ₆
        """
        gamma_theory = self.theory.gamma_theoretical(H)
        
        # Extract observed expansion factors
        n_detected = len(timeline_events)
        n_values = np.arange(1, n_detected + 1)  # Dynamic: [1, 2, ...] based on detected count
        f_observed = np.array([event['expansion_factor'] for event in timeline_events])
        
        # Exponential decay model: f(n) = f_∞ + (f_1 - f_∞) × exp(-n/n0)
        f_inf = 2.0  # Asymptotic limit
        
        def exp_decay(n, n0):
            return f_inf + (f_observed[0] - f_inf) * np.exp(-n / n0)
        
        # Fit decay constant (need at least 2 points)
        if n_detected < 2:
            # Cannot fit decay with < 2 points, use default
            n0_fit = 1.5
            self.output.log_message(f"Warning: Only {n_detected} transition detected, using default decay constant")
        else:
            popt, pcov = curve_fit(exp_decay, n_values, f_observed, p0=[1.5])
            n0_fit = popt[0]
        
        self.output.log_message(f"Expansion factor model:")
        self.output.log_message(f"  f(n) = {f_inf} + {f_observed[0]-f_inf:.2f} × exp(-n/{n0_fit:.2f})")
        self.output.log_message("")
        
        # Predict higher multipoles (starting from next undetected)
        predicted_multipoles = []
        n_start = n_detected + 1  # Start predictions after last detected
        n_end = max(7, n_start + 3)  # Predict at least 3 more
        for n in range(n_start, n_end):
            # Predicted expansion factor
            f_pred = exp_decay(n, n0_fit)
            
            # Predicted γ_obs
            gamma_pred = gamma_theory / f_pred
            
            # Predicted ℓ from quantization condition
            ell_pred = n * np.pi * H / (2 * gamma_pred)
            
            # Estimate uncertainty
            ell_error = errors[0] * np.sqrt(n)
            
            # Physical scale
            scale_pred = self.theory.physical_scale(ell_pred)
            
            # Vacuum energy ratio
            rho_ratio = f_pred**2
            
            self.output.log_message(f"Predicted transition n={n}:")
            self.output.log_message(f"  ℓ_{n} = {int(ell_pred)} ± {int(ell_error)}")
            self.output.log_message(f"  Expansion factor: {f_pred:.2f}×")
            self.output.log_message(f"  Physical scale: {scale_pred:.0f} pc")
            self.output.log_message(f"  ρ_Λ,eff/ρ_Λ,base = {rho_ratio:.2f}")
            self.output.log_message("")
            
            predicted_multipoles.append({
                'n': n,
                'ell': float(ell_pred),
                'ell_error': float(ell_error),
                'gamma_obs': float(gamma_pred),
                'expansion_factor': float(f_pred),
                'scale_pc': float(scale_pred),
                'rho_ratio': float(rho_ratio)
            })
        
        return {
            'predictions': predicted_multipoles,
            'model': f'f(n) = {f_inf} + (f₁-{f_inf}) × exp(-n/{n0_fit:.2f})',
            'f_infinity': float(f_inf),
            'decay_constant': float(n0_fit)
        }
    
    def analyze_instantiation(self, predicted_multipoles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze ℓ₆ as universe instantiation point.
        
        Parameters:
            predicted_multipoles (list): Predictions from predict_higher_multipoles
            
        Returns:
            dict: Instantiation analysis
        """
        # ℓ₆ is the instantiation point (no expansion)
        ell6_data = predicted_multipoles[2]  # n=6
        
        self.output.log_message(f"\nINSTANTIATION ANALYSIS:")
        self.output.log_message(f"  ℓ₆ = {ell6_data['ell']:.0f} (instantiation scale)")
        self.output.log_message(f"  Physical scale: {ell6_data['scale_pc']:.0f} pc = {ell6_data['scale_pc']*3.086e16:.3e} m")
        self.output.log_message(f"  This is the primordial boundary condition")
        self.output.log_message(f"  'ℓ₆ simply IS' - no expansion factor")
        self.output.log_message("")
        self.output.log_message(f"  ℓ₅ = {predicted_multipoles[1]['ell']:.0f} (first expansion after instantiation)")
        self.output.log_message(f"  Expected anomalous factor: ~3.11× (initiates cascade)")
        self.output.log_message("")
        self.output.log_message(f"  ℓ₄ = {predicted_multipoles[0]['ell']:.0f} (begins exponential decay)")
        self.output.log_message(f"  Predicted factor: {predicted_multipoles[0]['expansion_factor']:.2f}×")
        
        return {
            'instantiation_multipole': int(ell6_data['ell']),
            'instantiation_scale_pc': float(ell6_data['scale_pc']),
            'first_expansion_multipole': int(predicted_multipoles[1]['ell']),
            'first_expansion_anomaly': 3.11,  # Empirically required value
            'interpretation': 'ℓ₆ is universe instantiation, ℓ₅ is first expansion'
        }
    
    def calculate_model(self, transitions: np.ndarray, errors: np.ndarray,
                       gamma_values: np.ndarray, H: float = H_RECOMB) -> Dict[str, Any]:
        """
        Complete temporal cascade model calculation.
        
        Parameters:
            transitions (ndarray): Detected multipoles
            errors (ndarray): Uncertainties
            gamma_values (ndarray): Observed γ(ℓ) values
            H (float): Hubble parameter
            
        Returns:
            dict: Complete cascade results
        """
        self.output.log_section_header("TEMPORAL CASCADE MODEL")
        
        # Timeline
        self.output.log_subsection("TIMELINE OF EXPANSION EVENTS")
        timeline_events = self.calculate_timeline(transitions, errors, gamma_values, H)
        
        # Higher multipole predictions
        self.output.log_subsection("PREDICTIONS FOR HIGHER MULTIPOLES")
        higher_multipoles = self.predict_higher_multipoles(timeline_events, errors, H)
        
        # Instantiation analysis
        instantiation = self.analyze_instantiation(higher_multipoles['predictions'])
        
        return {
            'timeline': {
                'events': timeline_events,
                'chronological_order': 'Smallest scales (highest ℓ) expanded first'
            },
            'higher_multipoles': higher_multipoles,
            'instantiation': instantiation,
            'cascade_mechanism': {
                'equation': 'dI/dt = γ[1-I/I_max] - HI',
                'threshold': 'Expansion triggered when I → I_crit',
                'feedback': 'Each expansion modifies substrate for next scale'
            }
        }

