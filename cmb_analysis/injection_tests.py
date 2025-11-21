"""
Injection Testing Module
========================

Test detection sensitivity through signal injection and recovery.

Injects fake transitions of varying amplitudes into data or simulations
and measures recovery rate to establish minimum detectable signal.

Classes:
    InjectionTester: Injection/recovery sensitivity analysis

Paper reference: Methods, sensitivity analysis
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from .utils import OutputManager
from .phase_detector import PhaseTransitionDetector
from .monte_carlo_simulator import MonteCarloSimulator


class InjectionTester:
    """
    Injection and recovery testing for sensitivity analysis.
    
    Tests ability to detect transitions of varying amplitudes by:
    1. Injecting fake transitions into clean data
    2. Running detection pipeline
    3. Measuring recovery rate vs amplitude
    4. Establishing minimum detectable signal
    
    Attributes:
        output (OutputManager): For logging
        detector (PhaseTransitionDetector): Detection algorithm
        
    Example:
        >>> tester = InjectionTester()
        >>> sensitivity = tester.amplitude_scan(ell, C_ell_smooth, C_ell_err)
    """
    
    def __init__(self, output: OutputManager = None,
                 detector: PhaseTransitionDetector = None):
        """
        Initialize InjectionTester.
        
        Parameters:
            output (OutputManager, optional): For logging
            detector (PhaseTransitionDetector, optional): Detection algorithm
        """
        self.output = output if output is not None else OutputManager()
        self.detector = detector if detector is not None else PhaseTransitionDetector(output=self.output)
    
    def inject_transition(self, ell: np.ndarray, C_ell: np.ndarray,
                         location: float, amplitude: float,
                         transition_type: str = 'step') -> np.ndarray:
        """
        Inject artificial transition into spectrum.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Power spectrum
            location (float): Multipole location of transition
            amplitude (float): Fractional amplitude (e.g., -0.05 for 5% drop)
            transition_type (str): 'step' or 'smooth' transition
            
        Returns:
            ndarray: Spectrum with injected transition
        """
        C_ell_injected = C_ell.copy()
        
        if transition_type == 'step':
            # Sharp step function
            mask = ell > location
            C_ell_injected[mask] *= (1 + amplitude)
            
        elif transition_type == 'smooth':
            # Smooth transition (tanh profile)
            width = 50  # multipoles
            profile = 0.5 * (1 + np.tanh((ell - location) / width))
            C_ell_injected *= (1 + amplitude * profile)
            
        else:
            raise ValueError(f"Unknown transition type: {transition_type}")
        
        return C_ell_injected
    
    def inject_multiple_transitions(self, ell: np.ndarray, C_ell: np.ndarray,
                                   transitions: List[Tuple[float, float]]) -> np.ndarray:
        """
        Inject multiple transitions.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Power spectrum
            transitions (list): [(location_1, amplitude_1), ...]
            
        Returns:
            ndarray: Spectrum with all transitions injected
        """
        C_ell_injected = C_ell.copy()
        
        for location, amplitude in transitions:
            C_ell_injected = self.inject_transition(ell, C_ell_injected, location, amplitude)
        
        return C_ell_injected
    
    def test_recovery(self, ell: np.ndarray, C_ell_injected: np.ndarray,
                     true_location: float, tolerance: float = 100) -> Tuple[bool, Optional[float]]:
        """
        Test if injected transition is recovered.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell_injected (ndarray): Spectrum with injected transition
            true_location (float): True transition location
            tolerance (float): Tolerance for match (multipoles)
            
        Returns:
            tuple: (recovered, detected_location)
        """
        # Detect transitions
        detected_ells, _ = self.detector.detect_and_analyze(ell, C_ell_injected, n_transitions=5)
        
        if len(detected_ells) == 0:
            return False, None
        
        # Check if any detected transition matches injected location
        distances = np.abs(detected_ells - true_location)
        min_distance = np.min(distances)
        
        if min_distance < tolerance:
            detected_idx = np.argmin(distances)
            return True, detected_ells[detected_idx]
        
        return False, None
    
    def amplitude_scan(self, ell: np.ndarray, C_ell_smooth: np.ndarray,
                      C_ell_err: np.ndarray,
                      test_location: float = 1500,
                      amplitudes: Optional[np.ndarray] = None,
                      n_trials_per_amplitude: int = 100) -> Dict[str, Any]:
        """
        Scan recovery rate vs injection amplitude.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell_smooth (ndarray): Smooth spectrum (no transitions)
            C_ell_err (ndarray): Uncertainties
            test_location (float): Where to inject test transition (default: 1500)
            amplitudes (ndarray, optional): Amplitudes to test (default: -0.01 to -0.10)
            n_trials_per_amplitude (int): Trials per amplitude (default: 100)
            
        Returns:
            dict: Sensitivity results
        """
        self.output.log_section_header("INJECTION SENSITIVITY ANALYSIS")
        self.output.log_message(f"Test location: ℓ = {test_location:.0f}")
        self.output.log_message(f"Trials per amplitude: {n_trials_per_amplitude}")
        self.output.log_message("")
        
        if amplitudes is None:
            # Test range from 1% to 10% drops
            amplitudes = -np.linspace(0.01, 0.10, 10)
        
        # Results storage
        recovery_rates = []
        mean_deviations = []
        
        for amplitude in amplitudes:
            recovered_count = 0
            deviations = []
            
            for trial in range(n_trials_per_amplitude):
                # Add noise
                C_ell_noisy = C_ell_smooth + np.random.normal(0, C_ell_err)
                
                # Inject transition
                C_ell_injected = self.inject_transition(ell, C_ell_noisy, test_location, amplitude)
                
                # Test recovery
                recovered, detected_loc = self.test_recovery(ell, C_ell_injected, test_location)
                
                if recovered:
                    recovered_count += 1
                    deviations.append(abs(detected_loc - test_location))
            
            recovery_rate = recovered_count / n_trials_per_amplitude
            mean_deviation = np.mean(deviations) if len(deviations) > 0 else np.nan
            
            recovery_rates.append(recovery_rate)
            mean_deviations.append(mean_deviation)
            
            self.output.log_message(f"Amplitude: {amplitude:.3f} | Recovery: {recovery_rate*100:.1f}% | "
                                  f"Mean deviation: {mean_deviation:.1f if not np.isnan(mean_deviation) else 'N/A'} multipoles")
        
        self.output.log_message("")
        
        # Find minimum detectable amplitude (50% recovery)
        recovery_rates_array = np.array(recovery_rates)
        if np.any(recovery_rates_array >= 0.5):
            idx_50 = np.where(recovery_rates_array >= 0.5)[0][0]
            min_detectable_amp = abs(amplitudes[idx_50])
            self.output.log_message(f"Minimum detectable amplitude (50% recovery): {min_detectable_amp:.3f}")
        else:
            min_detectable_amp = None
            self.output.log_message("Minimum detectable amplitude: Not reached in scan")
        
        # Find 80% power threshold
        if np.any(recovery_rates_array >= 0.8):
            idx_80 = np.where(recovery_rates_array >= 0.8)[0][0]
            amp_80 = abs(amplitudes[idx_80])
            self.output.log_message(f"80% power threshold: {amp_80:.3f}")
        else:
            amp_80 = None
            self.output.log_message("80% power threshold: Not reached in scan")
        
        self.output.log_message("")
        
        return {
            'test_location': float(test_location),
            'amplitudes': amplitudes.tolist(),
            'recovery_rates': recovery_rates,
            'mean_deviations': mean_deviations,
            'n_trials_per_amplitude': int(n_trials_per_amplitude),
            'min_detectable_amplitude': float(min_detectable_amp) if min_detectable_amp is not None else None,
            'amplitude_80_percent_power': float(amp_80) if amp_80 is not None else None
        }
    
    def test_multiple_locations(self, ell: np.ndarray, C_ell_smooth: np.ndarray,
                               C_ell_err: np.ndarray,
                               test_locations: Optional[List[float]] = None,
                               amplitude: float = -0.05,
                               n_trials: int = 100) -> Dict[str, Any]:
        """
        Test sensitivity at multiple multipole locations.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell_smooth (ndarray): Smooth spectrum
            C_ell_err (ndarray): Uncertainties
            test_locations (list, optional): Locations to test
            amplitude (float): Fixed injection amplitude (default: -0.05)
            n_trials (int): Trials per location (default: 100)
            
        Returns:
            dict: Location-dependent sensitivity
        """
        self.output.log_section_header("LOCATION-DEPENDENT SENSITIVITY")
        self.output.log_message(f"Injection amplitude: {amplitude:.3f}")
        self.output.log_message(f"Trials per location: {n_trials}")
        self.output.log_message("")
        
        if test_locations is None:
            # Test at quartiles
            test_locations = [ell[len(ell)//4], ell[len(ell)//2], ell[3*len(ell)//4]]
        
        results_by_location = []
        
        for location in test_locations:
            recovered_count = 0
            
            for trial in range(n_trials):
                # Add noise
                C_ell_noisy = C_ell_smooth + np.random.normal(0, C_ell_err)
                
                # Inject
                C_ell_injected = self.inject_transition(ell, C_ell_noisy, location, amplitude)
                
                # Test recovery
                recovered, _ = self.test_recovery(ell, C_ell_injected, location)
                
                if recovered:
                    recovered_count += 1
            
            recovery_rate = recovered_count / n_trials
            
            results_by_location.append({
                'location': float(location),
                'recovery_rate': float(recovery_rate),
                'n_trials': int(n_trials)
            })
            
            self.output.log_message(f"Location ℓ = {location:.0f}: Recovery = {recovery_rate*100:.1f}%")
        
        self.output.log_message("")
        
        return {
            'amplitude': float(amplitude),
            'results': results_by_location
        }
    
    def compare_to_observed(self, ell: np.ndarray, C_ell_smooth: np.ndarray,
                           C_ell_err: np.ndarray,
                           observed_transitions: np.ndarray,
                           observed_amplitudes: Optional[np.ndarray] = None,
                           n_trials: int = 100) -> Dict[str, Any]:
        """
        Compare observed transitions to detection sensitivity.
        
        Tests whether observed transitions are above detection threshold.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell_smooth (ndarray): Smooth spectrum
            C_ell_err (ndarray): Uncertainties
            observed_transitions (ndarray): Observed transition locations
            observed_amplitudes (ndarray, optional): Estimated amplitudes
            n_trials (int): Trials for sensitivity test (default: 100)
            
        Returns:
            dict: Comparison results
        """
        self.output.log_section_header("OBSERVED vs SENSITIVITY THRESHOLD")
        
        if observed_amplitudes is None:
            # Assume typical 5% amplitude
            observed_amplitudes = np.full(len(observed_transitions), -0.05)
        
        results = []
        
        for location, amplitude in zip(observed_transitions, observed_amplitudes):
            # Test sensitivity at this location and amplitude
            recovered_count = 0
            
            for trial in range(n_trials):
                C_ell_noisy = C_ell_smooth + np.random.normal(0, C_ell_err)
                C_ell_injected = self.inject_transition(ell, C_ell_noisy, location, amplitude)
                recovered, _ = self.test_recovery(ell, C_ell_injected, location)
                
                if recovered:
                    recovered_count += 1
            
            recovery_rate = recovered_count / n_trials
            
            # Interpretation
            if recovery_rate >= 0.8:
                interpretation = "Well above threshold (>80% power)"
            elif recovery_rate >= 0.5:
                interpretation = "Above threshold (>50% detection)"
            elif recovery_rate >= 0.2:
                interpretation = "Near threshold (20-50% detection)"
            else:
                interpretation = "Below threshold (<20% detection)"
            
            results.append({
                'location': float(location),
                'amplitude': float(amplitude),
                'recovery_rate': float(recovery_rate),
                'interpretation': interpretation
            })
            
            self.output.log_message(f"ℓ = {location:.0f}, amp = {amplitude:.3f}:")
            self.output.log_message(f"  Recovery: {recovery_rate*100:.1f}% - {interpretation}")
        
        self.output.log_message("")
        
        return {
            'n_trials': int(n_trials),
            'transitions': results
        }
