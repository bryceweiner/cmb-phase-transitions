"""
Gamma Theoretical Analysis Module
==================================

Pure theoretical calculation of γ(z) and Λ_eff(z) from first principles.
NO parameter fitting - all values derived from fundamental constants.

Implements gamma_theoretical_derivation.tex equations:
- γ = H/ln(πc⁵/GℏH²)  [line 29, 159]
- Λ_eff(z) ∝ H(z)²/[ln H(z)]²  [line 245]

Classes:
    GammaTheoreticalAnalysis: Calculate γ and Λ at all redshifts

Usage:
    python main.py --gamma

Paper reference: gamma_theoretical_derivation.tex
"""

import numpy as np
import json
from typing import Dict, Any, List
from pathlib import Path

from .utils import OutputManager
from .constants import (
    C, HBAR, G, H0, OMEGA_M, OMEGA_LAMBDA,
    T_PLANCK, RHO_PLANCK, QTEP_RATIO, LAMBDA_OBS, Z_RECOMB
)


class GammaTheoreticalAnalysis:
    """
    Pure theoretical calculation of information processing rate.
    
    Calculates γ(z) and Λ_eff(z) from first principles without fitting.
    
    Attributes:
        output (OutputManager): For logging
        
    Example:
        >>> analysis = GammaTheoreticalAnalysis()
        >>> results = analysis.calculate_all()
        >>> print(f"γ at z=1100: {results['epochs'][1100]['gamma']:.3e} s⁻¹")
    """
    
    def __init__(self, output: OutputManager = None):
        """
        Initialize GammaTheoreticalAnalysis.
        
        Parameters:
            output (OutputManager, optional): For logging
        """
        self.output = output if output is not None else OutputManager()
    
    def gamma_at_redshift(self, z: float) -> float:
        """
        Calculate information processing rate at redshift z.
        
        Formula from gamma_theoretical_derivation.tex line 29, 159:
        γ = H/ln(πc⁵/GℏH²)
        
        Combines:
        - Bekenstein bound (holographic entropy)
        - Margolus-Levitin theorem (operational speed)
        - Addressing complexity (logarithmic suppression)
        
        Parameters:
            z (float): Redshift
            
        Returns:
            float: Information processing rate γ in s⁻¹
        """
        # Calculate H(z) from Friedmann equation
        H_z = H0 * np.sqrt(OMEGA_M * (1 + z)**3 + OMEGA_LAMBDA)
        
        # Theoretical γ formula
        arg = (np.pi * C**5) / (HBAR * G * H_z**2)
        ln_arg = np.log(arg)
        gamma = H_z / ln_arg
        
        return gamma
    
    def lambda_eff_at_redshift(self, z: float) -> float:
        """
        Calculate effective cosmological constant at redshift z.
        
        Formula from gamma_theoretical_derivation.tex line 245:
        Λ_eff(z) ∝ H(z)²/[ln H(z)]²
        
        Via energy density:
        ρ_Λ,eff = ρ_P × [γ(z)×t_P]² × QTEP_RATIO
        Λ_eff = (8πG/c²) × ρ_Λ,eff
        
        Parameters:
            z (float): Redshift
            
        Returns:
            float: Effective cosmological constant in m⁻²
        """
        # Calculate γ at this redshift
        gamma_z = self.gamma_at_redshift(z)
        
        # Dimensionless capacity
        gamma_t_P = gamma_z * T_PLANCK
        
        # Energy density scaling with QTEP ratio
        rho_Lambda_eff = RHO_PLANCK * (gamma_t_P**2) * QTEP_RATIO
        
        # Convert to Λ
        Lambda_eff = (8 * np.pi * G * rho_Lambda_eff) / (C**2)
        
        return Lambda_eff
    
    def calculate_at_key_epochs(self) -> Dict[str, Any]:
        """
        Calculate γ and Λ_eff at cosmologically significant epochs.
        
        Returns:
            dict: Results at each epoch
        """
        epochs = {
            'today': 0.0,
            'BAO_observations': 0.8,
            'reionization': 10.0,
            'recombination': 1100.0,
            'equality': 3402.0
        }
        
        results = {}
        
        self.output.log_section_header("GAMMA AT KEY COSMOLOGICAL EPOCHS")
        self.output.log_message("")
        self.output.log_message("Theoretical calculation from first principles:")
        self.output.log_message("  γ = H/ln(πc⁵/GℏH²)")
        self.output.log_message("  Λ_eff = (8πG/c²) × ρ_P × [γ×t_P]² × QTEP")
        self.output.log_message("")
        self.output.log_message(f"{'Epoch':<20} {'z':<8} {'H (s⁻¹)':<15} {'γ (s⁻¹)':<15} {'Λ_eff (m⁻²)':<15}")
        self.output.log_message("-" * 80)
        
        for name, z in epochs.items():
            H_z = H0 * np.sqrt(OMEGA_M * (1 + z)**3 + OMEGA_LAMBDA)
            gamma = self.gamma_at_redshift(z)
            lambda_eff = self.lambda_eff_at_redshift(z)
            
            self.output.log_message(
                f"{name:<20} {z:<8.1f} {H_z:<15.3e} {gamma:<15.3e} {lambda_eff:<15.3e}"
            )
            
            results[name] = {
                'z': float(z),
                'H': float(H_z),
                'gamma': float(gamma),
                'gamma_over_H': float(gamma / H_z),
                'lambda_eff': float(lambda_eff)
            }
        
        self.output.log_message("")
        
        return results
    
    def compare_to_observations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare predicted Λ_eff(z=0) to observed Λ.
        
        Parameters:
            results (dict): Results from calculate_at_key_epochs()
            
        Returns:
            dict: Comparison results
        """
        self.output.log_section_header("COMPARISON TO OBSERVATIONS")
        self.output.log_message("")
        
        Lambda_pred = results['today']['lambda_eff']
        ratio = Lambda_pred / LAMBDA_OBS
        orders_of_mag = np.log10(abs(ratio))
        
        self.output.log_message(f"Predicted Λ_eff(z=0): {Lambda_pred:.3e} m⁻²")
        self.output.log_message(f"Observed  Λ_obs:      {LAMBDA_OBS:.3e} m⁻²")
        self.output.log_message(f"Ratio (pred/obs):     {ratio:.3e}")
        self.output.log_message(f"Difference:           {abs(orders_of_mag):.1f} orders of magnitude")
        self.output.log_message("")
        
        if abs(orders_of_mag) < 2:
            verdict = "EXCELLENT - Within 2 orders of magnitude"
        elif abs(orders_of_mag) < 5:
            verdict = "GOOD - Within 5 orders of magnitude"
        elif abs(orders_of_mag) < 10:
            verdict = "ACCEPTABLE - Within 10 orders of magnitude"
        else:
            verdict = "POOR - More than 10 orders of magnitude off"
        
        self.output.log_message(f"Assessment: {verdict}")
        self.output.log_message("")
        
        return {
            'lambda_predicted': float(Lambda_pred),
            'lambda_observed': float(LAMBDA_OBS),
            'ratio': float(ratio),
            'log10_ratio': float(orders_of_mag),
            'verdict': verdict
        }
    
    def calculate_all(self, output_dir: str = "./results") -> Dict[str, Any]:
        """
        Complete theoretical gamma analysis.
        
        Parameters:
            output_dir (str): Output directory for results
            
        Returns:
            dict: Complete analysis results
        """
        self.output.log_message("=" * 70)
        self.output.log_message("THEORETICAL GAMMA AND LAMBDA ANALYSIS")
        self.output.log_message("=" * 70)
        self.output.log_message("")
        self.output.log_message("Reference: gamma_theoretical_derivation.tex")
        self.output.log_message("Pure theoretical prediction - NO parameter fitting")
        self.output.log_message("")
        
        # Calculate at key epochs
        epoch_results = self.calculate_at_key_epochs()
        
        # Compare to observations
        comparison = self.compare_to_observations(epoch_results)
        
        # Compile results
        results = {
            'methodology': 'pure_theoretical_prediction',
            'formula': 'γ = H/ln(πc⁵/GℏH²)',
            'reference': 'gamma_theoretical_derivation.tex',
            'epochs': epoch_results,
            'comparison_to_lambda_obs': comparison
        }
        
        # Save results
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_file = Path(output_dir) / "gamma_theoretical.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.output.log_message(f"Results saved to: {output_file}")
        self.output.log_message("")
        
        return results


def run_gamma_analysis(output_dir: str = "./results") -> Dict[str, Any]:
    """
    Main entry point for --gamma flag.
    
    Parameters:
        output_dir (str): Output directory
        
    Returns:
        dict: Analysis results
    """
    analysis = GammaTheoreticalAnalysis()
    return analysis.calculate_all(output_dir=output_dir)

