"""
BAO Theory Predictions Module
==============================

Pure theoretical predictions for BAO scale using three physical mechanisms.
NO parameter fitting - all predictions from γ = H/ln(πc⁵/GℏH²).

Three mechanisms tested:
A. Time-dependent vacuum: Λ_eff(z) ∝ H²/[ln H]²
B. Viscosity damping: γ creates effective viscosity at recombination
C. Direct Friedmann: γ enters expansion equations directly

Classes:
    BAOTheoryPredictions: Calculate D_M/r_d predictions via three mechanisms

Usage:
    python main.py --bao

Paper reference: gamma_theoretical_derivation.tex
"""

import numpy as np
from typing import Dict, Any, Callable
from scipy.integrate import quad, odeint

from .utils import OutputManager
from .antiviscosity_mechanism import AntiViscosityMechanism
from .constants import (
    C, HBAR, G, H0, OMEGA_M, OMEGA_LAMBDA,
    T_PLANCK, RHO_PLANCK, QTEP_RATIO, Z_RECOMB, Z_DRAG
)

from typing import Optional


class BAOTheoryPredictions:
    """
    Predict BAO observables from first-principles γ theory.
    
    Three independent mechanisms for how γ affects BAO scale:
    1. Modified vacuum energy Λ_eff(z)
    2. Viscosity damping of sound horizon
    3. Direct modification of Friedmann equations
    
    Attributes:
        output (OutputManager): For logging
        
    Example:
        >>> predictor = BAOTheoryPredictions()
        >>> dm_rd = predictor.predict_mechanism_a([0.65, 0.84, 1.02])
    """
    
    def __init__(self, output: OutputManager = None):
        """
        Initialize BAOTheoryPredictions.
        
        Parameters:
            output (OutputManager, optional): For logging
        """
        self.output = output if output is not None else OutputManager()
        self.antiviscosity = AntiViscosityMechanism(output=self.output)
    
    def gamma_at_redshift(self, z: float) -> float:
        """Calculate γ at redshift z using rigorous formula."""
        H_z = H0 * np.sqrt(OMEGA_M * (1 + z)**3 + OMEGA_LAMBDA)
        arg = (np.pi * C**5) / (HBAR * G * H_z**2)
        return H_z / np.log(arg)
    
    def lambda_eff_at_redshift(self, z: float) -> float:
        """Calculate Λ_eff at redshift z."""
        gamma_z = self.gamma_at_redshift(z)
        gamma_t_P = gamma_z * T_PLANCK
        rho_Lambda_eff = RHO_PLANCK * (gamma_t_P**2) * QTEP_RATIO
        return (8 * np.pi * G * rho_Lambda_eff) / (C**2)
    
    # ========================================================================
    # MECHANISM A: Time-Dependent Vacuum Energy
    # ========================================================================
    
    def H_with_lambda_eff(self, z: float) -> float:
        """
        Hubble parameter with time-dependent Λ_eff(z).
        
        From gamma_theoretical_derivation.tex line 245:
        Λ_eff(z) ∝ H(z)²/[ln H(z)]²
        
        Parameters:
            z (float): Redshift
            
        Returns:
            float: Modified Hubble parameter in s⁻¹
        """
        # Calculate Λ_eff at this redshift
        Lambda_eff_z = self.lambda_eff_at_redshift(z)
        
        # Convert to energy density
        rho_Lambda_eff = Lambda_eff_z * C**2 / (8 * np.pi * G)
        
        # Modified Friedmann equation
        # H² = (8πG/3c²)[ρ_m(z) + ρ_Λ,eff(z)]
        rho_m_z = OMEGA_M * (3 * H0**2 * C**2 / (8 * np.pi * G)) * (1 + z)**3
        
        H_eff_squared = (8 * np.pi * G / (3 * C**2)) * (rho_m_z + rho_Lambda_eff)
        
        return np.sqrt(H_eff_squared)
    
    def comoving_distance_lambda_eff(self, z: float) -> float:
        """
        Comoving angular diameter distance with Λ_eff(z).
        
        D_M(z) = c ∫₀^z dz'/H_eff(z')
        
        Parameters:
            z (float): Redshift
            
        Returns:
            float: Comoving distance in Mpc
        """
        def integrand(z_prime):
            return C / self.H_with_lambda_eff(z_prime)
        
        # Integrate (result in meters)
        distance_m, _ = quad(integrand, 0, z, limit=100)
        
        # Convert to Mpc
        distance_Mpc = distance_m / 3.086e22
        
        return distance_Mpc
    
    def predict_mechanism_a(self, redshifts: np.ndarray) -> np.ndarray:
        """
        Mechanism A: Predict D_M/r_d using time-dependent Λ_eff.
        
        Physics:
        - Λ_eff(z) varies with redshift (not constant)
        - Modifies expansion history H(z)
        - Standard sound horizon r_s ≈ 147.5 Mpc
        - Modified D_M from altered expansion
        
        Parameters:
            redshifts (ndarray): Observation redshifts
            
        Returns:
            ndarray: Predicted D_M/r_d values
        """
        self.output.log_message("MECHANISM A: Time-Dependent Vacuum Energy")
        self.output.log_message("-" * 50)
        self.output.log_message("Λ_eff(z) ∝ H(z)²/[ln H(z)]²")
        self.output.log_message("")
        
        # Standard sound horizon (unmodified for this mechanism)
        r_s_standard = 147.5  # Mpc (Planck 2018 value)
        
        predictions = np.zeros(len(redshifts))
        
        for i, z in enumerate(redshifts):
            D_M = self.comoving_distance_lambda_eff(z)
            predictions[i] = D_M / r_s_standard
            
            self.output.log_message(f"  z={z:.2f}: D_M={D_M:.1f} Mpc, D_M/r_d={predictions[i]:.2f}")
        
        self.output.log_message("")
        
        return predictions
    
    # ========================================================================
    # MECHANISM B: Viscosity Damping
    # ========================================================================
    
    def sound_horizon_with_viscosity(self, alpha: float = 10.0) -> float:
        """
        Sound horizon modified by information-theoretic viscosity.
        
        Physics:
        - γ(z=1100) creates effective viscosity in baryon-photon fluid
        - ~10⁹ decoherence events per Hubble time
        - ~10⁶⁰ events per m³ accumulated over recombination
        - Reduces effective sound horizon
        
        r_s,eff = r_s,LCDM × [1 - α × γ/H]
        
        Parameters:
            alpha (float): Coupling coefficient from Silk damping (~5-15)
            
        Returns:
            float: Modified sound horizon in Mpc
        """
        # Standard sound horizon
        r_s_standard = 147.5  # Mpc
        
        # Information processing at recombination
        gamma_recomb = self.gamma_at_redshift(Z_RECOMB)
        H_recomb = H0 * np.sqrt(OMEGA_M * (1 + Z_RECOMB)**3 + OMEGA_LAMBDA)
        viscosity_factor = gamma_recomb / H_recomb
        
        # Modified sound horizon
        # α ~ 10: ~10⁹ events/Hubble time create measurable viscosity
        r_s_eff = r_s_standard * (1.0 - alpha * viscosity_factor)
        
        self.output.log_message(f"Viscosity calculation:")
        self.output.log_message(f"  γ(z=1100) = {gamma_recomb:.3e} s⁻¹")
        self.output.log_message(f"  H(z=1100) = {H_recomb:.3e} s⁻¹")
        self.output.log_message(f"  γ/H = {viscosity_factor:.6f}")
        self.output.log_message(f"  α (coupling) = {alpha:.1f}")
        self.output.log_message(f"  r_s,LCDM = {r_s_standard:.2f} Mpc")
        self.output.log_message(f"  r_s,eff = {r_s_eff:.2f} Mpc")
        self.output.log_message(f"  Correction: {(r_s_eff/r_s_standard - 1)*100:.2f}%")
        self.output.log_message("")
        
        return r_s_eff
    
    def comoving_distance_standard(self, z: float) -> float:
        """
        Standard ΛCDM comoving distance (no γ modification).
        
        Parameters:
            z (float): Redshift
            
        Returns:
            float: Comoving distance in Mpc
        """
        def integrand(z_prime):
            H_z = H0 * np.sqrt(OMEGA_M * (1 + z_prime)**3 + OMEGA_LAMBDA)
            return C / H_z
        
        distance_m, _ = quad(integrand, 0, z, limit=100)
        return distance_m / 3.086e22  # Convert to Mpc
    
    def predict_mechanism_b(self, redshifts: np.ndarray, alpha: float = 10.0) -> np.ndarray:
        """
        Mechanism B: Predict D_M/r_d with viscosity-damped sound horizon.
        
        Physics:
        - Information processing creates effective viscosity
        - Reduces sound horizon by ~(α × γ/H) percent
        - Standard expansion (ΛCDM) for D_M
        - Net effect: larger D_M/r_d ratios
        
        Parameters:
            redshifts (ndarray): Observation redshifts
            alpha (float): Viscosity coupling (default: 10)
            
        Returns:
            ndarray: Predicted D_M/r_d values
        """
        self.output.log_message("MECHANISM B: Viscosity Damping")
        self.output.log_message("-" * 50)
        self.output.log_message("γ creates effective viscosity at recombination")
        self.output.log_message("")
        
        # Modified sound horizon
        r_s_eff = self.sound_horizon_with_viscosity(alpha)
        
        predictions = np.zeros(len(redshifts))
        
        for i, z in enumerate(redshifts):
            D_M = self.comoving_distance_standard(z)
            predictions[i] = D_M / r_s_eff
            
            self.output.log_message(f"  z={z:.2f}: D_M={D_M:.1f} Mpc, D_M/r_d={predictions[i]:.2f}")
        
        self.output.log_message("")
        
        return predictions
    
    # ========================================================================
    # MECHANISM C: Direct Friedmann Modification
    # ========================================================================
    
    def H_with_gamma_friedmann(self, z: float) -> float:
        """
        Hubble parameter with γ directly in Friedmann equation.
        
        H²(z) = (8πG/3c²)[ρ_m(z) + ρ_Λ,eff(γ(z))]
        
        Where ρ_Λ,eff includes information processing constraint.
        
        Parameters:
            z (float): Redshift
            
        Returns:
            float: Modified Hubble parameter in s⁻¹
        """
        # Calculate γ at this redshift
        gamma_z = self.gamma_at_redshift(z)
        
        # Effective vacuum energy density from information processing
        gamma_t_P = gamma_z * T_PLANCK
        rho_Lambda_eff = RHO_PLANCK * (gamma_t_P**2) * QTEP_RATIO
        
        # Matter density
        rho_m_z = OMEGA_M * (3 * H0**2 * C**2 / (8 * np.pi * G)) * (1 + z)**3
        
        # Modified Friedmann
        H_squared = (8 * np.pi * G / (3 * C**2)) * (rho_m_z + rho_Lambda_eff)
        
        return np.sqrt(H_squared)
    
    def sound_horizon_with_gamma_friedmann(self) -> float:
        """
        Sound horizon calculated with γ-modified expansion.
        
        r_s = ∫_{z_drag}^{z_∞} c_s(z)/H_modified(z) dz
        
        Returns:
            float: Sound horizon in Mpc
        """
        # Sound speed in baryon-photon fluid
        # c_s² ≈ c²/3(1 + R_b) where R_b = 3ρ_b/4ρ_γ
        # Simplified: c_s ≈ c/√3 before recombination
        c_s = C / np.sqrt(3)
        
        # Integration limits (drag epoch to high redshift)
        z_min = Z_DRAG
        z_max = 10000  # Effectively infinity
        
        def integrand(z):
            return c_s / self.H_with_gamma_friedmann(z)
        
        r_s_m, _ = quad(integrand, z_min, z_max, limit=200)
        r_s_Mpc = r_s_m / 3.086e22
        
        return r_s_Mpc
    
    def comoving_distance_with_gamma(self, z: float) -> float:
        """
        Comoving distance with γ-modified expansion.
        
        D_M(z) = c ∫₀^z dz'/H_modified(z')
        
        Parameters:
            z (float): Redshift
            
        Returns:
            float: Comoving distance in Mpc
        """
        def integrand(z_prime):
            return C / self.H_with_gamma_friedmann(z_prime)
        
        distance_m, _ = quad(integrand, 0, z, limit=100)
        return distance_m / 3.086e22
    
    def predict_mechanism_c(self, redshifts: np.ndarray) -> np.ndarray:
        """
        Mechanism C: Predict D_M/r_d with full γ-modified Friedmann.
        
        Physics:
        - γ(z) determines ρ_Λ,eff(z) at all redshifts
        - Modifies both expansion and sound horizon
        - Self-consistent calculation
        - Most complete theoretical treatment
        
        Parameters:
            redshifts (ndarray): Observation redshifts
            
        Returns:
            ndarray: Predicted D_M/r_d values
        """
        self.output.log_message("MECHANISM C: Direct Friedmann Modification")
        self.output.log_message("-" * 50)
        self.output.log_message("H²(z) = (8πG/3c²)[ρ_m(z) + ρ_Λ,eff(γ(z))]")
        self.output.log_message("")
        
        # Calculate sound horizon with γ-modified expansion
        self.output.log_message("Calculating sound horizon with γ-modified H(z)...")
        r_s_gamma = self.sound_horizon_with_gamma_friedmann()
        self.output.log_message(f"  r_s (γ-modified) = {r_s_gamma:.2f} Mpc")
        self.output.log_message(f"  r_s (ΛCDM) = 147.5 Mpc")
        self.output.log_message(f"  Difference: {(r_s_gamma/147.5 - 1)*100:.2f}%")
        self.output.log_message("")
        
        predictions = np.zeros(len(redshifts))
        
        for i, z in enumerate(redshifts):
            D_M = self.comoving_distance_with_gamma(z)
            predictions[i] = D_M / r_s_gamma
            
            self.output.log_message(f"  z={z:.2f}: D_M={D_M:.1f} Mpc, D_M/r_d={predictions[i]:.2f}")
        
        self.output.log_message("")
        
        return predictions
    
    # ========================================================================
    # MECHANISM D: Anti-Viscosity (Quantum Measurement Framework)
    # ========================================================================
    
    def predict_mechanism_d_antiviscosity(self, redshifts: np.ndarray) -> np.ndarray:
        """
        Mechanism D: Quantum anti-viscosity (PARAMETER-FREE).
        
        Pure theoretical prediction with ZERO free parameters.
        
        Physics:
        - γ(z=1100) from holographic formula
        - α=-5.7 from quantum Zeno effect
        - r_s enhanced by measurement-induced superfluidity
        - D_M from standard expansion + enhanced r_s
        
        Parameters:
            redshifts (ndarray): Observation redshifts
            
        Returns:
            ndarray: Predicted D_M/r_d values
        """
        self.output.log_message("MECHANISM D: Quantum Anti-Viscosity (PARAMETER-FREE)")
        self.output.log_message("-" * 50)
        self.output.log_message("Measurement-induced superfluidity at recombination")
        self.output.log_message("")
        
        # Document quantum framework
        framework = self.antiviscosity.quantum_measurement_framework()
        
        # Pure theoretical prediction (no fitting!)
        predictions = self.antiviscosity.predict_bao_scale(redshifts, antiviscosity_coefficient=None)
        
        return predictions
    
    # ========================================================================
    # Unified Prediction Interface
    # ========================================================================
    
    def predict_all_mechanisms(self, redshifts: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate predictions from all three mechanisms.
        
        Parameters:
            redshifts (ndarray): Observation redshifts
            
        Returns:
            dict: Predictions from each mechanism
        """
        self.output.log_section_header("BAO THEORETICAL PREDICTIONS")
        self.output.log_message("")
        self.output.log_message("Calculating D_M/r_d from pure theory (no fitting)")
        self.output.log_message(f"Observation redshifts: {redshifts}")
        self.output.log_message("")
        self.output.log_message("=" * 70)
        self.output.log_message("")
        
        # Mechanism A
        pred_a = self.predict_mechanism_a(redshifts)
        
        # Mechanism B removed - was parameter fitting, not pure prediction
        
        # Mechanism C
        pred_c = self.predict_mechanism_c(redshifts)
        
        # Mechanism D: Anti-viscosity (parameter-free!)
        pred_d = self.predict_mechanism_d_antiviscosity(redshifts)
        
        # ΛCDM baseline for comparison
        self.output.log_message("ΛCDM BASELINE (for comparison)")
        self.output.log_message("-" * 50)
        r_s_lcdm = 147.5
        pred_lcdm = np.zeros(len(redshifts))
        for i, z in enumerate(redshifts):
            D_M = self.comoving_distance_standard(z)
            pred_lcdm[i] = D_M / r_s_lcdm
            self.output.log_message(f"  z={z:.2f}: D_M/r_d={pred_lcdm[i]:.2f}")
        self.output.log_message("")
        
        return {
            'mechanism_a_lambda_eff': pred_a,
            'mechanism_c_friedmann': pred_c,
            'mechanism_d_antiviscosity': pred_d,  # Parameter-free quantum prediction
            'lcdm_baseline': pred_lcdm,
            'redshifts': redshifts
        }

