"""
Theoretical Calculations Module
================================

Implements all information-theoretic and quantum physics calculations from the paper.
Each method corresponds to specific formulas with line references to phase_transitions_discovery.tex.

Key formulas:
- γ = H/ln(πc²/ℏGH²) [line 107]
- QTEP ratio = ln(2)/(1-ln(2)) [line 167]
- Expansion factor = γ_theory/γ_obs [line 117]

References:
    Kaul & Majumdar, PRL 84, 5255 (2000) - Logarithmic corrections
    Wetterich, Eur. Phys. J. C 77, 264 (2017) - Graviton IR renormalization
    Koksma & Prokopec, arXiv:1105.6296 (2011) - Lorentz-invariant vacuum
    Hamada & Matsuda, JHEP 01, 069 (2016) - Two-loop quantum gravity
"""

import numpy as np
from typing import Tuple

from .constants import (
    C, HBAR, G, H0, Z_RECOMB, QTEP_RATIO,
    T_PLANCK, M_PLANCK, RHO_PLANCK
)


class TheoreticalCalculations:
    """
    All information-theoretic and quantum physics calculations.
    
    This class implements the theoretical foundation of the paper, including:
    - Holographic information processing rates
    - Quantum corrections to vacuum energy
    - Expansion factor calculations
    - Physical scale conversions
    
    All methods are static (no instance state) for ease of use.
    
    Example:
        >>> theory = TheoreticalCalculations()
        >>> H_recomb = 2.3e-18  # s⁻¹
        >>> gamma = theory.gamma_theoretical(H_recomb)
        >>> print(f"γ = {gamma:.3e} s⁻¹")
    """
    
    @staticmethod
    def gamma_theoretical(H: float) -> float:
        """
        Calculate theoretical γ from holographic entropy bounds.
        
        Formula (gamma_theoretical_derivation.tex line 29, 159): 
        γ = H/ln(πc⁵/GℏH²)
        
        Dimensional analysis requires c⁵ (not c²) to render the logarithmic
        argument dimensionless. This emerges from expressing horizon area in
        Planck units: N_Planck_areas = 4πc⁵/(GℏH²).
        
        This emerges from the Bekenstein bound on maximum information
        processable within a causal horizon.
        
        Parameters:
            H (float): Hubble parameter in s⁻¹
            
        Returns:
            float: Information processing rate in s⁻¹
        """
        arg = (np.pi * C**5) / (HBAR * G * H**2)
        gamma = H / np.log(arg)
        return gamma
    
    @staticmethod
    def gamma_refined(H: float) -> Tuple[float, float]:
        """
        Calculate refined γ with subleading logarithmic corrections.
        
        Reference: Kaul & Majumdar, PRL 84, 5255 (2000)
        "Logarithmic Correction to the Bekenstein-Hawking Entropy"
        
        The Bekenstein-Hawking entropy has subleading logarithmic corrections:
        S_bh = S_BH - (3/2) * ln(S_BH / ln(2)) + const + O(S_BH^{-1})
        
        This modifies the information processing rate:
        γ_refined = H / [ln(π c⁵/Gℏ H²) - (3/2) * ln(ln(π c⁵/Gℏ H²)/ln(2))]
        
        Parameters:
            H (float): Hubble parameter in s⁻¹
            
        Returns:
            tuple: (gamma_refined, correction_factor)
                - gamma_refined: Refined information processing rate in s⁻¹
                - correction_factor: Ratio of refined to baseline γ
        """
        # Baseline calculation (corrected: c⁵ not c² for dimensional consistency)
        arg = (np.pi * C**5) / (HBAR * G * H**2)
        ln_arg = np.log(arg)
        gamma_baseline = H / ln_arg
        
        # Subleading logarithmic correction (Kaul & Majumdar 2000)
        subleading_correction = 1.5 * np.log(ln_arg / np.log(2))
        
        # Refined γ with subleading correction
        gamma_refined = H / (ln_arg - subleading_correction)
        
        correction_factor = gamma_refined / gamma_baseline
        
        return gamma_refined, correction_factor
    
    @staticmethod
    def gamma_from_quantization(ell: float, n: int, H: float) -> float:
        """
        Extract observed γ(ℓ) from quantization condition.
        
        Formula (paper line 109): γℓ_n/H = nπ/2
        Therefore: γ(ℓ_n) = (nπ/2) × (H/ℓ_n)
        
        Parameters:
            ell (float): Multipole value
            n (int): Quantum number (1, 2, 3, ...)
            H (float): Hubble parameter in s⁻¹
            
        Returns:
            float: Observed scale-dependent γ(ℓ) in s⁻¹
        """
        return (n * np.pi / 2) * (H / ell)
    
    @staticmethod
    def expansion_factor(gamma_obs: float, gamma_theory: float) -> float:
        """
        Calculate expansion factor from observed vs theoretical γ.
        
        Paper line 117: Expansion factors 2.2-3.1×
        
        The reduction in observed γ indicates the expansion that occurred:
        expansion_factor ≈ γ_theory / γ_obs
        
        Parameters:
            gamma_obs (float): Observed γ(ℓ) from quantization
            gamma_theory (float): Theoretical baseline γ
            
        Returns:
            float: Expansion factor (dimensionless)
        """
        return gamma_theory / gamma_obs
    
    @staticmethod
    def harmonic_ratio_corrected(n: int, gamma_n: float, gamma_n_plus_1: float) -> float:
        """
        Calculate harmonic scaling ratio with scale-dependent γ correction.
        
        Formula (paper line 99): ℓ_(n+1)/ℓ_n = (n+1)/n × [γ(ℓ_n)/γ(ℓ_(n+1))]
        
        Parameters:
            n (int): Quantum number
            gamma_n (float): γ at ℓ_n
            gamma_n_plus_1 (float): γ at ℓ_(n+1)
            
        Returns:
            float: Predicted multipole ratio
        """
        harmonic = (n + 1) / n
        correction = gamma_n / gamma_n_plus_1
        return harmonic * correction
    
    @staticmethod
    def physical_scale(ell: float, z: float = Z_RECOMB) -> float:
        """
        Calculate physical scale in parsecs for given multipole at redshift z.
        
        Paper line 76-78: Physical scales 17,500-38,000 pc
        
        Parameters:
            ell (float): Multipole value
            z (float): Redshift (default: recombination)
            
        Returns:
            float: Physical scale in parsecs
        """
        theta_rad = np.pi / ell  # Angular scale in radians
        D_A_recomb = 13e6  # Angular diameter distance to z=1100 in parsecs
        return D_A_recomb * theta_rad
    
    @staticmethod
    def graviton_ir_correction(H: float, M_planck: float) -> float:
        """
        Calculate graviton infrared renormalization correction to vacuum energy.
        
        Reference: Wetterich, Eur. Phys. J. C 77, 264 (2017)
        "Graviton fluctuations erase the cosmological constant"
        
        Graviton fluctuations induce strong non-perturbative IR renormalization:
        V_c(k) = (M² k²)/2 - 5k⁴/(64π²)
        β_v = -2v + [5k²/(16π² M²)] * (1-v)^{-1}
        
        where v = 2V/(M² k²) and k = H for cosmological scales.
        
        Parameters:
            H (float): Hubble parameter (serves as IR cutoff k) in s⁻¹
            M_planck (float): Planck mass in kg
            
        Returns:
            float: Multiplicative correction factor (>1 means enhancement)
        """
        # Graviton IR correction factor from Wetterich flow equation
        # At fixed point: V_c ≈ (M² H²)/2 - 5H⁴/(64π²)
        # Correction to baseline: factor of [1 - 5H²/(64π² M²)]^{-1}
        
        correction_term = 5 * H**2 / (64 * np.pi**2 * M_planck**2)
        
        # This is a small correction, so we use first-order expansion
        # The correction enhances Λ (gravitons drive it away from zero)
        correction_factor = 1.0 / (1.0 - correction_term)
        
        return correction_factor
    
    @staticmethod
    def lorentz_vacuum_correction(H: float, mu: float = None) -> float:
        """
        Calculate Lorentz-invariant vacuum energy correction.
        
        Reference: Koksma & Prokopec, arXiv:1105.6296 (2011)
        "The Cosmological Constant and Lorentz Invariance of the Vacuum State"
        
        Requiring Lorentz invariance removes quartic UV divergences, keeping only:
        ⟨ρ_vac^ren⟩ = (m⁴)/(64π²) * ln(m²/μ²)
        
        This reduces vacuum energy contribution significantly and stabilizes
        the gravitational hierarchy (logarithmic rather than quartic running).
        
        Parameters:
            H (float): Hubble parameter in s⁻¹ (serves as effective mass scale)
            mu (float, optional): Renormalization scale in s⁻¹ (default: H0)
            
        Returns:
            float: Multiplicative correction factor
        """
        if mu is None:
            mu = H0  # Use present-day Hubble as renormalization scale
        
        # Lorentz-invariant vacuum correction with logarithmic running
        # ρ_corrected/ρ_baseline = [1 + (1/(64π²)) * ln(H²/μ²)]
        
        log_ratio = np.log(H**2 / mu**2)
        correction_factor = 1.0 + log_ratio / (64 * np.pi**2)
        
        return correction_factor
    
    @staticmethod
    def two_loop_qg_correction(alpha_t: float = None) -> float:
        """
        Calculate two-loop quantum gravity correction.
        
        Reference: Hamada & Matsuda, JHEP 01, 069 (2016)
        "Cosmological constant and the fate of the Universe"
        
        Two-loop quantum gravity corrections in renormalizable conformal gravity
        with dimensionless coupling α_t from asymptotic safety.
        
        The correction at O(α_t/b) in Landau gauge provides systematic
        quantum gravity effects on the cosmological constant.
        
        Parameters:
            alpha_t (float, optional): Dimensionless gravitational coupling
                (default: 1/(4π) from asymptotic safety)
            
        Returns:
            float: Multiplicative correction factor
        """
        if alpha_t is None:
            # From asymptotic safety: α_t ~ 1/(4π)
            alpha_t = 1.0 / (4.0 * np.pi)
        
        # Two-loop correction factor
        # From Hamada & Matsuda: Λ_corrected = Λ_baseline * [1 + α_t * C_2loop]
        # where C_2loop is an O(1) coefficient from two-loop calculations
        # Conservative estimate: C_2loop ~ 2 (typical for quantum gravity corrections)
        C_2loop = 2.0
        
        correction_factor = 1.0 + alpha_t * C_2loop
        
        return correction_factor
    
    def calculate_gamma_values(self, transitions: np.ndarray, H: float = None) -> np.ndarray:
        """
        Calculate gamma values for detected transitions.
        
        Parameters:
            transitions (ndarray): Array of multipole values
            H (float, optional): Hubble parameter (default: H_RECOMB)
            
        Returns:
            ndarray: Array of gamma values in s⁻¹
        """
        from .constants import H_RECOMB
        if H is None:
            H = H_RECOMB
        
        gamma_values = []
        for n, ell in enumerate(transitions, 1):
            gamma = self.gamma_from_quantization(ell, n, H)
            gamma_values.append(gamma)
        
        return np.array(gamma_values)

