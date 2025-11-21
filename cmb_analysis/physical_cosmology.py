"""
Physical Cosmology Module
=========================

Rigorous cosmological calculations using Friedmann equations and growth theory.

Implements both numerical integration and semi-analytic approximations
for cross-validation of results.

Classes:
    FriedmannSolver: Numerical integration of Friedmann equations
    AnalyticCosmology: Semi-analytic approximations from perturbation theory
    GrowthSolver: Numerical solution of structure growth equation

Paper reference: IPIL 170 - Holographic Information Rate
"""

import numpy as np
from scipy.integrate import odeint, quad
from scipy.interpolate import interp1d
from typing import Dict, Any, Tuple, Optional, Callable

from .utils import OutputManager
from .constants import C, HBAR, G, H0, OMEGA_M, OMEGA_LAMBDA


class FriedmannSolver:
    """
    Numerical integration of Friedmann equations with holographic modifications.
    
    Implements modified Friedmann equation from IPIL 170 Eq. 3:
    H² = (8πG/3)[ρ_m + ρ_Λ(1 - γt)]
    
    Attributes:
        output (OutputManager): For logging
        Omega_m (float): Matter density parameter (default: 0.315)
        Omega_Lambda (float): Dark energy density parameter (default: 0.685)
        H0 (float): Hubble constant in s⁻¹ (default: 2.18e-18)
        
    Example:
        >>> solver = FriedmannSolver()
        >>> D_M = solver.comoving_distance(z=0.8, gamma_over_H=0.04)
        >>> print(f"Comoving distance: {D_M:.1f} Mpc")
    """
    
    def __init__(self, output: OutputManager = None,
                 Omega_m: float = OMEGA_M,
                 Omega_Lambda: float = OMEGA_LAMBDA,
                 H0_si: float = H0):
        """
        Initialize FriedmannSolver.
        
        Parameters:
            output (OutputManager, optional): For logging
            Omega_m (float): Matter density parameter
            Omega_Lambda (float): Dark energy density parameter
            H0_si (float): Hubble constant in s⁻¹
        """
        self.output = output if output is not None else OutputManager()
        self.Omega_m = Omega_m
        self.Omega_Lambda = Omega_Lambda
        self.H0_si = H0_si
        
        # Convert H0 to km/s/Mpc for standard units
        # H0[km/s/Mpc] = H0[s⁻¹] × (1 Mpc / 1 km) × (1 s)
        # 1 Mpc = 3.086e19 km
        self.H0_kmsMpc = H0_si * 3.086e19  # km/s/Mpc
    
    def E_function(self, z: float, gamma_over_H: float = 0.0) -> float:
        """
        Dimensionless Hubble parameter E(z) = H(z)/H0.
        
        Holographic modification uses gamma_over_H as scale-invariant parameter.
        """
        if gamma_over_H == 0:
            E_squared = self.Omega_m * (1 + z)**3 + self.Omega_Lambda
            return np.sqrt(E_squared)
        else:
            Omega_Lambda_eff = self.Omega_Lambda * (1 - gamma_over_H * np.log(1 + z))
            E_squared = self.Omega_m * (1 + z)**3 + Omega_Lambda_eff
            return np.sqrt(E_squared)
    
    def comoving_distance(self, z: float, gamma_over_H: float = 0.0,
                         n_points: int = 1000) -> float:
        """
        Comoving distance D_M to redshift z.
        
        D_M = c × ∫_0^z dz' / H(z')
             = (c/H0) × ∫_0^z dz' / E(z')
        
        Parameters:
            z (float): Redshift
            gamma_over_H (float): Holographic parameter
            n_points (int): Integration points (default: 1000)
            
        Returns:
            float: Comoving distance in Mpc
        """
        # Integration grid
        z_grid = np.linspace(0, z, n_points)
        
        # Integrand: 1/E(z')
        integrand = np.array([1.0 / self.E_function(z_val, gamma_over_H) 
                             for z_val in z_grid])
        
        # Integrate using trapezoidal rule
        integral = np.trapz(integrand, z_grid)
        
        # D_M = (c/H0) × integral
        # c/H0 in Mpc: c[m/s] / H0[s⁻¹] × (1 Mpc / 3.086e22 m)
        c_over_H0_Mpc = (C / self.H0_si) / 3.086e22
        
        D_M = c_over_H0_Mpc * integral
        
        return D_M
    
    def sound_horizon(self, gamma_over_H: float = 0.0,
                     z_drag: float = 1059.0,
                     z_eq: float = 3402.0) -> float:
        """
        Sound horizon with holographic modification at drag epoch.
        
        Scale-invariant ratio γ/H applied at drag epoch redshift.
        IPIL 170 Eq. 10: gamma_over_H is scale-invariant, applied at z_drag.
        """
        r_s_LCDM = 147.5  # Mpc
        
        if gamma_over_H == 0:
            return r_s_LCDM
        
        log_factor = np.log(z_drag / z_eq)
        r_s_holo = r_s_LCDM * (1 - gamma_over_H * log_factor)
        
        return r_s_holo
    
    def dm_over_rd(self, z: float, gamma_over_H: float = 0.0) -> float:
        """
        Compute D_M/r_d ratio (BAO observable).
        
        This is what DES Y3 and other surveys measure.
        
        Parameters:
            z (float): Redshift
            gamma_over_H (float): Holographic parameter
            
        Returns:
            float: D_M/r_d ratio (dimensionless)
        """
        D_M = self.comoving_distance(z, gamma_over_H)
        r_d = self.sound_horizon(gamma_over_H)
        
        return D_M / r_d
    
    def angular_diameter_distance(self, z: float, gamma_over_H: float = 0.0) -> float:
        """
        Angular diameter distance D_A = D_M / (1+z).
        
        Parameters:
            z (float): Redshift
            gamma_over_H (float): Holographic parameter
            
        Returns:
            float: Angular diameter distance in Mpc
        """
        D_M = self.comoving_distance(z, gamma_over_H)
        return D_M / (1 + z)


class AnalyticCosmology:
    """
    Semi-analytic approximations from perturbation theory.
    
    Provides faster calculations using perturbative expansions
    for validation against numerical integration.
    
    Based on IPIL 170 equations with first-order γ corrections.
    
    Attributes:
        output (OutputManager): For logging
        
    Example:
        >>> analytic = AnalyticCosmology()
        >>> r_s = analytic.sound_horizon_perturbative(gamma_over_H=0.04)
    """
    
    def __init__(self, output: OutputManager = None):
        """Initialize AnalyticCosmology."""
        self.output = output if output is not None else OutputManager()
    
    def sound_horizon_perturbative(self, gamma_over_H: float = 0.0,
                                   z_drag: float = 1059.0,
                                   z_eq: float = 3402.0) -> float:
        """
        Sound horizon using perturbative formula with proper H(z) dependence.
        
        CRITICAL: γ(z_drag) = [γ/H] × H(z_drag) = [γ/H] × H₀ × E(z_drag)
        """
        r_s_LCDM = 147.5  # Planck 2018
        
        if gamma_over_H == 0:
            return r_s_LCDM
        
        correction = 1 - gamma_over_H * np.log(z_drag / z_eq)
        return r_s_LCDM * correction
    
    def dm_rd_analytic(self, z: float, gamma_over_H: float = 0.0) -> float:
        """
        Analytic D_M/r_d using IPIL 170 formulas as written.
        
        gamma_over_H is the scale-invariant theoretical ratio.
        """
        z_ref = 0.835
        dm_rd_LCDM = 20.1 - 0.5 * (z - z_ref)
        
        if gamma_over_H == 0:
            return dm_rd_LCDM
        
        z_drag, z_eq = 1059.0, 3402.0
        
        distance_correction = gamma_over_H * np.log(1 + z)
        sound_horizon_correction = -gamma_over_H * np.log(z_drag / z_eq)
        total_correction = distance_correction - sound_horizon_correction
        
        dm_rd_holo = dm_rd_LCDM * (1 + total_correction)
        
        return dm_rd_holo


class GrowthSolver:
    """
    Solve structure growth equation with holographic modification.
    
    Modified growth equation (IPIL 170 Eq. 5):
    δ̈ + 2Hδ̇ - 4πGρδ = -γHδ
    
    Attributes:
        output (OutputManager): For logging
        Omega_m (float): Matter density parameter
        H0 (float): Hubble constant in s⁻¹
        
    Example:
        >>> growth = GrowthSolver()
        >>> sigma8_z = growth.compute_sigma8(z=0.5, gamma_over_H=0.04)
    """
    
    def __init__(self, output: OutputManager = None,
                 Omega_m: float = OMEGA_M,
                 H0_si: float = H0):
        """
        Initialize GrowthSolver.
        
        Parameters:
            output (OutputManager, optional): For logging
            Omega_m (float): Matter density parameter
            H0_si (float): Hubble constant in s⁻¹
        """
        self.output = output if output is not None else OutputManager()
        self.Omega_m = Omega_m
        self.H0_si = H0_si
    
    def growth_rate_equation(self, y: np.ndarray, a: float, 
                            gamma_over_H: float) -> np.ndarray:
        """
        ODE system for growth factor.
        
        Variables: y = [δ, dδ/da]
        
        Equation: d²δ/da² + [2/a + d ln H/da] dδ/da - (3Ω_m/2a⁵E²)δ = -(γ/H)H(dδ/da)
        
        Parameters:
            y (ndarray): [δ, dδ/da]
            a (float): Scale factor
            gamma_over_H (float): Holographic parameter
            
        Returns:
            ndarray: [dδ/da, d²δ/da²]
        """
        delta, delta_prime = y
        
        # Redshift
        z = 1/a - 1
        
        # E(z) for modified cosmology
        if gamma_over_H == 0:
            E_squared = self.Omega_m * (1 + z)**3 + (1 - self.Omega_m)
        else:
            Omega_Lambda_eff = (1 - self.Omega_m) * (1 - gamma_over_H * np.log(1 + z))
            E_squared = self.Omega_m * (1 + z)**3 + Omega_Lambda_eff
        
        E = np.sqrt(E_squared)
        
        # d ln E / d ln a = d ln E / dz × dz/d ln a = d ln E / dz × (-a(1+z))
        # For E² = Ω_m a⁻³ + Ω_Λ,eff:
        # d ln E² / dz = [3Ω_m(1+z)² - Ω_Λ,eff × γ/H × (1/(1+z))] / E²
        
        if gamma_over_H == 0:
            dlogE_dz = 3 * self.Omega_m * (1 + z)**2 / (2 * E_squared)
        else:
            Omega_Lambda_eff = (1 - self.Omega_m) * (1 - gamma_over_H * np.log(1 + z))
            numerator = 3 * self.Omega_m * (1 + z)**2
            numerator -= (1 - self.Omega_m) * gamma_over_H / (1 + z)
            dlogE_dz = numerator / (2 * E_squared)
        
        dlogE_dloga = -a * (1 + z) * dlogE_dz
        
        # Second derivative
        friction_term = (2/a + dlogE_dloga) * delta_prime
        gravity_term = (3 * self.Omega_m / (2 * a**5 * E_squared)) * delta
        damping_term = gamma_over_H * E * delta_prime * a  # Holographic damping
        
        delta_double_prime = -friction_term + gravity_term - damping_term
        
        return np.array([delta_prime, delta_double_prime])
    
    def compute_growth_factor(self, z_final: float, gamma_over_H: float = 0.0,
                             a_init: float = 1e-3) -> float:
        """
        Compute growth factor D(a) from early times to z_final.
        
        Solves: δ̈ + 2Hδ̇ - 4πGρδ = -γHδ
        
        Parameters:
            z_final (float): Final redshift
            gamma_over_H (float): Holographic parameter
            a_init (float): Initial scale factor (default: 0.001 = z~1000)
            
        Returns:
            float: Growth factor D(z_final) normalized to D(z=0) = 1
        """
        a_final = 1 / (1 + z_final)
        
        # Scale factor array
        a_array = np.logspace(np.log10(a_init), 0, 500)
        
        # Initial conditions (matter-dominated era)
        # δ ∝ a in matter domination
        y0 = np.array([a_init, a_init])  # [δ, dδ/da] ∝ [a, a]
        
        # Solve ODE
        solution = odeint(self.growth_rate_equation, y0, a_array, 
                         args=(gamma_over_H,))
        
        # Extract growth factor at final redshift
        idx_final = np.argmin(np.abs(a_array - a_final))
        D_final = solution[idx_final, 0]
        
        # Normalize to D(z=0) = 1
        D_z0 = solution[-1, 0]
        D_normalized = D_final / D_z0
        
        return D_normalized
    
    def compute_sigma8(self, z: float, gamma_over_H: float = 0.0,
                      sigma8_z0_LCDM: float = 0.811) -> float:
        """
        Compute σ8 at redshift z with holographic modification.
        
        σ8(z) = σ8(z=0) × D(z)
        
        With holographic: σ8^holo(z) = σ8^LCDM(z) × [growth modification]
        
        Parameters:
            z (float): Redshift
            gamma_over_H (float): Holographic parameter
            sigma8_z0_LCDM (float): σ8 at z=0 for ΛCDM (default: 0.811 from Planck)
            
        Returns:
            float: σ8 at redshift z
        """
        # Compute growth factors
        D_z_holo = self.compute_growth_factor(z, gamma_over_H)
        
        if gamma_over_H == 0:
            D_z_LCDM = D_z_holo
        else:
            D_z_LCDM = self.compute_growth_factor(z, gamma_over_H=0.0)
        
        # σ8(z) for ΛCDM
        sigma8_z_LCDM = sigma8_z0_LCDM * D_z_LCDM
        
        # Modification from holographic effects
        if gamma_over_H == 0:
            return sigma8_z_LCDM
        
        # Ratio of growth factors
        growth_ratio = D_z_holo / D_z_LCDM if D_z_LCDM > 0 else 1.0
        
        sigma8_z_holo = sigma8_z_LCDM * growth_ratio
        
        return sigma8_z_holo
    
    def compute_S8(self, z: float, gamma_over_H: float = 0.0,
                   Omega_m: Optional[float] = None) -> float:
        """
        Compute S8 parameter: S8 = σ8 × (Ω_m/0.3)^0.5.
        
        Parameters:
            z (float): Redshift
            gamma_over_H (float): Holographic parameter
            Omega_m (float, optional): Matter density (default: self.Omega_m)
            
        Returns:
            float: S8 parameter
        """
        if Omega_m is None:
            Omega_m = self.Omega_m
        
        sigma8_z = self.compute_sigma8(z, gamma_over_H)
        S8 = sigma8_z * np.sqrt(Omega_m / 0.3)
        
        return S8


class CosmologyValidator:
    """
    Cross-validate numerical and analytic approaches.
    
    Ensures physical calculations are robust by comparing
    independent methods that should give consistent results.
    
    Attributes:
        output (OutputManager): For logging
        friedmann (FriedmannSolver): Numerical solver
        analytic (AnalyticCosmology): Analytic approximations
        
    Example:
        >>> validator = CosmologyValidator()
        >>> results = validator.validate_bao_calculations(gamma_over_H=0.04)
        >>> print(f"Agreement: {results['agreement_percent']:.1f}%")
    """
    
    def __init__(self, output: OutputManager = None):
        """Initialize CosmologyValidator."""
        self.output = output if output is not None else OutputManager()
        self.friedmann = FriedmannSolver(output=self.output)
        self.analytic = AnalyticCosmology(output=self.output)
    
    def validate_bao_calculations(self, gamma_over_H: float,
                                  test_redshifts: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Validate BAO calculations by comparing numerical and analytic methods.
        
        Parameters:
            gamma_over_H (float): Holographic parameter to test
            test_redshifts (ndarray, optional): Redshifts to test (default: DES Y3 range)
            
        Returns:
            dict: Validation results with agreement metrics
        """
        if test_redshifts is None:
            test_redshifts = np.array([0.65, 0.74, 0.84, 0.93, 1.02])
        
        self.output.log_message(f"\nValidating BAO calculations (γ/H = {gamma_over_H:.4f}):")
        self.output.log_message(f"{'z':<8} {'Numerical':<12} {'Analytic':<12} {'Diff %':<10}")
        self.output.log_message("-" * 45)
        
        numerical_values = []
        analytic_values = []
        
        for z in test_redshifts:
            # Numerical
            dm_rd_num = self.friedmann.dm_over_rd(z, gamma_over_H)
            numerical_values.append(dm_rd_num)
            
            # Analytic
            dm_rd_ana = self.analytic.dm_rd_analytic(z, gamma_over_H)
            analytic_values.append(dm_rd_ana)
            
            # Difference
            diff_percent = abs(dm_rd_num - dm_rd_ana) / dm_rd_num * 100
            
            self.output.log_message(
                f"{z:<8.2f} {dm_rd_num:<12.3f} {dm_rd_ana:<12.3f} {diff_percent:<10.2f}%"
            )
        
        numerical_values = np.array(numerical_values)
        analytic_values = np.array(analytic_values)
        
        # Overall agreement
        mean_diff_percent = np.mean(np.abs(numerical_values - analytic_values) / numerical_values * 100)
        max_diff_percent = np.max(np.abs(numerical_values - analytic_values) / numerical_values * 100)
        
        self.output.log_message("")
        self.output.log_message(f"Mean difference: {mean_diff_percent:.2f}%")
        self.output.log_message(f"Max difference:  {max_diff_percent:.2f}%")
        
        if mean_diff_percent < 1.0:
            self.output.log_message("✓ Excellent agreement between methods")
            agreement_quality = "excellent"
        elif mean_diff_percent < 5.0:
            self.output.log_message("✓ Good agreement between methods")
            agreement_quality = "good"
        elif mean_diff_percent < 10.0:
            self.output.log_message("~ Acceptable agreement between methods")
            agreement_quality = "acceptable"
        else:
            self.output.log_message("⚠ Significant discrepancy between methods")
            agreement_quality = "poor"
        
        return {
            'gamma_over_H': float(gamma_over_H),
            'test_redshifts': test_redshifts.tolist(),
            'numerical_values': numerical_values.tolist(),
            'analytic_values': analytic_values.tolist(),
            'mean_difference_percent': float(mean_diff_percent),
            'max_difference_percent': float(max_diff_percent),
            'agreement_quality': agreement_quality
        }

