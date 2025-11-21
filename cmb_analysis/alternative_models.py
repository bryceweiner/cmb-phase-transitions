"""
Alternative Models Comparison Module
=====================================

Test quantum anti-viscosity against other proposed BAO tension solutions.

Models tested:
- ΛCDM (baseline, k=0)
- w₀wₐCDM (varying dark energy, k=2)
- Early Dark Energy (k=2-3)
- Modified gravity f(R) (k=1-2)

Classes:
    AlternativeModels: Calculate predictions from alternative cosmologies
"""

import numpy as np
from typing import Dict, Any
from scipy.integrate import quad

from .utils import OutputManager
from .constants import C, H0, OMEGA_M, OMEGA_LAMBDA


class AlternativeModels:
    """
    Calculate BAO predictions from alternative cosmological models.
    
    Provides fair comparison to quantum anti-viscosity.
    
    Attributes:
        output (OutputManager): For logging
    """
    
    def __init__(self, output: OutputManager = None):
        """Initialize AlternativeModels."""
        self.output = output if output is not None else OutputManager()
    
    def H_lcdm(self, z: float) -> float:
        """Standard ΛCDM Hubble parameter."""
        return H0 * np.sqrt(OMEGA_M * (1 + z)**3 + OMEGA_LAMBDA)
    
    def H_w0wa(self, z: float, w0: float, wa: float) -> float:
        """
        w₀wₐCDM: Time-varying dark energy equation of state.
        
        w(z) = w₀ + wa × z/(1+z)
        
        Parameters:
            z (float): Redshift
            w0 (float): Present-day w
            wa (float): Evolution parameter
            
        Returns:
            float: Modified Hubble parameter
        """
        # Dark energy density evolution
        a = 1/(1+z)
        w_z = w0 + wa * (1 - a)
        
        # ρ_DE(a) = ρ_DE,0 × a^(-3(1+w₀+wa)) × exp(-3wa(1-a))
        exponent = -3 * (1 + w0 + wa) * np.log(a) - 3 * wa * (1 - a)
        rho_de_ratio = np.exp(exponent)
        
        # Modified Friedmann
        Omega_de_eff = OMEGA_LAMBDA * rho_de_ratio
        H_squared = H0**2 * (OMEGA_M * (1+z)**3 + Omega_de_eff)
        
        return np.sqrt(H_squared)
    
    def predict_lcdm(self, redshifts: np.ndarray) -> np.ndarray:
        """
        ΛCDM predictions for D_M/r_d.
        
        Parameters:
            redshifts (ndarray): Observation redshifts
            
        Returns:
            ndarray: ΛCDM predictions
        """
        r_s = 147.5  # Mpc
        predictions = np.zeros(len(redshifts))
        
        for i, z in enumerate(redshifts):
            def integrand(zp):
                return C / self.H_lcdm(zp)
            D_M_m, _ = quad(integrand, 0, z)
            D_M_Mpc = D_M_m / 3.086e22
            predictions[i] = D_M_Mpc / r_s
        
        return predictions
    
    def predict_w0wa(self, redshifts: np.ndarray, 
                     w0: float = -0.9, wa: float = 0.0) -> np.ndarray:
        """
        w₀wₐCDM predictions.
        
        Parameters:
            redshifts (ndarray): Observation redshifts
            w0 (float): Present w (default: -0.9)
            wa (float): Evolution (default: 0)
            
        Returns:
            ndarray: w₀wₐCDM predictions
        """
        r_s = 147.5  # Mpc (sound horizon assumed same)
        predictions = np.zeros(len(redshifts))
        
        for i, z in enumerate(redshifts):
            def integrand(zp):
                return C / self.H_w0wa(zp, w0, wa)
            D_M_m, _ = quad(integrand, 0, z)
            D_M_Mpc = D_M_m / 3.086e22
            predictions[i] = D_M_Mpc / r_s
        
        return predictions
    
    def H_ede(self, z: float, f_ede: float = 0.1, w_ede: float = -0.33) -> float:
        """
        Early Dark Energy Hubble parameter.
        
        EDE model (Poulin et al. 2019):
        Additional energy component at high-z that dilutes by today.
        
        Parameters:
            z (float): Redshift
            f_ede (float): EDE fraction at z~10⁴
            w_ede (float): EDE equation of state
            
        Returns:
            float: Modified Hubble parameter
        """
        # EDE density evolution
        # ρ_EDE ∝ (1+z)^(3(1+w_ede)) but only significant at high-z
        # For BAO (z<2), EDE is mostly diluted
        # Main effect is on r_s calculation
        
        # Simplified: EDE modifies early expansion, changes r_s
        # For low-z observations, use modified r_s with standard H(z)
        
        return self.H_lcdm(z)  # At observed z, EDE diluted
    
    def sound_horizon_ede(self, f_ede: float = 0.1, w_ede: float = -0.33) -> float:
        """
        Sound horizon with Early Dark Energy.
        
        EDE increases H at recombination → smaller r_s.
        
        Parameters:
            f_ede (float): EDE fraction
            w_ede (float): EDE equation of state
            
        Returns:
            float: Modified sound horizon in Mpc
        """
        # EDE increases expansion rate at recombination
        # r_s decreases by ~f_ede × factor
        # Typical: f_ede=0.1 → r_s decreases ~3-4%
        
        r_s_lcdm = 147.5
        ede_correction = 1 - 0.35 * f_ede  # Approximate scaling
        
        return r_s_lcdm * ede_correction
    
    def predict_ede(self, redshifts: np.ndarray,
                   f_ede: float = 0.1, w_ede: float = -0.33) -> np.ndarray:
        """
        EDE predictions for D_M/r_d.
        
        Parameters:
            redshifts (ndarray): Observation redshifts
            f_ede (float): EDE fraction (free parameter)
            w_ede (float): EDE equation of state (free parameter)
            
        Returns:
            ndarray: EDE predictions
        """
        r_s_ede = self.sound_horizon_ede(f_ede, w_ede)
        predictions = np.zeros(len(redshifts))
        
        for i, z in enumerate(redshifts):
            D_M_Mpc = self.comoving_distance_standard(z)
            predictions[i] = D_M_Mpc / r_s_ede
        
        return predictions
    
    def H_neff(self, z: float, delta_neff: float = 0.2) -> float:
        """
        Hubble parameter with extra neutrinos.
        
        N_eff = 3.046 + ΔN_eff modifies radiation density.
        
        Parameters:
            z (float): Redshift
            delta_neff (float): Extra relativistic species
            
        Returns:
            float: Modified Hubble parameter
        """
        # Extra radiation affects high-z expansion
        # Omega_rad increases by factor (N_eff / 3.046)
        # For z < 10, radiation is subdominant, small effect
        
        # Main effect is on r_s (radiation-matter equality shifts)
        
        return self.H_lcdm(z)  # At low-z, negligible
    
    def sound_horizon_neff(self, delta_neff: float = 0.2) -> float:
        """
        Sound horizon with extra neutrinos.
        
        More radiation → equality earlier → smaller r_s.
        
        Parameters:
            delta_neff (float): Extra species
            
        Returns:
            float: Modified sound horizon in Mpc
        """
        r_s_lcdm = 147.5
        
        # N_eff affects r_s by ~1-2% per ΔN_eff~0.1
        # r_s decreases with more radiation
        neff_correction = 1 - 0.015 * delta_neff  # ~1.5% per 0.1
        
        return r_s_lcdm * neff_correction
    
    def predict_neff(self, redshifts: np.ndarray,
                    delta_neff: float = 0.2) -> np.ndarray:
        """
        Extra neutrinos predictions.
        
        Parameters:
            redshifts (ndarray): Observation redshifts
            delta_neff (float): Extra species (free parameter)
            
        Returns:
            ndarray: N_eff predictions
        """
        r_s_neff = self.sound_horizon_neff(delta_neff)
        predictions = np.zeros(len(redshifts))
        
        for i, z in enumerate(redshifts):
            D_M_Mpc = self.comoving_distance_standard(z)
            predictions[i] = D_M_Mpc / r_s_neff
        
        return predictions
    
    def comoving_distance_standard(self, z: float) -> float:
        """
        Standard ΛCDM comoving distance.
        
        Parameters:
            z (float): Redshift
            
        Returns:
            float: Comoving distance in Mpc
        """
        def integrand(zp):
            return C / self.H_lcdm(zp)
        
        D_M_m, _ = quad(integrand, 0, z)
        return D_M_m / 3.086e22
    
    def compare_to_data(self, model_predictions: Dict[str, np.ndarray],
                       observations: np.ndarray,
                       covariance: np.ndarray) -> Dict[str, Any]:
        """
        Compare all models to data.
        
        Parameters:
            model_predictions (dict): {model_name: predictions}
            observations (ndarray): Observed values
            covariance (ndarray): Covariance matrix
            
        Returns:
            dict: Comparison for all models
        """
        cov_inv = np.linalg.inv(covariance)
        results = {}
        
        for name, predictions in model_predictions.items():
            residuals = observations - predictions
            chi2 = float(residuals @ cov_inv @ residuals)
            dof = len(observations)
            p_value = 1.0 - chi2_dist.cdf(chi2, dof)
            
            results[name] = {
                'chi2': chi2,
                'dof': dof,
                'p_value': p_value,
                'passes': p_value > 0.05
            }
        
        return results
    
    def compare_all_models_comprehensive(self, datasets: Dict,
                                        predictor_antiviscosity) -> Dict[str, Any]:
        """
        Comprehensive comparison of all alternative models.
        
        Tests: ΛCDM, w₀wₐCDM, EDE, N_eff, Quantum Anti-Viscosity
        
        Parameters:
            datasets (dict): All BAO datasets
            predictor_antiviscosity: Anti-viscosity predictor
            
        Returns:
            dict: Complete comparison with BIC/AIC for all models
        """
        self.output.log_section_header("COMPREHENSIVE MODEL COMPARISON")
        self.output.log_message("")
        self.output.log_message("Testing 5 models on all datasets:")
        self.output.log_message("  1. Quantum Anti-Viscosity (k=0)")
        self.output.log_message("  2. ΛCDM (k=0)")
        self.output.log_message("  3. w₀wₐCDM (k=2)")
        self.output.log_message("  4. Early Dark Energy (k=2)")
        self.output.log_message("  5. Extra Neutrinos (k=1)")
        self.output.log_message("")
        
        all_results = {}
        total_chi2 = {name: 0.0 for name in ['AntiViscosity', 'LCDM', 'w0wa', 'EDE', 'Neff']}
        total_dof = 0
        
        for dataset_name, dataset in datasets.items():
            # Skip QSO (known issue)
            if 'QSO' in dataset_name:
                continue
            
            z = dataset.redshifts
            
            # Generate predictions from each model
            pred_av = predictor_antiviscosity.predict_bao_scale(z, None)
            pred_lcdm = self.predict_lcdm(z)
            pred_w0wa = self.predict_w0wa(z, w0=-0.95, wa=-0.2)  # Best-fit-ish
            pred_ede = self.predict_ede(z, f_ede=0.09, w_ede=-0.33)
            pred_neff = self.predict_neff(z, delta_neff=0.15)
            
            # Calculate χ²
            cov = dataset.total_covariance(include_systematics=True)
            cov_inv = np.linalg.inv(cov)
            
            for model_name, pred in [('AntiViscosity', pred_av), ('LCDM', pred_lcdm),
                                     ('w0wa', pred_w0wa), ('EDE', pred_ede), ('Neff', pred_neff)]:
                residuals = dataset.values - pred
                chi2 = float(residuals @ cov_inv @ residuals)
                total_chi2[model_name] += chi2
            
            total_dof += dataset.dof
        
        # Calculate information criteria
        from .model_comparison_statistics import ModelComparisonStatistics
        comparator = ModelComparisonStatistics(output=self.output)
        
        models = {
            'QuantumAntiViscosity': {'chi2': total_chi2['AntiViscosity'], 'k_params': 0},
            'LCDM': {'chi2': total_chi2['LCDM'], 'k_params': 0},
            'w0waCDM': {'chi2': total_chi2['w0wa'], 'k_params': 2},
            'EarlyDarkEnergy': {'chi2': total_chi2['EDE'], 'k_params': 2},
            'ExtraNeutrinos': {'chi2': total_chi2['Neff'], 'k_params': 1}
        }
        
        comparison = comparator.compare_models(models, n_data=total_dof)
        full_results = comparator.summary_report(comparison)
        
        return full_results


