"""
Likelihood Model Module
=======================

Proper likelihood-based inference for CMB feature detection.

Replaces derivative peak-finding with rigorous statistical model comparison
using multivariate Gaussian likelihoods with full covariance matrices.

Classes:
    LikelihoodModel: Likelihood computation and model comparison

Paper reference: Statistical methodology
"""

import numpy as np
from scipy import optimize, stats
from typing import Dict, Any, List, Tuple, Optional, Callable

from .utils import OutputManager
from .covariance_matrix import CovarianceMatrix


class LikelihoodModel:
    """
    Likelihood-based detection of CMB features.
    
    Uses proper statistical inference:
    - Multivariate Gaussian likelihood with full covariance
    - Profile likelihood for parameter estimation
    - Wilks' theorem for significance testing
    - Model comparison via likelihood ratios
    
    Attributes:
        output (OutputManager): For logging
        cov_builder (CovarianceMatrix): Covariance construction
        
    Example:
        >>> likelihood = LikelihoodModel()
        >>> result = likelihood.compare_models(ell, C_ell, C_ell_err, cov)
    """
    
    def __init__(self, output: OutputManager = None):
        """
        Initialize LikelihoodModel.
        
        Parameters:
            output (OutputManager, optional): For logging
        """
        self.output = output if output is not None else OutputManager()
        self.cov_builder = CovarianceMatrix(output=self.output)
    
    def log_likelihood(self, data: np.ndarray, model: np.ndarray,
                      cov_inv: np.ndarray) -> float:
        """
        Compute log-likelihood for multivariate Gaussian.
        
        ln L = -0.5 * (data - model)ᵀ Cov⁻¹ (data - model) - 0.5 * ln|Cov| - (n/2) ln(2π)
        
        For model comparison, constant terms can be dropped.
        
        Parameters:
            data (ndarray): Observed data
            model (ndarray): Model prediction
            cov_inv (ndarray): Inverse covariance (precision matrix)
            
        Returns:
            float: Log-likelihood
        """
        residual = data - model
        chi2 = np.dot(residual, np.dot(cov_inv, residual))
        
        # For model comparison, we don't need the normalization
        # but include it for completeness
        n = len(data)
        log_det_cov = -np.linalg.slogdet(cov_inv)[1]  # log|Cov| = -log|Cov⁻¹|
        
        log_l = -0.5 * (chi2 + log_det_cov + n * np.log(2 * np.pi))
        
        return log_l
    
    def smooth_model(self, ell: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Smooth polynomial model (null hypothesis).
        
        Parameters:
            ell (ndarray): Multipole values
            params (ndarray): Polynomial coefficients
            
        Returns:
            ndarray: Smooth model prediction
        """
        return np.polyval(params, ell)
    
    def transition_model(self, ell: np.ndarray, smooth_params: np.ndarray,
                        transition_params: List[Tuple[float, float]]) -> np.ndarray:
        """
        Model with discrete transitions.
        
        Model = smooth(ℓ) × ∏_i [1 + amplitude_i × H(ℓ - ℓ_i)]
        where H is Heaviside step function
        
        Parameters:
            ell (ndarray): Multipole values
            smooth_params (ndarray): Polynomial coefficients for smooth part
            transition_params (list): [(location_1, amplitude_1), ...] for each transition
            
        Returns:
            ndarray: Model with transitions
        """
        # Start with smooth model
        model = self.smooth_model(ell, smooth_params)
        
        # Add step functions at transitions
        for ell_trans, amplitude in transition_params:
            step = np.where(ell > ell_trans, 1.0 + amplitude, 1.0)
            model = model * step
        
        # Ensure positivity
        model = np.maximum(model, 1e-20)
        
        return model
    
    def fit_smooth_model(self, ell: np.ndarray, C_ell: np.ndarray,
                        cov_inv: np.ndarray, poly_order: int = 9) -> Tuple[np.ndarray, float]:
        """
        Fit smooth polynomial model.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Observed spectrum
            cov_inv (ndarray): Inverse covariance
            poly_order (int): Polynomial order (default: 9)
            
        Returns:
            tuple: (best_fit_params, log_likelihood)
        """
        # Initial guess from simple polynomial fit
        init_params = np.polyfit(ell, C_ell, poly_order)
        
        # Objective function (negative log likelihood)
        def neg_log_l(params):
            model = self.smooth_model(ell, params)
            return -self.log_likelihood(C_ell, model, cov_inv)
        
        # Optimize
        result = optimize.minimize(neg_log_l, init_params, method='Powell')
        
        best_params = result.x
        best_log_l = -result.fun
        
        return best_params, best_log_l
    
    def fit_transition_model(self, ell: np.ndarray, C_ell: np.ndarray,
                            cov_inv: np.ndarray,
                            n_transitions: int,
                            init_locations: Optional[np.ndarray] = None,
                            poly_order: int = 9) -> Tuple[np.ndarray, List[Tuple[float, float]], float]:
        """
        Fit model with transitions.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Observed spectrum
            cov_inv (ndarray): Inverse covariance
            n_transitions (int): Number of transitions to fit
            init_locations (ndarray, optional): Initial transition locations
            poly_order (int): Polynomial order (default: 9)
            
        Returns:
            tuple: (smooth_params, transition_params, log_likelihood)
        """
        # Initial smooth fit
        smooth_params_init = np.polyfit(ell, C_ell, poly_order)
        
        # Initial transition locations (if not provided)
        if init_locations is None:
            # Space evenly in observed range
            ell_min, ell_max = ell[len(ell)//4], ell[3*len(ell)//4]
            init_locations = np.linspace(ell_min, ell_max, n_transitions)
        
        # Initial amplitudes (small negative steps)
        init_amplitudes = np.full(n_transitions, -0.05)
        
        # Pack parameters: [smooth_coeffs, trans_locs, trans_amps]
        init_params = np.concatenate([smooth_params_init, init_locations, init_amplitudes])
        
        # Bounds
        bounds = []
        # Smooth parameters: no bounds
        bounds.extend([(None, None)] * len(smooth_params_init))
        # Transition locations: within data range
        bounds.extend([(ell[10], ell[-10])] * n_transitions)
        # Amplitudes: -0.2 to +0.2 (20% steps)
        bounds.extend([(-0.2, 0.2)] * n_transitions)
        
        # Objective function
        def neg_log_l(params):
            n_smooth = poly_order + 1
            smooth_p = params[:n_smooth]
            trans_locs = params[n_smooth:n_smooth+n_transitions]
            trans_amps = params[n_smooth+n_transitions:]
            
            trans_params = list(zip(trans_locs, trans_amps))
            model = self.transition_model(ell, smooth_p, trans_params)
            
            return -self.log_likelihood(C_ell, model, cov_inv)
        
        # Optimize
        result = optimize.minimize(neg_log_l, init_params, method='L-BFGS-B',
                                  bounds=bounds)
        
        # Unpack results
        n_smooth = poly_order + 1
        smooth_params = result.x[:n_smooth]
        trans_locs = result.x[n_smooth:n_smooth+n_transitions]
        trans_amps = result.x[n_smooth+n_transitions:]
        
        transition_params = list(zip(trans_locs, trans_amps))
        best_log_l = -result.fun
        
        return smooth_params, transition_params, best_log_l
    
    def likelihood_ratio_test(self, log_l_null: float, log_l_alt: float,
                             df_diff: int) -> Tuple[float, float, float]:
        """
        Likelihood ratio test (Wilks' theorem).
        
        Test statistic: Λ = -2 ln(L_null / L_alt) = -2 (ln L_null - ln L_alt)
        Under null hypothesis, Λ ~ χ²(df_diff)
        
        Parameters:
            log_l_null (float): Log-likelihood of null model
            log_l_alt (float): Log-likelihood of alternative model
            df_diff (int): Difference in degrees of freedom
            
        Returns:
            tuple: (test_statistic, p_value, significance_sigma)
        """
        test_stat = -2 * (log_l_null - log_l_alt)
        
        # Ensure non-negative
        test_stat = max(test_stat, 0.0)
        
        # p-value from chi-squared distribution
        p_value = 1 - stats.chi2.cdf(test_stat, df_diff)
        
        # Convert to sigma
        from scipy.special import erfcinv
        if p_value > 0 and p_value < 1:
            significance_sigma = np.sqrt(2) * erfcinv(2 * p_value)
        elif p_value == 0:
            significance_sigma = np.inf
        else:
            significance_sigma = 0.0
        
        return test_stat, p_value, significance_sigma
    
    def compare_models(self, ell: np.ndarray, C_ell: np.ndarray,
                      C_ell_err: np.ndarray, cov: np.ndarray,
                      max_transitions: int = 4) -> Dict[str, Any]:
        """
        Compare models with varying numbers of transitions.
        
        Tests:
        - 0 transitions (smooth only)
        - 1 transition
        - 2 transitions
        - 3 transitions
        - etc. up to max_transitions
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Observed spectrum
            C_ell_err (ndarray): Uncertainties
            cov (ndarray): Covariance matrix
            max_transitions (int): Maximum transitions to test (default: 4)
            
        Returns:
            dict: Model comparison results
        """
        self.output.log_section_header("LIKELIHOOD-BASED MODEL COMPARISON")
        
        # Inverse covariance
        cov_inv = self.cov_builder.invert_covariance(cov)
        
        results = {
            'models': {},
            'likelihood_ratios': {},
            'best_model': None
        }
        
        # Fit smooth model (null hypothesis)
        self.output.log_message("Fitting smooth model (null hypothesis)...")
        smooth_params, log_l_smooth = self.fit_smooth_model(ell, C_ell, cov_inv)
        
        n_smooth_params = len(smooth_params)
        model_smooth = self.smooth_model(ell, smooth_params)
        chi2_smooth = np.dot(C_ell - model_smooth, np.dot(cov_inv, C_ell - model_smooth))
        
        results['models']['smooth'] = {
            'n_params': int(n_smooth_params),
            'n_transitions': 0,
            'log_likelihood': float(log_l_smooth),
            'chi2': float(chi2_smooth),
            'params': smooth_params.tolist()
        }
        
        self.output.log_message(f"  Log-likelihood: {log_l_smooth:.2f}")
        self.output.log_message(f"  χ²: {chi2_smooth:.1f}")
        self.output.log_message("")
        
        # Fit models with transitions
        best_log_l = log_l_smooth
        best_n_trans = 0
        
        for n_trans in range(1, max_transitions + 1):
            self.output.log_message(f"Fitting model with {n_trans} transition(s)...")
            
            try:
                smooth_p, trans_p, log_l = self.fit_transition_model(
                    ell, C_ell, cov_inv, n_trans
                )
                
                # Compute model and chi2
                model = self.transition_model(ell, smooth_p, trans_p)
                chi2 = np.dot(C_ell - model, np.dot(cov_inv, C_ell - model))
                
                # Degrees of freedom
                n_params = n_smooth_params + 2 * n_trans  # locations + amplitudes
                df_diff = 2 * n_trans
                
                # Likelihood ratio test vs smooth
                test_stat, p_value, sigma = self.likelihood_ratio_test(
                    log_l_smooth, log_l, df_diff
                )
                
                results['models'][f'{n_trans}_transitions'] = {
                    'n_params': int(n_params),
                    'n_transitions': int(n_trans),
                    'log_likelihood': float(log_l),
                    'chi2': float(chi2),
                    'transition_locations': [float(loc) for loc, _ in trans_p],
                    'transition_amplitudes': [float(amp) for _, amp in trans_p]
                }
                
                results['likelihood_ratios'][f'{n_trans}_vs_smooth'] = {
                    'test_statistic': float(test_stat),
                    'p_value': float(p_value),
                    'significance_sigma': float(sigma) if sigma < np.inf else None,
                    'df_diff': int(df_diff)
                }
                
                self.output.log_message(f"  Log-likelihood: {log_l:.2f}")
                self.output.log_message(f"  χ²: {chi2:.1f}")
                self.output.log_message(f"  vs smooth: Δχ² = {test_stat:.1f}, {sigma:.1f}σ")
                
                # Track best model
                if log_l > best_log_l:
                    best_log_l = log_l
                    best_n_trans = n_trans
                
                # Print transitions
                for i, (loc, amp) in enumerate(trans_p, 1):
                    self.output.log_message(f"    Transition {i}: ℓ={loc:.0f}, amplitude={amp:.3f}")
                
                self.output.log_message("")
                
            except Exception as e:
                self.output.log_message(f"  Fit failed: {e}")
                self.output.log_message("")
                continue
        
        # Determine best model
        results['best_model'] = {
            'n_transitions': int(best_n_trans),
            'model_name': 'smooth' if best_n_trans == 0 else f'{best_n_trans}_transitions',
            'log_likelihood': float(best_log_l)
        }
        
        self.output.log_message(f"Best model: {results['best_model']['model_name']}")
        self.output.log_message("")
        
        return results
    
    def profile_likelihood(self, ell: np.ndarray, C_ell: np.ndarray,
                          cov_inv: np.ndarray, transition_location: float,
                          poly_order: int = 9) -> Dict[str, Any]:
        """
        Compute profile likelihood for single transition location.
        
        Useful for confidence intervals on transition location.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Observed spectrum
            cov_inv (ndarray): Inverse covariance
            transition_location (float): Location to profile
            poly_order (int): Polynomial order
            
        Returns:
            dict: Profile likelihood results
        """
        # Fix transition location, optimize over other parameters
        smooth_params_init = np.polyfit(ell, C_ell, poly_order)
        init_amplitude = -0.05
        
        init_params = np.concatenate([smooth_params_init, [init_amplitude]])
        
        def neg_log_l(params):
            smooth_p = params[:poly_order+1]
            amp = params[-1]
            trans_p = [(transition_location, amp)]
            model = self.transition_model(ell, smooth_p, trans_p)
            return -self.log_likelihood(C_ell, model, cov_inv)
        
        result = optimize.minimize(neg_log_l, init_params, method='Powell')
        
        best_amplitude = result.x[-1]
        log_l = -result.fun
        
        return {
            'location': float(transition_location),
            'best_amplitude': float(best_amplitude),
            'log_likelihood': float(log_l)
        }
