"""
MCMC Parameter Estimation Module
=================================

Bayesian parameter estimation for phase transition locations and amplitudes
using Markov Chain Monte Carlo sampling.

Classes:
    MCMCAnalyzer: Bayesian parameter estimation via MCMC

Uses emcee (Goodman & Weare 2010) for affine-invariant ensemble sampling.

Paper reference: Statistical validation methodology
"""

import numpy as np
import math
from typing import Dict, Any, List, Tuple, Optional, Callable
from scipy.stats import norm, uniform
from scipy.interpolate import interp1d
import warnings

from .utils import OutputManager

# Check for emcee availability
try:
    import emcee
    EMCEE_AVAILABLE = True
except ImportError:
    EMCEE_AVAILABLE = False
    warnings.warn("emcee not available. Install with: pip install emcee")


class MCMCAnalyzer:
    """
    Bayesian parameter estimation for CMB phase transitions using MCMC.
    
    Samples posterior distributions of transition locations, amplitudes,
    and other model parameters to provide robust uncertainty estimates
    and model comparison via Bayes factors.
    
    Attributes:
        output (OutputManager): For logging
        ell (ndarray): Multipole values
        C_ell (ndarray): Power spectrum
        C_ell_err (ndarray): Uncertainties
        
    Example:
        >>> analyzer = MCMCAnalyzer(ell, C_ell, C_ell_err)
        >>> results = analyzer.run_mcmc(n_transitions=3)
        >>> print(f"Posterior mean: {results['posterior_means']}")
    """
    
    def __init__(self, ell: np.ndarray, C_ell: np.ndarray, 
                 C_ell_err: np.ndarray, output: OutputManager = None):
        """
        Initialize MCMC analyzer.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Observed power spectrum
            C_ell_err (ndarray): Uncertainties
            output (OutputManager, optional): For logging
        """
        self.output = output if output is not None else OutputManager()
        self.ell = ell
        self.C_ell = C_ell
        self.C_ell_err = C_ell_err
        
        if not EMCEE_AVAILABLE:
            raise ImportError("emcee is required for MCMC analysis. Install with: pip install emcee")
    
    def piecewise_model(self, ell: np.ndarray, transitions: np.ndarray,
                       amplitudes: np.ndarray, baseline: float) -> np.ndarray:
        """
        Piecewise linear model with discontinuities at transitions.
        
        Parameters:
            ell (ndarray): Multipole values
            transitions (ndarray): Transition locations
            amplitudes (ndarray): Jump amplitudes at each transition
            baseline (float): Baseline level
            
        Returns:
            ndarray: Model power spectrum
        """
        model = np.full_like(ell, baseline, dtype=float)
        
        # Sort transitions
        sorted_idx = np.argsort(transitions)
        trans_sorted = transitions[sorted_idx]
        amps_sorted = amplitudes[sorted_idx]
        
        # Apply jumps
        cumulative_amp = 0.0
        for trans, amp in zip(trans_sorted, amps_sorted):
            mask = ell >= trans
            cumulative_amp += amp
            model[mask] += cumulative_amp
        
        return model
    
    def smooth_model(self, ell: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """
        Smooth polynomial model (null hypothesis).
        
        Parameters:
            ell (ndarray): Multipole values
            coeffs (ndarray): Polynomial coefficients (highest order first)
            
        Returns:
            ndarray: Model power spectrum
        """
        return np.polyval(coeffs, ell)
    
    def log_likelihood_discontinuous(self, params: np.ndarray, 
                                    n_transitions: int) -> float:
        """
        Log-likelihood for discontinuous model.
        
        Parameters:
            params (ndarray): [transitions..., amplitudes..., baseline]
            n_transitions (int): Number of transitions
            
        Returns:
            float: Log-likelihood value
        """
        # Unpack parameters
        transitions = params[:n_transitions]
        amplitudes = params[n_transitions:2*n_transitions]
        baseline = params[2*n_transitions]
        
        # Compute model
        model = self.piecewise_model(self.ell, transitions, amplitudes, baseline)
        
        # Chi-squared
        chi2 = np.sum(((self.C_ell - model) / self.C_ell_err)**2)
        
        return -0.5 * chi2
    
    def log_prior_discontinuous(self, params: np.ndarray,
                               n_transitions: int) -> float:
        """
        Log-prior for discontinuous model.
        
        Priors:
        - Transitions: Uniform over ell range, ordered
        - Amplitudes: Normal(0, C_ell.std())
        - Baseline: Normal(C_ell.mean(), C_ell.std())
        
        Parameters:
            params (ndarray): Model parameters
            n_transitions (int): Number of transitions
            
        Returns:
            float: Log-prior probability
        """
        transitions = params[:n_transitions]
        amplitudes = params[n_transitions:2*n_transitions]
        baseline = params[2*n_transitions]
        
        # Check bounds
        ell_min, ell_max = self.ell[0], self.ell[-1]
        
        # Transitions must be in range and ordered
        if not all(ell_min <= t <= ell_max for t in transitions):
            return -np.inf
        if not all(transitions[i] < transitions[i+1] for i in range(len(transitions)-1)):
            return -np.inf
        
        # Uniform prior on transitions (ordered)
        # For ordered uniform, prior density is n!/(ell_max - ell_min)^n
        log_prior = np.log(math.factorial(n_transitions)) - n_transitions * np.log(ell_max - ell_min)
        
        # Gaussian prior on amplitudes (mean=0, std=typical C_ell scale)
        amp_scale = np.std(self.C_ell)
        for amp in amplitudes:
            log_prior += norm.logpdf(amp, loc=0, scale=amp_scale)
        
        # Gaussian prior on baseline
        baseline_mean = np.mean(self.C_ell)
        baseline_std = np.std(self.C_ell)
        log_prior += norm.logpdf(baseline, loc=baseline_mean, scale=baseline_std)
        
        return log_prior
    
    def log_posterior_discontinuous(self, params: np.ndarray,
                                   n_transitions: int) -> float:
        """
        Log-posterior = log-likelihood + log-prior.
        
        Parameters:
            params (ndarray): Model parameters
            n_transitions (int): Number of transitions
            
        Returns:
            float: Log-posterior probability
        """
        lp = self.log_prior_discontinuous(params, n_transitions)
        if not np.isfinite(lp):
            return -np.inf
        
        ll = self.log_likelihood_discontinuous(params, n_transitions)
        if not np.isfinite(ll):
            return -np.inf
        
        return lp + ll
    
    def initialize_walkers(self, n_transitions: int, n_walkers: int,
                          initial_transitions: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Initialize MCMC walker positions.
        
        Parameters:
            n_transitions (int): Number of transitions
            n_walkers (int): Number of MCMC walkers
            initial_transitions (ndarray, optional): Initial guesses for transitions
            
        Returns:
            ndarray: Initial positions (n_walkers x n_params)
        """
        n_params = 2 * n_transitions + 1  # transitions + amplitudes + baseline
        
        # Initialize near maximum likelihood if provided
        if initial_transitions is not None and len(initial_transitions) == n_transitions:
            pos = []
            for _ in range(n_walkers):
                # Perturb transitions
                trans_init = initial_transitions + np.random.randn(n_transitions) * 50
                trans_init = np.sort(trans_init)  # Keep ordered
                
                # Initialize amplitudes near zero with small scatter
                amp_scale = np.std(self.C_ell) * 0.1
                amps_init = np.random.randn(n_transitions) * amp_scale
                
                # Initialize baseline near mean
                baseline_init = np.mean(self.C_ell) + np.random.randn() * np.std(self.C_ell) * 0.1
                
                pos.append(np.concatenate([trans_init, amps_init, [baseline_init]]))
            
            return np.array(pos)
        else:
            # Random initialization
            ell_min, ell_max = self.ell[0], self.ell[-1]
            pos = []
            
            for _ in range(n_walkers):
                # Random ordered transitions
                trans_init = np.sort(np.random.uniform(ell_min, ell_max, n_transitions))
                
                # Random amplitudes
                amp_scale = np.std(self.C_ell) * 0.1
                amps_init = np.random.randn(n_transitions) * amp_scale
                
                # Random baseline
                baseline_init = np.mean(self.C_ell) + np.random.randn() * np.std(self.C_ell) * 0.1
                
                pos.append(np.concatenate([trans_init, amps_init, [baseline_init]]))
            
            return np.array(pos)
    
    def run_mcmc(self, n_transitions: int = 3,
                n_walkers: int = 32,
                n_steps: int = 5000,
                n_burn: int = 1000,
                initial_transitions: Optional[np.ndarray] = None,
                thin: int = 1) -> Dict[str, Any]:
        """
        Run MCMC sampling for transition parameters.
        
        Parameters:
            n_transitions (int): Number of transitions to fit (default: 3)
            n_walkers (int): Number of MCMC walkers (default: 32)
            n_steps (int): Number of MCMC steps (default: 5000)
            n_burn (int): Burn-in steps to discard (default: 1000)
            initial_transitions (ndarray, optional): Initial transition locations
            thin (int): Thinning factor (default: 1)
            
        Returns:
            dict: MCMC results including chains, posteriors, and diagnostics
        """
        self.output.log_section_header("MCMC BAYESIAN PARAMETER ESTIMATION")
        self.output.log_message(f"Model: {n_transitions} phase transitions")
        self.output.log_message(f"Walkers: {n_walkers}")
        self.output.log_message(f"Steps: {n_steps}")
        self.output.log_message(f"Burn-in: {n_burn}")
        self.output.log_message("")
        
        # Initialize
        n_params = 2 * n_transitions + 1
        pos_init = self.initialize_walkers(n_transitions, n_walkers, initial_transitions)
        
        # Create sampler
        sampler = emcee.EnsembleSampler(
            n_walkers, n_params,
            self.log_posterior_discontinuous,
            args=(n_transitions,)
        )
        
        # Run burn-in
        self.output.log_message("Running burn-in phase...")
        pos_burn, _, _ = sampler.run_mcmc(pos_init, n_burn, progress=False)
        sampler.reset()
        
        self.output.log_message(f"Burn-in complete ({n_burn} steps)")
        self.output.log_message("")
        
        # Production run
        self.output.log_message("Running production MCMC...")
        sampler.run_mcmc(pos_burn, n_steps, thin_by=thin, progress=False)
        
        self.output.log_message(f"MCMC sampling complete ({n_steps} steps)")
        self.output.log_message("")
        
        # Extract chains
        chain = sampler.get_chain(flat=True)  # (n_samples, n_params)
        log_prob = sampler.get_log_prob(flat=True)
        
        # Compute statistics
        posterior_means = np.mean(chain, axis=0)
        posterior_medians = np.median(chain, axis=0)
        posterior_stds = np.std(chain, axis=0)
        
        # Credible intervals (16th and 84th percentiles = 1-sigma)
        credible_lower = np.percentile(chain, 16, axis=0)
        credible_upper = np.percentile(chain, 84, axis=0)
        
        # Extract transitions specifically
        transitions_chain = chain[:, :n_transitions]
        transitions_mean = posterior_means[:n_transitions]
        transitions_median = posterior_medians[:n_transitions]
        transitions_std = posterior_stds[:n_transitions]
        
        # Acceptance fraction
        accept_frac = np.mean(sampler.acceptance_fraction)
        
        # Autocorrelation time (for convergence diagnostic)
        try:
            autocorr_time = sampler.get_autocorr_time(quiet=True)
            autocorr_mean = np.mean(autocorr_time)
        except Exception:
            autocorr_time = None
            autocorr_mean = None
        
        # Log results
        self.output.log_section_header("MCMC POSTERIOR RESULTS")
        self.output.log_message("Transition locations (posterior means ± std):")
        for i in range(n_transitions):
            self.output.log_message(
                f"  ℓ_{i+1} = {transitions_mean[i]:.0f} ± {transitions_std[i]:.0f} "
                f"[{credible_lower[i]:.0f}, {credible_upper[i]:.0f}]"
            )
        self.output.log_message("")
        
        self.output.log_message("Convergence diagnostics:")
        self.output.log_message(f"  Mean acceptance fraction: {accept_frac:.3f}")
        if autocorr_mean is not None:
            self.output.log_message(f"  Mean autocorrelation time: {autocorr_mean:.1f} steps")
            n_independent = len(chain) / autocorr_mean
            self.output.log_message(f"  Effective independent samples: {n_independent:.0f}")
        else:
            self.output.log_message("  Autocorrelation time: Could not compute")
        self.output.log_message("")
        
        # Interpretation
        if accept_frac < 0.2:
            self.output.log_message("⚠ Low acceptance fraction - consider adjusting proposal scale")
        elif accept_frac > 0.5:
            self.output.log_message("⚠ High acceptance fraction - chains may not be exploring efficiently")
        else:
            self.output.log_message("✓ Acceptance fraction in optimal range (0.2-0.5)")
        self.output.log_message("")
        
        # Compile results
        results = {
            'n_transitions': n_transitions,
            'n_walkers': n_walkers,
            'n_steps': n_steps,
            'n_burn': n_burn,
            'n_samples': len(chain),
            'chain': chain,
            'log_prob': log_prob,
            'posterior_means': posterior_means.tolist(),
            'posterior_medians': posterior_medians.tolist(),
            'posterior_stds': posterior_stds.tolist(),
            'credible_intervals_68': {
                'lower': credible_lower.tolist(),
                'upper': credible_upper.tolist()
            },
            'transitions': {
                'mean': transitions_mean.tolist(),
                'median': transitions_median.tolist(),
                'std': transitions_std.tolist(),
                'credible_68_lower': credible_lower[:n_transitions].tolist(),
                'credible_68_upper': credible_upper[:n_transitions].tolist()
            },
            'diagnostics': {
                'acceptance_fraction': float(accept_frac),
                'autocorr_time': autocorr_time.tolist() if autocorr_time is not None else None,
                'autocorr_time_mean': float(autocorr_mean) if autocorr_mean is not None else None,
                'n_effective_samples': float(len(chain) / autocorr_mean) if autocorr_mean else None
            }
        }
        
        return results
    
    def model_comparison(self, max_transitions: int = 5) -> Dict[str, Any]:
        """
        Compare models with different numbers of transitions using MCMC.
        
        Computes Bayesian evidence (marginal likelihood) for each model
        to determine optimal number of transitions.
        
        Parameters:
            max_transitions (int): Maximum number of transitions to test
            
        Returns:
            dict: Model comparison results including Bayes factors
        """
        self.output.log_section_header("MCMC MODEL COMPARISON")
        self.output.log_message(f"Testing models with 0 to {max_transitions} transitions")
        self.output.log_message("")
        
        results = {}
        
        for n in range(max_transitions + 1):
            if n == 0:
                self.output.log_message("Model 0: Smooth (no transitions)")
                # For null model, just compute BIC
                # Use polynomial fit
                poly_order = 9
                coeffs = np.polyfit(self.ell, self.C_ell, poly_order)
                model_smooth = np.polyval(coeffs, self.ell)
                chi2 = np.sum(((self.C_ell - model_smooth) / self.C_ell_err)**2)
                n_params = poly_order + 1
                n_data = len(self.ell)
                bic = chi2 + n_params * np.log(n_data)
                
                results[n] = {
                    'n_transitions': 0,
                    'bic': float(bic),
                    'chi2': float(chi2),
                    'n_params': n_params
                }
                
                self.output.log_message(f"  BIC = {bic:.1f}")
                self.output.log_message("")
            else:
                self.output.log_message(f"Model {n}: {n} transitions")
                
                # Run short MCMC for model comparison
                mcmc_results = self.run_mcmc(
                    n_transitions=n,
                    n_walkers=32,
                    n_steps=2000,
                    n_burn=500
                )
                
                # Approximate evidence using BIC
                # BIC ≈ -2 ln(evidence)
                max_log_prob = np.max(mcmc_results['log_prob'])
                n_params = 2 * n + 1
                n_data = len(self.ell)
                bic = -2 * max_log_prob + n_params * np.log(n_data)
                
                results[n] = {
                    'n_transitions': n,
                    'bic': float(bic),
                    'max_log_prob': float(max_log_prob),
                    'n_params': n_params,
                    'transitions_mean': mcmc_results['transitions']['mean']
                }
                
                self.output.log_message(f"  BIC = {bic:.1f}")
                self.output.log_message(f"  Max log-prob = {max_log_prob:.1f}")
                self.output.log_message("")
        
        # Find best model
        bic_values = [results[n]['bic'] for n in range(max_transitions + 1)]
        best_n = np.argmin(bic_values)
        
        # Compute Bayes factors relative to best model
        bic_best = bic_values[best_n]
        for n in range(max_transitions + 1):
            delta_bic = results[n]['bic'] - bic_best
            bayes_factor = np.exp(-0.5 * delta_bic)
            results[n]['delta_bic'] = float(delta_bic)
            results[n]['bayes_factor_vs_best'] = float(bayes_factor)
        
        self.output.log_section_header("MODEL COMPARISON SUMMARY")
        self.output.log_message(f"{'Model':<10} {'Transitions':<15} {'BIC':<12} {'ΔBIC':<12} {'Preference'}")
        self.output.log_message("-" * 65)
        
        for n in range(max_transitions + 1):
            delta_bic = results[n]['delta_bic']
            if delta_bic == 0:
                pref = "BEST"
            elif delta_bic < 2:
                pref = "Weak"
            elif delta_bic < 6:
                pref = "Positive"
            elif delta_bic < 10:
                pref = "Strong"
            else:
                pref = "Very Strong"
            
            self.output.log_message(
                f"{n:<10} {n:<15} {results[n]['bic']:<12.1f} "
                f"{delta_bic:<12.1f} {pref}"
            )
        
        self.output.log_message("")
        self.output.log_message(f"✓ Best model: {best_n} transitions (lowest BIC)")
        self.output.log_message("")
        
        return {
            'models': results,
            'best_model': best_n,
            'bic_values': bic_values
        }

