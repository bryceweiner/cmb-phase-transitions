"""
Null Hypothesis Tests Module
=============================

Bootstrap, shuffling, and other null tests to verify statistical robustness.

Uses Apple Silicon MPS for GPU acceleration when available.

Classes:
    NullHypothesisTests: Bootstrap, shuffling, randomization tests

Usage:
    python main.py --bao --null-tests
"""

import numpy as np
from typing import Dict, Any, Callable
from scipy.stats import chi2 as chi2_dist
import warnings

from .utils import OutputManager

# Try to import PyTorch for MPS acceleration
try:
    import torch
    MPS_AVAILABLE = torch.backends.mps.is_available()
except ImportError:
    MPS_AVAILABLE = False
    warnings.warn("PyTorch not available. Using CPU for null tests.")


class NullHypothesisTests:
    """
    Null hypothesis testing framework.
    
    Tests statistical robustness through:
    - Bootstrap resampling (with MPS acceleration)
    - Data shuffling
    - Randomization tests
    
    Attributes:
        output (OutputManager): For logging
        use_mps (bool): Use Apple Silicon GPU if available
        
    Example:
        >>> tester = NullHypothesisTests(use_mps=True)
        >>> results = tester.bootstrap_test(predictions, observations, n_bootstrap=10000)
    """
    
    def __init__(self, output: OutputManager = None, use_mps: bool = True):
        """
        Initialize NullHypothesisTests.
        
        Parameters:
            output (OutputManager, optional): For logging
            use_mps (bool): Use MPS acceleration if available
        """
        self.output = output if output is not None else OutputManager()
        self.use_mps = use_mps and MPS_AVAILABLE
        
        if self.use_mps:
            self.device = torch.device("mps")
            self.output.log_message("Using Apple Silicon MPS for acceleration")
        else:
            self.device = None
            self.output.log_message("Using CPU for null tests")
    
    def bootstrap_test(self, predictions: np.ndarray,
                      observations: np.ndarray,
                      errors: np.ndarray,
                      n_bootstrap: int = 10000) -> Dict[str, Any]:
        """
        Bootstrap resampling test.
        
        Resample observations within errors, recalculate χ².
        Tests stability of result.
        
        Parameters:
            predictions (ndarray): Theory predictions
            observations (ndarray): Observed values
            errors (ndarray): Measurement uncertainties
            n_bootstrap (int): Number of bootstrap iterations
            
        Returns:
            dict: Bootstrap results
        """
        self.output.log_message(f"\nBootstrap resampling ({n_bootstrap} iterations):")
        self.output.log_message("-" * 60)
        
        chi2_baseline = np.sum(((observations - predictions) / errors)**2)
        
        if self.use_mps:
            # GPU-accelerated bootstrap
            obs_tensor = torch.tensor(observations, device=self.device, dtype=torch.float32)
            pred_tensor = torch.tensor(predictions, device=self.device, dtype=torch.float32)
            err_tensor = torch.tensor(errors, device=self.device, dtype=torch.float32)
            
            # Generate all bootstrap samples at once
            noise = torch.randn(n_bootstrap, len(observations), device=self.device)
            obs_bootstrap = obs_tensor + noise * err_tensor
            
            # Calculate χ² for all samples
            residuals = obs_bootstrap - pred_tensor
            chi2_bootstrap = torch.sum((residuals / err_tensor)**2, dim=1)
            chi2_bootstrap = chi2_bootstrap.cpu().numpy()
        else:
            # CPU bootstrap
            chi2_bootstrap = np.zeros(n_bootstrap)
            for i in range(n_bootstrap):
                obs_resample = observations + np.random.randn(len(observations)) * errors
                chi2_bootstrap[i] = np.sum(((obs_resample - predictions) / errors)**2)
        
        # Statistics
        mean_chi2 = np.mean(chi2_bootstrap)
        std_chi2 = np.std(chi2_bootstrap)
        percentiles = np.percentile(chi2_bootstrap, [2.5, 16, 50, 84, 97.5])
        
        # Fraction worse than baseline
        frac_worse = np.sum(chi2_bootstrap > chi2_baseline) / n_bootstrap
        
        self.output.log_message(f"  Baseline χ²: {chi2_baseline:.2f}")
        self.output.log_message(f"  Bootstrap mean: {mean_chi2:.2f} ± {std_chi2:.2f}")
        self.output.log_message(f"  68% CI: [{percentiles[1]:.2f}, {percentiles[3]:.2f}]")
        self.output.log_message(f"  95% CI: [{percentiles[0]:.2f}, {percentiles[4]:.2f}]")
        self.output.log_message(f"  Fraction worse than baseline: {frac_worse*100:.1f}%")
        
        if abs(chi2_baseline - mean_chi2) < 2*std_chi2:
            self.output.log_message("  ✓ STABLE: Baseline within 2σ of bootstrap mean")
            stable = True
        else:
            self.output.log_message("  ⚠ UNSTABLE: Baseline deviates from bootstrap")
            stable = False
        
        self.output.log_message("")
        
        return {
            'n_bootstrap': n_bootstrap,
            'chi2_baseline': float(chi2_baseline),
            'chi2_mean': float(mean_chi2),
            'chi2_std': float(std_chi2),
            'chi2_percentiles': percentiles.tolist(),
            'fraction_worse': float(frac_worse),
            'stable': bool(stable),
            'used_mps': self.use_mps
        }
    
    def shuffling_test(self, predictions: np.ndarray,
                      observations: np.ndarray,
                      errors: np.ndarray,
                      n_shuffles: int = 1000) -> Dict[str, Any]:
        """
        Data shuffling test.
        
        Randomly permute observations. Should destroy any real signal.
        If χ² improves with shuffling, indicates overfitting.
        
        Parameters:
            predictions (ndarray): Theory predictions
            observations (ndarray): Observed values
            errors (ndarray): Measurement uncertainties
            n_shuffles (int): Number of shuffle iterations
            
        Returns:
            dict: Shuffling test results
        """
        self.output.log_message(f"\nShuffling test ({n_shuffles} permutations):")
        self.output.log_message("-" * 60)
        
        chi2_real = np.sum(((observations - predictions) / errors)**2)
        
        # Shuffle and recalculate
        chi2_shuffled = np.zeros(n_shuffles)
        for i in range(n_shuffles):
            obs_shuffled = np.random.permutation(observations)
            chi2_shuffled[i] = np.sum(((obs_shuffled - predictions) / errors)**2)
        
        # How many shuffles give better χ²?
        n_better = np.sum(chi2_shuffled < chi2_real)
        p_value_shuffle = n_better / n_shuffles
        
        self.output.log_message(f"  Real data χ²: {chi2_real:.2f}")
        self.output.log_message(f"  Shuffled mean χ²: {np.mean(chi2_shuffled):.2f}")
        self.output.log_message(f"  Shuffles with better χ²: {n_better}/{n_shuffles}")
        self.output.log_message(f"  p-value: {p_value_shuffle:.4f}")
        
        if p_value_shuffle < 0.05:
            self.output.log_message("  ✓ ROBUST: Real data significantly better than shuffled")
        else:
            self.output.log_message("  ✗ WARNING: Shuffled data comparable to real")
        
        self.output.log_message("")
        
        return {
            'n_shuffles': n_shuffles,
            'chi2_real': float(chi2_real),
            'chi2_shuffled_mean': float(np.mean(chi2_shuffled)),
            'n_better_shuffles': int(n_better),
            'p_value': float(p_value_shuffle),
            'robust': bool(p_value_shuffle < 0.05)
        }

