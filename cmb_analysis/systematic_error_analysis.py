"""
Systematic Error Analysis Module
==================================

Comprehensive systematic error treatment and robustness testing.

Tests:
1. Statistical vs statistical+systematic comparison
2. Individual systematic source impact
3. Robustness to cosmological assumptions (r_d, Ω_m)
4. Jack-knife resampling

Classes:
    SystematicErrorAnalysis: Full systematic error framework
"""

import numpy as np
from typing import Dict, Any, Callable
from scipy.stats import chi2 as chi2_dist

from .utils import OutputManager
from .bao_datasets import BAODataset


class SystematicErrorAnalysis:
    """
    Test robustness to systematic uncertainties.
    
    Comprehensive framework for:
    - Comparing statistical vs total errors
    - Isolating impact of individual systematics
    - Varying cosmological assumptions
    - Jack-knife stability tests
    
    Attributes:
        output (OutputManager): For logging
        
    Example:
        >>> sys_analysis = SystematicErrorAnalysis()
        >>> results = sys_analysis.test_with_systematics(dataset, predictions, alpha=-5)
    """
    
    def __init__(self, output: OutputManager = None):
        """
        Initialize SystematicErrorAnalysis.
        
        Parameters:
            output (OutputManager, optional): For logging
        """
        self.output = output if output is not None else OutputManager()
    
    def calculate_chi2(self, predictions: np.ndarray,
                      observations: np.ndarray,
                      covariance: np.ndarray) -> float:
        """Calculate χ²."""
        residuals = observations - predictions
        cov_inv = np.linalg.inv(covariance)
        return float(residuals @ cov_inv @ residuals)
    
    def test_with_systematics(self, dataset: BAODataset,
                             predictions: np.ndarray,
                             alpha: float) -> Dict[str, Any]:
        """
        Compare analysis with and without systematic errors.
        
        Parameters:
            dataset (BAODataset): Survey data
            predictions (ndarray): Theory predictions
            alpha (float): Alpha value used
            
        Returns:
            dict: Comparison results
        """
        self.output.log_message(f"\nSystematic error analysis for {dataset.name}:")
        self.output.log_message("-" * 60)
        
        # Test 1: Statistical only
        cov_stat = dataset.statistical_covariance()
        chi2_stat = self.calculate_chi2(predictions, dataset.values, cov_stat)
        p_stat = 1.0 - chi2_dist.cdf(chi2_stat, dataset.dof)
        
        # Test 2: With all systematics
        cov_total = dataset.total_covariance(include_systematics=True)
        chi2_total = self.calculate_chi2(predictions, dataset.values, cov_total)
        p_total = 1.0 - chi2_dist.cdf(chi2_total, dataset.dof)
        
        # Test 3: Individual systematic sources
        sys_impact = {}
        for sys_name, sys_frac in dataset.systematics.items():
            # Add this systematic to statistical covariance
            sys_abs = sys_frac * dataset.values
            cov_sys_single = cov_stat + np.diag(sys_abs**2)
            chi2_sys = self.calculate_chi2(predictions, dataset.values, cov_sys_single)
            p_sys = 1.0 - chi2_dist.cdf(chi2_sys, dataset.dof)
            
            sys_impact[sys_name] = {
                'chi2': float(chi2_sys),
                'p_value': float(p_sys),
                'impact': float(chi2_sys - chi2_stat)
            }
        
        self.output.log_message(f"  Statistical only:     χ² = {chi2_stat:.2f}, p = {p_stat:.4f}")
        self.output.log_message(f"  With all systematics: χ² = {chi2_total:.2f}, p = {p_total:.4f}")
        self.output.log_message(f"  Impact of systematics: Δχ² = {chi2_total - chi2_stat:.2f}")
        
        return {
            'alpha': float(alpha),
            'statistical_only': {
                'chi2': float(chi2_stat),
                'p_value': float(p_stat),
                'passes': bool(p_stat > 0.05)
            },
            'with_systematics': {
                'chi2': float(chi2_total),
                'p_value': float(p_total),
                'passes': bool(p_total > 0.05)
            },
            'systematic_impact': sys_impact,
            'robust_to_systematics': bool(p_total > 0.05) if p_stat > 0.05 else 'N/A'
        }
    
    def robustness_tests(self, dataset: BAODataset,
                        predict_function: Callable,
                        alpha_best: float) -> Dict[str, Any]:
        """
        Test robustness to cosmological assumptions.
        
        Varies:
        - r_d by ±1% (Planck uncertainty)
        - Ω_m by ±0.02 (Planck vs weak lensing)
        
        Parameters:
            dataset (BAODataset): Survey data
            predict_function (callable): Function(z, alpha, r_d_factor, omega_m)
            alpha_best (float): Best-fit alpha value
            
        Returns:
            dict: Robustness test results
        """
        self.output.log_message(f"\nRobustness tests for {dataset.name}:")
        self.output.log_message("-" * 60)
        
        results = {}
        cov = dataset.total_covariance(include_systematics=True)
        
        # Baseline (no variations)
        pred_baseline = predict_function(dataset.redshifts, alpha_best, r_d_factor=1.0, omega_m=0.315)
        chi2_baseline = self.calculate_chi2(pred_baseline, dataset.values, cov)
        
        # Vary r_d
        self.output.log_message("\n  Varying r_d (±1%):")
        for r_d_factor in [0.99, 1.00, 1.01]:
            pred = predict_function(dataset.redshifts, alpha_best, r_d_factor=r_d_factor, omega_m=0.315)
            chi2 = self.calculate_chi2(pred, dataset.values, cov)
            p_val = 1.0 - chi2_dist.cdf(chi2, dataset.dof)
            
            results[f'r_d_{r_d_factor:.2f}'] = {
                'chi2': float(chi2),
                'p_value': float(p_val),
                'passes': bool(p_val > 0.05)
            }
            
            self.output.log_message(f"    r_d × {r_d_factor:.2f}: χ² = {chi2:.2f}, p = {p_val:.4f}")
        
        # Vary Ω_m  
        self.output.log_message("\n  Varying Ω_m (±0.02):")
        for omega_m in [0.295, 0.315, 0.335]:
            pred = predict_function(dataset.redshifts, alpha_best, r_d_factor=1.0, omega_m=omega_m)
            chi2 = self.calculate_chi2(pred, dataset.values, cov)
            p_val = 1.0 - chi2_dist.cdf(chi2, dataset.dof)
            
            results[f'omega_m_{omega_m:.3f}'] = {
                'chi2': float(chi2),
                'p_value': float(p_val),
                'passes': bool(p_val > 0.05)
            }
            
            self.output.log_message(f"    Ω_m = {omega_m:.3f}: χ² = {chi2:.2f}, p = {p_val:.4f}")
        
        # Check if all variations pass
        all_pass = all(r['passes'] for r in results.values())
        
        self.output.log_message(f"\n  Robustness: {'PASS' if all_pass else 'MARGINAL'}")
        self.output.log_message(f"  {sum(r['passes'] for r in results.values())}/{len(results)} variations pass")
        
        return {
            'baseline_chi2': float(chi2_baseline),
            'variations': results,
            'all_pass': bool(all_pass),
            'pass_fraction': float(sum(r['passes'] for r in results.values()) / len(results))
        }
    
    def jackknife_test(self, dataset: BAODataset,
                      predictions: np.ndarray,
                      alpha: float) -> Dict[str, Any]:
        """
        Jack-knife resampling: drop each bin sequentially.
        
        Tests stability when individual data points removed.
        
        Parameters:
            dataset (BAODataset): Survey data
            predictions (ndarray): Theory predictions
            alpha (float): Alpha value
            
        Returns:
            dict: Jack-knife results
        """
        if dataset.n_bins < 3:
            self.output.log_message(f"\n  Jack-knife skipped: only {dataset.n_bins} bins")
            return {'skipped': True, 'reason': 'insufficient_bins'}
        
        self.output.log_message(f"\nJack-knife test for {dataset.name}:")
        self.output.log_message("-" * 60)
        
        cov = dataset.total_covariance(include_systematics=True)
        results = {}
        
        for i in range(dataset.n_bins):
            # Drop bin i
            mask = np.ones(dataset.n_bins, dtype=bool)
            mask[i] = False
            
            pred_jk = predictions[mask]
            obs_jk = dataset.values[mask]
            cov_jk = cov[np.ix_(mask, mask)]
            
            chi2_jk = self.calculate_chi2(pred_jk, obs_jk, cov_jk)
            dof_jk = dataset.n_bins - 1
            p_jk = 1.0 - chi2_dist.cdf(chi2_jk, dof_jk)
            
            results[f'drop_bin_{i}'] = {
                'dropped_z': float(dataset.redshifts[i]),
                'chi2': float(chi2_jk),
                'dof': int(dof_jk),
                'p_value': float(p_jk),
                'passes': bool(p_jk > 0.05)
            }
            
            self.output.log_message(f"  Drop z={dataset.redshifts[i]:.2f}: χ² = {chi2_jk:.2f}, p = {p_jk:.4f}")
        
        # Stability metric
        all_pass = all(r['passes'] for r in results.values())
        
        self.output.log_message(f"\n  Jack-knife stability: {'STABLE' if all_pass else 'UNSTABLE'}")
        
        return {
            'individual_tests': results,
            'all_pass': bool(all_pass),
            'stability': 'stable' if all_pass else 'unstable'
        }
    
    def systematic_budget_report(self, dataset: BAODataset) -> Dict[str, Any]:
        """
        Generate systematic error budget report.
        
        Parameters:
            dataset (BAODataset): Survey data
            
        Returns:
            dict: Systematic budget breakdown
        """
        self.output.log_message(f"\nSystematic Error Budget for {dataset.name}:")
        self.output.log_message("-" * 60)
        
        # Calculate total systematic per bin
        sys_total_per_bin = np.zeros(dataset.n_bins)
        sys_breakdown = {}
        
        for sys_name, sys_frac in dataset.systematics.items():
            sys_abs = sys_frac * dataset.values
            sys_total_per_bin += sys_abs**2
            
            mean_contribution = np.mean(sys_abs)
            sys_breakdown[sys_name] = {
                'fractional': float(sys_frac),
                'mean_absolute': float(mean_contribution),
                'per_bin': sys_abs.tolist()
            }
            
            self.output.log_message(f"  {sys_name:<20}: {sys_frac*100:>5.2f}% ({mean_contribution:.3f} mean)")
        
        sys_total_per_bin = np.sqrt(sys_total_per_bin)
        mean_sys = np.mean(sys_total_per_bin)
        mean_stat = np.mean(dataset.stat_errors)
        
        total_error = np.sqrt(mean_stat**2 + mean_sys**2)
        sys_fraction = mean_sys / total_error
        
        self.output.log_message(f"\n  Total systematic (quadrature): {mean_sys:.3f} ({sys_fraction*100:.1f}% of total)")
        self.output.log_message(f"  Statistical error:             {mean_stat:.3f}")
        self.output.log_message(f"  Total error:                   {total_error:.3f}")
        
        return {
            'systematic_sources': sys_breakdown,
            'mean_systematic': float(mean_sys),
            'mean_statistical': float(mean_stat),
            'systematic_fraction': float(sys_fraction),
            'dominant_systematic': max(dataset.systematics.items(), key=lambda x: x[1])[0]
        }

