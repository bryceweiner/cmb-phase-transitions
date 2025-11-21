"""
Cross-Validation for BAO Predictions Module
============================================

Leave-one-out and K-fold cross-validation.

CRITICAL: Since we have ZERO fitted parameters, "training" just means
verify that theoretical prediction works on held-out data.

Classes:
    CrossValidationBAO: LOO-CV and K-fold tests

Usage:
    python main.py --bao --cross-validation
"""

import numpy as np
from typing import Dict, Any, List, Callable
from scipy.stats import chi2 as chi2_dist

from .utils import OutputManager
from .bao_datasets import BAODataset


class CrossValidationBAO:
    """
    Cross-validation framework for BAO predictions.
    
    Since anti-viscosity is parameter-free, CV tests whether
    theoretical prediction generalizes across datasets.
    
    Attributes:
        output (OutputManager): For logging
        
    Example:
        >>> cv = CrossValidationBAO()
        >>> results = cv.leave_one_out(datasets, predictor)
    """
    
    def __init__(self, output: OutputManager = None):
        """
        Initialize CrossValidationBAO.
        
        Parameters:
            output (OutputManager, optional): For logging
        """
        self.output = output if output is not None else OutputManager()
    
    def leave_one_out(self, datasets: Dict[str, BAODataset],
                     predict_function: Callable) -> Dict[str, Any]:
        """
        Leave-one-out cross-validation.
        
        For each dataset:
        1. "Train" on all others (just verify theory works)
        2. Predict held-out dataset
        3. Compare prediction to observation
        
        Since we have no parameters, this tests generalization.
        
        Parameters:
            datasets (dict): All datasets
            predict_function (callable): Prediction function
            
        Returns:
            dict: LOO-CV results
        """
        self.output.log_section_header("LEAVE-ONE-OUT CROSS-VALIDATION")
        self.output.log_message("")
        self.output.log_message("Testing: Can theory predict each dataset independently?")
        self.output.log_message("(No training - pure theoretical prediction)")
        self.output.log_message("")
        
        results = {}
        
        for name, held_out in datasets.items():
            self.output.log_message(f"Holding out: {name}")
            
            # Predict held-out dataset using theoretical anti-viscosity
            predictions = predict_function(held_out.redshifts)
            
            # Calculate χ²
            cov = held_out.total_covariance(include_systematics=True)
            residuals = held_out.values - predictions
            cov_inv = np.linalg.inv(cov)
            chi2 = float(residuals @ cov_inv @ residuals)
            
            # p-value
            p_value = 1.0 - chi2_dist.cdf(chi2, held_out.dof)
            
            # Prediction error
            pred_error_sigma = np.abs(residuals) / np.sqrt(np.diag(cov))
            max_error = np.max(pred_error_sigma)
            mean_error = np.mean(pred_error_sigma)
            
            results[name] = {
                'chi2': float(chi2),
                'dof': int(held_out.dof),
                'p_value': float(p_value),
                'passes': bool(p_value > 0.05),
                'max_error_sigma': float(max_error),
                'mean_error_sigma': float(mean_error),
                'predictions': predictions.tolist(),
                'observations': held_out.values.tolist()
            }
            
            self.output.log_message(f"  χ² = {chi2:.2f}, p = {p_value:.4f}")
            self.output.log_message(f"  Max error: {max_error:.2f}σ, Mean: {mean_error:.2f}σ")
            self.output.log_message(f"  Verdict: {'PASS' if p_value > 0.05 else 'FAIL'}")
            self.output.log_message("")
        
        # Summary
        n_pass = sum(1 for r in results.values() if r['passes'])
        n_total = len(results)
        
        self.output.log_message("=" * 70)
        self.output.log_message(f"LOO-CV SUMMARY: {n_pass}/{n_total} datasets pass")
        self.output.log_message("")
        
        if n_pass == n_total:
            self.output.log_message("✓ PERFECT: Theory predicts ALL held-out datasets")
        elif n_pass >= n_total - 1:
            self.output.log_message("✓ EXCELLENT: Theory generalizes")
        elif n_pass >= n_total / 2:
            self.output.log_message("~ GOOD: Most datasets predicted")
        else:
            self.output.log_message("✗ POOR: Theory doesn't generalize")
        
        self.output.log_message("")
        
        return {
            'method': 'leave_one_out',
            'individual_results': results,
            'n_datasets': n_total,
            'n_passing': n_pass,
            'pass_fraction': n_pass / n_total
        }
    
    def k_fold(self, datasets: Dict[str, BAODataset],
               predict_function: Callable,
               k: int = 5) -> Dict[str, Any]:
        """
        K-fold cross-validation.
        
        Split datasets into k folds, predict each from others.
        
        Parameters:
            datasets (dict): All datasets
            predict_function (callable): Prediction function
            k (int): Number of folds
            
        Returns:
            dict: K-fold results
        """
        self.output.log_section_header(f"{k}-FOLD CROSS-VALIDATION")
        self.output.log_message("")
        
        # Split datasets into k folds
        dataset_list = list(datasets.items())
        np.random.shuffle(dataset_list)
        fold_size = len(dataset_list) // k
        
        results = {}
        
        for fold_idx in range(k):
            # Define held-out fold
            start = fold_idx * fold_size
            end = start + fold_size if fold_idx < k-1 else len(dataset_list)
            held_out_fold = dataset_list[start:end]
            
            self.output.log_message(f"FOLD {fold_idx+1}/{k}:")
            self.output.log_message(f"  Held out: {[name for name, _ in held_out_fold]}")
            
            # Predict each dataset in fold
            fold_results = {}
            for name, dataset in held_out_fold:
                predictions = predict_function(dataset.redshifts)
                cov = dataset.total_covariance(include_systematics=True)
                residuals = dataset.values - predictions
                chi2 = float(residuals @ np.linalg.inv(cov) @ residuals)
                p_value = 1.0 - chi2_dist.cdf(chi2, dataset.dof)
                
                fold_results[name] = {
                    'chi2': float(chi2),
                    'p_value': float(p_value),
                    'passes': bool(p_value > 0.05)
                }
                
                self.output.log_message(f"    {name}: χ² = {chi2:.2f}, p = {p_value:.4f}")
            
            results[f'fold_{fold_idx+1}'] = fold_results
            self.output.log_message("")
        
        # Overall statistics
        all_results = []
        for fold in results.values():
            all_results.extend(fold.values())
        
        n_pass = sum(1 for r in all_results if r['passes'])
        n_total = len(all_results)
        
        self.output.log_message(f"K-FOLD SUMMARY: {n_pass}/{n_total} pass")
        self.output.log_message("")
        
        return {
            'method': 'k_fold',
            'k': k,
            'fold_results': results,
            'n_passing': n_pass,
            'n_total': n_total,
            'pass_fraction': n_pass / n_total
        }

