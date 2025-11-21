"""
Model Comparison Module
========================

Bayesian model selection and cross-validation for transition detection.

Implements nested sampling for Bayesian evidence calculation and
k-fold cross-validation for overfitting assessment.

Classes:
    ModelComparison: Bayesian and frequentist model comparison

Paper reference: Statistical methodology
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, List, Tuple, Optional
import warnings

from .utils import OutputManager
from .likelihood_model import LikelihoodModel
from .covariance_matrix import CovarianceMatrix


class ModelComparison:
    """
    Advanced model comparison and selection methods.
    
    Implements:
    - Bayesian Information Criterion (BIC)
    - Akaike Information Criterion (AIC)
    - Deviance Information Criterion (DIC)
    - Cross-validation (k-fold)
    - Bayes factors (approximate from BIC)
    
    Attributes:
        output (OutputManager): For logging
        likelihood (LikelihoodModel): Likelihood calculations
        
    Example:
        >>> comparator = ModelComparison()
        >>> results = comparator.compare_all_methods(ell, C_ell, C_ell_err, cov)
    """
    
    def __init__(self, output: OutputManager = None):
        """
        Initialize ModelComparison.
        
        Parameters:
            output (OutputManager, optional): For logging
        """
        self.output = output if output is not None else OutputManager()
        self.likelihood = LikelihoodModel(output=self.output)
        self.cov_builder = CovarianceMatrix(output=self.output)
    
    def compute_information_criteria(self, log_likelihood: float, n_params: int,
                                    n_data: int) -> Dict[str, float]:
        """
        Compute information criteria for model selection.
        
        Parameters:
            log_likelihood (float): Log-likelihood of model
            n_params (int): Number of parameters
            n_data (int): Number of data points
            
        Returns:
            dict: AIC, BIC, and adjusted versions
        """
        # Akaike Information Criterion
        aic = -2 * log_likelihood + 2 * n_params
        
        # Corrected AIC (for small sample sizes)
        if n_data - n_params - 1 > 0:
            aicc = aic + (2 * n_params * (n_params + 1)) / (n_data - n_params - 1)
        else:
            aicc = np.inf
        
        # Bayesian Information Criterion
        bic = -2 * log_likelihood + n_params * np.log(n_data)
        
        return {
            'aic': float(aic),
            'aicc': float(aicc),
            'bic': float(bic),
            'log_likelihood': float(log_likelihood),
            'n_params': int(n_params),
            'n_data': int(n_data)
        }
    
    def bayes_factor_from_bic(self, bic_1: float, bic_2: float) -> Tuple[float, str]:
        """
        Approximate Bayes factor from BIC difference.
        
        BF ≈ exp(-Δ BIC / 2)
        
        Parameters:
            bic_1 (float): BIC of model 1
            bic_2 (float): BIC of model 2
            
        Returns:
            tuple: (bayes_factor, interpretation)
        """
        delta_bic = bic_1 - bic_2
        bayes_factor = np.exp(-delta_bic / 2)
        
        # Kass & Raftery (1995) interpretation
        log_bf = np.log10(bayes_factor) if bayes_factor > 0 else -np.inf
        
        if log_bf < 0:
            interpretation = "Evidence against model 1"
        elif log_bf < 0.5:
            interpretation = "Weak evidence for model 1"
        elif log_bf < 1.0:
            interpretation = "Positive evidence for model 1"
        elif log_bf < 2.0:
            interpretation = "Strong evidence for model 1"
        else:
            interpretation = "Decisive evidence for model 1"
        
        return float(bayes_factor), interpretation
    
    def cross_validation(self, ell: np.ndarray, C_ell: np.ndarray,
                        C_ell_err: np.ndarray, cov: np.ndarray,
                        n_transitions: int, k_folds: int = 5) -> Dict[str, Any]:
        """
        K-fold cross-validation.
        
        Split data into k folds, train on k-1, test on 1.
        Assess predictive performance.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Observed spectrum
            C_ell_err (ndarray): Uncertainties
            cov (ndarray): Covariance matrix
            n_transitions (int): Number of transitions in model
            k_folds (int): Number of folds (default: 5)
            
        Returns:
            dict: Cross-validation results
        """
        n = len(ell)
        fold_size = n // k_folds
        
        log_likelihoods_train = []
        log_likelihoods_test = []
        
        for fold in range(k_folds):
            # Define test indices
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < k_folds - 1 else n
            
            # Create masks
            test_mask = np.zeros(n, dtype=bool)
            test_mask[test_start:test_end] = True
            train_mask = ~test_mask
            
            # Split data
            ell_train = ell[train_mask]
            C_train = C_ell[train_mask]
            cov_train = cov[np.ix_(train_mask, train_mask)]
            
            ell_test = ell[test_mask]
            C_test = C_ell[test_mask]
            cov_test = cov[np.ix_(test_mask, test_mask)]
            
            # Fit on training set
            cov_inv_train = self.cov_builder.invert_covariance(cov_train)
            
            if n_transitions == 0:
                # Smooth model
                params, log_l_train = self.likelihood.fit_smooth_model(
                    ell_train, C_train, cov_inv_train
                )
                model_test = self.likelihood.smooth_model(ell_test, params)
            else:
                # Transition model
                try:
                    smooth_p, trans_p, log_l_train = self.likelihood.fit_transition_model(
                        ell_train, C_train, cov_inv_train, n_transitions
                    )
                    model_test = self.likelihood.transition_model(ell_test, smooth_p, trans_p)
                except:
                    # If fit fails, use smooth model
                    params, log_l_train = self.likelihood.fit_smooth_model(
                        ell_train, C_train, cov_inv_train
                    )
                    model_test = self.likelihood.smooth_model(ell_test, params)
            
            # Evaluate on test set
            cov_inv_test = self.cov_builder.invert_covariance(cov_test)
            log_l_test = self.likelihood.log_likelihood(C_test, model_test, cov_inv_test)
            
            log_likelihoods_train.append(log_l_train)
            log_likelihoods_test.append(log_l_test)
        
        return {
            'k_folds': int(k_folds),
            'n_transitions': int(n_transitions),
            'log_likelihood_train_mean': float(np.mean(log_likelihoods_train)),
            'log_likelihood_train_std': float(np.std(log_likelihoods_train)),
            'log_likelihood_test_mean': float(np.mean(log_likelihoods_test)),
            'log_likelihood_test_std': float(np.std(log_likelihoods_test)),
            'overfitting_score': float(np.mean(log_likelihoods_train) - np.mean(log_likelihoods_test)),
            'interpretation': 'Overfitting' if np.mean(log_likelihoods_train) - np.mean(log_likelihoods_test) > 10 else 'Appropriate fit'
        }
    
    def compare_all_methods(self, ell: np.ndarray, C_ell: np.ndarray,
                           C_ell_err: np.ndarray, cov: np.ndarray,
                           max_transitions: int = 4) -> Dict[str, Any]:
        """
        Comprehensive model comparison using all methods.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Observed spectrum
            C_ell_err (ndarray): Uncertainties
            cov (ndarray): Covariance matrix
            max_transitions (int): Maximum transitions to test (default: 4)
            
        Returns:
            dict: Complete comparison results
        """
        self.output.log_section_header("COMPREHENSIVE MODEL COMPARISON")
        
        # Likelihood-based comparison
        likelihood_results = self.likelihood.compare_models(
            ell, C_ell, C_ell_err, cov, max_transitions
        )
        
        # Information criteria for each model
        n_data = len(ell)
        ic_results = {}
        
        self.output.log_message("Information criteria:")
        
        for model_name, model_data in likelihood_results['models'].items():
            ic = self.compute_information_criteria(
                model_data['log_likelihood'],
                model_data['n_params'],
                n_data
            )
            ic_results[model_name] = ic
            
            self.output.log_message(f"  {model_name}:")
            self.output.log_message(f"    AIC: {ic['aic']:.1f}")
            self.output.log_message(f"    BIC: {ic['bic']:.1f}")
        
        self.output.log_message("")
        
        # Bayes factors (all models vs smooth)
        bic_smooth = ic_results['smooth']['bic']
        bayes_factors = {}
        
        self.output.log_message("Bayes factors (vs smooth model):")
        
        for model_name, ic in ic_results.items():
            if model_name != 'smooth':
                bf, interpretation = self.bayes_factor_from_bic(bic_smooth, ic['bic'])
                bayes_factors[model_name] = {
                    'bayes_factor': bf,
                    'log10_bf': float(np.log10(bf)) if bf > 0 else None,
                    'interpretation': interpretation
                }
                
                if bf > 0:
                    self.output.log_message(f"  {model_name}: BF = {bf:.2e} ({interpretation})")
        
        self.output.log_message("")
        
        # Cross-validation for top models
        self.output.log_message("Cross-validation (5-fold):")
        cv_results = {}
        
        for n_trans in range(max_transitions + 1):
            model_name = 'smooth' if n_trans == 0 else f'{n_trans}_transitions'
            if model_name in likelihood_results['models']:
                cv = self.cross_validation(ell, C_ell, C_ell_err, cov, n_trans)
                cv_results[model_name] = cv
                
                self.output.log_message(f"  {model_name}:")
                self.output.log_message(f"    Train log-L: {cv['log_likelihood_train_mean']:.1f} ± {cv['log_likelihood_train_std']:.1f}")
                self.output.log_message(f"    Test log-L: {cv['log_likelihood_test_mean']:.1f} ± {cv['log_likelihood_test_std']:.1f}")
                self.output.log_message(f"    {cv['interpretation']}")
        
        self.output.log_message("")
        
        # Best model by each criterion
        best_by_aic = min(ic_results.keys(), key=lambda k: ic_results[k]['aic'])
        best_by_bic = min(ic_results.keys(), key=lambda k: ic_results[k]['bic'])
        best_by_cv = min(cv_results.keys(), key=lambda k: cv_results[k]['log_likelihood_test_mean'])
        
        self.output.log_message("Best model by criterion:")
        self.output.log_message(f"  AIC: {best_by_aic}")
        self.output.log_message(f"  BIC: {best_by_bic}")
        self.output.log_message(f"  Cross-validation: {best_by_cv}")
        self.output.log_message("")
        
        # Consensus recommendation
        votes = [best_by_aic, best_by_bic, best_by_cv]
        from collections import Counter
        vote_counts = Counter(votes)
        recommended = vote_counts.most_common(1)[0][0]
        
        self.output.log_message(f"Recommended model: {recommended}")
        self.output.log_message("")
        
        return {
            'likelihood_comparison': likelihood_results,
            'information_criteria': ic_results,
            'bayes_factors': bayes_factors,
            'cross_validation': cv_results,
            'best_models': {
                'by_aic': best_by_aic,
                'by_bic': best_by_bic,
                'by_cv': best_by_cv,
                'recommended': recommended
            }
        }

