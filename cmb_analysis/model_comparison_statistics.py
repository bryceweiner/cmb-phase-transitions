"""
Model Comparison Statistics Module
===================================

Information criteria and Bayes factors for model comparison.

Implements:
- BIC (Bayesian Information Criterion)
- AIC (Akaike Information Criterion)
- DIC (Deviance Information Criterion)
- Bayes Factors via evidence calculation

Classes:
    ModelComparisonStatistics: Compare anti-viscosity vs alternatives

Usage:
    python main.py --bao --model-comparison
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from scipy.stats import chi2 as chi2_dist
import json

from .utils import OutputManager


class ModelComparisonStatistics:
    """
    Compare quantum anti-viscosity model to alternatives using information criteria.
    
    Key advantage: Anti-viscosity has ZERO new parameters!
    - k=0 for anti-viscosity (parameter-free)
    - k≥1 for alternatives (wCDM, EDE, etc.)
    
    This means anti-viscosity automatically wins on BIC/AIC if fit quality is comparable.
    
    Attributes:
        output (OutputManager): For logging
        
    Example:
        >>> comparator = ModelComparisonStatistics()
        >>> results = comparator.compare_all_models(datasets, predictions_dict)
    """
    
    def __init__(self, output: OutputManager = None):
        """
        Initialize ModelComparisonStatistics.
        
        Parameters:
            output (OutputManager, optional): For logging
        """
        self.output = output if output is not None else OutputManager()
    
    def calculate_bic(self, chi2: float, n_data: int, k_params: int) -> float:
        """
        Bayesian Information Criterion.
        
        BIC = χ² + k×ln(n)
        
        Lower is better. Penalizes extra parameters.
        
        Parameters:
            chi2 (float): Chi-squared value
            n_data (int): Number of data points
            k_params (int): Number of free parameters
            
        Returns:
            float: BIC value
        """
        return chi2 + k_params * np.log(n_data)
    
    def calculate_aic(self, chi2: float, k_params: int) -> float:
        """
        Akaike Information Criterion.
        
        AIC = χ² + 2k
        
        Lower is better. Lighter penalty than BIC.
        
        Parameters:
            chi2 (float): Chi-squared value
            k_params (int): Number of free parameters
            
        Returns:
            float: AIC value
        """
        return chi2 + 2 * k_params
    
    def calculate_aicc(self, chi2: float, k_params: int, n_data: int) -> float:
        """
        Corrected AIC for small sample sizes.
        
        AICc = AIC + 2k(k+1)/(n-k-1)
        
        Parameters:
            chi2 (float): Chi-squared value
            k_params (int): Number of free parameters
            n_data (int): Number of data points
            
        Returns:
            float: AICc value
        """
        aic = self.calculate_aic(chi2, k_params)
        if n_data - k_params - 1 > 0:
            correction = 2 * k_params * (k_params + 1) / (n_data - k_params - 1)
            return aic + correction
        return aic
    
    def compare_models(self, models: Dict[str, Dict[str, Any]], 
                      n_data: int) -> Dict[str, Any]:
        """
        Compare multiple models using information criteria.
        
        Parameters:
            models (dict): {model_name: {'chi2': ..., 'k_params': ...}}
            n_data (int): Total number of data points
            
        Returns:
            dict: Comparison results with all criteria
        """
        self.output.log_section_header("MODEL COMPARISON STATISTICS")
        self.output.log_message("")
        self.output.log_message(f"Comparing {len(models)} models on {n_data} data points")
        self.output.log_message("")
        
        results = {}
        
        # Calculate criteria for each model
        for name, model_info in models.items():
            chi2 = model_info['chi2']
            k = model_info['k_params']
            
            bic = self.calculate_bic(chi2, n_data, k)
            aic = self.calculate_aic(chi2, k)
            aicc = self.calculate_aicc(chi2, k, n_data)
            
            results[name] = {
                'chi2': float(chi2),
                'k_params': int(k),
                'BIC': float(bic),
                'AIC': float(aic),
                'AICc': float(aicc),
                'dof': int(n_data - k)
            }
        
        # Find best model for each criterion
        best_bic = min(results.items(), key=lambda x: x[1]['BIC'])
        best_aic = min(results.items(), key=lambda x: x[1]['AIC'])
        
        # Calculate deltas relative to ΛCDM (if present)
        if 'LCDM' in results:
            lcdm = results['LCDM']
            for name in results:
                if name != 'LCDM':
                    results[name]['ΔBIC_vs_LCDM'] = float(results[name]['BIC'] - lcdm['BIC'])
                    results[name]['ΔAIC_vs_LCDM'] = float(results[name]['AIC'] - lcdm['AIC'])
        
        # Report
        self.output.log_message(f"{'Model':<25} {'k':<5} {'χ²':<10} {'BIC':<10} {'AIC':<10} {'ΔBIC':<10}")
        self.output.log_message("-" * 75)
        
        for name, res in results.items():
            delta_bic = res.get('ΔBIC_vs_LCDM', 0.0)
            self.output.log_message(
                f"{name:<25} {res['k_params']:<5} {res['chi2']:<10.2f} "
                f"{res['BIC']:<10.2f} {res['AIC']:<10.2f} {delta_bic:<10.2f}"
            )
        
        self.output.log_message("")
        self.output.log_message(f"Best by BIC: {best_bic[0]}")
        self.output.log_message(f"Best by AIC: {best_aic[0]}")
        self.output.log_message("")
        
        # Interpretation
        if 'LCDM' in results and 'QuantumAntiViscosity' in results:
            delta_bic = results['QuantumAntiViscosity']['ΔBIC_vs_LCDM']
            
            self.output.log_message("BIC Interpretation (Kass & Raftery):")
            if delta_bic < -10:
                interp = "Very strong evidence for anti-viscosity"
            elif delta_bic < -6:
                interp = "Strong evidence for anti-viscosity"
            elif delta_bic < -2:
                interp = "Positive evidence for anti-viscosity"
            elif delta_bic < 2:
                interp = "Inconclusive"
            else:
                interp = "Evidence against anti-viscosity"
            
            self.output.log_message(f"  ΔBIC = {delta_bic:.2f}")
            self.output.log_message(f"  {interp}")
            self.output.log_message("")
        
        return {
            'models': results,
            'best_bic': best_bic[0],
            'best_aic': best_aic[0],
            'n_data': int(n_data),
            'comparison_type': 'information_criteria'
        }
    
    def calculate_bayes_factor_from_chi2(self, chi2_model1: float, chi2_model0: float,
                                        k1: int, k0: int, n_data: int) -> float:
        """
        Approximate Bayes factor from BIC.
        
        BF ≈ exp(-ΔBIC/2) for nested models
        
        Parameters:
            chi2_model1 (float): Model 1 chi-squared
            chi2_model0 (float): Model 0 (null) chi-squared
            k1 (int): Parameters in model 1
            k0 (int): Parameters in model 0
            n_data (int): Number of data points
            
        Returns:
            float: Approximate Bayes factor
        """
        bic1 = self.calculate_bic(chi2_model1, n_data, k1)
        bic0 = self.calculate_bic(chi2_model0, n_data, k0)
        
        delta_bic = bic1 - bic0
        log_bf = -delta_bic / 2.0
        
        # Prevent overflow
        if log_bf > 100:
            return np.inf
        elif log_bf < -100:
            return 0.0
        
        return np.exp(log_bf)
    
    def interpret_bayes_factor(self, bf: float) -> str:
        """
        Interpret Bayes factor using Jeffreys scale.
        
        Parameters:
            bf (float): Bayes factor
            
        Returns:
            str: Interpretation
        """
        if bf > 100:
            return "Decisive evidence"
        elif bf > 10:
            return "Strong evidence"
        elif bf > 3:
            return "Substantial evidence"
        elif bf > 1:
            return "Weak evidence"
        elif bf > 1/3:
            return "Inconclusive"
        elif bf > 1/10:
            return "Weak evidence against"
        elif bf > 1/100:
            return "Strong evidence against"
        else:
            return "Decisive evidence against"
    
    def calculate_evidence_ratio(self, models: Dict[str, Dict], n_data: int) -> Dict[str, float]:
        """
        Calculate all pairwise evidence ratios.
        
        Parameters:
            models (dict): Model comparison results
            n_data (int): Number of data points
            
        Returns:
            dict: Pairwise Bayes factors
        """
        evidence_ratios = {}
        model_names = list(models.keys())
        
        for i, name1 in enumerate(model_names):
            for name2 in model_names[i+1:]:
                bf = self.calculate_bayes_factor_from_chi2(
                    models[name1]['chi2'], models[name2]['chi2'],
                    models[name1]['k_params'], models[name2]['k_params'],
                    n_data
                )
                evidence_ratios[f'{name1}_vs_{name2}'] = {
                    'bayes_factor': float(bf),
                    'log_bf': float(np.log(bf)) if bf > 0 and bf < np.inf else None,
                    'interpretation': self.interpret_bayes_factor(bf)
                }
        
        return evidence_ratios
    
    def summary_report(self, comparison_results: Dict[str, Any],
                      output_file: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive comparison report.
        
        Parameters:
            comparison_results (dict): Results from compare_models()
            output_file (str, optional): Save to file if provided
            
        Returns:
            dict: Extended results with evidence ratios
        """
        self.output.log_message("=" * 70)
        self.output.log_message("MODEL COMPARISON SUMMARY")
        self.output.log_message("=" * 70)
        self.output.log_message("")
        
        models = comparison_results['models']
        n_data = comparison_results['n_data']
        
        # Calculate all evidence ratios
        evidence_ratios = self.calculate_evidence_ratio(models, n_data)
        
        # Key result
        if 'QuantumAntiViscosity' in models and 'LCDM' in models:
            qav = models['QuantumAntiViscosity']
            lcdm = models['LCDM']
            
            self.output.log_message("CRITICAL COMPARISON: Quantum Anti-Viscosity vs ΛCDM")
            self.output.log_message("-" * 70)
            self.output.log_message(f"  Anti-Viscosity: χ² = {qav['chi2']:.2f}, k = {qav['k_params']}")
            self.output.log_message(f"  ΛCDM:          χ² = {lcdm['chi2']:.2f}, k = {lcdm['k_params']}")
            self.output.log_message("")
            self.output.log_message(f"  ΔBIC = {qav.get('ΔBIC_vs_LCDM', 0):.2f}")
            self.output.log_message(f"  ΔAIC = {qav.get('ΔAIC_vs_LCDM', 0):.2f}")
            self.output.log_message("")
            
            # Bayes factor
            bf_key = 'QuantumAntiViscosity_vs_LCDM'
            if bf_key in evidence_ratios:
                bf_info = evidence_ratios[bf_key]
                self.output.log_message(f"  Bayes Factor: {bf_info['bayes_factor']:.2e}")
                self.output.log_message(f"  Interpretation: {bf_info['interpretation']}")
                self.output.log_message("")
            
            if qav['ΔBIC_vs_LCDM'] < -10:
                self.output.log_message("✓ VERY STRONG EVIDENCE for quantum anti-viscosity")
                verdict = "decisive"
            elif qav['ΔBIC_vs_LCDM'] < -6:
                self.output.log_message("✓ STRONG EVIDENCE for quantum anti-viscosity")
                verdict = "strong"
            elif qav['ΔBIC_vs_LCDM'] < 0:
                self.output.log_message("✓ Anti-viscosity preferred")
                verdict = "positive"
            else:
                self.output.log_message("✗ ΛCDM preferred")
                verdict = "negative"
        
        self.output.log_message("")
        
        # Extended results
        extended_results = {
            **comparison_results,
            'evidence_ratios': evidence_ratios,
            'verdict': verdict if 'QuantumAntiViscosity' in models else 'inconclusive'
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(extended_results, f, indent=2)
            self.output.log_message(f"Results saved to: {output_file}")
        
        return extended_results

