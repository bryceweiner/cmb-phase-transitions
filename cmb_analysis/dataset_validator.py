"""
Dataset-Specific Validation Module
====================================

Independent analysis pipeline for individual CMB datasets.

Classes:
    DatasetValidator: Complete validation for single dataset

Enables rigorous multi-level validation where each dataset is analyzed
independently before cross-comparison.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .utils import OutputManager
from .phase_detector import PhaseTransitionDetector
from .statistics import StatisticalAnalysis


@dataclass
class DatasetInfo:
    """Metadata for a CMB dataset."""
    name: str
    beam_fwhm_arcmin: float
    multipole_range: Tuple[int, int]
    n_points: int
    resolution_relative_to_act: float = 1.0


class DatasetValidator:
    """
    Complete validation pipeline for individual CMB datasets.
    
    Performs detection and full statistical validation on a single dataset
    independently. Designed for multi-level validation where datasets are
    analyzed separately before cross-comparison.
    
    Attributes:
        output (OutputManager): For logging
        dataset_info (DatasetInfo): Dataset metadata
        
    Example:
        >>> validator = DatasetValidator()
        >>> act_info = DatasetInfo("ACT", beam_fwhm_arcmin=1.4, ...)
        >>> results = validator.analyze_dataset(ell, C_ell, C_ell_err, act_info)
        >>> print(f"Significance: {results['significance_sigma']:.1f}σ")
    """
    
    def __init__(self, output: OutputManager = None):
        """
        Initialize DatasetValidator.
        
        Parameters:
            output (OutputManager, optional): For logging
        """
        self.output = output if output is not None else OutputManager()
        self.dataset_info = None
    
    def analyze_dataset(self, 
                       ell: np.ndarray,
                       C_ell: np.ndarray,
                       C_ell_err: np.ndarray,
                       dataset_info: DatasetInfo,
                       optimized_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run complete analysis pipeline on single dataset.
        
        Performs:
        1. Detection with optimized parameters
        2. Full statistical validation (significance, LEE, bootstrap)
        3. Internal validation (split-half, jackknife)
        4. Model comparison (BIC/AIC)
        5. Robustness assessment
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Power spectrum
            C_ell_err (ndarray): Uncertainties
            dataset_info (DatasetInfo): Dataset metadata
            optimized_params (dict, optional): Pre-optimized detection parameters
                If None, uses defaults
                
        Returns:
            dict: Complete validation results including:
                - transitions: Detected multipole locations
                - errors: Uncertainties on transitions
                - significance_sigma: Statistical significance
                - robustness_score: Overall validation score
                - internal_validation: Split-half and jackknife results
                - model_comparison: BIC/AIC comparisons
        """
        self.dataset_info = dataset_info
        
        self.output.log_subsection(f"INDEPENDENT ANALYSIS: {dataset_info.name}")
        self.output.log_message(f"Dataset: {dataset_info.name}")
        self.output.log_message(f"  Beam FWHM: {dataset_info.beam_fwhm_arcmin:.1f} arcmin")
        self.output.log_message(f"  Multipole range: {dataset_info.multipole_range[0]}-{dataset_info.multipole_range[1]}")
        self.output.log_message(f"  Data points: {dataset_info.n_points}")
        
        # Use optimized parameters or defaults
        if optimized_params is None:
            # Default parameters
            window_size = 25
            prominence_factor = 0.5
            min_distance = 500
        else:
            window_size = optimized_params.get('window_size', 25)
            prominence_factor = optimized_params.get('prominence_factor', 0.5)
            min_distance = optimized_params.get('min_distance', 500)
        
        self.output.log_message(f"\nDetection parameters:")
        self.output.log_message(f"  Window size: {window_size}")
        self.output.log_message(f"  Prominence factor: {prominence_factor}")
        self.output.log_message(f"  Min distance: {min_distance}")
        
        # 1. Detect transitions
        detector = PhaseTransitionDetector(output=self.output, window_size=window_size)
        transitions, errors = detector.detect_and_analyze(ell, C_ell)
        dC_dell = detector.compute_derivative(ell, C_ell)
        
        if len(transitions) == 0:
            self.output.log_message("\n⚠ No transitions detected in this dataset")
            return self._empty_results(dataset_info.name)
        
        self.output.log_message(f"\n✓ Detected {len(transitions)} transitions:")
        for i, (trans, err) in enumerate(zip(transitions, errors)):
            self.output.log_message(f"  ℓ_{i+1} = {trans:.0f} ± {err:.0f}")
        
        # 2. Statistical validation
        stats = StatisticalAnalysis(output=self.output)
        
        # Run full analysis without Planck cross-validation (dataset-specific)
        significance = stats.full_analysis(ell, C_ell, C_ell_err, transitions, planck_data=None)
        
        # Extract key metrics
        local_sigma = significance['significance']['local_significance_sigma']
        sig_sigma = significance['look_elsewhere']['methodology_comparison']['sigma_global_min']
        
        # Extract validation metrics
        validation = significance.get('cross_dataset_validation', {})
        split_half_consistency = validation.get('split_half_analysis', {}).get('consistency_rate', 0)
        jackknife_detection = validation.get('jackknife_validation', {}).get('mean_detection_rate', 0)
        filter_stability = validation.get('systematic_stability', {}).get('mean_match_rate', 0)
        robustness_score = validation.get('validation_summary', {}).get('overall_robustness_score', 0)
        
        # Model comparison
        alt_models = significance.get('alternative_models', {})
        
        self.output.log_message(f"\n{'='*60}")
        self.output.log_message(f"RESULTS SUMMARY: {dataset_info.name}")
        self.output.log_message(f"{'='*60}")
        self.output.log_message(f"Transitions detected: {len(transitions)}")
        self.output.log_message(f"Local significance: {local_sigma:.2f}σ")
        self.output.log_message(f"Global significance (post-LEE): {sig_sigma:.2f}σ")
        self.output.log_message(f"Robustness score: {robustness_score*100:.1f}%")
        self.output.log_message(f"  Split-half: {split_half_consistency*100:.1f}%")
        self.output.log_message(f"  Jackknife: {jackknife_detection*100:.1f}%")
        self.output.log_message(f"  Filter stability: {filter_stability*100:.1f}%")
        
        # Construct results dictionary
        results = {
            'dataset_name': dataset_info.name,
            'dataset_info': {
                'beam_fwhm_arcmin': dataset_info.beam_fwhm_arcmin,
                'multipole_range': dataset_info.multipole_range,
                'n_points': dataset_info.n_points,
                'resolution_relative': dataset_info.resolution_relative_to_act
            },
            'detection': {
                'n_transitions': len(transitions),
                'transitions': transitions.tolist() if isinstance(transitions, np.ndarray) else transitions,
                'errors': errors.tolist() if isinstance(errors, np.ndarray) else errors,
                'detection_params': {
                    'window_size': window_size,
                    'prominence_factor': prominence_factor,
                    'min_distance': min_distance
                }
            },
            'significance': {
                'local_sigma': float(local_sigma),
                'global_sigma': float(sig_sigma),
                'chi2_smooth': significance['significance'].get('chi2_smooth', 0),
                'chi2_discontinuous': significance['significance'].get('chi2_discontinuous', 0),
                'delta_chi2': significance['significance'].get('delta_chi2', 0)
            },
            'internal_validation': {
                'split_half_consistency': float(split_half_consistency),
                'jackknife_mean_detection': float(jackknife_detection),
                'filter_stability': float(filter_stability),
                'robustness_score': float(robustness_score)
            },
            'model_comparison': alt_models,
            'full_statistical_results': significance
        }
        
        return results
    
    def _empty_results(self, dataset_name: str) -> Dict[str, Any]:
        """
        Return empty results structure when no transitions detected.
        
        Parameters:
            dataset_name (str): Name of dataset
            
        Returns:
            dict: Empty results structure
        """
        return {
            'dataset_name': dataset_name,
            'dataset_info': {
                'beam_fwhm_arcmin': self.dataset_info.beam_fwhm_arcmin if self.dataset_info else 0,
                'multipole_range': self.dataset_info.multipole_range if self.dataset_info else (0, 0),
                'n_points': self.dataset_info.n_points if self.dataset_info else 0,
                'resolution_relative': self.dataset_info.resolution_relative_to_act if self.dataset_info else 1.0
            },
            'detection': {
                'n_transitions': 0,
                'transitions': [],
                'errors': [],
                'detection_params': {}
            },
            'significance': {
                'local_sigma': 0.0,
                'global_sigma': 0.0,
                'chi2_smooth': 0,
                'chi2_discontinuous': 0,
                'delta_chi2': 0
            },
            'internal_validation': {
                'split_half_consistency': 0.0,
                'jackknife_mean_detection': 0.0,
                'filter_stability': 0.0,
                'robustness_score': 0.0
            },
            'model_comparison': {},
            'full_statistical_results': {}
        }
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate human-readable report from validation results.
        
        Parameters:
            results (dict): Results from analyze_dataset()
            
        Returns:
            str: Formatted report text
        """
        dataset_name = results['dataset_name']
        n_trans = results['detection']['n_transitions']
        
        if n_trans == 0:
            return f"\n{'='*60}\n{dataset_name}: NO TRANSITIONS DETECTED\n{'='*60}\n"
        
        transitions = results['detection']['transitions']
        errors = results['detection']['errors']
        sig = results['significance']
        val = results['internal_validation']
        
        report = []
        report.append(f"\n{'='*60}")
        report.append(f"VALIDATION REPORT: {dataset_name}")
        report.append(f"{'='*60}\n")
        
        report.append("DETECTIONS:")
        for i, (trans, err) in enumerate(zip(transitions, errors)):
            report.append(f"  ℓ_{i+1} = {trans:.0f} ± {err:.0f}")
        
        report.append(f"\nSTATISTICAL SIGNIFICANCE:")
        report.append(f"  Local: {sig['local_sigma']:.2f}σ")
        report.append(f"  Global (post-LEE): {sig['global_sigma']:.2f}σ")
        report.append(f"  Δχ²: {sig['delta_chi2']:.1f}")
        
        report.append(f"\nINTERNAL VALIDATION:")
        report.append(f"  Split-half: {val['split_half_consistency']*100:.1f}%")
        report.append(f"  Jackknife: {val['jackknife_mean_detection']*100:.1f}%")
        report.append(f"  Filter stability: {val['filter_stability']*100:.1f}%")
        report.append(f"  Overall robustness: {val['robustness_score']*100:.1f}%")
        
        # Assessment
        report.append(f"\nASSESSMENT:")
        if sig['global_sigma'] >= 5.0 and val['robustness_score'] >= 0.5:
            report.append(f"  ✓ STRONG DETECTION in {dataset_name}")
            report.append(f"    Significance ≥5σ, robustness ≥50%")
        elif sig['global_sigma'] >= 3.0 and val['robustness_score'] >= 0.4:
            report.append(f"  ~ MODERATE DETECTION in {dataset_name}")
            report.append(f"    Significance ≥3σ, robustness ≥40%")
        else:
            report.append(f"  ✗ WEAK/MARGINAL in {dataset_name}")
            report.append(f"    Significance <3σ or robustness <40%")
        
        report.append(f"{'='*60}\n")
        
        return "\n".join(report)
    
    def compare_datasets(self, results_list: list) -> Dict[str, Any]:
        """
        Compare validation results across multiple datasets.
        
        Parameters:
            results_list (list): List of results dicts from analyze_dataset()
            
        Returns:
            dict: Cross-dataset comparison metrics
        """
        if len(results_list) < 2:
            return {'error': 'Need at least 2 datasets for comparison'}
        
        comparison = {
            'n_datasets': len(results_list),
            'datasets': [r['dataset_name'] for r in results_list],
            'detections': {},
            'significance': {},
            'robustness': {},
            'agreement': {}
        }
        
        # Collect metrics per dataset
        for result in results_list:
            name = result['dataset_name']
            comparison['detections'][name] = result['detection']['n_transitions']
            comparison['significance'][name] = result['significance']['global_sigma']
            comparison['robustness'][name] = result['internal_validation']['robustness_score']
        
        # Assess agreement
        n_detected = [r['detection']['n_transitions'] for r in results_list]
        all_detect = all(n > 0 for n in n_detected)
        same_count = len(set(n_detected)) == 1
        
        comparison['agreement']['all_datasets_detect'] = all_detect
        comparison['agreement']['consistent_count'] = same_count
        
        # Check significance levels
        sigs = [r['significance']['global_sigma'] for r in results_list]
        comparison['agreement']['all_significant_3sigma'] = all(s >= 3.0 for s in sigs)
        comparison['agreement']['all_significant_5sigma'] = all(s >= 5.0 for s in sigs)
        
        # Overall assessment
        if all_detect and comparison['agreement']['all_significant_3sigma']:
            comparison['overall_assessment'] = 'STRONG: All datasets detect with ≥3σ'
        elif all_detect:
            comparison['overall_assessment'] = 'MODERATE: All detect but some <3σ'
        else:
            comparison['overall_assessment'] = 'WEAK: Not all datasets detect features'
        
        return comparison

