"""
CMB Phase Transition Analysis Package
======================================

Object-oriented implementation of analysis code for:
'Pre-Recombination Spacetime Expansion Events Resolve the 
Cosmological Constant Problem Through Information Physics'

This package provides a complete pipeline for detecting quantum phase transitions
in CMB E-mode polarization data and deriving the cosmological constant from
information-theoretic first principles.

Main Components:
    - TheoreticalCalculations: Information theory and quantum physics
    - DataLoader: CMB data acquisition from public archives
    - PhaseTransitionDetector: Transition detection and analysis
    - StatisticalAnalysis: Statistical validation (bootstrap, LEE, etc.)
    - CosmologicalConstant: Λ derivation from first principles
    - TemporalCascade: Expansion history modeling
    - CosmologicalTensions: H₀, S₈, BAO resolution
    - Visualizer: Figure generation
    - OutputManager: Logging and I/O

Quick Start:
    >>> from cmb_analysis import run_analysis
    >>> results = run_analysis()

Advanced Usage:
    >>> from cmb_analysis import DataLoader, PhaseTransitionDetector
    >>> loader = DataLoader()
    >>> ell, C_ell, C_ell_err = loader.load_act_dr6()
    >>> detector = PhaseTransitionDetector()
    >>> transitions, errors = detector.detect_and_analyze(ell, C_ell)
"""

__version__ = "1.0.0"
__author__ = "Bryce Weiner"
__email__ = "bryce.weiner@informationphysicsinstitute.net"
__paper__ = "Pre-Recombination Spacetime Expansion Events Resolve the " \
            "Cosmological Constant Problem Through Information Physics"

# Standard library imports
import numpy as np

# Import core classes
from .constants import *
from .utils import OutputManager, format_scientific, format_error, safe_divide
from .theoretical import TheoreticalCalculations
from .data_loader import DataLoader
from .phase_detector import PhaseTransitionDetector
from .statistics import StatisticalAnalysis
from .temporal_cascade import TemporalCascade
from .cosmological_constant import CosmologicalConstant
# Tensions module removed - replaced with pure prediction modules
# from .tensions import CosmologicalTensions
from .visualizer import Visualizer


def run_analysis(output_dir: str = ".", extend: bool = False,
                 validation_level: str = "basic", run_monte_carlo: bool = False):
    """
    Run complete analysis pipeline.
    
    This convenience function orchestrates the entire analysis:
    
    Core analysis (always runs):
    1. Loads ACT DR6 CMB data
    2. Detects phase transitions
    3. Performs statistical validation
    4. Cross-dataset validation
    
    Extended analysis (only if extend=True):
    5. Calculates information processing rates (gamma)
    6. Derives cosmological constant
    7. Analyzes temporal cascade
    8. Resolves cosmological tensions
    9. Generates supplementary figures
    
    Multi-level validation (controlled by validation_level):
    - basic: Standard single-dataset validation (default)
    - standard: Add independent Planck analysis (Level 1)
    - comprehensive: All 3 levels (independent + combined + cross-validation)
    
    Parameters:
        output_dir (str): Directory for output files (default: current directory)
        extend (bool): Include extended theoretical analysis (default: False)
        validation_level (str): Validation rigor level (default: "basic")
        run_monte_carlo (bool): Run MC null simulations (default: False)
        
    Returns:
        dict: Complete analysis results
        
    Example:
        >>> # Core analysis only
        >>> results = run_analysis(output_dir="./results")
        >>> print(f"Detected {len(results['transitions'])} transitions")
        
        >>> # With extended analysis and comprehensive validation
        >>> results = run_analysis(output_dir="./results", extend=True, validation_level="comprehensive")
        >>> print(f"Λ₀ = {results['lambda_final']:.3e} m⁻²")
    """
    # Initialize components
    output = OutputManager(output_dir=output_dir)
    output.log_section_header("CMB PHASE TRANSITION ANALYSIS")
    output.log_message(f"Version: {__version__}")
    output.log_message(f"Paper: {__paper__}")
    output.log_message(f"Analysis mode: {'EXTENDED' if extend else 'CORE (observational only)'}")
    output.log_message("")
    
    # 1. Load data
    loader = DataLoader(output=output)
    ell, C_ell, C_ell_err = loader.load_act_dr6()
    loader.validate_data(ell, C_ell, C_ell_err)
    
    # Load Planck 2018 for cross-validation (optional)
    planck_data = loader.load_planck_2018()
    
    # 2. Detect transitions
    detector = PhaseTransitionDetector(output=output)
    transitions, errors = detector.detect_and_analyze(ell, C_ell)
    dC_dell = detector.compute_derivative(ell, C_ell)
    
    # 3. Core transition analysis (always runs - multipoles and physical scales)
    # Extended theoretical analysis (gamma extraction, expansion factors) is conditional
    if extend:
        output.log_message("Running extended theoretical analysis (gamma extraction, expansion factors)...")
        analysis_results = detector.analyze_transitions(transitions, errors)
        gamma_values = np.array(analysis_results['gamma_scale_dependent'])
    else:
        output.log_message("Skipping extended theoretical analysis (use --extend flag to enable)")
        analysis_results = None
        gamma_values = None
    
    # 4. Statistical validation (with optional Planck cross-validation)
    stats = StatisticalAnalysis(output=output)
    significance = stats.full_analysis(ell, C_ell, C_ell_err, transitions, planck_data)
    
    # Build advanced stats dictionary for figures
    advanced_stats = {
        'bootstrap_resampling': significance.get('bootstrap_resampling', {}),
        'alternative_models': significance.get('alternative_models', {}),
        'cross_dataset_validation': significance.get('cross_dataset_validation', {})
    }
    
    # 4b. Multi-level validation (if requested)
    multilevel_results = None
    if validation_level in ["standard", "comprehensive"]:
        from .dataset_validator import DatasetValidator, DatasetInfo
        from .parameter_optimizer import ParameterOptimizer
        from .dataset_combiner import DatasetCombiner
        
        output.log_section_header(f"MULTI-LEVEL VALIDATION ({validation_level.upper()})")
        
        # Level 1: Independent dataset analysis
        validator = DatasetValidator(output=output)
        
        # ACT analysis (already done above, create summary)
        act_info = DatasetInfo("ACT_DR6", beam_fwhm_arcmin=1.4, 
                              multipole_range=(int(ell[0]), int(ell[-1])), 
                              n_points=len(ell), resolution_relative_to_act=1.0)
        
        # Extract validation summary
        validation_summary = significance.get('cross_dataset_validation', {}).get('validation_summary', {})
        
        act_results = {
            'dataset_name': 'ACT_DR6',
            'dataset_info': {'beam_fwhm_arcmin': 1.4, 'multipole_range': (int(ell[0]), int(ell[-1])), 
                            'n_points': len(ell), 'resolution_relative': 1.0},
            'detection': {'n_transitions': len(transitions), 'transitions': transitions.tolist(), 
                         'errors': errors.tolist()},
            'significance': {'local_sigma': significance['significance']['local_significance_sigma'],
                           'global_sigma': significance['look_elsewhere']['methodology_comparison']['sigma_global_min']},
            'internal_validation': {
                'split_half_consistency': validation_summary.get('split_half_consistency', 0),
                'jackknife_mean_detection': validation_summary.get('jackknife_mean_detection', 0),
                'filter_stability': validation_summary.get('filter_stability', 0),
                'robustness_score': validation_summary.get('overall_robustness_score', 0)
            }
        }
        
        # Planck independent analysis
        if planck_data is not None:
            ell_planck, C_ell_planck, C_ell_err_planck = planck_data
            
            # Optimize parameters for Planck
            optimizer = ParameterOptimizer(output=output)
            planck_params = optimizer.optimize_for_planck(ell_planck, C_ell_planck, C_ell_err_planck)
            
            # Run independent Planck analysis
            planck_info = DatasetInfo("Planck_2018", beam_fwhm_arcmin=7.3,
                                     multipole_range=(int(ell_planck[0]), int(ell_planck[-1])),
                                     n_points=len(ell_planck), resolution_relative_to_act=5.21)
            
            planck_results = validator.analyze_dataset(ell_planck, C_ell_planck, C_ell_err_planck, 
                                                      planck_info, planck_params)
            
            # Compare datasets
            comparison = validator.compare_datasets([act_results, planck_results])
        else:
            planck_results = None
            comparison = None
            output.log_message("⚠ Planck data not available for Level 1 validation")
        
        multilevel_results = {
            'validation_level': validation_level,
            'level1_independent': {'ACT': act_results, 'Planck': planck_results, 'comparison': comparison}
        }
        
        # Level 2 & 3: Combined analysis and cross-validation (comprehensive only)
        if validation_level == "comprehensive" and planck_data is not None and planck_results is not None:
            from .cross_validator import CrossValidator
            
            # Level 2: Combined dataset analysis - COMPLETE IMPLEMENTATION
            output.log_section_header("LEVEL 2: COMBINED DATASET ANALYSIS")
            
            combiner = DatasetCombiner(output=output)
            act_data_dict = {'ell': ell, 'C_ell': C_ell, 'C_ell_err': C_ell_err}
            planck_data_dict = {'ell': ell_planck, 'C_ell': C_ell_planck, 'C_ell_err': C_ell_err_planck}
            
            # Merge datasets with all three methods
            # Resolution ratio: Planck FWHM / ACT FWHM = 7.3' / 1.4' = 5.21
            resolution_ratio = 7.3 / 1.4
            merged_datasets = combiner.merge_all_methods(act_data_dict, planck_data_dict, 
                                                         None, None, "ACT_DR6", "Planck_2018",
                                                         resolution_ratio=resolution_ratio)
            
            # Run complete detection and validation on each merged dataset
            combined_analysis_results = {}
            
            for method_name, merged_data in merged_datasets.items():
                output.log_subsection(f"ANALYZING COMBINED DATASET: {method_name.upper()}")
                
                # Extract merged data
                ell_merged = merged_data['ell']
                C_ell_merged = merged_data['C_ell']
                C_err_merged = merged_data['C_ell_err']
                
                output.log_message(f"Combined dataset: {len(ell_merged)} points")
                output.log_message(f"Range: {ell_merged[0]:.0f}-{ell_merged[-1]:.0f}")
                
                # Run detection on combined dataset
                detector_combined = PhaseTransitionDetector(output=output, window_size=25)
                transitions_combined, errors_combined = detector_combined.detect_and_analyze(
                    ell_merged, C_ell_merged
                )
                
                output.log_message(f"Detected {len(transitions_combined)} transitions in combined data")
                for i, (t, e) in enumerate(zip(transitions_combined, errors_combined)):
                    output.log_message(f"  ℓ_{i+1} = {t:.0f} ± {e:.0f}")
                
                # Run full statistical validation on combined dataset
                stats_combined = StatisticalAnalysis(output=output)
                significance_combined = stats_combined.full_analysis(
                    ell_merged, C_ell_merged, C_err_merged, transitions_combined, planck_data=None
                )
                
                # Extract key metrics
                local_sigma_combined = significance_combined['significance']['local_significance_sigma']
                global_sigma_combined = significance_combined['look_elsewhere']['methodology_comparison']['sigma_global_min']
                
                # Calculate significance boost/reduction
                act_global_sigma = act_results['significance']['global_sigma']
                planck_global_sigma = planck_results['significance']['global_sigma']
                max_individual_sigma = max(act_global_sigma, planck_global_sigma)
                
                boost_factor = global_sigma_combined / max_individual_sigma if max_individual_sigma > 0 else 0
                
                output.log_message(f"\nSignificance comparison:")
                output.log_message(f"  ACT alone: {act_global_sigma:.2f}σ")
                output.log_message(f"  Planck alone: {planck_global_sigma:.2f}σ")
                output.log_message(f"  Combined ({method_name}): {global_sigma_combined:.2f}σ")
                output.log_message(f"  Boost factor: {boost_factor:.2f}×")
                
                # Interpretation
                if boost_factor > 1.0:
                    output.log_message(f"  ✓ Combined analysis INCREASES significance (features likely real)")
                elif boost_factor > 0.8:
                    output.log_message(f"  ~ Combined analysis maintains significance (consistent)")
                else:
                    output.log_message(f"  ✗ Combined analysis DECREASES significance (inconsistent features)")
                
                # Store complete results
                combined_analysis_results[method_name] = {
                    'merged_data': {
                        'ell': ell_merged.tolist(),
                        'n_points': len(ell_merged),
                        'multipole_range': (float(ell_merged[0]), float(ell_merged[-1]))
                    },
                    'detection': {
                        'n_transitions': len(transitions_combined),
                        'transitions': transitions_combined.tolist(),
                        'errors': errors_combined.tolist()
                    },
                    'significance': {
                        'local_sigma': float(local_sigma_combined),
                        'global_sigma': float(global_sigma_combined),
                        'chi2_smooth': significance_combined['significance'].get('chi2_smooth', 0),
                        'chi2_discontinuous': significance_combined['significance'].get('chi2_discontinuous', 0),
                        'delta_chi2': significance_combined['significance'].get('delta_chi2', 0)
                    },
                    'comparison_to_individual': {
                        'act_global_sigma': float(act_global_sigma),
                        'planck_global_sigma': float(planck_global_sigma),
                        'combined_global_sigma': float(global_sigma_combined),
                        'boost_factor': float(boost_factor),
                        'interpretation': 'boost' if boost_factor > 1.0 else ('consistent' if boost_factor > 0.8 else 'reduced')
                    },
                    'full_statistical_results': significance_combined
                }
            
            multilevel_results['level2_combined'] = combined_analysis_results
            
            # Assess overall Level 2 results
            all_boost = all(r['comparison_to_individual']['boost_factor'] > 1.0 
                          for r in combined_analysis_results.values())
            
            output.log_message(f"\n{'='*60}")
            output.log_message("LEVEL 2 ASSESSMENT")
            output.log_message(f"{'='*60}")
            if all_boost:
                output.log_message("✓ ALL merging methods show significance boost")
                output.log_message("  Strong evidence: Features are real and consistent")
            else:
                output.log_message("⚠ Not all methods show boost")
                output.log_message("  Mixed evidence: Check method-specific results")
            
            # Level 3: Cross-validation
            output.log_section_header("LEVEL 3: CROSS-VALIDATION")
            
            cv = CrossValidator(output=output)
            planck_transitions = np.array(planck_results['detection']['transitions'])
            cv_results = cv.run_all_tests(act_data_dict, planck_data_dict, 
                                         transitions, planck_transitions)
            multilevel_results['level3_cross_validation'] = cv_results
        
        output.log_message(f"\n✓ {validation_level.capitalize()} validation complete")
    
    # 4c. Monte Carlo Null Simulations and MCMC (if requested)
    monte_carlo_results = None
    mcmc_results = None
    false_positive_results = None
    if run_monte_carlo:
        from .monte_carlo_simulator import MonteCarloSimulator
        from .false_positive_analysis import FalsePositiveAnalyzer
        from .mcmc_analysis import MCMCAnalyzer
        
        output.log_section_header("MONTE CARLO NULL SIMULATIONS AND MCMC")
        
        # Initialize simulator with MPS support
        simulator = MonteCarloSimulator(output=output, f_sky=0.4, use_mps=True)
        
        # ACT null simulations
        output.log_message("Running ACT null ensemble (10000 realizations)...")
        act_ensemble = simulator.run_act_simulations(
            ell, C_ell, C_ell_err, n_realizations=10000
        )
        
        # False positive analysis for ACT
        analyzer = FalsePositiveAnalyzer(output=output)
        act_fpr_results = analyzer.analyze_ensemble(
            act_ensemble,
            observed_transitions=transitions,
            n_observed=len(transitions)
        )
        
        # Store ACT results
        monte_carlo_results = {
            'act': {
                'ensemble_stats': {
                    'n_realizations': act_ensemble['n_realizations'],
                    'mean_null_transitions': act_ensemble.get('mean_detections', 0),
                    'std_null_transitions': act_ensemble.get('std_detections', 0)
                },
                'false_positive_analysis': act_fpr_results
            }
        }
        
        # Empirical LEE validation for ACT
        if 'null_significances' in act_ensemble and len(act_ensemble['null_significances']) > 0:
            from scipy import stats as scipy_stats
            local_sigma = significance['significance']['local_significance_sigma']
            local_pvalue = scipy_stats.norm.sf(local_sigma) * 2  # two-tailed
            null_pvalues = np.array([scipy_stats.norm.sf(s) * 2 for s in act_ensemble['null_significances']])
            
            # Empirical global p-value
            n_nulls = len(null_pvalues)
            n_more_significant = np.sum(null_pvalues <= local_pvalue)
            p_global_empirical = (n_more_significant + 1) / (n_nulls + 1)
            
            # Convert to sigma
            if p_global_empirical > 0 and p_global_empirical < 1:
                from scipy.special import erfcinv
                sigma_global_empirical = np.sqrt(2) * erfcinv(2 * p_global_empirical)
            else:
                sigma_global_empirical = 100.0  # Very high significance
            
            empirical_lee = {
                'method': 'empirical_from_null',
                'n_null_simulations': n_nulls,
                'n_more_significant': int(n_more_significant),
                'p_global_empirical': float(p_global_empirical),
                'sigma_global_empirical': float(sigma_global_empirical),
                'local_pvalue': float(local_pvalue)
            }
            monte_carlo_results['act']['empirical_lee'] = empirical_lee
            
            output.log_message(f"\n✓ ACT Empirical LEE: {empirical_lee['sigma_global_empirical']:.2f}σ")
        
            # Planck null simulations (if comprehensive validation)
            if planck_data is not None and validation_level == "comprehensive":
                output.log_message("\nRunning Planck null ensemble (10000 realizations)...")
                ell_planck, C_ell_planck, C_ell_err_planck = planck_data
                planck_transitions = np.array(planck_results['detection']['transitions'])
                
                planck_ensemble = simulator.run_planck_simulations(
                    ell_planck, C_ell_planck, C_ell_err_planck, n_realizations=10000
                )
            
            planck_fpr_results = analyzer.analyze_ensemble(
                planck_ensemble,
                observed_transitions=planck_transitions,
                n_observed=len(planck_transitions)
            )
            
            monte_carlo_results['planck'] = {
                'ensemble_stats': {
                    'n_realizations': planck_ensemble['n_realizations'],
                    'mean_null_transitions': planck_ensemble.get('mean_detections', 0),
                    'std_null_transitions': planck_ensemble.get('std_detections', 0)
                },
                'false_positive_analysis': planck_fpr_results
            }
            
            # Empirical LEE for Planck
            if 'null_significances' in planck_ensemble and len(planck_ensemble['null_significances']) > 0:
                planck_local_sigma = planck_results['significance']['local_sigma']
                planck_local_pvalue = scipy_stats.norm.sf(planck_local_sigma) * 2
                planck_null_pvalues = np.array([scipy_stats.norm.sf(s) * 2 for s in planck_ensemble['null_significances']])
                
                # Empirical global p-value
                planck_n_nulls = len(planck_null_pvalues)
                planck_n_more_significant = np.sum(planck_null_pvalues <= planck_local_pvalue)
                planck_p_global_empirical = (planck_n_more_significant + 1) / (planck_n_nulls + 1)
                
                # Convert to sigma
                if planck_p_global_empirical > 0 and planck_p_global_empirical < 1:
                    planck_sigma_global_empirical = np.sqrt(2) * erfcinv(2 * planck_p_global_empirical)
                else:
                    planck_sigma_global_empirical = 100.0
                
                planck_empirical_lee = {
                    'method': 'empirical_from_null',
                    'n_null_simulations': planck_n_nulls,
                    'n_more_significant': int(planck_n_more_significant),
                    'p_global_empirical': float(planck_p_global_empirical),
                    'sigma_global_empirical': float(planck_sigma_global_empirical),
                    'local_pvalue': float(planck_local_pvalue)
                }
                monte_carlo_results['planck']['empirical_lee'] = planck_empirical_lee
                
                output.log_message(f"✓ Planck Empirical LEE: {planck_empirical_lee['sigma_global_empirical']:.2f}σ")
        
        output.log_message(f"\n✓ Monte Carlo null simulations complete")
        
        # MCMC Bayesian parameter estimation
        output.log_message("\n" + "="*70)
        output.log_message("Running MCMC Bayesian parameter estimation...")
        output.log_message("="*70 + "\n")
        
        mcmc_analyzer = MCMCAnalyzer(ell, C_ell, C_ell_err, output=output)
        
        # Run MCMC with initial guesses from detection
        mcmc_results = mcmc_analyzer.run_mcmc(
            n_transitions=len(transitions),
            n_walkers=32,
            n_steps=5000,
            n_burn=1000,
            initial_transitions=transitions
        )
        
        # Model comparison (test 0-5 transitions)
        output.log_message("\n" + "="*70)
        output.log_message("Running MCMC model comparison...")
        output.log_message("="*70 + "\n")
        
        model_comparison_results = mcmc_analyzer.model_comparison(max_transitions=5)
        mcmc_results['model_comparison'] = model_comparison_results
        
        output.log_message(f"\n✓ MCMC analysis complete")
        
        # Store in advanced_stats for access by other components
        advanced_stats['monte_carlo'] = monte_carlo_results
        advanced_stats['mcmc'] = mcmc_results
    else:
        output.log_message("\nSkipping Monte Carlo and MCMC analysis (use --run-monte-carlo to enable)")
        output.log_message("  Note: This is the most rigorous validation")
    
    # 5-7. Extended analysis: Temporal cascade, Cosmological constant, Tensions
    if extend:
        output.log_message("Running extended cosmological analysis...")
        
        # 5. Temporal cascade
        cascade = TemporalCascade(output=output)
        cascade_results = cascade.calculate_model(transitions, errors, gamma_values)
        
        # 6. Cosmological constant
        expansion_factors = np.array(analysis_results['expansion_factors'])
        lambda_calc = CosmologicalConstant(output=output)
        lambda_results = lambda_calc.calculate_detailed(expansion_factors, cascade_results)
        
        # 7. Tensions analysis removed
        # Legacy fitting-based approach replaced with pure prediction modules
        # Use --gamma and --bao flags for theory-driven analysis
        tension_results = None
        tension_results_combined = None
        
        output.log_message("")
        output.log_message("NOTE: Tensions analysis moved to modular prediction modules")
        output.log_message("  Use --gamma for theoretical γ(z) and Λ_eff(z)")
        output.log_message("  Use --bao for BAO prediction tests")
        output.log_message("")
    else:
        output.log_message("Skipping extended cosmological analysis (Lambda, tensions)")
        cascade_results = None
        lambda_results = None
        tension_results = None
        tension_results_combined = None
    
    # 8. Visualize (conditional on extend flag)
    viz = Visualizer(output=output, figure_dir=output_dir)
    if extend:
        # Generate all figures including extended analysis
        figures = viz.create_all_figures(ell, C_ell, C_ell_err, dC_dell,
                                         transitions, errors, gamma_values,
                                         lambda_results, cascade_results, tension_results,
                                         advanced_stats, multilevel_results)
    else:
        # Generate core figures only (detection and statistics)
        output.log_message("Generating core analysis figures only")
        figures = viz.create_core_figures(ell, C_ell, C_ell_err, dC_dell,
                                          transitions, errors, advanced_stats, multilevel_results)
    
    # Build advanced_statistical_analysis section (conditional on extend flag)
    advanced_statistical_analysis = {
        # Detailed look-elsewhere calculation (always included)
        'detailed_look_elsewhere': significance.get('look_elsewhere', {}),
        
        # Statistical validation methods (always included)
        'bootstrap_resampling': advanced_stats.get('bootstrap_resampling', {}),
        'cross_dataset_validation': advanced_stats.get('cross_dataset_validation', {}),
        'alternative_models': advanced_stats.get('alternative_models', {}),
    }
    
    # Add multi-level validation results if performed
    if multilevel_results is not None:
        advanced_statistical_analysis['multilevel_validation'] = multilevel_results
    
    # Add extended analysis results if requested
    if extend:
        advanced_statistical_analysis.update({
            # Temporal cascade model
            'temporal_cascade': cascade_results,
            
            # Cosmological tensions resolution
            'cosmological_tensions': tension_results,
            
            # Cosmological constant calculation
            'cosmological_constant': lambda_results,
            
            # Extended figure paths
            'supplementary_figures': {
                's1': figures.get('supp_s1', ''),
                's2': figures.get('supp_s2', ''),
                's3': figures.get('supp_s3', '')
            },
            'tension_figures': {
                'bao': figures.get('bao', ''),
                's8': figures.get('s8', ''),
                'matter_density': figures.get('matter_density', '')
            }
        })
    
    # Compile results (structure depends on analysis mode)
    results = {
        'version': __version__,
        'analysis_type': 'extended' if extend else 'core',
        'transitions': transitions.tolist(),
        'errors': errors.tolist(),
        'significance': significance,
        'advanced_statistical_analysis': advanced_statistical_analysis,
        'figures': figures
    }
    
    # Add extended analysis results if requested
    if extend:
        results.update({
            'analysis': analysis_results,
            'lambda_results': lambda_results,
            'cascade_results': cascade_results,
            'tension_results': tension_results_combined if tension_results_combined else tension_results
        })
    
    # Save results
    output.save_json_output(results)
    output.print_summary(results)
    output.close()
    
    return results


__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__email__',
    '__paper__',
    
    # Classes
    'TheoreticalCalculations',
    'DataLoader',
    'PhaseTransitionDetector',
    'OutputManager',
    'StatisticalAnalysis',
    'CosmologicalConstant',
    'TemporalCascade',
    # 'CosmologicalTensions',  # Removed - use --gamma and --bao flags
    'Visualizer',
    
    # Utilities
    'format_scientific',
    'format_error',
    'safe_divide',
    
    # Main function
    'run_analysis',
]

