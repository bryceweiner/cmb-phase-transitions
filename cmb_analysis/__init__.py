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

# Import core classes
from .constants import *
from .utils import OutputManager, format_scientific, format_error, safe_divide
from .theoretical import TheoreticalCalculations
from .data_loader import DataLoader
from .phase_detector import PhaseTransitionDetector
from .statistics import StatisticalAnalysis
from .temporal_cascade import TemporalCascade
from .cosmological_constant import CosmologicalConstant
from .tensions import CosmologicalTensions
from .visualizer import Visualizer


def run_analysis(output_dir: str = "."):
    """
    Run complete analysis pipeline.
    
    This convenience function orchestrates the entire analysis:
    1. Loads ACT DR6 CMB data
    2. Detects phase transitions
    3. Performs statistical validation
    4. Calculates cosmological constant
    5. Analyzes temporal cascade
    6. Resolves cosmological tensions
    7. Generates all figures
    8. Saves results to JSON
    
    Parameters:
        output_dir (str): Directory for output files (default: current directory)
        
    Returns:
        dict: Complete analysis results
        
    Example:
        >>> results = run_analysis(output_dir="./my_results")
        >>> print(f"Detected {len(results['transitions'])} transitions")
        >>> print(f"Λ₀ = {results['lambda_final']:.3e} m⁻²")
    """
    # Initialize components
    output = OutputManager(output_dir=output_dir)
    output.log_section_header("CMB PHASE TRANSITION ANALYSIS")
    output.log_message(f"Version: {__version__}")
    output.log_message(f"Paper: {__paper__}")
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
    
    # 3. Analyze transitions
    analysis_results = detector.analyze_transitions(transitions, errors)
    gamma_values = np.array(analysis_results['gamma_scale_dependent'])
    
    # 4. Statistical validation (with optional Planck cross-validation)
    stats = StatisticalAnalysis(output=output)
    significance = stats.full_analysis(ell, C_ell, C_ell_err, transitions, planck_data)
    
    # Build advanced stats dictionary for figures
    advanced_stats = {
        'bootstrap_resampling': significance.get('bootstrap_resampling', {}),
        'alternative_models': significance.get('alternative_models', {}),
        'cross_dataset_validation': significance.get('cross_dataset_validation', {})
    }
    
    # 5. Temporal cascade
    cascade = TemporalCascade(output=output)
    cascade_results = cascade.calculate_model(transitions, errors, gamma_values)
    
    # 6. Cosmological constant
    expansion_factors = np.array(analysis_results['expansion_factors'])
    lambda_calc = CosmologicalConstant(output=output)
    lambda_results = lambda_calc.calculate_detailed(expansion_factors, cascade_results)
    
    # 7. Tensions
    tensions = CosmologicalTensions(output=output)
    tension_results = tensions.calculate_all()
    
    # 8. Visualize
    viz = Visualizer(output=output, figure_dir=output_dir)
    figures = viz.create_all_figures(ell, C_ell, C_ell_err, dC_dell,
                                     transitions, errors, gamma_values,
                                     lambda_results, cascade_results, tension_results,
                                     advanced_stats)
    
    # Build advanced_statistical_analysis section (matching original structure)
    advanced_statistical_analysis = {
        # Detailed look-elsewhere calculation
        'detailed_look_elsewhere': significance.get('look_elsewhere', {}),
        
        # Statistical validation methods
        'bootstrap_resampling': advanced_stats.get('bootstrap_resampling', {}),
        'cross_dataset_validation': advanced_stats.get('cross_dataset_validation', {}),
        'alternative_models': advanced_stats.get('alternative_models', {}),
        
        # Temporal cascade model
        'temporal_cascade': cascade_results,
        
        # Cosmological tensions resolution
        'cosmological_tensions': tension_results,
        
        # Cosmological constant calculation
        'cosmological_constant': lambda_results,
        
        # Figure paths
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
    }
    
    # Compile results (matching original structure)
    results = {
        'version': __version__,
        'transitions': transitions.tolist(),
        'errors': errors.tolist(),
        'analysis': analysis_results,
        'significance': significance,
        'lambda_results': lambda_results,
        'cascade_results': cascade_results,
        'tension_results': tension_results,
        'advanced_statistical_analysis': advanced_statistical_analysis,
        'figures': figures
    }
    
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
    'CosmologicalTensions',
    'Visualizer',
    
    # Utilities
    'format_scientific',
    'format_error',
    'safe_divide',
    
    # Main function
    'run_analysis',
]

