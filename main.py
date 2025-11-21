#!/usr/bin/env python3
"""
CMB Phase Transition Analysis - Main Runner
============================================

Main entry point for the analysis pipeline. Orchestrates all components
to perform complete analysis from data loading through figure generation.

Usage:
    python main.py                     # Run with defaults
    python main.py --output-dir ./results  # Custom output directory
    python main.py --quiet             # Suppress progress messages
    python main.py --version           # Show version

For more information, see README.md or examples/quickstart.md
"""

import argparse
import sys
from pathlib import Path

# Add package to path if running from source
sys.path.insert(0, str(Path(__file__).parent))

from cmb_analysis import (
    __version__, __paper__,
    run_analysis
)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='CMB E-mode Phase Transition Analysis',
        epilog=f'Paper: {__paper__}',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'CMB Analysis v{__version__}'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Output directory for results (default: ./results)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages (errors still shown)'
    )
    
    parser.add_argument(
        '--extend',
        action='store_true',
        help='Include extended theoretical analysis '
             '(gamma extraction, Lambda calculation, cosmological tensions). '
             'Default: core observational analysis only'
    )
    
    parser.add_argument(
        '--validation-level',
        choices=['basic', 'standard', 'comprehensive'],
        default='basic',
        help='Level of statistical validation:\n'
             '  basic: Current single-dataset validation (default)\n'
             '  standard: Add independent Planck validation (Level 1)\n'
             '  comprehensive: All 3 levels (independent + combined + cross-validation)'
    )
    
    parser.add_argument(
        '--run-monte-carlo',
        action='store_true',
        help='Run Monte Carlo null simulations (adds 2-8 hours runtime depending on hardware)'
    )
    
    parser.add_argument(
        '--gamma',
        action='store_true',
        help='Calculate theoretical γ(z) and Λ_eff(z) from first principles. '
             'Pure theory - no fitting. Runtime: ~1 minute'
    )
    
    parser.add_argument(
        '--bao',
        action='store_true',
        help='Test BAO predictions using 3 mechanisms (Λ_eff, viscosity, Friedmann). '
             'Pure prediction - no fitting. Runtime: ~5 minutes'
    )
    
    parser.add_argument(
        '--all-datasets',
        action='store_true',
        help='Test all available BAO datasets (BOSS, eBOSS, 6dFGS, WiggleZ, DESI). '
             'Requires --bao flag. Runtime: ~10 minutes'
    )
    
    parser.add_argument(
        '--alpha-analysis',
        action='store_true',
        help='Perform cross-dataset α consistency analysis. '
             'Requires --bao flag. Runtime: ~15 minutes'
    )
    
    parser.add_argument(
        '--with-systematics',
        action='store_true',
        default=True,
        help='Include systematic errors in BAO analysis (default: True)'
    )
    
    parser.add_argument(
        '--model-comparison',
        action='store_true',
        help='Run model comparison (BIC, AIC, Bayes factors). Requires --bao'
    )
    
    parser.add_argument(
        '--cross-validation',
        action='store_true',
        help='Run cross-validation (LOO-CV, K-fold). Requires --bao --all-datasets'
    )
    
    parser.add_argument(
        '--null-tests',
        action='store_true',
        help='Run null hypothesis tests (bootstrap, shuffling). Requires --bao'
    )
    
    parser.add_argument(
        '--full-validation',
        action='store_true',
        help='Run ALL statistical tests (model comparison, CV, null tests)'
    )
    
    parser.add_argument(
        '--forward-predictions',
        action='store_true',
        help='Generate forward predictions for DESI Y3 (pre-registration)'
    )
    
    parser.add_argument(
        '--alternative-models',
        action='store_true',
        help='Compare to alternative models (EDE, N_eff, w0wa). Requires --bao --all-datasets'
    )
    
    return parser.parse_args()


def main():
    """
    Execute complete analysis pipeline.
    
    Steps:
    1. Load ACT DR6 CMB E-mode data
    2. Detect phase transitions at ℓ = 1076, 1706, 2336
    3. Perform statistical validation (bootstrap, LEE corrections)
    4. Calculate cosmological constant from first principles
    5. Analyze temporal cascade and expansion history
    6. Resolve cosmological tensions (H₀, S₈, BAO)
    7. Generate publication-quality figures
    8. Save results to JSON and log files
    
    Runtime: ~3-5 minutes on standard hardware
    """
    args = parse_arguments()
    
    print("=" * 70)
    print("CMB E-MODE PHASE TRANSITION ANALYSIS")
    print(f"Version: {__version__}")
    print("=" * 70)
    print()
    print(f"Paper: {__paper__}")
    print()
    print(f"Output directory: {args.output_dir}")
    if args.quiet:
        print("Running in quiet mode...")
    print()
    print("Starting analysis pipeline...")
    print("=" * 70)
    print()
    
    try:
        # Check for modular theory analysis modes
        if args.gamma or args.bao:
            print("=" * 70)
            print("MODULAR THEORY ANALYSIS MODE")
            print("=" * 70)
            print()
            
            results_summary = {}
            
            # --gamma: Theoretical γ and Λ calculation
            if args.gamma:
                from cmb_analysis.gamma_theoretical_analysis import run_gamma_analysis
                
                print("Running theoretical γ(z) and Λ_eff(z) analysis...")
                print()
                
                gamma_results = run_gamma_analysis(output_dir=args.output_dir)
                results_summary['gamma_analysis'] = gamma_results
                
                print()
                print("✓ Gamma analysis complete")
                print(f"  Results: {args.output_dir}/gamma_theoretical.json")
                print()
            
            # --forward-predictions: DESI Y3 pre-registration
            if args.forward_predictions:
                from cmb_analysis.forward_predictions import run_forward_predictions
                
                print("Generating DESI Year 3 forward predictions...")
                print()
                
                forward_results = run_forward_predictions(output_dir=args.output_dir)
                results_summary['forward_predictions'] = forward_results
                
                print()
                print("✓ DESI Y3 predictions registered")
                print(f"  Timestamp: {forward_results['timestamp_utc']}")
                print(f"  Hash: {forward_results['sha256_hash'][:16]}...")
                print(f"  arXiv doc: DESI_Y3_PREDICTIONS.tex")
                print()
            
            # --bao: BAO prediction test
            if args.bao:
                from cmb_analysis.bao_prediction_test import run_bao_test
                
                if args.all_datasets or args.alpha_analysis:
                    print("Running multi-dataset BAO validation...")
                    if args.alpha_analysis:
                        print("  (includes α consistency analysis)")
                    if args.alternative_models:
                        print("  (includes alternative model comparison)")
                    print()
                else:
                    print("Running BAO theoretical prediction test...")
                    print()
                
                # Enable all tests if --full-validation
                if args.full_validation:
                    args.model_comparison = True
                    args.cross_validation = True
                    args.null_tests = True
                
                bao_results = run_bao_test(
                    output_dir=args.output_dir,
                    all_datasets=args.all_datasets,
                    with_systematics=args.with_systematics,
                    alpha_analysis=args.alpha_analysis,
                    model_comparison=args.model_comparison,
                    cross_validation=args.cross_validation,
                    null_tests=args.null_tests,
                    alternative_models=args.alternative_models
                )
                results_summary['bao_test'] = bao_results
                
                print()
                print("✓ BAO test complete")
                
                if args.all_datasets or args.alpha_analysis:
                    print(f"  Results: {args.output_dir}/bao_multi_dataset_validation.json")
                    if 'summary' in bao_results:
                        n_pass = bao_results['summary'].get('n_passing', 0)
                        n_total = bao_results['summary']['n_datasets']
                        print(f"  Theoretical (α=-5.7, PARAMETER-FREE): {n_pass}/{n_total} pass")
                    if args.alpha_analysis and bao_results.get('alpha_universality'):
                        alpha_univ = bao_results['alpha_universality']
                        print(f"  Universal α: {alpha_univ['alpha_universal']:.2f} ± {alpha_univ['alpha_uncertainty']:.2f}")
                        print(f"  Consistency: {alpha_univ['verdict']}")
                else:
                    print(f"  Results: {args.output_dir}/bao_theory_test.json")
                    # Summary
                    if bao_results['summary']['any_pass']:
                        print(f"  ✓ {bao_results['summary']['num_passing']} mechanism(s) PASS")
                        print(f"  Best: {bao_results['best_mechanism']}")
                    else:
                        print(f"  ✗ No mechanisms pass (need p > 0.05)")
                print()
            
            print("=" * 70)
            print("ANALYSIS COMPLETE")
            print("=" * 70)
            return 0
        
        # Run analysis with all flags
        results = run_analysis(
            output_dir=args.output_dir,
            extend=args.extend,
            validation_level=args.validation_level,
            run_monte_carlo=args.run_monte_carlo
        )
        
        # Print summary
        print()
        print("=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print()
        
        analysis_type = "extended" if args.extend else "core"
        print(f"Analysis type: {analysis_type}")
        print()
        
        if 'transitions' in results:
            print(f"Detected transitions: {len(results['transitions'])}")
            print(f"Multipoles: {results['transitions']}")
            print()
        
        print(f"Results saved to: {args.output_dir}/{analysis_type}_results.json")
        print(f"Detailed log: {args.output_dir}/cmb_analysis.log")
        print()
        
        if not args.extend:
            print("NOTE: Extended theoretical analysis not included.")
            print("      Run with --extend to include gamma extraction,")
            print("      Lambda calculation, and cosmological tensions.")
            print()
        
        print("For validation, see examples/expected_output.txt")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)
        print(f"Analysis failed with error: {e}")
        print()
        print("Check the log file for details:")
        print(f"  {args.output_dir}/cmb_analysis_unified.log")
        print()
        print("For help:")
        print("  python main.py --help")
        print("  See examples/quickstart.md")
        print("=" * 70)
        
        return 1


if __name__ == "__main__":
    sys.exit(main())

