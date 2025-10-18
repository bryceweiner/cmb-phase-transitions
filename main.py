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
        # Run complete analysis
        results = run_analysis(output_dir=args.output_dir)
        
        # Print summary
        print()
        print("=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print()
        print(f"Detected transitions: {len(results['transitions'])}")
        print(f"Multipoles: {results['transitions']}")
        print()
        print(f"Results saved to: {args.output_dir}/cmb_analysis_unified.json")
        print(f"Detailed log: {args.output_dir}/cmb_analysis_unified.log")
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

