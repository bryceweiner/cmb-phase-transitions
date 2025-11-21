#!/usr/bin/env python3
"""
Threshold Calibration Analysis
==============================

Comprehensive threshold calibration to determine statistically appropriate
detection threshold for CMB E-mode phase transitions.

Tests multiple false positive rate targets:
- 5% (standard statistical threshold)
- 1% (conservative)
- 0.27% (3-sigma equivalent)
- 0.0057% (5-sigma discovery threshold)

Usage:
    python run_threshold_calibration.py
"""

import numpy as np
import sys
from cmb_analysis import DataLoader, OutputManager
from cmb_analysis.threshold_calibration import ThresholdCalibrator

def main():
    """Run comprehensive threshold calibration analysis."""
    
    # Initialize output
    output = OutputManager(output_dir="./results/calibration")
    
    output.log_section_header("THRESHOLD CALIBRATION ANALYSIS")
    output.log_message("Comprehensive calibration for CMB E-mode phase transition detection")
    output.log_message("")
    
    # Load data
    loader = DataLoader(output=output)
    ell, C_ell, C_ell_err = loader.load_act_dr6()
    
    # Validate data
    loader.validate_data(ell, C_ell, C_ell_err)
    
    # Initialize calibrator
    calibrator = ThresholdCalibrator(output=output, f_sky=0.4)
    
    # Test multiple target FPR levels
    fpr_targets = [
        ('Standard (5%)', 0.05),
        ('Conservative (1%)', 0.01),
        ('3-sigma (0.27%)', 0.0027),
        ('Very conservative (0.1%)', 0.001)
    ]
    
    all_results = {}
    
    for name, target_fpr in fpr_targets:
        output.log_message("="*70)
        output.log_message(f"CALIBRATION FOR {name.upper()}")
        output.log_message("="*70)
        output.log_message("")
        
        # Run calibration
        results = calibrator.full_calibration_pipeline(
            ell, C_ell, C_ell_err,
            target_fpr=target_fpr,
            n_target=3,
            n_realizations=10000,
            window_size=25
        )
        
        all_results[name] = results
        
        # Summary
        calib = results['calibration']
        detect = results['detection']
        sig = results['significance']
        
        output.log_message("")
        output.log_message("SUMMARY:")
        output.log_message(f"  Target FPR: {target_fpr*100:.4f}%")
        output.log_message(f"  Calibrated prominence: {calib['calibrated_prominence_factor']:.2f}")
        output.log_message(f"  Detections: {sig['n_detected']}")
        
        if sig['n_detected'] > 0:
            output.log_message(f"  Transitions: {detect['transitions']}")
            if sig['empirical_sigma']:
                output.log_message(f"  Empirical significance: {sig['empirical_sigma']:.2f}σ")
        
        output.log_message("")
        output.log_message(f"Recommendation: {results['recommendation']}")
        output.log_message("")
    
    # Final comparison
    output.log_section_header("THRESHOLD CALIBRATION SUMMARY")
    output.log_message(f"{'FPR Target':<25} {'Prominence':<15} {'Detections':<12} {'Conclusion'}")
    output.log_message("-"*70)
    
    for name, results in all_results.items():
        prom = results['calibration']['calibrated_prominence_factor']
        n_det = results['significance']['n_detected']
        
        if n_det == 0:
            conclusion = "None"
        elif n_det < 3:
            conclusion = f"{n_det} (marginal)"
        else:
            conclusion = f"{n_det} (significant)"
        
        output.log_message(f"{name:<25} {prom:<15.2f} {n_det:<12} {conclusion}")
    
    output.log_message("")
    output.log_message("="*70)
    output.log_message("FINAL ASSESSMENT")
    output.log_message("="*70)
    output.log_message("")
    
    # Check consistency across thresholds
    detection_counts = [r['significance']['n_detected'] for r in all_results.values()]
    
    if all(d == 0 for d in detection_counts):
        output.log_message("CONCLUSION: NO STATISTICALLY SIGNIFICANT DETECTIONS")
        output.log_message("")
        output.log_message("No features exceed calibrated thresholds at any standard")
        output.log_message("significance level (5%, 1%, 0.27%, or 0.1%).")
        output.log_message("")
        output.log_message("The detected features at uncalibrated threshold are")
        output.log_message("consistent with noise fluctuations.")
        output.log_message("")
        output.log_message("RECOMMENDATION:")
        output.log_message("  Report as null result or upper limits on transition amplitudes.")
        output.log_message("  CMB E-mode power spectrum is consistent with smooth ΛCDM.")
        
    elif all(d >= 3 for d in detection_counts):
        output.log_message("CONCLUSION: ROBUST DETECTIONS ACROSS ALL THRESHOLDS")
        output.log_message("")
        output.log_message(f"Detected {detection_counts[0]} transitions at all tested")
        output.log_message("significance levels, indicating features significantly")
        output.log_message("above noise floor.")
        output.log_message("")
        output.log_message("RECOMMENDATION:")
        output.log_message("  Report as statistically significant discoveries.")
        output.log_message("  Proceed with physical interpretation.")
        
    else:
        output.log_message("CONCLUSION: THRESHOLD-DEPENDENT DETECTIONS")
        output.log_message("")
        output.log_message("Number of detections varies with threshold:")
        for name, n_det in zip(all_results.keys(), detection_counts):
            output.log_message(f"  {name}: {n_det} detections")
        output.log_message("")
        output.log_message("This indicates features are near the noise floor.")
        output.log_message("")
        output.log_message("RECOMMENDATION:")
        output.log_message("  Use most conservative threshold that yields detections.")
        output.log_message("  Report with appropriate caveats about marginal significance.")
        output.log_message("  Seek independent confirmation.")
    
    output.log_message("")
    output.close()
    
    return all_results


if __name__ == "__main__":
    try:
        results = main()
        print("\n" + "="*70)
        print("THRESHOLD CALIBRATION ANALYSIS COMPLETE")
        print("="*70)
        print("\nResults saved to: ./results/calibration/")
        print("  - cmb_analysis.log")
        print("  - cmb_analysis_unified.json")
        
    except Exception as e:
        print(f"\nAnalysis failed with error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

