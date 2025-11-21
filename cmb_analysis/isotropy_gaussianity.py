"""
Isotropy and Gaussianity Testing Module
========================================

Tests whether detected features are consistent with:
1. Statistical isotropy (no directional preference)
2. Gaussian statistics (consistent with inflation predictions)

These tests verify that detected phase transitions are genuine physical
features rather than artifacts of:
- Non-Gaussian foreground contamination
- Anisotropic systematic effects  
- Directional survey artifacts

References:
- Planck 2018 IX: Constraints on primordial non-Gaussianity
- ACT: Atacama Cosmology Telescope collaboration papers on systematics
"""

import numpy as np
from scipy import stats
from scipy.special import erfc
from typing import Dict, Any, Optional, Tuple

from .utils import OutputManager


class IsotropyGaussianityTests:
    """
    Statistical tests for isotropy and Gaussianity of CMB data.
    
    Methods:
        test_gaussianity: Tests if residuals follow Gaussian distribution
        test_isotropy: Tests for directional dependence
        test_non_gaussianity: Tests for higher-order correlations
        test_residuals_isotropy: Tests if detection residuals are isotropic
    """
    
    def __init__(self, output: OutputManager = None):
        """Initialize isotropy/Gaussianity tester."""
        self.output = output if output is not None else OutputManager()
    
    def test_gaussianity(self, residuals: np.ndarray, 
                        residual_errors: np.ndarray) -> Dict[str, Any]:
        """
        Test if residuals follow Gaussian distribution.
        
        Uses multiple tests:
        1. Kolmogorov-Smirnov test vs normal distribution
        2. Anderson-Darling test (more sensitive to tails)
        3. Shapiro-Wilk test
        4. Skewness and kurtosis tests
        5. Chi-squared test on binned residuals
        
        Parameters:
            residuals (ndarray): Data - Model residuals
            residual_errors (ndarray): Uncertainties on residuals
            
        Returns:
            dict: Gaussianity test results
        """
        self.output.log_subsection("GAUSSIANITY TESTS")
        
        # Normalize residuals by errors (should be standard normal if Gaussian)
        normalized_residuals = residuals / residual_errors
        
        results = {
            'n_points': len(normalized_residuals),
            'tests': {}
        }
        
        # 1. Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.kstest(normalized_residuals, 'norm')
        results['tests']['kolmogorov_smirnov'] = {
            'statistic': float(ks_stat),
            'p_value': float(ks_pvalue),
            'passes': bool(ks_pvalue > 0.05),
            'interpretation': 'Consistent with Gaussian' if ks_pvalue > 0.05 else 'Deviates from Gaussian'
        }
        
        self.output.log_message(f"\n1. Kolmogorov-Smirnov Test:")
        self.output.log_message(f"   Statistic: {ks_stat:.4f}")
        self.output.log_message(f"   p-value: {ks_pvalue:.4f}")
        self.output.log_message(f"   Result: {results['tests']['kolmogorov_smirnov']['interpretation']}")
        
        # 2. Anderson-Darling test (more sensitive to tails)
        ad_result = stats.anderson(normalized_residuals, dist='norm')
        # Check against 5% significance level (index 2 in critical values)
        ad_passes = ad_result.statistic < ad_result.critical_values[2]
        results['tests']['anderson_darling'] = {
            'statistic': float(ad_result.statistic),
            'critical_value_5pct': float(ad_result.critical_values[2]),
            'passes': bool(ad_passes),
            'interpretation': 'Consistent with Gaussian' if ad_passes else 'Deviates from Gaussian'
        }
        
        self.output.log_message(f"\n2. Anderson-Darling Test:")
        self.output.log_message(f"   Statistic: {ad_result.statistic:.4f}")
        self.output.log_message(f"   Critical value (5%): {ad_result.critical_values[2]:.4f}")
        self.output.log_message(f"   Result: {results['tests']['anderson_darling']['interpretation']}")
        
        # 3. Shapiro-Wilk test (best for small-medium samples)
        if len(normalized_residuals) < 5000:  # SW test limited to n < 5000
            sw_stat, sw_pvalue = stats.shapiro(normalized_residuals)
            results['tests']['shapiro_wilk'] = {
                'statistic': float(sw_stat),
                'p_value': float(sw_pvalue),
                'passes': bool(sw_pvalue > 0.05),
                'interpretation': 'Consistent with Gaussian' if sw_pvalue > 0.05 else 'Deviates from Gaussian'
            }
            
            self.output.log_message(f"\n3. Shapiro-Wilk Test:")
            self.output.log_message(f"   Statistic: {sw_stat:.4f}")
            self.output.log_message(f"   p-value: {sw_pvalue:.4f}")
            self.output.log_message(f"   Result: {results['tests']['shapiro_wilk']['interpretation']}")
        
        # 4. Skewness and Kurtosis
        skewness = stats.skew(normalized_residuals)
        kurtosis = stats.kurtosis(normalized_residuals)  # Excess kurtosis (0 for normal)
        
        # For Gaussian: skewness ≈ 0, kurtosis ≈ 0
        # Standard errors: SE(skew) ≈ √(6/n), SE(kurt) ≈ √(24/n)
        n = len(normalized_residuals)
        skew_se = np.sqrt(6.0 / n)
        kurt_se = np.sqrt(24.0 / n)
        
        skew_zscore = skewness / skew_se
        kurt_zscore = kurtosis / kurt_se
        
        results['tests']['moments'] = {
            'skewness': float(skewness),
            'skewness_zscore': float(skew_zscore),
            'skewness_passes': bool(abs(skew_zscore) < 2),
            'kurtosis': float(kurtosis),
            'kurtosis_zscore': float(kurt_zscore),
            'kurtosis_passes': bool(abs(kurt_zscore) < 2),
            'interpretation': 'Consistent with Gaussian moments' 
                            if (abs(skew_zscore) < 2 and abs(kurt_zscore) < 2) 
                            else 'Deviates from Gaussian moments'
        }
        
        self.output.log_message(f"\n4. Moment Tests:")
        self.output.log_message(f"   Skewness: {skewness:.4f} (z = {skew_zscore:.2f})")
        self.output.log_message(f"   Kurtosis: {kurtosis:.4f} (z = {kurt_zscore:.2f})")
        self.output.log_message(f"   Result: {results['tests']['moments']['interpretation']}")
        
        # 5. Chi-squared test on binned residuals
        # Bin the normalized residuals and compare to expected Gaussian counts
        n_bins = min(50, int(np.sqrt(len(normalized_residuals))))
        hist, bin_edges = np.histogram(normalized_residuals, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Expected counts from normal distribution
        bin_width = bin_edges[1] - bin_edges[0]
        expected = len(normalized_residuals) * bin_width * stats.norm.pdf(bin_centers)
        
        # Chi-squared (exclude bins with expected < 5)
        mask = expected >= 5
        valid_bins = np.sum(mask)
        
        if valid_bins > 2:  # Need at least 3 bins for meaningful chi-squared test
            chi2 = np.sum((hist[mask] - expected[mask])**2 / expected[mask])
            dof = valid_bins - 1  # bins - 1
            chi2_pvalue = 1.0 - stats.chi2.cdf(chi2, dof)
            
            results['tests']['chi_squared_binned'] = {
                'chi2': float(chi2),
                'dof': int(dof),
                'chi2_dof': float(chi2 / dof),
                'p_value': float(chi2_pvalue),
                'passes': bool(chi2_pvalue > 0.05),
                'valid_bins': int(valid_bins),
                'interpretation': 'Consistent with Gaussian distribution' 
                                if chi2_pvalue > 0.05 else 'Deviates from Gaussian distribution'
            }
            
            self.output.log_message(f"\n5. Chi-Squared Test (binned residuals):")
            self.output.log_message(f"   χ² = {chi2:.2f} (DOF = {dof}, valid bins = {valid_bins})")
            self.output.log_message(f"   χ²/DOF = {chi2/dof:.2f}")
            self.output.log_message(f"   p-value: {chi2_pvalue:.4f}")
            self.output.log_message(f"   Result: {results['tests']['chi_squared_binned']['interpretation']}")
        else:
            results['tests']['chi_squared_binned'] = {
                'passes': None,
                'valid_bins': int(valid_bins),
                'interpretation': f'Test skipped (only {valid_bins} bins with expected ≥ 5)',
                'note': 'Sample size too small or distribution too Gaussian for binned test'
            }
            
            self.output.log_message(f"\n5. Chi-Squared Test (binned residuals):")
            self.output.log_message(f"   Test skipped: only {valid_bins} bins with expected ≥ 5")
            self.output.log_message(f"   (Other gaussianity tests remain valid)")
        
        # Overall assessment (filter out None for skipped tests)
        test_results = [
            results['tests']['kolmogorov_smirnov']['passes'],
            results['tests']['anderson_darling']['passes'],
            results['tests'].get('shapiro_wilk', {}).get('passes', True),
            results['tests']['moments']['skewness_passes'],
            results['tests']['moments']['kurtosis_passes'],
            results['tests'].get('chi_squared_binned', {}).get('passes', True)
        ]
        # Filter out None (skipped tests) and count only actual passes
        passed_tests = sum([t for t in test_results if t is not None])
        
        # Count only non-skipped tests
        total_tests = sum([1 for t in test_results if t is not None])
        
        results['summary'] = {
            'tests_passed': int(passed_tests),
            'tests_total': int(total_tests),
            'fraction_passed': float(passed_tests / total_tests),
            'overall_assessment': (
                'Strongly consistent with Gaussian' if passed_tests == total_tests
                else 'Mostly consistent with Gaussian' if passed_tests >= total_tests - 1
                else 'Some deviations from Gaussian' if passed_tests >= total_tests - 2
                else 'Significant deviations from Gaussian'
            )
        }
        
        self.output.log_message(f"\n✓ Gaussianity Summary:")
        self.output.log_message(f"   Tests passed: {passed_tests}/{total_tests}")
        self.output.log_message(f"   Assessment: {results['summary']['overall_assessment']}")
        
        return results
    
    def test_isotropy(self, ell: np.ndarray, residuals: np.ndarray,
                     n_angular_bins: int = 8) -> Dict[str, Any]:
        """
        Test for isotropy by checking if residuals vary with multipole direction.
        
        For full-sky analysis, would check for directional dependence.
        For our 1D power spectrum, we test if residuals are uniformly distributed
        across multipole space (no preferred multipole ranges).
        
        Parameters:
            ell (ndarray): Multipole values
            residuals (ndarray): Data - model residuals
            n_angular_bins (int): Number of multipole bins to test
            
        Returns:
            dict: Isotropy test results
        """
        self.output.log_subsection("ISOTROPY TESTS")
        
        results = {
            'n_bins': n_angular_bins,
            'tests': {}
        }
        
        # 1. Runs test for randomness
        # Check if positive/negative residuals are randomly distributed
        signs = np.sign(residuals)
        runs, runs_pvalue = self._runs_test(signs)
        
        results['tests']['runs_test'] = {
            'n_runs': int(runs),
            'p_value': float(runs_pvalue),
            'passes': bool(runs_pvalue > 0.05),
            'interpretation': 'Random distribution' if runs_pvalue > 0.05 else 'Clustering detected'
        }
        
        self.output.log_message(f"\n1. Runs Test (randomness of residual signs):")
        self.output.log_message(f"   Number of runs: {runs}")
        self.output.log_message(f"   p-value: {runs_pvalue:.4f}")
        self.output.log_message(f"   Result: {results['tests']['runs_test']['interpretation']}")
        
        # 2. Chi-squared test for uniform distribution across multipole bins
        bin_edges = np.linspace(ell.min(), ell.max(), n_angular_bins + 1)
        bin_indices = np.digitize(ell, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_angular_bins - 1)
        
        # Variance in each bin
        bin_variances = []
        bin_counts = []
        for i in range(n_angular_bins):
            mask = bin_indices == i
            if np.sum(mask) > 1:
                bin_variances.append(np.var(residuals[mask]))
                bin_counts.append(np.sum(mask))
        
        if len(bin_variances) > 2:
            # Bartlett test for equality of variances
            # Use residuals from each bin
            bin_residuals = [residuals[bin_indices == i] for i in range(n_angular_bins) 
                           if np.sum(bin_indices == i) > 1]
            
            bartlett_stat, bartlett_pvalue = stats.bartlett(*bin_residuals)
            
            results['tests']['bartlett_variance'] = {
                'statistic': float(bartlett_stat),
                'p_value': float(bartlett_pvalue),
                'passes': bool(bartlett_pvalue > 0.05),
                'interpretation': 'Uniform variance (isotropic)' if bartlett_pvalue > 0.05 
                                else 'Non-uniform variance (anisotropic)'
            }
            
            self.output.log_message(f"\n2. Bartlett Test (variance uniformity across multipoles):")
            self.output.log_message(f"   Statistic: {bartlett_stat:.4f}")
            self.output.log_message(f"   p-value: {bartlett_pvalue:.4f}")
            self.output.log_message(f"   Result: {results['tests']['bartlett_variance']['interpretation']}")
        
        # 3. Levene test (more robust to non-Gaussianity)
        if len(bin_residuals) > 2:
            levene_stat, levene_pvalue = stats.levene(*bin_residuals)
            
            results['tests']['levene_variance'] = {
                'statistic': float(levene_stat),
                'p_value': float(levene_pvalue),
                'passes': bool(levene_pvalue > 0.05),
                'interpretation': 'Uniform variance (isotropic)' if levene_pvalue > 0.05 
                                else 'Non-uniform variance (anisotropic)'
            }
            
            self.output.log_message(f"\n3. Levene Test (robust variance uniformity):")
            self.output.log_message(f"   Statistic: {levene_stat:.4f}")
            self.output.log_message(f"   p-value: {levene_pvalue:.4f}")
            self.output.log_message(f"   Result: {results['tests']['levene_variance']['interpretation']}")
        
        # Overall assessment (filter out None for skipped tests)
        test_results_isotropy = [
            results['tests']['runs_test']['passes'],
            results['tests'].get('bartlett_variance', {}).get('passes', True),
            results['tests'].get('levene_variance', {}).get('passes', True)
        ]
        passed_tests = sum([t for t in test_results_isotropy if t is not None])
        
        total_tests = len([k for k in results['tests'].keys()])
        
        results['summary'] = {
            'tests_passed': int(passed_tests),
            'tests_total': int(total_tests),
            'fraction_passed': float(passed_tests / total_tests) if total_tests > 0 else 1.0,
            'overall_assessment': (
                'Consistent with isotropy' if passed_tests == total_tests
                else 'Mostly consistent with isotropy' if passed_tests >= total_tests - 1
                else 'Evidence for anisotropy'
            )
        }
        
        self.output.log_message(f"\n✓ Isotropy Summary:")
        self.output.log_message(f"   Tests passed: {passed_tests}/{total_tests}")
        self.output.log_message(f"   Assessment: {results['summary']['overall_assessment']}")
        
        return results
    
    def _runs_test(self, sequence: np.ndarray) -> Tuple[int, float]:
        """
        Wald-Wolfowitz runs test for randomness.
        
        A "run" is a sequence of consecutive identical values.
        Too few runs suggests clustering, too many suggests oscillation.
        
        Parameters:
            sequence (ndarray): Binary sequence (+1/-1 or similar)
            
        Returns:
            tuple: (number of runs, p-value)
        """
        # Count runs
        runs = 1
        for i in range(1, len(sequence)):
            if sequence[i] != sequence[i-1]:
                runs += 1
        
        # Count positives and negatives
        n_pos = np.sum(sequence > 0)
        n_neg = np.sum(sequence <= 0)
        
        # Expected runs and variance under null hypothesis
        n = len(sequence)
        expected_runs = (2 * n_pos * n_neg) / n + 1
        var_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n**2 * (n - 1))
        
        # Z-score
        z = (runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return runs, p_value
    
    def test_phase_randomness(self, ell: np.ndarray, C_ell: np.ndarray) -> Dict[str, Any]:
        """
        Test if phases are random (as expected for Gaussian field).
        
        Performs a proxy test by checking if fluctuations around smooth trend
        are consistent with random phases.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Power spectrum
            
        Returns:
            dict: Phase randomness results
        """
        self.output.log_subsection("PHASE RANDOMNESS TEST")
        
        # Remove smooth trend
        coeffs = np.polyfit(ell, C_ell, deg=5)
        C_smooth = np.polyval(coeffs, ell)
        fluctuations = C_ell - C_smooth
        
        # Autocorrelation test
        # For random phases, autocorrelation should decay quickly
        max_lag = min(50, len(fluctuations) // 4)
        autocorr = np.correlate(fluctuations, fluctuations, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags only
        autocorr = autocorr[:max_lag] / autocorr[0]  # Normalize
        
        # Expected: decay to ~0 within a few lags
        # Measure decay length: lag where |autocorr| < 1/e ≈ 0.37
        decay_lag = np.where(np.abs(autocorr) < 0.37)[0]
        decay_lag = decay_lag[0] if len(decay_lag) > 0 else max_lag
        
        results = {
            'autocorrelation_decay_lag': int(decay_lag),
            'max_autocorr_lag_10': float(np.max(np.abs(autocorr[1:11]))),  # Exclude lag=0
            'interpretation': (
                'Consistent with random phases' if decay_lag < 10 
                else 'Some phase coherence detected'
            )
        }
        
        self.output.log_message(f"\nAutocorrelation decay:")
        self.output.log_message(f"   Decay lag (|ρ| < 0.37): {decay_lag}")
        self.output.log_message(f"   Max |ρ| at lags 1-10: {results['max_autocorr_lag_10']:.4f}")
        self.output.log_message(f"   Result: {results['interpretation']}")
        
        return results
    
    def full_isotropy_gaussianity_analysis(self, ell: np.ndarray, C_ell: np.ndarray,
                                          C_ell_err: np.ndarray, 
                                          C_model: np.ndarray) -> Dict[str, Any]:
        """
        Complete isotropy and Gaussianity analysis.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Observed power spectrum
            C_ell_err (ndarray): Uncertainties
            C_model (ndarray): Model power spectrum (smooth or with transitions)
            
        Returns:
            dict: Complete isotropy/Gaussianity results
        """
        self.output.log_section_header("ISOTROPY & GAUSSIANITY VALIDATION")
        
        # Compute residuals
        residuals = C_ell - C_model
        
        # Run all tests
        gaussianity_results = self.test_gaussianity(residuals, C_ell_err)
        isotropy_results = self.test_isotropy(ell, residuals)
        phase_results = self.test_phase_randomness(ell, C_ell)
        
        # Combined assessment
        all_results = {
            'gaussianity': gaussianity_results,
            'isotropy': isotropy_results,
            'phase_randomness': phase_results
        }
        
        # Overall pass/fail
        gauss_pass = gaussianity_results['summary']['fraction_passed'] >= 0.5
        iso_pass = isotropy_results['summary']['fraction_passed'] >= 0.5
        
        all_results['overall_assessment'] = {
            'gaussianity_pass': bool(gauss_pass),
            'isotropy_pass': bool(iso_pass),
            'conclusion': (
                'Data consistent with isotropic Gaussian field' if (gauss_pass and iso_pass)
                else 'Some deviations from standard assumptions' if (gauss_pass or iso_pass)
                else 'Significant deviations from isotropic Gaussian field'
            )
        }
        
        self.output.log_message(f"\n{'='*70}")
        self.output.log_message("OVERALL ISOTROPY & GAUSSIANITY ASSESSMENT")
        self.output.log_message(f"{'='*70}")
        self.output.log_message(f"Gaussianity: {'✓ PASS' if gauss_pass else '✗ FAIL'}")
        self.output.log_message(f"Isotropy: {'✓ PASS' if iso_pass else '✗ FAIL'}")
        self.output.log_message(f"\nConclusion: {all_results['overall_assessment']['conclusion']}")
        
        return all_results

