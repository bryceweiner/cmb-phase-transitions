"""
Covariance Matrix Module
=========================

Proper covariance matrix construction for CMB power spectra.

Implements full covariance including cosmic variance, instrumental noise,
and foreground uncertainties. Critical for likelihood-based inference.

Classes:
    CovarianceMatrix: Construct and manipulate covariance matrices

Paper reference: Methods section, systematic uncertainties
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import warnings

from .utils import OutputManager


class CovarianceMatrix:
    """
    Construct and manage covariance matrices for CMB power spectra.
    
    Includes:
    - Cosmic variance (sample variance from finite sky)
    - Instrumental noise
    - Foreground uncertainties
    - Beam systematics
    - Correlations from filtering
    
    Attributes:
        output (OutputManager): For logging
        f_sky (float): Sky fraction observed
        
    Example:
        >>> cov_builder = CovarianceMatrix(f_sky=0.4)
        >>> cov = cov_builder.construct_full_covariance(ell, C_ell, C_ell_err)
    """
    
    def __init__(self, output: OutputManager = None, f_sky: float = 0.4):
        """
        Initialize CovarianceMatrix.
        
        Parameters:
            output (OutputManager, optional): For logging
            f_sky (float): Sky fraction (default: 0.4 for ACT)
        """
        self.output = output if output is not None else OutputManager()
        self.f_sky = f_sky
    
    def cosmic_variance(self, ell: np.ndarray, C_ell: np.ndarray) -> np.ndarray:
        """
        Compute cosmic variance contribution.
        
        For E-mode polarization: σ²_CV = 2 C_ℓ² / [(2ℓ+1) f_sky]
        Factor of 2 from E and B modes degeneracy.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Power spectrum
            
        Returns:
            ndarray: Cosmic variance (diagonal elements)
        """
        sigma_cv_sq = 2 * C_ell**2 / ((2 * ell + 1) * self.f_sky)
        return sigma_cv_sq
    
    def instrumental_noise(self, C_ell_err: np.ndarray) -> np.ndarray:
        """
        Instrumental noise variance.
        
        Parameters:
            C_ell_err (ndarray): Instrumental uncertainties
            
        Returns:
            ndarray: Noise variance
        """
        return C_ell_err**2
    
    def foreground_uncertainty(self, ell: np.ndarray, C_ell: np.ndarray,
                              foreground_fraction: float = 0.05) -> np.ndarray:
        """
        Foreground contamination uncertainty.
        
        Simplified model: constant fractional uncertainty from
        dust and synchrotron subtraction.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Power spectrum
            foreground_fraction (float): Fractional uncertainty (default: 5%)
            
        Returns:
            ndarray: Foreground variance
        """
        return (foreground_fraction * C_ell)**2
    
    def beam_uncertainty(self, ell: np.ndarray, C_ell: np.ndarray,
                        beam_fwhm_arcmin: float = 1.4,
                        beam_uncertainty_fraction: float = 0.01) -> np.ndarray:
        """
        Beam uncertainty contribution.
        
        Uncertainty in beam profile affects high-ℓ measurements.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Power spectrum
            beam_fwhm_arcmin (float): Beam FWHM in arcminutes (default: 1.4 for ACT)
            beam_uncertainty_fraction (float): Fractional beam uncertainty (default: 1%)
            
        Returns:
            ndarray: Beam variance
        """
        # Beam window function uncertainty increases with ℓ
        # Simplified: σ_beam ≈ (beam_unc × ℓ × beam_size) × C_ℓ
        beam_sigma_rad = beam_fwhm_arcmin * (np.pi / 180 / 60) / 2.355
        beam_effect = beam_uncertainty_fraction * ell * beam_sigma_rad
        return (beam_effect * C_ell)**2
    
    def construct_diagonal(self, ell: np.ndarray, C_ell: np.ndarray,
                          C_ell_err: np.ndarray,
                          include_cosmic_variance: bool = True,
                          include_foreground: bool = True,
                          include_beam: bool = True) -> np.ndarray:
        """
        Construct diagonal covariance (no correlations).
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Power spectrum
            C_ell_err (ndarray): Instrumental uncertainties
            include_cosmic_variance (bool): Include CV (default: True)
            include_foreground (bool): Include foreground unc (default: True)
            include_beam (bool): Include beam unc (default: True)
            
        Returns:
            ndarray: Diagonal variance
        """
        # Start with instrumental noise
        variance = self.instrumental_noise(C_ell_err)
        
        # Add cosmic variance
        if include_cosmic_variance:
            variance += self.cosmic_variance(ell, C_ell)
        
        # Add foreground uncertainty
        if include_foreground:
            variance += self.foreground_uncertainty(ell, C_ell)
        
        # Add beam uncertainty
        if include_beam:
            variance += self.beam_uncertainty(ell, C_ell)
        
        return variance
    
    def add_filter_correlations(self, cov: np.ndarray, ell: np.ndarray,
                               filter_window: int = 50,
                               correlation_strength: float = 0.1) -> np.ndarray:
        """
        Add correlations from Savitzky-Golay filtering.
        
        SG filter creates correlations between adjacent multipoles within window.
        
        Parameters:
            cov (ndarray): Covariance matrix (N x N)
            ell (ndarray): Multipole values
            filter_window (int): SG filter window size (default: 50)
            correlation_strength (float): Correlation coefficient (default: 0.1)
            
        Returns:
            ndarray: Covariance with correlations added
        """
        n = len(ell)
        
        # Correlation decays with distance
        for i in range(n):
            for j in range(i+1, min(i+filter_window, n)):
                distance = abs(ell[j] - ell[i])
                # Exponential decay
                corr_factor = correlation_strength * np.exp(-distance / filter_window)
                correlation = corr_factor * np.sqrt(cov[i, i] * cov[j, j])
                cov[i, j] = correlation
                cov[j, i] = correlation
        
        return cov
    
    def construct_full_covariance(self, ell: np.ndarray, C_ell: np.ndarray,
                                 C_ell_err: np.ndarray,
                                 add_correlations: bool = True,
                                 filter_window: int = 50) -> np.ndarray:
        """
        Construct full covariance matrix with all contributions.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Power spectrum
            C_ell_err (ndarray): Instrumental uncertainties
            add_correlations (bool): Include off-diagonal correlations (default: True)
            filter_window (int): SG filter window for correlations (default: 50)
            
        Returns:
            ndarray: Full covariance matrix (N x N)
        """
        self.output.log_message("Constructing covariance matrix...")
        
        n = len(ell)
        cov = np.zeros((n, n))
        
        # Diagonal: sum of variances
        diagonal = self.construct_diagonal(ell, C_ell, C_ell_err)
        np.fill_diagonal(cov, diagonal)
        
        self.output.log_message(f"  Diagonal elements: cosmic variance + noise + systematics")
        
        # Off-diagonal: correlations
        if add_correlations:
            cov = self.add_filter_correlations(cov, ell, filter_window=filter_window)
            self.output.log_message(f"  Off-diagonal: filter-induced correlations (window={filter_window})")
        
        # Check positive definiteness
        eigenvalues = np.linalg.eigvalsh(cov)
        min_eigenvalue = np.min(eigenvalues)
        
        if min_eigenvalue <= 0:
            warnings.warn(f"Covariance not positive definite (min eigenvalue: {min_eigenvalue})")
            self.output.log_message(f"  WARNING: Adding regularization")
            cov += np.eye(n) * abs(min_eigenvalue) * 1.01
        
        # Condition number
        cond_number = np.linalg.cond(cov)
        self.output.log_message(f"  Condition number: {cond_number:.2e}")
        
        if cond_number > 1e10:
            self.output.log_message(f"  WARNING: Covariance poorly conditioned")
        
        self.output.log_message(f"  Shape: {cov.shape}")
        self.output.log_message("")
        
        return cov
    
    def invert_covariance(self, cov: np.ndarray,
                         regularization: float = 1e-10) -> np.ndarray:
        """
        Safely invert covariance matrix.
        
        Uses regularization if matrix is singular or poorly conditioned.
        
        Parameters:
            cov (ndarray): Covariance matrix
            regularization (float): Regularization factor (default: 1e-10)
            
        Returns:
            ndarray: Inverse covariance (precision matrix)
        """
        try:
            cov_inv = np.linalg.inv(cov)
            return cov_inv
        except np.linalg.LinAlgError:
            warnings.warn("Covariance singular, using pseudo-inverse with regularization")
            cov_reg = cov + np.eye(len(cov)) * regularization * np.max(np.diag(cov))
            return np.linalg.inv(cov_reg)
    
    def decompose_variance(self, ell: np.ndarray, C_ell: np.ndarray,
                          C_ell_err: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Decompose total variance into components.
        
        Useful for understanding dominant uncertainty sources.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Power spectrum
            C_ell_err (ndarray): Instrumental uncertainties
            
        Returns:
            dict: Variance components
        """
        components = {
            'instrumental': self.instrumental_noise(C_ell_err),
            'cosmic_variance': self.cosmic_variance(ell, C_ell),
            'foreground': self.foreground_uncertainty(ell, C_ell),
            'beam': self.beam_uncertainty(ell, C_ell)
        }
        
        # Total
        components['total'] = sum(components.values())
        
        # Fractional contributions
        total_variance = components['total']
        components['fractions'] = {
            key: val / total_variance
            for key, val in components.items()
            if key not in ['total', 'fractions']
        }
        
        return components
    
    def get_effective_errors(self, ell: np.ndarray, C_ell: np.ndarray,
                            C_ell_err: np.ndarray) -> np.ndarray:
        """
        Get effective error bars including all uncertainties.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Power spectrum
            C_ell_err (ndarray): Instrumental uncertainties
            
        Returns:
            ndarray: Effective error bars (sqrt of diagonal)
        """
        diagonal = self.construct_diagonal(ell, C_ell, C_ell_err)
        return np.sqrt(diagonal)
    
    def analyze_covariance(self, cov: np.ndarray) -> Dict[str, Any]:
        """
        Analyze properties of covariance matrix.
        
        Parameters:
            cov (ndarray): Covariance matrix
            
        Returns:
            dict: Analysis results
        """
        # Eigenvalue decomposition
        eigenvalues = np.linalg.eigvalsh(cov)
        
        # Correlation matrix
        diag_sqrt = np.sqrt(np.diag(cov))
        corr = cov / np.outer(diag_sqrt, diag_sqrt)
        
        # Off-diagonal statistics
        mask = ~np.eye(len(cov), dtype=bool)
        off_diag = corr[mask]
        
        analysis = {
            'shape': cov.shape,
            'condition_number': float(np.linalg.cond(cov)),
            'determinant': float(np.linalg.det(cov)),
            'trace': float(np.trace(cov)),
            'eigenvalues': {
                'min': float(np.min(eigenvalues)),
                'max': float(np.max(eigenvalues)),
                'mean': float(np.mean(eigenvalues))
            },
            'correlation': {
                'mean_off_diagonal': float(np.mean(np.abs(off_diag))),
                'max_off_diagonal': float(np.max(np.abs(off_diag))),
                'fraction_significant': float(np.sum(np.abs(off_diag) > 0.1) / len(off_diag))
            },
            'positive_definite': bool(np.all(eigenvalues > 0))
        }
        
        return analysis
