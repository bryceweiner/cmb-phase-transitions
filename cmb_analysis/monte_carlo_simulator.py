"""
Monte Carlo Simulator Module
=============================

Generate realistic null CMB E-mode realizations for false positive testing.

This module creates mock datasets from ΛCDM predictions without transitions,
including proper covariance matrices (cosmic variance + instrumental noise).

Classes:
    MonteCarloSimulator: Generate null hypothesis CMB realizations

Paper reference: Statistical validation methodology
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple, Optional, Dict, Any
import warnings

from .utils import OutputManager
from .constants import H_RECOMB, C_LIGHT, HBAR, G_NEWTON

# Check for PyTorch with MPS support (Apple Silicon)
try:
    import torch
    TORCH_AVAILABLE = True
    MPS_AVAILABLE = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
except ImportError:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False


class MonteCarloSimulator:
    """
    Generate null hypothesis CMB E-mode realizations.
    
    Creates mock datasets from smooth ΛCDM power spectrum without transitions,
    properly accounting for cosmic variance and instrumental uncertainties.
    
    Attributes:
        output (OutputManager): For logging
        f_sky (float): Sky fraction observed
        
    Example:
        >>> simulator = MonteCarloSimulator()
        >>> ell_sim, C_ell_sim, C_err_sim = simulator.generate_realization(ell, C_ell_smooth)
    """
    
    def __init__(self, output: OutputManager = None, f_sky: float = 0.4, use_mps: bool = True):
        """
        Initialize MonteCarloSimulator.
        
        Parameters:
            output (OutputManager, optional): For logging
            f_sky (float): Sky fraction (default: 0.4 for ACT)
            use_mps (bool): Use Apple Silicon MPS if available (default: True)
        """
        self.output = output if output is not None else OutputManager()
        self.f_sky = f_sky
        
        # Check hardware acceleration availability
        self.use_mps = use_mps and MPS_AVAILABLE
        self.device = None
        
        if self.use_mps:
            self.device = torch.device("mps")
            self.output.log_message("✓ Apple Silicon MPS detected and enabled")
            self.output.log_message("  Monte Carlo simulations will use GPU acceleration")
        elif TORCH_AVAILABLE:
            self.device = torch.device("cpu")
            self.output.log_message("PyTorch available but MPS not detected")
        else:
            self.output.log_message("Running on CPU (PyTorch not available)")
    
    def compute_smooth_spectrum(self, ell: np.ndarray, C_ell: np.ndarray,
                               poly_order: int = 9) -> np.ndarray:
        """
        Compute smooth ΛCDM-like spectrum by polynomial fitting.
        
        Removes any sharp features/transitions to create null hypothesis spectrum.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Observed power spectrum
            poly_order (int): Polynomial order for smooth fit (default: 9)
            
        Returns:
            ndarray: Smooth power spectrum without transitions
        """
        # Fit high-order polynomial to capture smooth ΛCDM structure
        coeffs = np.polyfit(ell, C_ell, deg=poly_order)
        C_smooth = np.polyval(coeffs, ell)
        
        # Ensure positivity
        C_smooth = np.maximum(C_smooth, 1e-20)
        
        return C_smooth
    
    def construct_covariance_matrix(self, ell: np.ndarray, C_ell: np.ndarray,
                                   C_ell_err: np.ndarray,
                                   include_cosmic_variance: bool = True) -> np.ndarray:
        """
        Construct full covariance matrix for CMB power spectrum.
        
        Includes:
        - Cosmic variance (sample variance from finite sky coverage)
        - Instrumental noise (diagonal from error bars)
        - Foreground uncertainties (simplified as additional diagonal)
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Power spectrum
            C_ell_err (ndarray): Instrumental uncertainties
            include_cosmic_variance (bool): Include cosmic variance (default: True)
            
        Returns:
            ndarray: Covariance matrix (N x N)
        """
        n = len(ell)
        cov = np.zeros((n, n))
        
        # Diagonal: instrumental noise
        for i in range(n):
            cov[i, i] = C_ell_err[i]**2
        
        # Add cosmic variance if requested
        if include_cosmic_variance:
            for i in range(n):
                # Cosmic variance: σ²_cosmic ≈ 2/(2ℓ+1) × (C_ℓ)² / f_sky
                # For polarization, factor is 2 (E and B modes)
                cosmic_var = 2 * C_ell[i]**2 / ((2 * ell[i] + 1) * self.f_sky)
                cov[i, i] += cosmic_var
        
        # Add small off-diagonal correlations from beam and filtering
        # (simplified model: nearest neighbors have 5% correlation)
        for i in range(n-1):
            correlation = 0.05 * np.sqrt(cov[i, i] * cov[i+1, i+1])
            cov[i, i+1] = correlation
            cov[i+1, i] = correlation
        
        return cov
    
    def generate_realization(self, ell: np.ndarray, C_ell_smooth: np.ndarray,
                           cov_matrix: np.ndarray,
                           seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate single mock CMB realization.
        
        Draws from multivariate Gaussian with smooth spectrum and full covariance.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell_smooth (ndarray): Smooth ΛCDM spectrum (no transitions)
            cov_matrix (ndarray): Covariance matrix
            seed (int, optional): Random seed for reproducibility
            
        Returns:
            tuple: (C_ell_mock, C_ell_err_mock)
                - C_ell_mock: Mock power spectrum realization
                - C_ell_err_mock: Uncertainties (sqrt of diagonal)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Draw from multivariate Gaussian
        try:
            C_ell_mock = np.random.multivariate_normal(C_ell_smooth, cov_matrix)
        except np.linalg.LinAlgError:
            # If covariance is singular, add small regularization
            warnings.warn("Covariance matrix singular, adding regularization")
            cov_reg = cov_matrix + np.eye(len(cov_matrix)) * 1e-10 * np.max(np.diag(cov_matrix))
            C_ell_mock = np.random.multivariate_normal(C_ell_smooth, cov_reg)
        
        # Ensure positivity (physical requirement)
        C_ell_mock = np.maximum(C_ell_mock, 1e-20)
        
        # Uncertainties from diagonal of covariance
        C_ell_err_mock = np.sqrt(np.diag(cov_matrix))
        
        return C_ell_mock, C_ell_err_mock
    
    def generate_realization_mps(self, mean_tensor: torch.Tensor,
                                cov_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate mock realization using MPS acceleration.
        
        Uses Cholesky decomposition on GPU for faster sampling.
        
        Parameters:
            mean_tensor (torch.Tensor): Mean spectrum on MPS device
            cov_tensor (torch.Tensor): Covariance matrix on MPS device
            
        Returns:
            ndarray: Mock power spectrum realization
        """
        # Cholesky decomposition on MPS
        try:
            L = torch.linalg.cholesky(cov_tensor)
        except RuntimeError:
            # Add regularization if singular
            cov_reg = cov_tensor + torch.eye(cov_tensor.shape[0], device=self.device) * 1e-10
            L = torch.linalg.cholesky(cov_reg)
        
        # Sample standard normal
        z = torch.randn(mean_tensor.shape[0], device=self.device)
        
        # Transform: x = μ + L @ z
        sample = mean_tensor + L @ z
        
        # Ensure positivity
        sample = torch.maximum(sample, torch.tensor(1e-20, device=self.device))
        
        return sample.cpu().numpy()
    
    def generate_ensemble(self, ell: np.ndarray, C_ell: np.ndarray,
                         C_ell_err: np.ndarray, n_realizations: int = 10000,
                         use_smooth: bool = True) -> Dict[str, Any]:
        """
        Generate ensemble of null hypothesis realizations.
        
        Creates n_realizations mock datasets for false positive testing.
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Observed power spectrum
            C_ell_err (ndarray): Uncertainties
            n_realizations (int): Number of mock datasets (default: 10000)
            use_smooth (bool): Use smooth spectrum as mean (default: True)
            
        Returns:
            dict: Ensemble data with keys:
                - 'realizations': Array of mock spectra (n_realizations x n_multipoles)
                - 'ell': Multipole values
                - 'C_smooth': Smooth spectrum used as mean
                - 'covariance': Covariance matrix used
                - 'n_realizations': Number of realizations
        """
        self.output.log_section_header("MONTE CARLO NULL HYPOTHESIS GENERATION")
        self.output.log_message(f"Generating {n_realizations} null realizations...")
        self.output.log_message(f"Sky fraction: {self.f_sky}")
        self.output.log_message("")
        
        # Compute smooth spectrum (null hypothesis)
        if use_smooth:
            C_smooth = self.compute_smooth_spectrum(ell, C_ell)
            self.output.log_message("Smooth spectrum computed (9th order polynomial)")
        else:
            C_smooth = C_ell.copy()
            self.output.log_message("Using observed spectrum as template")
        
        # Construct covariance matrix
        cov_matrix = self.construct_covariance_matrix(ell, C_smooth, C_ell_err)
        self.output.log_message(f"Covariance matrix: {len(ell)} x {len(ell)}")
        
        # Check condition number
        cond_number = np.linalg.cond(cov_matrix)
        self.output.log_message(f"Condition number: {cond_number:.2e}")
        
        if cond_number > 1e10:
            self.output.log_message("WARNING: Covariance matrix poorly conditioned")
        
        # Generate ensemble (use MPS if available)
        realizations = []
        
        # Progress reporting
        checkpoints = [int(n_realizations * f) for f in [0.1, 0.25, 0.5, 0.75, 1.0]]
        
        if self.use_mps:
            # MPS-accelerated generation
            self.output.log_message("Using MPS (Apple Silicon GPU) for acceleration...")
            
            # Transfer data to MPS device
            mean_tensor = torch.tensor(C_smooth, dtype=torch.float32, device=self.device)
            cov_tensor = torch.tensor(cov_matrix, dtype=torch.float32, device=self.device)
            
            # Precompute Cholesky decomposition once
            try:
                L = torch.linalg.cholesky(cov_tensor)
            except RuntimeError:
                cov_reg = cov_tensor + torch.eye(cov_tensor.shape[0], device=self.device) * 1e-10
                L = torch.linalg.cholesky(cov_reg)
            
            for i in range(n_realizations):
                # Sample on GPU
                z = torch.randn(mean_tensor.shape[0], device=self.device)
                sample = mean_tensor + L @ z
                sample = torch.maximum(sample, torch.tensor(1e-20, device=self.device))
                realizations.append(sample.cpu().numpy())
                
                if (i+1) in checkpoints:
                    progress = (i+1) / n_realizations * 100
                    self.output.log_message(f"Progress: {progress:.0f}% ({i+1}/{n_realizations})")
        else:
            # CPU-based generation (original method)
            for i in range(n_realizations):
                C_ell_mock, _ = self.generate_realization(ell, C_smooth, cov_matrix, seed=i)
                realizations.append(C_ell_mock)
                
                if (i+1) in checkpoints:
                    progress = (i+1) / n_realizations * 100
                    self.output.log_message(f"Progress: {progress:.0f}% ({i+1}/{n_realizations})")
        
        realizations = np.array(realizations)
        
        self.output.log_message("")
        self.output.log_message("Ensemble generation complete")
        self.output.log_message(f"Shape: {realizations.shape}")
        self.output.log_message(f"Mean C_ell: {realizations.mean():.3e}")
        self.output.log_message(f"Std C_ell: {realizations.std():.3e}")
        self.output.log_message("")
        
        return {
            'realizations': realizations,
            'ell': ell.copy(),
            'C_smooth': C_smooth,
            'covariance': cov_matrix,
            'n_realizations': n_realizations,
            'f_sky': self.f_sky
        }
    
    def compute_ensemble_statistics(self, ensemble: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute statistical properties of ensemble.
        
        Parameters:
            ensemble (dict): Output from generate_ensemble()
            
        Returns:
            dict: Statistics including mean, std, percentiles
        """
        realizations = ensemble['realizations']
        
        stats = {
            'mean': np.mean(realizations, axis=0),
            'median': np.median(realizations, axis=0),
            'std': np.std(realizations, axis=0),
            'percentile_16': np.percentile(realizations, 16, axis=0),
            'percentile_84': np.percentile(realizations, 84, axis=0),
            'percentile_2p5': np.percentile(realizations, 2.5, axis=0),
            'percentile_97p5': np.percentile(realizations, 97.5, axis=0),
            'min': np.min(realizations, axis=0),
            'max': np.max(realizations, axis=0)
        }
        
        return stats
    
    def save_ensemble(self, ensemble: Dict[str, Any], filename: str):
        """
        Save ensemble to compressed numpy file.
        
        Parameters:
            ensemble (dict): Output from generate_ensemble()
            filename (str): Output filename (.npz)
        """
        np.savez_compressed(
            filename,
            realizations=ensemble['realizations'],
            ell=ensemble['ell'],
            C_smooth=ensemble['C_smooth'],
            covariance=ensemble['covariance'],
            n_realizations=ensemble['n_realizations'],
            f_sky=ensemble['f_sky']
        )
        self.output.log_message(f"Ensemble saved to: {filename}")
    
    def load_ensemble(self, filename: str) -> Dict[str, Any]:
        """
        Load ensemble from compressed numpy file.
        
        Parameters:
            filename (str): Input filename (.npz)
            
        Returns:
            dict: Ensemble data (same format as generate_ensemble())
        """
        data = np.load(filename)
        
        ensemble = {
            'realizations': data['realizations'],
            'ell': data['ell'],
            'C_smooth': data['C_smooth'],
            'covariance': data['covariance'],
            'n_realizations': int(data['n_realizations']),
            'f_sky': float(data['f_sky'])
        }
        
        self.output.log_message(f"Ensemble loaded from: {filename}")
        self.output.log_message(f"Realizations: {ensemble['n_realizations']}")
        
        return ensemble
    
    def run_dataset_specific_simulations(self,
                                        dataset_name: str,
                                        ell: np.ndarray,
                                        C_ell: np.ndarray,
                                        C_ell_err: np.ndarray,
                                        n_realizations: int = 1000,
                                        f_sky_override: Optional[float] = None) -> Dict[str, Any]:
        """
        Run Monte Carlo simulations specific to a single dataset.
        
        Generates null realizations using dataset-specific properties
        (sky fraction, covariance structure, multipole range).
        
        Parameters:
            dataset_name (str): Name of dataset (e.g., "ACT", "Planck")
            ell (ndarray): Multipole values
            C_ell (ndarray): Power spectrum
            C_ell_err (ndarray): Uncertainties
            n_realizations (int): Number of simulations (default: 1000)
            f_sky_override (float, optional): Override sky fraction
            
        Returns:
            dict: Dataset-specific ensemble with metadata
        """
        self.output.log_subsection(f"MONTE CARLO SIMULATIONS: {dataset_name}")
        self.output.log_message(f"Dataset: {dataset_name}")
        self.output.log_message(f"Multipoles: {len(ell)} ({ell[0]:.0f}-{ell[-1]:.0f})")
        self.output.log_message(f"Realizations: {n_realizations}")
        
        # Use dataset-specific sky fraction if provided
        original_f_sky = self.f_sky
        if f_sky_override is not None:
            self.f_sky = f_sky_override
            self.output.log_message(f"Sky fraction: {f_sky_override:.3f}")
        
        # Generate ensemble
        ensemble = self.generate_ensemble(
            ell, C_ell, C_ell_err,
            n_realizations=n_realizations,
            use_smooth=True
        )
        
        # Restore original sky fraction
        self.f_sky = original_f_sky
        
        # Add dataset metadata
        ensemble['dataset_name'] = dataset_name
        ensemble['multipole_range'] = (float(ell[0]), float(ell[-1]))
        
        return ensemble
    
    def run_act_simulations(self,
                           ell: np.ndarray,
                           C_ell: np.ndarray,
                           C_ell_err: np.ndarray,
                           n_realizations: int = 1000) -> Dict[str, Any]:
        """
        Convenience method for ACT DR6 simulations.
        
        Uses ACT-specific parameters (f_sky = 0.4).
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Power spectrum
            C_ell_err (ndarray): Uncertainties
            n_realizations (int): Number of simulations (default: 1000)
            
        Returns:
            dict: ACT-specific ensemble
        """
        return self.run_dataset_specific_simulations(
            "ACT_DR6",
            ell, C_ell, C_ell_err,
            n_realizations=n_realizations,
            f_sky_override=0.4  # ACT sky fraction
        )
    
    def run_planck_simulations(self,
                              ell: np.ndarray,
                              C_ell: np.ndarray,
                              C_ell_err: np.ndarray,
                              n_realizations: int = 1000) -> Dict[str, Any]:
        """
        Convenience method for Planck 2018 simulations.
        
        Uses Planck-specific parameters (f_sky ≈ 0.8 for full mission).
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Power spectrum
            C_ell_err (ndarray): Uncertainties
            n_realizations (int): Number of simulations (default: 1000)
            
        Returns:
            dict: Planck-specific ensemble
        """
        return self.run_dataset_specific_simulations(
            "Planck_2018",
            ell, C_ell, C_ell_err,
            n_realizations=n_realizations,
            f_sky_override=0.8  # Planck sky fraction
        )
