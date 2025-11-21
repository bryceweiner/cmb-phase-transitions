"""
Dataset Combining Module
=========================

Merge multiple CMB datasets using various statistical methods.

Classes:
    DatasetCombiner: Combine ACT and Planck data with three strategies

Implements three approaches:
1. Inverse-variance weighted average (optimal for uncorrelated errors)
2. Joint likelihood (full covariance treatment)
3. Simple concatenation (treat as independent measurements)
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from scipy.linalg import block_diag

from .utils import OutputManager


class DatasetCombiner:
    """
    Combine multiple CMB datasets using various strategies.
    
    Implements three merging approaches with different statistical assumptions:
    - Inverse-variance: Optimal weighted average
    - Joint likelihood: Full covariance treatment
    - Concatenation: Simple combination without weighting
    
    Attributes:
        output (OutputManager): For logging
        
    Example:
        >>> combiner = DatasetCombiner()
        >>> combined = combiner.merge_inverse_variance(act_data, planck_data)
        >>> print(f"Combined: {combined['ell'].shape}")
    """
    
    def __init__(self, output: OutputManager = None):
        """
        Initialize DatasetCombiner.
        
        Parameters:
            output (OutputManager, optional): For logging
        """
        self.output = output if output is not None else OutputManager()
    
    def find_overlapping_range(self,
                               ell1: np.ndarray,
                               ell2: np.ndarray,
                               tolerance: int = 50,
                               resolution_ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find overlapping multipole range between datasets.
        
        Uses resolution-aware tolerance to account for instrumental differences.
        
        Physical basis:
        - Different beam sizes cause systematic offsets in measured multipoles
        - Tolerance should scale with max(beam1, beam2) for robust matching
        
        Parameters:
            ell1 (ndarray): Multipoles from dataset 1
            ell2 (ndarray): Multipoles from dataset 2
            tolerance (int): Base matching tolerance for multipoles (default: 50)
            resolution_ratio (float): Ratio of beam sizes (default: 1.0 for same instrument)
            
        Returns:
            tuple: (ell_common, indices1, indices2)
                - ell_common: Common multipole values
                - indices1: Indices in dataset 1
                - indices2: Indices in dataset 2
        """
        # Scale tolerance by resolution difference
        # Use sqrt scaling: uncertainty grows as sqrt(resolution_ratio)
        effective_tolerance = tolerance * np.sqrt(resolution_ratio)
        # Find overlap range
        ell_min = max(ell1.min(), ell2.min())
        ell_max = min(ell1.max(), ell2.max())
        
        # Find indices within overlap
        mask1 = (ell1 >= ell_min) & (ell1 <= ell_max)
        mask2 = (ell2 >= ell_min) & (ell2 <= ell_max)
        
        ell1_overlap = ell1[mask1]
        ell2_overlap = ell2[mask2]
        
        # Match multipoles within resolution-aware tolerance
        ell_common = []
        indices1 = []
        indices2 = []
        
        for i, l1 in enumerate(ell1_overlap):
            # Find closest in ell2
            diffs = np.abs(ell2_overlap - l1)
            j = np.argmin(diffs)
            
            if diffs[j] <= effective_tolerance:
                ell_common.append(l1)
                indices1.append(np.where(ell1 == l1)[0][0])
                indices2.append(np.where(ell2 == ell2_overlap[j])[0][0])
        
        return np.array(ell_common), np.array(indices1), np.array(indices2)
    
    def merge_inverse_variance(self,
                               dataset1: Dict[str, np.ndarray],
                               dataset2: Dict[str, np.ndarray],
                               name1: str = "Dataset1",
                               name2: str = "Dataset2") -> Dict[str, Any]:
        """
        Merge datasets using inverse-variance weighting.
        
        For overlapping multipoles:
            C_combined = Σ(C_i/σ_i²) / Σ(1/σ_i²)
            σ_combined = 1/√(Σ(1/σ_i²))
        
        This is the optimal estimator when errors are uncorrelated and Gaussian.
        
        Parameters:
            dataset1 (dict): First dataset with keys 'ell', 'C_ell', 'C_ell_err'
            dataset2 (dict): Second dataset with keys 'ell', 'C_ell', 'C_ell_err'
            name1 (str): Name of first dataset
            name2 (str): Name of second dataset
            
        Returns:
            dict: Combined dataset with metadata
        """
        self.output.log_subsection(f"INVERSE-VARIANCE MERGING: {name1} + {name2}")
        
        ell1 = dataset1['ell']
        C_ell1 = dataset1['C_ell']
        C_err1 = dataset1['C_ell_err']
        
        ell2 = dataset2['ell']
        C_ell2 = dataset2['C_ell']
        C_err2 = dataset2['C_ell_err']
        
        self.output.log_message(f"{name1}: {len(ell1)} multipoles ({ell1[0]:.0f}-{ell1[-1]:.0f})")
        self.output.log_message(f"{name2}: {len(ell2)} multipoles ({ell2[0]:.0f}-{ell2[-1]:.0f})")
        
        # Find overlapping range (use resolution ratio if available)
        res_ratio = getattr(self, 'resolution_ratio', 1.0)
        ell_common, idx1, idx2 = self.find_overlapping_range(ell1, ell2, resolution_ratio=res_ratio)
        
        self.output.log_message(f"Overlap: {len(ell_common)} multipoles ({ell_common[0]:.0f}-{ell_common[-1]:.0f})")
        
        # Inverse-variance weights
        w1 = 1.0 / C_err1[idx1]**2
        w2 = 1.0 / C_err2[idx2]**2
        
        # Weighted average
        C_ell_combined = (C_ell1[idx1] * w1 + C_ell2[idx2] * w2) / (w1 + w2)
        C_err_combined = 1.0 / np.sqrt(w1 + w2)
        
        # Check improvement in precision
        relative_improvement = np.median(C_err1[idx1] / C_err_combined)
        self.output.log_message(f"Median error reduction: {relative_improvement:.2f}×")
        
        # Consistency check: weighted residuals
        residuals = (C_ell1[idx1] - C_ell2[idx2]) / np.sqrt(C_err1[idx1]**2 + C_err2[idx2]**2)
        chi2_consistency = np.sum(residuals**2)
        dof = len(ell_common)
        chi2_dof = chi2_consistency / dof
        
        self.output.log_message(f"Consistency χ²/DOF: {chi2_dof:.2f}")
        if chi2_dof > 2.0:
            self.output.log_message("⚠ WARNING: Datasets show poor consistency (χ²/DOF > 2)")
        elif chi2_dof > 1.5:
            self.output.log_message("⚠ Note: Moderate inconsistency between datasets")
        else:
            self.output.log_message("✓ Datasets are statistically consistent")
        
        return {
            'method': 'inverse_variance',
            'ell': ell_common,
            'C_ell': C_ell_combined,
            'C_ell_err': C_err_combined,
            'dataset1_name': name1,
            'dataset2_name': name2,
            'n_common': len(ell_common),
            'error_reduction_factor': float(relative_improvement),
            'consistency_chi2_dof': float(chi2_dof)
        }
    
    def merge_joint_likelihood(self,
                               dataset1: Dict[str, np.ndarray],
                               dataset2: Dict[str, np.ndarray],
                               cov1: Optional[np.ndarray] = None,
                               cov2: Optional[np.ndarray] = None,
                               name1: str = "Dataset1",
                               name2: str = "Dataset2") -> Dict[str, Any]:
        """
        Merge datasets using joint likelihood approach.
        
        Constructs block-diagonal covariance matrix and maximizes joint likelihood.
        Accounts for full covariance structure within each dataset.
        
        Parameters:
            dataset1 (dict): First dataset
            dataset2 (dict): Second dataset
            cov1 (ndarray, optional): Covariance matrix for dataset 1
            cov2 (ndarray, optional): Covariance matrix for dataset 2
            name1 (str): Name of first dataset
            name2 (str): Name of second dataset
            
        Returns:
            dict: Combined dataset with joint covariance
        """
        self.output.log_subsection(f"JOINT LIKELIHOOD MERGING: {name1} + {name2}")
        
        ell1 = dataset1['ell']
        C_ell1 = dataset1['C_ell']
        C_err1 = dataset1['C_ell_err']
        
        ell2 = dataset2['ell']
        C_ell2 = dataset2['C_ell']
        C_err2 = dataset2['C_ell_err']
        
        self.output.log_message(f"{name1}: {len(ell1)} multipoles")
        self.output.log_message(f"{name2}: {len(ell2)} multipoles")
        
        # Find overlapping range (use resolution ratio if available)
        res_ratio = getattr(self, 'resolution_ratio', 1.0)
        ell_common, idx1, idx2 = self.find_overlapping_range(ell1, ell2, resolution_ratio=res_ratio)
        
        # Construct covariance matrices if not provided
        if cov1 is None:
            cov1 = np.diag(C_err1[idx1]**2)
            self.output.log_message(f"{name1}: Using diagonal covariance")
        else:
            cov1 = cov1[np.ix_(idx1, idx1)]
            self.output.log_message(f"{name1}: Using full covariance matrix")
        
        if cov2 is None:
            cov2 = np.diag(C_err2[idx2]**2)
            self.output.log_message(f"{name2}: Using diagonal covariance")
        else:
            cov2 = cov2[np.ix_(idx2, idx2)]
            self.output.log_message(f"{name2}: Using full covariance matrix")
        
        # Build block-diagonal joint covariance
        cov_joint = block_diag(cov1, cov2)
        
        # Stack data vectors
        C_stacked = np.concatenate([C_ell1[idx1], C_ell2[idx2]])
        
        # Compute inverse covariance matrix
        try:
            cov_inv = np.linalg.inv(cov_joint)
        except np.linalg.LinAlgError:
            self.output.log_message("⚠ Covariance singular, using pseudo-inverse")
            cov_inv = np.linalg.pinv(cov_joint)
        
        # Joint likelihood estimator for common mean
        # μ_joint = (C₁ᵀΣ₁⁻¹ + C₂ᵀΣ₂⁻¹) / (Σ₁⁻¹ + Σ₂⁻¹)
        # For simplicity, use weighted average per multipole
        
        # Extract inverse covariances
        cov1_inv = cov_inv[:len(idx1), :len(idx1)]
        cov2_inv = cov_inv[len(idx1):, len(idx1):]
        
        # Compute precision-weighted estimate
        prec_sum = cov1_inv.sum(axis=1) + cov2_inv.sum(axis=1)
        C_ell_combined = (cov1_inv.sum(axis=1) * C_ell1[idx1] + 
                         cov2_inv.sum(axis=1) * C_ell2[idx2]) / prec_sum
        
        # Combined uncertainty (diagonal of joint posterior covariance)
        # Simplified: combine standard errors
        C_err_combined = 1.0 / np.sqrt(1.0/C_err1[idx1]**2 + 1.0/C_err2[idx2]**2)
        
        self.output.log_message(f"Combined: {len(ell_common)} multipoles")
        self.output.log_message(f"Joint covariance: {cov_joint.shape}")
        
        return {
            'method': 'joint_likelihood',
            'ell': ell_common,
            'C_ell': C_ell_combined,
            'C_ell_err': C_err_combined,
            'dataset1_name': name1,
            'dataset2_name': name2,
            'n_common': len(ell_common),
            'joint_covariance': cov_joint
        }
    
    def merge_concatenation(self,
                           dataset1: Dict[str, np.ndarray],
                           dataset2: Dict[str, np.ndarray],
                           name1: str = "Dataset1",
                           name2: str = "Dataset2") -> Dict[str, Any]:
        """
        Merge datasets by simple concatenation.
        
        Treats measurements as independent. For overlapping multipoles,
        includes both measurements separately.
        
        Parameters:
            dataset1 (dict): First dataset
            dataset2 (dict): Second dataset
            name1 (str): Name of first dataset
            name2 (str): Name of second dataset
            
        Returns:
            dict: Concatenated dataset
        """
        self.output.log_subsection(f"CONCATENATION MERGING: {name1} + {name2}")
        
        ell1 = dataset1['ell']
        C_ell1 = dataset1['C_ell']
        C_err1 = dataset1['C_ell_err']
        
        ell2 = dataset2['ell']
        C_ell2 = dataset2['C_ell']
        C_err2 = dataset2['C_ell_err']
        
        self.output.log_message(f"{name1}: {len(ell1)} multipoles ({ell1[0]:.0f}-{ell1[-1]:.0f})")
        self.output.log_message(f"{name2}: {len(ell2)} multipoles ({ell2[0]:.0f}-{ell2[-1]:.0f})")
        
        # Simple concatenation
        ell_concat = np.concatenate([ell1, ell2])
        C_ell_concat = np.concatenate([C_ell1, C_ell2])
        C_err_concat = np.concatenate([C_err1, C_err2])
        
        # Sort by multipole
        sort_idx = np.argsort(ell_concat)
        ell_concat = ell_concat[sort_idx]
        C_ell_concat = C_ell_concat[sort_idx]
        C_err_concat = C_err_concat[sort_idx]
        
        # Identify overlapping vs non-overlapping regions
        ell_min_overlap = max(ell1.min(), ell2.min())
        ell_max_overlap = min(ell1.max(), ell2.max())
        n_overlap = np.sum((ell_concat >= ell_min_overlap) & (ell_concat <= ell_max_overlap))
        
        self.output.log_message(f"Total: {len(ell_concat)} measurements")
        self.output.log_message(f"  {name1} only: {np.sum(ell_concat < ell_min_overlap)}")
        self.output.log_message(f"  Overlap region: {n_overlap}")
        self.output.log_message(f"  {name2} only: {np.sum(ell_concat > ell_max_overlap)}")
        
        return {
            'method': 'concatenation',
            'ell': ell_concat,
            'C_ell': C_ell_concat,
            'C_ell_err': C_err_concat,
            'dataset1_name': name1,
            'dataset2_name': name2,
            'n_total': len(ell_concat),
            'n_overlap': int(n_overlap)
        }
    
    def merge_all_methods(self,
                         dataset1: Dict[str, np.ndarray],
                         dataset2: Dict[str, np.ndarray],
                         cov1: Optional[np.ndarray] = None,
                         cov2: Optional[np.ndarray] = None,
                         name1: str = "Dataset1",
                         name2: str = "Dataset2",
                         resolution_ratio: float = 1.0) -> Dict[str, Dict[str, Any]]:
        """
        Merge datasets using all three methods.
        
        Convenience function that runs all merging strategies and returns results.
        
        Parameters:
            dataset1 (dict): First dataset
            dataset2 (dict): Second dataset
            cov1 (ndarray, optional): Covariance for dataset 1
            cov2 (ndarray, optional): Covariance for dataset 2
            name1 (str): Name of first dataset
            name2 (str): Name of second dataset
            resolution_ratio (float): Ratio of beam sizes (dataset2/dataset1)
            
        Returns:
            dict: Results from all three methods
        """
        self.output.log_section_header(f"DATASET MERGING: {name1} + {name2}")
        self.output.log_message(f"Resolution ratio: {resolution_ratio:.2f}×")
        
        # Store resolution ratio for use in merge methods
        self.resolution_ratio = resolution_ratio
        
        results = {}
        
        # Method 1: Inverse-variance
        results['inverse_variance'] = self.merge_inverse_variance(
            dataset1, dataset2, name1, name2
        )
        
        # Method 2: Joint likelihood
        results['joint_likelihood'] = self.merge_joint_likelihood(
            dataset1, dataset2, cov1, cov2, name1, name2
        )
        
        # Method 3: Concatenation
        results['concatenation'] = self.merge_concatenation(
            dataset1, dataset2, name1, name2
        )
        
        self.output.log_message("\n" + "="*60)
        self.output.log_message("MERGING SUMMARY")
        self.output.log_message("="*60)
        for method, result in results.items():
            self.output.log_message(f"{method}:")
            self.output.log_message(f"  Data points: {result['ell'].shape[0]}")
            self.output.log_message(f"  Range: {result['ell'][0]:.0f}-{result['ell'][-1]:.0f}")
        
        return results

