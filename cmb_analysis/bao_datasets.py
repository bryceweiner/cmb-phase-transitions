"""
BAO Datasets Module
===================

Centralized management of all BAO survey datasets with proper covariances
and systematic error budgets.

Datasets included:
- BOSS DR12 (z=0.38-0.61)
- eBOSS DR16 LRG, ELG, QSO (z=0.7-2.3)
- DESI Year 1 (z=0.4-4.2, if available)
- 6dFGS (z=0.1)
- WiggleZ (z=0.44-0.73)

Classes:
    BAODataset: Single survey with data, errors, correlations, systematics
    BAODatasetManager: Load and manage all surveys

References:
    BOSS DR12: arXiv:1607.03155
    eBOSS DR16: arXiv:2007.08991
    6dFGS: arXiv:1106.3366
    WiggleZ: arXiv:1108.2635
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .utils import OutputManager


@dataclass
class BAODataset:
    """
    Single BAO survey dataset with full error budget.
    
    Attributes:
        name (str): Survey name
        reference (str): Paper reference (arXiv number)
        redshifts (np.ndarray): Effective redshifts
        values (np.ndarray): D_M/r_d or D_V/r_d measurements
        stat_errors (np.ndarray): Statistical uncertainties
        correlation (np.ndarray): Correlation matrix
        systematics (Dict[str, float]): Systematic error budget (fractional)
        observable (str): 'D_M/r_d' or 'D_V/r_d'
        tracer (str): Galaxy type (LRG, ELG, QSO, etc.)
    """
    name: str
    reference: str
    redshifts: np.ndarray
    values: np.ndarray
    stat_errors: np.ndarray
    correlation: np.ndarray
    systematics: Dict[str, float]
    observable: str = 'D_M/r_d'
    tracer: str = 'LRG'
    
    def __post_init__(self):
        """Convert lists to arrays if needed."""
        self.redshifts = np.asarray(self.redshifts)
        self.values = np.asarray(self.values)
        self.stat_errors = np.asarray(self.stat_errors)
        self.correlation = np.asarray(self.correlation)
    
    @property
    def n_bins(self) -> int:
        """Number of redshift bins."""
        return len(self.redshifts)
    
    @property
    def dof(self) -> int:
        """Degrees of freedom for chi-squared test."""
        return self.n_bins  # No fitted parameters!
    
    def statistical_covariance(self) -> np.ndarray:
        """
        Build statistical covariance from correlation matrix.
        
        Returns:
            ndarray: Statistical covariance matrix
        """
        n = len(self.stat_errors)
        cov = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                cov[i, j] = self.correlation[i, j] * self.stat_errors[i] * self.stat_errors[j]
        
        return cov
    
    def systematic_covariance(self) -> np.ndarray:
        """
        Build systematic error covariance.
        
        Systematics add in quadrature and are assumed uncorrelated between bins.
        
        Returns:
            ndarray: Systematic covariance matrix (diagonal)
        """
        # Total systematic error per bin (quadrature sum)
        sys_total = np.zeros(self.n_bins)
        
        for sys_name, sys_frac in self.systematics.items():
            # Fractional systematic → absolute
            sys_abs = sys_frac * self.values
            sys_total += sys_abs**2
        
        sys_total = np.sqrt(sys_total)
        
        # Diagonal covariance (systematics assumed uncorrelated between bins)
        return np.diag(sys_total**2)
    
    def total_covariance(self, include_systematics: bool = True) -> np.ndarray:
        """
        Build total covariance matrix.
        
        Parameters:
            include_systematics (bool): Include systematic errors
            
        Returns:
            ndarray: Total covariance matrix
        """
        cov_stat = self.statistical_covariance()
        
        if include_systematics:
            cov_sys = self.systematic_covariance()
            return cov_stat + cov_sys
        
        return cov_stat
    
    def total_errors(self, include_systematics: bool = True) -> np.ndarray:
        """
        Total errors (diagonal elements of covariance).
        
        Parameters:
            include_systematics (bool): Include systematic errors
            
        Returns:
            ndarray: Total errors
        """
        cov = self.total_covariance(include_systematics)
        return np.sqrt(np.diag(cov))
    
    def summary(self) -> Dict[str, Any]:
        """
        Dataset summary for reporting.
        
        Returns:
            dict: Summary information
        """
        sys_total = self.total_errors(True) - self.total_errors(False)
        
        return {
            'name': self.name,
            'reference': self.reference,
            'n_bins': self.n_bins,
            'z_range': (float(self.redshifts.min()), float(self.redshifts.max())),
            'observable': self.observable,
            'tracer': self.tracer,
            'mean_stat_error': float(np.mean(self.stat_errors)),
            'mean_sys_error': float(np.mean(sys_total)),
            'systematic_sources': list(self.systematics.keys())
        }


class BAODatasetManager:
    """
    Manage all BAO survey datasets.
    
    Provides access to BOSS, eBOSS, DESI, 6dFGS, WiggleZ with proper
    error budgets and correlations.
    
    Attributes:
        output (OutputManager): For logging
        datasets (Dict[str, BAODataset]): All loaded datasets
        
    Example:
        >>> manager = BAODatasetManager()
        >>> boss = manager.get_dataset('BOSS_DR12')
        >>> print(f"BOSS has {boss.n_bins} redshift bins")
    """
    
    def __init__(self, output: OutputManager = None):
        """
        Initialize BAODatasetManager.
        
        Parameters:
            output (OutputManager, optional): For logging
        """
        self.output = output if output is not None else OutputManager()
        self.datasets = {}
        self._load_all_datasets()
    
    def _load_all_datasets(self):
        """Load all available BAO datasets."""
        self.datasets['BOSS_DR12'] = self._load_boss_dr12()
        self.datasets['eBOSS_DR16_LRG'] = self._load_eboss_lrg()
        self.datasets['eBOSS_DR16_ELG'] = self._load_eboss_elg()
        self.datasets['eBOSS_DR16_QSO'] = self._load_eboss_qso()
        self.datasets['6dFGS'] = self._load_6dfgs()
        self.datasets['WiggleZ_z044'] = self._load_wigglez_z044()
        self.datasets['WiggleZ_z060'] = self._load_wigglez_z060()
        self.datasets['WiggleZ_z073'] = self._load_wigglez_z073()
        
        # DESI Y1 (if data is finalized)
        try:
            self.datasets['DESI_Y1_BGS'] = self._load_desi_bgs()
            self.datasets['DESI_Y1_LRG'] = self._load_desi_lrg()
        except Exception:
            self.output.log_message("Note: DESI Y1 data not yet available")
    
    def _load_boss_dr12(self) -> BAODataset:
        """
        Load BOSS DR12 consensus BAO measurements.
        
        Reference: Alam et al. MNRAS 470, 2617 (2017), arXiv:1607.03155
        Table 2: Consensus measurements combining LOWZ and CMASS samples
        
        Returns:
            BAODataset: BOSS DR12 data
        """
        return BAODataset(
            name='BOSS DR12',
            reference='arXiv:1607.03155',
            redshifts=np.array([0.38, 0.51, 0.61]),
            values=np.array([10.27, 13.37, 15.23]),
            stat_errors=np.array([0.15, 0.15, 0.17]),
            correlation=np.array([
                [1.00, 0.61, 0.49],
                [0.61, 1.00, 0.71],
                [0.49, 0.71, 1.00]
            ]),
            systematics={
                'imaging': 0.003,      # 0.3% imaging systematics
                'fiber_collisions': 0.005,  # 0.5% fiber effects
                'nonlinear': 0.010     # 1.0% non-linear structure
            },
            observable='D_M/r_d',
            tracer='LRG+CMASS'
        )
    
    def _load_eboss_lrg(self) -> BAODataset:
        """
        Load eBOSS DR16 LRG sample BAO measurements.
        
        Reference: Alam et al. PRD 103, 083533 (2021), arXiv:2007.08991
        Table 3: LRG consensus measurements
        
        Returns:
            BAODataset: eBOSS LRG data
        """
        return BAODataset(
            name='eBOSS DR16 LRG',
            reference='arXiv:2007.08991',
            redshifts=np.array([0.698, 0.874]),
            values=np.array([17.65, 19.60]),  # D_M/r_d from consensus
            stat_errors=np.array([0.42, 0.48]),
            correlation=np.array([
                [1.00, 0.58],
                [0.58, 1.00]
            ]),
            systematics={
                'imaging': 0.004,
                'photo_z': 0.003,  # Photometric redshift uncertainties
                'nonlinear': 0.015
            },
            observable='D_M/r_d',
            tracer='LRG'
        )
    
    def _load_eboss_elg(self) -> BAODataset:
        """
        Load eBOSS DR16 ELG sample BAO measurements.
        
        Reference: arXiv:2007.08991, Table 5
        
        Returns:
            BAODataset: eBOSS ELG data
        """
        return BAODataset(
            name='eBOSS DR16 ELG',
            reference='arXiv:2007.08991',
            redshifts=np.array([0.845]),  # Single bin
            values=np.array([19.77]),
            stat_errors=np.array([0.99]),  # Larger errors for ELG
            correlation=np.array([[1.00]]),
            systematics={
                'imaging': 0.005,
                'elg_selection': 0.008,  # ELG-specific
                'nonlinear': 0.015
            },
            observable='D_M/r_d',
            tracer='ELG'
        )
    
    def _load_eboss_qso(self) -> BAODataset:
        """
        Load eBOSS DR16 Quasar sample BAO measurements.
        
        Reference: arXiv:2007.08991, Table 6
        High-redshift quasar Lyman-α forest BAO
        
        Returns:
            BAODataset: eBOSS QSO data
        """
        return BAODataset(
            name='eBOSS DR16 QSO',
            reference='arXiv:2007.08991',
            redshifts=np.array([1.48, 2.33]),  # Lyman-α forest
            values=np.array([26.07, 37.50]),  # D_M/r_d at high-z
            stat_errors=np.array([0.67, 1.85]),
            correlation=np.array([
                [1.00, 0.42],
                [0.42, 1.00]
            ]),
            systematics={
                'continuum_fitting': 0.010,  # Lyman-α specific
                'metal_contamination': 0.008,
                'nonlinear': 0.020  # Larger at high-z
            },
            observable='D_M/r_d',
            tracer='QSO_Lya'
        )
    
    def _load_6dfgs(self) -> BAODataset:
        """
        Load 6dFGS low-redshift BAO measurement.
        
        Reference: Beutler et al. MNRAS 416, 3017 (2011), arXiv:1106.3366
        Uses D_V (spherically averaged distance)
        
        Returns:
            BAODataset: 6dFGS data
        """
        # 6dFGS measures D_V/r_d, not D_M/r_d
        # D_V = [D_M² × D_H]^(1/3) (spherical average)
        # At low z, D_V ≈ D_M (to good approximation)
        
        return BAODataset(
            name='6dFGS',
            reference='arXiv:1106.3366',
            redshifts=np.array([0.106]),
            values=np.array([3.047]),  # D_V/r_d
            stat_errors=np.array([0.137]),  # ~4.5% error
            correlation=np.array([[1.00]]),
            systematics={
                'peculiar_velocities': 0.015,  # Large at low-z
                'photometry': 0.005,
                'sample_selection': 0.008
            },
            observable='D_V/r_d',
            tracer='Galaxies'
        )
    
    def _load_wigglez_z044(self) -> BAODataset:
        """Load WiggleZ z=0.44 bin."""
        return BAODataset(
            name='WiggleZ z=0.44',
            reference='arXiv:1108.2635',
            redshifts=np.array([0.44]),
            values=np.array([11.53]),  # D_V/r_d
            stat_errors=np.array([0.42]),
            correlation=np.array([[1.00]]),
            systematics={
                'elg_systematics': 0.012,  # Emission-line specific
                'photoz': 0.008,
                'nonlinear': 0.010
            },
            observable='D_V/r_d',
            tracer='ELG'
        )
    
    def _load_wigglez_z060(self) -> BAODataset:
        """Load WiggleZ z=0.60 bin."""
        return BAODataset(
            name='WiggleZ z=0.60',
            reference='arXiv:1108.2635',
            redshifts=np.array([0.60]),
            values=np.array([15.38]),  # D_V/r_d
            stat_errors=np.array([0.47]),
            correlation=np.array([[1.00]]),
            systematics={
                'elg_systematics': 0.012,
                'photoz': 0.008,
                'nonlinear': 0.010
            },
            observable='D_V/r_d',
            tracer='ELG'
        )
    
    def _load_wigglez_z073(self) -> BAODataset:
        """Load WiggleZ z=0.73 bin."""
        return BAODataset(
            name='WiggleZ z=0.73',
            reference='arXiv:1108.2635',
            redshifts=np.array([0.73]),
            values=np.array([18.57]),  # D_V/r_d
            stat_errors=np.array([0.57]),
            correlation=np.array([[1.00]]),
            systematics={
                'elg_systematics': 0.012,
                'photoz': 0.008,
                'nonlinear': 0.010
            },
            observable='D_V/r_d',
            tracer='ELG'
        )
    
    def _load_desi_bgs(self) -> BAODataset:
        """
        Load DESI Year 1 BGS (Bright Galaxy Sample).
        
        Reference: DESI Collaboration, arXiv:2404.03000 (2024)
        Note: Data may be preliminary - check publication status
        
        Returns:
            BAODataset: DESI BGS data
        """
        # NOTE: DESI Y1 final results released April 2024
        # Using preliminary values from early release
        return BAODataset(
            name='DESI Y1 BGS',
            reference='arXiv:2404.03000',
            redshifts=np.array([0.30]),
            values=np.array([7.93]),  # D_M/r_d
            stat_errors=np.array([0.16]),
            correlation=np.array([[1.00]]),
            systematics={
                'imaging': 0.002,  # Improved DESI pipeline
                'fiber': 0.003,
                'nonlinear': 0.008
            },
            observable='D_M/r_d',
            tracer='BGS'
        )
    
    def _load_desi_lrg(self) -> BAODataset:
        """Load DESI Year 1 LRG sample."""
        return BAODataset(
            name='DESI Y1 LRG',
            reference='arXiv:2404.03000',
            redshifts=np.array([0.51, 0.71]),
            values=np.array([13.62, 17.88]),  # D_M/r_d
            stat_errors=np.array([0.18, 0.24]),
            correlation=np.array([
                [1.00, 0.55],
                [0.55, 1.00]
            ]),
            systematics={
                'imaging': 0.002,
                'fiber': 0.003,
                'nonlinear': 0.012
            },
            observable='D_M/r_d',
            tracer='LRG'
        )
    
    def get_dataset(self, name: str) -> Optional[BAODataset]:
        """
        Get specific dataset by name.
        
        Parameters:
            name (str): Dataset name
            
        Returns:
            BAODataset or None: Requested dataset
        """
        return self.datasets.get(name)
    
    def get_all_datasets(self) -> Dict[str, BAODataset]:
        """
        Get all loaded datasets.
        
        Returns:
            dict: All datasets
        """
        return self.datasets
    
    def list_datasets(self) -> List[str]:
        """
        List available dataset names.
        
        Returns:
            list: Dataset names
        """
        return list(self.datasets.keys())
    
    def print_summary(self):
        """Print summary of all datasets."""
        self.output.log_section_header("AVAILABLE BAO DATASETS")
        self.output.log_message("")
        self.output.log_message(f"{'Name':<25} {'Reference':<20} {'z-range':<15} {'Bins':<6} {'Observable':<12}")
        self.output.log_message("-" * 85)
        
        for name, dataset in self.datasets.items():
            summary = dataset.summary()
            z_min, z_max = summary['z_range']
            z_range_str = f"{z_min:.2f}-{z_max:.2f}" if z_min != z_max else f"{z_min:.2f}"
            
            self.output.log_message(
                f"{summary['name']:<25} {summary['reference']:<20} "
                f"{z_range_str:<15} {summary['n_bins']:<6} {summary['observable']:<12}"
            )
        
        self.output.log_message("")
        self.output.log_message(f"Total datasets: {len(self.datasets)}")
        self.output.log_message(f"Total redshift bins: {sum(d.n_bins for d in self.datasets.values())}")
        self.output.log_message("")

