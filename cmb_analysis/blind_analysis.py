"""
Blind Analysis Manager Module
==============================

Implements blinding protocols for unbiased feature detection.

Prevents researcher bias by masking data during analysis development
and maintaining clear distinction between exploratory and confirmatory analysis.

Classes:
    BlindAnalysisManager: Manage blinding and unblinding protocols

Paper reference: Methods, analysis protocol
"""

import numpy as np
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from .utils import OutputManager


class BlindAnalysisManager:
    """
    Manage blind analysis protocols.
    
    Implements:
    - Data masking (hide specific multipole ranges)
    - Analysis pre-registration
    - Unblinding log
    - Validation/test splits
    
    Attributes:
        output (OutputManager): For logging
        is_blind (bool): Whether analysis is currently blind
        preregistration (dict): Pre-registered analysis plan
        
    Example:
        >>> blind = BlindAnalysisManager()
        >>> blind.preregister_analysis(search_range=(500, 3000), threshold=0.5)
        >>> ell_masked, C_ell_masked = blind.mask_data(ell, C_ell, mode='validation')
    """
    
    def __init__(self, output: OutputManager = None, output_dir: str = "."):
        """
        Initialize BlindAnalysisManager.
        
        Parameters:
            output (OutputManager, optional): For logging
            output_dir (str): Directory for saving logs (default: current directory)
        """
        self.output = output if output is not None else OutputManager()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.is_blind = True
        self.preregistration = {}
        self.unblinding_log = []
        
        # Load existing preregistration if exists
        self.prereg_file = self.output_dir / "preregistration.json"
        self.unblind_file = self.output_dir / "unblinding_log.json"
        
        if self.prereg_file.exists():
            with open(self.prereg_file, 'r') as f:
                self.preregistration = json.load(f)
            self.output.log_message(f"Loaded preregistration from {self.prereg_file}")
        
        if self.unblind_file.exists():
            with open(self.unblind_file, 'r') as f:
                self.unblinding_log = json.load(f)
            if len(self.unblinding_log) > 0:
                self.is_blind = False
                self.output.log_message("Analysis has been unblinded")
    
    def preregister_analysis(self, search_range: Tuple[float, float],
                            detection_threshold: float,
                            max_transitions: int,
                            statistical_methods: list,
                            description: str = "") -> str:
        """
        Pre-register analysis plan before looking at data.
        
        Parameters:
            search_range (tuple): (ell_min, ell_max) to search
            detection_threshold (float): Prominence factor for detection
            max_transitions (int): Maximum number of transitions to test
            statistical_methods (list): Methods to use (e.g., ['bootstrap', 'cross-validation'])
            description (str): Analysis description
            
        Returns:
            str: Hash of preregistration for verification
        """
        self.preregistration = {
            'timestamp': datetime.now().isoformat(),
            'search_range': search_range,
            'detection_threshold': detection_threshold,
            'max_transitions': max_transitions,
            'statistical_methods': statistical_methods,
            'description': description,
            'status': 'preregistered'
        }
        
        # Compute hash for tamper-proof verification
        prereg_str = json.dumps(self.preregistration, sort_keys=True)
        prereg_hash = hashlib.sha256(prereg_str.encode()).hexdigest()
        self.preregistration['hash'] = prereg_hash
        
        # Save to file
        with open(self.prereg_file, 'w') as f:
            json.dump(self.preregistration, f, indent=2)
        
        self.output.log_section_header("ANALYSIS PRE-REGISTRATION")
        self.output.log_message(f"Timestamp: {self.preregistration['timestamp']}")
        self.output.log_message(f"Search range: ℓ = {search_range[0]} to {search_range[1]}")
        self.output.log_message(f"Detection threshold: {detection_threshold}")
        self.output.log_message(f"Max transitions: {max_transitions}")
        self.output.log_message(f"Statistical methods: {', '.join(statistical_methods)}")
        self.output.log_message(f"Hash: {prereg_hash[:16]}...")
        self.output.log_message(f"Saved to: {self.prereg_file}")
        self.output.log_message("")
        
        return prereg_hash
    
    def mask_data(self, ell: np.ndarray, C_ell: np.ndarray,
                  mode: str = 'validation', mask_fraction: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mask data for blind analysis.
        
        Modes:
        - 'validation': Reserve random subset for final validation
        - 'region': Hide specific multipole ranges
        - 'shuffle': Randomize transition locations
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Power spectrum
            mode (str): Masking mode (default: 'validation')
            mask_fraction (float): Fraction to mask (default: 0.2)
            
        Returns:
            tuple: (ell_masked, C_ell_masked) with masked data removed
        """
        if not self.is_blind:
            self.output.log_message("WARNING: Analysis already unblinded, returning unmasked data")
            return ell, C_ell
        
        n = len(ell)
        
        if mode == 'validation':
            # Reserve random subset for validation
            n_mask = int(n * mask_fraction)
            mask_indices = np.random.choice(n, size=n_mask, replace=False)
            mask = np.ones(n, dtype=bool)
            mask[mask_indices] = False
            
            self.output.log_message(f"Masked {n_mask}/{n} points for validation ({mask_fraction*100:.0f}%)")
            
        elif mode == 'region':
            # Mask specific multipole regions (e.g., where transitions expected)
            # For demonstration, mask middle 20% of range
            start_idx = int(n * 0.4)
            end_idx = int(n * 0.6)
            mask = np.ones(n, dtype=bool)
            mask[start_idx:end_idx] = False
            
            self.output.log_message(f"Masked region: ℓ = {ell[start_idx]:.0f} to {ell[end_idx]:.0f}")
            
        elif mode == 'shuffle':
            # Return data with shuffled multipoles (breaks correlations)
            ell_masked = ell.copy()
            C_ell_masked = np.random.permutation(C_ell)
            
            self.output.log_message("Data shuffled (null test)")
            return ell_masked, C_ell_masked
            
        else:
            raise ValueError(f"Unknown masking mode: {mode}")
        
        return ell[mask], C_ell[mask]
    
    def unblind(self, reason: str, results_summary: Dict[str, Any]) -> bool:
        """
        Unblind analysis and record justification.
        
        Parameters:
            reason (str): Justification for unblinding
            results_summary (dict): Summary of results from blind analysis
            
        Returns:
            bool: True if unblinding successful
        """
        if not self.is_blind:
            self.output.log_message("WARNING: Analysis already unblinded")
            return False
        
        unblind_entry = {
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'results_summary': results_summary,
            'preregistration_hash': self.preregistration.get('hash', 'none')
        }
        
        self.unblinding_log.append(unblind_entry)
        
        # Save unblinding log
        with open(self.unblind_file, 'w') as f:
            json.dump(self.unblinding_log, f, indent=2)
        
        self.is_blind = False
        
        self.output.log_section_header("ANALYSIS UNBLINDED")
        self.output.log_message(f"Timestamp: {unblind_entry['timestamp']}")
        self.output.log_message(f"Reason: {reason}")
        self.output.log_message(f"Log saved to: {self.unblind_file}")
        self.output.log_message("")
        
        return True
    
    def check_adherence(self, actual_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if analysis adhered to preregistration.
        
        Parameters:
            actual_params (dict): Parameters actually used in analysis
            
        Returns:
            dict: Adherence report
        """
        if not self.preregistration:
            return {
                'preregistered': False,
                'message': 'No preregistration found'
            }
        
        adherence = {
            'preregistered': True,
            'deviations': [],
            'compliant': True
        }
        
        # Check each preregistered parameter
        for key in ['search_range', 'detection_threshold', 'max_transitions']:
            if key in self.preregistration and key in actual_params:
                if self.preregistration[key] != actual_params[key]:
                    adherence['deviations'].append({
                        'parameter': key,
                        'preregistered': self.preregistration[key],
                        'actual': actual_params[key]
                    })
                    adherence['compliant'] = False
        
        return adherence
    
    def generate_report(self) -> str:
        """
        Generate comprehensive blind analysis report.
        
        Returns:
            str: Markdown report
        """
        report = []
        report.append("# Blind Analysis Report")
        report.append("")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        # Preregistration
        report.append("## Preregistration")
        if self.preregistration:
            report.append(f"- Timestamp: {self.preregistration.get('timestamp', 'N/A')}")
            report.append(f"- Search range: {self.preregistration.get('search_range', 'N/A')}")
            report.append(f"- Detection threshold: {self.preregistration.get('detection_threshold', 'N/A')}")
            report.append(f"- Max transitions: {self.preregistration.get('max_transitions', 'N/A')}")
            report.append(f"- Hash: {self.preregistration.get('hash', 'N/A')}")
        else:
            report.append("No preregistration found.")
        report.append("")
        
        # Unblinding log
        report.append("## Unblinding History")
        if self.unblinding_log:
            for i, entry in enumerate(self.unblinding_log, 1):
                report.append(f"### Unblinding {i}")
                report.append(f"- Timestamp: {entry.get('timestamp', 'N/A')}")
                report.append(f"- Reason: {entry.get('reason', 'N/A')}")
                report.append("")
        else:
            report.append("Analysis remains blind or no unblinding log available.")
        report.append("")
        
        # Status
        report.append("## Current Status")
        report.append(f"- Blind: {'Yes' if self.is_blind else 'No'}")
        report.append("")
        
        return "\n".join(report)
    
    def save_report(self, filename: Optional[str] = None):
        """
        Save blind analysis report to file.
        
        Parameters:
            filename (str, optional): Output filename (default: blind_analysis_report.md)
        """
        if filename is None:
            filename = self.output_dir / "blind_analysis_report.md"
        else:
            filename = Path(filename)
        
        report = self.generate_report()
        
        with open(filename, 'w') as f:
            f.write(report)
        
        self.output.log_message(f"Blind analysis report saved to: {filename}")
