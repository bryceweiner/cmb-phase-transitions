"""
Utilities Module
================

Output management, logging, and utility functions for the analysis pipeline.

Classes:
    OutputManager: Handles all file I/O, logging, and result serialization
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

from .constants import (
    OUTPUT_LOG, OUTPUT_JSON, OUTPUT_FIGURE,
    QTEP_RATIO, C, HBAR, G, H0,
    OMEGA_M, OMEGA_LAMBDA, Z_RECOMB, H_RECOMB,
    PAPER_REFERENCES, __version__, __paper__
)


class OutputManager:
    """
    Manages all output operations including logging, JSON serialization, and file validation.
    
    This class provides a centralized interface for:
    - Logging messages to both console and file
    - Saving analysis results to JSON
    - Validating output file generation
    - Formatting section headers
    
    Attributes:
        output_dir (str): Directory for output files
        log_file (str): Path to log file
        json_file (str): Path to JSON output file
        log_handle: File handle for log (kept open during analysis)
        
    Example:
        >>> output = OutputManager(output_dir="./results")
        >>> output.log_section_header("DATA LOADING")
        >>> output.log_message("Loaded 1000 multipoles")
        >>> output.save_json_output(results)
        >>> output.close()
    """
    
    def __init__(self, output_dir: str = "."):
        """
        Initialize OutputManager.
        
        Parameters:
            output_dir (str): Directory for output files. Created if doesn't exist.
        """
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, OUTPUT_LOG)
        self.json_file = os.path.join(output_dir, OUTPUT_JSON)
        self.log_handle = None
        self.figures_generated = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize log file
        self._initialize_log()
    
    def _initialize_log(self):
        """Initialize log file with header."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("CMB E-MODE PHASE TRANSITION ANALYSIS\n")
            f.write(f"Paper: {__paper__}\n")
            f.write(f"Version: {__version__}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("=" * 70 + "\n\n")
        
        print("=" * 70)
        print("CMB E-MODE PHASE TRANSITION ANALYSIS")
        print(f"Version: {__version__}")
        print(f"Timestamp: {timestamp}")
        print("=" * 70)
        print()
    
    def log_message(self, message: str):
        """
        Log message to both console and file.
        
        Parameters:
            message (str): Message to log
        """
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def log_section_header(self, title: str):
        """
        Log a formatted section header.
        
        Parameters:
            title (str): Section title (e.g., "DATA LOADING")
        """
        header = f"\n{'='*70}\n{title}\n{'='*70}\n"
        self.log_message(header)
    
    def log_subsection(self, title: str):
        """
        Log a formatted subsection header.
        
        Parameters:
            title (str): Subsection title
        """
        header = f"\n{'-'*70}\n{title}\n{'-'*70}"
        self.log_message(header)
    
    def save_json_output(self, 
                        results: Dict[str, Any],
                        significance: Optional[Dict[str, Any]] = None,
                        look_elsewhere: Optional[Dict[str, Any]] = None,
                        advanced_stats: Optional[Dict[str, Any]] = None):
        """
        Save comprehensive results to JSON file.
        
        Parameters:
            results (dict): Main analysis results
            significance (dict, optional): Statistical significance results
            look_elsewhere (dict, optional): Look-elsewhere effect calculations
            advanced_stats (dict, optional): Advanced statistical analyses
        """
        output = {
            'metadata': {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'paper_title': __paper__,
                'analysis_version': __version__,
                'data_source': 'ACT DR6 + Planck 2018'
            },
            'paper_references': {
                'line_numbers': PAPER_REFERENCES,
                'description': 'Line numbers in phase_transitions_discovery.tex'
            },
            'physical_constants': {
                'c_m_per_s': float(C),
                'hbar_J_s': float(HBAR),
                'G_m3_kg_s2': float(G),
                'H0_per_s': float(H0)
            },
            'cosmology': {
                'Omega_m': float(OMEGA_M),
                'Omega_Lambda': float(OMEGA_LAMBDA),
                'z_recombination': float(Z_RECOMB),
                'H_recombination_per_s': float(H_RECOMB)
            },
            'qtep': {
                'ratio_theory': float(QTEP_RATIO),
                'formula': 'S_coh/|S_decoh| = ln(2)/(1-ln(2))',
                'value': float(QTEP_RATIO),
                'paper_line': PAPER_REFERENCES.get('qtep_ratio', 'line 167')
            },
            'analysis_results': results,
            'figures_generated': self.figures_generated
        }
        
        # Add optional components
        if significance is not None:
            output['statistical_significance'] = significance
        if look_elsewhere is not None:
            output['look_elsewhere_effect'] = look_elsewhere
        if advanced_stats is not None:
            output['advanced_statistical_analysis'] = advanced_stats
        
        with open(self.json_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        self.log_message(f"\nResults saved to {self.json_file}")
    
    def register_figure(self, figure_path: str):
        """
        Register a generated figure.
        
        Parameters:
            figure_path (str): Path to generated figure file
        """
        self.figures_generated.append(figure_path)
    
    def validate_outputs(self) -> Dict[str, bool]:
        """
        Validate that all expected output files exist.
        
        Returns:
            dict: Dictionary mapping file types to existence status
        """
        validation = {
            'log_file': os.path.exists(self.log_file),
            'json_file': os.path.exists(self.json_file),
            'figures': all(os.path.exists(fig) for fig in self.figures_generated)
        }
        return validation
    
    def print_summary(self, results: Dict[str, Any]):
        """
        Print a summary of key results.
        
        Parameters:
            results (dict): Analysis results dictionary
        """
        self.log_section_header("ANALYSIS SUMMARY")
        
        # Transitions
        if 'transitions' in results:
            self.log_message("Detected Transitions:")
            for i, ell in enumerate(results['transitions'], 1):
                self.log_message(f"  ℓ_{i} = {ell:.0f}")
        
        # QTEP ratio
        self.log_message(f"\nQTEP Ratio: {QTEP_RATIO:.4f}")
        
        # Cosmological constant
        if 'lambda_results' in results:
            lambda_res = results['lambda_results']
            if 'Lambda_final' in lambda_res:
                self.log_message(f"\nΛ₀ (final): {lambda_res['Lambda_final']:.3e} m⁻²")
            if 'Lambda_obs' in lambda_res:
                self.log_message(f"Λ₀ (observed): {lambda_res['Lambda_obs']:.3e} m⁻²")
        
        # Output files
        self.log_message(f"\nOutput Files:")
        self.log_message(f"  Log: {self.log_file}")
        self.log_message(f"  JSON: {self.json_file}")
        self.log_message(f"  Figures: {len(self.figures_generated)} generated")
        
        validation = self.validate_outputs()
        if all(validation.values()):
            self.log_message("\n✓ All outputs generated successfully")
        else:
            self.log_message("\n⚠ Some outputs missing:")
            for key, status in validation.items():
                if not status:
                    self.log_message(f"  ✗ {key}")
        
        self.log_message("\n" + "=" * 70)
        self.log_message("ANALYSIS COMPLETE")
        self.log_message("=" * 70)
    
    def close(self):
        """Close any open file handles."""
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None


def format_scientific(value: float, precision: int = 3) -> str:
    """
    Format a float in scientific notation.
    
    Parameters:
        value (float): Value to format
        precision (int): Number of decimal places
        
    Returns:
        str: Formatted string (e.g., "1.234e-05")
    """
    return f"{value:.{precision}e}"


def format_error(value: float, error: float, precision: int = 2) -> str:
    """
    Format value ± error.
    
    Parameters:
        value (float): Central value
        error (float): Uncertainty
        precision (int): Number of significant figures for error
        
    Returns:
        str: Formatted string (e.g., "1.23 ± 0.45")
    """
    return f"{value:.{precision}f} ± {error:.{precision}f}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Parameters:
        numerator (float): Numerator
        denominator (float): Denominator
        default (float): Value to return if denominator is zero
        
    Returns:
        float: Result of division or default
    """
    return numerator / denominator if denominator != 0 else default

