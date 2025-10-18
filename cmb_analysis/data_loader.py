"""
Data Loader Module
==================

Downloads and loads CMB E-mode polarization data from public archives.

Classes:
    DataLoader: Fetches ACT DR6 and Planck 2018 CMB data

Data Sources:
    - ACT DR6: LAMBDA Archive (NASA/GSFC)
    - Planck 2018: Planck Legacy Archive (ESA)
"""

import os
import tempfile
import tarfile
import requests
import numpy as np
from typing import Tuple, Optional

from .utils import OutputManager


class DataLoader:
    """
    Load CMB E-mode polarization data from public archives.
    
    Handles downloading, extracting, and parsing of:
    - ACT DR6 foreground-subtracted E-mode spectrum
    - Planck 2018 E-mode power spectrum
    
    Data is cached locally to avoid repeated downloads.
    
    Attributes:
        output (OutputManager): For logging
        cache_dir (str): Directory for cached data files
        
    Example:
        >>> loader = DataLoader()
        >>> ell, C_ell, C_ell_err = loader.load_act_dr6()
        >>> print(f"Loaded {len(ell)} multipoles")
    """
    
    def __init__(self, output: OutputManager = None, cache_dir: str = "downloaded_data"):
        """
        Initialize DataLoader.
        
        Parameters:
            output (OutputManager, optional): For logging. Creates new if None.
            cache_dir (str): Directory for caching downloaded data
        """
        self.output = output if output is not None else OutputManager()
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_act_dr6(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load ACT DR6 E-mode power spectrum from LAMBDA archive.
        
        Downloads and extracts ACT DR6 foreground-subtracted EE spectrum.
        Data is converted from D_ell to C_ell format. Caches data locally
        to avoid repeated downloads.
        
        Paper reference: Methods section, line ~268
        
        Returns:
            tuple: (ell, C_ell, C_ell_err)
                - ell: Multipole values (array)
                - C_ell: Power spectrum C_ℓ^EE (array)
                - C_ell_err: Uncertainties on C_ℓ (array)
                
        Raises:
            requests.HTTPError: If download fails
            FileNotFoundError: If EE spectrum file not found in archive
        """
        url = "https://lambda.gsfc.nasa.gov/data/act/pspipe/spectra_and_cov/" \
              "act_dr6.02_spectra_and_cov_binning_20.tar.gz"
        
        # Cached file paths
        cached_data = os.path.join(self.cache_dir, "act_dr6_fg_subtracted_EE.dat")
        
        self.output.log_section_header("DATA LOADING")
        self.output.log_message(f"Source: ACT DR6 (LAMBDA Archive)")
        
        # Check if cached data exists
        if os.path.exists(cached_data):
            self.output.log_message(f"Using cached data: {cached_data}")
            
            # Load from cache
            data = np.loadtxt(cached_data)
            ell = data[:, 0]
            D_ell = data[:, 1]
            D_ell_err = data[:, 2]
            
            # Convert D_ell to C_ell
            C_ell = D_ell * (2 * np.pi) / (ell * (ell + 1))
            C_ell_err = D_ell_err * (2 * np.pi) / (ell * (ell + 1))
            
            self.output.log_message(f"Data loaded from cache:")
            self.output.log_message(f"  Points: {len(ell)}")
            self.output.log_message(f"  Range: ℓ = {int(ell[0])} to {int(ell[-1])}")
            self.output.log_message(f"  C_ell range: {C_ell.min():.3e} to {C_ell.max():.3e}")
            self.output.log_message("")
            
            return ell, C_ell, C_ell_err
        
        # Download if not cached
        self.output.log_message(f"URL: {url}")
        self.output.log_message("Downloading and extracting...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with tempfile.TemporaryDirectory() as tmpdir:
                tar_path = os.path.join(tmpdir, "act_dr6_data.tar.gz")
                
                # Download
                with open(tar_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Extract
                with tarfile.open(tar_path, 'r:gz') as tar:
                    tar.extractall(path=tmpdir)
                
                # Find foreground-subtracted EE spectrum
                fg_file = None
                for root, dirs, files in os.walk(tmpdir):
                    for file in files:
                        if 'fg_subtracted' in file.lower() and 'EE' in file:
                            fg_file = os.path.join(root, file)
                            break
                    if fg_file:
                        break
                
                if not fg_file:
                    raise FileNotFoundError("Could not find fg_subtracted_EE.dat")
                
                self.output.log_message(f"Using: {os.path.basename(fg_file)}")
                
                # Load data (format: bin_center, D_ell_fg_sub, sigma)
                data = np.loadtxt(fg_file)
                
                # Cache the raw data for future use
                np.savetxt(cached_data, data, fmt='%.6e', 
                          header='ell D_ell_fg_sub sigma')
                self.output.log_message(f"Data cached to: {cached_data}")
                
                ell = data[:, 0]
                D_ell = data[:, 1]
                D_ell_err = data[:, 2]
                
                # Convert D_ell to C_ell: D_ell = ell(ell+1)C_ell/(2pi)
                C_ell = D_ell * (2 * np.pi) / (ell * (ell + 1))
                C_ell_err = D_ell_err * (2 * np.pi) / (ell * (ell + 1))
                
                self.output.log_message(f"Data loaded successfully:")
                self.output.log_message(f"  Points: {len(ell)}")
                self.output.log_message(f"  Range: ℓ = {int(ell[0])} to {int(ell[-1])}")
                self.output.log_message(f"  C_ell range: {C_ell.min():.3e} to {C_ell.max():.3e}")
                self.output.log_message("")
                
                return ell, C_ell, C_ell_err
        
        except Exception as e:
            self.output.log_message(f"Error loading data: {e}")
            raise
    
    def load_planck_2018(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Load Planck 2018 E-mode power spectrum for cross-validation.
        
        Downloads Planck 2018 EE power spectrum from Planck Legacy Archive.
        Uses COM_PowerSpect_CMB-EE-full_R3.01 from the final Planck 2018 release.
        
        Paper reference: Methods section, cross-dataset validation
        
        Returns:
            tuple or None: (ell, C_ell, C_ell_err) if available, else None
                - ell: Multipole values (array)
                - C_ell: Power spectrum C_ℓ^EE (array)  
                - C_ell_err: Uncertainties on C_ℓ (array)
        """
        # Planck Legacy Archive URL for EE spectrum
        url = "https://pla.esac.esa.int/pla/aio/product-action?" \
              "COSMOLOGY.FILE_ID=COM_PowerSpect_CMB-EE-full_R3.01.txt"
        
        # Cached file path
        cached_data = os.path.join(self.cache_dir, "planck_2018_EE_spectrum.dat")
        
        self.output.log_message("\nLoading Planck 2018 data for cross-validation...")
        
        # Check if cached
        if os.path.exists(cached_data):
            self.output.log_message(f"Using cached data: {cached_data}")
            
            try:
                data = np.loadtxt(cached_data)
                ell = data[:, 0]
                D_ell = data[:, 1]
                D_ell_err = data[:, 2]
                
                # Convert D_ell to C_ell
                C_ell = D_ell * (2 * np.pi) / (ell * (ell + 1))
                C_ell_err = D_ell_err * (2 * np.pi) / (ell * (ell + 1))
                
                self.output.log_message(f"Planck 2018 data loaded from cache:")
                self.output.log_message(f"  Points: {len(ell)}")
                self.output.log_message(f"  Range: ℓ = {int(ell[0])} to {int(ell[-1])}")
                self.output.log_message("")
                
                return ell, C_ell, C_ell_err
                
            except Exception as e:
                self.output.log_message(f"  Error loading cached Planck data: {e}")
                # Fall through to download
        
        # Try to download
        try:
            self.output.log_message(f"Downloading from PLA: {url}")
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse the Planck data file
            # Format: columns are ell, D_ell, error (and possibly others)
            lines = response.text.strip().split('\n')
            
            # Skip header lines (start with # or empty)
            data_lines = [line for line in lines if line and not line.startswith('#')]
            
            # Parse data
            ell_list, D_ell_list, D_ell_err_list = [], [], []
            for line in data_lines:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        ell_val = float(parts[0])
                        D_ell_val = float(parts[1])
                        D_ell_err_val = float(parts[2])
                        
                        # Only use multipoles in reasonable range
                        if 2 <= ell_val <= 3000:
                            ell_list.append(ell_val)
                            D_ell_list.append(D_ell_val)
                            D_ell_err_list.append(D_ell_err_val)
                    except ValueError:
                        continue
            
            if len(ell_list) == 0:
                raise ValueError("No valid data points found in Planck file")
            
            ell = np.array(ell_list)
            D_ell = np.array(D_ell_list)
            D_ell_err = np.array(D_ell_err_list)
            
            # Convert D_ell to C_ell
            C_ell = D_ell * (2 * np.pi) / (ell * (ell + 1))
            C_ell_err = D_ell_err * (2 * np.pi) / (ell * (ell + 1))
            
            # Cache the data
            cache_data = np.column_stack([ell, D_ell, D_ell_err])
            np.savetxt(cached_data, cache_data, fmt='%.6e',
                      header='ell D_ell_EE sigma')
            self.output.log_message(f"Planck data cached to: {cached_data}")
            
            self.output.log_message(f"Planck 2018 data loaded successfully:")
            self.output.log_message(f"  Points: {len(ell)}")
            self.output.log_message(f"  Range: ℓ = {int(ell[0])} to {int(ell[-1])}")
            self.output.log_message("")
            
            return ell, C_ell, C_ell_err
            
        except requests.RequestException as e:
            self.output.log_message(f"  Network error downloading Planck data: {e}")
            self.output.log_message("  Cross-validation with Planck will be skipped")
            return None
            
        except Exception as e:
            self.output.log_message(f"  Error parsing Planck data: {e}")
            self.output.log_message("  Cross-validation with Planck will be skipped")
            return None
    
    def validate_data(self, ell: np.ndarray, C_ell: np.ndarray, 
                     C_ell_err: np.ndarray) -> bool:
        """
        Validate loaded data for consistency and completeness.
        
        Checks:
        - Arrays have matching lengths
        - No NaN or Inf values
        - Multipole values are monotonically increasing
        - Error bars are positive
        - C_ell values are finite
        
        Parameters:
            ell (ndarray): Multipole values
            C_ell (ndarray): Power spectrum
            C_ell_err (ndarray): Uncertainties
            
        Returns:
            bool: True if data passes all checks
            
        Raises:
            ValueError: If validation fails with specific error message
        """
        # Check array lengths match
        if not (len(ell) == len(C_ell) == len(C_ell_err)):
            raise ValueError("Array lengths do not match")
        
        # Check for NaN or Inf
        if np.any(~np.isfinite(ell)) or np.any(~np.isfinite(C_ell)) or \
           np.any(~np.isfinite(C_ell_err)):
            raise ValueError("Data contains NaN or Inf values")
        
        # Check multipoles are monotonically increasing
        if not np.all(np.diff(ell) > 0):
            raise ValueError("Multipole values not monotonically increasing")
        
        # Check error bars are positive
        if np.any(C_ell_err <= 0):
            raise ValueError("Error bars must be positive")
        
        # Check C_ell values are finite
        if not np.all(np.isfinite(C_ell)):
            raise ValueError("C_ell contains non-finite values")
        
        self.output.log_message("✓ Data validation passed")
        return True

