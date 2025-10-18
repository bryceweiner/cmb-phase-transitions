"""
Physical Constants and Parameters
==================================

All physical constants, cosmological parameters, and configuration values
used throughout the analysis. Values are referenced to specific lines in
phase_transitions_discovery.tex where they appear.

References:
    - Paper Methods section, line ~193 for physical constants
    - Paper line ~75 for recombination parameters
    - Paper line ~117 for QTEP ratio
"""

import numpy as np


# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================
# Reference: Paper Methods section, line ~193

C = 2.998e8              # m/s - speed of light
HBAR = 1.055e-34         # J·s - reduced Planck constant
G = 6.674e-11            # m³ kg⁻¹ s⁻² - gravitational constant
H0 = 2.18e-18            # s⁻¹ - Hubble constant (H0 = 67 km/s/Mpc)
OMEGA_M = 0.315          # Matter density parameter
OMEGA_LAMBDA = 0.685     # Dark energy density parameter


# ============================================================================
# RECOMBINATION PARAMETERS
# ============================================================================
# Reference: Paper line ~75

Z_RECOMB = 1100          # Recombination redshift
H_RECOMB = 2.3e-18       # s⁻¹ - Hubble parameter at z=1100


# ============================================================================
# COSMOLOGICAL EPOCHS
# ============================================================================

Z_DRAG = 1059            # Drag epoch redshift (baryon-photon decoupling)
Z_EQ = 3402              # Matter-radiation equality redshift


# ============================================================================
# OBSERVATIONAL VALUES
# ============================================================================

# Planck 2018
SIGMA_8_PLANCK = 0.811   # σ8 amplitude
S8_PLANCK = 0.834        # S8 = σ8(Ωm/0.3)^0.5 (±0.016)
OMEGA_M_PLANCK = 0.315   # Matter density (±0.007)

# DES Year 3
S8_DES_Y3 = 0.776        # S8 value (±0.017)
OMEGA_M_DES_Y3 = 0.298   # Matter density (+0.007/-0.007)
RS_DES_Y3 = 143.6        # BAO scale in Mpc (±1.7)

# BOSS DR12
RS_BOSS = 147.47         # BAO scale in Mpc (±0.59)


# ============================================================================
# BAO DATA
# ============================================================================

# DES Y3 BAO data points (D_M/r_d measurements)
DES_Y3_BAO_DATA = [
    {'z': 0.65, 'value': 19.05, 'error': 0.55},
    {'z': 0.74, 'value': 18.92, 'error': 0.51},
    {'z': 0.84, 'value': 18.80, 'error': 0.48},
    {'z': 0.93, 'value': 18.68, 'error': 0.45},
    {'z': 1.02, 'value': 18.57, 'error': 0.42}
]


# ============================================================================
# MATTER DENSITY DATA
# ============================================================================

MATTER_DENSITY_DATA = {
    'surveys': ['DES Y1', 'DES Y3'],
    'omega_m_measured': [0.267, 0.298],
    'omega_m_hu': [0.268, 0.298],
    'measured_errors': [[0.017, 0.030], [0.007, 0.007]],
    'hu_errors': [[0.018, 0.018], [0.007, 0.007]]
}


# ============================================================================
# QUANTUM-THERMODYNAMIC ENTROPY PARTITION (QTEP)
# ============================================================================
# Reference: Paper line ~117

# Coherent entropy: S_coh = ln(2) ≈ 0.693 nats
S_COH = np.log(2)

# Decoherent entropy: S_decoh = ln(2) - 1 ≈ -0.307 nats
S_DECOH = np.log(2) - 1

# QTEP ratio: S_coh / |S_decoh| = ln(2) / (1 - ln(2)) ≈ 2.257
QTEP_RATIO = np.log(2) / (1 - np.log(2))


# ============================================================================
# PLANCK UNITS
# ============================================================================
# Reference: Supplementary Note 2, Step 2

T_PLANCK = np.sqrt(HBAR * G / C**5)   # Planck time ≈ 5.39×10⁻⁴⁴ s
M_PLANCK = np.sqrt(HBAR * C / G)      # Planck mass ≈ 2.18×10⁻⁸ kg
RHO_PLANCK = C**5 / (HBAR * G**2)     # Planck density ≈ 5.16×10⁹⁶ kg/m³
L_PLANCK = np.sqrt(HBAR * G / C**3)   # Planck length ≈ 1.62×10⁻³⁵ m


# ============================================================================
# COSMOLOGICAL CONSTANT
# ============================================================================
# Reference: Paper Introduction, line ~15

LAMBDA_OBS = 1.1e-52     # m⁻² - observed cosmological constant


# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

OUTPUT_LOG = 'cmb_analysis_unified.log'
OUTPUT_JSON = 'cmb_analysis_unified.json'
OUTPUT_FIGURE = 'cmb_phase_transitions_analysis.pdf'


# ============================================================================
# PAPER LINE REFERENCES
# ============================================================================
# Map formulas to specific lines in phase_transitions_discovery.tex

PAPER_REFERENCES = {
    'gamma_theoretical': 'line 107',           # γ = H/ln(πc²/ℏGH²)
    'qtep_ratio': 'line 167',                  # S_coh/|S_decoh| = 2.257
    'quantization_condition': 'line 109',      # γℓ/H = nπ/2
    'expansion_factor': 'line 117',            # f = γ_theory/γ_obs
    'vacuum_energy_scaling': 'line 115',       # ρ_Λ,eff ∝ [γ_theory/γ_obs]²
    'lambda_formula': 'line 134',              # Λ = 8πG ρ_P (γ×t_P)² × QTEP
    'physical_scale': 'line 212',              # Scale conversion at z_recomb
    'harmonic_ratio': 'line 99',               # ℓ_{n+1}/ℓ_n with corrections
}


# ============================================================================
# VERSION INFORMATION
# ============================================================================

__version__ = "1.0.0"
__author__ = "Bryce Weiner"
__paper__ = "Pre-Recombination Spacetime Expansion Events Resolve the " \
            "Cosmological Constant Problem Through Information Physics"

