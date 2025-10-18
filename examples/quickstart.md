# Quick Start Guide

This guide will help you run the CMB phase transition analysis in 5 simple steps.

## Prerequisites

Ensure you have Python 3.8 or higher installed:

```bash
python --version  # Should show 3.8 or higher
```

## Step 1: Install Dependencies

Navigate to the project directory and install required packages:

```bash
cd cmb-phase-transitions-code
pip install -r requirements.txt
```

This installs: NumPy, SciPy, Matplotlib, and Requests.

## Step 2: Run the Analysis

Execute the main analysis script:

```bash
python main.py
```

The script will:
- Download ACT DR6 and Planck 2018 CMB data (~5 MB total)
- Detect phase transitions at ℓ = 1076, 1706, 2336
- Perform statistical validation (>>5σ significance)
- Calculate cosmological constant: Λ₀ = 1.10×10⁻⁵² m⁻²
- Generate 8 publication-quality figures
- Save results to JSON and log files

**Runtime:** ~3-5 minutes on standard hardware

## Step 3: Check the Outputs

After completion, verify these files were created:

```bash
ls -lh *.json *.log *.pdf *.png
```

You should see:
- `cmb_analysis_unified.json` - Complete results
- `cmb_analysis_unified.log` - Detailed calculation log
- 8 figure files (PDF and PNG)

## Step 4: Validate Results

Compare key results in `cmb_analysis_unified.json` to expected values:

```python
import json

with open('cmb_analysis_unified.json') as f:
    results = json.load(f)

# Check transitions
print("Detected transitions:", results['transitions'])
# Expected: [1076±63, 1706±94, 2336±115]

# Check QTEP ratio
print("QTEP ratio:", results['qtep_ratio'])
# Expected: 2.2589

# Check cosmological constant
print("Λ₀ (complete):", results['lambda_complete'])
# Expected: ~1.10e-52 m⁻²
```

## Step 5: Explore the Code

Use individual components for custom analysis:

```python
from cmb_analysis import (
    DataLoader, PhaseTransitionDetector,
    TheoreticalCalculations
)

# Load CMB data
data = DataLoader()
ell, C_ell, C_ell_err = data.load_act_dr6()
print(f"Loaded {len(ell)} multipoles from ℓ={ell[0]} to ℓ={ell[-1]}")

# Detect phase transitions
detector = PhaseTransitionDetector()
dC_dell = detector.compute_derivative(ell, C_ell)
transitions, errors = detector.detect_transitions(ell, dC_dell)
print(f"Found transitions at ℓ = {transitions}")

# Calculate information processing rates
theory = TheoreticalCalculations()
H_recomb = 2.3e-18  # s⁻¹
gamma_theory = theory.gamma_theoretical(H_recomb)
print(f"Theoretical γ = {gamma_theory:.3e} s⁻¹")

# Extract observed rates from transitions
gamma_obs = [theory.gamma_from_quantization(ell, n+1, H_recomb) 
             for n, ell in enumerate(transitions)]
print(f"Observed γ = {gamma_obs}")

# Calculate expansion factors
expansion_factors = [theory.expansion_factor(g_obs, gamma_theory)
                    for g_obs in gamma_obs]
print(f"Expansion factors: {expansion_factors}")
# Expected: [3.07, 2.43, 2.22]
```

## Advanced Usage

### Custom Output Directory

```bash
python main.py --output-dir ./my_results
```

### Quiet Mode (suppress progress messages)

```bash
python main.py --quiet
```

### Using as Python Package

```python
# Import and run complete analysis
from cmb_analysis import run_analysis

results = run_analysis(output_dir="./results")
```

## Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'cmb_analysis'`  
**Solution:** Ensure you're in the correct directory and have run `pip install -r requirements.txt`

**Issue:** Download failures  
**Solution:** Check your internet connection. Data files are hosted on public servers (ACT, Planck archives)

**Issue:** Figures look different from paper  
**Solution:** Minor visual differences are expected due to matplotlib versions. Numerical results should match exactly.

## Next Steps

- Review `README.md` for detailed architecture documentation
- Explore individual modules in `cmb_analysis/` directory
- Modify detection parameters in `phase_detector.py` for sensitivity analysis
- Extend `cosmological_constant.py` for additional quantum corrections

## Getting Help

If you encounter issues:
1. Check `cmb_analysis_unified.log` for detailed error messages
2. Verify expected outputs in `examples/expected_output.txt`
3. Contact: bryce.weiner@informationphysicsinstitute.net

