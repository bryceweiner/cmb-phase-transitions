# CMB E-mode Phase Transition Analysis - Object-Oriented Implementation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Analysis code for the paper:

**"Pre-Recombination Spacetime Expansion Events Resolve the Cosmological Constant Problem Through Information Physics"**  
Bryce Weiner  
*Physical Review D* (submitted)

## Overview

This package implements a complete analysis pipeline for detecting quantum phase transitions in CMB E-mode polarization data and deriving the cosmological constant from information-theoretic first principles. The code discovers three discrete pre-recombination spacetime expansion events at multipoles ℓ = 1076±63, 1706±94, 2336±115 with overwhelming statistical significance (>>5σ).

**Key Results:**
- Transforms the cosmological constant problem from 120 orders of magnitude (10¹²² QFT over-prediction) to exact match with observations
- Predicts Λ₀ = 1.10×10⁻⁵² m⁻² from zero free parameters when including all expansion events from universe instantiation
- Resolves Hubble, S₈, and BAO tensions through single parameter γ
- Establishes universal quantum-thermodynamic entropy partition ratio (2.257) across 20 orders of magnitude

**Runtime:** <5 minutes on standard hardware

## Quick Start

```python
# Install dependencies
pip install -r requirements.txt

# Run complete analysis
python main.py

# Or use as a package
from cmb_analysis import run_analysis
results = run_analysis()
```

## Architecture

The codebase is organized into 9 logical classes with single responsibilities:

```
cmb_analysis/
├── constants.py              # Physical constants and parameters
├── theoretical.py            # TheoreticalCalculations - information theory
├── data_loader.py            # DataLoader - fetch CMB data
├── phase_detector.py         # PhaseTransitionDetector - find transitions
├── statistics.py             # StatisticalAnalysis - validation
├── cosmological_constant.py  # CosmologicalConstant - Λ derivation
├── temporal_cascade.py       # TemporalCascade - expansion history
├── tensions.py               # CosmologicalTensions - H₀, S₈, BAO
├── visualizer.py             # Visualizer - figure generation
└── utils.py                  # OutputManager - logging and I/O
```

**Data Flow:**
```
DataLoader → PhaseTransitionDetector → StatisticalAnalysis
                                   ↓
TheoreticalCalculations ← TemporalCascade
                ↓
CosmologicalConstant → Visualizer → OutputManager
                ↓
    CosmologicalTensions
```

## Installation

### Requirements
- Python 3.8 or higher
- NumPy ≥ 1.21.0
- SciPy ≥ 1.7.0
- Matplotlib ≥ 3.4.0
- Requests ≥ 2.26.0

### Install

```bash
git clone https://github.com/bryceweiner/cmb-phase-transitions
cd cmb-phase-transitions
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the complete analysis pipeline:

```bash
python main.py
```

This will:
1. Download ACT DR6 and Planck 2018 CMB data
2. Detect three phase transitions
3. Perform statistical validation (bootstrap, LEE corrections, cross-dataset)
4. Calculate cosmological constant from first principles
5. Generate all figures (main + supplementary)
6. Save results to JSON and log files

### Advanced Usage

Use individual components:

```python
from cmb_analysis import (
    DataLoader, PhaseTransitionDetector,
    TheoreticalCalculations, CosmologicalConstant
)

# Load data
data = DataLoader()
ell, C_ell, C_ell_err = data.load_act_dr6()

# Detect transitions
detector = PhaseTransitionDetector()
transitions, errors = detector.detect_and_analyze(ell, C_ell)

# Calculate gamma values
theory = TheoreticalCalculations()
gamma_values = theory.calculate_gamma_values(transitions)

# Derive cosmological constant
lambda_calc = CosmologicalConstant()
lambda_results = lambda_calc.calculate_detailed()
```

### Command-Line Options

```bash
python main.py --help                    # Show all options
python main.py --output-dir ./results   # Specify output directory
python main.py --quiet                   # Suppress progress messages
python main.py --version                 # Show version
```

## Output Files

The analysis generates:

**Data Files:**
- `cmb_analysis_unified.json` - Complete results in JSON format
- `cmb_analysis_unified.log` - Detailed execution log with all calculations

**Figures:**
- `cmb_phase_transitions_analysis.pdf` - Main analysis figure (3 panels)
- `supplementary_figure_s1_statistical_validation.pdf` - Statistical validation suite
- `supplementary_figure_s2_lambda_evolution.pdf` - Λ(z) evolution and resolution
- `supplementary_figure_s3_temporal_cascade.pdf` - Temporal cascade mechanism
- `cosmological_tension_bao.pdf` - BAO scale resolution
- `cosmological_tension_s8.pdf` - S₈ tension resolution
- `cosmological_tension_matter_density.pdf` - Matter density comparison
- `hubble_tension_resolution.png` - Hubble tension resolution

## Expected Results

To validate your installation, verify these key results:

| Result | Expected Value |
|--------|---------------|
| Transition 1 | ℓ = 1076±63 |
| Transition 2 | ℓ = 1706±94 |
| Transition 3 | ℓ = 2336±115 |
| Statistical Significance | >>5σ |
| QTEP Ratio | 2.2589 |
| Λ₀ (detected only) | 1.57×10⁻⁵³ m⁻² |
| Λ₀ (complete framework) | 1.10×10⁻⁵² m⁻² |
| Λ₀ (observed) | 1.10×10⁻⁵² m⁻² |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Weiner2025PRD,
  author = {Weiner, Bryce},
  title = {Pre-Recombination Spacetime Expansion Events Resolve the Cosmological Constant Problem Through Information Physics},
  journal = {Physical Review D},
  year = {2025},
  note = {Submitted}
}
```

**Previous work:**
- B. Weiner, "E-mode Polarization Phase Transitions Reveal a Fundamental Parameter of the Universe," *IPI Letters* **3**(1), 31-39 (2025). [doi:10.59973/ipil.150](https://doi.org/10.59973/ipil.150)
- B. Weiner, "Holographic Information Rate as a Resolution to Contemporary Cosmological Tensions," *IPI Letters* **3**(2), 1-12 (2025). [doi:10.59973/ipil.170](https://doi.org/10.59973/ipil.170)
- B. Weiner, "ATLAS Shrugged: Resolving Experimental Tensions in Particle Physics Through Holographic Theory," *IPI Letters* **3**(3), 1-15 (2025). [doi:10.59973/ipil.222](https://doi.org/10.59973/ipil.222)

## Documentation

See [`examples/quickstart.md`](examples/quickstart.md) for a step-by-step tutorial.

For detailed API documentation, see docstrings in individual modules.

## Design Rationale

This object-oriented refactoring provides:

1. **Readability**: Each class ~300-500 lines vs 4,297-line monolith
2. **Maintainability**: Changes isolated to relevant class
3. **Testability**: Each class can be tested independently
4. **Reusability**: Classes can be used in other projects
5. **Collaboration**: Different researchers can work on different components
6. **Peer Review**: Easier to review class-by-class

## License

MIT License - see [`LICENSE`](LICENSE) for details.

## Contact

**Bryce Weiner**  
Information Physics Institute  
Email: bryce.weiner@informationphysicsinstitute.net  
GitHub: [@bryceweiner](https://github.com/bryceweiner)

## Acknowledgments

This work uses publicly available data from:
- Atacama Cosmology Telescope (ACT) DR6
- Planck 2018 Legacy Release

Analysis follows standard techniques as described in:
- Gross & Vitells (2010) for look-elsewhere effect corrections
- Savitzky-Golay filtering for derivative computation
- Bootstrap resampling for uncertainty estimation

