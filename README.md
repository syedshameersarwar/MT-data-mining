# Cardiac Electrophysiology Multi-Modal Analysis Pipeline

A comprehensive computational pipeline for analyzing cardiac signal relationships across electrical, calcium, and mechanical modalities under various drug treatments. This project integrates feature extraction, statistical modeling, and advanced visualization to provide insights into cardiac drug mechanisms and physiological coupling.

## Table of Contents

- [Project Overview](#project-overview)
- [Scientific Background](#scientific-background)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Phase 1: Feature Extraction](#phase-1-feature-extraction)
- [Phase 2: Statistical Analysis](#phase-2-statistical-analysis)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)

## Project Overview

This pipeline provides a complete workflow for cardiac electrophysiology research, enabling:

- **Multi-modal Feature Extraction**: Extraction and integration of features from electrical (MEA/field potential), calcium (fluorescence), and mechanical (contractile force) signals
- **Statistical Modeling**: Advanced mixed-effects modeling and hypothesis testing for treatment effects
- **Dose-Response Curves**: Hill equation fitting for pharmacological parameter estimation
- **Correlation Analysis**: Bootstrap-based correlation analysis with uncertainty quantification
- **Publication-Quality Visualization**: LaTeX-compatible figures for thesis and journal submission

### Supported Drug Treatments
- **E-4031**: hERG potassium channel blocker (electrical effects)
- **Nifedipine**: L-type calcium channel blocker (calcium/mechanical effects)  
- **Ca²⁺ Titration**: Calcium concentration modulation experiments

### Key Analysis Capabilities
- Mixed-effects statistical modeling with R integration
- Hill equation fitting for EC50/IC50 estimation
- Bootstrap correlation analysis with confidence intervals
- Comprehensive visualization suite for publication

## Scientific Background

### Signal Modalities
1. **Electrical Signals (MEA/Field Potential)**: Action potential characteristics including duration, amplitude, and frequency
2. **Calcium Signals**: Intracellular calcium transients reflecting excitation-contraction coupling
3. **Mechanical Signals**: Contractile force measurements indicating cardiac output capacity

### Analysis Methods
- **Mixed-Effects Models**: Account for tissue-level random effects and nested experimental structure
- **Hill Equation Fitting**: Pharmacological dose-response relationship modeling
- **Bootstrap Correlation**: Robust uncertainty quantification for feature relationships
- **Multi-modal Integration**: Synchronized analysis across signal types

## Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 18.04+), macOS (10.14+), or Windows 10+ with WSL
- **Memory**: Minimum 8GB RAM (16GB+ recommended for large datasets)
- **Storage**: 15GB+ free space for dependencies and data
- **Python**: 3.10+ (3.11 recommended)
- **R**: 4.1+ for statistical modeling

### Required Input Data
The feature extraction phase requires preprocessed data from the [**arrythmia-classification**](https://github.com/myotwin/arrhythmia-classification) project:

1. **Processed HDF5 Files**: Time-series data for force, calcium, and MEA signals
   - Location: `<arrythmia-classification>/Preprocessed/HDFs/`
   - Format: HDF5 files with synchronized multi-modal signals
   - Naming: `{experiment_id}_{condition}.hdf`

2. **Force Peaks JSON Files**: Detected force peak locations and properties
   - Location: `<arrythmia-classification>/Preprocessed/Peaks/`
   - Format: JSON files with peak detection results
   - Content: Peak indices, amplitudes, and validation flags

3. **Arrhythmia Classification Models**: Pre-trained ML models for beat classification
   - Location: `<arrythmia-classification>/Models/`
   - Types: Force and calcium signal arrythmia classifiers
   - Format: Joblib-serialized scikit-learn models

## Environment Setup

### 1. Python Environment Setup

```bash
# Create virtual environment
python -m venv cardiac_analysis_env
source cardiac_analysis_env/bin/activate  # Linux/macOS
# cardiac_analysis_env\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 2. R Environment Setup

#### Install R (if not already installed)

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install r-base r-base-dev
```

**macOS:**
```bash
brew install r
```

**Windows:**
Download from [CRAN](https://cran.r-project.org/bin/windows/base/)

#### Install Required R Packages

```bash
# Start R session
R

# In R console:
install.packages(c(
    "lmerTest",      # Mixed-effects models with p-values
    "multcomp",      # Multiple comparisons (Dunnett's test)
    "broom.mixed",   # Tidy mixed model outputs
    "dplyr",         # Data manipulation
    "ggplot2"        # Plotting (optional)
    "Matrix",
    "report"
))

# Exit R
quit()
```

#### Verify R-Python Integration

```python
# Test rpy2 installation
python -c "import rpy2.robjects as ro; print('R-Python integration working!')"

# Test R package availability
python -c "
import rpy2.robjects.packages as rpackages
lmerTest = rpackages.importr('lmerTest')
print('R packages accessible from Python!')
"
```

## Project Structure

```
cardiac-analysis/
├── feature-extraction/          # Phase 1: Multi-modal feature extraction
│   ├── main.py                 # Main extraction pipeline
│   ├── common.py               # Common feature extraction utilities
│   ├── fp.py                   # Field potential feature extraction
├── analysis/                   # Phase 2: Statistical analysis
│   ├── significance/           # Mixed-effects statistical modeling
│   │   ├── main.py            # Statistical analysis pipeline
│   │   ├── statistical_modeling.py  # R/Python statistical methods
│   │   └── plotting.py        # Diagnostic and significance plots
│   ├── hill-fitting/          # Dose-response curve analysis
│   │   ├── main.py            # Hill fitting pipeline
│   │   └── visualizer.py      # Hill curve visualization
│   ├── correlation-bootstrapping/  # Feature correlation analysis
│   │   ├── main.py            # Bootstrap correlation pipeline
│   │   ├── bootstrapping.py   # Bootstrap statistical methods
│   │   ├── analyzer.py        # Correlation statistical analysis
│   │   └── visualizer.py      # Correlation visualization
│   └── utils.py               # Shared utilities and data handling for analysis
├── utils/                     # Shared utilities and data handling for feature extraction and general analysis
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Phase 1: Feature Extraction

### Overview
The feature extraction phase integrates multi-modal cardiac signals to create synchronized feature datasets. It processes electrical (MEA), calcium, and mechanical (force) signals to extract physiologically relevant parameters.

### Prerequisites
Ensure you have completed the [**arrythmia-classification**](https://github.com/myotwin/arrhythmia-classification) project to generate:
- Preprocessed HDF5 signal files
- Force peak detection results
- Trained arrhythmia classification models

### Running Feature Extraction

#### Basic Usage
```bash
cd feature-extraction

python main.py \
    --data_dir "/path/to/arrythmia-classification/Preprocessed/HDFs" \
    --force_peaks_dir "/path/to/arrythmia-classification/Preprocessed/Peaks" \
    --output_dir "./extracted_features" \
    --field_potential_case_filter "run1b"
```

#### Advanced Configuration
```bash
python main.py \
    --data_dir "/path/to/preprocessed/hdfs" \
    --force_peaks_dir "/path/to/force/peaks" \
    --output_dir "./features" \
    --field_potential_case_filter "run1b" \
    --dropped_case_file "./dropped_cases.json" \
    --force_arrythmia_classifier_path "/path/to/force_classifier.joblib" \
    --calcium_arrythmia_classifier_path "/path/to/calcium_classifier.joblib" \
    --raw_data_dir "/path/to/raw/data"
```

#### Parameter Descriptions

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data_dir` | Path to preprocessed HDF5 files | Required |
| `--force_peaks_dir` | Path to force peak JSON files | Required |
| `--output_dir` | Output directory for extracted features | `./Features` |
| `--field_potential_case_filter` | Filter for field potential cases | `run1b` |
| `--dropped_case_file` | JSON file with cases to exclude | `./dropped_cases.json` |
| `--force_arrythmia_classifier_path` | Force signal classifier model | See default path |
| `--calcium_arrythmia_classifier_path` | Calcium signal classifier model | See default path |
| `--raw_data_dir` | Raw data HDF5 directory | Required |

### Output Structure
```
extracted_features/
├── Data/
│   ├── case1.csv              # Merged features for each case
│   ├── case2.csv
│   └── ...
├── Plots/
│   ├── case1.html             # Interactive visualization
│   ├── case2.html
│   └── ...
└── skipped_cases.json         # Cases without field potential data
```

### Feature Categories Extracted

1. **Electrical Features (Field Potential)**:
   - Action potential duration
   - Amplitude characteristics
   - Frequency analysis
   - Morphological parameters

2. **Calcium Features**:
   - Transient amplitude
   - Rise and decay kinetics
   - Duration measurements
   - Peak characteristics

3. **Mechanical Features (Force)**:
   - Contractile amplitude
   - Force development kinetics
   - Duration parameters
   - Peak force characteristics

4. **Integrated Features**:
   - Cross-modal timing relationships
   - Arrhythmia classification results
   - Quality control flags (potential outliers, etc)

## Phase 2: Statistical Analysis

### Overview
The analysis phase provides three complementary approaches to understand cardiac signal relationships and drug effects.

> **Important**: All analysis commands must be run from the `analysis/` root directory due to the shared utilities import structure. Do not run commands from individual subdirectories (e.g., `analysis/significance/`, `analysis/hill-fitting/`, etc.).

### 1. Statistical Significance Analysis

#### Purpose
Mixed-effects statistical modeling to identify significant drug effects on cardiac parameters with proper accounting for tissue-level variability. The analysis automatically processes all three drug treatments (E-4031, Nifedipine, Ca²⁺ titration) with predefined feature sets and random effects structures.

#### Usage
```bash
cd analysis

# Run complete significance analysis for all drugs
python significance/main.py \
    --data-path "/path/to/extracted/features" \
    --output-path "./significance_results"

# With custom paths
python significance/main.py \
    --data-path "/path/to/features" \
    --output-path "./statistical_analysis"
```

#### Parameters

| Parameter | Description | Required | Default |
|-----------|-------------|----------|---------|
| `--data-path` | Path to directory containing HDF5 feature files | Yes | - |
| `--output-path` | Directory for saving results and plots | No | `./outputs/SignificanceAnalysis` |

#### Predefined Analysis Configuration

The analysis uses predefined configurations for each drug (can be changed in the code at line `#14` of `analysis/significance/main.py`):

- **E-4031** (hERG K⁺ channel blocker):
  - Features: `duration`, `force_peak_amplitude`, `calc_peak_amplitude`
  - Random effects: Tissue-only (simplified due to convergence issues as number of tissues and concentrations is too low)

- **Nifedipine** (L-type Ca²⁺ channel blocker):
  - Features: `force_rise_time_0.2_0.8 s`, `calc_rise_time_0.2_0.8 s`, `force_decay_time_0.2_0.8 s`, `calc_decay_time_0.2_0.8 s`
  - Random effects: Full nested structure (tissue + concentration within tissue)

- **Ca²⁺ Titration**:
  - Features: `force_peak_amplitude`
  - Random effects: Full nested structure

#### Output Structure
```
significance_results/
├── e-4031/
│   ├── data/frequency_table.csv
│   └── plots/
│       ├── qq_plots_lmer_3.pdf
│       ├── residual_lmer_3.pdf
│       └── significance_lmer_3.pdf
├── nifedipine/
│   ├── data/frequency_table.csv
│   └── plots/
│       ├── qq_plots_lmer_4.pdf
│       ├── residual_lmer_4.pdf
│       └── significance_lmer_4.pdf
└── ca_titration/
    ├── data/frequency_table.csv
    └── plots/
        ├── qq_plots_lmer_1.pdf
        ├── residual_lmer_1.pdf
        └── significance_lmer_1.pdf
```

### 2. Hill Fitting Analysis

#### Purpose
Dose-response curve fitting using the Hill equation to estimate drug potency (EC50/IC50) and cooperativity parameters.

#### Usage
```bash
cd analysis

# Compare nifedipine vs calcium titration
python hill-fitting/main.py \
    --data-path "/path/to/features" \
    --output-path "./hill_results" \
    --drug-pair "nifedipine-ca" \
    --features "duration" "force_peak_amplitude" "calc_peak_amplitude"

# E-4031 vs calcium titration comparison
python hill-fitting/main.py \
    --data-path "/path/to/features" \
    --output-path "./hill_e4031" \
    --drug-pair "e4031-ca" \
    --features "duration" "local_frequency[Hz]"

# Comprehensive multi-feature analysis
python hill-fitting/main.py \
    --drug-pair "nifedipine-e4031" \
    --features "duration" "force_peak_amplitude" "calc_peak_amplitude" "local_frequency[Hz]"
```

#### Supported Drug Pairs

| Pair | Description |
|------|-------------|
| `nifedipine-ca` | L-type Ca²⁺ blocker vs Ca²⁺ titration |
| `nifedipine-e4031` | Ca²⁺ blocker vs K⁺ blocker comparison |
| `e4031-ca` | hERG K⁺ blocker vs Ca²⁺ titration |

#### Output Files
- `hill_fitting_{drug_pair}.pdf`: Dose-response curves with fitted parameters
- Console output with EC50/IC50 values and fit statistics

### 3. Bootstrap Correlation Analysis

#### Purpose
Robust correlation analysis between cardiac features with uncertainty quantification using bootstrap resampling methods.

#### Usage
```bash
cd analysis

# Standard correlation analysis
python correlation-bootstrapping/main.py \
    --data-path "/path/to/features" \
    --output-path "./correlation_results"

# Custom feature subset analysis
python correlation-bootstrapping/main.py \
    --data-path "/path/to/features" \
    --output-path "./custom_correlations" \
    --optional-features "duration" "force_peak_amplitude" "calc_peak_amplitude"

# Full feature analysis
python correlation-bootstrapping/main.py \
    --optional-features "duration" "force_peak_amplitude" "calc_peak_amplitude" \
                       "force_width_0.5" "calc_width_0.5" "local_frequency[Hz]"
```

#### Output Structure
```
correlation_results/
├── data/
│   ├── bootstrapped_{treatment}_correlations.csv      # Raw bootstrap data
│   ├── bootstrapped_{treatment}_correlations_stats.csv  # Statistical summaries
│   ├── combined_drug_correlations.csv                 # Cross-treatment comparison
│   └── feature_pair_mapping.csv                       # Feature pair reference
└── plots/
    ├── drugs/
    │   ├── baseline/
    │   ├── e4031/
    │   ├── nifedipine/
    │   └── ca_titration/
    └── global/
        ├── correlation_venn.pdf                        # Correlation overlap diagram
        ├── drug_correlations_comparison.pdf            # Comparative analysis
        └── drug_correlations_comparison_numbered.pdf   # Compact numbered version
```

## Dependencies

### Python Dependencies (requirements.txt)

```txt
# Core scientific computing
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Statistical analysis
statsmodels>=0.12.0
pymer4==0.8.0

# R integration for mixed-effects modeling
rpy2>=3.4.0

# Machine learning (for feature extraction and analysis)
scikit-learn>=1.0.0
joblib>=1.0.0

# Signal processing and feature extraction
tsfel>=0.1.4

# Visualization
matplotlib>=3.4.0
plotly>=5.0.0
matplotlib-venn>=0.11.6
PyWavelets>=1.4.0

# Hill fitting for dose-response analysis
hillfit>=0.1.0
# File handling and utilities
pathlib2>=2.3.6
argparse  # Built-in but explicit for clarity
h5py>=3.1.0  # For HDF5 file handling in feature extraction
tables>=3.7.0
# Web framework for plotly interactive plots
kaleido>=0.2.1  # For plotly static image export

# Development and testing (optional)
pytest>=6.0.0
black>=21.0.0
flake8>=3.9.0
```

### R Dependencies

```r
# Install in R console
install.packages(c(
    "lmerTest",      # Mixed-effects models with p-values
    "multcomp",      # Multiple comparisons and post-hoc tests
    "broom.mixed",   # Tidy model outputs
    "great_tables",  # Publication-quality tables
    "dplyr",         # Data manipulation
    "tidyr",         # Data reshaping
    "ggplot2"        # Plotting (optional)
))
```

## Complete Analysis Workflow Example

### 1. Setup Environment
```bash
# Create and activate environment
python -m venv cardiac_env
source cardiac_env/bin/activate
pip install -r requirements.txt

# Install R packages (in R console)
R -e "install.packages(c('lmerTest', 'multcomp', 'broom.mixed', 'great_tables'))"
```

### 2. Feature Extraction
```bash
cd feature-extraction
python main.py \
    --data_dir "/path/to/arrythmia-classification/Preprocessed/HDFs" \
    --force_peaks_dir "/path/to/arrythmia-classification/Preprocessed/Peaks" \
    --output_dir "../extracted_features"
```

### 3. Statistical Analysis
```bash
cd analysis
python significance/main.py \
    --data-path "../extracted_features" \
    --output-path "./significance_analysis"
```

### 4. Hill Fitting Analysis
```bash
# Already in analysis directory
python hill-fitting/main.py \
    --data-path "../extracted_features" \
    --output-path "./dose_response_analysis" \
    --drug-pair "nifedipine-ca" \
    --features "duration" "force_peak_amplitude"
```

### 5. Correlation Analysis
```bash
# Already in analysis directory
python correlation-bootstrapping/main.py \
    --data-path "../extracted_features" \
    --output-path "./correlation_analysis"
```

## Troubleshooting

### Common Issues and Solutions

#### 1. R Integration Problems
```bash
# Error: rpy2 not working
# Solution: Reinstall with proper R path
pip uninstall rpy2
R_HOME=$(R RHOME) pip install rpy2

# Error: R packages not found
# Solution: Check R library path
R -e ".libPaths()"
```

#### 2. Memory Issues
```bash
# Error: Memory overflow during bootstrap
# Solution: Reduce bootstrap iterations or feature set
python main.py --n-bootstraps 500  # Instead of 1000
```

#### 3. Missing Input Data
```bash
# Error: HDF5 files not found
# Solution: Verify arrythmia-classification project completion
ls /path/to/arrythmia-classification/Preprocessed/HDFs/*.hdf
```

#### 4. Hill Fitting Failures
```bash
# Error: Hill fitting convergence issues
# Solution: Check concentration range and data quality
# Ensure sufficient concentration points and dose-response relationship
```

## Support and Contact

For questions, issues, or contributions:
- **Issues**: Please use the GitHub issue tracker
- **Documentation**: See individual module docstrings for detailed API documentation
- **Contributions**: Pull requests welcome following the existing code style
