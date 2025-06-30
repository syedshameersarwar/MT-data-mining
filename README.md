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
- [Phase 3: Advanced Analytics](#phase-3-advanced-analytics)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)

## Project Overview

This pipeline provides a complete workflow for cardiac electrophysiology research, enabling:

- **Multi-modal Feature Extraction**: Extraction and integration of features from electrical (MEA/field potential), calcium (fluorescence), and mechanical (contractile force) signals
- **Statistical Modeling**: Advanced mixed-effects modeling and hypothesis testing for treatment effects
- **Dose-Response Curves**: Hill equation fitting for pharmacological parameter estimation
- **Correlation Analysis**: Bootstrap-based correlation analysis with uncertainty quantification
- **Dimensionality Reduction and Feature Clustering**: t-SNE analysis for high-dimensional feature visualization and Principle Feature Analysis (PFA) based Feature Clustering
- **Feature Selection**: Principal Feature Analysis (PFA) for optimal feature subset identification
- **Regression Analysis**: GLMM Lasso regression for feature relationship modeling
- **Relative Comparison**: Multi-level comparison analysis for drug effects
- **Publication-Quality Visualization**: LaTeX-compatible figures for thesis and journal submission

### Supported Drug Treatments
- **E-4031**: hERG potassium channel blocker (electrical effects)
- **Nifedipine**: L-type calcium channel blocker (calcium/mechanical effects)  
- **Ca²⁺ Titration**: Calcium concentration modulation experiments

### Key Analysis Capabilities
- Mixed-effects statistical modeling with R integration
- Hill equation fitting for EC50/IC50 estimation
- Bootstrap correlation analysis with confidence intervals
- t-SNE dimensionality reduction with configurable parameters
- Principal Feature Analysis with automatic clustering
- GLMM Lasso regression for sparse feature selection
- Multi-level relative comparison analysis
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
├── analysis/                   # Phase 2 & 3: Statistical and advanced analysis
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
│   ├── tsne/                  # t-SNE dimensionality reduction analysis
│   │   ├── main.py            # t-SNE analysis pipeline
│   │   ├── tsne_analyzer.py   # t-SNE statistical methods
│   │   └── tsne_visualizer.py # t-SNE visualization
│   ├── regression/            # GLMM Lasso regression analysis
│   │   ├── main.py            # Regression analysis pipeline
│   │   ├── regression_analyzer.py  # Regression statistical methods
│   │   ├── regression_visualizer.py # Regression visualization
│   │   └── glmmLasso.R        # R script for GLMM Lasso
│   ├── pfa/                   # Principal Feature Analysis
│   │   ├── main.py            # PFA analysis pipeline
│   │   ├── pfa_analyzer.py    # PFA statistical methods
│   │   ├── pfa_visualizer.py  # PFA visualization
│   │   └── glmmLasso.R        # R script for GLMM Lasso
│   ├── relative-comparison/   # Relative comparison analysis
│   │   ├── main.py            # Relative comparison pipeline
│   │   ├── relative_comparison_analyzer.py  # Comparison methods
│   │   └── relative_comparison_visualizer.py # Comparison visualization
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

## Phase 3: Advanced Analytics

### Overview
The advanced analytics phase provides four complementary approaches for deep analysis of cardiac signal relationships, feature selection, and drug effect characterization.

> **Important**: All analysis commands must be run from the `analysis/` root directory due to the shared utilities import structure. Do not run commands from individual subdirectories.

### 1. t-SNE Dimensionality Reduction Analysis

#### Purpose
t-SNE (t-Distributed Stochastic Neighbor Embedding) analysis for visualizing high-dimensional cardiac feature relationships in 2D space. This analysis reveals drug-specific clustering patterns and tissue-level variability in feature space.

#### Usage
```bash
cd analysis

# Standard t-SNE analysis with default parameters
python tsne/main.py \
    --data-path "/path/to/extracted/features" \
    --output-path "./tsne_results"

# Custom feature subset analysis
python tsne/main.py \
    --data-path "/path/to/features" \
    --output-path "./custom_tsne" \
    --selected-features "duration" "force_peak_amplitude" "calc_peak_amplitude" \
    --perplexity 25.0 \
    --random-state 42

# Analysis without feature standardization
python tsne/main.py \
    --no-standard-scaling \
    --output-path "./tsne_no_scaling"
```

#### Parameters

| Parameter | Description | Required | Default |
|-----------|-------------|----------|---------|
| `--data-path` | Path to directory containing HDF5 feature files | Yes | - |
| `--output-path` | Directory for saving results and plots | No | `./outputs/TSneAnalysis` |
| `--selected-features` | Subset of features for analysis | No | All features |
| `--random-state` | Random seed for reproducible results | No | 42 |
| `--perplexity` | t-SNE perplexity parameter | No | 30.0 |
| `--n-components` | Number of t-SNE components | No | 2 |
| `--n-iterations` | Number of t-SNE iterations | No | 1000 |
| `--no-standard-scaling` | Disable standard scaling of features | No | False (scaling enabled) |

#### Special Requirements
- **Standard Scaling**: By default, features are standardized (mean=0, std=1) before t-SNE analysis. Use `--no-standard-scaling` to preserve original feature scales.
- **Feature Selection**: Large feature sets may require longer computation time. Consider using `--selected-features` for focused analysis.

#### Output Structure
```
tsne_results/
├── data/
│   ├── tsne_results.csv              # Complete t-SNE embeddings
│   ├── averaged_tsne_data.csv        # Tissue-concentration averaged data
│   └── analysis_summary.json         # Analysis metadata and statistics
└── plots/
    └── drug_wise_global_tsne_averaged.pdf  # Publication-quality t-SNE plot
```

### 2. GLMM Lasso Regression Analysis

#### Purpose
Generalized Linear Mixed Model (GLMM) Lasso regression analysis for identifying sparse feature relationships and understanding drug-specific changes in feature interdependencies.

#### Usage
```bash
cd analysis

# Complete regression analysis for all drugs
python regression/main.py \
    --data-path "/path/to/extracted/features" \
    --output-path "./regression_results"

# Analysis for specific drugs only
python regression/main.py \
    --data-path "/path/to/features" \
    --drugs "baseline" "e4031" "nifedipine" \
    --output-path "./selective_regression"

# Analysis without visualization
python regression/main.py \
    --no-visualization \
    --output-path "./regression_data_only"
```

#### Parameters

| Parameter | Description | Required | Default |
|-----------|-------------|----------|---------|
| `--data-path` | Path to directory containing HDF5 feature files | Yes | - |
| `--output-path` | Directory for saving data results | No | `./outputs/RegressionAnalysis/data` |
| `--plots-path` | Directory for saving plots | No | `./outputs/RegressionAnalysis/plots` |
| `--drugs` | Drugs to analyze | No | All drugs |
| `--r-script-path` | Path to GLMM Lasso R script | No | `glmmLasso.R` |
| `--no-visualization` | Skip visualization generation | No | False |
| `--verbose` | Enable verbose logging | No | False |

#### Special Requirements
- **R Dependencies**: Requires additional R packages: `glmmLasso`, `MASS`, `nlme`, `lme4`, `lmerTest`
- **R Script**: Uses custom R script (`glmmLasso.R`) for GLMM Lasso implementation
- **Memory**: May require significant memory for large datasets

#### Output Structure
```
regression_results/
├── data/
│   ├── baseline_data.csv              # Baseline regression data
│   ├── baseline_frequency_table.csv   # Baseline frequency table
│   ├── baseline_coefficient_matrix.csv # Baseline coefficient matrix
│   ├── baseline_weight_matrix.csv     # Baseline weight matrix (R output)
│   ├── e4031_data.csv                 # E-4031 regression data
│   ├── e4031_frequency_table.csv      # E-4031 frequency table
│   ├── e4031_coefficient_matrix.csv   # E-4031 coefficient matrix
│   ├── e4031_weight_matrix.csv        # E-4031 weight matrix (R output)
│   ├── nifedipine_data.csv            # Nifedipine regression data
│   ├── nifedipine_frequency_table.csv # Nifedipine frequency table
│   ├── nifedipine_coefficient_matrix.csv # Nifedipine coefficient matrix
│   ├── nifedipine_weight_matrix.csv   # Nifedipine weight matrix (R output)
│   ├── ca_titration_data.csv          # Ca²⁺ titration regression data
│   ├── ca_titration_frequency_table.csv # Ca²⁺ titration frequency table
│   ├── ca_titration_coefficient_matrix.csv # Ca²⁺ titration coefficient matrix
│   ├── ca_titration_weight_matrix.csv # Ca²⁺ titration weight matrix (R output)
│   ├── glmmLasso.R                    # R script used for analysis
│   └── regression_analysis_report.txt # Analysis summary report
└── plots/
    ├── baseline_nifedipine_comparison.pdf # Baseline vs Nifedipine comparison
    └── e4031_ca_titration_comparison.pdf  # E-4031 vs Ca²⁺ titration comparison
```

### 3. Principal Feature Analysis (PFA)

#### Purpose
Principal Feature Analysis for automatic feature selection and redundancy reduction. PFA combines PCA decomposition with DBSCAN clustering to identify representative features from each functional group.

#### Usage
```bash
cd analysis

# Standard PFA analysis with default parameters
python pfa/main.py \
    --data-path "/path/to/extracted/features" \
    --output-path "./pfa_results"

# Custom parameter analysis
python pfa/main.py \
    --data-path "/path/to/features" \
    --output-path "./custom_pfa" \
    --explained-var 0.90 \
    --min-samples 3 \
    --eps 0.1

# High precision analysis
python pfa/main.py \
    --explained-var 0.98 \
    --min-samples 2 \
    --output-path "./high_precision_pfa"
```

#### Parameters

| Parameter | Description | Required | Default |
|-----------|-------------|----------|---------|
| `--data-path` | Path to directory containing HDF5 feature files | Yes | - |
| `--output-path` | Directory for saving results and plots | No | `./outputs/PfaAnalysis` |
| `--explained-var` | Target explained variance for PCA | No | 0.95 |
| `--min-samples` | DBSCAN min_samples parameter | No | 2 |
| `--eps` | DBSCAN epsilon parameter (auto-determined if None) | No | None |

#### Special Requirements
- **Knee Detection**: Automatically determines optimal DBSCAN epsilon using knee detection algorithm
- **Feature Reduction**: Typically reduces feature set by 40-70% while preserving physiological information
- **Clustering**: Groups features with similar PCA contributions for redundancy elimination

#### Output Structure
```
pfa_results/
├── data/
│   ├── selected_features_e4031.json      # Selected features for E-4031
│   ├── selected_features_nifedipine.json # Selected features for Nifedipine
│   ├── selected_features_ca_titration.json # Selected features for Ca²⁺ titration
│   ├── analysis_summary_e4031.json       # Analysis summary for E-4031
│   ├── analysis_summary_nifedipine.json  # Analysis summary for Nifedipine
│   ├── analysis_summary_ca_titration.json # Analysis summary for Ca²⁺ titration
│   ├── pca_components_e4031.csv          # Subset of PCA components for E-4031
│   ├── pca_components_nifedipine.csv     # Subset of PCA components for Nifedipine
│   └── pca_components_ca_titration.csv   # Subset of PCA components for Ca²⁺ titration
└── plots/
    └── drugs/
        ├── e4031/
        │   ├── pfa_dbscan_cluster_report_e4031.pdf
        │   ├── pfa_dbscan_kneedle_curve_e4031.pdf
        │   └── pfa_dbscan_explained_variance_e4031.pdf
        ├── nifedipine/
        │   ├── pfa_dbscan_cluster_report_nifedipine.pdf
        │   ├── pfa_dbscan_kneedle_curve_nifedipine.pdf
        │   └── pfa_dbscan_explained_variance_nifedipine.pdf
        └── ca_titration/
            ├── pfa_dbscan_cluster_report_ca_titration.pdf
            ├── pfa_dbscan_kneedle_curve_ca_titration.pdf
            └── pfa_dbscan_explained_variance_ca_titration.pdf
```

### 4. Relative Comparison Analysis

#### Purpose
Multi-level relative comparison analysis for understanding drug effects across different analysis scales: tissue-specific changes, global averages, and target concentration comparisons (EC50/IC50).

#### Usage
```bash
cd analysis

# Standard relative comparison analysis
python relative-comparison/main.py \
    --data-path "/path/to/extracted/features" \
    --output-path "./relative_comparison_results"

# Analysis with EC50/IC50 target concentrations
python relative-comparison/main.py \
    --data-path "/path/to/features" \
    --output-path "./target_concentration_analysis" \
    --target-concentrations 0.03 1.0 1.0

# Analysis without saving results or plots
python relative-comparison/main.py \
    --data-path "/path/to/features" \
    --output-path "./analysis_only" \
    --no-save-results \
    --no-save-plots
```

#### Parameters

| Parameter | Description | Required | Default |
|-----------|-------------|----------|---------|
| `--data-path` | Path to directory containing HDF5 feature files | Yes | - |
| `--output-path` | Directory for saving results and plots | No | `./outputs/RelativeComparison` |
| `--target-concentrations` | Target concentrations for EC50/IC50 analysis [e4031, nifedipine, ca_titration] | No | [0.03, 1.0, 1.0] |
| `--no-save-results` | Do not save analysis results to CSV files | No | False |
| `--no-save-plots` | Do not save plots to PDF files | No | False |
| `--verbose` | Enable verbose logging | No | False |

#### Special Requirements
- **Target Concentrations**: Specify EC50/IC50 concentrations for each drug to enable concentration-specific analysis (Default: [0.03, 1.0, 1.0] for e4031, nifedipine, and ca_titration)
- **Multi-level Analysis**: Performs tissue-specific, global, and target concentration comparisons
- **Comprehensive Output**: Generates extensive comparison matrices and visualizations

#### Output Structure
```
relative_comparison_results/
├── tissue_specific_mean_features.csv      # Tissue-level mean feature values
├── tissue_specific_relative_differences.csv # Tissue-level relative changes
├── global_mean_features.csv               # Global mean feature values
├── global_relative_differences.csv        # Global relative changes
├── target_concentration_mean_features.csv # EC50/IC50 concentration mean features
├── target_concentration_relative_differences.csv # EC50/IC50 concentration changes
├── global/
│   ├── global_mean_relative_changes.pdf   # Global mean relative changes plot
│   └── ec_ic50_mean_relative_changes.pdf  # EC50/IC50 mean relative changes plot
└── drugs/
    ├── e-4031/
    │   └── relative_differences_e4031.pdf # E-4031 relative differences plot
    ├── nifedipine/
    │   └── relative_differences_nifedipine.pdf # Nifedipine relative differences plot
    └── ca-titration/
        └── relative_differences_ca_titration.pdf # Ca²⁺ titration relative differences plot
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
h5py>=3.1.0  # For HDF5 file handling in feature extraction
tables>=3.7.0

# Web framework for plotly interactive plots
kaleido>=0.2.1  # For plotly static image export

# Advanced analytics dependencies
kneed==0.8.5  # For knee detection in PFA analysis

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
    "ggplot2",       # Plotting (optional)
    "glmmLasso",     # GLMM Lasso regression (for regression analysis)
    "MASS",          # Statistical functions
    "nlme",          # Nonlinear mixed-effects models
    "lme4"           # Linear mixed-effects models
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
R -e "install.packages(c('lmerTest', 'multcomp', 'broom.mixed', 'great_tables', 'glmmLasso', 'MASS', 'nlme', 'lme4'))"
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

### 6. Advanced Analytics

#### t-SNE Analysis
```bash
# Already in analysis directory
python tsne/main.py \
    --data-path "../extracted_features" \
    --output-path "./tsne_analysis" \
    --selected-features "duration" "force_peak_amplitude" "calc_peak_amplitude"
```

#### Regression Analysis
```bash
# Already in analysis directory
python regression/main.py \
    --data-path "../extracted_features" \
    --output-path "./regression_analysis"
```

#### Principal Feature Analysis
```bash
# Already in analysis directory
python pfa/main.py \
    --data-path "../extracted_features" \
    --output-path "./pfa_analysis" \
    --explained-var 0.95
```

#### Relative Comparison Analysis
```bash
# Already in analysis directory
python relative-comparison/main.py \
    --data-path "../extracted_features" \
    --output-path "./relative_comparison_analysis" \
    --target-concentrations 0.03 1.0 1.0
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
