"""
t-SNE Analysis Pipeline for Cardiac Signal Feature Dimensionality Reduction

This module provides a comprehensive pipeline for t-SNE analysis of cardiac signal
features across different drug treatments. The analysis focuses on visualizing
high-dimensional feature relationships in 2D space to identify drug-specific
effects and tissue-level patterns.

Scientific Background:
    t-SNE (t-Distributed Stochastic Neighbor Embedding) is a nonlinear dimensionality
    reduction technique that:
    1. Preserves local structure and clusters in high-dimensional data
    2. Reveals patterns that may be hidden in the original feature space
    3. Enables visualization of complex relationships between cardiac parameters
    4. Helps identify drug-specific effects on cardiac signal coordination

    In cardiac electrophysiology, t-SNE analysis reveals:
    - Drug-specific clustering patterns in feature space
    - Tissue-level variability in drug responses
    - Concentration-dependent effects on cardiac function
    - Relationships between electrical, calcium, and mechanical parameters

Key Analysis Components:
    1. Data Preparation: Feature selection and normalization
    2. t-SNE Dimensionality Reduction: 2D embedding with configurable parameters
    3. Averaging: Tissue-concentration level aggregation for clarity
    4. Statistical Analysis: Cluster analysis and pattern identification
    5. Visualization: Publication-quality plots with proper legends

Pipeline Architecture:
    Data Loading → Feature Selection → t-SNE Embedding → Averaging → Visualization → Export

Supported Features:
    - Electrical: Action potential duration, frequency, local frequency
    - Calcium: Transient amplitude, kinetics, width measurements
    - Mechanical: Contractile force amplitude, duration, kinetics
    - Cross-modal: All combinations of signal type features

Drug Treatments Analyzed:
    - Baseline: Control conditions without drug intervention
    - E-4031: hERG potassium channel blocker (electrical effects)
    - Nifedipine: L-type calcium channel blocker (calcium/mechanical effects)
    - Ca²⁺ Titration: Calcium concentration modulation

Output Products:
    - t-SNE embeddings (2D coordinates)
    - Averaged tissue-concentration data points
    - Publication-quality PDF plots
    - Statistical summaries of clustering patterns

Command Line Usage:
    python main.py --data-path /path/to/features --output-path ./tsne_analysis
                   --selected-features duration force_peak_amplitude calc_peak_amplitude

Example Workflow:
    # Standard analysis with default parameters
    python main.py

    # Custom feature subset analysis
    python main.py --selected-features duration calc_peak_amplitude local_frequency[Hz]
                   --output-path ./custom_tsne_analysis

Scientific Applications:
    - Drug mechanism characterization through feature space analysis
    - Biomarker identification for cardiac conditions
    - Understanding physiological coupling mechanisms
    - Validation of computational cardiac models
    - Therapeutic target identification

Authors: Cardiac Electrophysiology Research Team
Version: 1.0
Dependencies: pandas, numpy, scikit-learn, matplotlib
"""

import sys
import argparse
from pathlib import Path
import logging

# Add parent directory to Python path to import utils.py
sys.path.append(str(Path(__file__).parent.parent))

from tsne_analyzer import TSneAnalyzer
from tsne_visualizer import TSneVisualizer
from utils import SignalData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    """
    Execute the complete t-SNE analysis pipeline.

    This function orchestrates a comprehensive t-SNE analysis workflow for investigating
    drug-specific effects on cardiac signal feature relationships. The pipeline integrates
    data loading, t-SNE dimensionality reduction, statistical analysis, and
    publication-quality visualization to provide deep insights into cardiac
    signal feature space organization.

    Analysis Workflow:
    1. Initialization Phase:
       - Configure output directories and logging
       - Initialize analysis classes with specified parameters
       - Validate input paths and create directory structure

    2. Data Loading Phase:
       - Read HDF5 feature files for all drug treatments
       - Validate data consistency and completeness
       - Merge cases by drug type and baseline conditions

    3. t-SNE Analysis Phase:
       - Prepare combined dataset with selected features
       - Perform t-SNE dimensionality reduction
       - Average coordinates by tissue and concentration
       - Generate statistical analysis of clustering patterns

    4. Visualization Generation Phase:
       - Create multi-drug t-SNE scatter plots
       - Generate concentration gradient visualizations
       - Produce tissue-specific marker plots
       - Save all visualizations in publication-ready format

    5. Output Organization Phase:
       - Structure results in organized directory hierarchy
       - Export statistical summaries and analysis results
       - Save high-resolution PDF plots for publication

    Key Statistical Outputs:
        For each treatment condition:
        - t-SNE coordinates (2D embeddings)
        - Averaged tissue-concentration data points
        - Clustering pattern statistics
        - Drug-specific effect analysis

    Visualization Outputs:
        - Multi-drug t-SNE plots with concentration gradients
        - Tissue-specific marker differentiation
        - Publication-quality legends and formatting
        - All plots formatted for LaTeX thesis integration

    Args:
        args (argparse.Namespace): Command line arguments containing:
            - data_path (str): Path to directory with HDF5 feature files
            - output_path (str): Directory for saving analysis results
            - selected_features (List[str]): Subset of features for analysis
            - random_state (int): Random seed for reproducible results
            - perplexity (float): t-SNE perplexity parameter
            - no_standard_scaling (bool): Disable standard scaling of features

    Directory Structure Created:
        output_path/
        ├── data/
        │   ├── tsne_results.csv
        │   ├── averaged_tsne_data.csv
        │   └── analysis_summary.json
        └── plots/
            └── drug_wise_global_tsne_averaged.pdf

    Example Usage:
        >>> import argparse
        >>> args = argparse.Namespace(
        ...     data_path='/path/to/features',
        ...     output_path='./tsne_results',
        ...     selected_features=['duration', 'force_peak_amplitude'],
        ...     random_state=42,
        ...     perplexity=30.0,
        ...     no_standard_scaling=True
        ... )
        >>> main(args)
        # Executes complete t-SNE analysis pipeline with specified parameters

    Console Output:
        The function provides detailed progress reporting including:
        - Configuration summary with all parameters
        - File discovery and data loading progress
        - t-SNE analysis progress and convergence
        - Visualization generation status
        - Final output file locations

    Note:
        The pipeline is designed for reproducibility with fixed random seeds
        and standardized output formats. All intermediate results are saved
        to enable post-hoc analysis and validation.
    """
    print("\n" + "=" * 80)
    print("Starting t-SNE Analysis Pipeline")
    print("=" * 80)

    print("\nInitializing...")
    print(f"Data path: {args.data_path}")
    print(f"Output path: {args.output_path}")
    print(f"Selected features: {args.selected_features}")
    print(f"Random state: {args.random_state}")
    print(f"t-SNE perplexity: {args.perplexity}")
    print(f"t-SNE n-components: {args.n_components}")
    print(f"Standard scaling: {not args.no_standard_scaling}")
    # Initialize classes
    output_path = Path(args.output_path)
    signal_data = SignalData(data_path=args.data_path)
    tsne_analyzer = TSneAnalyzer(
        random_state=args.random_state,
        perplexity=args.perplexity,
        n_components=args.n_components,
        standard_scaling=not args.no_standard_scaling,
    )
    visualizer = TSneVisualizer(output_path=str(output_path / "plots"))

    print("\nReading data files...")
    # Process data
    cases_dict = {
        "e-4031": signal_data.read_all_cases("run1b_e-4031"),
        "nifedipine": signal_data.read_all_cases("run1b_nifedipine"),
        "ca_titration": signal_data.read_all_cases("run1b_ca_titration"),
    }
    print("Files found:")
    for drug, files in cases_dict.items():
        print(f"- {drug}: {len(files)} files")

    print("\nMerging cases by drug and baseline...")
    signal_data.merge_cases_by_drug_and_baseline(cases_dict)

    # Prepare drug dataframes
    drug_dfs = [
        signal_data.e4031_cases,
        signal_data.nifedipine_cases,
        signal_data.ca_titration_cases,
    ]
    drug_names = ["e4031", "nifedipine", "ca_titration"]

    print("\nPerforming t-SNE analysis...")
    # Run complete t-SNE analysis
    analysis_results = tsne_analyzer.run_complete_analysis(
        baseline_df=signal_data.baseline_cases,
        drug_dfs=drug_dfs,
        drug_names=drug_names,
        selected_features=args.selected_features,
    )

    # Save analysis results
    print("\nSaving analysis results...")
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "data").mkdir(exist_ok=True)

    # Save t-SNE results
    tsne_results_file = output_path / "data" / "tsne_results.csv"
    analysis_results["plot_df"].to_csv(tsne_results_file, index=False)
    print(f"- Saved t-SNE results to: {tsne_results_file}")

    # Save averaged data
    averaged_data_file = output_path / "data" / "averaged_tsne_data.csv"
    analysis_results["averaged_df"].to_csv(averaged_data_file, index=False)
    print(f"- Saved averaged data to: {averaged_data_file}")

    # Save analysis summary
    import json

    summary_file = output_path / "data" / "analysis_summary.json"
    with open(summary_file, "w") as f:
        json.dump(analysis_results["analysis_results"], f, indent=2)
    print(f"- Saved analysis summary to: {summary_file}")

    print("\nGenerating visualization plots...")
    # Create comprehensive visualization
    visualization_results = visualizer.create_comprehensive_visualization(
        analysis_results=analysis_results, drug_names=drug_names, save_plots=True
    )

    print(f"- Saved main plot to: {visualization_results['main_plot']['filepath']}")

    # Print analysis summary
    print("\nAnalysis Summary:")
    summary = analysis_results["analysis_results"]
    print(f"- Total data points: {summary['overall']['total_points']}")
    print(f"- Total tissues: {summary['overall']['total_tissues']}")
    print(f"- Total drugs: {summary['overall']['total_drugs']}")

    for drug in drug_names:
        drug_summary = summary[drug]
        print(
            f"- {drug}: {drug_summary['n_points']} points, "
            f"{drug_summary['n_tissues']} tissues, "
            f"{drug_summary['n_concentrations']} concentrations"
        )

    print("\n" + "=" * 80)
    print("t-SNE Analysis Pipeline Completed Successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Perform t-SNE analysis on cardiac signal features"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the data directory containing HDF5 files",
        default="/home/syedshameersarwar/Desktop/Work/MyOfarm/Thesis/Experiments/Mea-Peak-Clustering/Analysis/FeatureGMM/Merge/Data/Features",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./outputs/TSneAnalysis",
        help="Path where outputs (plots and data) will be saved",
    )
    parser.add_argument(
        "--selected-features",
        nargs="+",
        default=[
            "duration",
            "force_peak_amplitude",
            "calc_peak_amplitude",
            "force_width_0.2 s",
            "force_width_0.5 s",
            "force_width_0.8 s",
            "calc_width_0.2 s",
            "calc_width_0.5 s",
            "calc_width_0.8 s",
            "force_rise_time_0.2_0.8 s",
            "force_rise_time_0.8_max s",
            "calc_rise_time_0.2_0.8 s",
            "calc_rise_time_0.8_max s",
            "force_decay_time_0.2_0.8 s",
            "force_decay_time_max_0.8 s",
            "calc_decay_time_0.2_0.8 s",
            "calc_decay_time_max_0.8 s",
            "local_frequency[Hz]",
        ],
        help="Subset of features to include in t-SNE analysis, Default: All features",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible results",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        help="Number of t-SNE components",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity parameter",
    )
    parser.add_argument(
        "--no-standard-scaling",
        action="store_true",
        help="Disable standard scaling of features (default: scaling enabled)",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=1000,
        help="t-SNE number of iterations",
    )

    args = parser.parse_args()
    main(args)
