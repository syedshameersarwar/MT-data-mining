"""
Bootstrap Correlation Analysis Pipeline for Cardiac Signal Feature Relationships

This module provides a comprehensive pipeline for analyzing correlations between cardiac
signal features using bootstrap statistical methods. The analysis focuses on understanding
how relationships between different cardiac parameters change under various drug treatments
and experimental conditions.

Scientific Background:
    Bootstrap correlation analysis is a robust statistical method that:
    1. Estimates correlation stability through resampling
    2. Provides confidence intervals for correlation estimates
    3. Handles non-normal distributions without parametric assumptions
    4. Quantifies uncertainty in correlation measurements

    In cardiac electrophysiology, feature correlations reveal:
    - Coupling between electrical, calcium, and mechanical processes
    - Drug-specific effects on physiological relationships
    - Tissue-level variability in signal coordination
    - Mechanistic insights into cardiac function

Key Analysis Components:
    1. Bootstrap Sampling: 1000 iterations with fixed seeds for reproducibility
    2. Spearman Correlation: Rank-based correlation robust to outliers
    3. Statistical Testing: Normality tests and confidence intervals
    4. Comparative Analysis: Cross-drug correlation comparisons
    5. Visualization: Publication-quality plots and diagrams

Pipeline Architecture:
    Data Loading → Bootstrap Analysis → Statistical Testing → Visualization → Export

Supported Features:
    - Electrical: Action potential duration, frequency
    - Calcium: Transient amplitude, kinetics, width
    - Mechanical: Contractile force amplitude, duration
    - Cross-modal: Correlations between different signal types

Drug Treatments Analyzed:
    - Baseline: Control conditions without drug intervention
    - E-4031: hERG potassium channel blocker (electrical effects)
    - Nifedipine: L-type calcium channel blocker (calcium/mechanical effects)
    - Ca²⁺ Titration: Calcium concentration modulation

Output Products:
    - Bootstrap correlation distributions (CSV files)
    - Statistical summaries with confidence intervals
    - Correlation grid plots (histograms and boxplots)
    - Comparative line plots across treatments
    - Venn diagrams of significant correlations

Command Line Usage:
    python main.py --data-path /path/to/features --output-path ./correlation_analysis
                   --optional-features duration force_peak_amplitude calc_peak_amplitude

Example Workflow:
    # Standard analysis with default parameters
    python main.py

    # Custom feature subset analysis
    python main.py --optional-features duration calc_peak_amplitude local_frequency[Hz]
                   --output-path ./custom_correlation_analysis

Scientific Applications:
    - Drug mechanism characterization
    - Biomarker identification for cardiac conditions
    - Understanding physiological coupling mechanisms
    - Validation of computational cardiac models
    - Therapeutic target identification

Authors: Cardiac Electrophysiology Research Team
Version: 1.0
Dependencies: pandas, numpy, scipy, matplotlib, matplotlib-venn
"""

import sys
import argparse
from pathlib import Path


from analyzer import CorrelationAnalyzer
from bootstrapping import BootstrapCorrelation
from visualizer import CorrelationVisualizer

# Add parent directory to Python path to import utils.py
sys.path.append(str(Path(__file__).parent.parent))
from utils import SignalData


def main(args):
    """
    Execute the complete bootstrap correlation analysis pipeline.

    This function orchestrates a comprehensive analysis workflow for investigating
    correlations between cardiac signal features across different drug treatments.
    The pipeline integrates data loading, bootstrap statistical analysis, and
    publication-quality visualization to provide deep insights into cardiac
    signal relationships.

    Analysis Workflow:
    1. Initialization Phase:
       - Configure output directories and logging
       - Initialize analysis classes with specified parameters
       - Validate input paths and create directory structure

    2. Data Loading Phase:
       - Read HDF5 feature files for all drug treatments
       - Validate data consistency and completeness
       - Merge cases by drug type and baseline conditions

    3. Bootstrap Analysis Phase:
       - For each treatment condition:
         * Perform 1000 bootstrap iterations with fixed seeds
         * Calculate Spearman correlations for all feature pairs
         * Generate statistical summaries and confidence intervals
         * Save raw correlation data and statistical results

    4. Visualization Generation Phase:
       - Create correlation grid plots (histograms + boxplots)
       - Generate comparative line plots across treatments
       - Produce Venn diagrams of significant correlations
       - Save all visualizations in publication-ready format

    5. Output Organization Phase:
       - Structure results in organized directory hierarchy
       - Export CSV files with detailed statistical summaries
       - Save high-resolution PDF plots for publication

    Key Statistical Outputs:
        For each treatment condition:
        - Bootstrap correlation distributions (1000 samples per feature pair)
        - Mean correlations with standard deviations
        - 95% confidence intervals for each correlation
        - Normality test results (Shapiro-Wilk)
        - Correlation type classification (positive/negative/inconclusive)

    Visualization Outputs:
        - Grid plots: Feature-by-feature correlation distributions
        - Comparison plots: Treatment effects on correlations
        - Venn diagrams: Overlap of significant correlations
        - All plots formatted for LaTeX thesis integration

    Args:
        args (argparse.Namespace): Command line arguments containing:
            - data_path (str): Path to directory with HDF5 feature files
            - output_path (str): Directory for saving analysis results
            - optional_features (List[str]): Subset of features for grid plots

    Directory Structure Created:
        output_path/
        ├── data/
        │   ├── bootstrapped_{treatment}_correlations.csv
        │   ├── bootstrapped_{treatment}_correlations_stats.csv
        │   ├── combined_drug_correlations.csv
        │   └── feature_pair_mapping.csv
        └── plots/
            ├── drugs/{treatment}/
            │   ├── hist_fig.pdf
            │   └── box_fig.pdf
            └── global/
                ├── correlation_venn.pdf
                ├── drug_correlations_comparison.pdf
                └── drug_correlations_comparison_numbered.pdf

    Example Usage:
        >>> import argparse
        >>> args = argparse.Namespace(
        ...     data_path='/path/to/features',
        ...     output_path='./correlation_results',
        ...     optional_features=['duration', 'force_peak_amplitude']
        ... )
        >>> main(args)
        # Executes complete analysis pipeline with specified parameters

    Console Output:
        The function provides detailed progress reporting including:
        - Configuration summary with all parameters
        - File discovery and data loading progress
        - Bootstrap iteration progress (every 100 iterations)
        - Visualization generation status
        - Final output file locations

    Note:
        The pipeline is designed for reproducibility with fixed random seeds
        and standardized output formats. All intermediate results are saved
        to enable post-hoc analysis and validation.
    """
    print("\n" + "=" * 80)
    print("Starting Correlation Analysis Pipeline")
    print("=" * 80)

    print("\nInitializing...")
    print(f"Data path: {args.data_path}")
    print(f"Output path: {args.output_path}")
    print(f"Optional features: {args.optional_features}")

    # Initialize classes
    output_path = Path(args.output_path)
    signal_data = SignalData(data_path=args.data_path)
    bootstrapped_correlations = BootstrapCorrelation()
    correlation_analyzer = CorrelationAnalyzer()
    visualizer = CorrelationVisualizer(output_path=output_path)

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

    # Analyze correlations and save results
    print("\nPerforming bootstrap correlation analysis...")
    correlation_analyses_df = {}
    correlation_analyses_dict = {}
    treatments = ["baseline", "e4031", "nifedipine", "ca_titration"]

    for treatment in treatments:
        print(f"\nProcessing {treatment}:")
        df = getattr(signal_data, f"{treatment}_cases")
        print(f"- Number of records: {len(df)}")

        print("- Calculating bootstrap correlations...")
        df_corr, corr_dict = bootstrapped_correlations.calculate_correlations(df)

        print("- Saving bootstrap correlations...")
        output_file = (
            output_path / "data" / f"bootstrapped_{treatment}_correlations.csv"
        )
        df_corr.to_csv(output_file, index=False)
        print(f"  Saved to: {output_file}")

        print("- Analyzing correlation statistics...")
        correlation_analyses_df[treatment] = (
            correlation_analyzer.analyze_bootstrap_correlations(df_corr)
        )
        correlation_analyses_dict[treatment] = corr_dict

        output_file = (
            output_path / "data" / f"bootstrapped_{treatment}_correlations_stats.csv"
        )
        correlation_analyses_df[treatment].to_csv(output_file, index=False)
        print(f"  Saved statistics to: {output_file}")

    print("\nGenerating visualization plots...")

    print("1. Creating correlation grid plots...")
    correlation_grid_plots = visualizer.create_correlation_grid_plots(
        *[correlation_analyses_dict[treatment] for treatment in treatments],
        feature_names=signal_data.features,
        optional_features=args.optional_features,
    )

    print("- Saving grid plots...")
    for i, (hist_fig, box_fig) in enumerate(correlation_grid_plots):
        treatment = treatments[i]
        hist_path = output_path / "plots" / "drugs" / treatment / "hist_fig.pdf"
        box_path = output_path / "plots" / "drugs" / treatment / "box_fig.pdf"

        hist_fig.savefig(hist_path, bbox_inches="tight", dpi=300)
        box_fig.savefig(box_path, bbox_inches="tight", dpi=300)
        print(f"  {treatment.title()}: Saved histogram and boxplot")

    print("\n2. Creating combined correlation analysis...")
    combined_correlations = correlation_analyzer.combine_drug_correlations(
        correlation_analyses_df["nifedipine"],
        correlation_analyses_df["e4031"],
        correlation_analyses_df["ca_titration"],
    )
    output_file = output_path / "data" / "combined_drug_correlations.csv"
    combined_correlations.to_csv(output_file, index=False)
    print(f"- Saved combined correlations to: {output_file}")

    print("\n3. Creating and saving Venn diagram...")
    venn_diagram = visualizer.create_venn_diagram(combined_correlations)
    output_file = output_path / "plots" / "global" / "correlation_venn.pdf"
    venn_diagram.savefig(output_file, bbox_inches="tight", dpi=300)
    print(f"- Saved Venn diagram to: {output_file}")

    print("\n4. Creating correlation comparison plots...")
    print("- Generating named feature pairs plot...")
    fig_named, _ = visualizer.plot_drug_correlations_comparison(
        correlation_analyses_df["baseline"],
        correlation_analyses_df["nifedipine"],
        correlation_analyses_df["e4031"],
        correlation_analyses_df["ca_titration"],
        use_numbered_pairs=False,
        width_fraction=2.2,
    )
    output_file = output_path / "plots" / "global" / "drug_correlations_comparison.pdf"
    fig_named.savefig(output_file, bbox_inches="tight", dpi=300)
    print(f"  Saved to: {output_file}")

    print("- Generating numbered feature pairs plot...")
    fig_numbered, _ = visualizer.plot_drug_correlations_comparison(
        correlation_analyses_df["baseline"],
        correlation_analyses_df["nifedipine"],
        correlation_analyses_df["e4031"],
        correlation_analyses_df["ca_titration"],
        use_numbered_pairs=True,
    )
    output_file = (
        output_path / "plots" / "global" / "drug_correlations_comparison_numbered.pdf"
    )
    fig_numbered.savefig(output_file, bbox_inches="tight", dpi=300)
    print(f"  Saved to: {output_file}")

    print("\n" + "=" * 80)
    print("Correlation Analysis Pipeline Completed Successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze correlations between signal features"
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
        default="./outputs/CorrelationAnalysis",
        help="Path where outputs (plots and data) will be saved",
    )
    parser.add_argument(
        "--optional-features",
        nargs="+",
        default=[
            "duration",
            "force_peak_amplitude",
            "calc_peak_amplitude",
            "force_width_0.5 s",
            "calc_width_0.5 s",
            "local_frequency[Hz]",
        ],
        help="Subset of features to include in grid correlation plots",
    )
    args = parser.parse_args()
    main(args)
