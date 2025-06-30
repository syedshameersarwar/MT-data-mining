"""
Principal Feature Analysis (PFA) Pipeline for Cardiac Signal Feature Selection

This module provides a comprehensive pipeline for PFA analysis of cardiac signal
features across different drug treatments. The analysis focuses on identifying
representative features using PCA and DBSCAN clustering to reduce redundancy
while preserving important physiological information.

Scientific Background:
    Principal Feature Analysis (PFA) is a feature selection technique that:
    1. Applies PCA to reduce dimensionality and identify principal components
    2. Uses DBSCAN clustering to group features with similar PCA contributions
    3. Selects representative features from each cluster to reduce redundancy
    4. Automatically determines optimal clustering parameters using knee detection

    In cardiac electrophysiology, PFA reveals:
    - Groups of features with similar physiological roles
    - Redundant features that can be eliminated
    - Representative features for each functional group
    - Optimal feature subsets for downstream analysis

Key Analysis Components:
    1. PCA Decomposition: Identify principal components and feature contributions
    2. Knee Detection: Automatically determine optimal DBSCAN epsilon parameter
    3. DBSCAN Clustering: Group features based on PCA contribution similarity
    4. Feature Selection: Choose representative features from each cluster
    5. Statistical Analysis: Analyze clustering patterns and feature importance

Pipeline Architecture:
    Data Loading → PCA Analysis → Knee Detection → DBSCAN Clustering → Feature Selection → Visualization → Export

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
    - Selected feature subsets for each drug treatment
    - PCA contribution matrices and heatmaps
    - Clustering analysis results
    - Knee detection curves and optimal parameters
    - Feature importance rankings and visualizations

Command Line Usage:
    python main.py --data-path /path/to/features --output-path ./pfa_analysis
                   --explained-var 0.95 --min-samples 2

Example Workflow:
    # Standard analysis with default parameters
    python main.py

    # Custom parameter analysis
    python main.py --explained-var 0.90 --min-samples 3
                   --output-path ./custom_pfa_analysis

Scientific Applications:
    - Feature selection for machine learning models
    - Biomarker identification for cardiac conditions
    - Understanding feature redundancy and relationships
    - Optimization of measurement protocols
    - Therapeutic target identification

Authors: Cardiac Electrophysiology Research Team
Version: 1.0
Dependencies: pandas, numpy, scikit-learn, kneed, matplotlib, seaborn
"""

import sys
import argparse
from pathlib import Path
import logging
import json
import pandas as pd

# Add parent directory to Python path to import utils.py
sys.path.append(str(Path(__file__).parent.parent))

from pfa_analyzer import PfaAnalyzer
from pfa_visualizer import PfaVisualizer
from utils import SignalData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    """
    Execute the complete PFA analysis pipeline.

    This function orchestrates a comprehensive PFA analysis workflow for identifying
    representative features from cardiac signal data. The pipeline integrates data
    loading, PCA analysis, knee detection, DBSCAN clustering, and publication-quality
    visualization to provide deep insights into feature relationships and redundancy.

    Analysis Workflow:
    1. Initialization Phase:
       - Configure output directories and logging
       - Initialize analysis classes with specified parameters
       - Validate input paths and create directory structure

    2. Data Loading Phase:
       - Read HDF5 feature files for all drug treatments
       - Validate data consistency and completeness
       - Merge cases by drug type and baseline conditions

    3. PFA Analysis Phase:
       - Perform PCA decomposition for each drug treatment
       - Determine optimal DBSCAN parameters using knee detection
       - Apply DBSCAN clustering to group similar features
       - Select representative features from each cluster

    4. Visualization Generation Phase:
       - Create PCA contribution heatmaps with cluster highlighting
       - Generate knee detection curves for parameter optimization
       - Produce explained variance plots with cumulative thresholds
       - Save all visualizations in publication-ready format

    5. Output Organization Phase:
       - Structure results in organized directory hierarchy
       - Export selected feature subsets and analysis results
       - Save high-resolution PDF plots for publication

    Key Statistical Outputs:
        For each treatment condition:
        - Selected feature subsets with redundancy reduction
        - PCA contribution matrices and clustering results
        - Optimal DBSCAN parameters and knee detection curves
        - Feature importance rankings and cluster summaries

    Visualization Outputs:
        - PCA cluster heatmaps showing feature contributions
        - Knee detection curves for epsilon parameter optimization
        - Explained variance plots with cumulative thresholds
        - All plots formatted for LaTeX thesis integration

    Args:
        args (argparse.Namespace): Command line arguments containing:
            - data_path (str): Path to directory with HDF5 feature files
            - output_path (str): Directory for saving analysis results
            - explained_var (float): Target explained variance for PCA
            - min_samples (int): DBSCAN min_samples parameter
            - eps (Optional[float]): DBSCAN epsilon parameter (auto-determined if None)

    Directory Structure Created:
        output_path/
        ├── data/
        │   ├── selected_features_{drug}.json
        │   ├── analysis_summary_{drug}.json
        │   └── pca_components_{drug}.csv
        └── plots/
            └── drugs/{drug}/
                ├── pfa_dbscan_cluster_report_{drug}.pdf
                ├── pfa_dbscan_kneedle_curve_{drug}.pdf
                └── pfa_dbscan_explained_variance_{drug}.pdf

    Example Usage:
        >>> import argparse
        >>> args = argparse.Namespace(
        ...     data_path='/path/to/features',
        ...     output_path='./pfa_results',
        ...     explained_var=0.95,
        ...     min_samples=2,
        ...     eps=None
        ... )
        >>> main(args)
        # Executes complete PFA analysis pipeline with specified parameters

    Console Output:
        The function provides detailed progress reporting including:
        - Configuration summary with all parameters
        - File discovery and data loading progress
        - PCA analysis progress and component selection
        - Clustering analysis and feature selection status
        - Visualization generation status
        - Final output file locations

    Note:
        The pipeline is designed for reproducibility with standardized output
        formats. All intermediate results are saved to enable post-hoc analysis
        and validation.
    """
    print("\n" + "=" * 80)
    print("Starting Principal Feature Analysis (PFA) Pipeline")
    print("=" * 80)

    print("\nInitializing...")
    print(f"Data path: {args.data_path}")
    print(f"Output path: {args.output_path}")
    print(f"Explained variance: {args.explained_var}")
    print(f"DBSCAN min_samples: {args.min_samples}")
    print(f"DBSCAN epsilon: {args.eps} (auto-determined if None)")

    # Initialize classes
    output_path = Path(args.output_path)
    signal_data = SignalData(data_path=args.data_path)
    pfa_analyzer = PfaAnalyzer(
        explained_var=args.explained_var, min_samples=args.min_samples, eps=args.eps
    )
    visualizer = PfaVisualizer(output_path=str(output_path / "plots"))

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

    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "data").mkdir(exist_ok=True)
    (output_path / "plots" / "drugs").mkdir(parents=True, exist_ok=True)

    # Analyze each drug treatment
    all_results = {}

    for drug_df, drug_name in zip(drug_dfs, drug_names):
        print(f"\nAnalyzing {drug_name}...")

        # Prepare feature data (exclude metadata columns)
        feature_cols = [
            col
            for col in drug_df.columns
            if col not in ["drug", "bct_id", "concentration[um]", "frequency[Hz]"]
        ]
        X = drug_df[feature_cols]

        print(f"- Original features: {len(feature_cols)}")

        # Perform PFA analysis
        print("- Running PFA analysis...")
        selected_features = pfa_analyzer.fit_transform(X)

        # Get analysis summary
        analysis_summary = pfa_analyzer.get_analysis_summary()
        print(f"- Selected features: {len(analysis_summary['selected_feature_names'])}")
        print(f"- PCA components: {analysis_summary['n_pca_components']}")
        print(f"- Clusters found: {analysis_summary['n_clusters']}")

        # Save results
        print("- Saving analysis results...")

        # Save selected features
        selected_features_file = (
            output_path / "data" / f"selected_features_{drug_name}.json"
        )
        with open(selected_features_file, "w") as f:
            json.dump(analysis_summary["selected_feature_names"], f, indent=2)
        print(f"  Saved selected features to: {selected_features_file}")

        # Save analysis summary
        summary_file = output_path / "data" / f"analysis_summary_{drug_name}.json"
        with open(summary_file, "w") as f:
            json.dump(analysis_summary, f, indent=2, default=str)
        print(f"  Saved analysis summary to: {summary_file}")

        # Save PCA components
        pca_components_file = output_path / "data" / f"pca_components_{drug_name}.csv"
        pca_components_df = pd.DataFrame(
            pfa_analyzer._pca_components,
            index=feature_cols,
            columns=[f"PC{i+1}" for i in range(pfa_analyzer.q)],
        )
        pca_components_df.to_csv(pca_components_file)
        print(f"  Saved PCA components to: {pca_components_file}")

        # Create visualizations
        print("- Generating visualizations...")
        drug_plot_path = output_path / "plots" / "drugs" / drug_name
        drug_plot_path.mkdir(parents=True, exist_ok=True)

        # Update visualizer output path for this drug
        drug_visualizer = PfaVisualizer(output_path=str(drug_plot_path))

        # Create comprehensive visualization
        visualization_results = drug_visualizer.create_comprehensive_visualization(
            pfa_analyzer=pfa_analyzer, drug_name=drug_name, save_plots=True
        )

        print(f"  Saved visualizations to: {drug_plot_path}")

        # Store results
        all_results[drug_name] = {
            "analysis_summary": analysis_summary,
            "selected_features": selected_features,
            "visualization_results": visualization_results,
        }

        # Print detailed summary
        print(f"\n{drug_name.upper()} Analysis Summary:")
        print(f"- Original features: {analysis_summary['n_original_features']}")
        print(f"- Selected features: {analysis_summary['n_selected_features']}")
        print(
            f"- Reduction: {((1 - analysis_summary['n_selected_features'] / analysis_summary['n_original_features']) * 100):.1f}%"
        )
        print(f"- PCA components: {analysis_summary['n_pca_components']}")
        print(f"- Clusters: {analysis_summary['n_clusters']}")
        print(f"- Independent features: {analysis_summary['n_independent_features']}")

        if analysis_summary["cluster_summary"]:
            print("- Cluster details:")
            for cluster_id, cluster_info in analysis_summary["cluster_summary"].items():
                print(
                    f"  Cluster {cluster_id}: {cluster_info['size']} features, "
                    f"representative: {cluster_info['representative_feature']}"
                )

    # Create combined summary
    print("\n" + "=" * 80)
    print("PFA Analysis Pipeline Completed Successfully!")
    print("=" * 80)

    # Print overall summary
    print("\nOverall Summary:")
    for drug_name, results in all_results.items():
        summary = results["analysis_summary"]
        reduction = (
            1 - summary["n_selected_features"] / summary["n_original_features"]
        ) * 100
        print(
            f"- {drug_name}: {summary['n_original_features']} → {summary['n_selected_features']} "
            f"features ({reduction:.1f}% reduction)"
        )

    print(f"\nResults saved to: {output_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Perform Principal Feature Analysis (PFA) on cardiac signal features"
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
        default="./outputs/PfaAnalysis",
        help="Path where outputs (plots and data) will be saved",
    )
    parser.add_argument(
        "--explained-var",
        type=float,
        default=0.95,
        help="Target explained variance for PCA (default: 0.95)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="DBSCAN min_samples parameter (default: 2)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="DBSCAN epsilon parameter (auto-determined if None)",
    )

    args = parser.parse_args()
    main(args)
