"""
Regression Analysis Pipeline for Cardiac Signal Feature Analysis

This script provides a complete pipeline for performing GLMM Lasso regression analysis
on cardiac signal features. It orchestrates data loading, preprocessing, R script
execution, and visualization generation in a modular and reproducible manner.

Scientific Background:
    This pipeline performs feature-wise regression analysis to understand:
    - Feature interdependencies and relationships
    - Drug-specific changes in feature correlations
    - Tissue and concentration level random effects
    - Sparse feature selection through Lasso shrinkage

Pipeline Architecture:
    Data Loading → Preprocessing → R Script Execution → Coefficient Analysis → Visualization

Key Components:
    1. SignalData: Data loading and preprocessing from utils.py
    2. RegressionAnalyzer: Core analysis logic and R script execution
    3. RegressionVisualizer: Publication-quality plot generation
    4. Main Pipeline: Orchestration and result management

Usage Examples:
    # Run complete analysis
    python main.py --data-path /path/to/data --output-path ./results

    # Run analysis for specific drugs
    python main.py --drugs baseline e4031 nifedipine

    # Run analysis without visualization
    python main.py --no-visualization

Dependencies:
    - Python: pandas, numpy, matplotlib, seaborn
    - R: glmmLasso, MASS, nlme, lme4, lmerTest
    - Data: HDF5 files with cardiac signal features

Authors: Cardiac Electrophysiology Research Team
Version: 1.0
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import List, Optional

# Import our modules
sys.path.append(str(Path(__file__).parent.parent))
from utils import SignalData
from regression_analyzer import RegressionAnalyzer
from regression_visualizer import RegressionVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("regression_analysis.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GLMM Lasso Regression Analysis Pipeline for Cardiac Signal Features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run complete analysis with default settings
    python main.py --data-path /path/to/data
    
    # Run analysis for specific drugs only
    python main.py --drugs baseline e4031 nifedipine
    
    # Run analysis without visualization
    python main.py --no-visualization
    
    # Run with custom output paths
    python main.py --data-path /path/to/data --output-path ./results --plots-path ./plots
        """,
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
        default="./outputs/RegressionAnalysis/data",
        help="Path where data will be saved (default: ./outputs/RegressionAnalysis/data)",
    )

    parser.add_argument(
        "--plots-path",
        type=str,
        default="./outputs/RegressionAnalysis/plots",
        help="Path for saving plots (default: ./outputs/RegressionAnalysis/plots)",
    )

    parser.add_argument(
        "--drugs",
        nargs="+",
        choices=["baseline", "e4031", "nifedipine", "ca_titration"],
        default=["baseline", "e4031", "nifedipine", "ca_titration"],
        help="Drugs to analyze (default: all drugs)",
    )

    parser.add_argument(
        "--r-script-path",
        type=str,
        default="glmmLasso.R",
        help="Path to the R script for GLMM Lasso (default: glmmLasso.R)",
    )

    parser.add_argument(
        "--no-visualization", action="store_true", help="Skip visualization generation"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def setup_directories(output_path: str, plots_path: str):
    """Create necessary directories for output."""
    Path(output_path).mkdir(parents=True, exist_ok=True)
    Path(plots_path).mkdir(parents=True, exist_ok=True)

    logger.info(f"Created output directories: {output_path}, {plots_path}")


def load_and_preprocess_data(data_path: str) -> SignalData:
    """
    Load and preprocess signal data using utils.SignalData.

    Args:
        data_path (str): Path to the data directory

    Returns:
        SignalData: Processed signal data object
    """
    logger.info(f"Loading and preprocessing data from {data_path}...")

    try:
        # Initialize SignalData
        signal_data = SignalData(data_path)

        # Read all cases for each drug
        cases_dict = {}
        drugs = ["e-4031", "nifedipine", "ca_titration"]

        for drug in drugs:
            cases = signal_data.read_all_cases(f"run1b_{drug}")
            cases_dict[drug] = cases
            logger.info(f"Found {len(cases)} cases for {drug}")

        # Merge cases by drug and baseline
        signal_data.merge_cases_by_drug_and_baseline(
            cases_dict, discard_concentrations=True, include_baseline=True
        )

        logger.info("Data loading and preprocessing completed successfully")
        logger.info(f"Features available: {len(signal_data.features)}")

        return signal_data

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def run_regression_analysis(
    signal_data: SignalData, drugs: List[str], r_script_path: str, output_path: str
) -> dict:
    """
    Run regression analysis for specified drugs.

    The R script runs for all drugs sequentially, so this method prepares data
    for all drugs and then executes the R script once.

    Args:
        signal_data (SignalData): Processed signal data
        drugs (List[str]): List of drugs to analyze
        r_script_path (str): Path to R script
        output_path (str): Output directory

    Returns:
        dict: Analysis results
    """
    logger.info(f"Starting regression analysis for drugs: {drugs}")

    try:
        # Initialize analyzer
        analyzer = RegressionAnalyzer(
            signal_data=signal_data,
            r_script_path=r_script_path,
            output_path=output_path,
        )

        # Run analysis (R script will process all drugs sequentially)
        results = analyzer.run_regression_analysis(drugs)

        # Get analysis summary
        summary = analyzer.get_analysis_summary()
        logger.info(f"Analysis summary: {summary}")

        # Save results
        saved_files = analyzer.save_analysis_results()
        logger.info(f"Saved analysis results: {list(saved_files.keys())}")

        return results

    except Exception as e:
        logger.error(f"Error in regression analysis: {e}")
        raise


def create_visualizations(analysis_results: dict, plots_path: str) -> dict:
    """
    Create visualizations for regression analysis results.

    Args:
        analysis_results (dict): Results from regression analysis
        plots_path (str): Path for saving plots

    Returns:
        dict: Visualization results
    """
    logger.info("Creating regression analysis visualizations...")

    try:
        # Initialize visualizer
        visualizer = RegressionVisualizer(output_path=plots_path)

        # Create comprehensive visualization suite
        visualization_results = visualizer.create_comprehensive_visualization(
            analysis_results=analysis_results, save_plots=True
        )

        logger.info("Visualization creation completed successfully")

        return visualization_results

    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        raise


def generate_analysis_report(
    analysis_results: dict, visualization_results: dict, output_path: str
):
    """
    Generate a comprehensive analysis report.

    Args:
        analysis_results (dict): Results from regression analysis
        visualization_results (dict): Results from visualization
        output_path (str): Output directory
    """
    logger.info("Generating analysis report...")

    try:
        report_path = Path(output_path) / "regression_analysis_report.txt"

        with open(report_path, "w") as f:
            f.write("GLMM Lasso Regression Analysis Report\n")
            f.write("=" * 50 + "\n\n")

            # Analysis summary
            f.write("ANALYSIS SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total drugs analyzed: {len(analysis_results)}\n")
            f.write(f"Drugs: {list(analysis_results.keys())}\n\n")

            # Results for each drug
            for drug, results in analysis_results.items():
                f.write(f"{drug.upper()} ANALYSIS\n")
                f.write("-" * 15 + "\n")

                if "data" in results:
                    f.write(f"Samples: {len(results['data'])}\n")
                    f.write(f"Tissues: {results['data']['bct_id'].nunique()}\n")
                    f.write(
                        f"Concentrations: {results['data']['concentration.um.'].nunique()}\n"
                    )

                if (
                    "coefficient_matrix" in results
                    and not results["coefficient_matrix"].empty
                ):
                    matrix = results["coefficient_matrix"]
                    f.write(f"Matrix shape: {matrix.shape}\n")
                    f.write(f"Non-zero coefficients: {(matrix != 0).sum().sum()}\n")

                f.write("\n")

            # Visualization summary
            if visualization_results:
                f.write("VISUALIZATION SUMMARY\n")
                f.write("-" * 25 + "\n")
                f.write(f"Total plots created: {len(visualization_results)}\n")

                if "analysis_summary" in visualization_results:
                    summary = visualization_results["analysis_summary"]
                    f.write(
                        f"Drugs visualized: {summary.get('drugs_visualized', [])}\n"
                    )
                    f.write(f"Plot types: {summary.get('plot_types', [])}\n")

                f.write("\n")

            # File paths
            f.write("OUTPUT FILES\n")
            f.write("-" * 15 + "\n")
            for drug, results in analysis_results.items():
                if "csv_path" in results:
                    f.write(f"{drug} data: {results['csv_path']}\n")
                if "matrix_path" in results:
                    f.write(f"{drug} coefficients: {results['matrix_path']}\n")

            f.write("\n")

        logger.info(f"Analysis report saved to: {report_path}")

    except Exception as e:
        logger.error(f"Error generating report: {e}")


def main():
    """Main pipeline execution function."""
    # Parse arguments
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting GLMM Lasso Regression Analysis Pipeline")
    logger.info(f"Arguments: {vars(args)}")

    try:
        # Setup directories
        setup_directories(args.output_path, args.plots_path)

        # Load and preprocess data
        signal_data = load_and_preprocess_data(args.data_path)

        # Run regression analysis
        # Convert r_script_path to absolute path if it's relative
        if not os.path.isabs(args.r_script_path):
            r_script_abs_path = os.path.join(
                os.path.dirname(__file__), args.r_script_path
            )
        else:
            r_script_abs_path = args.r_script_path

        analysis_results = run_regression_analysis(
            signal_data=signal_data,
            drugs=args.drugs,
            r_script_path=r_script_abs_path,
            output_path=args.output_path,
        )

        # Create visualizations (if requested)
        visualization_results = {}
        if not args.no_visualization:
            visualization_results = create_visualizations(
                analysis_results=analysis_results, plots_path=args.plots_path
            )

        # Generate analysis report
        generate_analysis_report(
            analysis_results=analysis_results,
            visualization_results=visualization_results,
            output_path=args.output_path,
        )

        logger.info("Regression analysis pipeline completed successfully!")

        # Print summary
        print("\n" + "=" * 60)
        print("REGRESSION ANALYSIS PIPELINE COMPLETED")
        print("=" * 60)
        print(f"Drugs analyzed: {list(analysis_results.keys())}")
        print(f"Results saved to: {args.output_path}")
        if not args.no_visualization:
            print(f"Plots saved to: {args.plots_path}")
        print(f"Report saved to: {args.output_path}/regression_analysis_report.txt")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
