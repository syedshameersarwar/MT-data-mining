"""
Relative Comparison Analysis Pipeline

This script provides a complete pipeline for relative comparison analysis of cardiac
signal features under different drug treatments. It performs three levels of analysis:
1. Tissue-specific relative changes
2. Global average changes
3. Target concentration comparisons (e.g., EC50/IC50)

Usage:
    python main.py --data-path /path/to/data --output-path /path/to/output [options]

Arguments:
    --data-path: Path to the features data directory
    --output-path: Path to save analysis results and plots
    --target-concentrations: Target concentrations for EC50/IC50 analysis (optional)
    --save-results: Whether to save analysis results to CSV files
    --save-plots: Whether to save plots to PDF files

Example:
    python main.py --data-path ../../data/features --output-path ./results --target-concentrations 0.03 1.0 1.0
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils import SignalData
from relative_comparison_analyzer import RelativeComparisonAnalyzer
from relative_comparison_visualizer import RelativeComparisonVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Relative Comparison Analysis Pipeline for Cardiac Signal Features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis with default settings
    python main.py --data-path ../../data/features --output-path ./results
    
    # Analysis with EC50/IC50 target concentrations
    python main.py --data-path ../../data/features --output-path ./results \\
                   --target-concentrations 0.03 1.0 1.0
    
    # Analysis without saving results
    python main.py --data-path ../../data/features --output-path ./results \\
                   --no-save-results --no-save-plots
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
        default="./outputs/RelativeComparison",
        help="Path where outputs (plots and data) will be saved",
    )

    parser.add_argument(
        "--target-concentrations",
        type=float,
        nargs=3,
        default=[0.03, 1.0, 1.0],
        metavar=("DRUG1_CONC", "DRUG2_CONC", "DRUG3_CONC"),
        help="Target concentrations for EC50/IC50 analysis [e4031, nifedipine, ca_titration]",
    )

    parser.add_argument(
        "--no-save-results",
        action="store_true",
        help="Do not save analysis results to CSV files",
    )

    parser.add_argument(
        "--no-save-plots", action="store_true", help="Do not save plots to PDF files"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def setup_output_directories(output_path: str):
    """Create output directories for results and plots."""
    output_dir = Path(output_path)

    # Create main output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for different plot types
    (output_dir / "drugs" / "e-4031").mkdir(parents=True, exist_ok=True)
    (output_dir / "drugs" / "nifedipine").mkdir(parents=True, exist_ok=True)
    (output_dir / "drugs" / "ca-titration").mkdir(parents=True, exist_ok=True)
    (output_dir / "global").mkdir(parents=True, exist_ok=True)

    logger.info(f"Created output directories in: {output_path}")


def load_and_preprocess_data(data_path: str):
    """Load and preprocess signal data."""
    logger.info(f"Loading data from: {data_path}")

    # Initialize signal data object
    signal_data = SignalData(data_path)

    # Read all cases for each drug
    e4031_cases = signal_data.read_all_cases("run1b_e-4031")
    nifedipine_cases = signal_data.read_all_cases("run1b_nifedipine")
    ca_titration_cases = signal_data.read_all_cases("run1b_ca_titration")

    # Merge cases by drug and baseline
    cases_dict = {
        "e-4031": e4031_cases,
        "nifedipine": nifedipine_cases,
        "ca_titration": ca_titration_cases,
    }

    signal_data.merge_cases_by_drug_and_baseline(cases_dict)

    logger.info("Data loading and preprocessing completed")
    logger.info(f"Loaded {len(signal_data.baseline_cases)} baseline cases")
    logger.info(f"Loaded {len(signal_data.e4031_cases)} E-4031 cases")
    logger.info(f"Loaded {len(signal_data.nifedipine_cases)} Nifedipine cases")
    logger.info(f"Loaded {len(signal_data.ca_titration_cases)} Ca²⁺ titration cases")

    return signal_data


def run_analysis(signal_data: SignalData, target_concentrations: list = None):
    """Run the relative comparison analysis."""
    logger.info("Starting relative comparison analysis...")

    # Initialize analyzer
    analyzer = RelativeComparisonAnalyzer(signal_data)

    # Run comprehensive analysis
    analysis_results = analyzer.run_comprehensive_analysis(target_concentrations)

    # Get analysis summary
    summary = analyzer.get_analysis_summary()

    logger.info("Analysis completed successfully")
    logger.info(f"Analysis summary: {summary}")

    return analyzer, analysis_results


def create_visualizations(
    analysis_results: dict, output_path: str, save_plots: bool = True
):
    """Create visualizations for the analysis results."""
    logger.info("Creating visualizations...")

    # Initialize visualizer
    visualizer = RelativeComparisonVisualizer(output_path)

    # Create comprehensive visualizations
    visualization_results = visualizer.create_comprehensive_visualization(
        analysis_results, save_plots=save_plots
    )

    logger.info("Visualization creation completed")

    return visualizer, visualization_results


def save_analysis_results(
    analyzer: RelativeComparisonAnalyzer, output_path: str, save_results: bool = True
):
    """Save analysis results to files."""
    if not save_results:
        logger.info("Skipping results saving as requested")
        return {}

    logger.info("Saving analysis results...")

    # Save results to CSV files
    saved_files = analyzer.save_results(output_path)

    logger.info(f"Analysis results saved to: {output_path}")

    return saved_files


def print_analysis_summary(analysis_results: dict, saved_files: dict):
    """Print a summary of the analysis results."""
    print("\n" + "=" * 80)
    print("RELATIVE COMPARISON ANALYSIS SUMMARY")
    print("=" * 80)

    # Analysis metadata
    metadata = analysis_results.get("metadata", {})
    print(f"Total features analyzed: {metadata.get('total_features', 'N/A')}")
    print(f"Drugs analyzed: {', '.join(metadata.get('drug_names', []))}")
    print(f"Analysis timestamp: {metadata.get('analysis_timestamp', 'N/A')}")

    # Analysis levels
    print("\nAnalysis Levels:")
    for level_info in metadata.get("analysis_levels", []):
        level = level_info.get("level", "Unknown")
        comparisons = level_info.get("total_comparisons", 0)
        print(f"  - {level}: {comparisons} comparisons")

        if "unique_tissues" in level_info:
            print(f"    Tissues: {level_info['unique_tissues']}")
        if "unique_concentrations" in level_info:
            print(f"    Concentrations: {level_info['unique_concentrations']}")
        if "target_concentrations" in level_info:
            print(f"    Target concentrations: {level_info['target_concentrations']}")

    # Saved files
    if saved_files:
        print(f"\nSaved Files:")
        for result_type, filepath in saved_files.items():
            print(f"  - {result_type}: {filepath}")

    print("=" * 80)


def main(args):
    """Main pipeline function."""
    logger.info("Starting Relative Comparison Analysis Pipeline")

    try:
        # Setup output directories
        setup_output_directories(args.output_path)

        # Load and preprocess data
        signal_data = load_and_preprocess_data(args.data_path)

        # Run analysis
        analyzer, analysis_results = run_analysis(
            signal_data, args.target_concentrations
        )

        # Save analysis results
        saved_files = save_analysis_results(
            analyzer, args.output_path, not args.no_save_results
        )

        # Create visualizations
        visualizer, visualization_results = create_visualizations(
            analysis_results, args.output_path, not args.no_save_plots
        )

        # Print summary
        print_analysis_summary(analysis_results, saved_files)

        logger.info("Relative Comparison Analysis Pipeline completed successfully")

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run pipeline
    exit_code = main(args)
    sys.exit(exit_code)
