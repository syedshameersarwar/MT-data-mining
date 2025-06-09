import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

# Add parent directory to Python path to import utils.py
sys.path.append(str(Path(__file__).parent.parent))
from utils import SignalData
from statistical_modeling import StatisticalAnalyzer
from plotting import DrugResponsePlotter


def setup_drug_parameters() -> Dict:
    """
    Setup drug-specific parameters including baseline and discarded concentrations.

    This function defines the experimental parameters for each drug treatment,
    including whether to use tissue-only random effects in mixed models and
    which features to analyze in subset analyses.

    Drug-specific configurations:
    - e-4031: hERG potassium channel blocker, uses tissue-only random effects
    - nifedipine: L-type calcium channel blocker, uses full random effects (tissue and nested concentration inside tissue)
    - ca_titration: Calcium concentration titration experiment

    Returns:
        Dict: Nested dictionary containing drug parameters with the following structure:
            {
                "drug_name": {
                    "tissue_random_only_effects": bool,
                    "feature_subsets": List[str]
                }
            }

    Example:
        >>> params = setup_drug_parameters()
        >>> params["e-4031"]["tissue_random_only_effects"]
        True
        >>> len(params["nifedipine"]["feature_subsets"])
        4
    """
    return {
        "e-4031": {
            "tissue_random_only_effects": True,
            "feature_subsets": [
                "duration",
                "force_peak_amplitude",
                "calc_peak_amplitude",
            ],
        },
        "nifedipine": {
            "tissue_random_only_effects": False,
            "feature_subsets": [
                "force_rise_time_0.2_0.8 s",
                "calc_rise_time_0.2_0.8 s",
                "force_decay_time_0.2_0.8 s",
                "calc_decay_time_0.2_0.8 s",
            ],
        },
        "ca_titration": {
            "tissue_random_only_effects": False,
            "feature_subsets": [
                "force_peak_amplitude",
            ],
        },
    }


def analyze_drug_significance(
    drug_df: pd.DataFrame,
    drug_params: Dict,
    signal_data: SignalData,
    drug_name: str,
    output_path: Path,
    feature_subsets: List[str] = [],
) -> None:
    """
    Analyze drug significance and create plots for a specific drug treatment.

    This function performs comprehensive statistical analysis including:
    1. Mixed effects modeling using lmer (R) and pymer4 (Python)
    2. Diagnostic plot generation (QQ plots and residual plots)
    3. Significance plot generation with post-hoc comparisons

    The analysis workflow:
    - Initialize statistical analyzer with drug-specific parameters
    - Run mixed effects models for selected features
    - Generate diagnostic plots to assess model assumptions
    - Create significance plots showing treatment effects
    - Save all outputs in organized directory structure

    Args:
        drug_df (pd.DataFrame): DataFrame containing drug measurement data with columns:
            - bct_id: Tissue/batch identifier
            - concentration[um]: Drug concentration in micromoles
            - Various feature columns (force, calcium, field potential metrics)
        drug_params (Dict): Drug-specific configuration parameters including:
            - tissue_random_only_effects: Whether to use simplified random effects
            - feature_subsets: List of features for subset analysis
        signal_data (SignalData): SignalData instance containing processed data
        drug_name (str): Name of the drug being analyzed (e.g., "e-4031", "nifedipine")
        output_path (Path): Base directory path for saving analysis outputs
        feature_subsets (List[str], optional): Specific features to analyze.
            If empty, analyzes all available features. Defaults to [].

    Returns:
        None: Function saves outputs to disk but returns nothing

    Side Effects:
        - Creates drug-specific output directory structure
        - Saves diagnostic plots (QQ and residual plots) as PDF files
        - Saves significance plots as PDF files
        - Saves frequency tables as CSV files
        - Prints progress information to console

    Directory Structure Created:
        output_path/
        └── {drug_name}/
            ├── data/
            │   └── frequency_table.csv
            └── plots/
                ├── qq_plots_lmer_{n_features}.pdf
                ├── residual_lmer_{n_features}.pdf
                └── significance_lmer_{n_features}.pdf

    Example:
        >>> drug_params = setup_drug_parameters()["e-4031"]
        >>> analyze_drug_significance(
        ...     drug_df=e4031_data,
        ...     drug_params=drug_params,
        ...     signal_data=signal_handler,
        ...     drug_name="e-4031",
        ...     output_path=Path("./results"),
        ...     feature_subsets=["duration", "force_peak_amplitude"]
        ... )
    """
    print(f"\nAnalyzing {drug_name}...")

    # Setup output directories
    drug_output_path = output_path / drug_name
    drug_output_path.mkdir(parents=True, exist_ok=True)

    # Initialize analyzers
    statistical_analyzer = StatisticalAnalyzer(
        output_path=drug_output_path,
        selected_concentrations=getattr(
            signal_data, f"{drug_name.replace('-', '')}_selected_concentrations"
        ),
        tissue_random_only_effects=drug_params["tissue_random_only_effects"],
    )

    # # Run analysis for all features
    if len(feature_subsets) == 0:
        print("- Running full feature analysis...")
    else:
        print(f"- Analyzing feature subset ({len(feature_subsets)} features)...")
    features = feature_subsets if len(feature_subsets) > 0 else drug_df.columns

    print("- Generating diagnostic plots...")

    # Analyze feature subsets if provided
    print(f"- Analyzing feature subset ({len(features)} features)...")
    subset_results, _ = statistical_analyzer.analyze_drug_significance(
        drug_df=drug_df,
        drug_name=drug_name,
        feature_subset=features,
    )

    subset_plotter = DrugResponsePlotter(
        drug_df=drug_df,
        statistical_results=subset_results,
        selected_concentrations=getattr(
            signal_data, f"{drug_name.replace('-', '')}_selected_concentrations"
        ),
    )

    # Generate subset plots
    qq_fig, resid_fig = subset_plotter.create_diagnostic_plots()
    (drug_output_path / "plots").mkdir(parents=True, exist_ok=True)
    qq_fig.savefig(
        drug_output_path / f"plots/qq_plots_lmer_{len(features)}.pdf",
        bbox_inches="tight",
    )
    resid_fig.savefig(
        drug_output_path / f"plots/residual_lmer_{len(features)}.pdf",
        bbox_inches="tight",
    )

    sig_fig = subset_plotter.create_significance_plot(title_prefix="Mixed Model")
    sig_fig.savefig(
        drug_output_path / f"plots/significance_lmer_{len(features)}.pdf",
        bbox_inches="tight",
    )


def main(args):
    """
    Main function to run the complete drug response significance analysis pipeline.

    This function orchestrates the entire analysis workflow:
    1. Data loading and preprocessing
    2. Statistical analysis for each drug treatment
    3. Plot generation and output organization

    The pipeline processes three drug treatments:
    - E-4031: hERG K+ channel blocker (cardiac repolarization)
    - Nifedipine: L-type Ca2+ channel blocker (cardiac contraction)
    - Ca2+ titration: Extracellular calcium concentration effects

    Analysis includes:
    - Linear mixed effects modeling with tissue-specific random effects
    - ANOVA analysis with Dunnett's post-hoc tests
    - Model diagnostic assessment (normality, homoscedasticity)
    - Comprehensive visualization of treatment effects

    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - data_path (str): Path to directory containing HDF5 feature files
            - output_path (str): Path for saving analysis results and plots

    Returns:
        None: Function performs analysis and saves results to disk

    Side Effects:
        - Reads multiple HDF5 files from data_path
        - Creates comprehensive output directory structure
        - Saves statistical analysis results and plots
        - Prints detailed progress information

    Console Output:
        - Pipeline initialization status
        - Data loading progress and file counts
        - Analysis progress for each drug
        - Completion confirmation

    Example Usage:
        From command line:
        >>> python main.py --data-path /path/to/data --output-path /path/to/results

        Programmatically:
        >>> import argparse
        >>> args = argparse.Namespace(
        ...     data_path="/path/to/features",
        ...     output_path="./analysis_results"
        ... )
        >>> main(args)
    """
    print("\n" + "=" * 80)
    print("Starting Drug Response Significance Analysis")
    print("=" * 80)

    # Setup paths
    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nInput path: {data_path}")
    print(f"Output path: {output_path}")

    # Initialize data handler
    print("\nInitializing data handler...")
    signal_data = SignalData(data_path=str(data_path))

    # Read and process data for each drug
    print("\nReading data files...")
    cases_dict = {
        "e-4031": signal_data.read_all_cases("run1b_e-4031"),
        "nifedipine": signal_data.read_all_cases("run1b_nifedipine"),
        "ca_titration": signal_data.read_all_cases("run1b_ca_titration"),
    }
    treatment_config = setup_drug_parameters()
    # Merge cases
    print("\nMerging cases...")
    signal_data.merge_cases_by_drug_and_baseline(cases_dict, include_baseline=True)

    # Analyze each drug
    for drug_name in ["e-4031", "nifedipine", "ca_titration"]:
        drug_df = getattr(signal_data, f"{drug_name.replace('-', '')}_cases")
        drug_params = treatment_config[drug_name]
        analyze_drug_significance(
            drug_df=drug_df,
            drug_params=drug_params,
            signal_data=signal_data,
            drug_name=drug_name,
            output_path=output_path,
            feature_subsets=drug_params["feature_subsets"],
        )

    print("\n" + "=" * 80)
    print("Drug Response Significance Analysis Completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    """
    Command-line interface for drug response significance analysis.

    This script provides a command-line interface for analyzing cardiac drug responses
    using mixed effects modeling and statistical testing. The analysis pipeline
    processes multimodal cardiac measurements (electrical, calcium, mechanical) and
    evaluates the statistical significance of drug effects across different concentrations.

    Required Arguments:
        --data-path: Path to directory containing HDF5 feature files

    Optional Arguments:
        --output-path: Directory for saving results (default: ./outputs/SignificanceAnalysis)

    Example Usage:
        # Basic usage with default output path
        python main.py --data-path /path/to/features

        # With custom output path
        python main.py --data-path /path/to/features --output-path /custom/output

    Output Structure:
        output_path/
        ├── e-4031/
        │   ├── data/frequency_table.csv
        │   └── plots/
        │       ├── qq_plots_lmer_*.pdf
        │       ├── residual_lmer_*.pdf
        │       └── significance_lmer_*.pdf
        ├── nifedipine/
        │   └── [similar structure]
        └── ca_titration/
            └── [similar structure]
    """
    parser = argparse.ArgumentParser(
        description="Analyze drug response significance and generate plots"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/home/syedshameersarwar/Desktop/Work/MyOfarm/Thesis/Experiments/Mea-Peak-Clustering/Analysis/FeatureGMM/Merge/Data/Features",
        help="Path to the directory containing feature data files",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./outputs/SignificanceAnalysis",
        help="Path where outputs (plots and data) will be saved",
    )

    args = parser.parse_args()
    main(args)
