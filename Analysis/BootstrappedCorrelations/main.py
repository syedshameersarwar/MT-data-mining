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
