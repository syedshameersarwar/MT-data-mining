import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import shutil
import matplotlib.pyplot as plt
import hillfit
import argparse


sys.path.append(str(Path(__file__).parent.parent))
from utils import SignalData, FeatureMapping


class HillFitting:
    def __init__(
        self,
        drug1_df: pd.DataFrame,
        drug2_df: pd.DataFrame,
        features_subset: list = None,
    ):
        self.drug1_df = drug1_df
        self.drug2_df = drug2_df
        self.features_subset = features_subset

    @staticmethod
    def normalize_values(values: np.ndarray) -> np.ndarray:
        """
        Normalize values between 0 and 1.

        Args:
            values: Array of values to normalize

        Returns:
            Normalized values between 0 and 1
        """
        min_val = np.min(values)
        max_val = np.max(values)
        return (values - min_val) / (max_val - min_val)

    def preprocess_data(
        self, drug1_df: pd.DataFrame, drug2_df: pd.DataFrame, features: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess drug response data by normalizing features.

        Args:
            drug1_df: DataFrame for first drug
            drug2_df: DataFrame for second drug
            features: List of features to analyze

        Returns:
            Tuple of preprocessed DataFrames for both drugs
        """
        drug1_df = drug1_df.copy()
        drug2_df = drug2_df.copy()

        for feature in features:
            drug1_df[feature] = self.normalize_values(drug1_df[feature])
            drug2_df[feature] = self.normalize_values(drug2_df[feature])

        return drug1_df, drug2_df

    @staticmethod
    def process_concentrations(df: pd.DataFrame) -> Tuple[np.ndarray, Dict, List]:
        """
        Process concentration values, handling zero concentrations.
        If set of concentrations contain zero, we replace it with a small value ~ (smallest non-zero concentration / 10)
        otherwise the mapping is the same as the original concentrations

        Args:
            df: DataFrame containing concentration data

        Returns:
            Tuple containing:
            - Modified concentration array
            - Mapping of original to modified concentrations
            - List of modified concentration values
        """
        unique_conc = sorted(df["concentration[um]"].unique())
        x = df["concentration[um]"].values

        if float(0) in unique_conc:
            # minimum non-zero concentration
            min_nonzero = min([c for c in unique_conc if c > 0])
            # small value ~ (smallest non-zero concentration / 10)
            small_value = min_nonzero / 10
            # replace zero with small value
            new_conc = [c if c != 0 else small_value for c in unique_conc]
            # create mapping of original to modified concentrations
            conc_mapping = dict(zip(unique_conc, new_conc))
            # replace original concentrations with modified concentrations
            x = np.array([conc_mapping[c] for c in x])
            return x, conc_mapping, new_conc
        # if no zero concentrations, return original concentrations
        return x, dict(zip(unique_conc, unique_conc)), unique_conc

    def check_sigmoid_direction_by_mean_slope(
        self, df: pd.DataFrame, feature: str
    ) -> Tuple[str, float]:
        """Check sigmoid direction using all data points across tissues.
           It takes a mean of the feature across tissues at each concentration.
        Args:
            df: DataFrame containing concentration-response data
            feature: Feature name

        Returns:
            Tuple containing:
            - Metric name (ec50 or ic50)
            - Slope sign (1 for positive, -1 for negative, 0 for no slope)
        """
        mean_df = df.groupby("concentration[um]", as_index=False)[feature].mean()
        mean_values = mean_df[feature].values
        slopes = np.diff(mean_values)
        average_slope = np.mean(slopes)
        if average_slope > 0:
            return "ec50", 1
        elif average_slope < 0:
            return "ic50", -1
        else:
            return "n/a", 0

    def fit_hill_equation(
        self, x: np.ndarray, y: np.ndarray, feature: str
    ) -> Optional[hillfit.HillFit]:
        """
        Fit Hill equation to concentration-response data.

        Args:
            x: Concentration values (will be log transformed)
            y: Response values
            feature: Feature name

        Returns:
            Fitted HillFit object or None if fitting fails
        """
        try:
            hf = hillfit.HillFit(x, y)
            hf.fitting(
                x_label="concentration[um]",
                y_label=feature,
                title=f"{feature}",
                log_x=True,
                generate_figure=False,
            )
            return hf
        except Exception as e:
            print(f"Failed to fit Hill equation: {str(e)}")
            return None

    def analyze_single_drug(self, df: pd.DataFrame, features: List[str]) -> Dict:
        """
        Analyze concentration-response relationships for a single drug.

        Args:
            df: Drug response DataFrame
            features: List of features to analyze
            drug_name: Name of the drug

        Returns:
            Dictionary containing analysis results
        """
        feature_results = {}

        for feature in features:
            # Get metric and sigmoid direction based on mean slope (ec50 or ic50)
            metric, sign = self.check_sigmoid_direction_by_mean_slope(df, feature)
            # Get mean and std of the feature across tissues at each concentration
            stats = (
                df.groupby("concentration[um]")[feature]
                .agg(["mean", "std"])
                .reset_index()
            )

            # Process concentrations (handling zero concentrations)
            x, conc_mapping, _ = self.process_concentrations(df)
            y = sign * df[feature].values

            # Update concentration values in stats
            stats["concentration[um]"] = stats["concentration[um]"].map(
                lambda c: conc_mapping.get(c, c)
            )

            # Fit Hill equation
            hf = self.fit_hill_equation(x, y, feature)

            if hf is not None:
                feature_results[feature] = {
                    "metric": metric.upper(),
                    "value": hf.ec50,
                    "R2": round(hf.r_2, 2),
                    "nH": round(hf.nH, 2),
                    "stats": stats,
                    "hill_fit": hf,
                    "sign": sign,
                }
            else:
                feature_results[feature] = {"error": "Failed to fit Hill equation"}

        return feature_results


class DrugResponseVisualizer:
    """Class to visualize drug concentration-response relationships."""

    def __init__(self):
        self.feature_colors = None

    def _set_figure_size(
        self,
        width: str = "thesis",
        width_fraction: float = 1.5,
        height_fraction: float = 0.6,
        subplots=(1, 1),
        golden_ratio=None,
    ) -> Tuple[float, float]:
        """Set figure dimensions to avoid scaling in LaTeX.
        Parameters
        ----------
        width: float or string
                Document width in points, or string of predined document type
        fraction: float, optional
                Fraction of the width which you wish the figure to occupy
        subplots: array-like, optional
                The number of rows and columns of subplots.
        Returns
        -------
        fig_dim: tuple
                Dimensions of figure in inches
        """
        if width == "thesis":
            width_pt = 430
            height_pt = 556
        elif width == "beamer":
            width_pt = 307.28987
            height_pt = 230
        else:
            width_pt = width
            height_pt = 230

        # Width of figure (in pts)
        inches_per_pt = 1 / 72.27
        fig_width_in = width_pt * inches_per_pt * width_fraction
        fig_height_in = height_pt * inches_per_pt * height_fraction
        # Convert from pt to inches

        # Adjust the golden ratio for better subplot proportions
        if golden_ratio is not None:
            golden_ratio = (5**0.5 - 1) / 2
            fig_height_in = (
                fig_width_in
                * golden_ratio
                * (subplots[0] / subplots[1])
                * height_fraction
            )

        return (fig_width_in, fig_height_in)

    def setup_figure(
        self, width_fraction: float = 1.5, height_fraction: float = 0.6
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Set up figure and axes for plotting.

        Returns:
            Tuple containing figure and list of axes
        """
        figsize = self._set_figure_size(
            width_fraction=width_fraction,
            height_fraction=height_fraction,
        )
        fig = plt.figure(figsize=figsize)

        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.2)
        axes = [fig.add_subplot(gs[i]) for i in range(2)]

        return fig, axes

    def setup_colors(self, n_features: int):
        """Set up color scheme for features."""
        self.feature_colors = plt.cm.Dark2(np.linspace(0, 1, n_features))

    def plot_single_drug(
        self,
        ax: plt.Axes,
        drug_results: Dict,
        drug_name: str,
        features: List[str],
        ax_idx: int,
    ):
        """
        Plot concentration-response relationships for a single drug.

        Args:
            ax: Matplotlib axes to plot on
            drug_results: Dictionary of drug analysis results
            drug_name: Name of the drug
            features: List of features to plot
            ax_idx: Index of the subplot
        """
        ec_ic_values = []

        # Basic axis setup
        self._setup_axis(ax, ax_idx)

        # Plot each feature
        for feature, color in zip(features, self.feature_colors):
            if feature not in drug_results or "error" in drug_results[feature]:
                continue

            result = drug_results[feature]
            self._plot_feature(ax, result, feature, color)
            ec_ic_values.append(result["hill_fit"].ec50)

        # Plot mean EC50/IC50
        if ec_ic_values:
            self._plot_mean_ec50(ax, ec_ic_values)

        # Customize plot
        self._customize_subplot(ax, drug_name, drug_results)

    def _setup_axis(self, ax: plt.Axes, ax_idx: int):
        """Set up basic axis properties."""
        ax.tick_params(axis="x", which="minor", bottom=False)
        if ax_idx == 0:
            ax.set_ylabel("Normalized Response", fontsize=12, labelpad=12)
        else:
            ax.set_ylabel("")

    def _plot_feature(
        self, ax: plt.Axes, result: Dict, feature: str, color: np.ndarray
    ):
        """Plot single feature data and fit."""
        stats = result["stats"]
        hf = result["hill_fit"]
        sign = result["sign"]

        # Plot data points
        ax.errorbar(
            stats["concentration[um]"],
            stats["mean"],
            yerr=stats["std"].fillna(0),
            fmt="o",
            alpha=0.8,
            capsize=5,
            color=color,
            ecolor=color,
            markersize=4,
            linewidth=1,
            label=f"{FeatureMapping.get_thesis_name(feature)}\n({result['metric']}={hf.ec50:.2f} µmol)",
        )

        # Plot fitted curve
        x_fit = np.logspace(
            np.log10(min(stats["concentration[um]"])),
            np.log10(max(stats["concentration[um]"])),
            100,
        )
        y_fit = [sign * eval(hf.equation, {"x": xi}) for xi in x_fit]
        ax.plot(x_fit, y_fit, "--", alpha=0.9, color=color, linewidth=1.5)

        # Plot EC50/IC50 line
        if hf.ec50 < max(x_fit):
            ax.axvline(x=hf.ec50, color=color, linestyle="--", alpha=0.5)

    def _plot_mean_ec50(self, ax: plt.Axes, ec_ic_values: List[float]):
        """Plot mean EC50/IC50 line."""
        mean_value = np.mean(ec_ic_values)
        ax.axvline(
            x=mean_value,
            color="red",
            linestyle="-",
            alpha=0.8,
            linewidth=1.4,
            label=f"Mean EC/IC50={mean_value:.2f} µmol",
        )

    def _customize_subplot(self, ax: plt.Axes, drug_name: str, drug_results: Dict):
        """Apply final customization to subplot."""
        ax.set_xscale("log")
        self._set_axis_limits(ax, drug_results)
        self._set_title(ax, drug_name)
        self._customize_ticks_and_grid(ax)
        self._add_legend(ax)

    def _set_axis_limits(self, ax: plt.Axes, drug_results: Dict):
        """
        Set appropriate axis limits based on concentration values.

        Args:
            ax: Matplotlib axes to customize
            drug_results: Dictionary containing drug analysis results
        """
        # Get any valid result to extract concentration information
        for result in drug_results.values():
            if "stats" in result:
                stats = result["stats"]
                conc_values = stats["concentration[um]"].values
                ax.set_xlim(min(conc_values) * 0.8, max(conc_values) * 1.2)
                ax.set_ylim(-0.1, 1.1)  # Normalized response limits

                # Set concentration ticks
                unique_conc = sorted(stats["concentration[um]"].unique())
                ax.set_xticks(unique_conc)
                ax.set_xticklabels(
                    [self.format_conc(c) for c in unique_conc], rotation=45
                )
                break

    def _set_title(self, ax: plt.Axes, drug_name: str):
        """
        Set subplot title with proper formatting.

        Args:
            ax: Matplotlib axes to customize
            drug_name: Name of the drug to display
        """
        # Convert internal drug name to display name
        title_mapping = {
            "drug1": "Nifedipine",
            "drug2": r"$\rm{Ca^{2+}}$" + " Titration",
            "nifedipine": "Nifedipine",
            "ca_titration": r"$\rm{Ca^{2+}}$" + " Titration",
            "e4031": "E-4031",
        }

        title = title_mapping.get(drug_name, drug_name)
        ax.set_title(title, fontsize=12, pad=12)

    def _customize_ticks_and_grid(self, ax: plt.Axes):
        """
        Customize tick labels and grid appearance.

        Args:
            ax: Matplotlib axes to customize
        """
        # Customize tick parameters
        ax.tick_params(axis="both", labelsize=9, which="major")
        ax.tick_params(axis="x", which="minor", bottom=False)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle="--", which="major")

    def _add_legend(self, ax: plt.Axes):
        """
        Add and customize legend for the subplot.

        Args:
            ax: Matplotlib axes to customize
        """
        ax.legend(
            bbox_to_anchor=(0.5, 1.28),  # Center above plot
            loc="center",
            borderaxespad=0,
            fontsize=7,
            edgecolor="black",
            ncol=2,  # Set to 2 columns
            columnspacing=1.0,  # Adjust space between columns
        )

    @staticmethod
    def finalize_figure(fig: plt.Figure):
        """Apply final adjustments to the figure."""
        fig.subplots_adjust(
            top=0.75,
            bottom=0.15,
            left=0.10,
            right=0.95,
            wspace=0.3,
        )
        fig.text(
            0.52, 0.02, "Concentration [µmol]", ha="center", va="center", fontsize=12
        )

    @staticmethod
    def format_conc(concentration: float) -> str:
        """
        Format concentration values for tick labels.

        Args:
            concentration: Concentration value to format

        Returns:
            Formatted concentration string
        """
        if concentration == 0:
            return "0"
        if concentration < 1:
            if concentration <= 0.001:
                return "0"
            if concentration <= 0.1:
                if concentration < 0.01:
                    return f"{concentration:.3f}"
                else:
                    if concentration == 0.1:
                        return "0.1"
                    else:
                        return f"{concentration:.2f}"
            else:
                return f"{concentration:.1f}"
        else:
            # check if there is a decimal part
            if concentration % 1 != 0:
                return f"{concentration:.1f}"
            else:
                return f"{int(concentration)}"


def compare_drug_responses(
    drug1_df: pd.DataFrame,
    drug2_df: pd.DataFrame,
    features: List[str],
    drug_names: List[str],
) -> Tuple[Dict, plt.Figure]:
    """
    Compare concentration-response relationships between two drugs.

    Args:
        drug1_df: DataFrame for first drug
        drug2_df: DataFrame for second drug
        features: List of features to analyze

    Returns:
        Tuple containing:
        - Dictionary of analysis results
        - Figure with plots
    """
    # Initialize analyzers
    analyzer = HillFitting(drug1_df, drug2_df, features)
    visualizer = DrugResponseVisualizer()

    # Preprocess data
    drug1_df, drug2_df = analyzer.preprocess_data(drug1_df, drug2_df, features)

    # Analyze drugs
    drug1_results = analyzer.analyze_single_drug(drug1_df, features)
    drug2_results = analyzer.analyze_single_drug(drug2_df, features)

    # Setup visualization
    fig, axes = visualizer.setup_figure()
    visualizer.setup_colors(len(features))

    # Plot results
    visualizer.plot_single_drug(axes[0], drug1_results, drug_names[0], features, 0)
    visualizer.plot_single_drug(axes[1], drug2_results, drug_names[1], features, 1)

    # Finalize figure
    visualizer.finalize_figure(fig)

    # Clean up temporary files
    cleanup_hillfit_files()

    return {"drug1": drug1_results, "drug2": drug2_results}, fig


def cleanup_hillfit_files():
    """Clean up temporary files created by HillFit."""
    for file in os.listdir():
        if file.startswith("Hillfit-reg"):
            shutil.rmtree(file, ignore_errors=True)


def setup_argument_parser():
    """
    Set up command line argument parser.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Analyze drug concentration-response relationships and generate Hill plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the directory containing feature data files",
        default="/home/syedshameersarwar/Desktop/Work/MyOfarm/Thesis/Experiments/Mea-Peak-Clustering/Analysis/FeatureGMM/Merge/Data/Features",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="./outputs/HillFitting",
        help="Path to store output files (plots and data)",
    )

    parser.add_argument(
        "--features",
        nargs="+",
        default=[
            "duration",
            "force_peak_amplitude",
            "calc_peak_amplitude",
            "local_frequency[Hz]",
        ],
        help="List of features to analyze and perform Hill fitting on",
    )

    parser.add_argument(
        "--drug-pair",
        choices=["nifedipine-ca", "nifedipine-e4031", "e4031-ca"],
        default="nifedipine-ca",
        help="Drug pair to compare",
    )

    return parser


def main():
    """
    Main function to run the drug response analysis pipeline.

    The pipeline consists of the following steps:
    1. Parse command line arguments
    2. Initialize SignalData with the provided data path
    3. Read and process drug response data
    4. Perform Hill fitting and generate comparison plots
    5. Save results to the specified output path
    """
    # Parse command line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("Drug Response Analysis Pipeline")
    print("=" * 80)

    print("\nConfiguration:")
    print(f"Data path: {args.data_path}")
    print(f"Output path: {args.output_path}")
    print(f"Features: {args.features}")
    print(f"Drug pair: {args.drug_pair}")

    # Initialize data handler
    print("\nInitializing data handler...")
    signal_data = SignalData(data_path=args.data_path)

    # Read data files
    print("\nReading data files...")
    cases_dict = {
        "e-4031": signal_data.read_all_cases("run1b_e-4031"),
        "nifedipine": signal_data.read_all_cases("run1b_nifedipine"),
        "ca_titration": signal_data.read_all_cases("run1b_ca_titration"),
    }

    # Print data summary
    print("\nFiles found:")
    for drug, files in cases_dict.items():
        print(f"- {drug}: {len(files)} files")

    # Process data
    print("\nProcessing data...")
    signal_data.merge_cases_by_drug_and_baseline(
        cases_dict, discard_concentrations=True
    )

    # Map drug pair choice to actual DataFrames and names
    drug_pair_mapping = {
        "nifedipine-ca": (
            signal_data.nifedipine_cases,
            signal_data.ca_titration_cases,
            ["Nifedipine", "Ca²⁺ Titration"],
        ),
        "nifedipine-e4031": (
            signal_data.nifedipine_cases,
            signal_data.e4031_cases,
            ["Nifedipine", "E-4031"],
        ),
        "e4031-ca": (
            signal_data.e4031_cases,
            signal_data.ca_titration_cases,
            ["E-4031", "Ca²⁺ Titration"],
        ),
    }

    drug1_df, drug2_df, drug_names = drug_pair_mapping[args.drug_pair]

    # Run analysis
    print("\nPerforming Hill fitting and generating plots...")
    results, fig = compare_drug_responses(
        drug1_df=drug1_df,
        drug2_df=drug2_df,
        features=args.features,
        drug_names=drug_names,
    )

    # Save results
    print("\nSaving results...")
    output_base = output_path / f"hill_fitting_{args.drug_pair}"

    # Save plot
    fig.savefig(f"{output_base}.pdf", bbox_inches="tight", dpi=300)

    # Cleanup
    print("\nCleaning up temporary files...")
    cleanup_hillfit_files()

    print("\nAnalysis complete! Results saved to:", output_path)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
