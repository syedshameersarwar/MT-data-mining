from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import numpy as np
import sys
from scipy import stats
from matplotlib_venn import venn3
import matplotlib.pyplot as plt
from itertools import combinations

sys.path.append(str(Path(__file__).parent.parent))
from utils import FeatureMapping


class CorrelationVisualizer:
    """
    Class to handle visualization of correlation analysis results.

    This class creates various visualizations including:
    - Venn diagrams of correlated features across drugs
    - Grid plots of correlation distributions (histograms and boxplots)
    - Line plots comparing correlations across different drugs

    The class handles both numbered and named feature pairs, and supports
    customization of plot dimensions and appearance.

    Attributes:
        feature_mapping (FeatureMapping): Instance for handling feature name mappings
        output_path (Path): Path for saving output files and plots
    """

    def __init__(self, output_path: str = "./"):
        """
        Initialize the CorrelationVisualizer.

        Args:
            output_path (str): Base path for storing outputs (data and plots)
        """
        self.feature_mapping = FeatureMapping()
        self.output_path = Path(output_path)
        self._setup_directories()

    def _setup_directories(self):
        """Create necessary directories for storing outputs"""
        # Create main directories
        (self.output_path / "data").mkdir(parents=True, exist_ok=True)
        (self.output_path / "plots" / "drugs").mkdir(parents=True, exist_ok=True)
        (self.output_path / "plots" / "global").mkdir(parents=True, exist_ok=True)

        # Create drug-specific directories
        for drug in ["baseline", "e4031", "nifedipine", "ca_titration"]:
            (self.output_path / "plots" / "drugs" / drug).mkdir(
                parents=True, exist_ok=True
            )

    @staticmethod
    def _get_drug_correlation_sets(combined_df: pd.DataFrame) -> Dict[str, Set[str]]:
        """Get sets of correlated features for each drug"""
        return {
            drug: set(
                # if correlated column is True for this feature pair considering a given drug, add the feature pair to the set
                combined_df[combined_df[f"{drug}_correlated"]].apply(
                    lambda x: f"{x['feature1']}_{x['feature2']}", axis=1
                )
            )
            for drug in ["nifedipine", "e4031", "ca_titration"]
        }

    def create_venn_diagram(self, combined_df: pd.DataFrame) -> plt.Figure:
        """Create Venn diagram of correlated features across drugs"""
        drug_sets = self._get_drug_correlation_sets(combined_df)

        fig_size = self._get_figure_size(width_fraction=0.8, height_fraction=0.5)
        fig = plt.figure(figsize=fig_size)

        venn = venn3(
            [drug_sets[drug] for drug in ["nifedipine", "e4031", "ca_titration"]],
            set_labels=("Nifedipine", "E-4031", r"$\rm{Ca^{2+}}$" + " Titration"),
            set_colors=("#000080", "#008000", "#800000"),
            alpha=0.6,
        )

        self._customize_venn_diagram(venn)
        return fig

    def create_correlation_grid_plots(
        self,
        baseline_corr_dict: Dict,
        e4031_corr_dict: Dict,
        nifedipine_corr_dict: Dict,
        ca_titration_corr_dict: Dict,
        feature_names: List[str],
        width: str = "thesis",
        width_fraction: float = 2.2,
        height_fraction: float = 4,
        optional_features: Optional[List[str]] = None,
        subset: bool = True,
    ) -> List[plt.Figure]:
        """
        Create correlation grid plots with histograms and boxplots.

        Generates visualization grids showing the distribution of correlations
        between feature pairs. Each grid contains:
        - Upper triangle: Hidden
        - Lower triangle: Histogram with KDE for correlation distribution
        - Diagonal: Empty

        Args:
            baseline_corr_dict: Dictionary with baseline correlation values between feature pair indices (i, j) -> list of correlations
            e4031_corr_dict: Dictionary with E-4031 correlation values between feature pair indices (i, j) -> list of correlations
            nifedipine_corr_dict: Dictionary with Nifedipine correlation values between feature pair indices (i, j) -> list of correlations
            ca_titration_corr_dict: Dictionary with Ca2+ titration correlation values between feature pair indices (i, j) -> list of correlations
            feature_names: List of feature names
            width: Plot width specification ('thesis' or 'beamer')
            width_fraction: Multiplier for plot width
            height_fraction: Multiplier for plot height
            optional_features: List of features to include/exclude
            subset: If True, only include optional_features; if False, exclude them

        Returns:
            List[plt.Figure]: List of figure pairs (histogram and boxplot) for each condition
        """

        def process_single_df(corrs_dict):
            if optional_features is not None:
                if subset:
                    features = [f for f in feature_names if f in optional_features]
                else:
                    features = [f for f in feature_names if f not in optional_features]
            else:
                features = feature_names

            n_features = len(features)
            feature_indices = {f: i for i, f in enumerate(feature_names)}
            selected_indices = [feature_indices[f] for f in features]

            figsize = self._get_figure_size(
                width=width,
                width_fraction=width_fraction,
                height_fraction=height_fraction,
                subplots=(n_features, n_features),
            )
            hist_fig, hist_axes = plt.subplots(n_features, n_features, figsize=figsize)
            box_fig, box_axes = plt.subplots(n_features, n_features, figsize=figsize)

            for i in range(n_features):
                for j in range(i):
                    orig_i = selected_indices[i]
                    orig_j = selected_indices[j]
                    corr_values = corrs_dict[(orig_i, orig_j)]
                    mean_corr = np.mean(corr_values)
                    std_corr = np.std(corr_values)

                    # Histogram subplot
                    hist_ax = hist_axes[i, j]
                    n_bins = 20
                    hist_ax.hist(
                        corr_values,
                        bins=n_bins,
                        alpha=0.3,
                        edgecolor="black",
                        linewidth=0.5,
                        rwidth=0.9,
                    )

                    # Add scaled KDE (scale = Number of samples * ((max - min) / number of bins))
                    # area under the KDE roughly match the total histogram counts, so it overlays correctly with a count-based histogram.
                    kernel = stats.gaussian_kde(corr_values)
                    x_range = np.linspace(min(corr_values), max(corr_values), 100)
                    kde_values = kernel(x_range)
                    hist_ax.plot(
                        x_range,
                        kde_values
                        * len(corr_values)
                        * (max(corr_values) - min(corr_values))
                        / n_bins,
                        color="red",
                        linewidth=1,
                        alpha=0.6,
                    )

                    # Add mean and std annotation
                    hist_ax.text(
                        0.5,
                        0.95,
                        r"$\mu=$"
                        + f"{mean_corr:.2f}\n"
                        + r"$\sigma=$"
                        + f"{std_corr:.3f}",
                        transform=hist_ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=6,
                    )

                    # Boxplot subplot
                    box_ax = box_axes[i, j]
                    box_ax.boxplot(corr_values, showmeans=True, meanline=True)

                    # Set axis limits and grid
                    hist_ax.set_xlim(-1.1, 1.1)
                    hist_ax.grid(True, alpha=0.2)
                    box_ax.grid(True, alpha=0.2)

                    # Adjust tick label size
                    hist_ax.tick_params(labelsize=6)
                    box_ax.tick_params(labelsize=6)

            # Hide upper triangle
            for i in range(n_features):
                for j in range(i, n_features):
                    hist_axes[i, j].set_visible(False)
                    box_axes[i, j].set_visible(False)

            # Add feature labels to the left and bottom of the plot (first column and last row)
            for idx, feature in enumerate(features):
                feature_name = self.feature_mapping.get_thesis_name(feature)
                hist_axes[idx, 0].set_ylabel(feature_name, fontsize=8, labelpad=10)
                box_axes[idx, 0].set_ylabel(feature_name, fontsize=8, labelpad=10)

                hist_axes[len(features) - 1, idx].set_xlabel(feature_name)
                box_axes[len(features) - 1, idx].set_xlabel(feature_name)

            hist_fig.tight_layout(rect=[0.08, 0.08, 0.92, 0.92], h_pad=1.5, w_pad=1.5)
            box_fig.tight_layout(rect=[0.08, 0.08, 0.92, 0.92], h_pad=1.5, w_pad=1.5)

            return hist_fig, box_fig

        results = []
        for corrs_dict in [
            baseline_corr_dict,
            e4031_corr_dict,
            nifedipine_corr_dict,
            ca_titration_corr_dict,
        ]:
            hist_fig, box_fig = process_single_df(corrs_dict)
            results.append((hist_fig, box_fig))

        return results

    def plot_drug_correlations_comparison(
        self,
        baseline_df: pd.DataFrame,
        nifedipine_df: pd.DataFrame,
        e4031_df: pd.DataFrame,
        catitration_df: pd.DataFrame,
        use_numbered_pairs: bool = False,
        width_fraction: float = 2.5,
        height_fraction: float = 0.6,
    ) -> Tuple[plt.Figure, pd.DataFrame]:
        """
        Create a line plot comparing bootstrap correlations across drugs.

        Generates a plot showing correlation trends across different drugs with:
        - Mean correlation values as lines with markers
        - 95% confidence intervals as shaded regions
        - Feature pairs on x-axis (numbered or named)
        - Correlation values on y-axis

        Args:
            baseline_df: DataFrame with baseline correlations
            nifedipine_df: DataFrame with Nifedipine correlations
            e4031_df: DataFrame with E-4031 correlations
            catitration_df: DataFrame with Ca2+ titration correlations
            use_numbered_pairs: If True, use numbered feature pairs instead of names
            width_fraction: Width fraction for figure size
            height_fraction: Height fraction for figure size

        Returns:
            Tuple[plt.Figure, pd.DataFrame]:
                - Figure object with the plot
                - DataFrame mapping feature pairs to their IDs/names
        """
        features_to_compare = list(
            set(baseline_df["feature1"]).union(set(baseline_df["feature2"]))
        )
        feature_pairs = list(combinations(sorted(features_to_compare), 2))

        # Create mapping DataFrame for all cases
        pair_mapping = pd.DataFrame(
            [
                {
                    "feature_pair": f"{feat1} vs {feat2}",
                    "feature1": feat1,
                    "feature2": feat2,
                    "thesis_feature_pair": (
                        f"{self.feature_mapping.get_thesis_name(feat1)} vs "
                        f"{self.feature_mapping.get_thesis_name(feat2)}"
                    ),
                }
                for idx, (feat1, feat2) in enumerate(feature_pairs, 1)
            ]
        )

        # Prepare data for plotting
        data_dict = {
            "baseline": {"data": baseline_df, "color": "black", "fill": "lightgray"},
            "nifedipine": {"data": nifedipine_df, "color": "blue", "fill": "lightblue"},
            "e4031": {"data": e4031_df, "color": "red", "fill": "mistyrose"},
            "ca_titration": {
                "data": catitration_df,
                "color": "green",
                "fill": "lightgreen",
            },
        }

        # Extract correlation data
        plot_data, sort_indices = self._prepare_correlation_plot_data(
            feature_pairs, data_dict
        )

        # Update pair mapping with sorted indices
        pair_mapping_sorted = pair_mapping.copy()  # Create a copy to avoid the warning
        pair_mapping_sorted = pair_mapping_sorted.iloc[sort_indices].reset_index(
            drop=True
        )
        pair_mapping_sorted["id"] = range(
            1, len(pair_mapping_sorted) + 1
        )  # Update IDs after sorting
        # sort the df based on the id column
        pair_mapping_sorted = pair_mapping_sorted.sort_values(by="id")
        # Save mapping to CSV
        pair_mapping_sorted.to_csv(
            self.output_path / "data" / "feature_pair_mapping.csv", index=False
        )

        # Create and customize plot
        fig = self._create_correlation_plot(
            plot_data,
            pair_mapping,
            use_numbered_pairs,
            sort_indices,
            width_fraction,
            height_fraction,
        )

        return fig, pair_mapping

    def _prepare_correlation_plot_data(self, feature_pairs, data_dict):
        """Prepare data for correlation plot"""
        plot_data = {
            name: {
                "mean": [],
                "ci_lower": [],
                "ci_upper": [],
                "color": data_dict[name]["color"],
                "fill": data_dict[name]["fill"],
            }
            for name in data_dict.keys()
        }

        for feat1, feat2 in feature_pairs:
            for name, drug_info in data_dict.items():
                mean, ci_lower, ci_upper = self._get_pair_correlation_data(
                    drug_info["data"], feat1, feat2
                )
                plot_data[name]["mean"].append(mean)
                plot_data[name]["ci_lower"].append(ci_lower)
                plot_data[name]["ci_upper"].append(ci_upper)

        # Sort feature pairs based on baseline correlation values
        sort_indices = self._get_sort_indices(plot_data["baseline"]["mean"])
        for name in plot_data:
            for key in ["mean", "ci_lower", "ci_upper"]:
                plot_data[name][key] = [plot_data[name][key][i] for i in sort_indices]

        return plot_data, sort_indices

    def _create_correlation_plot(
        self,
        plot_data,
        pair_mapping,
        use_numbered_pairs,
        sort_indices,
        width_fraction,
        height_fraction,
    ):
        """Create the correlation plot"""
        figsize = self._get_figure_size(
            width_fraction=width_fraction, height_fraction=height_fraction
        )
        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(next(iter(plot_data.values()))["mean"]))
        ax.axhline(y=0, color="black", linestyle="--", linewidth=1.8, zorder=3)

        # Plot each drug's data
        for name, data in plot_data.items():
            valid_indices = [i for i, m in enumerate(data["mean"]) if m is not None]
            if valid_indices:
                ax.fill_between(
                    x[valid_indices],
                    [data["ci_lower"][i] for i in valid_indices],
                    [data["ci_upper"][i] for i in valid_indices],
                    color=data["fill"],
                    alpha=0.6,
                    zorder=1,
                )
                ax.plot(
                    x[valid_indices],
                    [data["mean"][i] for i in valid_indices],
                    color=data["color"],
                    label=name.title(),
                    marker="o",
                    markersize=2,
                    linewidth=1,
                    zorder=2,
                )

        self._customize_correlation_plot(
            ax, x, pair_mapping, use_numbered_pairs, sort_indices
        )
        return fig

    @staticmethod
    def _get_pair_correlation_data(df, feat1, feat2):
        """Get correlation data for a feature pair"""
        mask = ((df["feature1"] == feat1) & (df["feature2"] == feat2)) | (
            (df["feature1"] == feat2) & (df["feature2"] == feat1)
        )
        if mask.any():
            row = df[mask].iloc[0]
            # ci_95 is a string like "(0.001, 0.002)", convert it to a tuple using eval
            ci = eval(row["ci_95"]) if isinstance(row["ci_95"], str) else row["ci_95"]
            return row["correlation_mean"], ci[0], ci[1]
        return None, None, None

    @staticmethod
    def _get_sort_indices(values):
        """Get indices for sorting correlation values"""
        valid_indices = [i for i, v in enumerate(values) if v is not None]
        return sorted(valid_indices, key=lambda i: values[i], reverse=True)

    def _customize_correlation_plot(
        self, ax, x, pair_mapping, use_numbered_pairs, sort_indices
    ):
        """Customize the correlation plot appearance"""
        ax.set_ylim(-1, 1)
        ax.grid(True, which="major", alpha=0.4)
        ax.grid(True, which="minor", alpha=0.2)
        ax.set_xlabel(
            "Feature Pair ID" if use_numbered_pairs else "Feature Pairs",
            fontsize=6,
            labelpad=10,
        )
        ax.set_ylabel("Bootstrap Correlation (Mean)", fontsize=6, labelpad=10)
        # sort the pair labels based on the sort_indices (sorted by baseline mean correlation)
        x_labels = [
            pair_mapping["thesis_feature_pair"].tolist()[i] for i in sort_indices
        ]
        if use_numbered_pairs:
            display_positions = [
                idx
                for idx, _ in enumerate(x_labels)
                if idx == 0
                or (idx + 1) % 10 == 0
                or idx
                == 76  # show after every 10th pair and FPD vs Force peak amplitude pair (77)
            ]
            display_labels = [idx + 1 for idx in display_positions]
            ax.set_xticks(display_positions)
            ax.set_xticklabels(display_labels, rotation=90, ha="center", fontsize=5)
            ax.set_xticks(range(len(x)), minor=True)
            ax.tick_params(axis="x", which="minor", length=2, width=0.5)
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=90, ha="center", fontsize=4)

        ax.legend()

    @staticmethod
    def _get_figure_size(
        width: str = "thesis",
        width_fraction: float = 1,
        height_fraction: float = 1,
        golden_ratio=None,
        subplots: Tuple[int, int] = (1, 1),
    ) -> Tuple[float, float]:
        """Calculate figure size based on thesis requirements"""
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

    @staticmethod
    def _customize_venn_diagram(venn) -> None:
        """Customize Venn diagram appearance"""
        for text in venn.set_labels:
            if text is not None:
                text.set_fontsize(12)

        for region_id in ["100", "010", "001", "110", "101", "011", "111"]:
            if venn.get_patch_by_id(region_id) is not None:
                label = venn.get_label_by_id(region_id)
                if label is not None:
                    label.set_fontsize(8)
                    label.set_color("black")
                    label.set_bbox(dict(facecolor="white", alpha=0.3))
