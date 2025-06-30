"""
PFA Visualization Module for Cardiac Signal Feature Analysis

This module provides comprehensive visualization capabilities for PFA analysis
results, including publication-quality plots for thesis presentation. The visualizer
handles PCA contribution heatmaps, knee detection curves, and explained variance
plots with proper legends and thesis-compatible formatting.

Scientific Background:
    PFA visualization reveals:
    - Feature contributions to principal components
    - Clustering patterns in feature space
    - Optimal parameter selection through knee detection
    - Explained variance distribution across components

Key Visualization Components:
    1. PCA contribution heatmaps with cluster highlighting
    2. Knee detection curves for epsilon parameter optimization
    3. Explained variance plots with cumulative variance thresholds
    4. Publication-quality legends and formatting
    5. LaTeX-compatible figure sizing and typography

Pipeline Architecture:
    PFA Results → Plot Configuration → Visualization Generation → Export

Supported Plot Types:
    - PCA cluster heatmaps
    - Knee detection curves
    - Explained variance plots
    - Thesis-formatted publication plots

Drug Treatments Visualized:
    - Baseline: Control conditions
    - E-4031: hERG potassium channel blocker
    - Nifedipine: L-type calcium channel blocker
    - Ca²⁺ Titration: Calcium concentration modulation

Output Products:
    - High-resolution PDF plots for publication
    - LaTeX-compatible figure formatting
    - Multi-panel visualization layouts
    - Interactive plot configurations

Authors: Cardiac Electrophysiology Research Team
Version: 1.0
Dependencies: matplotlib, pandas, numpy, seaborn
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
import logging

# Import from utils to avoid duplication
from utils import FeatureMapping

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PfaVisualizer:
    """
    Class for creating publication-quality PFA visualizations.

    This class handles all aspects of PFA plot generation including figure
    sizing, color schemes, legends, and thesis-compatible formatting. It provides
    methods for creating complex PCA heatmaps, knee detection curves, and
    explained variance plots with proper statistical representation.

    Attributes:
        output_path (str): Directory for saving plots
        cmaps (List): Color maps for different clusters
        latex_config (Dict): LaTeX configuration for thesis formatting
    """

    def __init__(self, output_path: str = "./plots"):
        """
        Initialize PfaVisualizer with output configuration.

        Args:
            output_path (str): Directory for saving plots
        """
        self.output_path = output_path

        # Define color maps for different clusters
        self.cmaps = [
            "Reds",
            "Blues",
            "Greens",
            "Purples",
            "Oranges",
            "YlOrBr",
            "PuBu",
            "RdPu",
        ]

        # Configure LaTeX formatting
        self._configure_latex()

        logger.info(f"Initialized PfaVisualizer with output path: {output_path}")

    def _configure_latex(self):
        """Configure matplotlib for LaTeX thesis formatting."""
        pgf_with_latex = {
            "pgf.texsystem": "pdflatex",
            "axes.labelsize": 8,
            "font.size": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 6,
            "axes.titlesize": 10,
            "ytick.labelsize": 6,
            "pgf.preamble": "\n".join(
                [
                    r"\usepackage[utf8x]{inputenc}",
                    r"\usepackage[T1]{fontenc}",
                ]
            ),
        }
        mpl.rcParams.update(pgf_with_latex)

    def set_figure_size(
        self,
        width: Union[str, float] = "thesis",
        width_fraction: float = 1.5,
        height_fraction: float = 0.8,
        subplots: Tuple[int, int] = (1, 1),
        golden_ratio: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Set figure dimensions to avoid scaling in LaTeX.

        Args:
            width (Union[str, float]): Document width type ("thesis", "beamer") or float
            width_fraction (float): Fraction of width to occupy
            height_fraction (float): Fraction of height to occupy
            subplots (Tuple[int, int]): Number of rows and columns
            golden_ratio (Optional[float]): Golden ratio adjustment

        Returns:
            Tuple[float, float]: Figure dimensions in inches
        """
        if width == "thesis":
            width_pt = 430.0
            height_pt = 556.0
        elif width == "beamer":
            width_pt = 307.28987
            height_pt = 230.0
        else:
            # If width is a float, use it directly
            width_pt = float(width)
            height_pt = 230.0

        # Convert from pt to inches
        inches_per_pt = 1 / 72.27
        fig_width_in = width_pt * inches_per_pt * width_fraction
        fig_height_in = height_pt * inches_per_pt * height_fraction

        # Adjust golden ratio for better subplot proportions
        if golden_ratio is not None:
            golden_ratio = (5**0.5 - 1) / 2
            fig_height_in = (
                fig_width_in
                * golden_ratio
                * (subplots[0] / subplots[1])
                * height_fraction
            )

        return (fig_width_in, fig_height_in)

    def _get_thesis_feature_name(self, feature: str) -> str:
        """
        Convert feature name to thesis format using utils.FeatureMapping.

        Args:
            feature (str): Original feature name

        Returns:
            str: Thesis-formatted feature name
        """
        return FeatureMapping.get_thesis_name(feature)

    def create_pca_cluster_heatmap(
        self, pfa_analyzer, width_fraction: float = 1.0, height_fraction: float = 0.4
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create PCA cluster heatmap showing feature contributions to principal components.

        Args:
            pfa_analyzer: Fitted PfaAnalyzer object
            width_fraction (float): Figure width fraction
            height_fraction (float): Figure height fraction

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        logger.info("Creating PCA cluster heatmap...")

        # Prepare data for plotting
        plot_data = []
        feature_names = []
        cluster_ids = []

        # Add independent features (noise points)
        for feature_info in pfa_analyzer.feature_groups_["independent_features"]:
            feature_name = feature_info["feature_name"]
            pca_components = pfa_analyzer._pca_components[feature_info["index"]]

            plot_data.append(pca_components)
            feature_names.append(self._get_thesis_feature_name(feature_name))
            cluster_ids.append(-1)

        # Add clustered features
        for cluster_id, cluster_info in pfa_analyzer.feature_groups_[
            "cluster_groups"
        ].items():
            for feature_info in cluster_info["features"]:
                feature_name = feature_info["feature_name"]
                pca_components = feature_info["pca_components"].values

                plot_data.append(pca_components)
                feature_names.append(self._get_thesis_feature_name(feature_name))
                cluster_ids.append(cluster_id)

        # Create DataFrame for plotting
        plot_df = pd.DataFrame(
            plot_data,
            index=feature_names,
            columns=[f"PC{i+1}" for i in range(pfa_analyzer.q)],
        )

        # Create figure
        figsize = self.set_figure_size(
            width_fraction=width_fraction, height_fraction=height_fraction
        )
        fig, ax = plt.subplots(
            figsize=figsize,
            gridspec_kw={
                "bottom": 0.10,
                "top": 0.8,
                "left": 0.10,
                "right": 0.85,
            },
        )

        # Calculate cluster positions
        cluster_positions = []
        current_pos = 0

        for cluster_id in sorted(set(cluster_ids)):
            cluster_size = cluster_ids.count(cluster_id)
            cluster_positions.append((current_pos, current_pos + cluster_size))
            current_pos += cluster_size

        # Plot heatmap for each cluster
        for (start, end), cmap in zip(
            cluster_positions,
            self.cmaps * (len(set(cluster_ids)) // len(self.cmaps) + 1),
        ):
            # Create mask for current cluster
            mask = np.ones_like(plot_df.values, dtype=bool)
            mask[start:end, :] = False

            # Plot heatmap for this cluster
            sns.heatmap(
                plot_df,
                mask=mask,
                cmap=cmap,
                center=0,
                annot=True,
                fmt=".2f",
                cbar=False,
                ax=ax,
                annot_kws={"size": 8},
            )

            # Add cluster label
            cluster_id = sorted(set(cluster_ids))[cluster_positions.index((start, end))]
            label = f"Cluster {cluster_id}\n({end-start} features)"
            if cluster_id == -1:
                label = f"Independent\n({end-start} features)"

            ax.text(
                1.02,
                (start + end) / 2,
                label,
                ha="left",
                va="center",
                transform=ax.get_yaxis_transform(),
            )

        # Set labels
        ax.set_xlabel("Principal Components")
        ax.set_ylabel("Features")

        # Adjust tick parameters
        ax.tick_params(axis="both")
        plt.setp(ax.get_yticklabels(), rotation=0)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()

        logger.info("PCA cluster heatmap created successfully")

        return fig, ax

    def create_knee_detection_plot(
        self,
        pfa_analyzer,
        drug_name: str,
        width_fraction: float = 0.5,
        height_fraction: float = 0.5,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create knee detection curve for epsilon parameter optimization.

        Args:
            pfa_analyzer: Fitted PfaAnalyzer object
            drug_name (str): Name of the drug treatment
            width_fraction (float): Figure width fraction
            height_fraction (float): Figure height fraction

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        logger.info("Creating knee detection plot...")

        if pfa_analyzer.kneedle is None:
            logger.warning("No knee detection data available")
            return None, None

        # Create figure
        figsize = self.set_figure_size(
            width_fraction=width_fraction, height_fraction=height_fraction
        )
        fig, ax = plt.subplots(figsize=figsize)

        # Plot normalized curve
        ax.plot(
            pfa_analyzer.kneedle.x_normalized,
            pfa_analyzer.kneedle.y_normalized,
            "b",
            label="Normalized Curve (" + r"$x_i^*$" + ", " + r"$d_i^*$" + ")",
        )

        # Set ticks
        ax.set_xticks(
            np.arange(
                pfa_analyzer.kneedle.x_normalized.min(),
                pfa_analyzer.kneedle.x_normalized.max() + 0.1,
                0.1,
            )
        )
        ax.set_yticks(
            np.arange(
                pfa_analyzer.kneedle.y_difference.min(),
                pfa_analyzer.kneedle.y_normalized.max() + 0.1,
                0.1,
            )
        )

        # Add knee line
        ax.vlines(
            pfa_analyzer.kneedle.norm_knee,
            ax.get_ylim()[0],
            ax.get_ylim()[1],
            linestyles="--",
            label=f"Knee ({np.round(pfa_analyzer.kneedle.norm_knee, 2)}, "
            f"{np.round(pfa_analyzer.kneedle.norm_knee_y, 2)}) ["
            f"$\\epsilon = {round(pfa_analyzer.kneedle.knee_y, 2)}$]",
            alpha=0.8,
            color="black",
        )

        # Add legend and labels
        ax.legend(loc="best", fontsize=5)
        ax.set_title("Knee Detection with [K=2]")
        ax.set_xlabel(
            r"Data Points ($x_i^*$ $=$ Min-Max-Scaled($x_{max} - x_i$))",
            fontsize=7,
        )
        ax.set_ylabel(
            r"$K^{th}$ distance ($d_i^*$ $=$ Min-Max-Scaled($d_{max} - d_i$))",
            fontsize=7,
        )

        plt.tight_layout()

        logger.info("Knee detection plot created successfully")

        return fig, ax

    def create_explained_variance_plot(
        self,
        pfa_analyzer,
        drug_name: str,
        width_fraction: float = 0.5,
        height_fraction: float = 0.4,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create explained variance plot showing cumulative variance by principal component.

        Args:
            pfa_analyzer: Fitted PfaAnalyzer object
            drug_name (str): Name of the drug treatment
            width_fraction (float): Figure width fraction
            height_fraction (float): Figure height fraction

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        logger.info("Creating explained variance plot...")

        # Prepare data
        pca_components_variance = pfa_analyzer.explained_variance_
        pca_components_variance_df = pd.DataFrame(
            {
                "PC": [f"{i+1}" for i in range(len(pca_components_variance))],
                "Variance": pca_components_variance,
            }
        )

        # Create figure
        figsize = self.set_figure_size(
            width_fraction=width_fraction, height_fraction=height_fraction
        )
        fig, ax = plt.subplots(figsize=figsize)

        # Plot variance bars
        sns.barplot(
            x="PC",
            y="Variance",
            data=pca_components_variance_df,
            ax=ax,
            color="lightgray",
            edgecolor="black",
        )

        # Add threshold line
        cumulative_variance = np.cumsum(pca_components_variance)
        pc_threshold = pfa_analyzer.explained_var
        pc_index = np.where(cumulative_variance >= pc_threshold)[0][0]

        ax.axvline(
            x=pc_index,
            color="red",
            linestyle="--",
            label=f"PC {pc_index + 1} ({int(pc_threshold*100)}\\% variance)",
        )

        # Set labels
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance")
        ax.tick_params(axis="both")
        ax.legend(loc="upper right", fontsize=5)

        plt.tight_layout()

        logger.info("Explained variance plot created successfully")

        return fig, ax

    def save_plot(
        self, fig: plt.Figure, filename: str, dpi: int = 300, bbox_inches: str = "tight"
    ) -> str:
        """
        Save plot to file with proper formatting.

        Args:
            fig (plt.Figure): Figure object to save
            filename (str): Output filename
            dpi (int): Resolution for saving
            bbox_inches (str): Bounding box setting

        Returns:
            str: Full path to saved file
        """
        import os

        os.makedirs(self.output_path, exist_ok=True)

        filepath = os.path.join(self.output_path, filename)
        fig.savefig(filepath, bbox_inches=bbox_inches, dpi=dpi)

        logger.info(f"Plot saved to: {filepath}")

        return filepath

    def create_comprehensive_visualization(
        self, pfa_analyzer, drug_name: str, save_plots: bool = True
    ) -> Dict:
        """
        Create comprehensive PFA visualization suite.

        This method generates all necessary plots for the PFA analysis,
        including PCA heatmaps, knee detection curves, and explained variance plots.

        Args:
            pfa_analyzer: Fitted PfaAnalyzer object
            drug_name (str): Name of the drug treatment
            save_plots (bool): Whether to save plots to files

        Returns:
            Dict: Dictionary containing all generated figures and file paths

        Example:
            >>> visualization_results = visualizer.create_comprehensive_visualization(
            ...     pfa_analyzer=results,
            ...     drug_name="e4031"
            ... )
        """
        logger.info("Creating comprehensive PFA visualization suite...")

        visualization_results = {}

        # Create PCA cluster heatmap
        fig_heatmap, ax_heatmap = self.create_pca_cluster_heatmap(pfa_analyzer)
        visualization_results["pca_heatmap"] = {
            "figure": fig_heatmap,
            "axes": ax_heatmap,
        }

        # Create knee detection plot
        fig_knee, ax_knee = self.create_knee_detection_plot(pfa_analyzer, drug_name)
        visualization_results["knee_detection"] = {"figure": fig_knee, "axes": ax_knee}

        # Create explained variance plot
        fig_variance, ax_variance = self.create_explained_variance_plot(
            pfa_analyzer, drug_name
        )
        visualization_results["explained_variance"] = {
            "figure": fig_variance,
            "axes": ax_variance,
        }

        # Save plots if requested
        if save_plots:
            # Save PCA heatmap
            if fig_heatmap is not None:
                filepath = self.save_plot(
                    fig_heatmap, f"pfa_dbscan_cluster_report_{drug_name}.pdf"
                )
                visualization_results["pca_heatmap"]["filepath"] = filepath

            # Save knee detection plot
            if fig_knee is not None:
                filepath = self.save_plot(
                    fig_knee, f"pfa_dbscan_kneedle_curve_{drug_name}.pdf"
                )
                visualization_results["knee_detection"]["filepath"] = filepath

            # Save explained variance plot
            if fig_variance is not None:
                filepath = self.save_plot(
                    fig_variance, f"pfa_dbscan_explained_variance_{drug_name}.pdf"
                )
                visualization_results["explained_variance"]["filepath"] = filepath

        # Add analysis summary
        visualization_results["analysis_summary"] = pfa_analyzer.get_analysis_summary()

        logger.info("Comprehensive PFA visualization suite completed")

        return visualization_results
