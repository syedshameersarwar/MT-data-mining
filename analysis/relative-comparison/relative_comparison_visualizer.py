"""
Relative Comparison Visualization Module for Cardiac Signal Feature Analysis

This module provides comprehensive visualization capabilities for relative comparison
analysis results, including publication-quality plots for thesis presentation. The visualizer
handles three types of plots: tissue-specific subplots, global average plots, and target
concentration comparison plots with proper legends and thesis-compatible formatting.

Scientific Background:
    Relative comparison visualization reveals:
    - Tissue-specific drug response patterns
    - Global drug effects across all tissues
    - Concentration-dependent feature changes
    - EC50/IC50 concentration effects

Key Visualization Components:
    1. Tissue-specific subplot layouts with concentration lines
    2. Global average plots with error bars
    3. Target concentration comparison plots
    4. Publication-quality legends and formatting
    5. LaTeX-compatible figure sizing and typography

Pipeline Architecture:
    Analysis Results → Plot Configuration → Visualization Generation → Export

Supported Plot Types:
    - Tissue-specific relative change subplots
    - Global average relative change plots
    - Target concentration comparison plots
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
import os

# Import from utils to avoid duplication
from utils import FeatureMapping

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RelativeComparisonVisualizer:
    """
    Class for creating publication-quality relative comparison visualizations.

    This class handles all aspects of relative comparison plot generation including figure
    sizing, color schemes, legends, and thesis-compatible formatting. It provides
    methods for creating tissue-specific subplots, global average plots, and target
    concentration comparison plots with proper statistical representation.

    Attributes:
        output_path (str): Directory for saving plots
        color_maps (List): Color maps for different plot types
        latex_config (Dict): LaTeX configuration for thesis formatting
    """

    def __init__(self, output_path: str = "./plots"):
        """
        Initialize RelativeComparisonVisualizer with output configuration.

        Args:
            output_path (str): Directory for saving plots
        """
        self.output_path = output_path
        self.color_maps = ["Dark2", "tab20", "Set3", "Paired"]

        # Configure LaTeX formatting
        self._configure_latex()

        logger.info(
            f"Initialized RelativeComparisonVisualizer with output path: {output_path}"
        )

    def _configure_latex(self):
        """Configure matplotlib for LaTeX thesis formatting."""
        pgf_with_latex = {
            "pgf.texsystem": "pdflatex",
            # "text.usetex": True,  # Commented out for compatibility
            # "font.family": "serif",
            # "font.serif": [],
            # "font.sans-serif": [],
            # "font.monospace": [],
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
        width_fraction: float = 1.0,
        height_fraction: float = 1.0,
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

    def _get_proper_drug_name(self, drug_name: str) -> str:
        """
        Get properly formatted drug name for plotting.

        Args:
            drug_name (str): Original drug name

        Returns:
            str: Formatted drug name
        """
        mapping = {
            "nifedipine": "Nifedipine",
            "ca_titration": r"$\rm{Ca^{2+}}$ Titration",
            "e4031": "E-4031",
        }
        return mapping.get(drug_name, drug_name)

    def create_tissue_specific_plots(
        self,
        relative_diff_df: pd.DataFrame,
        width_fraction: float = 1.0,
        height_fraction: float = 1.0,
        include_average: bool = True,
    ) -> Dict[str, Tuple[plt.Figure, plt.Axes]]:
        """
        Create tissue-specific relative change plots with subplots for each tissue.

        Args:
            relative_diff_df (pd.DataFrame): DataFrame with relative differences
            width_fraction (float): Figure width fraction
            height_fraction (float): Figure height fraction
            include_average (bool): Whether to include average subplot

        Returns:
            Dict[str, Tuple[plt.Figure, plt.Axes]]: Dictionary of figures and axes by drug
        """
        logger.info("Creating tissue-specific relative change plots...")

        # Get feature columns (excluding metadata)
        metadata_cols = ["drug", "bct_id", "concentration[um]"]
        feature_columns = [
            col
            for col in relative_diff_df.columns
            if col not in metadata_cols and col != "frequency[Hz]"
        ]

        drugs = sorted(list(set(relative_diff_df["drug"].unique())))
        color_map = plt.cm.tab20
        figures = {}

        for drug in drugs:
            drug_tissues = sorted(
                list(set(relative_diff_df[relative_diff_df["drug"] == drug]["bct_id"]))
            )
            n_tissues = len(drug_tissues)

            # Determine number of plots
            if include_average and n_tissues > 1:
                total_plots = n_tissues + 1
            else:
                total_plots = n_tissues

            # Create figure and subplots
            figsize = self.set_figure_size(
                width_fraction=width_fraction,
                height_fraction=height_fraction,
                subplots=(total_plots, 1),
            )
            fig, axes = plt.subplots(
                total_plots,
                1,
                figsize=figsize,
                gridspec_kw={
                    "hspace": 0.15,  # Increased spacing for legend
                    "top": 0.88,  # Reduced top margin for legend
                    "bottom": 0.1,
                    "left": 0.15,
                    "right": 0.95,
                },
            )

            # Convert to array if single subplot
            if total_plots == 1:
                axes = np.array([axes])

            # Process individual tissues
            for i, (tissue, ax) in enumerate(zip(drug_tissues, axes)):
                # Add baseline reference line
                ax.axhline(
                    y=100,
                    color="black",
                    linestyle="--",
                    label="Baseline" if i == 0 else None,
                )

                # Get data for this drug and tissue
                mask = (relative_diff_df["drug"] == drug) & (
                    relative_diff_df["bct_id"] == tissue
                )
                tissue_data = relative_diff_df[mask].sort_values("concentration[um]")

                if not tissue_data.empty:
                    concentrations = sorted(tissue_data["concentration[um]"].unique())
                    # Use lighter colors with better spacing
                    colors = [
                        (
                            color_map(i / (len(concentrations) - 1))
                            if len(concentrations) > 1
                            else color_map(0.5)
                        )
                        for i in range(len(concentrations))
                    ]

                    for idx, row in tissue_data.iterrows():
                        conc = row["concentration[um]"]
                        color_idx = concentrations.index(conc)
                        y_values = [
                            100 + row[feature]
                            for feature in feature_columns
                            if feature != "frequency[Hz]"
                        ]

                        # Plot lines and markers
                        label = f"{conc} µM" if i == 0 else None
                        if conc == 20 and label is not None:
                            label = r"1h$_{pt}$"
                        ax.plot(
                            range(len(feature_columns)),
                            y_values,
                            color=colors[color_idx],
                            marker="o",
                            label=label,
                            alpha=0.5,  # Slightly increased alpha for better visibility
                            linewidth=1,  # Slightly thicker lines
                            markersize=3,
                        )

                # Customize subplot
                ax.set_title(f"Tissue {tissue}", fontsize=8)
                ax.tick_params(axis="both", labelsize=7)

                # Only show x labels on bottom plot
                if i == len(axes) - 1:
                    feature_labels = [
                        self._get_thesis_feature_name(f) for f in feature_columns
                    ]
                    ax.set_xticks(range(len(feature_columns)))
                    ax.set_xticklabels(
                        feature_labels, rotation=90, ha="right", fontsize=7
                    )
                else:
                    ax.set_xticklabels([])

                ax.set_ylabel("Relative Change (\%)", fontsize=8)

            # Add average plot if needed
            if include_average and n_tissues > 1:
                ax = axes[-1]
                ax.axhline(y=100, color="black", linestyle="--", label="Baseline")

                # Calculate and plot averages
                drug_data = relative_diff_df[relative_diff_df["drug"] == drug]
                concentrations = sorted(drug_data["concentration[um]"].unique())
                # Use same color scheme for consistency
                colors = [
                    (
                        color_map(i / (len(concentrations) - 1))
                        if len(concentrations) > 1
                        else color_map(0.5)
                    )
                    for i in range(len(concentrations))
                ]

                for conc_idx, conc in enumerate(concentrations):
                    conc_data = drug_data[drug_data["concentration[um]"] == conc]
                    avg_changes = []
                    std_changes = []

                    for feature in feature_columns:
                        if feature != "frequency[Hz]":
                            avg_change = conc_data[feature].mean()
                            std_change = conc_data[feature].std()
                            avg_changes.append(100 + avg_change)
                            std_changes.append(std_change)

                    # Plot average with error bars
                    label = r"1h$_{pt}$" if conc == 20 else f"{conc} µmol"
                    label += " (avg)"

                    ax.errorbar(
                        range(len(feature_columns)),
                        avg_changes,
                        yerr=std_changes,
                        color=colors[conc_idx],
                        marker="o",
                        label=label,
                        alpha=0.5,
                        linewidth=1,
                        markersize=3,
                    )

                # Set labels for average plot
                feature_labels = [
                    self._get_thesis_feature_name(f) for f in feature_columns
                ]
                ax.set_xticks(range(len(feature_columns)))
                ax.set_xticklabels(feature_labels, rotation=90, ha="right", fontsize=7)
                ax.set_ylabel("Relative Change (\%)", fontsize=8)
                ax.set_xlabel("Features", fontsize=8, labelpad=15)
                ax.set_title("Average Across Tissues", fontsize=8)
                ax.tick_params(axis="both", labelsize=8)

            # Add legend in center top with multiple rows
            handles, labels = axes[0].get_legend_handles_labels()
            # Calculate number of columns for legend based on number of items
            n_items = len(handles)
            n_cols = min(3, n_items)  # Maximum 3 columns, adjust based on items

            fig.legend(
                handles,
                labels,
                bbox_to_anchor=(0.5, 0.98),  # Center top
                loc="lower center",
                borderaxespad=0,
                fontsize=7,
                ncol=n_cols,  # Multiple columns
                frameon=True,
                fancybox=True,
                shadow=True,
            )

            # Adjust layout to accommodate legend
            plt.tight_layout()

            figures[drug] = (fig, axes)

        logger.info(f"Created tissue-specific plots for {len(drugs)} drugs")

        return figures

    def create_global_average_plot(
        self,
        relative_diff_df: pd.DataFrame,
        width_fraction: float = 1.0,
        height_fraction: float = 0.5,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create global average relative change plot for all drugs.

        Args:
            relative_diff_df (pd.DataFrame): DataFrame with global relative differences
            width_fraction (float): Figure width fraction
            height_fraction (float): Figure height fraction

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        logger.info("Creating global average relative change plot...")

        # Get feature columns (excluding metadata)
        feature_columns = [col for col in relative_diff_df.columns if col != "drug"]

        # Create figure
        figsize = self.set_figure_size(
            width_fraction=width_fraction, height_fraction=height_fraction
        )
        fig, ax = plt.subplots(figsize=figsize)

        # Add baseline reference line at 100%
        ax.axhline(y=100, color="black", linestyle="--", label="Baseline", linewidth=1)

        # Create color map for drugs
        color_map = plt.cm.Dark2
        n_drugs = len(relative_diff_df)
        colors = [color_map(i / n_drugs) for i in range(n_drugs)]

        # Create x positions with increased spacing
        x = np.arange(0, len(feature_columns) * 1.5, 1.5)

        # Add line for each drug
        for idx, row in relative_diff_df.iterrows():
            drug = row["drug"]
            if ")" in drug:
                drug = (
                    self._get_proper_drug_name(drug.split(" ")[0])
                    + " "
                    + " ".join(drug.split(" ")[1:])
                )
                drug = drug.replace("µM", "µmol")
            y_values = [100 + row[feature] for feature in feature_columns]

            ax.plot(
                x,
                y_values,
                color=colors[idx],
                label=self._get_proper_drug_name(drug),
                marker="o",
                markersize=3,
                alpha=0.5,
                linewidth=1,
            )

        # Customize plot
        ax.set_ylim(0, 200)
        ax.grid(True, alpha=0.3)

        # Set x-ticks with thesis-friendly feature names
        feature_labels = [self._get_thesis_feature_name(f) for f in feature_columns]

        ax.set_xticks(x)
        ax.set_xticklabels(feature_labels, rotation=90, ha="right", fontsize=8)

        # Set labels
        ax.set_xlabel("Features", fontsize=9, labelpad=15)
        ax.set_ylabel("Relative Change (\%)", fontsize=9)

        # Customize tick parameters
        ax.tick_params(axis="both", labelsize=8)

        # Add legend
        ax.legend(
            loc="upper right",
            fontsize=6,
            ncol=1,
            framealpha=0.9,
            bbox_to_anchor=(0.92, 0.98),
        )

        # Adjust layout
        plt.tight_layout()

        logger.info("Global average relative change plot created successfully")

        return fig, ax

    def create_target_concentration_plot(
        self,
        relative_diff_df: pd.DataFrame,
        width_fraction: float = 1.0,
        height_fraction: float = 0.5,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create target concentration comparison plot (e.g., EC50/IC50 concentrations).

        Args:
            relative_diff_df (pd.DataFrame): DataFrame with target concentration relative differences
            width_fraction (float): Figure width fraction
            height_fraction (float): Figure height fraction

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        logger.info("Creating target concentration comparison plot...")

        # This is similar to global average plot but for specific concentrations
        return self.create_global_average_plot(
            relative_diff_df, width_fraction, height_fraction
        )

    def save_plot(
        self,
        fig: plt.Figure,
        filename: str,
        output_path: str = None,
        dpi: int = 300,
        bbox_inches: str = "tight",
    ) -> str:
        """
        Save plot to file with proper formatting.

        Args:
            fig (plt.Figure): Figure object to save
            filename (str): Output filename
            output_path (str): Custom output path (overrides self.output_path)
            dpi (int): Resolution for saving
            bbox_inches (str): Bounding box setting

        Returns:
            str: Full path to saved file
        """
        if output_path is None:
            output_path = self.output_path

        os.makedirs(output_path, exist_ok=True)

        filepath = os.path.join(output_path, filename)
        fig.savefig(filepath, bbox_inches=bbox_inches, dpi=dpi)

        logger.info(f"Plot saved to: {filepath}")

        return filepath

    def create_comprehensive_visualization(
        self, analysis_results: Dict, save_plots: bool = True
    ) -> Dict:
        """
        Create comprehensive relative comparison visualization suite.

        This method generates all necessary plots for the relative comparison analysis,
        including tissue-specific subplots, global average plots, and target concentration plots.

        Args:
            analysis_results (Dict): Results from RelativeComparisonAnalyzer
            save_plots (bool): Whether to save plots to files

        Returns:
            Dict: Dictionary containing all generated figures and file paths

        Example:
            >>> visualization_results = visualizer.create_comprehensive_visualization(
            ...     analysis_results=results
            ... )
        """
        logger.info("Creating comprehensive relative comparison visualization suite...")

        visualization_results = {}

        # Create tissue-specific plots
        if "tissue_specific" in analysis_results:
            tissue_results = analysis_results["tissue_specific"]
            tissue_figures = self.create_tissue_specific_plots(
                tissue_results["relative_differences"]
            )
            visualization_results["tissue_specific"] = tissue_figures

            # Save tissue-specific plots
            if save_plots:
                for drug_name, (fig, axes) in tissue_figures.items():
                    # Create drug-specific output directory
                    drug_folder = drug_name
                    if drug_folder == "ca_titration":
                        drug_folder = "ca-titration"
                    elif drug_folder == "e4031":
                        drug_folder = "e-4031"

                    drug_output_path = os.path.join(
                        self.output_path, "drugs", drug_folder
                    )

                    filepath = self.save_plot(
                        fig,
                        f"relative_differences_{drug_name}.pdf",
                        output_path=drug_output_path,
                    )
                    visualization_results["tissue_specific"][drug_name] = {
                        "figure": fig,
                        "axes": axes,
                        "filepath": filepath,
                    }

        # Create global average plot
        if "global_average" in analysis_results:
            global_results = analysis_results["global_average"]
            fig_global, ax_global = self.create_global_average_plot(
                global_results["relative_differences"]
            )
            visualization_results["global_average"] = {
                "figure": fig_global,
                "axes": ax_global,
            }

            # Save global average plot
            if save_plots:
                global_output_path = os.path.join(self.output_path, "global")

                filepath = self.save_plot(
                    fig_global,
                    "global_mean_relative_changes.pdf",
                    output_path=global_output_path,
                )
                visualization_results["global_average"]["filepath"] = filepath

        # Create target concentration plot
        if (
            analysis_results.get("target_concentration") is not None
            and analysis_results["target_concentration"]["relative_differences"]
            is not None
        ):
            target_results = analysis_results["target_concentration"]
            fig_target, ax_target = self.create_target_concentration_plot(
                target_results["relative_differences"]
            )
            visualization_results["target_concentration"] = {
                "figure": fig_target,
                "axes": ax_target,
            }

            # Save target concentration plot
            if save_plots:
                global_output_path = os.path.join(self.output_path, "global")

                filepath = self.save_plot(
                    fig_target,
                    "ec_ic50_mean_relative_changes.pdf",
                    output_path=global_output_path,
                )
                visualization_results["target_concentration"]["filepath"] = filepath

        # Add analysis summary
        visualization_results["analysis_summary"] = analysis_results.get("metadata", {})

        logger.info("Comprehensive relative comparison visualization suite completed")

        return visualization_results
