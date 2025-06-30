"""
t-SNE Visualization Module for Cardiac Signal Feature Analysis

This module provides comprehensive visualization capabilities for t-SNE analysis
results, including publication-quality plots for thesis presentation. The visualizer
handles complex multi-drug, multi-concentration plots with proper legends and
thesis-compatible formatting.

Scientific Background:
    t-SNE visualization reveals:
    - Drug-specific clustering patterns in cardiac feature space
    - Concentration-dependent effects on cardiac function
    - Tissue-level variability in drug responses
    - Relationships between electrical, calcium, and mechanical parameters

Key Visualization Components:
    1. Multi-drug t-SNE plots with concentration gradients
    2. Tissue-specific marker differentiation
    3. Publication-quality legends and formatting
    4. LaTeX-compatible figure sizing and typography
    5. Concentration-dependent size and alpha scaling

Pipeline Architecture:
    t-SNE Results → Plot Configuration → Visualization Generation → Export

Supported Plot Types:
    - Multi-drug t-SNE scatter plots
    - Concentration gradient visualizations
    - Tissue-specific marker plots
    - Thesis-formatted publication plots

Drug Treatments Visualized:
    - Baseline: Control conditions (black markers)
    - E-4031: hERG potassium channel blocker (red gradient)
    - Nifedipine: L-type calcium channel blocker (blue gradient)
    - Ca²⁺ Titration: Calcium concentration modulation (green gradient)

Output Products:
    - High-resolution PDF plots for publication
    - LaTeX-compatible figure formatting
    - Multi-panel visualization layouts
    - Interactive plot configurations

Authors: Cardiac Electrophysiology Research Team
Version: 1.0
Dependencies: matplotlib, pandas, numpy
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TSneVisualizer:
    """
    Class for creating publication-quality t-SNE visualizations.

    This class handles all aspects of t-SNE plot generation including figure
    sizing, color schemes, legends, and thesis-compatible formatting. It provides
    methods for creating complex multi-drug, multi-concentration visualizations
    with proper statistical representation.

    Attributes:
        output_path (str): Directory for saving plots
        drug_colors (Dict): Color mapping for different drugs
        markers (List): Marker styles for tissue differentiation
        latex_config (Dict): LaTeX configuration for thesis formatting
    """

    def __init__(self, output_path: str = "./plots"):
        """
        Initialize TSneVisualizer with output configuration.

        Args:
            output_path (str): Directory for saving plots
        """
        self.output_path = output_path

        # Define drug colors
        self.drug_colors = {
            "Baseline": "black",
            "e4031": "#E41A1C",  # Red
            "nifedipine": "#377EB8",  # Blue
            "ca_titration": "#4DAF4A",  # Green
        }

        # Define markers for tissues
        self.markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h"]

        # Configure LaTeX formatting
        self._configure_latex()

        logger.info(f"Initialized TSneVisualizer with output path: {output_path}")

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
        width: str = "thesis",
        width_fraction: float = 1.0,
        height_fraction: float = 1.0,
        subplots: Tuple[int, int] = (1, 1),
        golden_ratio: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Set figure dimensions to avoid scaling in LaTeX.

        Args:
            width (str): Document width type ("thesis", "beamer") or float
            width_fraction (float): Fraction of width to occupy
            height_fraction (float): Fraction of height to occupy
            subplots (Tuple[int, int]): Number of rows and columns
            golden_ratio (Optional[float]): Golden ratio adjustment

        Returns:
            Tuple[float, float]: Figure dimensions in inches
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

    def _create_drug_legend_handles(
        self, ax: plt.Axes, averaged_df: pd.DataFrame, drug_names: List[str]
    ) -> List:
        """
        Create legend handles for drug concentrations.

        Args:
            ax (plt.Axes): Matplotlib axes object
            averaged_df (pd.DataFrame): Averaged t-SNE data
            drug_names (List[str]): List of drug names

        Returns:
            List: Legend handles for drugs and concentrations
        """
        legend_handles = []

        # Add baseline first
        baseline_scatter = ax.scatter(
            [],
            [],
            c=self.drug_colors["Baseline"],
            marker="o",
            s=40,
            alpha=0.7,
            label="Baseline",
        )
        legend_handles.append(baseline_scatter)

        # Add drug concentrations with size gradients
        for data_type in drug_names:
            data_subset = averaged_df[averaged_df["data_type"] == data_type]
            concentrations = sorted(data_subset["concentration"].unique())

            if len(concentrations) > 1 and data_type != "Baseline":
                alphas = np.linspace(0.4, 0.9, len(concentrations))
                if len(concentrations) > 4:
                    sizes = np.exp(
                        np.linspace(np.log(20), np.log(60), len(concentrations))
                    )
                else:
                    sizes = np.linspace(20, 50, len(concentrations))
            else:
                alphas = [0.7]
                sizes = [30]

            for conc, alpha, size in zip(concentrations, alphas, sizes):
                # Create appropriate labels
                if data_type == "e4031":
                    label = f"E-4031 ({conc} µM)"
                elif data_type == "ca_titration":
                    label = r"Ca$^{2+}$" + f"Titration ({conc} µmol)"
                else:
                    label = f"{data_type.capitalize()} ({conc} µmol)"
                    if conc == 20:
                        label = f"{data_type.capitalize()} (" + r"1h$_{pt}$" + ")"

                scatter = ax.scatter(
                    [],
                    [],
                    c=self.drug_colors[data_type],
                    marker="o",
                    s=size,
                    alpha=alpha,
                    label=label,
                )
                legend_handles.append(scatter)

        return legend_handles

    def _create_tissue_legend_handles(
        self, ax: plt.Axes, averaged_df: pd.DataFrame
    ) -> Tuple[List, Dict]:
        """
        Create legend handles for tissue markers.

        Args:
            ax (plt.Axes): Matplotlib axes object
            averaged_df (pd.DataFrame): Averaged t-SNE data

        Returns:
            Tuple[List, Dict]: Legend handles for tissues and marker mapping
        """
        tissue_handles = []
        unique_tissues = sorted(averaged_df["bct_id"].unique())
        marker_map = dict(zip(unique_tissues, self.markers[: len(unique_tissues)]))

        for tissue in unique_tissues:
            tissue_scatter = ax.scatter(
                [],
                [],
                c="gray",
                marker=marker_map[tissue],
                s=40,
                label=f"Tissue {tissue}",
                alpha=0.7,
            )
            tissue_handles.append(tissue_scatter)

        return tissue_handles, marker_map

    def create_multi_drug_tsne_plot(
        self,
        averaged_df: pd.DataFrame,
        drug_names: List[str],
        width_fraction: float = 1.0,
        height_fraction: float = 0.85,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create publication-quality multi-drug t-SNE plot.

        This method creates a comprehensive t-SNE visualization showing drug-specific
        clustering patterns with concentration gradients and tissue differentiation.

        Args:
            averaged_df (pd.DataFrame): Averaged t-SNE data
            drug_names (List[str]): List of drug names to visualize
            width_fraction (float): Figure width fraction
            height_fraction (float): Figure height fraction

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects

        Example:
            >>> fig, ax = visualizer.create_multi_drug_tsne_plot(
            ...     averaged_df=results["averaged_df"],
            ...     drug_names=["e4031", "nifedipine", "ca_titration"]
            ... )
        """
        logger.info("Creating multi-drug t-SNE plot...")

        # Create figure with thesis-compatible sizing
        figsize = self.set_figure_size(
            width_fraction=width_fraction, height_fraction=height_fraction
        )
        fig = plt.figure(figsize=figsize)

        # Create main axes with specific position to accommodate legends
        ax = fig.add_axes([0.15, 0.15, 0.55, 0.75])

        # Get unique tissues for marker mapping
        unique_tissues = sorted(averaged_df["bct_id"].unique())

        # Create legend handles
        legend_handles = self._create_drug_legend_handles(ax, averaged_df, drug_names)
        tissue_handles, marker_map = self._create_tissue_legend_handles(ax, averaged_df)

        # Plot data points
        for data_type in ["Baseline"] + drug_names:
            data_subset = averaged_df[averaged_df["data_type"] == data_type]
            concentrations = sorted(data_subset["concentration"].unique())

            if len(concentrations) > 1 and data_type != "Baseline":
                alphas = np.linspace(0.4, 0.9, len(concentrations))
                if len(concentrations) > 4:
                    sizes = np.exp(
                        np.linspace(np.log(20), np.log(60), len(concentrations))
                    )
                else:
                    sizes = np.linspace(20, 50, len(concentrations))
            else:
                alphas = [0.7]
                sizes = [30]

            for conc, alpha, size in zip(concentrations, alphas, sizes):
                conc_subset = data_subset[data_subset["concentration"] == conc]

                for tissue in unique_tissues:
                    tissue_mask = conc_subset["bct_id"] == tissue
                    if not any(tissue_mask):
                        continue

                    ax.scatter(
                        conc_subset[tissue_mask]["TSNE1"],
                        conc_subset[tissue_mask]["TSNE2"],
                        c=self.drug_colors[data_type],
                        marker=marker_map[tissue],
                        s=size,
                        alpha=alpha,
                        edgecolor="white",
                        linewidth=0.5,
                    )

        # Set labels and formatting
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.tick_params(axis="both", labelsize=8)

        # Add legends
        drug_legend = ax.legend(
            handles=legend_handles,
            title="Drug \\& Concentration",
            bbox_to_anchor=(0.995, 1.0),
            loc="upper left",
            fontsize=4.25,
            title_fontsize=6,
            frameon=True,
            edgecolor="black",
            fancybox=False,
            labelspacing=1.3,
            borderpad=0.47,
            borderaxespad=0.62,
        )
        ax.add_artist(drug_legend)

        tissue_legend = ax.legend(
            handles=tissue_handles,
            title="Tissue ID",
            bbox_to_anchor=(1.02, 0.2),
            loc="center left",
            fontsize=5,
            title_fontsize=6,
            frameon=True,
            edgecolor="black",
            fancybox=False,
        )

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 0.75, 1])

        logger.info("Multi-drug t-SNE plot created successfully")

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
        self, analysis_results: Dict, drug_names: List[str], save_plots: bool = True
    ) -> Dict:
        """
        Create comprehensive t-SNE visualization suite.

        This method generates all necessary plots for the t-SNE analysis,
        including the main multi-drug plot and any additional visualizations.

        Args:
            analysis_results (Dict): Results from TSneAnalyzer
            drug_names (List[str]): List of drug names
            save_plots (bool): Whether to save plots to files

        Returns:
            Dict: Dictionary containing all generated figures and file paths

        Example:
            >>> visualization_results = visualizer.create_comprehensive_visualization(
            ...     analysis_results=results,
            ...     drug_names=["e4031", "nifedipine", "ca_titration"]
            ... )
        """
        logger.info("Creating comprehensive t-SNE visualization suite...")

        visualization_results = {}

        # Create main multi-drug t-SNE plot
        fig, ax = self.create_multi_drug_tsne_plot(
            averaged_df=analysis_results["averaged_df"], drug_names=drug_names
        )

        visualization_results["main_plot"] = {"figure": fig, "axes": ax}

        # Save plot if requested
        if save_plots:
            filepath = self.save_plot(fig, "drug_wise_global_tsne_averaged.pdf")
            visualization_results["main_plot"]["filepath"] = filepath

        # Add analysis summary
        visualization_results["analysis_summary"] = analysis_results["analysis_results"]

        logger.info("Comprehensive visualization suite completed")

        return visualization_results
