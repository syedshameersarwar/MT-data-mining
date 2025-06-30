"""
Regression Visualization Module for Cardiac Signal Feature Analysis

This module provides comprehensive visualization capabilities for regression analysis
results, including publication-quality coefficient heatmaps and comparative plots.
The visualizer handles coefficient matrix visualization with proper formatting and
thesis-compatible styling.

Scientific Background:
    Regression visualization reveals:
    - Feature relationship patterns and strengths
    - Drug-specific changes in feature correlations
    - Coefficient magnitude and sign patterns
    - Treatment effects on feature interdependencies

Key Visualization Components:
    1. Coefficient heatmaps with proper scaling
    2. Comparative heatmaps across treatments
    3. Pairwise treatment comparisons
    4. Publication-quality legends and formatting
    5. LaTeX-compatible figure sizing and typography

Pipeline Architecture:
    Analysis Results → Plot Configuration → Visualization Generation → Export

Supported Plot Types:
    - Individual coefficient heatmaps
    - Comparative 2x2 heatmap grids
    - Pairwise treatment comparisons
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
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import FeatureMapping

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegressionVisualizer:
    """
    Class for creating publication-quality regression analysis visualizations.

    This class handles all aspects of regression plot generation including figure
    sizing, color schemes, legends, and thesis-compatible formatting. It provides
    methods for creating coefficient heatmaps and comparative visualizations with
    proper statistical representation.

    Attributes:
        output_path (str): Directory for saving plots
        latex_config (Dict): LaTeX configuration for thesis formatting
    """

    def __init__(self, output_path: str = "./plots"):
        """
        Initialize RegressionVisualizer with output configuration.

        Args:
            output_path (str): Directory for saving plots
        """
        self.output_path = output_path

        # Configure LaTeX formatting
        self._configure_latex()

        logger.info(f"Initialized RegressionVisualizer with output path: {output_path}")

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

    def _process_coefficient_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Process coefficient matrix for visualization.

        Args:
            df (pd.DataFrame): Coefficient matrix

        Returns:
            np.ndarray: Processed matrix with diagonal set to None and small values filtered
        """
        matrix = df.values.copy()
        np.fill_diagonal(matrix, None)  # Set diagonal to None for visualization
        matrix[np.abs(matrix) < 1e-10] = None  # Filter very small values
        return matrix

    def _add_lambda_to_features(self, df: pd.DataFrame) -> List[str]:
        """
        Add lambda values to feature names for display.

        Args:
            df (pd.DataFrame): Coefficient matrix

        Returns:
            List[str]: Feature names with lambda values
        """
        new_index = []
        for feature in df.index:
            lambda_val = df.loc[feature, feature]
            new_index.append(
                f"{self._get_thesis_feature_name(feature)} ["
                + r"$\lambda$"
                + f" = {lambda_val:.3f}]"
            )
        return new_index

    def create_coefficient_heatmap(
        self,
        coefficient_matrix: pd.DataFrame,
        title: str = "Regression Coefficients",
        width_fraction: float = 1.0,
        height_fraction: float = 1.0,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a single coefficient heatmap.

        Args:
            coefficient_matrix (pd.DataFrame): Coefficient matrix
            title (str): Plot title
            width_fraction (float): Figure width fraction
            height_fraction (float): Figure height fraction

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        logger.info(f"Creating coefficient heatmap for {title}...")

        # Process matrix
        processed_values = self._process_coefficient_matrix(coefficient_matrix)

        # Create figure
        figsize = self.set_figure_size(
            width_fraction=width_fraction, height_fraction=height_fraction
        )
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        im = ax.imshow(
            processed_values,
            cmap="RdBu",
            aspect="auto",
            vmin=-np.max(np.abs(processed_values[~np.isnan(processed_values)])),
            vmax=np.max(np.abs(processed_values[~np.isnan(processed_values)])),
        )

        # Customize axes
        ax.set_xticks(np.arange(len(coefficient_matrix.columns)))
        ax.set_yticks(np.arange(len(coefficient_matrix.index)))
        ax.set_xticklabels(
            [self._get_thesis_feature_name(i) for i in coefficient_matrix.columns],
            rotation=90,
            ha="right",
            va="top",
        )
        ax.set_yticklabels(self._add_lambda_to_features(coefficient_matrix))

        # Add title
        ax.set_title(title, pad=25, fontsize=12)

        # Add gridlines
        ax.set_xticks(np.arange(-0.5, len(coefficient_matrix.columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(coefficient_matrix.index), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=2)

        # Customize tick parameters
        ax.tick_params(axis="both", which="major", labelsize=8)

        # Add axis labels
        ax.set_xlabel("Predictor Features", fontsize=10, labelpad=10)
        ax.set_ylabel("Target Features", fontsize=10, labelpad=10)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.01)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Coefficient", fontsize=10)

        plt.tight_layout()

        logger.info(f"Coefficient heatmap created for {title}")

        return fig, ax

    def create_comparative_heatmaps(
        self,
        coefficient_matrices: Dict[str, pd.DataFrame],
        drug_names: List[str] = None,
        width_fraction: float = 0.9,
        height_fraction: float = 1.6,
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Create 2x2 comparative heatmaps of regression coefficients.

        Args:
            coefficient_matrices (Dict[str, pd.DataFrame]): Dictionary of coefficient matrices
            drug_names (List[str]): List of drug names for titles
            width_fraction (float): Figure width fraction
            height_fraction (float): Figure height fraction

        Returns:
            Tuple[plt.Figure, List[plt.Axes]]: Figure and list of axes objects
        """
        logger.info("Creating comparative coefficient heatmaps...")

        if drug_names is None:
            drug_names = ["E-4031", "Nifedipine", r"Ca$^{2+}$ Titration"]

        # Create figure with thesis-compatible size
        figsize = self.set_figure_size(
            width_fraction=width_fraction, height_fraction=height_fraction
        )
        fig = plt.figure(figsize=(figsize[0] * 2, figsize[1] * 1.6))

        # Create GridSpec with adjusted spacing
        gs = fig.add_gridspec(
            2,
            3,
            width_ratios=[1, 1, 0.08],
            hspace=0.4,
            wspace=0.75,
            height_ratios=[1, 1],
        )

        # Get global min/max for consistent colorbar
        all_values = []
        for df in coefficient_matrices.values():
            processed = self._process_coefficient_matrix(df)
            all_values.extend(processed[~np.isnan(processed)])

        vmax = np.max(np.abs(all_values))
        vmin = -vmax

        # Create subplots
        axes = [
            fig.add_subplot(gs[0, 0]),  # Baseline
            fig.add_subplot(gs[0, 1]),  # Drug 1
            fig.add_subplot(gs[1, 0]),  # Drug 2
            fig.add_subplot(gs[1, 1]),  # Drug 3
        ]

        # Create colorbar axis
        cax = fig.add_subplot(gs[:, -1])

        # Plot heatmaps
        titles = ["Baseline"] + drug_names
        for ax, (drug, df), title in zip(axes, coefficient_matrices.items(), titles):
            processed_values = self._process_coefficient_matrix(df)

            # Create heatmap
            im = ax.imshow(
                processed_values, cmap="RdBu", aspect="auto", vmin=vmin, vmax=vmax
            )

            # Customize axes
            ax.set_xticks(np.arange(len(df.columns)))
            ax.set_yticks(np.arange(len(df.index)))
            ax.set_xticklabels(
                [self._get_thesis_feature_name(i) for i in df.columns],
                rotation=90,
                ha="right",
                va="top",
            )
            ax.set_yticklabels(self._add_lambda_to_features(df))

            # Add title
            ax.set_title(title, pad=25, fontsize=10)

            # Add gridlines
            ax.set_xticks(np.arange(-0.5, len(df.columns), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(df.index), 1), minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=2)

            # Customize tick parameters
            ax.tick_params(axis="both", which="major", labelsize=8)

            # Add axis labels only for left and bottom plots
            if ax in [axes[2], axes[3]]:  # Bottom plots
                ax.set_xlabel("Predictor Features", fontsize=10, labelpad=10)
            if ax in [axes[0], axes[2]]:  # Left plots
                ax.set_ylabel("Target Features", fontsize=10, labelpad=10)

        # Add colorbar
        cbar = plt.colorbar(
            im,
            cax=cax,
            label="Coefficient",
            fraction=0.046,
            pad=0.01,
        )
        cax.tick_params(labelsize=8)
        cax.set_ylabel("Coefficient", fontsize=10)

        plt.tight_layout()

        logger.info("Comparative coefficient heatmaps created successfully")

        return fig, axes

    def create_pairwise_comparison(
        self,
        matrix1: pd.DataFrame,
        matrix2: pd.DataFrame,
        drug_names: List[str] = None,
        width_fraction: float = 1.4,
        height_fraction: float = 0.8,
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Create 1x2 comparative heatmaps of regression coefficients using matplotlib.
        """
        logger.info("Creating pairwise coefficient comparison...")

        if drug_names is None:
            drug_names = ["Drug 1", "Drug 2"]

        def process_matrix(df):
            matrix = df.values.copy()
            np.fill_diagonal(matrix, None)
            matrix[np.abs(matrix) < 1e-10] = None
            return matrix

        def add_lambda_to_features(df):
            """Add lambda values to feature names"""
            new_index = []
            for feature in df.index:
                lambda_val = df.loc[feature, feature]
                new_index.append(
                    f"{self._get_thesis_feature_name(feature)} ["
                    + r"$\lambda$"
                    + f" = {lambda_val:.3f}]"
                )
            return new_index

        # Create figure with thesis-compatible size
        figsize = self.set_figure_size(
            width_fraction=width_fraction, height_fraction=height_fraction
        )
        fig = plt.figure(figsize=figsize)

        # Create GridSpec with adjusted spacing
        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=[1, 0.05],
            width_ratios=[1, 1],
            hspace=0.8,
            wspace=1.05,
        )

        # Get global min/max for consistent colorbar
        all_values = np.concatenate(
            [
                process_matrix(df)[~np.isnan(process_matrix(df))]
                for df in [matrix1, matrix2]
            ]
        )
        vmax = np.max(np.abs(all_values))
        vmin = -vmax

        # Create subplots
        axes = [
            fig.add_subplot(gs[0, 0]),  # First matrix
            fig.add_subplot(gs[0, 1]),  # Second matrix
        ]

        # Create colorbar axis at bottom, centered
        cax = fig.add_subplot(gs[1, :])

        dfs = [matrix1, matrix2]
        titles = drug_names

        # Plot heatmaps
        for ax, df, title in zip(axes, dfs, titles):
            processed_values = process_matrix(df)

            # Create heatmap
            im = ax.imshow(
                processed_values, cmap="RdBu", aspect="equal", vmin=vmin, vmax=vmax
            )

            # Customize axes
            ax.set_xticks(np.arange(len(df.columns)))
            ax.set_yticks(np.arange(len(df.index)))

            # Set tick labels with 90 degree rotation
            ax.set_xticklabels(
                [self._get_thesis_feature_name(i) for i in df.columns],
                rotation=90,
                ha="center",
                va="top",
            )
            ax.set_yticklabels(add_lambda_to_features(df))

            # Add title
            ax.set_title(title, pad=10, fontsize=12)

            # Add gridlines (only at cell boundaries)
            ax.set_xticks(np.arange(-0.5, len(df.columns), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(df.index), 1), minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=1)

            # Remove intermediate ticks
            ax.tick_params(which="minor", bottom=False, left=False)

            # Customize tick parameters
            ax.tick_params(axis="both", which="major", labelsize=8)

            # Add axis labels
            if ax == axes[1]:  # Right plot
                ax.set_ylabel("")  # Remove y-label for right plot
            if ax == axes[0]:  # Left plot
                ax.set_ylabel("Target Feature", fontsize=10, labelpad=8)

        # Add colorbar at bottom, centered
        cbar = plt.colorbar(
            im,
            cax=cax,
            orientation="horizontal",
            label="Predictor Feature",
        )
        cax.tick_params(labelsize=8)
        cax.set_xlabel("Predictor Feature", fontsize=10, labelpad=5)

        # Center the colorbar
        cbar.ax.set_position(
            [0.3, cax.get_position().y0, 0.4, cax.get_position().height]
        )

        # Adjust layout
        fig.subplots_adjust(
            top=0.95,
            bottom=0.15,
            left=0.12,
            right=0.95,
            wspace=0.15,
            hspace=0.15,
        )

        logger.info("Pairwise coefficient comparison created successfully")

        return fig, axes

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
        Create comprehensive regression visualization suite.

        This method generates all necessary plots for the regression analysis,
        including individual coefficient heatmaps and comparative visualizations.

        Args:
            analysis_results (Dict): Results from RegressionAnalyzer
            save_plots (bool): Whether to save plots to files

        Returns:
            Dict: Dictionary containing all generated figures and file paths

        Example:
            >>> visualization_results = visualizer.create_comprehensive_visualization(
            ...     analysis_results=results
            ... )
        """
        logger.info("Creating comprehensive regression visualization suite...")

        visualization_results = {}

        # Extract coefficient matrices
        coefficient_matrices = {}
        for drug, results in analysis_results.items():
            if (
                "coefficient_matrix" in results
                and not results["coefficient_matrix"].empty
            ):
                coefficient_matrices[drug] = results["coefficient_matrix"]
                print(
                    f"[DEBUG] Found coefficient matrix for {drug}: shape {results['coefficient_matrix'].shape}"
                )
            else:
                print(f"[DEBUG] No coefficient matrix for {drug} or matrix is empty.")

        if not coefficient_matrices:
            logger.warning("No coefficient matrices found for visualization")
            print("[DEBUG] No coefficient matrices found for visualization")
            return visualization_results

        # Create pairwise comparisons only
        drugs = list(coefficient_matrices.keys())
        print(f"[DEBUG] Drugs with coefficient matrices: {drugs}")
        if len(drugs) >= 2:
            # Baseline vs Nifedipine comparison
            if "baseline" in drugs and "nifedipine" in drugs:
                print("[DEBUG] Creating Baseline vs Nifedipine plot...")
                fig_pair, axes_pair = self.create_pairwise_comparison(
                    coefficient_matrices["baseline"],
                    coefficient_matrices["nifedipine"],
                    drug_names=["Baseline", "Nifedipine"],
                    width_fraction=2.0,  # Make it wider
                    height_fraction=0.8,
                )
                visualization_results["baseline_nifedipine_comparison"] = {
                    "figure": fig_pair,
                    "axes": axes_pair,
                }

                if save_plots:
                    filepath = self.save_plot(
                        fig_pair, "baseline_nifedipine_comparison.pdf"
                    )
                    print(f"[DEBUG] Saved Baseline vs Nifedipine plot to {filepath}")
                    visualization_results["baseline_nifedipine_comparison"][
                        "filepath"
                    ] = filepath

            # E-4031 vs Ca Titration comparison
            if "e4031" in drugs and "ca_titration" in drugs:
                print("[DEBUG] Creating E-4031 vs Ca Titration plot...")
                fig_pair2, axes_pair2 = self.create_pairwise_comparison(
                    coefficient_matrices["e4031"],
                    coefficient_matrices["ca_titration"],
                    drug_names=["E-4031", r"Ca$^{2+}$ Titration"],
                    width_fraction=2.0,  # Make it wider
                    height_fraction=0.8,
                )
                visualization_results["e4031_ca_titration_comparison"] = {
                    "figure": fig_pair2,
                    "axes": axes_pair2,
                }

                if save_plots:
                    filepath = self.save_plot(
                        fig_pair2, "e4031_ca_titration_comparison.pdf"
                    )
                    print(f"[DEBUG] Saved E-4031 vs Ca Titration plot to {filepath}")
                    visualization_results["e4031_ca_titration_comparison"][
                        "filepath"
                    ] = filepath

        # Add analysis summary
        visualization_results["analysis_summary"] = {
            "total_matrices": len(coefficient_matrices),
            "drugs_visualized": list(coefficient_matrices.keys()),
            "plot_types": list(visualization_results.keys()),
        }

        logger.info("Comprehensive regression visualization suite completed")
        print("[DEBUG] Visualization results keys:", list(visualization_results.keys()))

        return visualization_results
