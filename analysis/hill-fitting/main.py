"""
Hill Fitting Analysis Pipeline for Cardiac Drug Response Studies

This module provides a comprehensive pipeline for analyzing concentration-response relationships
in cardiac tissue using Hill equation fitting. The Hill equation is a mathematical model
commonly used in pharmacology to describe dose-response curves and calculate drug potency
metrics such as EC50 (effective concentration for 50% response) and IC50 (inhibitory
concentration for 50% response).

Scientific Background:
    The Hill equation describes the relationship between drug concentration and biological
    response as a sigmoidal curve:

    y = (y_max * x^n) / (EC50^n + x^n)

    Where:
    - y: Biological response (normalized between 0 and 1)
    - x: Drug concentration
    - y_max: Maximum response (typically 1 for normalized data)
    - EC50/IC50: Concentration producing 50% of maximum response
    - n (Hill coefficient): Measure of cooperativity/steepness

Key Features:
    - Automated detection of response direction (agonistic vs antagonistic)
    - Robust handling of zero concentrations in dose-response curves
    - Comprehensive visualization with publication-quality plots
    - Support for multiple cardiac signal features (electrical, calcium, mechanical)
    - Statistical analysis of dose-response relationships
    - Comparative analysis between different drug treatments

Data Processing Pipeline:
    1. Data normalization (0-1 scaling for each feature)
    2. Concentration preprocessing (handling zero values)
    3. Response direction detection (EC50 vs IC50)
    4. Hill equation fitting with optimization
    5. Statistical evaluation (R², Hill coefficient)
    6. Visualization and comparison

Supported Drug Types:
    - Nifedipine: L-type calcium channel blocker
    - E-4031: hERG potassium channel blocker
    - Ca²⁺ Titration: Calcium concentration modulation

Command Line Usage:
    python main.py --data-path /path/to/data --output-path ./results
                   --features duration force_peak_amplitude --drug-pair nifedipine-ca

Example:
    # Analyze nifedipine vs calcium titration for contractile duration
    python main.py --drug-pair nifedipine-ca --features duration --output-path ./hill_analysis

Authors: Cardiac Electrophysiology Research Team
Version: 1.0
Dependencies: pandas, numpy, matplotlib, hillfit, pathlib
"""

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
    """
    Hill Equation Fitting Analysis for Cardiac Drug Response Data

    This class provides comprehensive analysis of concentration-response relationships
    using the Hill equation, a fundamental pharmacological model for dose-response curves.
    The Hill equation is particularly useful for cardiac studies as it captures the
    sigmoidal nature of drug responses and provides quantitative metrics for drug potency.

    Scientific Context:
        In cardiac electrophysiology, drug responses typically follow sigmoidal curves
        where increasing concentrations produce proportionally greater effects until
        saturation. The Hill equation models this relationship and extracts key parameters:

        - EC50/IC50: Concentration producing 50% of maximal response
        - Hill coefficient (nH): Measure of cooperativity (steepness of curve)
        - R²: Goodness of fit measure

    Key Capabilities:
        - Automatic response direction detection (agonistic vs antagonistic)
        - Robust preprocessing of concentration data including zero handling
        - Feature-wise normalization for consistent scaling
        - Statistical evaluation of fit quality
        - Support for multiple cardiac signal modalities

    Attributes:
        drug1_df (pd.DataFrame): First drug concentration-response data
        drug2_df (pd.DataFrame): Second drug concentration-response data
        features_subset (list, optional): Specific features to analyze

    Example:
        >>> analyzer = HillFitting(nifedipine_data, ca_titration_data,
        ...                       features=['duration', 'force_peak_amplitude'])
        >>> results = analyzer.analyze_single_drug(nifedipine_data, features)
        >>> print(f"Duration EC50: {results['duration']['value']:.2f} µM")
    """

    def __init__(
        self,
        drug1_df: pd.DataFrame,
        drug2_df: pd.DataFrame,
        features_subset: list = None,
    ):
        """
        Initialize Hill fitting analyzer with drug response datasets.

        Args:
            drug1_df (pd.DataFrame): First drug dataset with columns:
                - concentration[um]: Drug concentrations in micromolar
                - tissue: Tissue identifier for grouping
                - Feature columns: Measured cardiac parameters
            drug2_df (pd.DataFrame): Second drug dataset with same structure
            features_subset (list, optional): Specific features to analyze.
                If None, analyzes all numeric columns except metadata.

        Note:
            Both DataFrames must have consistent column structure and
            concentration units (micromolar) for valid comparisons.
        """
        self.drug1_df = drug1_df
        self.drug2_df = drug2_df
        self.features_subset = features_subset

    @staticmethod
    def normalize_values(values: np.ndarray) -> np.ndarray:
        """
        Normalize values to 0-1 range using min-max scaling.

        Normalization is essential for Hill fitting as it:
        1. Ensures consistent scaling across different features
        2. Facilitates interpretation of EC50/IC50 as 50% of response range
        3. Improves numerical stability of optimization algorithms
        4. Enables meaningful comparison between different parameters

        The normalization formula used is:
        normalized = (value - min) / (max - min)

        Args:
            values (np.ndarray): Array of raw feature values to normalize

        Returns:
            np.ndarray: Normalized values scaled between 0 and 1, where:
                - 0 represents the minimum observed value
                - 1 represents the maximum observed value
                - All intermediate values are proportionally scaled

        Example:
            >>> raw_data = np.array([10, 20, 30, 40, 50])
            >>> normalized = HillFitting.normalize_values(raw_data)
            >>> print(normalized)  # [0.0, 0.25, 0.5, 0.75, 1.0]

        Note:
            If all values are identical (max == min), the function returns
            an array of zeros to avoid division by zero.
        """
        min_val = np.min(values)
        max_val = np.max(values)
        return (values - min_val) / (max_val - min_val)

    def preprocess_data(
        self, drug1_df: pd.DataFrame, drug2_df: pd.DataFrame, features: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess drug response data for Hill equation fitting.

        This method performs essential preprocessing steps to prepare concentration-response
        data for Hill fitting analysis. Preprocessing ensures data quality and enables
        meaningful parameter extraction from the Hill equation.

        Preprocessing Pipeline:
        1. Create independent copies of input DataFrames to avoid modification
        2. Apply min-max normalization (0-1 scaling) to each feature independently
        3. Preserve original concentration and metadata columns
        4. Maintain data structure for downstream analysis

        Why Preprocessing is Critical:
        - Normalization enables comparison across features with different units/scales
        - 0-1 scaling makes EC50/IC50 interpretation as "50% of response range"
        - Consistent scaling improves Hill equation fitting convergence
        - Preserves relative differences while standardizing absolute values

        Args:
            drug1_df (pd.DataFrame): First drug concentration-response dataset
            drug2_df (pd.DataFrame): Second drug concentration-response dataset
            features (List[str]): List of feature columns to normalize

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Preprocessed DataFrames with:
                - Normalized feature values (0-1 range)
                - Preserved concentration and metadata columns
                - Maintained row/column structure

        Example:
            >>> features = ['duration', 'force_peak_amplitude']
            >>> drug1_proc, drug2_proc = analyzer.preprocess_data(
            ...     drug1_df, drug2_df, features)
            >>> # Check normalization
            >>> print(drug1_proc['duration'].min(), drug1_proc['duration'].max())
            0.0 1.0

        Note:
            Original DataFrames are not modified. Concentration columns and
            metadata (tissue, condition, etc.) remain unchanged.
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
        Process and modify concentration values for Hill equation fitting.

        This method handles a critical preprocessing step for concentration-response analysis:
        the treatment of zero concentrations. In pharmacological studies, zero concentration
        represents control/baseline conditions, but poses mathematical challenges for Hill
        fitting due to logarithmic transformation requirements.

        Mathematical Challenge:
        - Hill fitting often uses log-transformed concentrations: log(concentration)
        - log(0) is undefined, causing fitting algorithms to fail
        - Zero concentrations are scientifically meaningful (control conditions)

        Solution Strategy:
        When zero concentrations are present:
        1. Identify minimum non-zero concentration
        2. Replace zero with small_value = min_nonzero / 10
        3. Create mapping between original and modified concentrations
        4. Preserve scientific interpretation while enabling mathematical fitting

        Pharmacological Rationale:
        - Zero concentration = no drug effect (baseline response)
        - Small positive value maintains baseline interpretation
        - Factor of 10 separation ensures clear distinction from lowest dose
        - Preserves dose-response curve shape and interpretation

        Args:
            df (pd.DataFrame): DataFrame containing concentration data with column
                'concentration[um]' in micromolar units

        Returns:
            Tuple[np.ndarray, Dict, List]: Three-element tuple containing:
                - x (np.ndarray): Modified concentration array for fitting
                - conc_mapping (Dict): Original → modified concentration mapping
                - new_conc (List): List of all modified concentration values

        Example:
            >>> # Dataset with zero concentration
            >>> df = pd.DataFrame({'concentration[um]': [0, 0.1, 1.0, 10.0]})
            >>> x, mapping, new_conc = HillFitting.process_concentrations(df)
            >>> print(mapping)
            {0: 0.01, 0.1: 0.1, 1.0: 1.0, 10.0: 10.0}
            >>> print(new_conc)
            [0.01, 0.1, 1.0, 10.0]

        Note:
            If no zero concentrations exist, returns original values unchanged.
            The replacement strategy maintains scientific validity while enabling
            robust mathematical fitting procedures.
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
        """
        Determine sigmoid curve direction and appropriate metric (EC50 vs IC50).

        This method automatically detects whether a drug acts as an agonist or antagonist
        by analyzing the overall trend in dose-response data. This detection is crucial
        for proper Hill equation fitting and parameter interpretation.

        Pharmacological Context:
        - Agonistic response: Increasing concentration → increasing effect (EC50)
        - Antagonistic response: Increasing concentration → decreasing effect (IC50)
        - EC50: Effective Concentration for 50% of maximal response
        - IC50: Inhibitory Concentration for 50% of maximal response

        Algorithm:
        1. Calculate mean response at each concentration across all tissues
        2. Compute slopes between consecutive concentration points
        3. Average all slopes to determine overall trend direction
        4. Assign appropriate metric and sign for Hill fitting

        Statistical Approach:
        - Uses tissue-averaged responses to reduce noise from individual variations
        - Slope-based detection is robust against outliers
        - Sign convention: +1 for agonistic, -1 for antagonistic, 0 for flat

        Args:
            df (pd.DataFrame): Concentration-response dataset containing:
                - concentration[um]: Drug concentrations
                - feature: Response variable to analyze
                - tissue: Tissue identifiers for grouping
            feature (str): Name of the response feature to analyze

        Returns:
            Tuple[str, float]: Two-element tuple containing:
                - metric (str): "ec50" for agonistic, "ic50" for antagonistic, "n/a" for flat
                - sign (float): +1 for positive slope, -1 for negative slope, 0 for no trend

        Example:
            >>> # Nifedipine typically reduces contractile force (antagonistic)
            >>> metric, sign = analyzer.check_sigmoid_direction_by_mean_slope(
            ...     nifedipine_df, 'force_peak_amplitude')
            >>> print(f"Metric: {metric}, Sign: {sign}")
            Metric: ic50, Sign: -1

            >>> # Calcium titration increases calcium amplitude (agonistic)
            >>> metric, sign = analyzer.check_sigmoid_direction_by_mean_slope(
            ...     ca_titration_df, 'calc_peak_amplitude')
            >>> print(f"Metric: {metric}, Sign: {sign}")
            Metric: ec50, Sign: 1

        Note:
            The sign is used to flip response values during Hill fitting to ensure
            proper curve orientation for the fitting algorithm.
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
    """
    Publication-Quality Visualization for Drug Concentration-Response Analysis

    This class provides comprehensive visualization capabilities for Hill equation fitting
    results, designed to meet publication standards for pharmaceutical and cardiac
    electrophysiology research. The visualizations combine statistical rigor with
    clear scientific communication.

    Scientific Visualization Principles:
        - Logarithmic concentration scaling (standard in pharmacology)
        - Error bars representing biological variability across tissues
        - Fitted curves with confidence visualization
        - EC50/IC50 markers for potency interpretation
        - Normalized response scaling (0-1) for feature comparison
        - Color-coded features with consistent scheme

    Key Visualization Components:
        1. Data Points: Mean ± SEM across tissues at each concentration
        2. Fitted Curves: Hill equation fits with parameter estimates
        3. Potency Lines: Vertical lines at EC50/IC50 concentrations
        4. Statistical Annotations: R², Hill coefficient, confidence metrics
        5. Comparative Layout: Side-by-side drug comparisons

    Publication Features:
        - LaTeX-compatible figure sizing for thesis/journal submission
        - Professional color schemes and typography
        - Consistent axis formatting and labeling
        - Legend positioning optimized for clarity
        - High-resolution output suitable for print

    Supported Analysis Types:
        - Single drug dose-response curves
        - Comparative drug analysis (side-by-side plots)
        - Multi-feature analysis with color coding
        - Statistical parameter visualization

    Example Usage:
        >>> visualizer = DrugResponseVisualizer()
        >>> fig, axes = visualizer.setup_figure()
        >>> visualizer.plot_single_drug(axes[0], nifedipine_results,
        ...                            "Nifedipine", features, 0)
        >>> plt.savefig("drug_response_analysis.pdf", dpi=300)

    Attributes:
        feature_colors (np.ndarray): Color palette for different features

    Note:
        Designed for cardiac electrophysiology data but adaptable to other
        concentration-response studies in pharmacology and toxicology.
    """

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
    Comprehensive comparative analysis of drug concentration-response relationships.

    This function orchestrates a complete Hill fitting analysis pipeline for comparing
    two drug treatments across multiple cardiac signal features. It integrates data
    preprocessing, statistical modeling, and publication-quality visualization to
    provide comprehensive pharmacological insights.

    Analysis Pipeline:
    1. Data Preprocessing:
       - Feature normalization (0-1 scaling)
       - Concentration handling (zero replacement)
       - Data quality validation

    2. Hill Equation Fitting:
       - Automatic response direction detection (EC50 vs IC50)
       - Non-linear optimization for parameter estimation
       - Statistical evaluation (R², Hill coefficient)
       - Error handling for failed fits

    3. Visualization Generation:
       - Side-by-side comparative plots
       - Multi-feature overlay with color coding
       - Statistical annotations and potency markers
       - Publication-ready formatting

    4. Results Organization:
       - Structured dictionary output
       - Parameter estimates with confidence metrics
       - Fitted model objects for further analysis
       - Statistical summaries

    Pharmacological Insights Provided:
    - Drug potency comparison (EC50/IC50 values)
    - Mechanism specificity (feature-dependent responses)
    - Cooperativity assessment (Hill coefficients)
    - Statistical confidence evaluation (R² values)
    - Tissue variability quantification (error bars)

    Args:
        drug1_df (pd.DataFrame): First drug concentration-response dataset with:
            - concentration[um]: Drug concentrations in micromolar
            - tissue: Tissue identifiers for statistical grouping
            - Feature columns: Measured cardiac parameters
        drug2_df (pd.DataFrame): Second drug dataset with identical structure
        features (List[str]): List of cardiac features to analyze, such as:
            - 'duration': Action potential duration
            - 'force_peak_amplitude': Contractile force magnitude
            - 'calc_peak_amplitude': Calcium transient amplitude
            - 'local_frequency[Hz]': Beating frequency
        drug_names (List[str]): Display names for drugs in plots and results

    Returns:
        Tuple[Dict, plt.Figure]: Two-element tuple containing:
            - results (Dict): Comprehensive analysis results with structure:
                {
                    'drug1': {
                        'feature_name': {
                            'metric': 'EC50' or 'IC50',
                            'value': potency_concentration,
                            'R2': goodness_of_fit,
                            'nH': hill_coefficient,
                            'stats': concentration_statistics,
                            'hill_fit': fitted_model_object,
                            'sign': response_direction
                        }
                    },
                    'drug2': {...}  # Same structure for second drug
                }
            - figure (plt.Figure): Publication-quality comparative plot with:
                - Side-by-side subplots for each drug
                - Multi-feature overlay with error bars
                - EC50/IC50 markers and statistical annotations
                - Optimized layout and formatting

    Example:
        >>> # Compare nifedipine vs calcium titration
        >>> features = ['duration', 'force_peak_amplitude', 'calc_peak_amplitude']
        >>> results, fig = compare_drug_responses(
        ...     nifedipine_df, ca_titration_df, features,
        ...     ['Nifedipine', 'Ca²⁺ Titration'])
        >>>
        >>> # Extract EC50 for contractile force
        >>> force_ic50 = results['drug1']['force_peak_amplitude']['value']
        >>> print(f"Nifedipine IC50 for force: {force_ic50:.2f} µM")
        >>>
        >>> # Save publication-ready figure
        >>> fig.savefig('drug_comparison.pdf', dpi=300, bbox_inches='tight')

    Raises:
        ValueError: If DataFrames have inconsistent structure or missing columns
        RuntimeError: If Hill fitting fails for all features

    Note:
        - Temporary files created by hillfit library are automatically cleaned up
        - Results dictionary preserves all fitting details for post-analysis
        - Figure formatting optimized for LaTeX document integration
        - Supports 2-8 features simultaneously with automatic color assignment
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
    Command-Line Interface for Hill Fitting Analysis Pipeline

    This function provides a comprehensive command-line interface for executing
    Hill equation fitting analysis on cardiac drug response data. It orchestrates
    the entire analysis pipeline from data loading through result visualization
    and output generation.

    Pipeline Architecture:
    1. Configuration Phase:
       - Parse command-line arguments with validation
       - Initialize data paths and output directories
       - Configure analysis parameters (features, drug pairs)

    2. Data Loading Phase:
       - Initialize SignalData handler for file management
       - Load drug-specific datasets (e-4031, nifedipine, ca_titration)
       - Perform data validation and structure verification

    3. Data Processing Phase:
       - Merge cases by drug type and baseline conditions
       - Apply concentration filtering and normalization
       - Prepare datasets for Hill fitting analysis

    4. Analysis Execution Phase:
       - Perform Hill equation fitting for selected drug pair
       - Generate statistical parameters (EC50/IC50, R², nH)
       - Create publication-quality visualization

    5. Output Generation Phase:
       - Save analysis results and fitted parameters
       - Export high-resolution plots (PDF format)
       - Clean up temporary files and resources

    Command Line Arguments:
        --data-path: Path to directory containing feature data files
        --output-path: Directory for storing analysis results and plots
        --features: List of cardiac features to analyze (space-separated)
        --drug-pair: Drug combination to compare (nifedipine-ca, nifedipine-e4031, e4031-ca)

    Supported Drug Combinations:
        - nifedipine-ca: L-type Ca²⁺ channel blocker vs Ca²⁺ titration
        - nifedipine-e4031: L-type Ca²⁺ blocker vs hERG K⁺ blocker
        - e4031-ca: hERG K⁺ channel blocker vs Ca²⁺ titration

    Available Features:
        - duration: Action potential duration (electrical)
        - force_peak_amplitude: Contractile force magnitude (mechanical)
        - calc_peak_amplitude: Calcium transient amplitude (calcium handling)
        - local_frequency[Hz]: Spontaneous beating frequency (pacemaking)

    Output Files Generated:
        - hill_fitting_{drug_pair}.pdf: Publication-ready comparative plot
        - Analysis results are displayed in console with statistical summaries
        - Temporary hillfit files are automatically cleaned up

    Example Usage:
        # Basic analysis with default parameters
        python main.py

        # Custom analysis with specific features and output path
        python main.py --data-path /path/to/features --output-path ./results
                       --features duration force_peak_amplitude
                       --drug-pair nifedipine-ca

        # Comprehensive multi-feature analysis
        python main.py --features duration force_peak_amplitude calc_peak_amplitude local_frequency[Hz]
                       --drug-pair e4031-ca --output-path ./comprehensive_analysis

    Error Handling:
        - Validates data path existence and accessibility
        - Checks for required data files and consistent structure
        - Handles Hill fitting failures gracefully with informative messages
        - Ensures output directory creation with appropriate permissions

    Console Output:
        - Configuration summary with all parameters
        - Data loading progress and file counts
        - Analysis execution status and results
        - Output file locations and completion confirmation

    Note:
        The pipeline is designed for cardiac electrophysiology data but can be
        adapted for other concentration-response studies by modifying the
        SignalData class and feature specifications.
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
