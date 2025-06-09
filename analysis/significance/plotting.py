from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import statsmodels.api as sm
from dataclasses import dataclass
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils import FeatureMapping


class DrugResponsePlotter:
    """
    A comprehensive visualization class for drug response analysis results.

    This class creates publication-quality plots for statistical analysis of cardiac drug responses,
    including diagnostic plots for model validation and significance plots for treatment effects.

    The class handles three types of visualizations:
    1. Diagnostic Plots: QQ plots and residual plots for model assumption validation
    2. Significance Plots: Treatment effect visualization with statistical annotations
    3. Custom Formatting: Thesis-quality formatting with proper scaling and labels

    Key Features:
    - Automatic layout calculation based on number of features
    - Logarithmic concentration scaling with custom tick formatting
    - Statistical significance annotation (Dunnett's test results)
    - Variance component display for mixed effects models
    - Tissue-specific data visualization with error bars
    - Customizable figure sizing for different publication requirements

    Attributes:
        drug_df (pd.DataFrame): Input data containing drug response measurements
        results (Dict): Statistical analysis results from mixed effects models
        selected_concentrations (List[float]): Concentration values used in analysis
        baseline (float): Reference concentration for statistical comparisons
        feature_mapping (FeatureMapping): Instance for converting feature names to thesis format

    Example:
        >>> plotter = DrugResponsePlotter(
        ...     drug_df=nifedipine_data,
        ...     statistical_results=lmer_results,
        ...     selected_concentrations=["0", "0.1", "1.0", "10.0"]
        ... )
        >>> qq_fig, resid_fig = plotter.create_diagnostic_plots()
        >>> sig_fig = plotter.create_significance_plot(title_prefix="Mixed Model")
    """

    def __init__(
        self,
        drug_df: pd.DataFrame,
        statistical_results: Dict,
        selected_concentrations: List[str],
    ):
        """
        Initialize the DrugResponsePlotter with data and analysis results.

        Args:
            drug_df (pd.DataFrame): DataFrame containing drug response measurements with columns:
                - bct_id: Tissue/batch identifier for grouping
                - concentration[um]: Drug concentration in micromoles
                - Feature columns: Various physiological measurements
            statistical_results (Dict): Dictionary containing statistical analysis results where
                keys are feature names and values are dictionaries with:
                - model: Fitted statistical model object
                - is_singular: Boolean indicating model singularity
                - anova_p_value: Overall ANOVA p-value
                - dunnett_results: Post-hoc test results (or None)
                - fitted_values: Model predicted values
                - residuals: Model residuals
                - variance_components: Random effects variance estimates
            selected_concentrations (List[str]): List of concentration values to include
                in analysis, with first element as baseline/control

        Raises:
            ValueError: If drug_df is empty or missing required columns
            KeyError: If statistical_results doesn't contain expected keys

        Example:
            >>> plotter = DrugResponsePlotter(
            ...     drug_df=drug_measurements,
            ...     statistical_results={"feature1": {"model": model, ...}, ...},
            ...     selected_concentrations=["0.0", "0.1", "1.0"]
            ... )
        """
        self.drug_df = drug_df
        self.results = statistical_results
        self.selected_concentrations = [float(c) for c in selected_concentrations]
        self.baseline = self.selected_concentrations[0]
        self.feature_mapping = FeatureMapping()

    def create_diagnostic_plots(
        self, selected_features: Optional[List[str]] = None
    ) -> Tuple[plt.Figure, plt.Figure]:
        """
        Create comprehensive diagnostic plots for model validation.

        Generates two types of diagnostic plots to assess statistical model assumptions:
        1. QQ Plots: Test normality of residuals using quantile-quantile plots
        2. Residual Plots: Test homoscedasticity using fitted vs. residual plots

        The plots are arranged in a grid layout that automatically adjusts based on
        the number of features being analyzed. Each subplot shows one feature with
        appropriate titles and formatting for publication quality.

        Model Assumptions Tested:
        - Normality: Residuals should follow normal distribution (QQ plot on line)
        - Homoscedasticity: Residual variance should be constant across fitted values
        - Independence: Assessed through random scatter in residual plots

        Args:
            selected_features (Optional[List[str]]): Subset of features to plot.
                If None, plots all features in statistical results.
                Useful for focusing on specific features or reducing plot complexity.

        Returns:
            Tuple[plt.Figure, plt.Figure]: Two matplotlib figures:
                - First figure: QQ plots for all selected features
                - Second figure: Residual plots for all selected features

        Side Effects:
            - Creates matplotlib figures that can be saved or displayed
            - Applies thesis-style formatting to all plots
            - Configures appropriate grid layout and spacing

        Example:
            >>> qq_fig, resid_fig = plotter.create_diagnostic_plots()
            >>> # Save plots
            >>> qq_fig.savefig("qq_plots.pdf", bbox_inches="tight")
            >>> resid_fig.savefig("residual_plots.pdf", bbox_inches="tight")

            >>> # Plot subset of features
            >>> subset_qq, subset_resid = plotter.create_diagnostic_plots(
            ...     selected_features=["duration", "force_peak_amplitude"]
            ... )
        """
        # Filter features if needed
        results_dict = (
            {k: v for k, v in self.results.items() if k in selected_features}
            if selected_features
            else self.results
        )

        # Calculate layout parameters
        n_features = len(results_dict)
        n_cols, n_rows, height_fraction, width_fraction = self._calculate_layout(
            n_features
        )

        # Create figures with GridSpec
        figsize = self._get_figure_size(width_fraction, height_fraction)
        fig_qq = plt.figure(figsize=figsize)
        fig_resid = plt.figure(figsize=figsize)

        # Setup GridSpec
        gs_qq = fig_qq.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.4)
        gs_resid = fig_resid.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.4)

        # Create plots for each feature
        for idx, (feature, result) in enumerate(results_dict.items()):
            row = 0 if n_features <= 4 else (idx // n_cols)
            col = idx if n_features <= 4 else (idx % n_cols)

            # Create QQ plot
            ax_qq = fig_qq.add_subplot(gs_qq[row, col])
            sm.graphics.qqplot(
                result["residuals"],
                line="s",
                ax=ax_qq,
                fit=True,
                alpha=0.5,
                markersize=2,
            )

            # Create Residuals plot
            ax_resid = fig_resid.add_subplot(gs_resid[row, col])
            ax_resid.scatter(
                result["fitted_values"],
                result["residuals"],
                alpha=0.5,
                color="blue",
                s=10,
                linewidth=1,
            )
            ax_resid.axhline(y=0, color="red", linestyle="--", alpha=0.8, linewidth=1)
            ax_resid.set_xlabel("Fitted Values")
            ax_resid.set_ylabel("Residuals")

            # Customize both plots
            thesis_feature = self.feature_mapping.get_thesis_name(feature)
            for ax in [ax_qq, ax_resid]:
                self._customize_diagnostic_subplot(ax, thesis_feature)

        # Adjust layout for both figures
        for fig in [fig_qq, fig_resid]:
            self._adjust_figure_layout(fig, n_features)

        return fig_qq, fig_resid

    def create_significance_plot(
        self, selected_features: Optional[List[str]] = None, title_prefix: str = ""
    ) -> plt.Figure:
        """
        Create comprehensive significance plots showing drug effects across features.

        Generates publication-quality plots displaying:
        1. Individual tissue responses with error bars
        2. Statistical significance annotations from Dunnett's tests
        3. Model statistics (p-values, variance components)
        4. Concentration-response relationships on log scale

        Plot Features:
        - Log-scaled concentration axis with custom formatting
        - Tissue-specific error bars showing mean ± standard deviation
        - Significance bars connecting baseline to treatment concentrations
        - Model singularity and variance component information
        - Baseline reference line for visual comparison

        Statistical Annotations:
        - Overall ANOVA p-value with significance stars (* p<0.05, ** p<0.01, *** p<0.001)
        - Dunnett's post-hoc test results with significance bars
        - Variance components (σ_α, σ_β, σ_ε) for mixed effects models
        - Model singularity status (Singular/Non-singular)

        Args:
            selected_features (Optional[List[str]]): Subset of features to include.
                If None, uses all features from statistical results.
            title_prefix (str): Prefix for subplot titles (e.g., "Mixed Model", "ANOVA").
                Helps distinguish between different analysis types.

        Returns:
            plt.Figure: Matplotlib figure containing significance plots arranged in grid layout.
                Figure includes global legend showing tissue identifiers.

        Side Effects:
            - Creates matplotlib figure with custom formatting
            - Applies log scaling to concentration axis
            - Adds statistical annotations and significance markers
            - Configures global legend for tissue identification

        Example:
            >>> # Create significance plot for mixed model results
            >>> sig_fig = plotter.create_significance_plot(
            ...     title_prefix="Mixed Model"
            ... )
            >>> sig_fig.savefig("significance_mixed_model.pdf", bbox_inches="tight")

            >>> # Focus on specific features
            >>> subset_fig = plotter.create_significance_plot(
            ...     selected_features=["duration", "force_peak_amplitude"],
            ...     title_prefix="ANOVA"
            ... )
        """
        features = selected_features or list(self.results.keys())
        n_features = len(features)

        # Calculate layout parameters
        n_cols, n_rows, height_fraction, width_fraction = self._calculate_layout(
            n_features
        )

        # Create figure with GridSpec
        figsize = self._get_figure_size(width_fraction, height_fraction)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.5, wspace=0.4)

        # Store legend information
        legend_handles = []
        legend_labels = []

        # Create subplot for each feature
        for idx, feature in enumerate(features, 1):
            row = 0 if n_features <= 4 else ((idx - 1) // n_cols)
            col = idx - 1 if n_features <= 4 else ((idx - 1) % n_cols)

            ax = fig.add_subplot(gs[row, col])

            # Plot data and collect legend info
            if idx == 1:
                handles, labels = self._plot_feature_data(
                    ax, feature, collect_legend=True
                )
                legend_handles.extend(handles)
                legend_labels.extend(labels)
            else:
                self._plot_feature_data(ax, feature)

            self._add_significance_markers(ax, feature, title_prefix)

            # Remove individual subplot legends
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        # Add global legend and adjust layout
        self._add_global_legend(fig, legend_handles, legend_labels)
        self._adjust_figure_layout(fig, n_features)

        return fig

    def _calculate_layout(self, n_features: int) -> Tuple[int, int, float, float]:
        """
        Calculate optimal subplot layout parameters based on number of features.

        Determines the best grid arrangement (rows × columns) and figure dimensions
        to accommodate the specified number of features while maintaining readability
        and proper aspect ratios.

        Layout Rules:
        - ≤3 features: Single row with variable columns
        - ≤4 features: Single row with 4 columns maximum
        - >4 features: Multiple rows with 3 columns per row

        Args:
            n_features (int): Number of features to display in grid layout

        Returns:
            Tuple[int, int, float, float]: Layout parameters:
                - n_cols: Number of columns in grid
                - n_rows: Number of rows in grid
                - height_fraction: Relative height scaling factor
                - width_fraction: Relative width scaling factor

        Example:
            >>> plotter._calculate_layout(3)
            (3, 1, 0.3, 1.0)
            >>> plotter._calculate_layout(6)
            (3, 2, 1.0, 2.0)
        """
        if n_features <= 3:
            n_cols = n_features
            n_rows = 1
            height_fraction = 0.3
            width_fraction = 1 * (n_features / 3)
            if n_features == 2:
                width_fraction = 1
        else:
            n_cols = 4 if n_features <= 4 else 3
            n_rows = 1 if n_features <= 4 else (n_features + 2) // 3
            height_fraction = n_rows * 0.5
            width_fraction = 2.0

        return n_cols, n_rows, height_fraction, width_fraction

    def _get_figure_size(
        self, width_fraction: float, height_fraction: float
    ) -> Tuple[float, float]:
        """
        Calculate figure dimensions in inches for thesis-quality output.

        Converts relative width and height fractions to absolute dimensions
        based on standard thesis page dimensions. Uses LaTeX document
        standards for consistent formatting across publications.

        Args:
            width_fraction (float): Relative width scaling factor (0.0-2.0+ typical)
            height_fraction (float): Relative height scaling factor (0.0-2.0+ typical)

        Returns:
            Tuple[float, float]: Figure dimensions in inches (width, height)

        Note:
            Based on standard thesis page dimensions:
            - Width: 430 points (~6.0 inches)
            - Height: 556 points (~7.7 inches)
            - Conversion: 1 point = 1/72.27 inches

        Example:
            >>> plotter._get_figure_size(1.0, 0.5)
            (5.95, 3.85)
        """
        width_pt = 430  # Thesis page width in points
        height_pt = 556  # Thesis page height in points
        inches_per_pt = 1 / 72.27

        fig_width = width_pt * inches_per_pt * width_fraction
        fig_height = height_pt * inches_per_pt * height_fraction

        return (fig_width, fig_height)

    def _customize_diagnostic_subplot(self, ax: plt.Axes, thesis_feature: str):
        """
        Apply consistent formatting to diagnostic plot subplots.

        Standardizes appearance of QQ plots and residual plots with:
        - Reduced font sizes appropriate for multi-panel figures
        - Grid lines for easier reading
        - Proper padding and margins
        - Thesis-formatted feature names as titles

        Args:
            ax (plt.Axes): Matplotlib axes object to customize
            thesis_feature (str): Formatted feature name for subplot title

        Side Effects:
            - Modifies axes properties in-place
            - Sets font sizes, grid, title, and label formatting

        Example:
            >>> ax = plt.subplot(2, 2, 1)
            >>> plotter._customize_diagnostic_subplot(ax, "FPD [s]")
        """
        ax.tick_params(axis="both", labelsize=6)
        ax.set_title(thesis_feature, fontsize=6, pad=4)
        ax.set_xlabel(ax.get_xlabel(), fontsize=6, labelpad=4)
        ax.set_ylabel(ax.get_ylabel(), fontsize=6, labelpad=4)
        ax.grid(True, alpha=0.3, linestyle="--")

    def _plot_feature_data(
        self, ax: plt.Axes, feature: str, collect_legend: bool = False
    ) -> Optional[Tuple[List, List]]:
        """
        Plot individual tissue responses with error bars for a specific feature.

        Creates scatter plots with error bars showing tissue-specific responses
        across drug concentrations. Each tissue is plotted with a unique color
        and marker style for identification.

        Data Processing:
        1. Groups data by concentration and tissue (bct_id)
        2. Calculates mean and standard deviation for each group
        3. Maps concentrations to log scale for display
        4. Plots error bars representing mean ± SD

        Args:
            ax (plt.Axes): Matplotlib axes for plotting
            feature (str): Feature name to plot (column in drug_df)
            collect_legend (bool): Whether to return legend handles for global legend.
                Set True for first subplot only to avoid duplicate legends.

        Returns:
            Optional[Tuple[List, List]]: If collect_legend=True, returns:
                - legend_handles: List of matplotlib Line2D objects
                - legend_labels: List of tissue identifiers
                Otherwise returns None.

        Side Effects:
            - Plots data points with error bars on provided axes
            - Configures log-scale concentration axis
            - Adds baseline reference line
            - Formats concentration tick labels

        Example:
            >>> handles, labels = plotter._plot_feature_data(
            ...     ax, "duration", collect_legend=True
            ... )
            >>> # Use for global legend
            >>> fig.legend(handles, labels, ...)
        """
        # Calculate summary statistics
        summary_stats = (
            self.drug_df.groupby(["concentration[um]", "bct_id"])[feature]
            .agg(["mean", "std"])
            .reset_index()
        )

        # Setup concentration mapping for log scale, replace 0 with small value ~ smallest non-zero concentration/10
        unique_conc = sorted(self.drug_df["concentration[um]"].unique())

        legend_handles = []
        legend_labels = []

        # Plot data for each tissue
        for tissue in self.drug_df["bct_id"].unique():
            tissue_data = summary_stats[summary_stats["bct_id"] == tissue]
            x_values = [
                self._get_mapped_concentration(c)
                for c in tissue_data["concentration[um]"]
            ]

            line = ax.errorbar(
                x_values,
                tissue_data["mean"],
                yerr=tissue_data["std"],
                label=tissue,
                marker="o",
                markersize=4,
                capsize=3,
                capthick=1,
                elinewidth=1,
                linestyle="None",
            )

            if collect_legend:
                legend_handles.append(line)
                legend_labels.append(tissue)

        # Customize axis
        self._customize_concentration_axis(ax, unique_conc, x_values)

        if collect_legend:
            return legend_handles, legend_labels
        return None

    def _customize_concentration_axis(
        self,
        ax: plt.Axes,
        unique_conc: List[float],
        x_values: List[float],
    ):
        """
        Configure concentration axis with logarithmic scaling and custom formatting.

        Sets up the x-axis to properly display drug concentrations:
        1. Applies logarithmic scaling for wide concentration ranges
        2. Formats tick labels with appropriate precision
        3. Handles special cases (baseline=0, post-treatment timepoints)
        4. Adds baseline reference line
        5. Rotates labels to prevent overlap

        Special Formatting Rules:
        - Concentration 0: Displayed as "0"
        - Concentration 20: Displayed as "1h_pt" (1-hour post-treatment)
        - Small concentrations (<0.01): 3 decimal places
        - Medium concentrations (0.01-1): 2 decimal places
        - Large concentrations (≥1): Integer or 1 decimal place

        Args:
            ax (plt.Axes): Matplotlib axes object to configure
            unique_conc (List[float]): Unique concentration values from data
            x_values (List[float]): Mapped x-coordinates for plotting

        Side Effects:
            - Sets logarithmic scale on x-axis
            - Configures tick positions and labels
            - Adds baseline reference line
            - Rotates tick labels 60 degrees

        Example:
            >>> unique_conc = [0.0, 0.1, 1.0, 10.0, 20.0]
            >>> x_values = [0.01, 0.1, 1.0, 10.0, 20.0]
            >>> plotter._customize_concentration_axis(ax, unique_conc, x_values)
            # Results in labels: ["0", "0.1", "1", "10", "1h_pt"]
        """
        ax.set_xscale("log")
        ax.set_xticks([self._get_mapped_concentration(c) for c in unique_conc])

        # Format concentration labels
        tick_labels = []
        for c in unique_conc:
            if c == 20:
                tick_labels.append(r"1h$_{pt}$")
            elif c == 0:
                tick_labels.append("0")
            elif c < 1:
                if c <= 0.1:
                    tick_labels.append(f"{c:.3f}" if c < 0.01 else f"{c:.2f}")
                else:
                    tick_labels.append(f"{c:.1f}")
            else:
                tick_labels.append(f"{int(c)}" if c.is_integer() else f"{c:.1f}")

        ax.set_xticklabels(tick_labels)
        ax.tick_params(axis="x", rotation=60)

        # Add reference lines
        ax.axvline(
            x=self._get_mapped_concentration(self.baseline),
            color="r",
            linestyle="--",
            alpha=0.3,
        )

    def _add_significance_markers(self, ax: plt.Axes, feature: str, title_prefix: str):
        """
        Add statistical significance annotations and model information to subplot.

        Enhances significance plots with comprehensive statistical information:
        1. Overall ANOVA p-value with significance stars
        2. Model variance components (for mixed effects models)
        3. Model singularity status
        4. Dunnett's post-hoc test results with significance bars

        Statistical Annotations:
        - P-value: Formatted to 3 decimal places
        - Significance stars: * (p<0.05), ** (p<0.01), *** (p<0.001)
        - Variance components: σ_α (tissue), σ_β (concentration×tissue), σ_ε (residual)
        - Singularity: Warning for models with convergence issues

        Post-hoc Visualization:
        - Horizontal bars connecting baseline to treatment concentrations
        - Significance indicated by stars or "ns" (not significant)
        - Vertical positioning to avoid overlap

        Args:
            ax (plt.Axes): Matplotlib axes for adding annotations
            feature (str): Feature name for statistical lookup
            title_prefix (str): Analysis type ("Mixed Model" or "ANOVA") for formatting

        Side Effects:
            - Adds title with statistical information
            - Draws significance bars above data points
            - Adjusts y-axis limits to accommodate annotations
            - Sets axis labels and formatting

        Example:
            >>> plotter._add_significance_markers(ax, "duration", "Mixed Model")
            # Adds title: "p=0.023 * (σ_α=0.15, σ_ε=0.08)\n(Non-singular)"
            # Draws significance bars for post-hoc comparisons
        """
        results = self.results[feature]

        # Add title with statistics
        p_val = results["anova_p_value"]
        stars = "*" * sum([p_val < cutoff for cutoff in [0.05, 0.01, 0.001]])
        title = f"p={p_val:.3f} {stars}"
        variance_components = results["variance_components"]
        if variance_components is not None:
            title += "("
            if "sigma_alpha" in variance_components:
                sigma_alpha = np.round(variance_components["sigma_alpha"], 4)
                title += f"$\sigma_{{\\alpha}}$={sigma_alpha}, "
            if "sigma_beta" in variance_components:
                sigma_beta = np.round(variance_components["sigma_beta"], 4)
                title += f"$\sigma_{{\\beta}}$={sigma_beta}, "
            sigma_epsilon = np.round(variance_components["sigma_epsilon"], 4)
            title += f"$\sigma_{{\\epsilon}}$={sigma_epsilon})"

        # # Add variance components if available
        # if "model" in results and hasattr(results["model"], "ranef_var"):
        #     variance_estimates = results["model"].ranef_var.iloc[:, -1]
        #     if len(variance_estimates) == 3:
        #         sigma_beta = np.round(variance_estimates.iloc[0], 4)
        #         sigma_alpha = np.round(variance_estimates.iloc[1], 4)
        #         sigma_epsilon = np.round(variance_estimates.iloc[2], 4)
        #         title += f" ($\sigma_{{\\alpha}}$={sigma_alpha}, $\sigma_{{\\beta}}$={sigma_beta}, $\sigma_{{\\epsilon}}$={sigma_epsilon})"
        #     elif len(variance_estimates) == 2:
        #         sigma_alpha = np.round(variance_estimates.iloc[0], 4)
        #         sigma_epsilon = np.round(variance_estimates.iloc[1], 4)
        #         title += f" ($\sigma_{{\\alpha}}$={sigma_alpha}, $\sigma_{{\\epsilon}}$={sigma_epsilon})"

        # Add model singularity information
        title += "\n(Singular)" if results["is_singular"] else "\n(Non-singular)"

        # Add Dunnett's test results
        if results["dunnett_results"] is not None:
            self._add_dunnett_markers(
                ax, results["dunnett_results"], feature, title_prefix
            )

        # Customize subplot
        thesis_feature = self.feature_mapping.get_thesis_name(feature)
        ax.tick_params(axis="both", labelsize=6)
        ax.set_xlabel("Concentration [µmol]", fontsize=6, labelpad=4)
        ax.set_ylabel(thesis_feature, fontsize=6, labelpad=4)
        ax.set_title(title, fontsize=6, pad=5)
        ax.xaxis.set_minor_locator(plt.NullLocator())

    def _add_dunnett_markers(
        self,
        ax: plt.Axes,
        dunnett_results: pd.DataFrame,
        feature: str,
        title_prefix: str,
    ):
        """
        Add Dunnett's test significance markers above data points.

        Creates horizontal bars with significance annotations for post-hoc
        comparisons between baseline and treatment concentrations. Each
        comparison is visually represented with:
        - Horizontal line connecting baseline to treatment concentration
        - Text annotation showing significance level
        - Proper vertical spacing to prevent overlap

        Significance Levels:
        - * : p < 0.05
        - ** : p < 0.01
        - *** : p < 0.001
        - ns : not significant (p ≥ 0.05)

        Args:
            ax (plt.Axes): Matplotlib axes for drawing markers
            dunnett_results (pd.DataFrame): Post-hoc test results with columns:
                - Contrast: Comparison description (e.g., "concentration1.0-concentration0.0")
                - P-val: Statistical p-value for comparison
                - Estimate: Effect size estimate (not used in visualization)
            feature (str): Feature name for calculating plot dimensions
            title_prefix (str): Analysis type for parsing contrast strings

        Side Effects:
            - Draws horizontal lines above data points
            - Adds text annotations for significance levels
            - Adjusts subplot y-axis limits to show all markers
            - Uses proportional spacing based on data range

        Example:
            >>> dunnett_df = pd.DataFrame({
            ...     'Contrast': ['concentration1.0-concentration0.0', 'concentration10.0-concentration0.0'],
            ...     'P-val': [0.023, 0.001]
            ... })
            >>> plotter._add_dunnett_markers(ax, dunnett_df, "duration", "Mixed Model")
            # Draws significance bars with "*" and "***" annotations
        """
        # Calculate y-axis limits for significance bars
        summary_stats = (
            self.drug_df.groupby(["concentration[um]", "bct_id"])[feature]
            .agg(["mean", "std"])
            .reset_index()
        )
        y_max = summary_stats["mean"].max() + summary_stats["std"].max()
        y_min = summary_stats["mean"].min() - summary_stats["std"].min()
        y_range = y_max - y_min

        # Spacing parameters
        spacing_factor = 0.18
        text_offset = y_range * 0.015

        # Add significance bars
        for i, row in dunnett_results.iterrows():
            x1 = self._get_mapped_concentration(self.baseline)
            x2 = self._get_mapped_concentration(
                float(row["Contrast"].split("-")[1].split("concentration")[1])
                if title_prefix == "Mixed Model"
                else float(row["Contrast"].split("-")[0].split("concentration")[1])
            )

            y = y_max + (int(i) + 1) * (y_range * spacing_factor)

            if row["P-val"] < 0.05:
                stars = "*" * sum(
                    [row["P-val"] < cutoff for cutoff in [0.05, 0.01, 0.001]]
                )
                ax.plot([x1, x2], [y, y], "k-", linewidth=1)
                ax.text((x1 + x2) / 2, y + text_offset, stars, ha="center", va="center")
            else:
                ax.plot([x2, x1], [y, y], "k-", linewidth=1)
                ax.text(
                    (x1 + x2) / 2,
                    y + text_offset,
                    "ns",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

        # Adjust y-axis limits
        current_ymin, current_ymax = ax.get_ylim()
        ax.set_ylim(current_ymin, current_ymax * 1.3)

    def _add_global_legend(self, fig: plt.Figure, handles: List, labels: List):
        """
        Add a single legend for the entire figure showing tissue identifiers.

        Creates a horizontal legend positioned above the subplots that identifies
        different tissues/batches used in the experiment. This provides a clean
        way to identify data series without cluttering individual subplots.

        Args:
            fig (plt.Figure): Matplotlib figure object
            handles (List): List of matplotlib artists (lines, markers) for legend
            labels (List): List of tissue/batch identifiers corresponding to handles

        Side Effects:
            - Adds legend positioned above subplots
            - Removes any existing individual subplot legends
            - Configures horizontal layout with appropriate spacing

        Example:
            >>> fig = plt.figure()
            >>> handles = [line1, line2, line3]
            >>> labels = ["tissue_1", "tissue_2", "tissue_3"]
            >>> plotter._add_global_legend(fig, handles, labels)
        """
        fig.legend(
            handles,
            labels,
            bbox_to_anchor=(0.5, 1.02),
            loc="lower center",
            fontsize=8,
            title="Tissue ID",
            title_fontsize=9,
            ncol=len(labels),
            borderaxespad=0,
        )

    def _adjust_figure_layout(self, fig: plt.Figure, n_features: int):
        """
        Adjust figure layout and spacing based on number of features displayed.

        Optimizes subplot spacing and margins to prevent overlap while maximizing
        use of available space. Adjustments are made based on the number of features
        to ensure readability across different plot configurations.

        Layout Adjustments:
        - Top margin: Reduced for >6 features to maximize space
        - Bottom/left margins: Consistent spacing for axis labels
        - Horizontal/vertical spacing: Increased for better separation

        Args:
            fig (plt.Figure): Matplotlib figure to adjust
            n_features (int): Number of features being displayed

        Side Effects:
            - Modifies figure subplot parameters
            - Adjusts margins and spacing
            - Optimizes layout for the specific number of subplots

        Example:
            >>> fig = plt.figure(figsize=(12, 8))
            >>> # ... create subplots ...
            >>> plotter._adjust_figure_layout(fig, 6)
        """
        top = 0.99 if n_features > 6 else (0.88 if n_features <= 3 else 0.95)
        fig.subplots_adjust(
            top=top, bottom=0.12, left=0.12, right=0.95, hspace=0.6, wspace=0.6
        )

    def _get_mapped_concentration(self, concentration: float) -> float:
        """
        Map concentration values to log-scale coordinates for plotting.

        Handles the special case of zero concentration (baseline) by mapping
        it to a small positive value that allows logarithmic scaling while
        maintaining visual separation from other concentrations.

        Mapping Rules:
        - Zero concentration: Mapped to smallest_nonzero_concentration / 10
        - Positive concentrations: Used as-is for log scaling

        Args:
            concentration (float): Original concentration value

        Returns:
            float: Mapped concentration value suitable for log-scale plotting

        Example:
            >>> # If concentrations are [0.0, 0.1, 1.0, 10.0]
            >>> plotter._get_mapped_concentration(0.0)
            0.01  # 0.1/10
            >>> plotter._get_mapped_concentration(1.0)
            1.0   # unchanged
        """
        if concentration == 0:
            min_nonzero = min(
                [c for c in self.drug_df["concentration[um]"].unique() if c > 0]
            )
            return min_nonzero / 10
        return concentration
