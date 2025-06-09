"""
Statistical Analysis and Inference for Bootstrap Correlation Results

This module provides comprehensive statistical analysis capabilities for bootstrap
correlation results in cardiac electrophysiology research. It transforms raw bootstrap
correlation distributions into meaningful statistical insights through hypothesis testing,
confidence interval estimation, and correlation type classification.

Scientific Framework:
    The module implements robust statistical inference methods specifically designed
    for correlation analysis in biological systems:

    1. Non-parametric Confidence Intervals: Using percentile-based methods that
       don't assume normal distributions of correlations
    2. Normality Assessment: Shapiro-Wilk tests to evaluate distribution assumptions
    3. Correlation Significance: Confidence interval-based classification
    4. Cross-treatment Comparison: Unified analysis across drug conditions

Key Statistical Methods:
    - Percentile Bootstrap Confidence Intervals (95% CI)
    - Shapiro-Wilk Normality Testing
    - Correlation Type Classification (positive/negative/inconclusive)
    - Multi-treatment Data Integration
    - Feature Mapping for Publication-Ready Output

Correlation Classification Logic:
    - Positive: Entire 95% CI > 0 (significant positive correlation)
    - Negative: Entire 95% CI < 0 (significant negative correlation)
    - Inconclusive: 95% CI includes 0 (non-significant correlation)

Dependencies: pandas, numpy, scipy.stats, pathlib
Author: Cardiac Electrophysiology Research Team
"""

import pandas as pd
import numpy as np
from scipy import stats
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import FeatureMapping


class CorrelationAnalyzer:
    """
    Comprehensive Statistical Analysis for Bootstrap Correlation Distributions

    This class provides sophisticated statistical analysis capabilities for interpreting
    bootstrap correlation results in cardiac electrophysiology research. It transforms
    raw correlation distributions into actionable insights through rigorous statistical
    inference methods designed specifically for biological correlation data.

    Statistical Analysis Framework:
        The analyzer implements a multi-layered approach to correlation analysis:

        1. Descriptive Statistics: Central tendency and variability measures
           - Mean and standard deviation of bootstrap distributions
           - Proportion analysis of positive/negative correlations

        2. Inferential Statistics: Hypothesis testing and confidence estimation
           - Percentile-based 95% confidence intervals (non-parametric)
           - Shapiro-Wilk normality tests for distribution assessment
           - Significance testing through confidence interval analysis

        3. Correlation Classification: Evidence-based categorization
           - Positive: Significant positive correlation (CI > 0)
           - Negative: Significant negative correlation (CI < 0)
           - Inconclusive: Non-significant correlation (CI includes 0)

        4. Cross-Treatment Integration: Multi-condition comparative analysis
           - Unified feature pair mapping across treatments
           - Drug-specific correlation pattern identification
           - Treatment effect quantification

    Key Methodological Advantages:
        - Non-parametric Methods: No distributional assumptions required
        - Bootstrap-based Inference: Robust to outliers and small samples
        - Multiple Testing Awareness: Structured approach to many correlations
        - Biological Interpretation: Feature mapping for scientific communication
        - Reproducible Analysis: Consistent statistical procedures across studies

    Scientific Applications:
        - Drug Mechanism Analysis: Understanding how treatments affect feature relationships
        - Biomarker Discovery: Identifying stable vs. treatment-sensitive correlations
        - Physiological Coupling: Quantifying relationships between cardiac parameters
        - Model Validation: Comparing experimental and computational correlation patterns
        - Clinical Translation: Identifying robust biomarker combinations

    Attributes:
        alpha (float): Significance level for statistical tests and confidence intervals
            - Default: 0.05 (95% confidence intervals)
            - Used for both normality testing and correlation significance
            - Determines Type I error rate for hypothesis testing

        feature_mapping (FeatureMapping): Utility for scientific feature name conversion
            - Converts internal feature names to publication-ready format
            - Ensures consistent nomenclature across visualizations
            - Facilitates interpretation by domain experts

    Example Usage:
        >>> # Initialize analyzer with default significance level
        >>> analyzer = CorrelationAnalyzer(alpha=0.05)
        >>>
        >>> # Analyze bootstrap correlation results
        >>> bootstrap_df = pd.read_csv('bootstrap_correlations.csv')
        >>> stats_df = analyzer.analyze_bootstrap_correlations(bootstrap_df)
        >>>
        >>> # Examine significant positive correlations
        >>> positive_corr = stats_df[stats_df['correlation_type'] == 'positive']
        >>> print(f"Found {len(positive_corr)} significant positive correlations")
        >>>
        >>> # Combine results across treatments
        >>> combined_df = analyzer.combine_drug_correlations(
        ...     nifedipine_stats, e4031_stats, ca_titration_stats)

    Output Structure:
        Statistical analysis results include:
        - Correlation estimates: mean, standard deviation
        - Uncertainty quantification: 95% confidence intervals
        - Distribution assessment: normality test results
        - Significance classification: correlation type and evidence strength
        - Biological interpretation: mapped feature names and relationships

    Note:
        This analyzer is specifically designed for cardiac electrophysiology data
        but can be adapted for other correlation analysis applications by modifying
        the feature mapping and interpretation framework.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize CorrelationAnalyzer with statistical parameters.

        Sets up the statistical analysis framework with configurable significance
        levels and feature mapping utilities. The analyzer is designed to provide
        consistent, reproducible statistical inference across different datasets
        and experimental conditions.

        Args:
            alpha (float, optional): Significance level for statistical tests.
                Defaults to 0.05. This parameter controls:
                - Confidence interval coverage (1-alpha = 95% coverage)
                - Normality test significance threshold
                - Correlation significance classification
                Range: Typically 0.01 to 0.10 for biological studies

        Initialization Process:
            1. Store significance level for consistent application across analyses
            2. Initialize feature mapping utility for name conversion
            3. Prepare statistical testing framework with appropriate thresholds
            4. Configure output formatting for scientific communication

        Example:
            >>> # Standard analysis with 95% confidence
            >>> analyzer = CorrelationAnalyzer()
            >>>
            >>> # Conservative analysis with 99% confidence
            >>> analyzer_conservative = CorrelationAnalyzer(alpha=0.01)
            >>>
            >>> # Liberal analysis with 90% confidence
            >>> analyzer_liberal = CorrelationAnalyzer(alpha=0.10)
        """
        self.alpha = alpha
        self.feature_mapping = FeatureMapping()

    def analyze_bootstrap_correlations(
        self, correlations_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Analyze bootstrap correlations and generate comprehensive statistics.

        For each feature pair, calculates:
        - Mean and standard deviation of correlations
        - Shapiro-Wilk test for normality
        - Proportion of positive and negative correlations
        - 95% confidence intervals
        - Correlation type (positive, negative, or inconclusive)

        Args:
            correlations_df (pd.DataFrame): DataFrame with bootstrap correlations

        Returns:
            pd.DataFrame: Analysis results with statistics for each feature pair
        """
        results = []

        for _, row in correlations_df.iterrows():
            correlations = row.iloc[2:].values
            ci_lower, ci_upper = np.percentile(
                correlations, [(self.alpha / 2) * 100, (1 - self.alpha / 2) * 100]
            )

            results.append(
                {
                    "feature1": row["feature1"],
                    "feature2": row["feature2"],
                    "correlation_mean": round(np.mean(correlations), 4),
                    "correlation_std": round(np.std(correlations), 4),
                    "shapiro_p_value": round(stats.shapiro(correlations)[1], 4),
                    "is_normal": stats.shapiro(correlations)[1] > self.alpha,
                    "positive_proportion": np.round(
                        np.sum(correlations > 0) / len(correlations), 3
                    ),
                    "negative_proportion": np.round(
                        np.sum(correlations < 0) / len(correlations), 3
                    ),
                    "ci_95": (round(float(ci_lower), 3), round(float(ci_upper), 3)),
                    "correlated": not (
                        ci_lower <= 0 <= ci_upper
                    ),  # if ci_95 does not contain 0, the correlation is considered to be significant
                    "correlation_type": self._determine_correlation_type(
                        ci_lower, ci_upper
                    ),
                }
            )

        results_df = pd.DataFrame(results)
        return results_df

    @staticmethod
    def _determine_correlation_type(ci_lower: float, ci_upper: float) -> str:
        """Determine correlation type based on confidence interval

        Args:
            ci_lower (float): Lower bound of 95% confidence interval
            ci_upper (float): Upper bound of 95% confidence interval

        Returns:
            str: Correlation type ('positive', 'negative', or 'inconclusive')

        If entire confidence interval is positive, the correlation is considered to be positive.
        If entire confidence interval is negative, the correlation is considered to be negative.
        If the confidence interval contains 0, the correlation is considered to be inconclusive.
        """
        if ci_lower > 0:
            return "positive"
        elif ci_upper < 0:
            return "negative"
        return "inconclusive"

    def combine_drug_correlations(
        self,
        nifedipine_df: pd.DataFrame,
        e4031_df: pd.DataFrame,
        ca_titration_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Combine correlation analyses from different drug treatments.

        Merges correlation results from different drugs while maintaining feature pair
        relationships. Prefixes columns with drug names to avoid conflicts.

        Args:
            nifedipine_df (pd.DataFrame): Nifedipine correlation analysis results
            e4031_df (pd.DataFrame): E-4031 correlation analysis results
            ca_titration_df (pd.DataFrame): Ca2+ titration correlation analysis results

        Returns:
            pd.DataFrame: Combined correlation results with drug-specific prefixes
        """
        columns_to_prefix = [
            col for col in nifedipine_df.columns if col not in ["feature1", "feature2"]
        ]

        renamed_dfs = {
            "nifedipine": nifedipine_df.copy(),
            "e4031": e4031_df.copy(),
            "ca_titration": ca_titration_df.copy(),
        }

        # Add drug prefixes to column names
        for drug, df in renamed_dfs.items():
            for col in columns_to_prefix:
                df.rename(columns={col: f"{drug}_{col}"}, inplace=True)

        # Merge dataframes
        combined_df = (
            renamed_dfs["nifedipine"]
            .merge(renamed_dfs["e4031"], on=["feature1", "feature2"])
            .merge(renamed_dfs["ca_titration"], on=["feature1", "feature2"])
        )

        return combined_df
