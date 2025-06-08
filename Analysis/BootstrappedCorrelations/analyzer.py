import pandas as pd
import numpy as np
from scipy import stats
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import FeatureMapping


class CorrelationAnalyzer:
    """
    Class to handle correlation analysis and statistics.

    This class analyzes bootstrap correlation distribution and generates various statistics
    including mean, standard deviation, confidence intervals, normality tests, and correlation types. It also
    combines the correlation analyses across different drug treatments into a single dataframe.

    Attributes:
        alpha (float): Significance level for statistical tests and confidence intervals
        feature_mapping (FeatureMapping): Instance for handling feature name mappings
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize CorrelationAnalyzer instance.

        Args:
            alpha (float): Significance level for statistical tests. Defaults to 0.05.
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
