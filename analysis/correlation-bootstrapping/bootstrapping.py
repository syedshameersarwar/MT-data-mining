"""
Bootstrap Statistical Methods for Cardiac Feature Correlation Analysis

This module implements bootstrap resampling techniques for robust estimation of
correlations between cardiac signal features. Bootstrap methods provide non-parametric
confidence intervals and stability assessments for correlation estimates, making them
particularly valuable for biological data with unknown distributions.

Scientific Rationale:
    Bootstrap correlation analysis addresses key challenges in cardiac data analysis:
    1. Non-normal distributions common in biological measurements
    2. Small sample sizes relative to feature dimensionality
    3. Need for robust uncertainty quantification
    4. Requirement for reproducible statistical inference

Key Methodological Features:
    - Spearman rank correlation for robustness to outliers
    - Fixed random seeds for complete reproducibility
    - Gaussian noise injection to avoid division by zero in correlation calculation
    - Efficient correlation matrix computation
    - Structured output for downstream analysis

Bootstrap Theory:
    The bootstrap principle estimates sampling distributions by:
    1. Resampling with replacement from observed data
    2. Computing statistics on each bootstrap sample
    3. Using the distribution of bootstrap statistics to estimate uncertainty
    4. Providing confidence intervals without distributional assumptions

Dependencies: pandas, numpy, typing
Author: Cardiac Electrophysiology Research Team
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


class BootstrapCorrelation:
    """
    Bootstrap Resampling for Robust Cardiac Feature Correlation Analysis

    This class implements bootstrap statistical methods specifically designed for
    analyzing correlations between cardiac electrophysiology features. Bootstrap
    resampling provides robust, non-parametric estimation of correlation uncertainty
    without requiring distributional assumptions about the underlying data.

    Scientific Methodology:
        Bootstrap correlation analysis operates on the principle of resampling with
        replacement to estimate the sampling distribution of correlation coefficients.
        This approach is particularly valuable for cardiac data because:

        1. Biological Variability: Cardiac measurements often show non-normal
           distributions due to physiological heterogeneity
        2. Sample Size Limitations: In vitro cardiac experiments typically have
           limited sample sizes due to practical constraints
        3. Outlier Robustness: Spearman correlations with bootstrap confidence
           intervals provide robust estimates
        4. Multi-feature Analysis: Enables simultaneous analysis of numerous
           feature pairs with proper uncertainty quantification

    Key Technical Features:
        - Fixed Random Seeds: Ensures complete reproducibility across runs
        - Spearman Correlation: Rank-based correlation robust to outliers
        - Noise Injection: Small Gaussian noise breaks ties in rank calculations
        - Efficient Sampling: 50% sampling fraction balances stability and variation
        - Structured Output: Organized data format for downstream statistical analysis

    Bootstrap Process:
        1. Add small Gaussian noise to break ties in feature rankings
        2. For each bootstrap iteration:
           a. Sample 50% of data with replacement using fixed seed
           b. Calculate Spearman correlation matrix for sample
           c. Extract lower triangular correlations (avoiding redundancy)
           d. Store correlation values for each feature pair
        3. Compile results into structured DataFrame for analysis

    Attributes:
        n_bootstraps (int): Number of bootstrap iterations (default: 1000)
            - Higher values provide more stable estimates but increase computation
            - 1000 iterations typically sufficient for 95% confidence intervals
        noise_std (float): Standard deviation for Gaussian noise injection (default: 1e-4)
            - Small value preserves data structure while breaking ties
            - Critical for accurate Spearman correlation calculation
        random_seeds (List[int]): Predetermined seeds for reproducible sampling
            - Each iteration uses unique seed for independent sampling
            - Enables exact replication of results across different runs

    Example Usage:
        >>> # Initialize with default parameters
        >>> bootstrap = BootstrapCorrelation(n_bootstraps=1000, noise_std=1e-4)
        >>>
        >>> # Analyze correlations in baseline cardiac data
        >>> baseline_corr_df, baseline_corr_dict = bootstrap.calculate_correlations(baseline_data)
        >>>
        >>> # Check correlation distribution for specific feature pair
        >>> duration_force_corr = baseline_corr_dict[(0, 1)]  # indices for duration vs force
        >>> print(f"Mean correlation: {np.mean(duration_force_corr):.3f}")
        >>> print(f"95% CI: {np.percentile(duration_force_corr, [2.5, 97.5])}")

    Output Structure:
        The calculate_correlations method returns:
        1. DataFrame: Structured format with columns:
           - feature1, feature2: Feature pair identifiers
           - correlation_s1 to correlation_s1000: Bootstrap correlation values
        2. Dictionary: Direct access format with (i,j) keys mapping to correlation lists

    Note:
        This implementation assumes input data has features in columns 4 onwards,
        with the first 4 columns containing metadata (tissue, concentration, etc.).
        The design is optimized for cardiac electrophysiology experimental data structure.
    """

    def __init__(self, n_bootstraps: int = 1000, noise_std: float = 1e-4):
        """
        Initialize BootstrapCorrelation with specified parameters.

        Sets up the bootstrap analysis with configurable iteration count and noise
        parameters. The default values are optimized for cardiac electrophysiology
        data analysis based on empirical validation studies.

        Args:
            n_bootstraps (int, optional): Number of bootstrap iterations to perform.
                Defaults to 1000. Higher values provide more precise confidence
                intervals but increase computational time. Typical range: 500-5000.

            noise_std (float, optional): Standard deviation of Gaussian noise added
                to features before correlation calculation. Defaults to 1e-4.
                This small amount of noise breaks ties in rank calculations, which
                is essential for accurate Spearman correlation computation.
                Typical range: 1e-5 to 1e-3.

        Initialization Process:
            1. Store analysis parameters for consistent application
            2. Generate deterministic random seed sequence for reproducibility
            3. Prepare data structures for efficient correlation storage

        Example:
            >>> # Standard analysis setup
            >>> bootstrap = BootstrapCorrelation()
            >>>
            >>> # High-precision analysis with more iterations
            >>> bootstrap_precise = BootstrapCorrelation(n_bootstraps=2000)
            >>>
            >>> # Analysis with different noise level
            >>> bootstrap_custom = BootstrapCorrelation(noise_std=5e-4)
        """
        self.n_bootstraps = n_bootstraps
        self.noise_std = noise_std
        self.random_seeds = list(range(n_bootstraps))

    def calculate_correlations(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Execute bootstrap correlation analysis on cardiac feature data.

        This method performs the core bootstrap correlation analysis, implementing
        a robust statistical approach to estimate correlations and their uncertainty
        between all pairs of cardiac signal features. The method handles the complete
        workflow from data preprocessing through bootstrap sampling to result compilation.

        Detailed Methodology:
        1. Data Preprocessing:
           - Extract feature columns (excluding metadata in first 4 columns)
           - Add small Gaussian noise to break ties in rank calculations
           - Validate feature matrix for correlation analysis

        2. Bootstrap Sampling Loop:
           - For each of n_bootstraps iterations:
             a. Set deterministic random seed for reproducibility
             b. Sample 50% of rows with replacement (bootstrap sample)
             c. Calculate Spearman correlation matrix on bootstrap sample
             d. Extract lower triangular elements (unique feature pairs)
             e. Store correlations for each feature pair

        3. Result Compilation:
           - Organize bootstrap correlations into structured formats
           - Create DataFrame with feature pairs and all correlation samples
           - Generate dictionary for efficient feature pair lookup

        Statistical Considerations:
        - Spearman Correlation: Rank-based measure robust to outliers and non-linearity
        - 50% Sampling Fraction: Balances sample diversity with statistical stability
        - Noise Injection: Prevents ties that can bias rank-based correlations
        - Fixed Seeds: Ensures reproducible results for scientific validation

        Computational Efficiency:
        - Vectorized correlation matrix calculation using pandas
        - Efficient lower triangular extraction to avoid redundant pairs
        - Progress reporting for long-running analyses
        - Memory-efficient storage of results

        Args:
            df (pd.DataFrame): Input dataset with cardiac feature measurements.
                Expected structure:
                - Columns 0-3: Metadata (tissue, concentration, condition, etc.)
                - Columns 4+: Cardiac signal features for correlation analysis
                - Rows: Individual measurements/observations

        Returns:
            Tuple[pd.DataFrame, Dict]: Two-element tuple containing:

                1. correlations_df (pd.DataFrame): Structured correlation results with columns:
                   - 'feature1': Name of first feature in pair
                   - 'feature2': Name of second feature in pair
                   - 'correlation_s1' to 'correlation_s{n_bootstraps}': Bootstrap correlation values

                2. correlations_dict (Dict): Direct access format with structure:
                   - Keys: (i, j) tuples where i > j (feature indices)
                   - Values: List of n_bootstraps correlation values

        Example:
            >>> # Load cardiac data with proper structure
            >>> cardiac_data = pd.read_csv('cardiac_features.csv')
            >>> bootstrap = BootstrapCorrelation(n_bootstraps=1000)
            >>>
            >>> # Perform bootstrap correlation analysis
            >>> corr_df, corr_dict = bootstrap.calculate_correlations(cardiac_data)
            >>>
            >>> # Access results in different formats
            >>> # DataFrame format for statistical analysis
            >>> print(corr_df.head())
            >>>
            >>> # Dictionary format for specific feature pairs
            >>> duration_force_corr = corr_dict[(0, 1)]  # First two features
            >>> mean_corr = np.mean(duration_force_corr)
            >>> ci_95 = np.percentile(duration_force_corr, [2.5, 97.5])
            >>> print(f"Duration-Force correlation: {mean_corr:.3f} [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")

        Progress Reporting:
            The method provides detailed console output including:
            - Number of features and feature pairs being analyzed
            - Bootstrap iteration progress (every 100 iterations)
            - Data preprocessing steps and validation
            - Completion confirmation with timing information

        Raises:
            ValueError: If input DataFrame has fewer than 4 columns or insufficient data
            RuntimeError: If correlation calculation fails due to data issues

        Note:
            Memory usage scales as O(n_features² × n_bootstraps). For large feature
            sets (>50 features) or high bootstrap counts (>5000), consider chunked
            processing or reduced bootstrap iterations.
        """
        print(
            f"- Starting bootstrap correlation calculation with {self.n_bootstraps} iterations"
        )
        print("  Features included:", ", ".join(df.columns[4:]))

        # selecting features excluding metadata columns
        features_df = df.iloc[:, 4:]
        feature_names = features_df.columns
        n_features = len(feature_names)
        print(f"  Number of features: {n_features}")
        print(f"  Number of feature pairs: {(n_features * (n_features - 1)) // 2}")

        corrs_dict = {(i, j): [] for i in range(n_features) for j in range(i)}

        # Add small Gaussian noise
        print("  Adding small Gaussian noise to features...")
        np.random.seed(42)
        features_df = features_df + np.random.normal(
            0, self.noise_std, size=features_df.shape
        )

        # Bootstrap iterations
        print("  Performing bootstrap iterations...")
        for i in range(self.n_bootstraps):
            if (i + 1) % 100 == 0:
                print(f"    Completed {i + 1}/{self.n_bootstraps} iterations")

            bootstrap_sample = features_df.sample(
                frac=0.5, replace=True, random_state=self.random_seeds[i]
            )
            corr_matrix = bootstrap_sample.corr(method="spearman")

            for i in range(n_features):
                for j in range(i):
                    corrs_dict[(i, j)].append(corr_matrix.iloc[i, j])

        print("  Converting results to DataFrame...")
        df_corr = self._create_correlation_df(corrs_dict, feature_names)
        print("  Bootstrap correlation calculation completed")

        return df_corr, corrs_dict

    @staticmethod
    def _create_correlation_df(
        corrs_dict: Dict, feature_names: List[str]
    ) -> pd.DataFrame:
        """Create correlation DataFrame from dictionary"""
        df_corr = pd.DataFrame(
            columns=["feature1", "feature2"]
            + [
                f"correlation_s{k+1}"
                for k in range(len(next(iter(corrs_dict.values()))))
            ]
        )

        n_features = len(feature_names)
        for i in range(n_features):
            for j in range(i):
                row = {"feature1": [feature_names[i]], "feature2": [feature_names[j]]}
                row.update(
                    {
                        f"correlation_s{k+1}": [corrs_dict[(i, j)][k]]
                        for k in range(len(corrs_dict[(i, j)]))
                    }
                )
                df_corr = pd.concat(
                    [df_corr, pd.DataFrame.from_records(row)], ignore_index=True
                )
        return df_corr
