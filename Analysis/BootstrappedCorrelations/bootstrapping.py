from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


class BootstrapCorrelation:
    """
    Class to handle bootstrap correlation analysis.

    This class performs bootstrap sampling on feature data and calculates Spearman
    correlations between feature pairs. It supports multiple bootstrap iterations
    with fixed random seeds for reproducibility.

    Attributes:
        n_bootstraps (int): Number of bootstrap iterations
        noise_std (float): Standard deviation of Gaussian noise added to features
        random_seeds (List[int]): List of random seeds for bootstrap sampling
    """

    def __init__(self, n_bootstraps: int = 1000, noise_std: float = 1e-4):
        """
        Initialize BootstrapCorrelation instance.

        Args:
            n_bootstraps (int): Number of bootstrap iterations. Defaults to 1000.
            noise_std (float): Standard deviation of Gaussian noise added to features.
                             Defaults to 1e-4.
        """
        self.n_bootstraps = n_bootstraps
        self.noise_std = noise_std
        self.random_seeds = list(range(n_bootstraps))

    def calculate_correlations(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Calculate bootstrap Spearman correlations between feature pairs."""
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
