"""
Principal Feature Analysis (PFA) Module for Cardiac Signal Feature Selection

This module provides comprehensive PFA analysis for selecting representative features
from cardiac signal data using PCA and DBSCAN clustering. The analysis identifies
clusters of features that contribute similarly to principal components and selects
representative features from each cluster.

Scientific Background:
    Principal Feature Analysis (PFA) is a feature selection technique that:
    1. Applies PCA to reduce dimensionality and identify principal components
    2. Uses DBSCAN clustering to group features with similar PCA contributions
    3. Automatically determines optimal clustering parameters using knee detection

    In cardiac electrophysiology, PFA reveals:
    - Groups of features with similar physiological roles
    - Redundant features that can be eliminated


Key Analysis Components:
    1. PCA Decomposition: Identify principal components and feature contributions
    2. Knee Detection: Automatically determine optimal DBSCAN epsilon parameter
    3. DBSCAN Clustering: Group features based on PCA contribution similarity
    4. Feature Selection: Choose representative features from each cluster
    5. Statistical Analysis: Analyze clustering patterns and feature importance

Pipeline Architecture:
    Data Loading → PCA Analysis → Knee Detection → DBSCAN Clustering → Feature Selection → Analysis

Supported Features:
    - Electrical: Action potential duration, frequency, local frequency
    - Calcium: Transient amplitude, kinetics, width measurements
    - Mechanical: Contractile force amplitude, duration, kinetics
    - Cross-modal: All combinations of signal type features

Drug Treatments Analyzed:
    - Baseline: Control conditions without drug intervention
    - E-4031: hERG potassium channel blocker (electrical effects)
    - Nifedipine: L-type calcium channel blocker (calcium/mechanical effects)
    - Ca²⁺ Titration: Calcium concentration modulation

Output Products:
    - Selected feature subsets for each drug treatment
    - PCA contribution matrices
    - Clustering analysis results
    - Knee detection curves and optimal parameters
    - Feature importance rankings

Authors: Cardiac Electrophysiology Research Team
Version: 1.0
Dependencies: pandas, numpy, scikit-learn, kneed
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from kneed import KneeLocator
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PfaAnalyzer:
    """
    Class for performing Principal Feature Analysis (PFA) on cardiac signal features.

    This class handles the complete PFA analysis pipeline including PCA decomposition,
    knee detection for DBSCAN parameter optimization, feature clustering, and
    representative feature selection. It provides methods for analyzing drug-specific
    feature relationships and identifying optimal feature subsets.

    Attributes:
        explained_var (float): Target explained variance for PCA (default: 0.95)
        diff_n_features (int): Number of features to select (if specified)
        eps (Optional[float]): DBSCAN epsilon parameter (auto-determined if None)
        min_samples (Optional[int]): DBSCAN min_samples parameter
        scaler (StandardScaler): Feature scaler for normalization
        pca_model (PCA): Fitted PCA model
        dbscan_model (DBSCAN): Fitted DBSCAN model
        kneedle (KneeLocator): Knee detection object for epsilon optimization
    """

    def __init__(
        self,
        explained_var: float = 0.95,
        eps: Optional[float] = None,
        min_samples: Optional[int] = None,
    ):
        """
        Initialize PfaAnalyzer with specified parameters.

        Args:
            explained_var (float): Target explained variance for PCA (default: 0.95)
            diff_n_features (int): Number of features to select (if specified)
            eps (Optional[float]): DBSCAN epsilon parameter (auto-determined if None)
            min_samples (Optional[int]): DBSCAN min_samples parameter
        """
        self.explained_var = explained_var
        self.eps = eps
        self.min_samples = min_samples

        # Initialize components
        self.scaler = StandardScaler()
        self.pca_model = None
        self.dbscan_model = None
        self.kneedle = None

        # Analysis results
        self.q = None  # Number of PCA components
        self._pca_components = None
        self.explained_variance_ = None
        self.feature_groups_ = {}
        self.indices_ = []
        self.features_ = None
        self.clusters_ = None

        logger.info(
            f"Initialized PfaAnalyzer with explained_var={explained_var}, "
            f"eps={eps}, min_samples={min_samples}"
        )

    def _get_pfa_features(self, X: pd.DataFrame) -> None:
        """
        Perform PCA analysis and determine optimal number of components.

        Args:
            X (pd.DataFrame): Input feature data
        """
        logger.info("Performing PCA analysis...")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit PCA
        self.pca_model = PCA().fit(X_scaled)

        # Determine number of components for target explained variance
        explained_variance = self.pca_model.explained_variance_ratio_
        cumulative_expl_var = [
            sum(explained_variance[: i + 1]) for i in range(len(explained_variance))
        ]

        for i, var in enumerate(cumulative_expl_var):
            if var >= self.explained_var:
                self.q = i
                break

        # Extract PCA components
        self._pca_components = self.pca_model.components_.T[
            :, : self.q
        ]  # Features x q matrix of PCA components
        self.explained_variance_ = explained_variance

        logger.info(
            f"Selected {self.q} PCA components for {self.explained_var} explained variance"
        )

    def _assign_dbscan_optimal_eps(self, X: np.ndarray) -> None:
        """
        Automatically determine optimal DBSCAN epsilon parameter using knee detection on nearest neighbors distances with respect to the min_samples parameter.

        Args:
            X (np.ndarray): PCA component matrix for feature clustering
        """
        logger.info("Determining optimal DBSCAN epsilon parameter...")

        # Set default min_samples if not specified
        if self.min_samples is None:
            self.min_samples = 2

        # Find optimal epsilon using knee detection
        nbrs = NearestNeighbors(n_neighbors=self.min_samples).fit(X)
        distances, _ = nbrs.kneighbors(X)

        # Sort distances in descending order
        distances = np.sort(distances, axis=0)[::-1]
        distances = distances[:, self.min_samples - 1]
        logger.info(f"Distances ({X.shape}): {distances}")

        # Use knee locator to find optimal epsilon
        self.kneedle = KneeLocator(
            list(range(1, len(distances) + 1)),
            distances,
            curve="convex",
            direction="decreasing",
        )

        self.eps = np.round(self.kneedle.knee_y, 2)
        logger.info(f"Optimal epsilon determined: {self.eps}")

    def _fit(self, X: pd.DataFrame) -> None:
        """
        Perform complete PFA analysis including PCA, clustering, and feature selection.

        Args:
            X (pd.DataFrame): Input feature data
        """
        # Step 1: PCA analysis
        self._get_pfa_features(X)

        # Step 2: Determine optimal epsilon based on the original features
        self._assign_dbscan_optimal_eps(X)

        # Step 3: DBSCAN clustering
        logger.info("Performing DBSCAN clustering...")
        self.dbscan_model = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(
            self._pca_components
        )
        clusters = self.dbscan_model.labels_  # Features x 1 array of cluster labels

        # Count clusters (excluding noise points labeled as -1)
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        logger.info(f"Found {n_clusters} clusters")

        # Step 4: Feature selection and grouping
        self._organize_feature_groups(X, clusters)

        # Step 5: Select representative features
        self._select_representative_features(X, clusters)

        logger.info("PFA analysis completed successfully")

    def _organize_feature_groups(self, X: pd.DataFrame, clusters: np.ndarray) -> None:
        """
        Organize features into groups based on clustering results.

        Args:
            X (pd.DataFrame): Original feature data
            clusters (np.ndarray): Cluster labels from DBSCAN
        """
        feature_names = X.columns

        # Initialize feature groups structure
        self.feature_groups_ = {
            "independent_features": [],  # Noise points
            "cluster_groups": {},  # Cluster information
            "selected_features": {},  # Representative features
            "cluster_pca_components": {},  # PCA components for each cluster
        }

        # Handle noise points (independent features)
        if -1 in clusters:
            noise_mask = clusters == -1
            noise_indices = np.where(noise_mask)[0]

            self.feature_groups_["independent_features"] = [
                {
                    "index": int(idx),
                    "feature_name": feature_names[idx],
                    "reason": "Noise point",
                }
                for idx in noise_indices
            ]
            logger.info(
                f"Found {len(noise_indices)} independent features (noise points)"
            )

        # Handle regular clusters
        for cluster_id in set(clusters) - {-1}:
            cluster_mask = clusters == cluster_id
            cluster_points = self._pca_components[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_points) == 0:
                continue

            # Calculate cluster center
            cluster_center = np.mean(cluster_points, axis=0)

            # Store cluster PCA components
            cluster_features = feature_names[cluster_indices]
            cluster_pca_df = pd.DataFrame(
                cluster_points,
                columns=[f"PC{i+1}" for i in range(self.q)],
                index=cluster_features,
            )
            self.feature_groups_["cluster_pca_components"][
                int(cluster_id)
            ] = cluster_pca_df

            # Store cluster information
            self.feature_groups_["cluster_groups"][int(cluster_id)] = {
                "features": [
                    {
                        "index": int(idx),
                        "feature_name": feature_names[idx],
                        "pca_components": cluster_pca_df.loc[feature_names[idx]],
                    }
                    for idx in cluster_indices
                ],
                "size": len(cluster_indices),
                "centroid": pd.Series(
                    cluster_center, index=[f"PC{i+1}" for i in range(self.q)]
                ),
            }

    def _select_representative_features(
        self, X: pd.DataFrame, clusters: np.ndarray
    ) -> None:
        """
        Select representative features from each cluster.

        Args:
            X (pd.DataFrame): Original feature data
            clusters (np.ndarray): Cluster labels from DBSCAN
        """
        feature_names = X.columns
        selected_indices = []

        # Add independent features (noise points)
        if -1 in clusters:
            noise_mask = clusters == -1
            noise_indices = np.where(noise_mask)[0]
            selected_indices.extend(noise_indices)

        # Select representative features from each cluster
        for cluster_id in set(clusters) - {-1}:
            cluster_mask = (
                clusters == cluster_id
            )  # Features x 1 array of with 1s for cluster features and 0s for other features
            cluster_points = self._pca_components[
                cluster_mask
            ]  # Cluster features x q matrix of PCA components
            cluster_indices = np.where(cluster_mask)[0]  # Cluster indices

            if len(cluster_points) == 0:
                continue

            # Calculate cluster center
            cluster_center = np.mean(cluster_points, axis=0)

            # Find closest point to center
            distances = []
            for idx, point in zip(cluster_indices, cluster_points):
                dist = euclidean_distances([point], [cluster_center])[0][0]
                distances.append((idx, dist))

            # Select closest point as representative
            if distances:
                selected_idx = min(distances, key=lambda x: x[1])[0]
                selected_indices.append(selected_idx)

                # Store selection information
                self.feature_groups_["selected_features"][int(cluster_id)] = {
                    "index": int(selected_idx),
                    "feature_name": feature_names[selected_idx],
                    "distance_to_center": float(
                        np.round(min(distances, key=lambda x: x[1])[1], 3)
                    ),
                    "cluster_size": len(cluster_indices),
                    "pca_components": pd.Series(
                        cluster_points[
                            distances.index(min(distances, key=lambda x: x[1]))
                        ],
                        index=[f"PC{i+1}" for i in range(self.q)],
                    ),
                }

        # Store final results
        self.indices_ = sorted(selected_indices)
        self.features_ = X.iloc[:, self.indices_]
        self.clusters_ = clusters

        # Print summary
        logger.info(f"Feature selection summary:")
        logger.info(f"- Total features selected: {len(self.indices_)}")
        logger.info(
            f"- Independent features: {len(self.feature_groups_['independent_features'])}"
        )
        logger.info(
            f"- Features from clusters: {len(self.feature_groups_['selected_features'])}"
        )

    def fit(self, X: pd.DataFrame) -> None:
        """
        Fit the PFA model to the data.

        Args:
            X (pd.DataFrame): Input feature data
        """
        self._fit(X)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the PFA model and return selected features.

        Args:
            X (pd.DataFrame): Input feature data

        Returns:
            pd.DataFrame: Selected features
        """
        self._fit(X)
        return self.features_

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted PFA model.

        Args:
            X (pd.DataFrame): Input feature data

        Returns:
            pd.DataFrame: Transformed data with selected features
        """
        return X.iloc[:, self.indices_]

    def get_selected_feature_names(self) -> List[str]:
        """
        Get names of selected features.

        Returns:
            List[str]: Names of selected features
        """
        if self.features_ is not None:
            return self.features_.columns.tolist()
        return []

    def get_analysis_summary(self) -> Dict:
        """
        Get comprehensive analysis summary.

        Returns:
            Dict: Analysis summary including clustering statistics and feature information
        """
        if self.features_ is None:
            return {}

        summary = {
            "n_original_features": len(self.pca_model.components_),
            "n_selected_features": len(self.indices_),
            "n_pca_components": self.q,
            "explained_variance": self.explained_var,
            "dbscan_eps": self.eps,
            "dbscan_min_samples": self.min_samples,
            "n_clusters": len(self.feature_groups_["cluster_groups"]),
            "n_independent_features": len(self.feature_groups_["independent_features"]),
            "selected_feature_names": self.get_selected_feature_names(),
            "cluster_summary": {
                cluster_id: {
                    "size": info["size"],
                    "representative_feature": info["features"][0]["feature_name"],
                }
                for cluster_id, info in self.feature_groups_["cluster_groups"].items()
            },
        }

        return summary
